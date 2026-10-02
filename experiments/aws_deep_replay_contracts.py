"""Depth4 corrected full-replay gradients and actual update/recovery contracts."""
import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_deep_replay as D
import dvs_critic_le_benchmark as CR
import dvs_local_expectation_benchmark as LE
import dvs_native_contracts as E
from sleeping_machines.factorized_race import factorized_race


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);tag=p.parse_args().tag;started=time.perf_counter();torch.set_num_threads(1)
    common=['--tag','contract','--data','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.json',
        '--controls','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_controls.json','--bins','4','--clock-step','.25',
        '--depth','4','--payload','4','--heads','2','--pool','2','--fit','4','--dev','2','--epochs','2','--update-targets','3','--seed','7']
    a=D.parse([*common,'--credit-mode','replay']);fit,dev,data=D.R.load(a);model=D.BL.make_model(a).double();reference=copy.deepcopy(model)
    rng=np.random.default_rng(681);rows=[dict(index=i,target=i%11,events=[((t+1)*.25,rng.standard_normal(33)) for t in range(3)]) for i in range(2)]
    a.route_samples=1000000;a.critic_width=8;a.critic_lr=.003;a.fork=True;a.lanes=False;a.no_critic=True
    LE.PATHWISE=factorized_race;CR.CRITIC.clear()
    CR.train_window(reference,torch.optim.SGD(reference.parameters(),lr=0),rows,a,epoch=1)
    D.BL.train_window(model,torch.optim.SGD(model.parameters(),lr=0),rows,a,epoch=1)
    for (name,p),(_,q) in zip(reference.named_parameters(),model.named_parameters()):
        x=torch.zeros_like(p) if p.grad is None else p.grad;y=torch.zeros_like(q) if q.grad is None else q.grad
        torch.testing.assert_close(x,y,rtol=1e-7,atol=1e-9,msg=name)
    cases=[]
    for mode in ('teacher','factorized','replay'):
        with tempfile.TemporaryDirectory(prefix='aws-deep-replay-') as tmp:
            def args(name):
                result=D.parse([*common,'--credit-mode',mode]);result.tag=name;return result
            full=D.run(args('full'),tmp);interrupted=args('recover');interrupted.stop_after_updates=1;D.run(interrupted,tmp)
            interrupted.stop_after_updates=None;interrupted.resume=True;second=D.run(interrupted,tmp)
            first_checkpoint=torch.load(Path(tmp)/'full.progress.pt',weights_only=False);second_checkpoint=torch.load(Path(tmp)/'recover.progress.pt',weights_only=False)
            for key in ('online_model','optimizer','best_state','best','cursor','torch_rng'):E.equal(first_checkpoint[key],second_checkpoint[key])
            for key in ('final','activity','work','work_samples','selected_epoch'):E.equal(full[key],second[key])
            for result in (full,second):
                for sample in result['work_samples']:assert all(s['formula_coverage_complete'] for s in sample['stages'].values())
            cases.append(dict(mode=mode,depth=4,update_recovery_and_complete_operator_contract=True))
    out=ROOT/'experiments/results/diagnostics'/f'{tag}.json';assert not out.exists()
    result=dict(status='completed',tag=tag,contracts_passed=True,deep_every_parameter_replay_gradient_equivalence=True,cases=cases,
        source_sha256=D.sources(),wall_s=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Deep integrated numerical admission, not a quality or full-risk-gradient theorem')
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
