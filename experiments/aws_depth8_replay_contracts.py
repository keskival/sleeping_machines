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
import aws_depth8_replay as S
D=S.D
import dvs_critic_le_benchmark as CR
import dvs_local_expectation_benchmark as LE
import dvs_native_contracts as E
from sleeping_machines.factorized_race import factorized_race


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);tag=p.parse_args().tag;started=time.perf_counter();torch.set_num_threads(1)
    common=['--tag','contract','--data','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.json',
        '--controls','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_controls.json','--bins','4','--clock-step','.25',
        '--depth','8','--receiver-sharing','depth','--payload','4','--heads','2','--pool','2','--fit','4','--dev','2','--epochs','2','--update-targets','3','--seed','7']
    a=S.parse([*common,'--credit-mode','replay']);fit,dev,data=D.R.load(a);model=S.make_model(a).double();reference=copy.deepcopy(model)
    rng=np.random.default_rng(681);rows=[dict(index=i,target=i%11,events=[((t+1)*.25,rng.standard_normal(33)) for t in range(3)]) for i in range(2)]
    a.route_samples=1000000;a.critic_width=8;a.critic_lr=.003;a.fork=True;a.lanes=False;a.no_critic=True
    LE.PATHWISE=factorized_race;CR.CRITIC.clear()
    CR.train_window(reference,torch.optim.SGD(reference.parameters(),lr=0),rows,a,epoch=1)
    D.BL.train_window(model,torch.optim.SGD(model.parameters(),lr=0),rows,a,epoch=1)
    for (name,p),(_,q) in zip(reference.named_parameters(),model.named_parameters()):
        x=torch.zeros_like(p) if p.grad is None else p.grad;y=torch.zeros_like(q) if q.grad is None else q.grad
        torch.testing.assert_close(x,y,rtol=1e-7,atol=1e-9,msg=name)
    # Gradient alias contract using the original teacher's native physical path.
    shared=S.make_model(a).double();untied=S.BASE_MAKE(a).double();untied.load_state_dict(shared.state_dict())
    outputs=[]
    for candidate in (shared,untied):
        candidate.zero_grad();values=[];states=[]
        for row in rows:
            value,state=D.N.predict(candidate,row,1729,True);values.append(value);states.append(state)
        torch.nn.functional.cross_entropy(torch.stack(values),torch.tensor([r['target'] for r in rows]),reduction='sum').backward()
        outputs.append((torch.stack(values),states))
    torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=0,atol=1e-10)
    for x,y in zip(outputs[0][1],outputs[1][1]):E.equal(x.memories,y.memories);E.equal(x.arrivals,y.arrivals);E.equal(x.contexts,y.contexts)
    old_parameters=dict(untied.named_parameters())
    for name,p in shared.named_parameters():
        parts=name.split('.')
        if parts[0]=='units' and parts[5] in S.S.SHARED:
            assert parts[1]=='0' and parts[4]=='0'
            names=['.'.join([parts[0],str(d),parts[2],parts[3],str(u),*parts[5:]]) for d in range(a.depth) for u in range(a.pool)]
            expected=sum((old_parameters[n].grad if old_parameters[n].grad is not None else torch.zeros_like(old_parameters[n])) for n in names)
        else:expected=old_parameters[name].grad
        actual=torch.zeros_like(p) if p.grad is None else p.grad
        expected=torch.zeros_like(p) if expected is None else expected
        torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-9,msg=name)
    assert len({id(p) for p in shared.parameters()})==len(list(shared.parameters()))
    for h in range(shared.heads):
        master=shared.units[0][h][0][0]
        assert all(u.input is master.input for d in shared.units for u in d[h][0])
        assert len({id(u.key) for d in shared.units for u in d[h][0]})==a.depth*a.pool
    a.receiver_sharing='private';private=S.make_model(a).double();sequential=copy.deepcopy(private)
    CR.CRITIC.clear();LE.PATHWISE=factorized_race
    CR.train_window(sequential,torch.optim.SGD(sequential.parameters(),lr=0),rows,a,epoch=1)
    D.BL.train_window(private,torch.optim.SGD(private.parameters(),lr=0),rows,a,epoch=1)
    for (name,p),(_,q) in zip(sequential.named_parameters(),private.named_parameters()):
        x=torch.zeros_like(p) if p.grad is None else p.grad;y=torch.zeros_like(q) if q.grad is None else q.grad
        torch.testing.assert_close(x,y,rtol=1e-7,atol=1e-9,msg=name)
    a.receiver_sharing='depth'
    cases=[]
    for family,mode in [(f,m) for f in ('private','depth') for m in ('teacher','factorized','replay')]:
        with tempfile.TemporaryDirectory(prefix='aws-deep-replay-') as tmp:
            def args(name):
                result=S.parse([*common,'--credit-mode',mode]);result.tag=name;result.receiver_sharing=family;return result
            full=S.run(args('full'),tmp);interrupted=args('recover');interrupted.stop_after_updates=1;S.run(interrupted,tmp)
            interrupted.stop_after_updates=None;interrupted.resume=True;second=S.run(interrupted,tmp)
            first_checkpoint=torch.load(Path(tmp)/'full.progress.pt',weights_only=False);second_checkpoint=torch.load(Path(tmp)/'recover.progress.pt',weights_only=False)
            for key in ('online_model','optimizer','best_state','best','cursor','torch_rng'):E.equal(first_checkpoint[key],second_checkpoint[key])
            for key in ('final','activity','work','work_samples','selected_epoch'):E.equal(full[key],second[key])
            for result in (full,second):
                for sample in result['work_samples']:assert all(s['formula_coverage_complete'] for s in sample['stages'].values())
            cases.append(dict(family=family,mode=mode,depth=8,update_recovery_and_complete_operator_contract=True))
    out=ROOT/'experiments/results/diagnostics'/f'{tag}.json';assert not out.exists()
    result=dict(status='completed',tag=tag,args=dict(tag=tag),contracts_passed=True,deep_every_parameter_replay_gradient_equivalence=True,shared_alias_gradient_and_private_state_contract=True,cases=cases,
        source_sha256=S.sources(),wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Deep integrated numerical admission, not a quality or full-risk-gradient theorem')
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
