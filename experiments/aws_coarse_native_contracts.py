"""Native coarse-adapter forward, gradient, data and interrupted recovery."""
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
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_coarse_native as A
import dvs_native_contracts as E

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);start=time.perf_counter()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    common=['--data','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.json','--controls','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_controls.json','--fit','4','--dev','2','--epochs','2','--update-targets','3']
    cases=[]
    for bins,step in [(20,.05),(4,.05),(4,.25)]:
        config=A.parser().parse_args(['--tag','contract',*common,'--bins',str(bins),'--clock-step',str(step)])
        fit,dev,info=A.load(config)
        if bins==20:
            original=A.BASE_LOAD(config)
            assert original[2]==info
            for x,y in zip(fit,original[0]):
                for e,f in zip(x['events'],y['events']):assert e[0]==f[0];np.testing.assert_array_equal(e[1],f[1])
        else:
            z=np.load(ROOT/'experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.data.npz',allow_pickle=False)
            compact=A.coalesce(z['fit_counts'],4);np.testing.assert_array_equal(compact.sum((1,2)),z['fit_counts'].sum((1,2)))
            changed=z['fit_counts'].copy();changed[:,-5:]+=17
            np.testing.assert_array_equal(A.coalesce(changed,4)[:,:3],compact[:,:3])
            assert [t for t,_ in fit[0]['events']]==[.25,.5,.75,1.,1.]
            mutated=copy.deepcopy(fit[0]);mutated['target']=(mutated['target']+1)%11
            for x,y in zip(fit[0]['events'],mutated['events']):np.testing.assert_array_equal(x[1],y[1])
        reference=A.C.make_model(config).double();batch=copy.deepcopy(reference);independent=[];states=[]
        for row in fit:
            value,state=A.N.predict(reference,row,1237,True);independent.append(value);states.append(state)
        labels=torch.tensor([r['target'] for r in fit]);F.cross_entropy(torch.stack(independent),labels,reduction='sum').backward()
        logits,state,_=A.B.forward(batch,fit,1237);F.cross_entropy(logits,labels,reduction='sum').backward()
        torch.testing.assert_close(logits,torch.stack(independent),rtol=1e-9,atol=1e-10)
        for x,y in zip(reference.parameters(),batch.parameters()):
            gx=torch.zeros_like(x) if x.grad is None else x.grad;gy=torch.zeros_like(y) if y.grad is None else y.grad
            torch.testing.assert_close(gx,gy,rtol=1e-8,atol=1e-9)
        for i,old in enumerate(states):
            for (d,h,s,u),v in old.memories.items():torch.testing.assert_close(state['memories'][d][i,h*config.pool+u],v,rtol=1e-9,atol=1e-10)
            old_context,old_times=old.contexts[0]
            torch.testing.assert_close(state['context'][i].reshape(-1),old_context,rtol=1e-9,atol=1e-10)
            torch.testing.assert_close(state['context_times'][i],old_times,rtol=1e-9,atol=1e-10)
        with tempfile.TemporaryDirectory(prefix='aws-coarse-recovery-') as temp:
            def args(tag):return A.parser().parse_args(['--tag',tag,*common,'--bins',str(bins),'--clock-step',str(step)])
            first=A.run(args('continuous'),temp);recovered=args('recovered');recovered.stop_after_updates=1;A.run(recovered,temp)
            recovered.stop_after_updates=None;recovered.resume=True;second=A.run(recovered,temp)
            for result in (first,second):
                for sample in result['work_samples']:
                    for stage in sample['stages'].values():assert stage['formula_coverage_complete']
            x=torch.load(Path(temp)/'continuous.progress.pt',weights_only=False);y=torch.load(Path(temp)/'recovered.progress.pt',weights_only=False)
            for key in ('online_model','optimizer','best_state','best','cursor','torch_rng'):E.equal(x[key],y[key])
            for key in ('final','activity','work','work_samples','selected_epoch'):E.equal(first[key],second[key])
        cases.append(dict(bins=bins,clock_step=step,forward_all_gradients_states_and_recovery_pass=True))
    out.write_text(json.dumps(dict(status='completed',args=vars(a),cases=cases,contracts_passed=True,source_sha256=A.sources(),wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Integrated numerical prerequisites; no held-out quality claim.'),indent=2)+'\n')
if __name__=='__main__':main()
