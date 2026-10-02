"""Numerical equivalence, optimizer recovery and measured replay work."""
import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_state_choice_credit_benchmark as C
import aws_checkpointed_choice_credit as B
import dvs_native_contracts as E

def compare(a,b,tol):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=tol,atol=tol)
    elif isinstance(a,dict):
        for k in a:
            if k not in ('events','_shadow_events'):compare(a[k],b[k],tol)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y,tol)
    else:assert a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    start=time.perf_counter();torch.set_num_threads(1)
    common=['--data','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.json','--controls','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_controls.json','--fit','4','--dev','2','--epochs','2','--update-targets','2']
    config=B.N.parser().parse_args(['--tag','contract']+common)
    fit,_,info=B.N.load(config);cases=[]
    for dtype,tol in [(torch.float64,1e-10),(torch.float32,1e-5)]:
        for epoch in range(1,5):
            x=B.N.make_model(config).to(dtype);y=copy.deepcopy(x)
            rng=torch.get_rng_state().clone()
            lx,sx,dx=C.corrected_loss(x,fit,991,epoch)
            assert torch.equal(torch.get_rng_state(),rng)
            ly,sy,dy=B.corrected_loss(y,fit,991,epoch)
            assert torch.equal(torch.get_rng_state(),rng)
            compare(lx,ly,tol);compare(sx,sy,tol)
            compare(dx['outcome_losses'],dy['outcome_losses'],tol)
            compare(dx['shadow_state'],dy['shadow_state'],tol)
            lx.backward();ly.backward()
            for (name,q),(name2,r) in zip(x.named_parameters(),y.named_parameters()):
                assert name==name2;compare(q.grad,r.grad,tol)
            ox=torch.optim.Adam(x.parameters(),lr=.003);oy=torch.optim.Adam(y.parameters(),lr=.003)
            ox.step();oy.step();compare(x.state_dict(),y.state_dict(),tol);compare(ox.state_dict(),oy.state_dict(),tol)
            cases.append(dict(dtype=str(dtype),epoch=epoch,reference_events=sx['events']+dx['shadow_state']['events'],optimized_events=sy['events']+dy['shadow_state']['events']))
    work=[]
    for module in (C,B):
        model=B.N.make_model(config);optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        begin=time.perf_counter()
        with module.activate():row=B.N.train_window(model,optimizer,fit,config,1,True)
        for stage in row['stages'].values():assert stage['formula_coverage_complete']
        work.append(dict(variant=module.__name__,wall_s=time.perf_counter()-begin,events=row['events'],arithmetic_flops=sum(v['arithmetic_flops'] for v in row['stages'].values()),specials=sum(v['special_function_evaluations'] for v in row['stages'].values()),stages=row['stages']))
    with tempfile.TemporaryDirectory(prefix='aws-prefix-recovery-') as temp:
        c=B.N.parser().parse_args(['--tag','continuous']+common);first=B.run(c,temp)
        r=B.N.parser().parse_args(['--tag','recovered']+common);r.stop_after_updates=1;B.run(r,temp)
        r.stop_after_updates=None;r.resume=True;second=B.run(r,temp)
        xc=torch.load(Path(temp)/'continuous.progress.pt',weights_only=False);yc=torch.load(Path(temp)/'recovered.progress.pt',weights_only=False)
        for name in ('online_model','optimizer','best_state','best','cursor','torch_rng'):E.equal(xc[name],yc[name])
        for name in ('final','activity','work','work_samples','selected_epoch','credit_protocol'):E.equal(first[name],second[name])
    result=dict(status='completed',args=vars(a),data=info,cases=cases,work=work,contracts_passed=True,
        next_adam_equal=True,interrupted_driver_recovery_equal=True,global_rng_preserved=True,
        factual_and_alternative_state_equal=True,all_parameter_gradients_equal=True,
        source_sha256={**B.sources(),'experiments/aws_prefix_replay_contracts.py':B.N.sha(Path(__file__))},
        wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Implementation equivalence on fixed real fitting prefixes and initialization; no held-out quality or cross-model advantage claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
