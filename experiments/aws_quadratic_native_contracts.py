"""Quadratic nesting, deep native derivatives and actual driver recovery."""
import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_quadratic_native as Q
import dvs_native_contracts as E

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();started=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    common=['--data','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.json','--controls','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_controls.json','--fit','4','--dev','2','--epochs','2','--update-targets','3','--bins','4','--clock-step','.25']
    config=Q.parser().parse_args(['--tag','contract',*common]);fit,dev,data=Q.A.load(config)
    reference=Q.A.C.make_model(config).double();rng=torch.get_rng_state().clone();nested=Q.replace_head(copy.deepcopy(reference));assert torch.equal(rng,torch.get_rng_state())
    labels=torch.tensor([r['target'] for r in fit])
    z,s,_=Q.A.B.forward(reference,fit,127);zn,sn,_=Q.A.B.forward(nested,fit,127)
    torch.testing.assert_close(z,zn,rtol=0,atol=0);E.equal(s,sn)
    F.cross_entropy(z,labels,reduction='sum').backward();F.cross_entropy(zn,labels,reduction='sum').backward()
    by=dict(nested.named_parameters())
    for name,parameter in reference.named_parameters():
        mapped=name.replace('head.weight','head.linear.weight').replace('head.bias','head.linear.bias')
        other=by[mapped];one=torch.zeros_like(parameter) if parameter.grad is None else parameter.grad;two=torch.zeros_like(other) if other.grad is None else other.grad
        torch.testing.assert_close(one,two,rtol=0,atol=0)
    assert float(nested.head.quadratic.weight.grad.norm())>0
    head=copy.deepcopy(nested.head);head.quadratic.weight.data.uniform_(-.1,.1)
    x=torch.randn(3,32,dtype=torch.float64,requires_grad=True)
    products=torch.stack([x[:,i]*x[:,j] for i in range(32) for j in range(i,32)],-1)/32**.5
    explicit=head.linear(x)+F.linear(products,head.quadratic.weight);actual=head(x)
    torch.testing.assert_close(actual,explicit,rtol=0,atol=0)
    params=[x,*head.parameters()];one=torch.autograd.grad(actual.square().sum(),params,retain_graph=True);two=torch.autograd.grad(explicit.square().sum(),params)
    for u,v in zip(one,two):torch.testing.assert_close(u,v,rtol=1e-12,atol=1e-12)
    model=Q.make_model(config).double();model.head.quadratic.weight.data.uniform_(-.02,.02);batch=copy.deepcopy(model);predictions=[];states=[]
    for row in fit:
        value,state=Q.N.predict(model,row,1237,True);predictions.append(value);states.append(state)
    F.cross_entropy(torch.stack(predictions),labels,reduction='sum').backward()
    logits,state,_=Q.A.B.forward(batch,fit,1237);F.cross_entropy(logits,labels,reduction='sum').backward()
    torch.testing.assert_close(logits,torch.stack(predictions),rtol=1e-9,atol=1e-10)
    for u,v in zip(model.parameters(),batch.parameters()):
        ug=torch.zeros_like(u) if u.grad is None else u.grad;vg=torch.zeros_like(v) if v.grad is None else v.grad
        torch.testing.assert_close(ug,vg,rtol=1e-8,atol=1e-9)
    for i,old in enumerate(states):
        for (d,h,s,u),v in old.memories.items():torch.testing.assert_close(state['memories'][d][i,h*config.pool+u],v,rtol=1e-9,atol=1e-10)
    model.eval();flipped=dict(dev[0],target=(dev[0]['target']+1)%11)
    with torch.no_grad():
        one,_=Q.N.predict(model,dev[0],619,False);two,_=Q.N.predict(model,flipped,619,False)
    torch.testing.assert_close(one,two,rtol=0,atol=0)
    with tempfile.TemporaryDirectory(prefix='aws-quadratic-recovery-') as temp:
        c=Q.parser().parse_args(['--tag','continuous',*common]);first=Q.run(c,temp)
        r=Q.parser().parse_args(['--tag','recovered',*common]);r.stop_after_updates=1;Q.run(r,temp);r.stop_after_updates=None;r.resume=True;second=Q.run(r,temp)
        for result in (first,second):
            for sample in result['work_samples']:
                for stage in sample['stages'].values():assert stage['formula_coverage_complete']
        x=torch.load(Path(temp)/'continuous.progress.pt',weights_only=False);y=torch.load(Path(temp)/'recovered.progress.pt',weights_only=False)
        for key in ('online_model','optimizer','best_state','best','cursor','torch_rng'):E.equal(x[key],y[key])
        for key in ('final','activity','work','work_samples','selected_epoch'):E.equal(first[key],second[key])
    out.write_text(json.dumps(dict(status='completed',args=vars(a),source_sha256=Q.sources(),data=data,initial_forward_and_all_original_gradients_exact=True,residual_learning_gradient_nonzero=True,explicit_polynomial_and_gradients_match=True,all_native_gradients_and_states_match_independent_clips=True,target_independent_forward=True,actual_driver_adam_rng_cursor_recovery=True,formula_coverage_complete=True,wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Integrated zero-nested polynomial readout prerequisites; no quality claim.'),indent=2)+'\n')
if __name__=='__main__':main()
