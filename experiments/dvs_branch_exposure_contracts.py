"""Independent ordinary-autograd reference for score-blocked legal branches."""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_branch_exposure_audit as A
import dvs_native_contracts as C


@contextmanager
def functional_route(site,option):
    previous=A.B.K.BatchedTemporalRoute;counter=0
    def apply(scores,values,noise):
        nonlocal counter
        first,winner=(noise[None]/scores.detach().double().exp()).min(-1)
        if counter==site and option is not None:
            winner=winner.clone();winner[:,0]=option
        counter+=1
        chosen=values.gather(2,winner[...,None,None].expand(*winner.shape,1,values.shape[-1])).squeeze(2)
        return chosen,.001+.010*first/(1+first),winner
    A.B.K.BatchedTemporalRoute=SimpleNamespace(apply=apply)
    try:yield
    finally:A.B.K.BatchedTemporalRoute=previous


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--data',required=True);p.add_argument('--controls',required=True)
    a=p.parse_args();started=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if out.exists() or Path(a.tag).name!=a.tag:raise ValueError('Unique tag required')
    config=A.B.N.parser().parse_args(['--tag','contract','--data',a.data,'--controls',a.controls,'--fit','2','--dev','2'])
    rows,_,metadata=A.B.N.load(config);model=A.B.N.make_model(config).double();targets=torch.tensor([r['target'] for r in rows]);trials=[]
    for event,depth in ((9,0),(19,1)):
        site=event*model.depth+depth
        for option in (None,0,1):
            first=copy.deepcopy(model);second=copy.deepcopy(model)
            with A.trace(site,option=option,blocked=True):z,state,_=A.B.K.forward(first,rows,701)
            with functional_route(site,option):rz,rs,_=A.B.K.forward(second,rows,701)
            torch.testing.assert_close(z,rz,rtol=0,atol=0);C.equal(state,rs)
            g=A.grads(F.cross_entropy(z,targets,reduction='sum'),first)
            rg=A.grads(F.cross_entropy(rz,targets,reduction='sum'),second)
            for name in g:torch.testing.assert_close(g[name],rg[name],rtol=1e-12,atol=1e-12)
            assert all(float(v.norm())==0 for n,v in g.items() if A.group(n)=='route_maps')
            assert any(float(v.norm())>0 for n,v in g.items() if A.group(n)=='message_maps')
            trials.append(dict(event=event,depth=depth,option=option,factual_and_state_match=True,every_parameter_gradient_matches_functional_autograd=True))
    # Conditional weighting identity; gradients of probabilities handled separately.
    theta=torch.tensor([.3,-.7],dtype=torch.float64,requires_grad=True)
    scores=torch.tensor([.2,-.1],dtype=torch.float64,requires_grad=True)
    pi=scores.softmax(0);losses=torch.stack(((theta[0]-1).square()+theta[1],(theta[1]+2).square()-theta[0]))
    direct=torch.autograd.grad((pi*losses).sum(),theta,retain_graph=True)[0]
    weighted=torch.autograd.grad((pi.detach()*losses).sum(),theta,retain_graph=True)[0]
    torch.testing.assert_close(direct,weighted,rtol=0,atol=0)
    choice=torch.autograd.grad((pi*losses.detach()).sum(),scores)[0]
    torch.testing.assert_close(choice,pi.detach()*(losses.detach()-(pi.detach()*losses.detach()).sum()),rtol=1e-12,atol=1e-12)
    result=dict(status='completed',args=vars(a),contracts_passed=2,integrated_trials=trials,
        exact_detached_weight_branch_derivative=True,separate_categorical_derivative=True,data=metadata,
        source_sha256={**A.B.sources(),'experiments/dvs_branch_exposure_audit.py':A.B.N.sha(ROOT/'experiments/dvs_branch_exposure_audit.py'),
            'experiments/dvs_branch_exposure_contracts.py':A.B.N.sha(Path(__file__)),
            'experiments/dvs_native_contracts.py':A.B.N.sha(ROOT/'experiments/dvs_native_contracts.py'),
            'experiments/dvs_route_content_audit.py':A.B.N.sha(ROOT/'experiments/dvs_route_content_audit.py')},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Six integrated factual/forced legal branches compare every parameter derivative and full state against independent ordinary-autograd gather with detached race scores/times. Explicit diagnostic clock/route score blocking, not a whole expected-risk derivative or training result.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
