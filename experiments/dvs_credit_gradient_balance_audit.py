"""Frozen paired credit changes, parameter-group balance and clipping diagnosis."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_state_choice_credit_benchmark as B
from dvs_route_content_audit import group


def flat(gradient):return torch.cat([g.double().flatten() for g in gradient.values()])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--native',action='append',required=True);a=p.parse_args();started=time.perf_counter()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    torch.set_num_threads(1);models=[];sources={}
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        for source,digest in parent['source_sha256'].items():
            if B.N.sha(ROOT/source)!=digest:raise ValueError('Changed frozen source '+source)
        sources.update(parent['source_sha256']);args=argparse.Namespace(**parent['args'])
        fitting,_,data=B.N.load(args);assert data==parent['data'];rows=fitting[:16]
        model=B.N.make_model(args);checkpoint=(ROOT/name).with_suffix('.progress.pt')
        saved=torch.load(checkpoint,weights_only=False);assert saved['source_sha256']==parent['source_sha256']
        model.load_state_dict(saved['best_state']);before={n:p.detach().clone() for n,p in model.named_parameters()}
        gradients={};results=[];reference_logits=None
        for mode in ('local','state_choice','state_clock'):
            model.zero_grad(set_to_none=True);box={}
            def forward():
                if mode=='local':
                    logits,state,_=B.K.forward(model,rows,314159)
                    loss=F.cross_entropy(logits,torch.tensor([r['target'] for r in rows]),reduction='sum')
                else:
                    fn=B.corrected_loss if mode=='state_choice' else B.S.corrected_loss
                    loss,state,detail=fn(model,rows,314159,1);logits=detail['logits']
                box.update(loss=loss,logits=logits)
            forward_work=B.N.capture(forward)
            # These drivers use summed loss and normalize AFTER all injected credit.
            backward_work=B.N.capture(lambda:box['loss'].backward())
            def normalize():
                for parameter in model.parameters():
                    if parameter.grad is not None:parameter.grad.div_(len(rows))
            normalization_work=B.N.capture(normalize)
            for work in (forward_work,backward_work,normalization_work):
                assert work['formula_coverage_complete'],work['unsupported_floating_operators']
            if reference_logits is None:reference_logits=box['logits'].detach()
            else:torch.testing.assert_close(box['logits'],reference_logits,rtol=0,atol=0)
            gradient={n:torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
            gradients[mode]=gradient;vector=flat(gradient);groups={}
            for n,g in gradient.items():groups[group(n)]=groups.get(group(n),0)+float(g.double().square().sum())
            norm=float(vector.norm());clip=min(1.,1./(norm+1e-6))
            results.append(dict(credit=mode,mean_factual_loss=float(box['loss'].detach())/len(rows),
                raw_parameter_gradient_norm=norm,native_global_clip_scale=clip,
                parameter_group_gradient_norms={k:v**.5 for k,v in groups.items()},
                stage_work=dict(forward_and_loss=forward_work,backward=backward_work,gradient_normalization=normalization_work)))
        base=flat(gradients['local']);choice=flat(gradients['state_choice']);joint=flat(gradients['state_clock'])
        for result in results:
            vector=flat(gradients[result['credit']]);denom=float(vector.norm()*base.norm())
            result['cosine_to_local_parameter_gradient']=None if denom==0 else float(vector.dot(base))/denom
            result['replacement_upstream_gradient_norm']=float((vector-base).norm())
        delta=joint-choice;group_delta={}
        for n,g in gradients['state_clock'].items():
            d=(g-gradients['state_choice'][n]).double()
            group_delta[group(n)]=group_delta.get(group(n),0)+float(d.square().sum())
        assert all(torch.equal(p.detach(),before[n]) for n,p in model.named_parameters())
        models.append(dict(native=name,result_sha256=B.N.sha(ROOT/name),checkpoint_sha256=B.N.sha(checkpoint),
            fitting_prefixes=len(rows),event=9,depth=0,head=0,credit_results=results,
            clock_likelihood_vs_native_timing_upstream_difference_norm=float(delta.norm()),
            clock_likelihood_vs_native_timing_upstream_difference_groups={k:v**.5 for k,v in group_delta.items()},
            optimizer_updates=0,weights_preserved=True,factual_forward_unchanged=True))
    for source in ['experiments/dvs_credit_gradient_balance_audit.py','experiments/dvs_route_content_audit.py']:
        sources[source]=B.N.sha(ROOT/source)
    sources.update(B.sources())
    result=dict(status='completed',args=vars(a),models=models,source_sha256=sources,
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Same selected weights/first16 fitting prefixes/draw314159/event9 head0 layer0 across local, full-write choice+native timing, and joint-clock correction. Sum loss then mean all gradients before clipping, no optimizer. Complete captured forward/replay/backward/normalization work. Single-draw norms/cosines and clip scales are diagnostics, not expected signal/noise, batch covariance, actual Adam updates, fitted normalization gains or causal attribution of regression.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
