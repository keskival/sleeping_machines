"""Fitting-only independent-draw branch exposure and parameter-credit audit."""
import argparse
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
import dvs_state_choice_credit_benchmark as B
from dvs_route_content_audit import group


class ScoreBlockedRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx,scores,values,noise,forced):
        rates=scores.double().exp();first,actual=(noise[None]/rates).min(-1)
        delivered=torch.where(forced>=0,forced,actual)
        selected=values.gather(2,delivered[...,None,None].expand(*actual.shape,1,values.shape[-1])).squeeze(2)
        ctx.save_for_backward(scores,values,delivered)
        return selected,.001+.010*first/(1+first),delivered

    @staticmethod
    def backward(ctx,error_value,error_delay,unused):
        scores,values,delivered=ctx.saved_tensors;vg=torch.zeros_like(values)
        if error_value is not None:
            vg.scatter_add_(2,delivered[...,None,None].expand(*delivered.shape,1,values.shape[-1]),error_value[:,:,None])
        # Explicit diagnostic: all race-score paths, including raw clock, blocked.
        return torch.zeros_like(scores),vg,None,None


@contextmanager
def trace(site,head=0,option=None,blocked=False):
    original=B.K.BatchedTemporalRoute;counter=0;records=[]
    def apply(scores,values,noise):
        nonlocal counter
        selected=counter==site;counter+=1
        forced=torch.full(scores.shape[:2],-1,dtype=torch.long,device=scores.device)
        if selected and option is not None:forced[:,head]=option
        out=ScoreBlockedRoute.apply(scores,values,noise,forced) if blocked else B.REFERENCE_ROUTE.apply(scores,values,noise)
        if selected:
            first,winner=(noise[None]/scores.detach().double().exp()).min(-1)
            rec=dict(scores=scores,values=values,first=first,winner=winner,delivered=out[2],error_value=None,error_delay=None)
            if out[0].requires_grad:out[0].register_hook(lambda g:rec.update(error_value=g.detach()))
            if out[1].requires_grad:out[1].register_hook(lambda g:rec.update(error_delay=g.detach()))
            records.append(rec)
        return out
    B.K.BatchedTemporalRoute=SimpleNamespace(apply=apply)
    try:
        yield records
        assert len(records)==1
    finally:B.K.BatchedTemporalRoute=original


def grads(loss,model,retain=False):
    raw=torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True,retain_graph=retain)
    return {n:torch.zeros_like(p) if g is None else g.detach() for (n,p),g in zip(model.named_parameters(),raw)}


def flat(g,names=None):
    return torch.cat([v.double().flatten() for n,v in g.items() if names is None or n in names])


def cosine(a,b):
    den=float(a.norm()*b.norm());return None if den<1e-20 else float(a.dot(b))/den


def stats(g):
    groups={}
    for n,v in g.items():groups[group(n)]=groups.get(group(n),0.)+float(v.double().square().sum())
    return {k:v**.5 for k,v in groups.items()}


def add(a,b):return {n:a[n]+b[n] for n in a}
def subtract(a,b):return {n:a[n]-b[n] for n in a}


def cell(model,rows,seed,event,depth,work):
    site=event*model.depth+depth;target=torch.tensor([r['target'] for r in rows]);box={};params=tuple(model.parameters())
    def capture(label,fn):
        w=B.N.capture(fn)
        if not w['formula_coverage_complete']:raise AssertionError((label,w['unsupported_floating_operators']))
        work.append(dict(stage=label,arithmetic_flops=w['arithmetic_flops'],special_function_evaluations=w['special_function_evaluations']))
    def factual():
        with trace(site) as records:z,state,_=B.K.forward(model,rows,seed)
        box.update(logits=z,state=state,record=records[0],loss=F.cross_entropy(z,target,reduction='sum'))
    capture('native_forward',factual)
    capture('native_backward',lambda:box.update(native=grads(box['loss'],model,True)))
    rec=box['record'];pi=rec['scores'].detach()[:,0].double().softmax(-1);winner=rec['winner'][:,0]
    branches=[];records=[];losses=[]
    for option in range(2):
        def branch():
            with trace(site,option=option,blocked=True) as rs:z,st,_=B.K.forward(model,rows,seed)
            branches.append((z,st));records.append(rs[0]);losses.append(F.cross_entropy(z,target,reduction='none'))
        capture('legal_score_blocked_forward',branch)
        for key in ('scores','values','first'):torch.testing.assert_close(records[-1][key],rec[key],rtol=0,atol=0)
        assert bool((records[-1]['delivered'][:,0]==option).all())
        assert torch.equal(records[-1]['delivered'][:,1],rec['winner'][:,1])
    factual_logits=torch.where(winner[:,None]==0,branches[0][0],branches[1][0])
    torch.testing.assert_close(factual_logits,box['logits'],rtol=0,atol=0)
    # Branch weights are detached; score derivatives are added separately.
    fitting_loss=sum((ls*(winner==i)).sum() for i,ls in enumerate(losses))
    expected_loss=sum((ls*pi[:,i]).sum() for i,ls in enumerate(losses))
    capture('sampled_branch_backward',lambda:box.update(sampled=grads(fitting_loss,model,True)))
    capture('enumerated_branch_backward',lambda:box.update(enumerated=grads(expected_loss,model)))
    outcomes=torch.stack([ls.detach() for ls in losses],-1).double()
    exact=pi*(outcomes-(pi*outcomes).sum(-1,keepdim=True))
    ctx=SimpleNamespace(saved_tensors=(rec['scores'].double().exp(),rec['first'],rec['winner'],rec['values']))
    local=B.REFERENCE_ROUTE.backward(ctx,rec['error_value'],None,None)[0][:,0].detach().double()
    def correction():
        credit=torch.zeros_like(rec['scores']);credit[:,0]=(exact-local).to(credit.dtype)
        raw=torch.autograd.grad(rec['scores'],params,grad_outputs=credit,allow_unused=True,retain_graph=True)
        box['correction']={n:torch.zeros_like(p) if g is None else g.detach() for (n,p),g in zip(model.named_parameters(),raw)}
    capture('actual_choice_residual_vjp',correction)
    sampled=box['sampled'];enum=box['enumerated'];native=box['native'];path=subtract(native,sampled)
    corrected=add(native,box['correction']);cv=flat(sampled);rv=flat(path);ev=flat(enum);nv=flat(native)
    grouped={}
    for g in stats(native):
        names={n for n in native if group(n)==g};c=flat(sampled,names);r=flat(path,names)
        grouped[g]=dict(sampled_branch_norm=float(c.norm())/len(rows),route_clock_difference_norm=float(r.norm())/len(rows),
            cosine=cosine(c,r),cross_inner_product=float(c.dot(r))/len(rows)**2,
            enumerated_branch_norm=float(flat(enum,names).norm())/len(rows))
    timing=B.REFERENCE_ROUTE.backward(ctx,None,rec['error_delay'],None)[0][:,0].detach().double()
    row=dict(draw_seed=seed,event=event,depth=depth,head=0,prefixes=len(rows),
        probabilities=pi.tolist(),winner=winner.tolist(),outcome_losses=outcomes.tolist(),
        local_choice_score_gradient=local.tolist(),exact_conditional_choice_score_gradient=exact.tolist(),
        choice_score_cosine=cosine(local.flatten(),exact.flatten()),native_timing_score_norm=float(timing.norm())/len(rows),
        sampled_branch_norm=float(cv.norm())/len(rows),enumerated_branch_norm=float(ev.norm())/len(rows),
        native_gradient_norm=float(nv.norm())/len(rows),corrected_gradient_norm=float(flat(corrected).norm())/len(rows),
        sampled_vs_enumerated_branch_cosine=cosine(cv,ev),sampled_vs_route_clock_difference_cosine=cosine(cv,rv),
        groups=grouped,factual_logits_preserved=True,alternative_entering_candidates_and_times_preserved=True)
    vectors={k:flat(v)/len(rows) for k,v in [('sampled',sampled),('enumerated',enum),('native',native),('corrected',corrected)]}
    return row,vectors


def run(a):
    started=time.perf_counter();torch.set_num_threads(1);out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique tag required')
    result_models=[];sources={**B.sources(),'experiments/dvs_branch_exposure_audit.py':B.N.sha(Path(__file__)),
        'experiments/dvs_route_content_audit.py':B.N.sha(ROOT/'experiments/dvs_route_content_audit.py')}
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        for n,h in parent['source_sha256'].items():assert B.N.sha(ROOT/n)==h,('Changed source',n)
        config=argparse.Namespace(**parent['args']);assert config.pool==2 and config.depth==2
        fitting,_,metadata=B.N.load(config);assert metadata==parent['data'];rows=fitting[:2]
        model=B.N.make_model(config);checkpoint=(ROOT/name).with_suffix('.progress.pt');saved=torch.load(checkpoint,weights_only=False)
        assert saved['source_sha256']==parent['source_sha256'];model.load_state_dict(saved['best_state'])
        before={n:p.detach().clone() for n,p in model.named_parameters()};cells=[];all_vectors={};work=[]
        for event in (9,19):
            for depth in range(model.depth):
                histories={k:[] for k in ('sampled','enumerated','native','corrected')}
                for draw in range(a.draws):
                    row,vectors=cell(model,rows,710003+997*draw,event,depth,work);cells.append(row)
                    for k,v in vectors.items():histories[k].append(v)
                for k,vs in histories.items():
                    mat=torch.stack(vs);mean=mat.mean(0);noise=float((mat-mean).square().sum())/(a.draws-1)
                    all_vectors[f'event{event}_depth{depth}_{k}']=dict(mean_norm=float(mean.norm()),
                        trace_sample_covariance=noise,second_moment=float(mat.square().sum(-1).mean()),
                        sample_count=a.draws,scope='Independent whole-history draws; entering prefixes/times vary. Not conditional current-node variance or a population convergence estimate.')
        assert all(torch.equal(p.detach(),before[n]) for n,p in model.named_parameters())
        comparable=[r for r in cells if r['choice_score_cosine'] is not None]
        result_models.append(dict(native=name,result_sha256=B.N.sha(ROOT/name),checkpoint_sha256=B.N.sha(checkpoint),
            fitting_prefixes=2,draws=a.draws,cells=cells,gradient_moments=all_vectors,stage_work=work,
            choice_score_opposed=sum(r['choice_score_cosine']<0 for r in comparable),choice_score_comparable=len(comparable),
            global_route_clock_opposed=sum((r['sampled_vs_route_clock_difference_cosine'] or 0)<0 for r in cells),
            weights_preserved=True,optimizer_updates=0,formula_coverage_complete=True))
    result=dict(status='completed',args=vars(a),models=result_models,source_sha256=sources,
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='First2 FITTING prefixes/model, events9/19, layers0/1, head0; independent whole-history draws. Native parameter gradients vs diagnostic ALL-race-score-blocked branch derivatives, exact selected-node categorical utility residual VJP, and detached-probability averaging of BOTH actual-write branches with all race-score paths blocked. Blocks raw clock score sensitivities and other route teachers deliberately; not a complete expected-risk gradient, a causal diagnosis of seed failure, fitted alternation, or practical advantage. All legal forwards/backwards/VJPs paid; reporting reductions and comparisons are outside captured model stages. No optimizer or development-label selection.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--native',action='append',required=True)
    p.add_argument('--draws',type=int,default=4);a=p.parse_args()
    if not 2<=a.draws<=8:raise ValueError('Bounded independent diagnostic draws required')
    run(a)
