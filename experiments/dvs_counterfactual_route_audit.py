"""Replay actual alternative commits and suffixes; audit scoped clock-choice credit."""
import argparse
from contextlib import contextmanager
import hashlib
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
import dvs_native_benchmark as N
import sleeping_machines.batched_addressed_fit as K

REFERENCE_ROUTE=K.BatchedTemporalRoute


class ConditionalRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx,scores,values,noise,forced,forced_write):
        rates=scores.double().exp();first,winner=(noise[None]/rates).min(-1)
        delivered=torch.where(forced>=0,forced,winner)
        committed=torch.where(forced_write>=0,forced_write,winner)
        selected=values.gather(2,delivered[...,None,None].expand(*winner.shape,1,values.shape[-1])).squeeze(2)
        ctx.save_for_backward(rates,first,delivered,values)
        return selected,.001+.010*first/(1+first),committed

    @staticmethod
    def backward(ctx,error_value,error_delay,unused):
        return (*REFERENCE_ROUTE.backward(ctx,error_value,error_delay,unused),None,None)


@contextmanager
def force_and_trace(site,clip,head,option,write_option=None):
    original=K.BatchedTemporalRoute;records=[]
    def apply(scores,values,noise):
        forced=torch.full(scores.shape[:2],-1,dtype=torch.long)
        if len(records)==site:forced[clip,head]=option
        forced_write=forced.clone()
        if len(records)==site and write_option is not None:forced_write[clip,head]=write_option
        chosen,delay,winner=ConditionalRoute.apply(scores,values,noise,forced,forced_write)
        actual=(noise[None]/scores.detach().double().exp()).min(-1).indices
        delivered=torch.where(forced>=0,forced,actual)
        record=dict(scores=scores.detach(),values=values.detach(),winner=delivered,
            time=(noise[None]/scores.detach().double().exp()).min(-1).values,error_value=None)
        if chosen.requires_grad:chosen.register_hook(lambda g:record.update(error_value=g.detach()))
        records.append(record);return chosen,delay,winner
    K.BatchedTemporalRoute=SimpleNamespace(apply=apply)
    try:yield records
    finally:K.BatchedTemporalRoute=original


def outcome(model,rows,site,clip,head,option,write_option=None,backward=True):
    model.zero_grad(set_to_none=True)
    with force_and_trace(site,clip,head,option,write_option) as records:
        logits,state,_=K.forward(model,rows,314159)
        losses=F.cross_entropy(logits,torch.tensor([r['target'] for r in rows]),reduction='none')
        if backward:losses.mean().backward()
    record=records[site];error=record['error_value']
    if error is None:policy=torch.zeros_like(record['scores'])
    else:
        ctx=SimpleNamespace(saved_tensors=(record['scores'].double().exp(),record['time'],record['winner'],record['values']))
        policy=REFERENCE_ROUTE.backward(ctx,error,None,None)[0]*len(rows)
    return float(losses[clip].detach()),record['scores'][clip,head],policy[clip,head],state,logits.detach()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--native',action='append',required=True);a=p.parse_args()
    start=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused plain tag required')
    models=[];sources={}
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        for f,h in parent['source_sha256'].items():
            if N.sha(ROOT/f)!=h:raise ValueError('Changed frozen model source')
        sources.update(parent['source_sha256']);args=argparse.Namespace(**parent['args'])
        _,dev,info=N.load(args);assert info==parent['data']
        model=N.make_model(args);cp=(ROOT/name).with_suffix('.progress.pt')
        ck=torch.load(cp,weights_only=False)
        assert ck['source_sha256']==parent['source_sha256'] and ck['data']==parent['data']
        model.load_state_dict(ck['best_state']);before={n:x.detach().clone() for n,x in model.named_parameters()}
        selected=dev[:4]
        baseline,_,_,bstats,z=outcome(model,selected,-1,0,0,0)
        torch.testing.assert_close(z.softmax(-1),torch.tensor(parent['final']['probabilities'][:4]),rtol=1e-4,atol=1e-5)
        samples=[]
        for event in (9,19):
            for depth in range(model.depth):
                site=event*model.depth+depth
                for head in range(model.heads):
                    for clip in range(len(selected)):
                        losses=[];teachers=[];states=[];entering=None
                        for option in range(model.pool):
                            loss,scores,teacher,state,_=outcome(model,selected,site,clip,head,option)
                            if entering is not None:torch.testing.assert_close(scores,entering,rtol=0,atol=0)
                            entering=scores;losses.append(loss);teachers.append(teacher);states.append(state)
                        pi=entering.double().softmax(-1);ls=torch.tensor(losses,dtype=torch.float64)
                        risk=(pi*ls).sum();exact=pi*(ls-risk)
                        local=(pi[:,None]*torch.stack(teachers).double()).sum(0)
                        norm=float(exact.norm());local_norm=float(local.norm())
                        cosine=None if norm*local_norm<1e-14 else float((exact*local).sum())/(norm*local_norm)
                        updated=(entering.double()-.01*local).softmax(-1)
                        delta=float(((updated-pi)*ls).sum())
                        # Both delivered content and selected persistent commit
                        # are changed by the actual winner mask in K.forward.
                        memory_gap=float((states[0]['memories'][depth][clip]-states[1]['memories'][depth][clip]).double().norm())
                        winner=int(bstats['winners'][site][clip,head]);alternative=1-winner
                        with torch.no_grad():
                            f10=outcome(model,selected,site,clip,head,alternative,write_option=winner,backward=False)[0]
                            f01=outcome(model,selected,site,clip,head,winner,write_option=alternative,backward=False)[0]
                        f00=losses[winner];f11=losses[alternative]
                        baseline_loss=float(F.cross_entropy(z[clip:clip+1],torch.tensor([selected[clip]['target']])))
                        assert abs(f00-baseline_loss)<1e-5
                        samples.append(dict(clip_index=clip,query_prediction_correct=bool(z[clip].argmax()==selected[clip]['target']),
                            event_index=event,depth=depth,head=head,probabilities=pi.tolist(),
                            actual_suffix_outcome_losses=losses,exact_conditional_choice_gradient=exact.tolist(),
                            mean_local_value_teacher_gradient=local.tolist(),choice_gradient_cosine=cosine,
                            conditional_risk_change_after_local_policy_step=delta,
                            final_persistent_memory_difference_l2=memory_gap,
                            delivered_value_only_effect=f10-f00,persistent_write_only_effect=f01-f00,
                            value_write_interaction=f11-f10-f01+f00,full_route_effect=f11-f00))
        eligible=[s for s in samples if s['choice_gradient_cosine'] is not None]
        opposed=sum(s['choice_gradient_cosine']<0 for s in eligible)
        assert all(torch.equal(x.detach(),before[n]) for n,x in model.named_parameters())
        models.append(dict(native=name,native_result_sha256=N.sha(ROOT/name),checkpoint_sha256=N.sha(cp),
            dev_prefixes=4,sites=len(samples),actual_outcomes_replayed=len(samples)*model.pool,
            diagnostic_hybrid_outcomes_replayed=2*len(samples),
            nonzero_comparable_policy_gradients=len(eligible),opposed_teacher_directions=opposed,
            opposed_fraction=None if not eligible else opposed/len(eligible),samples=samples,
            mean_absolute_value_only_effect=sum(abs(s['delivered_value_only_effect']) for s in samples)/len(samples),
            mean_absolute_write_only_effect=sum(abs(s['persistent_write_only_effect']) for s in samples)/len(samples),
            mean_absolute_value_write_interaction=sum(abs(s['value_write_interaction']) for s in samples)/len(samples),
            original_probabilities_reproduced=True,weights_preserved=True,optimizer_updates=0))
    for f in ['experiments/dvs_counterfactual_route_audit.py','sleeping_machines/batched_addressed_fit.py']:
        sources[f]=N.sha(ROOT/f)
    result=dict(status='completed',args=vars(a),models=models,source_sha256=sources,
        wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='First4 previously used dev prefixes, fixed events9/19, both layers/heads, all two candidate outcomes and two value/write diagnostic hybrids. Conditional first-arrival time is held fixed; forced realized payload and persistent memory commit alter the complete suffix with unchanged future random draws. Exact conditional categorical-choice component compared with winner-averaged current value teacher; time-density/pathwise credit is separate. Hybrid messages/commits are diagnostic, not legal routes. Absolute effects are not additive attribution percentages. Conditional suffix replay is not the full expected-history gradient, independent confirmation, a refit or a causal explanation of the observed quality gap.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
