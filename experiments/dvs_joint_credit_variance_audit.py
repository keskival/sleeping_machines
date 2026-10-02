"""Frozen actual route/time suffix quadrature; score-space variance diagnosis."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import sleeping_machines.batched_addressed_fit as K


@contextmanager
def intervention(site,head,option=None,times=None):
    original=K.BatchedTemporalRoute;record={};counter=0
    def apply(scores,values,noise):
        nonlocal counter
        selected=counter==site;counter+=1
        first,winner=(noise[None]/scores.double().exp()).min(-1)
        if selected:
            record.update(scores=scores,values=values,first=first,winner=winner)
            if option is not None:
                first=first.clone();winner=winner.clone()
                first[:,head]=times;winner[:,head]=option
        chosen=values.gather(2,winner[...,None,None].expand(*winner.shape,1,values.shape[-1])).squeeze(2)
        return chosen,.001+.010*first/(1+first),winner
    K.BatchedTemporalRoute=SimpleNamespace(apply=apply)
    try:
        yield record
        if not record:raise ValueError('Missing selected site')
    finally:K.BatchedTemporalRoute=original


def moments(scores,losses,nodes,weights,baseline):
    pi=scores.double().softmax(-1);u=torch.tensor(nodes,dtype=torch.float64)
    w=torch.tensor(weights,dtype=torch.float64)
    risk=(pi[None]*losses).sum(-1)
    choice=pi[None]*(losses-risk[...,None])
    k=pi[None]*(1-u[:,None,None])
    h0=choice+k*risk[...,None]
    mean=(w[:,None,None]*h0).sum(0)
    optimal=(w[:,None]*(k*h0).sum(-1)).sum(0)/(w[:,None]*k.square().sum(-1)).sum(0)
    def variance(b):
        return (w[:,None]*(h0-k*b[None,:,None]-mean[None]).square().sum(-1)).sum(0)
    actual=variance(baseline);minimum=variance(optimal)
    assert (minimum<=actual+1e-12).all()
    torch.testing.assert_close((w[:,None,None]*k).sum(0),torch.zeros_like(pi),rtol=0,atol=1e-12)
    return dict(prefix_baseline=baseline.tolist(),oracle_minimum_variance_baseline=optimal.tolist(),
        joint_mean_score_gradient=mean.tolist(),choice_mean_score_gradient=(w[:,None,None]*choice).sum(0).tolist(),
        score_space_variance_prefix_baseline=actual.tolist(),score_space_variance_oracle_baseline=minimum.tolist(),
        choice_rms=float((w[:,None]*choice.square().sum(-1)).sum(0).mean().sqrt()),
        common_clock_rms=float((w[:,None]*(k*(risk-baseline[None])[...,None]).square().sum(-1)).sum(0).mean().sqrt()),
        outcome_time_range_by_clip_and_option=(losses.max(0).values-losses.min(0).values).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--native',action='append',required=True);a=p.parse_args();start=time.perf_counter()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    torch.set_num_threads(1);sources={};models=[]
    # The time-independent outcome identity is exact, unlike nonlinear quadrature.
    nodes,weights=np.polynomial.laguerre.laggauss(16)
    constant=torch.tensor([.7,1.3],dtype=torch.float64)[None,None].expand(16,1,2)
    witness=moments(torch.zeros(1,2),constant,nodes,weights,torch.tensor([.2]))
    assert witness['score_space_variance_oracle_baseline'][0]<1e-24
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        for source,digest in parent['source_sha256'].items():
            if N.sha(ROOT/source)!=digest:raise ValueError('Changed frozen source '+source)
        sources.update(parent['source_sha256']);args=argparse.Namespace(**parent['args'])
        fit,_,metadata=N.load(args);assert metadata==parent['data'];rows=fit[:4]
        model=N.make_model(args);checkpoint=(ROOT/name).with_suffix('.progress.pt')
        saved=torch.load(checkpoint,weights_only=False);assert saved['source_sha256']==parent['source_sha256']
        model.load_state_dict(saved['best_state']);before={n:x.detach().clone() for n,x in model.named_parameters()}
        targets=torch.tensor([r['target'] for r in rows]);probes=[]
        with torch.no_grad():
            for depth in range(model.depth):
                for head in range(model.heads):
                    site=9*model.depth+depth
                    with intervention(site,head) as entering:K.forward(model,rows,314159)
                    scores=entering['scores'][:,head].double();total=scores.exp().sum(-1)
                    baseline=F.cross_entropy(model.head(entering['values'].mean(2).reshape(len(rows),-1)),targets,reduction='none').double()
                    resolutions=[]
                    for count in (8,16):
                        nodes,weights=np.polynomial.laguerre.laggauss(count);outcomes=[]
                        for node in nodes:
                            times=node/total;both=[]
                            for option in range(2):
                                with intervention(site,head,option,times) as replay:
                                    z,_,_=K.forward(model,rows,314159)
                                torch.testing.assert_close(replay['scores'],entering['scores'],rtol=0,atol=0)
                                both.append(F.cross_entropy(z,targets,reduction='none').double())
                            outcomes.append(torch.stack(both,-1))
                        losses=torch.stack(outcomes)
                        resolutions.append(dict(nodes=count,**moments(scores,losses,nodes,weights,baseline),
                            raw_times=(torch.tensor(nodes)[:,None]/total[None]).tolist(),outcome_losses=losses.tolist()))
                    coarse=torch.tensor(resolutions[0]['joint_mean_score_gradient']);fine=torch.tensor(resolutions[1]['joint_mean_score_gradient'])
                    probes.append(dict(event=9,depth=depth,head=head,resolutions=resolutions,
                        mean_score_gradient_8_16_l2_difference=float((coarse-fine).norm())))
        assert all(torch.equal(x.detach(),before[n]) for n,x in model.named_parameters())
        high=[x['resolutions'][-1] for x in probes]
        actual=sum(sum(x['score_space_variance_prefix_baseline']) for x in high)
        oracle=sum(sum(x['score_space_variance_oracle_baseline']) for x in high)
        models.append(dict(native=name,result_sha256=N.sha(ROOT/name),checkpoint_sha256=N.sha(checkpoint),
            fitting_prefixes=4,probes=probes,total_legal_suffix_replays=2*(8+16)*len(probes),
            mean_choice_rms=sum(x['choice_rms'] for x in high)/len(high),
            mean_common_clock_rms=sum(x['common_clock_rms'] for x in high)/len(high),
            summed_prefix_baseline_variance=actual,summed_oracle_variance=oracle,
            estimated_oracle_removable_variance_fraction=None if actual==0 else 1-oracle/actual,
            optimizer_updates=0,weights_preserved=True))
    for source in ['experiments/dvs_joint_credit_variance_audit.py','experiments/theory/82_joint_credit_variance_and_failure.md']:
        sources[source]=N.sha(ROOT/source)
    result=dict(status='completed',args=vars(a),models=models,constant_outcome_baseline_identity_passed=True,
        source_sha256=sources,wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='First4 previously used fitting prefixes, event9/both layers/heads, fixed future draws; actual legal payload/write/suffix replay at8/16 exponential-time quadrature nodes. Nonlinear/jumping-time quadrature is an estimate and resolutions can disagree. Oracle baseline pays all replay and minimizes score-space variance, not parameter/Adam noise. No observed batch covariance, optimizer, fit-quality benefit or causal attribution of regression.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
