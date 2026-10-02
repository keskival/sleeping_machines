"""Frozen all-site race support and exact race-scaled reception contracts."""
import argparse
import copy
import json
from pathlib import Path
import platform
import resource
import sys
import time
import types
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
from dvs_native_window_intervention import equal_state
from sleeping_machines.race_window import bounded_delay,residual_probability
from sleeping_machines.relative_race_window import relative_deadline,expected_relative_receivers,maximum_added_physical_delay


def tensor(x):return torch.tensor(x,dtype=torch.float64)


def numerical_contracts():
    scores=tensor(np.log([.3,1.1,2.8])).requires_grad_();extension=tensor(1.7).requires_grad_()
    noise=tensor([.8,.2,1.5]);raw=noise/scores.exp();first=raw.min()
    for c in (0.,1.,4.):
        mask=raw<=(1+c)*first
        for shift in (-4.,-.7,2.5):
            changed=noise/(scores+shift).exp()
            assert torch.equal(mask,changed<=(1+c)*changed.min())
            torch.testing.assert_close(expected_relative_receivers(scores,tensor(c)),
                expected_relative_receivers(scores+shift,tensor(c)),rtol=1e-13,atol=1e-14)
    # Independent integration over the common first Exp(1) draw and all winners.
    nodes,weights=np.polynomial.laguerre.laggauss(128);nodes=tensor(nodes);weights=tensor(weights)
    for s in (scores.detach(),tensor([0.,0.,0.]),tensor([-12.,12.,0.])):
        pi=s.softmax(0);T=nodes/s.exp().sum()
        for c in (.0,.3,1.,4.):
            explicit=tensor(0.)
            for winner in range(3):
                conditional=1+sum(-torch.expm1(-s[j].exp()*c*T) for j in range(3) if j!=winner)
                explicit=explicit+pi[winner]*(weights*conditional).sum()
            torch.testing.assert_close(explicit,expected_relative_receivers(s,tensor(c)),rtol=1e-10,atol=1e-11)
    assert torch.autograd.gradcheck(expected_relative_receivers,(scores,extension),eps=1e-6,atol=2e-7,rtol=2e-6)
    zero=tensor(0.).requires_grad_();count=expected_relative_receivers(scores,zero)
    torch.testing.assert_close(count,tensor(1.),rtol=0,atol=0)
    slope=torch.autograd.grad(count,zero)[0]
    torch.testing.assert_close(slope,1-scores.softmax(0).square().sum(),rtol=1e-12,atol=1e-13)
    for c in (0.,1.,4.,16.):
        times=torch.logspace(-6,6,400,dtype=torch.float64)
        added=relative_deadline(times,tensor(c))-bounded_delay(times)
        bound=maximum_added_physical_delay(tensor(c))
        assert bool((added<=bound+1e-14).all()) and bool((relative_deadline(times,tensor(c))<.011).all())
        peak=tensor(1/np.sqrt(1+c))
        torch.testing.assert_close(relative_deadline(peak,tensor(c))-bounded_delay(peak),bound,rtol=1e-12,atol=1e-14)
    # Common shift cancels membership credit, but real physical timing changes.
    T=tensor(.9)/scores.exp().sum();q=-torch.expm1(-scores.exp()*extension*T)
    membership=torch.autograd.grad(q.sum(),scores,retain_graph=True)[0]
    timing=torch.autograd.grad(relative_deadline(T,extension),scores)[0]
    torch.testing.assert_close(membership.sum(),tensor(0.),rtol=0,atol=1e-13)
    assert abs(float(timing.sum().detach()))>1e-5
    return dict(contracts_passed=6,scale_invariant_membership_and_count=True,
        quadrature_exact_count_law=True,every_count_score_extension_finite_difference=True,
        zero_extension_nesting_and_birth_derivative=True,tight_physical_delay_bound=True,
        common_shift_zero_membership_credit_nonzero_physical_clock_credit=True,
        maximum_added_physical_wait_ms={str(c):float(maximum_added_physical_delay(tensor(c))*1000) for c in (1.,4.)})


def collector(model,buffer):
    def race(self,scores,values=None):
        if values is not None:raise ValueError('Frozen winner-only evaluation collector required')
        rates=scores.to(torch.float64).exp();raw=torch.empty_like(rates).exponential_()/rates
        first,winner=raw.min(0)
        buffer.append((scores.detach().double().numpy().copy(),raw.numpy().copy(),int(winner)))
        return None,bounded_delay(first),winner
    model.race=types.MethodType(race,model)


def profile_summary(scores,raw,winner,sites):
    n,pool=scores.shape;s=tensor(scores);a=tensor(raw);pi=s.softmax(-1);first=a.min(-1).values
    physical=bounded_delay(a);first_physical=bounded_delay(first)
    actual={};expected={};waiting={};clock_cutoff={}
    loser_mask=torch.arange(pool)[None]!=torch.as_tensor(winner)[:,None]
    for h in (.001,.003):
        name=f'fixed_{h*1000:.0f}ms';actual[name]=(physical<=first_physical[:,None]+h).sum(-1).double()
        probability=residual_probability(s,first[:,None],tensor(h))
        expected[name]=1+(probability*loser_mask).sum(-1)
        waiting[name]=torch.full_like(first,h)
    for c in (1.,4.):
        name=f'relative_c{c:.0f}';actual[name]=(a<=(1+c)*first[:,None]).sum(-1).double()
        expected[name]=expected_relative_receivers(s,tensor(c))
        waiting[name]=relative_deadline(first,tensor(c))-first_physical
        clock_cutoff[name]=(1+c)*first
    entropy=-(pi*pi.log()).sum(-1)/np.log(pool)
    def summary(mask):
        def mean(x):return float(x[mask].mean())
        def quantiles(x):return [float(v) for v in torch.quantile(x[mask],tensor([.05,.5,.95]))]
        return dict(races=int(mask.sum()),mean_normalized_entropy=mean(entropy),
            top_probability_99_fraction=mean((pi.max(-1).values>=.99).double()),
            mean_effective_choices=mean(1/pi.square().sum(-1)),
            clamped_score_fraction=float((s[mask].abs()>=12.).double().mean()),
            first_physical_delay_ms_q05_q50_q95=quantiles(first_physical*1000),
            physical_loser_gap_ms_q05_q50_q95=quantiles((physical.max(-1).values-first_physical)*1000),
            receivers={name:dict(actual_mean=mean(actual[name]),expected_mean=mean(expected[name]),
                actual_extra_fraction=mean((actual[name]>1).double()),
                mean_extra_wait_ms=mean(waiting[name]*1000)) for name in actual})
    ids=torch.as_tensor(sites);allmask=torch.ones(n,dtype=torch.bool)
    grouped={f'layer{d}_head{h}':summary((ids[:,1]==d)&(ids[:,2]==h)) for d in range(2) for h in range(2)}
    selected=summary((ids[:,0]==19)&(ids[:,1]==0)&(ids[:,2]==0))
    per_site=[dict(event=e,depth=d,head=h,**summary((ids[:,0]==e)&(ids[:,1]==d)&(ids[:,2]==h)))
        for e in range(21) for d in range(2) for h in range(2)]
    return dict(all_sites=summary(allmask),by_layer_head=grouped,prior_intervention_site=selected,per_site=per_site)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--native',action='append',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    begin=time.perf_counter();torch.set_num_threads(1)
    # Contracts intentionally need gradients despite the frozen profile wrapper.
    with torch.enable_grad():checks=numerical_contracts()
    indices=np.linspace(256,983,16,dtype=int).tolist();drawseeds=[314159+1009*k for k in range(8)]
    rows=[];arrays={};parents=[];source={};forward_contracts=0
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        config=argparse.Namespace(**parent['args'])
        assert (config.fit,config.epochs,config.pool,config.depth,config.heads)==(256,4,2,2,2)
        for filename,digest in parent['source_sha256'].items():assert N.sha(ROOT/filename)==digest
        source.update(parent['source_sha256']);expanded=copy.copy(config);expanded.fit=984
        fit,_,metadata=N.load(expanded);assert metadata==parent['data'];selected=[fit[i] for i in indices]
        assert all(len(r['events'])==21 for r in selected)
        cp=(ROOT/name).with_suffix('.progress.pt');saved=torch.load(cp,weights_only=False)
        assert saved['cursor']['epoch']==5 and saved['source_sha256']==parent['source_sha256']
        parents.append(dict(result=name,result_sha256=N.sha(ROOT/name),checkpoint_sha256=N.sha(cp),
            original_core_fit_gflops=parent['work']['whole_fit_unit_special_flops_estimate']/1e9))
        initial=N.make_model(config,fast=False).state_dict()
        for encoder,weights in [('initial',initial),('fixed_pass4',saved['online_model'])]:
            model=N.make_model(config,fast=False);model.load_state_dict(weights)
            baseline,base_state=N.predict(model,selected[0],drawseeds[0],False)
            before={n:v.detach().clone() for n,v in model.state_dict().items()};buffer=[];collector(model,buffer)
            instrumented,new_state=N.predict(model,selected[0],drawseeds[0],False)
            assert torch.equal(baseline,instrumented);equal_state(base_state,new_state);forward_contracts+=1
            scores=[];raw=[];winners=[];sites=[];subjects=[];seeds=[];counts=dict(prefixes=0,events=0,key_scores=0,selected_updates=0)
            for drawseed in drawseeds:
                for row in selected:
                    buffer.clear();_,state=N.predict(model,row,drawseed,False)
                    assert len(buffer)==84
                    for r,(s,t,w) in enumerate(buffer):
                        scores.append(s);raw.append(t);winners.append(w)
                        sites.append((r//4,(r%4)//2,r%2));subjects.append(row['index']);seeds.append(drawseed)
                    counts['prefixes']+=1;counts['events']+=state.events
                    counts['key_scores']+=state.candidate_scores;counts['selected_updates']+=state.selected_updates
            assert all(torch.equal(before[n],v.detach()) for n,v in model.state_dict().items())
            result=profile_summary(np.array(scores),np.array(raw),winners,sites)
            key=f's{config.seed}_{encoder}'
            for suffix,values in [('scores',scores),('raw_arrivals',raw),('winners',winners),('site',sites),('fit_index',subjects),('noise_seed',seeds)]:
                arrays[key+'_'+suffix]=np.asarray(values)
            rows.append(dict(seed=config.seed,encoder=encoder,observed_work=counts,**result))
            print(json.dumps(dict(seed=config.seed,encoder=encoder,all_sites=result['all_sites'],selected_site=result['prior_intervention_site'])),flush=True)
    artifact=out.with_suffix('.profile.npz');np.savez_compressed(artifact,**arrays)
    own=['sleeping_machines/relative_race_window.py','sleeping_machines/race_window.py',
        'experiments/dvs_race_support_profile.py','experiments/dvs_native_window_intervention.py',
        'experiments/theory/99_race_scaled_reception_and_support.md']
    source.update({name:N.sha(ROOT/name) for name in own})
    result=dict(status='completed',args=vars(a),parents=parents,rows=rows,numerical_contracts=checks,
        original_logit_state_rng_contracts_passed=forward_contracts,all_weights_bitwise_preserved=True,
        indices=indices,noise_seeds=drawseeds,development_evaluations=0,optimizer_steps=0,
        profile_artifact=str(artifact.relative_to(ROOT)),profile_artifact_sha256=N.sha(artifact),
        source_sha256=source,whole_audit_flops=None,wall_s=time.perf_counter()-begin,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(host=platform.node(),device='CPU',torch_threads=1),
        scope='All84 actual native winner races,16 unused FIT prefixes times8 whole-history noise draws, '
            'two fixed-pass producers and initial reservoirs. Collector reproduces logits/state/RNG without any '
            'extra hypothetical value delivery or memory write. Relative support is not utility/quality; '
            'no training admission, optimizer, DEV/test evaluation, integrated relative-window learner or '
            'natural-silence result. Exact observed key/write counts and prior fits retained; '
            'full FLOPs/traffic/energy unknown, not zero.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
