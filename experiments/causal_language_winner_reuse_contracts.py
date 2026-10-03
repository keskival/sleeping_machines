"""Every-gradient equivalence for removing duplicated factual-winner shadows."""
import argparse
import copy
import io
import json
from pathlib import Path
import resource
import sys
import time
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import causal_language_replay_contracts as C
import causal_language_replay_helpers_rng as Old
import causal_language_replay_winner_reuse as New
import native_language_replay_benchmark as D
from causal_language_replay_winner_accumulator import WinnerReuseAccumulator
from native_language_replay_driver_contracts import exact
from race_language_screen import capture


def make(family,pool):
    a=argparse.Namespace(seed=108,payload=4,depth=8,pool=pool,heads=2,receiver_sharing=family)
    m=D.make(a).double()
    with torch.no_grad():
        for p in m.parameters():p.add_(torch.randn_like(p)*.03)
    return m


def sources():
    names=['experiments/causal_language_winner_reuse_contracts.py','experiments/causal_language_replay_winner_reuse.py',
        'experiments/causal_language_replay_winner_accumulator.py','sleeping_machines/causal_language_shadow_winner_reuse.py',
        'experiments/theory/115_factual_winner_return_reuse.md','experiments/native_language_replay_driver_contracts.py']
    return {**D.sources(),**{n:N.sha(ROOT/n) for n in names}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    torch.set_num_threads(1);begin=time.perf_counter();checks=[];rows=[];audits={}
    for family in ('private','depth'):
        for pool in (1,2,4):
            count=3 if pool==2 else 2;tokens=torch.tensor([3,1,4])[:count];targets=torch.tensor([1,4,2])[:count]
            m=make(family,pool);st=C.entering(m);seed=115329
            observed=Old.row(m,tokens,st);scores=[];winners=[];caller=torch.get_rng_state().clone()
            oz,os,orng=Old.batched_chunks(m,[observed],seed,[st])
            F.cross_entropy(oz[0],targets,reduction='sum').backward();og=C.grads(m);m.zero_grad(set_to_none=True)
            nz,ns,nrng=New.batched_chunks(m,[observed],seed,[st],record=scores,winners=winners)
            F.cross_entropy(nz[0],targets,reduction='sum').backward()
            C.close(nz,oz,True);C.state_close(ns[0],os[0],True);C.close(nrng,orng,True)
            C.grads_close(C.grads(m),og,True);C.close(torch.get_rng_state(),caller,True)
            checks.append(f'{family}/U{pool}: winner-recording kernel nests factual logits/all state/end RNG/every pathwise gradient bitwise')
            with torch.no_grad():
                forced=[(r,int(w[0])) for r,w in enumerate(winners)]
                same,ss,end=Old.batched_chunks(m,[observed]*len(forced),seed,[st]*len(forced),forced)
                for j in range(len(forced)):
                    C.close(same[j],nz[0]);C.state_close(ss[j],ns[0])
                C.close(end,nrng,True)
            checks.append(f'{family}/U{pool}: EVERY factual-winner first-time forced replay matches all factual predictions/private state')
            m.zero_grad(set_to_none=True);seq,z,ss,sa=Old.sequential_objective(m,tokens,targets,st,seed)
            seq.backward();sg=C.grads(m);m.zero_grad(set_to_none=True)
            old,zo,so,oa=Old.batched_objective(m,tokens,targets,st,seed);old.backward();og=C.grads(m)
            m.zero_grad(set_to_none=True);new,zn,sn,na=New.batched_objective(m,tokens,targets,st,seed);new.backward()
            C.close(new,old);C.close(new,seq);C.grads_close(C.grads(m),og);C.grads_close(C.grads(m),sg)
            C.close(zn,zo,True);C.state_close(sn,so,True);C.close(na['factual_end_rng'],oa['factual_end_rng'],True)
            assert na['shadow_lanes']==count*16*(pool-1) and na['shadow_events']==na['shadow_lanes']*count
            assert na['reused_winner_returns']==count*16
            checks.append(f'{family}/U{pool}: EVERY full-return parameter gradient/objective matches old batched AND independent sequential enumeration')
            m.zero_grad(set_to_none=True)
            _,changed,changed_state,act=New.batched_objective(m,tokens,(targets+7)%27,st,seed)
            C.close(changed,zn,True);C.state_close(changed_state,sn,True)
            later=tokens.clone();later[-1]=5
            _,future,_,_=New.batched_objective(m,later,targets,st,seed)
            C.close(future[:-1],zn[:-1],True);C.close(torch.get_rng_state(),caller,True)
            checks.append(f'{family}/U{pool}: labels never enter predictions/state and future observed tokens never enter earlier predictions; caller RNG unchanged')
            rows.append(dict(family=family,pool=pool,depth=8,heads=2,payload=4,targets=count,
                parameters=sum(p.numel() for p in m.parameters()),old_shadow_lanes=oa['shadow_lanes'],
                new_shadow_lanes=na['shadow_lanes'],old_shadow_events=oa['shadow_events'],new_shadow_events=na['shadow_events']))
        # Exact recovery of two pending microtargets and the next one-target partial update.
        m=make(family,2);st=C.entering(m);opt=torch.optim.Adam(m.parameters(),lr=.002)
        learner=WinnerReuseAccumulator(m,opt,.002,4);torch.manual_seed(115329)
        _,st,_=learner.accumulate(torch.tensor([3,1]),torch.tensor([1,4]),st)
        buf=io.BytesIO();torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),state=st,grads=C.grads(m),
            counters=learner.counters(),rng=torch.get_rng_state()),buf);buf.seek(0);saved=torch.load(buf,weights_only=False)
        other=make(family,2);other.load_state_dict(saved['model']);oopt=torch.optim.Adam(other.parameters(),lr=.002)
        oopt.load_state_dict(saved['optimizer']);recovered=WinnerReuseAccumulator(other,oopt,.002,4)
        recovered.restore_counters(saved['counters'])
        for n,par in other.named_parameters():par.grad=saved['grads'].get(n)
        torch.set_rng_state(saved['rng']);loss,st,z=learner.accumulate(torch.tensor([4]),torch.tensor([2]),st)
        end=torch.get_rng_state().clone();torch.set_rng_state(saved['rng'])
        loss2,ss,zz=recovered.accumulate(torch.tensor([4]),torch.tensor([2]),saved['state'])
        assert loss==loss2;C.close(z,zz,True);C.state_close(st,ss,True);C.grads_close(C.grads(m),C.grads(other),True)
        C.close(torch.get_rng_state(),end,True);learner.update();recovered.update()
        exact(m.state_dict(),other.state_dict());exact(opt.state_dict(),oopt.state_dict());exact(learner.counters(),recovered.counters())
        checks.append(f'{family}: pending gradient/private state/Adam/RNG/replay counter and next partial warmup update recover bitwise')
        trial=make(family,2);ts=C.entering(trial);to=torch.optim.Adam(trial.parameters(),lr=.002)
        learn=WinnerReuseAccumulator(trial,to,.002,4)
        stages=dict(full_credit_accumulate=capture(lambda:learn.accumulate(torch.tensor([3,1]),torch.tensor([1,4]),ts)),
                    normalize_clip_Adam=capture(learn.update))
        assert all(s['formula_coverage_complete'] for s in stages.values());audits[family]=stages
        checks.append(f'{family}: all optimized shadows/factual backward and actual target-normalize/clip/Adam operations are charged')
    r=dict(status='completed',args=vars(a),contracts_passed=len(checks),contracts=checks,cases=rows,
        source_sha256=sources(),work_audits=audits,wall_s=time.perf_counter()-begin,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Exact-arithmetic winner-return reuse equivalence, double integrated every-gradient contracts and test optimizer recovery; '
            'no trained text8 quality, no whole-stream exact-risk theorem, no changed active AWS driver.')
    out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r[k] for k in ('status','contracts_passed','wall_s','max_rss_kb')}))


if __name__=='__main__':main()
