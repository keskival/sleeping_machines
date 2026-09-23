#!/usr/bin/env python3
"""Sleeping Machines v3 benchmark.

Tests event-oriented computation AND event-oriented learning.

The central learning experiment is deliberately K-way (K>=3). With only two
alternatives, lowering A is equivalent to raising B after normalization. With
three alternatives, lowering A also benefits C, so winner-only learning cannot
in general implement target-specific redistribution.

The CPU is only a functional simulator. "Energy" is a transparent dynamic-work
proxy based on event/state transitions, not a claim about CPU joules.

Dependencies: numpy, torch, matplotlib

Example:
    python sleeping_machines_event_benchmark_v3.py --seeds 10
"""
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_dataset(n, templates, k, events, features, distractor, jitter, seed):
    """Generate a chronological marked-event stream; no post-hoc sorting."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, events, 2 + features), np.float32)
    y = rng.integers(k, size=n, dtype=np.int64)
    for b, target in enumerate(y):
        t = 0.0
        for j in range(events):
            t += float(rng.exponential(.035))
            signal = rng.random() > distractor
            if signal:
                amp = rng.normal(1.0, .15)
                feat = templates[target] + rng.normal(0, .65, features)
                te = t + rng.normal(0, jitter)
            else:
                amp = rng.normal(0, .65)
                feat = rng.normal(0, 1, features)
                te = t
            X[b,j,0] = max(0, te); X[b,j,1] = amp; X[b,j,2:] = feat
    return X, y


def make_templates(k, features, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(k,features)); q /= np.linalg.norm(q,axis=1,keepdims=True)
    return q


class DenseGRU(nn.Module):
    def __init__(self, features, k, hidden=64):
        super().__init__()
        self.rnn=nn.GRU(1+features,hidden,batch_first=True); self.head=nn.Linear(hidden,k)
    def forward(self,x):
        z=torch.cat([x[...,1:2],x[...,2:]],-1); h,_=self.rnn(z); return self.head(h[:,-1])


class TemporalRace(nn.Module):
    """Race with explicit delay parameters; weight cannot secretly alter delay."""
    def __init__(self, features, k, learn_delay):
        super().__init__(); self.k=k
        self.route=nn.Parameter(torch.randn(k,features)/math.sqrt(features))
        if learn_delay: self.delay=nn.Parameter(torch.full((k,),.15))
        else: self.register_buffer('delay',torch.full((k,),.15))
    def evidence(self,x):
        amp=x[...,1:2]; feat=x[...,2:]
        return (amp*torch.einsum('btf,kf->btk',feat,self.route)).sum(1)
    def candidate_times(self,x,beta=8.):
        # delay is the explicit temporal parameter; evidence is an input-dependent
        # temporal modulation. It is NOT represented as log(weight)/beta.
        base=x[...,0].min(1).values[:,None]
        return base+self.delay[None,:]-self.evidence(x)/beta
    def forward(self,x,beta=8.): return -beta*self.candidate_times(x,beta)


def train(model,Xtr,ytr,Xte,yte,epochs,device):
    model.to(device); opt=torch.optim.Adam(model.parameters(),lr=2e-3); curve=[]
    for ep in range(epochs):
        model.train(); perm=np.random.permutation(len(Xtr)); beta=4+20*ep/max(1,epochs-1)
        for s in range(0,len(perm),128):
            ii=perm[s:s+128]; xb=torch.from_numpy(Xtr[ii]).to(device); yb=torch.from_numpy(ytr[ii]).to(device)
            opt.zero_grad(); z=model(xb,beta) if isinstance(model,TemporalRace) else model(xb)
            F.cross_entropy(z,yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xb=torch.from_numpy(Xte).to(device); yt=torch.from_numpy(yte).to(device)
            z=model(xb,beta) if isinstance(model,TemporalRace) else model(xb)
            curve.append(float((z.argmax(1)==yt).float().mean()))
    return curve


@dataclass
class Race:
    winner:int
    times:np.ndarray
    margins:np.ndarray
    eligibility:np.ndarray


def resolve(times,beta):
    w=int(np.argmin(times)); margin=np.maximum(times-times[w],0)
    return Race(w,times,margin,np.exp(-beta*margin))


def episode(delays,target,mode,beta,lr,rng):
    """One race followed by a delayed local teaching event."""
    observed=np.asarray(delays)+rng.normal(0,.002,len(delays)); r=resolve(observed,beta); k=len(delays)
    if mode in ('winner_only','active_only'):
        elig=np.zeros(k); elig[r.winner]=1.
    elif mode=='binary_cf': elig=np.ones(k)
    elif mode=='margin_cf': elig=r.eligibility.copy()
    elif mode=='oracle':
        z=-beta*observed; p=np.exp(z-z.max()); elig=p/p.sum()
    else: raise ValueError(mode)

    if mode=='oracle':
        # Categorical gradient reference in delay coordinates.
        delta=lr*beta*(elig-np.eye(1,k,target).ravel())
    else:
        # Target-specific redistribution: desired route advances; alternatives retard.
        # In K=3 this is NOT equivalent to simply suppressing the winner.
        direction=np.ones(k); direction[target]=-1
        delta=lr*direction*elig
        if mode=='winner_only' and r.winner==target: delta[:]=0
    return np.clip(delays+delta,.01,2),r,elig


def counterfactual(k,trials,mode,seed,beta=10.):
    rng=np.random.default_rng(seed); d=np.array([.10,.22,.26]); wins=[]; dh=[]; eh=[]
    for _ in range(trials):
        target=int(rng.integers(k)); d,r,e=episode(d,target,mode,beta,.006,rng)
        wins.append(r.winner==target); dh.append(d.copy()); eh.append(e.copy())
    return np.asarray(wins,float),np.asarray(dh),np.asarray(eh)


def fixed_target_demo(k,trials,mode,seed):
    """Target is route 1; A and C initially beat it.

    Winner-only can suppress A, after which C can win. It cannot directly
    advance B while B is cancelled. Counterfactual learning can do so.
    """
    rng=np.random.default_rng(seed); d=np.array([.10,.30,.16]); dh=[]; eh=[]
    for _ in range(trials):
        d,r,e=episode(d,1,mode,10,.006,rng); dh.append(d.copy()); eh.append(e.copy())
    return np.asarray(dh),np.asarray(eh)


def timing_ablation(model,X,y,device):
    rng=np.random.default_rng(1234); xp=X.copy()
    for i in range(len(xp)): xp[i,:,0]=xp[i,rng.permutation(xp.shape[1]),0]
    model.eval(); xt=torch.from_numpy(X).to(device); xp=torch.from_numpy(xp).to(device); yt=torch.from_numpy(y).to(device)
    with torch.no_grad():
        a=model(xt,24) if isinstance(model,TemporalRace) else model(xt)
        b=model(xp,24) if isinstance(model,TemporalRace) else model(xp)
    return float((a.argmax(1)==yt).float().mean()),float((b.argmax(1)==yt).float().mean())


def dense_work(nodes,fanout,timesteps,episodes):
    syn=nodes*fanout; return 2*episodes*timesteps*(nodes+syn)


def event_work(fanout,input_events,active_fraction,race_width,teach,episodes):
    active=max(1.,input_events*active_fraction); scheduled=active*fanout
    races=max(1.,scheduled/race_width); fired=races; cancelled=max(0.,scheduled-fired)
    eligibility=cancelled+fired; plasticity=teach*race_width
    one=active+scheduled+2*races+eligibility+2*plasticity
    return episodes*one


def scaling():
    rows=[]
    for n in [1000,3000,10000,30000,100000,300000,1000000]:
        d=dense_work(n,8,100,1000); e=event_work(8,24,.20,8,1,1000)
        rows.append({'nodes':n,'dense_work':d,'event_work':e,'ratio':d/e})
    return rows


def main(a):
    os.makedirs(a.out,exist_ok=True); allr={'config':vars(a),'seeds':{}}
    templates=make_templates(a.k,a.features,777)
    for seed in range(a.seeds):
        seed_all(seed)
        Xtr,ytr=make_dataset(a.train,templates,a.k,a.events,a.features,a.distractors,a.jitter,1000+seed)
        Xte,yte=make_dataset(a.test,templates,a.k,a.events,a.features,a.distractors,a.jitter,2000+seed)
        gru=DenseGRU(a.features,a.k); race=TemporalRace(a.features,a.k,True); fixed=TemporalRace(a.features,a.k,False)
        gc=train(gru,Xtr,ytr,Xte,yte,a.epochs,a.device); rc=train(race,Xtr,ytr,Xte,yte,a.epochs,a.device); fc=train(fixed,Xtr,ytr,Xte,yte,a.epochs,a.device)
        intact,perm=timing_ablation(race,Xte,yte,a.device)
        cf={}
        for mode in ['winner_only','active_only','binary_cf','margin_cf','oracle']:
            w,d,e=counterfactual(a.k,a.cf_trials,mode,5000+seed)
            cf[mode]={'accuracy_last_500':float(w[-500:].mean()),'accuracy_all':float(w.mean()),'final_delays':d[-1].tolist(),'delay_history':d.tolist(),'eligibility_history':e.tolist()}
        demo={}
        for mode in ['winner_only','binary_cf','margin_cf']:
            d,e=fixed_target_demo(a.k,a.demo_trials,mode,9000+seed); demo[mode]={'delay_history':d.tolist(),'eligibility_history':e.tolist()}
        allr['seeds'][str(seed)]={'gru':gc[-1],'race':rc[-1],'fixed_delay':fc[-1],'timing_drop':intact-perm,'intact':intact,'permuted':perm,'delays':race.delay.detach().cpu().tolist(),'curves':{'gru':gc,'race':rc,'fixed':fc},'counterfactual':cf,'demo':demo}
    seeds=list(allr['seeds'].values())
    def ms(key):
        x=np.array([s[key] for s in seeds]); return {'mean':float(x.mean()),'std':float(x.std())}
    allr['summary']={'gru':ms('gru'),'race':ms('race'),'fixed_delay':ms('fixed_delay'),'timing_drop':ms('timing_drop')}
    allr['summary']['counterfactual']={m:{'mean':float(np.mean([s['counterfactual'][m]['accuracy_last_500'] for s in seeds])),'std':float(np.std([s['counterfactual'][m]['accuracy_last_500'] for s in seeds]))} for m in ['winner_only','active_only','binary_cf','margin_cf','oracle']}
    allr['energy_scaling']=scaling()
    with open(os.path.join(a.out,'results.json'),'w') as f: json.dump(allr,f,indent=2)

    s=allr['seeds']['0']
    plt.figure(figsize=(9,5)); plt.plot(s['curves']['gru'],label='dense GRU'); plt.plot(s['curves']['race'],label='temporal race'); plt.plot(s['curves']['fixed'],label='fixed-delay race'); plt.xlabel('Epoch'); plt.ylabel('Test accuracy'); plt.title('Functional viability'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(a.out,'benchmark_accuracy.png'),dpi=160); plt.close()
    plt.figure(figsize=(9,5))
    for m,style in [('winner_only','-'),('binary_cf','--'),('margin_cf','-.' )]: plt.plot(np.array(s['demo'][m]['delay_history'])[:,1],style=style,label=m)
    plt.xlabel('Training episode'); plt.ylabel('Desired-route delay'); plt.title('Multiway counterfactual credit'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(a.out,'counterfactual_learning.png'),dpi=160); plt.close()
    e=allr['energy_scaling']; n=np.array([x['nodes'] for x in e]); dw=np.array([x['dense_work'] for x in e]); ew=np.array([x['event_work'] for x in e])
    plt.figure(figsize=(9,5)); plt.loglog(n,dw,label='dense clocked training'); plt.loglog(n,ew,label='event-oriented training'); plt.xlabel('Dormant capacity (nodes)'); plt.ylabel('Dynamic-work proxy'); plt.title('Training activity scaling'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(a.out,'event_energy_scaling.png'),dpi=160); plt.close()
    print(json.dumps(allr['summary'],indent=2)); print('Results:',os.path.join(a.out,'results.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=150); p.add_argument('--seeds',type=int,default=5); p.add_argument('--train',type=int,default=4000); p.add_argument('--test',type=int,default=1500); p.add_argument('--k',type=int,default=3); p.add_argument('--events',type=int,default=32); p.add_argument('--features',type=int,default=8); p.add_argument('--distractors',type=float,default=.65); p.add_argument('--jitter',type=float,default=.035); p.add_argument('--cf-trials',type=int,default=4000); p.add_argument('--demo-trials',type=int,default=500); p.add_argument('--device',default='cpu'); p.add_argument('--out',default='sleeping_machines_v3_results'); main(p.parse_args())
