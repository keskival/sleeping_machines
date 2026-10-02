"""Saved native score-bound exposure and an exact bounded-race gradient trap."""
import argparse
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
from sleeping_machines.bounded_score_sensitivity import bounded_score_bridge
from sleeping_machines.race_window import bounded_delay


def summarize(scores,sites,mask,winners=None):
    s=torch.as_tensor(scores,dtype=torch.float64);prob=s.softmax(-1);cap=torch.as_tensor(mask)
    def one(selected):
        p=prob[selected];c=cap[selected];result=dict(races=int(selected.sum()),candidate_fraction=float(c.double().mean()),
            any_fraction=float(c.any(-1).double().mean()),both_fraction=float(c.all(-1).double().mean()),
            expected_capped_winner_probability=float((p*c).sum(-1).mean()))
        if winners is not None:
            win=torch.as_tensor(winners)[selected];result['observed_capped_winner_fraction']=float(c.gather(-1,win[:,None]).double().mean())
        return result
    allmask=np.ones(len(scores),dtype=bool)
    return dict(all_sites=one(allmask),layer_head={f'd{d}h{h}':one((sites[:,1]==d)&(sites[:,2]==h)) for d in (0,1) for h in (0,1)},
        first_event=one(sites[:,0]==0),last_packet=one(sites[:,0]==19),query=one(sites[:,0]==20))


def trap():
    x,w=np.polynomial.laguerre.laggauss(48);z=torch.tensor(x,dtype=torch.float64);weight=torch.tensor(w,dtype=torch.float64)
    def loss(raw,alpha):
        scores=bounded_score_bridge(raw,alpha);pi=scores.softmax(0);first=z/scores.exp().sum()
        return pi[0]+.1*(weight*bounded_delay(first)/.001).sum()
    initial=torch.tensor([13.,13.],dtype=torch.float64,requires_grad=True)
    original=loss(initial,0.);g=torch.autograd.grad(original,initial)[0];assert torch.equal(g,torch.zeros_like(g))
    hop=loss(torch.tensor([11.,13.],dtype=torch.float64),0.);assert hop<original-.2
    positive=torch.autograd.grad(loss(initial,.1),initial)[0];assert float(positive[0])>0 and float(positive[1])<0
    traces=[]
    for alpha in (0.,.1):
        raw=initial.detach().clone().requires_grad_();curve=[]
        for step in range(401):
            risk=loss(raw,alpha);gradient=torch.autograd.grad(risk,raw)[0]
            curve.append(dict(step=step,risk=float(risk.detach()),raw_scores=raw.detach().tolist(),gradient=gradient.detach().tolist()))
            if step<400:raw=(raw-gradient).detach().requires_grad_()
        if alpha==0.:assert torch.equal(raw,initial) and curve[-1]['risk']==curve[0]['risk']
        else:assert curve[-1]['risk']<curve[0]['risk']-.1
        traces.append(dict(alpha=alpha,step_size=1.,gradient_updates=400,initial_risk=curve[0]['risk'],final_risk=curve[-1]['risk'],curve=curve))
    return dict(hard_initial_zero_gradient=g.tolist(),finite_hop_risk=float(hop.detach()),positive_initial_gradient=positive.tolist(),traces=traces)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--decomposition',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    torch.set_num_threads(1);begin=time.perf_counter();parents=[];rows=[];sources={};margin=2e-5
    for role,name,field in [('cap_proxy',a.profile,'profile_artifact'),('raw_confirmed',a.decomposition,'decomposition_artifact')]:
        result=json.loads((ROOT/name).read_text());assert result['status']=='completed'
        for f,digest in result['source_sha256'].items():assert N.sha(ROOT/f)==digest
        sources.update(result['source_sha256']);artifact=ROOT/result[field];assert N.sha(artifact)==result[field+'_sha256']
        bank=np.load(artifact);parents.append(dict(role=role,result=name,result_sha256=N.sha(ROOT/name),artifact_sha256=N.sha(artifact)))
        for seed in (6,7):
            for encoder in ('initial','fixed_pass4'):
                key=f's{seed}_{encoder}';scores=bank[key+'_scores'];sites=bank[key+'_site']
                cap=np.abs(scores)>=12
                if role=='raw_confirmed':
                    raw=bank[key+'_static']+bank[key+'_dynamic'];mask=np.abs(raw)>12+margin
                    uncertain=cap & ~mask;winner=None
                else:mask=cap;winner=bank[key+'_winners'];uncertain=np.zeros_like(mask)
                rows.append(dict(role=role,seed=seed,encoder=encoder,summary=summarize(scores,sites,mask,winner),
                    ambiguous_cap_candidate_count=int(uncertain.sum())))
    witness=trap();own=['experiments/score_bound_exposure_and_trap.py','experiments/theory/105_score_bound_exposure_and_gradient_traps.md',
        'sleeping_machines/bounded_score_sensitivity.py','sleeping_machines/race_window.py']
    sources.update({f:N.sha(ROOT/f) for f in own})
    final=dict(status='completed',args=vars(a),parents=parents,rows=rows,raw_saturation_margin=margin,gradient_trap=witness,
        synthetic_gradient_updates=800,native_optimizer_steps=0,new_native_forwards=0,development_evaluations=0,
        source_sha256=sources,whole_audit_flops=None,wall_s=time.perf_counter()-begin,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,hardware=dict(host=platform.node(),threads=1),
        scope='Saved native cap occupancy proxy and separately raw-confirmed saturation; no new native model/quality. '
            'Synthetic exact conditional utility+computational latency risk shows a hard-bound flat gradient cell '
            'and bounded bridge escape. No stochastic estimator noise, DVS training nomination or benchmark advantage. '
            'All old fits, analysis and scalar gradient work retained; full FLOPs/traffic/energy unknown.')
    out.write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(rows=rows,trap=[{k:t[k] for k in ('alpha','initial_risk','final_risk')} for t in witness['traces']])),flush=True)


if __name__=='__main__':main()
