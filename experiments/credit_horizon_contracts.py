"""Exact delayed-feature truncation witnesses and finite sampled-route moments."""
import argparse
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    torch.set_num_threads(1);begin=time.perf_counter();rho=math.exp(-.01);D=32;H=16
    theta=torch.tensor(.3,dtype=torch.float64,requires_grad=True);x=torch.tensor(1.,dtype=torch.float64)
    z=theta*x
    for _ in range(D):z=rho*z
    loss=.5*(z-x).square();full,=torch.autograd.grad(loss,theta)
    expected=rho**D*(rho**D*.3-1)
    torch.testing.assert_close(full,torch.tensor(expected,dtype=torch.float64),rtol=1e-12,atol=1e-12)
    def risk(value):return .5*(rho**D*value-1).square()
    assert torch.autograd.gradcheck(risk,(theta,),eps=1e-5,atol=1e-10,rtol=1e-10)
    z=theta*x
    for event in range(D):
        if event==H:z=z.detach()
        z=rho*z
    truncated_loss=.5*(z-x).square()+0*theta
    omitted,=torch.autograd.grad(truncated_loss,theta)
    assert float(omitted)==0 and float(full)<0
    improved=theta.detach()+.01
    assert float(risk(improved))<float(loss.detach())
    route_logit=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    values=torch.tensor([.5*(rho**D-1)**2,.5],dtype=torch.float64)
    scores=torch.stack([route_logit,route_logit*0]);pi=scores.softmax(0)
    choice_risk=(pi*values).sum();choice,=torch.autograd.grad(choice_risk,route_logit)
    exact=pi.detach().prod()*(values[0]-values[1])
    torch.testing.assert_close(choice,exact,rtol=1e-12,atol=1e-12)
    assert float(choice)!=0
    scores=torch.stack([route_logit,route_logit*0]);before_label=(scores.softmax(0)*torch.zeros(2,dtype=torch.float64)).sum()
    prefix,=torch.autograd.grad(before_label,route_logit);assert float(prefix)==0
    # Exhaust every distinct subset for a finite sparse-signal population.
    R=12;k=3;g=torch.tensor([.2,-.4,.1],dtype=torch.float64)
    population=torch.zeros(R,3,dtype=torch.float64);population[0]=g
    estimates=torch.stack([population[list(indices)].sum(0)*(R/k)
                           for indices in itertools.combinations(range(R),k)])
    mean=estimates.mean(0);cov=(estimates-mean).T@(estimates-mean)/len(estimates)
    torch.testing.assert_close(mean,g,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(cov,(R/k-1)*g[:,None]*g[None,:],rtol=1e-12,atol=1e-12)
    tail=[]
    for horizon in (16,32,64,128):
        value=rho**horizon
        closed_tail=value/(1-rho)
        observed=sum(rho**i for i in range(horizon,4096))
        assert abs(closed_tail-observed)<1e-9
        tail.append(dict(horizon=horizon,geometric_tail_fraction=value,absolute_geometric_tail=closed_tail))
    schedules=[]
    for horizon,k in [(16,256),(64,8)]:
        races=horizon*8*2;lanes=k*2;events=lanes*horizon
        schedules.append(dict(horizon=horizon,races=races,selected_credit_races=k,
            selection_probability=k/races,shadow_lanes=lanes,shadow_events=events,
            shadow_events_per_target=events/horizon,factual_events=horizon,
            all_factual_plus_shadow_events_per_target=(events+horizon)/horizon,
            single_informative_route_variance_multiplier=races/k-1))
    result=dict(status='completed',args=vars(a),contracts_passed=7,
        delayed_content=dict(rho=rho,delay=D,credit_horizon=H,theta=.3,loss=float(loss.detach()),
            full_encoder_gradient=float(full),truncated_encoder_gradient=float(omitted),
            finite_step_theta=float(improved),finite_step_loss=float(risk(improved))),
        delayed_race=dict(logit=.2,utilities=values.tolist(),full_choice_gradient=float(choice),
            truncated_choice_gradient=float(prefix)),geometric_tail=tail,
        distinct_sampling=dict(races=R,k=3,exhaustive_subsets=len(estimates),true_gradient=g.tolist(),
            mean=mean.tolist(),covariance=cov.tolist(),variance_multiplier=3.),execution_counts=schedules,
        source_sha256={name:N.sha(ROOT/name) for name in ['experiments/credit_horizon_contracts.py',
            'experiments/theory/110_persistent_capacity_and_credit_horizon.md']},
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        optimizer_steps=0,development_evaluations=0,
        scope='Exact synthetic truncation/counterfactual utility and allocation-count witnesses; '
            'not an empirical native/text8 horizon diagnosis, native Jacobian bound or efficiency advantage.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
