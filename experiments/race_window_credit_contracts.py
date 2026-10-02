"""Exact arrival density and multi-arrival boundary-credit numerical reference."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sleeping_machines.race_window import bounded_delay,residual_cutoff,residual_probability,conditional_residual


def tensor(x):return torch.tensor(x,dtype=torch.float64)


def nodes(degree,count):
    if count==0:return tensor(np.empty((1,0))),tensor([1.])
    x,w=np.polynomial.legendre.leggauss(degree);x=(x+1)/2;w=w/2
    grid=np.array(list(itertools.product(x,repeat=count)))
    weights=np.prod(np.array(list(itertools.product(w,repeat=count))),axis=1)
    return tensor(grid),tensor(weights)


NODES={count:nodes(64,count) for count in range(3)}


def outcome(raw_times,mask,first,width,values,decay,write=True):
    deadline=bounded_delay(first)+width
    transport=torch.exp(-decay*(deadline-bounded_delay(raw_times)))
    received=(values[None]*transport[:,:,None]*mask[None,:,None]).sum(1)
    # Real separate persistent writes; a later fixed-address read sees address2.
    old=tensor([[.1,.2],[-.2,.4],[.3,-.1]])
    stored=old+values*mask[:,None] if write else old
    prediction=(received*tensor([.8,-.6])).sum(1)+.35*stored[2].sum()
    return torch.nn.functional.softplus(-prediction)+.07*deadline.square()


def timed_rows(scores,first,width,winner,heard,uniform,ordinary=False,boundary=None):
    residuals={j:conditional_residual(scores[j],first,width,uniform[:,k]) for k,j in enumerate(heard)}
    if ordinary:
        residuals={j:x.detach()*torch.exp(-(scores[j]-scores[j].detach())) for j,x in residuals.items()}
    if boundary is not None:
        j,cutoff=boundary;residuals[j]=cutoff.detach().expand(len(uniform))
    return torch.stack([first+residuals.get(j,torch.zeros(len(uniform),dtype=first.dtype)) for j in range(3)],1)


def expectation(scores,width,values,decay,ordinary=False):
    first=tensor(.9)/scores.exp().sum();pi=scores.softmax(0);conditional=[]
    for winner in range(3):
        losers=[j for j in range(3) if j!=winner];total=first*0
        q={j:residual_probability(scores[j],first,width) for j in losers}
        for bits in itertools.product((0,1),repeat=2):
            heard=[j for j,b in zip(losers,bits) if b];mask=tensor([j==winner or j in heard for j in range(3)])
            probability=torch.stack([q[j] if b else 1-q[j] for j,b in zip(losers,bits)]).prod()
            if ordinary:probability=probability.detach()
            u,weights=NODES[len(heard)]
            raw=timed_rows(scores,first,width,winner,heard,u,ordinary)
            total=total+probability*(weights*outcome(raw,mask,first,width,values,decay)).sum()
        conditional.append(total)
    conditional=torch.stack(conditional)
    return ((pi.detach() if ordinary else pi)*conditional).sum(),conditional


def boundary_credit(scores,width,values,decay):
    first=tensor(.9)/scores.exp().sum();pi=scores.softmax(0);credit=first*0;deltas=[]
    cutoff,covered=residual_cutoff(first,width)
    if covered:return credit,deltas
    for winner in range(3):
        for candidate in range(3):
            if candidate==winner:continue
            other=next(j for j in range(3) if j not in (winner,candidate));delta=first*0
            qother=residual_probability(scores[other],first,width).detach()
            for bit in (0,1):
                heard=[other] if bit else [];u,weights=NODES[len(heard)]
                raw=timed_rows(scores,first,width,winner,heard,u,boundary=(candidate,cutoff))
                mask=tensor([j==winner or j in heard for j in range(3)]);on=mask.clone();on[candidate]=1
                difference=outcome(raw,on,first,width,values,decay)-outcome(raw,mask,first,width,values,decay)
                delta=delta+(qother if bit else 1-qother)*(weights*difference).sum()
            deltas.append(dict(winner=winner,candidate=candidate,actual_write_boundary_loss_difference=float(delta.detach())))
            credit=credit+pi[winner].detach()*residual_probability(scores[candidate],first,width)*delta.detach()
    return credit,deltas


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    begin=time.perf_counter();torch.set_num_threads(1)
    scores=tensor(np.log([.7,1.3,2.2])).requires_grad_();width=tensor(.0011).requires_grad_()
    values=tensor([[.4,.7],[-.3,.2],[.6,-.5]]).requires_grad_();decay=tensor(8.).requires_grad_()
    parameters=(scores,width,values,decay)
    # Original and factorized joint densities; unit Jacobian in every winner cell.
    first=tensor(.17);residual=tensor([.24,.51,.83]);rates=scores.exp()
    for winner in range(3):
        times=first+residual;times=times.clone();times[winner]=first
        original=(rates*torch.exp(-rates*times)).prod()
        conditional=rates[winner]*torch.exp(-rates.sum()*first)
        for j in range(3):
            if j!=winner:conditional=conditional*rates[j]*torch.exp(-rates[j]*residual[j])
        torch.testing.assert_close(original,conditional,rtol=1e-13,atol=1e-14)
    # Physical and raw membership agree, including zero and complete-span widths.
    for T in (.0,.3,2.):
        for H in (.0,.0003,.0015,.0101):
            R,covered=residual_cutoff(tensor(T),tensor(H))
            if not covered:
                torch.testing.assert_close(bounded_delay(tensor(T)+R),bounded_delay(tensor(T))+H,rtol=1e-13,atol=1e-14)
                for x in (.0,.04,.4,3.):
                    assert bool(tensor(x)<=R)==bool(bounded_delay(tensor(T+x))<=bounded_delay(tensor(T))+H)
            else:assert bool(bounded_delay(tensor(T+1e12))<=bounded_delay(tensor(T))+H)
    T=tensor(.9)/rates.sum();activity=[]
    for winner in range(3):
        losers=[j for j in range(3) if j!=winner];q=[residual_probability(scores[j],T,width) for j in losers]
        normalization=T*0;selected=T*0
        for bits in itertools.product((0,1),repeat=2):
            probability=torch.stack([qj if bit else 1-qj for qj,bit in zip(q,bits)]).prod()
            normalization=normalization+probability;selected=selected+probability*(1+sum(bits))
        torch.testing.assert_close(normalization,tensor(1.),rtol=1e-13,atol=1e-14)
        torch.testing.assert_close(selected,1+sum(q),rtol=1e-13,atol=1e-14)
        activity.append(float(selected.detach()))
    exact,conditional=expectation(*parameters);ordinary,_=expectation(*parameters,ordinary=True)
    boundary,deltas=boundary_credit(*parameters)
    choice=(scores.softmax(0)*conditional.detach()).sum()
    expected=torch.autograd.grad(exact,parameters,retain_graph=True)
    corrected=torch.autograd.grad(ordinary+boundary+choice,parameters,retain_graph=True)
    plain=torch.autograd.grad(ordinary,parameters,retain_graph=True)
    for e,c in zip(expected,corrected):torch.testing.assert_close(e,c,rtol=1e-8,atol=2e-8)
    assert abs(float(expected[1]-plain[1]))>1.
    # Independent central differences of the complete conditional expectation.
    differences=[]
    for group,param in enumerate(parameters):
        numeric=torch.empty_like(param)
        for index in range(param.numel()):
            step=1e-7 if group==1 else 1e-5
            plus=[x.detach().clone() for x in parameters];minus=[x.detach().clone() for x in parameters]
            plus[group].view(-1)[index]+=step;minus[group].view(-1)[index]-=step
            with torch.no_grad():numeric.view(-1)[index]=(expectation(*plus)[0]-expectation(*minus)[0])/(2*step)
        torch.testing.assert_close(expected[group],numeric,rtol=2e-6,atol=2e-6)
        differences.append(float((expected[group]-numeric).abs().max()))
    # Zero-width law nests one winner and its actual write; width has birth credit.
    zero=tensor(0.).requires_grad_();rz,cz=expectation(scores,zero,values,decay)
    direct=0.
    for winner in range(3):
        mask=tensor([j==winner for j in range(3)]);raw=T.expand(1,3)
        direct=direct+scores.softmax(0)[winner]*outcome(raw,mask,T,zero,values,decay)[0]
    torch.testing.assert_close(rz,direct,rtol=1e-13,atol=1e-14)
    zordinary,_=expectation(scores,zero,values,decay,ordinary=True);zb,_=boundary_credit(scores,zero,values,decay)
    zg=torch.autograd.grad(rz,zero,retain_graph=True)[0];zc=torch.autograd.grad(zordinary+zb,zero)[0]
    torch.testing.assert_close(zg,zc,rtol=1e-10,atol=1e-9)
    with torch.no_grad():zfd=(expectation(scores,tensor(1e-9),values,decay)[0]-rz)/1e-9
    torch.testing.assert_close(zg,zfd,rtol=2e-6,atol=2e-5)
    # Complete-span probability and derivatives; no spurious membership credit.
    cap=tensor(.011).requires_grad_();q=residual_probability(scores[1],T,cap)
    torch.testing.assert_close(q,tensor(1.),rtol=0,atol=0)
    capgrad=torch.autograd.grad(q,(scores,cap))
    for gradient in capgrad:torch.testing.assert_close(gradient,torch.zeros_like(gradient),rtol=0,atol=0)
    # A delivery-only boundary is not an actual memory-write alternative.
    R,_=residual_cutoff(T,width);raw=torch.stack((T,T,T+R))[None]
    off=tensor([1.,0.,0.]);on=tensor([1.,0.,1.])
    actual=outcome(raw,on,T,width,values,decay)-outcome(raw,off,T,width,values,decay)
    delivery=outcome(raw,on,T,width,values,decay,False)-outcome(raw,off,T,width,values,decay,False)
    assert abs(float(actual-delivery))>.001
    sources=['sleeping_machines/race_window.py','experiments/race_window_credit_contracts.py',
        'experiments/theory/97_multi_arrival_race_window_credit.md']
    result=dict(status='completed',args=vars(a),contracts_passed=7,
        original_and_factorized_joint_density_equal=True,physical_cutoff_zero_width_and_cap_passed=True,
        conditional_membership_normalization_and_activity_passed=True,
        conditional_quadrature_and_boundary_choice_path_gradients_equal=True,
        every_score_width_message_and_decay_finite_difference_passed=True,
        zero_width_one_sided_boundary_credit_passed=True,actual_write_differs_from_delivery_only=True,
        conditional_noise_exp1=.9,quadrature_degree=64,candidates=3,membership_histories=12,
        boundary_pairs_per_other_history=12,expected_selected_given_winner=activity,
        exact_conditional_expected_loss=float(exact.detach()),
        exact_score_gradient=expected[0].tolist(),ordinary_history_score_gradient=plain[0].tolist(),
        exact_width_gradient=float(expected[1]),ordinary_history_width_gradient=float(plain[1]),
        zero_width_one_sided_gradient=float(zg),finite_difference_max_errors=differences,boundary_loss_differences=deltas,
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources},
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        numerical_reference_flops=None,
        scope='Three-candidate exact conditional arrival-law and full multi-arrival boundary-gradient reference; '
            'decaying content, actual separate writes and later fixed-address read. Conditional first-noise fixed, '
            'deterministic quadrature, no optimizer/fit/native window installation or benchmark advantage. '
            'Enumeration and numerical reference work paid; FLOPs/traffic/energy unmeasured, not zero. '
            'Fixed-first deadline differs from silence-reset popcorn; full-sequence credit/scheduler remain open.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('contracts_passed','exact_width_gradient','ordinary_history_width_gradient','wall_s','max_rss_kb')}),flush=True)


if __name__=='__main__':main()
