"""Exact finite witnesses for fixed versus winner-dependent downstream teachers."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
from sleeping_machines.exact_pi_race import ExactPiRoute
from sleeping_machines.sparse_race_language import TemporalRoute


def means(scores,values,errors):
    rates=scores.detach().exp();pi=rates/rates.sum();mean_time=1/rates.sum()
    old=[];new=[]
    for winner in range(len(rates)):
        ctx=SimpleNamespace(saved_tensors=(rates,mean_time,torch.tensor(winner),values))
        old.append(TemporalRoute.backward(ctx,errors[winner],None,None)[0])
        new.append(ExactPiRoute.backward(ctx,errors[winner],None,None)[0])
    return (pi[:,None]*torch.stack(old)).sum(0),(pi[:,None]*torch.stack(new)).sum(0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args();started=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique tag required')
    scores=torch.tensor([1.,3.],dtype=torch.float64).log().requires_grad_()
    values=torch.tensor([[0.],[1.]],dtype=torch.float64);pi=scores.softmax(0)
    fixed=torch.tensor([[2.],[2.]],dtype=torch.float64)
    before,after=means(scores,values,fixed)
    desired=torch.autograd.grad((pi*(values[:,0]*2)).sum(),scores,retain_graph=True)[0]
    torch.testing.assert_close(before,desired,rtol=0,atol=1e-12)
    torch.testing.assert_close(after,desired,rtol=0,atol=1e-12)
    target=.6;errors=values-target;losses=.5*(values[:,0]-target).square()
    exact=torch.autograd.grad((pi*losses).sum(),scores)[0]
    old,new=means(scores,values,errors)
    torch.testing.assert_close(old,exact,rtol=0,atol=1e-12)
    torch.testing.assert_close(new,torch.tensor([-.028125,.028125],dtype=torch.float64),rtol=0,atol=1e-12)
    assert float(old.dot(new))<0
    # In a quadratic, averaged realized error makes this teacher optimize mean-value risk.
    soft_scores=scores.detach().clone().requires_grad_();soft_pi=soft_scores.softmax(0)
    mean=(soft_pi*values[:,0]).sum();mean_risk=.5*(mean-target).square()
    value_variance=(soft_pi*values[:,0].square()).sum()-mean.square()
    mean_gradient=torch.autograd.grad(mean_risk,soft_scores,retain_graph=True)[0]
    variance_gradient=torch.autograd.grad(.5*value_variance,soft_scores)[0]
    torch.testing.assert_close(new,mean_gradient,rtol=0,atol=1e-12)
    torch.testing.assert_close(exact,mean_gradient+variance_gradient,rtol=0,atol=1e-12)
    # Forced candidate times versus conditional first time have different laws.
    total_rate=4.;candidate_rates=torch.tensor([1.,3.],dtype=torch.float64)
    conditional_mean_time=1/total_rate;forced_candidate_means=1/candidate_rates
    assert not bool((forced_candidate_means==conditional_mean_time).any())
    result=dict(status='completed',args=vars(a),contracts_passed=4,
        fixed_downstream_error_old_and_exact_pi_expectations_match=True,
        convex_quadratic_winner_dependent_error=dict(probabilities=[.25,.75],values=[0.,1.],target=target,
            legal_losses=losses.tolist(),true_categorical_gradient=exact.tolist(),
            original_teacher_expected_gradient=old.tolist(),exact_pi_teacher_expected_gradient=new.tolist(),
            expected_gradient_dot_product=float(old.dot(new)),direction_reversed=True),
        quadratic_mean_risk_decomposition=dict(mean_value_risk_gradient=mean_gradient.tolist(),
            half_value_variance_gradient=variance_gradient.tolist(),
            exact_pi_expected_gradient_equals_mean_risk_gradient=True,
            true_hard_outcome_risk_gradient_includes_variance_term=True),
        time_law=dict(conditional_first_time_mean=conditional_mean_time,
            forced_candidate_time_means=forced_candidate_means.tolist(),
            scope='Independent exponential races; conditional first time is Exp(sum rates), independent of winner. Forcing an unconditioned individual time changes the time law.'),
        source_sha256={n:N.sha(ROOT/n) for n in ['experiments/race_teacher_expectation_contracts.py',
            'sleeping_machines/exact_pi_race.py','sleeping_machines/sparse_race_language.py','experiments/dvs_credit_fidelity_audit.py']},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Finite exact categorical enumeration and analytic first-time mean against actual backward implementations; fixed-error equality and winner-dependent convex loss reversal. This disproves a general unchanged-expectation/zero-variance claim, not the empirical usefulness of the alternative teacher. Earlier fidelity audit also changes candidate time when forcing a route, so its losses are combined identity/time interventions rather than fixed-time categorical risk.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
