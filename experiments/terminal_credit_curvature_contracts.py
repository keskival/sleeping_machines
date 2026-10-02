"""Check scoped policy curvature claims through the actual route backwards."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sleeping_machines.sparse_race_language import TemporalRoute
from sleeping_machines.batched_addressed_fit import BatchedTemporalRoute


def losses(values):
    x = values.reshape(-1)
    logits = torch.stack((torch.zeros_like(x), x-3, 1-3*x), -1)
    return F.cross_entropy(logits, torch.zeros(len(x), dtype=torch.long), reduction='none')


def local_mean(scores, values, batched=False, affine=False):
    probabilities = scores.softmax(-1).detach()
    score_grad = torch.zeros_like(scores); value_grad = torch.zeros_like(values)
    mean_time = scores.exp().sum().reciprocal().detach()
    for winner in range(2):
        s = scores.detach().clone().requires_grad_()
        v = values.detach().clone().requires_grad_()
        # Backward depends only on winner, T and values. E[T]=1/Lambda
        # is sufficient here: no delay derivative and frozen prefix/time.
        chosen = v[winner].detach().clone().requires_grad_()
        loss = .7*chosen.sum()+.2 if affine else losses(chosen)[0]
        error = torch.autograd.grad(loss, chosen)[0]
        if batched:
            context=SimpleNamespace(saved_tensors=(s.detach().exp()[None,None],
                mean_time.reshape(1,1),torch.tensor([[winner]]),v.detach()[None,None]))
            gs,gv,_=BatchedTemporalRoute.backward(context,error[None,None],None,None)
            gs,gv=gs.reshape_as(s),gv.reshape_as(v)
        else:
            context=SimpleNamespace(saved_tensors=(s.detach().exp(),mean_time,
                torch.tensor(winner),v.detach()))
            gs,gv=TemporalRoute.backward(context,error,None,None)
        score_grad += probabilities[winner]*gs
        value_grad += probabilities[winner]*gv
    return score_grad, value_grad


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    a=p.parse_args();begin=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    rows=[]
    for first_probability in (.2,.5,.8):
        scores=torch.tensor([first_probability,1-first_probability],dtype=torch.float64).log().requires_grad_()
        values=torch.tensor([[0.],[5.]],dtype=torch.float64,requires_grad=True)
        endpoint=losses(values);risk=(scores.softmax(-1)*endpoint).sum()
        exact_s,exact_v=torch.autograd.grad(risk,(scores,values))
        reference_s,reference_v=local_mean(scores,values)
        batched_s,batched_v=local_mean(scores,values,batched=True)
        torch.testing.assert_close(reference_s,batched_s,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(reference_v,batched_v,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(reference_v,exact_v,rtol=1e-12,atol=1e-12)
        endpoint_derivative=torch.autograd.grad(losses(values).sum(),values)[0]
        pi=scores.softmax(-1).detach();difference=(values[1]-values[0]).detach()
        trapezoid=pi.prod()*(difference*(endpoint_derivative[0]+endpoint_derivative[1])/2).sum()
        torch.testing.assert_close(reference_s[1],trapezoid,rtol=1e-12,atol=1e-12)
        assert exact_s[1]>0 and reference_s[1]<0
        def updated_risk(gradient):return float(((scores.detach()-.01*gradient).softmax(-1)*endpoint.detach()).sum())
        assert updated_risk(exact_s)<float(risk.detach())<updated_risk(reference_s)
        affine_s,affine_v=local_mean(scores,values,affine=True)
        affine_risk=(scores.softmax(-1)*(.7*values[:,0]+.2)).sum()
        affine_exact=torch.autograd.grad(affine_risk,(scores,values))
        torch.testing.assert_close(affine_s,affine_exact[0],rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(affine_v,affine_exact[1],rtol=1e-12,atol=1e-12)
        rows.append(dict(first_probability=first_probability,endpoint_losses=endpoint.detach().tolist(),
            endpoint_derivatives=endpoint_derivative.flatten().tolist(),exact_score_gradient=exact_s.tolist(),
            mean_local_score_gradient=reference_s.tolist(),reference_batched_gradient_match=True,
            expected_winner_payload_gradient_is_exact=True,affine_loss_teacher_is_exact=True,
            risk_before=float(risk.detach()),risk_after_exact_step=updated_risk(exact_s),
            risk_after_local_step=updated_risk(reference_s)))
    # Numerical Hessian bound and logit-space Taylor remainder, alongside proof.
    z=torch.tensor([.7,-1.2,2.1],dtype=torch.float64,requires_grad=True)
    dz=torch.tensor([-.8,1.1,.3],dtype=torch.float64)
    def ce(x):return F.cross_entropy(x[None],torch.tensor([0]))
    hessian=torch.autograd.functional.hessian(ce,z)
    norm=float(torch.linalg.eigvalsh(hessian).max());assert norm<=.5+1e-12
    gradient=torch.autograd.grad(ce(z),z)[0]
    remainder=float((ce(z+dz)-ce(z)-(gradient*dz).sum()).detach())
    assert 0<=remainder<=float(dz.square().sum())/4+1e-12
    sources=['experiments/terminal_credit_curvature_contracts.py',
        'experiments/theory/78_terminal_curvature_and_credit.md',
        'sleeping_machines/sparse_race_language.py','sleeping_machines/batched_addressed_fit.py']
    result=dict(status='completed',args=vars(a),rows=rows,contracts_passed=5,
        maximum_tested_ce_hessian_eigenvalue=norm,tested_taylor_remainder=remainder,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Fixed-prefix/time two-candidate actual teacher mean, convex affine-head CE counterexample, exact winner payload and affine teacher contracts. Analytic conditions in theory78; no fit, test labels, whole-core gradient or empirical-cause claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
