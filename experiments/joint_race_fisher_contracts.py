"""Exact finite-moment contracts for the joint winner/time information metric."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    torch.set_num_threads(1);begin=time.perf_counter();dtype=torch.float64
    s=torch.tensor([-1.2,.4,.9],dtype=dtype,requires_grad=True);rates=s.exp();total=rates.sum();pi=rates/total
    # Gauss-Laguerre integrates Exp(1) expectation independently of algebra below.
    x,w=np.polynomial.laguerre.laggauss(48);z=torch.tensor(x,dtype=dtype);weight=torch.tensor(w,dtype=dtype)
    h=torch.eye(3,dtype=dtype)[:,None,:]-pi[None,None,:]*z[None,:,None]
    probability=pi[:,None]*weight[None,:]
    fisher=torch.einsum('wt,wti,wtj->ij',probability,h,h)
    torch.testing.assert_close(fisher,torch.diag(pi),rtol=1e-11,atol=1e-11)
    mean=torch.einsum('wt,wti->i',probability,h);torch.testing.assert_close(mean,torch.zeros(3,dtype=dtype),rtol=0,atol=1e-11)
    q=torch.eye(3,dtype=dtype)-pi;c=pi[None,:]*(1-z[:,None])
    cross=torch.einsum('w,t,wi,tj->ij',pi,weight,q,c)
    torch.testing.assert_close(cross,torch.zeros_like(cross),rtol=0,atol=1e-11)
    categorical=torch.diag(pi)-pi[:,None]*pi[None,:]
    torch.testing.assert_close(fisher,categorical+pi[:,None]*pi[None,:],rtol=1e-11,atol=1e-11)
    # Exact risk for a_w+v_w*T+.7*T^2, retaining winner/time coupling.
    av=torch.tensor([.8,-.4,1.3],dtype=dtype);v=torch.tensor([.7,1.1,-.2],dtype=dtype)
    risk=(pi*av).sum()+(pi*v).sum()/total+1.4/total.square()
    gradient=torch.autograd.grad(risk,s)[0]
    loss=av[:,None]+v[:,None]*z[None,:]/total+.7*(z[None,:]/total).square()
    likelihood_gradient=torch.einsum('wt,wt,wti->i',probability,loss,h)
    torch.testing.assert_close(gradient,likelihood_gradient,rtol=1e-10,atol=1e-10)
    ell=av+v/total+1.4/total.square();choice=pi*(ell-(pi*ell).sum())
    kappa=-(pi*v).sum()/total-2.8/total.square()
    torch.testing.assert_close(gradient,choice+pi*kappa,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close((choice/pi*pi).sum(),torch.zeros((),dtype=dtype),rtol=0,atol=1e-12)
    torch.testing.assert_close((pi*kappa)/pi,torch.ones_like(pi)*kappa,rtol=1e-12,atol=1e-12)
    # b,a pullback: ds/db=1, ds/da=I-1*pi^T; common-clock/choice blocks decouple.
    jac=torch.cat([torch.ones(3,1,dtype=dtype),torch.eye(3,dtype=dtype)-torch.ones(3,1,dtype=dtype)*pi[None,:]],1)
    block=jac.T@fisher@jac;expected=torch.zeros(4,4,dtype=dtype);expected[0,0]=1;expected[1:,1:]=categorical
    torch.testing.assert_close(block,expected,rtol=1e-11,atol=1e-11)
    shared=torch.tensor([[1.,.2],[.3,1.],[-.8,.4]],dtype=dtype)
    parameter_fisher=shared.T@fisher@shared
    pulled=torch.einsum('wt,wtk,wtl->kl',probability,h@shared,h@shared)
    torch.testing.assert_close(parameter_fisher,pulled,rtol=1e-11,atol=1e-11)
    native_step=torch.linalg.solve(parameter_fisher,shared.T@gradient)
    shortcut=shared.T@(gradient/pi)
    assert float((native_step-shortcut).abs().max())>.05
    noise=[]
    for tiny in (.1,.01,.001):
        pvec=torch.tensor([tiny,1-tiny],dtype=dtype);metric=torch.diag(pvec)
        amplification=float(torch.trace(torch.diag(1/pvec)@metric@torch.diag(1/pvec)))
        assert abs(amplification-float((1/pvec).sum()))<1e-10
        noise.append(dict(rare_probability=tiny,raw_score_mean_square=1.,inverse_metric_mean_square=amplification))
    names=['experiments/joint_race_fisher_contracts.py','experiments/theory/102_joint_race_fisher_and_calibration_limits.md']
    result=dict(status='completed',args=vars(a),contracts_passed=9,joint_fisher=fisher.detach().tolist(),
        joint_fisher_max_error=float((fisher-torch.diag(pi)).abs().max()),
        risk_gradient=gradient.tolist(),quadrature_gradient=likelihood_gradient.detach().tolist(),
        gradient_max_error=float((gradient-likelihood_gradient).abs().max()),
        parameter_step=native_step.detach().tolist(),score_division_shortcut=shortcut.detach().tolist(),noise_amplification=noise,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(host=platform.node(),threads=1),optimizer_steps=0,development_evaluations=0,whole_audit_flops=None,
        scope='Local common-start exponential winner+first-time law. Exact moment identities/independent quadrature, '
            'conditional-return gradient, common-clock/choice blocks, shared-parameter pullback and inverse-metric noise. '
            'No native estimator/optimizer replacement, fitted model, quality gain or free credit normalization.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
