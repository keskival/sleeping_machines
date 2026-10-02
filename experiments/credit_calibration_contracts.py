"""Exact common-clock/choice pullback and normalization-bias witnesses."""
import argparse
import json
from pathlib import Path
import resource
import time
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    a=p.parse_args();start=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    trials=[]
    for logits,clock,first,base in [([-.8,.7],.2,.4,.6),([0.,0.],-.4,1.8,0.),([.1,-.3,.8],.7,.6,2.)]:
        u=torch.tensor(logits,dtype=torch.float64,requires_grad=True)
        c=torch.tensor(clock,dtype=torch.float64,requires_grad=True)
        s=c+u-u.logsumexp(-1);rates=s.exp();pi=u.softmax(-1)
        losses=torch.arange(1,len(logits)+1,dtype=torch.float64).square()/3
        density=s-rates.sum()*first
        # Winner weights and actual suffix outcomes are held fixed in the score term.
        score_loss=(pi.detach()*(losses-base)*density).sum()
        gc,gu=torch.autograd.grad(score_loss,(c,u),retain_graph=True)
        risk=(pi.detach()*losses).sum()
        torch.testing.assert_close(gc,(1-c.detach().exp()*first)*(risk-base),rtol=1e-12,atol=1e-12)
        exact=torch.autograd.grad((pi*losses).sum(),u)[0]
        torch.testing.assert_close(gu,exact,rtol=1e-12,atol=1e-12)
        for offset in (-5.,.3,8.):
            shifted=u.detach().clone().requires_grad_()
            shifted_choice=torch.autograd.grad((shifted.softmax(-1)*(losses+offset)).sum(),shifted)[0]
            torch.testing.assert_close(shifted_choice,exact,rtol=1e-12,atol=1e-12)
        # Baseline error alters the common clock and leaves choice unchanged.
        for alternative_base in (-2.,0.,3.):
            uu=u.detach().clone().requires_grad_();cc=c.detach().clone().requires_grad_()
            ss=cc+uu-uu.logsumexp(-1)
            ll=(uu.detach().softmax(-1)*(losses-alternative_base)*(ss-ss.exp().sum()*first)).sum()
            _,choice=torch.autograd.grad(ll,(cc,uu))
            torch.testing.assert_close(choice,exact,rtol=1e-12,atol=1e-12)
        old=torch.tensor(logits,dtype=torch.float64,requires_grad=True)
        composed=old.logsumexp(-1)+old-old.logsumexp(-1)
        torch.testing.assert_close(composed,old,rtol=1e-12,atol=1e-12)
        weights=old.detach().softmax(-1)*(losses-base)
        direct=(weights*(old-old.exp().sum()*first)).sum()
        rewritten=(weights*(composed-composed.exp().sum()*first)).sum()
        one=torch.autograd.grad(direct,old,retain_graph=True)[0]
        two=torch.autograd.grad(rewritten,old)[0]
        torch.testing.assert_close(one,two,rtol=1e-12,atol=1e-12)
        trials.append(dict(logits=logits,common_log_rate=clock,time=first,baseline=base,
            common_clock_credit=float(gc),choice_credit=gu.tolist()))
    gradients=torch.tensor([2.,-1.],dtype=torch.float64);probabilities=torch.tensor([1/3,2/3],dtype=torch.float64)
    raw=(probabilities*gradients).sum();normalized=(probabilities*gradients.sign()).sum()
    torch.testing.assert_close(raw,torch.tensor(0.,dtype=torch.float64),rtol=0,atol=1e-15)
    torch.testing.assert_close(normalized,torch.tensor(-1/3,dtype=torch.float64),rtol=0,atol=1e-15)
    # Positive output-space scaling can oppose the true shared-parameter gradient.
    parameters=torch.tensor([.2,-.4],dtype=torch.float64,requires_grad=True)
    jacobian=parameters.new_tensor([[1.,0.],[2.,1.]])
    logits=jacobian@parameters
    loss=(logits*parameters.new_tensor([1.,-1.])).sum()
    true=torch.autograd.grad(loss,parameters,retain_graph=True)[0]
    scaled=torch.autograd.grad(logits,parameters,grad_outputs=parameters.new_tensor([10.,-1.]))[0]
    alignment=float(true.dot(scaled));assert alignment==-7
    before=float(loss.detach());after=float(((jacobian@(parameters.detach()-.01*scaled))*parameters.new_tensor([1.,-1.])).sum())
    assert abs(after-before-.07)<1e-12
    import hashlib
    names=['experiments/credit_calibration_contracts.py','experiments/theory/85_credit_calibration_coordinates.md']
    result=dict(status='completed',args=vars(a),contracts_passed=5,trials=trials,
        categorical_utility_and_loss_offset_invariance=True,complete_joint_density_coordinate_pullback=True,
        identity_reparameterization_preserves_original_gradient=True,
        zero_mean_sample_normalization_bias=dict(raw_mean=float(raw),unit_normalized_mean=float(normalized)),
        positive_score_scaling_parameter_ascent=dict(gradient_alignment=alignment,loss_before=before,loss_after=after),
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Finite isolated-rate mathematical contracts; no integrated clock-normalized model, fitted quality, optimizer-noise reduction, independent confirmation or advantage claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
