import numpy as np
import pytest
import torch

from sleeping_machines.joint_race_credit import enumerated_credit, sampled_credit


def tensor(x):
    return torch.tensor(x, dtype=torch.float64)


def test_quadratic_suffix_credit_matches_finite_difference():
    scores = tensor([-.7, .2, 1.1])
    a, b, c = tensor([.3, -.4, 1.2]), tensor([.2, .7, -.3]), tensor([.1, -.2, .4])
    surrogate_a, surrogate_b = tensor([1., 2., -1.]), tensor([-.3, .1, .8])
    rates = scores.exp(); total = rates.sum()
    nodes, weights = np.polynomial.laguerre.laggauss(24)
    estimate = torch.zeros_like(scores)
    for tau, weight in zip(nodes, weights):
        time = float(tau / total)
        estimate += weight * enumerated_credit(scores, time, a+b*time+c*time*time, surrogate_a, surrogate_b)
    def expected(s):
        r=s.exp(); z=r.sum()
        return float(((r/z)*(a+b/z+2*c/z**2)).sum())
    finite=[]
    for i in range(3):
        delta=torch.zeros_like(scores);delta[i]=1e-5
        finite.append((expected(scores+delta)-expected(scores-delta))/2e-5)
    torch.testing.assert_close(estimate, tensor(finite), atol=1e-9, rtol=1e-8)
    assert abs(float(estimate.sum())) > .01  # Genuine common-rate credit retained.


def test_importance_average_equals_enumeration_for_nonuniform_proposal():
    scores=tensor([-.2, .5, 1.]);q=tensor([.6, .3, .1]);losses=tensor([1., -2., 3.])
    a=tensor([.1, .2, .3]);b=tensor([.3, -.1, .6]);time=.7
    estimate=sum(q[i]*sampled_credit(scores,time,losses[i],i,q,a,b) for i in range(3))
    torch.testing.assert_close(estimate,enumerated_credit(scores,time,losses,a,b),atol=1e-12,rtol=1e-12)


def test_exact_affine_surrogate_has_zero_residual_and_no_draw_variance():
    scores=tensor([-.2, .5]);a=tensor([1., 3.]);b=tensor([.2, -.4]);q=tensor([.4, .6])
    reference=enumerated_credit(scores,0.,a,a,b)
    for time in (0., .1, 2., 10.):
        for i in range(2):
            torch.testing.assert_close(sampled_credit(scores,time,a[i]+b[i]*time,i,q,a,b),reference)
    assert not reference.requires_grad


def test_invalid_proposals_and_losses_rejected():
    s=tensor([0., 0.]);z=torch.zeros_like(s)
    with pytest.raises(ValueError):sampled_credit(s,1.,1.,0,tensor([1.,0.]),z,z)
    with pytest.raises(ValueError):sampled_credit(s,1.,1.,0,tensor([.2,.2]),z,z)
    with pytest.raises(ValueError):enumerated_credit(s,1.,tensor([float('nan'),1.]),z,z)
    with pytest.raises(ValueError):enumerated_credit(tensor([-10000.,0.]),1.,z,z,z)
