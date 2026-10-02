import pytest
import torch

from sleeping_machines.statistic_race_credit import delivery_credit, expected_write_gain, predictive


def test_actual_predictive_loss_gradient_matches_reference():
    counts=torch.tensor([[0.,3.,1.],[6.,0.,2.]],dtype=torch.float64)
    scores=torch.tensor([.4,-.7],dtype=torch.float64,requires_grad=True)
    losses=-predictive(counts)[:,1].log()
    (scores.softmax(0)*losses).sum().backward()
    torch.testing.assert_close(scores.grad,delivery_credit(scores,counts,1))


def test_closed_write_gain_matches_actual_expected_predictive_change():
    counts=torch.tensor([0.,5.,2.],dtype=torch.float64)
    q=torch.tensor([.6,.1,.3],dtype=torch.float64)
    before=predictive(counts)
    for symbol in range(3):
        changed=counts.clone();changed[symbol]+=1
        actual=(q*(predictive(changed).log()-before.log())).sum()
        torch.testing.assert_close(expected_write_gain(counts,symbol,q),actual,atol=1e-12,rtol=1e-12)


def test_equal_delivery_does_not_imply_equal_write_utility():
    counts=torch.tensor([2.,2.],dtype=torch.float64)
    positive=expected_write_gain(counts,0,torch.tensor([.9,.1],dtype=torch.float64))
    negative=expected_write_gain(counts,0,torch.tensor([.1,.9],dtype=torch.float64))
    assert positive>0>negative
    assert torch.equal(delivery_credit(torch.zeros(2),counts.repeat(2,1),0),torch.zeros(2,dtype=torch.float64))


def test_invalid_laws_and_counts_rejected():
    with pytest.raises(ValueError):predictive(torch.tensor([-1.,2.]))
    with pytest.raises(ValueError):expected_write_gain(torch.ones(2),0,torch.ones(2))
