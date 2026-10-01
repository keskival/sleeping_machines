"""Numerical proof contracts; no fitting or optimizer updates."""
import torch

from sleeping_machines.temporal_window import temporal_window


def explicit(times,values,queries,width,decay):
    age = queries[:,None]-times[None,:]
    u = age/width
    kernel = u.square()*(1-u).square()*torch.exp(-decay*age)
    return (kernel*torch.logical_and(age>0,age<width)) @ values


def test_moment_recurrence_matches_all_parameters_and_times():
    times = torch.tensor([.0,.21,.6,1.2,1.5,2.0],dtype=torch.float64,requires_grad=True)
    values = torch.arange(18,dtype=torch.float64).reshape(6,3).div(13).requires_grad_()
    queries = torch.tensor([.1,.5,.9,1.4,2.1,2.9],dtype=torch.float64,requires_grad=True)
    width = torch.tensor(1.1,dtype=torch.float64,requires_grad=True)
    decay = torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    arguments = times,values,queries,width,decay
    a = temporal_window(*arguments)
    b = explicit(*arguments)
    torch.testing.assert_close(a,b,rtol=1e-10,atol=2e-14)
    weight = torch.arange(a.numel(),dtype=a.dtype).reshape(a.shape)/11
    ga = torch.autograd.grad((a*weight).sum(),arguments,retain_graph=True)
    gb = torch.autograd.grad((b*weight).sum(),arguments)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=2e-9,atol=2e-12)
    assert torch.autograd.gradcheck(temporal_window,arguments,eps=1e-6,atol=2e-6)


def test_window_entry_and_expiry_have_zero_value_and_slope():
    times = torch.tensor([0.],dtype=torch.float64)
    values = torch.ones((1,1),dtype=torch.float64)
    for position in (-1e-6,0.,1e-6,1.-1e-6,1.,1.+1e-6):
        query = torch.tensor([position],dtype=torch.float64,requires_grad=True)
        width = torch.tensor(1.,dtype=torch.float64,requires_grad=True)
        decay = torch.tensor(.5,dtype=torch.float64,requires_grad=True)
        result = temporal_window(times,values,query,width,decay)
        gq,gh = torch.autograd.grad(result.sum(),(query,width))
        assert abs(float(result.detach().sum())) < 2e-12
        assert abs(float(gq)) < 3e-6 and abs(float(gh)) < 3e-6


def test_empty_windows_and_causality():
    times = torch.tensor([0.,1.,2.],dtype=torch.float64)
    values = torch.tensor([[1.,2.],[3.,4.],[5.,6.]],dtype=torch.float64)
    queries = torch.tensor([-.1,.5,3.,100.],dtype=torch.float64)
    width,decay = torch.tensor(1.,dtype=torch.float64),torch.tensor(.4,dtype=torch.float64)
    result = temporal_window(times,values,queries,width,decay)
    torch.testing.assert_close(result,explicit(times,values,queries,width,decay),rtol=1e-9,atol=1e-12)
    changed = values.clone(); changed[1:] *= 1000
    torch.testing.assert_close(temporal_window(times,changed,queries[:2],width,decay),result[:2],rtol=0,atol=0)
