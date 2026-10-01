import torch
from sleeping_machines.threshold_spike_train import threshold_spike_train


def test_multi_spike_times_and_all_parameter_derivatives():
    knots = torch.tensor([0.,.45,1.3,2.5],dtype=torch.float64,requires_grad=True)
    currents = torch.tensor([.2,2.3,1.8],dtype=torch.float64,requires_grad=True)
    threshold = torch.tensor(.4,dtype=torch.float64,requires_grad=True)
    leak = torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    initial = torch.tensor(.1,dtype=torch.float64,requires_grad=True)
    args = knots,currents,threshold,leak,initial
    times,voltage = threshold_spike_train(*args)
    assert len(times)>4 and bool((times[1:]>times[:-1]).all())
    assert 0 <= float(voltage.detach()) < float(threshold.detach())
    assert torch.autograd.gradcheck(threshold_spike_train,args,eps=1e-6,atol=2e-6)


def test_constant_current_closed_form_and_partition_invariance():
    threshold,leak,initial = [torch.tensor(v,dtype=torch.float64) for v in (.5,.8,0.)]
    single = threshold_spike_train(torch.tensor([0.,2.],dtype=torch.float64),
        torch.tensor([3.],dtype=torch.float64),threshold,leak,initial)
    split = threshold_spike_train(torch.tensor([0.,.37,.83,2.],dtype=torch.float64),
        torch.tensor([3.,3.,3.],dtype=torch.float64),threshold,leak,initial)
    for a,b in zip(single,split):torch.testing.assert_close(a,b,rtol=1e-13,atol=1e-13)
    period = -torch.log1p(-leak*threshold/3.)/leak
    torch.testing.assert_close(single[0],period*torch.arange(1,len(single[0])+1,dtype=period.dtype))


def test_silence_and_learnable_window_compose_without_time_steps():
    from sleeping_machines.temporal_window import temporal_window
    knots = torch.tensor([0.,1.],dtype=torch.float64)
    current = torch.tensor([2.],dtype=torch.float64,requires_grad=True)
    threshold,leak,initial = [torch.tensor(v,dtype=torch.float64) for v in (.3,.8,0.)]
    times,_ = threshold_spike_train(knots,current,threshold,leak,initial)
    queries = torch.tensor([.55,.92],dtype=torch.float64)
    width = torch.tensor(.41,dtype=torch.float64,requires_grad=True)
    decay = torch.tensor(.4,dtype=torch.float64)
    def compose(c,w):
        t,_ = threshold_spike_train(knots,c,threshold,leak,initial)
        return temporal_window(t,torch.ones((len(t),1),dtype=t.dtype),queries,w,decay)
    assert torch.autograd.gradcheck(compose,(current,width),eps=1e-6,atol=2e-6)
    assert len(times)>1
    no_spikes,voltage = threshold_spike_train(knots,current.detach()*0,threshold,leak,initial)
    assert len(no_spikes)==0 and float(voltage)==0.


def test_later_spikes_carry_information_absent_from_identical_first_spikes():
    from sleeping_machines.temporal_window import temporal_window
    knots = torch.tensor([0.,.25,.45],dtype=torch.float64)
    threshold,leak,initial = [torch.tensor(v,dtype=torch.float64) for v in (.4,.7,0.)]
    slow = torch.tensor([2.3,.8],dtype=torch.float64,requires_grad=True)
    fast = torch.tensor([2.3,2.],dtype=torch.float64,requires_grad=True)
    a,_ = threshold_spike_train(knots,slow,threshold,leak,initial)
    b,_ = threshold_spike_train(knots,fast,threshold,leak,initial)
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    assert len(a)==1 and len(b)==2
    first_gradient, = torch.autograd.grad(b[0],fast,retain_graph=True)
    second_gradient, = torch.autograd.grad(b[1],fast,retain_graph=True)
    assert float(first_gradient[1]) == 0. and float(second_gradient[1]) < 0.
    query = torch.tensor([.45],dtype=torch.float64)
    width,decay = [torch.tensor(v,dtype=torch.float64) for v in (.18,.4)]
    def read(current):
        t,_ = threshold_spike_train(knots,current,threshold,leak,initial)
        return temporal_window(t,torch.ones((len(t),1),dtype=t.dtype),query,width,decay)
    assert abs(float(read(slow).detach())) < 1e-12
    assert float(read(fast).detach()) > .01
    gradient, = torch.autograd.grad(read(fast).sum(),fast)
    assert abs(float(gradient[1])) > .01
    assert torch.autograd.gradcheck(read,(fast,),eps=1e-6,atol=2e-6)
