"""Exact timing-adjoint and nested-model contracts, without optimizer steps."""
import copy
import torch

from sleeping_machines.clock_feature_races import ContentAndClockRaces
from sleeping_machines.clock_feature_event_heads import ClockFeatureLanguageModel
from sleeping_machines.native_stream_language import NativeStreamLanguageModel


def test_clock_only_adjoint_matches_fixed_noise_finite_difference():
    torch.manual_seed(9)
    scores = torch.tensor([[.1,-.8,.5],[.3,.8,-.2],[1.,-.1,.3]],dtype=torch.float64,requires_grad=True)
    values = torch.zeros((3,0),dtype=torch.float64)
    weights = torch.tensor([.7,-.5,.3],dtype=torch.float64)
    noise = torch.get_rng_state()
    _,delays,_ = ContentAndClockRaces.apply(scores,values)
    derivative, = torch.autograd.grad(delays @ weights,scores)
    for i in range(scores.numel()):
        results = []
        for sign in (-1,1):
            shifted = scores.detach().clone();shifted.flatten()[i] += sign*1e-6
            torch.set_rng_state(noise)
            _,time,_ = ContentAndClockRaces.apply(shifted,values)
            results.append(float(time @ weights))
        torch.testing.assert_close(derivative.flatten()[i],scores.new_tensor((results[1]-results[0])/2e-6),rtol=1e-5,atol=1e-10)


def test_content_teacher_conserved_and_confined_to_primary_race():
    torch.manual_seed(10)
    scores = torch.randn(5,3,dtype=torch.float64,requires_grad=True)
    values = torch.randn(3,4,dtype=torch.float64,requires_grad=True)
    output,_,winner = ContentAndClockRaces.apply(scores,values)
    gs,gv = torch.autograd.grad(output.sum(),(scores,values))
    torch.testing.assert_close(gs[0].sum(),scores.new_tensor(0.),rtol=0,atol=1e-12)
    assert not bool(gs[1:].any())
    expected = torch.zeros_like(values);expected[winner] = 1.
    torch.testing.assert_close(gv,expected,rtol=0,atol=0)


def test_zero_clock_model_exactly_nests_parent_rng_values_and_gradients():
    torch.manual_seed(12);parent = NativeStreamLanguageModel(payload=4,depth=2,heads=2)
    torch.manual_seed(12);nested = ClockFeatureLanguageModel(payload=4,depth=2,heads=2,clock_features=0)
    assert list(parent.state_dict()) == list(nested.state_dict())
    tokens = torch.tensor([1,2,3,1])
    rng = torch.get_rng_state()
    a,sa = parent.forward_chunk(tokens);a.square().sum().backward();next_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    b,sb = nested.forward_chunk(tokens);b.square().sum().backward()
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(next_rng,torch.get_rng_state(),rtol=0,atol=0)
    for p,q in zip(parent.parameters(),nested.parameters()):
        if p.grad is None: assert q.grad is None
        else:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    assert sa.storage() == sb.storage()


def test_extra_clocks_keep_content_sparse_causal_and_match_waiting_control():
    torch.manual_seed(13)
    model = ClockFeatureLanguageModel(payload=4,depth=2,heads=2,clock_features=4)
    control = copy.deepcopy(model);control.clock_readout = False
    tokens = torch.tensor([1,2,1,3])
    rng = torch.get_rng_state()
    taught,state = model.forward_chunk(tokens)
    torch.set_rng_state(rng);other,_ = control.forward_chunk(tokens)
    torch.testing.assert_close(taught,other,rtol=0,atol=0)
    torch.set_rng_state(rng);model.eval()
    with torch.no_grad():inferred,st = model.forward_chunk(tokens)
    torch.testing.assert_close(taught,inferred,rtol=0,atol=0)
    assert st.counterfactual_values == 0
    assert state.counterfactual_values == state.candidate_scores == len(tokens)*2*2*2
    assert state.selected_updates == len(tokens)*2*2
    assert state.extra_clock_races == 4*state.selected_updates
    assert state.extra_rate_settings == 4*state.candidate_scores
    assert all(3. < float(t.min()) <= float(t.max()) < 3.+2*.011 for _,t in st.contexts.values())
    model.train();taught.square().sum().backward()
    assert all(bool(layer.weight.grad.abs().sum()>0) for row in model.clock_projection for layer in row)


def test_content_dependent_temporal_reception_has_gradients():
    torch.manual_seed(14)
    model = ClockFeatureLanguageModel(payload=4,depth=2,heads=2,clock_features=2)
    # Read-only nonzero decoder fixture exposes reception gradients without
    # claiming that an untrained zero decoder already teaches the gate.
    with torch.no_grad():
        for row in model.clock_projection:
            for layer in row:layer.weight.copy_(torch.randn_like(layer.weight)*.1)
    logits,_ = model.forward_chunk(torch.tensor([1,2,3,4]))
    logits.square().sum().backward()
    for row in model.clock_reception:
        for layer in row:assert bool(layer.weight.grad.abs().sum()>0)
    assert bool(model.clock_frequency[0].grad.abs().sum()>0)


def test_native_arrival_coordinates_have_no_absolute_time_origin():
    from dataclasses import replace
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
    from native_event_contracts import predict
    from native_event_tasks import episodes
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.clock_feature_event_heads import ClockFeatureEventHeads
    row = episodes('order',2,2,44)[0]
    shifted = [replace(event,time=event.time+100.) for event in row]
    for constructor in (AddressedEventHeads,ClockFeatureEventHeads):
        torch.manual_seed(15)
        model = constructor(sources=2,payload=4,depth=2).eval()
        if hasattr(model,'clock_projection'):
            with torch.no_grad():
                for layer in model.clock_projection:
                    for head in layer:head.weight.normal_(0.,.1)
        rng = torch.get_rng_state()
        with torch.no_grad():a,_,_,ta = predict(model,row)
        torch.set_rng_state(rng)
        with torch.no_grad():b,_,_,tb = predict(model,shifted)
        torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(ta+100.,tb,rtol=0,atol=1e-12)


def test_temporal_projection_rotates_relevance_and_composes_message():
    model = ClockFeatureLanguageModel(payload=4,depth=1,heads=2,clock_features=2).double()
    with torch.no_grad():
        model.clock_reception[0][0].weight.copy_(torch.eye(4,dtype=torch.float64))
        model.clock_projection[0][0].weight.copy_(torch.eye(4,dtype=torch.float64))
    message = torch.tensor([2.,0.,0.,0.],dtype=torch.float64,requires_grad=True)
    first = torch.tensor([1.,0.,0.,0.],dtype=torch.float64)
    rotated = torch.tensor([0.,1.,0.,0.],dtype=torch.float64)
    a = model.temporal_reception(message,first,0,0)
    b = model.temporal_reception(message,rotated,0,0)
    assert float((a-message)[0].detach()) > .5
    torch.testing.assert_close((b-message)[1],message.new_tensor(.5/2**.5))
    torch.testing.assert_close((a-message)[1],message.new_tensor(0.))
    assert torch.autograd.gradcheck(lambda v:model.temporal_reception(v,rotated,0,0),(message,))


def test_late_capacity_matches_uniform_budget_and_preserves_sparse_updates():
    torch.manual_seed(17)
    uniform = ClockFeatureLanguageModel(payload=4,depth=8,heads=2,clock_features=2)
    torch.manual_seed(17)
    late = ClockFeatureLanguageModel(payload=4,depth=8,heads=2,clock_features=4,clock_allocation='late')
    assert uniform.clock_counts == [2]*8 and late.clock_counts == [0]*4+[4]*4
    assert sum(p.numel() for p in uniform.parameters()) == sum(p.numel() for p in late.parameters())
    tokens = torch.tensor([1,2,3])
    for model in (uniform,late):
        logits,state = model.forward_chunk(tokens)
        logits.square().sum().backward()
        assert state.extra_clock_races == len(tokens)*2*16
        assert state.extra_rate_settings == len(tokens)*2*16*2
        assert state.selected_updates == len(tokens)*8*2
        assert state.counterfactual_values == len(tokens)*8*2*2
