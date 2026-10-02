import torch

from sleeping_machines.dilated_delay_taps import TappedNativeStreamLanguageModel
from sleeping_machines.native_stream_language import NativeStreamLanguageModel


def _models():
    torch.manual_seed(0)
    plain = NativeStreamLanguageModel(payload=4, depth=3, pool=2, heads=2).double()
    torch.manual_seed(0)
    tapped = TappedNativeStreamLanguageModel(payload=4, depth=3, pool=2, heads=2).double()
    tapped.load_state_dict({**tapped.state_dict(), **plain.state_dict()})
    return plain, tapped


def _run(model, tokens, chunks):
    with torch.random.fork_rng():
        torch.manual_seed(1)
        state, out, s = model.new_state(), [], 0
        for size in chunks:
            z, state = model.forward_chunk(tokens[s:s + size], state); out.append(z); s += size
            state = state.detach()
    return torch.cat(out)


def test_zero_taps_nest_the_native_core_exactly():
    plain, tapped = _models()
    tokens = torch.randint(0, 27, (30,), generator=torch.Generator().manual_seed(2))
    assert torch.equal(_run(plain, tokens, (30,)), _run(tapped, tokens, (30,)))


def test_tapped_forward_is_chunk_invariant_and_delays_learn():
    _, tapped = _models()
    with torch.no_grad():
        for m in tapped.channel_mix:
            m.tap.weight.normal_(0, .3); m.raw_delay.add_(.37)  # non-integer delays
    tokens = torch.randint(0, 27, (40,), generator=torch.Generator().manual_seed(3))
    whole = _run(tapped, tokens, (40,))
    assert torch.allclose(whole, _run(tapped, tokens, (16, 16, 8)), atol=1e-10)
    tapped.train()
    with torch.random.fork_rng():
        torch.manual_seed(1)
        z, _ = tapped.forward_chunk(tokens[:-1])
    torch.nn.functional.cross_entropy(z, tokens[1:]).backward()
    for m in tapped.channel_mix[:2]:  # deepest delay may exceed the 39-event stream
        assert m.raw_delay.grad is not None and m.raw_delay.grad.abs() > 0
        assert m.tap.weight.grad.abs().sum() > 0


def test_integer_delay_reads_the_layer_input_that_many_events_back():
    _, tapped = _models()
    m = tapped.channel_mix[1]
    with torch.no_grad():
        m.raw_delay.copy_(torch.tensor(3.).expm1().log())  # tau = 3
    state = tapped.new_state(); tapped._tap_state = state
    xs = [torch.randn(8, dtype=torch.float64) for _ in range(5)]
    for x in xs[:4]:
        m(x)
    with torch.no_grad():
        m.tap.weight.copy_(torch.eye(8))
    out = m(xs[4])
    assert torch.allclose(out - torch.nn.functional.linear(xs[4], m.weight), xs[1], atol=1e-9)
