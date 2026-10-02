import numpy as np
import torch

from sleeping_machines import count_carrying_language as M
from sleeping_machines import count_escape_gate as G


def _pair(orders=2):
    torch.manual_seed(0)
    plain = M.CountCarryingNativeModel(payload=4, depth=2, pool=2, heads=2, orders=orders).double()
    torch.manual_seed(0)
    gated = G.GatedCountCarryingNativeModel(payload=4, depth=2, pool=2, heads=2, orders=orders).double()
    gated.load_state_dict({**gated.state_dict(), **plain.state_dict()})
    rng = np.random.default_rng(4)
    tokens = torch.tensor(rng.integers(0, M.A, 40))
    counts = M.eval_stream_counts(rng.integers(0, M.A, 200), tokens.numpy(), orders)
    for m in (plain, gated):
        m.register_stream('s', counts); m.use_stream('s')
    return plain, gated, tokens


def _run(model, tokens, chunks=(40,)):
    with torch.random.fork_rng():
        torch.manual_seed(1)
        state, out, s = model.new_state(), [], 0
        for size in chunks:
            o, state = model.forward_chunk(tokens[s:s + size], state); out.append(o); s += size
    return torch.cat(out)


def test_zero_gate_nests_scalar_escape_exactly():
    plain, gated, tokens = _pair()
    assert torch.allclose(_run(plain, tokens), _run(gated, tokens), atol=1e-12, rtol=0)


def test_gated_cascade_is_normalized_and_chunk_invariant():
    _, gated, tokens = _pair()
    with torch.no_grad():
        gated.escape_gate.weight.normal_(0, .5); gated.count_message.weight.normal_(0, .1)
    whole = _run(gated, tokens)
    assert torch.allclose(whole.exp().sum(-1), torch.ones(40, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(whole, _run(gated, tokens, (16, 16, 8)), atol=1e-10)


def test_gate_and_base_receive_gradient():
    _, gated, tokens = _pair()
    lp = _run(gated, tokens)
    (-lp[:-1].gather(1, tokens[1:, None]).sum()).backward()
    assert gated.escape_gate.weight.grad.abs().sum() > 0
    assert gated.count_message.weight.grad.abs().sum() > 0
    assert gated.embedding.weight.grad.abs().sum() > 0


def test_composed_wrapper_nests_exactly():
    from sleeping_machines.count_composed_stream import CountComposedModel
    base = M.CountCarryingNativeModel(payload=4, depth=2, pool=2, heads=2, orders=1).double()
    plain, gated = CountComposedModel(base, 2).double(), G.GatedCountComposedModel(base, 2).double()
    rng = np.random.default_rng(5)
    tokens = torch.tensor(rng.integers(0, M.A, 30))
    counts = M.eval_stream_counts(rng.integers(0, M.A, 100), tokens.numpy(), 2)
    base.register_stream('s', M.eval_stream_counts(rng.integers(0, M.A, 100), tokens.numpy(), 1)); base.use_stream('s')
    for m in (plain, gated):
        m.register_stream('s', counts); m.use_stream('s')
    assert torch.allclose(_run(plain, tokens, (30,)), _run(gated, tokens, (30,)), atol=1e-12, rtol=0)


def test_each_repair_can_be_disabled_and_nests():
    plain, _, tokens = _pair()
    for flags in (dict(escape_gate=False), dict(count_message=False), dict(escape_gate=False, count_message=False)):
        torch.manual_seed(0)
        m = G.GatedCountCarryingNativeModel(payload=4, depth=2, pool=2, heads=2, orders=2, **flags).double()
        m.load_state_dict({**m.state_dict(), **plain.state_dict()})
        m.streams, m.active = plain.streams, plain.active
        assert torch.allclose(_run(plain, tokens), _run(m, tokens), atol=1e-12, rtol=0)


def test_base_only_diagnostic_returns_the_base_predictive():
    _, gated, tokens = _pair()
    gated.base_only = True
    lp = _run(gated, tokens)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(40, dtype=torch.float64), atol=1e-10)
