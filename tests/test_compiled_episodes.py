import numpy as np
import torch

from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.batched_episodes import batched_logits
from sleeping_machines.compiled_episodes import compiled_logits, layer_step
from sleeping_machines.fast_native_core import fast_class


def _case(dtype, skip=False):
    torch.manual_seed(0)
    m = fast_class(AddressedEventHeads)(sources=1, content_dim=5, classes=5, payload=8, depth=3, heads=2, pool=3).to(dtype)
    if skip:   # near-identity gates and closed intermediate transport, as the language/DVS skip-init arms use
        with torch.no_grad():
            for d in range(1, 3):
                for h in m.units[d]:
                    for pool in h:
                        for u in pool:
                            u.gate.bias.fill_(-4.)
            m.transport_rate[1:-1] = -20.
    g = np.random.default_rng(1)
    rows = [dict(events=[(float(t) + g.random(), g.normal(size=5).astype(np.float32)) for t in range(L)]) for L in (7, 3, 9, 1, 9)]
    return m, rows


def _run(m, rows, fn):
    m.zero_grad()
    z = fn(m, rows, 123)
    (z.square().sum() + z[:, -1, 0].sum()).backward()
    return z.detach().clone(), [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in m.parameters()]


def _close(a, b, rtol):
    za, ga = a; zb, gb = b
    assert torch.allclose(za, zb, rtol=rtol, atol=rtol)
    for x, y in zip(ga, gb):
        assert torch.allclose(x, y, rtol=rtol, atol=rtol * max(1., float(y.abs().max())))


def test_traceable_race_formulation_matches_batched_episodes_float64():
    for skip in (False, True):
        m, rows = _case(torch.float64, skip)
        ref = _run(m, rows, lambda m, r, s: batched_logits(m, r, s, all_logits=True))
        eager = _run(m, rows, lambda m, r, s: compiled_logits(m, r, s, all_logits=True, step=layer_step))
        _close(eager, ref, 1e-10)


def test_compiled_step_matches_batched_episodes():
    m, rows = _case(torch.float64)
    ref = _run(m, rows, lambda m, r, s: batched_logits(m, r, s, all_logits=True))
    comp = _run(m, rows, lambda m, r, s: compiled_logits(m, r, s, all_logits=True))
    _close(comp, ref, 1e-9)
    m, rows = _case(torch.float32)
    ref = _run(m, rows, lambda m, r, s: batched_logits(m, r, s, all_logits=True))
    comp = _run(m, rows, lambda m, r, s: compiled_logits(m, r, s, all_logits=True))
    _close(comp, ref, 2e-4)
    final = compiled_logits(m, rows, 9)
    assert torch.allclose(final, batched_logits(m, rows, 9), rtol=2e-4, atol=2e-4)


def test_linear_route_credit_leaves_values_and_value_gradients_and_adds_score_credit():
    from sleeping_machines.batched_episodes import linear_route_credit
    torch.manual_seed(3)
    s = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(4, 3, 5, dtype=torch.float64, requires_grad=True)
    g = torch.randn(4, 5, dtype=torch.float64)
    z = linear_route_credit(s, v)
    assert torch.equal(z, torch.zeros_like(z))
    (z * g).sum().backward()
    pi = torch.softmax(s.detach(), -1)
    vbar = (pi[..., None] * v.detach()).sum(1, keepdim=True)
    expected = pi * ((v.detach() - vbar) * g[:, None]).sum(-1)
    assert torch.allclose(s.grad, expected, atol=1e-14) and v.grad is None or torch.equal(v.grad, torch.zeros_like(v))


def test_route_credit_forward_unchanged_and_compiled_matches_eager():
    m, rows = _case(torch.float64)
    plain = batched_logits(m, rows, 123, all_logits=True)
    ref = _run(m, rows, lambda m, r, s: batched_logits(m, r, s, all_logits=True, route_credit='linear'))
    assert torch.equal(ref[0], plain.detach())
    eager = _run(m, rows, lambda m, r, s: compiled_logits(m, r, s, all_logits=True, step=layer_step, route_credit='linear'))
    _close(eager, ref, 1e-10)
    comp = _run(m, rows, lambda m, r, s: compiled_logits(m, r, s, all_logits=True, route_credit='linear'))
    _close(comp, ref, 1e-9)
    base = _run(m, rows, lambda m, r, s: batched_logits(m, r, s, all_logits=True))
    keys = [i for i, (n, _) in enumerate(m.named_parameters()) if n.endswith('.key') or 'queries' in n]
    assert any(not torch.allclose(ref[1][i], base[1][i]) for i in keys)        # routing now receives value credit


def test_linear_write_credit_is_zero_and_has_the_expected_score_gradient():
    from sleeping_machines.batched_episodes import linear_write_credit
    torch.manual_seed(4)
    s = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    old, new = torch.randn(4, 3, 5, dtype=torch.float64), torch.randn(4, 3, 5, dtype=torch.float64)
    G = torch.randn(4, 3, 5, dtype=torch.float64)
    active = torch.tensor([True, True, False, True])
    z = linear_write_credit(s, old, new, active)
    assert torch.equal(z, torch.zeros_like(z))
    (z * G).sum().backward()
    pi = torch.softmax(s.detach(), -1); gd = (G * (new - old)).sum(-1)
    expected = pi * (gd - (pi * gd).sum(-1, keepdim=True)) * active[:, None]
    assert torch.allclose(s.grad, expected, atol=1e-14)


def test_read_write_route_credit_compiled_matches_eager():
    m, rows = _case(torch.float64)
    plain = batched_logits(m, rows, 123, all_logits=True)
    ref = _run(m, rows, lambda m, r, s: batched_logits(m, r, s, all_logits=True, route_credit='linear_rw'))
    assert torch.equal(ref[0], plain.detach())
    eager = _run(m, rows, lambda m, r, s: compiled_logits(m, r, s, all_logits=True, step=layer_step, route_credit='linear_rw'))
    _close(eager, ref, 1e-10)
    comp = _run(m, rows, lambda m, r, s: compiled_logits(m, r, s, all_logits=True, route_credit='linear_rw'))
    _close(comp, ref, 1e-9)
    lin = _run(m, rows, lambda m, r, s: batched_logits(m, r, s, all_logits=True, route_credit='linear'))
    keys = [i for i, (n, _) in enumerate(m.named_parameters()) if n.endswith('.key') or 'queries' in n]
    assert any(not torch.allclose(ref[1][i], lin[1][i]) for i in keys)          # write credit adds score credit


def test_written_content_route_credit_compiled_matches_eager():
    m, rows = _case(torch.float64)
    plain = batched_logits(m, rows, 123, all_logits=True)
    ref = _run(m, rows, lambda m, r, s: batched_logits(m, r, s, all_logits=True, route_credit='linear_rwn'))
    assert torch.equal(ref[0], plain.detach())
    eager = _run(m, rows, lambda m, r, s: compiled_logits(m, r, s, all_logits=True, step=layer_step, route_credit='linear_rwn'))
    _close(eager, ref, 1e-10)
    comp = _run(m, rows, lambda m, r, s: compiled_logits(m, r, s, all_logits=True, route_credit='linear_rwn'))
    _close(comp, ref, 1e-9)
    rw = _run(m, rows, lambda m, r, s: batched_logits(m, r, s, all_logits=True, route_credit='linear_rw'))
    keys = [i for i, (n, _) in enumerate(m.named_parameters()) if n.endswith('.key')]
    assert any(not torch.allclose(ref[1][i], rw[1][i]) for i in keys)
