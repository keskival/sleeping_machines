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
