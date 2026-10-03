import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, 'experiments')
import dvs_clock_calibrated_benchmark as C  # noqa: E402
import dvs_grow_depth_benchmark as G  # noqa: E402


def _args(depth, pool=1):
    return SimpleNamespace(payload=8, depth=depth, heads=2, pool=pool, clock_step=.05, seed=6)


def _run(model, events, seed=1):
    model.eval(); state = model.new_state(); out = []
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(seed)
        for t, x in events:
            out.append(model.consume_event(0, t, x, state)[0])
    return torch.stack(out)


def test_grown_model_starts_near_the_parent_and_keeps_parent_weights():
    parent = C.make_model(_args(2), fast=False)
    with torch.no_grad():
        for p in parent.parameters():
            p.add_(torch.randn_like(p) * .05)
    grown = G.grow(C.make_model(_args(4), fast=False), parent.state_dict(), 2, appended_gate_bias=G.CLOSED_GATE)  # legacy
    for name, value in parent.state_dict().items():
        parts = name.split('.')
        if parts[0] in ('transport_rate', 'transport_frequency'):
            assert torch.equal(grown.state_dict()[name][:2], value)
        else:
            assert torch.equal(grown.state_dict()[name], value), name
    events = [((k + 1) * .05, torch.randn(33, generator=torch.Generator().manual_seed(k))) for k in range(20)]
    a, b = _run(parent, events), _run(grown, events)
    # appended layers shift the race-noise stream of the parent's layers; the deviation must stay at the level of a
    # race-seed change of the parent itself (the noise floor), not beyond it
    floor = (a - _run(parent, events, seed=2)).abs().max()
    assert (a - b).abs().max() < 1.5 * floor, (float((a - b).abs().max()), float(floor))
    assert all(float(u.gate.bias.mean()) == G.CLOSED_GATE for layer in grown.units[2:] for h in layer for pool in h for u in pool)
    assert all(abs(u.gain - .5 / 2 ** .5) < 1e-12 for layer in grown.units[:2] for h in layer for pool in h for u in pool)


def test_new_layers_receive_credit():
    grown = G.grow(C.make_model(_args(4, pool=2), fast=False), C.make_model(_args(2, pool=2), fast=False).state_dict(), 2)
    grown.train(); state = grown.new_state()
    with torch.random.fork_rng():
        torch.manual_seed(2)
        for k in range(10):
            z, _ = grown.consume_event(0, (k + 1) * .05, torch.randn(33, generator=torch.Generator().manual_seed(k)), state)
    torch.nn.functional.cross_entropy(z[None], torch.tensor([3])).backward()
    assert any(u.gate.bias.grad is not None and u.gate.bias.grad.abs().sum() > 0
               for layer in grown.units[2:] for h in layer for pool in h for u in pool)


def test_appended_gates_default_live_and_lineage_gains_survive_progressive_growth(tmp_path, monkeypatch):
    import json
    grown = G.grow(C.make_model(_args(4), fast=False), C.make_model(_args(2), fast=False).state_dict(), 2)
    assert all(float(u.gate.bias.mean()) == G.APPENDED_GATE_BIAS for layer in grown.units[2:] for h in layer for pool in h for u in pool)
    root = tmp_path; monkeypatch.setattr(G, 'ROOT', root)
    (root / 'd2.json').write_text(json.dumps(dict(args=dict(depth=2))))
    (root / 'd4.json').write_text(json.dumps(dict(args=dict(depth=4, parent='d2.json'))))
    gains = G.lineage_gains('d4.json')
    assert gains[:2] == [.5 / 2 ** .5] * 2 and gains[2:] == [.25, .25]
    d6 = G.grow(C.make_model(_args(6), fast=False), grown.state_dict(), 4, gains)
    assert [d6.units[l][0][0][0].gain for l in range(6)][:4] == gains
