import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, 'experiments')
import dvs_tied_pool_benchmark as T  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402


def _model(pool, fast=True, tied=True):
    torch.manual_seed(0)
    m = (fast_class(AddressedEventHeads) if fast else AddressedEventHeads)(
        sources=1, content_dim=33, classes=11, payload=4, depth=2, heads=2, pool=pool).double()
    return T.tie_pools(m) if tied else m


def test_tying_shares_maps_but_keeps_routing_identity_and_state_private():
    m = _model(4)
    pool = m.units[0][0][0]
    for name in T.SHARED:
        assert all(getattr(u, name) is getattr(pool[0], name) for u in pool)
    assert len({id(u.key) for u in pool}) == 4 and len({id(u.raw_rate) for u in pool}) == 4
    untied = sum(p.numel() for p in _model(4, tied=False).parameters())
    tied = sum(p.numel() for p in m.parameters())
    base = sum(p.numel() for p in _model(1, tied=False).parameters())
    assert tied < untied and tied - base < .2 * (untied - base)   # extra receivers add only keys, biases and timescales


def test_tied_fast_path_equals_tied_reference_and_shared_maps_learn_from_every_winner():
    ref, fast = _model(4, fast=False), _model(4, fast=True)
    fast.load_state_dict(ref.state_dict())
    events = [(float(i) * .3, torch.randn(33, dtype=torch.float64, generator=torch.Generator().manual_seed(i)))
              for i in range(12)]
    outs = []
    for m in (ref, fast):
        m.train(); m.zero_grad()
        if hasattr(m, '_fast_layers'):
            m._fast_layers = {}
        with torch.random.fork_rng():
            torch.manual_seed(3); state = m.new_state(); z = []
            for t, x in events:
                z.append(m.consume_event(0, t, x, state)[0])
        if hasattr(m, '_fast_layers'):
            m._fast_layers = None
        torch.nn.functional.cross_entropy(torch.stack(z), torch.arange(12) % 11).backward(); outs.append(torch.stack(z).detach())
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=1e-11)
    for (name, a), b in zip(ref.named_parameters(), fast.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(b.grad, a.grad, rtol=1e-9, atol=1e-11, msg=name)
    assert ref.units[0][0][0][0].input.weight.grad.abs().sum() > 0
