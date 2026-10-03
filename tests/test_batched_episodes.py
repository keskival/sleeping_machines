import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, 'experiments')
import dvs_local_expectation_benchmark as LE  # noqa: E402
from sleeping_machines.batched_episodes import batched_logits  # noqa: E402


def _rows(n, rng):
    rows = []
    for j in range(n):
        length = int(rng.integers(3, 8))
        rows.append(dict(index=j, target=int(rng.integers(11)),
                         events=[((k + 1) * .05 * (1 + .3 * j), rng.standard_normal(33)) for k in range(length)]))
    return rows


def test_batched_episodes_equal_sequential_runs_in_loss_and_summed_gradient():
    a = SimpleNamespace(payload=8, depth=2, heads=2, pool=2, clock_step=.05, seed=6)
    m = LE.make_model(a, fast=True).double(); m.train()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * .05)
    rows = _rows(5, np.random.default_rng(3))
    m.zero_grad(); seq = []
    for r in rows:
        loss, _, _ = LE.run(m, r, 41); loss.backward(); seq.append(float(loss))
    g_seq = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    logits = batched_logits(m, rows, 41)
    losses = F.cross_entropy(logits, torch.tensor([r['target'] for r in rows]), reduction='none')
    losses.sum().backward()
    np.testing.assert_allclose(losses.detach().numpy(), seq, rtol=0, atol=1e-10)
    for n, g in g_seq.items():
        torch.testing.assert_close(dict(m.named_parameters())[n].grad, g, rtol=1e-8, atol=1e-10, msg=n)


def test_all_logits_equal_sequential_per_event_logits():
    from sleeping_machines.fast_native_core import fast_class
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.factorized_race import factorized_race
    torch.manual_seed(0)
    m = fast_class(AddressedEventHeads)(sources=1, content_dim=27, classes=27, payload=4, depth=2, heads=2, pool=2).double()
    m.train()
    rng = np.random.default_rng(1)
    segs = [rng.integers(0, 27, 9) for _ in range(3)]
    rows = [dict(events=[(float(t), np.eye(27)[c]) for t, c in enumerate(seg)]) for seg in segs]
    allz = batched_logits(m, rows, 77, all_logits=True)
    for j, r in enumerate(rows):
        m.race = factorized_race; m._fast_layers = {}
        try:
            with torch.random.fork_rng():
                torch.manual_seed(77); st = m.new_state()
                seq = torch.stack([m.consume_event(0, t, torch.tensor(c), st)[0] for t, c in r['events']])
        finally:
            del m.race; m._fast_layers = None
        torch.testing.assert_close(allz[j], seq, rtol=0, atol=1e-10)
