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
