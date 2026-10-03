import numpy as np
import torch

from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.batched_episodes import batched_logits
from sleeping_machines.fast_native_core import fast_class
from sleeping_machines.sparse_training import sparse_layer_step, sparse_train_logits


def _case(pool=3, depth=3):
    torch.manual_seed(0)
    m = fast_class(AddressedEventHeads)(sources=1, content_dim=5, classes=5, payload=8, depth=depth, heads=2,
                                        pool=pool).double()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * .1)
    g = np.random.default_rng(1)
    rows = [dict(events=[(float(t) + g.random(), g.normal(size=5)) for t in range(L)]) for L in (7, 3, 9, 1, 9)]
    return m, rows


def _grads(m, rows, fn):
    m.zero_grad()
    z = fn(m, rows)
    (z.square().sum() + z[:, -1, 0].sum()).backward()
    return z.detach().clone(), [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in m.parameters()]


def test_without_credit_equals_batched_path_exactly():
    for pool in (1, 3):
        m, rows = _case(pool)
        ref = _grads(m, rows, lambda m, r: batched_logits(m, r, 11, all_logits=True))
        got = _grads(m, rows, lambda m, r: sparse_train_logits(m, r, 11, all_logits=True, step=sparse_layer_step))
        assert torch.allclose(got[0], ref[0], rtol=1e-11, atol=1e-11)
        for a, b in zip(got[1], ref[1]):
            assert torch.allclose(a, b, rtol=1e-9, atol=1e-11)


def test_sampled_credit_is_unbiased_for_the_linear_credit():
    m, rows = _case(4, 2)
    lin = _grads(m, rows, lambda m, r: batched_logits(m, r, 11, all_logits=True, route_credit='linear'))
    base = _grads(m, rows, lambda m, r: batched_logits(m, r, 11, all_logits=True))
    acc = None; M = 400
    for s in range(M):
        z, g = _grads(m, rows, lambda m, r: sparse_train_logits(m, r, 11, all_logits=True, route_credit='sampled',
                                                                  step=sparse_layer_step, alt_seed=1000 + s))
        assert torch.allclose(z, lin[0], rtol=1e-11, atol=1e-11)              # forward unchanged
        acc = g if acc is None else [x + y for x, y in zip(acc, g)]
    mean = [x / M for x in acc]
    credit_lin = torch.cat([(a - b).flatten() for a, b in zip(lin[1], base[1])])
    credit_mc = torch.cat([(a - b).flatten() for a, b in zip(mean, base[1])])
    assert credit_lin.norm() > 1e-6
    assert (credit_mc - credit_lin).norm() / credit_lin.norm() < .15                # Monte-Carlo error of 400 draws
