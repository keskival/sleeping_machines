import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, 'experiments')
import dvs_batched_le_benchmark as BL  # noqa: E402
import dvs_batched_reg_benchmark as RG  # noqa: E402


def _rows():
    rng = np.random.default_rng(0)
    rows = []
    for j in range(3):
        ev = [((k + 1) * .05, np.r_[rng.standard_normal(32), 0.].astype(np.float32)) for k in range(4)]
        ev.append((1., np.r_[np.zeros(32), 1.].astype(np.float32)))
        rows.append(dict(index=j, target=j, events=ev))
    return rows


def test_noise_touches_only_packet_content_and_is_deterministic():
    rows = _rows()
    a, b = RG.noisy(rows, .3, 5), RG.noisy(rows, .3, 5)
    for r0, r1, r2 in zip(rows, a, b):
        assert np.array_equal(r1['events'][-1][1], r0['events'][-1][1])          # query unchanged
        assert not np.array_equal(r1['events'][0][1], r0['events'][0][1])
        assert all(np.array_equal(x[1], y[1]) for x, y in zip(r1['events'], r2['events']))
    assert RG.noisy(rows, 0., 5) is rows


def test_zero_options_equal_the_batched_driver_and_adamw_patch_is_restored():
    a = SimpleNamespace(payload=8, depth=2, heads=2, pool=2, clock_step=.05, seed=6, route_credit=False, route_races=0,
                        weight_decay=0., input_noise=0., bins=20)
    rows = _rows()
    grads = []
    for fn in (BL.train_window, RG.train_window):
        torch.manual_seed(0); m = BL.make_model(a).double()
        opt = torch.optim.SGD(m.parameters(), lr=0.); fn(m, opt, rows, a, epoch=1)
        grads.append({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
    for n in grads[0]:
        torch.testing.assert_close(grads[1][n], grads[0][n], rtol=0, atol=0)
    original = torch.optim.Adam
    with RG.activate(SimpleNamespace(weight_decay=.01)):
        assert torch.optim.Adam is not original
    assert torch.optim.Adam is original


def test_clip_option_changes_the_clip_and_restores_it():
    a = SimpleNamespace(payload=8, depth=2, heads=2, pool=2, clock_step=.05, seed=6, route_credit=False, route_races=0,
                        weight_decay=0., input_noise=0., bins=20, clip=1e-6)
    rows = _rows(); torch.manual_seed(0); m = BL.make_model(a).double()
    before = [p.detach().clone() for p in m.parameters()]
    opt = torch.optim.SGD(m.parameters(), lr=1.)
    original = torch.nn.utils.clip_grad_norm_
    RG.train_window(m, opt, rows, a, epoch=1)
    assert torch.nn.utils.clip_grad_norm_ is original
    moved = sum(float((p.detach() - b).norm()) for p, b in zip(m.parameters(), before))
    assert moved < 1e-5   # a tiny clip scales the SGD step to almost nothing
