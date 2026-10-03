import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, 'experiments')
import dvs_batched_large_benchmark as LG  # noqa: E402
import dvs_batched_reg_benchmark as RG  # noqa: E402
import dvs_tied_pool_benchmark as T  # noqa: E402


def test_tied_batched_model_shares_maps_and_trains_on_the_batched_path():
    a = SimpleNamespace(payload=8, depth=2, heads=2, pool=4, clock_step=.05, seed=6, route_credit=False, route_races=0,
                        weight_decay=0., input_noise=0., bins=20, clip=1., tie_pools=True)
    m = LG.make_model(a).double()
    pool = m.units[0][0][0]
    assert all(getattr(u, n) is getattr(pool[0], n) for u in pool for n in T.SHARED)
    rng = np.random.default_rng(0)
    rows = [dict(index=j, target=j, events=[((k + 1) * .05, np.r_[rng.standard_normal(32), 0.]) for k in range(4)]
                 + [(1., np.r_[np.zeros(32), 1.])]) for j in range(3)]
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    out = RG.train_window(m, opt, rows, a, epoch=1)
    assert np.isfinite(out['loss_sum']) and pool[0].input.weight.grad.abs().sum() > 0


def test_skip_init_starts_upper_layers_near_identity_and_stays_learnable():
    a = SimpleNamespace(payload=8, depth=4, heads=2, pool=2, clock_step=.05, seed=6, route_credit=False, route_races=0,
                        weight_decay=0., input_noise=0., bins=20, clip=1., tie_pools=False, skip_init_from=2, skip_gate_bias=-4.)
    m = LG.make_model(a).double()
    assert all(float(u.gate.bias.mean()) == -4. for layer in m.units[2:] for h in layer for pool in h for u in pool)
    assert all(float(u.gate.bias.mean()) != -4. for layer in m.units[:2] for h in layer for pool in h for u in pool)
    assert float(m.transport_frequency[2:3].abs().max()) == 0. and float(m.transport_frequency[3].abs().max()) > 0
    rng = np.random.default_rng(0)
    rows = [dict(index=j, target=j, events=[((k + 1) * .05, np.r_[rng.standard_normal(32), 0.]) for k in range(4)]
                 + [(1., np.r_[np.zeros(32), 1.])]) for j in range(3)]
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    RG.train_window(m, opt, rows, a, epoch=1)
    assert m.units[3][0][0][0].gate.bias.grad.abs().sum() > 0   # learnable from the near-closed start
