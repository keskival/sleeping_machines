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
