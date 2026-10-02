import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, 'experiments')
import dvs_fork_replay as FR  # noqa: E402
import dvs_local_expectation_benchmark as LE  # noqa: E402
from sleeping_machines.shadow_lanes import shadow_losses  # noqa: E402


def test_every_shadow_lane_equals_its_sequential_forked_replay():
    for depth, pool in ((2, 2), (3, 3)):
        a = SimpleNamespace(payload=8, depth=depth, heads=2, pool=pool, clock_step=.05, seed=6)
        m = LE.make_model(a, fast=True).double(); m.train()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * .05)
        rng = np.random.default_rng(1)
        row = dict(index=2, target=5, events=[((k + 1) * .05, rng.standard_normal(33)) for k in range(6)])
        record = []
        loss, races, _, snaps = FR.realized(m, row, 23, record)
        forces = [None] + [(r, i) for r in (0, races // 3, races - 1) for i in range(pool)]
        lanes = shadow_losses(m, row, 23, forces)
        assert abs(float(lanes[0]) - float(loss)) < 1e-10
        for k, f in enumerate(forces[1:], 1):
            assert abs(float(lanes[k]) - float(FR.forked(m, row, snaps, f)[0])) < 1e-10, (depth, pool, f)
