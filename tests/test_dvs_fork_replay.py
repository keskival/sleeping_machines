import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, 'experiments')
import dvs_fork_replay as FR  # noqa: E402
import dvs_local_expectation_benchmark as LE  # noqa: E402


def test_forked_replay_equals_the_full_replay_exactly_and_replays_only_the_suffix():
    for fast in (False, True):
        a = SimpleNamespace(payload=8, depth=2, heads=2, pool=2, clock_step=.05, seed=6)
        m = LE.make_model(a, fast=fast).double(); m.train()
        rng = np.random.default_rng(0)
        row = dict(index=1, target=3, events=[((k + 1) * .05, rng.standard_normal(33)) for k in range(7)])
        record = []
        loss, races, _, snaps = FR.realized(m, row, 17, record)
        full_loss = LE.run(m, row, 17)[0]
        assert float(loss) == float(full_loss)
        steps = 0
        for r in range(races):
            for i in range(2):
                with torch.no_grad():
                    expected = float(LE.run(m, row, 17, force=(r, i))[0])
                got, n = FR.forked(m, row, snaps, (r, i))
                assert float(got) == expected, (fast, r, i)
                steps += n
        assert steps < 2 * races * len(row['events']) * .6   # suffix replays: well under full-episode cost
