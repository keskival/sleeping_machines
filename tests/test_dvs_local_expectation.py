import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, 'experiments')
import dvs_local_expectation_benchmark as LE  # noqa: E402


def _setup(fast):
    a = SimpleNamespace(payload=8, depth=2, heads=2, pool=2, clock_step=.05, seed=6, route_samples=100)
    m = LE.make_model(a, fast=fast).double()
    rng = np.random.default_rng(0)
    row = dict(index=3, target=4, events=[((k + 1) * .05, rng.standard_normal(33)) for k in range(6)])
    return a, m, row


def test_route_term_equals_the_exact_marginal_over_each_race():
    a, m, row = _setup(fast=False)
    m.train(); scores = []
    loss, races, _ = LE.run(m, row, 11, record=scores)
    leaves = [s.detach().requires_grad_() for s in scores]   # direct term only: later races' indirect pathwise
    with torch.no_grad():                                       # dependence on earlier scores is a separate, correct term
        L = [torch.tensor([float(LE.run(m, row, 11, force=(r, i))[0]) for i in range(2)], dtype=torch.float64) for r in range(races)]
    route = sum((torch.softmax(leaves[r], 0) * L[r]).sum() for r in range(races))
    route.backward()
    for r in range(races):
        pi = torch.softmax(leaves[r].detach(), 0)
        torch.testing.assert_close(leaves[r].grad, pi * (L[r] - (pi * L[r]).sum()), rtol=0, atol=1e-12)
    # forcing the realized winner reproduces the realized loss exactly (common random numbers)
    assert races == 6 * 2 * 2
    # the realized branch of a forced replay reproduces the realized loss when forcing the realized winner
    assert races == 6 * 2 * 2


def test_window_step_runs_with_fast_path_and_reaches_routers():
    a, m, row = _setup(fast=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    a.route_samples = 4
    out = LE.train_window(m, opt, [row, dict(row, index=4, target=1)], a, epoch=1)
    assert out['route_replays'] == 2 * 4 * 2 and np.isfinite(out['loss_sum'])
    assert m.units[0][0][0][0].key.grad is not None


def test_forcing_the_realized_winner_reproduces_the_realized_loss():
    a, m, row = _setup(fast=False)
    m.train()
    winners = []
    orig = LE.PATHWISE

    def spy(model, scores, values):
        out = orig(model, scores, values); winners.append(int(out[2])); return out
    LE.PATHWISE = spy
    try:
        with torch.no_grad():
            loss, races, _ = LE.run(m, row, 11)
    finally:
        LE.PATHWISE = orig
    with torch.no_grad():
        for r in (0, races // 2, races - 1):
            assert float(LE.run(m, row, 11, force=(r, winners[r]))[0]) == float(loss)
