import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, 'experiments')
import dvs_batched_le_benchmark as BL  # noqa: E402
import dvs_critic_le_benchmark as CR  # noqa: E402
import dvs_local_expectation_benchmark as LE  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402
from sleeping_machines.factorized_race import factorized_race  # noqa: E402

LE.PATHWISE = factorized_race   # the sequential replay driver's realized-branch race (set by LE.make_model)


def _setup():
    a = SimpleNamespace(payload=8, depth=2, heads=2, pool=2, clock_step=.05, seed=6, route_credit=True,
                        route_samples=10 ** 6, critic_width=8, critic_lr=.01, fork=True, lanes=False, no_critic=True)
    m = BL.make_model(a).double()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * .05)
    rng = np.random.default_rng(2)
    rows = [dict(index=j, target=j % 11, events=[((k + 1) * .05, rng.standard_normal(33)) for k in range(int(rng.integers(3, 6)))])
            for j in range(4)]
    return a, m, rows


def test_batched_evaluation_equals_sequential_evaluation():
    a, m, rows = _setup()
    seq, bat = N.evaluate(m, rows), BL.evaluate(m, rows)
    assert seq['accuracy'] == bat['accuracy'] and abs(seq['nll'] - bat['nll']) < 1e-10
    np.testing.assert_allclose(bat['per_target_nll'], seq['per_target_nll'], rtol=0, atol=1e-10)


def test_batched_all_race_credit_equals_sequential_all_race_credit():
    a, m, rows = _setup()
    opt = torch.optim.SGD(m.parameters(), lr=0.)
    m.zero_grad(); CR.CRITIC.clear(); CR.train_window(m, opt, rows, a, epoch=1)
    seq = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(); BL.train_window(m, opt, rows, a, epoch=1)
    for n, g in seq.items():
        torch.testing.assert_close(dict(m.named_parameters())[n].grad, g, rtol=1e-7, atol=1e-9, msg=n)
