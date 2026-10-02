import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, 'experiments')
import dvs_critic_le_benchmark as CR  # noqa: E402
import dvs_local_expectation_benchmark as LE  # noqa: E402


def _setup():
    a = SimpleNamespace(payload=8, depth=2, heads=2, pool=2, clock_step=.05, seed=6, route_samples=100,
                        critic_width=8, critic_lr=.01)
    m = LE.make_model(a, fast=False).double()
    rng = np.random.default_rng(0)
    rows = [dict(index=j, target=j % 11, events=[((k + 1) * .05, rng.standard_normal(33)) for k in range(5)]) for j in range(2)]
    return a, m, rows


def _grads(m, a, rows, fn):
    torch.manual_seed(0); CR.CRITIC.clear(); m.zero_grad()
    opt = torch.optim.SGD(m.parameters(), lr=0.)                 # zero step: compare accumulated gradients only
    fn(m, opt, rows, a, epoch=1)
    return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def test_critic_terms_cancel_when_every_race_is_corrected():
    a, m, rows = _setup()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * .05)
    full = _grads(m, a, rows, LE.train_window)        # replay credit on every race (route_samples >= races)
    crit = _grads(m, a, rows, CR.train_window)        # critic + replay correction on every race
    for n in full:
        torch.testing.assert_close(crit[n], full[n], rtol=1e-6, atol=1e-7, msg=n)   # LE replays are float32


def test_critic_is_cross_fitted_after_the_model_update():
    a, m, rows = _setup()
    a.route_samples = 2
    CR.CRITIC.clear()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    out = CR.train_window(m, opt, rows, a, epoch=1)
    net = CR.CRITIC[id(m)][0]
    assert out['route_replays'] == 2 * 2 * 2 and all(p.grad is not None for p in net.parameters())


def test_forked_replays_give_identical_gradients():
    a, m, rows = _setup()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * .05)
    a.route_samples = 3
    a.fork = False; plain = _grads(m, a, rows, CR.train_window)
    a.fork = True; forked = _grads(m, a, rows, CR.train_window)
    for n in plain:
        torch.testing.assert_close(forked[n], plain[n], rtol=0, atol=1e-12, msg=n)
