"""Checks the claims the experiments rest on.

    python -m pytest tests/ -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from sleeping_machines.energy import PROFILES, break_even_synop_pj, energy_joules  # noqa: E402
from sleeping_machines.sim import Engine  # noqa: E402
import e5_capacity as E5  # noqa: E402
import e6_hidden as E6  # noqa: E402


# ── Event engine ──────────────────────────────────────────────────────────────

def test_engine_delivers_in_time_order_and_skips_cancelled():
    e, seen = Engine(), []
    e.on("x", seen.append)
    e.schedule(0.3, "x", "c")
    b = e.schedule(0.2, "x", "b")
    e.schedule(0.1, "x", "a")
    e.cancel(b)
    e.run()
    assert seen == ["a", "c"]
    assert e.work["scheduled"] == 3 and e.work["cancelled"] == 1 and e.work["fired"] == 2


def test_engine_refuses_the_past_and_keeps_fifo_for_ties():
    e, seen = Engine(), []
    e.on("x", seen.append)
    with pytest.raises(ValueError):
        e.schedule(-0.1, "x")
    for tag in "abc":
        e.schedule(0.5, "x", tag)
    e.run()
    assert seen == ["a", "b", "c"]


def test_engine_run_until_leaves_later_events_pending():
    e, seen = Engine(), []
    e.on("x", seen.append)
    e.schedule(0.2, "x", 1)
    e.schedule(0.8, "x", 2)
    e.run(until=0.5)
    assert seen == [1] and e.now == 0.5
    e.run()
    assert seen == [1, 2]


# ── E6: closed-form races agree with the discrete-event engine ────────────────

def _latency_batch(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.random((n, 784)) * (rng.random((n, 784)) < 0.2)).astype(np.float32)
    x[:, :40] = np.where(rng.random((n, 40)) < 0.5, 1.0, x[:, :40])    # saturated pixels create ties at t = 0
    return E6.latency_code(x)


@pytest.mark.parametrize("kw", [
    dict(psp="step"),
    dict(psp="step", deadline=1, lateral=1, theta_out=0.3),
    dict(psp="step", patch=10, stride=3),
    dict(psp="ramp"),
    dict(psp="ramp", deadline=1),
    dict(psp="ramp", patch=10, stride=3),
    dict(psp="ramp", fanin=40),
])
def test_closed_form_matches_event_engine(kw):
    xm = _latency_batch()
    if kw["psp"] == "ramp":
        drive = np.where(np.isfinite(xm), E6.HORIZON - xm, 0).mean(0)
    else:
        drive = np.isfinite(xm).mean(0)
    net = E6.RaceNet(E6.Config(hidden=120, winners=3, hid_frac=0.6, **kw), 784, 10, float(drive.sum()),
                     np.random.default_rng(1), drive)
    assert E6.test_equivalence(net, xm, len(xm)) == 1.0


def test_training_step_changes_only_existing_patch_synapses():
    xm = _latency_batch(32)
    drive = np.isfinite(xm).mean(0)
    net = E6.RaceNet(E6.Config(hidden=100, winners=3, patch=10, stride=3), 784, 10, float(drive.sum()),
                     np.random.default_rng(2), drive)
    before = net.W1.copy()
    t, idx = E6.to_events(xm)
    net.teach(net.forward(t, idx), np.arange(32) % 10)
    changed = net.W1 != before
    assert changed.any()
    assert not (changed & ~net.M1).any()


def test_growth_keeps_fanin_and_connects_only_active_inputs():
    xm = _latency_batch(32)
    drive = np.where(np.isfinite(xm), E6.HORIZON - xm, 0).mean(0)
    net = E6.RaceNet(E6.Config(hidden=100, winners=3, psp="ramp", fanin=20, grow=2), 784, 10,
                     float(drive.sum()), np.random.default_rng(3), drive)
    assert (net.M1.sum(1) == 20).all()
    before = net.M1.copy()
    t, idx = E6.to_events(xm)
    for _ in range(3):
        net.teach(net.forward(t, idx), np.arange(32) % 10)
    assert (net.M1.sum(1) == 20).all()
    assert net.work.get("rewire", 0) > 0
    added = net.M1 & ~before
    assert np.isfinite(xm[:, np.flatnonzero(added.any(0)[:784])]).any(0).all()
    assert not (net.W1[:, :784][~net.M1[:, :784]]).any()


# ── E5: the vectorised race equals a direct simulation ───────────────────────

def _direct_race(net, ch, t, theta, beta):
    """Spike-by-spike reference: integrate, check the rising threshold, first wins."""
    v = {}
    for j, c in enumerate(ch):
        for sid in net.rev[c]:
            node = sid // E5.FANIN
            v[node] = v.get(node, 0.0) + net.w.flat[sid]
        th = theta + beta * (j + 1)
        over = {n: x - th for n, x in v.items() if x >= th}
        if over:
            return max(over, key=over.get)
    return max(v, key=v.get) if v else None


@pytest.mark.parametrize("beta", [0.0, 0.06])
def test_e5_race_matches_direct_simulation(beta):
    rng = np.random.default_rng(3)
    k, m = 256, 256
    protos = [rng.choice(m, E5.PROTO, replace=False) for _ in range(k)]
    net = E5.Net(k, m, np.random.default_rng(4), 0.05)
    for _ in range(200):
        y, ch, t = E5.sample(protos, m, rng)
        net.connect(y, ch.tolist(), 1 / 11)
    for _ in range(200):
        _, ch, t = E5.sample(protos, m, rng)
        winner, *_ = E5.race_episode(net, ch, t, 1.0, 0.15, beta, deadline=True)
        assert winner == _direct_race(net, ch, t, 1.0, beta)


# ── Energy model ─────────────────────────────────────────────────────────────

def test_energy_prices_counts_linearly():
    counts = {"synops": 1000, "spikes": 10, "plasticity": 5}
    p = PROFILES["event_ideal"]
    expected = 1e-12 * (1000 * p.synop + 10 * p.spike + 5 * p.plasticity)
    assert energy_joules(counts, "event_ideal") == pytest.approx(expected)
    assert energy_joules({"macs": 100}, "dense_int8") == pytest.approx(100 * PROFILES["dense_int8"].mac * 1e-12)


def test_break_even_is_where_costs_meet():
    dense = energy_joules({"macs": 10_000}, "dense_int8")
    counts = {"synops": 2000, "spikes": 50}
    pj = break_even_synop_pj(dense, counts, spike_pj=2.0)
    assert (counts["synops"] * pj + counts["spikes"] * 2.0) * 1e-12 == pytest.approx(dense)
