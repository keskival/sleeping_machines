"""E7 stream learner: with every mechanism off it is E6 at one frame per update,
and each mechanism does only what it claims.

    python -m pytest tests/ -q
"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import e6_hidden as E6  # noqa: E402
import e7_stream as E7  # noqa: E402


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.random((n, 784)) * (rng.random((n, 784)) < 0.2)).astype(np.float32)
    return x, rng.integers(0, 10, n)


def _nets(mech, seed=3):
    times = E6.latency_code(_data()[0])
    drive = np.where(np.isfinite(times), 1.0 - times, 0).mean(0)
    cfg = E6.Config(hidden=100, winners=3, hid_frac=0.6, psp="ramp", deadline=1, batch=1)
    a = E6.RaceNet(cfg, 784, 10, float(drive.sum()), np.random.default_rng(seed), drive)
    b = E7.StreamRace(E6.Config(**vars(cfg)), mech, 784, 10, float(drive.sum()),
                      np.random.default_rng(seed), drive)
    return a, b, times


def test_stream_with_mechanisms_off_is_e6_one_frame_at_a_time():
    x, y = _data()
    a, b, times = _nets(E7.Mech())
    for i in range(len(y)):
        t, idx = E6.to_events(times[i:i + 1])
        a.teach(a.forward(t, idx), y[i:i + 1])
        st = b.predict(times[i])
        b.learn(st, y[i])
        b.after(st, False)
    assert np.array_equal(a.W1, b.W1) and np.array_equal(a.W2, b.W2)
    assert np.allclose(a.th1, b.th1)


def test_consolidation_only_slows_change():
    x, y = _data()
    _, fast, times = _nets(E7.Mech())
    _, slow, _ = _nets(E7.Mech(consolidation=100.0))
    moved = {}
    for net in (fast, slow):
        before = net.W1.copy()
        for i in range(len(y)):
            net.learn(net.predict(times[i]), y[i])
        moved[net] = np.abs(net.W1 - before).sum()
    assert 0 < moved[slow] < moved[fast]


def test_prior_lowers_only_the_last_winner_and_resets_between_episodes():
    _, net, times = _nets(E7.Mech(prior=0.3))
    st = net.predict(times[0])
    net.after(st, reset_next=False)
    w = st["winner"][0]
    assert net.prior[w] == 1.0 and net.prior.sum() == 1.0
    net.after(net.predict(times[1]), reset_next=True)
    assert not net.prior.any()


def test_shuffled_stream_has_the_same_frames_and_labels_in_another_order():
    x, y = _data(200)
    ep = E7.make_stream(E7.Stream(order="episodes", frames=100, length=5), x, y)
    sh = E7.make_stream(E7.Stream(order="shuffled", frames=100, length=5), x, y)
    key = lambda f, l: sorted(zip(map(bytes, f), l))   # noqa: E731
    assert key(ep[0], ep[1]) == key(sh[0], sh[1])
    assert not np.array_equal(ep[1], sh[1])
    assert np.array_equal(ep[2], sh[2])                 # resets stay where they were


def test_episode_views_share_their_label_and_blocked_streams_keep_task_order():
    x, y = _data(200)
    frames, labels, reset, _ = E7.make_stream(E7.Stream(order="episodes", frames=100, length=5), x, y)
    assert reset.sum() == 20
    assert all(len(set(labels[i:i + 5])) == 1 for i in range(0, 100, 5))
    _, labels, _, task = E7.make_stream(E7.Stream(order="blocked", frames=100, length=1), x, y)
    assert np.all(np.diff(task) >= 0)
    assert all(labels[task == k].min() >= 2 * k and labels[task == k].max() <= 2 * k + 1 for k in range(5))


def test_asked_labels_respect_the_budget():
    lab = E7.Labeller(E7.Stream(labels="ask", label_rate=0.1, bucket=5), 1000)
    given = sum(lab(i, True) for i in range(1000))
    assert given <= 100 and given >= 95
    lab = E7.Labeller(E7.Stream(labels="ask", label_rate=0.1), 1000)
    assert sum(lab(i, False) for i in range(1000)) == 0
