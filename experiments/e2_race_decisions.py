#!/usr/bin/env python3
"""E2: integrate-and-race decisions compute adaptively.

See experiments/E2_PREREGISTRATION.md.

    python experiments/e2_race_decisions.py
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sleeping_machines.sim import Engine  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e2")
COHERENCES = np.array([0.05, 0.1, 0.2, 0.4, 0.8])


def make_trial(rng, k, g, r0, t_max):
    """Independent Poisson channels; returns per-channel spike-time arrays."""
    target = int(rng.integers(k))
    c = float(rng.choice(COHERENCES))
    rates = np.full(k * g, r0 * (1 - c / (k - 1)))
    rates[target * g:(target + 1) * g] = r0 * (1 + c)
    trains = []
    for lam in rates:
        n = rng.poisson(lam * t_max)
        trains.append(np.sort(rng.uniform(0, t_max, n)))
    return target, c, trains


def merged(trains, g):
    times = np.concatenate(trains)
    groups = np.concatenate([np.full(len(t), j // g) for j, t in enumerate(trains)])
    order = np.argsort(times, kind="stable")
    return times[order], groups[order]


# ── Event-driven race ─────────────────────────────────────────────────────────

def race_trial(trains, k, g, theta, t_max):
    """Each channel lazily schedules its next spike; the first node at θ fires and
    cancels every pending input event. Returns (decision, time, work counters)."""
    e = Engine()
    v = np.zeros(k)
    pending = {}
    result = {}

    def spike(ch):
        idx = pending_idx[ch]
        pending_idx[ch] = idx + 1
        if idx + 1 < len(trains[ch]):
            pending[ch] = e.schedule(trains[ch][idx + 1] - e.now, "spike", ch)
        else:
            pending.pop(ch, None)
        grp = ch // g
        v[:] -= 1.0 / (k - 1)
        v[grp] += 1.0 + 1.0 / (k - 1)
        e.count("synops", k)
        if "decision" not in result and v.max() >= theta:
            result["decision"], result["time"] = int(v.argmax()), e.now
            for ev in pending.values():
                e.cancel(ev)
            pending.clear()
            e.count("inhibition", k - 1)

    def deadline(_):
        if "decision" not in result:
            result["decision"], result["time"] = int(v.argmax()), t_max
            for ev in pending.values():
                e.cancel(ev)
            pending.clear()

    e.on("spike", spike)
    e.on("deadline", deadline)
    pending_idx = [0] * len(trains)
    for ch, t in enumerate(trains):
        if len(t):
            pending[ch] = e.schedule(t[0], "spike", ch)
    e.schedule(t_max, "deadline")
    e.run()
    return result["decision"], result["time"], dict(e.work)


# ── References ────────────────────────────────────────────────────────────────

def fixed_time(times, groups, k, t, rng):
    counts = np.bincount(groups[times <= t], minlength=k).astype(float)
    counts += rng.random(k) * 1e-6          # random tie-break
    return int(counts.argmax())


def msprt_path(groups, k):
    """Posterior max and argmax after each spike, marginalised over coherence."""
    n = len(groups)
    if n == 0:
        return np.zeros(0), np.zeros(0, int)
    onehot = np.zeros((n, k))
    onehot[np.arange(n), groups] = 1
    nh = np.cumsum(onehot, 0)                         # spikes in each group so far
    ntot = np.arange(1, n + 1)[:, None]
    ll = (nh[None] * np.log1p(COHERENCES)[:, None, None]
          + (ntot - nh)[None] * np.log1p(-COHERENCES / (k - 1))[:, None, None])
    ll = np.logaddexp.reduce(ll - np.log(len(COHERENCES)), axis=0)   # (n, k)
    post = np.exp(ll - ll.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    return post.max(1), post.argmax(1)


def run_seed(args):
    seed, a = args
    rng = np.random.default_rng(seed)
    trials = [make_trial(rng, a.k, a.g, a.r0, a.t_max) for _ in range(a.trials)]
    rows = []
    for theta in a.thetas:
        for target, c, trains in trials:
            d, t, w = race_trial(trains, a.k, a.g, theta, a.t_max)
            total = sum(len(x) for x in trains)
            rows.append({"decoder": "race", "param": theta, "c": c, "correct": d == target, "time": t,
                         "events": w.get("fired", 0), "cancelled": w.get("cancelled", 0),
                         "synops": w.get("synops", 0), "spikes_to_tmax": total})
    tie_rng = np.random.default_rng(seed + 1)
    for target, c, trains in trials:
        times, groups = merged(trains, a.g)
        for t in a.times:
            rows.append({"decoder": "fixed_time", "param": t, "c": c,
                         "correct": fixed_time(times, groups, a.k, t, tie_rng) == target, "time": t,
                         "events": int((times <= t).sum()), "synops": int((times <= t).sum()) * a.k})
        pmax, pargmax = msprt_path(groups, a.k)
        for alpha in a.alphas:
            hit = np.flatnonzero(pmax >= 1 - alpha)
            if len(hit):
                i = hit[0]
                d, t, ev = int(pargmax[i]), float(times[i]), int(i + 1)
            else:
                d, t, ev = int(pargmax[-1]) if len(pargmax) else 0, a.t_max, len(times)
            rows.append({"decoder": "msprt", "param": alpha, "c": c, "correct": d == target,
                         "time": t, "events": ev})
    return seed, rows


def main(a):
    os.makedirs(OUT, exist_ok=True)
    with Pool(a.procs) as pool:
        out = pool.map(run_seed, [(s, a) for s in range(a.seeds)])
    rows = [dict(r, seed=s) for s, rs in out for r in rs]
    with open(os.path.join(OUT, "rows.json"), "w") as f:
        json.dump({"config": vars(a), "rows": rows}, f)
    summarize(rows)


def summarize(rows):
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["decoder"], r["param"])].append(r)
    print(f"{'decoder':11s} {'param':>7s} {'acc':>6s} {'time':>6s} {'events':>7s}")
    for (dec, p), rs in sorted(agg.items()):
        print(f"{dec:11s} {p:7.3g} {np.mean([r['correct'] for r in rs]):6.3f} "
              f"{np.mean([r['time'] for r in rs]):6.3f} {np.mean([r['events'] for r in rs]):7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--g", type=int, default=5)
    ap.add_argument("--r0", type=float, default=10.0)
    ap.add_argument("--t-max", type=float, default=4.0)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--thetas", type=float, nargs="+", default=[1, 2, 3, 4, 6, 8, 11, 15, 20, 26])
    ap.add_argument("--times", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
    ap.add_argument("--procs", type=int, default=4)
    main(ap.parse_args())
