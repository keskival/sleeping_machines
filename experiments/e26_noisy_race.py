"""E26: can timing noise resolve the frustration of sparse, error-driven delay learning? (THEORY §54)

Substrate (native, event-driven): a passive delay ring of period p (a wire; no events while a spike travels).
A query is two operand spikes. Operand a's spike enters the ring after a learnable delay d_a; operand b's spike
arrives after a learnable delay e_b and opens a coincidence window; class detector c listens at ring phase t_c.
The ring phase at the read is (e_b - d_a) mod p. Detectors race: the first whose phase the ring reaches after the
read fires, every other detector is cancelled. Events per query: 2 operand spikes + 1 winner.

Every latency is jittered: each spike's arrival gets Gaussian noise of width sigma (the stochastic weaving W: a
pending event has a latency distribution, not a time).

Learning is sparse and local: nothing happens unless the winner is wrong. Then the teacher detector c* and the
winner are the only nodes involved. Each of the three participating delays moves by a fixed step eta toward
closing the teacher's timing gap (the phase distance between where the ring was and where c* listens), and the
wrong winner's listening phase steps away. No sums over samples, no epochs, no normalisation, no replay.
Noise annealing: sigma follows the recent error rate (sigma = sigma0 * err_rate, an exponential trace of errors),
so the system is hot while wrong and cools as it becomes right. Samples arrive as an endless stream.
"""
import argparse
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e26")


def wrap(x, p):
    return (x + p / 2) % p - p / 2


def race(d, e, t, a, b, p, sigma, rng):
    """One query. Returns the winning class and the ring phase at the read (both after jitter)."""
    phase = (e[b] + sigma * rng.standard_normal() - d[a] - sigma * rng.standard_normal()) % p
    listen = (t + sigma * rng.standard_normal(p)) % p
    wait = (listen - phase) % p                      # time until the ring reaches each detector's phase
    order = np.argsort(wait)[:2]                     # the first to coincide fires; the rest are cancelled
    return int(order[0]), phase, int(order[1]), float(wait[order[1]] - wait[order[0]])


def run(p, frac, seed, steps, eta, sigma0, tau, anneal, give="", repel=1, push=1, cool=20000.0):
    rng = np.random.default_rng(seed)
    A, B = np.divmod(np.arange(p * p), p)
    perm = rng.permutation(p * p)
    n = int(round(frac * p * p))
    tr, te = perm[:n], perm[n:]
    d, e, t = (rng.uniform(0, p, p) for _ in range(3))
    k = np.arange(p, dtype=float)                   # sanity: hand some maps their true values
    if "d" in give: d = (-k) % p
    if "e" in give: e = k.copy()
    if "t" in give: t = (k + 0.5) % p
    err, updates, curve = 1.0, 0, []
    for step in range(1, steps + 1):
        i = tr[rng.integers(n)]
        a, b, y = A[i], B[i], (A[i] + B[i]) % p
        if anneal == 2:                             # cool to zero with the learner's own update count
            sigma = sigma0 * np.exp(-updates / cool)
        else:
            sigma = sigma0 * (err if anneal else 1.0)
        c, phase, r2, gap2 = race(d, e, t, a, b, p, sigma, rng)
        wrong = c != y
        err += (wrong - err) / tau
        if wrong:                                   # sparse: only errors teach
            gap = wrap(t[y] - phase - 0.5, p)       # teacher should listen half a unit after the read
            s = eta * np.sign(gap)
            if "e" not in give: e[b] += s / 3
            if "d" not in give: d[a] -= s / 3
            if "t" not in give: t[y] -= s / 3
            if push and "t" not in give: t[c] += eta / 3                         # the wrong winner came too early: it listens later
            if repel and "t" not in give and gap2 < 1.0:   # crowding: the runner-up, cancelled within one
                t[r2] += eta * (1.0 - gap2)            # unit of the winner (its near-miss), steps later
            updates += 1
        if step % (steps // 20) == 0:
            def acc(idx):
                return float(np.mean([race(d, e, t, A[j], B[j], p, 0.0, rng)[0] == (A[j] + B[j]) % p for j in idx]))
            curve.append({"step": step, "train": acc(tr), "test": acc(te), "err_trace": round(err, 4),
                          "sigma": round(sigma, 4), "updates": updates})
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=31)
    ap.add_argument("--fracs", default="0.2,0.5")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=500)
    ap.add_argument("--anneal", type=int, default=1, help="0 fixed, 1 sigma follows the error trace, 2 cools with updates")
    ap.add_argument("--cool", type=float, default=20000.0)
    ap.add_argument("--repel", type=int, default=1)
    ap.add_argument("--push", type=int, default=1)
    ap.add_argument("--give", default="", help="sanity: maps given their true values, e.g. 'dt'")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0, rows = time.time(), []
    for frac in map(float, a.fracs.split(",")):
        for s in range(a.seeds):
            c = run(a.p, frac, s, a.steps, a.eta, a.sigma, a.tau, a.anneal, a.give, a.repel, a.push, a.cool)
            rows.append({"frac": frac, "seed": s, "final": c[-1], "curve": c})
            print(json.dumps({"frac": frac, "seed": s, **c[-1]}), flush=True)
    with open(os.path.join(OUT, f"p{a.p}_s{a.sigma}_a{a.anneal}_r{a.repel}{'_' + a.tag if a.tag else ''}.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "wall_s": round(time.time() - t0, 1)}, f)
    print("EXIT-OK", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
