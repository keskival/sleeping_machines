#!/usr/bin/env python3
"""M18 (THEORY.md §13): learning as history repair.

On a mistake, the learner looks for the cheapest single-event repair of the
history that makes the right answer win, and verifies each candidate by re-running
the output race:

  output repair   the target output is projected into "fires by the decision time"
                  and the winner out of it (the M13 half-space projection);
  hidden repair   one hidden node is rescheduled: a near-miss node that would help the
                  target fires just before its group's cancel time (taking the place of
                  the group's last winner), or a firing node that helps the wrong winner
                  stops firing (and its group's next strand fires in its place).

Cost is the size of the weight change (a half-space distance). The cheapest verified
repair is applied, with a PA-I step cap. Everything else stays untouched.
Compared with crl_fa, fired-only and frozen-hidden on the small network of M3.

    python experiments/m18_repair.py --rule repair
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import HORIZON, Config, layer_race, race_order, to_events  # noqa: E402
from theory_checks import make_net, small_mnist  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "theory")


def output_race(net, h_times):
    """Winner of the output race for each row of hidden spike times (inf = silent)."""
    t2, idx2 = to_events(h_times)
    T2, over2, v_at = layer_race(t2, idx2, net.W2, net.th2, "ramp")
    has = np.isfinite(T2).any(1)
    winner = np.where(has, race_order(T2, over2)[:, 0], -1)
    if net.cfg.deadline:
        v, _ = v_at(np.full(T2.shape, HORIZON))
        winner = np.where(has, winner, v.argmax(1))
    t_dec = np.where(has, T2.min(1), HORIZON)
    return winner, t_dec


def charges(t, idx, c, d):
    """Charge each input had injected by time c: the half-space normal for 'fired by c'."""
    q = np.zeros(d + 1)
    ok = np.isfinite(t) & (t < c)
    q[idx[ok]] = c - t[ok]
    return q[:d]


class Repairer:
    def __init__(self, net, cap, margin, max_candidates):
        self.net, self.cap, self.mu, self.R = net, cap, margin, max_candidates
        self.stats = {"output": 0, "hidden_fire": 0, "hidden_silence": 0, "none": 0, "weights_touched": 0}

    def project(self, W, row, q, target, up):
        """Move W[row] to the half-space  w·q ≥ target (up) or ≤ target (down)."""
        v, n2 = float(W[row, :len(q)] @ q), float(q @ q)
        if n2 == 0 or (up and v >= target) or (not up and v <= target):
            return 0
        step = (target - v) / n2
        step = min(step, self.cap) if up else max(step, -self.cap)
        W[row, :len(q)] += step * q
        return int((q > 0).sum())

    def repair(self, x_t, y):
        net, cfg = self.net, self.net.cfg
        t, idx = to_events(x_t[None])
        st = net.forward(t, idx)
        if self.homeo and net.h:                               # label-free, every sample (as E7)
            net.th1 += cfg.homeo * (st["fired"][0] - cfg.winners / cfg.group)
            np.maximum(net.th1, 0.05, out=net.th1)
        if st["winner"][0] == y:
            if not self.thin:
                return
            rival = st["snap2"][0].copy()
            rival[y] = np.inf
            if rival.min() >= cfg.margin:
                return
            b = int(rival.argmin())                           # a thin margin: widen it at the output only
            q2 = np.zeros(net.h)
            h_t = np.where(st["fired"][0], st["freeze1"][0], np.inf)
            c = st["freeze2"][0, 0]
            ok = np.isfinite(h_t) & (h_t < c)
            q2[ok] = c - h_t[ok]
            touched = self.project(net.W2, y, q2, net.th2[y] * (1 + self.mu), True)
            touched += self.project(net.W2, b, q2, net.th2[b] * (1 - self.mu), False)
            self.stats["weights_touched"] += touched
            self.stats["thin"] = self.stats.get("thin", 0) + 1
            return
        b = st["winner"][0]
        fired, freeze1 = st["fired"][0], st["freeze1"][0]
        h_times = np.where(fired, freeze1, np.inf)
        t_dec = st["freeze2"][0, 0]
        # projected crossing time of every hidden strand (the pool), for groups' next-in-line
        T1, _, _ = layer_race(t, idx, net.W1, net.th1, "ramp")
        T1 = T1[0]
        G, size = net.h // cfg.group, cfg.group
        cancel = freeze1.reshape(G, size).max(1)             # group cancel times (k-th winner)

        cands, costs, rows = [], [], []
        help_y = net.W2[y, :net.h] - (net.W2[b, :net.h] if b >= 0 else 0)
        # hidden nodes that would help if they fired: silent, positive help, nearest first
        silent = np.flatnonzero(~fired & (help_y > 0))
        for h in silent:
            g = h // size
            c = cancel[g] - 1e-4
            if c <= 0:
                continue
            q = charges(t[0], idx[0], c, net.d)
            v = float(net.W1[h, :net.d] @ q)
            dist = max(0.0, net.th1[h] * (1 + self.mu) - v) / max(np.linalg.norm(q), 1e-9)
            new = h_times.copy()
            new[h] = c
            grp = np.arange(g * size, (g + 1) * size)
            last = grp[np.argmax(np.where(fired[grp], freeze1[grp], -np.inf))]
            new[last] = np.inf                               # h takes the last winner's place
            cands.append(("hidden_fire", h, c, True)); costs.append(dist); rows.append(new)
        # hidden nodes that help the wrong winner and fired: silence them
        loud = np.flatnonzero(fired & (help_y < 0))
        for h in loud:
            g = h // size
            grp = np.arange(g * size, (g + 1) * size)
            c = cancel[g]
            q = charges(t[0], idx[0], c + 1e-4, net.d)
            v = float(net.W1[h, :net.d] @ q)
            dist = max(0.0, v - net.th1[h] * (1 - self.mu)) / max(np.linalg.norm(q), 1e-9)
            new = h_times.copy()
            new[h] = np.inf
            nxt = [m for m in grp[np.argsort(T1[grp])] if not fired[m] and np.isfinite(T1[m])]
            if nxt:                                          # the next strand in line fires instead
                new[nxt[0]] = T1[nxt[0]]
            cands.append(("hidden_silence", h, c + 1e-4, False)); costs.append(dist); rows.append(new)
        # keep the cheapest few, verify them together
        best = None
        if cands and self.R:
            order = np.argsort(costs)[:self.R]
            wins, _ = output_race(net, np.stack([rows[k] for k in order]).astype(np.float32))
            ok = [k for k, w in zip(order, wins) if w == y]
            if ok:
                best = ok[0]
        # output repair cost: target into "fires by t_dec", winner out of it
        q2 = np.zeros(net.h)
        spk = np.isfinite(h_times) & (h_times < t_dec)
        q2[spk] = t_dec - h_times[spk]
        n2 = max(np.linalg.norm(q2), 1e-9)
        v2 = net.W2[:, :net.h] @ q2
        out_cost = (max(0.0, net.th2[y] * (1 + self.mu) - v2[y])
                    + (max(0.0, v2[b] - net.th2[b] * (1 - self.mu)) if b >= 0 else 0)) / n2
        if best is not None and costs[best] <= out_cost:
            kind, h, c, up = cands[best]
            q = charges(t[0], idx[0], c, net.d)
            target = net.th1[h] * (1 + self.mu) if up else net.th1[h] * (1 - self.mu)
            self.stats["weights_touched"] += self.project(net.W1, h, q, target, up)
            self.stats[kind] += 1
        else:
            touched = self.project(net.W2, y, q2, net.th2[y] * (1 + self.mu), True)
            if b >= 0:
                touched += self.project(net.W2, b, q2, net.th2[b] * (1 - self.mu), False)
            self.stats["weights_touched"] += touched
            self.stats["output" if touched else "none"] += 1


def accuracy(net, times, y):
    t, idx = to_events(times)
    return float((net.forward(t, idx)["winner"] == y).mean())


def main(a):
    cfg = Config(hidden=a.hidden, group=10, winners=3, hid_frac=0.6, psp="ramp", deadline=1,
                 eta_out=0.01, eta_hid=0.01, homeo=0.001 / 32, batch=1,
                 variant={"repair": "crl_fa", "output_only": "frozen_hidden"}.get(a.rule, a.rule), seed=a.seed)
    xtr, ytr = small_mnist(a.train)
    xte, yte = small_mnist(a.test, offset=50000)
    net = make_net(cfg, xtr, a.seed)
    rep = Repairer(net, a.cap, a.margin, a.candidates)
    rep.homeo, rep.thin = a.homeo, a.thin
    rng = np.random.default_rng(a.seed)
    curve = []
    t0 = time.time()
    for ep in range(a.epochs):
        before = net.work["plasticity"]
        for i in rng.permutation(len(ytr)):
            if a.rule == "repair":
                rep.repair(xtr[i], ytr[i])
            elif a.rule == "output_only":
                rep.R = 0                                   # never consider hidden repairs
                rep.repair(xtr[i], ytr[i])
            else:
                t, idx = to_events(xtr[i:i + 1])
                net.teach(net.forward(t, idx), ytr[i:i + 1])
        curve.append(accuracy(net, xte, yte))
        print(f"{a.rule} epoch {ep + 1}: held-out {curve[-1]:.4f}  ({time.time() - t0:.0f}s)  "
              f"{rep.stats if a.rule in ('repair', 'output_only') else net.work['plasticity'] - before}",
              flush=True)
    res = {"config": vars(a), "curve": curve, "acc": curve[-1],
           "repairs": rep.stats if a.rule in ("repair", "output_only") else None,
           "plasticity": net.work["plasticity"]}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"m18_{a.rule}_{a.tag or 'run'}_s{a.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="repair",
                    choices=("repair", "output_only", "crl_fa", "crl_fired_only", "frozen_hidden"))
    ap.add_argument("--hidden", type=int, default=60)
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--cap", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--homeo", type=int, default=0, help="run hidden homeostasis on every sample")
    ap.add_argument("--thin", type=int, default=0, help="also widen thin margins on correct samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    main(ap.parse_args())
