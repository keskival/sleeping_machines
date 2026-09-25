#!/usr/bin/env python3
"""M13 (THEORY.md §11.1): learning by rescheduling, not by gradient.

For ramp synapses, "node n has fired by time c" is the half-space
Σ_i w_ni (c − t_i)⁺ ≥ θ. On a mistake (or a thin margin) at decision time c, the
target is projected into "fires by c with margin μ" and every competitor whose
potential at c is within the margin is projected out of it. Each projection is the
smallest weight change that satisfies its constraint; there is no loss and no
learning rate. Single racing layer, round-3 settings otherwise.

    python experiments/m13_projection.py --epochs 3 --margin 0.1
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import HORIZON, Config, RaceNet, latency_code, mnist, to_events, evaluate  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "theory")


class ProjectionNet(RaceNet):
    """Single racing layer whose teaching step is a half-space projection."""

    def teach(self, st, y):
        mu, W, th = self.mu, self.W2, self.th2
        for b in range(len(y)):
            t, idx = st["t_in"][b], st["idx_in"][b]
            c = st["freeze2"][b, 0]                        # decision time (horizon if forced)
            if self.cfg.deadline and st["urgent"][b]:
                c = HORIZON
            c = max(c, self.c_min)                          # a deadline too early leaves no charge to act on
            charge = np.where(t < c, c - t, 0.0)            # constraint normal, per arrived input
            nrm = float(charge @ charge)
            if nrm == 0:
                continue
            v = W[:, idx] @ charge                          # every node's potential at c
            lo, hi = th * (1 + mu), th * (1 - mu)
            fix_up = v[y[b]] < lo[y[b]]
            down = v > hi
            down[y[b]] = False
            if not fix_up and not down.any():
                continue
            # exact projection, with the step capped (passive-aggressive PA-I): tiny charges
            # would otherwise demand huge weight changes
            if fix_up:
                W[y[b], idx] += min((lo[y[b]] - v[y[b]]) / nrm, self.cap) * charge
            for n in np.flatnonzero(down):
                W[n, idx] += max((hi[n] - v[n]) / nrm, -self.cap) * charge
            W[:, -1] = 0
            self.work["plasticity"] += int((charge > 0).sum()) * (int(fix_up) + int(down.sum()))
            self.mistakes += 1


def main(a):
    x, y = mnist("train")
    if a.val:
        xtr, ytr, xte, yte = x[:-a.val], y[:-a.val], x[-a.val:], y[-a.val:]
    else:
        (xtr, ytr), (xte, yte) = (x, y), mnist("test")
    ttr, tte = latency_code(xtr), latency_code(xte)
    cfg = Config(variant="single_layer", psp="ramp", deadline=1, seed=a.seed)
    drive = np.where(np.isfinite(ttr), HORIZON - ttr, 0).mean(0)
    rng = np.random.default_rng(a.seed)
    net = ProjectionNet(cfg, 784, 10, float(drive.sum()), rng, drive)
    net.mu, net.mistakes, net.cap, net.c_min = a.margin, 0, a.cap, a.c_min
    curve, mistakes = [], []
    t0 = time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(len(ytr))
        before = net.mistakes
        for i in range(0, len(perm), a.batch):
            ii = perm[i:i + a.batch]
            t, idx = to_events(ttr[ii])
            net.teach(net.forward(t, idx), ytr[ii])
        curve.append(evaluate(net, tte, yte))
        mistakes.append(net.mistakes - before)
        print(f"projection epoch {ep + 1}: test {curve[-1]:.4f}  corrections {mistakes[-1]}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    res = {"config": vars(a), "curve": curve, "test_acc": curve[-1], "corrections_per_epoch": mistakes,
           "plasticity_total": net.work["plasticity"]}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"m13_{a.tag or 'run'}_s{a.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--cap", type=float, default=0.05, help="largest projection step (PA-I)")
    ap.add_argument("--c-min", type=float, default=0.0, help="earliest deadline used for a correction")
    ap.add_argument("--val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    main(ap.parse_args())
