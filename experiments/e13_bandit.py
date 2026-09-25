#!/usr/bin/env python3
"""E13a: learning from reward alone (contextual bandit).

The learner picks a class by its race and is told only whether it was right (reward 1)
or not (0), never the label. Output teaching signals s (as in E6's apply_signal):

  rstdp        reward-modulated, winner only: s_w = +1 if rewarded, −1 if not
  pool_pg      policy gradient from the pool: π = softmax(−Δ/σ) over outputs (the
               winner has Δ = 0), s_o = (r − r̄)(1[o = w] − π_o), r̄ a running mean reward
  nearmiss     on a miss, push the winner down and pull the near misses up in proportion
               to π (the right answer is probably among the close calls); on a hit,
               reinforce the winner only when the margin was thin
  supervised   the E6 rule with the label (reference, not a bandit)

Small network of M3/M18 (14×14 MNIST, 60 hidden nodes), one sample at a time.

    python experiments/e13_bandit.py --rule pool_pg
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import Config, to_events  # noqa: E402
from theory_checks import make_net, small_mnist  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e13")


def signal(rule, st, y, rbar, sigma):
    w = int(st["winner"][0])
    r = float(w == y)
    k = st["snap2"].shape[1]
    delta = st["snap2"][0].copy()
    pi = np.exp(-delta / sigma)
    pi /= pi.sum()
    s = np.zeros(k)
    if rule == "rstdp":
        s[w] = 1.0 if r else -1.0
    elif rule == "pool_pg":
        s = (r - rbar) * (np.eye(k)[w] - pi)
    elif rule == "nearmiss":
        if r:
            rival = delta.copy()
            rival[w] = np.inf
            if rival.min() < 0.1:                            # a thin win: widen it
                s[w] = 1.0
                s[int(rival.argmin())] = -1.0
        else:
            others = pi.copy()
            others[w] = 0.0
            s = others / max(others.sum(), 1e-9)             # pull the close calls up
            s[w] = -1.0
    return s[None], r


def main(a):
    cfg = Config(hidden=a.hidden, group=10, winners=3, hid_frac=0.6, psp="ramp", deadline=1,
                 eta_out=a.eta, eta_hid=a.eta, homeo=0.001 / 32, batch=1, variant=a.hidden_rule, seed=a.seed)
    xtr, ytr = small_mnist(a.train)
    xte, yte = small_mnist(a.test, offset=50000)
    net = make_net(cfg, xtr, a.seed)
    rng = np.random.default_rng(a.seed)
    rbar, rewards, curve = 0.1, [], []
    t0 = time.time()
    for ep in range(a.epochs):
        for i in rng.permutation(len(ytr)):
            t, idx = to_events(xtr[i:i + 1])
            st = net.forward(t, idx)
            if a.rule == "supervised":
                net.teach(st, ytr[i:i + 1])
                rewards.append(float(st["winner"][0] == ytr[i]))
                continue
            s, r = signal(a.rule, st, ytr[i], rbar, a.sigma)
            rbar += 0.01 * (r - rbar)
            rewards.append(r)
            if np.any(s):
                net.apply_signal(st, s)
            if cfg.homeo and net.h:                          # homeostasis runs whether or not it taught
                net.th1 += cfg.homeo * (st["fired"][0] - cfg.winners / cfg.group)
                np.maximum(net.th1, 0.05, out=net.th1)
        t, idx = to_events(xte)
        curve.append(float((net.forward(t, idx)["winner"] == yte).mean()))
        print(f"{a.rule} epoch {ep + 1}: held-out {curve[-1]:.4f}  reward rate {np.mean(rewards[-len(ytr):]):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    res = {"config": vars(a), "curve": curve, "acc": curve[-1],
           "reward_rate_by_epoch": [float(np.mean(rewards[e * len(ytr):(e + 1) * len(ytr)]))
                                    for e in range(a.epochs)],
           "plasticity": net.work["plasticity"]}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"bandit_{a.rule}_{a.tag or 'run'}_s{a.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="pool_pg", choices=("rstdp", "pool_pg", "nearmiss", "supervised"))
    ap.add_argument("--hidden-rule", default="crl_fa", help="E6 variant used for hidden credit")
    ap.add_argument("--hidden", type=int, default=60)
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=0.15, help="pool temperature for π")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    main(ap.parse_args())
