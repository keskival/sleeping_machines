#!/usr/bin/env python3
"""E5 diagnosis: is the race's accuracy gap in its weights (learning) or its readout?

Trains the race with the tuned settings, then reads the same weights out three ways:
the race itself, and an argmax over all input evidence (no time pressure).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e5_capacity as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=1024)
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--deadline", type=int, default=0)
ap.add_argument("--beta", type=float, default=0.06)
a = ap.parse_args()
cfg = argparse.Namespace(eta=0.1, beta=a.beta, margin=0.3, sigma=0.15, tau=1.0, teach_delay=0.05, w0=0.005,
                         theta=1.0, train_per_class=8, test=1000, models=["race"], lr=0.5)
k, m = a.k, max(256, a.k // 4)
rng = np.random.default_rng(a.seed)
protos = [rng.choice(m, E.PROTO, replace=False) for _ in range(k)]

# Re-run the race training loop from e5_capacity.run, keeping the network.
net = E.Net(k, m, np.random.default_rng(a.seed + 1), cfg.w0)
w_new = 1.0 / (E.PROTO * E.KEEP)
for _ in range(cfg.train_per_class * k):
    y, ch, t = E.sample(protos, m, rng)
    winner, dec, touched, delta, _, urgent = E.race_episode(net, ch, t, cfg.theta, cfg.sigma, cfg.beta, a.deadline)
    rivals = delta[touched != y]
    if winner == y and not urgent and not (len(rivals) and rivals.min() < cfg.margin):
        continue
    xs = dict(zip(ch.tolist(), np.exp(-(1.0 + cfg.teach_delay - t) / cfg.tau).tolist()))
    net.connect(y, ch.tolist(), w_new)
    for node, sign in [(y, 1.0)] + [(int(n), -float(np.exp(-d / cfg.sigma))) for n, d in zip(touched, delta) if n != y]:
        if sign < 0 and -sign < 0.05:
            continue
        for slot, c in enumerate(net.ch[node]):
            if c in xs:
                net.w[node, slot] += cfg.eta * sign * xs[c]

test_rng = np.random.default_rng(10_000 + a.seed)
race_ok = argmax_ok = timeouts = work = 0
for _ in range(cfg.test):
    y, ch, t = E.sample(protos, m, test_rng)
    winner, _, _, _, synops, _ = E.race_episode(net, ch, t, cfg.theta, cfg.sigma, cfg.beta, a.deadline)
    work += synops
    race_ok += winner == y
    timeouts += winner is None
    ids = np.concatenate(net.gather(ch))
    touched, inv = np.unique(ids // E.FANIN, return_inverse=True)
    z = np.bincount(inv, weights=net.w.flat[ids], minlength=len(touched))
    argmax_ok += touched[z.argmax()] == y
print(f"K={k}: race readout {race_ok / cfg.test:.3f} | argmax over all evidence, same weights {argmax_ok / cfg.test:.3f} "
      f"| timeouts {timeouts / cfg.test:.3f} | synaptic events per input {work / cfg.test:.0f}")
