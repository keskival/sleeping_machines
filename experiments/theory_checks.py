#!/usr/bin/env python3
"""Numerical checks of THEORY.md on small networks.

M3 (first version): which learning rule points along the true gradient?

The true objective is the expected error of the race under timing noise
(input spike times jittered by σ), which is smooth although each race is not.
Its gradient is estimated by central finite differences with common random
numbers, on a random subset of weight coordinates. Each rule's update, computed
on the same samples without noise, is compared with it by cosine similarity on
those coordinates.

    python experiments/theory_checks.py m3 --coords 200
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import Config, RaceNet, latency_code, mnist, to_events  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "theory")


def small_mnist(n, offset=0):
    """14×14 MNIST (2×2 average pooling), latency-coded."""
    x, y = mnist("train")
    x = x[offset:offset + n].reshape(-1, 14, 2, 14, 2).mean((2, 4)).reshape(-1, 196)
    return latency_code(x), y[offset:offset + n]


def make_net(cfg, times, seed):
    drive = np.where(np.isfinite(times), 1.0 - times, 0).mean(0)
    return RaceNet(cfg, times.shape[1], 10, float(drive.sum()), np.random.default_rng(seed), drive)


def pretrain(net, times, y, epochs, batch=32):
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        for i in range(0, len(y), batch):
            ii = rng.permutation(len(y))[:batch]
            t, idx = to_events(times[ii])
            net.teach(net.forward(t, idx), y[ii])


def error_under_noise(net, noisy_events, y_rep):
    t, idx = noisy_events
    return float((net.forward(t, idx)["winner"] != y_rep).mean())


def rule_update(net, variant, t, idx, y):
    """ΔW1 and ΔW2 a rule would make on this batch (no noise, no homeostasis)."""
    W1, W2, th1 = net.W1.copy(), net.W2.copy(), net.th1.copy()
    saved = net.cfg
    net.cfg = replace(saved, variant=variant, homeo=0.0)
    net.teach(net.forward(t, idx), y)
    d1, d2 = net.W1 - W1, net.W2 - W2
    net.W1, net.W2, net.th1, net.cfg = W1, W2, th1, saved
    return d1, d2


def m3(a):
    cfg = Config(hidden=a.hidden, group=10, winners=3, hid_frac=0.6, psp="ramp", deadline=1,
                 eta_out=0.01, eta_hid=0.01, homeo=0.001, seed=a.seed)
    times, y = small_mnist(a.train)
    net = make_net(cfg, times, a.seed)
    pretrain(net, times, y, a.pretrain)
    ct, cy = small_mnist(a.samples, offset=50000)

    rng = np.random.default_rng(a.seed + 7)
    coords = {}
    for name, W in (("W1", net.W1), ("W2", net.W2)):
        n, m = W.shape
        flat = rng.choice(n * (m - 1), a.coords, replace=False)          # never the dummy column
        coords[name] = np.stack(np.unravel_index(flat, (n, m - 1)), 1)
    y_rep = np.repeat(cy, a.draws)
    rep = np.repeat(ct, a.draws, 0)

    def fd_gradient(noise_seed):
        """Central differences of the expected error, common random numbers within one estimate."""
        r = np.random.default_rng(noise_seed)
        noisy = np.clip(rep + a.sigma * r.standard_normal(rep.shape).astype(np.float32), 0, 0.999)
        events = to_events(np.where(np.isfinite(rep), noisy, np.inf).astype(np.float32))
        out = {"base": error_under_noise(net, events, y_rep)}
        for name in ("W1", "W2"):
            W = getattr(net, name)
            eps = a.rel_eps * float(np.abs(W[:, :-1]).mean())
            g = np.empty(len(coords[name]))
            for k, (i, j) in enumerate(coords[name]):
                old = W[i, j]
                W[i, j] = old + eps
                up = error_under_noise(net, events, y_rep)
                W[i, j] = old - eps
                down = error_under_noise(net, events, y_rep)
                W[i, j] = old
                g[k] = (up - down) / (2 * eps)
            out[name] = g
        return out

    def cos(u, g):
        nu, ng = np.linalg.norm(u), np.linalg.norm(g)
        return float(u @ g / (nu * ng)) if nu and ng else None

    t0 = time.time()
    ga, gb = fd_gradient(a.seed + 100), fd_gradient(a.seed + 200)
    grad = {k: (ga[k] + gb[k]) / 2 for k in ("W1", "W2")}
    fd_time = time.time() - t0

    t, idx = to_events(ct)
    res = {"config": vars(a), "base_error": (ga["base"] + gb["base"]) / 2, "fd_seconds": round(fd_time, 1),
           # reliability: agreement of two independent noise estimates (the ceiling any rule can reach
           # against a single estimate is about sqrt of this against the average)
           "fd_split_cosine": {k: cos(ga[k], gb[k]) for k in ("W1", "W2")},
           "grad_nonzero_frac": {k: float((v != 0).mean()) for k, v in grad.items()}, "rules": {}}
    for variant in ("crl_fired_only", "crl_fa", "crl_sym", "crl_sign", "frozen_hidden"):
        d1, d2 = rule_update(net, variant, t, idx, cy)
        row = {}
        for name, d in (("W1", d1), ("W2", d2)):
            u = -d[coords[name][:, 0], coords[name][:, 1]]      # an update descends: compare −Δw with ∇
            row[name] = {"cosine": cos(u, grad[name]), "update_nonzero_frac": float((u != 0).mean())}
        res["rules"][variant] = row
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"m3_{a.tag or 'run'}_s{a.seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=("m3",))
    ap.add_argument("--hidden", type=int, default=60)
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--pretrain", type=int, default=1)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--coords", type=int, default=200)
    ap.add_argument("--rel-eps", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    m3(ap.parse_args())
