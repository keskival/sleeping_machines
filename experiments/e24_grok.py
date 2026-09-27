"""E24: grokking in race networks (THEORY §52, M53).

Task: (a + b) mod p from two one-hot tokens; a fraction of the p² pairs is the training set.
Each token is one input spike at t = 0 (on line a, and on line p + b).

  mlp   dense reference, full-batch AdamW with weight decay (the regime known to grok)
  race  the E6 race network with its local error-gated rule, trained epoch by epoch;
        with --sleep λ every epoch ends with a sleep phase that shrinks all weights by (1 − λ)
        (synaptic downscaling). Without it the rule stops once every sample wins with margin.

Prints and saves train/test accuracy curves and weight norms.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import Config, RaceNet, to_events, evaluate  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e24")


def dataset(p, frac, seed):
    a, b = np.divmod(np.arange(p * p), p)
    y = (a + b) % p
    x = np.full((p * p, 2 * p), np.inf, np.float32)
    x[np.arange(p * p), a] = 0.0
    x[np.arange(p * p), p + b] = 0.0
    perm = np.random.default_rng(seed).permutation(p * p)
    n = int(frac * p * p)
    tr, te = perm[:n], perm[n:]
    return x[tr], y[tr], x[te], y[te]


def run_mlp(a, xtr, ytr, xte, yte):
    rng = np.random.default_rng(a.seed)
    f = lambda t: np.isfinite(t).astype(np.float32)          # noqa: E731
    ftr, fte = f(xtr), f(xte)
    d, h, k = ftr.shape[1], a.hidden, a.p
    P = {"W1": rng.normal(0, a.init / np.sqrt(d), (d, h)), "b1": np.zeros(h),
         "W2": rng.normal(0, a.init / np.sqrt(h), (h, k)), "b2": np.zeros(k)}
    m = {n: np.zeros_like(v) for n, v in P.items()}
    v = {n: np.zeros_like(x) for n, x in P.items()}
    b1, b2, eps = 0.9, 0.98, 1e-8
    Bfb = rng.normal(0, a.init / np.sqrt(h), (h, k))          # fixed random feedback (FA)
    if a.credit == "fa_fourier":                              # feedback rows: few low frequencies of the label
        c = np.arange(k)
        w = rng.integers(1, a.freqs + 1, h)[:, None]
        Bfb = a.init / np.sqrt(h) * np.cos(2 * np.pi * w * c / k + rng.uniform(0, 2 * np.pi, (h, 1)))
    curve = []
    for step in range(1, a.steps + 1):
        z1 = ftr @ P["W1"] + P["b1"]
        a1 = np.maximum(z1, 0)
        z = a1 @ P["W2"] + P["b2"]
        pr = np.exp(z - z.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        pr[np.arange(len(ytr)), ytr] -= 1
        pr /= len(ytr)
        g1 = (pr @ (P["W2"] if a.credit == "bp" else Bfb).T) * (z1 > 0)
        if step % a.every == 0 and a.credit in ("kp", "fa", "fa_fourier"):
            cos = float((Bfb * P["W2"]).sum() / (np.linalg.norm(Bfb) * np.linalg.norm(P["W2"]) + 1e-12))
        G = {"W2": a1.T @ pr, "b2": pr.sum(0), "W1": ftr.T @ g1, "b1": g1.sum(0)}
        for n in (("W2", "b2") if a.credit == "frozen" else P):
            m[n] = b1 * m[n] + (1 - b1) * G[n]
            v[n] = b2 * v[n] + (1 - b2) * G[n] ** 2
            mh, vh = m[n] / (1 - b1 ** step), v[n] / (1 - b2 ** step)
            step_dir = mh / (np.sqrt(vh) + eps)
            P[n] -= a.lr * (step_dir + a.wd * P[n])
            if n == "W2" and a.credit == "kp":            # Kolen–Pollack: feedback gets W2's update and decay
                Bfb -= a.lr * (step_dir + a.wd * Bfb)
        if step % a.every == 0 or step == a.steps:
            def acc(F, Y):
                return float(((np.maximum(F @ P["W1"] + P["b1"], 0) @ P["W2"] + P["b2"]).argmax(1) == Y).mean())
            norm = float(np.sqrt(sum((P[n] ** 2).sum() for n in ("W1", "W2"))))
            curve.append({"step": step, "train": acc(ftr, ytr), "test": acc(fte, yte), "norm": norm,
                          "align": cos if a.credit in ("kp", "fa", "fa_fourier") else 1.0})
            print(json.dumps(curve[-1]), flush=True)
    return curve


def run_race(a, xtr, ytr, xte, yte):
    cfg = Config(variant=a.variant, hidden=a.hidden, group=a.group, winners=a.winners, eta_out=a.eta,
                 eta_hid=a.eta, psp="step", batch=a.batch, epochs=1, seed=a.seed, margin=a.margin,
                 homeo=a.homeo, init_frac=a.init_frac, sigma=a.sigma, deadline=a.deadline, compete=a.compete)
    rng = np.random.default_rng(a.seed)
    drive = np.isfinite(xtr).mean(0)
    net = RaceNet(cfg, xtr.shape[1], a.p, float(drive.sum()), rng, drive)
    curve = []
    for ep in range(1, a.epochs + 1):
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), cfg.batch):
            j = perm[i:i + cfg.batch]
            t, idx = to_events(xtr[j])
            st = net.forward(t, idx)
            net.teach(st, ytr[j])
        if a.sleep:                                       # sleep: synaptic downscaling
            for W in [net.W2] + ([net.W1] if net.h else []):
                W *= 1 - a.sleep
        if ep % a.every == 0 or ep == a.epochs:
            norm = float(np.sqrt((net.W2 ** 2).sum() + ((net.W1 ** 2).sum() if net.h else 0)))
            curve.append({"epoch": ep, "train": evaluate(net, xtr, ytr), "test": evaluate(net, xte, yte),
                          "norm": norm, "plasticity": int(net.work["plasticity"])})
            print(json.dumps(curve[-1]), flush=True)
    return curve


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("mlp", "race"))
    ap.add_argument("--p", type=int, default=31)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--tag", default="")
    # mlp
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--init", type=float, default=1.0)
    ap.add_argument("--credit", default="bp", choices=("bp", "fa", "frozen", "fa_fourier", "kp"))
    ap.add_argument("--freqs", type=int, default=3)
    # race
    ap.add_argument("--variant", default="crl_fa")
    ap.add_argument("--group", type=int, default=10)
    ap.add_argument("--winners", type=int, default=3)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--homeo", type=float, default=0.0)
    ap.add_argument("--init-frac", type=float, default=0.3)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--compete", type=int, default=1)
    a = ap.parse_args()
    data = dataset(a.p, a.frac, a.seed)
    t0 = time.time()
    curve = (run_mlp if a.kind == "mlp" else run_race)(a, *data)
    os.makedirs(OUT, exist_ok=True)
    res = {"args": vars(a), "curve": curve, "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, f"{a.kind}_{a.tag or 'run'}_s{a.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("EXIT-OK", res["wall_s"])
