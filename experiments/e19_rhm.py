"""E19: can race networks exploit a matched hierarchy? The Random Hierarchy Model (Cagnetta, Petrini,
Tomasini, Favero & Wyart, PRX 2024) as a controlled test of depth.

    python experiments/e19_rhm.py --model race|mlp --depth D --train N [--residual 1] [--seed S]

RHM(v, n_c, m, s, L): each class is a top-level symbol; every symbol at every level rewrites to one of m
synonymous s-tuples of lower-level symbols (vocabulary v), rules unambiguous (no tuple shared between
symbols). After L levels the input is a string of s^L symbols, one-hot (s^L · v channels). Deep networks
trained by backprop learn it with polynomially many samples by building synonym-invariant representations
level by level; shallow/kernel methods need exponentially many. Here: race networks (depth 1–4, plain and
residual stream) against backprop MLPs on the same data. Present symbols spike at t = 0.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import HORIZON, Config, to_events  # noqa: E402
from e14_depth import DeepRaceNet  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e19")


def make_rules(v, n_c, m, s, L, rng):
    """rules[l][a] = (m, s) tuples for symbol a at level l (l = L−1 is the top, with n_c symbols)."""
    rules = []
    for level in range(L):
        n_sym = n_c if level == L - 1 else v
        pick = rng.choice(v ** s, n_sym * m, replace=False)            # unambiguous: all tuples distinct
        tuples = np.stack(np.unravel_index(pick, (v,) * s), 1)
        rules.append(tuples.reshape(n_sym, m, s))
    return rules


def sample(rules, n, n_c, m, rng):
    y = rng.integers(0, n_c, n)
    x = y[:, None]
    for level in reversed(range(len(rules))):
        r = rng.integers(0, m, x.shape)
        x = rules[level][x, r].reshape(n, -1)                          # (n, len·s)
    return x, y


def one_hot(x, v):
    n, length = x.shape
    out = np.zeros((n, length * v), np.float32)
    out[np.arange(n)[:, None], np.arange(length)[None, :] * v + x] = 1
    return out


def run_race(a, xtr, ytr, xte, yte, n_c):
    rng = np.random.default_rng(a.seed)
    Ttr = np.where(xtr > 0, 0.0, np.inf).astype(np.float32)          # present symbol: a spike at t = 0
    Tte = np.where(xte > 0, 0.0, np.inf).astype(np.float32)
    drive = np.where(np.isfinite(Ttr), HORIZON - Ttr, 0).mean(0)
    cfg = Config(variant="crl_fa", winners=3, hid_frac=0.6, eta_out=0.01, eta_hid=0.01, deadline=1, psp="ramp",
                 homeo=0.001, sigma=0.15, zero_sum=1, seed=a.seed)
    net = DeepRaceNet(cfg, [a.width] * a.depth, Ttr.shape[1], n_c, drive, rng, residual=bool(a.residual))
    for attr, val in dict(window=0.15, nonneg=False, eg=0.0, homeo_mode="linear", homeo_rate=0.001,
                          info_capacity=False, group_conserve=False, pivot_top=False, share_jac=False,
                          causal=False, center_credit=0, gauge=False, self_sigma=0).items():
        setattr(net, attr, val)

    def acc(T, y):
        c = 0
        for i in range(0, len(y), 500):
            c += int((net.forward(*to_events(T[i:i + 500]))["winner"] == y[i:i + 500]).sum())
        return c / len(y)

    curve = []
    for ep in range(a.epochs):
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), 32):
            ii = perm[i:i + 32]
            net.teach(net.forward(*to_events(Ttr[ii])), ytr[ii])
        curve.append(acc(Tte, yte))
    return {"test_acc": curve[-1], "curve": curve, "train_acc": acc(Ttr[:5000], ytr[:5000]),
            "synops_per_sample": net.work["synops"] / max(net.work["samples"], 1)}


def run_mlp(a, xtr, ytr, xte, yte, n_c):
    """Backprop reference: ReLU MLP with `depth` hidden layers, Adam, cross-entropy."""
    rng = np.random.default_rng(a.seed)
    dims = [xtr.shape[1]] + [a.width] * a.depth + [n_c]
    Ws = [rng.normal(0, np.sqrt(2 / dims[i]), (dims[i], dims[i + 1])).astype(np.float32) for i in range(len(dims) - 1)]
    bs = [np.zeros(d, np.float32) for d in dims[1:]]
    params = Ws + bs
    mom = [np.zeros_like(p) for p in params]
    vel = [np.zeros_like(p) for p in params]
    step = 0

    def fwd(x):
        hs = [x]
        for i, (W, b) in enumerate(zip(Ws, bs)):
            z = hs[-1] @ W + b
            hs.append(np.maximum(z, 0) if i < len(Ws) - 1 else z)
        return hs

    def acc(x, y):
        return float((fwd(x)[-1].argmax(1) == y).mean())

    curve = []
    for ep in range(a.epochs):
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), 64):
            ii = perm[i:i + 64]
            hs = fwd(xtr[ii])
            p = np.exp(hs[-1] - hs[-1].max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            g = p
            g[np.arange(len(ii)), ytr[ii]] -= 1
            g /= len(ii)
            gW, gb = [None] * len(Ws), [None] * len(Ws)
            for layer in reversed(range(len(Ws))):
                gW[layer], gb[layer] = hs[layer].T @ g, g.sum(0)
                if layer:
                    g = (g @ Ws[layer].T) * (hs[layer] > 0)
            step += 1
            for j, (prm, gr) in enumerate(zip(params, gW + gb)):
                mom[j] = 0.9 * mom[j] + 0.1 * gr
                vel[j] = 0.999 * vel[j] + 0.001 * gr * gr
                prm -= 1e-3 * (mom[j] / (1 - 0.9 ** step)) / (np.sqrt(vel[j] / (1 - 0.999 ** step)) + 1e-8)
        curve.append(acc(xte, yte))
    return {"test_acc": curve[-1], "curve": curve, "train_acc": acc(xtr[:5000], ytr[:5000])}


def main(a):
    rules = make_rules(a.v, a.nc, a.m, a.s, a.L, np.random.default_rng(1000 + a.rule_seed))
    rng = np.random.default_rng(a.seed)
    xtr, ytr = sample(rules, a.train, a.nc, a.m, rng)
    xte, yte = sample(rules, 10000, a.nc, a.m, np.random.default_rng(10_000 + a.seed))
    Xtr, Xte = one_hot(xtr, a.v), one_hot(xte, a.v)
    res = run_race(a, Xtr, ytr, Xte, yte, a.nc) if a.model == "race" else run_mlp(a, Xtr, ytr, Xte, yte, a.nc)
    res["config"] = vars(a)
    os.makedirs(OUT, exist_ok=True)
    name = f"{a.model}_d{a.depth}{'_res' if a.residual else ''}_P{a.train}_L{a.L}_s{a.seed}"
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    print(name, "test", res["test_acc"], "train", res["train_acc"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("race", "mlp"), required=True)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--residual", type=int, default=0)
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule-seed", type=int, default=0)
    ap.add_argument("--v", type=int, default=8)
    ap.add_argument("--nc", type=int, default=8)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--s", type=int, default=2)
    ap.add_argument("--L", type=int, default=3)
    main(ap.parse_args())
