#!/usr/bin/env python3
"""E16: the counterfactual routing gradient in an ordinary mixture of experts (THEORY §19).

Top-1 routing over E linear-softmax experts on 14×14 MNIST intensities (no spikes).
Router variants:

  gate       Switch-style: the router learns only through the chosen expert's gate value
  gate_lb    gate + the Switch load-balancing auxiliary loss
  boundary   gate + the boundary term: for the m near-miss experts (highest router scores
             after the chosen one) the loss is measured by running them in shadow, and the
             router is moved by ρ_σ(margin) · (L_i − L_chosen) · ∇(s_i − s_chosen)
  top2       dense top-2 reference (two experts run per sample, standard gradients)

Measures: accuracy, router quality (share of test inputs sent to the expert whose loss is
lowest), load balance (normalised entropy of expert usage), expert runs per training sample.

    python experiments/e16_moe.py --router boundary --shadow 2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import mnist  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e16")


def data(n_train, n_test):
    x, y = mnist("train")
    x = x.reshape(-1, 14, 2, 14, 2).mean((2, 4)).reshape(-1, 196)
    x = np.hstack([x, np.ones((len(x), 1), np.float32)])        # bias
    return x[:n_train], y[:n_train], x[50000:50000 + n_test], y[50000:50000 + n_test]


def piecewise(n_train, n_test, rng, d=20, clusters=8, k=4, spread=4.0):
    """A task that needs routing: inputs come from `clusters` Gaussian clusters, and each
    cluster labels its points by its own random linear rule. One linear expert cannot fit it;
    correctly routed experts can."""
    centers = rng.normal(0, spread, (clusters, d))
    teachers = rng.normal(0, 1, (clusters, d, k))

    def draw(n):
        cl = rng.integers(0, clusters, n)
        x = centers[cl] + rng.normal(0, 1, (n, d))
        y = np.einsum("nd,ndk->nk", x - centers[cl], teachers[cl]).argmax(1)
        return np.hstack([x / spread, np.ones((n, 1))]).astype(np.float32), y
    xtr, ytr = draw(n_train)
    xte, yte = draw(n_test)
    return xtr, ytr, xte, yte


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class MoE:
    def __init__(self, d, k, E, rng):
        self.R = rng.normal(0, 0.01, (d, E))
        self.X = rng.normal(0, 0.01, (E, d, k))
        self.E, self.k = E, k
        self.runs = 0

    def expert_probs(self, x, e):
        """Class probabilities of expert e[i] for each x[i]."""
        return softmax(np.einsum("nd,ndk->nk", x, self.X[e]))

    def logits(self, x, e):
        return np.einsum("nd,ndk->nk", x, self.X[e])

    def step(self, x, y, a, lr):
        """Output = softmax(g_e · u_e(x)), the gate scaling the chosen expert's logits
        (Switch-style), so the router learns through the gate value: the pathwise term."""
        n, rows, onehot = len(y), np.arange(len(y)), np.eye(self.k)[y]
        s = x @ self.R
        g = softmax(s)
        order = np.argsort(-s, 1)
        ds = np.zeros((n, self.E))                                   # ∂L/∂s, accumulated
        if a.router == "top2":
            e2 = order[:, :2]
            w = softmax(s[rows[:, None], e2])                        # renormalised gate over the two
            u = [self.logits(x, e2[:, j]) for j in range(2)]
            p = [softmax(u[j]) for j in range(2)]
            self.runs += 2 * n
            mix = w[:, 0] * p[0][rows, y] + w[:, 1] * p[1][rows, y]
            for j in range(2):
                dz = (w[:, j] * p[j][rows, y] / mix)[:, None] * (p[j] - onehot)
                np.add.at(self.X, e2[:, j], -lr * np.einsum("nd,nk->ndk", x, dz) / n)
            dLdw = -np.stack([p[j][rows, y] for j in range(2)], 1) / mix[:, None]
            dsw = w * (dLdw - (w * dLdw).sum(1, keepdims=True))
            np.add.at(ds, (rows[:, None].repeat(2, 1), e2), dsw)
            self.R -= lr * x.T @ ds / n
            return
        c = order[:, 0]
        gc = g[rows, c]
        u = self.logits(x, c)
        p = softmax(gc[:, None] * u)
        self.runs += n
        dz = p - onehot                                              # ∂L/∂(g_c u)
        np.add.at(self.X, c, -lr * np.einsum("nd,nk->ndk", x, dz * gc[:, None]) / n)
        dLdg = (dz * u).sum(1)
        ds += (np.eye(self.E)[c] - g) * (gc * dLdg)[:, None]          # softmax Jacobian, row c
        if a.router == "gate_lb":
            f = np.bincount(c, minlength=self.E) / n
            dP = a.lb * self.E * f / n                               # ∂(lb·E·Σ f·P̄)/∂g, per sample
            ds += g * (dP[None, :] - (g * dP[None, :]).sum(1, keepdims=True))
        if a.router == "boundary":
            Lc = -np.log(np.maximum(p[rows, y], 1e-9))
            for j in range(1, a.shadow + 1):
                alt = order[:, j]
                pa = softmax(g[rows, alt][:, None] * self.logits(x, alt))
                self.runs += n
                La = -np.log(np.maximum(pa[rows, y], 1e-9))
                z = (s[rows, c] - s[rows, alt]) / a.sigma
                rho = np.exp(-z) / (1 + np.exp(-z)) ** 2 / a.sigma   # logistic density at the margin
                coef = rho * (La - Lc)                               # ∂E[L]/∂(s_alt − s_c)
                ds[rows, alt] += coef
                ds[rows, c] -= coef
        self.R -= lr * x.T @ ds / n

    def evaluate(self, x, y):
        rows = np.arange(len(y))
        s = x @ self.R
        c = s.argmax(1)
        acc = float((self.logits(x, c).argmax(1) == y).mean())
        losses = np.stack([-np.log(np.maximum(self.expert_probs(x, np.full(len(y), e))[rows, y], 1e-9))
                           for e in range(self.E)], 1)
        quality = float((c == losses.argmin(1)).mean())
        use = np.bincount(c, minlength=self.E) / len(y)
        nz = use[use > 0]
        balance = float(-(nz * np.log(nz)).sum() / np.log(self.E))
        return acc, quality, balance


def main(a):
    rng = np.random.default_rng(a.seed)
    if a.task == "mnist":
        xtr, ytr, xte, yte = data(a.train, a.test)
    else:
        xtr, ytr, xte, yte = piecewise(a.train, a.test, np.random.default_rng(1000 + a.seed))
    m = MoE(xtr.shape[1], int(ytr.max()) + 1, a.experts, rng)
    curve = []
    t0 = time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), a.batch):
            ii = perm[i:i + a.batch]
            m.step(xtr[ii], ytr[ii], a, a.lr)
        acc, q, bal = m.evaluate(xte, yte)
        curve.append({"acc": acc, "router_quality": q, "balance": bal})
        print(f"{a.router} m={a.shadow} epoch {ep + 1}: acc {acc:.4f} router quality {q:.3f} "
              f"balance {bal:.3f} ({time.time() - t0:.0f}s)", flush=True)
    res = {"config": vars(a), "curve": curve, **curve[-1], "expert_runs_per_sample": m.runs / (a.epochs * len(ytr))}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"moe_{a.task}_{a.router}_m{a.shadow}_{a.tag or 'run'}_s{a.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="boundary", choices=("gate", "gate_lb", "boundary", "top2"))
    ap.add_argument("--shadow", type=int, default=1, help="near-miss experts run in shadow (boundary)")
    ap.add_argument("--sigma", type=float, default=1.0, help="routing noise scale for the boundary density")
    ap.add_argument("--lb", type=float, default=0.01, help="load-balancing coefficient (gate_lb)")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--task", default="piecewise", choices=("piecewise", "mnist"))
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--test", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    main(ap.parse_args())
