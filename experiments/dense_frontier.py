#!/usr/bin/env python3
"""Dense accuracy-vs-cost frontier on latency-coded MNIST (same input as E6).

Energy comparisons must be made at matched accuracy: a small dense network may
match the event network for far fewer operations than a large one.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import Config, load, train_mlp, OUT  # noqa: E402


def pooled(t, pool):
    """Intensity image average-pooled by `pool` x `pool` (fewer inputs, fewer MACs)."""
    x = np.where(np.isfinite(t), 1 - t, 0).astype(np.float32).reshape(-1, 28, 28)
    n = 28 // pool
    return x[:, :n * pool, :n * pool].reshape(-1, n, pool, n, pool).mean((2, 4)).reshape(len(x), -1)


def train_linear(xtr, ytr, xte, yte, epochs=10, lr=0.05, batch=32, seed=0, pool=1):
    rng = np.random.default_rng(seed)
    ftr, fte = pooled(xtr, pool), pooled(xte, pool)
    d = ftr.shape[1]
    W = np.zeros((d, 10), np.float32); b = np.zeros(10, np.float32)
    curve = []
    for _ in range(epochs):
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), batch):
            ii = perm[i:i + batch]
            z = ftr[ii] @ W + b
            p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            p[np.arange(len(ii)), ytr[ii]] -= 1
            p /= len(ii)
            W -= lr * ftr[ii].T @ p; b -= lr * p.sum(0)
        curve.append(float(((fte @ W + b).argmax(1) == yte).mean()))
    macs = d * 10
    return {"hidden": 0, "pool": pool, "curve": curve, "test_acc": curve[-1],
            "inference_work_per_sample": {"macs": macs},
            "train_work": {"macs": 2 * macs * epochs * len(ytr)}}


if __name__ == "__main__":
    xtr, ytr, xte, yte = load("mnist", 0)
    rows = []
    for pool in (1, 2, 4):
        rows.append(train_linear(xtr, ytr, xte, yte, pool=pool))
        print("linear pool", pool, rows[-1]["test_acc"], flush=True)
    for h in (16, 32, 64, 128, 256):
        r = train_mlp(Config(variant="mlp", hidden=h, epochs=10), xtr, ytr, xte, yte)
        r["hidden"] = h
        rows.append(r)
        print("mlp", h, r["test_acc"], flush=True)
    with open(os.path.join(OUT, "dense_frontier.json"), "w") as f:
        json.dump(rows, f, indent=1)
