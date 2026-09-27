"""E67: learning from the timing of races (THEORY §101) on MNIST.

A race layer: class (or hidden) nodes with clock rates exp(score); one race per example: each node draws an exponential
clock, the first to fire is the winner, T the decision time. Learning rules compared at equal examples:
  exact      softmax gradient descent: dW ∝ (onehot(y) - softmax(s)) x^T
  race-time  the local estimate of §101: dW ∝ (onehot(y) - lambda * T) x^T   (each node: its rate times the decision time)
  winner     race perceptron: dW ∝ (onehot(y) - onehot(winner)) x^T         (identity only)
Two layers (--hidden H): hidden payloads h = relu(U x); the output race's error e (from time, as above) reaches the hidden
units through the transpose of the forward synapses (--feedback sym) or through fixed random feedback (--feedback random).
"""
import argparse
import gzip
import json
import os
import time

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "results", "e67")


def mnist():
    def imgs(f):
        with gzip.open(os.path.join(DATA, f)) as g:
            return np.frombuffer(g.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
    def labs(f):
        with gzip.open(os.path.join(DATA, f)) as g:
            return np.frombuffer(g.read(), np.uint8, offset=8).astype(np.int64)
    return (imgs("train-images-idx3-ubyte.gz"), labs("train-labels-idx1-ubyte.gz"),
            imgs("t10k-images-idx3-ubyte.gz"), labs("t10k-labels-idx1-ubyte.gz"))


def error_signal(s, y, rule, rng, races):
    """(K,) error for one example: exact softmax, race-time estimate (averaged over `races` races), or winner-only."""
    z = s - s.max(); lam = np.exp(z)
    if rule == "exact":
        p = lam / lam.sum()
    elif rule == "race_time":
        p = np.zeros_like(lam)
        for _ in range(races):
            T = (rng.exponential(size=len(lam)) / lam).min()           # the decision time of one race
            p += lam * T                                               # each node: own rate x decision time
        p /= races
    else:                                                              # winner only: the identity of the first to fire
        w = int(np.argmin(rng.exponential(size=len(lam)) / lam)); p = np.zeros_like(lam); p[w] = 1.0
    e = -p; e[y] += 1.0
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="race_time", choices=("exact", "race_time", "winner"))
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--feedback", default="sym", choices=("sym", "random"))
    ap.add_argument("--races", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); rng = np.random.default_rng(a.seed)
    Xtr, Ytr, Xte, Yte = mnist(); K = 10
    if a.hidden:
        U = rng.normal(0, 1 / np.sqrt(784), (a.hidden, 784)); W = rng.normal(0, 1 / np.sqrt(a.hidden), (K, a.hidden))
        B = rng.normal(0, 1 / np.sqrt(K), (a.hidden, K))                # fixed random feedback
    else:
        W = np.zeros((K, 784))
    curve = []; Wavg = None; Uavg = None; navg = 0                                # iterate averaging: a slow synaptic trace
    for ep in range(a.epochs):
        for i in rng.permutation(len(Xtr)):
            x = Xtr[i]
            if a.hidden:
                pre = U @ x; h = np.maximum(pre, 0); s = W @ h
                e = error_signal(s, Ytr[i], a.rule, rng, a.races)
                back = (W.T @ e) if a.feedback == "sym" else (B @ e)
                W += a.lr * np.outer(e, h); U += a.lr * np.outer(back * (pre > 0), x)
            else:
                s = W @ x; e = error_signal(s, Ytr[i], a.rule, rng, a.races)
                W += a.lr * np.outer(e, x)
            if ep == a.epochs - 1:                                  # running mean over the last epoch
                navg = navg + 1 if Wavg is not None else 1
                Wavg = W.copy() if Wavg is None else Wavg + (W - Wavg) / navg
                if a.hidden:
                    Uavg = U.copy() if Uavg is None else Uavg + (U - Uavg) / navg
        Wm = Wavg if (ep == a.epochs - 1 and Wavg is not None) else W; Um = Uavg if (ep == a.epochs - 1 and Uavg is not None) else (U if a.hidden else None)
        S = (np.maximum(Xte @ Um.T, 0) @ Wm.T) if a.hidden else Xte @ Wm.T
        acc = float(np.mean(S.argmax(1) == Yte)); curve.append({"epoch": ep + 1, "test_acc": acc})
        print(json.dumps({"rule": a.rule, "hidden": a.hidden, "feedback": a.feedback, "epoch": ep + 1, "test_acc": acc,
                          "wall_s": round(time.time() - t0)}), flush=True)
    with open(os.path.join(OUT, f"{a.rule}_h{a.hidden}_{a.feedback}_r{a.races}_s{a.seed}.json"), "w") as f:
        json.dump({"args": vars(a), "curve": curve}, f)


if __name__ == "__main__":
    main()
