"""E32: accuracy vs compute on a sparse temporal task, event learner vs a clocked dense model (THEORY §55).

Task: E27's. N = 12 channels spike at most once per episode of length H = 10; K = 4 patterns "B within Δ after A
unless C", plus none (a tenth of episodes are vetoed near-misses). Optionally the episode sits inside S units of
silence (padding), as in a stream where informative events are rare.

Dense side (clocked): spikes binned at resolution δ into an (N, H/δ) raster, a 1-D temporal convolution (F filters,
receptive field R bins covering the longest pattern, ReLU), global max over time, a linear readout; trained with
backprop + Adam on the same number of episodes. Cost per episode = multiply-adds of the forward pass:
F·N·R per bin × (H + S)/δ bins, plus the readout (training adds ~2× for the backward pass).
Event side: E27's learner (delays, coincidence windows, veto; errors-only updates). Cost per episode = synaptic
events (input spikes × fan-out + detector spikes), independent of δ and of silence; updates counted separately.

Output: accuracy and ops/episode for each δ and padding, for the dense model; E27's numbers for the event model.
"""
import argparse
import json
import os
import time

import numpy as np

import e27_veto as E27

OUT = os.path.join(os.path.dirname(__file__), "results", "e32")


def raster(t, H, dt):
    nb = int(round(H / dt))
    X = np.zeros((len(t), nb), np.float32)
    ok = np.isfinite(t)
    X[np.flatnonzero(ok), np.minimum((t[ok] / dt).astype(int), nb - 1)] = 1.0
    return X


class Conv:
    """1-D temporal conv (N in-channels, F filters, width R) -> ReLU -> max over time -> linear (K+1)."""

    def __init__(self, N, F, R, C, rng):
        self.W = rng.normal(0, 1 / np.sqrt(N * R), (F, N, R)).astype(np.float32)
        self.b = np.zeros(F, np.float32)
        self.V = rng.normal(0, 1 / np.sqrt(F), (C, F)).astype(np.float32)
        self.c = np.zeros(C, np.float32)
        self.P = [self.W, self.b, self.V, self.c]
        self.m = [np.zeros_like(p) for p in self.P]
        self.v = [np.zeros_like(p) for p in self.P]
        self.step = 0

    def forward(self, X):                                   # X: (B, N, T)
        B, N, T = X.shape
        F, _, R = self.W.shape
        Xp = np.pad(X, ((0, 0), (0, 0), (R - 1, 0)))       # causal
        cols = np.lib.stride_tricks.sliding_window_view(Xp, R, axis=2)   # (B, N, T, R)
        Z = np.einsum("bntr,fnr->bft", cols, self.W) + self.b[None, :, None]
        A = np.maximum(Z, 0)
        am = A.argmax(2)
        h = np.take_along_axis(A, am[:, :, None], 2)[:, :, 0]
        return h @ self.V.T + self.c, (cols, Z, am, h)

    def train_step(self, X, y, lr=3e-3):
        logits, (cols, Z, am, h) = self.forward(X)
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1; p /= len(y)
        gV = p.T @ h; gc = p.sum(0)
        gh = p @ self.V                                      # (B, F)
        gh = gh * (np.take_along_axis(Z, am[:, :, None], 2)[:, :, 0] > 0)
        colsel = cols[np.arange(len(y))[:, None], :, am, :]  # (B, F, N, R): input window at each filter's max
        gW = np.einsum("bf,bfnr->fnr", gh, colsel); gb = gh.sum(0)
        self.step += 1
        for i, g in enumerate((gW, gb, gV, gc)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.step); vh = self.v[i] / (1 - 0.999 ** self.step)
            self.P[i] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def macs(self, T_bins):
        F, N, R = self.W.shape
        return F * N * R * T_bins + self.V.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dts", default="0.05,0.1,0.25,0.5,1.0")
    ap.add_argument("--filters", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=200000, help="training episodes (E27 used 200k)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pads", default="0,90,990", help="silence around an episode, in time units")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    N, K, H, q = 12, 4, 10.0, 0.4
    rows = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        pats = E27.make_task(N, K, H, rng)                   # same task generator and seeds as E27
        ev = np.random.default_rng(99)
        test = [E27.sample(pats, N, H, q, ev) for _ in range(2000)]
        for dt in map(float, a.dts.split(",")):
            R = int(np.ceil(4.0 / dt))                       # receptive field covers the longest pattern (Δ ≤ 3)
            net = Conv(N, a.filters, R, K + 1, np.random.default_rng(seed))
            t0 = time.time()
            for _ in range(a.episodes // a.batch):
                eps = [E27.sample(pats, N, H, q, rng) for _ in range(a.batch)]
                X = np.stack([raster(t, H, dt) for t, _ in eps]); y = np.array([y for _, y in eps])
                net.train_step(X, y)
            Xt = np.stack([raster(t, H, dt) for t, _ in test]); yt = np.array([y for _, y in test])
            acc = float((net.forward(Xt)[0].argmax(1) == yt).mean())
            for pad in map(float, a.pads.split(",")):
                bins = int(round((H + pad) / dt))
                r = {"seed": seed, "filters": a.filters, "dt": dt, "pad": pad, "acc": acc, "macs_per_episode": net.macs(bins),
                     "train_macs_per_episode": 3 * net.macs(bins), "wall_s": round(time.time() - t0, 1)}
                rows.append(r)
                print(json.dumps(r), flush=True)
    with open(os.path.join(OUT, f"dense_F{a.filters}.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows}, f, indent=1)
    print("EXIT-OK")


if __name__ == "__main__":
    main()
