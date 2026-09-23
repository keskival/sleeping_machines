#!/usr/bin/env python3
"""E6 diagnosis: is a hidden-layer race network limited by its representation or its readout?

Trains CRL briefly, then compares the output race's accuracy with an offline linear
readout of the complete hidden spike pattern (no race, no time pressure), for the
trained network and for its random initialisation.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import Config, RaceNet, load, to_events  # noqa: E402


def hidden_codes(net, x, batch=250):
    fired, first, dec = [], [], []
    for i in range(0, len(x), batch):
        t, idx = to_events(x[i:i + batch])
        st = net.forward(t, idx)
        fired.append(st["fired"].astype(np.float32))
        # hidden spikes that arrived before the output decision
        before = st["fired"] & (st["freeze1"] <= st["freeze2"][:, :1])
        first.append(before.sum(1) / np.maximum(st["fired"].sum(1), 1))
        dec.append(st["freeze2"][:, 0])
    return np.concatenate(fired), np.concatenate(first), np.concatenate(dec)


def linear_readout(ftr, ytr, fte, yte, epochs=10, lr=0.1):
    rng = np.random.default_rng(0)
    W = np.zeros((ftr.shape[1], 10), np.float32); b = np.zeros(10, np.float32)
    for _ in range(epochs):
        for i in range(0, len(ytr), 64):
            ii = rng.permutation(len(ytr))[:64] if False else slice(i, i + 64)
            z = ftr[ii] @ W + b
            p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            p[np.arange(len(p)), ytr[ii]] -= 1
            W -= lr * ftr[ii].T @ p / len(p); b -= lr * p.mean(0)
    return float(((fte @ W + b).argmax(1) == yte).mean())


if __name__ == "__main__":
    xtr, ytr, xva, yva = load("mnist", 10000)
    xtr, ytr, xva, yva = xtr[:12000], ytr[:12000], xva[:3000], yva[:3000]
    cfg = Config(variant="crl_fa", hidden=500, eta_hid=0.003)
    net = RaceNet(cfg, 784, 10, float(np.isfinite(xtr).sum(1).mean()), np.random.default_rng(0))
    for label, epochs in (("random init", 0), ("after 3 epochs of CRL", 3)):
        for _ in range(epochs):
            perm = np.random.default_rng(1).permutation(len(ytr))
            for i in range(0, len(perm), 32):
                ii = perm[i:i + 32]
                t, idx = to_events(xtr[ii]); net.teach(net.forward(t, idx), ytr[ii])
        ftr, _, _ = hidden_codes(net, xtr)
        fva, frac, dec = hidden_codes(net, xva)
        race = float(np.mean([net.forward(*to_events(xva[i:i + 250]))["winner"] == yva[i:i + 250]
                              for i in range(0, len(yva), 250)]))
        print(f"{label}: race readout {race:.3f} | linear readout of all hidden spikes {linear_readout(ftr, ytr, fva, yva):.3f} "
              f"| decision at median t={np.median(dec):.3f}, after {np.median(frac) * 100:.0f}% of hidden spikes")
