"""E40: SHD with the architecture §71 prescribes: classes as unions of zones over onset-referenced parts.

Parts (depth 1): "band p fired within [k·tau, (k+1)·tau) after the utterance's first event" (the onset is the
reference, §56.2); a part's hold window is tau, and each event also feeds the neighbouring bin with weight `blur`
(a wider, weaker PSP: timing tolerance).
Zones (depth 2): prototype nodes, each a k-of-n coincidence over parts: charge = sum of weights of the parts that fired;
conserved weight budget per prototype; a price (threshold) per prototype.
Classes (depth 3): a class fires when the first of its prototypes crosses (a union of zones). Readout: the prototype
with the largest margin charge - price (the first to cross a rising threshold).
Learning (errors only, native): the teacher's nearest prototype (its largest charge: the near miss, §57) is pulled
toward the utterance's parts (conserved, fractional); the false winner is weakened on the same parts (conserved);
prices: missed teacher's prototype cheaper, false winner dearer. Prototypes start as copies of random training
utterances of their class (each prototype's weights = that utterance's parts, normalized).
"""
import argparse
import json
import os
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "shd")
OUT = os.path.join(os.path.dirname(__file__), "results", "e40")
BANDS, SLOTS = 70, 10


def onset_parts(X, pool, tau, blur):
    nb, nk = BANDS // pool, int(np.ceil(1.0 / tau))
    F = np.zeros((len(X), nb, nk), np.float32)
    for n, x in enumerate(X):
        idx = np.flatnonzero(np.isfinite(x))
        if not len(idx):
            continue
        b, t = (idx // SLOTS) // pool, x[idx]
        k = np.minimum(((t - t.min()) / tau).astype(int), nk - 1)
        np.add.at(F[n], (b, k), 1.0)
    F = np.minimum(F, 1.0)                                   # a part fires once (one-shot)
    if blur:
        G = F.copy()
        G[:, :, 1:] += blur * F[:, :, :-1]; G[:, :, :-1] += blur * F[:, :, 1:]
        F = np.minimum(G, 1.0)
    return F.reshape(len(X), -1)


class Zones:
    def __init__(self, F, y, K, per, rng, kappa):
        self.cls = np.repeat(np.arange(K), per)
        W = []
        for c in range(K):
            pick = rng.choice(np.flatnonzero(y == c), per, replace=False)
            W.append(F[pick] + 1e-3)
        self.W = np.concatenate(W)
        self.W /= self.W.sum(1, keepdims=True)                  # budget 1 per prototype
        self.price = np.zeros(len(self.W)); self.kappa = kappa; self.updates = 0

    def margins(self, F):
        return F @ self.W.T - self.price

    def predict(self, F):
        return self.cls[self.margins(F).argmax(1)]

    def teach(self, f, y, eta):
        m = f @ self.W.T - self.price
        w = int(m.argmax())
        if self.cls[w] == y:
            return
        own = np.flatnonzero(self.cls == y)
        near = int(own[np.argmax(m[own])])                       # the teacher's near miss
        g = f / max(f.sum(), 1e-9)
        self.W[near] = (1 - eta) * self.W[near] + eta * g         # conserved, fractional pull
        self.W[w] *= 1 - eta * f / max(f.max(), 1e-9)             # conserved weakening
        self.W[w] /= self.W[w].sum()
        self.price[near] -= self.kappa; self.price[w] += self.kappa
        self.updates += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--blur", type=float, default=0.5)
    ap.add_argument("--per", type=int, default=20, help="prototypes (zones) per class")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--kappa", type=float, default=0.0005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    d = np.load(os.path.join(ROOT, "shd_700.npz"))
    F, Ft = onset_parts(d["Xtr"], a.pool, a.tau, a.blur), onset_parts(d["Xte"], a.pool, a.tau, a.blur)
    y, yt = d["ytr"], d["yte"]
    rng = np.random.default_rng(a.seed)
    net = Zones(F, y, 20, a.per, rng, a.kappa)
    curve = [{"epoch": 0, "test": float((net.predict(Ft) == yt).mean())}]
    print(json.dumps(curve[-1]), flush=True)
    for ep in range(1, a.epochs + 1):
        for i in rng.permutation(len(y)):
            net.teach(F[i], y[i], a.eta)
        curve.append({"epoch": ep, "train": float((net.predict(F) == y).mean()),
                      "test": float((net.predict(Ft) == yt).mean()), "updates": net.updates})
        print(json.dumps(curve[-1]), flush=True)
    res = {"args": vars(a), "parts": int(F.shape[1]), "part_events_per_utterance": float((F > 0).sum(1).mean()),
           "prototypes": int(len(net.W)), "curve": curve, "final_test": curve[-1]["test"],
           "best_test": max(c["test"] for c in curve), "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, f"zones_per{a.per}_tau{a.tau}_b{a.blur}_s{a.seed}{'_' + a.tag if a.tag else ''}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}))


if __name__ == "__main__":
    main()
