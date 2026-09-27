"""E38: the compositional architecture on real asynchronous data: Spiking Heidelberg Digits (THEORY §65).

Input: E22's encoding (70 frequency bands × 10 slots; the first spike of each band in each slot; ~356 events per
utterance, times in seconds within the first second). Train 8156, test 2264 utterances, 20 classes.

Parts (a generic basis, nothing about digits): one directional hold node per ordered band pair (p, q) and window
scale Δ: it fires each time a band-q event follows a band-p event within (0, Δ]. Bands are pooled by `--pool` to
keep the basis small (70 / pool bands). Part events per utterance are counted (the node's spikes).
Classes: each class node accumulates the spikes of the parts routed to it with weights W[c, part] (non-leaky; the
readout is the class with the most charge at the end, i.e. the first to cross a rising threshold).
Learning (native, §60): errors only; the teacher's weights on the parts that fired are pulled (conserved budget per
class, fractional step), the false winner's weakened (conserved), per-class prices. Control (--rule softmax): the
same part counts into a dense softmax layer trained by backprop (Adam), to separate the representation from the rule.
"""
import argparse
import json
import os
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "shd")
OUT = os.path.join(os.path.dirname(__file__), "results", "e38")
BANDS, SLOTS = 70, 10


def events(x, pool):
    """(band, time) events of one utterance from the (band, slot) first-spike encoding."""
    idx = np.flatnonzero(np.isfinite(x))
    return (idx // SLOTS) // pool, x[idx]


def part_counts(x, pool, scales):
    b, t = events(x, pool)
    nb = BANDS // pool
    o = np.argsort(t); b, t = b[o], t[o]
    feats = []
    for w in scales:
        C = np.zeros((nb, nb), np.float32)
        j = 0
        for i in range(len(t)):                              # pairs (i -> k) with 0 < t_k - t_i <= w
            k = np.searchsorted(t, t[i], "right"); k2 = np.searchsorted(t, t[i] + w, "right")
            if k2 > k:
                np.add.at(C[b[i]], b[k:k2], 1.0)
        feats.append(C.ravel())
    return np.concatenate(feats)


def onset_parts(x, pool, tau, nbins):
    """reference parts (§56.2): band p fired within [k tau, (k+1) tau) after the utterance's first event."""
    b, t = events(x, pool)
    nb = BANDS // pool
    C = np.zeros((nb, nbins), np.float32)
    if len(t):
        k = np.minimum(((t - t.min()) / tau).astype(int), nbins - 1)
        np.add.at(C, (b, k), 1.0)
    return C.ravel()


def deep_parts(x, pool, w1, w2):
    """depth 2 (§64): a pair of pair-parts. Layer-1 part (p->q within w1) fires at q's time; layer 2 counts
    part (p,q) followed by part (r,s) within w2, on coarse bands to keep the basis small."""
    b, t = events(x, pool)
    nb = BANDS // pool
    o = np.argsort(t); b, t = b[o], t[o]
    P1b, P1t = [], []
    for i in range(len(t)):
        k = np.searchsorted(t, t[i], "right"); k2 = np.searchsorted(t, t[i] + w1, "right")
        for j in range(k, k2):
            P1b.append(b[i] * nb + b[j]); P1t.append(t[j])
    P1b, P1t = np.array(P1b, int), np.array(P1t)
    o = np.argsort(P1t); P1b, P1t = P1b[o], P1t[o]
    C = np.zeros((nb * nb, nb * nb), np.float32)
    for i in range(len(P1t)):
        k = np.searchsorted(P1t, P1t[i] + 1e-4); k2 = np.searchsorted(P1t, P1t[i] + w2, "right")
        if k2 > k:
            np.add.at(C[P1b[i]], P1b[k:k2], 1.0)
    return C.ravel()


FEAT = {"pairs": True, "onset": 0.0, "deep": 0}


def build(X, pool, scales):
    out = []
    for x in X:
        f = [part_counts(x, pool, scales)] if FEAT["pairs"] else []
        if FEAT["onset"]:
            f.append(onset_parts(x, pool, FEAT["onset"], int(np.ceil(1.0 / FEAT["onset"]))))
        if FEAT["deep"]:
            f.append(deep_parts(x, FEAT["deep"], 0.02, 0.15))
        out.append(np.concatenate(f))
    return np.stack(out)


class Native:
    def __init__(self, K, P, rng, budget=None, kappa=0.02):
        self.W = rng.uniform(0, 1e-3, (K, P)); self.budget = budget or P * 1e-2; self.kappa = kappa
        self.thr = np.zeros(K); self.updates = 0

    def scores(self, F):
        return F @ self.W.T - self.thr

    def _pull(self, c, f, eta):
        tot = self.W[c].sum()
        self.W[c] += eta * f / max(f.sum(), 1e-9) * max(tot, self.budget * 0.01)
        if self.W[c].sum() > self.budget:
            self.W[c] *= self.budget / self.W[c].sum()

    def _drop(self, c, f, eta):
        tot = self.W[c].sum()
        self.W[c] *= 1 - eta * f / max(f.max(), 1e-9)
        if tot >= self.budget:
            self.W[c] *= tot / max(self.W[c].sum(), 1e-12)

    def teach(self, f, y, eta):
        c = int(np.argmax(self.scores(f[None])[0]))
        if c == y:
            return
        self._pull(y, f, eta); self._drop(c, f, eta)
        self.thr[y] -= self.kappa; self.thr[c] += self.kappa
        self.updates += 1


def softmax_train(F, y, Ft, K, epochs, rng, lr=1e-3):
    import torch
    torch.set_num_threads(1)
    Xt, yt = torch.tensor(F), torch.tensor(y)
    lin = torch.nn.Linear(F.shape[1], K)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        perm = torch.tensor(rng.permutation(len(y)))
        for i in range(0, len(y), 64):
            j = perm[i:i + 64]
            loss = torch.nn.functional.cross_entropy(lin(Xt[j]), yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return lin(torch.tensor(Ft)).argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=5, help="bands pooled per node (70/pool bands)")
    ap.add_argument("--scales", default="0.02,0.08")
    ap.add_argument("--onset", type=float, default=0.0, help="> 0: add onset-referenced parts with this bin (s)")
    ap.add_argument("--deep", type=int, default=0, help="> 0: add depth-2 pair-of-pair parts on bands pooled by this")
    ap.add_argument("--nopairs", type=int, default=0)
    ap.add_argument("--rule", default="native", choices=("native", "softmax"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    FEAT.update(pairs=not a.nopairs, onset=a.onset, deep=a.deep)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    d = np.load(os.path.join(ROOT, "shd_700.npz"))
    scales = [float(s) for s in a.scales.split(",")]
    cache = os.path.join(OUT, f"feats_p{a.pool}_s{a.scales}_o{a.onset}_d{a.deep}_np{a.nopairs}.npz")
    if os.path.exists(cache):
        c = np.load(cache); F, Ft = c["F"], c["Ft"]
    else:
        F, Ft = build(d["Xtr"], a.pool, scales), build(d["Xte"], a.pool, scales)
        np.savez_compressed(cache, F=F, Ft=Ft)
    y, yt = d["ytr"], d["yte"]
    F = np.log1p(F); Ft = np.log1p(Ft)                        # dendritic saturation of repeated part events
    rng = np.random.default_rng(a.seed)
    K = 20
    curve = []
    if a.rule == "softmax":
        pred = softmax_train(F, y, Ft, K, a.epochs, rng)
        curve.append({"epoch": a.epochs, "test": float((pred == yt).mean())})
    else:
        net = Native(K, F.shape[1], rng)
        for ep in range(1, a.epochs + 1):
            for i in rng.permutation(len(y)):
                net.teach(F[i], y[i], a.eta)
            te = float((net.scores(Ft).argmax(1) == yt).mean()); tr = float((net.scores(F).argmax(1) == y).mean())
            curve.append({"epoch": ep, "train": tr, "test": te, "updates": net.updates})
            print(json.dumps(curve[-1]), flush=True)
    res = {"args": vars(a), "parts": int(F.shape[1]), "part_events_per_utterance": float(np.expm1(F).sum(1).mean()),
           "curve": curve, "final_test": curve[-1]["test"], "wall_s": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}))
    with open(os.path.join(OUT, f"{a.rule}_p{a.pool}_s{a.scales}_o{a.onset}_d{a.deep}_np{a.nopairs}{'_' + a.tag if a.tag else ''}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
