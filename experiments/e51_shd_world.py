"""E51: SHD with class-conditional world models: one semi-Markov event network per class (E48's model on band spikes).

Stream: every spike of an utterance, channels pooled into B frequency bands, time-ordered. Per class, an event network
whose state is (last band, window since the last spike [bank 1, 3, 10, 30 ms, longer], onset bucket [time since the
utterance's first spike, O buckets over 1 s]) and whose detectors give (a) the hazard of the next spike in the current
window and (b) the band of the next spike, learned by counts over the class's training utterances (Dirichlet
smoothing a). An utterance is assigned to the class whose network gives its spike stream the highest likelihood.
Native and event-driven: per spike, one state update, the expiry events crossed, and one synapse read per class.
"""
import argparse
import json
import os
import time

import h5py
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "shd")
OUT = os.path.join(os.path.dirname(__file__), "results", "e51")
GAPS = np.array([0.001, 0.003, 0.01, 0.03])
EDGES = np.r_[0.0, GAPS, np.inf]


VAL_SPEAKERS = (3, 6)                                      # held-out speakers: validation like the test (unseen voices)


def utterances(split, B, part=None):
    """part: None (all), "fit"/"val" (random 10% of train), "fit_spk"/"val_spk" (train minus / only VAL_SPEAKERS)."""
    with h5py.File(os.path.join(ROOT, f"shd_{split}.h5"), "r") as f:
        times, units, labels = f["spikes"]["times"], f["spikes"]["units"], np.array(f["labels"])
        val = np.random.default_rng(0).random(len(labels)) < 0.1
        spk = np.isin(np.array(f["extra"]["speaker"]), VAL_SPEAKERS)
        for i in range(len(labels)):
            if part == "fit" and val[i]: continue
            if part == "val" and not val[i]: continue
            if part == "fit_spk" and spk[i]: continue
            if part == "val_spk" and not spk[i]: continue
            t = np.asarray(times[i], np.float64); u = (np.asarray(units[i]).astype(np.int64) * B) // 700
            o = np.argsort(t, kind="stable")
            yield t[o], u[o], int(labels[i])


def features(t, u, B, O, rel=0):
    """per spike (after the first): context index and window bucket, next band, and the gap's span over windows.
    rel=1: bands coded relative to the utterance's running centroid (sum and count of bands so far, per spike),
    for the context and for the predicted next band alike (speaker-invariant coordinates)."""
    gap = np.diff(t); last = u[:-1]; nxt = u[1:]
    if rel:
        m = np.round(np.cumsum(u) / np.arange(1, len(u) + 1)).astype(np.int64)[:-1]
        last = np.clip(last - m + B // 2, 0, B - 1); nxt = np.clip(nxt - m + B // 2, 0, B - 1)
    k = np.searchsorted(GAPS, gap)
    ob = np.minimum(((t[:-1] - t[0]) * O).astype(int), O - 1)
    ctx = last * O + ob
    span = np.clip(gap[:, None] - EDGES[None, :-1], 0, np.diff(EDGES)[None, :])
    return ctx, k, nxt, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=35)
    ap.add_argument("--O", type=int, default=5)
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--timing", type=int, default=1)
    ap.add_argument("--eval", default="val", choices=("val", "test", "spk"),
                    help="val: random 10% of train; spk: held-out training speakers; test: the test set")
    ap.add_argument("--rel", type=int, default=0, help="1: speaker-invariant relative band coding")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    K, nb, NCX = 20, len(GAPS) + 1, a.B * a.O
    TP = np.full((K, NCX, nb, a.B), a.a, np.float32); HN = np.full((K, NCX, nb), a.a); HE = np.full((K, NCX, nb), 1e-3)
    fit_part = {"val": "fit", "spk": "fit_spk", "test": None}[a.eval]
    for t, u, y in utterances("train", a.B, fit_part):
        if len(t) < 2: continue
        ctx, k, nxt, span = features(t, u, a.B, a.O, a.rel)
        np.add.at(TP[y], (ctx, k, nxt), 1.0); np.add.at(HN[y], (ctx, k), 1.0); np.add.at(HE[y], ctx, span)
    logP = np.log(TP / TP.sum(-1, keepdims=True)); H = HN / HE; logH = np.log(H)
    ok = n = 0; conf = np.zeros((K, K), int)
    score_it = {"val": lambda: utterances("train", a.B, "val"), "spk": lambda: utterances("train", a.B, "val_spk"),
                "test": lambda: utterances("test", a.B)}[a.eval]()
    for t, u, y in score_it:
        if len(t) < 2: continue
        ctx, k, nxt, span = features(t, u, a.B, a.O, a.rel)
        ll = logP[:, ctx, k, nxt].sum(1)
        if a.timing:
            ll = ll + logH[:, ctx, k].sum(1) - (H[:, ctx, :] * span[None]).sum((1, 2))
        pred = int(np.argmax(ll)); ok += pred == y; n += 1; conf[y, pred] += 1
    res = {"args": vars(a), f"{a.eval}_acc": ok / n, "n_test": n, "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(res), flush=True)
    with open(os.path.join(OUT, f"world_B{a.B}_O{a.O}_a{a.a:g}_t{a.timing}{'_rel' if a.rel else ''}_{a.eval}.json"), "w") as f:
        json.dump({**res, "confusion": conf.tolist()}, f)


if __name__ == "__main__":
    main()
