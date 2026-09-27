"""E60: DVS128 Gesture with native event machinery (counting only, no gradients).

Pipeline per gesture (all event-driven, costs counted):
  cells    128x128 pixels -> G x G cells (polarity ignored); refractory hold: a cell emits at most one event per R ms
  motion   Reichardt pairs = hold/trigger nodes: a cell event whose neighbour in direction d (8 directions) fired between
           lo and hi ms earlier emits a motion event (motion from the neighbour toward this cell), with speed band by lag
  tokens   motion event -> token (region, direction, speed); region = coarse position (4x4) either absolute or relative to
           the running centroid of the gesture's cell events (--rel, the person-relative code)
  classify (a) bag of motion events: class-conditional token counts (multinomial naive Bayes, Dirichlet a)
           (b) class-conditional event world models over the token sequence: state = (last token, onset bucket), predict
               next token (backed off to the token frequencies)
Protocol: official split (users 1-23 train, 24-29 test); --eval val selects on users 20-23 held out from 1-19.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import dvs_data as D  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e60")
DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def motion_tokens(x, y, t, G, R, lo, hi, rel, nreg=4, S=50.0):
    """cell events (refractory R ms) -> onsets (cell silent >= S ms before) -> motion events: an onset whose neighbour in
    direction d had an onset lo..hi ms earlier while the opposite neighbour did not (opponent veto). Returns token ids
    (time-ordered), times (s), and the number of onset events."""
    if len(t) == 0:
        return np.zeros(0, np.int64), np.zeros(0), 0
    c = (x.astype(np.int64) * G // 128) * G + (y.astype(np.int64) * G // 128)
    b = t // int(R * 1000)
    key = c * (int(b.max()) + 1) + b
    _, first = np.unique(key, return_index=True)                     # refractory: first event per (cell, R-ms bin)
    cc, tt = c[first], t[first].astype(np.int64)
    o = np.lexsort((tt, cc)); cc, tt = cc[o], tt[o]                   # grouped by cell, time-ordered
    prev_same = np.r_[-1, np.where(cc[1:] == cc[:-1], tt[:-1], -1)]
    onset = (prev_same < 0) | (tt - prev_same >= S * 1000)            # the cell was silent for >= S ms
    cc, tt = cc[onset], tt[onset]
    cx, cy = cc // G, cc % G
    big = int(tt.max()) + 10
    skey = cc * big + tt                                              # sorted (cells grouped, times ordered)
    def recent(nx, ny):
        ok = (nx >= 0) & (nx < G) & (ny >= 0) & (ny < G)
        ncell = nx * G + ny
        j = np.searchsorted(skey, ncell * big + tt, side="left") - 1
        jj = np.clip(j, 0, len(skey) - 1); prev_t = skey[jj] - ncell * big
        valid = ok & (j >= 0) & (prev_t >= 0) & (prev_t < big)
        lag = np.where(valid, (tt - prev_t) / 1000.0, np.inf)
        return lag
    lags = [recent(cx - dx, cy - dy) for dx, dy in DIRS]
    if rel:
        order_t = np.argsort(tt, kind="stable"); inv = np.empty_like(order_t); inv[order_t] = np.arange(len(tt))
        mx = (np.cumsum(cx[order_t]) / np.arange(1, len(tt) + 1))[inv]; my = (np.cumsum(cy[order_t]) / np.arange(1, len(tt) + 1))[inv]
        rx = np.clip(((cx - mx) / G + 0.5) * nreg, 0, nreg - 1).astype(np.int64)
        ry = np.clip(((cy - my) / G + 0.5) * nreg, 0, nreg - 1).astype(np.int64)
    else:
        rx, ry = cx * nreg // G, cy * nreg // G
    toks, times = [], []
    for di in range(8):
        lag = lags[di]; opp = lags[(di + 4) % 8]
        valid = (lag >= lo) & (lag <= hi) & ~((opp >= lo) & (opp <= hi))   # opponent veto
        speed = (lag > (lo + hi) / 2).astype(np.int64)
        tok = ((rx * nreg + ry) * 8 + di) * 2 + speed
        toks.append(tok[valid]); times.append(tt[valid])
    tok = np.concatenate(toks); tim = np.concatenate(times)
    o = np.argsort(tim, kind="stable")
    return tok[o], tim[o] / 1e6, len(cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=32)
    ap.add_argument("--R", type=float, default=10.0, help="refractory hold per cell (ms)")
    ap.add_argument("--lo", type=float, default=2.0)
    ap.add_argument("--hi", type=float, default=40.0)
    ap.add_argument("--S", type=float, default=50.0, help="onset: the cell was silent for >= S ms")
    ap.add_argument("--rel", type=int, default=1)
    ap.add_argument("--O", type=int, default=6, help="onset buckets over the gesture")
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--eval", default="val", choices=("val", "test"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    train = D.split("train")
    if a.eval == "val":
        fit = [s for s in train if s[5] <= 19]; ev = [s for s in train if s[5] >= 20]
    else:
        fit, ev = train, D.split("test")
    V = 4 * 4 * 8 * 2; K = 11
    def enc(samples):
        out = []
        for lab, x, y, p, t, u in samples:
            tok, tim, ncell = motion_tokens(x, y, t, a.G, a.R, a.lo, a.hi, a.rel, S=a.S)
            dur = max(tim[-1] - tim[0], 1e-3) if len(tim) else 1.0
            ob = np.minimum(((tim - (tim[0] if len(tim) else 0)) / dur * a.O).astype(int), a.O - 1) if len(tim) else tim
            out.append((lab, tok, ob, len(t), ncell))
        return out
    F, E = enc(fit), enc(ev)
    bag = np.full((K, V), a.a); T = np.full((K, V * a.O, V), a.a / V)
    for lab, tok, ob, _, _ in F:
        np.add.at(bag[lab], tok, 1.0)
        if len(tok) > 1:
            np.add.at(T[lab], (tok[:-1] * a.O + ob[:-1], tok[1:]), 1.0)
    lbag = np.log(bag / bag.sum(1, keepdims=True))
    Tn = T + 5.0 * (bag / bag.sum(1, keepdims=True))[:, None, :]      # back off transitions to the class's token rates
    lT = np.log(Tn / Tn.sum(2, keepdims=True))
    ok_bag = ok_seq = 0; raw_ev = cell_ev = mot_ev = 0
    for lab, tok, ob, nraw, ncell in E:
        if len(tok) == 0:
            continue
        sb = lbag[:, tok].sum(1)
        ss = lT[:, tok[:-1] * a.O + ob[:-1], tok[1:]].sum(1) if len(tok) > 1 else sb
        ok_bag += int(np.argmax(sb) == lab); ok_seq += int(np.argmax(ss) == lab)
        raw_ev += nraw; cell_ev += ncell; mot_ev += len(tok)
    n = len(E)
    res = {"args": vars(a), f"{a.eval}_bag_acc": ok_bag / n, f"{a.eval}_seq_acc": ok_seq / n, "n": n,
           "raw_events_per_gesture": raw_ev / n, "onset_events_per_gesture": cell_ev / n,
           "motion_events_per_gesture": mot_ev / n, "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(res), flush=True)
    with open(os.path.join(OUT, f"dvs_G{a.G}_R{a.R:g}_S{a.S:g}_lo{a.lo:g}_hi{a.hi:g}_rel{a.rel}_O{a.O}_{a.eval}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
