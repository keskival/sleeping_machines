"""E66: word-keyed experts for the event LM: context detectors and copy memories keyed on words, not exact suffixes.

Keys per position (the character to predict is x_t):
  P  = the current partial word (characters since the last space)
  W1 = the previous complete word (hashed), W2 = the two previous words (hashed)
Experts added to the stage-2 mixture (E63): KT counts of the next character given (W1, P) and (W2, P); copy memories
keyed on (W1, P) and (W2, P): the character that followed the most recent earlier position with the same key (found by one
stable sort: the previous element of the same key group). These match on partial, word-level context ("after this word,
this prefix continued like this last time"), a step from exact-suffix copying toward race attention with fuzzy keys.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e62_charlm as S1  # noqa: E402
import e63_mixlm as S2  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e66")
A = S1.A
MOD = (1 << 61) - 1


def word_keys(x):
    """per position t: hashes of (W1, P) and (W2, P) describing the context before x_t (polynomial hashes mod 2^64,
    computed by one vectorized pass per position-in-word, so memory stays O(n) machine words)."""
    n = len(x); x = np.asarray(x); is_sp = x == 0
    wid = np.cumsum(np.r_[0, is_sp[:-1]])                             # word index of each position (space ends a word)
    start = np.maximum.accumulate(np.where(is_sp, np.arange(n) + 1, 0))   # first position of the current word
    pos = np.arange(n) - start                                         # position inside the word (spaces: -1 -> 0 below)
    hl = np.zeros(n, np.uint64); c = x.astype(np.uint64); m = ~is_sp
    k = 0
    with np.errstate(over="ignore"):
        while True:
            idx = np.flatnonzero(m & (pos == k))
            if not len(idx):
                break
            hl[idx] = (hl[idx - 1] * np.uint64(131) if k else np.uint64(0)) + c[idx]
            k += 1
        part = np.r_[np.uint64(0), hl[:-1]]                            # partial-word hash before x_t
        ends = np.flatnonzero(is_sp)
        wh = np.r_[np.where(ends > 0, hl[np.maximum(ends - 1, 0)], np.uint64(0)), hl[-1]]
        w1 = np.where(wid >= 1, wh[np.maximum(wid - 1, 0)], np.uint64(0x9E3779B97F4A7C15))
        w2 = np.where(wid >= 2, wh[np.maximum(wid - 2, 0)], np.uint64(0xC2B2AE3D27D4EB4F))
        k1 = w1 * np.uint64(1000003) + part
        k2 = (w2 * np.uint64(1000003) + w1) * np.uint64(1000033) + part
    return k1.view(np.int64), k2.view(np.int64)


class KeyCounts:
    def __init__(self, keys, y):
        pair = keys.astype(np.int64) * 0 + keys                        # keys < 2^61
        order = np.lexsort((y, keys)); ks, ys = keys[order], y[order]
        new = np.r_[True, (ks[1:] != ks[:-1]) | (ys[1:] != ys[:-1])]
        grp = np.cumsum(new) - 1
        self.pk = ks[new]; self.py = ys[new]; self.pc = np.bincount(grp)
        newk = np.r_[True, self.pk[1:] != self.pk[:-1]]
        self.ctx = self.pk[newk]; first = np.flatnonzero(newk)
        self.n = np.add.reduceat(self.pc, first); self.u = np.diff(np.r_[first, len(self.pk)])

    def kt(self, keys, y, back=None):
        """KT estimate, or Witten-Bell backoff to `back` (probabilities of y under a coarser expert) when given."""
        j = np.searchsorted(self.ctx, keys); jj = np.clip(j, 0, len(self.ctx) - 1)
        hit = self.ctx[jj] == keys
        n = np.where(hit, self.n[jj], 0); u = np.where(hit, self.u[jj], 0)
        # count of (key, y): search in the (pk, py) lexicographic list
        lo = np.searchsorted(self.pk, keys, side="left"); hi = np.searchsorted(self.pk, keys, side="right")
        c = np.zeros(len(keys))
        for a in range(A):                                             # small alphabet: probe each symbol's position
            m = (y == a) & (hi > lo)
            if m.any():
                seg_lo, seg_hi = lo[m], hi[m]
                pos = seg_lo + np.array([np.searchsorted(self.py[s0:s1], a) for s0, s1 in zip(seg_lo, seg_hi)])
                ok = (pos < seg_hi) & (self.py[np.minimum(pos, len(self.py) - 1)] == a)
                c[np.flatnonzero(m)[ok]] = self.pc[pos[ok]]
        if back is not None:
            return np.where(n > 0, (c + u * back) / np.maximum(n + u, 1e-9), back)
        return (c + 0.5) / (n + A / 2)


def copy_by_key(keys, stream, eps):
    """probability of the true character under 'what followed the most recent earlier position with the same key'."""
    order = np.argsort(keys, kind="stable"); ks = keys[order]
    prev = np.full(len(keys), -1); same = np.r_[False, ks[1:] == ks[:-1]]
    prev[order[1:][same[1:]]] = order[:-1][same[1:]]
    P = np.full(len(keys), 1.0 / A)
    has = prev >= 0
    P[has] = np.where(stream[prev[has]] == stream[has], 1 - eps, eps / (A - 1))
    return P


def sleeping_hedge(P, awake, sel, eta, share=0.0):
    """sleeping experts (Freund, Schapire, Singer, Warmuth 1997), one weight vector per selector context: the mixture uses
    only awake experts; after each event, awake experts are multiplied by (p_i / p_mix)^eta, which conserves the awake
    experts' total weight (conserved multiplicative credit); sleeping experts neither vote nor learn."""
    T, E = P.shape; out = np.empty(T)
    W = {}
    Pl = P.tolist(); Al = awake.tolist(); Sl = sel.tolist()
    for t in range(T):
        w = W.get(Sl[t])
        if w is None:
            w = W[Sl[t]] = np.ones(E) / E
        a = np.array(Al[t]); p = np.maximum(np.array(Pl[t]), 1e-12)
        wa = w[a]; tot = wa.sum(); pm = float((wa * p[a]).sum() / tot)
        out[t] = pm
        nw = wa * (p[a] / pm) ** eta
        nw *= tot / nw.sum()                                           # the awake experts' total weight is conserved
        if share:                                                      # fixed share: tracking a switching best expert
            nw = (1 - share) * nw + share * nw.mean()
        w[a] = nw
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=1_000_000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--test", type=int, default=1_000_000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    x = S1.load(); train = x[:a.D]
    valid = x[90_000_000:90_000_000 + a.test]; test = x[95_000_000:95_000_000 + a.test]
    orders = [S1.Order(train, k) for k in range(a.K + 1)]; word = S2.WordOrder(train)
    tk1, tk2 = word_keys(train); C1 = KeyCounts(tk1, train); C2 = KeyCounts(tk2, train)
    def experts(stream, eps):
        base = S2.experts(orders, word, train, stream, a.K, eps)
        tail = 400                                                       # the keys need only the last two words of context
        k1, k2 = word_keys(np.r_[train[-tail:], stream]); s1, s2 = k1[tail:], k2[tail:]
        full = np.r_[train, stream]
        cp1 = copy_by_key(np.r_[tk1, s1], full, eps)[len(train):]; cp2 = copy_by_key(np.r_[tk2, s2], full, eps)[len(train):]
        wp = base[:, a.K + 2]                                            # the partial-word expert of E63 (key P)
        p1 = C1.kt(s1, stream, back=wp); p2 = C2.kt(s2, stream, back=p1)  # (W1,P) -> P ; (W2,P) -> (W1,P)
        return np.column_stack([base, p1, p2, cp1, cp2])

    def awake_mask(Pm):
        """an expert is awake when it has information: count experts whose probability differs from uniform-ish fallbacks
        are always awake (they back off); copy experts only when they found a match."""
        aw = np.ones(Pm.shape, bool)
        nc = len(S2.COPY_L)
        cols = list(range(Pm.shape[1] - 4 - nc, Pm.shape[1] - 4)) + [Pm.shape[1] - 2, Pm.shape[1] - 1]
        for c in cols:
            aw[:, c] = np.abs(Pm[:, c] - 1.0 / A) > 1e-12
        return aw
    res = {"args": vars(a)}
    PV, PT = experts(valid, 0.05), experts(test, 0.05)                   # computed once, shared by all mixers
    for with_w in (0, 1):
        best = None
        Pv = PV if with_w else PV[:, :-4]
        nc = len(S2.COPY_L); cpcols = Pv[:, -nc - 4:-4] if with_w else Pv[:, -nc:]
        found = np.abs(cpcols - 1.0 / A) > 1e-12; longest = np.where(found.any(1), nc - np.argmax(found[:, ::-1], 1), 0)
        prev_ch = np.r_[train[-1], valid[:-1]]
        for eta in (0.03, 0.1, 0.3):
            for W in (10, 25, 50):
                bpc = float(np.mean(-np.log2(S2.hedge(Pv, longest * A + prev_ch, eta, W))))
                if best is None or bpc < best[0]:
                    best = (bpc, eta, W)
        bpc_v, eta, W = best
        Pt = PT if with_w else PT[:, :-4]
        cpcols = Pt[:, -nc - 4:-4] if with_w else Pt[:, -nc:]
        found = np.abs(cpcols - 1.0 / A) > 1e-12; longest = np.where(found.any(1), nc - np.argmax(found[:, ::-1], 1), 0)
        prev_ch = np.r_[valid[-1], test[:-1]]
        key = "word_keys" if with_w else "base"
        res[key] = {"valid_bpc": bpc_v, "test_bpc": float(np.mean(-np.log2(S2.hedge(Pt, longest * A + prev_ch, eta, W)))),
                    "chosen": {"eta": eta, "W": W}}
        if with_w:
            res[key]["expert_test_bpc"] = [float(np.mean(-np.log2(np.maximum(Pt[:, i], 1e-12)))) for i in range(-4, 0)]
    # sleeping-experts mixer on the full expert set (eta chosen on validation)
    Pv, Pt = PV, PT
    def longest_sel(Pm, prev):
        nc = len(S2.COPY_L); cp = Pm[:, -nc - 4:-4]; found = np.abs(cp - 1.0 / A) > 1e-12
        return np.where(found.any(1), nc - np.argmax(found[:, ::-1], 1), 0) * A + prev
    selv = longest_sel(Pv, np.r_[train[-1], valid[:-1]]); selt = longest_sel(Pt, np.r_[valid[-1], test[:-1]])
    best = min(((float(np.mean(-np.log2(sleeping_hedge(Pv, awake_mask(Pv), selv, eta, sh)))), eta, sh)
                for eta in (0.3, 1.0) for sh in (0.01, 0.05, 0.2)))
    res["sleeping"] = {"valid_bpc": best[0], "eta": best[1], "share": best[2],
                       "test_bpc": float(np.mean(-np.log2(sleeping_hedge(Pt, awake_mask(Pt), selt, best[1], best[2]))))}
    res["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)
    with open(os.path.join(OUT, f"wordkeys_D{a.D}_K{a.K}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
