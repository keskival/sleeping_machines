"""E63: character LM stage 2: many counting and copy experts mixed by conserved multiplicative credit (Hedge; §83, §94).

Experts (each gives the probability of the next character):
  order-k context counts, k = 0..K (Krichevsky-Trofimov estimate (c + 1/2) / (n + A/2))
  the Witten-Bell blend of stage 1 (E62)
  the current partial word (characters since the last space, up to 10) as a context
  copy memories: the character that followed the previous occurrence of the last L characters, L in {3,5,8,12,16,24,32}
    (probability 1 - e on the copied character, e spread over the rest; uniform when there is no earlier occurrence)
Mixture: weights w_i(t) ∝ exp(eta * sum of log p_i over the last W characters that had the same selector context) (Hedge
with forgetting, one weight set per selector = the previous character): multiplicative, local, O(experts) per character.
eta, W, e chosen on validation; test scored once. Vectorized: only each expert's probability of the true character is needed.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e62_charlm as S1  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e63")
A = S1.A
COPY_L = (3, 5, 8, 12, 16, 24, 32)


def kt_true(order, codes, y):
    """KT probability of the true next character y under an order's counts (codes: context codes, -1 = none)."""
    ok = codes >= 0
    j = np.searchsorted(order.ctx, codes); jj = np.clip(j, 0, len(order.ctx) - 1)
    hit = ok & (order.ctx[jj] == codes)
    n = np.where(hit, order.n[np.clip(jj, 0, len(order.n) - 1)], 0)
    q = codes * A + y; i = np.searchsorted(order.pairs, q); ii = np.clip(i, 0, len(order.pairs) - 1)
    c = np.where(hit & (order.pairs[ii] == q), order.cnt[ii], 0)
    return (c + 0.5) / (n + A / 2)


def word_codes(x):
    """context code of the current partial word (characters since the last space, up to 10) for each position."""
    sp = np.where(x == 0, np.arange(len(x)), -1); last = np.maximum.accumulate(sp)
    plen = np.minimum(np.arange(len(x)) - last - 1, 10); plen[plen < 0] = 0
    out = np.full(len(x), -1, np.int64)
    for k in range(0, 11):
        m = plen == k
        if m.any():
            out[m] = S1.ctx_codes(x, k)[m] + (A ** 10) * (k + 1) if k > 0 else (A ** 10) * 1
    return out


class WordOrder:
    def __init__(self, train):
        c = word_codes(train); m = c >= 0; pair = c[m] * A + train[m]
        self.pairs, self.cnt = np.unique(pair, return_counts=True)
        ctx = self.pairs // A; self.ctx, first = np.unique(ctx, return_index=True)
        self.n = np.add.reduceat(self.cnt, first)


def copy_true(stream_u8, start, stop, y, eps):
    """probability of the true character under each copy expert (lengths COPY_L)."""
    s = stream_u8.tobytes(); last = {L: {} for L in COPY_L}
    P = np.full((stop - start, len(COPY_L)), 1.0 / A)
    for i in range(max(COPY_L), stop):
        for li, L in enumerate(COPY_L):
            key = s[i - L:i]
            if i >= start:
                j = last[L].get(key)
                if j is not None:
                    P[i - start, li] = (1 - eps) if s[j] == s[i] else eps / (A - 1)
            last[L][key] = i
    return P


def experts(orders, word, train, stream, K, eps):
    """(T, E) probabilities of the true next character for every expert on `stream` (context from the train tail)."""
    full = np.r_[train[-100_000:], stream]; off = 100_000
    y = stream
    cols = []
    for k in range(K + 1):
        codes = S1.ctx_codes(full, k)[off:]
        cols.append(kt_true(orders[k], codes, y))
    CH = 250_000; wb = []
    for s0 in range(0, len(stream), CH):
        s1 = min(len(stream), s0 + CH)
        p = S1.count_probs(orders, np.r_[train[-K:], stream], K, K + s0, K + s1)
        wb.append(p[np.arange(s1 - s0), stream[s0:s1]])
    cols.append(np.concatenate(wb))
    wc = word_codes(full)[off:]
    cols.append(kt_true(word, wc, y))
    Pc = copy_true(full.astype(np.uint8), off, len(full), y, eps)
    return np.column_stack(cols + [Pc[:, i] for i in range(Pc.shape[1])])


def hedge(P, sel, eta, W):
    """windowed Hedge mixture per selector group: weights from the last W positions of the same group (exclusive)."""
    T, E = P.shape; L = np.log(np.maximum(P, 1e-12)); out = np.empty(T)
    for g in np.unique(sel):
        idx = np.flatnonzero(sel == g)
        cs = np.vstack([np.zeros((1, E)), np.cumsum(L[idx], 0)])
        n = len(idx); lo = np.maximum(np.arange(n) - W, 0)
        S = cs[np.arange(n)] - cs[lo]                                  # sum over the previous <= W same-group positions
        z = eta * S; z -= z.max(1, keepdims=True); w = np.exp(z); w /= w.sum(1, keepdims=True)
        out[idx] = (w * P[idx]).sum(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=10_000_000)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--test", type=int, default=1_000_000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    x = S1.load(); train = x[:a.D]
    valid = x[90_000_000:90_000_000 + a.test]; test = x[95_000_000:95_000_000 + a.test]
    orders = [S1.Order(train, k) for k in range(a.K + 1)]; word = WordOrder(train)
    res = {"args": vars(a)}
    def selectors(P, prev):
        """0: none; 1: previous character; 2: longest copy match (index of the longest copy expert that is not uniform);
        3: both."""
        nc = len(COPY_L); cp = P[:, -nc:]
        found = np.abs(cp - 1.0 / A) > 1e-12
        longest = np.where(found.any(1), nc - np.argmax(found[:, ::-1], 1), 0)
        return {0: np.zeros(len(P), int), 1: prev, 2: longest, 3: longest * A + prev}
    best = None
    for eps in (0.05, 0.2):
        Pv = experts(orders, word, train, valid, a.K, eps); sels = selectors(Pv, np.r_[train[-1], valid[:-1]])
        res.setdefault("valid_expert_bpc", {})[f"eps{eps}"] = (-np.log2(np.maximum(Pv, 1e-12))).mean(0).round(3).tolist()
        for eta in (0.03, 0.1, 0.3):
            for W in (10, 25, 50, 200):
                for use_sel in (0, 1, 2, 3):
                    bpc = float(np.mean(-np.log2(hedge(Pv, sels[use_sel], eta, W))))
                    if best is None or bpc < best[0]:
                        best = (bpc, eps, eta, W, use_sel)
    bpc_v, eps, eta, W, use_sel = best
    Pt = experts(orders, word, train, test, a.K, eps); selt = selectors(Pt, np.r_[valid[-1], test[:-1]])[use_sel]
    res.update({"valid_bpc_mix": bpc_v, "chosen": {"eps": eps, "eta": eta, "W": W, "selector": use_sel},
                "test_bpc_mix": float(np.mean(-np.log2(hedge(Pt, selt, eta, W)))),
                "test_bpc_wb_stage1": float(np.mean(-np.log2(np.maximum(Pt[:, a.K + 1], 1e-12)))),
                "experts": [f"order{k}" for k in range(a.K + 1)] + ["wb_blend", "word"] + [f"copy{L}" for L in COPY_L],
                "wall_s": round(time.time() - t0, 1)})
    print(json.dumps({k: v for k, v in res.items() if k != "valid_expert_bpc"}), flush=True)
    with open(os.path.join(OUT, f"mix_D{a.D}_K{a.K}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
