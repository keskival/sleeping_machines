"""E79: geometric mixing of the native experts, i.e. a race whose clock rates are summed log-probabilities (THEORY §108(e)):
p_mix(c) ∝ exp( Σ_e w_{s,e} log p_e(c) ), with a weight set per selector s (longest copy match × previous character) and the
exact local log-loss gradient w_{s,e} += lr (log p_e(y) - Σ_c p_mix(c) log p_e(c)). This is the product-of-experts
mixing of PAQ/cmix-style compressors (logistic mixing), written as a race.

Experts (full next-character distributions): KT counts of orders 0..K, the Witten-Bell blend, the partial-word expert, and
a copy expert (the character after the most recent earlier occurrence of the longest matching context, among COPY_L, found
within a window of `copy_window` characters; 0 = unbounded).
Reported on the same 1M test characters as E62-E66: each expert alone, linear Hedge over the same experts (E63's mixer),
and the geometric race mixer (a) frozen after an online pass over the 1M validation characters and (b) online on the
test stream (prequential, as compressors are scored). Copy memory unbounded and restricted to 256 characters (the E64
Transformer's context), so that memory length does not decide the comparison.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e62_charlm as S1  # noqa: E402
import e63_mixlm as S2  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e79")
A = S1.A


def lookup(obj, codes):
    """counts (len, A) and totals for context codes under an object with pairs/cnt/ctx/n (Order or WordOrder)."""
    L = len(codes); C = np.zeros((L, A), np.float32); n = np.zeros(L, np.float32)
    ok = codes >= 0
    j = np.searchsorted(obj.ctx, codes); jj = np.clip(j, 0, len(obj.ctx) - 1)
    hit = ok & (obj.ctx[jj] == codes); n[hit] = obj.n[jj[hit]]
    for a in range(A):
        q = codes * A + a; i = np.searchsorted(obj.pairs, q); ii = np.clip(i, 0, len(obj.pairs) - 1)
        h = hit & (obj.pairs[ii] == q); C[h, a] = obj.cnt[ii[h]]
    return C, n


def copy_pred(full_u8, start, stop, window):
    """predicted character (or -1) and the index of the longest matching length, for positions start..stop of `full`."""
    s = full_u8.tobytes(); last = {L: {} for L in S2.COPY_L}
    pred = np.full(stop - start, -1, np.int64); li_ = np.zeros(stop - start, np.int64)
    for i in range(max(S2.COPY_L), stop):
        if i >= start:
            for li in range(len(S2.COPY_L) - 1, -1, -1):
                L = S2.COPY_L[li]; j = last[L].get(s[i - L:i])
                if j is not None and (window == 0 or i - j <= window):
                    pred[i - start] = s[j]; li_[i - start] = li + 1; break
        for L in S2.COPY_L:
            last[L][s[i - L:i]] = i
    return pred, li_


def expert_logp(orders, word, train, stream, K, window, eps=0.05):
    """(T, E, A) log-probabilities of every expert, and the selector (longest copy length index * A + previous char)."""
    off = 100_000; full = np.r_[train[-off:], stream]; T = len(stream)
    E = K + 1 + 1 + 1 + 1; LP = np.empty((T, E, A), np.float32)
    for k in range(K + 1):
        C, n = lookup(orders[k], S1.ctx_codes(full, k)[off:])
        LP[:, k] = np.log((C + 0.5) / (n[:, None] + A / 2))
    CH = 200_000
    for s0 in range(0, T, CH):
        s1 = min(T, s0 + CH)
        LP[s0:s1, K + 1] = np.log(np.maximum(S1.count_probs(orders, np.r_[train[-K:], stream], K, K + s0, K + s1), 1e-9))
    C, n = lookup(word, S2.word_codes(full)[off:]); LP[:, K + 2] = np.log((C + 0.5) / (n[:, None] + A / 2))
    pred, li = copy_pred(full.astype(np.uint8), off, len(full), window)
    cp = np.full((T, A), np.log(1.0 / A), np.float32); has = pred >= 0
    cp[has] = np.log(eps / (A - 1)); cp[np.flatnonzero(has), pred[has]] = np.log(1 - eps)
    LP[:, K + 3] = cp
    sel = li * A + np.r_[train[-1], stream[:-1]]
    return LP, sel


def race_mix(LP, y, sel, lr, W=None, update=True):
    """geometric (race) mixer, one weight set per selector; returns log2-loss per position and the weights."""
    T, E, _ = LP.shape; nS = int(sel.max()) + 1 if W is None else W.shape[0]
    W = np.full((max(nS, int(sel.max()) + 1), E), 1.0 / E, np.float32) if W is None else W.copy()
    loss = np.empty(T)
    for t in range(T):
        lp = LP[t]; w = W[sel[t]]
        z = w @ lp; z -= z.max(); p = np.exp(z); p /= p.sum()
        loss[t] = -np.log2(max(p[y[t]], 1e-12))
        if update:
            W[sel[t]] = w + lr * (lp[:, y[t]] - lp @ p)
    return loss, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=1_000_000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n", type=int, default=1_000_000, help="test characters")
    ap.add_argument("--nvalid", type=int, default=1_000_000)
    ap.add_argument("--e77", default="none", help="E77 run tag: add its full distribution as one more racing expert")
    ap.add_argument("--windows", default="0,256")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    x = S1.load(); train = x[:a.D]
    valid = x[90_000_000:90_000_000 + a.nvalid]; test = x[95_000_000:95_000_000 + a.n]
    tv = {}
    if a.e77 != "none":                                                     # E77's distributions; unscored positions uniform
        R77 = os.path.join(os.path.dirname(__file__), "results", "e77")
        for nm, ln in (("valid", len(valid)), ("test", len(test))):
            lp = np.load(os.path.join(R77, f"{a.e77}_logp_{nm}.npy")).astype(np.float32)[:ln]
            tv[nm] = np.where(np.isfinite(lp), lp, np.log(1.0 / A)).astype(np.float32)
    def with_tv(LP, nm):
        return np.concatenate([LP, tv[nm][:, None, :]], 1) if tv else LP
    orders = [S1.Order(train, k) for k in range(a.K + 1)]; word = S2.WordOrder(train)
    res = {"args": vars(a)}
    for window in [int(w) for w in a.windows.split(",")]:
        r = {}
        LPv, sv = expert_logp(orders, word, train, valid, a.K, window); LPv = with_tv(LPv, "valid")
        Pv = np.exp(LPv[np.arange(len(valid)), :, valid])
        best = min(((float(np.mean(-np.log2(S2.hedge(Pv, sv, eta, Wn)))), eta, Wn) for eta in (0.03, 0.1, 0.3) for Wn in (10, 25, 50)))
        cand = {}
        for lr in (0.002, 0.01, 0.05):
            lv, Wv = race_mix(LPv, valid, sv, lr); cand[lr] = (float(lv.mean()), Wv)
        lr = min(cand, key=lambda k: cand[k][0]); Wv = cand[lr][1]
        r["race_lr"] = lr; r["race_valid_bpc"] = cand[lr][0]
        del LPv, Pv
        LPt, st = expert_logp(orders, word, train, test, a.K, window); LPt = with_tv(LPt, "test")
        r["experts_test_bpc"] = [round(float(np.mean(-LPt[np.arange(len(test)), e, test]) / np.log(2)), 4) for e in range(LPt.shape[1])]
        Pt = np.exp(LPt[np.arange(len(test)), :, test])
        r["linear_hedge_test_bpc"] = float(np.mean(-np.log2(S2.hedge(Pt, st, best[1], best[2])))); r["linear_hedge_choice"] = best[1:]
        r["race_frozen_test_bpc"] = float(race_mix(LPt, test, st, lr, Wv, update=False)[0].mean())
        r["race_online_test_bpc"] = float(race_mix(LPt, test, st, lr, Wv, update=True)[0].mean())
        res[f"copy_window_{window or 'unbounded'}"] = r
        print(json.dumps({f"window_{window}": {k: v for k, v in r.items()}}), flush=True)
        del LPt, Pt
    with open(os.path.join(OUT, f"race_mixer_D{a.D}_K{a.K}_e77{a.e77}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
