"""E78: the lower envelope of THEORY §108 on text8. The native experts of E66 (counting orders, blend, word experts,
copy memories, word-keyed counts and copies) and the time-vector language model of E77, each alone and mixed:
  bayes       Bayes mixture, eta = 1, uniform prior (the theorem's regret: log2(E) / T bits per character)
  share       fixed share, alpha chosen on validation (tracks the locally best expert)
  hedge_sel   E63/E66's windowed Hedge keyed by the longest copy match (heuristic, tuned on validation)
each with and without the E77 column. Evaluated on the test positions E77 scored (all but its first character and last
half-window), validation on the 200k characters E77 used for early stopping.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e62_charlm as S1  # noqa: E402
import e63_mixlm as S2  # noqa: E402
from e66_wordkeys import word_keys, KeyCounts, copy_by_key  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e78")
A = S1.A


def native_experts(train, K, stream, eps=0.05):
    orders = [S1.Order(train, k) for k in range(K + 1)]; word = S2.WordOrder(train)
    tk1, tk2 = word_keys(train); C1 = KeyCounts(tk1, train); C2 = KeyCounts(tk2, train)
    base = S2.experts(orders, word, train, stream, K, eps)
    tail = 400
    k1, k2 = word_keys(np.r_[train[-tail:], stream]); s1, s2 = k1[tail:], k2[tail:]
    full = np.r_[train, stream]
    cp1 = copy_by_key(np.r_[tk1, s1], full, eps)[len(train):]; cp2 = copy_by_key(np.r_[tk2, s2], full, eps)[len(train):]
    p1 = C1.kt(s1, stream, back=base[:, K + 2]); p2 = C2.kt(s2, stream, back=p1)
    P = np.column_stack([base, p1, p2, cp1, cp2])
    nc = len(S2.COPY_L); cp = base[:, -nc:]; found = np.abs(cp - 1.0 / A) > 1e-12
    longest = np.where(found.any(1), nc - np.argmax(found[:, ::-1], 1), 0)
    return P, longest


def bayes(P):
    L = np.log(np.maximum(P, 1e-12)); cs = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(L, 0)[:-1]])
    z = cs - cs.max(1, keepdims=True); w = np.exp(z); w /= w.sum(1, keepdims=True)
    return (w * P).sum(1)


def share(P, alpha):
    T, E = P.shape; w = np.full(E, 1.0 / E); out = np.empty(T); Pl = np.maximum(P, 1e-12)
    for t in range(T):
        p = Pl[t]; m = float(w @ p); out[t] = m
        w = w * p / m; w = (1 - alpha) * w + alpha / E
    return out


def bpc(p):
    return float(np.mean(-np.log2(np.maximum(p, 1e-12))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=1_000_000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--e77", default="tvlm_D1000000_p5_r1_M128-128-64_s0")
    ap.add_argument("--valid", type=int, default=200_000)
    ap.add_argument("--test", type=int, default=1_000_000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    x = S1.load(); train = x[:a.D]
    valid = x[90_000_000:90_000_000 + a.valid]; test = x[95_000_000:95_000_000 + a.test]
    R77 = os.path.join(os.path.dirname(__file__), "results", "e77")
    if a.e77 == "none":                                                          # native experts only
        tv_v = np.full(len(valid), 1.0 / A, np.float32); tv_t = np.full(len(test), 1.0 / A, np.float32)
    else:
        tv_v = np.load(os.path.join(R77, a.e77 + "_ptrue_valid.npy")); tv_t = np.load(os.path.join(R77, a.e77 + "_ptrue_test.npy"))
    res = {"args": vars(a)}
    data = {}
    for name, stream, tv, prev0 in (("valid", valid, tv_v, train[-1]), ("test", test, tv_t, valid[-1])):
        P, longest = native_experts(train, a.K, stream)
        m = np.isfinite(tv)                                                     # positions E77 scored
        sel = longest * A + np.r_[prev0, stream[:-1]]
        data[name] = (P[m], np.column_stack([P, np.nan_to_num(tv, nan=1.0 / A)])[m], sel[m])
    Pv, Pvx, sv = data["valid"]; Pt, Ptx, st = data["test"]
    res["experts_test_bpc"] = [round(bpc(Pt[:, i]), 4) for i in range(Pt.shape[1])]
    res["tv_alone_test_bpc"] = bpc(Ptx[:, -1]); res["best_native_expert_test_bpc"] = min(res["experts_test_bpc"])
    for tag, (PV, PT) in (("native", (Pv, Pt)),) + ((("native+tv", (Pvx, Ptx)),) if a.e77 != "none" else ()):
        r = {"bayes": bpc(bayes(PT))}
        al = min((0.0001, 0.001, 0.01, 0.03), key=lambda al_: bpc(share(PV, al_)))
        r["share"] = bpc(share(PT, al)); r["share_alpha"] = al
        best = min(((bpc(S2.hedge(PV, sv, eta, W)), eta, W) for eta in (0.03, 0.1, 0.3) for W in (10, 25, 50)))
        r["hedge_sel"] = bpc(S2.hedge(PT, st, best[1], best[2])); r["hedge_sel_choice"] = best[1:]
        res[tag] = r
        print(json.dumps({tag: r}), flush=True)
    res["positions"] = int(len(Pt))
    print(json.dumps({k: res[k] for k in ("tv_alone_test_bpc", "best_native_expert_test_bpc", "positions")}), flush=True)
    with open(os.path.join(OUT, f"envelope_D{a.D}_{a.e77}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
