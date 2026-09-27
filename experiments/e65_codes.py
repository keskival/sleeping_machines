"""E65: learned shared codes for the event LM (THEORY §99c): contexts merged by predictive similarity, used as new contexts.

Training contexts of order k (count >= 2) are clustered into m code units by likelihood k-means (assignment: the code unit
whose next-character distribution gives the context's observed continuations the highest likelihood, i.e. the winner of the
race of §99c; update: pooled counts). Two experts are added to the stage-2 mixture (E63):
  code(last k characters)                                   -> next character (counts per code)
  code(k characters before the last one) x last character   -> next character
Unseen contexts get an extra "unknown" code. Same Hedge mixture and validation-chosen settings as E63; the gain from the
codes is read against E63 at the same data size.
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

OUT = os.path.join(os.path.dirname(__file__), "results", "e65")
A = S1.A


def cluster(order, m, iters, rng):
    """likelihood k-means of an order's contexts (count >= 2) into m codes; returns (ctx codes, code per ctx, q (m, A))."""
    ctx_of_pair = order.pairs // A; sym = order.pairs % A
    j = np.searchsorted(order.ctx, ctx_of_pair)
    keep = order.n >= 2
    Nc = np.zeros((len(order.ctx), A), np.float32); np.add.at(Nc, (j, sym), order.cnt)
    Nc = Nc[keep]; ctxs = order.ctx[keep]
    init = rng.choice(len(Nc), m, replace=False, p=Nc.sum(1) / Nc.sum())
    Q = (Nc[init] + 0.5); Q /= Q.sum(1, keepdims=True)
    for _ in range(iters):
        lq = np.log(Q).T.astype(np.float32)                              # (A, m)
        asg = np.empty(len(Nc), np.int64)
        for s0 in range(0, len(Nc), 200_000):                           # the race: highest likelihood claims the context
            asg[s0:s0 + 200_000] = np.argmax(Nc[s0:s0 + 200_000] @ lq, 1)
        Q = np.full((m, A), 0.5); np.add.at(Q, asg, Nc); Q /= Q.sum(1, keepdims=True)
    return ctxs, asg, Q


def code_of(codes, ctxs, asg, m):
    """code index for context codes (m = unknown)."""
    j = np.searchsorted(ctxs, codes); jj = np.clip(j, 0, len(ctxs) - 1)
    return np.where((codes >= 0) & (ctxs[jj] == codes), asg[jj], m)


def code_expert_counts(train, key, nkeys):
    C = np.full((nkeys, A), 0.5); np.add.at(C, (key, train), 1.0)
    return C / C.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=10_000_000)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--k", type=int, default=4, help="order of the clustered contexts")
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--test", type=int, default=1_000_000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); rng = np.random.default_rng(0)
    x = S1.load(); train = x[:a.D]
    valid = x[90_000_000:90_000_000 + a.test]; test = x[95_000_000:95_000_000 + a.test]
    orders = [S1.Order(train, k) for k in range(a.K + 1)]; word = S2.WordOrder(train)
    ctxs, asg, Q = cluster(orders[a.k], a.m, a.iters, rng)
    m1 = a.m + 1
    def keys(stream_full):
        c1 = code_of(S1.ctx_codes(stream_full, a.k), ctxs, asg, a.m)                 # code of the last k characters
        prev = np.r_[0, stream_full[:-1]]
        c2 = code_of(np.r_[-1, S1.ctx_codes(stream_full, a.k)[:-1]], ctxs, asg, a.m) * A + prev   # code before the last char
        return c1, c2
    k1, k2 = keys(train)
    E1 = code_expert_counts(train, k1, m1); E2 = code_expert_counts(train, k2, m1 * A)
    def experts(stream, eps):
        base = S2.experts(orders, word, train, stream, a.K, eps)
        full = np.r_[train[-100_000:], stream]; c1, c2 = keys(full); c1, c2 = c1[100_000:], c2[100_000:]
        return np.column_stack([base, E1[c1, stream], E2[c2, stream]])
    def selectors(P, prev):
        nc = len(S2.COPY_L); cp = P[:, -nc - 2:-2]
        found = np.abs(cp - 1.0 / A) > 1e-12
        longest = np.where(found.any(1), nc - np.argmax(found[:, ::-1], 1), 0)
        return {0: np.zeros(len(P), int), 1: prev, 2: longest, 3: longest * A + prev}
    res = {"args": vars(a), "clustered_contexts": int(len(ctxs))}
    out = {}
    for with_codes in (0, 1):
        best = None
        for eps in (0.05, 0.2):
            Pv = experts(valid, eps); Pv = Pv if with_codes else Pv[:, :-2]
            sels = selectors(Pv if with_codes else np.column_stack([Pv, np.ones((len(Pv), 2))]), np.r_[train[-1], valid[:-1]])
            for eta in (0.03, 0.1, 0.3):
                for W in (10, 25, 50, 200):
                    for use_sel in (0, 1, 2, 3):
                        bpc = float(np.mean(-np.log2(S2.hedge(Pv, sels[use_sel], eta, W))))
                        if best is None or bpc < best[0]:
                            best = (bpc, eps, eta, W, use_sel)
        bpc_v, eps, eta, W, use_sel = best
        Pt = experts(test, eps); Pt = Pt if with_codes else Pt[:, :-2]
        selt = selectors(Pt if with_codes else np.column_stack([Pt, np.ones((len(Pt), 2))]), np.r_[valid[-1], test[:-1]])[use_sel]
        out["codes" if with_codes else "no_codes"] = {"valid_bpc": bpc_v, "test_bpc": float(np.mean(-np.log2(S2.hedge(Pt, selt, eta, W)))),
                                                      "chosen": {"eps": eps, "eta": eta, "W": W, "selector": use_sel}}
        if with_codes:
            out["codes"]["expert_test_bpc"] = {"code_k": float(np.mean(-np.log2(Pt[:, -2]))), "code_k_x_last": float(np.mean(-np.log2(Pt[:, -1])))}
    res.update(out); res["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)
    with open(os.path.join(OUT, f"codes_D{a.D}_K{a.K}_k{a.k}_m{a.m}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
