"""E65b: a learned discrete recurrent state (probabilistic automaton) as the event LM's representation (THEORY §99d).

State nodes s = 0..m-1; a character event moves the armed state node to s' = T[s, x] (one table lookup per event: the
stateful arm/disarm nodes of §75). Learning by predictive merging (approximating causal states):
  1. run the automaton over the training text (states s_t);
  2. for every (state, character) pair, count the next character that followed; merge the pairs into m new states by
     likelihood k-means (the race of §99c), which defines the new transition table T[s, x] = cluster of (s, x);
  repeat. Initialization: the state is the code of the last k characters (hashed into m).
Experts added to the stage-2 mixture: next character given the state, and given (state, last character).
Unseen contexts are handled: the state is computed from content, not looked up.
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
import e65_codes as C  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e65")
A = S1.A


def run(T, x, s0=0):
    """states s_t (the state *after* reading x_t) for a character stream."""
    s = np.empty(len(x), np.int64); cur = s0; Tl = T.tolist(); xl = x.tolist()
    for i, c in enumerate(xl):
        cur = Tl[cur][c]; s[i] = cur
    return s


def learn(train, m, iters, rng, horizon=2):
    """init: the exact order-2 automaton (state = last two characters, (a, b) + c -> (b, c)); then iterate: merge
    (state, character) pairs by the distribution of the next `horizon` characters (finite-horizon causal states)."""
    base = A * A
    T = np.zeros((m, A), np.int64)
    ab = np.arange(base); b = ab % A
    for c in range(A):
        T[:base, c] = b * A + c                                           # consistent, no collisions (needs m >= 729)
    for it in range(iters):
        st = run(T, train); prev = np.r_[0, st[:-1]]
        pair = prev * A + train
        fut = np.r_[train[1:], 0, 0][:len(train)] * (A if horizon == 2 else 1)
        if horizon == 2:
            fut = fut + np.r_[train[2:], 0, 0][:len(train)]
        nf = A ** horizon
        N = np.zeros((m * A, nf), np.float32); np.add.at(N, (pair[:-horizon], fut[:-horizon]), 1.0)
        used = N.sum(1) > 0; Nu = N[used]; idx = np.flatnonzero(used)
        k = min(m, len(Nu))
        init = rng.choice(len(Nu), k, replace=False, p=Nu.sum(1) / Nu.sum())
        Q = Nu[init] + 0.5 / nf; Q /= Q.sum(1, keepdims=True)
        for _ in range(6):                                                # merge by the likelihood race over futures
            asg = np.argmax(Nu @ np.log(Q).T.astype(np.float32), 1)
            Q = np.full((k, nf), 0.5 / nf); np.add.at(Q, asg, Nu); Q /= Q.sum(1, keepdims=True)
        newT = np.zeros(m * A, np.int64); newT[idx] = asg
        for c in range(A):
            col = np.arange(m) * A + c; seen = used[col]
            if seen.any():
                newT[col[~seen]] = np.bincount(newT[col[seen]], minlength=k).argmax()
        T = newT.reshape(m, A)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=1_000_000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--test", type=int, default=1_000_000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); rng = np.random.default_rng(0)
    x = S1.load(); train = x[:a.D]
    valid = x[90_000_000:90_000_000 + a.test]; test = x[95_000_000:95_000_000 + a.test]
    orders = [S1.Order(train, k) for k in range(a.K + 1)]; word = S2.WordOrder(train)
    T = learn(train, a.m, a.iters, rng, a.horizon)
    st_train = run(T, train); prev_train = np.r_[0, st_train[:-1]]
    E1 = C.code_expert_counts(train, prev_train, a.m)                                     # state -> next char
    E2 = C.code_expert_counts(train, prev_train * A + np.r_[0, train[:-1]], a.m * A)      # (state, last char) -> next
    def experts(stream, eps):
        base = S2.experts(orders, word, train, stream, a.K, eps)
        full = np.r_[train[-100_000:], stream]; st = run(T, full); prev = np.r_[0, st[:-1]][100_000:]
        last = np.r_[0, full[:-1]][100_000:]
        return np.column_stack([base, E1[prev, stream], E2[prev * A + last, stream]])
    res = {"args": vars(a)}
    for with_state in (0, 1):
        best = None
        for eps in (0.05,):
            Pv = experts(valid, eps); Pv = Pv if with_state else Pv[:, :-2]
            pad = Pv if with_state else np.column_stack([Pv, np.ones((len(Pv), 2))])
            nc = len(S2.COPY_L); cp = pad[:, -nc - 2:-2]; found = np.abs(cp - 1.0 / A) > 1e-12
            longest = np.where(found.any(1), nc - np.argmax(found[:, ::-1], 1), 0); prev_ch = np.r_[train[-1], valid[:-1]]
            for eta in (0.03, 0.1, 0.3):
                for W in (10, 25, 50):
                    bpc = float(np.mean(-np.log2(S2.hedge(Pv, longest * A + prev_ch, eta, W))))
                    if best is None or bpc < best[0]:
                        best = (bpc, eps, eta, W)
        bpc_v, eps, eta, W = best
        Pt = experts(test, eps); Pt = Pt if with_state else Pt[:, :-2]
        pad = Pt if with_state else np.column_stack([Pt, np.ones((len(Pt), 2))])
        nc = len(S2.COPY_L); cp = pad[:, -nc - 2:-2]; found = np.abs(cp - 1.0 / A) > 1e-12
        longest = np.where(found.any(1), nc - np.argmax(found[:, ::-1], 1), 0); prev_ch = np.r_[valid[-1], test[:-1]]
        key = "state" if with_state else "no_state"
        res[key] = {"valid_bpc": bpc_v, "test_bpc": float(np.mean(-np.log2(S2.hedge(Pt, longest * A + prev_ch, eta, W)))),
                    "chosen": {"eta": eta, "W": W}}
        if with_state:
            res[key]["expert_test_bpc"] = {"state": float(np.mean(-np.log2(Pt[:, -2]))), "state_x_last": float(np.mean(-np.log2(Pt[:, -1])))}
    res["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)
    with open(os.path.join(OUT, f"automaton_D{a.D}_m{a.m}_h{a.horizon}_it{a.iters}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
