"""E60b: DVS128 Gesture with depth over motion events (THEORY §84 applied to vision) and a discriminative native learner.

Motion events as in E60 (refractory cells -> onsets -> Reichardt pairs with opponent veto), absolute regions R x R.
Tokens: unigrams (region, direction) and composite "motion bigrams" (region, previous direction, current direction): a
motion event whose region had another motion event within T ms before; these separate rotation senses and wave phases.
Classifiers on per-gesture feature counts (log(1 + count) as the event feature strength):
  (a) naive Bayes on unigram + bigram counts (counting only);
  (b) class nodes with summed potentials over the features, trained by mistake-driven conserved multiplicative updates
      (Winnow/Hedge-type, §83): on a mistake the true class's weights on active features are multiplied by exp(eta * x),
      the wrong winner's by exp(-eta * x), each class's weights renormalized to a fixed budget. Several passes.
Protocol: select on users 20-23 (held out from 1-19); test (users 24-29) once via --eval test.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import dvs_data as D  # noqa: E402
import e60_dvs as E60  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e60")


def features(x, y, t, G, R, T_ms, reg):
    tok, tim, _ = E60.motion_tokens(x, y, t, G, 10.0, 2.0, 40.0, 0, nreg=reg, S=50.0)
    region = tok // 16; direction = (tok // 2) % 8                    # E60 token = ((region)*8 + dir)*2 + speed
    uni = region * 8 + direction
    nU = reg * reg * 8
    # bigrams: previous motion event in the same region within T ms
    order = np.lexsort((tim, region)); r_s, t_s, d_s = region[order], tim[order], direction[order]
    same = np.r_[False, (r_s[1:] == r_s[:-1]) & ((t_s[1:] - t_s[:-1]) * 1000 <= T_ms)]
    big = (r_s * 64 + np.r_[0, d_s[:-1]] * 8 + d_s)[same]
    cu = np.bincount(uni, minlength=nU); cb = np.bincount(big, minlength=reg * reg * 64)
    return np.r_[cu, cb].astype(np.float64), len(tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=32)
    ap.add_argument("--reg", type=int, default=8)
    ap.add_argument("--T", type=float, default=100.0, help="bigram window (ms)")
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--eval", default="val", choices=("val", "test"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    def enc(it):
        X, Y, U, ev = [], [], [], []
        for lab, x, y, p, t, u in it:
            f, n = features(x, y, t, a.G, 10.0, a.T, a.reg); X.append(f); Y.append(lab); U.append(u); ev.append(n)
        return np.array(X), np.array(Y), np.array(U), np.array(ev)
    Xtr, Ytr, Utr, Etr = enc(D.iter_split("train"))
    if a.eval == "val":
        m = Utr <= 19; Xf, Yf, Xe, Ye = Xtr[m], Ytr[m], Xtr[~m], Ytr[~m]
    else:
        Xf, Yf = Xtr, Ytr; Xe, Ye, _, _ = enc(D.iter_split("test"))
    K = 11; nU = a.reg * a.reg * 8
    res = {"args": vars(a), "n_eval": len(Ye), "motion_events_per_gesture": float(Etr.mean())}
    for name, cols in (("unigram", slice(0, nU)), ("unigram+bigram", slice(None))):
        C = np.full((K, Xf[:, cols].shape[1]), 0.5)
        for k in range(K):
            C[k] += Xf[Yf == k][:, cols].sum(0)
        lp = np.log(C / C.sum(1, keepdims=True))
        res[f"nb_{name}"] = float(np.mean(np.argmax(Xe[:, cols] @ lp.T, 1) == Ye))
        # discriminative native learner: conserved multiplicative updates on log(1 + count) features
        F = np.log1p(Xf[:, cols]); Fe = np.log1p(Xe[:, cols])
        F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9; Fe /= np.linalg.norm(Fe, axis=1, keepdims=True) + 1e-9
        Wt = np.full((K, F.shape[1]), 1.0 / F.shape[1]); rng = np.random.default_rng(0); mistakes = 0
        for _ in range(a.passes):
            for i in rng.permutation(len(F)):
                s = Wt @ F[i]; c = int(np.argmax(s))
                if c != Yf[i]:
                    mistakes += 1
                    Wt[Yf[i]] *= np.exp(a.eta * F[i] * F.shape[1] ** 0.5); Wt[c] *= np.exp(-a.eta * F[i] * F.shape[1] ** 0.5)
                    Wt[Yf[i]] /= Wt[Yf[i]].sum(); Wt[c] /= Wt[c].sum()
        res[f"winnow_{name}"] = float(np.mean(np.argmax(Fe @ Wt.T, 1) == Ye)); res[f"winnow_{name}_mistakes"] = mistakes
    res["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)
    with open(os.path.join(OUT, f"depth_G{a.G}_reg{a.reg}_T{a.T:g}_eta{a.eta:g}_{a.eval}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
