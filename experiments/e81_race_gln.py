"""E81: a race Gated Linear Network over the native experts (THEORY §109): layers of race neurons, each a geometric
(product-of-experts) mixture p ∝ exp(W[c(t)] · log P) of its inputs' distributions over the next character, gated by its
own context c(t), and each trained on its OWN log loss by the exact local gradient (no backpropagation between neurons;
Veness et al. 2021). Layer 1 mixes the native experts of E79; layer 2 mixes layer 1's outputs; one final neuron.

Contexts (side information, all from the past): previous character; previous two characters; longest copy-match length x
previous character; position in the current word x previous character; confidence bucket of the Witten-Bell blend x copy
length; and a constant context. Weights learned online over the validation text, then (a) frozen for the test text and
(b) kept learning on it (prequential). Copy memory unbounded and restricted to 256 characters.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e62_charlm as S1  # noqa: E402
import e63_mixlm as S2  # noqa: E402
from e79_race_mixer import expert_logp, race_mix  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e81")
A = S1.A


def contexts(stream, prev0, LP, K):
    """(T, 6) context ids for the layer-1 neurons and (T, 3) for layer 2; all computed from the past."""
    x1 = np.r_[prev0[-1], stream[:-1]]; x2 = np.r_[prev0[-2:], stream[:-2]][:len(stream)]
    sp = np.r_[prev0[-1:] == 0, stream[:-1] == 0]
    last = np.maximum.accumulate(np.where(sp, np.arange(len(stream)), -1)); wpos = np.minimum(np.arange(len(stream)) - last, 7)
    blend = LP[:, K + 1]                                                      # Witten-Bell blend: its top probability
    conf = np.clip((np.exp(blend.max(1)) * 8).astype(int), 0, 7)
    copy_top = LP[:, K + 3].max(1) > np.log(0.5)                             # the copy expert has a match
    c1 = np.stack([x1, x1 * A + x2, copy_top * A + x1, wpos * A + x1, conf * 2 + copy_top, np.zeros_like(x1)], 1)
    c2 = np.stack([copy_top * 8 + conf, x1, np.zeros_like(x1)], 1)
    return c1.astype(np.int64), c2.astype(np.int64)


class RaceGLN:
    def __init__(self, E, sizes1=(A, A * A, 2 * A, 8 * A, 16, 1), sizes2=(16, A, 1), lr=(0.01, 0.01, 0.005)):
        self.W1 = [np.full((s, E), 1.0 / E, np.float32) for s in sizes1]
        m = len(sizes1)
        self.W2 = [np.full((s, m), 1.0 / m, np.float32) for s in sizes2]
        self.W3 = np.full(len(sizes2), 1.0 / len(sizes2), np.float32); self.lr = lr

    @staticmethod
    def neuron(W, lp):
        z = W @ lp; z = z - z.max(-1, keepdims=True); p = np.exp(z); p /= p.sum(-1, keepdims=True)
        return p

    def run(self, LP, y, c1, c2, update=True):
        T = len(y); loss = np.empty(T); l1, l2, l3 = self.lr
        m1, m2 = len(self.W1), len(self.W2)
        for t in range(T):
            lp = LP[t]                                                        # (E, A)
            w1 = np.stack([self.W1[i][c1[t, i]] for i in range(m1)])        # (m1, E)
            p1 = self.neuron(w1, lp); h1 = np.log(np.maximum(p1, 1e-9))      # (m1, A)
            w2 = np.stack([self.W2[i][c2[t, i]] for i in range(m2)])
            p2 = self.neuron(w2, h1); h2 = np.log(np.maximum(p2, 1e-9))
            p3 = self.neuron(self.W3[None], h2)[0]
            loss[t] = -np.log2(max(p3[y[t]], 1e-12))
            if update:                                                        # each neuron: its own log-loss gradient
                g1 = lp[:, y[t]][None] - p1 @ lp.T                            # (m1, E)
                for i in range(m1):
                    self.W1[i][c1[t, i]] += l1 * g1[i]
                g2 = h1[:, y[t]][None] - p2 @ h1.T
                for i in range(m2):
                    self.W2[i][c2[t, i]] += l2 * g2[i]
                self.W3 += l3 * (h2[:, y[t]] - p3 @ h2.T)
        return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=1_000_000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--windows", default="256,0")
    ap.add_argument("--wordkeys", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    x = S1.load(); train = x[:a.D]
    valid = x[90_000_000:90_000_000 + a.n]; test = x[95_000_000:95_000_000 + a.n]
    orders = [S1.Order(train, k) for k in range(a.K + 1)]; word = S2.WordOrder(train)
    tk = None
    if a.wordkeys:
        from e66_wordkeys import word_keys, KeyCounts
        tk1, tk2 = word_keys(train); tk = (tk1, tk2, KeyCounts(tk1, train), KeyCounts(tk2, train))
    res = {"args": vars(a)}
    for window in [int(w) for w in a.windows.split(",")]:
        r = {}
        LPv, sv = expert_logp(orders, word, train, valid, a.K, window, tk=tk)
        c1v, c2v = contexts(valid, train, LPv, a.K)
        best = None
        for lr in ((0.005, 0.005, 0.002), (0.01, 0.01, 0.005), (0.02, 0.02, 0.01)):
            g = RaceGLN(LPv.shape[1], lr=lr); lv = g.run(LPv, valid, c1v, c2v)
            if best is None or lv.mean() < best[0]:
                best = (float(lv.mean()), lr, g)
        r["valid_bpc"], r["lr"] = best[0], best[1]
        _, W1lr = race_mix(LPv, valid, sv, 0.002)                             # E79's one-neuron mixer, for reference
        del LPv
        LPt, st = expert_logp(orders, word, train, test, a.K, window, tk=tk)
        c1t, c2t = contexts(test, valid, LPt, a.K)
        import copy
        g = best[2]; gf = copy.deepcopy(g)
        r["gln_frozen_test_bpc"] = float(gf.run(LPt, test, c1t, c2t, update=False).mean())
        r["gln_online_test_bpc"] = float(g.run(LPt, test, c1t, c2t, update=True).mean())
        r["one_neuron_frozen_test_bpc"] = float(race_mix(LPt, test, st, 0.002, W1lr, update=False)[0].mean())
        res[f"copy_window_{window or 'unbounded'}"] = r
        print(json.dumps({f"window_{window}": r}), flush=True)
        del LPt
    with open(os.path.join(OUT, f"race_gln_D{a.D}_K{a.K}_wk{a.wordkeys}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
