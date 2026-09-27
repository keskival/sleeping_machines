"""E62: character-level language modelling on text8, stage 1: the counting event model (THEORY §94-§95).

Context detectors of order 0..K (the characters just seen) with counts of the next character, blended by Witten-Bell
backoff (each order backs off to the one below in proportion to how many different continuations it has seen); a copy
memory (induction trace): the character that followed the previous occurrence of the longest recent context (lengths
32/16/8/4), mixed in with a weight per match length and copy-agreement, set from its track record on validation. Counts are
computed in batch (vectorized); natively they are the same counts accumulated event by event.
Measures bits per character on the test set (last 5M characters) and the number of stored contexts S(D) for training sizes
D, to test §95: S(D) ~ D^h and an excess loss falling like D^-(1-h).
"""
import argparse
import json
import os
import time

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "text8", "text8")
OUT = os.path.join(os.path.dirname(__file__), "results", "e62")
A = 27


def load():
    raw = np.frombuffer(open(DATA, "rb").read(), dtype=np.uint8)
    return np.where(raw == 32, 0, raw - 96).astype(np.int64)          # space 0, a..z 1..26


def ctx_codes(x, k):
    """code of the k characters before each position (positions < k get -1)."""
    code = np.zeros(len(x), np.int64)
    for j in range(1, k + 1):
        code[j:] += x[:-j] * (A ** (j - 1)) if j < len(x) else 0
    code[:k] = -1
    return code


class Order:
    def __init__(self, train, k):
        c = ctx_codes(train, k)
        m = c >= 0
        pair = c[m] * A + train[m]
        self.pairs, self.cnt = np.unique(pair, return_counts=True)
        ctx = self.pairs // A
        self.ctx, first = np.unique(ctx, return_index=True)
        self.n = np.add.reduceat(self.cnt, first); self.u = np.diff(np.r_[first, len(ctx)])

    def lookup(self, codes):
        """counts (len, A), totals n and distinct continuations u for context codes (-1 or unseen: zeros)."""
        L = len(codes); C = np.zeros((L, A)); n = np.zeros(L); u = np.zeros(L)
        ok = codes >= 0
        j = np.searchsorted(self.ctx, codes); jj = np.clip(j, 0, len(self.ctx) - 1)
        hit = ok & (self.ctx[jj] == codes)
        n[hit] = self.n[jj[hit]]; u[hit] = self.u[jj[hit]]
        for a in range(A):
            q = codes * A + a; i = np.searchsorted(self.pairs, q); ii = np.clip(i, 0, len(self.pairs) - 1)
            h = hit & (self.pairs[ii] == q)
            C[h, a] = self.cnt[ii[h]]
        return C, n, u


def count_probs(orders, stream, K, start, stop):
    """Witten-Bell blended next-character distributions for positions start..stop of `stream`."""
    seg = np.arange(start, stop)
    p = np.full((len(seg), A), 1.0 / A)
    for k in range(K + 1):
        codes = ctx_codes(stream[max(0, start - K):stop], k)[start - max(0, start - K):]
        C, n, u = orders[k].lookup(codes)
        seen = n > 0
        p = np.where(seen[:, None], (C + u[:, None] * p) / np.maximum(n + u, 1e-9)[:, None], p)
    return p


def copy_predictions(stream, start, stop, lens=(32, 16, 8, 4)):
    """for each position: the character that followed the most recent earlier occurrence of the longest recent context
    (searched in the stream before the position), and the length matched (0: none). One pass, a dict per length."""
    pred = np.full(stop - start, -1); mlen = np.zeros(stop - start, np.int64)
    s = stream.tobytes() if stream.dtype == np.uint8 else stream.astype(np.uint8).tobytes()
    last = {L: {} for L in lens}
    for i in range(max(lens), stop):
        if i >= start:
            for L in lens:
                j = last[L].get(s[i - L:i])
                if j is not None:
                    pred[i - start] = s[j]; mlen[i - start] = L; break
        for L in lens:
            last[L][s[i - L:i]] = i
    return pred, mlen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=10_000_000, help="training characters (from the start of text8)")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--test", type=int, default=1_000_000, help="test characters scored (from the last 5M)")
    ap.add_argument("--copy", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    x = load(); train = x[:a.D]; valid = x[90_000_000:90_000_000 + a.test]; test = x[95_000_000:95_000_000 + a.test]
    orders = [Order(train, k) for k in range(a.K + 1)]
    S = int(sum(len(o.ctx) for o in orders)); pairs = int(sum(len(o.pairs) for o in orders))
    res = {"args": vars(a), "stored_contexts": S, "stored_pairs": pairs}
    def score(stream, fit_w=None):
        out = {}
        CH = 250_000; logs = []; logs_mix = []; stats = {}
        # the copy memory reads the scored stream's own past (as a cache), plus the tail of training
        pre = train[-100_000:]; full = np.r_[pre, stream].astype(np.uint8)
        if a.copy:
            pred, mlen = copy_predictions(full, len(pre), len(full))
        for s0 in range(0, len(stream), CH):
            s1 = min(len(stream), s0 + CH)
            p = count_probs(orders, np.r_[train[-a.K:], stream], a.K, a.K + s0, a.K + s1)
            y = stream[s0:s1]
            pc = p[np.arange(len(y)), y]; logs.append(-np.log2(pc))
            if a.copy:
                pr, ml = pred[s0:s1], mlen[s0:s1]
                agree = (pr >= 0) & (pr == p.argmax(1))
                bucket = np.where(pr < 0, 0, np.searchsorted([4, 8, 16, 32], ml) + 1) * 2 + agree
                hitc = (pr == y)
                if fit_w is None:
                    for b in np.unique(bucket):
                        m = bucket == b; st = stats.setdefault(int(b), [0, 0, []])
                        st[0] += int(hitc[m].sum()); st[1] += int(m.sum())
                        st[2].append((pc[m], hitc[m]))
                else:
                    w = np.array([fit_w.get(int(b), 0.0) for b in bucket])
                    pm = (1 - w) * pc + w * hitc
                    logs_mix.append(-np.log2(np.maximum(pm, 1e-12)))
        out["bpc_counts"] = float(np.mean(np.concatenate(logs)))
        if a.copy and fit_w is None:
            W = {}
            for b, (h, n, parts) in stats.items():                     # choose the weight per bucket on validation
                pc_all = np.concatenate([q for q, _ in parts]); hit_all = np.concatenate([hh for _, hh in parts])
                grid = np.linspace(0, 0.95, 20)
                W[b] = float(grid[np.argmin([np.mean(-np.log2(np.maximum((1 - g) * pc_all + g * hit_all, 1e-12))) for g in grid])])
            out["weights"] = W
        if logs_mix:
            out["bpc_counts_copy"] = float(np.mean(np.concatenate(logs_mix)))
        return out
    v = score(valid)
    res["valid_bpc_counts"] = v["bpc_counts"]
    tst = score(test, fit_w=v.get("weights") if a.copy else None)
    res["test_bpc_counts"] = tst["bpc_counts"]
    if a.copy:
        res["test_bpc_counts_copy"] = tst["bpc_counts_copy"]; res["copy_weights"] = v["weights"]
    res["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps({k: v2 for k, v2 in res.items() if k != "copy_weights"}), flush=True)
    with open(os.path.join(OUT, f"counts_D{a.D}_K{a.K}_copy{a.copy}.json"), "w") as f:  # (K fixed across D: the h series)
        json.dump(res, f)


if __name__ == "__main__":
    main()
