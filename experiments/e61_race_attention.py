"""E61: is attention trainable by local credit? Associative recall with a learned query-key map (THEORY §96, §97).

Task: n key-value pairs (keys from a vocabulary of K, values from a vocabulary of K, all keys distinct), then a query token
from a third vocabulary of K, q = pi(k_j) for a fixed unknown permutation pi; the answer is v_j. Neither the match (query ->
key) nor the read (which neighbour of the key holds the answer) is given.

Event network (race attention): every context token leaves a trace on its identity channel (with its position). A route
is (item a, read offset o), a in the 2K context identities, o in {-2, -1, +1, +2}; query channel c has conserved weights
Z[c, a, o]. At the query event the routes whose item is present race; the winner (largest weight: first to fire) reads the
token at the item's position + o. Credit (§83, §97), on a mistake only: promote, by (1 + alpha), every route consistent
with the answer (items at the answer's positions - o, for each o); demote, by (1 - beta), the route that fired; renormalize.
Transformer: token + learned position embeddings, 2 layers, trained by Adam on the same sequences (cross-entropy at the
query position). Both: accuracy vs sequences seen (one pass for the event network), and length extrapolation (train with
n pairs, test with 4n).
"""
import argparse
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e61")
OFFS = np.array([-2, -1, 1, 2])


def make_perm(K, rng):
    return rng.permutation(K)


def sample(K, n, perm, rng):
    keys = rng.choice(K, n, replace=False); vals = rng.integers(0, K, n)
    j = int(rng.integers(n))
    seq = np.empty(2 * n, np.int64); seq[0::2] = keys; seq[1::2] = K + vals          # keys 0..K-1, values K..2K-1
    return seq, int(perm[keys[j]]), int(vals[j])                                       # query in its own vocabulary


class RaceAttention:
    def __init__(self, K, alpha, beta):
        self.K, self.alpha, self.beta = K, alpha, beta
        self.Z = np.full((K, 2 * K, len(OFFS)), 1.0 / (2 * K * len(OFFS)))
        self.mistakes = 0; self.events = 0

    def forward(self, seq, q):
        pos = np.arange(len(seq))
        a = seq; w = self.Z[q][a]                                                      # (L, 4) weights of present routes
        rd = pos[:, None] + OFFS[None, :]
        ok = (rd >= 0) & (rd < len(seq))
        w = np.where(ok, w, -1.0)
        i, o = np.unravel_index(int(np.argmax(w)), w.shape)                            # first to fire
        self.events += int(ok.sum())
        return int(seq[rd[i, o]]) - self.K, (int(a[i]), o)

    def teach(self, seq, q, v):
        out, (ai, oi) = self.forward(seq, q)
        if out == v:
            return True
        self.mistakes += 1
        target = np.flatnonzero(seq == self.K + v)
        for p in target:                                                               # consistent routes
            for oi2, off in enumerate(OFFS):
                src = p - off
                if 0 <= src < len(seq):
                    self.Z[q, seq[src], oi2] *= 1 + self.alpha
        self.Z[q, ai, oi] *= 1 - self.beta
        self.Z[q] /= self.Z[q].sum()
        return False


def run_event(a):
    rows = []
    for s in range(a.seeds):
        rng = np.random.default_rng(s); perm = make_perm(a.K, rng)
        net = RaceAttention(a.K, a.alpha, a.beta); curve = []; seen = 0
        for target in a.checkpoints:
            while seen < target:
                seq, q, v = sample(a.K, a.n, perm, rng); net.teach(seq, q, v); seen += 1
            ev = np.random.default_rng(1000 + s)
            acc = {}
            for n_test in (a.n, 4 * a.n):
                if 4 * a.n > a.K and n_test > a.K:
                    continue
                ok = 0
                for _ in range(1000):
                    seq, q, v = sample(a.K, n_test, perm, ev); ok += net.forward(seq, q)[0] == v
                acc[f"n{n_test}"] = ok / 1000
            curve.append({"seen": seen, "mistakes": net.mistakes, **acc})
        rows.append({"seed": s, "curve": curve}); print(json.dumps({"model": "event", "seed": s, **curve[-1]}), flush=True)
    return rows


def run_tf(a):
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    rows = []
    for s in range(a.seeds):
        rng = np.random.default_rng(s); perm = make_perm(a.K, rng); torch.manual_seed(s)
        V = 3 * a.K; Lmax = 2 * 4 * a.n + 1
        class TF(nn.Module):
            def __init__(self, d=64, L=2):
                super().__init__()
                self.tok = nn.Embedding(V, d); self.pos = nn.Embedding(Lmax, d)
                layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
                self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False); self.out = nn.Linear(d, a.K)
            def forward(self, x):
                h = self.enc(self.tok(x) + self.pos.weight[:x.shape[1]])
                return self.out(h[:, -1])
        net = TF(a.d, a.layers); opt = torch.optim.Adam(net.parameters(), lr=a.lr)
        def batch(n, B, r):
            X, Y = [], []
            for _ in range(B):
                seq, q, v = sample(a.K, n, perm, r); X.append(np.r_[seq, 2 * a.K + q]); Y.append(v)
            return torch.tensor(np.array(X)), torch.tensor(Y)
        curve = []; seen = 0; B = 32
        for target in a.checkpoints:
            while seen < target:
                X, Y = batch(a.n, B, rng)
                loss = nn.functional.cross_entropy(net(X), Y); opt.zero_grad(); loss.backward(); opt.step(); seen += B
            ev = np.random.default_rng(1000 + s); acc = {}
            with torch.no_grad():
                for n_test in (a.n, 4 * a.n):
                    if n_test > a.K:
                        continue
                    X, Y = batch(n_test, 1000, ev); acc[f"n{n_test}"] = float((net(X).argmax(1) == Y).float().mean())
            curve.append({"seen": seen, **acc})
        rows.append({"seed": s, "curve": curve}); print(json.dumps({"model": "transformer", "seed": s, **curve[-1]}), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="event", choices=("event", "tf"))
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--checkpoints", default="250,500,1000,2000,5000,10000,20000")
    a = ap.parse_args()
    a.checkpoints = [int(x) for x in a.checkpoints.split(",")]
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    rows = run_event(a) if a.model == "event" else run_tf(a)
    tag = "" if a.model == "event" else f"_d{a.d}_L{a.layers}_lr{a.lr:g}"
    with open(os.path.join(OUT, f"{a.model}_K{a.K}_n{a.n}{tag}.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "wall_s": round(time.time() - t0, 1)}, f)


if __name__ == "__main__":
    main()
