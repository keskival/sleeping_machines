"""E68: training a race Transformer vs a softmax Transformer (THEORY §103).

Same architecture (embeddings d = 32, 2 layers, 4 heads, feed-forward 4d, learned positions); attention is either softmax
or time-normalized races (§102): per head and query, keys fire exponential clocks with rates exp(q·k/sqrt(dh)), T = the
first arrival, output = sum_j lambda_j T v_j, averaged over R independent races. Gradients by autograd through T = min(E /
lambda) are exactly the pathwise race gradients of §103 (the local rule). Tasks: associative recall with a learned query-key
map (E61) and character-level text (text8, first 1M characters; test on held-out text).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e68")


class Attn(nn.Module):
    def __init__(self, d, h, race_R):
        super().__init__()
        self.h, self.dh, self.R = h, d // h, race_R
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)

    def forward(self, x, causal):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)     # (B, h, L, dh)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)                                # (B, h, L, L)
        if causal:
            s = s.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf"))
        if self.R == 0:
            a = torch.softmax(s, -1)
        else:                                                                          # time-normalized races
            lam = torch.exp(s - s.max(-1, keepdim=True).values.detach())
            a = 0
            for _ in range(self.R):
                E = torch.empty_like(lam).exponential_()
                T = (E / lam).masked_fill(lam == 0, float("inf")).min(-1, keepdim=True).values   # decision time
                a = a + lam * T                                                            # each key: rate x time
            a = a / self.R
        return self.o((a @ v).transpose(1, 2).reshape(B, L, d))


class Block(nn.Module):
    def __init__(self, d, h, R):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.a = Attn(d, h, R); self.n2 = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, causal):
        x = x + self.a(self.n1(x), causal)
        return x + self.f(self.n2(x))


class Model(nn.Module):
    def __init__(self, V, n_out, d, L, h, R, Lmax):
        super().__init__()
        self.e = nn.Embedding(V, d); self.p = nn.Embedding(Lmax, d)
        self.b = nn.ModuleList([Block(d, h, R) for _ in range(L)]); self.n = nn.LayerNorm(d); self.out = nn.Linear(d, n_out)

    def forward(self, x, causal):
        z = self.e(x) + self.p.weight[:x.shape[1]]
        for blk in self.b:
            z = blk(z, causal)
        return self.out(self.n(z))


def recall_batch(K, n, perm, B, rng):
    X, Y = [], []
    for _ in range(B):
        keys = rng.choice(K, n, replace=False); vals = rng.integers(0, K, n); j = int(rng.integers(n))
        seq = np.empty(2 * n + 1, np.int64); seq[0:2 * n:2] = keys; seq[1:2 * n:2] = K + vals; seq[-1] = 2 * K + perm[keys[j]]
        X.append(seq); Y.append(vals[j])
    return torch.tensor(np.array(X)), torch.tensor(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="recall", choices=("recall", "text"))
    ap.add_argument("--R", type=int, default=0, help="0: softmax attention; R >= 1: race attention with R races")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    curve = []
    if a.task == "recall":
        K, n = 32, 8; perm = rng.permutation(K)
        net = Model(3 * K, K, 32, 2, 4, a.R, 2 * n + 1); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for step in range(1, a.steps + 1):
            X, Y = recall_batch(K, n, perm, 64, rng)
            loss = nn.functional.cross_entropy(net(X, False)[:, -1], Y); opt.zero_grad(); loss.backward(); opt.step()
            if step % (a.steps // 8) == 0:
                with torch.no_grad():
                    X, Y = recall_batch(K, n, perm, 1000, np.random.default_rng(99))
                    acc = float((net(X, False)[:, -1].argmax(1) == Y).float().mean())
                curve.append({"step": step, "sequences": step * 64, "test_acc": acc})
    else:
        import e62_charlm as S1
        x = S1.load(); train = torch.tensor(x[:1_000_000]); test = torch.tensor(x[95_000_000:95_000_000 + 100_000])
        Lc = 64; net = Model(27, 27, 32, 2, 4, a.R, Lc); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        for step in range(1, a.steps + 1):
            idx = rng.integers(0, len(train) - Lc - 1, 32)
            xb = torch.stack([train[i:i + Lc] for i in idx]); yb = torch.stack([train[i + 1:i + Lc + 1] for i in idx])
            loss = nn.functional.cross_entropy(net(xb, True).reshape(-1, 27), yb.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
            if step % (a.steps // 8) == 0:
                with torch.no_grad():
                    tot = 0.0; cnt = 0
                    for s0 in range(0, len(test) - Lc - 1, Lc):
                        xb = test[s0:s0 + Lc][None]; yb = test[s0 + 1:s0 + Lc + 1][None]
                        tot += float(nn.functional.cross_entropy(net(xb, True)[0], yb[0], reduction="sum")); cnt += Lc
                curve.append({"step": step, "test_bpc": tot / cnt / math.log(2)})
    res = {"args": vars(a), "curve": curve, "wall_s": round(time.time() - t0, 1)}
    print(json.dumps({"task": a.task, "R": a.R, "seed": a.seed, **curve[-1], "wall_s": res["wall_s"]}), flush=True)
    with open(os.path.join(OUT, f"{a.task}_R{a.R}_s{a.seed}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
