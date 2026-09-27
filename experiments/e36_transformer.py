"""E36: Transformer baseline on event tokens, for E27's task (and E28's with --task e28).

One token per input spike: learned channel embedding + sinusoidal encoding of the spike's continuous time (so the
model, like the event learner, pays per spike and nothing for silence). L pre-norm Transformer encoder layers
(d_model d, 2 heads, FFN 4d), mean-pool over tokens, linear readout to K + 1 classes. Adam, same number of training
episodes as the event learners. Cost per episode = forward multiply-adds for the episode's n tokens:
  per layer  n·4d² (Q, K, V, out) + 2·n²·d (scores and mixing) + n·8d² (FFN)   plus readout d·(K+1).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

import e27_veto as E27
import e28_routing as E28

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e36")


class RelTimeLayer(nn.Module):
    """pre-norm attention layer whose scores get a learned bias b_h(t_j - t_i) per head (a small MLP on the lag)."""
    def __init__(self, d, heads=2):
        super().__init__()
        self.h, self.d = heads, d
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.bias = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, heads))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def forward(self, x, times, mask):
        B, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        s = q @ k.transpose(-1, -2) / math.sqrt(d // self.h)
        lag = (times[:, None, :] - times[:, :, None])[..., None]
        s = s + self.bias(lag).permute(0, 3, 1, 2)
        s = s.masked_fill(mask[:, None, None, :], -1e9)
        x = x + self.o((s.softmax(-1) @ v).transpose(1, 2).reshape(B, n, d))
        return x + self.ff(self.n2(x))


class EventTransformer(nn.Module):
    def __init__(self, N, C, d, L, H, reltime=0):
        super().__init__()
        self.reltime = reltime
        if reltime:
            self.layers = nn.ModuleList([RelTimeLayer(d) for _ in range(L)])
        self.ch = nn.Embedding(N, d)
        self.freq = torch.exp(torch.linspace(0, math.log(H * 4), d // 2)) / (H * 4) * 2 * math.pi * 8
        layer = nn.TransformerEncoderLayer(d, 2, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L)
        self.out = nn.Linear(d, C)
        self.d, self.L = d, L

    def forward(self, chans, times, mask):              # mask: True = padding
        ang = times[..., None] * self.freq
        x = self.ch(chans) + torch.cat([ang.sin(), ang.cos()], -1)
        if self.reltime:
            for layer in self.layers:
                x = layer(x, times, mask)
        else:
            x = self.enc(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask[..., None], 0).sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.out(x)

    def macs(self, n, C):
        d = self.d
        rel = n * n * (16 + 16 * 2) if self.reltime else 0          # the lag-bias MLP, evaluated per token pair
        return self.L * (n * 4 * d * d + 2 * n * n * d + n * 8 * d * d + rel) + d * C


def batchify(eps):
    n = max(1, max(int(np.isfinite(t).sum()) for t, _ in eps))
    B = len(eps)
    ch = torch.zeros(B, n, dtype=torch.long); tm = torch.zeros(B, n); mask = torch.ones(B, n, dtype=torch.bool)
    for b, (t, _) in enumerate(eps):
        sp = np.flatnonzero(np.isfinite(t)); sp = sp[np.argsort(t[sp])]
        ch[b, :len(sp)] = torch.tensor(sp); tm[b, :len(sp)] = torch.tensor(t[sp], dtype=torch.float32)
        mask[b, :len(sp)] = False
    return ch, tm, mask, torch.tensor([y for _, y in eps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="e27", choices=("e27", "e28"))
    ap.add_argument("--sizes", default="8x1,16x1,32x1,32x2", help="d x layers")
    ap.add_argument("--episodes", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reltime", type=int, default=0, help="1: learned relative-time attention bias (§70)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        if a.task == "e27":
            N, K, H, q = 12, 4, 10.0, 0.4
            pats = E27.make_task(N, K, H, rng)
            draw = lambda r: E27.sample(pats, N, H, q, r)                      # noqa: E731
        else:
            N, K, H, q = 16, 15, 12.0, 0.25
            motifs, classes = E28.make_task(N, 6, K, rng)
            draw = lambda r: E28.sample(motifs, classes, N, H, q, r)           # noqa: E731
        ev = np.random.default_rng(99)
        test = [draw(ev) for _ in range(2000)]
        for size in a.sizes.split(","):
            d, L = map(int, size.split("x"))
            torch.manual_seed(seed)
            net = EventTransformer(N, K + 1, d, L, H, a.reltime)
            opt = torch.optim.Adam(net.parameters(), lr=a.lr)
            t0 = time.time()
            for _ in range(a.episodes // a.batch):
                ch, tm, mask, y = batchify([draw(rng) for _ in range(a.batch)])
                loss = nn.functional.cross_entropy(net(ch, tm, mask), y)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                ch, tm, mask, y = batchify(test)
                acc = float((net(ch, tm, mask).argmax(1) == y).float().mean())
            ntok = np.mean([np.isfinite(t).sum() for t, _ in test])
            r = {"task": a.task, "reltime": a.reltime, "seed": seed, "d": d, "layers": L, "acc": acc,
                 "params": sum(p.numel() for p in net.parameters()),
                 "tokens_per_episode": float(ntok), "macs_per_episode": float(net.macs(ntok, K + 1)),
                 "wall_s": round(time.time() - t0, 1)}
            rows.append(r)
            print(json.dumps(r), flush=True)
    with open(os.path.join(OUT, f"transformer_{a.task}{'_' + a.tag if a.tag else ''}.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows}, f, indent=1)
    print("EXIT-OK")


if __name__ == "__main__":
    main()
