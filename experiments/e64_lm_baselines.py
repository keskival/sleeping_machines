"""E64: gradient-trained character LM baselines on text8 at the same training sizes as E62/E63.

LSTM (1 layer) and a causal Transformer (2-4 layers, learned positions, context 256), trained by Adam on the first D
characters for a fixed number of passes, scored on the same 1M test characters (bits per character).
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
import e62_charlm as S1  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e64")
A = S1.A


class LSTMLM(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.emb = nn.Embedding(A, 64); self.rnn = nn.LSTM(64, h, batch_first=True); self.out = nn.Linear(h, A)

    def forward(self, x, state=None):
        h, state = self.rnn(self.emb(x), state)
        return self.out(h), state


class TfLM(nn.Module):
    def __init__(self, d, L, ctx):
        super().__init__()
        self.emb = nn.Embedding(A, d); self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False); self.out = nn.Linear(d, A)
        self.register_buffer("mask", torch.triu(torch.full((ctx, ctx), float("-inf")), 1))

    def forward(self, x, state=None):
        n = x.shape[1]
        return self.out(self.enc(self.emb(x) + self.pos.weight[:n], mask=self.mask[:n, :n], is_causal=True)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=("lstm", "tf"))
    ap.add_argument("--D", type=int, default=1_000_000)
    ap.add_argument("--passes", type=float, default=3.0)
    ap.add_argument("--size", type=int, default=256, help="LSTM hidden size or Transformer width")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--test", type=int, default=1_000_000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(0); rng = np.random.default_rng(0)
    x = S1.load(); train = torch.tensor(x[:a.D]); test = torch.tensor(x[95_000_000:95_000_000 + a.test])
    net = LSTMLM(a.size) if a.model == "lstm" else TfLM(a.size, a.layers, a.ctx)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3 if a.model == "lstm" else 1e-3)
    B, T = 32, a.ctx
    steps = int(a.passes * a.D / (B * T))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1))
    for step in range(steps):
        idx = rng.integers(0, a.D - T - 1, B)
        xb = torch.stack([train[i:i + T] for i in idx]); yb = torch.stack([train[i + 1:i + T + 1] for i in idx])
        logits, _ = net(xb)
        loss = nn.functional.cross_entropy(logits.reshape(-1, A), yb.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % max(steps // 10, 1) == 0:
            print(json.dumps({"step": step, "of": steps, "train_bpc": float(loss) / math.log(2), "wall_s": round(time.time() - t0)}), flush=True)
    net.eval(); tot = 0.0; n = 0
    with torch.no_grad():                                               # score in windows of T with T/2 of context
        state = None
        if a.model == "lstm":
            for s0 in range(0, len(test) - 1, 4096):
                xb = test[s0:s0 + 4096][None]; yb = test[s0 + 1:s0 + 4097][None]
                xb = xb[:, :yb.shape[1]]
                logits, state = net(xb, state)
                tot += float(nn.functional.cross_entropy(logits.reshape(-1, A), yb.reshape(-1), reduction="sum")); n += yb.numel()
        else:
            half = T // 2
            for s0 in range(0, len(test) - T - 1, half):
                xb = test[s0:s0 + T][None]; yb = test[s0 + 1:s0 + T + 1][None]
                logits, _ = net(xb)
                lo = 0 if s0 == 0 else half
                tot += float(nn.functional.cross_entropy(logits[0, lo:], yb[0, lo:], reduction="sum")); n += T - lo
    nparam = sum(p.numel() for p in net.parameters())
    res = {"args": vars(a), "params": nparam, "test_bpc": tot / n / math.log(2), "steps": steps, "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(res), flush=True)
    with open(os.path.join(OUT, f"{a.model}_D{a.D}_s{a.size}_p{a.passes:g}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
