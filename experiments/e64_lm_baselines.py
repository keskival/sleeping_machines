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
    def __init__(self, h, drop=0.0):
        super().__init__()
        self.emb = nn.Embedding(A, 64); self.rnn = nn.LSTM(64, h, batch_first=True); self.out = nn.Linear(h, A)
        self.drop = nn.Dropout(drop)

    def forward(self, x, state=None):
        h, state = self.rnn(self.drop(self.emb(x)), state)
        return self.out(self.drop(h)), state


class TfLM(nn.Module):
    def __init__(self, d, L, ctx, drop=0.0):
        super().__init__()
        self.emb = nn.Embedding(A, d); self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False); self.out = nn.Linear(d, A)
        self.register_buffer("mask", torch.triu(torch.full((ctx, ctx), float("-inf")), 1))

    def forward(self, x, state=None):
        n = x.shape[1]
        return self.out(self.enc(self.emb(x) + self.pos.weight[:n], mask=self.mask[:n, :n], is_causal=True)), None


def score(net, test, a, T):
    """bits per character on `test`: LSTM statefully in blocks of 4096; Transformer in windows of T with T/2 of context."""
    was = net.training
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
    net.train(was)
    return tot / n / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=("lstm", "tf"))
    ap.add_argument("--D", type=int, default=1_000_000)
    ap.add_argument("--passes", type=float, default=3.0)
    ap.add_argument("--size", type=int, default=256, help="LSTM hidden size or Transformer width")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--test", type=int, default=1_000_000)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--valid", type=int, default=0, help="if > 0: score this many validation characters at 10 checkpoints and "
                    "test the best checkpoint (early stopping on validation)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(0); rng = np.random.default_rng(0)
    x = S1.load(); train = torch.tensor(x[:a.D]); test = torch.tensor(x[95_000_000:95_000_000 + a.test])
    net = LSTMLM(a.size, a.dropout) if a.model == "lstm" else TfLM(a.size, a.layers, a.ctx, a.dropout)
    valid = torch.tensor(x[90_000_000:90_000_000 + a.valid]) if a.valid else None
    best = (float("inf"), None, -1); vcurve = []
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
        if step % max(steps // 10, 1) == 0 or step == steps - 1:
            row = {"step": step, "of": steps, "train_bpc": float(loss) / math.log(2), "wall_s": round(time.time() - t0)}
            if valid is not None and step > 0:
                row["valid_bpc"] = score(net, valid, a, T); vcurve.append(row)
                if row["valid_bpc"] < best[0]:
                    best = (row["valid_bpc"], {k: v.clone() for k, v in net.state_dict().items()}, step)
            print(json.dumps(row), flush=True)
    if best[1] is not None:
        net.load_state_dict(best[1])
    tbpc = score(net, test, a, T)
    nparam = sum(p.numel() for p in net.parameters())
    res = {"args": vars(a), "params": nparam, "test_bpc": tbpc, "steps": steps,
           "valid_curve": vcurve, "best_step": best[2], "best_valid_bpc": best[0] if best[1] is not None else None, "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(res), flush=True)
    name = f"{a.model}_D{a.D}_s{a.size}_p{a.passes:g}" + (f"_dr{a.dropout:g}_v" if a.valid else "")
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(res, f)
    torch.save({"args": vars(a), "state": net.state_dict()}, os.path.join(OUT, name + ".pt"))      # for E76


if __name__ == "__main__":
    main()
