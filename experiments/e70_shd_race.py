"""E70: spoken digits (SHD) with a race Transformer over band-onset events (THEORY §102-§103).

Tokens: B frequency bands (700 channels pooled); an onset event = a spike in a band after >= S ms of silence in that band;
payload: band (embedding), time since utterance onset (sinusoidal features), burst strength (log spikes in the band over the
next 10 ms). ~320 events per utterance. Encoder: blocks of time-normalized race attention (R races; R = 0: softmax control)
and feed-forward cores, width d; mean over events -> 20 classes. Optional band-shift augmentation (random shift of all
bands by up to +-k: the voice-relative code of §92, learned). Selection on held-out speakers 3 and 6; test once.
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
import e51_shd_world as S  # noqa: E402
from e68_race_transformer import Block  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e70")


def tokens(t, u, B, S_ms, maxlen):
    o = np.lexsort((t, u)); uu, tt = u[o], t[o]
    prev = np.r_[-1, np.where(uu[1:] == uu[:-1], tt[:-1], -1)]
    on = (prev < 0) | (tt - prev >= S_ms / 1000)
    ob, ot = uu[on], tt[on]
    # burst strength: spikes of the same band within 10 ms after the onset
    burst = np.zeros(len(ob))
    for b in np.unique(ob):
        bt = tt[uu == b]; oi = np.flatnonzero(ob == b)
        burst[oi] = np.searchsorted(bt, ot[oi] + 0.01) - np.searchsorted(bt, ot[oi])
    k = np.argsort(ot, kind="stable")[:maxlen]
    t0 = ot.min() if len(ot) else 0.0
    return ob[k], (ot[k] - t0), np.log1p(burst[k])


class Net(nn.Module):
    def __init__(self, B, d, L, R, heads=4):
        super().__init__()
        self.band = nn.Embedding(B, d); self.feat = nn.Linear(9, d)
        self.blocks = nn.ModuleList([Block(d, heads, R) for _ in range(L)]); self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, 20)

    def forward(self, band, feats, mask):
        z = self.band(band) + self.feat(feats)
        km = mask > 0
        for b in self.blocks:
            z = b(z, False, km)
        z = self.norm(z) * mask[..., None]
        return self.out(z.sum(1) / mask.sum(1, keepdim=True).clamp(min=1))


def featurize(tt, bu):
    fr = torch.tensor([1.0, 2.0, 4.0, 8.0]) * math.pi
    tt = torch.tensor(tt, dtype=torch.float32)[:, None]
    return torch.cat([torch.sin(tt * fr), torch.cos(tt * fr), torch.tensor(bu, dtype=torch.float32)[:, None]], 1)


def batchify(items, B, shift_max, rng):
    L = max(len(b) for b, _, _, _ in items)
    band = torch.zeros(len(items), L, dtype=torch.long); feats = torch.zeros(len(items), L, 9); mask = torch.zeros(len(items), L)
    y = torch.tensor([lab for *_, lab in items])
    for i, (b, tt, bu, _) in enumerate(items):
        sh = int(rng.integers(-shift_max, shift_max + 1)) if shift_max else 0
        band[i, :len(b)] = torch.tensor(np.clip(b + sh, 0, B - 1)); feats[i, :len(b)] = featurize(tt, bu); mask[i, :len(b)] = 1
    return band, feats, mask, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--B", type=int, default=35)
    ap.add_argument("--S", type=float, default=10.0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--shift", type=int, default=2)
    ap.add_argument("--eval", default="spk", choices=("spk", "test"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    def load(split, part):
        return [(*tokens(t, u, a.B, a.S, 512), y) for t, u, y in S.utterances(split, a.B, part) if len(t) > 1]
    tr = load("train", "fit_spk" if a.eval == "spk" else None)
    ev = load("train", "val_spk") if a.eval == "spk" else load("test", None)
    net = Net(a.B, a.d, a.layers, a.R); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / 32); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    curve = []
    for ep in range(a.epochs):
        net.train()
        for i0 in range(0, len(tr), 32):
            batch = [tr[j] for j in rng.permutation(len(tr))[:32]]
            band, feats, mask, y = batchify(batch, a.B, a.shift, rng)
            loss = nn.functional.cross_entropy(net(band, feats, mask), y); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval(); ok = 0
        with torch.no_grad():
            for i0 in range(0, len(ev), 64):
                band, feats, mask, y = batchify(ev[i0:i0 + 64], a.B, 0, rng); ok += int((net(band, feats, mask).argmax(1) == y).sum())
        curve.append({"epoch": ep + 1, f"{a.eval}_acc": ok / len(ev), "wall_s": round(time.time() - t0)})
        print(json.dumps(curve[-1]), flush=True)
    with open(os.path.join(OUT, f"shd_race_R{a.R}_d{a.d}_L{a.layers}_sh{a.shift}_{a.eval}_s{a.seed}.json"), "w") as f:
        json.dump({"args": vars(a), "curve": curve}, f)


if __name__ == "__main__":
    main()
