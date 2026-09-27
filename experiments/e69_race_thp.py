"""E69: the market world model with race attention: the E52 Transformer Hawkes process with its attention replaced by
time-normalized races (THEORY §102-§103), same hazard family, same protocol (select on day 5 after training on days 1-4;
test: train on days 1-5, score days 6-7 once). --R 0 reproduces softmax attention in the same code (control).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from e17_market import load_day, PILOT  # noqa: E402
from e44_tpp import day_events, NT, SemiMarkov  # noqa: E402
import e52_thp as T52  # noqa: E402
from e68_race_transformer import Block  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e69")


class RaceTHP(nn.Module):
    def __init__(self, d, L, nf, nb, R, layers=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(NT, d); self.tf = nn.Linear(nf, d); self.pos = nn.Embedding(L, d)
        self.blocks = nn.ModuleList([Block(d, heads, R) for _ in range(layers)]); self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, nb * NT); self.nb = nb

    def forward(self, y, x):
        n = y.shape[1]
        z = self.emb(y) + self.tf(x) + self.pos.weight[:n]
        for b in self.blocks:
            z = b(z, True)
        return nn.functional.softplus(self.head(self.norm(z))).view(*z.shape[:2], self.nb, NT) + 1e-6

    ll = T52.THP.ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="val", choices=("val", "test"))
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--fine", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(0); rng = np.random.default_rng(0)
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT]); big_q = float(np.quantile(qs, 0.99))
    days = [day_events(d, big_q) for d in PILOT]; gaps = T52.bank(a.fine)
    encs = [T52.encode(T, Y, gaps) for T, Y in days]
    ntr = 4 if a.mode == "val" else 5; evals = [4] if a.mode == "val" else [5, 6]
    chunks = [(di, s) for di in range(ntr) for s in T52.windows(len(encs[di][0]) - 1, a.L, a.L)]
    net = RaceTHP(a.d, a.L, encs[0][1].shape[1], len(gaps) + 1, a.R); opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()), "epochs": []}
    for ep in range(a.epochs):
        net.train(); order = rng.permutation(len(chunks))
        for i in range(0, len(order), 16):
            bt = [chunks[k] for k in order[i:i + 16]]
            parts = [[e[s:s + a.L] for e in encs[di]] for di, s in bt]
            y, x, b, sp, yn = (torch.stack([p[j] for p in parts]) for j in range(5))
            loss = -net.ll(y, x, b, sp, yn).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); row = {"epoch": ep + 1, "wall_s": round(time.time() - t0, 1)}
        for di in evals:
            row[f"day{di + 1}"] = T52.score_day(net, encs[di], a.L)[0]
        res["epochs"].append(row); print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"race_thp_{a.mode}_R{a.R}_d{a.d}_L{a.L}_f{a.fine}_e{a.epochs}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
