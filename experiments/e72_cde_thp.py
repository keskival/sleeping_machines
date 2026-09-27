"""E72: the market world model with event-CDE units (THEORY §104) in place of attention: a neural point process whose
state is a stack of diagonal complex linear CDEs driven by the trade events (closed-form flow over the true gap, jump at
each event, content-selective time advance), with the E52 hazard head (piecewise-constant rates per type over the gap
window bank). Same protocol as E52/E69: --mode val trains on days 1-4 and scores day 5 (selection); --mode test trains on
days 1-5 and scores days 6-7 once. Recurrence makes long context cheap (--L), unlike attention.
Work per event: layers x (2 D N complex projections + 2 D^2 gate) multiply-adds, reported.
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
from e44_tpp import day_events, NT  # noqa: E402
import e52_thp as T52  # noqa: E402
from e71_event_cde import CDELayer  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e72")


class CDETHP(nn.Module):
    def __init__(self, d, N, nf, nb, layers, sel):
        super().__init__()
        self.emb = nn.Embedding(NT, d); self.tf = nn.Linear(nf, d)
        self.layers = nn.ModuleList([CDELayer(d, N, sel, 0, (1e-2, 1e3)) for _ in range(layers)]); self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, nb * NT); self.nb = nb

    def forward(self, y, x):
        dt = torch.exp(5 * x[..., 0]) - 1e-4                                 # the gap before each event (encode: log(g)/5)
        dt = torch.cat([torch.zeros_like(dt[:, :1]), dt[:, 1:]], 1)          # no flow into the first event of a window
        z = self.emb(y) + self.tf(x); mask = torch.ones(y.shape, dtype=z.dtype)
        for layer in self.layers:
            z = layer(z, dt, mask)
        return nn.functional.softplus(self.head(self.norm(z))).view(*z.shape[:2], self.nb, NT) + 1e-6

    ll = T52.THP.ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="val", choices=("val", "test"))
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--sel", type=int, default=1)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--fine", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(0); rng = np.random.default_rng(0)
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT]); big_q = float(np.quantile(qs, 0.99))
    days = [day_events(d, big_q) for d in PILOT]; gaps = T52.bank(a.fine)
    encs = [T52.encode(T, Y, gaps) for T, Y in days]
    ntr = 4 if a.mode == "val" else 5; evals = [4] if a.mode == "val" else [5, 6]
    chunks = [(di, s) for di in range(ntr) for s in T52.windows(len(encs[di][0]) - 1, a.L, a.L)]
    net = CDETHP(a.d, a.N, encs[0][1].shape[1], len(gaps) + 1, a.layers, a.sel)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    madds = a.layers * (2 * a.d * a.N * 2 + 2 * a.d * a.d + a.d * (a.N if a.sel else 0)) + a.d * (len(gaps) + 1) * NT
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()), "madds_per_event": madds, "epochs": []}
    for ep in range(a.epochs):
        net.train(); order = rng.permutation(len(chunks))
        for i in range(0, len(order), a.batch):
            bt = [chunks[k] for k in order[i:i + a.batch]]
            parts = [[e[s:s + a.L] for e in encs[di]] for di, s in bt]
            y, x, b, sp, yn = (torch.stack([p[j] for p in parts]) for j in range(5))
            loss = -net.ll(y, x, b, sp, yn).mean(); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval(); row = {"epoch": ep + 1, "wall_s": round(time.time() - t0, 1)}
        for di in evals:
            row[f"day{di + 1}"] = T52.score_day(net, encs[di], a.L)[0]
        res["epochs"].append(row); print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"cde_thp_{a.mode}_d{a.d}_N{a.N}_l{a.layers}_s{a.sel}_L{a.L}_f{a.fine}_e{a.epochs}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
