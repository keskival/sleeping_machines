"""E49: the strongest dense baseline for the market world model: a GRU neural point process trained OFFLINE on the pilot
days 1–5 (several epochs, truncated backprop through time over chunks of events, Adam), then evaluated frozen on days 6–7
by the same exact-per-event log-likelihood as E44/E48 (compensator by 6-point quadrature). Inputs per event: type
one-hot and log gap (as E44's online GRU). Compares with the semi-Markov event network frozen after days 1–5.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e17_market import load_day, PILOT  # noqa: E402
from e44_tpp import day_events, NT, SemiMarkov  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e49")


class GRUTPP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.cell = torch.nn.GRUCell(NT + 1, d); self.out = torch.nn.Linear(d, NT)
        self.log_gamma = torch.nn.Parameter(torch.zeros(d)); self.d = d

    def chunk_ll(self, h, T, Y):
        """sum of per-event log-likelihoods over a chunk (events 1..n given event 0), returns (ll, n, h_end)."""
        g = torch.exp(self.log_gamma); ll = 0.0
        for i in range(1, len(T)):
            dt = max(T[i] - T[i - 1], 1e-6)
            ts = torch.linspace(0, dt, 6)
            lam = torch.nn.functional.softplus(self.out(h * torch.exp(-g * ts[:, None])))      # (6, NT)
            comp = torch.trapezoid(lam.sum(1), ts)
            ll = ll + torch.log(lam[-1, Y[i]] + 1e-9) - comp
            x = torch.zeros(1, NT + 1); x[0, Y[i]] = 1.0; x[0, NT] = float(np.log(dt + 1e-3))
            h = self.cell(x, h * torch.exp(-g * dt))
        return ll, len(T) - 1, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT]); big_q = float(np.quantile(qs, 0.99))
    data = [day_events(d, big_q) for d in PILOT]
    torch.manual_seed(0)
    net = GRUTPP(a.d); opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    for ep in range(a.epochs):
        tot = n = 0
        for T, Y in data[:5]:
            h = torch.zeros(1, a.d)
            for s in range(0, len(T) - 1, a.chunk):
                ll, k, h = net.chunk_ll(h, T[s:s + a.chunk + 1], Y[s:s + a.chunk + 1])
                opt.zero_grad(); (-ll / k).backward(); opt.step(); h = h.detach()
                tot += float(ll); n += k
        print(json.dumps({"epoch": ep + 1, "train_ll_per_event": tot / n}), flush=True)
    res = {"args": vars(a), "heldout": []}
    sm = SemiMarkov(order=2)
    for T, Y in data[:5]:
        for t, y in zip(T, Y):
            sm.step(t, y)
    sm.learn = False
    with torch.no_grad():
        for di in (5, 6):
            T, Y = data[di]
            ll_g, k, _ = net.chunk_ll(torch.zeros(1, a.d), T, Y)
            ll_s = [sm.step(t, y) for t, y in zip(T, Y)]
            row = {"day": str(PILOT[di]), "gru_offline_frozen": float(ll_g) / k,
                   "semimarkov_frozen": float(np.mean([v for v in ll_s if v is not None]))}
            res["heldout"].append(row); print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"gru_offline_d{a.d}_e{a.epochs}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
