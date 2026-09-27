"""E52: the Transformer Hawkes Process baseline (Zuo et al. 2020) for the market world model, same protocol as E48/E49.

Stream: E44's 4 event types (price up, price down, large buy, large sell). Model: causal Transformer over the last L events;
input per event = type embedding + a projection of time features (log gap, its window-bank one-hot, sinusoids of the log
gap). Output: from the context vector h_j after event j, one intensity per event type per gap window of the SAME window
bank as the semi-Markov event network (0.01, 0.05, 0.2, 1, 5 s): lambda_k(tau) = softplus(W_b h_j + c_b)_k for tau in
window b. The compensator is then exact (sum of rate x span), so the likelihood is exactly comparable, and the only
difference from the event network is the context encoder (attention over L events and times vs the last two types).
--fine 1 uses 12 log-spaced windows (1 ms to 30 s) instead, a strictly richer hazard family.
Protocol: --mode val trains on days 1-4 and reports day 5 per epoch (model selection); --mode test trains on days 1-5
for --epochs and reports days 6-7 frozen, with the semi-Markov table model (frozen after days 1-5) alongside.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e17_market import load_day, PILOT  # noqa: E402
from e44_tpp import day_events, NT, SemiMarkov  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e52")


def bank(fine):
    return np.geomspace(0.001, 30.0, 12) if fine else np.asarray(SemiMarkov.GAPS, float)


def encode(T, Y, gaps):
    """per event j: (type, time features of the gap before j), and the gap after j with its window spans."""
    edges = np.r_[0.0, gaps, np.inf]
    g_in = np.r_[1.0, np.diff(T)]                                       # gap before each event
    lg = np.log(g_in + 1e-4)
    feats = [lg[:, None] / 5]
    feats.append(np.eye(len(gaps) + 1)[np.searchsorted(gaps, g_in)])
    fr = np.array([0.5, 1, 2, 4])
    feats += [np.sin(lg[:, None] * fr), np.cos(lg[:, None] * fr)]
    X = np.concatenate(feats, 1).astype(np.float32)
    g_out = np.r_[np.diff(T), np.nan]                                   # gap to the next event (target)
    b_out = np.searchsorted(gaps, np.nan_to_num(g_out))
    span = np.clip(np.nan_to_num(g_out)[:, None] - edges[None, :-1], 0, np.diff(edges)[None, :]).astype(np.float32)
    y_next = np.r_[Y[1:], 0]
    return (torch.tensor(Y, dtype=torch.long), torch.tensor(X), torch.tensor(b_out, dtype=torch.long),
            torch.tensor(span), torch.tensor(y_next, dtype=torch.long))


class THP(torch.nn.Module):
    def __init__(self, d, L, nf, nb, layers=2, heads=4):
        super().__init__()
        self.emb = torch.nn.Embedding(NT, d); self.tf = torch.nn.Linear(nf, d); self.pos = torch.nn.Embedding(L, d)
        layer = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = torch.nn.Linear(d, nb * NT); self.nb = nb
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

    def forward(self, y, x):
        n = y.shape[1]
        z = self.emb(y) + self.tf(x) + self.pos.weight[:n]
        h = self.enc(z, mask=self.mask[:n, :n], is_causal=True)
        return torch.nn.functional.softplus(self.head(h)).view(*h.shape[:2], self.nb, NT) + 1e-6   # (B, n, nb, NT)

    def ll(self, y, x, b, span, yn):
        """per-position log-likelihood of the NEXT event (type yn after gap in window b; exposure span)."""
        lam = self(y, x)
        at = lam.gather(2, b[..., None, None].expand(*b.shape, 1, NT)).squeeze(2)          # rates in the event's window
        return torch.log(at.gather(2, yn[..., None]).squeeze(2)) - (lam.sum(3) * span).sum(2)


def windows(n, L, stride):
    starts = list(range(0, max(n - L, 0) + 1, stride))
    if starts[-1] + L < n:
        starts.append(n - L)
    return starts


def score_day(net, enc, L):
    """every event (but the first L/2 of the day) scored with a context of >= L/2 previous events."""
    y, x, b, sp, yn = enc; n = len(y) - 1; tot = 0.0; cnt = 0; done = L // 2
    with torch.no_grad():
        for s in windows(n, L, L // 2):
            sl = slice(s, s + L)
            ll = net.ll(y[None, sl], x[None, sl], b[None, sl], sp[None, sl], yn[None, sl])[0]
            lo = max(done, s + L // 2) - s; hi = min(s + L, n) - s
            if hi > lo:
                tot += float(ll[lo:hi].sum()); cnt += hi - lo; done = s + hi
    return tot / cnt, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="val", choices=("val", "test"))
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fine", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); torch.manual_seed(0); rng = np.random.default_rng(0)
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT]); big_q = float(np.quantile(qs, 0.99))
    days = [day_events(d, big_q) for d in PILOT]
    gaps = bank(a.fine)
    encs = [encode(T, Y, gaps) for T, Y in days]
    ntr = 4 if a.mode == "val" else 5
    evals = [4] if a.mode == "val" else [5, 6]
    chunks = [(di, s) for di in range(ntr) for s in windows(len(encs[di][0]) - 1, a.L, a.L)]
    net = THP(a.d, a.L, encs[0][1].shape[1], len(gaps) + 1)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    nparam = sum(p.numel() for p in net.parameters())
    res = {"args": vars(a), "params": nparam, "epochs": []}
    for ep in range(a.epochs):
        net.train(); order = rng.permutation(len(chunks)); tl = 0.0; tn = 0
        for i in range(0, len(order), a.batch):
            bt = [chunks[k] for k in order[i:i + a.batch]]
            parts = [[e[s:s + a.L] for e in encs[di]] for di, s in bt]
            y, x, b, sp, yn = (torch.stack([p[j] for p in parts]) for j in range(5))
            ll = net.ll(y, x, b, sp, yn)                            # targets are precomputed per position
            loss = -ll.mean(); opt.zero_grad(); loss.backward(); opt.step()
            tl += float(ll.sum()); tn += ll.numel()
        net.eval()
        row = {"epoch": ep + 1, "train_ll": tl / tn, "wall_s": round(time.time() - t0, 1)}
        for di in evals:
            row[f"day{di + 1}"] = score_day(net, encs[di], a.L)[0]
        res["epochs"].append(row); print(json.dumps(row), flush=True)
    if True:                                                          # reference: semi-Markov frozen after the same days
        sm = SemiMarkov(order=2)
        for T, Y in days[:ntr]:
            for t, yy in zip(T, Y):
                sm.step(t, yy)
        sm.learn = False
        for di in evals:
            T, Y = days[di]; ll = [sm.step(t, yy) for t, yy in zip(T, Y)]
            res[f"semimarkov_day{di + 1}"] = float(np.mean([v for v in ll if v is not None]))
    d, L = a.d, a.L
    res["macs_per_event"] = 2 * (12 * d * d + 2 * L * d) + d * (len(gaps) + 1) * NT
    print(json.dumps({k: v for k, v in res.items() if k != "epochs"}), flush=True)
    with open(os.path.join(OUT, f"thp_{a.mode}_d{a.d}_L{a.L}_f{a.fine}_e{a.epochs}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
