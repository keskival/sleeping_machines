"""E71: spoken digits (SHD) with event-CDE units (THEORY §104): each layer is a diagonal complex linear CDE driven by the
event stream, flowing in closed form between events and jumping at them.

Events: spikes pooled into B bands and merged within `--merge` seconds per band (payload: log count). Layer:
    z_k = exp(Lambda * (delta * dt_k + sel_k)) * z_{k-1} + B x_k,     y_k = Re(C z_k) + D x_k -> GELU -> GLU, residual
  dt_k   the true time since the previous event (the time channel of the gate path)
  sel_k  = softplus(W_s x_k + b_s) >= 0: a content-driven advance of internal time (--sel 1), i.e. the gate path
         omega = (t, sum over events of softplus(W_s x)): the diagonal closure of §104(c) weights each past event by a function
         of its age and of what arrived since. --sel 0 is the non-selective gate omega = t (Event-SSM).
Sparse subscription (--sub W > 0, first layer): the state is split into G units with tonotopic windows of W bands; an event
on band b drives (input and jump) only the units whose window contains b; the others only flow, which the lazy simulation
reproduces exactly (§104 proposition), so first-layer work is counted as events x subscribed units.
Pooling (--pool p): after each layer except the last, keep every p-th event (the state at kept events is exact).
Readout: mean over the last layer's events -> 20 classes. Selection on held-out speakers (--eval spk); test once (--eval test).
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

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e71")


def events(t, u, merge):
    """merge spikes of one band closer than `merge` s into one event (time of the first, payload log(1 + count))."""
    o = np.lexsort((t, u)); uu, tt = u[o], t[o]
    new = np.r_[True, (uu[1:] != uu[:-1]) | (np.diff(tt) > merge)]
    g = np.cumsum(new) - 1; cnt = np.bincount(g)
    eb, et = uu[new], tt[new]
    k = np.argsort(et, kind="stable")
    return eb[k].astype(np.int16), (et[k] - et[k][0]).astype(np.float32), np.log1p(cnt[k]).astype(np.float32)


def scan(loga, b, C=None):
    """z_k = exp(loga_k) z_{k-1} + b_k along dim 1 by a log-depth associative scan (Hillis-Steele): after the step with
    shift s, (A_k, b_k) summarise events k-2s+1..k; A is a product of decays (|A| <= 1), so the scan is stable."""
    A, bb = torch.exp(loga), b; Bn, L, N = b.shape; s = 1
    while s < L:
        bb = bb + A * torch.cat([bb.new_zeros(Bn, s, N), bb[:, :L - s]], 1)
        A = A * torch.cat([A.new_ones(Bn, s, N), A[:, :L - s]], 1)
        s *= 2
    return bb


class CDELayer(nn.Module):
    def __init__(self, D, N, sel, chunk):
        super().__init__()
        self.N, self.sel, self.chunk = N, sel, chunk
        tau = torch.exp(torch.linspace(math.log(1e-3), math.log(1.0), N))       # time constants 1 ms .. 1 s
        self.log_rate = nn.Parameter(-torch.log(tau))                           # Re Lambda = -exp(log_rate)
        self.freq = nn.Parameter(torch.rand(N) * math.pi / tau)                  # Im Lambda
        self.log_delta = nn.Parameter(torch.zeros(N))
        self.Bre = nn.Parameter(torch.randn(D, N) / math.sqrt(D)); self.Bim = nn.Parameter(torch.randn(D, N) / math.sqrt(D))
        self.Cre = nn.Parameter(torch.randn(N, D) / math.sqrt(N)); self.Cim = nn.Parameter(torch.randn(N, D) / math.sqrt(N))
        self.Dskip = nn.Parameter(torch.ones(D))
        if sel:
            self.Ws = nn.Linear(D, N); nn.init.zeros_(self.Ws.weight); nn.init.constant_(self.Ws.bias, -8.0)
        self.norm = nn.LayerNorm(D); self.g1 = nn.Linear(D, D); self.g2 = nn.Linear(D, D)

    def forward(self, x, dt, mask, sub=None):
        """x (B, L, D), dt (B, L) seconds since the previous event, mask (B, L); sub (B, L, N) subscription mask or None."""
        h = self.norm(x)
        lam = torch.complex(-torch.exp(self.log_rate), self.freq)               # (N,)
        adv = torch.exp(self.log_delta)[None, None] * dt[..., None]            # internal-time advance by the flow
        if self.sel:
            jump = nn.functional.softplus(self.Ws(h))                           # content-driven advance (the jump)
            adv = adv + (jump * sub if sub is not None else jump)
        loga = lam * adv.to(torch.complex64)
        bin_ = torch.complex(h @ self.Bre, h @ self.Bim) * mask[..., None]
        if sub is not None:
            bin_ = bin_ * sub
        z = scan(loga, bin_)
        y = (z.real @ self.Cre - z.imag @ self.Cim) + self.Dskip * h
        y = nn.functional.gelu(y)
        return x + self.g1(y) * torch.sigmoid(self.g2(y))


class Net(nn.Module):
    def __init__(self, B, D, N, layers, sel, chunk, sub_w, groups):
        super().__init__()
        self.band = nn.Embedding(B, D); self.cnt = nn.Linear(1, D)
        self.layers = nn.ModuleList([CDELayer(D, N, sel, chunk) for _ in range(layers)])
        self.norm = nn.LayerNorm(D); self.out = nn.Linear(D, 20)
        self.register_buffer("sub", None)
        if sub_w:                                                               # tonotopic windows over bands, per state group
            centers = (torch.arange(groups) + 0.5) * B / groups
            win = (torch.arange(B)[:, None] - centers[None]).abs() <= sub_w / 2  # (B, G)
            self.register_buffer("sub", win.float().repeat_interleave(N // groups, 1))   # (B, N)

    def forward(self, band, t, cnt, mask, pool):
        x = self.band(band.long()) + self.cnt(cnt[..., None])
        work = []
        for i, layer in enumerate(self.layers):
            dt = torch.diff(t, dim=1, prepend=t[:, :1])
            sub = self.sub[band.long()] if (i == 0 and self.sub is not None) else None
            x = layer(x, dt, mask, sub)
            work.append(float(mask.sum()) * (float(sub.mean()) if sub is not None else 1.0))
            if pool > 1 and i < len(self.layers) - 1:
                x, t, mask, band = x[:, pool - 1::pool], t[:, pool - 1::pool], mask[:, pool - 1::pool], band[:, pool - 1::pool]
        z = self.norm(x) * mask[..., None]
        return self.out(z.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)), work


def batchify(items, B, shift, rng, drop):
    items = [(b, t, c, y) for b, t, c, y in items]
    if drop:
        keep = [rng.random(len(b)) >= drop for b, *_ in items]
        items = [(b[k], t[k], c[k], y) for (b, t, c, y), k in zip(items, keep)]
    L = max(len(b) for b, *_ in items)
    band = torch.zeros(len(items), L, dtype=torch.long); tt = torch.zeros(len(items), L); cc = torch.zeros(len(items), L)
    mask = torch.zeros(len(items), L)
    for i, (b, t, c, _) in enumerate(items):
        sh = int(rng.integers(-shift, shift + 1)) if shift else 0
        n = len(b); band[i, :n] = torch.from_numpy(np.clip(b.astype(np.int64) + sh, 0, B - 1))
        tt[i, :n] = torch.from_numpy(t); tt[i, n:] = float(t[-1]) if n else 0.0
        cc[i, :n] = torch.from_numpy(c); mask[i, :n] = 1
    return band, tt, cc, mask, torch.tensor([y for *_, y in items])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=140)
    ap.add_argument("--merge", type=float, default=0.002)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--sel", type=int, default=1)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--sub", type=int, default=0, help="first-layer tonotopic window in bands (0: dense)")
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--shift", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval", default="spk", choices=("spk", "test"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many training utterances")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    def load(split, part):
        return [(*events(t, u, a.merge), y) for t, u, y in S.utterances(split, a.B, part) if len(t) > 1]
    tr = load("train", "fit_spk" if a.eval == "spk" else None)
    ev = load("train", "val_spk") if a.eval == "spk" else load("test", None)
    if a.limit:
        tr = tr[:a.limit]; ev = ev[:max(64, a.limit // 4)]
    print(json.dumps({"train": len(tr), "eval": len(ev), "median_events": int(np.median([len(b) for b, *_ in tr])),
                      "load_s": round(time.time() - t0)}), flush=True)
    net = Net(a.B, a.D, a.N, a.layers, a.sel, a.chunk, a.sub, a.groups)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    order = np.argsort([len(b) for b, *_ in tr])                               # length buckets: batches of similar length
    buckets = [order[i:i + a.bs] for i in range(0, len(order), a.bs)]
    steps = a.epochs * len(buckets); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()), "curve": []}
    for ep in range(a.epochs):
        net.train(); tl = 0.0
        for bi in rng.permutation(len(buckets)):
            band, tt, cc, mask, y = batchify([tr[j] for j in buckets[bi]], a.B, a.shift, rng, a.drop)
            logits, _ = net(band, tt, cc, mask, a.pool)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tl += float(loss)
        net.eval(); ok = 0; work = np.zeros(a.layers); nev = 0
        eo = np.argsort([len(b) for b, *_ in ev])
        with torch.no_grad():
            for i0 in range(0, len(ev), 64):
                idx = eo[i0:i0 + 64]
                band, tt, cc, mask, y = batchify([ev[j] for j in idx], a.B, 0, rng, 0.0)
                logits, w = net(band, tt, cc, mask, a.pool); ok += int((logits.argmax(1) == y).sum())
                work += np.array(w); nev += len(idx)
        row = {"epoch": ep + 1, "train_loss": round(tl / len(buckets), 4), f"{a.eval}_acc": round(ok / len(ev), 4),
               "unit_events_per_utt_by_layer": (work / nev).round(1).tolist(), "wall_s": round(time.time() - t0)}
        res["curve"].append(row); print(json.dumps(row), flush=True)
    tag = f"sel{a.sel}_sub{a.sub}_D{a.D}_N{a.N}_L{a.layers}_p{a.pool}_{a.eval}_s{a.seed}"
    with open(os.path.join(OUT, f"cde_{tag}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
