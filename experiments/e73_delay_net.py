"""E73: computing with delays. Spoken digits (SHD) with a network whose values are spike TIMES: every synapse has a
learnable weight and a learnable delay; a unit integrates its delayed arrivals (membrane kernel of §104: a two-mode linear
flow, jump at each arrival) and fires when the potential crosses its threshold; the crossing times are the next layer's
inputs. Learning is exact in the spike times (THEORY §104d, the time-change / implicit-function gradient, EventProp's jump):
    T = T0 - (V(T0) - theta) / dV/dt(T0)        (V differentiable, dV/dt detached)
so gradients flow only through spikes that occurred: to their weights, their delays, and the spike times feeding them.

Forward (exact up to the grid used to detect crossings): arrivals t_s + d_ij are scattered onto a 1 ms grid with their exact
sub-bin decay; two traces per unit (membrane and synaptic time constants) give V on the grid; crossings are located by
interpolation and refined by the differentiable Newton step above, which evaluates V(T0) exactly from each input's
filtered spike train D_i (continuous: kernel(0) = 0) tabulated at 1 ms and interpolated at T0 - d_ij.
Readout: 20 non-spiking units with delays (V_c(t) = sum_i w_ci D_i(t - d_ci)); loss = -log mean_t softmax_c V_c(t).
Selection on held-out speakers (--eval spk); test once (--eval test). Work: input arrivals and spikes per layer, reported.
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
from e71_event_cde import events  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e73")
TM, TS = 20.0, 5.0                                                      # membrane and synaptic time constants (ms)
_sp = math.log(TM / TS) * TM * TS / (TM - TS)
A = 1.0 / (math.exp(-_sp / TM) - math.exp(-_sp / TS))                  # kernel peak normalized to 1
KWIN = 100                                                              # kernel window (ms): exp(-100/20) = 0.007


def kernel(u):
    return A * (torch.exp(-u / TM) - torch.exp(-u / TS))


def build_table(sb, si, st, B, n, G):
    """D[b, i, g] = sum over spikes s of (b, i) of kernel(g - t_s), g on a 1 ms grid; differentiable in the spike times."""
    g = torch.floor(st.detach())[:, None] + 1 + torch.arange(KWIN, dtype=st.dtype)[None]      # grid points after t_s
    val = kernel(g - st[:, None]); ok = g < G
    idx = ((sb[:, None] * n + si[:, None]) * G + g.long().clamp(max=G - 1))
    return torch.zeros(B * n * G, dtype=st.dtype).index_add(0, idx[ok], val[ok]).view(B, n, G)


def interp(D, b, i, u):
    """value and slope of D[b, i, .] at continuous time u (ms), linear between grid points; zero before 0."""
    B, n, G = D.shape
    pos = u.clamp(0, G - 1.001); f = pos.detach().floor(); fr = pos - f
    flat = (b * n + i) * G + f.long(); Df = D.reshape(-1)
    v0, v1 = Df[flat], Df[flat + 1]
    return v0 + fr * (v1 - v0), (v1 - v0)


class DelayLayer(nn.Module):
    def __init__(self, n_in, M, dmax, w_mu, w_sd, theta=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_in, M) * w_sd + w_mu)
        self.D = nn.Parameter(torch.rand(n_in, M) * dmax); self.dmax = dmax; self.theta = theta; self.M = M

    def delays(self):
        return self.D.clamp(0, self.dmax)

    @torch.no_grad()
    def simulate(self, sb, si, st, B, G):
        """spikes (b, unit, T0, R(T0)) from input spikes (sb, si, st), on a 1 ms grid with exact sub-bin arrival decay."""
        M, W, Dl = self.M, self.W, self.delays()
        ta = st[:, None] + Dl[si]                                             # (S, M) arrival times
        g = torch.ceil(ta).clamp(max=G - 1); off = (g - ta).clamp(min=0); wv = W[si]
        idx = ((g.long() * B + sb[:, None]) * M + torch.arange(M)[None]).reshape(-1)          # time-major (G, B, M)
        Xm = torch.zeros(G * B * M).index_add_(0, idx, (wv * torch.exp(-off / TM)).reshape(-1)).view(G, B, M)
        Xs = torch.zeros(G * B * M).index_add_(0, idx, (wv * torch.exp(-off / TS)).reshape(-1)).view(G, B, M)
        em, es, th = math.exp(-1 / TM), math.exp(-1 / TS), self.theta
        Pm = torch.zeros(B, M); Ps = torch.zeros(B, M); R = torch.zeros(B, M); Vp = torch.zeros(B, M)
        F = torch.zeros(G, B, M, dtype=torch.bool); FR = torch.zeros(G, B, M); RP = torch.zeros(G, B, M)
        for k in range(G):
            Pm.mul_(em).add_(Xm[k]); Ps.mul_(es).add_(Xs[k]); R.mul_(em)
            V = A * (Pm - Ps) - th * R
            fire = V >= th
            frac = ((th - Vp) / (V - Vp).clamp(min=1e-6)).clamp(0, 1)
            RP[k] = R * torch.exp(frac / TM) * fire                          # R at the crossing (before its own reset)
            FR[k] = frac; F[k] = fire
            jump = fire * torch.exp(-(1 - frac) / TM)
            R.add_(jump); Vp = V - th * jump
        kk, bb, jj = F.nonzero(as_tuple=True)
        return bb, jj, (kk - 1).float() + FR[kk, bb, jj], RP[kk, bb, jj]

    def refine(self, kb, kj, T0, Rpre, table):
        """differentiable spike times: one Newton step on V(T) = theta from the detected T0 (implicit-function gradient)."""
        n_in = self.W.shape[0]; i = torch.arange(n_in)[None]
        u = T0[:, None] - self.delays()[:, kj].T                              # (K, n_in)
        val, slope = interp(table, kb[:, None], i, u)
        Wk = self.W[:, kj].T
        V = (Wk * val).sum(1) - self.theta * Rpre
        Vdot = ((Wk * slope).sum(1) + self.theta * Rpre / TM).detach().clamp(min=0.02)
        return T0 - (V - self.theta) / Vdot


class Net(nn.Module):
    def __init__(self, n_in, hidden, dmax, w_mu, w_sd, rstep):
        super().__init__()
        sizes = [n_in] + hidden
        self.layers = nn.ModuleList([DelayLayer(sizes[l], sizes[l + 1], dmax, w_mu[l], w_sd[l]) for l in range(len(hidden))])
        self.Wo = nn.Parameter(torch.randn(hidden[-1], 20) / math.sqrt(hidden[-1]))
        self.Do = nn.Parameter(torch.rand(hidden[-1], 20) * dmax); self.dmax = dmax; self.rstep = rstep

    def forward(self, sb, si, st, B, G):
        counts = []
        for layer in self.layers:
            table = build_table(sb, si, st, B, layer.W.shape[0], G)
            kb, kj, T0, Rpre = layer.simulate(sb, si, st.detach(), B, G)
            T = layer.refine(kb, kj, T0, Rpre, table) if len(kb) else T0
            ok = (T > 0) & (T < G - 1)
            sb, si, st = kb[ok], kj[ok], T[ok]; counts.append(len(st) / B)
        table = build_table(sb, si, st, B, self.layers[-1].M, G)
        tr = torch.arange(0, G, self.rstep, dtype=torch.float32)                  # readout times
        H = self.Wo.shape[0]
        u = tr[None, None, :] - self.Do.clamp(0, self.dmax)[:, :, None]            # (H, C, R)
        val, _ = interp(table, torch.arange(B)[:, None, None, None], torch.arange(H)[None, :, None, None], u[None])
        Vc = (self.Wo[None, :, :, None] * val).sum(1)                               # (B, C, R)
        p = torch.softmax(Vc, 1).mean(2)
        return p, counts


def to_spikes(items, B_bands, shift, rng, drop):
    sb, si, st, ys = [], [], [], []
    for n, (b, t, c, y) in enumerate(items):
        k = rng.random(len(b)) >= drop if drop else np.ones(len(b), bool)
        sh = int(rng.integers(-shift, shift + 1)) if shift else 0
        sb.append(np.full(k.sum(), n)); si.append(np.clip(b[k].astype(np.int64) + sh, 0, B_bands - 1)); st.append(t[k] * 1000.0)
        ys.append(y)
    st = np.concatenate(st)
    return (torch.from_numpy(np.concatenate(sb)).long(), torch.from_numpy(np.concatenate(si)).long(),
            torch.from_numpy(st).float(), torch.tensor(ys), int(st.max()) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=140)
    ap.add_argument("--merge", type=float, default=0.001)
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--dmax", type=float, default=100.0)
    ap.add_argument("--w_mu", default="0.004,0.01")
    ap.add_argument("--w_sd", default="0.06,0.1")
    ap.add_argument("--rstep", type=int, default=5)
    ap.add_argument("--shift", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr_delay", type=float, default=0.5)
    ap.add_argument("--eval", default="spk", choices=("spk", "test"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    def load(split, part):
        return [(*events(t, u, a.merge), y) for t, u, y in S.utterances(split, a.B, part) if len(t) > 1]
    tr = load("train", "fit_spk" if a.eval == "spk" else None)
    ev = load("train", "val_spk") if a.eval == "spk" else load("test", None)
    if a.limit:
        tr = tr[:a.limit]; ev = ev[:max(64, a.limit // 4)]
    hidden = [int(h) for h in a.hidden.split(",")]
    net = Net(a.B, hidden, a.dmax, [float(v) for v in a.w_mu.split(",")], [float(v) for v in a.w_sd.split(",")], a.rstep)
    dparams = [l.D for l in net.layers] + [net.Do]
    wparams = [p for p in net.parameters() if all(p is not q for q in dparams)]
    opt = torch.optim.Adam([{"params": wparams, "lr": a.lr}, {"params": dparams, "lr": a.lr_delay}])
    nb = math.ceil(len(tr) / a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.lr_delay], total_steps=a.epochs * nb, pct_start=0.1)
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()), "curve": []}
    print(json.dumps({"train": len(tr), "eval": len(ev), "params": res["params"], "load_s": round(time.time() - t0)}), flush=True)
    for ep in range(a.epochs):
        net.train(); tl = 0.0; cnt = np.zeros(len(hidden))
        perm = rng.permutation(len(tr))
        for i0 in range(0, len(tr), a.bs):
            items = [tr[j] for j in perm[i0:i0 + a.bs]]
            sb, si, st, y, tmax = to_spikes(items, a.B, a.shift, rng, a.drop)
            p, c = net(sb, si, st, len(items), tmax + int(a.dmax) + KWIN)
            loss = -torch.log(p[torch.arange(len(y)), y] + 1e-8).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tl += float(loss); cnt += np.array(c)
        net.eval(); ok = 0; ecnt = np.zeros(len(hidden)); nbe = 0
        with torch.no_grad():
            for i0 in range(0, len(ev), a.bs):
                items = ev[i0:i0 + a.bs]
                sb, si, st, y, tmax = to_spikes(items, a.B, 0, rng, 0.0)
                p, c = net(sb, si, st, len(items), tmax + int(a.dmax) + KWIN)
                ok += int((p.argmax(1) == y).sum()); ecnt += np.array(c) * len(items); nbe += len(items)
        row = {"epoch": ep + 1, "train_loss": round(tl / nb, 4), f"{a.eval}_acc": round(ok / len(ev), 4),
               "spikes_per_utt_by_layer": (ecnt / nbe).round(1).tolist(), "wall_s": round(time.time() - t0)}
        res["curve"].append(row); print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"delay_h{a.hidden.replace(',', '-')}_d{a.dmax:g}_{a.eval}_s{a.seed}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
