"""E75: the equivariant time-vector network of THEORY §106(c) on SHD.

Units sit on a lattice of cochlear positions p and log time-scales m. Unit (p, m) has mode rates Lambda_0 rho^-m, content-delay
scale tau_0 rho^m and reset time tau_R rho^m; every other parameter depends only on the offset between sender and receiver.
So the network is equivariant to channel shifts (vocal-tract length) and to time dilations by rho (tempo), exactly (§106(c)),
and its readout pools over the lattice (invariant). The mechanisms are those of §105 / E74: content-gated, content-delayed
messages; transported payloads; threshold firing with exact spike-time gradients; snapshot emission.
  layer 1: P1 positions (every `stride1` bands) x S scales; a band event reaches units whose window covers it; the message's
           gate/delay bias and written offset embedding depend on (band - centre).
  layer 2: P2 positions (every `stride2` layer-1 positions) x S scales; fed by layer-1 units within +-w2 positions and +-1 scale,
           offset-dependent gate/delay bias and offset embedding.
  readout: 20 non-spiking class units fed by every layer-2 unit, with parameters that ignore the sender's position (pooling).
Reports held-out-speaker accuracy and accuracy on 500 training utterances (the generalization gap of §106(d)).
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
OUT = os.path.join(os.path.dirname(__file__), "results", "e75")
TAU_R = 20.0


class EqLayer(nn.Module):
    """time-vector layer with offset-shared parameters; offs[i, j] = offset id of sender i -> unit j (-1: no synapse)."""
    def __init__(self, offs, scale, d, n, spiking=True, w_sd=0.05, theta=1.0, gate_bias=0.5, dmax=50.0, per_unit=False):
        super().__init__()
        self.per_unit = per_unit                    # readout classes: own parameters, still blind to the sender's position
        n_off = int(offs.max()) + 1
        self.register_buffer("offs", offs); self.register_buffer("scale", scale)      # scale: (M,) = rho^m
        self.register_buffer("sent", torch.zeros(offs.shape, dtype=torch.bool), persistent=False)   # §107(e) diagnostic
        self.M, self.n, self.spiking, self.theta, self.dmax = offs.shape[1], n, spiking, theta, dmax
        tau = torch.exp(torch.linspace(math.log(5.0), math.log(100.0), n))
        self.log_rate = nn.Parameter(-torch.log(tau)); self.freq = nn.Parameter(torch.rand(n) * 0.3)
        self.Bre = nn.Parameter(torch.randn(n, d) / math.sqrt(d)); self.Bim = nn.Parameter(torch.randn(n, d) / math.sqrt(d))
        M_ = offs.shape[1] if per_unit else None
        self.wre = nn.Parameter(torch.randn(*((M_, n) if per_unit else (n,))) * w_sd)
        self.wim = nn.Parameter(torch.randn(*((M_, n) if per_unit else (n,))) * w_sd)
        self.q = nn.Parameter(torch.randn(*((M_, d) if per_unit else (d,))) / math.sqrt(d))
        self.c = nn.Parameter(torch.full((M_,) if per_unit else (n_off,), gate_bias))
        self.e_off = nn.Parameter(torch.randn(n_off, d) * 0.5)                            # offset embedding written in
        self.log_td = nn.Parameter(torch.tensor(math.log(10.0)))
        if spiking:
            self.Cre = nn.Parameter(torch.randn(d, n) / math.sqrt(n)); self.Cim = nn.Parameter(torch.randn(d, n) / math.sqrt(n))
            self.y0 = nn.Parameter(torch.zeros(d))

    def forward(self, eb, ei, et, ev, B, G):
        M, n, th = self.M, self.n, self.theta
        o = self.offs[ei]                                                    # (E, M)
        pe, pj = (o >= 0).nonzero(as_tuple=True); oid = o[pe, pj]
        v = ev[pe] + self.e_off[oid]                                         # payload as seen from this receiver
        r = (v * self.q[pj]).sum(-1) + self.c[pj] if self.per_unit else v @ self.q + self.c[oid]
        keep = r.detach() > 0
        pe, pj, oid, v, r = pe[keep], pj[keep], oid[keep], v[keep], r[keep]
        if not self.training:
            self.sent[ei[pe], pj] = True
        sc = self.scale[pj]
        a = et[pe] + (torch.exp(self.log_td) * sc * r).clamp(max=self.dmax * sc)            # delay scales with the unit
        g = torch.ceil(a.detach()); ok = (g <= G - 1) & (g >= 0)
        pe, pj, a, g, v, sc = pe[ok], pj[ok], a[ok], g[ok], v[ok], sc[ok]
        lam0 = torch.complex(-torch.exp(self.log_rate), self.freq)
        lamj = lam0[None, :] / self.scale[:, None]                           # (M, n): rates scale as rho^-m
        Bv = torch.complex(v @ self.Bre.T, v @ self.Bim.T)
        val = torch.exp(lamj[pj] * (g - a)[:, None]) * Bv
        idx = (g.long() * B + eb[pe]) * M + pj
        X = torch.zeros(G * B * M, n, dtype=torch.complex64).index_add(0, idx, val).view(G, B, M, n)
        E1 = torch.exp(lamj); wc = torch.complex(self.wre, self.wim)
        if not self.spiking:                                                    # normalized read z / (count + 1) (§105)
            Xc = torch.zeros(G * B * M, n).index_add(0, idx, torch.exp(lamj.real[pj] * (g - a)[:, None])).view(G, B, M, n).unbind(0)
            c = torch.zeros(B, M, n); Ec = torch.exp(lamj.real)
        eR = torch.exp(-1.0 / (TAU_R * self.scale))                          # (M,)
        z = torch.zeros(B, M, n, dtype=torch.complex64); zs = []; Vs = []
        R = torch.zeros(B, M); Vp = torch.zeros(B, M)
        if self.spiking:
            F = torch.zeros(G, B, M, dtype=torch.bool); FR = torch.zeros(G, B, M); RP = torch.zeros(G, B, M)
        Xk = X.unbind(0)
        for k in range(G):
            z = z * E1 + Xk[k]
            if not self.spiking:
                c = c * Ec + Xc[k]
                Vs.append((wc * z / (c + 1.0)).real.sum(-1)); continue
            V = (wc * z).real.sum(-1)
            zs.append(z)
            with torch.no_grad():
                R.mul_(eR); Vd = V - th * R
                fire = Vd >= th
                frac = ((th - Vp) / (Vd - Vp).clamp(min=1e-6)).clamp(0, 1)
                F[k] = fire; FR[k] = frac
                RP[k] = R * torch.exp((1 - frac) / (TAU_R * self.scale)) * fire
                jump = fire * torch.exp(-(1 - frac) / (TAU_R * self.scale))
                R.add_(jump); Vp = Vd - th * jump
        if not self.spiking:
            return torch.stack(Vs), len(pe) / B
        kk, bb, jj = F.nonzero(as_tuple=True)
        good = kk >= 1; kk, bb, jj = kk[good], bb[good], jj[good]
        frac, Rpre = FR[kk, bb, jj], RP[kk, bb, jj]
        zprev = torch.stack(zs)[kk - 1, bb, jj]
        lj = lamj[jj]
        zT0 = torch.exp(lj * frac[:, None]) * zprev
        V0 = (wc * zT0).real.sum(-1) - th * Rpre
        Vdot = ((wc * lj * zT0).real.sum(-1) + th * Rpre / (TAU_R * self.scale[jj])).detach().clamp(min=0.02)
        s = frac - ((V0 - th) / Vdot).clamp(-1.0, 1.0)
        zT = torch.exp(lj * s[:, None]) * zprev
        y = nn.functional.gelu((zT @ torch.complex(self.Cre, self.Cim).T).real) + self.y0
        return (bb, jj, (kk - 1).float() + s, y), len(pe) / B


def lattice(bands, stride1, win1, S_, stride2, win2, rho):
    """offset tables and scales for layer 1 (bands -> P1 x S) and layer 2 (P1 x S -> P2 x S)."""
    c1 = torch.arange(stride1 // 2, bands, stride1); P1 = len(c1)
    m1 = torch.arange(S_).repeat(P1); p1 = torch.arange(P1).repeat_interleave(S_)             # unit j = p * S + m
    off = torch.arange(bands)[:, None] - c1[p1][None]                                          # (bands, M1)
    # layer-1 offset ids depend on band offset only (all scales see the same offset table)
    offs1 = torch.where(off.abs() <= win1 // 2, off + win1 // 2, torch.full_like(off, -1))
    c2 = torch.arange(0, P1, stride2); P2 = len(c2)
    m2 = torch.arange(S_).repeat(P2); p2 = torch.arange(P2).repeat_interleave(S_)
    dp = p1[:, None] - c2[p2][None]; dm = m1[:, None] - m2[None]
    ok = (dp.abs() <= win2) & (dm.abs() <= 1)
    offs2 = torch.where(ok, (dp + win2) * 3 + (dm + 1), torch.full_like(dp, -1))
    scale1 = rho ** m1.float(); scale2 = rho ** m2.float()
    return offs1, scale1, offs2, scale2


class Net(nn.Module):
    def __init__(self, bands, d, n, stride1, win1, S_, stride2, win2, rho, w_sd):
        super().__init__()
        offs1, sc1, offs2, sc2 = lattice(bands, stride1, win1, S_, stride2, win2, rho)
        self.v0 = nn.Parameter(torch.randn(d) * 0.5); self.vc = nn.Parameter(torch.randn(d) * 0.5)
        self.l1 = EqLayer(offs1, sc1, d, n, True, w_sd[0])
        self.l2 = EqLayer(offs2, sc2, d, n, True, w_sd[1])
        self.ro = EqLayer(torch.zeros(len(sc2), 20, dtype=torch.long), torch.ones(20), d, n, False, 0.3, gate_bias=1.0, per_unit=True)
        self.M = (len(sc1), len(sc2))

    def forward(self, eb, ei, et, cnt, B, G, rstep=4):
        ev = self.v0[None] + cnt[:, None] * self.vc[None]                  # input payload carries no band identity
        (b1, j1, t1, y1), m1 = self.l1(eb, ei, et, ev, B, G)
        (b2, j2, t2, y2), m2 = self.l2(b1, j1, t1, y1, B, G)
        V, m3 = self.ro(b2, j2, t2, y2, B, G)
        return torch.softmax(V[::rstep], -1).mean(0), {"msgs": [m1, m2, m3], "spikes": [len(t1) / B, len(t2) / B]}


def to_events_cnt(items, bands, shift, rng, drop):
    eb, ei, et, ec, ys = [], [], [], [], []
    for n_, (b, t, c, y) in enumerate(items):
        k = rng.random(len(b)) >= drop if drop else np.ones(len(b), bool)
        sh = int(rng.integers(-shift, shift + 1)) if shift else 0
        eb.append(np.full(k.sum(), n_)); ei.append(np.clip(b[k].astype(np.int64) + sh, 0, bands - 1)); et.append(t[k] * 1000.0)
        ec.append(c[k]); ys.append(y)
    et = np.concatenate(et)
    return (torch.from_numpy(np.concatenate(eb)).long(), torch.from_numpy(np.concatenate(ei)).long(),
            torch.from_numpy(et).float(), torch.from_numpy(np.concatenate(ec)).float(), torch.tensor(ys), int(et.max()) + 1)


def evaluate(net, data, a, rng):
    ok = 0; st = {"msgs": np.zeros(3), "spikes": np.zeros(2)}; ne = 0
    with torch.no_grad():
        for i0 in range(0, len(data), a.bs):
            items = data[i0:i0 + a.bs]
            eb, ei, et, ec, y, tmax = to_events_cnt(items, a.bands, 0, rng, 0.0)
            p, info = net(eb, ei, et, ec, len(items), tmax + 2 * 50 * int(a.rho ** (a.scales - 1)) + 60)
            ok += int((p.argmax(1) == y).sum()); ne += len(items)
            for k_ in st:
                st[k_] += np.array(info[k_]) * len(items)
    return ok / len(data), {k: (v / ne).round(0).tolist() for k, v in st.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=140)
    ap.add_argument("--merge", type=float, default=0.002)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--stride1", type=int, default=4)
    ap.add_argument("--win1", type=int, default=24)
    ap.add_argument("--scales", type=int, default=3)
    ap.add_argument("--rho", type=float, default=1.5)
    ap.add_argument("--stride2", type=int, default=2)
    ap.add_argument("--win2", type=int, default=3)
    ap.add_argument("--w_sd", default="0.2,0.03")
    ap.add_argument("--shift", type=int, default=0, help="band-shift augmentation (0: rely on equivariance)")
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval", default="spk", choices=("spk", "test"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    def load(split, part):
        return [(*events(t, u, a.merge), y) for t, u, y in S.utterances(split, a.bands, part) if len(t) > 1]
    tr = load("train", "fit_spk" if a.eval == "spk" else None)
    ev = load("train", "val_spk") if a.eval == "spk" else load("test", None)
    if a.limit:
        tr = tr[:a.limit]; ev = ev[:max(64, a.limit // 4)]
    probe = [tr[i] for i in np.random.default_rng(1).choice(len(tr), min(500, len(tr)), replace=False)]
    net = Net(a.bands, a.d, a.n, a.stride1, a.win1, a.scales, a.stride2, a.win2, a.rho, [float(v) for v in a.w_sd.split(",")])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    nb = math.ceil(len(tr) / a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb, pct_start=0.1)
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()), "units": net.M, "curve": []}
    print(json.dumps({"train": len(tr), "eval": len(ev), "params": res["params"], "units": net.M, "load_s": round(time.time() - t0)}), flush=True)
    for ep in range(a.epochs):
        net.train(); tl = 0.0; perm = rng.permutation(len(tr))
        for i0 in range(0, len(tr), a.bs):
            items = [tr[j] for j in perm[i0:i0 + a.bs]]
            eb, ei, et, ec, y, tmax = to_events_cnt(items, a.bands, a.shift, rng, a.drop)
            p, _ = net(eb, ei, et, ec, len(items), tmax + 2 * 50 * int(a.rho ** (a.scales - 1)) + 60)
            loss = -torch.log(p[torch.arange(len(y)), y] + 1e-8).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tl += float(loss)
        net.eval()
        for l_ in (net.l1, net.l2, net.ro):
            l_.sent.zero_()
        acc, info = evaluate(net, ev, a, rng)
        sending = [round(float(l_.sent[l_.offs >= 0].float().mean()), 3) for l_ in (net.l1, net.l2, net.ro)]
        tr_acc, _ = evaluate(net, probe, a, rng)
        row = {"epoch": ep + 1, "train_loss": round(tl / nb, 4), f"{a.eval}_acc": round(acc, 4), "train_probe_acc": round(tr_acc, 4),
               "msgs_per_utt": info["msgs"], "spikes_per_utt": info["spikes"], "synapses_sending": sending,
               "wall_s": round(time.time() - t0)}
        res["curve"].append(row); print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"eq_S{a.scales}_rho{a.rho:g}_st{a.stride1}_w{a.win1}_sh{a.shift}_bs{a.bs}_{a.eval}_s{a.seed}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
