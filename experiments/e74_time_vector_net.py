"""E74: a time-vector network (THEORY §105): events carry small vectors, and time and content compute together.

Event: (t, v), v in R^d. Synapse sender -> unit j:
  content -> time   the message is sent only if its score r = q_j . v + c_ij > 0, and arrives after a delay tau_j * r
                    (a matching payload arrives later and fresher: with the receiver's decay this is softmax weighting,
                    §105 lemma; a non-matching payload is never sent, so messaging is sparse by content)
  time -> content   the receiver's state z_j in C^n flows between arrivals (z <- e^{Lambda_j dt} z: decay and rotation by
                    elapsed time) and jumps by B v at each arrival: every payload is weighted and phase-rotated by its
                    age after its content-dependent delay
  content -> time   the unit fires when its potential Re<w_j, z_j(t)> crosses the threshold (reset by subtraction)
  time -> content   it emits the snapshot y = GELU(Re C_j z_j(T)) + e_j at its firing time T
All four couplings are differentiable: spike times by the implicit-function step T = T0 - (V(T0) - theta) / V'(T0) (§104d),
payloads through z(T) (including its dependence on T), arrival times through the content-dependent delays.
Layer 1 units listen to tonotopic windows of bands; layer 2 units to a random quarter of layer 1; a non-spiking readout layer
(20 class units) is trained by -log mean_t softmax_c V_c(t). Training simulates on a 1 ms grid (arrivals keep their exact
sub-bin phase); inference is event-driven. Reported: messages sent and spikes per utterance per layer.
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
OUT = os.path.join(os.path.dirname(__file__), "results", "e74")
TAU_R = 20.0                                                         # reset trace time constant (ms)


class TVLayer(nn.Module):
    def __init__(self, n_in, M, d_in, d_out, n, dmax, mask=None, spiking=True, w_sd=0.1, theta=1.0, gate_bias=0.5,
                 cdelay=1, gate=1, snapshot=1, causal=False, tau_range=(5.0, 100.0), td0=10.0, normalize=False):
        super().__init__()
        self.normalize = normalize                  # non-spiking read z / (count channel + 1): the §105 normalizer
        self.cdelay, self.gate, self.snapshot, self.causal = cdelay, gate, snapshot, causal
        self.vdot_min = 0.02                       # floor on dV/dt at a crossing: bounds 1/V' for grazing spikes
        self.register_buffer("sent", torch.zeros(n_in, M, dtype=torch.bool), persistent=False)   # §107(e) diagnostic
        self.M, self.n, self.dmax, self.spiking, self.theta = M, n, dmax, spiking, theta
        tau = torch.exp(torch.empty(M, n).uniform_(math.log(tau_range[0]), math.log(tau_range[1])))
        self.log_rate = nn.Parameter(-torch.log(tau)); self.freq = nn.Parameter(torch.rand(M, n) * 0.3)   # rad per time unit
        self.Bre = nn.Parameter(torch.randn(n, d_in) / math.sqrt(d_in)); self.Bim = nn.Parameter(torch.randn(n, d_in) / math.sqrt(d_in))
        self.wre = nn.Parameter(torch.randn(M, n) * w_sd); self.wim = nn.Parameter(torch.randn(M, n) * w_sd)
        self.q = nn.Parameter(torch.randn(M, d_in) / math.sqrt(d_in)); self.c = nn.Parameter(torch.full((n_in, M), gate_bias))
        self.log_td = nn.Parameter(torch.full((M,), math.log(td0)))        # delay per unit of score (time units)
        if spiking:
            self.Cre = nn.Parameter(torch.randn(M, d_out, n) / math.sqrt(n)); self.Cim = nn.Parameter(torch.randn(M, d_out, n) / math.sqrt(n))
            self.emb = nn.Parameter(torch.randn(M, d_out) * 0.5)
        self.register_buffer("mask", mask)

    def lam(self):
        return torch.complex(-torch.exp(self.log_rate), self.freq)

    def forward(self, eb, ei, et, ev, B, G, force_route=None, return_routes=False):
        M, n, th = self.M, self.n, self.theta
        E = len(et)
        if self.mask is not None:
            pe, pj = self.mask[ei].nonzero(as_tuple=True)
        else:
            pe = torch.arange(E).repeat_interleave(M); pj = torch.arange(M).repeat(E)
        r = (self.q[pj] * ev[pe]).sum(-1) + self.c[ei[pe], pj]                  # content score of each message
        route_info = None
        if return_routes:
            route_info = {"event_index": pe.detach(), "receiver": pj.detach(),
                          "score": r.detach()}
        if self.gate:
            keep = r.detach() > 0                                              # non-matching content: no message
            if force_route is not None:
                fe, fj = map(int, force_route)
                forced = (pe == fe) & (pj == fj)
                if not bool(forced.any()):
                    raise ValueError("forced route is not in the candidate connectivity mask")
                if bool((forced & keep).any()):
                    raise ValueError("force_route must name a currently closed route")
                keep = keep | forced
            pe, pj, r = pe[keep], pj[keep], r[keep]
        if not self.training:
            self.sent[ei[pe], pj] = True
        rd = r if self.cdelay else nn.functional.softplus(self.c[ei[pe], pj])    # ablation: static per-synapse delay
        a = et[pe] + (torch.exp(self.log_td[pj]) * rd.clamp(min=0)).clamp(max=self.dmax)   # arrival time
        g = torch.ceil(a.detach()); ok = (g <= G - 1) & (g >= 0)
        pe, pj, a, g = pe[ok], pj[ok], a[ok], g[ok]
        lam = self.lam()
        Bv = torch.complex(ev @ self.Bre.T, ev @ self.Bim.T)                  # (E, n) payload written into states
        val = torch.exp(lam[pj] * (g - a)[:, None]) * Bv[pe]                   # exact sub-bin flow to the grid point
        idx = (g.long() * B + eb[pe]) * M + pj
        X = torch.zeros(G * B * M, n, dtype=torch.complex64).index_add(0, idx, val).view(G, B, M, n)
        E1 = torch.exp(lam); wc = torch.complex(self.wre, self.wim)
        if self.normalize and not self.spiking:                                # count channel: same decay, weight 1 per message
            Xc = torch.zeros(G * B * M, n).index_add(0, idx, torch.exp(lam.real[pj] * (g - a)[:, None])).view(G, B, M, n).unbind(0)
            c = torch.zeros(B, M, n); Ec = torch.exp(lam.real)
        z = torch.zeros(B, M, n, dtype=torch.complex64); zs = []; Vs = []
        R = torch.zeros(B, M); Vp = torch.zeros(B, M); eR = math.exp(-1 / TAU_R)
        if self.spiking:
            F = torch.zeros(G, B, M, dtype=torch.bool); FR = torch.zeros(G, B, M); RP = torch.zeros(G, B, M)
        Xk = X.unbind(0)                                                       # one backward op instead of G slices
        for k in range(G):
            z = z * E1 + Xk[k]
            if self.normalize and not self.spiking:
                c = c * Ec + Xc[k]
                Vs.append((wc * z / (c + 1.0)).real.sum(-1)); continue
            V = (wc * z).real.sum(-1)
            if not self.spiking:
                Vs.append(V); continue
            zs.append(z)
            with torch.no_grad():
                R.mul_(eR); Vd = V - th * R
                fire = Vd >= th
                frac = ((th - Vp) / (Vd - Vp).clamp(min=1e-6)).clamp(0, 1)
                F[k] = fire; FR[k] = frac; RP[k] = R * torch.exp((1 - frac) / TAU_R) * fire
                jump = fire * torch.exp(-(1 - frac) / TAU_R)
                R.add_(jump); Vp = Vd - th * jump
        if not self.spiking:
            output = (torch.stack(Vs), len(pe) / B)
            return (*output, route_info) if return_routes else output
        kk, bb, jj = F.nonzero(as_tuple=True)
        good = kk >= 1; kk, bb, jj = kk[good], bb[good], jj[good]
        frac, Rpre = FR[kk, bb, jj], RP[kk, bb, jj]
        Z = torch.stack(zs); zprev = Z[kk - 1, bb, jj]                          # state at the grid point before the crossing
        lj, wj = lam[jj], wc[jj]
        zT0 = torch.exp(lj * frac[:, None]) * zprev
        V0 = (wj * zT0).real.sum(-1) - th * Rpre
        Vdot = ((wj * lj * zT0).real.sum(-1) + th * Rpre / TAU_R).detach().clamp(min=self.vdot_min)
        s = frac - ((V0 - th) / Vdot).clamp(-1.0, 1.0)                         # refined time since grid point k-1
        if self.causal:                                                        # never earlier than the detecting step
            s = s.clamp(1e-3, 1.0)
        zT = torch.exp(lj * s[:, None]) * zprev                                # state at the firing time
        C = torch.complex(self.Cre, self.Cim)[jj]                              # (S, d_out, n)
        y = nn.functional.gelu((C @ zT[..., None]).squeeze(-1).real) * self.snapshot + self.emb[jj]
        output = ((bb, jj, (kk - 1).float() + s, y), len(pe) / B)
        return (*output, route_info) if return_routes else output


class Net(nn.Module):
    def __init__(self, bands, d, n, M1, M2, window, fan2, dmax, w_sd, seed=0, cdelay=1, gate=1, snapshot=1):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(bands, d)
        centers = (torch.arange(M1) + 0.5) * bands / M1
        m1 = (torch.arange(bands)[:, None] - centers[None]).abs() <= window / 2
        m2 = torch.rand(M1, M2, generator=gen) < fan2
        ab = dict(cdelay=cdelay, gate=gate, snapshot=snapshot)
        self.l1 = TVLayer(bands, M1, d, d, n, dmax, m1, True, w_sd[0], **ab)
        self.l2 = TVLayer(M1, M2, d, d, n, dmax, m2, True, w_sd[1], **ab)
        self.ro = TVLayer(M2, 20, d, d, n, dmax, None, False, 0.3, gate_bias=1.0, cdelay=cdelay, gate=gate, normalize=True)

    def forward(self, eb, ei, et, B, G, rstep=4):
        ev = self.emb(ei)
        (b1, j1, t1, y1), m1 = self.l1(eb, ei, et, ev, B, G)
        (b2, j2, t2, y2), m2 = self.l2(b1, j1, t1, y1, B, G)
        V, m3 = self.ro(b2, j2, t2, y2, B, G)                                   # (G, B, 20)
        p = torch.softmax(V[::rstep], -1).mean(0)
        return p, {"msgs": [m1, m2, m3], "spikes": [len(t1) / B, len(t2) / B]}


def to_events(items, bands, shift, rng, drop):
    eb, ei, et, ys = [], [], [], []
    for n_, (b, t, c, y) in enumerate(items):
        k = rng.random(len(b)) >= drop if drop else np.ones(len(b), bool)
        sh = int(rng.integers(-shift, shift + 1)) if shift else 0
        eb.append(np.full(k.sum(), n_)); ei.append(np.clip(b[k].astype(np.int64) + sh, 0, bands - 1)); et.append(t[k] * 1000.0)
        ys.append(y)
    et = np.concatenate(et)
    return (torch.from_numpy(np.concatenate(eb)).long(), torch.from_numpy(np.concatenate(ei)).long(),
            torch.from_numpy(et).float(), torch.tensor(ys), int(et.max()) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=140)
    ap.add_argument("--merge", type=float, default=0.002)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--M1", type=int, default=64)
    ap.add_argument("--M2", type=int, default=64)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--fan2", type=float, default=0.25)
    ap.add_argument("--dmax", type=float, default=50.0)
    ap.add_argument("--w_sd", default="0.1,0.1")
    ap.add_argument("--shift", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval", default="spk", choices=("spk", "test"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cdelay", type=int, default=1, help="0: static per-synapse delays (no content -> time)")
    ap.add_argument("--gate", type=int, default=1, help="0: every message is sent")
    ap.add_argument("--snapshot", type=int, default=1, help="0: emitted payload = unit identity only")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    def load(split, part):
        return [(*events(t, u, a.merge), y) for t, u, y in S.utterances(split, a.bands, part) if len(t) > 1]
    tr = load("train", "fit_spk" if a.eval == "spk" else None)
    ev = load("train", "val_spk") if a.eval == "spk" else load("test", None)
    if a.limit:
        tr = tr[:a.limit]; ev = ev[:max(64, a.limit // 4)]
    net = Net(a.bands, a.d, a.n, a.M1, a.M2, a.window, a.fan2, a.dmax, [float(v) for v in a.w_sd.split(",")], a.seed,
              a.cdelay, a.gate, a.snapshot)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    nb = math.ceil(len(tr) / a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb, pct_start=0.1)
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()), "curve": []}
    print(json.dumps({"train": len(tr), "eval": len(ev), "params": res["params"], "load_s": round(time.time() - t0)}), flush=True)
    for ep in range(a.epochs):
        net.train(); tl = 0.0; perm = rng.permutation(len(tr))
        for i0 in range(0, len(tr), a.bs):
            items = [tr[j] for j in perm[i0:i0 + a.bs]]
            eb, ei, et, y, tmax = to_events(items, a.bands, a.shift, rng, a.drop)
            p, _ = net(eb, ei, et, len(items), tmax + 2 * int(a.dmax) + 60)
            loss = -torch.log(p[torch.arange(len(y)), y] + 1e-8).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tl += float(loss)
        net.eval(); ok = 0; st = {"msgs": np.zeros(3), "spikes": np.zeros(2)}; ne = 0
        for l_ in (net.l1, net.l2, net.ro):
            l_.sent.zero_()
        with torch.no_grad():
            for i0 in range(0, len(ev), a.bs):
                items = ev[i0:i0 + a.bs]
                eb, ei, et, y, tmax = to_events(items, a.bands, 0, rng, 0.0)
                p, info = net(eb, ei, et, len(items), tmax + 2 * int(a.dmax) + 60)
                ok += int((p.argmax(1) == y).sum()); ne += len(items)
                for k_ in st:
                    st[k_] += np.array(info[k_]) * len(items)
        row = {"epoch": ep + 1, "train_loss": round(tl / nb, 4), f"{a.eval}_acc": round(ok / len(ev), 4),
               "msgs_per_utt": (st["msgs"] / ne).round(0).tolist(), "spikes_per_utt": (st["spikes"] / ne).round(0).tolist(),
               "synapses_sending": [round(float(l_.sent[l_.mask].float().mean() if l_.mask is not None else l_.sent.float().mean()), 3)
                                    for l_ in (net.l1, net.l2, net.ro)],
               "wall_s": round(time.time() - t0)}
        res["curve"].append(row); print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"tv_d{a.d}_n{a.n}_M{a.M1}-{a.M2}_w{a.window}_cd{a.cdelay}g{a.gate}sn{a.snapshot}_bs{a.bs}_{a.eval}_s{a.seed}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
