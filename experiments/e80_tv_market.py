"""E80: the market stream with a time-vector network (THEORY §105), as a world model and as a trading signal.

World model. BTC spot events (price up / down, large buy / sell; E44) are vector events: payload = type embedding + the
E52 gap features (+ optionally the BTC perpetual's and ETH's last price move and its age, computed strictly before the
event: other markets as more vector content). Time-vector units step once per event; each step flows the state by the
real time elapsed before the event (seconds), and an event arrives just before the end of its step, so the state read at
step j + 1 has seen events 0..j and nothing later. Content-gated, content-delayed messages (delays in events), threshold
firing with exact spike-time gradients and snapshot payloads (two spiking layers), then non-spiking state units and E52's
hazard head (piecewise-constant rate per type and gap window). Protocol of E52 / E69: --mode val trains on days 1-4 and
scores day 5 every epoch; --mode test trains on days 1-5 and scores days 6-7 once; --mode confirm trains on the 7 pilot
days and scores the 21 confirmatory days (preregistered: run only if the pilot audit shows an edge above 2 bp fees).

Edge audit (E55's executable round trips: buy at the last buyer-initiated price, sell at the last seller-initiated price
h seconds later, or the reverse). At each BTC price event the model's probability that the next price move is up, from
its own hazards. Policy: go long / short when the probability is beyond a threshold. Horizon h and threshold are chosen on
day 5 predictions of the val-mode model (best epoch) at each fee; read once on days 6-7 (test) or the confirmatory days.
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
from e17_market import load_day, PILOT  # noqa: E402
from e44_tpp import day_events, NT  # noqa: E402
import e52_thp as T52  # noqa: E402
import e55_edge as E55  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e80")
TAU_R = 20.0                                                           # reset trace time constant (s)
H_SET = (1.0, 5.0, 30.0, 120.0)


class TVLayerW(nn.Module):
    """E74's time-vector layer with a per-step flow: step k advances real time by w[k, b] seconds (the gap before the
    event of that step). Delays are in steps (events); sub-step arrival phases use the step's real duration."""
    def __init__(self, n_in, M, d_in, d_out, n, dmax, mask=None, spiking=True, w_sd=0.1, theta=1.0, gate_bias=0.5):
        super().__init__()
        self.M, self.n, self.dmax, self.spiking, self.theta = M, n, dmax, spiking, theta
        tau = torch.exp(torch.empty(M, n).uniform_(math.log(0.5), math.log(600.0)))       # 0.5 s .. 10 min
        self.log_rate = nn.Parameter(-torch.log(tau)); self.freq = nn.Parameter(torch.rand(M, n) * 0.5 / tau)
        self.Bre = nn.Parameter(torch.randn(n, d_in) / math.sqrt(d_in)); self.Bim = nn.Parameter(torch.randn(n, d_in) / math.sqrt(d_in))
        self.wre = nn.Parameter(torch.randn(M, n) * w_sd); self.wim = nn.Parameter(torch.randn(M, n) * w_sd)
        self.q = nn.Parameter(torch.randn(M, d_in) / math.sqrt(d_in)); self.c = nn.Parameter(torch.full((n_in, M), gate_bias))
        self.log_td = nn.Parameter(torch.full((M,), math.log(1.0)))            # delay per unit of score (events)
        if spiking:
            self.Cre = nn.Parameter(torch.randn(M, d_out, n) / math.sqrt(n)); self.Cim = nn.Parameter(torch.randn(M, d_out, n) / math.sqrt(n))
            self.emb = nn.Parameter(torch.randn(M, d_out) * 0.5)
        self.register_buffer("mask", mask)

    def forward(self, eb, ei, et, ev, B, G, w):
        M, n, th = self.M, self.n, self.theta
        E = len(et)
        if self.mask is not None:
            pe, pj = self.mask[ei].nonzero(as_tuple=True)
        else:
            pe = torch.arange(E).repeat_interleave(M); pj = torch.arange(M).repeat(E)
        r = (self.q[pj] * ev[pe]).sum(-1) + self.c[ei[pe], pj]
        keep = r.detach() > 0
        pe, pj, r = pe[keep], pj[keep], r[keep]
        a = et[pe] + (torch.exp(self.log_td[pj]) * r).clamp(max=self.dmax)
        g = torch.ceil(a.detach()); ok = (g <= G - 1) & (g >= 1)
        pe, pj, a, g = pe[ok], pj[ok], a[ok], g[ok]
        lam = torch.complex(-torch.exp(self.log_rate), self.freq)
        Bv = torch.complex(ev @ self.Bre.T, ev @ self.Bim.T)
        wg = w[g.long(), eb[pe]]                                                # real duration of the arrival's step
        val = torch.exp(lam[pj] * ((g - a) * wg)[:, None]) * Bv[pe]
        idx = (g.long() * B + eb[pe]) * M + pj
        X = torch.zeros(G * B * M, n, dtype=torch.complex64).index_add(0, idx, val).view(G, B, M, n)
        wc = torch.complex(self.wre, self.wim)
        z = torch.zeros(B, M, n, dtype=torch.complex64); zs = []; Vs = []
        R = torch.zeros(B, M); Vp = torch.zeros(B, M)
        if self.spiking:
            F = torch.zeros(G, B, M, dtype=torch.bool); FR = torch.zeros(G, B, M); RP = torch.zeros(G, B, M)
        Xk = X.unbind(0)
        for k in range(G):
            wk = w[k][:, None, None]
            z = z * torch.exp(lam[None] * wk) + Xk[k]
            V = (wc * z).real.sum(-1)
            if not self.spiking:
                Vs.append(V); continue
            zs.append(z)
            with torch.no_grad():
                eR = torch.exp(-w[k] / TAU_R)[:, None]
                R.mul_(eR); Vd = V - th * R
                fire = Vd >= th
                frac = ((th - Vp) / (Vd - Vp).clamp(min=1e-6)).clamp(0, 1)
                F[k] = fire; FR[k] = frac; RP[k] = R * torch.exp((1 - frac) * w[k][:, None] / TAU_R) * fire
                jump = fire * torch.exp(-(1 - frac) * w[k][:, None] / TAU_R)
                R.add_(jump); Vp = Vd - th * jump
        if not self.spiking:
            return torch.stack(Vs), len(pe) / B
        kk, bb, jj = F.nonzero(as_tuple=True)
        good = kk >= 1; kk, bb, jj = kk[good], bb[good], jj[good]
        frac, Rpre = FR[kk, bb, jj], RP[kk, bb, jj]
        zprev = torch.stack(zs)[kk - 1, bb, jj]
        lj, wj = lam[jj], wc[jj]; ws = w[kk, bb]
        zT0 = torch.exp(lj * (frac * ws)[:, None]) * zprev
        V0 = (wj * zT0).real.sum(-1) - th * Rpre
        Vdot = ((wj * lj * zT0).real.sum(-1) * ws + th * Rpre * ws / TAU_R).detach().clamp(min=0.02)     # per step
        s = (frac - ((V0 - th) / Vdot).clamp(-1.0, 1.0)).clamp(1e-3, 1.0)                                # causal
        zT = torch.exp(lj * (s * ws)[:, None]) * zprev
        C = torch.complex(self.Cre, self.Cim)[jj]
        y = nn.functional.gelu((C @ zT[..., None]).squeeze(-1).real) + self.emb[jj]
        return (bb, jj, (kk - 1).float() + s, y), len(pe) / B


class TVPP(nn.Module):
    def __init__(self, nf, nb, d=16, n=8, M1=64, M2=64, Ms=64, fan2=0.25, dmax=16.0, w_sd=(0.05, 0.05), seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(NT, d); self.tf = nn.Linear(nf, d)
        self.l1 = TVLayerW(NT, M1, d, d, n, dmax, None, True, w_sd[0])
        self.l2 = TVLayerW(M1, M2, d, d, n, dmax, torch.rand(M1, M2, generator=gen) < fan2, True, w_sd[1])
        self.st = TVLayerW(M2, Ms, d, d, n, dmax, None, False, 0.3, gate_bias=1.0)
        self.norm = nn.LayerNorm(Ms); self.head = nn.Linear(Ms + d, nb * NT); self.nb = nb

    def forward(self, y, x):
        """y (B, L) types, x (B, L, nf) features (x[..., 0] = log(gap + 1e-4) / 5): hazards (B, L, nb, NT) after each event."""
        B, L = y.shape; G = L + 1
        gap = (torch.exp(5 * x[..., 0]) - 1e-4).clamp(min=0)                   # seconds before each event
        w = torch.zeros(G, B); w[1:] = gap.T                                    # step j + 1 carries event j
        v = self.emb(y) + self.tf(x)
        eb = torch.arange(B).repeat_interleave(L); ei = y.reshape(-1); et = torch.arange(L).float().repeat(B) + 1 - 1e-3
        ev = v.reshape(B * L, -1)
        (b1, j1, t1, y1), m1 = self.l1(eb, ei, et, ev, B, G, w)
        (b2, j2, t2, y2), m2 = self.l2(b1, j1, t1, y1, B, G, w)
        V, m3 = self.st(b2, j2, t2, y2, B, G, w)
        h = torch.cat([self.norm(V[1:].permute(1, 0, 2)), v], -1)
        self.work = {"msgs_per_event": [m1 / L, m2 / L, m3 / L], "spikes_per_event": [len(t1) / (B * L), len(t2) / (B * L)]}
        return nn.functional.softplus(self.head(h)).view(B, L, self.nb, NT) + 1e-6

    ll = T52.THP.ll


def cross_features(day, T):
    """the BTC perpetual's and ETH's last price move before each event: direction and log age (strictly earlier)."""
    at = (T * 1e6).astype(np.int64) - 1; cols = []
    for sym in ("perp", "eth"):
        ts, ps, _ = E55.load(sym, day)
        dr, tm = E55.last_move(ts, ps, at)
        age = np.clip((at - tm) / 1e6, 1e-3, 1e4)
        cols += [dr.astype(np.float32), (np.log(age) / 5).astype(np.float32)]
    return np.stack(cols, 1)


def magnitude_features(day, big_q, T, Y):
    """transaction magnitudes as vector content of each event (user's proposal, 2026-09-28), from the trade tape up to and
    including the event's own trade: log size of that trade, log buy and sell volume and log trade count since the
    previous event (sizes relative to the day-1..7 99% quantile scale)."""
    from e42_when import decision_points
    t, p, q, sell = load_day(day)
    dp = decision_points(t, p)[1:]; big = np.flatnonzero(q >= big_q)
    idx = np.concatenate([dp, big]); Tt = np.concatenate([t[dp], t[big]]) / 1e6
    o = np.argsort(Tt, kind="stable"); idx = idx[o]
    assert np.allclose(Tt[o], T)                                            # same events, same order as day_events
    cb = np.r_[0.0, np.cumsum(np.where(sell, 0.0, q))]; cs = np.r_[0.0, np.cumsum(np.where(sell, q, 0.0))]
    prev = np.r_[-1, np.maximum.accumulate(idx)[:-1]]                     # latest trade already covered
    idx_ = np.maximum(idx, prev)
    scale = big_q
    f = [np.log1p(q[idx] / scale), np.log1p((cb[idx_ + 1] - cb[prev + 1]) / scale), np.log1p((cs[idx_ + 1] - cs[prev + 1]) / scale),
         np.log1p(idx_ - prev) / 3]
    return np.stack(f, 1).astype(np.float32)


def encode_day(day, big_q, gaps, cross, mag=0):
    T, Y = day_events(day, big_q)
    enc = list(T52.encode(T, Y, gaps))
    if cross:
        enc[1] = torch.cat([enc[1], torch.from_numpy(cross_features(day, T))], 1)
    if mag:
        enc[1] = torch.cat([enc[1], torch.from_numpy(magnitude_features(day, big_q, T, Y))], 1)
    return T, Y, enc


def p_up_day(net, enc, L, gaps):
    """probability that the next price move is up, after every event (from the model's hazards; windows of L as E52)."""
    y, x, b, sp, yn = enc; n = len(y); edges = np.r_[0.0, gaps, np.inf]; P = np.full(n, np.nan)
    with torch.no_grad():
        done = 0
        for s in T52.windows(n, L, L // 2):
            lam = net(y[None, s:s + L], x[None, s:s + L])[0].numpy()        # (L, nb, NT)
            up, dn = lam[..., 0], lam[..., 1]; tot = up + dn                       # price-move rates per gap window
            width = np.diff(edges); width[-1] = 1e9
            Hs = np.concatenate([np.zeros((len(lam), 1)), np.cumsum(tot[:, :-1] * width[None, :-1], 1)], 1)
            leave = np.exp(-Hs) * (1 - np.exp(-tot * width[None]))                 # first price move falls in window b
            P_up = (leave * up / np.maximum(tot, 1e-12)).sum(1) / np.maximum(leave.sum(1), 1e-12)
            lo = max(done, s + L // 2) - s if s > 0 else 0; hi = min(s + L, n) - s
            P[s + lo:s + hi] = P_up[lo:hi]; done = s + hi
    return P


def returns_at(day, T, Y):
    """E55's executable round-trip returns (bp) at each BTC price event, for every horizon."""
    t, p, sell = E55.load("btc", day)
    idx = np.arange(len(t))
    ask_i = np.maximum.accumulate(np.where(~sell, idx, -1)); bid_i = np.maximum.accumulate(np.where(sell, idx, -1))
    ev = np.flatnonzero(Y <= 1); te = (T[ev] * 1e6).astype(np.int64)
    k = np.searchsorted(t, te, side="right") - 1
    out = {"ev": ev}
    for h in H_SET:
        j = np.searchsorted(t, te + int(h * 1e6), side="right") - 1
        valid = (k >= 0) & (j > k) & (ask_i[np.maximum(k, 0)] >= 0) & (bid_i[np.maximum(k, 0)] >= 0) & (ask_i[j] >= 0) & (bid_i[j] >= 0)
        a0, b0 = p[ask_i[np.maximum(k, 0)]], p[bid_i[np.maximum(k, 0)]]; a1, b1 = p[ask_i[j]], p[bid_i[j]]
        out[f"long{h}"] = np.where(valid, (b1 - a0) / a0 * 1e4, np.nan)
        out[f"short{h}"] = np.where(valid, (b0 - a1) / b0 * 1e4, np.nan)
    return out


def choose_policy(P, R, fees, qs=(0.01, 0.05, 0.2), min_trades=50):
    """on selection-day predictions: horizon and confidence threshold maximizing mean net return at each fee."""
    pu = P[R["ev"]]; conf = np.abs(pu - 0.5); best = {}
    for fee in fees:
        cand = []
        for h in H_SET:
            for q in qs:
                thr = float(np.nanquantile(conf, 1 - q))
                sel = (conf >= thr) & np.isfinite(pu)
                ret = np.where(pu > 0.5, R[f"long{h}"], R[f"short{h}"])[sel]; ret = ret[np.isfinite(ret)]
                if len(ret) >= min_trades:
                    cand.append((float(ret.mean()) - fee, h, thr, len(ret)))
        best[fee] = max(cand) if cand else None
    return best


def apply_policy(P, R, h, thr, fee):
    pu = P[R["ev"]]; conf = np.abs(pu - 0.5); sel = (conf >= thr) & np.isfinite(pu)
    ret = np.where(pu > 0.5, R[f"long{h}"], R[f"short{h}"])[sel]; ret = ret[np.isfinite(ret)]
    return {"trades": int(len(ret)), "net_bp": float(ret.mean() - fee) if len(ret) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="val", choices=("val", "test", "confirm"))
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--fine", type=int, default=1)
    ap.add_argument("--cross", type=int, default=0)
    ap.add_argument("--mag", type=int, default=0, help="1: transaction magnitudes in the event vectors")
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fees", default="0,2,5")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(0); rng = np.random.default_rng(0)
    fees = [float(f) for f in a.fees.split(",")]
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT]); big_q = float(np.quantile(qs, 0.99))
    gaps = T52.bank(a.fine)
    if a.mode == "confirm":
        from e17_market import CONF
        train_days, eval_days = list(PILOT), list(CONF)
    else:
        ntr = 4 if a.mode == "val" else 5
        train_days, eval_days = PILOT[:ntr], ([PILOT[4]] if a.mode == "val" else PILOT[5:7])
    tr = [encode_day(d, big_q, gaps, a.cross, a.mag) for d in train_days]; ev = [encode_day(d, big_q, gaps, a.cross, a.mag) for d in eval_days]
    chunks = [(di, s) for di in range(len(tr)) for s in T52.windows(len(tr[di][2][0]) - 1, a.L, a.L)]
    net = TVPP(tr[0][2][1].shape[1], len(gaps) + 1, a.d, 8, a.M, a.M, a.M)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    tag = f"{a.mode}_L{a.L}_f{a.fine}_x{a.cross}_m{a.mag}_d{a.d}_M{a.M}_e{a.epochs}"
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()), "epochs": []}
    best = (-1e9, None, 0)
    for ep in range(a.epochs):
        net.train(); order = rng.permutation(len(chunks))
        for i in range(0, len(order), a.batch):
            parts = [[e[s:s + a.L] for e in tr[di][2]] for di, s in (chunks[k] for k in order[i:i + a.batch])]
            y, x, b, sp, yn = (torch.stack([p[j] for p in parts]) for j in range(5))
            loss = -net.ll(y, x, b, sp, yn).mean(); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval(); row = {"epoch": ep + 1, "wall_s": round(time.time() - t0, 1)}
        if a.mode == "val" or ep == a.epochs - 1:
            scores = [T52.score_day(net, e[2], a.L)[0] for e in ev]
            row.update({f"eval_day{i}": s for i, s in enumerate(scores)}); row["work"] = net.work
            if a.mode == "val" and scores[0] > best[0]:
                best = (scores[0], {k: v.clone() for k, v in net.state_dict().items()}, ep + 1)
        res["epochs"].append(row); print(json.dumps(row), flush=True)
    if a.mode == "val":                                                       # day-5 predictions of the best epoch
        net.load_state_dict(best[1]); res["best_epoch"] = best[2]; res["best_day5"] = best[0]
        T, Y, enc = ev[0]; P = p_up_day(net, enc, a.L, gaps); R = returns_at(eval_days[0], T, Y)
        pol = choose_policy(P, R, fees)
        res["policy"] = {str(f): (None if v is None else {"net_bp_day5": v[0], "h": v[1], "thr": v[2], "trades": v[3]}) for f, v in pol.items()}
    else:                                                                     # read once, with the policy chosen on day 5
        vtag = f"val_L{a.L}_f{a.fine}_x{a.cross}_m{a.mag}_d{a.d}_M{a.M}_e30"
        pol = json.load(open(os.path.join(OUT, f"tv_market_{vtag}.json")))["policy"]
        res["audit"] = {}
        for di, d in enumerate(eval_days):
            T, Y, enc = ev[di]; P = p_up_day(net, enc, a.L, gaps); R = returns_at(d, T, Y)
            for f in fees:
                v = pol[str(f)]
                if v is not None:
                    res["audit"].setdefault(str(f), []).append(apply_policy(P, R, v["h"], v["thr"], f))
    print(json.dumps({k: res[k] for k in res if k in ("best_epoch", "best_day5", "policy", "audit")}), flush=True)
    with open(os.path.join(OUT, f"tv_market_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
