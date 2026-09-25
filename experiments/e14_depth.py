#!/usr/bin/env python3
"""E14: does counterfactual credit start to matter with more than one hidden layer?

With one hidden layer, counterfactual (near-miss) and fired-only hidden credit tie (E6 r3,
M18). Hypothesis: in deeper networks a near-miss node influences the output only through
downstream nodes its spike would have tipped over threshold, a path fired-only credit cannot
see, so the gap should grow with depth.

A stack of racing hidden layers (groups of 10, k winners, ramp synapses, as E6), an output
race, and the output error sent to every hidden layer through its own fixed random feedback
(direct feedback alignment). Hidden credit per layer:

  crl_fa           fired nodes and near misses (exp(−Δ/σ)), as E6
  crl_fired_only   fired nodes only
  frozen_hidden    hidden layers do not learn

    python experiments/e14_depth.py --depth 2 --variant crl_fa --val 10000
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import (HORIZON, Config, RaceNet, group_race, latency_code, layer_race, mnist,  # noqa: E402
                       race_order, to_events)

OUT = os.path.join(os.path.dirname(__file__), "results", "e14")


class DeepRaceNet:
    _apply_add = RaceNet._apply

    def _apply(self, W, idx, coef, mask, dummy, exists=None):
        if not self.eg:
            return self._apply_add(W, idx, coef, mask, dummy, exists)
        # M28 (THEORY §24): each geometry gets its own update: the evidence mix u = w/ρ moves
        # multiplicatively on the simplex; the urgency ρ = Σw moves additively by the credit's total.
        for bi in range(len(idx)):
            active = np.flatnonzero(coef[bi])
            if not len(active):
                continue
            m = mask[bi, active]
            if exists is not None:
                m = m * exists[np.ix_(active, idx[bi])]
            rho = W[active, :dummy].sum(1, keepdims=True)
            upd = coef[bi, active, None] * m
            # urgency: the credit's additive effect on the node's total drive
            rho_new = np.maximum(rho + upd.sum(1, keepdims=True), 1e-6)
            # evidence mix: multiplicative step on the simplex
            r = self.eg * upd / np.maximum(rho, 1e-9)
            sub = W[np.ix_(active, idx[bi])]
            W[np.ix_(active, idx[bi])] = sub * np.exp(np.clip(r, -5, 5))
            W[active, :dummy] *= rho_new / np.maximum(W[active, :dummy].sum(1, keepdims=True), 1e-9)
            self.work["plasticity"] += int((m > 0).sum()) + len(active)
        W[:, dummy] = 0
    _elig = RaceNet._elig

    def __init__(self, cfg, widths, d_in, k, drive, rng, feedback="dfa", fanin=0, fanin_in=0):
        self.cfg, self.k, self.feedback = cfg, k, feedback
        self.rng, self.window, self.nonneg, self.eg = rng, 0.15, False, 0.0
        self.homeo_mode, self.homeo_rate = "linear", 0.001
        self.usage = [np.full(h, cfg.winners / cfg.group) for h in widths]
        self.info_capacity = False
        self.counts = [np.full((h, k), 1.0) for h in widths]
        self.ycount = [np.full(k, 10.0) for _ in widths]
        self.W, self.th, self.B, self.M, self.R, dims = [], [], [], [], [], [d_in] + list(widths)
        expected = float(drive.sum())
        for l, h in enumerate(widths):
            W = np.zeros((h, dims[l] + 1), np.float32)
            M = None
            f = fanin if l > 0 else fanin_in                       # sparse connectivity (inputs: fanin_in)
            if f:
                M = np.zeros((h, dims[l] + 1), bool)
                live = np.flatnonzero(drive > 0) if l == 0 else np.arange(dims[l])
                for n in range(h):
                    M[n, rng.choice(live, min(f, len(live)), replace=False)] = True
                expected_l = expected * min(f, len(live)) / len(live)
            else:
                expected_l = expected
            mu = 1.0 / (cfg.hid_frac * expected_l)
            W[:, :dims[l]] = rng.normal(mu, mu, (h, dims[l]))
            if M is not None:
                W *= M
            self.W.append(W)
            self.M.append(M)
            # local feedback runs backwards along existing synapses, through fixed random weights
            R = rng.normal(0, 1.0, (h, dims[l])).astype(np.float32) / np.sqrt(max(fanin or dims[l], 1))
            self.R.append(R * (M[:, :dims[l]] if M is not None else 1.0))
            self.th.append(np.ones(h, np.float32))
            expected = h // cfg.group * cfg.winners * 0.5          # charge from the next layer's input spikes
        mu2 = 1.0 / (cfg.init_frac * expected)
        self.Wo = np.zeros((k, dims[-1] + 1), np.float32)
        self.Wo[:, :dims[-1]] = rng.normal(mu2, mu2, (k, dims[-1]))
        self.tho = np.full(k, cfg.theta_out, np.float32)
        for h in widths:
            self.B.append(rng.normal(0, mu2, (k, h)).astype(np.float32))
        self.dims = dims
        self.work = {"synops": 0, "plasticity": 0, "samples": 0}
        self.reach = [[0, 0] for _ in widths]                      # [credited nodes, all nodes] per layer

    def forward(self, t, idx):
        cfg, layers = self.cfg, []
        self.work["samples"] += len(t)
        shadow = cfg.variant == "crl_shadow"
        st_, sidx = t, idx                                      # shadow channel: real + upstream shadow spikes
        for l, W in enumerate(self.W):
            T, over, v_at = layer_race(t, idx, W, self.th[l], "ramp")
            fired, freeze = group_race(T, over, W.shape[0] // cfg.group, cfg.winners)
            v, n_before = v_at(freeze)
            if self.M[l] is None:
                self.work["synops"] += int(n_before.sum())
            else:                                               # only existing synapses carry events
                before = t[:, None, :] <= freeze[:, :, None]
                self.work["synops"] += int((np.transpose(self.M[l][:, idx], (1, 0, 2)) & before).sum())
            L = dict(t=t, idx=idx, fired=fired, freeze=freeze, snap=np.clip((self.th[l] - v) / self.th[l], 0, None))
            if shadow:
                # the shadow compartment is never inhibited: it integrates real and shadow inputs and
                # emits a shadow spike when it would have crossed; only near misses (within the window
                # after the group's decision) count, so closeness is selected by time, not by sorting
                Ts, _, _ = layer_race(st_, sidx, W, self.th[l], "ramp")
                sfire = ~fired & np.isfinite(Ts) & (Ts < freeze + self.window) & (Ts < HORIZON)
                L["shadow_fired"] = sfire
                self.work["shadow_spikes"] = self.work.get("shadow_spikes", 0) + int(sfire.sum())
                st_, sidx = to_events(np.where(fired, freeze, np.where(sfire, Ts, np.inf)))
            layers.append(L)
            t, idx = to_events(np.where(fired, freeze, np.inf))
        T2, over2, v_at2 = layer_race(t, idx, self.Wo, self.tho, "ramp")
        winner = np.where(np.isfinite(T2).any(1), race_order(T2, over2)[:, 0], -1)
        t_dec = np.where(winner >= 0, T2.min(1), HORIZON)
        freeze2 = np.repeat(t_dec[:, None], self.k, 1)
        v2, n2 = v_at2(freeze2)
        self.work["synops"] += int(n2.sum())
        urgent = winner < 0
        if cfg.deadline:
            winner = np.where(urgent, v2.argmax(1), winner)
        snap2 = np.clip((self.tho - v2) / self.tho, 0, None)
        snap2[winner >= 0, winner[winner >= 0]] = 0
        return dict(layers=layers, t2=t, idx2=idx, freeze2=freeze2, snap2=snap2, winner=winner, urgent=urgent)

    def teach(self, st, y):
        cfg = self.cfg
        rows = np.arange(len(y))
        elig2 = np.exp(-st["snap2"] / cfg.sigma)
        rival = st["snap2"].copy()
        rival[rows, y] = np.inf
        update = (st["winner"] != y) | (rival.min(1) < cfg.margin) | st["urgent"]
        s = -elig2 * (elig2 >= 0.05)
        s[rows, y] = 0.0
        if cfg.zero_sum:                                   # credit conserved at the collapse (THEORY §22)
            s /= np.maximum(-s.sum(1, keepdims=True), 1e-9)
        s[rows, y] = 1.0
        s *= update[:, None]
        mask2 = self._elig(st["t2"], st["freeze2"])
        self._apply(self.Wo, st["idx2"], cfg.eta_out * s, mask2, self.Wo.shape[1] - 1)
        if self.nonneg:
            np.maximum(self.Wo, 0, out=self.Wo)
        carried = None                                             # local feedback, from the top layer down
        for l in reversed(range(len(st["layers"]))):
            L = st["layers"][l]
            if cfg.variant != "frozen_hidden":
                if cfg.variant == "crl_drtp":                 # random projection of the label alone
                    delta = np.eye(self.k, dtype=np.float32)[y] @ self.B[l]
                elif self.feedback == "local" and carried is not None:
                    delta = carried @ self.R[l + 1]            # only across synapses that exist
                else:
                    delta = s @ self.B[l]
                if cfg.variant == "crl_fired_only":
                    elig = L["fired"].astype(np.float32)
                elif cfg.variant == "crl_stoch":              # sampled binary near-miss events, no weighting
                    p_near = np.exp(-L["snap"] / cfg.sigma)
                    elig = np.where(L["fired"], 1.0, (self.rng.random(p_near.shape) < p_near)).astype(np.float32)
                elif cfg.variant == "crl_shadow":             # two-channel neuron: fired, or shadow-fired
                    elig = (L["fired"] | L["shadow_fired"]).astype(np.float32)
                elif cfg.variant == "crl_window":             # a hard window: near misses within δ, unweighted
                    elig = (L["fired"] | (L["snap"] < self.window)).astype(np.float32)
                else:
                    elig = np.where(L["fired"], 1.0, np.exp(-L["snap"] / cfg.sigma))
                    elig *= elig >= 0.05
                coef = cfg.eta_hid * delta * elig
                carried = delta * elig                         # credit passes only through eligible nodes
                self.reach[l][0] += int((coef != 0).sum())
                self.reach[l][1] += coef.size
                mask = self._elig(L["t"], L["freeze"])
                self._apply(self.W[l], L["idx"], coef, mask, self.dims[l], self.M[l])
            if self.nonneg:                                    # no inhibitory weights: a monotone network
                np.maximum(self.W[l], 0, out=self.W[l])
            target = cfg.winners / cfg.group
            if self.info_capacity:
                # §25: learnt capacities: a node's target rate grows with the information its firing
                # carries about the label, I(fires; class), from running class-conditional counts
                cnt = self.counts[l]
                cnt *= 0.999
                np.add.at(cnt, (np.arange(cnt.shape[0])[None, :].repeat(len(y), 0), y[:, None].repeat(cnt.shape[0], 1)),
                          L["fired"].astype(np.float64))
                self.ycount[l] = 0.999 * self.ycount[l] + np.bincount(y, minlength=self.k)
                p_y = self.ycount[l] / self.ycount[l].sum()
                p_f_y = np.clip(cnt / np.maximum(self.ycount[l][None, :], 1e-9), 1e-4, 1 - 1e-4)
                p_f = (p_f_y * p_y[None, :]).sum(1, keepdims=True)
                h = lambda q: -(q * np.log(q) + (1 - q) * np.log(1 - q))   # noqa: E731
                info = h(p_f[:, 0]) - (h(p_f_y) * p_y[None, :]).sum(1)
                target = target * np.clip(info / max(info.mean(), 1e-9), 0.25, 2.5)
            if self.homeo_mode == "sinkhorn":
                # M29 (THEORY §25): thresholds are dual potentials of a balanced-usage constraint;
                # the Sinkhorn dual step is a log-ratio of usage to capacity (running usage estimate)
                self.usage[l] += 0.02 * (L["fired"].mean(0) - self.usage[l])
                self.th[l] += self.homeo_rate * np.log(np.maximum(self.usage[l], 1e-3) / target)
            else:
                self.th[l] += cfg.homeo * (L["fired"].mean(0) - target)
            np.maximum(self.th[l], 0.05, out=self.th[l])


def code_diagnostics(net, times):
    """Early-evidence monopoly check (THEORY §18): per hidden layer, the mean absolute
    correlation between nodes' firing across samples (redundancy), and the share of the
    layer's input events integrated before its nodes froze (evidence used)."""
    t, idx = to_events(times[:200])                      # small batch: arrays are samples × nodes × events
    st = net.forward(t, idx)
    out = []
    for L in st["layers"]:
        f = L["fired"].astype(np.float32)
        live = f.std(0) > 0
        if live.sum() > 1:
            c = np.corrcoef(f[:, live].T)
            red = float(np.abs(c[~np.eye(len(c), dtype=bool)]).mean())
        else:
            red = None
        arrived = np.isfinite(L["t"])[:, None, :] & (L["t"][:, None, :] <= L["freeze"][:, :, None])
        used = float(arrived.sum(2).mean() / max(np.isfinite(L["t"]).sum(1).mean(), 1))
        out.append({"redundancy": red, "evidence_used": used, "fire_rate": float(f.mean())})
    return out


def evaluate(net, times, y, batch=250):
    correct = 0
    for i in range(0, len(y), batch):
        t, idx = to_events(times[i:i + batch])
        correct += int((net.forward(t, idx)["winner"] == y[i:i + batch]).sum())
    return correct / len(y)


def main(a):
    cfg = Config(variant=a.variant, winners=3, hid_frac=0.6, eta_out=0.01, eta_hid=0.01, deadline=1, psp="ramp",
                 homeo=0.001, sigma=a.sigma, zero_sum=a.zero_sum, seed=a.seed)
    x, y = mnist("train")
    if a.val:
        xtr, ytr, xte, yte = x[:-a.val], y[:-a.val], x[-a.val:], y[-a.val:]
    else:
        (xtr, ytr), (xte, yte) = (x, y), mnist("test")
    if a.train_limit:
        xtr, ytr = xtr[:a.train_limit], ytr[:a.train_limit]
    ttr, tte = latency_code(xtr), latency_code(xte)
    drive = np.where(np.isfinite(ttr), HORIZON - ttr, 0).mean(0)
    rng = np.random.default_rng(a.seed)
    net = DeepRaceNet(cfg, [a.width] * a.depth, 784, 10, drive, rng, a.feedback, a.fanin, a.fanin_in)
    net.window = a.window
    net.nonneg = bool(a.nonneg)
    net.eg = a.eg
    net.homeo_mode, net.homeo_rate = a.homeo_mode, a.homeo_rate
    net.info_capacity = bool(a.info_capacity)
    if net.nonneg:
        for W in net.W:
            np.maximum(W, 0, out=W)
        np.maximum(net.Wo, 0, out=net.Wo)
    curve = []
    t0 = time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), 32):
            ii = perm[i:i + 32]
            t, idx = to_events(ttr[ii])
            net.teach(net.forward(t, idx), ytr[ii])
        curve.append(evaluate(net, tte, yte))
        print(f"depth {a.depth} {a.variant} epoch {ep + 1}: {curve[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    res_diag = code_diagnostics(net, tte)
    # balance of hidden usage (normalised entropy of how often each node fires), per layer
    res_balance = []
    for L in res_diag:
        pass
    t_b, i_b = to_events(tte[:200])
    for L in net.forward(t_b, i_b)["layers"]:
        u = L["fired"].mean(0)
        q = u / max(u.sum(), 1e-9)
        nz = q[q > 0]
        res_balance.append(float(-(nz * np.log(nz)).sum() / np.log(len(q))))
    res = {"config": vars(a), "curve": curve, "acc": curve[-1], "code": res_diag, "usage_balance": res_balance,
           "credit_reach": [c / n if n else None for c, n in net.reach],
           "synops_per_sample": net.work["synops"] / max(net.work["samples"], 1)}
    os.makedirs(OUT, exist_ok=True)
    extras = "".join(f"_{k}{v}" for k, v in (("fb", a.feedback if a.feedback != "dfa" else ""), ("f", a.fanin or ""),
                                             ("fi", a.fanin_in or ""), ("sg", a.sigma if a.sigma != 0.15 else ""),
                                             ("w", a.window if a.variant in ("crl_shadow", "crl_window") else ""),
                                             ("zs", a.zero_sum or ""), ("nn", a.nonneg or ""), ("eg", a.eg or ""),
                                             ("hm", a.homeo_mode if a.homeo_mode != "linear" else "")) if v != "")
    # every setting that varies is in the name, so runs never overwrite each other
    with open(os.path.join(OUT, f"d{a.depth}_{a.variant}{extras}_{a.tag or 'run'}_s{a.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--variant", default="crl_fa", choices=("crl_fa", "crl_fired_only", "frozen_hidden", "crl_drtp",
                                                             "crl_stoch", "crl_window", "crl_shadow"))
    ap.add_argument("--window", type=float, default=0.15, help="near-miss window (crl_window: in Δ; crl_shadow: time)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--feedback", default="dfa", choices=("dfa", "local"),
                    help="E15: dfa = output error to every layer; local = layer by layer along existing synapses")
    ap.add_argument("--fanin", type=int, default=0, help="E15: hidden-to-hidden fan-in (0 = dense)")
    ap.add_argument("--fanin-in", type=int, default=0, help="input-to-first-hidden fan-in (0 = dense)")
    ap.add_argument("--sigma", type=float, default=0.15, help="near-miss temperature")
    ap.add_argument("--zero-sum", type=int, default=0, help="conserve credit at each collapse")
    ap.add_argument("--nonneg", type=int, default=0, help="clamp all weights to be non-negative (monotone net)")
    ap.add_argument("--homeo-mode", default="linear", choices=("linear", "sinkhorn"))
    ap.add_argument("--info-capacity", type=int, default=0, help="§25: learnt capacities from I(fires; class)")
    ap.add_argument("--homeo-rate", type=float, default=0.001, help="step of the Sinkhorn (log-ratio) threshold update")
    ap.add_argument("--eg", type=float, default=0.0, help="M28: multiplicative simplex updates with this scale (0 = additive)")
    ap.add_argument("--val", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    main(ap.parse_args())
