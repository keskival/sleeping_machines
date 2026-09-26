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
        self.time_sigma = 0.1
        self.group_conserve = False
        self.pivot_top = False
        self.share_jac = False
        self.causal, self.noncausal = False, []
        self.center_credit = False
        self.gauge = False
        self.cm = [[] for _ in widths]
        self.pm = [[] for _ in widths]                 # §29 diagnostic: share of credit along the Perron direction
        self.self_sigma, self.sig_l = False, [cfg.sigma for _ in widths]
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
            L = dict(t=t, idx=idx, fired=fired, freeze=freeze, snap=np.clip((self.th[l] - v) / self.th[l], 0, None),
                     T=np.where(fired, freeze, T))
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
        return dict(layers=layers, t2=t, idx2=idx, freeze2=freeze2, snap2=snap2, winner=winner, urgent=urgent,
                    t_dec=t_dec)

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
                elif cfg.variant == "crl_pivot" and carried is not None and self.pivot_top:
                    delta = s @ self.B[l]                      # deeper layers: random feedback (pivot at the top)
                elif cfg.variant == "crl_pivot":
                    # pivotal credit (THEORY §26): the first-order jump of the decision. A spike (real or
                    # projected) matters only if it arrives before its consumer decides; its effect is carried
                    # back through the real forward weights (reciprocal synapses), not a random projection.
                    def jac(Wm):                                # ∂T_consumer/∂t_input = w/A: the evidence share (§22.1, §24)
                        if not self.share_jac:
                            return Wm
                        return Wm / np.maximum(np.abs(Wm).sum(1, keepdims=True), 1e-9)
                    if self.causal and carried is not None:
                        # exact causal Jacobian (§27): ∂T_j/∂t_i = w_ji/A_j · [t_i < T_j], per sample and edge;
                        # input spikes that reach j after j crossed are in j's woven past and carry no credit
                        Wn = jac(self.W[l + 1][:, :self.dims[l + 1]])
                        Tj = st["layers"][l + 1]["T"]                          # consumers' (projected) crossings
                        causal = L["T"][:, None, :] < Tj[:, :, None]          # (b, consumers, inputs)
                        full = np.einsum("bj,ji->bi", carried, Wn)
                        delta = np.einsum("bj,bji->bi", carried, Wn[None, :, :] * causal)
                        self.noncausal.append(float(np.abs(full - delta).sum() / max(np.abs(full).sum(), 1e-12)))
                        deadline = np.full((len(delta), 1), np.inf)
                    elif carried is None:
                        delta = s @ jac(self.Wo[:, :self.dims[l + 1]])
                        deadline = st["t_dec"][:, None]
                    else:
                        Wn = self.W[l + 1][:, :self.dims[l + 1]]
                        delta = carried @ jac(Wn)
                        deadline = st["layers"][l + 1]["freeze"].max(1, keepdims=True)
                    # arriving before the consumer decides is itself a boundary: a late spike gets the credit it
                    # would have had in time, weighted by its time residue (how much earlier it had to be)
                    if np.isfinite(deadline).all():                # (the causal path already gates per edge)
                        late = np.clip(np.where(np.isfinite(L["T"]), L["T"], 10.0) - deadline, 0, None)
                        delta = delta * np.exp(-late / self.time_sigma)
                    ref = np.sqrt((s @ self.B[l]) ** 2).mean() if l < len(self.B) else 1.0
                    delta = delta * (ref / max(np.sqrt((delta ** 2).mean()), 1e-12))   # DFA-comparable scale
                elif self.feedback == "local" and carried is not None:
                    delta = carried @ self.R[l + 1]            # only across synapses that exist
                else:
                    delta = s @ self.B[l]
                sig = cfg.sigma
                if self.self_sigma:
                    # §28: the race's natural temperature is its own runner-up scale (extreme-value theory):
                    # each layer tracks the mean residue of the closest loser per group and uses it as σ
                    d_ = np.where(L["fired"], np.inf, L["snap"]).reshape(len(y), -1, cfg.group).min(2)
                    d_ = d_[np.isfinite(d_)]
                    if d_.size:
                        # the closest loser's gap is the k-th Gumbel spacing, mean σ/k (§28): mode 2 undoes the 1/k
                        scale = cfg.winners if self.self_sigma == 2 else 1
                        self.sig_l[l] += 0.05 * (scale * float(d_.mean()) - self.sig_l[l])
                    sig = self.sig_l[l]
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
                    elig = np.where(L["fired"], 1.0, np.exp(-L["snap"] / sig))
                    elig *= elig >= 0.05
                credit = delta * elig
                e_ = elig > 0
                n_ = np.maximum(e_.sum(1), 1)
                mu_ = (credit * e_).sum(1) / n_
                self.cm[l].append(float((mu_ ** 2).sum() / max(((credit ** 2) * e_).sum(1).__truediv__(n_).sum(), 1e-30)))
                # §29: through real (mostly positive) weights the credit's dominant mode is the backward
                # operator's Perron direction, v_i = Σ_j [consumer j eligible] W_ji (node i's outgoing weight)
                pv = None
                if cfg.variant == "crl_pivot" and not (self.pivot_top and carried is not None):
                    if carried is None:
                        pv = np.broadcast_to(self.Wo[:, :self.dims[l + 1]].sum(0), credit.shape)
                    else:
                        pv = (carried != 0).astype(np.float32) @ self.W[l + 1][:, :self.dims[l + 1]]
                    pe = pv * e_
                    self.pm[l].append(float(np.mean((credit * pe).sum(1) ** 2 / np.maximum(
                        (pe ** 2).sum(1) * ((credit * e_) ** 2).sum(1), 1e-30))))
                if self.center_credit:
                    # §27: the credit's layer mean is an urgency (activity) signal, owned by the prices
                    # (homeostasis); the weights get only its zero-mean part, the evidence signal.
                    # Mode 2 (§29) removes the Perron direction instead of the plain mean.
                    if self.center_credit == 2 and pv is not None:
                        pe = pv * e_
                        c_ = (credit * pe).sum(1, keepdims=True) / np.maximum((pe ** 2).sum(1, keepdims=True), 1e-30)
                        credit = (credit - c_ * pe) * e_
                    else:
                        m_ = (credit * e_).sum(1, keepdims=True) / np.maximum(e_.sum(1, keepdims=True), 1)
                        credit = (credit - m_) * e_
                if self.group_conserve:
                    # §22.3 applied to hidden races: each group is a collapse, so its credit sums to zero:
                    # pushing one member earlier means pushing its rivals in the group later
                    b_, h_ = credit.shape
                    g = credit.reshape(b_, h_ // cfg.group, cfg.group)
                    e_ = (elig.reshape(b_, h_ // cfg.group, cfg.group) > 0)
                    mean = (g * e_).sum(2, keepdims=True) / np.maximum(e_.sum(2, keepdims=True), 1)
                    credit = ((g - mean) * e_).reshape(b_, h_)
                coef = cfg.eta_hid * credit
                carried = credit                               # credit passes only through eligible nodes
                self.reach[l][0] += int((coef != 0).sum())
                self.reach[l][1] += coef.size
                mask = self._elig(L["t"], L["freeze"])
                if self.gauge:
                    rho0 = self.W[l][:, :self.dims[l]].sum(1)
                self._apply(self.W[l], L["idx"], coef, mask, self.dims[l], self.M[l])
                if self.gauge:
                    # §30 gauge fixing: (θ, w) → (αθ, αw) is an exact symmetry, so urgency θ/ρ is one degree of
                    # freedom that credit and prices would both steer. Credit keeps only the evidence mix w/ρ;
                    # the urgency is left to the prices (thresholds)
                    rho1 = self.W[l][:, :self.dims[l]].sum(1)
                    ok = (rho0 > 1e-6) & (rho1 > 1e-6)
                    self.W[l][ok, :self.dims[l]] *= (rho0[ok] / rho1[ok])[:, None]
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


def certify(net, times, y, n=1000, eps_list=(0.001, 0.003, 0.01, 0.03), seed=0):
    """§33 timing-jitter certificate. With non-negative weights every crossing time is a monotone,
    shift-equivariant (topical) function of its input times, hence 1-Lipschitz in the sup norm; the
    decision can change only if some race's order at a boundary changes. Certified radius per sample:
    ε* = min over races of (half the gap between the k-th and (k+1)-th crossing, distance of those
    crossings to the horizon) and, at the output, half the winner's margin. Checked by jitter."""
    times, y = times[:n], y[:n]
    k, G = net.cfg.winners, net.cfg.group
    np_err = np.seterr(invalid="ignore")                        # inf − inf in branches np.where discards
    radius, pred, base_fired = [], [], []
    for i in range(0, len(y), 250):
        t, idx = to_events(times[i:i + 250])
        st = net.forward(t, idx)
        b = len(t)
        r = np.full(b, np.inf)
        for L in st["layers"]:
            T = np.where(np.isfinite(L["T"]) & (L["T"] < HORIZON), L["T"], np.inf)
            srt = np.sort(T.reshape(b, -1, G), 2)[:, :, :k + 1]
            gap = np.where(np.isfinite(srt[:, :, k - 1]), (srt[:, :, k] - srt[:, :, k - 1]) / 2, np.inf)
            hz = np.abs(np.where(np.isfinite(srt), srt, np.inf) - HORIZON).min(2)
            r = np.minimum(r, np.minimum(gap, hz).min(1))
        T2, _, _ = layer_race(st["t2"], st["idx2"], net.Wo, net.tho, "ramp")
        T2 = np.where(np.isfinite(T2) & (T2 < HORIZON), T2, np.inf)
        s2 = np.sort(T2, 1)
        r = np.minimum(r, np.minimum((s2[:, 1] - s2[:, 0]) / 2, np.abs(s2[:, 0] - HORIZON)))
        r[st["urgent"]] = 0.0
        radius.append(r)
        pred.append(st["winner"])
        base_fired.append([L["fired"] for L in st["layers"]])
    radius, pred = np.concatenate(radius), np.concatenate(pred)
    rng = np.random.default_rng(seed)
    out = {"radius_quantiles": [float(q) for q in np.quantile(radius, [0.1, 0.25, 0.5, 0.75, 0.9])],
           "acc": float((pred == y).mean()), "jitter": []}
    for eps in eps_list:
        jt = times + np.where(np.isfinite(times), rng.uniform(-eps, eps, times.shape), 0)
        pj, diff = [], np.zeros(len(net.W))
        for bi, i in enumerate(range(0, len(y), 250)):
            stj = net.forward(*to_events(jt[i:i + 250]))
            pj.append(stj["winner"])
            for l, L in enumerate(stj["layers"]):                  # §41: fraction of groups whose winners changed
                b_ = len(L["fired"])
                ch = (L["fired"] != base_fired[bi][l]).reshape(b_, -1, G).any(2)
                diff[l] += ch.sum()
        pj = np.concatenate(pj)
        rho = [float(d / (len(y) * (W_.shape[0] // G))) for d, W_ in zip(diff, net.W)]
        cert = radius > eps
        flip = pj != pred
        out["jitter"].append({"eps": eps, "certified": float(cert.mean()), "flips": float(flip.mean()),
                              "flips_among_certified": int((flip & cert).sum()),
                              "acc_jittered": float((pj == y).mean()), "rho_per_layer": rho})
    np.seterr(**np_err)
    return out


def speed_accuracy(net, times, y, n=1000, steps=100):
    """§38: when should the output race weave? Absolute rule: decide at the first time any output
    potential reaches λ·θ (λ = 1 is the network's own race, up to the time grid). Relative (MSPRT) rule:
    decide when the leader's share softmax(v/θ/τ) reaches 1 − ε, i.e. when committing to it costs
    less than −log(1 − ε) (§35.1). Evaluation only: the trained network is unchanged."""
    times, y = times[:n], y[:n]
    grid = np.linspace(0, HORIZON, steps + 1)[1:]
    V, Y = [], []
    for i in range(0, len(y), 250):
        st = net.forward(*to_events(times[i:i + 250]))
        _, _, vat = layer_race(st["t2"], st["idx2"], net.Wo, net.tho, "ramp")
        b = len(st["winner"])
        V.append(np.stack([vat(np.full((b, net.k), g))[0] / net.tho for g in grid]))   # (steps, b, k)
        Y.append(y[i:i + 250])
    V, Y = np.concatenate(V, 1), np.concatenate(Y)

    def decide(stop):                                   # stop: (steps, b) bool
        first = np.where(stop.any(0), stop.argmax(0), steps - 1)
        win = V[first, np.arange(V.shape[1])].argmax(1)
        return float((win == Y).mean()), float(grid[first].mean())

    out = {"absolute": [], "relative": []}
    for lam in (0.3, 0.5, 0.7, 1.0, 1.3):
        acc, t = decide(V.max(2) >= lam)
        out["absolute"].append({"lambda": lam, "acc": acc, "t": t})
    for tau in (0.05, 0.1, 0.2):
        z = V / tau
        p = np.exp(z - z.max(2, keepdims=True))
        p /= p.sum(2, keepdims=True)
        for eps in (0.3, 0.1, 0.03, 0.01):
            acc, t = decide(p.max(2) >= 1 - eps)
            out["relative"].append({"tau": tau, "eps": eps, "acc": acc, "t": t})
    return out


def main(a):
    cfg = Config(variant=a.variant, winners=a.winners, hid_frac=0.6, eta_out=0.01, eta_hid=0.01, deadline=1, psp="ramp",
                 homeo=a.homeo, sigma=a.sigma, zero_sum=a.zero_sum, seed=a.seed)
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
    widths = [int(w) for w in a.widths.split(",")] if a.widths else [a.width] * a.depth
    assert len(widths) == a.depth and all(w % 10 == 0 for w in widths), "widths: one multiple of 10 per layer"
    net = DeepRaceNet(cfg, widths, 784, 10, drive, rng, a.feedback, a.fanin, a.fanin_in)
    net.window = a.window
    net.nonneg = bool(a.nonneg)
    net.eg = a.eg
    net.homeo_mode, net.homeo_rate = a.homeo_mode, a.homeo_rate
    net.info_capacity = bool(a.info_capacity)
    net.group_conserve = bool(a.group_conserve)
    net.pivot_top = bool(a.pivot_top)
    net.share_jac = bool(a.share_jac)
    net.causal = bool(a.causal)
    net.center_credit = a.center_credit
    q0 = [float(np.mean(net.th[l] ** 2 + (net.W[l][:, :net.dims[l]] ** 2).sum(1))) for l in range(len(net.W))]
    net.gauge = bool(a.gauge)
    net.self_sigma = a.self_sigma
    if net.nonneg:
        for W in net.W:
            np.maximum(W, 0, out=W)
        np.maximum(net.Wo, 0, out=net.Wo)
    curve = []
    t0 = time.time()
    for ep in range(a.epochs):
        if a.anneal:                                            # continuation from a wide tree to the race
            frac = ep / max(a.epochs - 1, 1)
            net.cfg = type(cfg)(**{**vars(cfg), "sigma": float(np.exp((1 - frac) * np.log(a.anneal)
                                                                    + frac * np.log(a.sigma)))})
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
           "synops_per_sample": net.work["synops"] / max(net.work["samples"], 1),
           "noncausal_credit_share": float(np.mean(net.noncausal)) if net.noncausal else None,
           "sigma_per_layer": [float(v) for v in net.sig_l],
           "common_mode_share": [float(np.mean(c)) if c else None for c in net.cm],
           "perron_share": [float(np.mean(c)) if c else None for c in net.pm],
           "noether_q": [[q, float(np.mean(net.th[l] ** 2 + (net.W[l][:, :net.dims[l]] ** 2).sum(1)))]   # §30.4
                         for l, q in enumerate(q0)],
           "f_eff": [float(np.mean(1 / np.maximum(((np.abs(W_[:, :d_]) / np.maximum(np.abs(W_[:, :d_]).sum(1, keepdims=True),
                     1e-12)) ** 2).sum(1), 1e-12))) for W_, d_ in zip(net.W, net.dims)]}   # §30: evidence-share participation
    def dobrushin(W_, d_):
        # §31: contraction of zero-sum timing credit (and of forward time contrast) through one layer's
        # evidence-share kernel P = |w|/Σ|w|: max and mean total-variation distance between consumers' rows
        P = np.abs(W_[:, :d_]).astype(np.float64)
        P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
        tv = np.array([0.5 * np.abs(P[j + 1:] - P[j]).sum(1).max(initial=0) for j in range(len(P) - 1)])
        tm = np.array([0.5 * np.abs(P[j + 1:] - P[j]).sum(1).mean() for j in range(len(P) - 1)])
        return [float(tv.max()), float(tm.mean())]
    res["dobrushin"] = [dobrushin(W_, d_) for W_, d_ in zip(net.W, net.dims)]
    if a.speed:
        res["speed_accuracy"] = speed_accuracy(net, tte, yte)
        print("speed_accuracy:", json.dumps(res["speed_accuracy"]), flush=True)
    if a.certify:
        res["certificate"] = certify(net, tte, yte)
        res["train_acc_1k"] = evaluate(net, ttr[:1000], ytr[:1000])     # §40: generalisation gap
        print("certificate:", json.dumps(res["certificate"]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    extras = "".join(f"_{k}{v}" for k, v in (("fb", a.feedback if a.feedback != "dfa" else ""), ("f", a.fanin or ""),
                                             ("ws", a.widths.replace(",", "-")), ("k", a.winners if a.winners != 3 else ""),
                                             ("fi", a.fanin_in or ""), ("sg", a.sigma if a.sigma != 0.15 else ""),
                                             ("w", a.window if a.variant in ("crl_shadow", "crl_window") else ""),
                                             ("zs", a.zero_sum or ""), ("gc", a.group_conserve or ""), ("pt", a.pivot_top or ""), ("sj", a.share_jac or ""), ("ca", a.causal or ""), ("cc", a.center_credit or ""), ("gf", a.gauge or ""), ("ss", a.self_sigma or ""), ("ho", a.homeo if a.homeo != 0.001 else ""), ("nn", a.nonneg or ""), ("eg", a.eg or ""),
                                             ("hm", a.homeo_mode if a.homeo_mode != "linear" else "")) if v != "")
    # every setting that varies is in the name, so runs never overwrite each other
    with open(os.path.join(OUT, f"d{a.depth}_{a.variant}{extras}_{a.tag or 'run'}_s{a.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--variant", default="crl_fa", choices=("crl_fa", "crl_fired_only", "frozen_hidden", "crl_drtp",
                                                             "crl_stoch", "crl_window", "crl_shadow", "crl_pivot"))
    ap.add_argument("--anneal", type=float, default=0.0, help="§21.7: start σ at this value, anneal to --sigma")
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
    ap.add_argument("--self-sigma", type=int, default=0, help="§28: per-layer σ from the closest-loser residue; 1 raw mean, 2 k × mean (EVT-corrected)")
    ap.add_argument("--widths", default="", help="§37: per-layer widths, e.g. 800,400,200 (overrides --width)")
    ap.add_argument("--winners", type=int, default=3, help="§37: winners k per group of 10")
    ap.add_argument("--speed", type=int, default=0, help="§38: speed–accuracy of absolute vs relative (MSPRT) stopping")
    ap.add_argument("--certify", type=int, default=0, help="§33: timing-jitter certificate and jitter check")
    ap.add_argument("--gauge", type=int, default=0, help="§30: credit may not change a node's total weight (urgency left to prices)")
    ap.add_argument("--center-credit", type=int, default=0, help="§27/§29: 1 remove the layer mean of credit, 2 remove its Perron direction; activity left to prices")
    ap.add_argument("--homeo", type=float, default=0.001, help="homeostasis (price) step")
    ap.add_argument("--causal", type=int, default=0, help="§27: exact per-edge causal mask between hidden layers")
    ap.add_argument("--share-jac", type=int, default=0, help="pivotal credit through evidence shares w/A (exact Jacobian)")
    ap.add_argument("--pivot-top", type=int, default=0, help="pivotal credit only for the top hidden layer")
    ap.add_argument("--group-conserve", type=int, default=0, help="§22.3 in hidden races: zero-sum credit per group")
    ap.add_argument("--info-capacity", type=int, default=0, help="§25: learnt capacities from I(fires; class)")
    ap.add_argument("--homeo-rate", type=float, default=0.001, help="step of the Sinkhorn (log-ratio) threshold update")
    ap.add_argument("--eg", type=float, default=0.0, help="M28: multiplicative simplex updates with this scale (0 = additive)")
    ap.add_argument("--val", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    main(ap.parse_args())
