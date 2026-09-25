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
    _apply = RaceNet._apply
    _elig = RaceNet._elig

    def __init__(self, cfg, widths, d_in, k, drive, rng, feedback="dfa", fanin=0):
        self.cfg, self.k, self.feedback = cfg, k, feedback
        self.W, self.th, self.B, self.M, self.R, dims = [], [], [], [], [], [d_in] + list(widths)
        expected = float(drive.sum())
        for l, h in enumerate(widths):
            W = np.zeros((h, dims[l] + 1), np.float32)
            M = None
            if fanin and l > 0:                                    # sparse hidden-to-hidden connectivity
                M = np.zeros((h, dims[l] + 1), bool)
                for n in range(h):
                    M[n, rng.choice(dims[l], min(fanin, dims[l]), replace=False)] = True
                expected_l = expected * min(fanin, dims[l]) / dims[l]
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
        for l, W in enumerate(self.W):
            T, over, v_at = layer_race(t, idx, W, self.th[l], "ramp")
            fired, freeze = group_race(T, over, W.shape[0] // cfg.group, cfg.winners)
            v, n_before = v_at(freeze)
            self.work["synops"] += int(n_before.sum())
            layers.append(dict(t=t, idx=idx, fired=fired, freeze=freeze,
                               snap=np.clip((self.th[l] - v) / self.th[l], 0, None)))
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
        s[rows, y] = 1.0
        s *= update[:, None]
        mask2 = self._elig(st["t2"], st["freeze2"])
        self._apply(self.Wo, st["idx2"], cfg.eta_out * s, mask2, self.Wo.shape[1] - 1)
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
                else:
                    elig = np.where(L["fired"], 1.0, np.exp(-L["snap"] / cfg.sigma))
                    elig *= elig >= 0.05
                coef = cfg.eta_hid * delta * elig
                carried = delta * elig                         # credit passes only through eligible nodes
                self.reach[l][0] += int((coef != 0).sum())
                self.reach[l][1] += coef.size
                mask = self._elig(L["t"], L["freeze"])
                self._apply(self.W[l], L["idx"], coef, mask, self.dims[l], self.M[l])
            self.th[l] += cfg.homeo * (L["fired"].mean(0) - cfg.winners / cfg.group)
            np.maximum(self.th[l], 0.05, out=self.th[l])


def code_diagnostics(net, times):
    """Early-evidence monopoly check (THEORY §18): per hidden layer, the mean absolute
    correlation between nodes' firing across samples (redundancy), and the share of the
    layer's input events integrated before its nodes froze (evidence used)."""
    t, idx = to_events(times)
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
                 homeo=0.001, sigma=a.sigma, seed=a.seed)
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
    net = DeepRaceNet(cfg, [a.width] * a.depth, 784, 10, drive, rng, a.feedback, a.fanin)
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
    res_diag = code_diagnostics(net, tte[:1000])
    res = {"config": vars(a), "curve": curve, "acc": curve[-1], "code": res_diag,
           "credit_reach": [c / n if n else None for c, n in net.reach],
           "synops_per_sample": net.work["synops"] / max(net.work["samples"], 1)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"d{a.depth}_{a.variant}_{a.feedback}_f{a.fanin}_sg{a.sigma}_{a.tag or 'run'}"
                                f"_s{a.seed}.json" if a.feedback != "dfa" or a.fanin or a.sigma != 0.15 else
                                f"d{a.depth}_{a.variant}_{a.tag or 'run'}_s{a.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--variant", default="crl_fa", choices=("crl_fa", "crl_fired_only", "frozen_hidden", "crl_drtp"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--feedback", default="dfa", choices=("dfa", "local"),
                    help="E15: dfa = output error to every layer; local = layer by layer along existing synapses")
    ap.add_argument("--fanin", type=int, default=0, help="E15: hidden-to-hidden fan-in (0 = dense)")
    ap.add_argument("--sigma", type=float, default=0.15, help="near-miss temperature")
    ap.add_argument("--val", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    main(ap.parse_args())
