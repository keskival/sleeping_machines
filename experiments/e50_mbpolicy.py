"""E50: a world-model-based trading policy (model-predictive control on the event network's own rollouts).

World model: the semi-Markov event network of E48 on 6 channels (up small, up large, down small, down large, large buy,
large sell; E47's encoding), learned online by counts; each price channel also learns its mean size in bp. At every price
event (E42's decision points) the policy samples R rollouts of the race forward over the holding horizon H (the network
run forward: sample the next event's time from the piecewise-constant hazard of the current state, its type from the
type detectors, update the state, repeat), giving a distribution of the price change over H. It targets position +1 if
E[change] > c + kappa·sd/sqrt(R_eff), -1 if E[change] < -(c + kappa·sd/...), else keeps the position; a change costs c.
Metric: realized net bp after costs per day (prequential, pilot days), vs flat, buy-and-hold and E42's priced learner.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e17_market import load_day, PILOT  # noqa: E402
from e42_when import decision_points, pnl  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e50")
GAPS = np.array([0.01, 0.05, 0.2, 1.0, 5.0]); NC = 6
EDGES = np.r_[0.0, GAPS, np.inf]


def stream(day, big_q):
    t, p, q, sell = load_day(day)
    dp = decision_points(t, p)
    dlp = 1e4 * np.diff(np.log(p[dp]))
    tt = [t[dp][1:]]; ch = [np.where(dlp > 0, np.where(dlp > 1.5, 1, 0), np.where(dlp < -1.5, 3, 2))]
    mv = [dlp]
    big = np.flatnonzero(q >= big_q)
    tt.append(t[big]); ch.append(np.where(sell[big], 5, 4)); mv.append(np.zeros(len(big)))
    T = np.concatenate(tt) / 1e6; C = np.concatenate(ch); MV = np.concatenate(mv)
    o = np.argsort(T, kind="stable")
    return T[o], C[o], MV[o]


class World:
    def __init__(self):
        nb = len(GAPS) + 1
        self.hN = np.full((NC, NC, nb), 0.5); self.hE = np.full((NC, NC, nb), 1.0)
        self.P = np.ones((NC, NC, nb, NC)); self.size = np.array([1.0, 2.5, -1.0, -2.5, 0.0, 0.0]); self.sn = np.ones(NC)
        self.l1 = self.l2 = None; self.t0 = None

    def learn(self, t, c, mv):
        if self.t0 is not None:
            d = t - self.t0; ctx = (self.l2 if self.l2 is not None else 0, self.l1)
            k = int(np.searchsorted(GAPS, d)); span = np.clip(d - EDGES[:-1], 0, np.diff(EDGES))
            self.hN[ctx][k] += 1; self.hE[ctx] += span; self.P[ctx][k][c] += 1
        if c < 4:
            self.sn[c] += 1; self.size[c] += (mv - self.size[c]) / self.sn[c]
        self.t0 = t; self.l2, self.l1 = self.l1, c

    def rollout(self, H, R, rng):
        """R sampled futures over H seconds from the current state; returns the price change (bp) of each."""
        out = np.zeros(R)
        for r in range(R):
            l2, l1, el, tot = self.l2 if self.l2 is not None else 0, self.l1, 0.0, 0.0
            elapsed = 0.0
            while True:
                h = self.hN[l2, l1] / self.hE[l2, l1]
                # sample the waiting time from the piecewise-constant hazard over the window bank
                u = rng.exponential(); t_ev = None; acc = 0.0
                for k in range(len(h)):
                    span = EDGES[k + 1] - EDGES[k] if np.isfinite(EDGES[k + 1]) else np.inf
                    if acc + h[k] * span >= u:
                        t_ev = EDGES[k] + (u - acc) / h[k]; kk = k; break
                    acc += h[k] * span
                if elapsed + t_ev > H:
                    break
                elapsed += t_ev
                pt = self.P[l2, l1, kk] / self.P[l2, l1, kk].sum()
                c = rng.choice(NC, p=pt)
                tot += self.size[c]; l2, l1 = l1, c
            out[r] = tot
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=2.0)
    ap.add_argument("--H", type=float, default=60.0)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--ndays", type=int, default=7)
    ap.add_argument("--every", type=int, default=5, help="act at every n-th price event (cost of rollouts)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT]); big_q = float(np.quantile(qs, 0.99))
    rng = np.random.default_rng(0); W = World(); rows = []
    for day in PILOT[:a.ndays]:
        T, C, MV = stream(day, big_q)
        price_idx = np.flatnonzero(C < 4)
        pos = 0; P_log = []; pos_log = []; n_dec = 0
        for i, (t, c, mv) in enumerate(zip(T, C, MV)):
            W.learn(t, c, mv)
            if c < 4:
                n_dec += 1
                if n_dec % a.every == 0 and W.t0 is not None:
                    fut = W.rollout(a.H, a.R, rng)
                    m, sd = fut.mean(), fut.std() + 1e-9
                    margin = a.c + a.kappa * sd / np.sqrt(a.R)
                    if m > margin: pos = 1
                    elif m < -margin: pos = -1
                P_log.append(mv); pos_log.append(pos)
        r = np.r_[np.array(P_log[1:]), 0.0]                   # move from each price event to the next
        net, ch = pnl(np.array(pos_log), r, a.c)
        hold, _ = pnl(np.ones(len(r), int), r, a.c)
        row = {"day": str(day), "c": a.c, "net_bp": net, "changes": ch, "buy_hold_bp": hold,
               "time_in_market": float(np.mean(np.array(pos_log) != 0))}
        rows.append(row); print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"mb_c{a.c:g}_H{a.H:g}_k{a.kappa:g}.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
