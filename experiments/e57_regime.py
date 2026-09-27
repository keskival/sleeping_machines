"""E57: a native world model with slow regime state: the semi-Markov event network (E44/E48) plus leaky event counters.

Regime nodes are leaky counters updated only at events, c <- c * exp(-dt / tau) + 1 (tau = 5 s and 60 s): their level
c / tau is the recent event rate at that scale, quantized into 5 buckets (log-spaced, fixed); an optional flow counter
integrates price moves up (+1) minus down (-1) with tau = 10 s, quantized into 3 buckets (sell pressure, neutral, buy
pressure). The state held between events is (last two types, regime buckets); in each gap window of the bank the hazard
and the next-type distribution are counts, backed off to the plain semi-Markov estimate (last two types only) with prior
strength m (hierarchical Dirichlet smoothing), so rare regime cells borrow from the coarse model. Exact likelihood as in
E44; cost per event: a few counter updates and two table reads per level.
Protocol as E52: frozen after days 1-4, scored on day 5 (validation), and frozen after days 1-5, scored on days 6-7.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e17_market import load_day, PILOT  # noqa: E402
from e44_tpp import day_events, NT, SemiMarkov  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e57")
GAPS = SemiMarkov.GAPS; EDGES = np.r_[0.0, GAPS, np.inf]; NB = len(GAPS) + 1


def set_bank(fine):
    """window bank: the event network's 6 windows, or the THP's fine bank (12 log-spaced edges, 1 ms - 30 s)."""
    global GAPS, EDGES, NB
    GAPS = np.geomspace(0.001, 30.0, 12) if fine else SemiMarkov.GAPS
    EDGES = np.r_[0.0, GAPS, np.inf]; NB = len(GAPS) + 1
RATE_EDGES = np.array([0.5, 2.0, 8.0, 32.0])            # events per second
FLOW_EDGES = np.array([-1.5, 1.5])


class RegimeWorld:
    def __init__(self, taus, flow, m):
        self.taus, self.flow, self.m = list(taus), flow, m
        self.nr = [len(RATE_EDGES) + 1] * len(self.taus) + ([len(FLOW_EDGES) + 1] if flow else [])
        R = int(np.prod(self.nr)) if self.nr else 1
        self.cN = np.full((NT, NT, NB), 0.5); self.cE = np.full((NT, NT, NB), 1.0); self.cP = np.ones((NT, NT, NB, NT))
        self.fN = np.zeros((NT, NT, R, NB)); self.fE = np.zeros((NT, NT, R, NB)); self.fP = np.zeros((NT, NT, R, NB, NT))
        self.c = np.zeros(len(self.taus)); self.fl = 0.0
        self.t0 = None; self.l1 = self.l2 = None; self.learn = True

    def regime(self):
        idx = [int(np.searchsorted(RATE_EDGES, self.c[i] / tau)) for i, tau in enumerate(self.taus)]
        if self.flow:
            idx.append(int(np.searchsorted(FLOW_EDGES, self.fl)))
        return int(np.ravel_multi_index(idx, self.nr)) if idx else 0

    def step(self, t, y):
        if self.t0 is None:
            self.t0, self.l1 = t, y; self._count(0.0, y)
            return None
        d = max(t - self.t0, 1e-6)
        ctx = (self.l2 if self.l2 is not None else 0, self.l1); r = self.regime()
        k = int(np.searchsorted(GAPS, d)); span = np.clip(d - EDGES[:-1], 0, np.diff(EDGES))
        hc = self.cN[ctx] / self.cE[ctx]                                       # coarse hazard per bucket
        h = (self.fN[ctx][r] + self.m * hc) / (self.fE[ctx][r] + self.m)       # backed-off (m: pseudo-exposure, s)
        pc = self.cP[ctx][k] / self.cP[ctx][k].sum()
        p = (self.fP[ctx][r][k] + self.m * pc) / (self.fP[ctx][r][k].sum() + self.m)
        ll = float(np.log(h[k] * p[y]) - (h * span).sum())
        if self.learn:
            self.cN[ctx][k] += 1; self.cE[ctx] += span; self.cP[ctx][k][y] += 1
            self.fN[ctx][r][k] += 1; self.fE[ctx][r] += span; self.fP[ctx][r][k][y] += 1
        self._count(d, y)
        self.t0 = t; self.l2, self.l1 = self.l1, y
        return ll

    def _count(self, d, y):                                                    # regime nodes: leaky counters (events only)
        for i, tau in enumerate(self.taus):
            self.c[i] = self.c[i] * np.exp(-d / tau) + 1.0
        if self.flow:
            self.fl = self.fl * np.exp(-d / 10.0) + (1.0 if y == 0 else -1.0 if y == 1 else 0.0)


def run(days, train, evals, taus, flow, m):
    w = RegimeWorld(taus, flow, m)
    for di in train:
        for t, y in zip(*days[di]):
            w.step(t, y)
    w.learn = False
    out = {}
    for di in evals:
        ll = [w.step(t, y) for t, y in zip(*days[di])]
        out[f"day{di + 1}"] = float(np.mean([v for v in ll if v is not None]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, default=5.0)
    ap.add_argument("--fine", type=int, default=0)
    a = ap.parse_args()
    set_bank(a.fine)
    os.makedirs(OUT, exist_ok=True)
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT]); big_q = float(np.quantile(qs, 0.99))
    days = [day_events(d, big_q) for d in PILOT]
    rows = []
    for name, taus, flow in (("semi-Markov (no regime)", [], False), ("+ rate 5 s", [5.0], False),
                             ("+ rate 60 s", [60.0], False), ("+ rates 5 s, 60 s", [5.0, 60.0], False),
                             ("+ rates 5 s, 60 s + flow", [5.0, 60.0], True), ("+ rate 5 s + flow", [5.0], True)):
        val = run(days, range(4), [4], taus, flow, a.m)
        test = run(days, range(5), [5, 6], taus, flow, a.m)
        row = {"model": name, "m": a.m, "fine": a.fine, **{f"val_{k}": v for k, v in val.items()}, **{f"test_{k}": v for k, v in test.items()}}
        rows.append(row); print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"regime_m{a.m:g}{'_fine' if a.fine else ''}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
