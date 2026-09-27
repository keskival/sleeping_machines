"""E58: a factorized native world model: the semi-Markov event network times multiplicative regime factors.

Hazard of type k in gap window b:   lambda_k(b) = base[l2, l1, b, k] * prod_f phi[f, bucket_f, b, k]
  base   counts (events / exposure per window, per type) given the last two types (the E44/E48 event network)
  phi    one factor per regime feature and bucket, per window and type; regime features are leaky event counters
         updated only at events (c <- c * exp(-dt / tau) + 1): per-type rates (4 types) at tau = 2 s and 30 s and the
         total rate at 300 s, and order flow (up minus down moves, tau = 10 s), each quantized into 4 buckets
Learning, at events only (online Poisson regression by exponentiated gradient, i.e. multiplicative credit):
  after a gap with exposure span[b] and an event of type y in window b*: for every active factor
      phi[., ., b, k] *= exp(eta * (1[b = b*, k = y] - lambda_k(b) * span[b]))     (clipped, per window touched)
  base counts are updated as in E44.
Exact likelihood: log lambda_y(b*) - sum_b sum_k lambda_k(b) span[b]. Protocol as E52/E57.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e17_market import load_day, PILOT  # noqa: E402
from e44_tpp import day_events, NT, SemiMarkov  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e58")
GAPS = SemiMarkov.GAPS; EDGES = np.r_[0.0, GAPS, np.inf]; NB = len(GAPS) + 1


class FactorWorld:
    def __init__(self, eta, feats, bins=4):
        self.eta, self.feats, self.bins = eta, feats, bins
        self.N = np.full((NT, NT, NB, NT), 0.5 / NT); self.E = np.full((NT, NT, NB), 1.0)
        self.F = len(feats); self.logphi = np.zeros((self.F, bins, NB, NT))
        self.c = np.zeros(self.F); self.edges = [None] * self.F; self.hist = [[] for _ in range(self.F)]
        self.t0 = None; self.l1 = self.l2 = None; self.learn = True; self.n = 0

    def _buckets(self):
        out = []
        for f in range(self.F):
            v = self.c[f] if self.feats[f][0] == "flow" else np.log1p(self.c[f])
            if self.edges[f] is None:                        # quantile edges from the first 5,000 events (online)
                out.append(0); continue
            out.append(int(np.searchsorted(self.edges[f], v)))
        return out

    def step(self, t, y):
        if self.t0 is None:
            self.t0, self.l1 = t, y; self._count(0.0, y)
            return None
        d = max(t - self.t0, 1e-6); ctx = (self.l2 if self.l2 is not None else 0, self.l1)
        k = int(np.searchsorted(GAPS, d)); span = np.clip(d - EDGES[:-1], 0, np.diff(EDGES))
        bk = self._buckets()
        lphi = sum(self.logphi[f, bk[f]] for f in range(self.F)) if self.F else 0.0      # (NB, NT)
        lam = self.N[ctx] / self.E[ctx][:, None] * np.exp(lphi)                            # (NB, NT)
        ll = float(np.log(lam[k, y]) - (lam * span[:, None]).sum())
        if self.learn:
            self.N[ctx][k][y] += 1.0; self.E[ctx] += span
            touched = span > 0
            grad = -lam * span[:, None]; grad[k, y] += 1.0
            for f in range(self.F):
                self.logphi[f, bk[f]][touched] += self.eta * np.clip(grad[touched], -3, 3)
        self._count(d, y)
        self.t0 = t; self.l2, self.l1 = self.l1, y
        return ll

    def _count(self, d, y):
        for f, (kind, tau, typ) in enumerate(self.feats):
            if kind == "flow":
                self.c[f] = self.c[f] * np.exp(-d / tau) + (1.0 if y == 0 else -1.0 if y == 1 else 0.0)
            else:
                self.c[f] = self.c[f] * np.exp(-d / tau) + (1.0 if (typ is None or y == typ) else 0.0)
            v = self.c[f] if kind == "flow" else np.log1p(self.c[f])
            if self.edges[f] is None:
                self.hist[f].append(v)
                if len(self.hist[f]) >= 5000:
                    self.edges[f] = np.quantile(self.hist[f], np.linspace(0, 1, self.bins + 1)[1:-1]); self.hist[f] = None


def feature_set(name):
    per_type = [("rate", tau, k) for tau in (2.0, 30.0) for k in range(NT)]
    return {"none": [], "total": [("rate", 5.0, None), ("rate", 60.0, None)],
            "per-type": per_type, "per-type+slow+flow": per_type + [("rate", 300.0, None), ("flow", 10.0, None)]}[name]


def run(days, train, evals, feats, eta):
    w = FactorWorld(eta, feats)
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
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--sets", default="none,total,per-type,per-type+slow+flow")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT]); big_q = float(np.quantile(qs, 0.99))
    days = [day_events(d, big_q) for d in PILOT]
    rows = []
    for name in a.sets.split(","):
        feats = feature_set(name)
        val = run(days, range(4), [4], feats, a.eta)
        test = run(days, range(5), [5, 6], feats, a.eta)
        row = {"features": name, "eta": a.eta, **{f"val_{k}": v for k, v in val.items()}, **{f"test_{k}": v for k, v in test.items()}}
        rows.append(row); print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"factor_eta{a.eta:g}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
