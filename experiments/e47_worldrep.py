"""E47: a native world model that discovers its own representation from prediction errors (THEORY §79, §80, §81).

Stream: BTCUSDT events on 6 input channels: up-move small (0.5–1.5 bp), up-move large (> 1.5 bp), down small, down large,
large aggressive buy, large aggressive sell. Targets (the point process being modeled): 4 types, as E44: up, down,
large buy, large sell, so likelihoods are comparable with E44.

Base intensity (E44's native model): lambda_e = mu_e + sum_{j,b} alpha[e, j, b] S[j, b], with traces S of the 4 target
types at 1 s and 30 s, learned by exponentiated-gradient steps on the online log-likelihood.
Discovered routes: a route is a hold/trigger node "input channel i, then channel j within window w" (w in 0.1, 1, 5 s); it
fires at j's time and keeps its own trace R (decay 1 s). Each target type e has a menu of up to M routes; a route feeds
lambda_e through a weight beta[e, r] (EG, like alpha). Routes are proposed from misses: when an event of type e occurs
with low predicted probability (lambda_e / sum lambda below its running median), every route pattern that fired in the
preceding window is counted for e; when a pattern's count reaches `min_count` and is not yet in e's menu, it replaces
e's weakest route. Sleep between days: route weights decay by `lam` and routes whose weight falls below a floor are
freed. Scored prequentially (before learning from each event).
Generalization test (--freeze_after d): after d days all learning stops; later days are scored with frozen structure.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e17_market import load_day, PILOT  # noqa: E402
from e42_when import decision_points  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e47")
BETAS = np.array([1.0, 1 / 30.0]); NT, NC = 4, 6
WINS = (0.1, 1.0, 5.0)


def day_events(day, big_q):
    t, p, q, sell = load_day(day)
    dp = decision_points(t, p)
    dlp = 1e4 * np.diff(np.log(p[dp]))
    tt = [t[dp][1:]]
    ch = [np.where(dlp > 0, np.where(dlp > 1.5, 1, 0), np.where(dlp < -1.5, 3, 2))]
    big = np.flatnonzero(q >= big_q)
    tt.append(t[big]); ch.append(np.where(sell[big], 5, 4))
    T = np.concatenate(tt) / 1e6; C = np.concatenate(ch)
    o = np.argsort(T, kind="stable")
    T, C = T[o], C[o]
    Y = np.array([0, 0, 1, 1, 2, 3])[C]                      # target type of each event
    return T, C, Y


class Model:
    def __init__(self, eta=0.005, M=16, min_count=30, routes=True, lam=0.3):
        self.eta, self.M, self.min_count, self.routes_on, self.lam = eta, M, min_count, routes, lam
        self.mu = np.full(NT, 0.1); self.alpha = np.full((NT, NT, len(BETAS)), 0.01)
        self.routes = [[] for _ in range(NT)]                 # per target: list of (i, j, w)
        self.beta = [np.zeros(0) for _ in range(NT)]
        self.R = {}                                           # route pattern -> trace value at self.t0
        self.cnt = [Counter() for _ in range(NT)]
        self.S = np.zeros((NT, len(BETAS))); self.t0 = None
        self.last = np.full(NC, -np.inf); self.p_hist = [[] for _ in range(NT)]
        self.learn = True

    def route_vals(self, d):
        """trace of every active route just before now (decay 1 s over the gap d)."""
        return {k: v * np.exp(-d) for k, v in self.R.items()}

    def step(self, t, c, y):
        if self.t0 is None:
            self.t0 = t; self.S[y] += 1; self.last[c] = t
            return None
        d = max(t - self.t0, 1e-6)
        decay = np.exp(-BETAS * d)
        Sb = self.S * decay; integ = self.S * (1 - decay) / BETAS
        Rb = self.route_vals(d); Rint = {k: v * (1 - np.exp(-d)) for k, v in self.R.items()}
        lam = self.mu + np.einsum("ejb,jb->e", self.alpha, Sb)
        comp = self.mu * d + np.einsum("ejb,jb->e", self.alpha, integ)
        for e in range(NT):
            for r, key in enumerate(self.routes[e]):
                lam[e] += self.beta[e][r] * Rb.get(key, 0.0); comp[e] += self.beta[e][r] * Rint.get(key, 0.0)
        ll = float(np.log(max(lam[y], 1e-12)) - comp.sum())
        if self.learn:
            g_mu = -d * np.ones(NT); g_mu[y] += 1 / lam[y]
            g_a = -np.repeat(integ[None], NT, 0); g_a[y] += Sb / lam[y]
            self.mu = np.maximum(self.mu * np.exp(np.clip(self.eta * g_mu * self.mu, -.5, .5)), 1e-4)
            self.alpha = np.maximum(self.alpha, 1e-4) * np.exp(np.clip(self.eta * g_a * np.maximum(self.alpha, 1e-4), -.5, .5))
            for e in range(NT):
                if len(self.routes[e]):
                    gb = np.array([-Rint.get(k, 0.0) + (Rb.get(k, 0.0) / lam[y] if e == y else 0.0) for k in self.routes[e]])
                    self.beta[e] = np.maximum(self.beta[e], 1e-4) * np.exp(np.clip(self.eta * gb * np.maximum(self.beta[e], 1e-4), -.5, .5))
            if self.routes_on:                                # propose routes from misses
                pr = lam[y] / lam.sum()
                h = self.p_hist[y]; h.append(pr)
                if len(h) > 200: h.pop(0)
                if pr < np.median(h):
                    recent = np.flatnonzero(t - self.last <= max(WINS))
                    for i in recent:
                        for j in recent:
                            if i != j and self.last[i] < self.last[j]:
                                for w in WINS:
                                    if self.last[j] - self.last[i] <= w:
                                        self.cnt[y][(int(i), int(j), w)] += 1
                    if self.cnt[y]:
                        key, n = self.cnt[y].most_common(1)[0]
                        if n >= self.min_count and key not in self.routes[y]:
                            if len(self.routes[y]) < self.M:
                                self.routes[y].append(key); self.beta[y] = np.r_[self.beta[y], 1e-3]
                            else:
                                k = int(np.argmin(self.beta[y])); self.routes[y][k] = key; self.beta[y][k] = 1e-3
                            self.cnt[y][key] = 0
        # advance state: traces decay, the new event updates target traces, route nodes fire (hold/trigger)
        self.S = Sb; self.S[y] += 1
        self.R = Rb
        for key in {k for rs in self.routes for k in rs}:
            i, j, w = key
            if c == j and 0 < t - self.last[i] <= w:
                self.R[key] = self.R.get(key, 0.0) + 1.0
        self.last[c] = t; self.t0 = t
        return ll

    def sleep(self):
        for e in range(NT):
            if len(self.routes[e]):
                self.beta[e] *= 1 - self.lam
                keep = self.beta[e] > 1e-4
                self.routes[e] = [k for k, kk in zip(self.routes[e], keep) if kk]; self.beta[e] = self.beta[e][keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndays", type=int, default=7)
    ap.add_argument("--freeze_after", type=int, default=0)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT]); big_q = float(np.quantile(qs, 0.99))
    models = {"native": Model(routes=False), "routes": Model(M=a.M)}
    rows = []
    for di, day in enumerate(PILOT[:a.ndays]):
        if a.freeze_after and di == a.freeze_after:
            for m in models.values():
                m.learn = False
        T, C, Y = day_events(day, big_q)
        tot = {k: 0.0 for k in models}; n = 0
        for t, c, y in zip(T, C, Y):
            lls = {k: m.step(t, c, y) for k, m in models.items()}
            if lls["native"] is not None:
                n += 1
                for k in models:
                    tot[k] += lls[k]
        for m in models.values():
            if m.learn:
                m.sleep()
        row = {"day": str(day), "frozen": bool(a.freeze_after and di >= a.freeze_after), "events": n,
               **{f"ll_{k}": tot[k] / n for k in models},
               "routes": [len(r) for r in models["routes"].routes],
               "example_routes": [[list(map(float, k)) for k in r[:3]] for r in models["routes"].routes]}
        rows.append(row); print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"worldrep_M{a.M}_f{a.freeze_after}{'_' + a.tag if a.tag else ''}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
