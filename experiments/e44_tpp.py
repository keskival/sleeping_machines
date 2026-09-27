"""E43/E44 stage 1: an asynchronous world model of the BTCUSDT trade stream as a temporal point process (THEORY §79).

Event types: 0 price up-move, 1 price down-move (E42's price events: >= 0.5 bp from the previous one, >= 1 s apart),
2 large aggressive buy, 3 large aggressive sell (top 1% of trade sizes on the pilot days). Events are processed in time
order; every model predicts the next event's type and time, is scored by the log-likelihood of what happened
(prequential: scored before it learns from the event), and then learns from it.

State: exponentially decaying traces S[j, b] of each type j at time scales 1/beta_b (1 s and 30 s): leaky
integrators updated at events (the Hawkes excitation; a continuous-time latent state).
Intensity: lambda_e(t) = mu_e + sum_{j,b} alpha[e, j, b] * S[j, b](t).
Log-likelihood of an event of type e at t after the previous event at t0: log lambda_e(t-) - int_{t0}^{t} sum_e' lambda_e'.

Models: poisson (alpha = 0); hawkes (additive online gradient ascent); native (exponentiated-gradient, i.e. normalized
multiplicative updates on positive parameters, §68); native_meta (native + fast/slow parameters, surprise-gated fast
plasticity, sleep consolidation between days).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e17_market import load_day, PILOT, CONF  # noqa: E402
from e42_when import decision_points  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e44")
BETAS = np.array([1.0, 1 / 30.0])                              # per second
NT = 4


def day_events(day, big_q):
    t, p, q, sell = load_day(day)
    dp = decision_points(t, p)
    dlp = np.r_[0.0, np.diff(np.log(p[dp]))]
    ev_t = [t[dp][1:]]; ev_y = [np.where(dlp[1:] > 0, 0, 1)]
    big = np.flatnonzero(q >= big_q)
    ev_t.append(t[big]); ev_y.append(np.where(sell[big], 3, 2))
    T = np.concatenate(ev_t) / 1e6; Y = np.concatenate(ev_y)
    o = np.argsort(T, kind="stable")
    return T[o], Y[o]


class TPP:
    def __init__(self, kind, eta=0.01, meta=False):
        self.kind, self.eta, self.meta = kind, eta, meta
        self.mu = np.full(NT, 0.1)
        self.alpha = np.full((NT, NT, len(BETAS)), 0.0 if kind == "poisson" else 0.01)
        self.fast = np.zeros_like(self.alpha)                   # fast weights, log domain (native_meta)
        self.m = [np.zeros_like(self.mu), np.zeros_like(self.alpha)]; self.v = [np.zeros_like(self.mu),
                                                                                np.zeros_like(self.alpha)]
        self.k = 0
        self.ll_fast, self.ll_slow = 0.0, 0.0
        self.S = np.zeros((NT, len(BETAS))); self.t0 = None

    def A(self):
        return self.alpha * np.exp(self.fast)

    def step(self, t, y):
        """score the event, then learn from it. Returns its log-likelihood."""
        if self.t0 is None:
            self.t0 = t; self.S[y] += 1.0
            return None
        d = max(t - self.t0, 1e-6)
        decay = np.exp(-BETAS * d)
        S_before = self.S * decay                               # traces just before the event
        integ = self.S * (1 - decay) / BETAS                    # integral of the traces over (t0, t]
        A = self.A()
        lam = self.mu + np.einsum("ejb,jb->e", A, S_before)
        comp = self.mu * d + np.einsum("ejb,jb->e", A, integ)
        ll = float(np.log(max(lam[y], 1e-12)) - comp.sum())
        if self.kind != "poisson" or True:
            g_mu = -d * np.ones(NT); g_mu[y] += 1.0 / max(lam[y], 1e-12)
            g_a = -integ[None, :, :].repeat(NT, 0)
            g_a[y] += S_before / max(lam[y], 1e-12)
            eta = self.eta
            if self.meta:                                       # surprise: recent vs long-run likelihood
                self.ll_fast += 0.05 * (ll - self.ll_fast); self.ll_slow += 0.001 * (ll - self.ll_slow)
                surprise = np.clip(np.exp(self.ll_slow - self.ll_fast), 0.25, 4.0)
            if self.kind == "hawkes":                           # Adam on (mu, alpha), projected to >= 0
                self.k += 1
                for i, (P, g) in enumerate(((self.mu, g_mu), (self.alpha, g_a))):
                    self.m[i] = 0.9 * self.m[i] + 0.1 * g; self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
                    mh = self.m[i] / (1 - 0.9 ** self.k); vh = self.v[i] / (1 - 0.999 ** self.k)
                    P += 1e-3 * mh / (np.sqrt(vh) + 1e-8)
                np.maximum(self.mu, 1e-4, out=self.mu); np.maximum(self.alpha, 0.0, out=self.alpha)
            elif self.kind == "poisson":
                self.mu = np.maximum(self.mu * np.exp(np.clip(eta * g_mu * self.mu, -0.5, 0.5)), 1e-4)
            else:                                               # native: exponentiated gradient (multiplicative)
                self.mu = np.maximum(self.mu * np.exp(np.clip(eta * g_mu * self.mu, -0.5, 0.5)), 1e-4)
                step = np.clip(eta * g_a * np.maximum(self.alpha, 1e-4), -0.5, 0.5)
                self.alpha = np.maximum(self.alpha, 1e-4) * np.exp(step)
                if self.meta:                                   # fast weights: log-domain correction, surprise-gated,
                    self.fast = 0.995 * self.fast + np.clip(eta * surprise * g_a * self.A(), -0.2, 0.2)   # leaky
        self.S = S_before; self.S[y] += 1.0; self.t0 = t
        return ll

    def sleep(self):
        if self.meta:                                           # consolidate a share of the fast weights, reset
            self.alpha = np.maximum(self.alpha * np.exp(0.3 * self.fast), 1e-4); self.fast[:] = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndays", type=int, default=7)
    ap.add_argument("--eta", type=float, default=0.005)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT]); big_q = float(np.quantile(qs, 0.99))
    models = {"poisson": TPP("poisson", a.eta), "hawkes": TPP("hawkes", a.eta), "native": TPP("native", a.eta),
              "native_meta": TPP("native", a.eta, meta=True)}
    rows = []
    for day in PILOT[:a.ndays]:
        T, Y = day_events(day, big_q)
        tot = {k: 0.0 for k in models}; n = 0
        for t, y in zip(T, Y):
            lls = {k: m.step(t, y) for k, m in models.items()}
            if lls["poisson"] is not None:
                n += 1
                for k in models:
                    tot[k] += lls[k]
        for m in models.values():
            m.sleep()
        row = {"day": str(day), "events": int(n), "counts": np.bincount(Y, minlength=NT).tolist(),
               **{f"ll_per_event_{k}": tot[k] / max(n, 1) for k in models}}
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, "pilot_stage1.json"), "w") as f:
        json.dump({"args": vars(a), "big_q": big_q, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
