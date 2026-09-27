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


HOLD = 2.0                                                      # part window (s): k within HOLD after j


class StateTPP:
    """native intensity + a stateful memory of the last one and two event types (arm/disarm nodes held until the next
    event, §75): lambda_e = mu_e + sum alpha S + g1[e, last] + g2[e, last2, last]; all weights positive, learned by
    exponentiated-gradient steps on the online likelihood. The state is constant between events, so its compensator
    term is its value times the gap."""

    def __init__(self, eta=0.005, order=2):
        self.eta, self.order = eta, order
        self.base = TPP("native", eta)
        self.g1 = np.full((NT, NT), 1e-3); self.g2 = np.full((NT, NT, NT), 1e-3)
        self.l1 = self.l2 = None

    def step(self, t, y):
        b = self.base
        if b.t0 is None:
            b.step(t, y); self.l1 = y
            return None
        d = max(t - b.t0, 1e-6)
        decay = np.exp(-BETAS * d); Sb = b.S * decay; integ = b.S * (1 - decay) / BETAS
        st = self.g1[:, self.l1] + (self.g2[:, self.l2, self.l1] if (self.order > 1 and self.l2 is not None) else 0)
        lam = b.mu + np.einsum("ejb,jb->e", b.alpha, Sb) + st
        comp = b.mu * d + np.einsum("ejb,jb->e", b.alpha, integ) + st * d
        ll = float(np.log(max(lam[y], 1e-12)) - comp.sum())
        self.split = (float(np.log(lam[y] / lam.sum())), float(np.log(lam.sum()) - comp.sum()))
        # learning: base parameters as in TPP, plus the state weights (EG)
        g_mu = -d * np.ones(NT); g_mu[y] += 1 / lam[y]
        g_a = -np.repeat(integ[None], NT, 0); g_a[y] += Sb / lam[y]
        b.mu = np.maximum(b.mu * np.exp(np.clip(self.eta * g_mu * b.mu, -.5, .5)), 1e-4)
        b.alpha = np.maximum(b.alpha, 1e-4) * np.exp(np.clip(self.eta * g_a * np.maximum(b.alpha, 1e-4), -.5, .5))
        gs = -d * np.ones(NT); gs[y] += 1 / lam[y]
        self.g1[:, self.l1] *= np.exp(np.clip(self.eta * gs * self.g1[:, self.l1], -.5, .5))
        if self.order > 1 and self.l2 is not None:
            v = self.g2[:, self.l2, self.l1]; self.g2[:, self.l2, self.l1] = v * np.exp(np.clip(self.eta * gs * v, -.5, .5))
        b.S = Sb; b.S[y] += 1; b.t0 = t
        self.l2, self.l1 = self.l1, y
        return ll

    def sleep(self):
        pass


class SemiMarkov:
    """native semi-Markov marked point process: state = identity of the last event (or last two), held by nodes with a
    bank of window durations (GAPS); in each window bucket k the hazard h[ctx, k] and the next-type distribution
    P[ctx, k, :] are learned online as normalized counts (events / exposure time; type counts). Exact likelihood:
    log(h[ctx, k*] * P[ctx, k*, y]) - sum_k h[ctx, k] * (time spent in bucket k). `learn=False` freezes it."""
    GAPS = np.array([0.01, 0.05, 0.2, 1.0, 5.0])

    def __init__(self, order=2):
        self.order = order; nb = len(self.GAPS) + 1
        self.N = np.full((NT, NT, nb), 0.5); self.E = np.full((NT, NT, nb), 1.0)   # events, exposure per bucket
        self.P = np.ones((NT, NT, nb, NT)); self.l1 = self.l2 = None; self.t0 = None; self.learn = True
        self.edges = np.r_[0.0, self.GAPS, np.inf]

    def step(self, t, y):
        if self.t0 is None:
            self.t0, self.l1 = t, y
            return None
        d = max(t - self.t0, 1e-6)
        c2 = (self.l2 if (self.order > 1 and self.l2 is not None) else 0)
        ctx = (c2, self.l1)
        k = int(np.searchsorted(self.GAPS, d))
        span = np.clip(d - self.edges[:-1], 0, np.diff(self.edges))                # time spent in each bucket
        h = self.N[ctx] / self.E[ctx]
        pk = self.P[ctx][k] / self.P[ctx][k].sum()
        comp = float((h * span).sum())
        ll = float(np.log(h[k] * pk[y]) - comp)
        self.split = (float(np.log(pk[y])), float(np.log(h[k]) - comp))
        if self.learn:
            self.N[ctx][k] += 1.0; self.E[ctx] += span; self.P[ctx][k][y] += 1.0
        self.t0 = t; self.l2, self.l1 = self.l1, y
        return ll

    def sleep(self):
        pass


class MarkovMarks:
    """baseline: total rate as the native model; next type from online counts given the last two types (gap=0), or
    given (last type, bucket of the time since the last event) (gap=1), or (last two types, gap bucket) (gap=2)."""
    GAPS = np.array([0.01, 0.05, 0.2, 1.0, 5.0])

    def __init__(self, eta=0.005, gap=0):
        self.rate = TPP("native", eta); self.gap = gap
        self.C = np.ones((NT, NT, len(self.GAPS) + 1, NT)); self.l1 = self.l2 = None

    def step(self, t, y):
        r = self.rate
        if r.t0 is None:
            r.step(t, y); self.l1 = y
            return None
        d = max(t - r.t0, 1e-6)
        decay = np.exp(-BETAS * d); Sb = r.S * decay; integ = r.S * (1 - decay) / BETAS
        lam = r.mu + np.einsum("ejb,jb->e", r.alpha, Sb); comp = r.mu * d + np.einsum("ejb,jb->e", r.alpha, integ)
        tot, ctot = lam.sum(), comp.sum()
        gb = int(np.searchsorted(self.GAPS, d)) if self.gap else 0
        a2 = (self.l2 if self.l2 is not None else self.l1) if self.gap != 1 else 0
        ctx = self.C[a2, self.l1, gb]
        pt = ctx / ctx.sum()
        ll = float(np.log(pt[y]) + np.log(tot) - ctot)
        self.split = (float(np.log(pt[y])), float(np.log(tot) - ctot))
        r.step(t, y)                                          # learns the rates (its own ll is discarded)
        ctx[y] += 1
        self.l2, self.l1 = self.l1, y
        return ll

    def sleep(self):
        pass


class TPP:
    def __init__(self, kind, eta=0.01, meta=False, parts=False):
        self.kind, self.eta, self.meta, self.parts = kind, eta, meta, parts
        self.NS = NT + (NT * NT if parts else 0)                  # state channels: types, then (j, k) parts
        self.mu = np.full(NT, 0.1)
        self.alpha = np.full((NT, self.NS, len(BETAS)), 0.0 if kind == "poisson" else 0.01)
        self.last = np.full(NT, -np.inf)                          # last time of each type (for the hold nodes)
        if parts:
            self.alpha[:, NT:] = 1e-4                              # parts start silent and must earn their weight
        self.fast = np.zeros_like(self.alpha)                   # fast weights, log domain (native_meta)
        self.m = [np.zeros_like(self.mu), np.zeros_like(self.alpha)]; self.v = [np.zeros_like(self.mu),
                                                                                np.zeros_like(self.alpha)]
        self.k = 0
        self.ll_fast, self.ll_slow = 0.0, 0.0
        self.S = np.zeros((self.NS, len(BETAS))); self.t0 = None

    def A(self):
        return self.alpha * np.exp(self.fast)

    def step(self, t, y):
        """score the event, then learn from it. Returns its log-likelihood."""
        if self.t0 is None:
            self.t0 = t; self.S[y] += 1.0; self.last[y] = t
            return None
        d = max(t - self.t0, 1e-6)
        decay = np.exp(-BETAS * d)
        S_before = self.S * decay                               # traces just before the event
        integ = self.S * (1 - decay) / BETAS                    # integral of the traces over (t0, t]
        A = self.A()
        lam = self.mu + np.einsum("ejb,jb->e", A, S_before)
        comp = self.mu * d + np.einsum("ejb,jb->e", A, integ)
        ll = float(np.log(max(lam[y], 1e-12)) - comp.sum())
        self.split = (float(np.log(max(lam[y], 1e-12) / lam.sum())), float(np.log(lam.sum()) - comp.sum()))
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
        if self.parts:                                            # hold/trigger parts: y triggers (j, y) if j is held
            held = np.flatnonzero(t - self.last <= HOLD)
            self.S[NT + held * NT + y] += 1.0
        self.last[y] = t
        return ll

    def sleep(self):
        if self.meta:                                           # consolidate a share of the fast weights, reset
            self.alpha = np.maximum(self.alpha * np.exp(0.3 * self.fast), 1e-4); self.fast[:] = 0.0


class NeuralTPP:
    """continuous-time GRU point process (neural-Hawkes style), trained online by Adam, one step per event."""

    def __init__(self, d=16, lr=3e-3, seed=0):
        import torch
        self.torch = torch
        torch.manual_seed(seed); torch.set_num_threads(1)
        self.cell = torch.nn.GRUCell(NT + 1, d)
        self.out = torch.nn.Linear(d, NT)
        self.log_gamma = torch.nn.Parameter(torch.zeros(d))
        self.opt = torch.optim.Adam(list(self.cell.parameters()) + list(self.out.parameters()) + [self.log_gamma], lr=lr)
        self.h = torch.zeros(1, d); self.t0 = None

    def lam(self, h, dt):
        g = self.torch.exp(self.log_gamma)
        return self.torch.nn.functional.softplus(self.out(h * self.torch.exp(-g * dt)))

    def step(self, t, y):
        torch = self.torch
        if self.t0 is None:
            self.t0 = t
            x = torch.zeros(1, NT + 1); x[0, y] = 1.0
            self.h = self.cell(x, self.h).detach()
            return None
        d = max(t - self.t0, 1e-6)
        ts = torch.linspace(0, d, 6)[:, None]                   # quadrature for the compensator
        lams = torch.cat([self.lam(self.h, tt) for tt in ts])  # (6, NT)
        comp = torch.trapezoid(lams.sum(1), ts[:, 0])
        ll = torch.log(lams[-1, y] + 1e-9) - comp
        lt = lams[-1].detach()
        self.split = (float(torch.log(lt[y] / lt.sum())), float(torch.log(lt.sum()) - comp.detach()))
        self.opt.zero_grad(); (-ll).backward(); self.opt.step()
        with torch.no_grad():
            x = torch.zeros(1, NT + 1); x[0, y] = 1.0; x[0, NT] = float(np.log(d + 1e-3))
            g = torch.exp(self.log_gamma)
            self.h = self.cell(x, self.h * torch.exp(-g * d)).detach()
        self.t0 = t
        return float(ll.detach())

    def sleep(self):
        pass


class InhibTPP:
    """native intensity with excitation AND inhibition: lambda_e(t) = mu_e * exp(sum W_exc S - sum W_inh S), both weight
    sets positive and learned by multiplicative (exponentiated-gradient) updates; compensator by trapezoid quadrature."""

    def __init__(self, eta=0.005, parts=False, q=6):
        self.eta, self.parts, self.q = eta, parts, q
        self.NS = NT + (NT * NT if parts else 0)
        self.mu = np.full(NT, 0.1)
        self.We = np.full((NT, self.NS, len(BETAS)), 1e-3); self.Wi = np.full((NT, self.NS, len(BETAS)), 1e-3)
        self.S = np.zeros((self.NS, len(BETAS))); self.t0 = None; self.last = np.full(NT, -np.inf)

    def lam(self, S):
        return self.mu * np.exp(np.clip(np.einsum("ejb,jb->e", self.We - self.Wi, S), -8, 8))

    def step(self, t, y):
        if self.t0 is None:
            self.t0 = t; self.S[y] += 1.0; self.last[y] = t
            return None
        d = max(t - self.t0, 1e-6)
        taus = np.linspace(0, d, self.q)
        Ss = [self.S * np.exp(-BETAS * u) for u in taus]
        lams = np.array([self.lam(S_) for S_ in Ss])            # (q, NT)
        wts = np.full(self.q, d / (self.q - 1)); wts[[0, -1]] /= 2
        comp = (wts[:, None] * lams).sum(0)
        ll = float(np.log(max(lams[-1, y], 1e-12)) - comp.sum())
        # gradients of ll w.r.t. log-intensity parameters
        gz = -np.einsum("q,qe,qjb->ejb", wts, lams, np.array(Ss))  # d(-compensator)/d(W_e) summed over time
        gz[y] += Ss[-1]
        g_mu = -comp / self.mu; g_mu[y] += 1.0 / self.mu[y]
        self.mu *= np.exp(np.clip(self.eta * g_mu * self.mu, -0.5, 0.5))
        # log-domain steps normalized per synapse by its own running gradient scale (local, keeps weights positive)
        self.v = 0.999 * getattr(self, "v", np.zeros_like(gz)) + 0.001 * gz * gz
        st = 1e-3 * gz / (np.sqrt(self.v) + 1e-8)
        self.We *= np.exp(np.clip(st, -0.1, 0.1))                # excitatory: + direction
        self.Wi *= np.exp(np.clip(-st, -0.1, 0.1))               # inhibitory: - direction
        np.clip(self.We, 1e-5, 5, out=self.We); np.clip(self.Wi, 1e-5, 5, out=self.Wi)
        self.S = Ss[-1]; self.S[y] += 1.0; self.t0 = t
        if self.parts:
            held = np.flatnonzero(t - self.last <= HOLD)
            self.S[NT + held * NT + y] += 1.0
        self.last[y] = t
        return ll

    def sleep(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndays", type=int, default=7)
    ap.add_argument("--eta", type=float, default=0.005)
    ap.add_argument("--neural", type=int, default=1)
    ap.add_argument("--freeze_after", type=int, default=0, help="stop learning after this many days (held-out test)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT]); big_q = float(np.quantile(qs, 0.99))
    models = {"poisson": TPP("poisson", a.eta), "hawkes": TPP("hawkes", a.eta), "native": TPP("native", a.eta),
              "native_meta": TPP("native", a.eta, meta=True),
              "native_parts": TPP("native", a.eta, parts=True), "hawkes_parts": TPP("hawkes", a.eta, parts=True),
              }
    models = {k: models[k] for k in ("poisson", "hawkes", "native")}
    models["native_state1"] = StateTPP(a.eta, order=1); models["native_state2"] = StateTPP(a.eta, order=2)
    models = {k: models[k] for k in ("native", "neural")} if "neural" in models else {k: models[k] for k in ("native",)}
    models["markov_last2"] = MarkovMarks(a.eta, gap=0); models["markov_last1_gap"] = MarkovMarks(a.eta, gap=1)
    models["markov_last2_gap"] = MarkovMarks(a.eta, gap=2)
    models["semimarkov1"] = SemiMarkov(order=1); models["semimarkov2"] = SemiMarkov(order=2)
    if a.neural:
        models["neural"] = NeuralTPP()
    rows = []
    for di, day in enumerate(PILOT[:a.ndays]):
        if a.freeze_after and di == a.freeze_after:
            for m in models.values():
                m.learn = False
                if hasattr(m, "opt"):
                    for g in m.opt.param_groups: g["lr"] = 0.0
                if hasattr(m, "eta"): m.eta = 0.0
                if hasattr(m, "rate"): m.rate.eta = 0.0
        T, Y = day_events(day, big_q)
        tot = {k: 0.0 for k in models}; n = 0; split = {k: np.zeros(2) for k in models}
        for t, y in zip(T, Y):
            lls = {k: m.step(t, y) for k, m in models.items()}
            if lls["native"] is not None:
                n += 1
                for k in models:
                    tot[k] += lls[k]; split[k] += np.array(getattr(models[k], "split", (0.0, 0.0)))
        for m in models.values():
            m.sleep()
        row = {"day": str(day), "events": int(n), "counts": np.bincount(Y, minlength=NT).tolist(),
               **{f"ll_per_event_{k}": tot[k] / max(n, 1) for k in models},
               **{f"split_type_time_{k}": (split[k] / max(n, 1)).round(3).tolist() for k in models}}
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, "pilot_stage1.json"), "w") as f:
        json.dump({"args": vars(a), "big_q": big_q, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
