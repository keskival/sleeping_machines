"""E37: grokking as a route change: a memorizing route vs a shared route through a rhythm (THEORY §66).

Task: E24's (a + b) mod p, operands as one spike each at t = 0 on lines a and p + b; a fraction of pairs trains.

Two routes to every class, both native, both learned, errors only:
  lookup   pair nodes (one hold/trigger node per operand-line pair, as E34's parts) routed to classes by weights
           r[c, pair] (conserved budget per class): it fires at once if some r[c, (a, b)] > thr. A table: memorizers
           exist (§66), and since it fires first it wins whenever it has an answer.
  rhythm   E26's ring (period p, a free-running rhythm): operand a sets the phase through a learned delay d[a], operand b
           reads it through a learned delay e[b]; class detectors listen at learned phases t[c]; first to coincide
           wins. Shared by all classes. It must wait for the phase, so it answers only if the lookup is silent.
Learning on an error (teacher y): pull y's lookup weights on (a, b) (conserved) and pull the rhythm (E26's pull-only
rule on d[a], e[b], t[y]); a false lookup winner is weakened (conserved). No learning when the answer is right: once
the lookup memorizes, the rhythm stops learning (the absorbing state of §52).
Sleep: after every epoch the lookup weights decay toward uniform by a fraction lam (synaptic downscaling). A lookup
synapse is reinforced by one training pair per epoch; the rhythm's delays by every pair that errs.
"""
import argparse
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e37")
INF = np.inf


def wrap(x, p):
    return (x + p / 2) % p - p / 2


class Net:
    def __init__(self, p, rng, thr=0.5, budget=12.0):
        self.p, self.thr = p, thr
        # lookup = pair nodes (one hold/trigger node per operand-line pair, E34's parts) routed to classes: r[c, pair]
        self.r = rng.uniform(0, 1e-3, (p, p * p)); self.budget = budget   # rows grow to the budget, then conserve
        self.d, self.e, self.t = (rng.uniform(0, p, p) for _ in range(3))                   # rhythm
        self.updates = 0

    def lookup(self, a, b):
        col = self.r[:, a * self.p + b]
        return int(col.argmax()) if col.max() > self.thr else None

    def rhythm(self, a, b):
        phase = (self.e[b] - self.d[a]) % self.p
        wait = (self.t - phase) % self.p
        return int(wait.argmin()), phase

    def predict(self, a, b):
        c = self.lookup(a, b)
        return (c, "lookup") if c is not None else (self.rhythm(a, b)[0], "rhythm")

    def _pull(self, W, c, j, eta):
        tot = W[c].sum()
        if tot < self.budget:                                     # §60: grow freely up to the budget ...
            W[c, j] += eta
        else:                                                     # ... then a pull is paid by the row's others
            W[c, j] += eta * tot; W[c] *= tot / W[c].sum()

    def _drop(self, W, c, j, eta):
        tot = W[c].sum(); W[c, j] *= 1 - eta; W[c] *= tot / max(W[c].sum(), 1e-12)

    def teach(self, a, b, eta, eta_r):
        y = (a + b) % self.p
        c, route = self.predict(a, b)
        if c == y:
            return False
        self._pull(self.r, y, a * self.p + b, eta)
        if route == "lookup":
            self._drop(self.r, c, a * self.p + b, eta)
        _, phase = self.rhythm(a, b)                              # E26: pull only, toward the teacher's phase
        gap = wrap(self.t[y] - phase - 0.5, self.p)
        s = eta_r * np.sign(gap)
        self.e[b] += s / 3; self.d[a] -= s / 3; self.t[y] -= s / 3
        self.updates += 1
        return True

    def sleep(self, lam):
        tot = self.r.sum(1, keepdims=True)
        self.r += lam * (tot / self.r.shape[1] - self.r)          # decay toward uniform, budget conserved


def acc(net, A, B):
    return float(np.mean([net.predict(a, b)[0] == (a + b) % net.p for a, b in zip(A, B)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=31)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--eta-r", type=float, default=0.3)
    ap.add_argument("--budget", type=float, default=12.0, help="lookup synaptic budget per class row")
    ap.add_argument("--lam", type=float, default=0.0, help="sleep: lookup decay per epoch")
    ap.add_argument("--rhythm", type=int, default=1, help="0: no rhythm route (lookup only)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows, t0 = [], time.time()
    for s in range(a.seeds):
        rng = np.random.default_rng(s)
        A_, B_ = np.divmod(np.arange(a.p * a.p), a.p)
        perm = rng.permutation(a.p * a.p); n = int(a.frac * a.p * a.p)
        A, B, At, Bt = A_[perm[:n]], B_[perm[:n]], A_[perm[n:]], B_[perm[n:]]
        net = Net(a.p, rng, budget=a.budget)
        if not a.rhythm:
            net.rhythm = lambda aa, bb: (-1, 0.0)                # rhythm route absent: silent lookup = wrong
        curve = []
        for ep in range(1, a.epochs + 1):
            for i in rng.permutation(n):
                net.teach(A[i], B[i], a.eta, a.eta_r if a.rhythm else 0.0)
            net.sleep(a.lam)
            if ep % a.every == 0:
                lk = np.mean([net.lookup(x, y) is not None for x, y in zip(At, Bt)])
                curve.append({"epoch": ep, "train": acc(net, A, B), "test": acc(net, At, Bt),
                              "test_lookup_answers": float(lk), "updates": net.updates})
        rows.append({"seed": s, "final": curve[-1], "curve": curve})
        print(json.dumps({"seed": s, **curve[-1]}), flush=True)
    name = f"p{a.p}_f{a.frac}_lam{a.lam}_r{a.rhythm}{'_' + a.tag if a.tag else ''}.json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "rho": a.frac * a.p * a.p / (a.p ** 3 + 3 * a.p),
                   "wall_s": round(time.time() - t0, 1)}, f)
    print("EXIT-OK", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
