"""E45: structure discovery: given a menu of generic routes, does the network pick the composition the data needs?

Operands a, b, c (p values each) always spike; the task decides which matter:
  ab    (a + b) mod p          (c irrelevant)
  abc   (a + b + c) mod p
  rand  a random table over (a, b, c)
Routes (all learned by errors-only pulls, as E37/E41; cooled timing noise on the rhythm routes):
  lookup   one node per triple routed to classes (memorizes; fires first when it has an answer), decayed in sleep
  r_ab, r_ac, r_bc   single-stage rhythms over one operand pair (E26's ring: phase = e[y] - d[x])
  chain    two-stage rhythm (a, b) -> c (E41)
Route selection (native): each rhythm route keeps a price that tracks its reliability: after every example each route's
own answer is scored (a free counterfactual: all routes are evaluated anyway) and its price moves down when right, up
when wrong. When the lookup is silent, the route with the lowest price answers.
Reports the route that answers test examples, and test accuracy, per task.
"""
import argparse
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e45")


def wrap(x, p):
    return (x + p / 2) % p - p / 2


class Ring:
    """one or two rhythm stages over operands `ops` (indices into (a, b, c))."""
    def __init__(self, p, ops, rng):
        self.p, self.ops = p, ops
        self.d = [rng.uniform(0, p, p) for _ in ops]          # one delay table per operand
        self.off = rng.uniform(0, p) if len(ops) == 3 else 0.0
        self.t = rng.uniform(0, p, p)

    def phase(self, x, noise=0.0, rng=None):
        v = [self.d[i][x[o]] for i, o in enumerate(self.ops)]
        ph = v[1] - v[0] if len(v) == 2 else v[2] - (v[1] - v[0]) - self.off
        if noise:
            ph += noise * rng.standard_normal()
        return ph % self.p

    def answer(self, x, noise=0.0, rng=None):
        return int(((self.t - self.phase(x, noise, rng)) % self.p).argmin())

    def pull(self, x, y, eta, noise, rng):
        ph = self.phase(x, noise, rng)
        s = eta * np.sign(wrap(self.t[y] - ph - 0.5, self.p)) / (len(self.ops) + 1)
        if len(self.ops) == 2:                                # phase = d1[x1] - d0[x0]
            self.d[1][x[self.ops[1]]] += s; self.d[0][x[self.ops[0]]] -= s
        else:                                                 # phase = d2[c] - d1[b] + d0[a] - off
            self.d[2][x[self.ops[2]]] += s; self.d[1][x[self.ops[1]]] -= s; self.d[0][x[self.ops[0]]] += s
            self.off -= s
        self.t[y] -= s


class Net:
    def __init__(self, p, rng, sigma, budget):
        self.p, self.rng, self.sigma = p, rng, sigma
        self.r = rng.uniform(0, 1e-4, (p, p ** 3)).astype(np.float32); self.budget = budget
        self.routes = {"r_ab": Ring(p, (0, 1), rng), "r_ac": Ring(p, (0, 2), rng), "r_bc": Ring(p, (1, 2), rng),
                       "chain": Ring(p, (0, 1, 2), rng)}
        self.price = {k: 1.0 for k in self.routes}; self.n_upd = 0

    def lookup(self, x):
        col = self.r[:, (x[0] * self.p + x[1]) * self.p + x[2]]
        return int(col.argmax()) if col.max() > 0.5 else None

    def predict(self, x):
        k = self.lookup(x)
        if k is not None:
            return k, "lookup"
        best = min(self.price, key=self.price.get)
        return self.routes[best].answer(x), best

    def teach(self, x, y, eta, eta_r):
        noise = self.sigma * np.exp(-self.n_upd / 20000.0)
        for k, R in self.routes.items():                      # every route is scored (counterfactual) ...
            right = R.answer(x) == y
            self.price[k] += 0.01 * ((0.0 if right else 1.0) - self.price[k])
        c, route = self.predict(x)
        if c == y:
            return
        j = (x[0] * self.p + x[1]) * self.p + x[2]
        row = self.r[y]; tot = row.sum()
        if tot < self.budget: row[j] += eta
        else: row[j] += eta * tot; row *= tot / row.sum()
        if route == "lookup":
            tk = self.r[c].sum(); self.r[c, j] *= 1 - eta; self.r[c] *= tk / max(self.r[c].sum(), 1e-12)
        for R in self.routes.values():                        # ... and every route learns from the error
            if R.answer(x) != y:
                R.pull(x, y, eta_r, noise, self.rng)
        self.n_upd += 1

    def sleep(self, lam):
        tot = self.r.sum(1, keepdims=True); self.r += lam * (tot / self.r.shape[1] - self.r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="abc", choices=("ab", "abc", "rand"))
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=2.2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows, t0 = [], time.time()
    for s in range(a.seeds):
        rng = np.random.default_rng(s)
        T = np.array(np.unravel_index(np.arange(a.p ** 3), (a.p,) * 3)).T
        table = rng.integers(0, a.p, a.p ** 3)
        lab = {"ab": (T[:, 0] + T[:, 1]) % a.p, "abc": T.sum(1) % a.p, "rand": table}[a.task]
        perm = rng.permutation(len(T)); n = int(a.frac * len(T))
        tr, te = perm[:n], perm[n:][:3000]
        net = Net(a.p, rng, a.sigma, budget=1.2 * n / a.p)
        for ep in range(a.epochs):
            for i in rng.permutation(tr):
                net.teach(T[i], lab[i], 0.3, 0.3)
            net.sleep(a.lam)
        used = {}
        ok = 0
        for i in te:
            c, route = net.predict(T[i]); ok += c == lab[i]; used[route] = used.get(route, 0) + 1
        tr_ok = np.mean([net.predict(T[i])[0] == lab[i] for i in tr[:3000]])
        row = {"seed": s, "train": float(tr_ok), "test": ok / len(te),
               "route_share": {k: v / len(te) for k, v in used.items()},
               "prices": {k: round(v, 3) for k, v in net.price.items()}}
        rows.append(row); print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, f"{a.task}_p{a.p}_f{a.frac}_lam{a.lam}{'_' + a.tag if a.tag else ''}.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    print("EXIT-OK", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
