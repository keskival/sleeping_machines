"""E41: grokking with depth: (a + b + c) mod p needs two arithmetic stages composed in time (THEORY §56, §66, §72).

Operands a, b, c each spike once at t = 0 on their lines; a fraction of the p³ triples trains.

Two routes to every class, errors-only learning (as E37):
  lookup   one node per triple (a, b, c) routed to classes, r[c, triple] (conserved budget per class, grows to it):
           memorizes every training triple; fires first when it has an answer.
  chain    depth-2 rhythm route, all delays shared across classes:
             stage 1: a resets rhythm 1 through delay d1[a], b reads it through delay e1[b]: an output spike whose
                      phase is phi1 = e1[b] - d1[a] (mod p), i.e. a time that encodes a + b once learned;
             stage 2: that spike resets rhythm 2 through a shared delay d2; c reads it through delay e2[c]:
                      phase2 = e2[c] - phi1 - d2 (mod p);
             classes listen on rhythm 2 at learned phases t[k]; the first to coincide wins.
           Credit: the teacher's timing gap is pulled along the one causal chain (§56.4): every delay on it moves by
           the same small step in the direction that closes the gap (pull only, §54). Cooled timing noise on both
           stages (E26c, E37).
Sleep: after every epoch the lookup decays toward uniform (§72): a triple's entry is reused by one example, the chain's
delays by many.
"""
import argparse
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e41")


def wrap(x, p):
    return (x + p / 2) % p - p / 2


class Net:
    def __init__(self, p, rng, thr=0.5, budget=None, sigma=0.0, cool=20000.0):
        self.p, self.thr, self.rng = p, thr, rng
        self.r = rng.uniform(0, 1e-4, (p, p ** 3)).astype(np.float32)
        self.budget = budget or 1.2 * p ** 2 * 0.3               # ~ trained triples per class × 1.2
        self.d1, self.e1, self.e2, self.t = (rng.uniform(0, p, p) for _ in range(4))
        self.d2 = rng.uniform(0, p)
        self.sigma, self.cool, self.r_updates = sigma, cool, 0

    def lookup(self, a, b, c):
        col = self.r[:, (a * self.p + b) * self.p + c]
        return int(col.argmax()) if col.max() > self.thr else None

    def chain(self, a, b, c, noisy=False):
        s = self.sigma * np.exp(-self.r_updates / self.cool) if (noisy and self.sigma) else 0.0
        phi1 = (self.e1[b] - self.d1[a] + s * self.rng.standard_normal()) % self.p
        phase2 = (self.e2[c] - phi1 - self.d2 + s * self.rng.standard_normal()) % self.p
        wait = (self.t - phase2) % self.p
        return int(wait.argmin()), phase2

    def predict(self, a, b, c):
        k = self.lookup(a, b, c)
        return (k, "lookup") if k is not None else (self.chain(a, b, c)[0], "chain")

    def teach(self, a, b, c, eta, eta_r):
        y = (a + b + c) % self.p
        k, route = self.predict(a, b, c)
        if k == y:
            return False
        j = (a * self.p + b) * self.p + c
        row = self.r[y]; tot = row.sum()
        if tot < self.budget:
            row[j] += eta
        else:
            row[j] += eta * tot; row *= tot / row.sum()
        if route == "lookup":
            tk = self.r[k].sum(); self.r[k, j] *= 1 - eta; self.r[k] *= tk / max(self.r[k].sum(), 1e-12)
        if eta_r:
            _, phase2 = self.chain(a, b, c, noisy=True)
            s = eta_r * np.sign(wrap(self.t[y] - phase2 - 0.5, self.p)) / 6
            # phase2 = e2[c] - (e1[b] - d1[a]) - d2: move every delay on the chain to close the gap
            self.e2[c] += s; self.e1[b] -= s; self.d1[a] += s; self.d2 -= s; self.t[y] -= s
            self.r_updates += 1
        return True

    def sleep(self, lam):
        tot = self.r.sum(1, keepdims=True)
        self.r += lam * (tot / self.r.shape[1] - self.r)


def acc(net, T):
    return float(np.mean([net.predict(a, b, c)[0] == (a + b + c) % net.p for a, b, c in T]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=31)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--eta-r", type=float, default=0.3)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--chain", type=int, default=1, help="0: lookup only")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows, t0 = [], time.time()
    for s in range(a.seeds):
        rng = np.random.default_rng(s)
        allT = np.array(np.unravel_index(np.arange(a.p ** 3), (a.p,) * 3)).T
        perm = rng.permutation(len(allT)); n = int(a.frac * len(allT))
        tr, te = allT[perm[:n]], allT[perm[n:]][:3000]
        net = Net(a.p, rng, sigma=a.sigma)
        curve = []
        for ep in range(1, a.epochs + 1):
            for i in rng.permutation(n):
                net.teach(*tr[i], a.eta, a.eta_r if a.chain else 0.0)
            net.sleep(a.lam)
            if ep % a.every == 0:
                curve.append({"epoch": ep, "train": acc(net, tr[:3000]), "test": acc(net, te)})
        rows.append({"seed": s, "final": curve[-1], "curve": curve})
        print(json.dumps({"seed": s, **curve[-1]}), flush=True)
    name = f"p{a.p}_f{a.frac}_lam{a.lam}_sig{a.sigma:g}_c{a.chain}{'_' + a.tag if a.tag else ''}.json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "rho": a.frac * a.p ** 3 / (a.p ** 4 + 4 * a.p + 1),
                   "wall_s": round(time.time() - t0, 1)}, f)
    print("EXIT-OK", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
