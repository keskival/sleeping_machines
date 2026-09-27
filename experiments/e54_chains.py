"""E54: deeper order by chain composites with synapses grown on activity (THEORY §85), verified against dense weights.

Task (generalizes E53 to D motifs per class): M motifs on disjoint channel pairs; classes are ORDERS of D-motif sets
(S sets x R orders); an example of class (m1..mD) plants the motifs in that order, each starting 0.5-2.5 after the
previous one ends; 'none' examples (1/3) are mostly decoys (other orders of the same sets); noise spikes (prob q).

Units (event nodes; no clock):
  level 0  parts: one per ordered channel pair, window [0, 1.5]; P = N(N-1)
  level k  chain units (u, p): u a level-(k-1) unit, p a part firing within W2 after u; fires at p's spike.
           Candidate basis Q = P + P^2 + ... + P^(L+1) (5.5e7 at N = 20, L = 2); a unit exists as an event only when it
           fires, and it gets a synapse on a class node only when it is first credited (synaptogenesis, §85).
  classes  summed hold/trigger nodes over units (§83), instant credit with cooled exploration (§84).
Weights: w = s_k * v for grown synapses, s_k * v0 for all others (v0 = 1/Q): exactly dense Winnow with a conserved
budget (§85), with memory and time proportional to the units that ever fired in a credited instant. --dense 1 stores
all Q weights explicitly (feasible at L = 1) to verify that the lazy network makes identical decisions.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e54")
INF = np.inf
PART_HI, W2, W = 1.5, 4.2, 4.2


def make_task(N, M, D, S, R, rng):
    ch = rng.permutation(N)[:2 * M]
    motifs = [(int(ch[2 * m]), int(ch[2 * m + 1]), float(rng.uniform(0.8, 1.5))) for m in range(M)]
    sets = []
    while len(sets) < S:
        st = tuple(sorted(int(x) for x in rng.choice(M, D, replace=False)))
        if st not in sets:
            sets.append(st)
    classes, decoys = [], []
    for st in sets:
        perms = [tuple(p) for p in itertools.permutations(st)]
        idx = rng.permutation(len(perms))
        classes += [perms[i] for i in idx[:R]]; decoys += [perms[i] for i in idx[R:]]
    return motifs, classes, decoys


def holds(motifs, cls, t):
    o = []
    for m in cls:
        i, j, D = motifs[m]
        if not (np.isfinite(t[i]) and np.isfinite(t[j]) and 0.3 <= t[j] - t[i] <= D):
            return False
        o.append((t[i], t[j]))
    return all(0.5 <= o[k + 1][0] - o[k][1] <= 2.5 for k in range(len(o) - 1))


def plant_seq(t, motifs, seq, start, rng):
    s = start
    for m in seq:
        i, j, D = motifs[m]
        t[i] = s; t[j] = s + rng.uniform(0.3, D); s = t[j] + rng.uniform(0.5, 2.5)


def horizon(D):
    return 4.0 * D + 4.0


def sample(task, N, q, rng):
    motifs, classes, decoys = task
    K = len(classes); D = len(classes[0]); H = horizon(D)
    for _ in range(1000):
        none = rng.random() < 1 / 3
        y = K if none else int(rng.integers(K))
        t = np.where(rng.random(N) < q, rng.uniform(0, H, N), INF)
        if not none:
            plant_seq(t, motifs, classes[y], rng.uniform(0, H - 4.0 * D), rng)
        elif rng.random() < 0.9:
            plant_seq(t, motifs, decoys[int(rng.integers(len(decoys)))], rng.uniform(0, H - 4.0 * D), rng)
        true = [k for k in range(K) if holds(motifs, classes[k], t)]
        if (y < K and true == [y]) or (y == K and not true):
            return t, y
    raise RuntimeError("sampling failed")


class Syn:
    """one role's synapses for all K class nodes: w[k, j] = s[k] * V[k, j] for grown synapses j and s[k] * u[k] for every
    ungrown one (all ungrown synapses of a node share one value: they have never been updated)."""

    def __init__(self, K, Q, dense):
        self.K, self.Q = K, Q
        self.u = np.full(K, 1.0 / Q)
        self.V = np.full((K, Q if dense else 1024), 1.0 / Q); self.n = Q if dense else 0
        self.s = np.ones(K); self.sumV = self.u * self.n

    def grow(self, n_new):                                   # synaptogenesis: new synapses start at the ungrown value
        if n_new <= self.n:
            return
        if n_new > self.V.shape[1]:
            V = np.empty((self.K, max(n_new, 2 * self.V.shape[1]))); V[:, :self.n] = self.V[:, :self.n]; self.V = V
        self.V[:, self.n:n_new] = self.u[:, None]
        self.sumV += self.u * (n_new - self.n); self.n = n_new

    def w(self, idx):                                        # idx < 0: a synapse not grown yet (ungrown value)
        return self.s[:, None] * np.where(idx[None, :] >= 0, self.V[:, np.maximum(idx, 0)], self.u[:, None])

    def mul(self, k, idx, fac):                              # multiplicative update of a set, then conserve the budget
        if len(idx) == 0:
            return
        idx = np.unique(idx)
        old = self.V[k, idx].sum(); self.V[k, idx] *= fac; self.sumV[k] += self.V[k, idx].sum() - old
        self.s[k] = 1.0 / (self.sumV[k] + (self.Q - self.n) * self.u[k])
        if not 1e-100 < self.s[k] < 1e100:                   # change of representation (exact): fold s into storage
            self.V[k, :self.n] *= self.s[k]; self.u[k] *= self.s[k]; self.sumV[k] = self.V[k, :self.n].sum()
            self.s[k] = 1.0


class Net:
    def __init__(self, N, K, L, rng, thr, alpha, beta, temp, dense):
        self.N, self.K, self.L, self.thr, self.alpha, self.beta, self.temp, self.rng = N, K, L, thr, alpha, beta, temp, rng
        self.pp = np.array([(p, q) for p in range(N) for q in range(N) if p != q]); self.P = P = len(self.pp)
        self.Q = sum(P ** (k + 1) for k in range(L + 1))
        self.dense = dense
        self.reg = {}                                                        # unit key -> synapse index (lazy)
        self.h = Syn(K, self.Q, dense); self.g = Syn(K, self.Q, dense)
        self.events = 0; self.syn = 0; self.updates = 0; self.margin = 0.0

    def _idx(self, key, k):
        if self.dense:                                                       # exact dense index of the unit
            return key if k == 0 else self._dense_base(k) + key
        return self.reg.get((k, key), -1)                                    # -1: no synapse grown yet

    def _grow(self, U, keys, sel):
        """synaptogenesis at credit: units in the credited set without a synapse get one (on every class node)."""
        for i in np.flatnonzero(sel & (U < 0)):
            U[i] = self.reg.setdefault(keys[i], len(self.reg))
        self.h.grow(len(self.reg)); self.g.grow(len(self.reg))

    def _dense_base(self, k):
        return sum(self.P ** (i + 1) for i in range(k))

    def units(self, t):
        dt = t[self.pp[:, 1]] - t[self.pp[:, 0]]
        f = np.flatnonzero(np.isfinite(dt) & (dt >= 0) & (dt <= PART_HI)); xf = t[self.pp[f, 1]]
        keys = [(0, int(p)) for p in f]; xs = list(xf)
        level = [(int(p), float(x)) for p, x in zip(f, xf)]                  # (flat key within level, time)
        for k in range(1, self.L + 1):
            nxt = []
            for u, xu in level:
                for p, xp in zip(f, xf):
                    if xu < xp <= xu + W2:
                        nxt.append((u * self.P + int(p), float(xp)))
            keys += [(k, u) for u, _ in nxt]; xs += [x for _, x in nxt]; level = nxt
        ids = np.array([self._idx(u, k) for k, u in keys], dtype=np.int64)
        x = np.array(xs); o = np.argsort(x, kind="stable")
        self.events += int(np.isfinite(t).sum()) + len(ids)
        self._keys = [keys[i] for i in o]
        return ids[o], x[o]

    def forward(self, t):
        U, x = self.units(t)
        ft = np.full(self.K, INF); inst = np.full(self.K, -1)
        win = (x[None, :] < x[:, None]) & (x[None, :] >= x[:, None] - W); same = x[None, :] == x[:, None]
        if len(U) >= 2:
            hU, gU = self.h.w(U), self.g.w(U)
            ok = ((hU @ win.T) > self.thr) & ((gU @ same.T) > self.thr)
            anyc = ok.any(1); first = ok.argmax(1)
            ft[anyc] = x[first[anyc]]; inst[anyc] = first[anyc]
            self.syn += int(((hU > 0.01) | (gU > 0.01)).sum())
        self.events += int(np.isfinite(ft).sum())
        c = int(ft.argmin()) if np.isfinite(ft).any() else self.K
        return c, U, win, same, inst, ft

    def teach(self, t, y):
        c, U, win, same, inst, ft = self.forward(t)
        if c == y:
            if self.margin and y < self.K:                                  # §86: near-miss credit
                i = inst[y]
                if self.h.w(U[win[i]])[y].sum() < self.margin:
                    self._grow(U, self._keys, win[i]); self.h.mul(y, U[win[i]], 1 + self.alpha)
                if self.g.w(U[same[i]])[y].sum() < self.margin:
                    self._grow(U, self._keys, same[i]); self.g.mul(y, U[same[i]], 1 + self.alpha)
            return
        if y < self.K and not np.isfinite(ft[y]) and len(U) >= 2:           # miss: instant credit (§84)
            hy, gy = self.h.w(U)[y], self.g.w(U)[y]
            HS = win @ hy; GS = same @ gy; cand = win.any(1); score = np.minimum(HS, GS)
            if self.temp > 0:
                z = np.where(cand, (score - score[cand].max()) / self.temp, -np.inf); pr = np.exp(z)
                i = int(self.rng.choice(len(pr), p=pr / pr.sum()))
            else:
                i = int(np.argmax(np.where(cand, score, -1.0)))
            if HS[i] <= self.thr:
                self._grow(U, self._keys, win[i]); self.h.mul(y, U[win[i]], 1 + self.alpha)
            if GS[i] <= self.thr:
                self._grow(U, self._keys, same[i]); self.g.mul(y, U[same[i]], 1 + self.alpha)
        if c < self.K:                                                       # false fire: demote contributors
            i = inst[c]
            self._grow(U, self._keys, win[i] | same[i])
            self.h.mul(c, U[win[i]], 1 - self.beta); self.g.mul(c, U[same[i]], 1 - self.beta)
        self.updates += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--D", type=int, default=4, help="motifs per class")
    ap.add_argument("--S", type=int, default=5)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--L", type=int, default=-1, help="chain levels above parts (default D - 2)")
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--q", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.0, help="§86 near-miss margin (0 = off)")
    ap.add_argument("--dense", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    L = a.D - 2 if a.L < 0 else a.L
    os.makedirs(OUT, exist_ok=True)
    rows, t0 = [], time.time()
    for s in range(a.seeds):
        rng = np.random.default_rng(s)
        task = make_task(a.N, a.M, a.D, a.S, a.R, rng)
        net = Net(a.N, len(task[1]), L, rng, a.thr, a.alpha, a.beta, a.temp, a.dense); net.margin = a.margin
        curve, decisions = [], []
        for step in range(1, a.steps + 1):
            t, y = sample(task, a.N, a.q, rng)
            net.teach(t, y)
            if step % (a.steps // 8) == 0:
                ev = np.random.default_rng(99); n = 1000; ok = 0; e0, s0 = net.events, net.syn
                for _ in range(n):
                    t, y = sample(task, a.N, a.q, ev); c = net.forward(t)[0]; ok += c == y; decisions.append(c)
                curve.append({"step": step, "test": ok / n, "updates": net.updates, "grown_synapses": len(net.reg),
                              "events_per_episode": (net.events - e0) / n, "synapses_per_episode": (net.syn - s0) / n})
                net.events, net.syn = e0, s0
        rows.append({"seed": s, "Q": net.Q, "final": curve[-1], "curve": curve,
                     "decision_hash": int(np.sum(np.array(decisions) * (np.arange(len(decisions)) % 9973 + 1)))})
        print(json.dumps({"seed": s, "Q": net.Q, **curve[-1], "decision_hash": rows[-1]["decision_hash"]}), flush=True)
    with open(os.path.join(OUT, f"D{a.D}_L{L}_S{a.S}R{a.R}_T{a.temp:g}{'_dense' if a.dense else ''}{f'_m{a.margin:g}' if a.margin else ''}"
                                f"{'_' + a.tag if a.tag else ''}.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "wall_s": round(time.time() - t0, 1)}, f)
    print("EXIT-OK", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
