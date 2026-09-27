"""E53: order among three parts needs depth 3; the expanded candidate basis is paid for by activity, not size (THEORY §84).

Task: M motifs on disjoint channel pairs ("j within [0.3, D] after i", D in [0.8, 1.5]); classes are ORDERS of motif sets:
S sets of 3 motifs, R of the 6 orders of each set -> K = S*R classes, so classes share motifs AND sets, and only order
separates them. An example of class (a, b, c): a, then b starting 0.5-2.5 after a ends, then c 0.5-2.5 after b ends; noise
spikes on every channel with probability q; 'none' examples (prob 1/3) are mostly decoys: a set's motifs in an order that
is not a class. Labels are checked (exactly one class holds, or none).

Network (all event nodes, no clock):
  parts       one node per ordered channel pair, window [0, 1.5] (E34); P = N(N-1) nodes
  composites  (--depth 3) one node per ordered part pair (u, v): fires at v's spike if u fired within W2 before; P^2 nodes,
              materialized only when they fire (a composite costs nothing unless both its parts fire in its window)
  classes     summed hold/trigger nodes over units (parts, and composites at depth 3), §83: fire at the first unit
              instant where the held input (units in [tau - W, tau)) and the coincident trigger input both exceed thr
  readout     classes race; none if no class fires
Credit: §83 full-information Winnow with conserved budgets (miss: promote every hold/trigger candidate x(1 + alpha);
false fire: demote the contributors x(1 - beta)).
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e53")
INF = np.inf
PART_HI, W2, W = 1.5, 4.2, 4.2


def make_task(N, M, S, R, rng):
    ch = rng.permutation(N)[:2 * M]
    motifs = [(int(ch[2 * m]), int(ch[2 * m + 1]), float(rng.uniform(0.8, 1.5))) for m in range(M)]
    sets = []
    while len(sets) < S:
        st = tuple(sorted(int(x) for x in rng.choice(M, 3, replace=False)))
        if st not in sets:
            sets.append(st)
    classes, decoys = [], []
    for st in sets:
        perms = [tuple(p) for p in itertools.permutations(st)]
        idx = rng.permutation(6)
        classes += [perms[i] for i in idx[:R]]; decoys += [perms[i] for i in idx[R:]]
    return motifs, classes, decoys


def occ(motifs, m, t):
    i, j, D = motifs[m]
    return (t[i], t[j]) if np.isfinite(t[i]) and np.isfinite(t[j]) and 0.3 <= t[j] - t[i] <= D else None


def holds(motifs, cls, t):
    o = [occ(motifs, m, t) for m in cls]
    if any(x is None for x in o):
        return False
    return all(0.5 <= o[k + 1][0] - o[k][1] <= 2.5 for k in range(2))


def plant_seq(t, motifs, seq, start, rng):
    s = start
    for m in seq:
        i, j, D = motifs[m]
        t[i] = s; t[j] = s + rng.uniform(0.3, D); s = t[j] + rng.uniform(0.5, 2.5)


def sample(task, N, H, q, rng):
    motifs, classes, decoys = task
    K = len(classes)
    for _ in range(1000):
        none = rng.random() < 1 / 3
        y = K if none else int(rng.integers(K))
        t = np.where(rng.random(N) < q, rng.uniform(0, H, N), INF)
        if not none:
            plant_seq(t, motifs, classes[y], rng.uniform(0, H - 10), rng)
        elif rng.random() < 0.9:
            plant_seq(t, motifs, decoys[int(rng.integers(len(decoys)))], rng.uniform(0, H - 10), rng)
        true = [k for k in range(K) if holds(motifs, classes[k], t)]
        if (y < K and true == [y]) or (y == K and not true):
            return t, y
    raise RuntimeError("sampling failed")


class Net:
    def __init__(self, N, K, depth, rng, thr, alpha, beta, credit="instant"):
        self.N, self.K, self.depth, self.thr, self.alpha, self.beta = N, K, depth, thr, alpha, beta
        self.credit = credit; self.temp = 0.0; self.rng = rng; self.margin = 0.0; self.nm = 0
        self.gate = 0.0; self.ok_c = np.zeros(K); self.bad_c = np.zeros(K)   # §86b: recent precision per node
        self.pp = np.array([(p, q) for p in range(N) for q in range(N) if p != q]); self.P = len(self.pp)
        self.Q = self.P + (self.P ** 2 if depth == 3 else 0)
        self.h = rng.uniform(0.5, 1.5, (K, self.Q)); self.h /= self.h.sum(1, keepdims=True)
        self.g = rng.uniform(0.5, 1.5, (K, self.Q)); self.g /= self.g.sum(1, keepdims=True)
        self.events = 0; self.syn = 0; self.updates = 0

    def units(self, t):
        dt = t[self.pp[:, 1]] - t[self.pp[:, 0]]
        f = np.flatnonzero(np.isfinite(dt) & (dt >= 0) & (dt <= PART_HI))
        xf = t[self.pp[f, 1]]
        ids, xs = [f], [xf]
        if self.depth == 3 and len(f) >= 2:
            d = xf[None, :] - xf[:, None]                                     # d[u, v] = x_v - x_u
            u, v = np.nonzero((d > 0) & (d <= W2))
            ids.append(self.P + f[u] * self.P + f[v]); xs.append(xf[v])
        ids = np.concatenate(ids); xs = np.concatenate(xs)
        o = np.argsort(xs, kind="stable")
        self.events += int(np.isfinite(t).sum()) + len(ids)
        return ids[o], xs[o]

    def forward(self, t):
        U, x = self.units(t)
        ft = np.full(self.K, INF); inst = np.full(self.K, -1)
        win = (x[None, :] < x[:, None]) & (x[None, :] >= x[:, None] - W); same = x[None, :] == x[:, None]
        if len(U) >= 2:
            hU, gU = self.h[:, U], self.g[:, U]
            ok = ((hU @ win.T) > self.thr) & ((gU @ same.T) > self.thr)
            anyc = ok.any(1); first = ok.argmax(1)
            ft[anyc] = x[first[anyc]]; inst[anyc] = first[anyc]
            self.syn += int(((hU > 0.01) | (gU > 0.01)).sum())
        self.events += int(np.isfinite(ft).sum())
        c = int(ft.argmin()) if np.isfinite(ft).any() else self.K
        return c, U, win, same, inst, ft

    def _mul(self, W_, c, idx, fac):
        if len(idx):
            W_[c, idx] *= fac; W_[c] /= W_[c].sum()

    def _near_miss(self, y, U, win, same, i):
        """§86: a correct fire below the margin promotes its own contributors (keeps the route > theta_m)."""
        if self.h[y, U[win[i]]].sum() < self.margin:
            self._mul(self.h, y, U[win[i]], 1 + self.alpha); self.nm += 1
        if self.g[y, U[same[i]]].sum() < self.margin:
            self._mul(self.g, y, U[same[i]], 1 + self.alpha); self.nm += 1

    def teach(self, t, y):
        c, U, win, same, inst, ft = self.forward(t)
        if c < self.K:                                      # the firing node's record (decaying counts, local)
            self.ok_c[c] = 0.98 * self.ok_c[c] + (c == y); self.bad_c[c] = 0.98 * self.bad_c[c] + (c != y)
        if c == y:
            prec = self.ok_c[y] / max(self.ok_c[y] + self.bad_c[y], 1e-9) if y < self.K else 0.0
            if self.margin and y < self.K and prec >= self.gate:
                i = inst[y]
                if self.credit == "latest" and win.any():   # §89: margin for the complete route (latest instant)
                    i = int(np.flatnonzero(win.any(1))[-1])
                self._near_miss(y, U, win, same, i)
            return
        if y < self.K and not np.isfinite(ft[y]) and len(U) >= 2:
            if self.credit == "union":                      # §83 as stated: every candidate in the episode
                self._mul(self.h, y, U[win.any(0)], 1 + self.alpha); self._mul(self.g, y, U[win.any(1)], 1 + self.alpha)
            else:                                           # §84: at the instant closest to firing, deficient roles only
                HS = win @ self.h[y, U]; GS = same @ self.g[y, U]
                cand = win.any(1)
                score = np.minimum(HS, GS)
                if self.credit == "latest":                 # §89: the last candidate instant (complete evidence)
                    i = int(np.flatnonzero(cand)[-1])
                elif self.temp > 0:                         # cooled exploration over instants (§76, §84)
                    z = np.where(cand, (score - score[cand].max()) / self.temp, -np.inf)
                    pr = np.exp(z); i = int(self.rng.choice(len(pr), p=pr / pr.sum()))
                else:
                    i = int(np.argmax(np.where(cand, score, -1.0)))
                if HS[i] <= self.thr:
                    self._mul(self.h, y, U[win[i]], 1 + self.alpha)
                if GS[i] <= self.thr:
                    self._mul(self.g, y, U[same[i]], 1 + self.alpha)
        if c < self.K:
            i = inst[c]
            self._mul(self.h, c, U[win[i]], 1 - self.beta); self._mul(self.g, c, U[same[i]], 1 - self.beta)
        self.updates += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--S", type=int, default=5)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--credit", default="instant", choices=("instant", "union", "latest"))
    ap.add_argument("--temp", type=float, default=0.0, help="instant credit: softmax temperature (0 = greedy)")
    ap.add_argument("--margin", type=float, default=0.0, help="§86 near-miss margin theta_m (0 = off)")
    ap.add_argument("--gate", type=float, default=0.0, help="§86b: near-miss only if the node's recent precision >= gate")
    ap.add_argument("--q", type=float, default=0.25, help="noise spike probability per channel")
    ap.add_argument("--eval-at", default="", help="comma list of steps to evaluate at (default: 8 even checkpoints)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    H, q, rows, t0 = 16.0, a.q, [], time.time()
    evals = set(int(x) for x in a.eval_at.split(",")) if a.eval_at else set()
    for s in range(a.seeds):
        rng = np.random.default_rng(s)
        task = make_task(a.N, a.M, a.S, a.R, rng)
        net = Net(a.N, len(task[1]), a.depth, rng, a.thr, a.alpha, a.beta, a.credit); net.temp = a.temp; net.margin = a.margin; net.gate = a.gate
        curve = []
        for step in range(1, a.steps + 1):
            t, y = sample(task, a.N, H, q, rng)
            net.teach(t, y)
            if (step in evals) if evals else (step % (a.steps // 8) == 0):
                ev = np.random.default_rng(99); n = 1500; ok = 0; e0, s0 = net.events, net.syn; trail = npos = 0
                for _ in range(n):
                    t, y = sample(task, a.N, H, q, ev); ok += net.forward(t)[0] == y
                    if y < len(task[1]):                     # §91: is the latest candidate instant after the pattern?
                        ev0 = net.events; U, x = net.units(t); net.events = ev0; last_end = t[task[0][task[1][y][-1]][1]]
                        cand = x[((x[None, :] < x[:, None]) & (x[None, :] >= x[:, None] - W)).any(1)]
                        npos += 1; trail += bool(len(cand) and cand.max() > last_end + 1e-9)
                curve.append({"step": step, "test": ok / n, "updates": net.updates, "near_miss_updates": net.nm,
                              "q_trail": trail / max(npos, 1),
                              "events_per_episode": (net.events - e0) / n, "synapses_per_episode": (net.syn - s0) / n})
                net.events, net.syn = e0, s0
        rows.append({"seed": s, "final": curve[-1], "curve": curve})
        print(json.dumps({"seed": s, **curve[-1]}), flush=True)
    with open(os.path.join(OUT, f"d{a.depth}_S{a.S}R{a.R}_t{a.thr:g}{f'_q{a.q:g}' if a.q != 0.25 else ''}{'_curve' if a.eval_at else ''}_{a.credit}_T{a.temp:g}_b{a.beta:g}{f'_m{a.margin:g}' if a.margin else ''}{f'_g{a.gate:g}' if a.gate else ''}{'_' + a.tag if a.tag else ''}.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "wall_s": round(time.time() - t0, 1)}, f)
    print("EXIT-OK", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
