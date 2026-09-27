"""E34: depth by composition in time: hold/trigger chains learned natively (THEORY §64, §65).

Task: E28's hierarchical motifs (N channels; M sub-motifs "j within [0.3, D] after i"; K classes = ordered pairs of
sub-motifs, the second starting 0.5–2.5 after the first ends; decoys = a class's motifs in reversed order).

Network (all nodes directional hold/trigger nodes, §64):
  parts    one node per ordered channel pair (p, q): fires at q's spike if q came within [lo, hi] after p; the
           window is [0, part_hi]; with --learn-win it tightens on parts used by pulled teacher routes (duration
           credit, §61). N(N-1) nodes, most of them useless.
  classes  node c fires at the spike of a part in its TRIGGER role if a part in its HOLD role fired within W before.
           Roles are learned weights: h[c, part], g[c, part] (conserved budgets). A single part cannot satisfy both
           roles (its hold must precede the trigger strictly), so a class node is a conjunction and an order.
  readout  classes race; a none node at the anchor.
Credit (errors only, native): teacher c missed -> among fired parts, the pair (earlier e, later l, l - e <= W) with the
largest current h[c, e] + g[c, l] is the counterfactual route: pull h[c, e] and g[c, l] (conserved). False winner c ->
weaken (conserved) the hold and trigger weights of the pair that fired it.
Depth-1 control (--depth 1): classes are hold/trigger nodes directly on channels.
"""
import argparse
import json
import os
import time

import numpy as np

import e28_routing as T

OUT = os.path.join(os.path.dirname(__file__), "results", "e34")
INF = np.inf


class Compose:
    def __init__(self, N, K, depth, W, rng, budget=1.0, thr=0.5, part_hi=1.5, learn_win=0, scales=None, mult=0, recruit=0):
        self.mult, self.recruit = mult, recruit
        self.learn_win = learn_win
        self.N, self.K, self.depth, self.W, self.thr = N, K, depth, W, thr
        if depth == 2:
            base = [(p, q) for p in range(N) for q in range(N) if p != q]
            scales = scales or [part_hi]                     # a bank of window scales per pair (no learning)
            self.pairs = [pq for w in scales for pq in base]
            P = len(self.pairs)
            self.lo = np.zeros(P); self.hi = np.concatenate([np.full(len(base), w) for w in scales])
        else:
            P = N
        self.P = P
        self.h = rng.uniform(0, 2 * budget / P, (K, P))     # hold-role weights
        self.g = rng.uniform(0, 2 * budget / P, (K, P))     # trigger-role weights
        self.updates = 0; self.events = 0

    def parts(self, t):
        if self.depth == 1:
            return t.copy()
        p = np.array([a for a, _ in self.pairs]); q = np.array([b for _, b in self.pairs])
        dt = t[q] - t[p]
        ok = np.isfinite(dt) & (dt >= self.lo) & (dt <= self.hi)
        return np.where(ok, t[q], INF)

    def classes(self, x):
        """class c fires at the earliest trigger part l whose g[c,l] > thr and some hold part e with h[c,e] > thr fired
        in [x_l - W, x_l). Returns fire times and the (e, l) pair used."""
        fired = np.flatnonzero(np.isfinite(x))
        ft = np.full(self.K, INF); used = [None] * self.K
        if len(fired) < 2:
            return ft, used
        order = fired[np.argsort(x[fired])]
        for c in range(self.K):
            hs = order[self.h[c, order] > self.thr]
            if not len(hs):
                continue
            for l in order[self.g[c, order] > self.thr]:
                e = hs[(x[hs] < x[l]) & (x[hs] >= x[l] - self.W)]
                if len(e):
                    ft[c] = x[l]; used[c] = (int(e[np.argmax(self.h[c, e])]), int(l))
                    break
        return ft, used

    def forward(self, t):
        x = self.parts(t)
        ft, used = self.classes(x)
        self.events += int(np.isfinite(t).sum() + np.isfinite(x).sum() + np.isfinite(ft).sum())
        c = int(ft.argmin()) if np.isfinite(ft).any() else self.K
        return c, x, ft, used

    def _pull(self, W, c, j, eta):                          # conserved pull of synapse j on node c
        tot = W[c].sum()
        if self.mult:                                       # Winnow (§68): multiplicative, then renormalize;
            W[c, j] *= 1 + eta * 10                          # concentrates on consistently pulled synapses
            W[c, j] = max(W[c, j], eta * tot / 10)           # (a floor so a never-pulled synapse can start)
        else:
            W[c, j] += eta * tot
        W[c] *= tot / W[c].sum()

    def _drop(self, W, c, j, eta):                          # conserved weakening
        tot = W[c].sum(); W[c, j] *= 1 - eta; W[c] *= tot / max(W[c].sum(), 1e-12)

    def teach(self, t, y, eta):
        c, x, ft, used = self.forward(t)
        if c == y:
            return False
        if y < self.K and not np.isfinite(ft[y]):          # teacher missed: pull its best counterfactual route
            fired = np.flatnonzero(np.isfinite(x))
            if len(fired) >= 2:
                xe, xl = x[fired][:, None], x[fired][None, :]
                ok = (xe < xl) & (xe >= xl - self.W)
                if ok.any():
                    score = self.h[y, fired][:, None] + self.g[y, fired][None, :]
                    score = np.where(ok, score, -INF)
                    i, j = np.unravel_index(np.argmax(score), score.shape)
                    self._pull(self.h, y, fired[i], eta); self._pull(self.g, y, fired[j], eta)
                    if self.recruit and not ((self.h[y] > self.thr).any() and (self.g[y] > self.thr).any()):
                        for Wt, part in ((self.h, fired[i]), (self.g, fired[j])):   # one-shot recruitment of a
                            tot = Wt[y].sum()                                        # dead class: its route is
                            Wt[y] *= (tot - 1.5 * self.thr) / max(tot - Wt[y, part], 1e-9)  # set above threshold
                            Wt[y, part] = 1.5 * self.thr
                    if self.depth == 2 and self.learn_win:        # duration credit (§61): parts on the pulled route
                        for part in (fired[i], fired[j]):          # tighten their window toward what they observed
                            pp, qq = self.pairs[part]
                            dt = t[qq] - t[pp]
                            self.hi[part] += 0.2 * (1.2 * dt + 0.1 - self.hi[part])
        if c < self.K and c != y:                           # false winner: weaken the route that fired it
            e, l = used[c]
            self._drop(self.h, c, e, eta); self._drop(self.g, c, l, eta)
        self.updates += 1
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--W", type=float, default=3.5, help="class hold window (covers the gap between parts)")
    ap.add_argument("--part-hi", type=float, default=1.5, help="part-node window [0, part_hi] (fixed)")
    ap.add_argument("--scales", default="", help="comma list: bank of part-window scales, e.g. 1,2,4")
    ap.add_argument("--learn-win", type=int, default=0, help="1: part windows learned from pulled routes")
    ap.add_argument("--recruit", type=int, default=0, help="1: one-shot recruitment of classes with no live route")
    ap.add_argument("--mult", type=int, default=0, help="1: multiplicative (Winnow) routing pulls (§68)")
    ap.add_argument("--lam", type=float, default=0.0, help="sleep: routing-weight decay per 1000 episodes (§72)")
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    H, q, rows = 12.0, 0.25, []
    t0 = time.time()
    for s in range(a.seeds):
        rng = np.random.default_rng(s)
        motifs, classes = T.make_task(a.N, a.M, a.K, rng)
        net = Compose(a.N, a.K, a.depth, a.W, rng, part_hi=a.part_hi, learn_win=a.learn_win,
                      scales=[float(x) for x in a.scales.split(",")] if a.scales else None, mult=a.mult, recruit=a.recruit)
        curve = []
        for step in range(1, a.steps + 1):
            t, y = T.sample(motifs, classes, a.N, H, q, rng)
            net.teach(t, y, a.eta)
            if a.lam and step % 1000 == 0:                    # sleep (§72): routing weights decay toward uniform
                for Wt in (net.h, net.g):
                    Wt += a.lam * (Wt.sum(1, keepdims=True) / Wt.shape[1] - Wt)
            if step % (a.steps // 5) == 0:
                ev = np.random.default_rng(99); n = 1500; ok = 0; e0 = net.events
                for _ in range(n):
                    t, y = T.sample(motifs, classes, a.N, H, q, ev); ok += net.forward(t)[0] == y
                curve.append({"step": step, "test": ok / n, "updates": net.updates,
                              "events_per_episode": (net.events - e0) / n})
                net.events = e0
        rows.append({"seed": s, "final": curve[-1], "curve": curve})
        print(json.dumps({"seed": s, **curve[-1]}), flush=True)
    with open(os.path.join(OUT, f"d{a.depth}_K{a.K}_ph{a.scales.replace(",", "-") if a.scales else a.part_hi}{'_lw' if a.learn_win else ''}{f'_lam{a.lam:g}' if a.lam else ''}{'_mult' if a.mult else ''}{'_rec' if a.recruit else ''}{'_' + a.tag if a.tag else ''}.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "wall_s": round(time.time() - t0, 1)}, f)
    print("EXIT-OK", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
