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
    def __init__(self, N, K, depth, W, rng, budget=1.0, thr=0.5, part_hi=1.5, learn_win=0, scales=None, mult=0, recruit=0,
                 summed=0, alpha=1.0, beta=0.3, ibank=None):
        self.mult, self.recruit = mult, recruit
        self.summed, self.alpha, self.beta = summed, alpha, beta
        self.credit, self.temp, self.rng, self.margin, self.nm = "union", 0.0, rng, 0.0, 0
        self.gate = 0.0; self.ok_c = np.zeros(K); self.bad_c = np.zeros(K)
        self.learn_win = learn_win; self.learn_iv = 0; self.iv_min = 30
        self.N, self.K, self.depth, self.W, self.thr = N, K, depth, W, thr
        if depth == 2:
            base = [(p, q) for p in range(N) for q in range(N) if p != q]
            if ibank:                                        # §88: every interval [i·δ, j·δ] per pair, a fixed-window copy
                d_, mx = ibank; n_ = int(round(mx / d_))
                iv = [(i * d_, j * d_) for i in range(n_) for j in range(i + 1, n_ + 1)]
                self.pairs = [pq for _ in iv for pq in base]
                self.lo = np.concatenate([np.full(len(base), lo) for lo, _ in iv])
                self.hi = np.concatenate([np.full(len(base), hi) for _, hi in iv])
                P = len(self.pairs)
            else:
                scales = scales or [part_hi]                 # a bank of window scales per pair (no learning)
                self.pairs = [pq for w in scales for pq in base]
                P = len(self.pairs)
                self.lo = np.zeros(P); self.hi = np.concatenate([np.full(len(base), w) for w in scales])
        else:
            P = N
        self.P = P
        self.h = rng.uniform(0, 2 * budget / P, (K, P))     # hold-role weights
        self.g = rng.uniform(0, 2 * budget / P, (K, P))     # trigger-role weights
        if summed:
            self.h /= self.h.sum(1, keepdims=True); self.g /= self.g.sum(1, keepdims=True)
        self.updates = 0; self.events = 0; self.syn = 0

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

    def classes_summed(self, x):
        """§83: summed potentials. Class c fires at the first fired instant tau where the held input
        H = sum h[c, e] over parts fired in [tau - W, tau) and the coincident trigger input G = sum g[c, l] over parts
        firing at tau both exceed thr. Returns fire times, (fired parts, window matrix, instant index) for credit."""
        f = np.flatnonzero(np.isfinite(x)); f = f[np.argsort(x[f], kind="stable")]; xf = x[f]
        win = (xf[None, :] < xf[:, None]) & (xf[None, :] >= xf[:, None] - self.W)     # win[i, j]: j held at i
        same = xf[None, :] == xf[:, None]
        ft = np.full(self.K, INF); inst = np.full(self.K, -1)
        if len(f) >= 2:
            HS = self.h[:, f] @ win.T; GS = self.g[:, f] @ same.T
            ok = (HS > self.thr) & (GS > self.thr)
            anyc = ok.any(1); first = ok.argmax(1)
            ft[anyc] = xf[first[anyc]]; inst[anyc] = first[anyc]
            self.syn += int(((self.h[:, f] > 0.01) | (self.g[:, f] > 0.01)).sum())   # effective synapses read
        return ft, (f, win, same, inst)

    def _iv_update(self, t, units, positive):
        """§88: parametric duration learning at shared parts (version space). A correct fire extends each contributing
        part's observed positive lag range; a false fire moves a window edge halfway toward a lag outside that range,
        never excluding a lag seen in a positive."""
        if not self.learn_iv or len(units) == 0:
            return
        pp = np.array([self.pairs[u] for u in units]); dt = t[pp[:, 1]] - t[pp[:, 0]]
        for u, d in zip(units, dt):
            if positive:
                self.pos_lo[u] = min(self.pos_lo[u], d); self.pos_hi[u] = max(self.pos_hi[u], d); self.npos[u] += 1
            elif self.npos[u] >= self.iv_min:                # edges of the estimated support (uniform MVUE)
                ext = (self.pos_hi[u] - self.pos_lo[u]) / (self.npos[u] - 1)
                a_hat, b_hat = self.pos_lo[u] - ext, self.pos_hi[u] + ext
                if d > b_hat:
                    self.hi[u] = min(self.hi[u], (d + b_hat) / 2)
                elif d < a_hat:
                    self.lo[u] = max(self.lo[u], (d + a_hat) / 2)

    def _contrib(self, k, f, win, same, i):
        return np.r_[f[win[i]][self.h[k, f[win[i]]] > 0.05], f[same[i]][self.g[k, f[same[i]]] > 0.05]].astype(int)

    def _mul(self, W, c, idx, fac):                         # multiplicative update of a set, conserved budget
        if len(idx):
            W[c, idx] *= fac; W[c] /= W[c].sum()

    def teach_summed(self, t, y):
        c, x, ft, (f, win, same, inst) = self.forward(t)
        if c < self.K:                                      # §86b: the firing node's record
            self.ok_c[c] = 0.98 * self.ok_c[c] + (c == y); self.bad_c[c] = 0.98 * self.bad_c[c] + (c != y)
            if self.learn_iv:                               # §88: durations tuned at the contributing parts
                self._iv_update(t, self._contrib(c, f, win, same, inst[c]), c == y)
        prec = self.ok_c[y] / max(self.ok_c[y] + self.bad_c[y], 1e-9) if y < self.K else 0.0
        if c == y:
            if self.margin and y < self.K and prec >= self.gate:   # §86: near-miss credit keeps a margin
                i = inst[y]
                if self.h[y, f[win[i]]].sum() < self.margin:
                    self._mul(self.h, y, f[win[i]], 1 + self.alpha); self.nm += 1
                if self.g[y, f[same[i]]].sum() < self.margin:
                    self._mul(self.g, y, f[same[i]], 1 + self.alpha); self.nm += 1
            return False
        if y < self.K and not np.isfinite(ft[y]) and len(f) >= 2 and self.credit == "union":   # §83: every candidate
            self._mul(self.h, y, f[win.any(0)], 1 + self.alpha); self._mul(self.g, y, f[win.any(1)], 1 + self.alpha)
        elif y < self.K and not np.isfinite(ft[y]) and len(f) >= 2 and win.any():               # §84: one instant
            HS = win @ self.h[y, f]; GS = same @ self.g[y, f]; cand = win.any(1); score = np.minimum(HS, GS)
            if self.temp > 0:
                z = np.where(cand, (score - score[cand].max()) / self.temp, -np.inf); pr = np.exp(z)
                i = int(self.rng.choice(len(pr), p=pr / pr.sum()))
            else:
                i = int(np.argmax(np.where(cand, score, -1.0)))
            if HS[i] <= self.thr:
                self._mul(self.h, y, f[win[i]], 1 + self.alpha)
            if GS[i] <= self.thr:
                self._mul(self.g, y, f[same[i]], 1 + self.alpha)
        if c < self.K:                                      # false fire: demote every contributor at its instant
            i = inst[c]
            self._mul(self.h, c, f[win[i]], 1 - self.beta); self._mul(self.g, c, f[same[i]], 1 - self.beta)
        self.updates += 1
        return True

    def forward(self, t):
        x = self.parts(t)
        ft, used = self.classes_summed(x) if self.summed else self.classes(x)
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
    ap.add_argument("--learn-iv", type=float, default=0.0, help="§88: part windows start at [0, x] and are tuned (0 = off)")
    ap.add_argument("--ibank", default="", help="§88 interval bank 'delta,max': every window [i*delta, j*delta] per pair")
    ap.add_argument("--learn-win", type=int, default=0, help="1: part windows learned from pulled routes")
    ap.add_argument("--recruit", type=int, default=0, help="1: one-shot recruitment of classes with no live route")
    ap.add_argument("--mult", type=int, default=0, help="1: multiplicative (Winnow) routing pulls (§68)")
    ap.add_argument("--summed", type=int, default=0, help="1: summed hold/trigger potentials + full-information Winnow (§83)")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--credit", default="union", choices=("union", "instant"))
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.0, help="§86 near-miss margin (0 = off)")
    ap.add_argument("--gate", type=float, default=0.0, help="§86b precision gate for the margin")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
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
                      scales=[float(x) for x in a.scales.split(",")] if a.scales else None, mult=a.mult, recruit=a.recruit,
                      summed=a.summed, thr=a.thr, alpha=a.alpha, beta=a.beta,
                      ibank=tuple(float(x) for x in a.ibank.split(",")) if a.ibank else None)
        net.credit, net.temp, net.margin, net.gate = a.credit, a.temp, a.margin, a.gate
        if a.learn_iv:                                        # §88: broad initial windows, tuned by the version space
            net.learn_iv = 1; net.lo[:] = 0.0; net.hi[:] = a.learn_iv
            net.pos_lo = np.full(net.P, np.inf); net.pos_hi = np.full(net.P, -np.inf); net.npos = np.zeros(net.P)
        curve = []
        for step in range(1, a.steps + 1):
            t, y = T.sample(motifs, classes, a.N, H, q, rng)
            net.teach_summed(t, y) if a.summed else net.teach(t, y, a.eta)
            if a.lam and step % 1000 == 0:                    # sleep (§72): routing weights decay toward uniform
                for Wt in (net.h, net.g):
                    Wt += a.lam * (Wt.sum(1, keepdims=True) / Wt.shape[1] - Wt)
            if step % (a.steps // 5) == 0:
                ev = np.random.default_rng(99); n = 1500; ok = 0; e0 = net.events; s0 = net.syn
                for _ in range(n):
                    t, y = T.sample(motifs, classes, a.N, H, q, ev); ok += net.forward(t)[0] == y
                curve.append({"step": step, "test": ok / n, "updates": net.updates, "near_miss_updates": getattr(net, "nm", 0),
                              "events_per_episode": (net.events - e0) / n, "synapses_per_episode": (net.syn - s0) / n})
                net.events = e0; net.syn = s0
        rows.append({"seed": s, "final": curve[-1], "curve": curve})
        print(json.dumps({"seed": s, **curve[-1]}), flush=True)
    with open(os.path.join(OUT, f"d{a.depth}_K{a.K}_ph{('ib' + a.ibank.replace(',', '-')) if a.ibank else (a.scales.replace(",", "-") if a.scales else a.part_hi)}{'_lw' if a.learn_win else ''}{f'_lam{a.lam:g}' if a.lam else ''}{'_mult' if a.mult else ''}{'_rec' if a.recruit else ''}{f'_sum_t{a.thr:g}_a{a.alpha:g}_b{a.beta:g}' if a.summed else ''}{f'_{a.credit}_T{a.temp:g}_W{a.W:g}' if a.summed and a.credit == 'instant' else ''}{f'_m{a.margin:g}' if a.margin else ''}{f'_g{a.gate:g}' if a.gate else ''}{f'_iv{a.learn_iv:g}' if a.learn_iv else ''}{'_' + a.tag if a.tag else ''}.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "wall_s": round(time.time() - t0, 1)}, f)
    print("EXIT-OK", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
