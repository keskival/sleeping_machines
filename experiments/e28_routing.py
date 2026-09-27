"""E28: depth in time needs counterfactual routing credit; can cancelled near-misses supply it for free? (THEORY §57)

Task (hierarchical temporal motifs): N input channels, each spiking at most once in an episode of length H.
M sub-motifs, each "channel j within [0.3, Δ] after channel i". K classes, each an ORDERED pair of sub-motifs
(m_a, then m_b starting 0.5–2.5 after m_a ends); classes share sub-motifs and differ in order, so a class is a
composition: stage 1 has to detect sub-motifs, stage 2 their order. Distractor channels spike at random.

Network: every node is a windowed integrate-to-threshold unit: it fires at the first time the weights of the
inputs that arrived within the last W reach 1. Synapses carry a weight in [0, 1] and a delay.
  stage 1   Hn hidden nodes on all N channels, in groups of G that race: the first k in a group fire, the rest are
            cancelled and keep their near-miss (the largest windowed charge they reached before cancellation).
  stage 2   K class nodes on all hidden nodes; they race; a "none" node fires at the anchor H + 1.
  depth 1   (--depth 1) the class nodes sit directly on the channels.

Learning: only on errors, pull-only (§56.5). Output: the teacher's synapses from inputs that arrived are
strengthened and their delays moved toward its latest arrival, so they coincide (critical path + coincidence).
Hidden credit ("which route should have been taken"), by arm:
  path      only hidden nodes that fired and feed the teacher (w2 > 0.5) are pulled: timing along the taken route
  fired     + fired hidden nodes that don't yet feed the teacher get w2 strengthened: top-k MoE credit, which
              needs the k winners to fire (k events per group)
  nearmiss  + cancelled hidden nodes the teacher wants (w2 > 0.5; under --fix60, > 2x the node's mean) whose near-miss was above 1/2 are pulled so
              they win their group next time: counterfactual routing from cancellation, no extra events
  recur     hidden layer learns label-free: every hidden winner pulls its own firing window (conserved), on
            every episode; labels train only the readout. Parts from recurrence (§62)
  push      = nearmiss, but a wrong winner is displaced in time (its delays lengthened) instead of specialized
A wrong winner is otherwise specialized: the synapses in its firing window are weakened.
"""
import argparse
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e28")
INF = np.inf


def make_task(N, M, K, rng):
    motifs = []
    for _ in range(M):
        i, j = rng.choice(N, 2, replace=False)
        motifs.append((int(i), int(j), float(rng.uniform(0.8, 1.5))))
    classes, seen = [], set()
    while len(classes) < K:
        a, b = (int(x) for x in rng.choice(M, 2, replace=False))
        if (a, b) not in seen:
            seen.add((a, b)); classes.append((a, b))
    return motifs, classes


def plant(t, motif, start, rng):
    i, j, D = motif
    t[i] = start
    t[j] = start + rng.uniform(0.3, D)
    return t[j]


def holds(motifs, cls, t):
    """Class holds if motif a occurs, then motif b starts 0.5–2.5 after a ends."""
    def occ(m):
        i, j, D = motifs[m]
        return (t[i], t[j]) if np.isfinite(t[i]) and np.isfinite(t[j]) and 0.3 <= t[j] - t[i] <= D else None
    a, b = occ(cls[0]), occ(cls[1])
    return a is not None and b is not None and 0.5 <= b[0] - a[1] <= 2.5


def sample(motifs, classes, N, H, q, rng):
    K = len(classes)
    for _ in range(1000):
        y = int(rng.integers(K + 1))
        t = np.where(rng.random(N) < q, rng.uniform(0, H, N), INF)
        if y < K:
            a, b = classes[y]
            end = plant(t, motifs[a], rng.uniform(0, H - 6), rng)
            plant(t, motifs[b], end + rng.uniform(0.5, 2.5), rng)
        elif rng.random() < 0.5:                               # decoy: a class's motifs in the wrong order
            a, b = classes[int(rng.integers(K))]
            end = plant(t, motifs[b], rng.uniform(0, H - 6), rng)
            plant(t, motifs[a], end + rng.uniform(0.5, 2.5), rng)
        true = [k for k in range(K) if holds(motifs, classes[k], t)]
        if (y < K and true == [y]) or (y == K and not true):
            return t, y
    raise RuntimeError("sampling failed")


def integrate(arr, w, W, thr=1.0):
    """Windowed integrate-to-threshold. arr: arrival times per synapse, w: weights. Returns (fire time, charge
    reached, the synapses in the firing window)."""
    ok = np.isfinite(arr) & (w > 0.02)
    if not ok.any():
        return INF, 0.0, np.array([], int)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(arr[idx])]
    best, bwin, lo, s = 0.0, idx[:0], 0, 0.0
    for hi in range(len(idx)):
        s += w[idx[hi]]
        while arr[idx[hi]] - arr[idx[lo]] > W:
            s -= w[idx[lo]]; lo += 1
        if s >= thr:
            return arr[idx[hi]], s, idx[lo:hi + 1]
        if s > best:
            best, bwin = s, idx[lo:hi + 1]
    return INF, best, bwin                                     # a near miss keeps its best partial window


class Net:
    def __init__(self, N, K, H, depth, Hn, G, k, W, rng, fix60=0, kappa=0.02):
        self.fix60, self.kappa = fix60, kappa
        self.thr = np.ones(K)
        self.N, self.K, self.H, self.depth, self.G, self.k, self.W = N, K, H, depth, G, k, W
        nin = N if depth == 1 else Hn
        self.Hn = Hn
        if depth == 2:
            self.w1 = rng.uniform(0, 0.8, (Hn, N)); self.d1 = rng.uniform(0, 2, (Hn, N))
        self.w2 = rng.uniform(0, 0.4, (K, nin)); self.d2 = rng.uniform(0, 2, (K, nin))
        self.events = 0; self.updates = 0

    def hidden(self, t):
        ft = np.full(self.Hn, INF); charge = np.zeros(self.Hn); win = [None] * self.Hn
        for h in range(self.Hn):
            ft[h], charge[h], win[h] = integrate(t + self.d1[h], self.w1[h], self.W)
        fired = np.zeros(self.Hn, bool)
        for g0 in range(0, self.Hn, self.G):                   # group races: first k fire, the rest cancelled
            grp = np.arange(g0, min(g0 + self.G, self.Hn))
            order = grp[np.argsort(ft[grp])][:self.k]
            fired[order[np.isfinite(ft[order])]] = True
        out = np.where(fired, ft, INF)
        return out, fired, charge, win

    def forward(self, t):
        st = {}
        if self.depth == 2:
            x, fired, charge, win = self.hidden(t)
            st.update(fired=fired, charge=charge, win=win)
        else:
            x = t
        ft = np.full(self.K, INF); owin = [None] * self.K
        for c in range(self.K):
            ft[c], _, owin[c] = integrate(x + self.d2[c], self.w2[c], self.W if not self.fix60 else np.inf,
                                          self.thr[c])
        c = int(ft.argmin()) if ft.min() < self.H + 1 else self.K
        self.events += int(np.isfinite(t).sum() + np.isfinite(x).sum() * (self.depth == 2) + np.isfinite(ft).sum())
        st.update(x=x, ft=ft, owin=owin, winner=c)
        return c, st

    def _pull1(self, h, wi, eta):
        if self.fix60:                                          # §60 for hidden nodes too: conserved budget
            tot = self.w1[h].sum(); self.w1[h, wi] += eta * tot / len(wi); self.w1[h] *= tot / self.w1[h].sum()
        else:
            self.w1[h, wi] += eta

    def teach(self, t, y, eta, arm):
        c, st = self.forward(t)
        if arm == "recur" and self.depth == 2:                  # parts from recurrence: every hidden winner pulls
            for h in np.flatnonzero(st["fired"]):               # its own firing window, label-free, conserved
                wi = st["win"][h]
                if len(wi):
                    self._pull1(h, wi, eta * 0.25)
        if c == y:
            return False
        x = st["x"]
        if y < self.K:                                          # pull the teacher's route (critical path)
            arr = x + self.d2[y]
            got = np.isfinite(arr)
            wants = self.w2[y] > (2 * self.w2[y].mean() if self.fix60 else 0.5)   # teacher wants h (relative
            if self.depth == 1 or arm != "path":                                # to its budget under §60)
                cand = got
            else:
                cand = got & wants
            if cand.any():
                target = np.min(arr[cand])                      # pull the late arrivals earlier, to coincide
                if self.fix60:                                  # §60: conserved budget, fractional step
                    tot = self.w2[y].sum(); self.w2[y, cand] += eta * tot / cand.sum(); self.w2[y] *= tot / self.w2[y].sum()
                else:
                    self.w2[y, cand] += eta
                self.d2[y, cand] += 0.5 * (target - arr[cand])
            if self.depth == 2 and arm in ("nearmiss", "push"):
                want = (~st["fired"]) & wants & (st["charge"] > 0.5)
                for h in np.flatnonzero(want):                  # counterfactual route: make it win its group,
                    wi = st["win"][h]                           # on the partial coincidence that nearly fired it
                    if len(wi):
                        self._pull1(h, wi, eta)
            if self.depth == 2 and arm not in ("path", "recur"):             # fired contributors: sharpen their own inputs
                for h in np.flatnonzero(st["fired"] & wants):
                    wi = st["win"][h]
                    if len(wi):
                        self._pull1(h, wi, eta * 0.5)
        if self.fix60:                                          # §60 prices: false winner dearer, missed cheaper
            if y < self.K: self.thr[y] -= self.kappa
            if c < self.K and c != y: self.thr[c] += self.kappa
            np.clip(self.thr, 0.2, 5.0, out=self.thr)
        if c < self.K and c != y:                               # a wrong class fired
            if arm == "push":                                   # displace it in time (the loser push)
                self.d2[c] += eta
            else:                                               # specialize it: drop the synapses that fired it
                wi = st["owin"][c]
                if self.fix60:                                  # conserving specialization
                    tot = self.w2[c].sum(); self.w2[c, wi] *= 1 - eta
                    self.w2[c] *= tot / max(self.w2[c].sum(), 1e-9)
                else:
                    self.w2[c, wi] -= eta
        np.clip(self.w2, 0, 1, out=self.w2); np.maximum(self.d2, 0, out=self.d2)
        if self.depth == 2:
            np.clip(self.w1, 0, 1, out=self.w1); np.maximum(self.d1, 0, out=self.d1)
        self.updates += 1
        return True


def evaluate(net, motifs, classes, N, H, q, n=1500):
    rng = np.random.default_rng(99)
    return float(np.mean([net.forward(t)[0] == y for t, y in (sample(motifs, classes, N, H, q, rng)
                                                               for _ in range(n))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--H", type=float, default=12.0)
    ap.add_argument("--q", type=float, default=0.25)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--W", type=float, default=0.6)
    ap.add_argument("--fix60", type=int, default=0, help="1: §60 readout (non-leaky, conserved, priced)")
    ap.add_argument("--kappa", type=float, default=0.02)
    ap.add_argument("--arm", default="nearmiss", choices=("path", "fired", "nearmiss", "push", "recur"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0, rows = time.time(), []
    for s in range(a.seeds):
        rng = np.random.default_rng(s)
        motifs, classes = make_task(a.N, a.M, a.K, rng)
        net = Net(a.N, a.K, a.H, a.depth, a.hidden, a.group, a.k, a.W, rng, a.fix60, a.kappa)
        curve = []
        for step in range(1, a.steps + 1):
            t, y = sample(motifs, classes, a.N, a.H, a.q, rng)
            net.teach(t, y, a.eta, a.arm)
            if step % (a.steps // 10) == 0:
                ev0 = net.events
                acc = evaluate(net, motifs, classes, a.N, a.H, a.q)
                curve.append({"step": step, "test": acc, "updates": net.updates,
                              "events_per_episode": (net.events - ev0) / 1500})
                net.events = ev0
        rows.append({"seed": s, "final": curve[-1], "curve": curve})
        print(json.dumps({"seed": s, **curve[-1]}), flush=True)
    name = f"d{a.depth}_{a.arm}_k{a.k}{'_f60' if a.fix60 else ''}{'_' + a.tag if a.tag else ''}.json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "wall_s": round(time.time() - t0, 1)}, f)
    print("EXIT-OK", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
