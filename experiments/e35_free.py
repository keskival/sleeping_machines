"""E35: E27's task with nothing given: each detector learns its channels, its hold duration and its vetoes.

E27 told each detector which two channels were its A and B; only timing and veto were learned. Here a detector sees
all N channels through three sets of learned weights, exactly the input a dense model gets:
  hold     hA[k, n]  (conserved budget)  a hold channel's spike opens a PSP of length w[k]
  trigger  hB[k, n]  (conserved budget)  a trigger channel's spike fires the detector if a hold PSP is open
  veto     g[k, n]   (strength in [0, 1]) a veto channel spiking inside the held interval blocks it
Detectors race; none fires -> "none". Learning is errors-only and local:
  teacher k missed -> among spiked channel pairs (e before l), pull the pair with the largest hA[k,e] + hB[k,l]
      (conserved); if that pair came later than w[k], lengthen w[k] toward it (duration credit); if it was blocked by
      vetoes, weaken those vetoes.
  false winner c   -> strengthen veto on channels that spiked inside its held interval; if there are none, shorten
      w[c] below the interval; and weaken its route a little (conserved), since its channels may be wrong.
"""
import argparse
import json
import os
import time

import numpy as np

import e27_veto as E27

OUT = os.path.join(os.path.dirname(__file__), "results", "e35")
INF = np.inf


class Free:
    def __init__(self, N, K, rng, thr=0.5, w0=0.5):
        self.N, self.K, self.thr = N, K, thr
        self.hA = rng.uniform(0, 2.0 / N, (K, N)); self.hB = rng.uniform(0, 2.0 / N, (K, N))
        self.w = np.full(K, w0); self.g = np.full((K, N), 0.25)
        self.updates = 0

    def detector(self, k, t):
        sp = np.flatnonzero(np.isfinite(t))
        if len(sp) < 2:
            return INF, None
        hold = sp[self.hA[k, sp] > self.thr]; trig = sp[self.hB[k, sp] > self.thr]
        vet = sp[self.g[k, sp] > 0.5]
        for l in trig[np.argsort(t[trig])]:
            e = hold[(t[hold] < t[l]) & (t[hold] >= t[l] - self.w[k])]
            for ee in e[np.argsort(-t[e])]:                     # latest opening hold first
                if not ((t[vet] > t[ee]) & (t[vet] < t[l]) & (vet != ee) & (vet != l)).any():
                    return t[l], (int(ee), int(l))
        return INF, None

    def forward(self, t):
        ft, used = zip(*(self.detector(k, t) for k in range(self.K)))
        ft = np.array(ft)
        c = int(ft.argmin()) if np.isfinite(ft).any() else self.K
        return c, ft, used

    def _pull(self, W, k, j, eta):
        tot = W[k].sum(); W[k, j] += eta * tot; W[k] *= tot / W[k].sum()

    def _drop(self, W, k, j, eta):
        tot = W[k].sum(); W[k, j] *= 1 - eta; W[k] *= tot / max(W[k].sum(), 1e-12)

    def teach(self, t, y, eta):
        c, ft, used = self.forward(t)
        if c == y:
            return False
        sp = np.flatnonzero(np.isfinite(t))
        if y < self.K and not np.isfinite(ft[y]) and len(sp) >= 2:
            te, tl = t[sp][:, None], t[sp][None, :]
            score = np.where(te < tl, self.hA[y, sp][:, None] + self.hB[y, sp][None, :], -INF)
            i, j = np.unravel_index(np.argmax(score), score.shape)
            e, l = sp[i], sp[j]
            self._pull(self.hA, y, e, eta); self._pull(self.hB, y, l, eta)
            gap = t[l] - t[e]
            if gap > self.w[y]:                                  # the partner came too late: hold longer
                self.w[y] += eta * (gap * 1.05 - self.w[y]) + 0.01
            blk = sp[(t[sp] > t[e]) & (t[sp] < t[l]) & (self.g[y, sp] > 0.5) & (sp != e) & (sp != l)]
            self.g[y, blk] -= eta                                # release vetoes that blocked the teacher
        if c < self.K and c != y:
            e, l = used[c]
            inside = sp[(t[sp] > t[e]) & (t[sp] < t[l]) & (sp != e) & (sp != l)]
            if len(inside):
                self.g[c, inside] += eta                          # specialize by veto
            else:
                self.w[c] += eta * ((t[l] - t[e]) * 0.95 - self.w[c])   # shorten the hold
            self._drop(self.hA, c, e, eta * 0.5); self._drop(self.hB, c, l, eta * 0.5)
        np.clip(self.g, 0, 1, out=self.g); np.clip(self.w, 0.05, 6.0, out=self.w)
        self.updates += 1
        return True

    def synaptic_events(self, t, ft):
        """events delivered: each spike to the detectors where it has a live synapse (any role), + detector spikes."""
        sp = np.flatnonzero(np.isfinite(t))
        live = (self.hA[:, sp] > self.thr) | (self.hB[:, sp] > self.thr) | (self.g[:, sp] > 0.5)
        return int(live.sum()) + int(np.isfinite(ft).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12, help="channels (more = more distractors)")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    N, K, H, q = a.N, 4, 10.0, 0.4 * 12 / a.N              # same expected number of spikes per episode
    rows, t0 = [], time.time()
    for s in range(a.seeds):
        rng = np.random.default_rng(s)
        pats = E27.make_task(N, K, H, rng)                     # same tasks and seeds as E27 / E32
        for _ in range(20):                                    # redraw task sets in which some class cannot be
            try:                                               # sampled (two patterns sharing their A and B
                chk = np.random.default_rng(12345)             # channels); leaves every earlier task unchanged
                [E27.sample(pats, N, H, q, chk) for _ in range(300)]
                break
            except RuntimeError:
                pats = E27.make_task(N, K, H, rng)
        net = Free(N, K, rng)
        curve = []
        for step in range(1, a.steps + 1):
            tt, y = E27.sample(pats, N, H, q, rng)
            net.teach(tt, y, a.eta)
            if step % (a.steps // 10) == 0:
                ev = np.random.default_rng(99); ok = 0; syn = 0
                for _ in range(2000):
                    tt, y = E27.sample(pats, N, H, q, ev)
                    c, ft, _ = net.forward(tt); ok += c == y; syn += net.synaptic_events(tt, ft)
                curve.append({"step": step, "test": ok / 2000, "updates": net.updates,
                              "synaptic_events": syn / 2000})
        rows.append({"seed": s, "final": curve[-1], "curve": curve})
        print(json.dumps({"seed": s, **curve[-1]}), flush=True)
    with open(os.path.join(OUT, f"free_N{a.N}{'_' + a.tag if a.tag else ''}.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "wall_s": round(time.time() - t0, 1)}, f)
    print("EXIT-OK", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
