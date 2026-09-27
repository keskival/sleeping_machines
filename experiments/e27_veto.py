"""E27: learning temporal patterns with delays, coincidence windows and veto (THEORY §56).

Task: N input channels, each spiking at most once in an episode of length H. K patterns; pattern k holds when
channel i_k spikes, channel j_k spikes within Delta_k after it, and channel v_k does NOT spike in between.
Exactly one pattern holds per episode, or none. The network must name it.

Network (native primitives only):
  detector k   coincidence node (2-of-2 with window): its two excitatory synapses carry learnable delays
               d1, d2 and it fires at the second arrival if the first arrived at most w_k before (w_k = its
               integration window, learnable). Which channels feed it is given (i_k, j_k): E27 is about timing.
  veto         every channel n has an inhibitory synapse onto every detector with learnable delay u[k, n] and
               learnable strength g[k, n] in [0, 1]; it blocks the detector if it arrives while the detector is
               armed (after the first excitatory arrival, before the second) and g > 1/2.
  readout      the detectors race; the first to fire wins. A "none" node fires at the anchor H + margin (the
               reference, the end of the episode). Events per episode: input spikes + fired detectors.

Learning happens only on errors and only on the causal chain of the event that should have won (§56.5):
  teacher k did not fire, no coincidence   align: move its delays together; hold (--tol hold): lengthen its
                                           PSP window until the second input falls inside it
  teacher k did not fire, it was vetoed     weaken the veto synapses that blocked it (they were on its chain)
  a wrong detector c fired  (--fix veto)   specialize c by inhibition: strengthen the veto synapses of channels
                                           that spiked while c was armed and move their delays into that span
                            (--fix push)   displace c in time instead: lengthen its delays (the §54 push)
"""
import argparse
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e27")
INF = np.inf


def make_task(N, K, H, rng):
    ch = rng.permutation(N)
    pats = []
    for k in range(K):
        i, j, v = rng.choice(N, 3, replace=False)
        pats.append((int(i), int(j), int(v), float(rng.uniform(1.0, 3.0))))
    return pats


def holds(pat, t):
    i, j, v, D = pat
    if not (np.isfinite(t[i]) and np.isfinite(t[j])):
        return False
    if not (0 < t[j] - t[i] <= D):
        return False
    return not (t[i] < t[v] < t[j])


def sample(pats, N, H, q, rng):
    """One episode with exactly one true pattern (or none), by rejection."""
    K = len(pats)
    want = int(rng.integers(K + 1))                       # K = none
    for _ in range(1000):
        t = np.where(rng.random(N) < q, rng.uniform(0, H, N), INF)
        near = want == K and rng.random() < 0.5              # half the "none" episodes are vetoed near-misses
        k = want if want < K else int(rng.integers(K))
        if want < K or near:                                # plant the pattern (or its vetoed near-miss)
            i, j, v, D = pats[k]
            t[i] = rng.uniform(0, H - D)
            t[j] = t[i] + rng.uniform(0.05, D)
            t[v] = rng.uniform(t[i], t[j]) if near else (INF if t[i] < t[v] < t[j] else t[v])
        true = [k for k in range(K) if holds(pats[k], t)]
        if (want < K and true == [want]) or (want == K and not true):
            return t, want
    raise RuntimeError("could not sample")


class Net:
    def __init__(self, pats, N, H, rng, tol="align"):
        self.pats, self.N, self.H, K = pats, N, H, len(pats)
        self.tol = tol
        self.d = rng.uniform(0, 3, (K, 2)) if tol == "align" else np.zeros((K, 2))   # excitatory delays
        self.w = np.full(K, 0.5)                            # integration windows
        self.u = rng.uniform(0, 3, (K, N))                  # veto delays
        self.g = np.full((K, N), 0.25)                      # veto strengths (active above 1/2)
        self.updates = 0
        self.events = 0

    def detector(self, k, t):
        """Returns (fire time or INF, info) for detector k on input times t."""
        i, j = self.pats[k][0], self.pats[k][1]
        a1, a2 = t[i] + self.d[k, 0], t[j] + self.d[k, 1]
        first, second = min(a1, a2), max(a1, a2)
        if not np.isfinite(second) or second - first > self.w[k]:
            return INF, ("nocoinc", a1, a2)
        veto_t = t + self.u[k]
        blockers = np.flatnonzero((self.g[k] > 0.5) & (veto_t > first) & (veto_t < second))
        if len(blockers):
            return INF, ("vetoed", blockers, first, second)
        return second, ("fired", first, second)

    def forward(self, t):
        times, infos = [], []
        for k in range(len(self.pats)):
            ft, info = self.detector(k, t)
            times.append(ft); infos.append(info)
        times = np.array(times)
        self.events += int(np.isfinite(t).sum() + np.isfinite(times).sum())
        win = int(times.argmin()) if times.min() < self.H + 1 else len(self.pats)
        return win, times, infos

    def teach(self, t, y, eta, fix):
        win, times, infos = self.forward(t)
        if win == y:
            return False
        K = len(self.pats)
        if y < K and not np.isfinite(times[y]):             # the teacher should have fired
            kind = infos[y][0]
            if kind == "nocoinc":
                _, a1, a2 = infos[y]
                gap = a2 - a1
                if self.tol == "hold":                      # hold: lengthen the PSP until B falls inside it
                    self.w[y] += eta * max(abs(gap) - self.w[y], 0) + eta * 0.1
                else:                                       # align: bring the two arrivals together (pull)
                    self.d[y, 0] += eta * np.sign(gap) * 0.5
                    self.d[y, 1] -= eta * np.sign(gap) * 0.5
                    self.w[y] += eta * 0.1
                    np.maximum(self.d, 0, out=self.d)
            else:                                            # wrongly vetoed: the blockers were on its chain
                blockers = infos[y][1]
                self.g[y, blockers] -= eta
        if win < K and win != y:                            # a wrong detector fired
            c = win
            _, first, second = infos[c]
            if self.tol == "hold" and second - first > 0.5 * self.w[c]:
                # a late partner fired it: shorten the hold toward just below that span (duration credit, §61)
                self.w[c] += eta * ((second - first) * 0.95 - self.w[c])
            if fix == "veto":                                # specialize it by inhibition
                i, j = self.pats[c][0], self.pats[c][1]
                lo, hi = min(t[i], t[j]), max(t[i], t[j])     # channels that spiked inside the armed interval
                armed = np.flatnonzero(np.isfinite(t) & (t > lo) & (t < hi))
                armed = armed[[n not in (i, j) for n in armed]]
                if len(armed):
                    mid = 0.5 * (first + second)                # land the veto inside the armed span
                    self.g[c, armed] += eta
                    self.u[c, armed] += 0.5 * (mid - (t[armed] + self.u[c, armed]))
                    np.maximum(self.u, 0, out=self.u)
                else:
                    self.w[c] *= 1 - eta                    # nothing to veto with: narrow its window
            else:                                            # displace it in time (the push)
                self.d[c] += eta
        np.clip(self.g, 0, 1, out=self.g)
        np.clip(self.w, 0.05, 5, out=self.w)
        self.updates += 1
        return True


def evaluate(net, pats, N, H, q, rng, n=2000):
    ok = 0
    for _ in range(n):
        t, y = sample(pats, N, H, q, rng)
        ok += net.forward(t)[0] == y
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--H", type=float, default=10.0)
    ap.add_argument("--q", type=float, default=0.4, help="spike probability per distractor channel")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--fix", default="veto", choices=("veto", "push"))
    ap.add_argument("--tol", default="align", choices=("align", "hold"),
                    help="temporal tolerance by aligning delays, or by holding a long PSP (§61)")
    ap.add_argument("--noveto", type=int, default=0, help="1: veto synapses disabled (monotone network)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0, rows = time.time(), []
    for s in range(a.seeds):
        rng = np.random.default_rng(s)
        pats = make_task(a.N, a.K, a.H, rng)
        net = Net(pats, a.N, a.H, rng, a.tol)
        curve = []
        for step in range(1, a.steps + 1):
            t, y = sample(pats, a.N, a.H, a.q, rng)
            net.teach(t, y, a.eta, a.fix)
            if a.noveto:
                net.g[:] = 0
            if step % (a.steps // 10) == 0:
                curve.append({"step": step, "test": evaluate(net, pats, a.N, a.H, a.q, np.random.default_rng(99)),
                              "updates": net.updates})
        r = {"seed": s, "final": curve[-1], "curve": curve, "events_per_episode": net.events / (a.steps + 20000)}
        rows.append(r)
        print(json.dumps({"seed": s, **curve[-1]}), flush=True)
    name = f"N{a.N}_K{a.K}_{a.fix}{'_hold' if a.tol == 'hold' else ''}{'_noveto' if a.noveto else ''}{'_' + a.tag if a.tag else ''}.json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "wall_s": round(time.time() - t0, 1)}, f)
    print("EXIT-OK", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
