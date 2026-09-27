"""E29: true grokking test (THEORY §58): a general race network that can memorize, given recurrent delay loops.

Task: E24's. (a + b) mod p; operand lines a and p + b each spike once at t = 0; a fraction of pairs trains.

Network (every node a windowed integrate-to-threshold unit, as in E28; synapses carry weight and delay):
  loops    R recurrent delay loops with learnable periods P_r (random init, nothing tuned to p). Operand line a
           injects into loop r after a learnable delay d[r, a]; the loop re-emits a spike every P_r (a lap)
           until the horizon.
  hidden   Hn nodes in racing groups (k winners, near-miss kept). Each has synapses from every loop (one delay
           per loop, applied to every lap) and from every b-line and every a-line (a delay each). It fires when
           the windowed weight of arrivals reaches 1, e.g. a loop lap coinciding with a b-spike.
  output   p class nodes accumulate hidden evidence without leak (the main race node); they race; none at the
           anchor. Coincidence windows are used only where timing is the computation, in the hidden layer.
Capacity: R·p + Hn·(R + 2p) + p·Hn delays and as many weights, against n training pairs (rho = n / params).
Without loops (--loops 0) the hidden layer can only react to operand coincidences: a pair memorizer.

Learning (native, E28's rule): errors only; the teacher output pulls its arrived hidden inputs (weights up, late
arrivals earlier); near-miss routing: the cancelled hidden node the teacher wants with the best partial window
is pulled on that window; arrivals in a pulled window move toward each other (the partner, not 'earlier'); a loop's period moves by
its lap count (d(arrival)/dP = laps, §56.4); every pull conserves the node's total synaptic weight
(heterosynaptic: the node's other synapses pay); a false winner is specialized (its firing-window synapses weakened).
"""
import argparse
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e29")
INF = np.inf


def split(p, frac, seed):
    a, b = np.divmod(np.arange(p * p), p)
    perm = np.random.default_rng(seed).permutation(p * p)
    n = int(round(frac * p * p))
    return (a[perm[:n]], b[perm[:n]]), (a[perm[n:]], b[perm[n:]])


def integrate(arr, w, W, thr=1.0, v=None):
    """arr, w: flat arrays of arrival times and weights (a synapse may contribute several arrivals).
    Returns fire time, charge, the arrivals in the firing (or best partial) window."""
    eff = w if v is None else w - v                            # inhibitory synapses subtract (veto)
    ok = np.isfinite(arr) & (np.abs(eff) > 0.02)
    if not ok.any():
        return INF, 0.0, np.array([], int)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(arr[idx])]
    best, bwin, lo, s = 0.0, idx[:0], 0, 0.0
    for hi in range(len(idx)):
        s += eff[idx[hi]]
        while arr[idx[hi]] - arr[idx[lo]] > W:
            s -= eff[idx[lo]]; lo += 1
        if s >= thr:
            return arr[idx[hi]], s, idx[lo:hi + 1]
        if s > best:
            best, bwin = s, idx[lo:hi + 1]
    return INF, best, bwin


class Net:
    def __init__(self, p, R, Hn, G, k, W, H, rng, dinit=1.0, budget=8.0, kappa=0.02, veto=0):
        self.dinit, self.budget, self.kappa = dinit, budget, kappa
        self.p, self.R, self.Hn, self.G, self.k, self.W, self.H = p, R, Hn, G, k, W, H
        self.P = rng.uniform(3, 12, R)                          # loop periods
        self.d = rng.uniform(0, 12, (R, p))                     # a -> loop injection delays
        self.L = int(np.ceil(H / 3)) + 1                        # max laps considered
        nin = R + 2 * p                                         # hidden inputs: loops, b-lines, a-lines
        self.wh = rng.uniform(0.3, 0.7, (Hn, nin))                  # any two coinciding inputs can fire a node
        self.dh = rng.uniform(0, self.dinit, (Hn, nin))
        self.wo = rng.uniform(0, 2 * self.budget / Hn, (p, Hn))  # class node synaptic budget (sum of weights)
        self.wo = self.wo; self.do = rng.uniform(0, 2, (p, Hn))
        self.vo = np.zeros((p, Hn)); self.veto = veto           # output veto synapses (inhibitory)
        self.thr = np.ones(p)                                   # per-class prices (thresholds), §36 / §51
        self.events = 0; self.updates = 0

    def loop_spikes(self, a):
        """lap spike times of each loop (R x L) after operand a's injection."""
        if self.R == 0:
            return np.zeros((0, self.L)), np.zeros((0, self.L))
        laps = np.arange(self.L)
        t = self.d[:, a][:, None] + laps[None, :] * self.P[:, None]
        return np.where(t <= self.H, t, INF), np.broadcast_to(laps, t.shape)

    def hidden_arrivals(self, h, a, b, ls):
        """flat arrivals at hidden node h: loops (every lap), then b-line, then a-line. Returns arr, w, src, lap."""
        R, p = self.R, self.p
        arr = [ls[r] + self.dh[h, r] for r in range(R)]
        arr = np.concatenate(arr + [np.array([self.dh[h, R + b]]), np.array([self.dh[h, R + p + a]])]) \
            if R else np.array([self.dh[h, R + b], self.dh[h, R + p + a]])
        src = np.concatenate([np.repeat(np.arange(R), self.L), [R + b, R + p + a]]).astype(int)
        lap = np.concatenate([np.tile(np.arange(self.L), R), [0, 0]]).astype(int)
        return arr, self.wh[h, src], src, lap

    def forward(self, a, b):
        ls, _ = self.loop_spikes(a)
        ft = np.full(self.Hn, INF); charge = np.zeros(self.Hn); win = [None] * self.Hn; info = [None] * self.Hn
        for h in range(self.Hn):
            arr, w, src, lap = self.hidden_arrivals(h, a, b, ls)
            ft[h], charge[h], wi = integrate(arr, w, self.W)
            win[h] = wi; info[h] = (arr, src, lap)
        fired = np.zeros(self.Hn, bool)
        for g0 in range(0, self.Hn, self.G):
            grp = np.arange(g0, min(g0 + self.G, self.Hn))
            order = grp[np.argsort(ft[grp])][:self.k]
            fired[order[np.isfinite(ft[order])]] = True
        x = np.where(fired, ft, INF)
        fo = np.full(self.p, INF); owin = [None] * self.p
        for c in range(self.p):                                 # output: non-leaky evidence accumulation
            fo[c], _, owin[c] = integrate(x + self.do[c], self.wo[c], INF, self.thr[c],
                                          self.vo[c] if self.veto else None)
        c = int(fo.argmin()) if fo.min() < self.H + 2 else self.p
        self.events += 2 + int(np.isfinite(ls).sum()) + int(fired.sum()) + int(np.isfinite(fo).sum())
        return c, dict(x=x, fired=fired, charge=charge, win=win, info=info, fo=fo, owin=owin)

    def pull_hidden(self, h, wi, info, eta):
        """pull hidden node h on arrivals wi: weights up, late arrivals earlier (delays / loop periods)."""
        arr, src, lap = info
        if not len(wi):
            return
        target = arr[wi].mean()                                 # coincidence: arrivals align to each other
        tot = self.wh[h].sum()
        for j in wi:
            s = src[j]
            self.wh[h, s] += eta * tot / len(wi)
            shift = 0.5 * (target - arr[j])
            if s < self.R and lap[j] > 0:                       # a loop lap: split between delay and period
                self.dh[h, s] += shift / 2
                self.P[s] += shift / (2 * lap[j])
            else:
                self.dh[h, s] += shift
        self.wh[h] *= tot / self.wh[h].sum()                    # conserve the node's synaptic resource

    def teach(self, a, b, eta, credit):
        y = (a + b) % self.p
        c, st = self.forward(a, b)
        if c == y:
            return False
        x = st["x"]
        arr = x + self.do[y]
        got = np.isfinite(arr)
        wants = self.wo[y] > 2 * self.wo[y].mean()              # the teacher wants h (relative to its budget)
        if got.any():                                           # teacher output pull, resource-conserving:
            tot = self.wo[y].sum()                              # the node's other synapses pay for it;
            self.wo[y, got] += eta * tot / got.sum()            # a pull moves a fraction eta of the budget
            self.wo[y] *= tot / self.wo[y].sum()
            self.do[y, got] += 0.5 * (arr[got].mean() - arr[got])
            for h in np.flatnonzero(st["fired"] & wants):
                if credit != "frozen":
                    self.pull_hidden(h, st["win"][h], st["info"][h], eta * 0.5)
        if credit == "nearmiss":                                # counterfactual routing from cancellation
            want = (~st["fired"]) & wants & (st["charge"] > 0.5)
            for h in np.flatnonzero(want):
                self.pull_hidden(h, st["win"][h], st["info"][h], eta)
        if y < self.p:                                          # prices: a missed teacher gets cheaper,
            self.thr[y] -= self.kappa                           # a false winner dearer (event-local)
        if c < self.p and c != y:
            self.thr[c] += self.kappa
        np.clip(self.thr, 0.2, 5.0, out=self.thr)
        if self.veto:                                           # §56.5: losers are vetoed, not weakened
            if c < self.p and c != y:
                self.vo[c, st["owin"][c]] += eta * self.budget / max(len(st["owin"][c]), 1)
            if y < self.p and got.any():                        # a teacher blocked by its own veto: release it
                self.vo[y, got] *= 1 - eta
            np.clip(self.vo, 0, 1, out=self.vo)
        elif c < self.p and c != y:                             # specialize the false winner, conserving:
            tot = self.wo[c].sum()                              # weight leaves the synapses that fired it
            self.wo[c, st["owin"][c]] *= 1 - eta                # and returns to the node's other synapses
            np.clip(self.wo[c], 0, 1, out=self.wo[c])
            self.wo[c] *= tot / max(self.wo[c].sum(), 1e-9)
        np.clip(self.wh, 0, 1, out=self.wh); np.clip(self.wo, 0, 1, out=self.wo)
        np.maximum(self.dh, 0, out=self.dh); np.maximum(self.do, 0, out=self.do)
        np.clip(self.P, 1.0, 30.0, out=self.P)
        self.updates += 1
        return True

    def params(self):
        return 2 * (self.wh.size + self.wo.size) + self.d.size + self.P.size


def accuracy(net, A, B):
    return float(np.mean([net.forward(a, b)[0] == (a + b) % net.p for a, b in zip(A, B)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=31)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--W", type=float, default=0.5)
    ap.add_argument("--H", type=float, default=40.0)
    ap.add_argument("--credit", default="nearmiss", choices=("path", "nearmiss", "frozen"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--budget", type=float, default=4.0, help="class node synaptic budget")
    ap.add_argument("--veto", type=int, default=1, help="1: false winners get veto synapses (§56.5)")
    ap.add_argument("--kappa", type=float, default=0.02, help="price step")
    ap.add_argument("--dinit", type=float, default=1.0, help="initial hidden delay spread")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    (A, B), (At, Bt) = split(a.p, a.frac, a.seed)
    net = Net(a.p, a.loops, a.hidden, a.group, a.k, a.W, a.H, rng, a.dinit, a.budget, a.kappa, a.veto)
    rho = len(A) / net.params()
    curve = []
    for ep in range(1, a.epochs + 1):                           # a stream of training pairs, reshuffled
        for i in rng.permutation(len(A)):
            net.teach(A[i], B[i], a.eta, a.credit)
        if ep % a.every == 0 or ep == a.epochs:
            curve.append({"epoch": ep, "train": accuracy(net, A, B), "test": accuracy(net, At, Bt),
                          "updates": net.updates, "periods": [round(float(x), 2) for x in net.P]})
            print(json.dumps(curve[-1]), flush=True)
    res = {"args": vars(a), "rho": rho, "params": net.params(), "curve": curve, "wall_s": round(time.time() - t0, 1)}
    name = f"L{a.loops}_{a.credit}_s{a.seed}{'_' + a.tag if a.tag else ''}.json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(res, f)
    print("rho", round(rho, 4), "EXIT-OK", res["wall_s"])


if __name__ == "__main__":
    main()
