"""E25: modular addition with delays instead of lookup (README "Sleep Sort", THEORY §53).

The E24 task, (a + b) mod p, re-posed in the time domain. A ring of p nodes, each relaying a spike to the next
after one delay unit, is a cyclic clock: a spike injected at node a sits at node (a + t) mod p at time t. Read it at
t = b and the answer is where the spike is. The mod comes free from the cycle; the addition from waiting.

  compiled  the hand-wired ring, event-driven, no learning: checks all p² pairs and counts events
  learn     a bank of rings of periods L (a generic substrate of oscillators). Per ring, three maps are learned
            by races with local error-gated near-miss credit: operand a -> injection node, operand b -> read
            delay, ring node -> output class. Output = first class to cross threshold over the summed votes.
            On an error the teacher's branch is woven (§35): the readout node that votes most for the label,
            reached by moving whichever of (injection, delay) is the nearer miss.
            (Negative: frustrated, fails even on the training set; kept as the discrete baseline.)
  online    continuous delays (phases on the ring): theta_a (operand-a delay), phi_b (read delay), tau_c (class
            detector phase), learned online from the timing error, error-gated. Phasor form, no wrap.
  sync      the same delays learned by replay ("sleep"): every stored sample votes, each delay moves to the circular
            mean of what its co-active partners say it should be, theta_a <- arg sum_{(a,b)} (tau_y - phi_b). This is
            the power method for angular synchronization; restarts are chosen by training error only.
  trace     fully online and local: each delay keeps a phasor trace S that every sample it takes part in adds its
            vote to (S_a += z_y conj(z_b), ...); the delay is the trace's phase. Between epochs a sleep phase
            downscales all traces by (1 - decay), so old, inconsistent votes fade (§52's downscaling, §53.3).
  table     control: a race over (a, b) conjunction nodes, i.e. memory indexing. Cannot generalize.

Prints and saves test accuracy vs training fraction.
"""
import argparse
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e25")


_PERM = {}


def label(A, B, p, op="add"):
    """add: (a+b) mod p. sub: (a-b). mul: a*b (operands 1..p-1: a cyclic group of order p-1, needs the discrete log).
    perm: add with the classes relabelled at random. sq: a^2+b^2 (still f(a)+g(b)). poly: a^2+ab+b^2
    (not separable). rand: a random table."""
    if op == "add":
        return (A + B) % p
    if op == "sub":
        return (A - B) % p
    if op == "mul":
        return (A * B) % p
    if op == "sq":
        return (A * A + B * B) % p
    if op == "poly":
        return (A * A + A * B + B * B) % p
    if op not in _PERM:
        r = np.random.default_rng(12345)
        _PERM[op] = r.permutation(p) if op == "perm" else r.integers(0, p, (p, p))
    return _PERM[op][(A + B) % p] if op == "perm" else _PERM[op][A, B]


OP = "add"


def split(p, frac, seed):
    a, b = np.divmod(np.arange(p * p), p)
    if OP == "mul":
        keep = (a > 0) & (b > 0)
        a, b = a[keep], b[keep]
    perm = np.random.default_rng(seed).permutation(len(a))
    n = max(1, int(round(frac * len(a))))
    tr, te = perm[:n], perm[n:]
    return (a[tr], b[tr]), (a[te], b[te])


def compiled(p):
    """Event-driven simulation of the hand-wired ring. Returns accuracy and mean synaptic events per query."""
    correct, events = 0, 0
    for a in range(p):
        for b in range(p):
            # event queue: (time, kind, node). Input a -> ring node a at t=0; input b -> read pulse at t=b.
            q = [(0.0, 0, a), (float(b), 1, -1)]      # kind 0 = ring spike, 1 = read (ring first on ties)
            ev, out, pos = 0, None, None
            while q:
                q.sort()
                t, kind, n = q.pop(0)
                if kind == 0:
                    pos = n
                    ev += 2                                   # to next ring node, to its coincidence detector
                    if t + 1 <= b:
                        q.append((t + 1, 0, (n + 1) % p))
                else:
                    ev += p                                   # read pulse fans out to the p coincidence detectors
                    out = pos                                 # only the detector whose ring node is live fires
                    break
            correct += out == (a + b) % p
            events += ev
    return correct / p ** 2, events / p ** 2


class RingBank:
    def __init__(self, p, periods, rng, init=0.01):
        self.p, self.L = p, list(periods)
        self.Wa = [rng.uniform(0, init, (p, L)) for L in self.L]      # operand a -> injection node
        self.Wb = [rng.uniform(0, init, (p, L)) for L in self.L]      # operand b -> read delay 0..L-1
        self.Wo = [rng.uniform(0, init, (L, p)) for L in self.L]      # ring node -> class
        self.updates = 0

    def route(self, a, b):
        pos = [W[a].argmax() for W in self.Wa]
        dly = [W[b].argmax() for W in self.Wb]
        return pos, dly, [(x + d) % L for x, d, L in zip(pos, dly, self.L)]

    def predict(self, a, b):
        _, _, node = self.route(a, b)
        return int(sum(W[k] for W, k in zip(self.Wo, node)).argmax())

    def teach(self, a, b, y, eta, margin):
        pos, dly, node = self.route(a, b)
        v = sum(W[k] for W, k in zip(self.Wo, node))
        rival = v.copy(); rival[y] = -np.inf
        yhat = int(rival.argmax())
        if v[y] - v[yhat] > margin:                               # error-gated: silent once right with margin
            return False
        for i, L in enumerate(self.L):
            Wo = self.Wo[i]
            Wo[node[i], y] += eta; Wo[node[i], yhat] -= eta
            target = int(Wo[:, y].argmax())                        # the node that votes most for the label
            if target == node[i]:
                continue
            # weave the teacher's branch: move whichever of injection / delay is the nearer miss
            pa, db = (target - dly[i]) % L, (target - pos[i]) % L
            gap_a = self.Wa[i][a, pos[i]] - self.Wa[i][a, pa]
            gap_b = self.Wb[i][b, dly[i]] - self.Wb[i][b, db]
            if gap_a <= gap_b:
                self.Wa[i][a, pa] += eta; self.Wa[i][a, pos[i]] -= eta
            else:
                self.Wb[i][b, db] += eta; self.Wb[i][b, dly[i]] -= eta
        self.updates += 1
        return True

    def params(self):
        return sum(W.size for Ws in (self.Wa, self.Wb, self.Wo) for W in Ws)


class Table:
    """(a, b) conjunction nodes racing to classes: memory indexing."""
    def __init__(self, p, rng, init=0.01):
        self.p, self.W, self.updates = p, rng.uniform(0, init, (p * p, p)), 0

    def predict(self, a, b):
        return int(self.W[a * self.p + b].argmax())

    def teach(self, a, b, y, eta, margin):
        w = self.W[a * self.p + b]
        r = w.copy(); r[y] = -np.inf
        yh = int(r.argmax())
        if w[y] - w[yh] > margin:
            return False
        w[y] += eta; w[yh] -= eta
        self.updates += 1
        return True

    def params(self):
        return self.W.size


def _agg(idx, v, p):
    return np.bincount(idx, v.real, p) + 1j * np.bincount(idx, v.imag, p)


def _unit(z):
    return z / np.maximum(np.abs(z), 1e-12)


class Phases:
    """Delays as unit phasors on a ring of period p: z = exp(2 pi i delay / p). Parameters: 3p delays."""
    def __init__(self, p, rng):
        self.p, self.updates = p, 0
        u = lambda: np.exp(2j * np.pi * rng.uniform(0, 1, p))   # noqa: E731
        self.za, self.zb, self.zy = u(), u(), u()

    def scores(self, A, B):
        return np.real(np.conj(self.zy)[None, :] * (self.za[A] * self.zb[B])[:, None])

    def predict(self, a, b):
        return int(self.scores(np.array([a]), np.array([b]))[0].argmax())

    def params(self):
        return 3 * self.p

    def teach(self, a, b, y, eta, margin):                     # online, error-gated (mode "online")
        s = self.scores(np.array([a]), np.array([b]))[0]
        r = s.copy(); r[y] = -np.inf
        yh = int(r.argmax())
        if s[y] - s[yh] > margin:
            return False
        x, g = self.za[a] * self.zb[b], self.zy[y] - self.zy[yh]
        self.za[a] = _unit(self.za[a] + eta * g * np.conj(self.zb[b]))
        self.zb[b] = _unit(self.zb[b] + eta * g * np.conj(self.za[a]))
        self.zy[y] = _unit(self.zy[y] + eta * x)
        self.zy[yh] = _unit(self.zy[yh] - eta * x)
        self.updates += 1
        return True

    def sleep(self, A, B, iters):                              # replay (mode "sync")
        p, Y = self.p, label(A, B, self.p, OP)
        for _ in range(iters):
            self.za = _unit(_agg(A, self.zy[Y] * np.conj(self.zb[B]), p))
            self.zb = _unit(_agg(B, self.zy[Y] * np.conj(self.za[A]), p))
            self.zy = _unit(_agg(Y, self.za[A] * self.zb[B], p))
            self.updates += 3 * len(A)


def trace(p, tr, te, seed, restarts, epochs, decay):
    A, B = tr
    Y = label(A, B, p, OP)
    best = None
    for r in range(restarts):
        rng = np.random.default_rng(1000 * seed + r)
        net = Phases(p, rng)
        S = [1e-3 * net.za, 1e-3 * net.zb, 1e-3 * net.zy]
        for _ in range(epochs):
            for x in S:
                x *= 1 - decay
            for i in rng.permutation(len(A)):
                a, b, y = A[i], B[i], Y[i]
                za, zb, zy = _unit(S[0][a]), _unit(S[1][b]), _unit(S[2][y])
                S[0][a] += zy * np.conj(zb); S[1][b] += zy * np.conj(za); S[2][y] += za * zb
            net.updates += 3 * len(A)
        net.za, net.zb, net.zy = (_unit(x) for x in S)
        tra = accuracy(net, *tr, p)
        if best is None or tra > best[0]:
            best = (tra, net)
    net = best[1]
    return {"epochs": epochs, "train": best[0], "test": accuracy(net, *te, p), "updates": net.updates * restarts,
            "params": net.params(), "restarts": restarts}


def sync(p, tr, te, seed, restarts, iters):
    best = None
    for r in range(restarts):
        net = Phases(p, np.random.default_rng(1000 * seed + r))
        net.sleep(*tr, iters)
        tra = accuracy(net, *tr, p)
        if best is None or tra > best[0]:
            best = (tra, net)
    net = best[1]
    return {"epochs": iters, "train": best[0], "test": accuracy(net, *te, p), "updates": net.updates * restarts,
            "params": net.params(), "restarts": restarts}


def accuracy(net, A, B, p):
    if len(A) == 0:
        return float("nan")
    if hasattr(net, "scores"):
        return float((net.scores(A, B).argmax(1) == label(A, B, p, OP)).mean())
    Y = label(A, B, p, OP)
    return float(np.mean([net.predict(a, b) == y for a, b, y in zip(A, B, Y)]))


def train(net, tr, te, p, epochs, eta, margin, rng):
    A, B = tr
    Y = label(A, B, p, OP)
    for ep in range(1, epochs + 1):
        errs = 0
        for i in rng.permutation(len(A)):
            errs += net.teach(A[i], B[i], Y[i], eta, margin)
        if errs == 0:
            break
    return {"epochs": ep, "train": accuracy(net, *tr, p), "test": accuracy(net, *te, p),
            "updates": net.updates, "params": net.params()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("compiled", "learn", "online", "sync", "trace", "table"))
    ap.add_argument("--p", type=int, default=31)
    ap.add_argument("--fracs", default="0.02,0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--periods", default="p", help="'p', or 'lo:hi' for a bank of ring periods lo..hi")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--op", default="add", choices=("add", "sub", "mul", "perm", "sq", "poly", "rand"))
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global OP
    OP = a.op
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if a.mode == "compiled":
        acc, ev = compiled(a.p)
        res = {"p": a.p, "accuracy": acc, "synaptic_events_per_query": ev, "params_synapses": 4 * a.p}
        print(json.dumps(res))
    else:
        if a.periods == "p":
            periods = [a.p]
        else:
            lo, hi = map(int, a.periods.split(":"))
            periods = list(range(lo, hi + 1))
        rows = []
        for frac in map(float, a.fracs.split(",")):
            for s in range(a.seeds):
                rng = np.random.default_rng(1000 + s)
                tr, te = split(a.p, frac, s)
                if a.mode == "sync":
                    out = sync(a.p, tr, te, s, a.restarts, a.iters)
                elif a.mode == "trace":
                    out = trace(a.p, tr, te, s, a.restarts, a.epochs, a.decay)
                else:
                    net = {"learn": lambda: RingBank(a.p, periods, rng), "online": lambda: Phases(a.p, rng),
                           "table": lambda: Table(a.p, rng)}[a.mode]()
                    out = train(net, tr, te, a.p, a.epochs, a.eta, a.margin, rng)
                r = {"frac": frac, "seed": s, "n_train": len(tr[0]), **out}
                rows.append(r)
                print(json.dumps(r), flush=True)
        res = {"args": vars(a), "periods": periods, "rows": rows}
    res["wall_s"] = round(time.time() - t0, 1)
    name = f"{a.mode}_p{a.p}{'_' + a.op if a.op != 'add' else ''}{'_' + a.tag if a.tag else ''}.json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(res, f, indent=1)
    print("EXIT-OK", res["wall_s"])


if __name__ == "__main__":
    main()
