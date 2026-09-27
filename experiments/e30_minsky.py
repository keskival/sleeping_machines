"""E30: a two-counter (Minsky) machine wired from the temporal operator basis (THEORY §59).

Two-counter Minsky machines are Turing-complete. Here one is a netlist of four node types and nothing else:
  Delay(d)        re-emits each input spike d later (with timing jitter sigma, the only noise)
  Or              re-emits every input spike (first-of, for merging lines)
  And(w0, w1)     fires at the second of its two inputs if the first is still within its PSP window (w0 for
                  port 0, w1 for port 1); one-shot (coincidence; a long window is a long PSP)
  Veto(window)    re-emits its input unless an inhibitory spike arrived at most `window` before it
plus a reference oscillator (a spike every T). A hold loop is a Delay whose output feeds back to its input. No node
reads a clock, stores a number, or branches in code: all control is spike timing.

Encoding. Cycle k spans [kT, (k+1)T). Counter c holds value n as a spike at phase OFF + n·q in every cycle; it
circulates through one of three paths per cycle, each a Delay:
  hold   T          (vetoed when this cycle's instruction operates on c)
  inc    T + q      (gated: And(enable_inc_c, counter))
  dec    T - q      (gated: And(enable_dec_c, counter), vetoed by the zero detector; a blocked decrement
                     returns to the hold path through And(zero, gated counter))
Zero detector: And(counter, reference delayed by OFF, window q/2). The program state is a one-hot spike on the
line of the current instruction, at phase 0. Instruction i drives its enable lines at phase 0, and its successor
line one cycle later: Delay(T) for inc; for dec, Veto(Delay(state), Z) -> next, And(Delay(state), Delay(Z)) ->
next_if_zero. Lines targeted by several instructions are merged with Or.

Precision is the tape: n < (T - OFF)/q, and jitter sigma on each hop random-walks the phases; the machine fails
once the accumulated jitter crosses q/2 (at the zero test or the final readout).
"""
import argparse
import heapq
import itertools
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e30")


class Net:
    def __init__(self, sigma, rng):
        self.q, self.sigma, self.rng, self.events, self.seq = [], sigma, rng, 0, itertools.count()

    def at(self, t, target, port=0):
        heapq.heappush(self.q, (t, next(self.seq), target, port))
        self.events += 1

    def run(self, until):
        while self.q and self.q[0][0] <= until:
            t, _, target, port = heapq.heappop(self.q)
            target.recv(self, t, port)


class Node:
    def __init__(self):
        self.out = []

    def fire(self, net, t):
        for tgt, port in self.out:
            net.at(t, tgt, port)

    def to(self, tgt, port=0):
        self.out.append((tgt, port))
        return tgt


class Delay(Node):
    def __init__(self, d):
        super().__init__(); self.d = d

    def recv(self, net, t, port):
        jit = net.sigma * net.rng.standard_normal() if net.sigma else 0.0
        self.fire(net, t + self.d + jit)


class Or(Node):
    def recv(self, net, t, port):
        self.fire(net, t)


class And(Node):
    """Coincidence with a PSP length per input: fires on a spike at one port if the other port's last spike is
    still within that other port's PSP window. One-shot: both inputs are consumed when it fires."""
    def __init__(self, w0, w1=None):
        super().__init__(); self.w = (w0, w0 if w1 is None else w1); self.last = [None, None]

    def recv(self, net, t, port):
        other = self.last[1 - port]
        if other is not None and t - other <= self.w[1 - port]:
            self.last = [None, None]
            self.fire(net, t)
        else:
            self.last[port] = t


class Veto(Node):
    def __init__(self, window):
        super().__init__(); self.w = window; self.inh = -np.inf

    def recv(self, net, t, port):
        if port == 1:
            self.inh = t
        elif t - self.inh > self.w:
            self.fire(net, t)


class Probe(Node):
    def __init__(self):
        super().__init__(); self.times = []

    def recv(self, net, t, port):
        self.times.append(t)


def build(prog, T, q, OFF, restore=False, comb=None):
    """Wire the machine. Returns (entry points, probes)."""
    n_ins = len(prog)
    EPS = q / 4                                                  # small alignment delays
    state = [Or() for _ in range(n_ins)]                         # state line i fires at phase 0 of its cycle
    ref = Or()                                                   # the reference oscillator's output
    counters = {}
    for c in (0, 1):
        inp = Or()                                               # counter spike arrives here each cycle
        hold = Veto(window=T)                                    # hold path, vetoed by any enable for c
        inc_g, dec_g = And(T, 0.0), And(T, 0.0)                  # enable: long PSP; counter: instantaneous
        dec_v = Veto(window=q)                                   # decrement blocked by the zero detector
        zero = And(q / 2)                                        # counter coincides with reference + OFF
        ref.to(Delay(OFF)).to(zero, 0)
        inp.to(zero, 1)
        inp.to(hold, 0); inp.to(inc_g, 1); inp.to(Delay(EPS)).to(dec_g, 1)
        dec_g.to(dec_v, 0)
        zero.to(dec_v, 1)
        back = Or()                                              # all paths merge back into the loop input
        if restore:                                              # restoration: the loop paths deliver the spike
            snap = And(q, 0.0)                                   # half a quantum early; the comb tick at the
            back.to(snap, 0)                                     # nominal phase re-emits it exactly on time
            comb.to(snap, 1)                                     # (digital restoration, in time)
            snap.to(inp)
        R = q / 2 if restore else 0.0                            # restoring loops run half a quantum early
        hold.to(Delay(T - R)).to(back); inc_g.to(Delay(T + q - R)).to(back)
        dec_v.to(Delay(T - q - EPS - R)).to(back)
        keep = And(q, 0.0)                                       # a decrement blocked at zero keeps the spike:
        zero.to(keep, 0); dec_g.to(keep, 1)                      # zero AND gated counter -> back on the hold path
        keep.to(Delay(T - EPS - R)).to(back)
        if not restore:
            back.to(inp)
        probe = Probe(); inp.to(probe)
        counters[c] = dict(inp=inp, hold=hold, inc=inc_g, dec=dec_g, zero=zero, probe=probe)
    zprobe = {c: Probe() for c in (0, 1)}
    for c in (0, 1):
        counters[c]["zero"].to(zprobe[c])
    halt = Probe()
    for i, op in enumerate(prog):
        s = state[i]
        if op[0] == "halt":
            s.to(halt)
            continue
        c = op[1]
        s.to(counters[c]["hold"], 1)                             # this cycle, c leaves its hold path
        if op[0] == "inc":
            s.to(counters[c]["inc"], 0)
            s.to(Delay(T)).to(state[op[2]])
        else:
            s.to(counters[c]["dec"], 0)
            # successor: next unless zero; next_if_zero if zero. Z fires at phase OFF (+ jitter); align to T.
            nz = Veto(window=T)
            s.to(Delay(T - EPS)).to(nz, 0)                       # arrives just before the next cycle starts
            counters[c]["zero"].to(Delay(0)).to(nz, 1)           # the zero detector inhibits "next"
            nz.to(Delay(EPS)).to(state[op[2]])
            z = And(0.0, T)                                      # state (next cycle start) after a zero this cycle
            s.to(Delay(T)).to(z, 0)
            counters[c]["zero"].to(z, 1)
            z.to(state[op[3]])
    return state, ref, counters, halt


def run_program(prog, c0, c1, T=1.0, q=0.02, OFF=0.05, sigma=0.0, max_cycles=5000, seed=0, restore=False):
    net = Net(sigma, np.random.default_rng(seed))
    comb = Or()                                                  # restoring comb: ticks at phases OFF + n q
    state, ref, counters, halt = build(prog, T, q, OFF, restore, comb)
    for k in range(max_cycles + 1):                              # the reference oscillator
        net.at(k * T, ref)
    if restore:                                                  # comb ticks (a second, faster oscillator)
        nmax = int((T - OFF) / q)
        for k in range(1, 200):
            for n in range(nmax):
                net.q.append((k * T + OFF + n * q, next(net.seq), comb, 0))
        heapq.heapify(net.q)
    net.at(0.0, state[0])
    net.at(OFF + c0 * q, counters[0]["inp"]); net.at(OFF + c1 * q, counters[1]["inp"])
    cyc = 0
    while cyc < max_cycles:
        net.run((cyc + 1) * T - 1e-9)
        if halt.times:
            break
        cyc += 1
    if not halt.times:
        return None, cyc, False, net.events
    k = int(round(halt.times[0] / T))                            # read the counters in the halting cycle
    net.run((k + 1) * T - 1e-9)                                  # (simulate through it: jitter can put the halt
                                                                 #  spike just before the cycle boundary)
    vals = []
    for c in (0, 1):
        ts = [t for t in counters[c]["probe"].times if k * T <= t < (k + 1) * T]
        vals.append(int(round((ts[-1] - k * T - OFF) / q)) if ts else -1)
    return tuple(vals), k, True, net.events


ADD = [("dec", 1, 1, 2), ("inc", 0, 0, None), ("halt",)]                         # c0 += c1
DOUBLE = [("dec", 0, 1, 3), ("inc", 1, 2, None), ("inc", 1, 0, None), ("halt",)]  # c1 += 2·c0
PARITY = [("dec", 1, 1, 4), ("dec", 1, 2, 3), ("dec", 0, 0, 0), ("inc", 0, 4, None), ("halt",)]  # c0 = c1 mod 2


def reference(prog, c0, c1, max_steps=5000):
    c = [c0, c1]; pc = 0; steps = 0
    while prog[pc][0] != "halt" and steps < max_steps:
        op = prog[pc]
        if op[0] == "inc":
            c[op[1]] += 1; pc = op[2]
        elif c[op[1]] == 0:
            pc = op[3]
        else:
            c[op[1]] -= 1; pc = op[2]
        steps += 1
    return tuple(c), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.02)
    ap.add_argument("--sigmas", default="0,0.0005,0.001,0.002,0.003,0.004,0.006")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--steps-long", type=int, default=0, help="also test a long program (add with large c1)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows, allok = [], True
    progs = {"add": (ADD, [(3, 5), (10, 12), (0, 7), (5, 0)]),
             "double": (DOUBLE, [(4, 0), (10, 3), (0, 0)]),
             "parity": (PARITY, [(0, 9), (0, 14), (0, 0)])}
    for name, (prog, inputs) in progs.items():
        for c0, c1 in inputs:
            got, cycles, halted, ev = run_program(prog, c0, c1, q=a.q)
            ref, steps = reference(prog, c0, c1)
            ok = halted and got == ref and cycles == steps
            allok &= ok
            r = {"prog": name, "in": [c0, c1], "got": got, "ref": ref, "cycles": cycles, "steps": steps, "ok": ok,
                 "events": ev, "sigma": 0.0}
            rows.append(r); print(json.dumps(r))
    prog, (c0, c1) = ADD, (10, 12)
    ref, steps = reference(prog, c0, c1)
    for restore in (False, True):
        for (c0, c1) in ((10, 12), (2, 40)):
            ref, steps = reference(prog, c0, c1)
            for s in map(float, a.sigmas.split(",")):
                succ = sum(run_program(prog, c0, c1, q=a.q, sigma=s, seed=k, restore=restore)[0] == ref
                           for k in range(a.trials)) / a.trials
                r = {"prog": "add", "in": [c0, c1], "steps": steps, "sigma": s, "restore": restore,
                     "q_over_sigma": a.q / s if s else None, "success": succ}
                rows.append(r); print(json.dumps(r))
    with open(os.path.join(OUT, "minsky.json"), "w") as f:
        json.dump({"args": vars(a), "exact_all": allok, "rows": rows}, f, indent=1)
    print("ALL EXACT" if allok else "SOME WRONG")


if __name__ == "__main__":
    main()
