#!/usr/bin/env python3
"""E4: learning from counterfactual traces of cancelled events.

See experiments/E4_PREREGISTRATION.md for the claim and predictions.

    python experiments/e4_counterfactual.py tune
    python experiments/e4_counterfactual.py main
    python experiments/e4_counterfactual.py delay
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sleeping_machines.sim import Engine  # noqa: E402

RULES = ("fired_only", "fired_reward", "cf_winner", "cf_uniform", "cf_margin")
OUT = os.path.join(os.path.dirname(__file__), "results", "e4")


# ── Task ──────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    k: int
    m: int = 256            # input channels
    proto: int = 20         # channels per class prototype
    keep: float = 0.5       # probability a prototype channel spikes
    distractors: int = 20   # random extra channels per sample
    window: float = 1.0     # input spikes arrive uniformly in [0, window)

    def prototypes(self, rng):
        return [rng.choice(self.m, self.proto, replace=False) for _ in range(self.k)]

    def sample(self, protos, rng):
        target = int(rng.integers(self.k))
        signal = protos[target][rng.random(self.proto) < self.keep]
        channels = np.unique(np.concatenate([signal, rng.choice(self.m, self.distractors)]))
        times = rng.uniform(0, self.window, len(channels))
        return target, channels, times


# ── Event-driven race layer ───────────────────────────────────────────────────

@dataclass
class Params:
    rule: str
    eta: float = 0.03
    sigma: float = 0.15     # cf_margin: eligibility = exp(-Δ/σ), Δ normalised by θ
    eps: float = 0.05       # traces below this are not touched
    tau: float = 1.0        # presynaptic trace time constant
    teach_delay: float = 0.05
    gap: float = 0.2        # silence between the end of one window and the next episode
    fire_delay: float = 0.002
    theta: float = 1.0
    init_total: float = 1.5  # expected summed input per node at init, in units of θ
    homeo: float = 0.0       # adaptive threshold: θ_k += homeo * (fired_k - 1/K) per episode


class RaceLayer:
    """K integrate-to-threshold nodes racing; the first to fire inhibits the rest."""

    def __init__(self, engine, task, p, rng):
        self.e, self.task, self.p = engine, task, p
        mean_active = task.proto * task.keep + task.distractors
        w0 = p.init_total * p.theta / mean_active
        self.W = rng.uniform(0, 2 * w0, (task.k, task.m))
        self.v = np.zeros(task.k)
        self.theta = np.full(task.k, p.theta)
        self.pending = {}
        self.snapshot = np.ones(task.k)      # Δ_k / θ frozen at inhibition
        self.last_spike = np.full(task.m, -np.inf)
        self.decided = False
        self.winner = None
        self.log = []                        # (target, winner) per teaching event
        for kind in ("start", "input", "fire", "timeout", "teach"):
            engine.on(kind, getattr(self, "_" + kind))

    # Handlers -----------------------------------------------------------------

    def _start(self, _):
        for ev in self.pending.values():
            self.e.cancel(ev)
        self.pending.clear()
        self.v[:] = 0
        self.decided, self.winner = False, None

    def _input(self, channel):
        self.last_spike[channel] = self.e.now
        if self.decided:
            return                           # layer is inhibited for the rest of the episode
        self.v += self.W[:, channel]
        self.e.count("synops", self.task.k)
        for k in np.flatnonzero(self.v >= self.theta):
            if k not in self.pending:
                self.pending[k] = self.e.schedule(self.p.fire_delay, "fire", int(k))

    def _fire(self, k):
        self.decided, self.winner = True, k
        del self.pending[k]
        for ev in self.pending.values():     # the other futures are made counterfactual
            self.e.cancel(ev)
        self.pending.clear()
        self.e.count("inhibition", self.task.k - 1)
        self._freeze()

    def _timeout(self, _):
        if not self.decided:
            self.decided, self.winner = True, None
            self._freeze()

    def _freeze(self):
        self.snapshot = np.clip((self.theta - self.v) / self.theta, 0, None)
        if self.winner is not None:
            self.snapshot[self.winner] = 0.0
        self.v[:] = 0

    def _teach(self, target):
        p, w = self.p, self.winner
        self.log.append((target, w))
        x = np.exp(-(self.e.now - self.last_spike) / p.tau)
        active = np.flatnonzero(x > p.eps)
        x = x[active]
        n = len(active)
        self.e.count("teach_broadcast", self.task.k)
        wrong = w != target
        if p.homeo:
            fired = np.zeros(self.task.k)
            if w is not None:
                fired[w] = 1.0
            self.theta = np.maximum(self.theta + p.homeo * (fired - 1.0 / self.task.k), 0.05)
            self.e.count("homeostasis", self.task.k)

        def update(node, scale):
            self.W[node, active] += scale * p.eta * x
            self.e.count("plasticity", n)

        if p.rule == "fired_only":
            if wrong and w is not None:
                update(w, -1)
        elif p.rule == "fired_reward":
            if w is not None:
                update(w, -1 if wrong else +1)
        elif wrong:
            update(target, +1)
            if p.rule == "cf_winner":
                if w is not None:
                    update(w, -1)
            else:
                for k in range(self.task.k):
                    if k == target:
                        continue
                    if p.rule == "cf_uniform":
                        elig = 1.0 / (self.task.k - 1)   # balanced: total depression = potentiation
                    else:
                        elig = np.exp(-self.snapshot[k] / p.sigma)
                    if p.rule == "cf_uniform" or elig >= p.eps:
                        update(k, -elig)


def run_race(task, p, seed, n_train, n_test):
    rng = np.random.default_rng(seed)
    protos = task.prototypes(np.random.default_rng(10_000 + seed))
    engine = Engine()
    layer = RaceLayer(engine, task, p, rng)

    def episode(target, channels, times, teach):
        t0 = engine.now
        engine.schedule(0.0, "start")
        for c, t in zip(channels, times):
            engine.schedule(t, "input", int(c))
        engine.schedule(task.window, "timeout")
        if teach:
            engine.schedule(task.window + p.teach_delay, "teach", target)
        engine.run(until=t0 + task.window + p.gap)

    for _ in range(n_train):
        episode(*task.sample(protos, rng), teach=True)
    engine.run()                              # deliver any teaching still in flight
    train_log = list(layer.log)
    train_work = dict(engine.work)

    correct, timeouts = 0, 0
    test_rng = np.random.default_rng(20_000 + seed)
    for _ in range(n_test):
        target, channels, times = task.sample(protos, test_rng)
        episode(target, channels, times, teach=False)
        correct += layer.winner == target
        timeouts += layer.winner is None
    online = np.array([t == w for t, w in train_log], float)
    return {
        "test_acc": correct / n_test,
        "test_timeout": timeouts / n_test,
        "online": online,
        "train_work": train_work,
    }


def run_softmax(task, seed, n_train, n_test, lr=0.5):
    """Non-local reference: online softmax regression on the bag of active channels."""
    rng = np.random.default_rng(seed)
    protos = task.prototypes(np.random.default_rng(10_000 + seed))
    W = np.zeros((task.k, task.m))
    online = np.zeros(n_train)
    updates = 0
    for i in range(n_train):
        target, channels, _ = task.sample(protos, rng)
        z = W[:, channels].sum(1)
        online[i] = z.argmax() == target
        pr = np.exp(z - z.max()); pr /= pr.sum()
        pr[target] -= 1
        W[:, channels] -= lr * pr[:, None]
        updates += task.k * len(channels)    # a dense gradient touches every output row
    test_rng = np.random.default_rng(20_000 + seed)
    correct = 0
    for _ in range(n_test):
        target, channels, _ = task.sample(protos, test_rng)
        correct += W[:, channels].sum(1).argmax() == target
    return {"test_acc": correct / n_test, "online": online, "train_work": {"plasticity": updates}}


# ── Experiments ───────────────────────────────────────────────────────────────

def _job(args):
    kind, task_kw, p_kw, seed, n_train, n_test = args
    task = Task(**task_kw)
    if kind == "softmax":
        r = run_softmax(task, seed, n_train, n_test)
    else:
        r = run_race(task, Params(**p_kw), seed, n_train, n_test)
    r["online_curve"] = [float(c.mean()) for c in np.array_split(r.pop("online"), 50)]
    return args, r


def pool_map(jobs, procs):
    with Pool(procs) as pool:
        return list(pool.imap(_job, jobs, chunksize=1))


def tune(a):
    """Pick eta, sigma and homeostasis per (rule, K) on tuning seeds 100-102."""
    grid = []
    for rule in a.rules:
        for eta in (0.001, 0.003, 0.01, 0.03, 0.1):
            sigmas = (0.05, 0.15, 0.4) if rule == "cf_margin" else (0.15,)
            for sigma in sigmas:
                for homeo in (0.0, 0.003, 0.01, 0.05):
                    grid.append((rule, eta, sigma, homeo))
    jobs = [("race", {"k": k}, {"rule": r, "eta": e, "sigma": s, "homeo": h}, seed, a.train, a.test)
            for k in a.ks for r, e, s, h in grid for seed in (100, 101, 102)]
    results = pool_map(jobs, a.procs)
    scores = {}
    for (kind, t_kw, p_kw, seed, *_), r in results:
        key = (p_kw["rule"], t_kw["k"], p_kw["eta"], p_kw["sigma"], p_kw["homeo"])
        scores.setdefault(key, []).append(r["test_acc"])
    table = [{"rule": r, "k": k, "eta": e, "sigma": s, "homeo": h, "acc": float(np.mean(v))}
             for (r, k, e, s, h), v in sorted(scores.items())]
    path = os.path.join(OUT, "tuned.json")
    best = load_tuned() if os.path.exists(path) else {}
    for rule in a.rules:
        best[rule] = {}
        for k in a.ks:
            top = max((x for x in table if x["rule"] == rule and x["k"] == k), key=lambda x: x["acc"])
            best[rule][str(k)] = {x: top[x] for x in ("eta", "sigma", "homeo", "acc")}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tuning_grid_per_k.json"), "w") as f:
        json.dump(table, f, indent=1)
    with open(os.path.join(OUT, "tuned.json"), "w") as f:
        json.dump(best, f, indent=2)
    for rule in a.rules:
        print(rule, {k: round(v["acc"], 3) for k, v in best[rule].items()})


def load_tuned():
    with open(os.path.join(OUT, "tuned.json")) as f:
        return json.load(f)


def tuned_params(tuned, rule, k, **overrides):
    t = tuned[rule][str(k)]
    return {"rule": rule, "eta": t["eta"], "sigma": t["sigma"], "homeo": t["homeo"], **overrides}


def main(a):
    tuned = load_tuned()
    jobs = []
    for k in a.ks:
        for seed in range(a.seeds):
            jobs.append(("softmax", {"k": k}, {}, seed, a.train, a.test))
            for rule in RULES:
                jobs.append(("race", {"k": k}, tuned_params(tuned, rule, k), seed, a.train, a.test))
    results = pool_map(jobs, a.procs)
    rows = [{"kind": kind, "k": tk["k"], "rule": pk.get("rule", "softmax"), "seed": seed, **r}
            for (kind, tk, pk, seed, *_), r in results]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "main.json"), "w") as f:
        json.dump({"task": asdict(Task(k=0)), "tuned": tuned, "train": a.train, "rows": rows}, f)
    summarize_main(rows)


def summarize_main(rows):
    rules = ("softmax",) + RULES
    ks = sorted({r["k"] for r in rows})
    print(f"\n{'rule':13s}" + "".join(f"   K={k:<14}" for k in ks))
    for rule in rules:
        line = f"{rule:13s}"
        for k in ks:
            acc = [r["test_acc"] for r in rows if r["rule"] == rule and r["k"] == k]
            line += f"   {np.mean(acc):.3f}±{1.96 * np.std(acc) / np.sqrt(len(acc)):.3f}   "
        print(line)
    print("\nplasticity synapse-updates during training (mean):")
    for rule in rules:
        line = f"{rule:13s}"
        for k in ks:
            pl = [r["train_work"].get("plasticity", 0) for r in rows if r["rule"] == rule and r["k"] == k]
            line += f"   {np.mean(pl):>14.3g}   "
        print(line)


def delay(a):
    """E4b: teaching delay with and without an intervening race."""
    tuned = load_tuned()
    jobs = []
    for gap in (5.0, 0.2):                   # 5.0 = sleep until taught; 0.2 = next race intervenes
        for d in (0.05, 0.15, 0.5, 1.0, 2.0, 4.0):
            for seed in range(a.seeds):
                p = tuned_params(tuned, "cf_margin", 32, teach_delay=d, gap=gap, tau=1.0)
                jobs.append(("race", {"k": 32}, p, seed, a.train, a.test))
    results = pool_map(jobs, a.procs)
    rows = [{"gap": pk["gap"], "delay": pk["teach_delay"], "seed": seed, "test_acc": r["test_acc"]}
            for (_, _, pk, seed, *_), r in results]
    with open(os.path.join(OUT, "delay.json"), "w") as f:
        json.dump(rows, f)
    for gap in (5.0, 0.2):
        print(f"gap={gap}: " + "  ".join(
            f"D={d}: {np.mean([r['test_acc'] for r in rows if r['gap'] == gap and r['delay'] == d]):.3f}"
            for d in sorted({r['delay'] for r in rows})))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tune", "main", "delay"))
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 8, 32, 128])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--rules", nargs="+", default=list(RULES), choices=RULES)
    a = ap.parse_args()
    {"tune": tune, "main": main, "delay": delay}[a.cmd](a)
