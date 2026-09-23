#!/usr/bin/env python3
"""E5: does work track activity rather than capacity?

See experiments/E5_PREREGISTRATION.md (and its amendment on connectivity).

Three learners on the same sparse, fixed fan-in connectivity, with the same
structural rule (a taught class node connects to the active inputs, replacing
its weakest synapse):

    race     integrate-to-threshold race + cf_margin; work counted up to the decision
    sparse   masked softmax regression; touches the synapses of active inputs
    dense    K x active-input work, counted analytically (same accuracy class as sparse)

Non-leaky single-spike nodes use the exact closed form (validated against the
event engine in E6).

    python experiments/e5_capacity.py
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e5")
PROTO, KEEP, DISTRACT, FANIN = 16, 0.7, 10, 64


class Net:
    """Fan-in lists W_ch / W_w (K x F) plus a reverse index channel -> synapse ids."""

    def __init__(self, k, m, rng, w0):
        self.k, self.m = k, m
        self.ch = np.stack([rng.choice(m, FANIN, replace=False) for _ in range(k)])
        self.w = rng.uniform(0, 2 * w0, (k, FANIN)).astype(np.float32)
        self.rev = [set() for _ in range(m)]
        for node in range(k):
            for slot, c in enumerate(self.ch[node]):
                self.rev[c].add(node * FANIN + slot)

    def gather(self, channels):
        """Synapse ids reached by each active channel, in spike order."""
        return [np.fromiter(self.rev[c], np.int64, len(self.rev[c])) for c in channels]

    def connect(self, node, channels, w_new):
        """Structural plasticity: connect `node` to active channels it lacks,
        replacing its weakest synapses. Returns the number of synapses rewired."""
        have = set(self.ch[node].tolist())
        new = [c for c in channels if c not in have]
        if not new:
            return 0
        weakest = np.argsort(np.abs(self.w[node]))[:len(new)]
        for slot, c in zip(weakest, new):
            self.rev[self.ch[node, slot]].discard(node * FANIN + slot)
            self.ch[node, slot] = c
            self.w[node, slot] = w_new
            self.rev[c].add(node * FANIN + slot)
        return len(new)


def sample(protos, m, rng):
    target = int(rng.integers(len(protos)))
    sig = protos[target][rng.random(PROTO) < KEEP]
    ch = np.unique(np.concatenate([sig, rng.choice(m, DISTRACT)]))
    t = rng.uniform(0, 1, len(ch))
    order = np.argsort(t)
    return target, ch[order], t[order]


def race_episode(net, ch, t, theta, sigma, beta=0.0):
    """First-to-threshold race over the reached synapses. Returns winner, decision
    index (number of input spikes integrated), touched nodes and their Δ.

    With beta > 0 the threshold rises by beta per input spike: global inhibition
    shared by all nodes (one counter, O(1) per spike), so the race runs on
    relative rather than absolute evidence."""
    syn = net.gather(ch)
    lens = np.array([len(s) for s in syn])
    if lens.sum() == 0:
        return None, len(ch), np.zeros(0, np.int64), np.zeros(0), 0
    ids = np.concatenate(syn)
    spike = np.repeat(np.arange(len(ch)), lens)
    nodes, w = ids // FANIN, net.w.flat[ids]
    order = np.lexsort((spike, nodes))
    nodes_s, spike_s, w_s = nodes[order], spike[order], w[order]
    starts = np.r_[0, np.flatnonzero(np.diff(nodes_s)) + 1]
    cum = np.cumsum(w_s)
    cum -= np.repeat(np.r_[0, cum[starts[1:] - 1]], np.diff(np.r_[starts, len(cum)]))
    th = theta + beta * (spike_s + 1)
    crossed = cum >= th
    if crossed.any():
        first = np.flatnonzero(crossed)
        seg = np.searchsorted(starts, first, side="right") - 1
        _, uniq = np.unique(seg, return_index=True)
        cand = first[uniq]                          # first crossing per node
        j = cand[np.lexsort((-(cum[cand] - th[cand]), spike_s[cand]))[0]]
        winner, dec = int(nodes_s[j]), int(spike_s[j])
    else:
        winner, dec = None, len(ch) - 1
    before = spike_s <= dec
    synops = int(before.sum())
    touched, inv = np.unique(nodes_s[before], return_inverse=True)
    v = np.bincount(inv, weights=w_s[before], minlength=len(touched))
    th_dec = theta + beta * (dec + 1)
    delta = np.clip((th_dec - v) / th_dec, 0, None)
    if winner is not None:
        delta[touched == winner] = 0
    return winner, dec, touched, delta, synops


def run(args):
    k, seed, a = args
    m = max(256, k // 4)
    rng = np.random.default_rng(seed)
    protos = [rng.choice(m, PROTO, replace=False) for _ in range(k)]
    w_new = 1.0 / (PROTO * KEEP)
    out = {"k": k, "m": m, "seed": seed}

    # Race + cf_margin -------------------------------------------------------
    if "race" not in a.models:
        return sparse_softmax(out, k, m, seed, protos, a)
    net = Net(k, m, np.random.default_rng(seed + 1), a.w0)
    work = {"synops": 0, "plasticity": 0, "rewired": 0, "inputs": 0, "inputs_used": 0}
    n_train = a.train_per_class * k
    for _ in range(n_train):
        y, ch, t = sample(protos, m, rng)
        winner, dec, touched, delta, synops = race_episode(net, ch, t, a.theta, a.sigma, a.beta)
        work["synops"] += synops; work["inputs"] += len(ch); work["inputs_used"] += dec + 1
        rivals = delta[touched != y]
        if winner == y and not (len(rivals) and rivals.min() < a.margin):
            continue
        x = np.exp(-(1.0 + a.teach_delay - t) / a.tau)       # presynaptic traces at teaching time
        xs = dict(zip(ch.tolist(), x.tolist()))
        work["rewired"] += net.connect(y, ch.tolist(), w_new)
        for node, sign in [(y, 1.0)] + [(int(n), -float(np.exp(-d / a.sigma))) for n, d in zip(touched, delta) if n != y]:
            if sign < 0 and -sign < 0.05:
                continue
            for slot, c in enumerate(net.ch[node]):
                if c in xs:
                    net.w[node, slot] += a.eta * sign * xs[c]
                    work["plasticity"] += 1
    test_rng = np.random.default_rng(10_000 + seed)
    correct, inf = 0, {"synops": 0, "inputs": 0, "inputs_used": 0}
    for _ in range(a.test):
        y, ch, t = sample(protos, m, test_rng)
        winner, dec, _, _, synops = race_episode(net, ch, t, a.theta, a.sigma, a.beta)
        correct += winner == y
        inf["synops"] += synops; inf["inputs"] += len(ch); inf["inputs_used"] += dec + 1
    out["race"] = {"acc": correct / a.test, "train_per_episode": {q: v / n_train for q, v in work.items()},
                   "inference_per_sample": {q: v / a.test for q, v in inf.items()}}
    return sparse_softmax(out, k, m, seed, protos, a) if "sparse" in a.models else out


def sparse_softmax(out, k, m, seed, protos, a):
    n_train = a.train_per_class * k
    rng = np.random.default_rng(seed + 2)
    net = Net(k, m, np.random.default_rng(seed + 1), a.w0)
    net.w[:] = 0
    work = {"synops": 0, "plasticity": 0, "rewired": 0}
    for _ in range(n_train):
        y, ch, t = sample(protos, m, rng)
        work["rewired"] += net.connect(y, ch.tolist(), 0.0)
        syn = net.gather(ch)
        ids = np.concatenate(syn) if syn else np.zeros(0, np.int64)
        nodes = ids // FANIN
        work["synops"] += len(ids)
        touched, inv = np.unique(nodes, return_inverse=True)
        z = np.bincount(inv, weights=net.w.flat[ids], minlength=len(touched))
        # untouched nodes have logit 0; the partition function counts them analytically
        zmax = max(z.max() if len(z) else 0.0, 0.0)
        denom = np.exp(z - zmax).sum() + (k - len(touched)) * np.exp(-zmax)
        p = np.exp(z - zmax) / denom
        g = p - (touched == y)
        net.w.flat[ids] -= a.lr * g[inv]
        work["plasticity"] += len(ids)
    test_rng = np.random.default_rng(10_000 + seed)
    correct, syn_total = 0, 0
    for _ in range(a.test):
        y, ch, t = sample(protos, m, test_rng)
        ids = np.concatenate(net.gather(ch))
        syn_total += len(ids)
        touched, inv = np.unique(ids // FANIN, return_inverse=True)
        z = np.bincount(inv, weights=net.w.flat[ids], minlength=len(touched))
        correct += len(z) and z.max() > 0 and touched[z.argmax()] == y
    out["sparse"] = {"acc": correct / a.test, "train_per_episode": {q: v / n_train for q, v in work.items()},
                     "inference_per_sample": {"synops": syn_total / a.test}}

    active = PROTO * KEEP + DISTRACT
    out["dense"] = {"inference_per_sample": {"macs": k * active}, "train_per_episode": {"macs": 2 * k * active}}
    return out


def tune(a):
    import copy
    grid = []
    for eta in (0.1, 0.3):
        for beta in (0.06, 0.1, 0.15):
            for margin in (0.3, 0.5):
                b = copy.copy(a); b.eta, b.beta, b.margin, b.w0, b.models = eta, beta, margin, 0.005, ["race"]
                grid.append((1024, 100, b))
    for lr in (0.5,):
        b = copy.copy(a); b.lr, b.models = lr, ["sparse"]
        grid.append((1024, 100, b))
    with Pool(a.procs) as pool:
        rows = pool.map(run, grid, chunksize=1)
    best = {}
    for (_, _, b), r in zip(grid, rows):
        model = b.models[0]
        cfg = {"eta": b.eta, "beta": b.beta, "margin": b.margin, "w0": b.w0} if model == "race" else {"lr": b.lr}
        print(model, cfg, round(r[model]["acc"], 3))
        if model not in best or r[model]["acc"] > best[model]["acc"]:
            best[model] = dict(cfg, acc=r[model]["acc"])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tuned.json"), "w") as f:
        json.dump(best, f, indent=1)
    print(best)


def main(a):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(k, s, a) for k in a.ks for s in range(a.seeds)]
    with Pool(a.procs) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    with open(os.path.join(OUT, "rows.json"), "w") as f:
        json.dump({"config": {q: v for q, v in vars(a).items()}, "rows": rows}, f, indent=1)
    print(f"{'K':>6} {'model':7} {'acc':>6} {'inf work':>10} {'learn work':>11} {'inputs used':>12}")
    for k in a.ks:
        rs = [r for r in rows if r["k"] == k]
        for model in ("race", "sparse", "dense"):
            acc = np.mean([r[model].get("acc", np.nan) for r in rs])
            inf = np.mean([sum(v for q, v in r[model]["inference_per_sample"].items() if q in ("synops", "macs")) for r in rs])
            tr = np.mean([sum(v for q, v in r[model]["train_per_episode"].items() if q in ("plasticity", "macs")) for r in rs])
            used = np.mean([r[model]["inference_per_sample"].get("inputs_used", np.nan) / r[model]["inference_per_sample"].get("inputs", np.nan)
                            if "inputs" in r[model]["inference_per_sample"] else np.nan for r in rs])
            print(f"{k:6d} {model:7} {acc:6.3f} {inf:10.1f} {tr:11.1f} {used:12.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train-per-class", type=int, default=8)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--teach-delay", type=float, default=0.05)
    ap.add_argument("--w0", type=float, default=0.005)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--models", nargs="+", default=["race", "sparse"])
    ap.add_argument("--tune", action="store_true", help="grid on tuning seed 100 at K=1024")
    a = ap.parse_args()
    if a.tune:
        tune(a)
    else:
        main(a)
