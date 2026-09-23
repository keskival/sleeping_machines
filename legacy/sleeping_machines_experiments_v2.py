#!/usr/bin/env python3
"""Sleeping Machines v2 reproducible feasibility experiments.

Dependencies: numpy, torch, matplotlib

Example:
    python sleeping_machines_experiments_v2.py --epochs 200 --seeds 5

Outputs:
    results.json
    learning_curves.png
    counterfactual_delays.png

This is an experimental viability suite, not a proof of general superiority.
"""

import argparse, json, math, os, random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_dataset(n, noise=0.08, n_distractors=8, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 2 + n_distractors, 2), dtype=np.float32)
    y = rng.integers(0, 2, size=n, dtype=np.int64)
    for i in range(n):
        base = rng.uniform(.35, .55)
        gap = rng.uniform(.18, .38)
        if y[i] == 0:
            ta = base + rng.normal(0, noise)
            tb = base + gap + rng.normal(0, noise)
        else:
            tb = base + rng.normal(0, noise)
            ta = base + gap + rng.normal(0, noise)
        X[i, 0] = [0., ta]
        X[i, 1] = [1., tb]
        for j in range(n_distractors):
            X[i, 2+j] = [rng.integers(2, 8), rng.uniform(.05, .95)]
        X[i] = X[i, np.argsort(X[i, :, 1])]
    return X, y


def scramble_times(X, seed=0):
    rng = np.random.default_rng(seed)
    Y = X.copy()
    for i in range(len(Y)):
        typ = Y[i, :, 0].copy()
        rng.shuffle(typ)
        Y[i, :, 0] = typ
        Y[i] = Y[i, np.argsort(Y[i, :, 1])]
    return Y


class RateMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, x):
        typ = x[..., 0].long().clamp(0, 7)
        counts = F.one_hot(typ, 8).float().sum(1)
        return self.net(counts)


class GRUModel(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.emb = nn.Embedding(8, 12)
        self.gru = nn.GRU(13, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        typ = x[..., 0].long().clamp(0, 7)
        z = torch.cat([self.emb(typ), x[..., 1:2]], -1)
        h, _ = self.gru(z)
        return self.head(h[:, -1])


class TemporalRace(nn.Module):
    def __init__(self, learn_delay=True, beta=5.):
        super().__init__()
        self.beta = beta
        self.raw_weight = nn.Parameter(torch.zeros(8, 2))
        if learn_delay:
            self.raw_delay = nn.Parameter(torch.full((8, 2), -1.0))
        else:
            self.register_buffer("raw_delay", torch.full((8, 2), -1.0))
        self.learn_delay = learn_delay

    def delays(self):
        return F.softplus(self.raw_delay)

    def candidate_times(self, x):
        typ = x[..., 0].long().clamp(0, 7)
        t = x[..., 1]
        w = F.softplus(self.raw_weight[typ]) + 1e-5
        d = self.delays()[typ]
        return t[..., None] + d - torch.log(w) / self.beta

    def forward(self, x):
        b = self.beta
        typ = x[..., 0].long().clamp(0, 7)
        t = x[..., 1]
        w = F.softplus(self.raw_weight[typ]) + 1e-5
        d = self.delays()[typ]
        T = t[..., None] + d - torch.log(w) / b
        softmin = -torch.logsumexp(-b * T, dim=1) / b
        return -softmin


def batches(X, y, size=128):
    idx = np.arange(len(X)); np.random.shuffle(idx)
    for s in range(0, len(idx), size):
        ii = idx[s:s+size]
        yield torch.from_numpy(X[ii]), torch.from_numpy(y[ii])


def train(model, Xtr, ytr, Xte, yte, epochs, lr=3e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    for ep in range(epochs):
        model.train()
        if isinstance(model, TemporalRace):
            p = ep / max(1, epochs-1)
            model.beta = 5. + p * 25.
        for xb, yb in batches(Xtr, ytr):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            if isinstance(model, TemporalRace):
                loss = loss + 1e-3 * (model.delays() ** 2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            z = model(torch.from_numpy(Xte).to(device))
            acc = (z.argmax(-1) == torch.from_numpy(yte).to(device)).float().mean()
            curve.append(float(acc))
    return {"test_accuracy": curve[-1], "curve": curve}


@dataclass
class Candidate:
    name: str
    time: float
    status: str = "pending"
    margin: float = None
    eligibility: float = 0.


@dataclass
class LocalHistory:
    fired: List[Candidate] = field(default_factory=list)
    cancelled: List[Candidate] = field(default_factory=list)


def resolve_race(candidates, beta=6., remember=True, use_margin=True):
    candidates = sorted(candidates, key=lambda e: e.time)
    winner = candidates[0]; winner.status = "fired"
    hist = LocalHistory(fired=[winner])
    for e in candidates[1:]:
        e.status = "cancelled"
        e.margin = e.time - winner.time
        e.eligibility = math.exp(-beta * e.margin) if use_margin else 1.
        if remember: hist.cancelled.append(e)
    return winner, hist


def counterfactual_learning(condition, trials=500, lr=.02, beta=6., seed=0):
    rng = np.random.default_rng(seed)
    d = {"A": .20, "B": .30}
    trajectory, successes = [], []
    for trial in range(trials):
        A = Candidate("A", d["A"] + rng.normal(0, .004))
        B = Candidate("B", d["B"] + rng.normal(0, .004))
        winner, hist = resolve_race(
            [A, B], beta=beta,
            remember=(condition != "none"),
            use_margin=(condition == "margin"))
        desired = "B" if (trial // 50) % 2 == 0 else "A"
        candidates = {"A": A, "B": B}
        for name, e in candidates.items():
            if name == winner.name:
                elig = 1.
            elif condition == "none":
                elig = 0.
            elif condition == "binary":
                elig = 1.
            else:
                elig = e.eligibility
            signal = 1. if name == desired else (-1. if name == winner.name else 0.)
            d[name] -= lr * signal * elig
        d["A"] = float(np.clip(d["A"], .02, .90))
        d["B"] = float(np.clip(d["B"], .02, .90))
        trajectory.append([d["A"], d["B"]])
        successes.append(winner.name == desired)
    return {
        "final_delay_A": d["A"], "final_delay_B": d["B"],
        "desired_winner_accuracy_last_200": float(np.mean(successes[-200:])),
        "trajectory": trajectory
    }


def event_economy(nodes=1000, input_events=100, fanout=8, race_width=8, timesteps=1000):
    dense = nodes * timesteps
    scheduled1 = input_events * fanout
    fired1 = max(1, scheduled1 // race_width)
    scheduled2 = fired1 * fanout
    fired2 = max(1, scheduled2 // race_width)
    scheduled = scheduled1 + scheduled2
    fired = fired1 + fired2
    cancelled = scheduled - fired
    event_ops = scheduled + fired + cancelled * 0 + fired1 + fired2
    return {
        "dense_ops": dense,
        "scheduled_events": scheduled,
        "fired_events": fired,
        "cancelled_events": cancelled,
        "event_ops_estimate": event_ops,
        "dense_to_event_ratio": dense / max(1, event_ops)
    }


def save_learning_plot(curves, path):
    plt.figure(figsize=(7, 4))
    for name, curve in curves.items(): plt.plot(curve, label=name)
    plt.xlabel("Epoch"); plt.ylabel("Test accuracy")
    plt.title("Sleeping Machines: temporal learning benchmark")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def save_cf_plot(results, path):
    plt.figure(figsize=(7, 4))
    for name, r in results.items():
        a = np.asarray(r["trajectory"])
        plt.plot(a[:, 0], label=f"{name}: dA")
        plt.plot(a[:, 1], "--", label=f"{name}: dB")
    plt.xlabel("Trial"); plt.ylabel("Delay")
    plt.title("Counterfactual local delay learning")
    plt.legend(ncol=2); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def run(args):
    os.makedirs(args.out, exist_ok=True)
    results = {"config": vars(args), "seeds": {}}

    for seed in range(args.seeds):
        seed_all(seed)
        Xtr, ytr = make_dataset(args.train, args.noise, args.distractors, 1000+seed)
        Xte, yte = make_dataset(args.test, args.noise, args.distractors, 2000+seed)

        count = train(RateMLP(), Xtr, ytr, Xte, yte, args.epochs, device=args.device)
        gru = train(GRUModel(), Xtr, ytr, Xte, yte, args.epochs, device=args.device)
        race_model = TemporalRace(True)
        race = train(race_model, Xtr, ytr, Xte, yte, args.epochs, device=args.device)
        fixed = train(TemporalRace(False), Xtr, ytr, Xte, yte, args.epochs, device=args.device)

        race_model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(Xte).to(args.device)
            yt = torch.from_numpy(yte).to(args.device)
            intact = (race_model(xt).argmax(-1) == yt).float().mean().item()
            scr = torch.from_numpy(scramble_times(Xte, 3000+seed)).to(args.device)
            scrambled = (race_model(scr).argmax(-1) == yt).float().mean().item()
            T = race_model.candidate_times(xt)
            hard = (T.min(1).values.argmin(-1) == yt).float().mean().item()

        cf = {c: counterfactual_learning(c, args.cf_trials, seed=4000+seed)
              for c in ["none", "binary", "margin"]}

        results["seeds"][str(seed)] = {
            "count_mlp": count,
            "gru": gru,
            "temporal_race": race,
            "fixed_delay_race": fixed,
            "timing_ablation": {
                "intact_accuracy": intact,
                "scrambled_accuracy": scrambled,
                "timing_drop": intact - scrambled},
            "soft_vs_hard": {
                "soft_accuracy": race["test_accuracy"],
                "hard_accuracy": hard,
                "soft_hard_gap": race["test_accuracy"] - hard},
            "delay_advantage": {
                "learned_delay_accuracy": race["test_accuracy"],
                "fixed_delay_accuracy": fixed["test_accuracy"],
                "advantage": race["test_accuracy"] - fixed["test_accuracy"],
                "mean_learned_delay": float(race_model.delays().mean())},
            "counterfactual": cf,
            "event_economy": event_economy(args.nodes, args.input_events,
                                           args.fanout, args.race_width,
                                           args.timesteps)
        }

    def vals(path):
        out = []
        for sr in results["seeds"].values():
            x = sr
            for p in path: x = x[p]
            out.append(float(x))
        return np.asarray(out)

    summary = {}
    for key, path in {
        "count_mlp_accuracy": ["count_mlp", "test_accuracy"],
        "gru_accuracy": ["gru", "test_accuracy"],
        "temporal_race_accuracy": ["temporal_race", "test_accuracy"],
        "fixed_delay_accuracy": ["fixed_delay_race", "test_accuracy"],
        "timing_drop": ["timing_ablation", "timing_drop"],
        "learned_delay_advantage": ["delay_advantage", "advantage"],
        "soft_hard_gap": ["soft_vs_hard", "soft_hard_gap"],
        "event_economy_ratio": ["event_economy", "dense_to_event_ratio"]
    }.items():
        v = vals(path); summary[key] = {"mean": float(v.mean()), "std": float(v.std())}

    summary["counterfactual_accuracy"] = {}
    for c in ["none", "binary", "margin"]:
        v = vals(["counterfactual", c, "desired_winner_accuracy_last_200"])
        summary["counterfactual_accuracy"][c] = {"mean": float(v.mean()), "std": float(v.std())}

    results["summary"] = summary
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    s0 = results["seeds"]["0"]
    save_learning_plot({
        "count MLP": s0["count_mlp"]["curve"],
        "GRU": s0["gru"]["curve"],
        "temporal race": s0["temporal_race"]["curve"],
        "fixed delay race": s0["fixed_delay_race"]["curve"]},
        os.path.join(args.out, "learning_curves.png"))
    save_cf_plot(s0["counterfactual"], os.path.join(args.out, "counterfactual_delays.png"))

    print(json.dumps(summary, indent=2))
    print("Results:", os.path.join(args.out, "results.json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--train", type=int, default=4000)
    p.add_argument("--test", type=int, default=1500)
    p.add_argument("--noise", type=float, default=.08)
    p.add_argument("--distractors", type=int, default=8)
    p.add_argument("--cf-trials", type=int, default=500)
    p.add_argument("--nodes", type=int, default=1000)
    p.add_argument("--input-events", type=int, default=100)
    p.add_argument("--fanout", type=int, default=8)
    p.add_argument("--race-width", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="sleeping_machines_results")
    run(p.parse_args())


if __name__ == "__main__":
    main()
