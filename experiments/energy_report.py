#!/usr/bin/env python3
"""Energy estimates for E2, E4 and E6 from measured operation counts.

Headline rule: compare each event network with the cheapest dense model that is
at least as accurate (matched accuracy), for inference and for training, on
several hardware profiles. See sleeping_machines/energy.py for the sources.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sleeping_machines.energy import PROFILES, energy_joules, profiles_table  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")
EVENT_PROFILES = ("event_ideal", "event_shared_sram", "loihi")
CLOCK_STEPS = 256          # timesteps a clocked simulation needs to resolve an 8-bit latency code
TRAIN_SAMPLES = 60000


def run_round(name):
    """E6 rounds. 1: first full-data runs; 2: higher hidden threshold, 3 winners per
    group, LR decay (tag v2); 3: ramp synapses and collapsing bound (tags r3, r3patch)."""
    return 3 if "_r3" in name else 2 if name.endswith("_v2") else 1


def e6():
    out = {"race": [], "dense": []}
    for path in sorted(glob.glob(os.path.join(RES, "e6", "mnist_*_s0.json"))):
        name = os.path.basename(path)[6:-8]
        r = json.load(open(path))
        if r.get("val") or any(tag in name for tag in ("pilot", "check", "timing")):
            continue                           # tuning runs on the validation split are not results
        cfg = r["config"]
        inf, tr = r["inference_work_per_sample"], r["train_work"]
        if cfg["variant"] == "mlp":
            out["dense"].append(dense_row(f"MLP {cfg['hidden']} hidden (backprop)", r, cfg["hidden"], cfg["epochs"]))
            continue
        h = cfg["hidden"] if cfg["variant"] != "single_layer" else 0
        inf_counts = {"synops": inf["synops"], "spikes": inf["input_spikes"] + inf["hidden_spikes"] + 1}
        tr_counts = {"synops": tr["synops"], "spikes": tr["input_spikes"] + tr["hidden_spikes"],
                     "plasticity": tr["plasticity"]}
        n_trained = cfg["epochs"] * TRAIN_SAMPLES
        clocked = dict(inf_counts, neuron_steps=(h + 10) * CLOCK_STEPS)
        clocked_tr = dict(tr_counts, neuron_steps=(h + 10) * CLOCK_STEPS * n_trained)
        row = {"name": name, "variant": cfg["variant"], "round": run_round(name), "patch": cfg.get("patch", 0),
               "hidden": h, "winners": cfg["winners"],
               "acc": r["test_acc"], "curve": r["curve"], "counts_inference": inf_counts,
               "counts_training": tr_counts,
               "inference_J": {p: energy_joules(inf_counts, p) for p in EVENT_PROFILES},
               "training_J": {p: energy_joules(tr_counts, p) for p in EVENT_PROFILES}}
        row["inference_J"]["clocked_snn"] = energy_joules(clocked, "clocked_snn")
        row["training_J"]["clocked_snn"] = energy_joules(clocked_tr, "clocked_snn")
        out["race"].append(row)
    frontier = os.path.join(RES, "e6", "dense_frontier.json")
    if os.path.exists(frontier):
        for r in json.load(open(frontier)):
            label = (f"linear, {28 // r.get('pool', 1)}x{28 // r.get('pool', 1)} input" if r["hidden"] == 0
                     else f"MLP {r['hidden']} hidden (backprop)")
            out["dense"].append(dense_row(label, r, r["hidden"], 10))
    for row in out["race"]:
        row["matched"] = matched(row, out["dense"])
    return out


def dense_row(label, r, hidden, epochs):
    macs = r["inference_work_per_sample"]["macs"]
    params = macs                               # one weight per MAC in these dense nets
    steps = int(np.ceil(TRAIN_SAMPLES / 32)) * epochs
    tr = {"macs": r["train_work"]["macs"], "plasticity": params * steps}
    return {"name": label, "hidden": hidden, "acc": r["test_acc"], "curve": r["curve"],
            "macs": macs, "counts_training": tr,
            "inference_J": {p: energy_joules({"macs": macs}, p) for p in ("dense_int8", "dense_int8_batched")},
            "training_J": {p: energy_joules(tr, p) for p in ("dense_fp16_train", "dense_fp16_train_batched")}}


def matched(race, dense):
    """Cheapest dense model at least as accurate; training energy is to the first
    epoch that reaches the race network's accuracy (a fraction of the full run)."""
    ok = [d for d in dense if d["acc"] >= race["acc"]]
    if not ok:
        return None
    best = min(ok, key=lambda d: d["macs"])
    epochs_needed = next(i + 1 for i, a in enumerate(best["curve"]) if a >= race["acc"])
    frac = epochs_needed / len(best["curve"])
    tr = {p: v * frac for p, v in best["training_J"].items()}
    ev_inf, ev_tr = race["inference_J"]["event_ideal"], race["training_J"]["event_ideal"]
    return {"dense": best["name"], "dense_acc": best["acc"], "dense_epochs_to_match": epochs_needed,
            "inference_ratio_vs_int8_batch1": best["inference_J"]["dense_int8"] / ev_inf,
            "inference_ratio_vs_int8_batched": best["inference_J"]["dense_int8_batched"] / ev_inf,
            "training_ratio_vs_fp16": tr["dense_fp16_train"] / ev_tr,
            "training_ratio_vs_fp16_batched": tr["dense_fp16_train_batched"] / ev_tr,
            "break_even_synop_pj_batch1": best["inference_J"]["dense_int8"] / 1e-12
            / (race["counts_inference"]["synops"] + 2.0 * race["counts_inference"]["spikes"] / 1.3)}


def e4():
    path = os.path.join(RES, "e4", "main.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    rows = []
    for k in sorted({r["k"] for r in d["rows"]}):
        for rule in ("cf_margin", "fired_reward", "softmax"):
            rs = [r for r in d["rows"] if r["k"] == k and r["rule"] == rule]
            acc = float(np.mean([r["test_acc"] for r in rs]))
            w = {key: float(np.mean([r["train_work"].get(key, 0) for r in rs]))
                 for key in ("synops", "plasticity", "fired", "inhibition")}
            n = d["train"]
            if rule == "softmax":
                counts = {"macs": w["plasticity"], "plasticity": w["plasticity"]}   # forward K*n + update K*n
                J = {p: energy_joules(counts, p) for p in ("dense_fp16_train", "dense_fp16_train_batched")}
            else:
                counts = {"synops": w["synops"] + w["inhibition"], "spikes": w["fired"],
                          "plasticity": w["plasticity"]}
                J = {p: energy_joules(counts, p) for p in EVENT_PROFILES}
            rows.append({"k": k, "rule": rule, "acc": acc, "per_episode_counts": {c: v / n for c, v in counts.items()},
                         "training_J_per_episode": {p: v / n for p, v in J.items()}})
    return rows


def e2():
    path = os.path.join(RES, "e2", "rows.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))["rows"]
    agg = {}
    for r in d:
        agg.setdefault((r["decoder"], r["param"]), []).append(r)
    curves = {}
    for (dec, p), rs in sorted(agg.items()):
        curves.setdefault(dec, []).append({
            "param": p, "acc": float(np.mean([r["correct"] for r in rs])),
            "time": float(np.mean([r["time"] for r in rs])),
            "events": float(np.mean([r["events"] for r in rs])),
            "by_coherence": {str(c): {"time": float(np.mean([r["time"] for r in rs if r["c"] == c])),
                                      "acc": float(np.mean([r["correct"] for r in rs if r["c"] == c]))}
                             for c in sorted({r["c"] for r in rs})}})
    return curves


if __name__ == "__main__":
    report = {"profiles": profiles_table(), "e6": e6(), "e4": e4(), "e2": e2()}
    with open(os.path.join(RES, "energy.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("E6 (MNIST) — event network vs cheapest dense model of equal or better accuracy")
    for r in report["e6"]["race"]:
        m = r["matched"]
        line = (f"  {r['name']:28s} acc {r['acc']:.4f}  synops/inf {r['counts_inference']['synops']:9.0f}  "
                f"E_inf(ideal) {r['inference_J']['event_ideal'] * 1e9:7.2f} nJ  "
                f"E_train(ideal) {r['training_J']['event_ideal']:.3g} J")
        if m:
            line += (f"\n      vs {m['dense']} ({m['dense_acc']:.4f}): inference x{m['inference_ratio_vs_int8_batch1']:.2f} "
                     f"(batched x{m['inference_ratio_vs_int8_batched']:.2f}), training x{m['training_ratio_vs_fp16']:.2f} "
                     f"(batched x{m['training_ratio_vs_fp16_batched']:.2f}); break-even synop {m['break_even_synop_pj_batch1']:.2f} pJ")
        print(line)
    for d in report["e6"]["dense"]:
        print(f"  {d['name']:28s} acc {d['acc']:.4f}  MACs {d['macs']:8d}  E_inf(int8) {d['inference_J']['dense_int8'] * 1e9:7.2f} nJ")
