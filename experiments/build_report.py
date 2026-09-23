#!/usr/bin/env python3
"""Assemble results into report/data.json and inline it into report/index.html."""
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "experiments", "results")
REPORT = os.path.join(ROOT, "report")
E4_RULES = ("softmax", "fired_only", "fired_reward", "cf_winner", "cf_uniform", "cf_margin")


def ci95(x):
    x = np.asarray(x, float)
    return float(1.96 * x.std() / np.sqrt(len(x))) if len(x) > 1 else 0.0


def e4():
    d = json.load(open(os.path.join(RES, "e4", "main.json")))
    ks = sorted({r["k"] for r in d["rows"]})
    table = []
    for rule in E4_RULES:
        for k in ks:
            rs = [r for r in d["rows"] if r["rule"] == rule and r["k"] == k]
            acc = [r["test_acc"] for r in rs]
            table.append({"rule": rule, "k": k, "acc": float(np.mean(acc)), "ci": ci95(acc),
                          "plasticity": float(np.mean([r["train_work"].get("plasticity", 0) for r in rs])),
                          "curve": np.mean([r["online_curve"] for r in rs], 0).round(4).tolist()})
    delay = json.load(open(os.path.join(RES, "e4", "delay.json")))
    drows = []
    for gap in sorted({r["gap"] for r in delay}):
        for dl in sorted({r["delay"] for r in delay}):
            acc = [r["test_acc"] for r in delay if r["gap"] == gap and r["delay"] == dl]
            drows.append({"gap": gap, "delay": dl, "acc": float(np.mean(acc)), "ci": ci95(acc)})
    return {"ks": ks, "train_episodes": d["train"], "table": table, "delay": drows}


def e6():
    runs = []
    for path in sorted(glob.glob(os.path.join(RES, "e6", "mnist_*_s0.json"))):
        name = os.path.basename(path)[6:-8]
        r = json.load(open(path))
        kind = "pilot" if "pilot" in name else ("tuning" if "_sl" in name else "final")
        runs.append({"name": name, "kind": kind, "variant": r["config"]["variant"],
                     "hidden": r["config"]["hidden"], "winners": r["config"]["winners"],
                     "train_samples": 6000 if kind == "pilot" else 60000,
                     "acc": r["test_acc"], "curve": r["curve"], "val": r.get("val", 0)})
    xor = []
    for path in sorted(glob.glob(os.path.join(RES, "e6", "xor_*_smoke_s0.json"))):
        r = json.load(open(path))
        xor.append({"variant": r["config"]["variant"], "acc": r["test_acc"]})
    return {"runs": runs, "xor": xor}


if __name__ == "__main__":
    energy = json.load(open(os.path.join(RES, "energy.json")))
    data = {"e4": e4(), "e2": energy["e2"], "e6": e6(), "energy": {
        "profiles": energy["profiles"], "e4": energy["e4"], "e6": energy["e6"]}}
    os.makedirs(REPORT, exist_ok=True)
    blob = json.dumps(data, separators=(",", ":"))
    with open(os.path.join(REPORT, "data.json"), "w") as f:
        f.write(blob)
    template = open(os.path.join(REPORT, "template.html")).read()
    with open(os.path.join(REPORT, "index.html"), "w") as f:
        f.write(template.replace("/*__DATA__*/null", blob))
    print(f"report/index.html written ({len(blob) / 1024:.0f} KB of data)")
