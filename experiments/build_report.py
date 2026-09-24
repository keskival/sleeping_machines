#!/usr/bin/env python3
"""Assemble results into report/data.json and inline it into report/index.html."""
import glob
import json
import os

import numpy as np

from energy_report import run_round

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
        kind = "pilot" if "pilot" in name else ("tuning" if name.endswith("_sl") or name.endswith("v2sl") else "final")
        if kind == "final" and r.get("val"):
            kind = "tuning"
        cfg = r["config"]
        runs.append({"name": name, "kind": kind, "variant": cfg["variant"], "round": run_round(name),
                     "patch": cfg.get("patch", 0), "psp": cfg.get("psp", "step"), "deadline": cfg.get("deadline", 0),
                     "hidden": 0 if cfg["variant"] == "single_layer" else cfg["hidden"],
                     "winners": cfg["winners"], "lateral": cfg.get("lateral", 0), "lr_decay": cfg.get("lr_decay", 1.0),
                     "hid_frac": cfg.get("hid_frac", 0.0), "train_samples": 6000 if kind == "pilot" else 60000,
                     "acc": r["test_acc"], "peak": max(r["curve"]), "peak_epoch": int(np.argmax(r["curve"])) + 1,
                     "curve": r["curve"], "val": r.get("val", 0),
                     "synops": r["inference_work_per_sample"].get("synops"),
                     "macs": r["inference_work_per_sample"].get("macs")})
    xor = []
    for path in sorted(glob.glob(os.path.join(RES, "e6", "xor_*_smoke_s0.json"))):
        r = json.load(open(path))
        xor.append({"variant": r["config"]["variant"], "acc": r["test_acc"]})
    return {"runs": runs, "xor": xor}


def e5_rows(path):
    d = json.load(open(path))
    out = []
    for k in sorted({r["k"] for r in d["rows"]}):
        rs = [r for r in d["rows"] if r["k"] == k]
        row = {"k": k, "m": rs[0]["m"]}
        for model in ("race", "sparse"):
            row[model] = {
                "acc": float(np.mean([r[model]["acc"] for r in rs])), "acc_ci": ci95([r[model]["acc"] for r in rs]),
                "inference_synops": float(np.mean([r[model]["inference_per_sample"]["synops"] for r in rs])),
                "learning_updates": float(np.mean([r[model]["train_per_episode"]["plasticity"] for r in rs])),
                "rewired": float(np.mean([r[model]["train_per_episode"]["rewired"] for r in rs]))}
        row["race"]["inputs_used"] = float(np.mean([r["race"]["inference_per_sample"]["inputs_used"]
                                                    / r["race"]["inference_per_sample"]["inputs"] for r in rs]))
        row["dense"] = {"inference_macs": rs[0]["dense"]["inference_per_sample"]["macs"],
                        "learning_macs": rs[0]["dense"]["train_per_episode"]["macs"]}
        out.append(row)
    return {"config": d["config"], "rows": out}


def e5():
    """Round 1 is the preregistered run; round 2 (exploratory) adds the collapsing bound."""
    r1, r2 = (os.path.join(RES, "e5", f) for f in ("rows.json", "rows_r2.json"))
    if not os.path.exists(r1):
        return None
    out = e5_rows(r1)
    if os.path.exists(r2):
        out["round2"] = e5_rows(r2)
    return out


if __name__ == "__main__":
    energy = json.load(open(os.path.join(RES, "energy.json")))
    data = {"e4": e4(), "e2": energy["e2"], "e5": e5(), "e6": e6(), "energy": {
        "profiles": energy["profiles"], "e4": energy["e4"], "e6": energy["e6"]}}
    os.makedirs(REPORT, exist_ok=True)
    blob = json.dumps(data, separators=(",", ":"))
    with open(os.path.join(REPORT, "data.json"), "w") as f:
        f.write(blob)
    template = open(os.path.join(REPORT, "template.html")).read()
    with open(os.path.join(REPORT, "index.html"), "w") as f:
        f.write(template.replace("/*__DATA__*/null", blob))
    print(f"report/index.html written ({len(blob) / 1024:.0f} KB of data)")
