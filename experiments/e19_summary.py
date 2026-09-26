"""E19 summary: test accuracy vs training-set size per model (table + figure)."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results", "e19")
FIG = os.path.join(os.path.dirname(__file__), "..", "report", "figures", "e19_rhm.png")


def label(c):
    if c["model"] == "mlp":
        return f"backprop MLP, depth {c['depth']}"
    skip = {0: "", 1: ", plain skips", 2: ", delay-matched skips"}[c.get("residual", 0)]
    return f"race, depth {c['depth']}{skip}"


def main():
    rows = {}
    for p in glob.glob(os.path.join(RES, "*.json")):
        r = json.load(open(p))
        c = r["config"]
        rows.setdefault(label(c), {})[c["train"]] = (r["test_acc"], r["train_acc"])
    sizes = sorted({P for v in rows.values() for P in v})
    print("model".ljust(38) + "".join(f"P={P:<8}" for P in sizes))
    for k in sorted(rows):
        print(k.ljust(38) + "".join(f"{rows[k][P][0]:<10.3f}" if P in rows[k] else " " * 10 for P in sizes))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k in sorted(rows):
        Ps = sorted(rows[k])
        style = "--" if "MLP" in k else "-"
        ax.plot(Ps, [rows[k][P][0] for P in Ps], style, marker="o", ms=4, label=k)
    ax.axhline(1 / 8, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("training examples")
    ax.set_ylabel("test accuracy")
    ax.set_title("Random Hierarchy Model (8 classes, 3 levels): does depth exploit the hierarchy?", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG, dpi=160)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
