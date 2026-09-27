"""E25 figure: test accuracy vs training fraction at p = 97, delay substrate vs dense backprop vs lookup."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R25, R24 = os.path.join(ROOT, "experiments/results/e25"), os.path.join(ROOT, "experiments/results/e24")
FIG = os.path.join(ROOT, "report/figures")
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
SERIES = [("sync_p97.json", "delays, replay (sleep)", "#2a78d6"),
          ("trace_p97_d0.5.json", "delays, online trace + downscaling", "#eb6834"),
          ("trace_p97_d0.json", "delays, online trace, no downscaling", "#1baf7a"),
          ("table_p97.json", "lookup table (race)", "#eda100")]


def curve(rows):
    fr = sorted({r["frac"] for r in rows})
    t = [[r["test"] for r in rows if r["frac"] == f] for f in fr]
    return np.array(fr), np.array([np.mean(x) for x in t]), np.array([np.min(x) for x in t]), \
        np.array([np.max(x) for x in t])


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "lines.linewidth": 2, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for fn, name, col in SERIES:
        f = os.path.join(R25, fn)
        if not os.path.exists(f):
            continue
        x, m, lo, hi = curve(json.load(open(f))["rows"])
        ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(x, m, color=col, marker="o", ms=5, label=name)
    mlp = []
    for f in glob.glob(os.path.join(R24, "mlp_p97_f*_s0.json")):
        d = json.load(open(f))
        mlp.append((d["args"]["frac"], d["curve"][-1]["test"]))
    if mlp:
        mlp.sort()
        ax.plot(*zip(*mlp), color="#e87ba4", marker="s", ms=5, label="dense MLP, backprop + AdamW (30k steps)")
    ax.axhline(1 / 97, color=MUTED, lw=1, ls=":")
    ax.text(0.5, 1 / 97 + 0.02, "chance", color=MUTED, ha="right", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    ax.set_xticklabels(["1%", "2%", "5%", "10%", "20%", "50%"])
    ax.set_xlabel("fraction of the p² pairs used for training (p = 97)")
    ax.set_ylabel("test accuracy on unseen pairs")
    ax.set_ylim(-0.02, 1.04)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="center right", fontsize=8)
    ax.set_title("(a + b) mod 97: a phase-restricted delay model vs a dense MLP", color=INK, fontsize=11, loc="left")
    fig.savefig(os.path.join(FIG, "e25_generalization.png"), dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
