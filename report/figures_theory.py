#!/usr/bin/env python3
"""Figures for the theory checks and the newest experiments.

Each function draws only from results that exist and returns None otherwise, so the
report builds at any stage of the queue.

    python report/figures_theory.py      # writes report/figures/*.png
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "experiments", "results")
FIG = os.path.join(ROOT, "report", "figures")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, MUTED, GRID = "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False, "savefig.dpi": 200,
})


def load(p):
    with open(p) as f:
        return json.load(f)


def fig_alignment():
    """M3: cosine between each rule's update and the true gradient of expected error."""
    runs = [load(p) for p in sorted(glob.glob(os.path.join(RES, "theory", "m3_main_s*.json")))]
    if not runs:
        return None
    rules = [("crl_fired_only", "fired-only credit"), ("crl_fa", "counterfactual, random feedback"),
             ("crl_sign", "counterfactual, sign feedback"), ("crl_sym", "counterfactual, true weights")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    for ax, layer, title in ((axes[0], "W1", "hidden weights"), (axes[1], "W2", "output weights")):
        ceiling = np.mean([r["fd_split_cosine"][layer] for r in runs])
        ax.axvspan(np.sqrt(max(ceiling, 0)), 1.0, color=GRID, alpha=0.6, lw=0)
        ax.axvline(0, color=INK, lw=0.8)
        for i, (rule, name) in enumerate(rules):
            xs = [r["rules"][rule][layer]["cosine"] for r in runs]
            ax.plot(xs, [i] * len(xs), "o", color=BLUE if layer == "W1" else AQUA, alpha=0.55, ms=5)
            ax.plot([np.mean(xs)], [i], "|", color=INK, ms=14, mew=2)
        ax.set_xlim(-0.4, 1.0)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("cosine with the true gradient")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rules)), [n for _, n in rules])
    axes[0].invert_yaxis()
    fig.suptitle(f"M3 · does the update point downhill? ({len(runs)} seeds; bar = mean; "
                 "shaded = beyond the estimate's reliability)", x=0.02, ha="left",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_blind_spot():
    """Share of weights whose true gradient is nonzero, by timing-noise level."""
    pts = []
    for tag, sigma in (("sig01", 0.01), ("main", 0.03), ("sig10", 0.1)):
        rs = [load(p) for p in glob.glob(os.path.join(RES, "theory", f"m3_{tag}_s*.json"))]
        if rs:
            pts.append((sigma, np.mean([r["grad_nonzero_frac"]["W1"] for r in rs]),
                        np.mean([r["grad_nonzero_frac"]["W2"] for r in rs])))
    if len(pts) < 2:
        return None
    s, w1, w2 = map(np.array, zip(*pts))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(s, w1, "o-", color=BLUE)
    ax.plot(s, w2, "o-", color=AQUA)
    ax.annotate("hidden weights", (s[-1], w1[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                fontsize=8, color=MUTED)
    ax.annotate("output weights", (s[-1], w2[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(s[0] / 1.5, s[-1] * 4)
    ax.set_ylim(0, 1)
    ax.set_xlabel("timing noise σ (log): how wide the tree of histories is")
    ax.set_ylabel("weights with any gradient")
    ax.set_title("The blind spot: most hidden weights get no gradient at all")
    return fig


def fig_ladder():
    """Where E6 round 3's accuracy comes from."""
    e6 = os.path.join(RES, "e6")
    get = lambda n: load(os.path.join(e6, n))["test_acc"] if os.path.exists(os.path.join(e6, n)) else None  # noqa: E731
    rows = [("frozen random hidden layer", get("mnist_frozen_hidden_r3_s0.json"), GRAY),
            ("single racing layer", get("mnist_single_layer_r3_s0.json"), GRAY),
            ("hidden 1000, counterfactual credit", get("mnist_crl_fa_r3_s0.json"), BLUE),
            ("hidden 1000, fired-only credit", get("mnist_crl_fired_only_r3_s0.json"), BLUE),
            ("hidden 2000, counterfactual credit", get("mnist_crl_fa_r3_h2000_s0.json"), BLUE),
            ("dense MLP, backprop (reference)", 0.9783, INK)]
    rows = [r for r in rows if r[1] is not None]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for i, (name, acc, c) in enumerate(rows):
        ax.barh(i, acc - 0.85, left=0.85, height=0.62, color=c, alpha=0.9 if c != GRAY else 0.55)
        ax.text(acc + 0.002, i, f"{acc:.3f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.85, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MNIST test accuracy (round-3 settings, seed 0)")
    ax.set_title("E6 round 3 · depth adds ~4 points; which credit, and width, add nothing")
    return fig


def fig_projection():
    """M13: half-space projection vs the near-miss rule (single layer, validation)."""
    base = os.path.join(RES, "e6", "mnist_single_layer_m13base_s0.json")
    runs = sorted(glob.glob(os.path.join(RES, "theory", "m13_*_s0.json")))
    if not runs or not os.path.exists(base):
        return None
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    for p in runs:
        c = load(p)["curve"]
        ax.plot(range(1, len(c) + 1), c, color=BLUE, alpha=0.35, lw=1.3)
    b = load(base)["curve"]
    ax.plot(range(1, len(b) + 1), b, color=ORANGE, lw=2.2)
    ax.text(len(b) + 0.1, b[-1], "near-miss rule", color=MUTED, fontsize=8, va="center")
    ax.text(len(b) + 0.1, np.mean([load(p)["curve"][-1] for p in runs]) - 0.002,
            f"projection\n({len(runs)} settings)", color=MUTED, fontsize=8, va="top")
    ax.set_xlim(0.8, len(b) + 1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title("M13 · learning by rescheduling ties the near-miss rule")
    return fig


def fig_repair():
    """M18: accuracy against weights touched, per rule and seed."""
    runs = [load(p) for p in glob.glob(os.path.join(RES, "theory", "m18_*_main_s*.json"))]
    if not runs:
        return None
    runs = [load(p) for p in glob.glob(os.path.join(RES, "theory", "m18_*_s[0-9].json"))]
    style = {("repair", "main"): ("repair", BLUE), ("repair", "homeothin"): ("repair + thin margins", "#174a8c"),
             ("output_only", "main"): ("output-only repair", AQUA),
             ("crl_fa", "main"): ("counterfactual credit", ORANGE),
             ("crl_fired_only", "main"): ("fired-only credit", YELLOW),
             ("frozen_hidden", "main"): ("frozen hidden", GRAY),
             ("unsup_hidden", "main"): ("label-free hidden", GRAY)}
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for (rule, tag), (name, c) in style.items():
        pts = [(r["repairs"]["weights_touched"] if r["repairs"] else r["plasticity"], r["acc"])
               for r in runs if r["config"]["rule"] == rule and r["config"]["tag"] == tag]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, "o", color=c, ms=6, alpha=0.85)
            dx, dy, ha = {"fired-only credit": (-8, -12, "right"),
                          "counterfactual credit": (-8, 12, "right")}.get(name, (7, 0, "left"))
            ax.annotate(name, (np.mean(x), np.mean(y)), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                        color=MUTED, va="center", ha=ha)
    ax.set_xscale("log")
    ax.set_xlabel("weights changed during training (log)")
    ax.set_ylabel("held-out accuracy")
    ax.set_title("M18 · accuracy vs how much of the network was touched")
    return fig


def fig_tree():
    """Schematic: the tree of histories and its three readings."""
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 4.3)
    # realised path
    path = [(0.4, 2), (2.2, 2), (4.0, 2), (5.8, 2), (7.6, 2)]
    ax.plot(*zip(*path), "-", color=INK, lw=2.4)
    for x, y in path[1:-1]:
        ax.plot(x, y, "o", color=INK, ms=6)
    ax.plot(*path[-1], "s", color=ORANGE, ms=9)
    ax.text(7.85, 2, "realised outcome (wrong)", va="center", fontsize=7.5, color=INK)
    # forks
    forks = [(2.2, 2, 3.6, 0.30, "right"), (4.0, 2, 0.6, 0.08, "wrong"), (5.8, 2, 2.9, 0.22, "right")]
    for x, y, y2, p, outcome in forks:
        ax.plot([x, x + 1.4, 7.6], [y, y2, y2], "--", color=BLUE, lw=1.4)
        ax.plot(7.6, y2, "s", color=AQUA if outcome == "right" else ORANGE, ms=8)
        ax.text(x + 0.2, (y + y2) / 2 + (0.15 if y2 > y else -0.3), f"p={p:.2f}", fontsize=7, color=BLUE)
    ax.text(7.85, 3.6, "right (repair cost 0.9)", va="center", fontsize=7.5, color=MUTED)
    ax.text(7.85, 2.9, "right (repair cost 0.4)", va="center", fontsize=7.5, color=MUTED)
    ax.text(7.85, 0.6, "wrong", va="center", fontsize=7.5, color=MUTED)
    ax.text(0.4, 4.15, "collapses along the realised history (solid); close calls fork (dashed)",
            fontsize=7.5, color=MUTED)
    ax.text(0.4, -0.2, "greedy backprop: slope at the solid path only · holistic: sum over forks weighted by p · "
            "repair: cheapest fork that is right", fontsize=7.2, color=INK)
    return fig


def fig_depth():
    """E14: accuracy vs depth per credit type, and the share of hidden nodes that receive credit."""
    runs = [load(p) for p in glob.glob(os.path.join(RES, "e14", "d*_main_s*.json"))]
    if not runs:
        return None
    style = {"crl_fa": ("counterfactual credit", BLUE), "crl_fired_only": ("fired-only credit", ORANGE),
             "crl_drtp": ("label templates (DRTP)", AQUA), "frozen_hidden": ("frozen hidden", GRAY)}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for v, (name, c) in style.items():
        pts = {}
        for r in runs:
            if r["config"]["variant"] == v:
                pts.setdefault(r["config"]["depth"], []).append(r)
        if not pts:
            continue
        ds = sorted(pts)
        acc = [np.mean([r["acc"] for r in pts[d]]) for d in ds]
        axes[0].plot(ds, acc, "o-", color=c, label=name)
        if v != "frozen_hidden":
            reach = [np.mean([np.mean([x for x in r["credit_reach"] if x is not None]) for r in pts[d]]) for d in ds]
            axes[1].plot(ds, reach, "o-", color=c)
    axes[0].set_xlabel("hidden layers")
    axes[0].set_ylabel("validation accuracy")
    axes[0].set_xticks([1, 2, 3])
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_title("accuracy", fontsize=8.5)
    axes[1].set_xlabel("hidden layers")
    axes[1].set_ylabel("share of hidden nodes credited")
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("credit reach", fontsize=8.5)
    n = len({r["config"]["seed"] for r in runs})
    fig.suptitle(f"E14 · the advantage of counterfactual credit grows with depth ({n} seed{'s' if n > 1 else ''})",
                 x=0.02,
                 ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_bandit():
    """E13a: reward-only learning, accuracy per rule."""
    runs = [load(p) for p in glob.glob(os.path.join(RES, "e13", "bandit_*_main_s*.json"))]
    if not runs:
        return None
    names = {"supervised": "labels (reference)", "nearmiss": "near-miss guess", "rstdp": "reward-modulated",
             "pool_pg": "pool policy gradient"}
    rules = sorted({r["config"]["rule"] for r in runs}, key=lambda k: -np.mean(
        [r["acc"] for r in runs if r["config"]["rule"] == k]))
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    for i, k in enumerate(rules):
        xs = [r["acc"] for r in runs if r["config"]["rule"] == k]
        ax.barh(i, np.mean(xs), color=GRAY if k == "supervised" else BLUE, height=0.6)
        ax.plot(xs, [i] * len(xs), "o", color=INK, ms=3)
        ax.text(np.mean(xs) + 0.01, i, f"{np.mean(xs):.3f}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(rules)), [names.get(k, k) for k in rules])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("held-out accuracy (reward only, except the reference)")
    ax.set_title("E13a · learning from reward alone")
    return fig


FIGS = {"e14_depth": fig_depth, "e13_bandit": fig_bandit,"m3_alignment": fig_alignment, "m3_blind_spot": fig_blind_spot, "e6_ladder": fig_ladder,
        "m13_projection": fig_projection, "m18_repair": fig_repair, "history_tree": fig_tree}


def main():
    for name, fn in FIGS.items():
        fig = fn()
        if fig is None:
            print("skip", name)
            continue
        fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    main()
