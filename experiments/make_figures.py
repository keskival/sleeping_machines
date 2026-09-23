#!/usr/bin/env python3
"""Render the report figures as PNGs (for REPORT.md on GitHub) from report/data.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "report", "figures")
S1, S2, S3, S4, S5, REF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6b7280"
INK, INK2, MUTED, HAIR = "#13161b", "#4a515c", "#7d848e", "#e1e4e9"
RULES = [("cf_margin", S1, "-", 2.6), ("cf_winner", S2, "-", 2), ("cf_uniform", S3, "-", 2),
         ("fired_reward", S4, "-", 2), ("fired_only", S5, "-", 2), ("softmax", REF, "--", 2)]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c4c9d1", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": HAIR, "grid.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11.5,
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def end_labels(ax, items, gap):
    """Direct labels at line ends, pushed apart so they do not collide."""
    items = sorted(items, key=lambda t: t[1])
    ys = []
    for _, y, *_ in items:
        ys.append(max(y, ys[-1] + gap) if ys else y)
    for (x, _, text, color, bold), y in zip(items, ys):
        ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK if bold else INK2, fontweight="bold" if bold else "normal", fontsize=9)


def race_diagram():
    fig, axes = plt.subplots(3, 1, figsize=(8, 3.6), sharex=True)
    lanes = [("A", [0, .10, .22, .30, .38, .45], S2, .55, False),
             ("B", [0, .18, .40, .58, .80, 1.0], S1, None, False),
             ("C", [0, .14, .30, .52, .66, .78], S3, .22, True)]
    for ax, (name, pts, color, delta, target) in zip(axes, lanes):
        ax.plot([0, 5.35], [1, 1], color="#9aa1ab", ls=(0, (3, 3)), lw=1)
        ax.plot(range(6), pts, color=color, lw=2.6, solid_joinstyle="round")
        ax.set_ylim(-.05, 1.25); ax.set_yticks([]); ax.grid(False)
        ax.set_ylabel(name, rotation=0, fontsize=13, fontweight="bold", color=INK, labelpad=14, va="center")
        ax.spines["left"].set_visible(False)
        if delta is None:
            ax.annotate("fires → inhibits A and C", (5, 1), xytext=(10, 4), textcoords="offset points", color=INK, fontweight="bold")
        else:
            ax.plot(5, pts[-1], "o", ms=7, mfc="white", mec=color, mew=2)
            ax.annotate("", xy=(5.2, 1), xytext=(5.2, pts[-1]), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
            label = f"cancelled · Δ = {delta:.2f} kept"
            if target:
                label += "\nteacher later names C → promote C"
            ax.annotate(label, (5.3, min((1 + pts[-1]) / 2, pts[-1] - .05)), xytext=(6, 0), textcoords="offset points",
                        va="top" if target else "center", color=INK2, fontsize=9)
    axes[-1].set_xticks([]); axes[-1].set_xlim(-.2, 8)
    axes[-1].set_xlabel("input spikes arrive →  (dashed line: threshold θ; the race ends at B's fire event)", color=MUTED, fontsize=9)
    axes[0].set_title("One race: B fires, A and C are cancelled but keep their distance to threshold")
    fig.savefig(os.path.join(FIG, "race.png")); plt.close(fig)


def e4(D):
    t = D["e4"]["table"]; ks = D["e4"]["ks"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    items = []
    for rule, c, ls, lw in RULES:
        rows = [next(r for r in t if r["rule"] == rule and r["k"] == k) for k in ks]
        y = [r["acc"] for r in rows]; e = [r["ci"] for r in rows]
        ax.errorbar(ks, y, yerr=e, color=c, ls=ls, lw=lw, marker="o", ms=5 if rule == "cf_margin" else 4, capsize=0,
                    mec="white", mew=1.2)
        items.append((ks[-1], y[-1], rule + (" (reference)" if rule == "softmax" else ""), c, rule == "cf_margin"))
    end_labels(ax, items, .045)
    ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks); ax.set_xlim(2.5, 400); ax.set_ylim(-.02, 1.02)
    ax.set_xlabel("number of classes K (log scale)"); ax.set_ylabel("test accuracy")
    ax.set_title("E4 · Accuracy as the number of classes grows (10 seeds, 95% CI)")
    fig.savefig(os.path.join(FIG, "e4_accuracy.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for rule, c, _, _ in RULES:
        r = next(r for r in t if r["rule"] == rule and r["k"] == 128)
        ax.scatter(r["plasticity"], r["acc"], s=70 if rule == "cf_margin" else 50, color=c, edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(rule, (r["plasticity"], r["acc"]), xytext=(8, -3 if rule != "cf_winner" else 5), textcoords="offset points",
                    fontsize=9, color=INK if rule == "cf_margin" else INK2, fontweight="bold" if rule == "cf_margin" else "normal")
    ax.set_xscale("log"); ax.set_ylim(-.03, 1.02); ax.set_xlim(8e4, 2e8)
    ax.set_xlabel("synaptic weight updates during training (log)"); ax.set_ylabel("test accuracy")
    ax.set_title("E4 · Accuracy vs plasticity work at K = 128")
    fig.savefig(os.path.join(FIG, "e4_work.png")); plt.close(fig)

    d = D["e4"]["delay"]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for gap, name, c in [(5.0, "quiet until taught", S1), (0.2, "next race intervenes", S2)]:
        rows = [r for r in d if r["gap"] == gap]
        ax.errorbar([r["delay"] for r in rows], [r["acc"] for r in rows], yerr=[r["ci"] for r in rows], color=c, marker="o",
                    lw=2.4 if gap == 5 else 2, ms=5, mec="white", label=name)
    ax.axvline(1, color=MUTED, lw=1, ls=":"); ax.annotate("τ", (1, .02), xytext=(4, 0), textcoords="offset points", color=MUTED)
    ax.set_xscale("log"); ax.set_ylim(-.02, 1.02); ax.legend(loc="lower left")
    ax.set_xlabel("teaching delay D (log scale)"); ax.set_ylabel("test accuracy")
    ax.set_title("E4 · Credit under a delayed teacher (cf_margin, K = 32)")
    fig.savefig(os.path.join(FIG, "e4_delay.png")); plt.close(fig)


def e2(D):
    E = D["e2"]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for key, name, c, ls in [("race", "race", S1, "-"), ("fixed_time", "fixed-time", S2, "-"), ("msprt", "MSPRT (reference)", REF, "--")]:
        rows = sorted(E[key], key=lambda r: r["time"])
        ax.plot([r["time"] for r in rows], [r["acc"] for r in rows], color=c, ls=ls, marker="o", ms=4, lw=2.4 if key == "race" else 2,
                mec="white", label=name)
    ax.set_xscale("log"); ax.legend(loc="upper left")
    ax.set_xlabel("mean decision time, s (log)"); ax.set_ylabel("accuracy")
    ax.set_title("E2 · Speed–accuracy frontier")
    fig.savefig(os.path.join(FIG, "e2_frontier.png")); plt.close(fig)

    race = next(r for r in E["race"] if r["param"] == 15); fixed = next(r for r in E["fixed_time"] if r["param"] == 1)
    cs = sorted(float(c) for c in race["by_coherence"])
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(cs, [race["by_coherence"][str(c)]["time"] for c in cs], color=S1, marker="o", lw=2.4, mec="white", label="race θ = 15 (acc 0.79)")
    ax.plot(cs, [1.0] * len(cs), color=S2, marker="o", lw=2, mec="white", label="fixed-time T = 1 (acc 0.74)")
    for c in cs:
        ax.annotate(f"{race['by_coherence'][str(c)]['acc']:.2f}", (c, race["by_coherence"][str(c)]["time"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xscale("log"); ax.set_xticks(cs); ax.set_xticklabels(cs); ax.set_ylim(0, 2); ax.legend(loc="upper right")
    ax.set_xlabel("coherence (evidence strength, log)"); ax.set_ylabel("mean decision time, s")
    ax.set_title("E2 · Decision time follows difficulty (labels: race accuracy)")
    fig.savefig(os.path.join(FIG, "e2_coherence.png")); plt.close(fig)


NAMES = {"mlp": "dense MLP 1000, backprop", "single_layer": "single racing layer", "frozen_hidden": "frozen random hidden",
         "crl_fired_only": "CRL · fired-only hidden credit", "crl_fa": "CRL · feedback alignment", "crl_sym": "CRL · symmetric feedback"}
COLORS = {"mlp": REF, "single_layer": S2, "frozen_hidden": S5, "crl_fired_only": S4, "crl_fa": S3, "crl_sym": S1}


def e6(D):
    runs = [r for r in D["e6"]["runs"] if r["kind"] == "final" and r["variant"] in NAMES]
    if not runs:
        return
    runs.sort(key=lambda r: r["acc"])
    fig, ax = plt.subplots(figsize=(7.6, .45 * len(runs) + 1.2))
    labels = [NAMES[r["variant"]] + (f" · {r['winners']} winners" if r["winners"] > 1 else "") + f"  ({len(r['curve'])} ep)" for r in runs]
    ax.barh(labels, [r["acc"] for r in runs], color=[COLORS[r["variant"]] for r in runs], height=.6)
    for i, r in enumerate(runs):
        ax.annotate(f"{r['acc']:.3f}", (r["acc"], i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 1.08); ax.grid(axis="y", visible=False); ax.set_xlabel("MNIST test accuracy")
    ax.set_title("E6 · Latency-coded MNIST, 60k training images")
    fig.savefig(os.path.join(FIG, "e6_mnist.png")); plt.close(fig)


def energy(D):
    E = D["energy"]["e4"]
    cf = next(r for r in E if r["rule"] == "cf_margin" and r["k"] == 128)
    fr = next(r for r in E if r["rule"] == "fired_reward" and r["k"] == 128)
    sm = next(r for r in E if r["rule"] == "softmax" and r["k"] == 128)
    rows = [("cf_margin · ideal event", cf["training_J_per_episode"]["event_ideal"], S1),
            ("cf_margin · event, shared SRAM", cf["training_J_per_episode"]["event_shared_sram"], S1),
            ("cf_margin · Loihi (measured ops)", cf["training_J_per_episode"]["loihi"], S1),
            ("fired_reward · ideal event", fr["training_J_per_episode"]["event_ideal"], S4),
            ("softmax · dense fp16", sm["training_J_per_episode"]["dense_fp16_train"], REF),
            ("softmax · dense fp16, batch 256", sm["training_J_per_episode"]["dense_fp16_train_batched"], REF)]
    rows.reverse()
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.barh([r[0] for r in rows], [r[1] * 1e9 for r in rows], color=[r[2] for r in rows], height=.6)
    for i, r in enumerate(rows):
        ax.annotate(f"{r[1] * 1e9:.1f} nJ", (r[1] * 1e9, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_xscale("log"); ax.set_xlim(.5, 1000); ax.grid(axis="y", visible=False)
    ax.set_xlabel("training energy per teaching event, nJ (log)")
    ax.set_title(f"Energy · E4 at K = 128 (accuracy: cf_margin {cf['acc']:.2f}, softmax {sm['acc']:.2f})")
    fig.savefig(os.path.join(FIG, "energy_e4.png")); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    D = json.load(open(os.path.join(ROOT, "report", "data.json")))
    race_diagram(); e4(D); e2(D); e6(D); energy(D)
    print("figures:", sorted(os.listdir(FIG)))
