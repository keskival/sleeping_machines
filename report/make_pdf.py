#!/usr/bin/env python3
"""Build the status report PDF (report/sleeping_machines_status.pdf) from result files.

    python report/make_pdf.py

Charts are drawn with matplotlib from experiments/results and report/data.json;
the document is assembled with reportlab. Rerun after new results arrive.
"""
import glob
import io
import json
import os
from datetime import date

import matplotlib
import matplotlib.patches  # noqa: F401
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(ROOT, "report", "sleeping_machines_status.pdf")

# Validated categorical palette (dataviz reference instance, light mode); gray for references.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, MUTED, GRID = "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#a9c8ef", "#5f9be3", "#2a78d6", "#174a8c"]            # one hue, light -> dark (rounds)

FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False, "savefig.dpi": 200,
})


def load(path):
    with open(path) as f:
        return json.load(f)


def fig_image(fig, width_mm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_mm * mm, height=width_mm * mm * h / w)


def label_end(ax, x, y, text, color=INK, dx=4, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", fontsize=8,
                color=color)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_e2(d):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    series = [("race", "race (additions + threshold)", BLUE, "-"), ("fixed_time", "fixed-time decoder", ORANGE, "-"),
              ("msprt", "MSPRT (optimal reference)", GRAY, "--")]
    for key, name, c, ls in series:
        pts = sorted((p["time"], p["acc"]) for p in d["e2"][key])
        x, y = zip(*pts)
        ax.plot(x, y, ls, color=c, marker="o", markersize=3.5, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("mean decision time (s, log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title("E2 · speed–accuracy frontier (4 classes, Poisson evidence)")
    ax.legend(loc="lower right")
    return fig


def fig_e4(d):
    rules = [("cf_margin", "counterfactual, near misses", BLUE), ("cf_winner", "counterfactual, winner only", AQUA),
             ("fired_reward", "reward-modulated (fired only)", ORANGE), ("fired_only", "fired only", YELLOW)]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    tab = d["e4"]["table"]
    for rule, name, c in rules + [("softmax", "softmax SGD (non-local reference)", GRAY)]:
        rows = sorted((r["k"], r["acc"], r["ci"]) for r in tab if r["rule"] == rule)
        k, a, ci = map(np.array, zip(*rows))
        ls = "--" if rule == "softmax" else "-"
        ax.plot(k, a, ls, color=c, marker="o", markersize=4, label=name)
        ax.fill_between(k, a - ci, a + ci, color=c, alpha=0.15, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(d["e4"]["ks"], [str(k) for k in d["e4"]["ks"]])
    ax.set_xlabel("number of classes K (log scale)")
    ax.set_ylabel("test accuracy (10 seeds, 95% CI)")
    ax.set_title("E4 · a cancelled node can be taught")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
    return fig


def fig_e5():
    r = load(os.path.join(RES, "e5", "rows_r2.json"))["rows"]
    ks = sorted({row["k"] for row in r})

    def mean(model, part, key):
        return [np.mean([row[model][part][key] for row in r if row["k"] == k]) for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    panels = [("inference_per_sample", "inference work per input", ("synops", "synops", "macs")),
              ("train_per_episode", "learning work per teaching event", ("plasticity", "plasticity", "macs"))]
    for ax, (part, title, keys) in zip(axes, panels):
        for (model, name, c, ls), key in zip((("race", "race", BLUE, "-"), ("sparse", "sparse softmax", ORANGE, "-"),
                                              ("dense", "dense", GRAY, "--")), keys):
            y = mean(model, part, key)
            ax.plot(ks, y, ls, color=c, marker="o", markersize=4)
            nudge = {"race": -5, "sparse": 5}.get(model, 0) if part == "inference_per_sample" else 0
            label_end(ax, ks[-1], y[-1], name, MUTED, dy=nudge)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ks, [f"{k // 1024}k" if k >= 1024 else str(k) for k in ks])
        ax.set_xlim(ks[0] / 1.5, ks[-1] * 6)
        ax.set_xlabel("classes K")
        ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel("operations (log)")
    fig.suptitle("E5 round 2 · work as capacity grows 64× (3 seeds)", x=0.02, ha="left", fontsize=9.5,
                 fontweight="bold")
    fig.tight_layout()
    return fig


def e6_runs():
    runs = {}
    for p in glob.glob(os.path.join(RES, "e6", "mnist_*_s0.json")):
        name = os.path.basename(p)[6:-8]
        r = load(p)
        if r.get("val") or any(t in name for t in ("pilot", "check", "timing", "s0_")):
            continue
        variant = r["config"]["variant"]
        rnd = 3 if "_r3" in name else 2 if name.endswith("_v2") else 1 if name == variant else None
        if rnd is None:
            continue
        runs[(variant, rnd, r["config"]["hidden"])] = r["test_acc"]
    return runs


def fig_e6(runs):
    variants = [("single_layer", "single racing layer"), ("frozen_hidden", "frozen random hidden layer"),
                ("crl_fired_only", "CRL, fired-only credit"), ("crl_fa", "CRL, counterfactual credit"),
                ("crl_sym", "CRL, symmetric feedback")]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    h = 0.26
    for i, (v, name) in enumerate(variants):
        for j, rnd in enumerate((1, 2, 3)):
            acc = runs.get((v, rnd, 1000 if v != "single_layer" else runs and 1000))
            acc = acc if acc is not None else next((a for (vv, rr, _), a in runs.items() if vv == v and rr == rnd), None)
            y = i + (j - 1) * h
            if acc is None:
                ax.text(0.705, y, "queued" if (v == "single_layer" and rnd == 3) else "—", va="center", fontsize=7,
                        color=MUTED)
                continue
            ax.barh(y, acc - 0.70, left=0.70, height=h * 0.85, color=SEQ[j + 1 if j else 0], edgecolor="white",
                    linewidth=1)
            ax.text(acc + 0.003, y, f"{acc:.3f}", va="center", fontsize=7, color=INK)
    ax.axvline(0.9783, color=GRAY, ls="--", lw=1.2)
    ax.text(0.9803, -0.75, "dense MLP\n0.978", ha="left", fontsize=6.5, color=MUTED, va="center")
    ax.axvline(0.95, color=GRAY, ls=":", lw=1.2)
    ax.text(0.948, -0.75, "STDP-WTA\n≈0.95", ha="right", fontsize=6.5, color=MUTED, va="center")
    ax.set_yticks(range(len(variants)), [n for _, n in variants])
    ax.set_xlim(0.70, 1.0)
    ax.set_ylim(-0.95, len(variants) - 0.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MNIST test accuracy (seed 0; hidden = 1000)")
    ax.set_title("E6 · three rounds of the hidden-layer network")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEQ[k]) for k in (0, 2, 3)]
    ax.legend(handles, ["round 1", "round 2", "round 3 (ramp synapses, collapsing bound)"], loc="upper center",
              bbox_to_anchor=(0.4, -0.16), ncol=3, fontsize=7)
    return fig


def pretty(name):
    rnd = "round 3" if name.endswith("_r3") else "round 2" if name.endswith("_v2") else "round 1"
    base = name.removesuffix("_r3").removesuffix("_v2")
    return {"single_layer": "single layer", "frozen_hidden": "frozen hidden", "crl_fa": "CRL counterfactual",
            "crl_fired_only": "CRL fired-only", "crl_sym": "CRL symmetric"}.get(base, base) + " · " + rnd


def fig_energy():
    e = load(os.path.join(RES, "energy.json"))["e6"]["race"]
    rows = [(r["name"], r["acc"], r["matched"]) for r in e if r.get("matched")]
    rows.sort(key=lambda t: t[2]["inference_ratio_vs_int8_batch1"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (name, acc, m) in enumerate(rows):
        x1, x2 = m["inference_ratio_vs_int8_batch1"], m["training_ratio_vs_fp16"]
        ax.plot([x1], [i], "o", color=BLUE, markersize=6)
        ax.plot([x2], [i], "D", color=ORANGE, markersize=5)
    ax.axvline(1, color=INK, lw=1)
    ax.text(1.08, len(rows) - 0.4, "event cheaper →", fontsize=7.5, color=MUTED)
    ax.text(0.92, len(rows) - 0.4, "← dense cheaper", fontsize=7.5, color=MUTED, ha="right")
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)), [f"{pretty(n)}  ({a:.3f})" for n, a, _ in rows], fontsize=7.5)
    ax.set_ylim(-0.7, len(rows) + 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("dense energy ÷ event energy at matched accuracy (log; ideal event hardware)")
    ax.set_title("Energy · E6 networks vs the cheapest dense model at least as accurate")
    ax.legend([plt.Line2D([], [], marker="o", color=BLUE, ls=""), plt.Line2D([], [], marker="D", color=ORANGE, ls="")],
              ["inference vs int8, batch 1", "training vs fp16, unbatched"], loc="lower right", fontsize=7)
    return fig


def fig_e7():
    """Schematic: a stream of frames with episode resets; fast state per frame, slow state across."""
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    ax.set_axis_off()
    ax.set_xlim(0, 16.5)
    ax.set_ylim(-1.2, 3.2)
    x = 0.3
    for ep, n in enumerate((5, 5, 3)):
        for v in range(n):
            labelled = (ep, v) in ((0, 1), (1, 3))
            ax.add_patch(plt.Rectangle((x, 1.2), 0.8, 0.8, facecolor=SEQ[0] if not labelled else BLUE,
                                       edgecolor="white", lw=1.5))
            if v == 0:
                ax.plot([x - 0.08, x - 0.08], [0.9, 2.3], color=ORANGE, lw=2)
            x += 0.95
        x += 0.35
    ax.text(0.3, 2.55, "frames (views of one image) →   dark = label arrives   orange bar = episode reset",
            fontsize=7.5, color=MUTED)
    ax.annotate("", xy=(15.8, 0.55), xytext=(0.3, 0.55), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.text(0.3, 0.1, "slow state carried across frames: prior · tags · carried label · continuity · "
            "homeostasis · consolidation", fontsize=7.5, color=INK)
    ax.text(0.3, -0.45, "fast state (one race per frame) restarts at every frame, so the exact solver still applies",
            fontsize=7.5, color=MUTED)
    ax.text(0.3, -0.95, "each frame is predicted before it may be learned from (prequential); no epochs, no batches",
            fontsize=7.5, color=MUTED)
    return fig


# ── Document ──────────────────────────────────────────────────────────────────

def fig_e26():
    """E26: test accuracy vs training fraction, p = 31, by learning rule (3 seeds each; line = mean, dots = seeds)."""
    arms = [("p31_s0.0_a1_r0_push1.json", "push on the wrong winner", GRAY),
            ("p31_s0.0_a1_r0_push0.json", "pull only", BLUE),
            ("p31_s1.0_a1_r0_push0_noise.json", "pull only + noise (annealed by error)", YELLOW),
            ("p31_s1.0_a2_r0_cool.json", "pull only + noise cooled to zero", ORANGE)]
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for fn, name, col in arms:
        path = os.path.join(RES, "e26", fn)
        if not os.path.exists(path):
            continue
        rows = load(path)["rows"]
        fr = sorted({r["frac"] for r in rows})
        vals = [[r["final"]["test"] for r in rows if r["frac"] == f] for f in fr]
        for f, v in zip(fr, vals):
            ax.scatter([f] * len(v), v, color=col, s=10, alpha=0.5, lw=0)
        ax.plot(fr, [np.mean(v) for v in vals], color=col, marker="o", ms=4, label=name)
    ax.axhline(1 / 31, color=GRAY, lw=0.8, ls=":")
    ax.set_xlabel("fraction of the p² pairs used for training (p = 31)")
    ax.set_ylabel("test accuracy, unseen pairs")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7, loc="center right")
    ax.set_title("E26: sparse, error-driven delay learning with a race readout")
    return fig


def fig_e30():
    """E30: two-counter machine success vs timing precision q/sigma, with and without restoration."""
    path = os.path.join(RES, "e30", "minsky.json")
    if not os.path.exists(path):
        return None
    rows = [r for r in load(path)["rows"] if "success" in r and r.get("q_over_sigma")]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for restore, steps, col, ls in ((False, 25, GRAY, "-"), (False, 81, GRAY, "--"),
                                    (True, 25, BLUE, "-"), (True, 81, BLUE, "--")):
        pts = sorted((r["q_over_sigma"], r["success"]) for r in rows
                     if r.get("restore") == restore and r["steps"] == steps)
        if pts:
            ax.plot(*zip(*pts), color=col, ls=ls, marker="o", ms=4,
                    label=f"{'restored' if restore else 'plain'}, {steps} steps")
    ax.set_xlabel("timing precision q / σ (quantum over jitter per hop)")
    ax.set_ylabel("programs computed exactly")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("E30: a Minsky machine wired from delay, or, and, veto + one oscillator")
    return fig


def fig_e32():
    """accuracy vs operations per episode on E27's task: clocked conv nets, event-token Transformers, event network."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    pts = {}
    for path in glob.glob(os.path.join(RES, "e32", "dense_F*.json")):
        for r in load(path)["rows"]:
            if r["pad"] == 0.0:
                pts.setdefault((r.get("filters", 16), r["dt"]), []).append((r["acc"], r["macs_per_episode"]))
    if pts:
        xs = [np.mean([m for _, m in v]) for v in pts.values()]; ys = [np.mean([a for a, _ in v]) for v in pts.values()]
        ax.scatter(xs, ys, color=GRAY, s=18, label="clocked conv net (backprop, 200k episodes)", zorder=3)
    tf = {}
    for path in glob.glob(os.path.join(RES, "e36", "transformer_e27*.json")):
        long = "long" in path or "rel" in path
        for r in load(path)["rows"]:
            tf.setdefault((long, r["d"], r["layers"], r.get("reltime", 0)), []).append((r["acc"], r["macs_per_episode"]))
    for long, col, name in ((False, YELLOW, "event-token Transformer, 200k episodes"),
                            (True, BLUE, "event-token Transformer, 2M episodes")):
        v = [(np.mean([m for _, m in vv]), np.mean([a for a, _ in vv])) for k, vv in tf.items() if k[0] == long]
        if v:
            ax.scatter(*zip(*v), color=col, s=18, marker="s", label=name, zorder=3)
    ax.scatter([7.5], [1.0], color=ORANGE, s=70, marker="D", label="event network (E35), nothing given", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("operations per episode (synaptic events or multiply-adds), log scale")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Timing task: equal or better accuracy at 10⁴–10⁵× fewer operations")
    return fig


def fig_depth_theorem():
    """§71: a node's accept set in lag coordinates is a product set; the order a < b is not."""
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for ax in axs:
        ax.set_xlim(-4, 0.3); ax.set_ylim(-4, 0.3); ax.set_aspect("equal")
        ax.set_xlabel("lag of input a to the trigger"); ax.set_ylabel("lag of input b")
    axs[0].add_patch(matplotlib.patches.Rectangle((-3, -2.5), 2.2, 1.8, color=BLUE, alpha=0.35, lw=0))
    axs[0].set_title("one node: a product set")
    axs[1].fill_between([-4, 0], [-4, 0], [0, 0], color=ORANGE, alpha=0.35, lw=0)
    axs[1].plot([-2, -1], [-1.5, -0.5], "o", color=INK, ms=4)
    axs[1].plot([-1], [-1.5], "x", color="#c0392b", ms=7, mew=2)
    axs[1].annotate("in any product set\ncontaining both dots,\nbut violates a < b", (-1, -1.5), (-3.9, -3.6), fontsize=7,
                    arrowprops=dict(arrowstyle="->", lw=0.7))
    axs[1].set_title("a before b: not a product set")
    fig.tight_layout()
    return fig


def fig_e34():
    """depth and composition on hierarchical motifs (15 classes from 6 motifs)."""
    rows = [("depth 1, hold/trigger on channels", 0.39, GRAY), ("E28 accumulating readout, depth 2", 0.44, GRAY),
            ("depth 2, single wide window", 0.79, BLUE), ("depth 2, window bank {1, 2, 4}", 0.86, BLUE),
            ("depth 2, tuned part window", 0.97, BLUE), ("Transformer, 2M episodes, ~175k MACs", 0.998, YELLOW)]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    for yi, (_, v, _) in zip(y, rows):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7, color=INK)
    ax.axvline(1 / 16, color=GRAY, lw=0.8, ls=":")
    ax.set_xlim(0, 1.1); ax.set_xlabel("test accuracy (5 seeds; chance 0.06)")
    ax.set_title("Depth by composition: hold/trigger chains (≈ 14 events per episode)")
    ax.grid(axis="y", visible=False)
    return fig

def fig_e37():
    """E37: train and test accuracy over epochs, with and without sleep (p = 31, half the pairs, seed 0)."""
    runs = [("p31_add_f0.5_lam0.0_r1.json", "no sleep", GRAY), ("p31_add_f0.5_lam0.02_r1.json", "sleep λ = 0.02", BLUE),
            ("p31_add_f0.5_lam0.2_r1.json", "sleep λ = 0.2", ORANGE)]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for fn, name, col in runs:
        path = os.path.join(RES, "e37", fn)
        if not os.path.exists(path):
            continue
        c = load(path)["rows"][0]["curve"]
        ep = [x["epoch"] for x in c]
        ax.plot(ep, [x["train"] for x in c], color=col, ls="--", lw=1.4)
        ax.plot(ep, [x["test"] for x in c], color=col, label=name)
    ax.axhline(1 / 31, color=GRAY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("epoch (log scale)")
    ax.set_ylabel("accuracy (dashed: train, solid: test)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7, loc="center right")
    ax.set_title("E37: memorized by epoch 5; with sleep, unseen pairs follow after a delay")
    return fig


def time_pages(st, W):
    """E24–E30 and THEORY §53–§62: computing with time (26–27 September)."""
    s = [Paragraph("Computing with time (E24–E30, 26–27 September)", st["h1"]),
         Paragraph("Until E24 the learned part of every race network was synaptic weights; time was a code, never a "
                   "resource. These experiments ask what time itself computes and how a system can learn it "
                   "natively: every mechanism an event handler with local state, cost proportional to events, no "
                   "batches, epochs, replay or global sums. Dense rescues are kept only as diagnostics.",
                   st["body"])]
    s += [Paragraph("E24: grokking (complete)", st["h2"]),
          Paragraph("(a + b) mod 31, half the pairs. Every local race variant, with or without sleep downscaling, "
                    "memorizes: train ≈ 1.0, test ≤ 0.008 (chance 0.032, 17 runs). Dense MLPs: backprop 0.87, "
                    "Kolen–Pollack 0.956, feedback alignment 0.000. At p = 97 backprop fails at 10% and 20% of pairs "
                    "and groks at 30% (0.937).", st["body"])]
    s += [Paragraph("E25: delays instead of lookup", st["h2"]),
          Paragraph("A ring of relays computes (a + b) mod p for all pairs with no learning: 4p synapses, about 2p "
                    "events per query. Delays learned as phases generalize to 1.000 on unseen pairs from 15–20% of "
                    "pairs (5 seeds, p = 31–97), including a·b mod p (the delays find the discrete logarithm); "
                    "a² + ab + b² and random tables are out of reach. <b>This is restriction, not grokking</b> "
                    "(§58): the single-phase readout can only express one-character relations, and the learner "
                    "(replay) is dense.", st["body"])]
    im = png("e25_generalization", W * 0.55)
    if im:
        s.append(im)
    s += [Paragraph("E26: in a race, winning is positional", st["h2"]),
          Paragraph("The same ring, learned natively: updates only on errors, only on the delays involved, a race "
                    "readout. Pushing the wrong winner later collapses every detector onto one phase (chance in all "
                    "15 runs, ~386k updates each). Pulling only the teacher learns the relation (p = 31: 0.97–0.98 at "
                    "50% of pairs; p = 59: 0.95) with ~40k updates. Timing noise makes generalization at 20% reliable; "
                    "cooled to zero it also lifts the accuracy cap.", st["body"])]
    s.append(fig_image(fig_e26(), W))
    s.append(PageBreak())
    s += [Paragraph("E27: false positives are specialized, never displaced", st["h2"]),
          table([["false positives handled by", "test (5 seeds)", "updates"],
                 ["veto (inhibition)", "0.914 (0.887–0.934)", "15–20k"],
                 ["none (no veto synapses)", "0.896 (0.881–0.918)", "18–23k"],
                 ["push (delays lengthened)", "0.195 (0.18–0.20)", "~155k"]], [70, 55, 30], st),
          Paragraph("Patterns “B within Δ after A unless C”. Displacement is destructive; veto helps only a little, "
                    "because aligning by delay destroys the interval the veto must see (§61: tolerance by holding a "
                    "long PSP keeps it).", st["small"])]
    s += [Paragraph("E28: depth needs routing credit, and does not pay yet", st["h2"]),
          table([["arm (hierarchical motifs)", "6 classes, 3 seeds", "15 classes from 6 motifs, 5 seeds, 60k"],
                 ["depth 1", "0.68", "0.51"],
                 ["depth 2, counterfactual routing credit", "0.72–0.74", "0.44"],
                 ["depth 2, critical-path credit only", "0.17 (chance 0.14)", "–"]], [70, 40, 64], st),
          Paragraph("A race is a routing network: without counterfactual credit (runners-up or cancelled near-misses) "
                    "depth stays at chance. With it, depth is learnable but not better. Receptive fields show the hidden "
                    "nodes do learn parts (86 of 124 draw both strongest inputs from one motif), but the readout fails "
                    "at conjunction: 56% of depth-2 answers are a class sharing one motif with the true one, and none "
                    "are order errors; synapse caps, windowed readouts and global competition did not fix it (§62). Getting here required §60: pull-only on weights needs a "
                    "conserved per-node budget in fractional steps, conserving weakening, per-node prices, and enough "
                    "hidden nodes for Cover's capacity bound (depth 1 rose from 0.29 to 0.68).", st["small"])]
    s += [Paragraph("E34: depth by composition in time", st["h2"]),
          table([["hierarchical motifs, 15 classes from 6 motifs, 5 seeds", "test"],
                 ["depth 2, hold/trigger chains, part window [0, 1.5]", "0.97 (0.92–0.99)"],
                 ["depth 2, generic window bank {1, 2, 4}", "0.86 (0.79–0.96)"],
                 ["depth 2, single wide window 4", "0.79 (0.68–0.93)"],
                 ["depth 1, same nodes on channels", "0.39 (0.32–0.48)"],
                 ["E28 (accumulating readout), depth 1 / 2", "0.51 / 0.44"]], [110, 50], st),
          Paragraph("Parts are directional hold nodes on channel pairs; a class node holds one part's spike and is "
                    "triggered by another's, so a single part cannot satisfy it: conjunction and order by construction "
                    "(§64). Routing is learned by counterfactual pulls under conserved budgets. Learning the part windows "
                    "from class routes failed (0.08–0.13): a shared part must not be shaped by one class.", st["small"])]
    s += [Paragraph("E35: nothing given", st["h2"]),
          Paragraph("E27's detectors were told their two channels. With learned hold, trigger and veto weights over all "
                    "channels and learned hold durations: 1.000 on four seeds and 0.9995 on the fifth, 7.5 synaptic events "
                    "per episode, 443–1,530 updates in 200k episodes. This is the headline result.", st["body"])]
    s += [Paragraph("E37: grokking as a route change", st["h2"]),
          Paragraph("A network that can memorize (one pair node per operand pair routed to classes; ρ = n/params ≈ "
                    "0.016) and also has a generic rhythm resource (E26's ring with learned delays). Errors-only learning "
                    "makes memorization absorbing: without sleep, train 1.0 and test at chance in all seeds. Sleep "
                    "(decay of the per-pair weights) prunes parameters that are not reused (§72); the relation becomes "
                    "the absorbing state and test reaches 0.93–0.97 in 2 of 3 seeds at every λ > 0, after a delay at "
                    "small λ: at λ = 0.02 it memorizes by epoch 5, partly forgets as sleep erodes the lookup (train dips near epoch 10), then recovers through the relation, with unseen pairs following from epoch ≈ 50. The third seed collapses (its shared route never becomes correct), as does the network with "
                    "sleep but no rhythm (train 0.45, test 0).", st["body"])]
    f37 = fig_e37()
    if f37 is not None:
        s.append(fig_image(f37, W))
    s += [Paragraph("E29: true grokking test, not passed", st["h2"]),
          Paragraph("A general race network that can memorize (ρ = n/params ≈ 0.006), given recurrent delay loops, on "
                    "E24's encoding. Frozen random loops only memorize (test 0.015); with hidden learning, training "
                    "collapses whether loop periods are learned or fixed. No grokking yet.", st["body"])]
    s += [Paragraph("E30: the operator basis is Turing-complete, and restoration makes it scale", st["h2"]),
          Paragraph("A two-counter Minsky machine as a netlist of Delay, Or, And (a PSP window per input) and Veto "
                    "nodes plus one reference oscillator; counters are phases of spikes in hold loops. Exact on every "
                    "test program. Under timing jitter, one comb coincidence per counter per cycle makes reliability "
                    "independent of program length: precision is the tape, and restoration is its price.",
                    st["body"])]
    f30 = fig_e30()
    if f30 is not None:
        s.append(fig_image(f30, W))
    s.append(PageBreak())
    s += [Paragraph("E32: a measured frontier against a clocked dense model", st["h2"]),
          Paragraph("E27's task, against a 1-D temporal conv net on binned spikes trained by backprop on the same 200k "
                    "episodes. The cheapest dense model at the event learner's accuracy (≈ 0.91) needs 1.9k "
                    "multiply-adds per episode; the event learner 10.2 synaptic events (≈ 190× fewer operations, ≈ 80× "
                    "less energy at published per-operation costs). In 99% silence the clocked cost grows 100× and the "
                    "event cost does not (≈ 19,000×). Training: ≈ 6,000× fewer operations. <b>Caveats:</b> the dense "
                    "model reaches 0.995 at 3M multiply-adds, which the event learner cannot yet buy; and most of the gap "
                    "is the clock: a sparse, event-driven implementation of the same conv would cost ≈ 106 "
                    "multiply-adds, only ≈ 10× more (§63).", st["body"])]
    f32 = fig_e32()
    if f32 is not None:
        s.append(fig_image(f32, W))
    s += [Paragraph("Theory added (THEORY §53–§63)", st["h1"])]
    s += bullets([
        "<b>Clockless means shift-equivariant (§56).</b> A network of delays, first-ofs, coincidences and vetoes "
        "commutes with time shifts, so it cannot add two times, only compare them. One oscillator reference breaks the "
        "symmetry to shifts by a period and makes time a cyclic group: exactly one character, which is E25's "
        "learnable class. The space-time algebra itself is prior art (Smith 2018; race logic).",
        "<b>Credit follows one causal chain (§56.4).</b> A spike's time depends on one critical path, with derivative "
        "1 along it: exact credit is sparse by construction and does not contract with depth. Routing choices need "
        "counterfactuals on top (§57); cancelled near-misses supply them without extra events.",
        "<b>Teach the event that should have won (§54, §56.5, §60).</b> Never displace a loser in time; remove false "
        "positives by veto where codes are specific. For timing parameters pull-only suffices; for weights, pull "
        "under a conserved budget with prices.",
        "<b>Aligning destroys the interval, holding keeps it (§61).</b> Delay says where in time an event acts, PSP "
        "duration how long a node remembers it; veto and ordering need duration.",
        "<b>What counts as generalization (§58).</b> Restriction by the model class, forced generalization above "
        "capacity, and grokking (the relation reached while memorizers exist) are different claims; report ρ = n / "
        "params with every result.",
        "<b>Where supremacy can live (§55).</b> Not in operation counts for static functions (an encoding effect). On "
        "the input side the factor is 1 / (spikes per channel per precision bin): SHD gives only ~6× at 10 ms bins, "
        "so the benchmark must be far sparser in time.",
        "<b>Completeness and restoration (§59).</b> {delay, first-of, coincidence, veto, hold} + one reference is "
        "Turing-complete (Minsky; prior art for spiking nets: Maass 1996); a random-walk model with ~6 jittered hops "
        "per cycle fits the unrestored failures.",
    ], st)
    return s


def styles():
    base = ParagraphStyle("b", fontName="DV", fontSize=9.2, leading=13.2, textColor=colors.HexColor(INK),
                          spaceAfter=5)
    return {
        "title": ParagraphStyle("t", parent=base, fontName="DVB", fontSize=20, leading=25, spaceAfter=4),
        "sub": ParagraphStyle("s", parent=base, fontSize=10, textColor=colors.HexColor(MUTED), spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=base, fontName="DVB", fontSize=13.5, leading=18, spaceBefore=8,
                             spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base, fontName="DVB", fontSize=10.5, leading=14, spaceBefore=6,
                             spaceAfter=3),
        "body": base,
        "bullet": ParagraphStyle("bu", parent=base, leftIndent=12, bulletIndent=2, spaceAfter=3),
        "small": ParagraphStyle("sm", parent=base, fontSize=7.8, leading=10.5, textColor=colors.HexColor(MUTED)),
        "cell": ParagraphStyle("c", parent=base, fontSize=7.8, leading=10, spaceAfter=0),
    }


def bullets(items, st):
    return [Paragraph(t, st["bullet"], bulletText="•") for t in items]


def table(rows, widths, st, head=True):
    data = [[Paragraph(str(c), st["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f0ec")),
                  ("FONTNAME", (0, 0), (-1, 0), "DVB")]
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(18 * mm, 10 * mm, "Sleeping Machines · status report")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def png(name, width_mm):
    from reportlab.lib.utils import ImageReader
    path = os.path.join(ROOT, "report", "figures", name + ".png")
    if not os.path.exists(path):
        return None
    w, h = ImageReader(path).getSize()
    return Image(path, width=width_mm * mm, height=width_mm * mm * h / w)


def e13_summary():
    rows = {}
    for p in glob.glob(os.path.join(RES, "e13", "bandit_*_main_s*.json")):
        r = load(p)
        rows.setdefault(r["config"]["rule"], []).append(r["acc"])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in rows.items()}


def promising_page(st, W):
    s = [Paragraph("Most promising so far", st["h1"])]
    img = png("promising", W)
    if img:
        s.append(img)
    s += bullets([
        "<b>Learning by repairing history</b> (top left, 3 seeds): fix each mistake at its pivotal branch point "
        "with the smallest change. Within ~2 points of gradient-like rules while changing <b>31× fewer weights</b>, "
        "which matters for continual learning (less interference) and for hardware (fewer writes).",
        "<b>Counterfactual credit pays with depth</b> (top right, full length): no gain with one hidden "
        "layer, +1.2 points with two and +1.5 with three (2 seeds, both positive), +1.9 with four and +2.5 with "
        "five (1 seed): the gap grows monotonically with depth. Near-miss nodes influence the output only through events that did "
        "not happen, which fired-only credit cannot see.",
        "<b>Credit conservation, confirmed at full length</b> (bottom left; 2 seeds, width 400, 3 epochs): "
        "normalising competitor credit at each race, as the theory's soft-minimum derivative requires, gives "
        "0.960 / 0.952 / 0.941 at depths 1 / 2 / 3 against 0.949 / 0.942 / 0.924 without: +1.0 to +1.7 points at "
        "every depth, both seeds. (The debug runs had suggested +8 to +10; full training shrinks it.) "
        "<b>Correction:</b> the shadow neuron's debug lead did <i>not</i> hold at full length.",
        "<b>Sparse connectivity</b> (bottom right, debug): 14–22× fewer synaptic events per image. In one layer it "
        "even raised accuracy; with two layers it cost accuracy in a very short run. Full runs are queued. For "
        "scale, the MLP that matched round-3 accuracy (0.96) needs ~25k multiply-accumulates; sparse race networks "
        "at 4–8k events would win at inference if they hold accuracy.",
        "<b>Theory that predicts:</b> reading the network's unrealised futures as a dequantized tropical computation "
        "predicted credit conservation (confirmed in debug) and exact time-shift invariance (confirmed); its "
        "asynchronous form (shadow events) reproduces batch computation exactly (M20).",
    ], st)
    s.append(Paragraph("Evidence levels: \u201c3 seeds\u201d and \u201cfull length\u201d results are confirmatory-grade "
                       "within their setting; \u201cdebug\u201d results are single short runs, reported as leads.",
                       st["small"]))
    s.append(PageBreak())
    return s


def theory_pages(st, W):
    s = [Paragraph("The theory in five principles", st["h1"]),
         Paragraph("The theory note (experiments/THEORY.md) has grown to some twenty-five sections. They reduce to "
                   "five principles; each result follows from one of them. Colours give the evidence status.",
                   st["body"])]
    img = png("principles", W)
    if img:
        s.append(img)
    s.append(PageBreak())
    s += [Paragraph("What the theory added since (THEORY §27–41)", st["h1"]),
          Paragraph("Each result is labelled by what kind of claim it is. <b>Exact</b>: a derivation that holds "
                    "without approximation. <b>Scaling</b>: an order-of-magnitude argument whose exponent or sign is "
                    "the prediction. <b>Prior art</b>: found in the literature, credited. None of the predictions below "
                    "has been confirmed yet; the tests are queued.", st["body"]),
          table([
              ["result", "kind", "prediction / status"],
              ["Timing credit sums to the deadline's credit (Ward identity, §30.1)", "exact",
               "centring timing credit is the symmetry, not a heuristic"],
              ["Excitatory race nets are topical maps: timing noise never amplified; certified jitter radius "
               "(§34)", "exact", "zero flips among certified samples (M36)"],
              ["Committing to a branch costs σ × surprisal; near-miss credit is its gradient (§35)", "exact",
               "supervised loss = cost of weaving the teacher's branch"],
              ["Deep credit contracts like a Markov chain; exact kernels conserve errors in the sum (§30–31)",
               "scaling", "share Jacobian + centring trains depth 3 (M34)"],
              ["Credit through positive weights collapses to an activity (Perron) mode (§27, §29)",
               "scaling; outlier mode is prior art", "Perron centring ≥ mean centring (M33)"],
              ["Prices must be the faster timescale; our default is 10× too slow (§36)", "scaling",
               "threshold near κ ≈ η for pivotal credit (M38)"],
              ["Firing patterns are chaotic: ρ' ≈ A√ρ, no ordered phase; A ∝ 1/√k (§41)", "scaling",
               "slope ½ in log ρ across layers (M43)"],
              ["Winner–fan-in coupling k·F ≥ G; widths from the data's entropy exponent (§37)", "scaling",
               "entropy exponent measured: α ≈ 0.95, so no pyramid from input redundancy (refuted a guess)"],
              ["Optimal weaving prices commitment cost (MSPRT); race neurons are blind to absence (§38)",
               "scaling; MSPRT is prior art", "relative stopping beats the absolute race (M40)"],
              ["Concave piecewise-linear firing time, one piece per causal set (§34.1)", "prior art",
               "polyhedral geometry of TTFS networks (2026)"],
              ["Deep exact training collapses; fast homeostasis + Ward centering rescue it (§27, §30, §36)",
               "scaling, confirmed", "depth-3 exact training 0.10 → 0.87 (debug)"],
              ["Skips must be delay-matched or the shallow path wins the race (§47)", "scaling, partly refuted",
               "0.915 vs 0.898 plain, but no skips best (0.937) under local learning"],
              ["The entropic k-winner race is Fermi–Dirac; chemical potential = price (§48)",
               "exact; soft top-k prior art", "recovers ~60% of the cancellation cost"],
              ["Dilation equivariance; temporal collapse; temporal normalisation as a gauge (§49)",
               "exact symmetry; scaling", "queued"],
              ["Conserved near-miss rule is ultraconservative: forgetting bounded by the new task's mistakes, vs "
               "O(log T) for softmax SGD (§50)", "bound prior art; consequence new", "E23 running"],
          ], [86, 30, 58], st),
          Paragraph("Honest summary: the mathematics used is borrowed (Noether and Ward identities, "
                    "Perron–Frobenius and topical maps, Birkhoff contraction, Gibbs/Landauer identities, "
                    "two-timescale stochastic approximation, mean-field propagation). The applications to race "
                    "networks were not found in prior work in a few targeted searches (THEORY §32), which is not a "
                    "claim of priority.", st["small"])]
    s.append(PageBreak())
    s += [Paragraph("Theory: collapsing futures, trees of histories, repair", st["h1"]),
         Paragraph("At any moment the network holds a pool of pending futures: each unfired node's projected "
                   "firing time. A firing collapses the pool: one future becomes history and inhibition cancels its "
                   "competitors, each leaving a residue (how close it came). Because the order of events depends on "
                   "the weights, the network's computation is a <b>tree of possible histories</b>, and a small weight "
                   "change can move it to another branch. In a dense network the tree has one branch, so the "
                   "derivative along it is the whole story. Here it is not.", st["body"])]
    img = png("history_tree", W)
    if img:
        s.append(img)
    s += bullets([
        "<b>Greedy backprop</b> (EventProp in our setting) follows the realised branch only, and misses how the "
        "weights move the forks.",
        "<b>Holistic backprop</b> sums over the close calls, weighted by their probability. Branches are cheap here: "
        "a fork changes only events downstream of it, and cancellation ends it early. They can run "
        "asynchronously as <i>shadow events</i> in the same event queue.",
        "<b>History repair</b> takes the cheapest branch that gives the right answer and changes just that tie. "
        "Credit becomes a causal question (would the outcome differ but for this event?) rather than a sensitivity.",
        "Known ingredients (EventProp, perturbed argmax, surrogate gradients, race logic, Madaline Rule II, spike "
        "discontinuity estimation) are credited in experiments/RELATED_WORK.md; the tree-of-histories reading and "
        "cancellation residues as its sufficient statistic were not found in prior work.",
    ], st)
    s.append(PageBreak())
    s += [Paragraph("Checking the theory", st["h1"])]
    for name, text in (
            ("m3_alignment", "M3 measures, on a small network, whether each rule's update points along the true "
                             "gradient of expected error (estimated by finite differences under timing noise). The "
                             "output rule does; hidden credit barely does, for every feedback type."),
            ("m3_blind_spot", "Most hidden weights have no gradient at all along the realised history. Widening the "
                              "tree (more timing noise) enlarges the set that gets any signal: the blind spot that "
                              "greedy backprop cannot see.")):
        img = png(name, W * 0.92)
        if img:
            s += [img, Paragraph(text, st["body"])]
    s += [Paragraph("A correction", st["h2"]),
          Paragraph("From M3 we first concluded that the hidden layer learns mainly by its own competition, not from "
                    "labels. M21 contradicted this: a label-free competitive hidden layer ends far below a frozen "
                    "random one (0.47 vs 0.66), while label-driven hidden credit adds 16 points. Labels matter to the "
                    "hidden layer even though their alignment with the gradient of expected error is weak, an open "
                    "puzzle we are now testing.", st["body"])]
    s.append(PageBreak())
    s += [Paragraph("New learning rules", st["h1"])]
    img = png("m18_repair", W * 0.92)
    if img:
        s += [img, Paragraph("History repair on a small network (14×14 MNIST, 60 hidden nodes, 3 seeds): each "
                             "mistake is fixed by the cheapest verified single-event change, either rescheduling "
                             "the output or one hidden node. With thin-margin repairs it reaches 0.80 against 0.82 for "
                             "the gradient-like rules, while changing 31× fewer weights. Label-free hidden learning "
                             "(grey, bottom right) is the worst of all.", st["body"])]
    img = png("m13_projection", W * 0.62)
    if img:
        s += [img, Paragraph("Learning by rescheduling: \u201cfires by time c\u201d is a half-space in the weights, so a "
                             "mistake can be fixed by an exact projection with no learning rate. It ties the "
                             "near-miss rule: a valid reformulation, not an improvement, and it is now the geometric "
                             "core of history repair.", st["body"])]
    img = png("e14_depth", W * 0.95)
    if img:
        s += [Paragraph("Depth: where counterfactual credit pays (E14)", st["h2"]), img,
              Paragraph("Hidden credit per layer through fixed random feedback. With one hidden layer, counterfactual "
                        "and fired-only credit tie. Deeper, a near-miss node matters only through nodes its spike would "
                        "have tipped over threshold: a path made of events that did not happen, which only counterfactual "
                        "credit reaches. At full training the gap is modest (about 2 points at depth 3) but grows steadily "
                        "with depth, while a frozen hidden stack collapses.",
                        st["body"])]
    img = png("e13_bandit", W * 0.8)
    e13 = e13_summary()
    if img:
        s += [Paragraph("Reinforcement learning: reward only (E13a)", st["h2"]), img]
    elif e13:
        names = {"supervised": "supervised (reference)", "nearmiss": "near-miss guess", "rstdp": "reward-modulated",
                 "pool_pg": "pool policy gradient"}
        rows = [["rule (reward only)", "held-out accuracy", "seeds"]] + [
            [names.get(k, k), f"{m:.3f} ± {sd:.3f}", str(n)] for k, (m, sd, n) in sorted(e13.items(),
                                                                                          key=lambda kv: -kv[1][0])]
        s += [Paragraph("Reinforcement learning: reward only (E13a)", st["h2"]), table(rows, [70, 50, 20], st)]
    else:
        s += [Paragraph("Reinforcement learning: reward only (E13a)", st["h2"]),
              Paragraph("In a first small test, crediting the near misses when wrong reached 0.30, against 0.19 for "
                        "reward-modulated (winner only) learning and 0.59 with labels. Full runs are queued.",
                        st["body"])]
    s.append(PageBreak())
    return s


def build():
    """The report: what is known, organized by claim (mirrors REPORT.md)."""
    st = styles()
    W = 174

    def fig(fn, w=W):
        f = fn()
        return [fig_image(f, w)] if f is not None else []

    def P(t, style="body"):
        return Paragraph(t, st[style])

    s = [P("Sleeping Machines: what is known", "title"),
         P(f"Computing in time with races, holds and vetoes · report, {date.today():%d %B %Y}", "sub"),
         P("Sleeping Machines proposes that computation can happen <b>in time rather than memory</b>: candidate events "
           "race, the first to fire cancels the rest, and what a node computes is set by delays, by how long it holds an "
           "input, and by inhibition that arrives in time. This report states what is now known about such networks, "
           "why, and what remains open. Derivations and proofs are in experiments/THEORY.md (cited as §n)."),
         P("Summary", "h1")]
    s += bullets([
        "<b>What one node computes is exactly characterized, and it says where depth is needed.</b> A node with a trigger, "
        "hold and veto inputs accepts a product set in lag coordinates; one node orders at most three events; two layers "
        "compute every conjunction of bounded time differences, three layers every union (§71, proved; exhaustive "
        "search agrees). Nodes with state (arm/disarm) are strictly stronger.",
        "<b>Event networks learn timing with very few mistakes</b> if winning is positional (pull the event that should "
        "have won, never push a loser later) and order uses held intervals, not aligned delays: O(log N + a few) "
        "mistakes per node.",
        "<b>On a timing task a learned event network matches or beats dense models at 10⁴–10⁵× lower cost</b>, with "
        "nothing given: 1.000 at 7.5 synaptic events per episode vs 0.989–0.996 for event-token Transformers at "
        "146k–1.16M multiply-adds and 0.995 for a clocked conv net at 3.07M.",
        "<b>Depth pays when composition is a hold/trigger chain</b> (0.97 vs 0.39 at depth 1), but a well-trained "
        "Transformer is more accurate on the composition task (0.998) at ≈ 10⁴× the cost.",
        "<b>Grokking occurs, by a route change under sleep, and only for relations the substrate can express.</b> "
        "Without sleep the network memorizes; with sleep it generalizes after a delay (0.93–0.99, reliable with cooled "
        "timing noise); a data × sleep phase diagram shows memorization, grokking and collapse; with depth, (a + b + c) "
        "mod p through two composed rhythm stages reaches 0.99–1.00 (3 seeds, p = 17 and 31).",
        "<b>Learning cost follows activity, not model size:</b> 12 → 48 input channels leaves learning mistakes flat and "
        "makes inference cheaper (§77).",
        "<b>Not yet: real asynchronous benchmarks.</b> SHD below dense baselines; on the market, correctly posed as "
        "trading with costs, no learner profits and the native one learns to stay out; an online world model of the "
        "stream matches a Hawkes process but not a neural point process.",
    ], st)
    s.append(PageBreak())

    s += [P("1. What an event node computes", "h1"),
          P("<b>Primitives.</b> Spike times; delay; first-of (min) and all-of (max); hold (an input opens a window of given "
            "duration); veto (an input blocks the node while within its window). A node fires at its trigger if its holds "
            "are met and no veto is active."),
          P("<b>Theorem (§71).</b> In lag coordinates relative to the trigger a stateless node accepts a product set. So one "
            "node orders at most three events (trigger the middle one, hold the first, veto the last until it arrives); two "
            "layers compute every conjunction of bounded differences with exclusions (every zone of timed-automata "
            "verification); three layers every finite union. <i>Evidence:</i> an exhaustive search over single nodes finds "
            "the proof's own node for 3 events and none for 4 on grids of 1,680 and 11,880 configurations; all thirteen "
            "Allen interval relations built at depth 2 are exact on ~19,900 interval pairs (E39).")]
    s += fig(fig_depth_theorem, W * 0.9)
    s += [P("<b>Stateful nodes are stronger.</b> A node armed by A, disarmed by C and fired by B computes an XNOR of two "
            "order relations, which no stateless node computes (exhaustive search, E39b); a toggle node computes parity."),
          P("<b>Clockless computation computes relations, not sums</b>: shift symmetry forbids adding two times; one shared "
            "rhythm makes cyclic arithmetic computable (§56)."),
          P("<b>Completeness and reliability (E30).</b> A two-counter machine wired from these primitives plus one oscillator "
            "runs exactly (Turing-complete given timing precision). Jitter accumulates like a random walk; one restoring "
            "coincidence per cycle makes reliability independent of program length.")]
    s += fig(fig_e30, W * 0.85)
    s.append(PageBreak())

    s += [P("2. How event networks learn", "h1"),
          P("<b>Winning is positional (§54).</b> Pull the event that should have won; never push a loser later, which "
            "hands the win to the next loser and collapses the detectors. With learned delays on (a + b) mod p: push at "
            "chance in all 15 runs, pull-only 0.97–0.98 (E26). Timing noise cooled to zero makes small-data "
            "generalization reliable.")]
    s += fig(fig_e26, W * 0.85)
    s += [P("<b>Order needs held intervals, not aligned delays (§61).</b> Delaying A to meet B compresses the interval so a "
            "veto cannot see where C fell; A opening a window that B must hit keeps it. With holding, the detector for "
            "“B within Δ after A unless C” reaches 1.000 on 5 seeds with ~450 updates in 200k episodes, and veto is worth "
            "19 points (alignment: 0.914, veto worth 1.8)."),
          P("<b>The rules are online learning on the simplex (§68):</b> pulls moving a fixed fraction of a conserved budget "
            "are Winnow/Hedge-type updates with O(k log N) mistake bounds; durations are interval learning, vetoes "
            "monotone disjunctions. With nothing given, detectors learn channels, durations and vetoes: 1.000 ×4, 0.9995, "
            "443–1,530 updates (E35)."),
          P("<b>Routing needs counterfactuals (§57).</b> Credit along a spike's causal path cannot say whether another route "
            "should have been taken; cancelled near misses supply it. Without it a two-layer network stays at chance "
            "(0.17 vs 0.72–0.74, E28).")]
    s.append(PageBreak())

    s += [P("3. Against dense models and Transformers", "h1"),
          table([["model", "test accuracy", "cost per episode", "learning"],
                 ["<b>event network, nothing given (E35)</b>", "<b>1.000 ×4, 0.9995</b>", "<b>7.5 synaptic events</b>",
                  "443–1,530 updates, 200k episodes"],
                 ["event-token Transformer, 2M episodes", "0.989–0.996", "146k–1.16M MACs", "backprop"],
                 ["event-token Transformer, 200k episodes", "0.65–0.97", "5k–576k MACs", "backprop"],
                 ["clocked conv net (E32)", "0.995 / 0.984 / 0.895", "3.07M / 123k / 1.9k MACs", "backprop, 200k"]],
                [56, 36, 42, 40], st)]
    s += fig(fig_e32, W * 0.95)
    s += [P("<b>Scaling with the input basis (E35, §77).</b> At fixed spikes per episode, 12 → 48 candidate channels "
            "leaves accuracy ≈ 1.0 and learning updates flat while synaptic events per episode fall: learning and "
            "inference cost follow activity, not the size of the basis.")]
    s += [P("<b>Why.</b> Attention cannot see order without position information and must synthesize time comparisons from "
            "dot products at O(n²·d) per layer; a hold/trigger node computes the comparison as its primitive, at the cost "
            "of its input events (§70). The clocked model also pays per time bin, so silence multiplies its cost; a conv "
            "net evaluated only where spikes are would cost ≈ 100 multiply-adds, not millions (§63). At published "
            "per-operation energies the gap at equal accuracy is ≈ 10⁴×. <i>Limits:</i> one task built around the "
            "primitives; the Transformer needed 10× more training for parity; a relative-time-attention Transformer "
            "is being run.")]
    s.append(PageBreak())

    s += [P("4. Depth and composition", "h1"),
          P("Classes that are ordered combinations of shared parts are zones over part events, so the theorem prescribes two "
            "layers: parts (hold nodes on channel pairs) and class nodes that hold one part's spike and are triggered by "
            "another's, so a single part cannot satisfy them. Learned natively: 0.97 with part windows at the motif scale, "
            "0.86 with a generic bank of window scales, depth 1 0.39 (5 seeds, E34). A readout that accumulates evidence "
            "fails at conjunction even though its hidden nodes learn parts (E28). An event-token Transformer trained on 2M "
            "episodes reaches 0.998 at ≈ 175k–690k multiply-adds vs ≈ 14 events. The chains are not training-limited: at 200k "
            "episodes they stay at 0.954 (tuned) and 0.87 (bank); some class routes lock onto wrong parts.")]
    s += fig(fig_e34, W * 0.9)
    s += [P("5. Generalization and grokking", "h1"),
          P("<b>What counts (§58):</b> restriction, forced generalization above capacity, and grokking (the relation reached "
            "while memorizers are available) are different claims; each reports ρ = n/params. <b>Per-class parameters "
            "cannot generalize on (a + b) mod p (§66):</b> each operand occurs once per class, so generalization needs shared "
            "intermediates; in time, a sum needs a rhythm."),
          P("<b>Grokking as a route change (E37).</b> Each class has a pair-node lookup that can memorize everything "
            "(ρ ≈ 0.016) and a shared route through a rhythm with learned delays; learning is errors-only. Without sleep: "
            "train 1.0, test 0.03–0.04. With sleep (λ = 0.02–0.2): test 0.93–0.97 in 2 of 3 seeds, after a delay; the "
            "third seed collapses. Sleep without the rhythm: train 0.45, test 0. The rhythm is used only where it fits: "
            "a − b and relabelled sums grok (0.95–0.98), a·b in 1 of 3 seeds, while a² + ab + b² and random tables stay "
            "at chance on unseen pairs (0.01–0.04) and their training accuracy erodes under sleep. A data × sleep phase "
            "diagram (4 × 4, 3 seeds) shows no grokking below 20–30% of pairs, and above it a minimum sleep that falls "
            "with data. Larger problems grok more reliably: 2/3 seeds at p = 31 and 59, 3/3 at p = 97 (ρ ≈ 0.005). With depth (E41, 3 seeds): (a + b + c) mod p through two composed rhythm stages, 0.99–1.00 on unseen "
            "triples from 30% of them at p = 17 and 31, and from 10% in 2 of 3 seeds at p = 31 (ρ ≈ 0.003); chance "
            "without sleep; collapse without the chain.")]
    s += fig(fig_e37, W * 0.9)
    s += [P("<b>Why (§69, §72).</b> Error-gated learning makes memorization absorbing. Sleep keeps a parameter only if it "
            "is used by more than m* = λθ/(eη) examples: lookup entries serve one and die, the rhythm's delays serve many "
            "and survive, and once the rhythm answers a pair its lookup entry is never relearned. This predicts "
            "memorization, grokking and collapse regimes (all observed) and a data threshold n* ∝ pλθ/(eη).")]
    s.append(PageBreak())

    s += [P("6. The weight race", "h1"),
          P("The original architecture (integrate-to-threshold nodes with learned weights, racing in groups) holds its first "
            "results: a cancelled node can be taught (E4: 0.79 at K = 128 where reward-modulated rules are at chance); races "
            "decide as fast as the evidence allows (E2, matching the optimal MSPRT); learning work tracks activity (E5: "
            "≈ 300× fewer updates); local learning reaches ≈ 0.96 on latency-coded MNIST with counterfactual credit that "
            "pays more with depth (E6, E14).")]
    im = png("e14_depth", W * 0.7)
    if im:
        s.append(im)
    s += [P("Its limits: depth still costs accuracy; only its single racing layer beats an equal dense model at inference "
            "(≈ 2.4×); it forgets more than SGD (E23); it memorizes instead of grokking (E24); and the rules that fixed the "
            "timing networks do not transfer to its weights (SHD 0.04–0.29 vs 0.35).")]
    s += [P("7. Real data", "h1")]
    s += bullets([
        "<b>Spiking Heidelberg Digits:</b> the weight race 0.35 vs 0.56–0.59 for a dense MLP. For the timing architecture the "
        "representation is the bottleneck: onset-referenced parts give a dense readout 0.566, and the native learner "
        "overfits (0.27–0.33). SHD is only ≈ 6× sparser than a 10 ms raster, a weak test of the paradigm.",
        "<b>Market stream posed as trading with costs (E42, pilot):</b> imitating a hindsight teacher over-trades and "
        "loses; a profit-priced event learner makes 26 changes in 7 days (−170 bp): it learns that trading does not pay.",
        "<b>Online world model (E44):</b> a temporal point process of four event types learned from every event; the "
        "native model (−2.64/−2.85/−2.70 nats per event) matches Hawkes (−2.62/−2.94/−2.84) and trails a GRU neural "
        "point process (−2.61/−2.52) by ≈ 0.2 nats.",
    ], st)
    s += [P("8. Open problems and next steps", "h1")]
    s += bullets([
        "Grokking theory tests (running): phase diagram over data × sleep (§72); relations the rhythm cannot express; sleep "
        "as the pressure toward reusable parts (§73); O(log N) learning as the candidate basis grows (§74).",
        "Fair baselines (running): relative-time-attention Transformers; the chains with 5–25× more training.",
        "Composition accuracy against Transformers; grokking reliability; native learning of sparse parity (§75, open); "
        "a real stream with rare, precisely timed events (§55).",
    ], st)
    s.append(Spacer(1, 6))
    s.append(P("Every mechanism is an event handler (local state, triggered by events, cost proportional to events); dense "
               "procedures are diagnostics only. Reproduce: python report/figures_time.py && python report/make_pdf.py.",
               "small"))
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Sleeping Machines — what is known",
                            author="Sleeping Machines project")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
