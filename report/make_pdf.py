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
from matplotlib.lines import Line2D  # noqa: E402
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


def fig_potential_evidence():
    """Show the two strongest current learning signals on their own task scales."""
    fig, (ax_lm, ax_retrieval) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.45, 1]})

    # Real text8 test results: E79 expert count changes with data (K=5/6/7).
    e79 = []
    for path in glob.glob(os.path.join(RES, "e79", "race_mixer_D*_K*_e77none.json")):
        row = load(path)
        D = int(row["args"]["D"])
        if D in (1_000_000, 10_000_000, 90_000_000):
            e79.append((D, int(row["args"]["K"]), row["copy_window_256"]["race_frozen_test_bpc"],
                        row["copy_window_256"]["race_online_test_bpc"]))
    e79.sort()
    ds = np.array([x[0] for x in e79], dtype=float)
    frozen = np.array([x[2] for x in e79])
    online = np.array([x[3] for x in e79])
    ax_lm.plot(ds, frozen, color=BLUE, marker="o", label="E79 race mixture · frozen")
    ax_lm.plot(ds, online, color=AQUA, marker="o", ls="--", label="E79 · online adaptation")
    ax_lm.annotate("E79 frozen", (ds[0], frozen[0]), xytext=(7, 8), textcoords="offset points",
                   fontsize=5.8, color=BLUE)
    ax_lm.annotate("E79 online", (ds[0], online[0]), xytext=(7, -12), textcoords="offset points",
                   fontsize=5.8, color=AQUA)

    base = os.path.join(RES, "e64")
    lstm = load(os.path.join(base, "lstm_D1000000_s256_p20_dr0.2_v.json"))["test_bpc"]
    lstm_10m = load(os.path.join(base, "lstm_D10000000_s512_p6_dr0.1_v.json"))["test_bpc"]
    tf = load(os.path.join(base, "tf_D1000000_s256_p20_dr0.2_v.json"))["test_bpc"]
    tf_10m_final = load(os.path.join(base, "tf_D10000000_s256_L4_p4_dr0.1_v.json"))
    tf_10m_test = tf_10m_final["test_bpc"]
    ax_lm.scatter([1_000_000, 10_000_000], [lstm, lstm_10m], color=ORANGE, marker="s", s=36, zorder=4,
                  label="E64b LSTM · 1M and 10M")
    ax_lm.annotate(f"10M LSTM {lstm_10m:.3f}", (10_000_000, lstm_10m), xytext=(5, 7),
                   textcoords="offset points", fontsize=6.8, color=ORANGE)
    ax_lm.scatter([1_000_000], [tf], color=GRAY, marker="D", s=34, zorder=4, label="E64b Transformer · 1M")
    ax_lm.annotate(f"1M TF test {tf:.3f}", (1_000_000, tf), xytext=(6, 7),
                   textcoords="offset points", fontsize=5.8, color=GRAY)
    ax_lm.scatter([10_000_000], [tf_10m_test], color=INK, marker="^", s=38, zorder=5,
                  label="E64b Transformer · 10M test")
    ax_lm.annotate(f"10M 4L TF test {tf_10m_test:.3f}", (10_000_000, tf_10m_test), xytext=(7, -12),
                   textcoords="offset points", fontsize=6.2, color=INK)
    ax_lm.set_xscale("log")
    ax_lm.set_xticks([1_000_000, 10_000_000, 90_000_000], ["1M", "10M", "90M"])
    ax_lm.set_xlim(700_000, 130_000_000)
    ax_lm.set_ylim(1.35, 2.55)
    ax_lm.set_xlabel("training characters")
    ax_lm.set_ylabel("bits per character · lower is better")
    ax_lm.set_title("A · Real text8 language modeling")

    # Learned retrieval on E61's separate synthetic recall task.
    event = load(os.path.join(RES, "e61", "event_K32_n8.json"))
    tf_paths = glob.glob(os.path.join(RES, "e61", "tf_K32_n8*.json"))
    event_at_4k = []
    for row in event["rows"]:
        vals = [p["n32"] for p in row["curve"] if p["seen"] <= 4_000]
        event_at_4k.append(max(vals))
    tf_max = max(p["n32"] for path in tf_paths for row in load(path)["rows"] for p in row["curve"])
    labels = ["Local race attention", "Best of 7 Transformers"]
    vals = [min(event_at_4k), tf_max]
    cols = [BLUE, GRAY]
    y = np.arange(2)
    ax_retrieval.barh(y, vals, color=cols, height=0.55)
    ax_retrieval.set_yticks(y, labels, fontsize=7)
    ax_retrieval.invert_yaxis()
    ax_retrieval.set_xlim(0, 1.14)
    ax_retrieval.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_retrieval.axvline(1 / 32, color=INK, ls=":", lw=1)
    ax_retrieval.text(vals[0] - 0.02, 0, "100% · 5/5 by 4k", ha="right", va="center", fontsize=6.5, color="white")
    ax_retrieval.text(vals[1] + 0.025, 1, f"{tf_max:.0%} · up to 1M", ha="left", va="center", fontsize=6.5, color=INK)
    ax_retrieval.set_xlabel("accuracy at 4× context")
    ax_retrieval.set_title("B · Synthetic key retrieval")

    fig.suptitle("Measured signals for the frontier-model hypothesis", x=0.02, ha="left", fontsize=9.5,
                 fontweight="bold")
    fig.text(0.02, 0.015,
             f"A: Every plotted BPC is held-out test; the 10M Transformer checkpoint was selected on validation "
             f"(step {tf_10m_final['best_step']:,}/{tf_10m_final['steps']:,}). "
             "E79 K rises 5→6→7. B: synthetic E61 recall; dotted line = chance (1/32).", fontsize=6.0, color=MUTED)
    fig.text(0.02, -0.018,
             "Theory: vector-delay retrieval computes exact softmax; fixed-schedule memory scan has O(G) work and O(log G) span.",
             fontsize=6.0, color=MUTED)
    fig.text(0.02, -0.051,
             f"10M 4-layer Transformer: {tf_10m_test:.4f} held-out test bpc after "
             f"{tf_10m_final['steps']:,} updates.", fontsize=6.0, color=MUTED)
    fig.tight_layout(rect=(0, 0.11, 1, 0.91))
    out = os.path.join(os.path.dirname(__file__), "figures", "potential_evidence.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_route_diagnostics():
    """Summarize the matched depth-4 route-shadow and gradient diagnostics."""
    matches = glob.glob(os.path.join(
        RES, "e83", "*objevent_prefix_cfnorm1_b0.5_sg0.25_w1_dl5_lr0.001_gc1_spk_s6.json"))
    if not matches:
        raise FileNotFoundError("missing matched E83 local-counterfactual result")
    rows = load(matches[0])["curve"]
    fig, (ax_delta, ax_grad) = plt.subplots(1, 2, figsize=(7.2, 2.65))
    layers = np.arange(1, 5)
    colors_ep = [BLUE, ORANGE]
    offsets = [-0.08, 0.08]
    for ei, row in enumerate(rows[:2]):
        mean = np.asarray(row["counterfactual_layer_mean_open_minus_closed_loss"], dtype=float)
        spread = np.asarray(row["counterfactual_layer_shadow_delta_std"], dtype=float)
        ax_delta.errorbar(layers + offsets[ei], mean, yerr=spread, color=colors_ep[ei],
                          marker="o", capsize=2.5, lw=1.3, label=f"epoch {ei + 1}")
    ax_delta.axhline(0, color=INK, lw=0.9, ls=":")
    ax_delta.set_xticks(layers)
    ax_delta.set_xlabel("content-route layer")
    ax_delta.set_ylabel("$L_{open}-L_{closed}$")
    ax_delta.set_title("A · Shadow utility by layer")
    ax_delta.legend(fontsize=6.5)

    for ei, row in enumerate(rows[:2]):
        ratio = np.asarray(row["counterfactual_layer_to_pathwise_norm_ratios"], dtype=float)
        cosine = np.asarray(row["counterfactual_layer_pathwise_cosines"], dtype=float)
        ax_grad.plot(layers, ratio, color=colors_ep[ei], marker="o", lw=1.3,
                     label=f"norm ratio · epoch {ei + 1}")
        ax_grad.plot(layers, cosine, color=colors_ep[ei], marker="^", ls="--", lw=1.0,
                     label=f"cosine · epoch {ei + 1}")
    ax_grad.axhline(0, color=INK, lw=0.9, ls=":")
    ax_grad.set_xticks(layers)
    ax_grad.set_ylim(-0.02, 0.09)
    ax_grad.set_xlabel("hidden layer")
    ax_grad.set_ylabel("dimensionless gradient statistic")
    ax_grad.set_title("B · Counterfactual vs pathwise gradient")
    ax_grad.legend(fontsize=5.8, ncol=2, loc="upper right")

    fig.suptitle("E83 · matched depth-4 local counterfactual pilot", x=0.02,
                 ha="left", fontsize=9.2, fontweight="bold")
    fig.text(0.02, 0.005,
             "A: positive loss difference disfavors opening; bars show ±1 SD across sampled routes. "
             "B: norm ratio and cosine. One seed; held-out accuracy stayed near chance.",
             fontsize=6.0, color=MUTED)
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    out = os.path.join(os.path.dirname(__file__), "figures", "e83_route_gradient_diagnostics.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_d4_support():
    """Show matched 128-example D4 support, accuracy, race, and calibration."""
    runs = {}
    for fusion in ("all_depths", "deepest"):
        pattern = os.path.join(RES, "e83", f"*rf{fusion}*_eval128*_spk_s6.json")
        matches = []
        for path in glob.glob(pattern):
            result = load(path)
            args = result.get("args", {})
            if (args.get("eval_limit") == 128 and args.get("epochs") == 4
                    and args.get("seed") == 6
                    and args.get("input_count_payload", "off") == "off"
                    and not args.get("early_event_skip", False)):
                matches.append((path, result))
        if not matches:
            raise FileNotFoundError(f"missing seed-6 eval128 D4 {fusion} run")
        runs[fusion] = sorted(matches, key=lambda row: row[0])[0][1]["curve"]
        result = sorted(matches, key=lambda row: row[0])[0][1]
        runs[f"{fusion}_outputs"] = result["eval_output_payloads"]
    skip_matches = []
    for path in glob.glob(os.path.join(RES, "e83", "*rfdeepest_skfirst*eval128*_spk_s6.json")):
        result = load(path)
        args = result.get("args", {})
        if (args.get("eval_limit") == 128 and args.get("epochs") == 4
                and args.get("seed") == 6 and args.get("early_event_skip", False)):
            skip_matches.append((path, result))
    if not skip_matches:
        raise FileNotFoundError("missing seed-6 eval128 D4 sparse-skip run")
    skip_result = sorted(skip_matches, key=lambda row: row[0])[0][1]
    runs["skipfirst"] = skip_result["curve"]
    runs["skipfirst_outputs"] = skip_result["eval_output_payloads"]

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 4.5))
    ax_support, ax_acc, ax_race, ax_nll = axes.flat
    layers = np.arange(1, 5)
    width = 0.25
    end_support = {
        key: np.asarray(runs[key][-1]["event_support_coverage"], dtype=float) * 100
        for key in ("all_depths", "deepest", "skipfirst")}
    ax_support.bar(layers - width, end_support["all_depths"], width,
                   color=BLUE, label="all depths")
    ax_support.bar(layers, end_support["deepest"], width,
                   color=ORANGE, label="deepest only")
    ax_support.bar(layers + width, end_support["skipfirst"], width,
                   color=AQUA, label="layer-1 skip")
    ax_support.set_xticks(layers)
    ax_support.set_ylim(0, 108)
    ax_support.set_xlabel("hidden layer")
    ax_support.set_ylabel("examples with ≥1 event (%)")
    ax_support.set_title("A · Endpoint event support", fontsize=8)
    for x, key in ((layers - width, "all_depths"), (layers, "deepest"),
                   (layers + width, "skipfirst")):
        for xi, yi in zip(x, end_support[key]):
            ax_support.text(xi, yi + 2, f"{yi:.0f}", ha="center", fontsize=6, color=INK)

    epochs = np.arange(1, len(runs["all_depths"]) + 1)
    for key, color, name in (("all_depths", BLUE, "all depths"),
                             ("deepest", ORANGE, "deepest only"),
                             ("skipfirst", AQUA, "layer-1 skip")):
        terminal = [100 * row["event_terminal_accuracy"] for row in runs[key]]
        anytime = [100 * sum(x["predicted_class"] == x["true_class"]
                             for x in outputs) / len(outputs)
                   for outputs in runs[f"{key}_outputs"]]
        ax_acc.plot(epochs, terminal, color=color, marker="o", label=f"{name} · terminal")
        ax_acc.plot(epochs, anytime, color=color, marker="s", ls="--",
                    label=f"{name} · race + fallback")
        late_nll = [row["prefix_window_nll_by_stratum"][-1] for row in runs[key]]
        ax_nll.plot(epochs, late_nll, color=color, marker="o", label=name)
    ax_acc.axhline(5, color=GRAY, ls=":", lw=1, label="20-class chance")
    ax_acc.set_xticks(epochs)
    ax_acc.set_ylim(0, 24)
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("held-out accuracy (%)")
    ax_acc.set_title("B · Accuracy (solid: terminal; dashed: race + fallback)", fontsize=7.4)
    model_handles = [Line2D([0], [0], color=color, marker="o", label=name)
                     for color, name in ((BLUE, "all depths"), (ORANGE, "deepest"),
                                         (AQUA, "layer-1 skip"))]
    ax_acc.legend(handles=model_handles, fontsize=5.5, ncol=3, loc="upper left")
    model_keys = ("all_depths", "deepest", "skipfirst")
    last = [runs[key][-1] for key in model_keys]
    x = np.arange(len(model_keys))
    race_width = 0.19
    coverage = [100 * row["race_coverage"] for row in last]
    emitted_accuracy = [100 * (row["race_acc_when_emitted"] or 0) for row in last]
    mean_confidence = []
    for key in model_keys:
        emitted = [row for row in runs[f"{key}_outputs"][-1]
                   if row["emission"] == "event_race"]
        mean_confidence.append(100 * np.mean([
            max(row["value_vector"]) for row in emitted]) if emitted else 0.0)
    ax_race.bar(x - race_width, coverage, race_width,
                color=BLUE, label="emission coverage")
    ax_race.bar(x, emitted_accuracy, race_width,
                color=ORANGE, label="accuracy if emitted")
    ax_race.bar(x + race_width, mean_confidence, race_width,
                color=AQUA, label="mean emitted confidence")
    ax_race.set_xticks(x, ["all depths", "deepest only", "layer-1 skip"])
    ax_race.set_ylim(0, 90)
    ax_race.set_ylabel("fixed 0.6 threshold (%)")
    ax_race.set_title("C · Race bars: coverage · accuracy · confidence", fontsize=7.4)
    for xi, values in enumerate(zip(coverage, emitted_accuracy, mean_confidence)):
        for offset, val in zip((-race_width, 0, race_width), values):
            ax_race.text(xi + offset, val + 0.8, f"{val:.1f}", ha="center", fontsize=6)

    ax_nll.axhline(np.log(20), color=GRAY, ls=":", lw=1, label="uniform NLL")
    ax_nll.set_yscale("log")
    ax_nll.set_xticks(epochs)
    ax_nll.set_xlabel("epoch")
    ax_nll.set_ylabel("late-prefix NLL · log scale")
    ax_nll.set_title("D · Posterior quality", fontsize=8)
    ax_nll.legend(fontsize=6.2, loc="upper right")

    fig.suptitle("E83 · matched depth-4 readout comparison", x=0.02,
                 ha="left", fontsize=9.2, fontweight="bold")
    fig.text(0.02, 0.095,
             "Seed 6; 128 train / 128 held-out examples from two speakers; fixed epoch 4. All-depth: 17/128 terminal, "
             "20/128 race + fallback; deepest: 7/128; skip: 9/128, 99.2% layer-4 support.",
             fontsize=5.2, color=MUTED)
    fig.text(0.02, 0.068,
             "Paired race + fallback: all-depth vs deepest p=0.0146; skip vs strict p=0.791. Emitted confidence "
             "63.8%, accuracy 23.8%.", fontsize=5.2, color=MUTED)
    fig.text(0.02, 0.041,
             "Seed 7 did not replicate the paired gain: 9 vs 6 race + fallback correct (p=0.607); layer-4 support "
             "22.7% vs 16.4%; 0/8 emissions correct at 63.1% mean confidence.",
             fontsize=5.2, color=MUTED)
    fig.text(0.02, 0.014,
             "Leave-one-head-out terminal accuracy in seed 6 (omit layers 1–4): 10.2 / 7.0 / 10.2 / 13.3%. "
             "Poor NLL; two-speaker validation; candidate-score work is not energy.",
             fontsize=5.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93), h_pad=1.0, w_pad=1.0)
    out = os.path.join(os.path.dirname(__file__), "figures", "e83_d4_readout_support.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_equal_updates():
    """Compare data diversity at a fixed update budget, showing both seed directions."""
    arms = {}
    for seed in (6, 7):
        for limit, tag in ((120, "120x4"), (480, "480x1")):
            pattern = os.path.join(
                RES, "e83", f"*rngsplit_split_u120_d{limit}*_s{seed}.json")
            matches = []
            for path in glob.glob(pattern):
                result = load(path)
                args = result.get("args", {})
                expected_epochs = 4 if limit == 120 else 1
                if (args.get("seed") == seed and args.get("limit") == limit
                        and args.get("epochs") == expected_epochs
                        and args.get("eval_limit") == 128
                        and args.get("rng_protocol") == "split"):
                    matches.append(result)
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"expected one equal-update E83 arm for seed {seed}, limit {limit}; found {len(matches)}")
            arms[(seed, tag)] = matches[0]["curve"][-1]

    x = np.arange(2)
    width = 0.32
    series = (("120x4", "120 examples × 4 epochs", BLUE, -width / 2),
              ("480x1", "480 examples × 1 epoch", ORANGE, width / 2))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25))
    metrics = (
        ("event_terminal_accuracy", "A · Terminal accuracy (%)", 100, (0, 24)),
        ("event_support_coverage", "B · Layer-4 support (%)", None, (0, 10)),
        ("prefix_window_nll_by_stratum", "C · Late-prefix NLL", None, None),
    )
    for ax, (key, title, scale, ylim) in zip(axes, metrics):
        for tag, label, color, offset in series:
            vals = []
            for seed in (6, 7):
                record = arms[(seed, tag)]
                if key == "event_support_coverage":
                    value = 100 * record[key][-1]
                elif key == "prefix_window_nll_by_stratum":
                    value = record[key][-1]
                else:
                    value = 100 * record[key]
                vals.append(value)
            bars = ax.bar(x + offset, vals, width, color=color, label=label)
            for bar, value in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.15 if key != "prefix_window_nll_by_stratum" else 0.08),
                        f"{value:.1f}", ha="center", va="bottom", fontsize=6)
        ax.set_title(title, fontsize=7.4)
        ax.set_xticks(x, ["seed 6", "seed 7"])
        if ylim:
            ax.set_ylim(*ylim)
        if key == "event_terminal_accuracy":
            ax.axhline(5, color=GRAY, ls=":", lw=1)
            ax.set_ylabel("accuracy (%)")
        elif key == "event_support_coverage":
            ax.set_ylabel("active examples (%)")
        else:
            ax.axhline(np.log(20), color=GRAY, ls=":", lw=1)
            ax.set_ylim(0, 7.5)
            ax.set_ylabel("NLL (uniform = ln 20)")
    axes[0].legend(fontsize=5.3, loc="upper left")
    fig.suptitle("E83 · more data at equal optimizer updates does not give a stable gain",
                 x=0.02, ha="left", fontsize=8.7, fontweight="bold")
    fig.text(0.02, 0.015,
             "120 updates per arm; nested 120/480 training examples; same held-out set within seed. Accuracy direction "
             "reverses: seed 6 favors 480 (20 vs 8, p=.023), seed 7 favors 120 (20 vs 8, p=.036). Two held-out speakers.",
             fontsize=5.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.12, 1, 0.88), w_pad=1.5)
    out = os.path.join(os.path.dirname(__file__), "figures", "e83_equal_update_data_budget.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_route_reachability():
    """Contrast static mask paths with realized deep support and depth survival needs."""
    arms = {}
    for seed in (6, 7):
        for limit, tag in ((120, "120x4"), (480, "480x1")):
            pattern = os.path.join(
                RES, "e83", f"*rngsplit_split_u120_d{limit}*_s{seed}.json")
            matches = []
            for path in glob.glob(pattern):
                result = load(path)
                args = result.get("args", {})
                if (args.get("seed") == seed and args.get("limit") == limit
                        and args.get("eval_limit") == 128):
                    matches.append(result)
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"expected one E83 support record for seed {seed}, limit {limit}; found {len(matches)}")
            arms[(seed, tag)] = matches[0]["curve"][-1]["event_support_coverage"][-1] * 100

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    x = np.arange(2)
    width = 0.31
    axes[0].bar(x - width / 2, [90.2, 98.0], width,
                 color=BLUE, label="L1→L4 unit pairs with a path")
    axes[0].bar(x + width / 2, [100, 100], width,
                 color=AQUA, label="input bands reaching any L4 unit")
    axes[0].set_xticks(x, ["seed 6", "seed 7"])
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("candidate-graph reachability (%)")
    axes[0].set_title("A · Static routes exist", fontsize=7.4)
    axes[0].legend(fontsize=4.8, loc="lower right")
    for ax in axes[:1]:
        for bars in ax.containers:
            ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=5.8)

    for tag, label, color, offset in (
            ("120x4", "120 × 4 epochs", BLUE, -width / 2),
            ("480x1", "480 × 1 epoch", ORANGE, width / 2)):
        vals = [arms[(seed, tag)] for seed in (6, 7)]
        bars = axes[1].bar(x + offset, vals, width, color=color, label=label)
        axes[1].bar_label(bars, fmt="%.1f", padding=2, fontsize=5.8)
    axes[1].set_xticks(x, ["seed 6", "seed 7"])
    axes[1].set_ylim(0, 10)
    axes[1].set_ylabel("active examples (%)")
    axes[1].set_title("B · Realized layer-4 support", fontsize=7.4)
    axes[1].legend(fontsize=5.1, loc="upper left")
    axes[1].text(0.5, 0.48, "Sparse layer-1 skip: 99–100% support\nwithout paired accuracy gain",
                 transform=axes[1].transAxes, ha="center", va="center", fontsize=5.3,
                 color=MUTED, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})

    depths = np.array([4, 8, 16])
    survival = 100 * 0.5 ** (1 / (depths - 1))
    axes[2].plot(depths, survival, marker="o", color=ORANGE)
    for d, s in zip(depths, survival):
        axes[2].annotate(f"{s:.1f}%", (d, s), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=5.8)
    axes[2].set_xticks(depths)
    axes[2].set_ylim(72, 100)
    axes[2].set_xlabel("hidden depth")
    axes[2].set_ylabel("per-layer survival needed")
    axes[2].set_title("C · Keep half alive to the end", fontsize=7.4)

    fig.suptitle("E83 · candidate connectivity is broad; event support is not",
                 x=0.02, ha="left", fontsize=8.8, fontweight="bold")
    fig.text(0.02, 0.015,
             "Exact seed-6/7 masks: 90.2%/98.0% of first-to-fourth unit pairs connected; all 140 bands reach L4. "
             "Equal-update arms have 1.6–7.0% L4 support. Panel C assumes equal conditional survival and full L1 support.",
             fontsize=5.1, color=MUTED)
    fig.tight_layout(rect=(0, 0.12, 1, 0.88), w_pad=1.25)
    out = os.path.join(os.path.dirname(__file__), "figures", "e83_route_reachability.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_route_bundle_pair():
    """Compare global, layer-balanced, and late-layer bundle sampling."""
    by_tag = {}
    for run_tag in ("bundle_control", "bundle_pair", "bundle_pair_lbal", "bundle_pair_late"):
        matches = []
        for path in glob.glob(os.path.join(RES, "e83", "*bundle*.json")):
            result = load(path)
            if result.get("args", {}).get("run_tag") == run_tag:
                matches.append(result)
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one E83 route-bundle result for {run_tag}; found {len(matches)}")
        by_tag[run_tag] = matches[0]

    labels = {"bundle_control": "no pair", "bundle_pair": "global",
              "bundle_pair_lbal": "layer-balanced", "bundle_pair_late": "late-layer-only"}
    colors_by_tag = {"bundle_control": GRAY, "bundle_pair": BLUE,
                     "bundle_pair_lbal": ORANGE, "bundle_pair_late": YELLOW}
    epochs = np.arange(1, len(by_tag["bundle_control"]["curve"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.0))

    for tag in labels:
        rows = by_tag[tag]["curve"]
        acc = [100 * r["event_terminal_accuracy"] for r in rows]
        support = [100 * r["event_support_coverage"][-1] for r in rows]
        axes[0, 0].plot(epochs, acc, marker="o", color=colors_by_tag[tag],
                        label=labels[tag], linewidth=1.6)
        axes[0, 1].plot(epochs, support, marker="o", color=colors_by_tag[tag],
                        label=labels[tag], linewidth=1.6)
    activity_ax = axes[0, 1].twinx()
    for tag, label, color in (("bundle_pair_lbal", "balanced L4 events", ORANGE),
                              ("bundle_pair_late", "late-only L4 events", YELLOW)):
        rows = by_tag[tag]["curve"]
        activity_ax.plot(epochs, [r["spikes_per_utt"][-1] for r in rows],
                         marker="x", linestyle="--", color=color, alpha=0.75, label=label)
    axes[0, 0].axhline(5, color=GRAY, linestyle=":", linewidth=1)
    axes[0, 0].set_ylim(0, 25)
    axes[0, 0].set_ylabel("accuracy (%)")
    axes[0, 0].set_title("A · Held-out SHD accuracy", fontsize=7.5)
    axes[0, 1].set_ylim(0, 115)
    axes[0, 1].set_ylabel("L4 support (%)")
    activity_ax.set_ylim(0, max(100, max(r["spikes_per_utt"][-1]
                                        for tag in labels
                                        for r in by_tag[tag]["curve"]) * 1.12))
    activity_ax.set_ylabel("layer-4 events / utterance", color=ORANGE)
    activity_ax.tick_params(axis="y", labelcolor=ORANGE)
    axes[0, 1].set_title("B · Deep support and event rate", fontsize=7.5)
    for ax in axes[0]:
        ax.set_xlabel("epoch")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.2)
        if ax is axes[0, 1]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = activity_ax.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=4.6, loc="best", ncol=2)
        else:
            ax.legend(fontsize=5.5, loc="best")

    for tag, color in (("bundle_pair", BLUE), ("bundle_pair_lbal", ORANGE),
                       ("bundle_pair_late", YELLOW)):
        rows = by_tag[tag]["curve"]
        sampled = [r["counterfactual_route_pairs_shadowed_per_epoch"] > 0 for r in rows]
        helpful = [r["counterfactual_pair_fraction_joint_opening_improves"] if has_sample else np.nan
                   for r, has_sample in zip(rows, sampled)]
        synergistic = [r["counterfactual_pair_fraction_synergistic_gamma_negative"] if has_sample else np.nan
                       for r, has_sample in zip(rows, sampled)]
        axes[1, 0].plot(epochs, helpful, marker="o", color=color,
                        label=f"{labels[tag]} · joint helps")
        if tag != "bundle_pair_late":
            axes[1, 0].plot(epochs, synergistic, marker="x", linestyle="--", color=color,
                            label=f"{labels[tag]} · Γ < 0")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xticks(epochs)
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("pair fraction")
    axes[1, 0].set_title("C · Measured pair utility", fontsize=7.5)
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=5.0, loc="upper right")

    pair_tags = ("bundle_pair", "bundle_pair_lbal", "bundle_pair_late")
    pair_labels = ("global", "layer-balanced", "late-only")
    pair_colors = (BLUE, ORANGE, YELLOW)
    x = np.arange(4)
    width = 0.23
    for offset, tag, label, color in zip((-width, 0, width), pair_tags, pair_labels, pair_colors):
        rows = by_tag[tag]["curve"]
        counts = np.sum([r["counterfactual_pair_shadow_counts_by_layer"] for r in rows], axis=0)
        bars = axes[1, 1].bar(x + offset, counts, width, color=color, label=label)
        axes[1, 1].bar_label(bars, fmt="%.0f", padding=1, fontsize=5.2)
    axes[1, 1].set_xticks(x, ["L1", "L2", "L3", "L4"])
    axes[1, 1].set_ylabel("pairs")
    axes[1, 1].set_title("D · Where route credit went", fontsize=7.5)
    axes[1, 1].legend(fontsize=5.5)

    fig.suptitle("E83 · route-pair density collapses with depth and training",
                 x=0.02, ha="left", fontsize=8.8, fontweight="bold")
    fig.text(0.02, 0.033,
             "Matched seed-6 D4: 120 training examples, 120 updates, one pair budget per batch; 128 held-out examples.",
             fontsize=5.0, color=MUTED)
    fig.text(0.02, 0.012,
             "Balanced: 100% L4 support but collapsed class output. Late-only: 6/128; only 9 deep pairs in epoch 1, "
             "then none. Global: 14/128 vs control 8/128 (p=.180).",
             fontsize=5.0, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91), w_pad=2.0, h_pad=1.3)
    out = os.path.join(os.path.dirname(__file__), "figures", "e83_route_bundle_pair.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_route_cost_audit():
    """Visualize validation utility and downstream work from matched pair shadows."""
    result = load(os.path.join(RES, "e83", "route_cost_audit_pair_lbal.json"))
    rows = result["summary_by_layer"]
    layers = np.arange(1, len(rows) + 1)
    n = np.asarray([row["sampled_pairs"] for row in rows], dtype=np.float64)
    helpful = np.asarray([
        row["fraction_joint_opening_improves_class_loss"] for row in rows
    ], dtype=np.float64)
    # Wilson 95% intervals keep the uncertainty visible at these small n.
    z = 1.96
    denom = 1.0 + z * z / np.maximum(n, 1)
    center = (helpful + z * z / (2 * np.maximum(n, 1))) / denom
    half = z * np.sqrt(
        helpful * (1 - helpful) / np.maximum(n, 1)
        + z * z / (4 * np.maximum(n, 1) ** 2)
    ) / denom
    lower = np.maximum(0, center - half)
    upper = np.minimum(1, center + half)
    # The compact JSON summary stores means; use the pair records for medians
    # because early-layer mean deltas have a small number of large outliers.
    pairs = result["pairs"]
    median_delta = np.asarray([
        np.median([pair["joint_class_loss_delta"] for pair in pairs
                   if pair["layer"] == layer])
        for layer in layers
    ], dtype=np.float64)
    l4_spikes = np.asarray([
        row["mean_joint_work_delta_per_example"]["hidden_spikes_by_layer"][-1]
        for row in rows
    ], dtype=np.float64)
    l4_readout = np.asarray([
        row["mean_joint_work_delta_per_example"]["readout_edge_updates_by_layer"][-1]
        for row in rows
    ], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.85))
    colors = [GRAY, BLUE, AQUA, ORANGE]
    axes[0].bar(layers, helpful, color=colors, width=0.68)
    axes[0].errorbar(layers, helpful,
                     yerr=np.vstack((helpful - lower, upper - helpful)),
                     fmt="none", ecolor=INK, capsize=2, linewidth=0.8)
    for layer, value, count in zip(layers, helpful, n.astype(int)):
        axes[0].text(layer, value + 0.04, f"{100 * value:.0f}%\nn={count}",
                     ha="center", va="bottom", fontsize=5.3)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("pair fraction")
    axes[0].set_title("A · Joint opening helps", fontsize=7.2)

    axes[1].axhline(0, color=GRAY, linestyle=":", linewidth=1)
    axes[1].bar(layers, median_delta, color=colors, width=0.68)
    axes[1].set_ylabel("median Δ prefix loss")
    axes[1].set_title("B · Matched loss change", fontsize=7.2)

    width = 0.34
    axes[2].bar(layers - width / 2, l4_spikes, width, color=BLUE,
                label="L4 spikes")
    axes[2].bar(layers + width / 2, l4_readout, width, color=ORANGE,
                label="L4 readout updates")
    axes[2].set_ylabel("added work / example")
    axes[2].set_title("C · Downstream activity", fontsize=7.2)
    axes[2].legend(fontsize=4.8, loc="upper right")

    for ax in axes:
        ax.set_xlabel("source layer")
        ax.set_xticks(layers, [f"L{k}" for k in layers])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("E83 · utility and event growth vary by route depth",
                 x=0.02, ha="left", fontsize=8.7, fontweight="bold")
    fig.text(0.02, 0.025,
             "Frozen balanced checkpoint; 128 speaker-held-out examples, 115 pair shadows, no updates. "
             "Primitive event counts are work proxies, not energy measurements.",
             fontsize=5.0, color=MUTED)
    fig.tight_layout(rect=(0, 0.12, 1, 0.88), w_pad=1.1)
    out = os.path.join(os.path.dirname(__file__), "figures", "e83_route_cost_audit.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_route_option_value():
    """Show how different counterfactual actions expose deep option value."""
    result_paths = [
        os.path.join(RES, "e83", "route_option_value_moe_top2_tree_paired_band0p5.json"),
        os.path.join(RES, "e83", "route_option_value_moe_top2_tree_paired_band5.json"),
        os.path.join(RES, "e83", "route_option_value_moe_top2_tree_birth_band0p5.json"),
        os.path.join(RES, "e83", "route_option_value_moe_top2_tree_spike_band0p5.json"),
        os.path.join(RES, "e83", "route_option_value_moe_top2_tree_all_band0p5.json"),
    ]
    results = [load(path) for path in result_paths]
    replacement, _, birth, spike, combined = results
    layers = np.arange(1, replacement["depth"] + 1)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.55))

    proposal_results = [replacement, birth, spike, combined]
    proposal_names = ["route swap", "route birth", "spike birth", "combined"]
    proposal_colors = [BLUE, ORANGE, AQUA, YELLOW]
    width = 0.18
    for idx, (result, name, color) in enumerate(zip(
            proposal_results, proposal_names, proposal_colors)):
        counts = np.asarray(result["near_boundary_candidates_seen_by_layer"],
                            dtype=np.float64)
        x = layers + (idx - 1.5) * width
        axes[0, 0].bar(x, np.log10(counts + 1), width=width, color=color,
                       label=name)
        for xpos, count in zip(x, counts.astype(int)):
            if count:
                label = f"{count / 1000:.1f}k" if count >= 1000 else f"{count:,}"
                axes[0, 0].text(xpos, np.log10(count + 1) + 0.07,
                                label, ha="center", va="bottom",
                                fontsize=4.7, rotation=35 if count >= 1000 else 0)
    axes[0, 0].set_xticks(layers, [f"L{k}" for k in layers])
    axes[0, 0].set_ylabel("log10(1 + candidate visits)")
    axes[0, 0].set_title("A · Candidate support by action family", fontsize=7.4)
    axes[0, 0].legend(fontsize=4.8, ncol=2)

    for result, name, color in ((spike, "spike only", AQUA),
                                (combined, "combined pool", YELLOW)):
        branches = [leaf for tree in result["trees"] for leaf in tree["leaves"]
                    if leaf["swap_count"] > 0]
        fractions = [100 * np.mean([
            row["hidden_event_delta_by_layer"][i] > 0 for row in branches])
                     if branches else 0.0 for i in range(len(layers))]
        axes[0, 1].plot(layers, fractions, marker="o", linewidth=1.1,
                        color=color, label=f"{name} (n={len(branches)})")
    axes[0, 1].set_xticks(layers, [f"L{k}" for k in layers])
    axes[0, 1].set_ylim(-3, 65)
    axes[0, 1].set_ylabel("counterfactual leaves adding events (%)")
    axes[0, 1].set_title("B · Verified event cascades", fontsize=7.4)
    axes[0, 1].legend(fontsize=4.9)

    all_branches = [leaf for tree in combined["trees"] for leaf in tree["leaves"]
                    if leaf["swap_count"] > 0]
    for has_route, name, color, marker in (
            (False, "spike-only path", AQUA, "o"),
            (True, "path includes route swap", ORANGE, "s")):
        rows = [row for row in all_branches
                if any(action["mechanism"] == "replacement"
                       for action in row["path"]) == has_route]
        axes[1, 0].scatter(
            [row["immediate_advantage_vs_factual"] for row in rows],
            [row["learning_option_advantage"] for row in rows],
            s=18, alpha=0.7, color=color, marker=marker,
            label=f"{name} (n={len(rows)})")
    axes[1, 0].axhline(0, color=GRAY, linewidth=0.8, linestyle=":")
    axes[1, 0].axvline(0, color=GRAY, linewidth=0.8, linestyle=":")
    axes[1, 0].set_xlabel("immediate deepest-head loss advantage")
    axes[1, 0].set_ylabel("matched suffix-step progress advantage")
    axes[1, 0].set_title("C · Immediate loss vs suffix-step progress", fontsize=7.3)
    axes[1, 0].legend(fontsize=4.7)

    weights = ["0.0", "1.0", "10.0", "100.0"]
    positive_counts = [sum(tree["backup_value_by_learning_weight"][w] > 0
                           for tree in combined["trees"]) for w in weights]
    axes[1, 1].bar(np.arange(len(weights)), positive_counts, color=[BLUE, BLUE, BLUE, ORANGE])
    for i, count in enumerate(positive_counts):
        axes[1, 1].text(i, count + 0.12, f"{count}/8", ha="center", fontsize=5.6)
    axes[1, 1].set_xticks(np.arange(len(weights)), ["0", "1", "10", "100"])
    axes[1, 1].set_ylim(0, 3)
    axes[1, 1].set_xlabel("learning-option weight λ")
    axes[1, 1].set_ylabel("error trees with positive scalar value")
    axes[1, 1].set_title("D · Scalar backup is weight-sensitive", fontsize=7.4)
    axes[1, 1].text(0.03, 0.97, "the second positive tree flips only at λ = 100",
                    transform=axes[1, 1].transAxes, va="top", fontsize=4.8,
                    color=MUTED)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("E83 · scalar optionality across route and spike counterfactuals",
                 x=0.02, ha="left", fontsize=8.7, fontweight="bold")
    fig.text(0.02, 0.015,
             "Frozen seed-6 top-2 checkpoint; 8 error-conditioned held-out-speaker examples. "
             "One clipped suffix-SGD step; no persistent updates. Candidate counts repeat across visited states; λ is uncalibrated.",
             fontsize=4.9, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91), h_pad=1.35, w_pad=1.0)
    out = os.path.join(os.path.dirname(__file__), "figures", "e83_route_option_value.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_spike_option_training():
    """Compare trained scalar threshold-option arms on SHD."""
    specs = [
        ("*spike_option_pathwise_control*spk_s6.json", "pathwise", GRAY),
        ("*spike_option_immediate*spk_s6.json", "immediate", BLUE),
        ("*spike_option_scalar_lambda10*spk_s6.json", "scalar λ=10", AQUA),
    ]
    rows = []
    for pattern, label, color in specs:
        matches = glob.glob(os.path.join(RES, "e83", pattern))
        if not matches:
            raise FileNotFoundError(f"missing E83 spike-option result: {pattern}")
        result = load(matches[0])
        rows.append((label, color, result["curve"]))

    fig = plt.figure(figsize=(7.1, 4.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                            width_ratios=[1.0, 1.25], hspace=0.48, wspace=0.32)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]
    x = np.arange(len(rows))
    width = 0.28
    for idx, (label, color, curve) in enumerate(rows):
        accuracy = [100 * row["event_terminal_accuracy"] for row in curve]
        axes[0].bar(idx - width / 2, accuracy[0], width, facecolor="white",
                    edgecolor=color, linewidth=1.2,
                    label="epoch 1" if idx == 0 else None)
        axes[0].bar(idx + width / 2, accuracy[-1], width, color=color,
                    label="epoch 2" if idx == 0 else None)
        axes[0].text(idx + width / 2, accuracy[-1] + 0.3, "6/128",
                     ha="center", va="bottom", fontsize=5.0, color=INK)
    axes[0].axhline(5, color=ORANGE, linewidth=1.0, linestyle="--",
                    label="5% chance")
    axes[0].set_xticks(x, ["control", "immediate", "λ=10"])
    axes[0].set_ylim(0, 9)
    axes[0].set_ylabel("accuracy (%)", fontsize=6)
    axes[0].set_title("A · No recognition gain", fontsize=7.5)
    axes[0].tick_params(axis="x", labelsize=5.7)
    axes[0].legend(fontsize=4.6, ncol=2, loc="upper right")

    offsets = [-0.23, 0.0, 0.23]
    for offset, (label, color, curve) in zip(offsets, rows):
        coverage = np.asarray(curve[-1]["event_support_coverage"], dtype=float) * 100
        axes[1].bar(np.arange(len(coverage)) + offset, coverage, width=0.22,
                    color=color, label=label)
        if len(coverage) >= 4:
            axes[1].text(3 + offset, coverage[3] + 1.2,
                         f"{coverage[3]:.2f}%", ha="center", fontsize=4.8,
                         color=color, rotation=45)
    axes[1].set_xticks(np.arange(4), ["L1", "L2", "L3", "L4"])
    axes[1].set_ylim(0, 112)
    axes[1].set_ylabel("examples with an event (%)", fontsize=6)
    axes[1].set_title("B · L2 moves; L4 stays scarce", fontsize=7.5)
    axes[1].legend(fontsize=4.6, ncol=1, loc="upper right")
    axes[1].tick_params(axis="x", labelsize=5.7)

    for offset, (label, color, curve) in zip(offsets, rows):
        norms = np.asarray(curve[-1]["layer_grad_norms"], dtype=float)
        plotted = np.maximum(norms, 1e-6)
        axes[2].bar(np.arange(len(norms)) + offset, plotted, width=0.22,
                    color=color, label=label)
        for li, norm in enumerate(norms):
            if norm == 0:
                axes[2].text(li + offset, 1.3e-6, "0", ha="center",
                             va="bottom", fontsize=4.4, color=color)
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-6, 3)
    axes[2].set_xticks(np.arange(4), ["L1", "L2", "L3", "L4"])
    axes[2].set_ylabel("pathwise gradient norm (log scale)", fontsize=6)
    axes[2].set_title("C · Credit vanishes at L3/L4", fontsize=7.5)
    axes[2].legend(fontsize=4.8, ncol=3, loc="upper right")
    axes[2].tick_params(axis="x", labelsize=6)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("E83 · scalar spike-option training pilot", x=0.02,
                 ha="left", fontsize=8.7, fontweight="bold")
    fig.text(0.02, 0.015,
             "Seed 6; depth 4; 120 train / 128 held-out speakers; deepest readout; two epochs. "
             "Each arm ended at 6/128 accuracy; race coverage was zero. Zero L4 norms are plotted at the axis floor. One seed.",
             fontsize=4.9, color=MUTED)
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.1, right=0.99)
    out = os.path.join(os.path.dirname(__file__), "figures",
                       "e83_spike_option_training.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_optionality_state_value():
    """Compare six-epoch immediate and state-value route-credit runs."""
    patterns = {
        "immediate": ("*optionality_immediate_6ep*spk_s6.json",
                      "optionality_immediate_6ep"),
        "continuation-aware": ("*optionality_rollout_aware_6ep*spk_s6.json",
                               "optionality_rollout_aware_6ep"),
    }
    results = {}
    for label, (pattern, run_tag) in patterns.items():
        candidates = glob.glob(os.path.join(RES, "e83", pattern))
        matches = [path for path in candidates
                   if load(path).get("args", {}).get("run_tag") == run_tag]
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one E83 optionality result for {label}: {pattern}")
        results[label] = load(matches[0])

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.85),
                             gridspec_kw={"width_ratios": [0.9, 1.25, 1.1]})
    epochs = np.arange(1, 7)
    colors_by_arm = {"immediate": BLUE, "continuation-aware": AQUA}
    for label, result in results.items():
        accuracy = [100 * row["event_terminal_accuracy"] for row in result["curve"]]
        axes[0].plot(epochs, accuracy, marker="o", linewidth=1.2,
                     color=colors_by_arm[label],
                     label=label)
    axes[0].axhline(5, color=ORANGE, linestyle="--", linewidth=1,
                    label="20-class chance")
    axes[0].set_xticks(epochs)
    axes[0].set_ylim(0, 7)
    axes[0].set_xlim(0.75, 6.25)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("A · Chance through epoch 6")
    axes[0].legend(fontsize=5.4, loc="lower right")

    layers = np.arange(4)
    width = 0.36
    for offset, label in zip((-width / 2, width / 2), patterns):
        coverage = (100 * np.asarray(results[label]["curve"][-1]["event_support_coverage"],
                                     dtype=float))
        axes[1].bar(layers + offset, coverage, width=width,
                    color=colors_by_arm[label], label=label)
    axes[1].set_xticks(layers, ["L1", "L2", "L3", "L4"])
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("held-out support (%)")
    axes[1].set_title("B · Final support remains shallow")
    axes[1].legend(fontsize=5.4, loc="upper right")

    aware_curve = results["continuation-aware"]["curve"]
    parent_mass = [100 * row["spike_option_mean_parent_beneficial_continuation_mass"]
                   for row in aware_curve]
    child_mass = [100 * row["spike_option_mean_child_beneficial_continuation_mass"]
                  for row in aware_curve]
    axes[2].plot(epochs, parent_mass, marker="o", linewidth=1.2,
                 color=BLUE, label="parent state")
    axes[2].plot(epochs, child_mass, marker="o", linewidth=1.2,
                 color=AQUA, label="child state")
    axes[2].set_xticks(epochs)
    axes[2].set_xlim(0.75, 6.25)
    axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel("sampled helpful futures (%)")
    axes[2].set_title("C · Few futures clear the loss cutoff")
    axes[2].legend(fontsize=5.4, loc="upper left")

    fig.suptitle("E83 · six-epoch state-conditioned optionality check",
                 x=0.02, ha="left", fontsize=8.7, fontweight="bold")
    fig.text(0.02, 0.005,
             "Seed 6; depth 4; 120 train / 128 held-out speakers; six epochs; two future rollouts/action. "
             "One seed; rollout reserve is proposal-conditioned.",
             fontsize=4.8, color=MUTED)
    fig.subplots_adjust(top=0.79, bottom=0.2, left=0.08, right=0.985, wspace=0.55)
    out = os.path.join(os.path.dirname(__file__), "figures",
                       "e83_optionality_state_value.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_pair_occupancy():
    """Show that wider pair windows do not restore deep proposal support."""
    result = load(os.path.join(RES, "e83", "route_pair_occupancy_late_seed6.json"))
    windows = np.asarray(result["windows_ms"], dtype=np.float64)
    counts = np.asarray([
        result["pair_candidates_by_window_layer"][str(float(w))]
        for w in windows
    ], dtype=np.float64)
    batches = np.asarray([
        result["batches_with_pair_by_window_layer"][str(float(w))]
        for w in windows
    ], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    axes[0].plot(windows, counts[:, 0] / 1000.0, marker="o", color=GRAY)
    axes[0].set_xscale("log")
    axes[0].set_xticks(windows, [f"{int(w)}" for w in windows])
    axes[0].set_xlabel("pair arrival window (ms)")
    axes[0].set_ylabel("candidate pairs (thousands)")
    axes[0].set_title("A · L1 already dominates", fontsize=7.5)

    for layer, color in ((1, BLUE), (2, AQUA), (3, ORANGE)):
        axes[1].plot(windows, counts[:, layer], marker="o", color=color,
                     label=f"L{layer + 1} candidates")
    axes[1].set_xscale("log")
    axes[1].set_xticks(windows, [f"{int(w)}" for w in windows])
    axes[1].set_xlabel("pair arrival window (ms)")
    axes[1].set_ylabel("candidate pairs across 30 batches")
    axes[1].set_ylim(-0.35, max(7, float(counts[:, 1:].max()) + 1))
    axes[1].set_yticks(range(0, int(axes[1].get_ylim()[1]) + 1))
    axes[1].set_title("B · Deep pairs remain scarce", fontsize=7.5)
    axes[1].legend(fontsize=5.5, loc="upper left")
    for layer, color in ((1, BLUE), (2, AQUA), (3, ORANGE)):
        for idx, value in enumerate(counts[:, layer]):
            if value > 0:
                axes[1].annotate(str(int(value)), (windows[idx], value),
                                 xytext=(0, 4), textcoords="offset points",
                                 ha="center", fontsize=5.2, color=color)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("E83 · time-window widening does not create deep pair support",
                 x=0.02, ha="left", fontsize=8.7, fontweight="bold")
    fig.text(0.02, 0.02,
             "Frozen final late-only checkpoint on its 120-example fit subset. At 1,000 ms: L2=6, L3=0, L4=1; "
             "only 6/30 batches had any L2 pair and 1/30 had an L4 pair. Candidate counts are not utility estimates.",
             fontsize=4.9, color=MUTED)
    fig.tight_layout(rect=(0, 0.14, 1, 0.88), w_pad=1.4)
    out = os.path.join(os.path.dirname(__file__), "figures", "e83_route_pair_occupancy.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_spike_boundary_late():
    """Show the shallow readout shortcut in all-depth spike-boundary utility."""
    result_paths = {
        ("control", "all_depths"): os.path.join(RES, "e83", "spike_boundary_audit_bundle_control_refined_s6_n128.json"),
        ("late-only", "all_depths"): os.path.join(RES, "e83", "spike_boundary_audit_late_only_s6_n128.json"),
        ("control", "deepest"): os.path.join(RES, "e83", "spike_boundary_audit_bundle_control_deepest_s6_n128.json"),
        ("late-only", "deepest"): os.path.join(RES, "e83", "spike_boundary_audit_late_only_deepest_s6_n128.json"),
    }
    results = {tag: load(path) for tag, path in result_paths.items()}
    rows = {tag: result["paired_shadows"] for tag, result in results.items()}
    layers = np.arange(1, len(rows[("control", "all_depths")]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.9))
    tags = ("control", "late-only")
    labels = {"control": "control", "late-only": "late-only pair"}
    colors = {"control": GRAY, "late-only": YELLOW}
    width = 0.34

    for offset, tag in zip((-width / 2, width / 2), tags):
        counts = []
        for layer_rows in rows[(tag, "all_depths")]:
            valid = [r for r in layer_rows
                     if r["within_spike_band"] and not r["refractory_blocked"]]
            n = len(valid)
            counts.append(n / 32.0)
        axes[0].bar(layers + offset, counts, width, color=colors[tag], label=labels[tag])
        for ax_idx, fusion in ((1, "all_depths"), (2, "deepest")):
            means, mean_errs = [], []
            for layer_rows in rows[(tag, fusion)]:
                valid = [r for r in layer_rows
                         if r["within_spike_band"] and not r["refractory_blocked"]]
                delta = np.asarray([r["main_L_on_minus_L_off"] for r in valid], dtype=float)
                n = len(delta)
                means.append(float(delta.mean()) if n else np.nan)
                mean_errs.append(1.96 * float(delta.std(ddof=1)) / np.sqrt(n)
                                 if n > 1 else 0.0)
            means = np.asarray(means)
            axes[ax_idx].bar(layers + offset, means, width, color=colors[tag], label=labels[tag])
            axes[ax_idx].errorbar(layers + offset, means, yerr=mean_errs,
                                  fmt="none", ecolor=INK, capsize=2, linewidth=0.7)

    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("valid candidates / 32 batches")
    axes[0].set_title("A · Valid boundary candidates", fontsize=7.3)
    for idx, title in ((1, "B · All-depth answer loss"), (2, "C · Deepest-only answer loss")):
        axes[idx].axhline(0, color=GRAY, linestyle=":", linewidth=1)
        axes[idx].set_ylabel("mean main $L_{on}-L_{off}$")
        axes[idx].set_title(title, fontsize=7.3)
    for ax in axes:
        ax.set_xlabel("event layer")
        ax.set_xticks(layers, [f"L{k}" for k in layers])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=5.0, loc="upper right")
    axes[1].legend(fontsize=5.0, loc="upper right")
    axes[2].legend(fontsize=5.0, loc="upper right")
    fig.suptitle("E83 · all-depth readout exposes a shallow shortcut, not deep credit",
                 x=0.02, ha="left", fontsize=8.4, fontweight="bold")
    fig.text(0.02, 0.015,
             "Same checkpoints, examples, valid nonrefractory candidates. Late-only L1: all-depth ΔL=−0.00936, "
             "deepest-only ΔL=0. L1 toggles changed no downstream hidden spikes; only the L1 readout edges changed.",
             fontsize=4.6, color=MUTED)
    fig.tight_layout(rect=(0, 0.12, 1, 0.88), w_pad=1.1)
    out = os.path.join(os.path.dirname(__file__), "figures", "e83_spike_boundary_late.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_spike_pair_audit():
    """Separate downstream event propagation from pair-specific loss credit."""
    result_paths = [
        ("L2 control", "spike_pair_audit_L2_bundle_control_s6_n1024.json", GRAY),
        ("L2 late-only", "spike_pair_audit_L2_late_only_s6_n1024.json", YELLOW),
        ("L3 control", "spike_pair_audit_L3_bundle_control_s6_n1024.json", GRAY),
        ("L3 late-only", "spike_pair_audit_L3_late_only_s6_n1024.json", YELLOW),
    ]
    results = [load(os.path.join(RES, "e83", filename)) for _, filename, _ in result_paths]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.95))
    labels = [name.replace(" ", "\n") for name, _, _ in result_paths]
    labels_with_n = [f"{name.replace(' ', chr(10))}\nn={n}" for
                     (name, _, _), n in zip(result_paths, [64, 34, 10, 5])]
    colors_by_arm = [color for _, _, color in result_paths]

    # Panel A: available paired spike interventions by layer/checkpoint.
    eligible = [result["pairs_attempted"] for result in results]
    natural_off = [sum(row.get("pair_found", False) and row["natural_state"] == [0, 0]
                       for row in result["per_batch"]) for result in results]
    x = np.arange(len(results))
    axes[0].bar(x, eligible, color=colors_by_arm, width=0.62)
    for i, (n, n00) in enumerate(zip(eligible, natural_off)):
        axes[0].text(i, n + 3, f"{n}\n00={n00}", ha="center", va="bottom", fontsize=5.4)
    axes[0].set_ylim(0, max(eligible) * 1.25)
    axes[0].set_ylabel("selected pairs / 256 batches")
    axes[0].set_title("A · Pair availability", fontsize=7.2)

    # Panel B: outcomes conditional on both natural events being absent.
    propagation, deep_help = [], []
    pair_only_help = 0
    for result in results:
        rows = [row for row in result["per_batch"]
                if row.get("pair_found", False) and row["natural_state"] == [0, 0]]
        propagated = helped = 0
        for row in rows:
            work = row["selected_example_work_by_corner"]
            before = np.asarray(work["00"]["hidden_spikes"], dtype=float)
            after = np.asarray(work["11"]["hidden_spikes"], dtype=float)
            layer = int(row["layer"]) - 1
            propagated += bool(np.sum(after[layer + 1:] - before[layer + 1:]) > 0)
            loss = row["deepest"]
            joint = loss["L11"] - loss["L00"]
            singleton = (loss["L10"] - loss["L00"], loss["L01"] - loss["L00"])
            helped += joint < -1e-12
            pair_only_help += (joint < -1e-12 and singleton[0] >= -1e-12
                               and singleton[1] >= -1e-12)
        propagation.append(propagated / len(rows) if rows else np.nan)
        deep_help.append(helped / len(rows) if rows else np.nan)
    width = 0.34
    axes[1].bar(x - width / 2, propagation, width, color=BLUE, label="suffix spikes increase")
    axes[1].bar(x + width / 2, deep_help, width, color=AQUA, label="deep loss improves")
    axes[1].set_ylim(0, 1.12)
    axes[1].set_ylabel("fraction of natural-off pairs")
    axes[1].set_title("B · Opening both events", fontsize=7.2)
    axes[1].legend(fontsize=5.1, loc="upper left")

    # Panel C: difference-in-differences, not the raw joint intervention delta.
    interaction_counts = [sum(abs(row["deepest"]["interaction_gamma"]) > 0.01
                              for row in result["per_batch"] if row.get("pair_found", False))
                          for result in results]
    totals = [result["pairs_attempted"] for result in results]
    axes[2].bar(x, interaction_counts, color=colors_by_arm, width=0.62)
    for i, (n, total) in enumerate(zip(interaction_counts, totals)):
        axes[2].text(i, n + 0.08, f"{n}/{total}", ha="center", va="bottom", fontsize=5.4)
    axes[2].set_ylim(0, max(interaction_counts, default=0) + 1.25)
    axes[2].set_ylabel(r"pairs with $|\Gamma|>0.01$")
    axes[2].set_title("C · Pair-only interaction", fontsize=7.2)
    axes[2].text(0.5, 0.88, f"joint-only helpful: {pair_only_help}/113",
                 ha="center", va="center", transform=axes[2].transAxes, fontsize=5.6,
                 color=MUTED)

    for i, ax in enumerate(axes):
        ax.set_xticks(x, labels_with_n if i == 1 else labels, fontsize=5.4)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("E83 · downstream spikes are commoner than pair-specific class credit",
                 x=0.02, ha="left", fontsize=8.3, fontweight="bold")
    fig.text(0.02, 0.015,
             "Frozen held-out-speaker checkpoints; one near-boundary pair per batch. Distinct hidden units within 50 ms; "
             "no shared-receiver condition. Validation diagnostic, not a trainability result.",
             fontsize=4.6, color=MUTED)
    fig.tight_layout(rect=(0, 0.14, 1, 0.87), w_pad=1.2)
    out = os.path.join(os.path.dirname(__file__), "figures", "e83_spike_pair_audit.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e77_bootstrap_diagnostics():
    """Show exact rate calibration's effect and the depth-8 gradient-reach smoke."""
    e77_dir = os.path.join(RES, "e77")
    quantile = load(os.path.join(
        e77_dir, "tvlm_D10000_p0.5_r1_M128-128-64_depth4_c0_eh1_ek0_s77_boot0.1_cb4.json"))
    rate_match = load(os.path.join(
        e77_dir, "tvlm_D10000_p0.5_r1_M128-128-64_depth4_c0_eh1_ek0_s77_boot0.1_cb4_cs6.json"))
    depth4 = load(os.path.join(
        e77_dir, "tvlm_D4096_p0.25_r1_M8-8-8_depth4_c0_eh1_ek0_s77_cf4_b0.5_sg0.25_w1_dl5_lr0_gc1_boot0.1_cb4_cs6.json"))
    depth8 = load(os.path.join(
        e77_dir, "tvlm_D4096_p0.25_r1_M8-8-8_depth8_c0_eh1_ek0_s77_boot0.1_cb4_cs6.json"))

    def active_fraction(result, layer):
        curve = result["valid_curve"]
        return 100 * np.mean([row["event_layer_grad_norms"][layer] > 0 for row in curve])

    layers4 = np.arange(1, 5)
    quantile_active = [active_fraction(quantile, i) for i in range(4)]
    matched_active = [active_fraction(rate_match, i) for i in range(4)]
    quantile_spikes = np.asarray(quantile["test_work"]["spikes_per_char"][:4], dtype=float)
    matched_spikes = np.asarray(rate_match["test_work"]["spikes_per_char"][:4], dtype=float)
    depth4_active = [active_fraction(depth4, i) for i in range(4)]
    depth8_active = [active_fraction(depth8, i) for i in range(8)]

    fig, (ax_grad, ax_spikes, ax_depth) = plt.subplots(1, 3, figsize=(10.0, 3.2))
    width = 0.34
    ax_grad.bar(layers4 - width / 2, quantile_active, width, color=GRAY, label="raw voltage quantile")
    ax_grad.bar(layers4 + width / 2, matched_active, width, color=BLUE, label="exact reset-rate match")
    ax_grad.set_xticks(layers4)
    ax_grad.set_ylim(0, 108)
    ax_grad.set_ylabel("validation checkpoints with gradient (%)")
    ax_grad.set_xlabel("event layer")
    ax_grad.set_title("A · Default width, depth 4")

    ax_spikes.bar(layers4 - width / 2, quantile_spikes, width, color=GRAY)
    ax_spikes.bar(layers4 + width / 2, matched_spikes, width, color=BLUE)
    ax_spikes.axhline(0.1, color=INK, ls=":", lw=1)
    ax_spikes.set_xticks(layers4)
    ax_spikes.set_ylim(0, 0.9)
    ax_spikes.set_ylabel("selected-checkpoint test spikes / char")
    ax_spikes.set_xlabel("event layer")
    ax_spikes.set_title("B · Activity after learning")

    layers8 = np.arange(1, 9)
    ax_depth.bar(layers8[:4] - width / 2, depth4_active, width, color=GRAY, label="depth 4")
    ax_depth.bar(layers8 + width / 2, depth8_active, width, color=AQUA, label="depth 8")
    ax_depth.set_xticks(layers8)
    ax_depth.set_ylim(0, 108)
    ax_depth.set_ylabel("validation checkpoints with gradient (%)")
    ax_depth.set_xlabel("event layer")
    ax_depth.set_title("C · Beyond four layers")
    ax_depth.legend(fontsize=6.3)

    fig.suptitle("E77 · matching realized firing carries gradients through eight event layers",
                 x=0.02, ha="left", fontsize=9.2, fontweight="bold")
    fig.legend(handles=[
        matplotlib.patches.Patch(color=GRAY, label="raw voltage quantile / depth-4 reference"),
        matplotlib.patches.Patch(color=BLUE, label="exact reset-rate match"),
        matplotlib.patches.Patch(color=AQUA, label="depth-8 rate match"),
        Line2D([0], [0], color=INK, ls=":", lw=1, label="initial target 0.1 spikes / char / layer"),
    ], loc="upper center", bbox_to_anchor=(0.52, 0.89), ncol=4, fontsize=6.1,
       frameon=False, handlelength=1.4, columnspacing=0.9)
    fig.text(0.02, 0.005,
             "One seed. A/B: default width, 10k train characters; 39 updates; 1k test characters. "
             "C: width 8, 4,096 train characters, 16 updates, 512 test characters. "
             "Gradient reach only; BPC is too small-sample for a quality claim.", fontsize=5.8, color=MUTED)
    fig.tight_layout(rect=(0.01, 0.12, 0.99, 0.82), w_pad=1.3)
    out = os.path.join(os.path.dirname(__file__), "figures", "e77_depth_trainability_bootstrap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
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
    for long, rel, col, name in ((False, 0, YELLOW, "event-token Transformer, 200k episodes"),
                                 (True, 0, BLUE, "event-token Transformer, 2M episodes"),
                                 (True, 1, AQUA, "Transformer + relative-time bias, 1M episodes")):
        v = [(np.mean([m for _, m in vv]), np.mean([a for a, _ in vv])) for k, vv in tf.items()
             if k[0] == long and k[3] == rel]
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


def fig_phase():
    """E37 phase diagram: mean test accuracy over training fraction × sleep strength (3 seeds; min–max annotated)."""
    cells = {}
    for path in glob.glob(os.path.join(RES, "e37", "p31_add_f*_lam*_r1_sig2_pd.json")):
        d = load(path); v = [r["final"]["test"] for r in d["rows"]]
        cells[(d["args"]["frac"], d["args"]["lam"])] = v
    if not cells:
        return None
    fr = sorted({k[0] for k in cells}); la = sorted({k[1] for k in cells})
    M = np.array([[np.mean(cells[(f, l)]) for l in la] for f in fr])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#a9c8ef", "#2a78d6", "#174a8c"])
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="lower")
    for i, f in enumerate(fr):
        for j, l in enumerate(la):
            v = cells[(f, l)]
            ax.text(j, i, f"{np.mean(v):.2f}\n{min(v):.2f}–{max(v):.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if np.mean(v) > 0.6 else INK)
    ax.set_xticks(range(len(la))); ax.set_xticklabels([f"{l:g}" for l in la])
    ax.set_yticks(range(len(fr))); ax.set_yticklabels([f"{int(f * 100)}%" for f in fr])
    ax.set_xlabel("sleep strength λ (decay of the lookup per epoch)"); ax.set_ylabel("training pairs")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="test accuracy")
    ax.set_title("Grokking phase diagram, (a + b) mod 31: memorize / grok / collapse")
    return fig


def fig_e41():
    """E41: test accuracy on unseen triples, (a + b + c) mod p, by condition (3 seeds; dots = seeds)."""
    conds = [("30%, sleep", "f0.3_lam0.05"), ("10%, sleep", "f0.1_lam0.05"), ("30%, no sleep", "f0.3_lam0.0"),
             ("30%, lookup only", "c0")]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for k, (p, col) in enumerate(((17, BLUE), (31, ORANGE))):
        for i, (name, key) in enumerate(conds):
            fs = [f for f in glob.glob(os.path.join(RES, "e41", f"p{p}_*.json")) if key in f and "smoke" not in f
                  and "fsweep" not in f and "tlong" not in f]
            if key == "c0":
                fs = [f for f in fs if f.endswith("_c0.json")]
            else:
                fs = [f for f in fs if f.endswith("_c1.json")]
            if not fs:
                continue
            v = [r["final"]["test"] for r in load(fs[0])["rows"]]
            x = i + (k - 0.5) * 0.3
            ax.bar(x, np.mean(v), width=0.28, color=col, label=f"p = {p}" if i == 0 else None)
            ax.scatter([x] * len(v), v, color=INK, s=8, zorder=3)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([c[0] for c in conds], fontsize=7.5)
    ax.set_ylabel("test accuracy, unseen triples"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Grokking with depth: (a + b + c) mod p through a two-stage rhythm chain")
    ax.grid(axis="x", visible=False)
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
    s = [Paragraph("Theory foundations in five principles", st["h1"]),
         Paragraph("The foundational calculus in §§1–20 reduces to five working principles. The remaining derivations "
                   "are organized into eight linked themes in experiments/THEORY.md, with global section numbers. "
                   "Colours give the evidence status of the foundational principles.",
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

    tf_10m_final = load(os.path.join(RES, "e64", "tf_D10000000_s256_L4_p4_dr0.1_v.json"))

    import figures_mech as FM                                   # explanatory figures (plain-language front)
    s = [P("Sleeping Machines: what is known", "title"),
         P(f"Computing in time with races, holds and vetoes · report, {date.today():%d %B %Y}", "sub"),
         P("Frontier signals", "h1"),
         P("Three measured capability leads make a concrete case for this architecture's potential: a real-language lead "
           "on a shared text8 split, locally learned retrieval that generalizes to longer contexts, and deep compositional "
           "networks that learn structured tasks with far less data and counted computation. A new E77 depth-8 pilot adds "
           "a distinct trainability signal: gradients reached all eight event layers.")]
    s += fig(fig_potential_evidence, W)
    s += bullets([
        "<b>Real language:</b> on the same 1M-character text8 training and test split, E79's native race mixture scores "
        "1.808 bpc frozen, versus 2.179 for the completed LSTM and 2.367 for the 2-layer Transformer. It is a strong combined "
        "expert-and-copy-memory result. At 10M, E79 scores 1.613 frozen versus 1.799 for the completed LSTM on the same "
        "test segment, a 0.186 bpc lead. Both comparisons are single-seed; parameter count, training budget, and inference "
        f"work are not matched. The four-layer 10M Transformer checkpoint was selected on validation and scored "
        f"{tf_10m_final['test_bpc']:.4f} held-out test bpc after {tf_10m_final['steps']:,} updates, "
        f"{tf_10m_final['test_bpc'] - 1.7993:.4f} above the LSTM's 1.7993 "
        "test bpc. The Transformer has 3.24M parameters and four passes; the LSTM has 1.20M parameters and six passes. "
        "E79 is a mixture of expert predictors plus copy memory, not a deep hidden event stack; E77 is the separate "
        "deep time-vector language model. A default-width depth-4 pilot showed that a raw voltage quantile left its "
        "deeper layers nearly silent. Matching realized post-reset activity restored gradients in all four layers, and "
        "a width-8 depth-8 pilot reached all eight layers at every validation point. These are trainability checks; "
        "a meaningful E77 language-model result remains pending.",
        "<b>Learned retrieval:</b> on E61's synthetic recall task, local race attention reaches 100% at 4× context after "
        "at most 4,000 examples in all five runs. The best of seven Transformer settings reaches 71.6% after as many as "
        "1M examples.",
        "<b>Depth and composition:</b> on a depth-4 order task, the event model reaches 99.9–100% after 10–15k examples "
        "(5/5 runs); a Transformer reaches 99.0% after 40k examples repeated 50 times and 99.2–99.6% on 2M fresh "
        "examples. On shared-motif composition, it averages 99.65% after one pass at roughly 10,000× lower counted work.",
        "<b>Deep-stack trainability (not yet a supremacy result):</b> exact replay of the spike reset dynamics gave "
        "nonzero gradients in all eight E77 event layers at all 16 validation points. Test activity stayed between "
        "0.075 and 0.217 spikes per character per layer. This was one width-8 seed with 4,096 training and 512 test "
        "characters; its 4.319 BPC is only a micro-pilot diagnostic, not a quality or scaling result.",
        "<b>Deep SHD route analysis (§§138–143):</b> the exact sparse masks have candidate paths from every input band "
        "to layer 4 in both measured seeds, yet only 1.6–7.0% of examples actually reach layer 4 in the equal-update "
        "runs. Sparse skips restore support to 99–100% but do not improve paired accuracy. The analysis shows why: "
        "path existence, event realization, and loss credit are separate; competing routes need paired outcome evaluation, "
        "and two subthreshold messages can require a joint shadow to create a useful spike. A layer-balanced pair run "
        "restored 100% layer-4 support but finished at 6/128 versus 8/128 control (paired p=.791); it emitted 1,647 "
        "layer-4 events per utterance and predicted only two classes. The global pair run reached 14/128 versus 8/128 "
        "(p=.180), still inconclusive. More depth adds possible expert compositions, while both support survival and "
        "event-rate control must be maintained. A frozen validation audit found joint pair openings helpful in "
        "58% of sampled L3 pairs and 50% of L4 pairs, but only 16% of L1 and 31% of L2. L2 pairs added 4.24 L4 "
        "spikes and 10.76 L4 readout updates per example on average; early-layer means are outlier-sensitive. A "
        "matched late-only run produced nine L3/L4 pair shadows in epoch 1 and none in epochs 2–4; final accuracy "
        "was 6/128 (4.69%) with 0.78% L4 support, versus 8/128 (6.25%) for control. Training loss fell "
        "59.78→3.07, but held-out prefix NLL remained 2.985/3.167 across its two time strata. This isolates sparse pair "
        "co-occupancy: occasional deep events remained, but no two near-time closed routes converged on one deep "
        "receiver. The pair estimator had no late proposals to assess for three epochs. The current SHD implementation "
        "scans hidden states on a 1 ms grid; it does not establish sparse asynchronous training efficiency.",
    ], st)
    s += fig(fig_e83_route_reachability, W * 0.96)
    s += fig(fig_e83_route_bundle_pair, W * 0.96)
    s += fig(fig_e83_route_cost_audit, W * 0.96)
    s += fig(fig_e83_route_option_value, W * 0.96)
    s += fig(fig_e83_pair_occupancy, W * 0.96)
    s += fig(fig_e83_spike_boundary_late, W * 0.96)
    s += fig(fig_e83_spike_pair_audit, W * 0.96)
    s += [P("<b>What the SHD intervention teaches:</b> exposing deeper route alternatives can restore activity, but "
            "activity is not trainability: the layer-balanced model's classifier collapsed while event counts grew. "
            "Section 143 derives a constrained route utility that weighs class-loss change against per-layer work. "
            "The frozen audit found late pairs more often helpful, then a matched late-only run found just nine deep "
            "pair proposals in epoch 1 and none afterward; its accuracy was 6/128 versus 8/128 for control. Section 144 "
            "derives why pair availability can vanish quadratically with sparse event flux. A no-update sweep from 25 "
            "to 1,000 ms found just 6 L2, zero L3, and one L4 pair in the 120-example fit subset at the widest "
            "window, so widening time alone cannot restore support. Reweighting cannot repair an empty proposal set. "
            "A refractory-aware audit compared matched control and late-only spike shadows. After filtering to "
            "in-band, nonrefractory events, L1 main-loss spike-on helped 12/22 control cases (mean ΔL=+0.0266) "
            "and 16/21 late-only cases (mean −0.0094, median −0.0020). The auxiliary loss agrees. Only 13 batches "
            "had valid L1 candidates in both arms; the two selected unit/times can differ, and their mean utility "
            "difference was −0.037 (SE 0.035), so this is inconclusive. L2 late-only main-loss mean "
            "ΔL was +0.0040 despite 12/19 improvements; the matched deepest-only replay found zero L1/L2 utility. "
            "A larger four-corner replay found occasional L2/L3 effects on deepest-only loss, but no pair-only helpful "
            "case among 113 natural-off pairs. L2 openings often added downstream spikes without improving class loss. "
            "This supports a topology-conditioned shared-receiver test, not an L1 update against the fused shallow head. "
            "These validation-informed runs are development evidence, not untouched-test "
            "results. Neither pair strategy has established an SHD accuracy gain.", "small"),
         P("<b>Counterfactual optionality through descendants (§§149–150):</b> six bounded audits compared route swaps, "
           "route births, spike births, and a combined proposal pool on eight wrong held-out-speaker examples. The "
           "single-action arms had zero measured class-loss or suffix-learning advantage. In the combined pool, "
           "5/31 verified counterfactual leaves added layer-4 events; three leaves improved both deepest-head loss and "
           "one-step suffix-SGD progress. The recursive scalar value was positive on one of eight error trees at "
           "learning-option weights 0, 1, and 10, and on two only at weight 100. That additional tree's current loss "
           "worsened, so the large weight is not calibrated. This is a small mechanism signal that a sparse cascade can "
           "expose a deep learning option, not a trained update or SHD accuracy gain.", "small"),
         P("<b>Training check (§151):</b> the first local scalar threshold update was tested against pathwise and "
           "immediate-only controls on the same seed, with 120 training examples and 128 held-out speakers. All three "
           "depth-4 arms ended at 6/128 accuracy after two epochs. The scalar arm raised held-out L2 event support to "
           "27.3% from 8.6% in control, while L4 support stayed at 0.78% in every arm. The epoch-2 suffix-learning "
           "advantage was slightly negative. Epoch-2 L3/L4 pathwise gradient norms were at most 6e−5/0 across all arms. "
           "Race coverage was zero, so all answers used the terminal fallback. The scalar update moved intermediate "
           "activity but did not establish a persistent route or useful gradient "
           "at the classifier; candidate support and downstream survival remain open.", "small"),
         *fig(fig_e83_spike_option_training, W),
         P("<b>State-conditioned optionality (§152):</b> optionality is a value of the current event state under a stated "
           "label, route proposal, horizon, and continuation budget; rollout samples estimate that value. It is distinct "
           "from class entropy, true-label surprise, route entropy, and optimizer momentum. A matched immediate-only and "
           "two-rollout lambda=1 depth-4 comparison both remained at 6/128 accuracy. The rollout arm's mean reserve "
           "change was about 0.001 loss/action by epoch 2. Neither parent nor child had a sampled future beat the "
           "0.05 cutoff: 0/48 per side in epoch 1 and 0/54 per side in epoch 2. "
           "Layer-4 support was 0.78%, and almost all measured counterfactual correction still landed in the readout. "
           "This validates the measurement path, not a training gain; the run used one seed and the reserve depends on "
           "the sparse proposal. The two-rollout arm took 224 s versus 177 s for control.", "small"),
         *fig(fig_e83_optionality_state_value, W),
         P("<b>Six-epoch follow-up (§153):</b> both matched depth-4 arms remained at 6/128 held-out accuracy through "
           "epoch 6 (the continuation arm reached 7/128 only in epoch 1); race coverage stayed at zero and final loss "
           "was approximately the uniform 20-class value, log 20. The continuation update briefly expanded L3/L4 "
           "support and retained more L2 activity, but finished with zero L3/L4 support and no accuracy gain. This rules "
           "out extra epochs alone as a remedy for this configuration. Section 153 derives the proposal-support limit: "
           "more rollouts cannot recover an improving route that the proposal assigns probability zero.", "small"),
         P("<b>What this establishes:</b> these are clear measured capability leads on the tested tasks and a promising "
            "real-language result. E79 is a single-seed expert mixture without matched compute, while the strongest depth "
            "and retrieval comparisons are synthetic tasks built around event primitives. A general-language-model scaling "
            "advantage and lower training energy remain to be demonstrated.")]
    s.append(PageBreak())

    s += [P("Accuracy and work across controlled tasks", "h1")]
    s += fig(FM.fig_supremacy_map, W)
    s += [P("These task-level comparisons show where learned event computation has a measured lead. Operation counts "
            "are not end-to-end energy measurements; real-stream results and current gaps appear in section 7.", "small")]
    s.append(PageBreak())

    s += [P("In plain terms", "h1"),
         P("Today's neural networks are <b>clocked and dense</b>: at every step, every input is multiplied by every weight, "
           "whether or not anything happened. Many real signals are the opposite: long silences broken by precisely timed "
           "events (nerve spikes, trades on a market, sensor alarms), where <i>when</i> something happens is the information."),
         P("<b>Sleeping Machines are networks that only work when an event arrives.</b> A node waits. It fires when the right "
           "inputs arrive in the right time window (“B within 1.5 s after A”), and the first node to fire gives the answer, "
           "a <i>race</i>. Silence costs nothing, and time itself does the computing: a delay or a waiting window plays the "
           "role that a weight matrix plays in a dense network. A signal can also carry a small vector (a few numbers), and "
           "its content sets its own delay: when it arrives decides how much it counts."),
         P("The questions are whether such networks can <b>learn</b> (with credit that flows only along the events that "
           "actually happened: a node adjusts only its few connections that were active, like moving money between accounts under a fixed budget) and "
           "whether they can <b>match or beat</b> MLPs and Transformers.")]
    s += fig(FM.fig_concept, W)
    s += [P("Highlights", "h1")]
    s += bullets([
        "<b>Same accuracy, 10,000–100,000× less computation.</b> On timing-pattern recognition a learned event network is "
        "perfect (1.000) using ≈ 7.5 events per example; Transformers reach 0.989–0.998 at 150k–1.2M multiply-adds after "
        "1–2M training examples.",
        "<b>It groks where a Transformer does not.</b> Trained on 30% of all (a, b, c) triples, it learns (a + b + c) mod 17 "
        "and is 99.4–99.9% correct on the triples it never saw; a Transformer with weight decay stays at 3–63%.",
        "<b>Ten times less data.</b> On the depth-3 order task the event network is 99.8–99.9% correct after 1,000–2,000 "
        "examples seen once; a Transformer allowed as many passes as it likes needs about 10,000–20,000 for 99% (2,000: "
        "33–41%; 5,000: 82–90%).",
        "<b>Learning to retrieve from ≥ 250× less data.</b> In a recall task where the network must learn which stored key a "
        "query refers to (what attention learns), race attention trained by local credit is 100% correct after 1,000–4,000 "
        "examples (5 of 5 runs) and stays 100% on contexts four times longer. Of seven Transformer configurations (width "
        "64–128, 2–4 layers, absolute or relative positions), only the two largest solve it, after 400k–1M examples, and they "
        "reach at most 72% on the longer contexts (E61; the best configuration twice, solving it between 400k and 1M examples).",
        "<b>Deep order from a few thousand examples.</b> Recognizing which of 20 orders of four patterns occurred needs four "
        "levels of “this, then that”. The network finds the right detectors among 55 million candidates and is 99.9–100% "
        "correct on 5 of 5 runs after 10–15k examples, with ≈ 2,000 learning updates, ≈ 150 events per example and only "
        "≈ 80k connections ever created. <b>At depth 4 a Transformer makes about ten times as many errors with the same 40k "
        "examples, and still 4–8 times as many with 2M</b> (0.990; 0.992–0.996 vs 0.999–1.000), at ≈ 7,000× the computation; "
        "at depth 3 it matches only with 8–400× more data.",
        "<b>Composing parts: Transformer-level accuracy from one pass over the data, at ≈ 10⁴× less computation.</b> 15 classes built from "
        "ordered pairs of shared motifs: 0.990–0.999 (mean 0.9965) from 40k examples seen once, with learned timing windows, "
        "at ≈ 20 events per example; a Transformer needs 2M examples for 0.9955–0.998 and reaches 0.9935–0.9965 given the "
        "same 40k examples 50 times.",
        "<b>A world model of a real market stream within 0.08–0.19 nats of a Transformer at ≈ 1/3000 of the computation.</b> "
        "On days it never saw, a small event network with slow regime counters beats a GRU point process and comes within "
        "0.08–0.19 nats per event of a Transformer point process, at ≈ 40 operations per event instead of ≈ 110k–130k; the "
        "Transformer is the more accurate model.",
        "<b>Learning cost follows activity, not size.</b> Eight times more inputs (12 → 96 channels) costs no more learning "
        "mistakes.",
        "<b>New theory, proved:</b> exactly what one event node can compute and where depth is needed; why a fixed weight "
        "budget lets a node learn an AND without knowing which half was wrong; that deep order is trainable with the fewest "
        "mistakes any learner can guarantee; and that a race of clocks carries both the softmax (which fires) and its "
        "normalizer (when), so Transformer attention and its gradients are computed on average exactly by local rules: "
        "Transformers, including their training, are a limit of these networks. And event networks are exactly controlled "
        "differential equations driven by their events: the order detectors they learn are the universal features of event "
        "streams, the state-space units behind today's best event-stream models are a special case, and a race run in "
        "continuous time is more expressive than a softmax, the more so the longer it deliberates (§104). When signals carry "
        "vectors whose content sets their delays, a receiver whose state fades computes exactly softmax attention, paying "
        "only for the messages that match (§105), and the work attention then costs is set by how sharp it is, not by how "
        "much context there is (§106).",
    ], st)
    s += [P("<b>Scope of the evidence.</b> The supremacy results (timing, composition, deep order, grokking) are on synthetic "
            "tasks built to test one capability at a time, where the target is exactly expressible by the primitives. The theory "
            "behind them (what a node computes, mistake bounds logarithmic in the candidate basis, cost proportional to events) "
            "is not task-specific: the reason to expect them to carry over to sparse, precisely timed real streams. On the real "
            "data tested so far the event network is competitive at a small fraction of the computation, not ahead: spoken digits "
            "0.675 vs ≈ 0.70 (LSTM) and 95–96% (event-by-event state-space models); a market world model 0.08–0.19 nats behind a Transformer point process; no trading edge "
            "after fees in four markets; on a real event-camera benchmark (DVS128 Gesture) far behind: 0.70 vs 94–98% published.")]
    s += [P("<b>What the network actually does</b> on one example: spikes arrive; part detectors fire when two spikes are "
            "close enough in time; an order detector fires when part B follows part A; the class node holds that and fires "
            "when C arrives. With the same motifs in another order, the “A then B” detector still fires but nothing "
            "completes the pattern.")]
    s += fig(FM.fig_anatomy, W)
    s += [P("<b>Why learning it is not trivial.</b> When a detector for “A, then B, then C” fails to fire, which of its "
            "connections should change? Crediting every candidate spreads the weight so thinly that the node never fires; "
            "crediting the tempting shortcut (“A, then B” is shared with another class) traps it; exploring a little, then "
            "settling, finds the right order and keeps it (§84).")]
    s += fig(FM.fig_credit, W)
    s += [P("<b>Where it does not win yet:</b> event-camera gestures (0.70 vs 94–98% published), spoken digits (0.675 vs 0.70 "
            "for a published LSTM and 95–96% for event-by-event state-space models, whose unit the new theory identifies as a "
            "special case of ours; E74's initial time-vector pilot reached 0.146 peak held-out speaker accuracy, while E82's "
            "partial readout sweep reached 0.184), and trading, where no "
            "learner beats buy-and-hold on this data (an audit shows why: the predictable edge, ≈ 1 bp per trade, is below "
            "any taker fee). <b>Next:</b> a path to generative language models built this way (section 10).")]
    s.append(PageBreak())
    s += [P("Sleeping Machines proposes that computation can happen <b>in time rather than memory</b>: candidate events "
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
        "<b>Depth is learned natively when credit is right (§83–§86).</b> Summed potentials with conserved multiplicative "
        "credit, credit to one instant with cooled exploration, and synapses grown only when credited (provably the same "
        "decisions as dense weights) learn order among three and four parts: 0.999–1.000 on 5/5 seeds at depth 4 with "
        "≈ 2,000 updates and ≈ 80k grown synapses out of 5.5·10⁷ candidates. With learned timing windows and credit to the "
        "latest instant, the composition task a Transformer led reaches 0.990–0.999 (mean 0.9965) from 40k examples seen "
        "once, vs 0.9955–0.998 for a Transformer after 2M (§88–§89).",
        "<b>Grokking occurs, by a route change under sleep, and only for relations the substrate can express.</b> "
        "Without sleep the network memorizes; with sleep it generalizes after a delay (0.93–0.99, reliable with cooled "
        "timing noise); a data × sleep phase diagram shows memorization, grokking and collapse; with depth, (a + b + c) "
        "mod p through two composed rhythm stages reaches 0.99–1.00 (3 seeds, p = 17 and 31).",
        "<b>Learning cost follows activity, not model size:</b> 12 → 96 input channels leaves learning mistakes flat and "
        "makes inference cheaper (§77).",
        "<b>Not yet: real asynchronous benchmarks.</b> SHD below dense baselines; on the market, correctly posed as "
        "trading with costs, no learner profits and the native one learns to stay out; the online world model, built as "
        "an event network, beats a neural point process by 0.5–0.9 nats per event, also on held-out days.",
        "<b>Transformers are a limit of these networks, including their training, and time adds what they lack (§96, "
        "§101–§106).</b> Races of random clocks compute softmax attention and its gradient on average from local quantities; "
        "event networks are controlled differential equations whose universal features are the order detectors they learn; "
        "content-dependent delays compute softmax attention exactly at a cost set by its sharpness; a network laid out on "
        "positions and time scales is exactly equivariant to shifts and tempo changes, the two ways speakers differ. E79's "
        "race mixture leads the completed 1M text8 LSTM and Transformer baselines on the same split, and at 10M is ahead of "
        "the completed LSTM by 0.186 bpc on the same test segment; compute is not matched. The 10M four-layer Transformer "
        f"scores {tf_10m_final['test_bpc']:.4f} held-out test bpc after validation-based checkpoint selection. "
        "Deep E77 language-model results are still pending.",
    ], st)
    s.append(PageBreak())
    s += [P("Where the event paradigm wins, and where it does not", "h1"),
          P("“Supremacy” here means a measured advantage over dense networks given the same data; each row states its "
            "caveat.")]
    s.append(table([
        ["claim", "evidence", "caveat"],
        ["<b>Equal or better accuracy at 10⁴–10⁵× lower cost on timing tasks</b>",
         "E35: 1.000 at 7.5 synaptic events, nothing given; conv net 0.995 at 3.07M MACs; event-token Transformer "
         "0.989–0.996 at 146k–1.16M after 10× more training",
         "one task family; much of the gap is the clock (an event-driven conv net ≈ 10×)"],
        ["<b>Groks composed arithmetic where a Transformer does not</b>",
         "E41, (a + b + c) mod 17, 30% of triples: 0.994–0.999 (3/3) in 200 epochs; Transformer (AdamW, weight decay, "
         "100k steps): 0.29 / 0.63 (seed 0, d = 32 / 64), 0.06 / 0.03 (seed 1; chance 0.06)",
         "the two-stage rhythm route is a provided resource (E45: it can choose among routes); more steps might help the "
         "Transformer"],
        ["<b>World model: beats a GRU; within 0.08–0.19 nats of a Transformer point process at ≈ 1/3000 of its cost</b>",
         "E48: online −2.11 vs GRU −2.62 nats/event; held-out frozen −2.38 / −2.10 vs −3.15 / −2.98; ≈ 19 synaptic ops "
         "vs thousands of MACs",
         "the Transformer Hawkes process is more accurate (held-out −1.97 / −1.82 vs −2.16 / −1.97, same hazard family)"],
        ["<b>Learning cost follows activity, not model size</b>",
         "E35: 12 → 96 channels, 0.999–1.000, mistakes flat, inference cheaper (7.5 → 3.2–4.0 synaptic events); §77, §81", "measured to 96 channels"],
        ["<b>Structure discovery, implicit Occam razor</b>",
         "E45 pilot: one rhythm for a + b (1.000), the chain for a + b + c (0.999), nothing for random tables", "pilot, 2 seeds; 5-seed runs queued"],
        ["<b>Learned retrieval from ≥ 250× less data</b>", "E61: race attention with a learned query–key match, 5/5 runs 100% "
         "after 1–4k examples, 100% on 4× longer contexts; of 7 Transformer configurations only d = 128, 4 layers solves it, "
         "after 400k–1M examples (4× length: 0.25 absolute positions, 0.72 ALiBi)", "the event learner's candidate routes are "
         "(item, offset) pairs (ALiBi gives the Transformer relative offsets too); one run per configuration, two for the best (solved at 400k–700k and 700k–1M)"],
        ["<b>Deep order learned from few examples</b>", "E54: 20 orders of four motifs, 0.999–1.000 (5/5) after 10–15k "
         "examples, ≈ 2,000 updates, ≈ 150 events, 80k of 5.5·10⁷ candidate synapses grown", "depth 3: a Transformer matches (0.996–0.999) at ≈ 5,000× the "
         "computation; depth 4: 0.990 at equal data, 0.992–0.996 with 2M (4–10× the error rate)"],
        ["<b>Composition: Transformer-level accuracy from one pass</b>", "learned windows + latest-instant credit "
         "0.990–0.999 (mean 0.9965) from 40k examples once (E89); Transformer 0.9955–0.998 after 2M, 0.9935–0.9965 given the "
         "same 40k × 50", "one seed at 0.990; dips without a margin"],
        ["<i>Not supremacy:</i> spoken digits (SHD)", "E59 class-conditional event world models, speaker-relative bands, selected on held-out speakers: "
         "0.675 test (E51 0.647); LSTM ≈ 0.70; state of the art 95.9–96.3% (event-by-event state-space models)", "unseen test speakers"],
        ["<i>Not supremacy:</i> trading profit", "E42 (21 unseen days): no learner beats buy-and-hold (+932 bp); the priced native one +226 bp, others lose",
         "the predictable edge is ≈ 1 bp per trade, below any taker fee (E55, E55b): staying out is correct"],
    ], [48, 76, 50], st))
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
    s += [P("<b>A learning calculus for deep event networks (§83–§88).</b> Learning deep order natively needed five rules, "
            "each derived from a failure, each local to a node and paid for by events:")]
    s.append(table([
        ["problem", "rule", "why it works", "evidence"],
        ["an AND fails: which half was wrong?", "sum inputs; multiplicative credit under a conserved budget",
         "conservation moves weight toward the target on every false fire iff θ > ½ budget (§83, proved)",
         "4× candidates → 1.3× updates (E34w)"],
        ["candidates fire at different instants", "credit one instant, chosen by cooled exploration",
         "all-instant credit deadlocks (proved); greedy cycles on shared prefixes; valid routes absorb (§84)",
         "depth 3: 0.998–0.999 vs 0.71–0.85 greedy, 0.33 all-instant (E53)"],
        ["the candidate basis grows as P^depth", "grow a synapse when first credited",
         "exactly the decisions of dense weights (§85, proved, checked)", "depth 4: 80k of 5.5·10⁷ grown, 0.999–1.000 (E54)"],
        ["converged nodes sit on their threshold", "near-miss margin, earned by recent precision",
         "margins survive r demotions (§86); unearned margins protect wrong routes (§86b)", "0.997–0.999 at every checkpoint (E53g)"],
        ["timing precision", "tune one window per part from its own lags",
         "a window bank costs activity quadratic in resolution; tuning costs none (§88)", "learned = hand-tuned windows (E56)"],
        ["first-to-fire commits on partial evidence", "credit the latest candidate instant on a miss",
         "the pattern is complete only at its last event; earlier instants are prefixes (§89)",
         "composition 0.990–0.999; depth 3 at half the updates (E89)"],
    ], [34, 40, 62, 38], st))
    s += [P("<b>A derived law, measured (§91).</b> Latest-instant credit works because noise after the pattern is inconsistent "
            "between examples; the bound says learning slows as 1/(1 − q − 2f), q the share of examples whose last event is "
            "noise. Sweeping the noise and measuring q, the updates needed follow the law with one fitted parameter (f ≈ 0.03):")]
    s += fig(FM.fig_drift, W * 0.8)
    s.append(PageBreak())

    s += [P("3. Against dense models and Transformers", "h1"),
          table([["model", "test accuracy", "cost per episode", "learning"],
                 ["<b>event network, nothing given (E35)</b>", "<b>1.000 ×4, 0.9995</b>", "<b>7.5 synaptic events</b>",
                  "443–1,530 updates, 200k episodes"],
                 ["event-token Transformer, 2M episodes", "0.989–0.996", "146k–1.16M MACs", "backprop"],
                 ["Transformer + relative-time bias, 1M episodes", "0.996, 0.998", "≈ 150k MACs", "backprop"],
                 ["event-token Transformer, 200k episodes", "0.65–0.97", "5k–576k MACs", "backprop"],
                 ["clocked conv net (E32)", "0.995 / 0.984 / 0.895", "3.07M / 123k / 1.9k MACs", "backprop, 200k"]],
                [56, 36, 42, 40], st)]
    s += fig(fig_e32, W * 0.95)
    s += [P("<b>Scaling with the input basis (E35, §77).</b> At fixed spikes per episode, 12 → 96 candidate channels "
            "leaves accuracy ≈ 1.0 (96: 1.000, 0.999, 1.000) and learning updates flat while synaptic events per episode fall: learning and "
            "inference cost follow activity, not the size of the basis.")]
    s += [P("<b>Why.</b> Attention cannot see order without position information and must synthesize time comparisons from "
            "dot products at O(n²·d) per layer; a hold/trigger node computes the comparison as its primitive, at the cost "
            "of its input events (§70). The clocked model also pays per time bin, so silence multiplies its cost; a conv "
            "net evaluated only where spikes are would cost ≈ 100 multiply-adds, not millions (§63). At published "
            "per-operation energies the gap at equal accuracy is ≈ 10⁴×. <i>Limits:</i> one task built around the "
            "primitives; the Transformer needed 10× more training for parity; with a learned relative-time attention bias "
            "(time differences enter attention directly) it reaches 0.996–0.998 at ≈ 150k MACs, still below the event "
            "network at ≈ 2·10⁴× its cost.")]
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
    s += [P("<b>What the credit rule must be (§83–§84).</b> A class node should sum its held and its coincident trigger inputs "
            "and learn by full-information multiplicative updates under a conserved budget. Proved: conservation resolves the "
            "AND's credit ambiguity (a false fire does not say which half was wrong) iff the threshold exceeds half the "
            "budget; the mistake bound grows with the log of the candidate basis (16 → 32 channels: 1.3× the updates). "
            "Crediting every candidate deadlocks when units fire in every positive at different instants (window bank: "
            "0.05–0.10); crediting the one instant closest to firing can cycle between invalid prefixes; cooled exploration "
            "over instants reaches the valid route, which is absorbing. With a class window covering the task's span, E34's "
            "task plateaus at 0.988–0.995 per seed and the generic bank at 0.97–0.99 with fixed windows. <b>Learned windows "
            "(§88) plus credit to the latest instant (§89)</b> remove the last errors, lost races to a shortcut that fires "
            "when the second motif merely begins: 0.990–0.999 (mean 0.9965) from 40k examples seen once, against a "
            "Transformer's 0.9955–0.998 after 2M examples and 0.9935–0.9965 given the same 40k examples 50 times: parity at "
            "equal data, from one pass, at ≈ 10⁴× less computation."),
          P("<b>Order among three parts: depth 3 (E53, E54).</b> Classes that are different orders of the same motif sets "
            "(20 classes, decoys = other orders) need ordered intermediates: composite units u → v over every ordered pair of "
            "parts (57,840 candidates, ≈ 41 events per episode). 5 seeds, 40k episodes:")]
    s.append(table([
        ["E53", "test accuracy", "updates"],
        ["<b>depth 3, + margin earned by reliability (§86b)</b>", "<b>0.997–0.999 at every checkpoint, 5/5 seeds</b>", "≈ 1,100–1,400"],
        ["depth 3, instant credit, exploration T = 0.3", "0.998–0.999 on all seeds by 5k episodes; final 0.963–0.999 (dips)", "≈ 1,100"],
        ["depth 3, greedy instant credit", "0.71–0.85 (cycles between shared prefixes)", "4k–11k"],
        ["depth 3, credit to every candidate", "0.33–0.36 (never fires)", "≈ 26k"],
        ["depth 2, same credit", "0.32–0.42", "≈ 24k"],
        ["<i>Transformer, 2M fresh examples</i>", "<i>0.997–0.999</i>", "<i>215k–843k MACs/example</i>"],
        ["<i>Transformer, 40k × 50 passes (wd 0 / 0.1)</i>", "<i>0.9975, 0.996 / 0.981, 0.994</i>", "<i>215k MACs/example</i>"],
    ], [70, 74, 30], st))
    s += [P("Without a margin, converged classes dip (a node sits just above threshold; one demotion knocks it under). A margin "
            "kept by near-miss credit removes the dips but also protects wrong routes (3/5 seeds freeze on a shared prefix); a "
            "margin earned by reliability, applied only when the node's recent fires were mostly correct, is stable at "
            "0.997–0.999 (§86b). With "
            "synapses grown only when first credited (§85, provably the same decisions as dense weights), depth 3 at 20 "
            "channels gives 0.981–1.000 with 9.5k–13.7k of 145k candidate synapses ever grown. <b>Inference cost follows the "
            "learned structure (§93):</b> extending a unit only if one of its children carries weight (checked periodically, "
            "like sleep) keeps accuracy (depth 3 identical; depth 4 0.995–1.000) while events per example fall 42% at depth 3 "
            "and 75% at depth 4 (≈ 155 → 39).")]
    s += [P("5. Generalization and grokking", "h1"),
          P("<b>What counts (§58):</b> restriction, forced generalization above capacity, and grokking (the relation reached "
            "while memorizers are available) are different claims; each reports ρ = n/params. <b>Per-class parameters "
            "cannot generalize on (a + b) mod p (§66):</b> each operand occurs once per class, so generalization needs shared "
            "intermediates; in time, a sum needs a rhythm."),
          P("<b>Grokking as a route change (E37).</b> Each class has a pair-node lookup that can memorize everything "
            "(ρ ≈ 0.016) and a shared route through a rhythm with learned delays; learning is errors-only. Without sleep: "
            "train 1.0, test 0.03–0.04. With sleep: test 0.93–0.97 in 2 of 3 seeds without timing noise, and 0.95–0.99 in "
            "5 of 5 seeds with cooled timing noise on the shared route (σ = 2), after a delay. Sleep without the rhythm: "
            "train 0.45, test 0. The rhythm is used only where it fits: "
            "a − b and relabelled sums grok (0.95–0.98), a·b in 1 of 3 seeds, while a² + ab + b² and random tables stay "
            "at chance on unseen pairs (0.01–0.04) and their training accuracy erodes under sleep. A data × sleep phase "
            "diagram (4 × 4, 3 seeds) shows no grokking below 20–30% of pairs, and above it a minimum sleep that falls "
            "with data. Larger problems grok more reliably: 2/3 seeds at p = 31 and 59, 3/3 at p = 97 (ρ ≈ 0.005). With depth (E41, 3 seeds): (a + b + c) mod p through two composed rhythm stages, 0.99–1.00 on unseen "
            "triples from 30% of them at p = 17 and 31, and from 10% in 2 of 3 seeds at p = 31 (ρ ≈ 0.003); chance "
            "without sleep; collapse without the chain. The data it needs is far above the Occam bound (§78): at 1–7% of "
            "triples it memorizes and stays at chance, also when trained 7–10× longer; the threshold at p = 17 lies "
            "between 7% and 30%.")]
    s += fig(fig_e37, W * 0.9)
    s += fig(fig_phase, W * 0.9)
    s += fig(fig_e41, W * 0.9)
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
        "<b>Spiking Heidelberg Digits</b> (spoken digits as cochlear spike trains, 700 channels, 20 classes, unseen test "
        "speakers): class-conditional event world models (E51; state = last spike's band, time since it, time since onset; "
        "one counting pass) reach 0.647 test (0.734 on held-in speakers); timing +0.06, onset reference +0.21. The weight "
        "race reached 0.35; a published LSTM ≈ 0.70; the state of the art is 95.1% (learned delays), 95.9% (Event-SSM) and "
        "96.3% (S7): the last two process spikes one event at a time with linear state-space units, which §104 shows are "
        "event units of our kind with every unit updated on every event (both select checkpoints on the test set); time only fades their state, so they do not "
        "compute with delays. E74 tests the paradigm's own design: events carry small vectors whose content sets their "
        "delays and whether they are sent; a 2k-train, 500-held-out pilot reached 0.146 peak speaker accuracy and 0.120 at "
        "its final epoch after six epochs. E82's partial readout sweep reached 0.184 held-out accuracy after 240 updates. "
        "An E83 audit found its old sequence loss averaged softmax through silent and batch-padded time, so depth-2 runs are "
        "excluded as trainability evidence. In guarded depth-4 screening (512 train / 128 held-out-speaker examples, two "
        "epochs), integral and max objectives ended at 4.69% max-over-time accuracy (20-way chance 5%). The anytime "
        "race/fallback reached 6.25% max-potential accuracy (8/128; chance-tail p=0.31) and 5.47% emitted accuracy at "
        "100% coverage. Every tested threshold emitted every item and confidence saturated at 1.0; Layer 4 rose from "
        "932 to 1,024 spikes per utterance. This diagnoses false confidence/activity growth, not reliable learning. The "
        "stable-cause race-only control ended at 3.12% max accuracy, 97.66% coverage, and 4.8% emitted accuracy with "
        "0.981 peak confidence. A readout-only shadow probe at seed-2 initialization (not trained weights) forced 32 "
        "near-gate routes on four held-out utterances: 31 changed max-pooled CE by exactly zero and one reduced it by "
        "0.045; boundary-gradient norm was 2.2% of pathwise norm, cosine 0.012. Section 129 derives why max pooling can "
        "erase routes that remain below the temporal winner. `TVLayer` still detaches its hard content gate and computes "
        "spike identities in no_grad, so closed routes and silent units get no pathwise task gradient. E83 now trains a "
        "causal prefix posterior with a fixed 0–1000 ms query window and shadows near-boundary routes through the full "
        "downstream stack. The first unbiased total estimator over about 480k eligible routes and 128 shadows per epoch "
        "destabilized the depth-4 model. A clipped normalized local rule avoided that explosion but did not improve "
        "recognition: the matched 128/32 seed-6 run stayed at 6.25% terminal accuracy and had epoch-2 prefix NLL "
        "[2.996, 19.735]. Only 11.6% of sampled openings helped; the counterfactual/pathwise cosine was 0.0038. "
        "Layerwise shadow deltas and gradient norms are still noisy, so update gain needs uncertainty-aware calibration. "
        "The E74/E82 results are above the 0.05 chance level for 20 classes and set the current learning target. SHD is only ≈ 6× sparser than a 10 ms raster, "
        "a weak test of the paradigm's cost advantage. Validating on held-out speakers and coding bands relative to each "
        "voice (a running centroid per utterance) raises held-out-speaker accuracy from 0.36–0.38 to 0.44–0.46 and the test "
        "to 0.675 (E59, §92).",
        "<b>Market stream posed as trading with costs (E42, confirmed on 21 unseen days, preregistered):</b> at 2 bp, "
        "imitating a hindsight teacher over-trades and loses (event learner −15,236 bp, logistic −24,302); the "
        "profit-priced event learner nets +226 bp with 8 changes; buy-and-hold +932; at 10 bp all stay out. "
        "No learner beats buy-and-hold; pricing the decision is what stops the losses.",
        "<b>Is staying out right? An edge audit (E55, §87).</b> From executable round trips on the tape (buy at the ask, "
        "sell at the bid), BTC spot's own event states carry a real held-out edge of +0.3 to +0.9 bp per trade before fees, "
        "and lead–lag states of BTC perpetual futures and ETH raise it to +1.1 to +1.5 bp: more markets carry more "
        "information. A 2 bp round-trip fee (a tenth of a realistic taker fee) removes it: staying out is correct for a "
        "taker here; the edge would need market-making economics. <b>Confirmed on the 21 untouched days</b> (preregistered): "
        "before fees +0.26 to +1.05 bp from the own state, +0.47 to +1.40 bp with perp and ETH; at 2 bp every selected "
        "state loses; at 5 bp none qualifies. <b>Across four markets</b> (ETH spot, SOL spot or the BTC perpetual traded, the "
        "others as leaders; choices made on the pilot days): at most ≈ 1 bp before fees, nothing at 2 bp (E55b).",
        "<b>Event-camera gestures (DVS128 Gesture, E60; official split).</b> Native motion events (refractory cells, onsets, "
        "direction-selective pair detectors with an opponent veto) turn ≈ 410k raw events per gesture into ≈ 15k. A bag of "
        "motion events reaches 0.663 on unseen people; depth (successive motions per region) with the multiplicative learner "
        "reaches 0.776 on held-out training users and 0.701 on the test users (chosen on validation). Published systems reach "
        "94–98%: not yet competitive; the representation misses rotation sense and trajectory shape.",
        "<b>Against a Transformer point process (E52, E57, §90).</b> A Transformer Hawkes process with the event network's own "
        "hazard family and 128 events of context, selected on day 5, scores −1.97 / −1.82 nats per event on the held-out days "
        "(≈ 108k multiply-adds per event; −1.83 / −1.65 with 12 finer windows). The semi-Markov event network scores −2.38 / "
        "−2.10; slow regime state (leaky event counters at 5 s and 60 s, an order-flow counter, backoff) brings it to −2.18 / "
        "−2.00, and per-type counters as a third backoff level to −2.16 / −1.97 (fine windows −1.91 / −1.73; chosen on day 5) at ≈ 30 operations per event. The Transformer is the better world model by "
        "0.08–0.19 nats; the event network gets within that at ≈ 1/3000 of the computation. Counted slow state transfers to "
        "unseen days; constant-step multiplicative factors track the end of training and do not (E58).",
        "<b>The world model is an event network, and it beats a neural point process (E44, E48).</b> A likelihood "
        "decomposition located the GRU's lead in which event comes next; count baselines located the missing information "
        "(the time since the last event). A semi-Markov event network (state nodes for the last two types, window nodes "
        "from a delay line, count-learned detectors) reproduces that model exactly and scores −2.11 nats per event online "
        "(days 1–5) and −2.38 / −2.10 frozen on held-out days 6 / 7, against −2.62 and −3.15 / −2.98 for the GRU, at ≈ 19 "
        "synaptic operations per event. The model class is classical; an offline-trained GRU scores −2.72 / −2.53 on the "
        "held-out days, also behind (E49); the Transformer point process is ahead of both (above).",
        "<b>Earlier native world model (E44), prequential log-likelihood per event (nats; days 1 / 2 / 3):</b> Poisson "
        "−3.00 / −3.42 / −3.32; Hawkes (Adam) −2.62 / −2.94 / −2.84; native (multiplicative) −2.64 / −2.85 / −2.70; "
        "GRU neural point process – / −2.61 / −2.52. Pair-part state neutral; learned inhibition below excitation-only; the "
        "semi-Markov network of E48 (above) closed the gap to the GRU.",
    ], st)
    s += fig(fig_e83_route_diagnostics, W * 0.92)
    s += fig(fig_e83_d4_support, W * 0.96)
    s += fig(fig_e83_equal_updates, W * 0.96)
    s += [P("<b>Route reachability and depth (§§139–140).</b> The exact D4 candidate masks connect 90.2% and 98.0% "
            "of first-layer/fourth-layer unit pairs in seeds 6 and 7, with every input band connected to layer 4. "
            "Realized layer-4 support is only 1.6–7.0% in the equal-update arms. This locates the loss after static "
            "wiring: route gating and thresholded event generation determine which paths exist on a sample. A skip "
            "raises support to 99–100% without paired recognition gain. The MoE calculation requires paired losses for "
            "competing routes; the two-gate calculation shows how singleton shadows miss subthreshold route pairs that "
            "jointly cause a spike. For strict chains, retaining half the samples through depth 8 or 16 requires mean "
            "per-transition survival of 90.6% or 95.5%. So deeper stacks add combinatorial route choices, but they "
            "need route acquisition and per-layer survival to make those choices trainable.", "body")]
    s += [P("<b>Depth-credit redesign in evaluation (§136).</b> E83's deepest-only readout makes an early event's final-loss "
            "effect depend on surviving every later hard route. The new sparse `all_depths` option adds the class-evidence "
            "streams from every layer at the same causal prefix and scores lost routes against that fused objective; "
            "deepest-only remains the matched control. This reduces the serial credit burden but could let shallow branches "
            "carry the classifier. In the matched 128-example seed-6 pair, all-depth fusion reached 17/128 terminal and "
            "20/128 race-plus-fallback correct versus 7/128 deepest-only (paired McNemar p=0.0146). This signal did not "
            "replicate at seed 7: 9 versus 6 correct (p=0.607). Layer-4 removal did not hurt seed-6 fused accuracy, and "
            "sparse layer-1 skips restored layer-4 support to 99–100% without paired classification gain. The race was "
            "overconfident in both all-depth seeds (63% mean confidence; 23.8% emitted accuracy at seed 6, 0/8 correct at "
            "seed 7). These are two-speaker exploratory comparisons, not supremacy evidence. A separate gap remains: class "
            "evidence does not change during silence until another hidden event arrives or EOS is reached.", "body")]
    s += [P("<b>Readout shortcut resolved (§145).</b> A matched frozen replay tested the same valid hidden-spike toggles "
            "against both all-depth and deepest-only main loss. The late-only L1 all-depth delta averaged −0.00936 "
            "(16/21 helpful), but all 21 matched deepest-only deltas were exactly zero; every valid L1/L2 deepest-only "
            "delta was zero in both arms. The L1 toggles added 0.1429 L1 hidden spikes/example, no downstream hidden "
            "spikes, and 1.2381 L1 readout-edge updates/example. So the fused classifier rewarded its shallow L1 head; "
            "this local utility did not demonstrate serial credit through depth. Deepest-only L3/L4 samples were only "
            "2/1. The next test must train with deepest-only primary loss or explicitly replay a sparse suffix and show "
            "downstream event changes. This is a frozen audit, not a training gain.", "body")]
    s += [P("<b>Pair propagation is not pair synergy (§146).</b> In 1,024 held-out-speaker examples, opening both "
            "natural-off L2 events increased suffix spike count in 38/64 control and 19/34 late-only pairs, but improved "
            "deepest-only loss in only 8/64 and 4/34. At L3, 5/10 and 4/5 double openings improved loss, but the "
            "late-only pool had only five cases and one +1.40 loss outlier. The four-corner interaction "
            "Γ = L11 − L10 − L01 + L00 was zero in most sampled pairs: no natural-off pair succeeded "
            "when both singleton openings failed, and only 2/42 L3 pairs exceeded |Γ| = 0.01; none of 210 L2 pairs did. "
            "The audit paired distinct spike events without requiring a shared receiver. A true topology-conditioned "
            "route-pair experiment must compare shared-receiver arrivals to time-matched nonshared controls and track "
            "accepted messages, event payload/timing, deepest loss, and replay work. These are validation diagnostics, "
            "not a trained accuracy gain.", "body")]
    s += [P("8. Open problems and next steps", "h1")]
    s += bullets([
        "<b>Stability of the full rule set on every task at once:</b> the margin earned by reliability is stable at depth 3–4 "
        "and with fixed windows but hurts when windows are learned (it entrenches early shortcuts at the firing instant and "
        "promotes trailing noise at the latest instant, §89): the margin needs another anchor.",
        "<b>Depth beyond four and denser streams:</b> depth costs activity n·r^L (§85); extending a unit only toward children "
        "that carry weight cuts events by 42% at depth 3 and 75% at depth 4 at unchanged accuracy (§93); depth 5 is queued.",
        "<b>Deep real-stream trainability (E83/E84):</b> E83 now scores proper class posteriors at sampled causal prefixes, "
        "but its strict event chain has nested per-utterance support and silent units have no pathwise firing gradient. "
        "Depth-4 all-depth readout produced one paired seed-6 result that did not replicate at seed 7, and its L1 "
        "boundary utility was a shallow-head shortcut with zero deepest-only loss change. A sparse layer-1 skip restored "
        "layer-4 support to 99–100%, and an additive count mark raised it to 56–94%, "
        "without class improvement in either tested arm; the count-mark run ended at 0/32. The trained-checkpoint "
        "spike-boundary audit found few deep near-threshold events "
        "and mixed single-spike loss effects, so a boundary update is not yet justified. Prefix NLL and race calibration "
        "remain poor. Historical eval-size runs coupled evaluation selection to training RNG; new runs default to split "
        "streams, pending cross-limit validation. Next tests should hold the training stream fixed while expanding speaker "
        "coverage, then compare a calibrated event representation and boundary-credit rule against matched controls. E84's "
        "market stream remains a separate depth experiment.",
        "<b>Anytime sparse stream classification (§§119–§129):</b> if the prefix scores are calibrated posteriors, first "
        "crossing confidence 1−ε bounds error among emitted answers by ε. Calibration must hold at the policy-selected "
        "prefixes. Point-process likelihood uses both observed events and silence, so class-specific absence evidence can "
        "cause between-event threshold crossings. With sparse class-logit updates, max and log-sum-exp trees maintain the "
        "exact threshold statistic in O(r log C); full posterior-vector output still costs O(C). Sampled-prefix log loss "
        "is proper for the conditional class posterior even with only a stream label, but the race objective alone does "
        "not identify those prefix probabilities. Max pooling creates zero-credit regions for routes below the current "
        "temporal winner; smooth-max weighting can soften them with sparse event updates. A reference readout is written "
        "but not integrated or benchmarked.",
        "<b>Time-vector networks on real streams (§105–§106):</b> E74's first speech pilot reached 0.146 peak held-out "
        "speaker accuracy, and E82's partial readout sweep reached 0.184. E75 verified exact band-shift covariance but its "
        "pilot was resource-limited. Improve the learning signal, then test whether delays and vectors together close the "
        "remaining gap to strong speech models.",
        "<b>The work law of attention in language (§106a):</b> how many keys a trained character-level Transformer's queries "
        "actually need as the context grows (E76), which fixes what delay-coded attention saves on text.",
        "<b>A time-vector language model (§107, E77):</b> does sparse time-vector memory plus delay-coded retrieval reach the "
        "converged LSTM and Transformer at equal data, and how many keys does it retrieve per character?",
        "<b>Structure discovery for grokking</b> (E45 pilot picks correctly) and <b>the data threshold of grokking</b> (7–30% "
        "of triples, far above the Occam bound; the sleep reuse filter is the candidate constraint).",
        "<b>Native learning of sparse parity</b> (§75) and <b>a real benchmark with rare, precisely timed events</b> (§55).",
        "<b>Joules, not operation counts:</b> run trained networks on neuromorphic hardware (§9).",
    ], st)
    s += [P("9. Hardware: what these networks need, and what exists", "h1"),
          P("Each primitive the theory settled on maps to a hardware feature:")]
    s.append(table([
        ["what the network does", "what hardware must provide", "why"],
        ["work only when an event arrives", "event-driven execution; memory next to compute", "cost follows activity (§55, §77)"],
        ["delays and hold windows", "per-synapse programmable delays, per-node hold timers, timestamps", "order is held intervals (§61, §71)"],
        ["first to fire wins, the rest cancelled", "fast arrival-order resolution and inhibition", "the race readout (§59)"],
        ["multiplicative credit under a conserved budget", "per-synapse multiply, per-node renormalize, contribution tags, local "
         "random source, precision counter", "§83–§86b, §89"],
        ["synapses grown when first credited", "run-time allocation in a sparse synapse store", "basis 10⁵–10¹⁰, 10⁴–10⁵ grown (§85)"],
        ["part windows tuned from their own lags", "per-synapse window edges with a local rule", "§88"],
        ["signals carrying small vectors whose content sets their delay", "8–32 values per event, a small dense core per unit, "
         "a delay computed per message, jitter small against the delay scale", "delay-coded attention (§105); latency × "
         "weight error = jitter × logit range (§106b)"],
    ], [48, 76, 50], st))
    s += [P("<b>The optimal machine (a sketch).</b> Clockless digital cores with timestamped events, per-synapse delay and window "
            "fields and per-node timers; arrival-order comparators and inhibition trees for the race; a small event-triggered "
            "learning engine per core; a content-addressed sparse synapse store with allocation on credit. This is close to "
            "<i>race logic</i> (first arrival = minimum, delay = addition: the max-plus algebra of §79) plus learning. At ≈ 24 pJ "
            "per synaptic event (Loihi, 2018) these networks would spend ≈ 0.2 nJ per example on the timing task, ≈ 0.5 nJ on "
            "composition and ≈ 4 nJ at depth 4; a Transformer at 175k multiply-adds per example on a GPU is on the order of "
            "a microjoule (orders of magnitude, not measurements)."),
          P("<b>What exists (September 2026):</b>")]
    s.append(table([
        ["system", "availability", "fit"],
        ["Intel Loihi 2 / Hala Point", "research access; Hala Point a prototype (Sandia); Loihi 3 announced, no public specs",
         "best for prototyping: event-driven; weight, delay (≤ 62 steps), tag per synapse; microcode learning rules. Missing: "
         "run-time synapse allocation, long delays, timestamps"],
        ["SpiNNaker2 (SpiNNcloud)", "commercial systems; 152 ARM cores/chip", "most flexible: every rule incl. synapse growth in "
         "software; less efficient per event; time-stepped"],
        ["BrainChip Akida / Akida Pico", "commercial; Pico in FPGA-cloud evaluation (2026)", "converted CNN-style SNNs, limited "
         "learning, no suitable delays: poor fit"],
        ["Innatera Pulsar", "volume production (2026), µW–mW", "deploy small trained networks at the sensor"],
        ["SynSense Speck / Xylo", "commercial dev kits", "inference-only spiking ASICs for vision / audio"],
        ["DYNAP-SE2, BrainScaleS-2", "research", "analog continuous time; device mismatch"],
        ["IBM NorthPole", "research", "synchronous dense inference: not a fit"],
        ["FPGAs", "commercial", "delays as timestamp queues, synapses in BRAM hash tables: the most faithful full implementation today"],
        ["event sensors (Sony/Prophesee, iniVation)", "commercial", "natural front ends: they emit these event streams"],
    ], [38, 58, 78], st))
    s += [P("<b>What to do with it.</b> Measure joules: trained networks on Loihi 2, the full learning calculus on SpiNNaker2 or "
            "an FPGA, against a Transformer on a GPU; deploy frozen networks on sensor-edge chips. The feature no commercial "
            "chip offers, and the one these results say matters most, is <b>run-time synapse allocation on credit</b> with "
            "per-node conserved budgets: it lets a network search 10⁷–10¹⁰ candidates while storing only what it uses. "
            "Sources are listed in REPORT.md §9.", "small")]
    s += [P("10. Next frontier: generative language models", "h1"),
          P("<b>The aim</b> is not to approximate Transformers but to exceed them: the same or better quality, with work per word "
            "that does not grow with model size or text length, learned by local rules from less data."),
          P("<b>Why that is a reasonable aim</b> (theory, §96–§98):")]
    s += bullets([
        "<b>Nothing a Transformer computes is out of reach.</b> Query–key similarity is the overlap of spike codes; a race among "
        "stored keys picks the best match, and a race of randomly ticking clocks picks each key with exactly its "
        "softmax-attention probability; relative position is native; the feed-forward block is threshold units over codes; "
        "stacking layers is composition. A Sleeping Machines network can express any Transformer.",
        "<b>It has freedoms a Transformer lacks:</b> the order in which signals fire carries up to log₂ n! extra bits for n "
        "signals; only active units work, so a model can be very large while each word stays cheap; structure grows where "
        "it proves useful; sampling is a race; retrieval reaches only keys sharing a channel with the query.",
        "<b>Local learning is not a handicap in principle.</b> A race computes with minima and sums; the exact gradient "
        "backpropagation would compute runs only along the chain of spikes that caused the output, which each node traces "
        "locally: credit along that chain is backpropagation for these networks, and near misses supply the signal gradients "
        "cannot give to losing paths. For races of random clocks the exact gradient is local too.",
        "<b>Depth is trainable, optimally.</b> For ordered-pattern detectors of depth d, mistakes grow as d × log(candidate "
        "pool), and no learner can do better in the worst case (§97–§98).",
    ], st)
    s += [P("<b>Training, subsumed (theory, §101–§103).</b> A race of randomly ticking clocks splits its output into two independent "
            "channels: which clock fires first (a sample from the softmax) and when (the decision time, which carries the "
            "softmax's normalizer). Proved: every competitor computes its own softmax probability locally (its rate times the "
            "decision time); keys emitting values scaled by that product give, on average, exactly softmax attention; and the "
            "locally computed gradient of this race is, on average, exactly the gradient of softmax attention. A network of such "
            "races with small dense cores is trained by local message passing as stochastic gradient descent on the Transformer "
            "objective, up to an error shrinking as 1/R with R races per head: Transformers, including their training, are a "
            "limit of these networks. Numerical check below; training curves compared in E68 (queued).")]
    s += fig(FM.fig_race_theory, W)
    s += [P("<b>Time and content, one system (theory, §104).</b> Continuous-time neural models describe a hidden state that "
            "flows and is pushed by its input: neural ODEs, controlled differential equations, and the state-space models "
            "behind Mamba-class language models. An event network is exactly such a system. Between events its state flows in "
            "closed form, and at each event it jumps. The only nonlinearity is in <i>which</i> units fire <i>when</i>. Four "
            "consequences follow:")]
    s += bullets([
        "<b>Order detectors are the natural features of event streams.</b> The iterated integrals that make these models "
        "universal (the “signature” of a path) are, for event streams, the counts of ordered event patterns: exactly what our "
        "“this, then that” detectors learn. Stacking such detectors builds the universal feature set.",
        "<b>Selection comes free.</b> Mamba-class models gain their power by letting the input set how fast the state forgets. "
        "In an event network, which channel fired is that signal. A unit that an event does not address need not be touched "
        "at all, and skipping it is exact (sleeping execution). The best published models on spoken digits (95.9–96.3%) are "
        "such units with every unit updated on every event, and time only fades their state: they do not compute with delays.",
        "<b>A race unit is an integrate-and-fire neuron with a random threshold.</b> Its gradient is the event-based "
        "backpropagation used for spiking networks, but the random threshold keeps the expected loss smooth even when "
        "spikes appear or vanish. At the moment of decision, each unit's own integral is on average exactly its probability "
        "of winning, for any time-varying rates.",
        "<b>Time adds expressiveness.</b> If the scores change while the race runs, the race outputs a mixture of softmaxes "
        "over its own decision time. That is more expressive than the single softmax at the end of every Transformer (the "
        "“softmax bottleneck”). A fast race is one softmax; a slower race buys expressiveness with time rather than with "
        "parameters.",
    ], st)
    s += fig(FM.fig_race_time, W)
    s += [P("<b>Delays and vectors, computing together (theory, §105).</b> An event carries a small vector, and its content decides "
            "when it arrives: a message whose content matches the receiver is delayed in proportion to the match, and one that "
            "does not match is never sent. The receiver's state fades with time, so a later arrival counts more. Proved and "
            "checked: the receiver holds exactly softmax attention over the matching messages, with no multiplications for the "
            "weights and no sampling, paying only for messages sent. Races compute the same softmax by sampling (fast, slightly "
            "noisy); delays compute it by waiting (exact, slower for a wider range of scores). A unit fires when its evidence "
            "crosses threshold and sends on its state at that moment, so what it says and when it says it are one computation. "
            "Such networks compute in the log semiring: delays add, gains multiply. E74's initial spoken-digit pilot has "
            "a measurable but small learning signal; further optimization is needed."),
          P("<b>Two memories (theory, §107).</b> A time-vector unit has a restricted affine-accumulator resemblance to "
            "exponential-gated recurrent memories: elapsed time supplies decay, content-dependent delay supplies an "
            "exponential write factor, and a count channel normalizes the read. This is not an identity with a full xLSTM "
            "layer: E74/E77 do not implement sLSTM's learned gate/memory-mixing cell or mLSTM's matrix state of key/value "
            "outer products. E77's event-Hopfield update and token query/key/value retrieval are separate associative "
            "operations. xLSTM is a useful topology and scaling precedent: its 7B model was trained on 2.3T tokens, and a "
            "672-run study covered 80M–7B parameters and 2B–2T tokens, reporting better compute/loss trade-offs than its "
            "tested Llama2-style Transformer baseline (<link href='https://arxiv.org/abs/2405.04517' color='blue'>architecture</link>; "
            "<link href='https://arxiv.org/abs/2503.13427' color='blue'>7B model</link>; "
            "<link href='https://arxiv.org/abs/2510.02228' color='blue'>scaling study</link>). These results show that a "
            "nonstandard recurrent/matrix-memory family can be built and scaled deeply; they do not transfer to E77. We can "
            "apply the lessons—repeatable blocks, stable gate/residual settings, systematic parameter/data/context sweeps, "
            "and kernel performance treated as part of the architecture—while retaining event-state layers and sparse event "
            "routes. A write-time memory cannot answer arbitrary questions asked later: remembering N facts for any future "
            "question needs at least N × (bits per fact) of state. A language model therefore also needs retrieval: a query "
            "is sent to stored keys, which reply sooner the better they match; the first reply opens a short window, and "
            "replies inside are weighted exponentially, exactly softmax attention over the good matches. Aggregating values "
            "costs the number of good matches; finding them still costs linear work without an index. E77's adaptive path "
            "is not yet measured against the converged language baselines."),
          P("<b>Parallel state scan (theory, §107(i)).</b> With event arrivals/topology fixed, each time-vector memory step "
            "is an affine map z_k = A_k z_(k−1) + x_k. These maps compose associatively, so an exact prefix scan computes "
            "all states in logarithmic parallel depth with linear arithmetic work; the count normalizer has the same form. "
            "This gives a concrete route to parallelizing the recurrent state path without dense pairwise attention. It "
            "does not remove the current dense time-by-batch-by-unit tensors, parallelize hard event births or reset "
            "decisions, or establish an energy advantage. Next: compare outputs and gradients on fixed event schedules, "
            "then run a small, safe timing pilot.")]
    s += [P("<b>Potential: a layer can choose its memory operation.</b> Softmax key/value attention is a one-step modern "
            "Hopfield retrieval rule under the associative-memory interpretation. E77 now applies a separate causal "
            "query/key/value update to emitted event payloads, then passes the retrieved vector onward through the event "
            "stream. Its score gradient is centered on the retrieved value, teaching both where to read and what to send. "
            "The temporal layers remain event-state layers; the design is heterogeneous by intent. A learned gate mixes "
            "this event-level lookup with an identity payload path. In the auto-associative case, retrieval is the gradient "
            "of a convex log-partition function with a positive-semidefinite covariance Hessian; repeated retrieval is "
            "contractive when inverse temperature times key-diameter squared / 4 is below one. Separate keys and values "
            "enable hetero-association but need not preserve this energy structure, so E77 uses one gated update per layer."),
          P("<b>A depth result with explicit assumptions.</b> For fixed keys and values, one-query sensitivity is a "
            "cross-covariance bounded by inverse temperature times key diameter times value diameter / 4. In a full "
            "event sequence, a key reused by many later queries can amplify credit. Theory §107h gives a conservative "
            "operator bound using this maximum accumulated key attention mass plus query/key/value, gate and output gains. "
            "E77 logs key fan-out, the sequence-level bound, its residual-scaled value and layer sum, along with score "
            "spread, retrieval entropy, update size, diameters and role-specific gradients. The certificate fixes event "
            "order and candidate membership; hard event births and route discovery remain outside it."),
          P("<b>Precision and learning credit trade off.</b> With score margin m, non-winner mass is at most "
            "(N−1)e^(−βm), so sharper scores improve retrieval. For two keys, however, score sensitivity is "
            "βp(1−p): it peaks at a tie and vanishes when the route is certain. An excluded key gets no gradient. "
            "This mathematically motivates broad early retrieval, near-miss credit, and gradual sparsification."),
          P("<b>Deep sparse-stack stability (theory with the E114 diagnostic).</b> "
            "With 1/depth residual scaling and bounded local errors, §113 keeps forward and gradient perturbations "
            "depth-independent. Section 114 lifts fixed-support softmax truncation to the full sequence Jacobian: "
            "its operator error is at most √(R C), where R is a per-query row-sum bound and C is a shared-key "
            "column-sum bound. The C term measures truncation's accumulated influence through each reused key. This "
            "closes the local-to-sequence certificate for linearly projected attention. E77 still needs uniform bounds "
            "over its state region, a sparse-Jacobian Lipschitz bound, parameter-VJP and input-dependent gate terms; "
            "route changes remain the established §§19/57 counterfactual problem. This is theory, not evidence of "
            "training success. E114 checked 144 fixed-support synthetic cases by central finite differences: no absolute "
            "violation exceeded 1e−8. Among cases with bounds at least 1e−8, maximum Jacobian and forward-error ratios "
            "were 0.897 and 0.687; the largest absolute Jacobian excess was 1.32e−10. Exact-support cases have a zero "
            "bound, so their finite-difference residue is judged absolutely. Autodiff checks, learned support changes, "
            "and architecture-scale uniform constants remain open."),
          P("<b>If these mechanisms scale.</b> Deep event stacks that preserve associative recall and learn useful sparse "
            "routes could grow model memory and reasoning capacity without making every token pay for every possible "
            "interaction. Training would follow predictive routes; inference would follow emitted events and retrieved "
            "candidates. Capability growth would depend less on dense matrix size and more on temporal composition, "
            "associative memory, and sparse routing. A stronger accuracy-per-compute curve could shift frontier investment "
            "toward fast memory, event-capable processors, and distributed associative stores, while reducing dependence "
            "on ever-larger dense GPU clusters."),
          P("<b>Cost is still a live question.</b> The event Hopfield update runs only at emitted events, but it scores "
            "every eligible event pair. Token retrieval also scores every query/key pair, and TVLayer still allocates "
            "dense time-by-batch-by-unit state. Sparse wiring is not yet sparse execution or a measured energy win."),
          P("<b>Scaling depth.</b> E77 retains earlier event payloads, appends each layer's new events, and sends sparse "
            "raw-input skips; event-Hopfield updates use 1/depth residual scaling. This gives a fixed-topology identity "
            "inclusion path across event layers without adding dense tokenwise residual blocks. The queued 2/4/8/16-layer "
            "Transformer sweep matches context, training windows, token exposure, optimizer updates and parameter count "
            "to within 4%. A wider, longer depth-4 Transformer is a separate stronger reference. The two-layer configuration "
            "is a starting point, not a depth limit.")]
    s += [P("<b>Local learning that provably suffices (theory, §108–§109).</b> When units predict the next character by racing "
            "(each candidate's clock rate a weighted sum of the log-probabilities its inputs assign), a network of such units is "
            "a gated linear network: every unit predicts the target itself and learns only its own convex loss, so no error is "
            "sent backwards, and such networks are known to be universal and to learn well in a single pass (Veness et al., "
            "2021). The network is never worse than its best part. These units do not build features; here features come from "
            "the time-vector layers and native detectors, and the race units combine them.")]
    s += [P("<b>First evidence.</b> Deep order is learned from about ten times less data than a Transformer needs (section 4). "
            "Attention is learnable by local credit, from far less data: in a recall task where the network must learn which key "
            "a query refers to and which neighbour to read, a race-attention layer trained by local credit alone is 100% correct "
            "after 1–4k examples and 64–68 mistakes (5/5 runs) and 100% on contexts four times longer; the Transformers that solve "
            "it need 400k–1M examples and reach at most 72% on the longer contexts (E61). <b>Language, stage 1</b> (counting experts, copy memories and word-keyed memories, mixed "
            "by conserved multiplicative credit), measured on text8 test text at 1M, 10M and 90M training characters: the "
            "mixture reaches 2.00, 1.79 and 1.65 bits per character (E63), and word-keyed experts bring it to 1.98 and 1.73 at "
            "1M and 10M (E66). Stored contexts grow as D^0.41 and pairs as D^0.49. Counting alone (E62, 2.31 → 1.81) fits a "
            "floor near 1.73 bpc, which the mixture already passes: the floor belongs to the component, not to the design. "
            "<b>Mixing by a race</b> (§108: each candidate's clock rate is the weighted sum of the experts' log-probabilities, "
            "a product of experts, as the best compressors mix) takes the same experts at 10M characters from 1.80 to "
            "<b>1.61 bits per character</b>, with weights frozen after the validation text, at a few hundred operations "
            "per character (E79). Across single-seed runs, the 256-character-copy-window mixture scores 1.808 / 1.613 / "
            "1.504 frozen and 1.782 / 1.593 / 1.483 online at 1M / 10M / 90M characters. Expert count also rises 5 / 6 / 7, "
            "so this is a data-and-capacity scaling signal, not an isolated data-scaling law. Unbounded copy versus the "
            "256-character window gives 1.779 vs 1.808 at 1M, 1.612 vs 1.613 at 10M, and 1.512 vs 1.504 at 90M. Long-range "
            "copy helps modestly at 1M but has no measured advantage at larger scales; these single-seed differences have "
            "no uncertainty estimates. This supports bounded copy spans in scale tests while retaining long-range associative "
            "retrieval as a separate path. Published text8: standard LSTM ≈ 1.43, stronger recurrent models 1.27–1.36, large Transformers ≈ 1.08: "
            "at full scale near an LSTM, behind Transformers). For "
            "scale, large Transformers reach ≈ 1.1 on text8 from 90M characters. E64b's 1M-character, 20-pass, "
            "validation-selected test scores are 2.179 for the 256-unit LSTM and 2.367 for the 2-layer width-256 Transformer; "
            "both best checkpoints are at the final validation point, so strict convergence is not established. At 10M, the "
            "two-layer 512-unit LSTM scores 1.7993 held-out test bpc (1,199,323 parameters, six passes), also with "
            "its best checkpoint at the final validation point. E79 scores 1.613 frozen on the same test segment, a single-seed "
            f"0.186 bpc lead without matched compute. The four-layer Transformer completed 4,882 updates with "
            f"{tf_10m_final['test_bpc']:.4f} held-out test bpc after validation-based checkpoint selection. "
            f"It has 3.24M parameters and four passes, while the LSTM has "
            "1.20M parameters and six passes. "
            "E77 has not yet produced a scale-level language-model result. A default-width depth-4 diagnostic found "
            "that the raw voltage quantile left activity nearly absent: test rates were [0, 0.007, 0.009, 0.004], "
            "with nonzero gradients at only [1, 9, 12, 6] of 13 validation points. Exact replay of threshold/reset "
            "dynamics matched initial rates to [0.096, 0.082, 0.094, 0.100] and reached all four layers at all 13 "
            "points. The width-8 depth-8 pilot reached all eight layers at all 16 points, with test rates "
            "[0.115, 0.125, 0.217, 0.075, 0.081, 0.138, 0.124, 0.121]. This demonstrates gradient reach in a "
            "small model, not language-model quality or a scaling law.")]
    s += fig(FM.fig_lm_topology, W)
    s += [P("<b>E77 exact activity matching and depth-8 gradient reach (§§133–134).</b> At default width, the raw "
            "voltage-quantile initialization yielded selected-checkpoint test activity [0, 0.007, 0.009, 0.004] "
            "spikes/character and nonzero gradients at only [1, 9, 12, 6] of 13 validation points. Replaying the "
            "exact reset recurrence on saved voltage traces matched initial activity to [0.096, 0.082, 0.094, 0.100] "
            "around the 0.1 target and restored gradients at all four layers at all 13 points. Rates later rose to "
            "[0.204, 0.190, 0.456, 0.802], indicating representation drift; online homeostasis is not implemented. "
            "A width-8 depth-8 pilot gave nonzero gradients at all eight layers for all 16 validation points; test "
            "rates were [0.115, 0.125, 0.217, 0.075, 0.081, 0.138, 0.124, 0.121]. Its test BPC was 4.319 on "
            "512 characters (the depth-4 control was 4.254), too small to compare quality. A fresh depth-4 route "
            "shadow pass found mean opening-minus-closing loss deltas ± trajectory-cluster SE of [+0.000360 ± "
            "0.000218, −0.001115 ± 0.000325, +0.000220 ± 0.000278, −0.000435 ± 0.000290]; adjacent updates "
            "share a moving model. Counterfactual/pathwise norm ratio was 0.77%, cosine 0.0040, so the correction "
            "remains disabled. The result is a trainability indication, not model-quality or supremacy evidence.")]
    s += fig(fig_e77_bootstrap_diagnostics, W * 0.96)
    s += [P("<b>The plan, in stages, on character-level text (text8):</b> (1) a counting baseline with a copy memory (measured, "
            "E62–E66, above), not the goal but a measurement of how memory and loss scale with data; (2) attention over the stream, by "
            "races (E61 at scale; E68) or by content-dependent delays (§105: exact, and as cheap as the attention is sharp; E76 "
            "measures how sharp a trained model's attention on text is; E77 is the first full model of this kind); (3) learned shared codes; (4) stacked layers with credit along causal chains and near "
            "misses. At each stage: a recurrent network and a Transformer trained by gradients on the same text; bits per "
            "character, examples needed, work per character.")]
    s += fig(FM.fig_lm_scaling, W)
    s += [P("<b>Established and not.</b> Established: expressive equivalence, locality of exact credit for races, optimality of "
            "the depth bound, exact delay-coded attention and its work law (theory, checked), learned attention on a recall task from "
            "≥ 250× less data than a Transformer, data efficiency on deep order. Not yet shown: that stacked "
            "race-attention layers with learned codes, trained by local credit, match or beat a Transformer on language itself; "
            "the stages decide it.")]
    s.append(Spacer(1, 6))
    s.append(P("Every mechanism is an event handler (local state, triggered by events, cost proportional to events); dense "
               "procedures are diagnostics only; time-vector networks (E73–E75) are trained by gradients that flow only through "
               "spikes that occurred, simulated on a 1 ms grid for speed (an event-driven adjoint form exists). Reproduce: python report/figures_time.py && python report/make_pdf.py.",
               "small"))
    s += [PageBreak(), P("11. Potential applications and the transformation", "h1"),
          P("If deep event models learn Transformer-level representations and remain trainable as data, depth, and memory "
            "grow, this could open a different route to frontier AI. Training and inference would spend computation on "
            "messages that fire, memories that are retrieved, and parameters that receive useful credit. A model could keep "
            "a large associative store and spend work on the information each prediction actually uses. Lower cost per token "
            "would expand the number and scale of experiments a fixed research budget can support, and make capable models "
            "cheaper to serve continuously."),
          P("Applications", "h2")]
    s += bullets([
        "<b>Language and knowledge work.</b> Deep models could combine persistent event memory, recurrent state, and "
        "key/value retrieval to reason across long-running projects without reprocessing every token in a large dense "
        "context. Lower inference cost would make capable personal and organizational assistants practical to run more often.",
        "<b>Autonomous mobile platforms.</b> Phones, wearables, vehicles, and robots receive asynchronous camera, audio, "
        "motion, and location streams. Event routing could keep perception and decisions local, respond to salient changes, "
        "and retrieve relevant past observations. That could reduce cloud round trips, conserve battery during quiet periods, "
        "keep sensitive sensor data on the device, and preserve useful autonomy while disconnected.",
        "<b>Robotics and industry.</b> Machines could pair fast event reactions with selective recall, adapting to changing "
        "workflows without running a dense model over every sensor frame. Applications include inspection, logistics, "
        "process control, and collaborative machines.",
        "<b>Scientific and environmental sensing.</b> Instruments could analyze rare events continuously, retain causal "
        "context, and coordinate through compact messages rather than transmitting every raw sample.",
    ], st)
    s += [P("Economics and industry shift", "h2"),
          P("At frontier quality, the main benefit would be a new compute scaling curve for both training and inference. "
            "Fewer dense operations and unnecessary weight updates would reduce accelerator-hours and energy per useful "
            "token. The same capital and power envelope could support larger training runs, broader ablations, more continual "
            "adaptation, or more users. Demand would move toward high-bandwidth memory near compute, sparse routing networks, "
            "rapid event resolution, and associative stores. Data centers could become heterogeneous: GPUs for dense kernels, "
            "event-capable processors for sparse temporal work, and memory-centric accelerators for associative retrieval. "
            "Architecture and investment decisions would track useful learning and retrieval throughput alongside dense FLOPs."),
          P("This would change the economics of frontier development. Research teams could explore more architectures at the "
            "same budget, service providers could lower inference cost, and capable models could reach devices and organizations "
            "that cannot justify today's energy and infrastructure footprint. Scaling learned routes and associative memory "
            "would reshape model software, accelerator design, data-center layout, and the products built on them."),
          P("Mobile autonomy", "h2"),
          P("The most visible change could be a device that watches and listens continuously while using little power between "
            "meaningful events. A phone or robot could build a persistent local model of people, places, and ongoing tasks; "
            "retrieve relevant observations when something changes; and coordinate applications or physical actions without "
            "shipping a continuous sensor feed to a remote service. Fast local response, longer battery life, offline capability, "
            "and user-controlled memory could make autonomy a property of the platform itself. As autonomy grows, dependable "
            "permission boundaries, memory controls, and clear action records become core product capabilities."),
          P("The route from current evidence", "h2"),
          P("E79's native expert mixture leads the completed 1M text8 gradient baselines on the same split. E61 learns "
            "associative retrieval and context extrapolation with local credit. The theory supplies exact delay-coded attention "
            "and a linear-work associative scan for fixed event schedules. The decisive step is to make these capabilities "
            "work together in deep E77 language models, then measure matched quality, training cost, inference work, and energy "
            "on real hardware. This is a concrete path from promising mechanisms to a new frontier-computing paradigm.")]
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Sleeping Machines — what is known",
                            author="Sleeping Machines project")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
