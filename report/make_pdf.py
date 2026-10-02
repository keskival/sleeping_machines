#!/usr/bin/env python3
"""Build the status report PDF (report/sleeping_machines_status.pdf) from result files.

    python report/make_pdf.py

Charts are drawn with matplotlib from experiments/results and report/data.json;
the document is assembled with reportlab. Rerun after new results arrive.
"""
import glob
import html
import io
import json
import os
from datetime import date

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sleeping_machines-mpl")

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


def load_tf_10m_final():
    base = os.path.join(RES, "e64")
    for name in ("tf_D10000000_s256_L4_p4_dr0.1_v.json",
                 "tf_D10000000_s256_L4_p4_dr0.1_v_checkpoint.json"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            return load(path)
    raise FileNotFoundError("No completed E64 10M Transformer result JSON")


def aws_tf_1m_test_bpc():
    """Return held-out scores from completed AWS reruns of the E64 1M Transformer."""
    values = []
    for provenance_path in glob.glob(os.path.join(RES, "aws_20260929", "*", "provenance.json")):
        try:
            provenance = load(provenance_path)
            args = provenance.get("arguments", [])
            if (provenance.get("status") != "completed" or
                    provenance.get("script") != "experiments/e64_lm_baselines.py" or
                    "--model" not in args or args[args.index("--model") + 1] != "tf" or
                    "--D" not in args or args[args.index("--D") + 1] != "1000000"):
                continue
            for result_path in glob.glob(os.path.join(os.path.dirname(provenance_path), "*.json")):
                if result_path == provenance_path:
                    continue
                result = load(result_path)
                value = result.get("test_bpc")
                if isinstance(value, (int, float)) and np.isfinite(value):
                    values.append(float(value))
        except (OSError, ValueError, TypeError, IndexError):
            continue
    return values


def aws_e68_text_test_bpc():
    """Return completed E68 text8 controls as (race count, test BPC) pairs."""
    values = []
    for provenance_path in glob.glob(os.path.join(RES, "aws_20260929", "*", "provenance.json")):
        try:
            provenance = load(provenance_path)
            args = provenance.get("arguments", [])
            if (provenance.get("status") != "completed" or
                    provenance.get("script") != "experiments/e68_race_transformer.py" or
                    "--task" not in args or args[args.index("--task") + 1] != "text" or
                    "--R" not in args):
                continue
            races = int(args[args.index("--R") + 1])
            for result_path in glob.glob(os.path.join(os.path.dirname(provenance_path), "text_R*_s*.json")):
                result = load(result_path)
                curve = result.get("curve", [])
                if curve and isinstance(curve[-1].get("test_bpc"), (int, float)):
                    value = float(curve[-1]["test_bpc"])
                    if np.isfinite(value):
                        values.append((races, value))
        except (OSError, ValueError, TypeError, IndexError):
            continue
    return values


def aws_e77_scale_test_bpc():
    """Return completed E77 text8 runs with at least 1M training characters."""
    values = []
    for provenance_path in glob.glob(os.path.join(RES, "aws_20260929", "*", "provenance.json")):
        try:
            provenance = load(provenance_path)
            args = provenance.get("arguments", [])
            if (provenance.get("status") != "completed" or
                    provenance.get("script") != "experiments/e77_tv_lm.py" or
                    "--D" not in args):
                continue
            data_size = int(args[args.index("--D") + 1])
            if data_size < 1_000_000:
                continue
            for result_path in glob.glob(os.path.join(os.path.dirname(provenance_path), "*.json")):
                if result_path == provenance_path:
                    continue
                result = load(result_path)
                value = result.get("test_bpc")
                if isinstance(value, (int, float)) and np.isfinite(value):
                    values.append((data_size, float(value)))
        except (OSError, ValueError, TypeError, IndexError):
            continue
    return values


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
    raise RuntimeError('Retired: target-leaked evidence cannot be plotted; use the readable report')


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
             "Seed 6; depth 4; 120 train / 128 held-out-speaker utterances; deepest readout; two epochs. "
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
             "Seed 6; depth 4; 120 train / 128 held-out-speaker utterances; six epochs; two future rollouts/action. "
             "One seed; rollout reserve is proposal-conditioned.",
             fontsize=4.8, color=MUTED)
    fig.subplots_adjust(top=0.79, bottom=0.2, left=0.08, right=0.985, wspace=0.55)
    out = os.path.join(os.path.dirname(__file__), "figures",
                       "e83_optionality_state_value.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def fig_e83_conditioned_margin_support():
    """Show the depthwise collapse in actual-message-conditioned spike support."""
    result = load(os.path.join(
        RES, "e83", "conditioned_margin_support_bundle_control_s6_n128.json"))
    layers = np.arange(1, 5)
    examples = result["examples"]
    strict = np.asarray([x["examples_with_candidate"][0]
                         for x in result["per_layer"]], dtype=float)
    band = np.asarray([x["examples_with_candidate_within_0p5"]
                       for x in result["per_layer"]], dtype=float)
    strict_cells = np.asarray([x["candidate_counts"][0]
                               for x in result["per_layer"]], dtype=float)
    extra_cells = np.asarray([x["candidate_counts"][1]
                              for x in result["per_layer"]], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.65))
    axes[0].plot(layers, 100 * strict / examples, marker="o", linewidth=1.3,
                 color=BLUE, label="margin [−0.25, 0)")
    axes[0].plot(layers, 100 * band / examples, marker="o", linewidth=1.3,
                 color=ORANGE, label="margin [−0.5, 0)")
    axes[0].set_xticks(layers, ["L1", "L2", "L3", "L4"])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("utterances with ≥1 candidate (%)")
    axes[0].set_title("A · In-band proposal coverage")
    axes[0].legend(fontsize=5.8, loc="upper right")
    width = 0.34
    axes[1].bar(layers - width / 2, strict_cells, width=width, color=BLUE,
                label="[−0.25, 0)")
    axes[1].bar(layers + width / 2, extra_cells, width=width, color=ORANGE,
                label="[−0.5, −0.25)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(layers, ["L1", "L2", "L3", "L4"])
    axes[1].set_ylabel("time-receiver cells (log scale)")
    axes[1].set_title("B · Candidate cells by margin")
    axes[1].legend(fontsize=5.8, loc="upper right")
    fig.suptitle("E83 · near-threshold proposal support contracts with depth",
                 x=0.02, ha="left", fontsize=8.7, fontweight="bold")
    fig.text(0.02, 0.005,
             "Frozen seed-6 depth-4 control; 128 held-out-speaker utterances. Each candidate has an actual selected upstream message arrival. "
             "Time cells are correlated; no updates.", fontsize=4.8, color=MUTED)
    fig.subplots_adjust(top=0.83, bottom=0.2, left=0.09, right=0.99, wspace=0.45)
    out = os.path.join(os.path.dirname(__file__), "figures",
                       "e83_conditioned_margin_support.png")
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
    paths = glob.glob(os.path.join(RES, "e36", "transformer_e27*.json"))
    paths += glob.glob(os.path.join(RES, "aws_20260929", "*", "transformer_e27*.json"))
    for path in paths:
        if os.path.join("aws_20260929", "") in path:
            try:
                if load(os.path.join(os.path.dirname(path), "provenance.json")).get("status") != "completed":
                    continue
            except (OSError, ValueError, TypeError):
                continue
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


def fig_e119_work_and_learning():
    audit_path = os.path.join(RES, "e119", "scan_audit_s6.json")
    training_path = os.path.join(RES, "e119", "race_d8_linear_n1024_e8_s6.json")
    if not all(os.path.exists(p) for p in (audit_path, training_path)):
        return None
    audit, training = load(audit_path), load(training_path)
    if any(x.get("status") != "completed" for x in (audit, training)):
        return None
    f, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    curve = training["curve"]
    epochs = [r["epoch"] for r in curve]
    for split, label, color in (("fit", "Fitting speakers", AQUA), ("dev", "Held-out speakers", BLUE)):
        axes[0, 0].plot(epochs, [r[split]["nll"] for r in curve], "o-", color=color, label=label)
        axes[0, 1].plot(epochs, [100*r[split]["accuracy"] for r in curve], "o-", color=color, label=label)
    axes[0, 0].set_title("Terminal classification loss")
    axes[0, 0].set_ylabel("NLL (nats)")
    axes[0, 1].set_title("Eight-layer recognition")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].axhline(40.625, ls=":", color=GRAY, label="Earlier 512-fit pilot")
    for ax in axes[0]:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=6)
    for i, kind in enumerate(("doubling", "linear")):
        timing = audit["median_timings"][kind]
        axes[1, 0].bar(i-.16, timing["inference_s"]*1000, width=.3, color=AQUA,
                       label="Inference" if i == 0 else None)
        axes[1, 0].bar(i+.16, timing["training_step_s"]*1000, width=.3, color=BLUE,
                       label="Forward + backward" if i == 0 else None)
        count = audit["dev"][kind]["scan_compositions"]/1e6
        axes[1, 1].bar(i, count, color=(GRAY, BLUE)[i])
        axes[1, 1].text(i, count+.3, f"{count:.2f}", ha="center", fontsize=8)
    axes[1, 0].set_title("Same checkpoint: measured CPU time")
    axes[1, 0].set_ylabel("ms / batch of 4")
    axes[1, 0].legend(fontsize=6)
    for ax in axes[1]:
        ax.set_xticks([0, 1], ["Doubling scan", "Linear-work scan"])
    pareto_path = os.path.join(RES, "e119", "packet_pareto_d8_n1024_s6.json")
    pareto = load(pareto_path) if os.path.exists(pareto_path) else None
    if pareto and pareto.get("status") == "completed":
        ax = axes[1, 1]
        ax.clear()
        rows = [pareto["rows"][str(k)] for k in (1, 2, 4, 8)]
        ax.plot([100*r["packet_ratio_to_10ms"] for r in rows],
                [100*r["accuracy"] for r in rows], "o-", color=BLUE)
        for row in rows:
            ax.annotate(f'{row["packet_window_ms"]} ms',
                        (100*row["packet_ratio_to_10ms"], 100*row["accuracy"]),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
        ax.set_title("Frozen final model: causal coalescing")
        ax.set_xlabel("Packets (% of 10 ms input)")
        ax.set_ylabel("Held-out accuracy (%)")
        ax.margins(x=.16, y=.3)
    else:
        axes[1, 1].set_title("Memory scan: counted work")
        axes[1, 1].set_ylabel("Million vector combines / 256 examples")
        axes[1, 1].set_ylim(0, 25)
    f.suptitle("More learning per unit of event computation", fontsize=11)
    f.text(.02, .015, "Top: 1,024 fitting / 256 development examples; fixed cosine schedule, seed 6; no official test access.\n"
           "Bottom-left: exact scan change, earlier checkpoint, 8 warm timing repeats. Bottom-right: input coalescing changes the model's input.\n"
           "CPU timing and packet counts are distinct from joules. Coalescing can add input delay; official test set remains untouched.", fontsize=6.5)
    f.tight_layout(rect=(0, .095, 1, .95))
    f.savefig(os.path.join(os.path.dirname(__file__), "figures", "e119_work_and_learning.png"), dpi=180)
    return f


def fig_e118_race_carriers():
    paths = [os.path.join(RES, "e118", f"race_d8_cf{k}_n128_e8_s6.json") for k in (0, 1)]
    probe_path = os.path.join(RES, "e117", "probe_serial_d8_n128_e8_s6.json")
    if not all(os.path.exists(p) for p in [*paths, probe_path]):
        return None
    arms = [load(p) for p in paths]
    if any(a.get("status") != "completed" for a in arms):
        return None
    probes = load(probe_path)["rows"]
    conditioning = load(os.path.join(RES, "e117", "readout_conditioning_d8_n128_e8_s6.json"))["arms"]
    f, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    for arm, color, label in zip(arms, [GRAY, BLUE], ["Winner pathwise", "+ loser score credit"]):
        curve = arm["curve"]
        epochs = [row["epoch"] for row in curve]
        axes[0, 0].plot(epochs, [r["fit"]["nll"] for r in curve], color=color, label=label + " fit")
        axes[0, 0].plot(epochs, [r["dev"]["nll"] for r in curve], "--", color=color)
        axes[0, 1].plot(epochs, [100*r["dev"]["accuracy"] for r in curve], "o-", color=color, label=label)
    axes[0, 0].axhline(np.log(20), color=INK, ls=":")
    axes[0, 0].set_title("Terminal NLL: fit solid / held-out dashed")
    axes[0, 0].legend(fontsize=6)
    axes[0, 1].axhline(5, color=INK, ls=":")
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].set_title("Held-out-speaker accuracy (%)")
    axes[0, 1].legend(fontsize=6)
    for i, (name, label) in enumerate((("raw", "Raw features"), ("whitened", "Fit-only whitening"))):
        last = conditioning[name][-1]
        axes[1, 0].bar(i-.16, 100*last["fit_correct"]/128, width=.3, color=AQUA,
                       label="Fit" if i == 0 else None)
        axes[1, 0].bar(i+.16, 100*last["dev_correct"]/128, width=.3, color=ORANGE,
                       label="Held-out" if i == 0 else None)
    axes[1, 0].set_xticks([0, 1], ["Raw features", "Fit-only whitening"])
    axes[1, 0].set_title("Frozen features: conditioning (%)")
    axes[1, 0].legend(fontsize=6)
    for i, depth in enumerate((1, 8)):
        scaled = load(os.path.join(RES, "e118", f"race_d{depth}_cf1_n512_e4_s6.json"))
        last = scaled["curve"][-1]
        axes[1, 1].bar(i-.16, 100*last["fit"]["accuracy"], width=.3, color=AQUA,
                       label="Fit" if i == 0 else None)
        axes[1, 1].bar(i+.16, 100*last["dev"]["accuracy"], width=.3, color=BLUE,
                       label="Held-out" if i == 0 else None)
    axes[1, 1].set_xticks([0, 1], ["1 layer", "8 layers"])
    axes[1, 1].set_title("More data: 1 vs 8 layers (%)")
    axes[1, 1].legend(fontsize=6)
    axes[1, 1].set_ylim(bottom=0)
    for ax in axes[0]:
        ax.set_xlabel("Epoch")
    f.suptitle("Eight layers with real winning continuations: a development diagnostic", fontsize=10)
    f.text(.02, .015, "Top and bottom-left: 128 fit / 128 held-out, 8 epochs. Bottom-right: 512 / 256, 4 epochs. One seed.\n"
           "Depth comparison shares width/data/updates, with 7,537 vs 53,296 parameters and more event work at depth 8.", fontsize=6.5)
    f.tight_layout(rect=(0, .095, 1, .95))
    f.savefig(os.path.join(os.path.dirname(__file__), "figures", "e118_race_carriers.png"), dpi=180)
    return f


def fig_e83_countmark_coupling():
    path = os.path.join(RES, "e83", "countmark_matched_20260929.json")
    if not os.path.exists(path):
        return None
    data = load(path)
    names = ["off", "additive", "address_neutral"]
    labels = ["No mark", "Shared vector", "Payload only"]
    palette = [GRAY, BLUE, ORANGE]
    f, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    accuracy = [100 * data["arms"][name]["terminal_correct"] / 256 for name in names]
    axes[0, 0].bar(labels, accuracy, color=palette)
    axes[0, 0].axhline(5, color=INK, ls=":")
    axes[0, 0].set_ylim(0, 10)
    axes[0, 0].set_title("Final terminal accuracy (%)")
    for i, name in enumerate(names):
        rows = data["arms"][name]["curve"]
        epochs = [row["epoch"] for row in rows]
        axes[0, 1].plot(epochs, [100 * row["event_support_coverage"][-1] for row in rows],
                        "o-", color=palette[i], label=labels[i])
        axes[1, 0].plot(epochs, [row["train_loss"] for row in rows], "o-", color=palette[i])
        axes[1, 1].plot(epochs, [row["prefix_window_nll_by_stratum"][-1] for row in rows],
                        "o-", color=palette[i])
    axes[0, 1].set_title("Utterances reaching layer 4 (%)")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].set_title("Training primary loss (log scale)")
    axes[1, 1].set_title("Late-prefix validation NLL (log scale)")
    for ax in axes[1]:
        ax.set_yscale("log"); ax.axhline(np.log(20), color=INK, ls=":")
    for ax in [axes[0, 1], *axes[1]]:
        ax.set_xticks([1, 2]); ax.set_xlabel("Epoch")
    f.suptitle("Input marks change deep activity; recognition remains at chance", fontsize=10)
    f.text(.02, .015, "512 fit / 256 held-out-speaker utterances; D4; corrected grid emission; one seed.\n"
           "The payload-only arm was chosen after the first pair. Dotted references: 5% accuracy and log(20) loss.", fontsize=6.5)
    f.tight_layout(rect=(0, .095, 1, .95))
    f.savefig(os.path.join(os.path.dirname(__file__), "figures", "e83_countmark_coupling.png"), dpi=180)
    return f


def fig_emission_contract():
    audit = load(os.path.join(RES, "e83", "emission_contract_s6_n128.json"))
    prefix = "deep_d8_n4_M16-16_depth4_aux0.2_objevent_prefix_rfall_depths_rngsplit_"
    suffix = "_cfnorm1_b0.5_sg0.25_w1_dl5_lr0.001_gc1_spk_s6.json"
    control = load(os.path.join(RES, "e83", prefix + "bundle_control" + suffix))
    grid = load(os.path.join(RES, "e83", prefix + "emission_grid_recongrid" + suffix))
    if not audit or not control or not grid:
        return None
    f, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for ax, key, title in zip(axes[0], ["payload", "payload_derivative"],
                              ["Single-arrival emitted payload", "Derivative with respect to payload"]):
        vals = [r[key][0][0] for r in audit["witness"]]
        ax.bar(["Legacy", "Grid reference"], vals, color=[GRAY, BLUE])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_title(title, fontsize=9)
    for rows, label, color in [(control["curve"], "Legacy", GRAY), (grid["curve"], "Grid reference", BLUE)]:
        epochs = [r["epoch"] for r in rows]
        axes[1, 0].plot(epochs, [100 * r["event_support_coverage"][-1] for r in rows], "o-", color=color, label=label)
        axes[1, 1].plot(epochs, [100 * r["event_terminal_accuracy"] for r in rows], "o-", color=color, label=label)
    axes[1, 0].set_title("Layer-4 event coverage (%)", fontsize=9)
    axes[1, 0].set_ylim(0, 105)
    axes[1, 1].set_title("Held-out terminal accuracy (%)", fontsize=9)
    axes[1, 1].axhline(5, color=INK, ls=":", label="20-class chance")
    axes[1, 1].set_ylim(0, 20)
    for ax in axes[1]:
        ax.set_xlabel("Epoch"); ax.set_xticks([1, 2, 3, 4]); ax.legend(fontsize=7)
    f.suptitle("A corrected payload contract restores propagation; recognition remains unresolved", fontsize=9)
    f.text(0.02, 0.01, "Top: exact one-event diagnostic. Bottom: one matched seed, 120 fit / 128 held-out-speaker utterances; four epochs.\n"
           "The grid reference also changes spike time and reset discretization. This is not a supremacy result.", fontsize=6.5)
    f.tight_layout(rect=(0, .09, 1, .94))
    f.savefig(os.path.join(os.path.dirname(__file__), "figures", "e83_emission_contract.png"), dpi=180)
    return f


def fig_optionality_contract():
    data = load(os.path.join(RES, "e116", "optionality_contract_v2.json"))
    if not data:
        return None
    f, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    labels = ["Complementary", "One volatile", "Duplicates"]
    vals = [data[k]["premium"] for k in ["complementary_routes", "one_volatile_route", "duplicate_routes"]]
    axes[0].bar(labels, vals, color=[BLUE, GRAY, GRAY]); axes[0].set_ylim(0, 1.3)
    axes[0].set_title("Value of choosing after observed evidence", fontsize=8)
    x = np.arange(2)
    for i, (key, label, color) in enumerate([("same_sample", "Same-sample score", ORANGE),
                                           ("independent_sample_transfer", "Independent transfer", BLUE)]):
        axes[1].bar(x + (i - .5) * .32, [data[k][key] for k in ["zero_mean_noisy_gradient", "consistent_gradient"]],
                    width=.32, label=label, color=color)
    axes[1].set_xticks(x, ["Noisy ±2", "Consistent +1"])
    axes[1].set_title("Virtual learning can reward gradient noise", fontsize=8)
    axes[1].legend(fontsize=6.5)
    f.suptitle("Optionality contracts: exact mathematical examples", fontsize=10)
    f.text(.02, .015, "Equiprobable scenarios; identity update metric. These illustrate theory, not SHD performance.", fontsize=6.5)
    f.tight_layout(rect=(0, .07, 1, .94))
    f.savefig(os.path.join(os.path.dirname(__file__), "figures", "optionality_contract.png"), dpi=180)
    return f


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


def build_legacy():
    raise RuntimeError('Retired: legacy report included target-leaked comparisons; use build()')


def build():
    from readable_report import build as build_readable
    build_readable(globals())


if __name__ == "__main__":
    build()
