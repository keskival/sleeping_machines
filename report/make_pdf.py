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
    import figures_theory
    figures_theory.main()
    d = load(os.path.join(ROOT, "report", "data.json"))
    runs = e6_runs()
    st = styles()
    W = 174
    r3 = {v: runs.get((v, 3, 1000)) for v in ("crl_fa", "crl_fired_only", "frozen_hidden", "single_layer")}
    h2000 = runs.get(("crl_fa", 3, 2000))
    e7_done = sorted(glob.glob(os.path.join(RES, "e7", "*.json")))

    def pct(x):
        return f"{100 * x:.1f}%" if x is not None else "pending"

    s = []
    s += [Paragraph("Sleeping Machines", st["title"]),
          Paragraph(f"Computing with races, cancellations and near misses · status report, {date.today():%d %B %Y}",
                    st["sub"])]
    s += [Paragraph("In one page", st["h1"]),
          Paragraph("Sleeping Machines proposes that computation can happen <b>in time rather than memory</b>: "
                    "candidate events race, the first to fire cancels the rest, and the cancelled ones keep a trace of "
                    "how close they came, which a later teaching signal can use. Every experiment below had its "
                    "predictions written down before evaluation, and the report includes what did not work.",
                    st["body"]),
          Paragraph("What holds up", st["h2"])]
    s += bullets([
        "<b>A cancelled node can be taught (E4).</b> Keeping a loser's distance to threshold lets a delayed teacher "
        "promote the right answer. At K = 128 classes: 0.79 accuracy where the reward-modulated rule is at chance, "
        "with 6% of the weight updates of uniform credit.",
        "<b>Races decide as fast as the evidence allows (E2).</b> An accumulator race beats a fixed-time decoder at "
        "every decision time and tracks the optimal MSPRT, using only additions and a threshold.",
        "<b>Learning work tracks activity, not capacity (E5):</b> about 300× fewer weight updates than a sparse "
        "softmax on the same connectivity.",
        f"<b>Local, event-driven learning reaches about 96% on MNIST (E6, E14).</b> One hidden layer "
        f"{pct(r3['crl_fa'])} (test, round 3); with credit conservation 96.0% (validation, 2 seeds).",
        "<b>Counterfactual credit pays with depth (E14, full length).</b> Over fired-only credit: +0.2, +1.2, +1.5 "
        "points at depths 1–3 (2 seeds), +1.9 and +2.5 at depths 4–5 (1 seed). A frozen hidden stack collapses.",
        "<b>Credit conservation, predicted by the theory, holds at full length:</b> +1.0 to +1.7 points at every "
        "depth, both seeds.",
    ], st)
    s += [Paragraph("What does not (yet)", st["h2"])]
    s += bullets([
        "<b>Depth still costs accuracy</b> (0.960, 0.952, 0.941 at depths 1–3 with conservation). The theory now "
        "names three reasons (credit contraction, activity drift, pattern chaos), each with a predicted remedy, "
        "all untested.",
        "<b>Energy.</b> Only the single racing layer beats an equally accurate dense model at inference (about 2.4×). "
        "Dense hidden layers cost more than a small MLP; sparse fan-in (14–22× fewer events) is the candidate fix.",
        "<b>Leads that reversed:</b> the shadow neuron (debug +5 points, full length −0.8); the counterfactual "
        "routing gradient in MoE (a tie with load balancing at 10 seeds).",
        "<b>Market stream (E17): no edge.</b> The race matches simple baselines while deciding a third earlier, "
        "but continual learning did not help, learned trade selection had no skill, and every learner loses money "
        "after costs.",
        "<b>Not yet run:</b> the E7 stream learner; most theory predictions (M31–M43).",
    ], st)
    s += [Paragraph("What is new in the theory (details in the theory pages)", st["h2"])]
    s += bullets([
        "<b>Exact results:</b> timing credit sums to the deadline's credit (a Ward identity); excitatory race "
        "networks are topical maps, so timing noise is never amplified and a decision comes with a certified "
        "jitter radius; committing to a branch costs temperature × surprisal, and near-miss credit is the gradient "
        "of that cost.",
        "<b>Mechanisms, predicted and queued for test:</b> why deep credit dies (contraction by a Markov kernel; "
        "an activity mode that swamps evidence), why prices must be the faster timescale, and why firing "
        "<i>patterns</i> are chaotic even though firing times are not.",
        "<b>Checked against the literature:</b> several pieces turned out to be prior art and are credited (for "
        "example the polyhedral geometry of first-spike networks, the outlier mode in non-negative backprop).",
    ], st)
    s.append(PageBreak())
    s += promising_page(st, W)

    s += [Paragraph("The idea", st["h1"]),
          Image(os.path.join(ROOT, "report", "figures", "race.png"), width=W * mm, height=W * mm * 500 / 1080),
          Paragraph("Nodes are non-leaky integrate-to-threshold units. The first to reach threshold fires and cancels "
                    "the others; each cancelled node freezes Δ, its normalised distance to threshold. A teaching event "
                    "arriving later can then credit near misses, not only the node that fired. The simulator has no "
                    "global clock: it jumps from event to event and counts every operation, and all work and energy "
                    "figures come from those counts.", st["body"]),
          Paragraph("<b>How a decision unfolds (weaving).</b> Each group of neurons holds an open set of possible "
                    "futures: every member's projected crossing time, which only moves earlier as input arrives. A "
                    "crossing is <i>woven</i>, fixed history. When k members have crossed, the group closes, and each "
                    "loser's distance to threshold is frozen as a near miss. Spikes from closed groups drive the next "
                    "layer, so the settled region spreads through the network as a diagonal front in layers and time, "
                    "until the first output crossing decides. Learning rereads the woven record, including the near "
                    "misses. (experiments/WEAVING.md)", st["body"])]
    s += [Paragraph("E2 · decisions that take as long as they need", st["h2"]), fig_image(fig_e2(d), W),
          Paragraph("The race dominates a fixed-time decoder at every matched decision time and follows the MSPRT, "
                    "which knows the exact likelihoods. Easy trials end in 0.29 s, hard ones in 1.69 s, with no "
                    "controller deciding when to stop.", st["body"])]
    s.append(PageBreak())
    s += [Paragraph("E4 · counterfactual credit", st["h1"]), fig_image(fig_e4(d), W),
          Paragraph("Rules that credit only nodes that fired can only punish a wrong winner; as K grows the right "
                    "answer almost never fires, and they collapse to chance. Counterfactual rules can promote the "
                    "target from its frozen record. Crediting only competitors that came close is both the most "
                    "accurate and the cheapest local rule. It still trails the global-gradient reference by 13 points "
                    "at K = 128, and credit survives a delayed teacher only until another race overwrites the record.",
                    st["body"])]
    s += [Paragraph("E5 · capacity that costs nothing while it sleeps", st["h2"]), fig_image(fig_e5(), W),
          Paragraph("The fair baseline is a sparse softmax on the same connectivity, because most of what event "
                    "computing saves comes from sparsity itself. Against it, inference work is similar (the race "
                    "saves about 20% by stopping early), while learning work is about 300× lower: only near misses "
                    "are updated. Dense models grow linearly with K.", st["body"])]
    s.append(PageBreak())

    s += [Paragraph("E6 · hidden layers on MNIST", st["h1"]), fig_image(fig_e6(runs), W)]
    s += bullets([
        "Round 3 (current-based ramp synapses, a collapsing decision bound, 3 winners per group) lifted the hidden "
        "networks from about 0.89–0.90 to about 0.96.",
        "Hidden learning matters: the frozen random hidden layer stays at 0.895.",
        "With one hidden layer, fired-only ≈ counterfactual credit. From two layers on, counterfactual credit "
        "leads, and the gap grows with depth (E14, on the depth page). Symmetric feedback was worst in rounds 1–2 "
        "and was not carried forward.",
        f"A wider hidden layer (2000 nodes): {pct(h2000) if h2000 else 'pending'}, no gain. The single-layer "
        f"control reaches {pct(r3['single_layer'])}: the new synapse model gave ~2.5 points, depth ~4 more.",
    ], st)
    s += [Paragraph("Energy, measured by counting", st["h2"]), fig_image(fig_energy(), W),
          Paragraph("Operation counts priced with published per-operation energies (45 nm logic and SRAM; measured "
                    "Loihi). These are order-of-magnitude estimates, not chip measurements. At inference, the single "
                    "racing layer is the only network that beats an equally accurate dense model; the round-3 hidden "
                    "networks are 1.2–1.5× cheaper to <i>train</i> than unbatched dense training. Hidden layers integrate about "
                    "100 input events in each of 1000 nodes before inhibition stops them, and that swamps the savings. "
                    "Against batched dense hardware every advantage disappears.", st["body"])]
    s.append(PageBreak())

    s += theory_pages(st, W)

    s += [Paragraph("E7 · a learner that lives in a causal stream", st["h1"]),
          Paragraph("Everything above trained the way clocked hardware likes: minibatches, IID epochs, a global "
                    "learning-rate schedule, a label for every sample, a separate test phase. None of that is needed "
                    "by an event-driven learner; it is inherited. E7 drops it. The guiding rule is not biological "
                    "fidelity but this: <b>where we deviate from biology only because of synchronous-hardware habits, "
                    "we do not carry the deviation over.</b>", st["body"]),
          fig_image(fig_e7(), W)]
    s.append(table([
        ["habit", "origin", "in E7"],
        ["minibatch updates", "GPU throughput", "one update per frame (Stage 0 checks this changes nothing)"],
        ["epochs, reshuffling", "batched SGD", "one pass over a stream"],
        ["global learning-rate decay", "SGD convergence", "per-synapse consolidation"],
        ["homeostasis from batch means", "vectorised statistic", "per node, every frame, label or not"],
        ["dense feedback matrix", "cheap matmul", "counted as events to eligible nodes only"],
        ["train phase, then test", "benchmark protocol", "predict every frame before learning from it"],
        ["a label for every sample", "benchmark convention", "10% labels, late labels, or labels the learner asks for"],
        ["frames with a global t = 0", "datasets are frames", "kept as episode resets; slow state crosses them"],
        ["latency-coded dense images", "MNIST is a frame", "kept as a benchmark, not a sensor model"],
    ], [48, 38, 88], st))
    s += [Spacer(1, 6), Paragraph("Hypotheses (preregistered)", st["h2"])]
    s += bullets([
        "<b>H0</b> one frame per update learns as well as minibatches of 32.",
        "<b>H1</b> a carried prior cuts the input events needed to decide on later views, without hurting first views.",
        "<b>H2</b> at 10% labels, carried labels and continuity help on real episodes and not on the "
        "<b>shuffled-time control</b> (same frames, order destroyed).",
        "<b>H3</b> labels the learner asks for (when its race is uncertain) beat random labels on the same budget.",
        "<b>H4</b> synaptic tags make labels that arrive 2 frames late usable.",
        "<b>H5</b> on class-blocked streams the race forgets less than an MLP trained by SGD; replay is reported "
        "alongside and may beat both.",
    ], st)
    status = (f"{len(e7_done)} E7 result files so far." if e7_done else
              "Code, tests and the pilot queue are in place; the pilots run after the E6 jobs finish.")
    s.append(Paragraph(f"<b>Status.</b> {status} Pilots use the validation split to set thresholds, which are written "
                       "into the preregistration before 5-seed confirmatory runs on the test set.", st["body"]))
    s.append(PageBreak())

    s += [Paragraph("E17 · a continually learning race on a live market stream", st["h1"]),
          Paragraph("A test of the model class on real, non-stationary, asynchronous event data: Binance BTCUSDT "
                    "trades (millisecond timestamps), 7 pilot days and 21 confirmatory days, one stream. This is a "
                    "crypto market, not a stock market: freely available stock tick data with raw timestamps was not "
                    "found. Every 10 s the race restarts, trades stream in as spikes (side × size × tick direction), "
                    "and the network commits at its first output crossing, or abstains. The label is the price move "
                    "over the next 10 s <i>from the moment it decided</i>. Look-ahead is impossible by construction: "
                    "a decision uses only trades that already arrived, and weights are taught only with labels "
                    "revealed before the prediction. <b>Not a trading system; no live trading.</b>", st["body"])]
    s += bullets([
        "<b>Learners:</b> continual race; the same race frozen after the pilot days; a three-output race "
        "(up / down / <b>hold</b>) that learns <i>whether</i> a move will pay the 2 bp cost, i.e. when to trade; "
        "online logistic regression deciding at the end of each window; momentum.",
        "<b>Decision rules (preregistered):</b> competitive if within 1 point of logistic regression while deciding "
        "at least 20% earlier; continual learning helps if it beats frozen by 1 point (day-block interval excludes "
        "0); trade selection helps if the hold race's profit proxy beats the others at the same trade fraction. "
        "Near 50% for everyone is the expected null for 10 s direction.",
    ], st)
    e17 = {os.path.basename(p_)[:-5]: load(p_) for p_ in glob.glob(os.path.join(RES, "e17", "*.json"))
           if not p_.endswith("analysis.json")}
    if e17:
        rows = [["learner", "accuracy", "coverage", "profit bp/episode (2 bp)", "(10 bp)"]]
        for k, r in sorted(e17.items()):
            c = r["confirmatory"]
            rows.append([k, f"{c['acc']:.4f}", f"{c['coverage']:.3f}", f"{c['profit_bp_per_episode_c2']:+.3f}",
                         f"{c['profit_bp_per_episode_c10']:+.3f}"])
        s += [table(rows, [52, 26, 26, 42, 28], st),
              Paragraph("Confirmatory days only, prequential. Profit is a diagnostic proxy, not tradable P&amp;L.",
                        st["small"])]
        an_p = os.path.join(RES, "e17", "analysis.json")
        if os.path.exists(an_p):
            an = load(an_p)
            f = lambda v: f"{100 * v[0]:+.1f} points [{100 * v[1][0]:+.1f}, {100 * v[1][1]:+.1f}]"  # noqa: E731
            s += bullets([
                f"<b>Preregistered verdicts:</b> competitive (met: {100 * an['race_earlier_frac']:.0f}% earlier "
                f"decisions); nominally better than logistic regression ({f(an['race_minus_b1_acc'])}).",
                f"<b>A fairness check made after seeing the results overturns \u201cbetter\u201d:</b> at the race's "
                f"coverage, logistic regression is as accurate ({f(an['race_minus_b1_same_coverage_acc'])}); the race "
                f"ties momentum ({f(an['race_minus_b0_acc'])}). The edge came from abstaining on hard windows.",
                f"<b>Continual learning did not help:</b> continual − frozen {f(an['continual_minus_frozen_acc'])}.",
                f"<b>Learning when to trade:</b> the hold race traded {100 * an['hold_trade_fraction']:.1f}% of "
                f"episodes, at chance accuracy; its profit ties logistic regression's most confident trades at "
                f"the same rate. No trade-selection skill.",
                "<b>Every learner loses money after costs.</b> Direction on moves of at least 1 bp is predictable "
                "at about 59% (above the ~50% null the preregistration expected; checked for look-ahead), not "
                "enough to pay even a 2 bp cost.",
            ], st)
    else:
        s.append(Paragraph("<b>Status.</b> Data downloaded (28 days), preregistration written before any data was "
                           "inspected (experiments/E17_PREREGISTRATION.md), runs queued.", st["body"]))
    s.append(PageBreak())

    s += [Paragraph("Lessons learned along the way", st["h1"])]
    s += bullets([
        "<b>Compute discipline.</b> Running two heavy jobs at once hung the host three times (no swap, and no memory "
        "limit on the container; two hard reboots corrupted the filesystem). The third time, the queue ran one job while "
        "an ad-hoc debug run ran beside it. Now every computation, debug snippets included, goes through the one-job "
        "queue (watchdog at 6 GB free, checked every second), and the container gets hard memory and CPU caps.",
        "<b>Debug leads reverse.</b> The shadow neuron led by 5 points in 1-epoch runs and trailed by 0.8 at full "
        "length; credit conservation's +8–10 became +1–1.7. Short runs are reported as leads only.",
        "<b>Theory can be wrong in informative ways.</b> A predicted input-redundancy pyramid was refuted by measuring "
        "the data (entropy exponent 0.95, not 0.5–0.8); an early smoke test contradicts the predicted size of the "
        "activity mode. Both are recorded next to the claims they test.",
        "<b>Hidden batch assumptions.</b> E6 ran homeostasis inside the teaching step, which silently assumed every "
        "sample is taught. With scarce labels that would have switched homeostasis off. Batch-derived constants "
        "(homeostasis rate) also have to be rescaled when moving to one frame per update.",
        "<b>A case that batching hides.</b> A frame where no hidden node fires crashed the per-frame learner; in "
        "batches, some other sample always fired.",
        "<b>Undercounted training work.</b> The feedback fan-out and homeostasis updates were not counted. They are "
        "now, as events to eligible nodes only.",
        "<b>Controls before claims.</b> Round 3 looked like a win for counterfactual hidden credit until the fired-only "
        "ablation matched it. The single-layer and shuffled-time controls exist for the same reason.",
    ], st)
    s += [Paragraph("Where this could go", st["h1"])]
    s += bullets([
        "<b>Make depth pay:</b> the theory's three remedies, each with a queued test: centre credit in time "
        "coordinates (Ward identity), run the prices on the faster timescale, and use sparse fan-in with enough "
        "winners (k·F ≥ G) or topographic codes to damp pattern chaos.",
        "<b>Decide better, not only faster:</b> relative (MSPRT) stopping via a shared free-energy inhibition, and "
        "onset-referenced inhibition so the network can use the <i>absence</i> of expected spikes.",
        "<b>Make the hidden layer cheap:</b> sparse fan-in cut synaptic events 14–22× in pilots; if accuracy holds, "
        "the inference-energy verdict flips.",
        "<b>Live in time:</b> E7 (a causal stream with scarce, late labels) and E17 (a real market stream, with the "
        "network learning when to act) test the asynchronous, continual side of the proposal.",
        "<b>Seeds:</b> 3–5 seeds for every headline number before any claim.",
    ], st)
    s.append(Spacer(1, 8))
    s.append(Paragraph("Reproduce: <font face='DV'>python report/make_pdf.py</font> rebuilds this document from "
                       "experiments/results and report/data.json. Preregistrations: experiments/E*_PREREGISTRATION.md.",
                       st["small"]))

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Sleeping Machines — status report",
                            author="Sleeping Machines project")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
