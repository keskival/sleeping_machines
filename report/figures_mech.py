"""Explanatory and mechanism figures for the report (plain-language front section and §4).
Each fig_* returns a matplotlib figure; data come from experiments/results and report/data/mech.json."""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

import make_pdf as M

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "experiments", "results")
MECH = os.path.join(os.path.dirname(__file__), "data", "mech.json")
BLUE, ORANGE, AQUA, YELLOW, GRAY, INK, MUTED, GRID = M.BLUE, M.ORANGE, M.AQUA, M.YELLOW, M.GRAY, M.INK, M.MUTED, M.GRID
LIGHT_BLUE = "#cde2fb"
EVENT = ORANGE                      # the event network, in every figure
DENSE_T = BLUE                      # Transformers
DENSE_O = GRAY                      # other dense models
MOTIF_COLORS = (BLUE, ORANGE, AQUA)  # motifs A, B, C (first three validated slots)


def _load(p):
    with open(p) as f:
        return json.load(f)


# ── 1. The idea: clocked dense computation vs an event race ──────────────────────────────────────────────
def fig_concept():
    C, T = 12, 48
    spikes = [(1, 6), (3, 9), (7, 21), (8, 24), (4, 35), (2, 43)]      # (channel, time bin)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"wspace": 0.1})
    for ax in (a, b):
        ax.set_xlim(-0.5, T - 0.5); ax.set_ylim(-0.8, C - 0.2); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(T - 1, -0.75, "time →", ha="right", fontsize=7.5, color=MUTED)
    for c in range(C):
        for t in range(T):
            a.add_patch(Rectangle((t - 0.42, c - 0.38), 0.84, 0.76, color="#e8e7e2", lw=0))
    for c, t in spikes:
        a.add_patch(Rectangle((t - 0.42, c - 0.38), 0.84, 0.76, color=INK, lw=0))
    a.set_title("Clocked network: every cell, every tick", fontsize=9)
    a.set_ylabel("input channels", fontsize=7.5)
    for c in range(C):
        b.plot([0, T - 1], [c, c], color="#efeee9", lw=0.8, zorder=1)
    for c, t in spikes:
        b.plot(t, c, "o", color=INK, ms=4.5, zorder=3)
    b.add_patch(Rectangle((6, 0.5), 5, 3, color=BLUE, alpha=0.14, lw=0, zorder=0))
    b.plot(9, 3, "D", color=BLUE, ms=7, zorder=4)
    b.annotate("part detector: spike 2 arrives\nwithin a window after spike 1", (9.4, 3), xytext=(13, 1.2), fontsize=7,
               color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    b.add_patch(Rectangle((21, 6.5), 5, 2, color=AQUA, alpha=0.16, lw=0, zorder=0))
    b.plot(24, 8, "D", color=AQUA, ms=7, zorder=4)
    b.annotate("", xy=(23.6, 8.1), xytext=(9.3, 3.3),
               arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, connectionstyle="arc3,rad=-0.3"))
    b.plot(24.2, 9.6, marker="*", color=EVENT, ms=16, zorder=5)
    b.text(26, 9.6, "order detector: part 2 after part 1\n→ the first detector to fire is the answer", fontsize=7,
           color=INK, va="center")
    b.set_title("Event network: work only when a spike arrives", fontsize=9)
    fig.text(0.29, 0.02, "576 cells × every weight ≈ 10⁵–10⁶ multiply-adds", ha="center", fontsize=7.8, color=MUTED)
    fig.text(0.71, 0.02, "≈ 10 events; silence costs nothing", ha="center", fontsize=7.8, color=MUTED)
    return fig


# ── 2. The supremacy map: one panel per task, accuracy against work ───────────────────────────────────────
def _tf_points(pattern, long_only=None):
    pts = []
    for path in glob.glob(os.path.join(RES, "e36", pattern)):
        rows = _load(path)["rows"] if path.endswith(".json") else [json.loads(l) for l in open(path)]
        for r in rows:
            pts.append((r["macs_per_episode"], r["acc"], r.get("reltime", 0)))
    return pts


def _panel(ax, title, xlabel, ylabel):
    ax.set_xscale("log"); ax.set_title(title, fontsize=8.6)
    ax.set_xlabel(xlabel, fontsize=7.5); ax.set_ylabel(ylabel, fontsize=7.5)
    ax.tick_params(labelsize=7)


def fig_supremacy_map():
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6), gridspec_kw={"hspace": 0.62, "wspace": 0.3})
    # (a) timing patterns (E27 task): E35 event network vs Transformers and clocked conv nets
    ax = axs[0, 0]
    _panel(ax, "Timing patterns:\nsame accuracy, ~10⁵× less work", "operations per example (log)", "test accuracy")
    conv = []
    for path in glob.glob(os.path.join(RES, "e32", "dense_F*.json")):
        conv += [(r["macs_per_episode"], r["acc"]) for r in _load(path)["rows"] if r["pad"] == 0.0]
    conv = [p for p in conv if p[1] > 0.9]
    ax.scatter(*zip(*conv), s=14, color=DENSE_O, label="clocked conv nets", zorder=3)
    tf = [p for p in _tf_points("transformer_e27_long.json")] + [p for p in _tf_points("transformer_e27_rel.json")]
    ax.scatter([p[0] for p in tf], [p[1] for p in tf], s=16, marker="s", color=DENSE_T, label="Transformers (1–2M examples)", zorder=3)
    ax.scatter([7.5], [1.0], s=70, marker="D", color=EVENT, label="event network", zorder=4)
    ax.annotate("1.000 at 7.5 events", (7.5, 1.0), xytext=(9, -12), textcoords="offset points", fontsize=7, color=INK)
    ax.set_ylim(0.9, 1.006); ax.set_xlim(2, 1e7)
    ax.legend(fontsize=6.3, loc="lower left", bbox_to_anchor=(0.0, 0.28), markerscale=0.7)
    # (b) composition (E28/E34 task): chains vs Transformers at several data budgets (ordinal: light = less data)
    ax = axs[0, 1]
    _panel(ax, "Composing parts: Transformer-level accuracy\nfrom one pass, ~10⁴× less work", "operations per example (log)", "test accuracy")
    small = [(r["macs_per_episode"], r["acc"]) for r in _load(os.path.join(RES, "e36", "transformer_e28.json"))["rows"]]
    fixed = []
    for tag in ("n10k_wd0.1", "n40k_wd0.1", "n40k_wd0"):
        p = os.path.join(RES, "e36", f"transformer_e28_{tag}.json")
        if os.path.exists(p):
            fixed += [(r["macs_per_episode"], r["acc"]) for r in _load(p)["rows"]]
    big = [(p[0], p[1]) for p in _tf_points("transformer_e28_long_partial.jsonl")] + \
          [(p[0], p[1]) for p in _tf_points("transformer_e28_rel.json")]
    lo_s, hi_s = min(a for _, a in small), max(a for _, a in small)
    ax.text(0.98, 0.04, f"Transformer, 40k examples once: {lo_s:.2f}–{hi_s:.2f} (below the axis)", transform=ax.transAxes,
            ha="right", fontsize=6.6, color=MUTED)
    if fixed:
        ax.scatter(*zip(*fixed), s=18, marker="s", color="#86b6ef", label="Transformer, 10k–40k examples × 50–200 passes", zorder=3)
    ax.scatter(*zip(*big), s=16, marker="s", color=DENSE_T, label="Transformer, 1–2M examples", zorder=3)
    chains = _chain_plateau()
    if chains:
        ax.scatter([20] * len(chains), chains, s=40, marker="D", color=EVENT, label="event chains, 40k examples once", zorder=4)
        ax.annotate(f"{min(chains):.3f}–{max(chains):.3f}", (20, min(chains)), xytext=(8, -12), textcoords="offset points", fontsize=7)
    ax.set_ylim(0.94, 1.003); ax.set_xlim(5, 3e6)
    ax.legend(fontsize=6.3, loc="lower left", bbox_to_anchor=(0.02, 0.1), markerscale=0.7)
    # (c) world model of a real market stream (held-out days)
    ax = axs[1, 0]
    ax.set_title("World model of a real market stream:\nbeats a GRU at 80–200× less work", fontsize=8.6)
    rows = [("event network (semi-Markov)", [-2.38, -2.10], 19, EVENT, "D"),
            ("GRU point process, online", [-3.15, -2.98], 1.5e3, DENSE_O, "o")]
    for f, c, lab in _extra_world():
        rows.append((lab, f, c, DENSE_T, "s"))
    for name, vals, cost, col, mk in rows:
        ax.scatter([cost] * len(vals), vals, s=40 if col == EVENT else 18, marker=mk, color=col, zorder=4, label=name)
    ax.set_xscale("log"); ax.set_xlim(5, 3e5)
    ax.set_xlabel("operations per event (log)", fontsize=7.5); ax.set_ylabel("log-likelihood per event, held-out days\n(higher = better)", fontsize=7.2)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.3, loc="lower right", markerscale=0.7)
    # (d) grokking (a + b + c) mod 17 from 30% of the triples
    ax = axs[1, 1]
    ax.set_title("Grokking (a + b + c) mod 17 from 30%:\nthe event chain generalizes", fontsize=8.6)
    ev = [r["final"]["test"] for r in _load(os.path.join(RES, "e41", "p17_f0.3_lam0.05_sig2.21_c1.json"))["rows"]]
    tfr = _load(os.path.join(RES, "e36", "transformer_add3_add3.json"))["rows"]
    groups = [("event\nchain", ev, EVENT, "D"),
              ("Trans-\nformer 32", [r["acc"] for r in tfr if r["d"] == 32], DENSE_T, "s"),
              ("Trans-\nformer 64", [r["acc"] for r in tfr if r["d"] == 64], DENSE_T, "s")]
    for i, (name, vals, col, mk) in enumerate(groups):
        ax.scatter([i] * len(vals), vals, s=40 if col == EVENT else 22, marker=mk, color=col, zorder=4)
    ax.axhline(1 / 17, color=MUTED, lw=0.8); ax.text(2.45, 1 / 17 + 0.02, "chance", fontsize=7, color=MUTED, ha="right")
    ax.annotate(f"{min(ev):.3f}–{max(ev):.3f}", (0, max(ev)), xytext=(10, -3), textcoords="offset points", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_xlim(-0.5, 2.5); ax.set_ylabel("accuracy on unseen triples", fontsize=7.5)
    ax.tick_params(labelsize=7); ax.grid(axis="x", visible=False)
    return fig


def _chain_plateau():
    """E34's task, final checkpoint: learned windows + latest-instant credit (§88–§89) if present, else earlier rules."""
    for pat in ("d2_K15_ph1.5_sum_t0.6_a1_b0.5_latest_T0_W4.2_iv3.json",
                "d2_K15_ph1.5_sum_t0.6_a1_b0.5_instant_T0.3_W4.2_m0.9.json", "d2_K15_ph1.5_sum_t0.6_a1_b0.5_W4.2.json"):
        p = os.path.join(RES, "e34", pat)
        if os.path.exists(p):
            return [r["final"]["test"] for r in _load(p)["rows"]]
    return []


def _extra_world():
    out = []
    for p in glob.glob(os.path.join(RES, "e49", "gru_offline_*.json")):
        r = _load(p); out.append(([h["gru_offline_frozen"] for h in r["heldout"]], 1.5e3 * (r["args"]["d"] / 16) ** 2,
                                  f"GRU offline, d = {r['args']['d']}"))
    for p in glob.glob(os.path.join(RES, "e52", "thp_test_*.json")):
        r = _load(p); out.append(([r["epochs"][-1]["day6"], r["epochs"][-1]["day7"]], r["macs_per_event"], "Transformer Hawkes process"))
    return out


# ── 3. Anatomy of one decision (E53, depth 3) ─────────────────────────────────────────────────────────────
def fig_anatomy():
    d = _load(MECH)
    motifs = d["motifs"]; order = d["class_order"]; letters = "ABC"
    mchan = {}
    for li, m in enumerate(order):
        i, j, _ = motifs[m]; mchan[i] = (li, "start"); mchan[j] = (li, "end")
    part_of = {(motifs[m][0], motifs[m][1]): li for li, m in enumerate(order)}
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, gridspec_kw={"hspace": 0.45})
    names = {"positive": f"Class ({', '.join(letters)}): the three motifs in the right order → the class node fires",
             "decoy": f"Decoy (same noise): the motifs in the order ({', '.join(letters[order.index(m)] for m in d['decoy_order'])}) → nothing fires"}
    rows_y = {"ch": 5, "part": 2.6, "comp": 1.4, "cls": 0.2}
    for ax, key in zip(axs, ("positive", "decoy")):
        a = d["anatomy"][key]; t = np.array(a["spikes"])
        ax.set_ylim(-0.5, 7.4); ax.grid(False); ax.set_yticks([])
        for sp in ("left",):
            ax.spines[sp].set_visible(False)
        # channel spikes: motif channels by letter, others grey
        for c, tc in enumerate(t):
            if not np.isfinite(tc):
                continue
            if c in mchan:
                li, kind = mchan[c]
                ax.plot(tc, rows_y["ch"] + (0.9 if kind == "start" else 0.3), "|", ms=13, mew=2.2, color=MOTIF_COLORS[li])
            else:
                ax.plot(tc, rows_y["ch"] - 0.45, "|", ms=10, mew=1.4, color="#b9b8b2")
        for u in a["units"]:
            if u["level"] == 0:
                li = part_of.get(tuple(u["parts"][0]))
                col = MOTIF_COLORS[li] if li is not None else "#b9b8b2"
                ax.plot(u["t"], rows_y["part"], "o", ms=6 if li is not None else 4, color=col, zorder=3)
                if li is not None:
                    ax.text(u["t"], rows_y["part"] + 0.32, letters[li], ha="center", fontsize=7.5, color=col, weight="bold")
            else:
                l1 = part_of.get(tuple(u["parts"][0])); l2 = part_of.get(tuple(u["parts"][1]))
                key_unit = (l1, l2) == (0, 1)
                ax.plot(u["t"], rows_y["comp"], "s", ms=6.5 if key_unit else 4, color=INK if key_unit else "#b9b8b2", zorder=3)
                if key_unit:
                    ax.text(u["t"], rows_y["comp"] + 0.32, "A→B", ha="center", fontsize=7.5, color=INK, weight="bold")
        ft = a["fire_times"][d["class"]]
        if ft is not None:
            ax.add_patch(Rectangle((ft - d["W"], rows_y["cls"] - 0.35), d["W"], 3.3, color=LIGHT_BLUE, alpha=0.45, lw=0, zorder=0))
            ax.plot(ft, rows_y["cls"], marker="*", ms=16, color=EVENT, zorder=5)
            ax.text(ft + 0.2, rows_y["cls"] + 0.05, "class node fires: holds A→B, triggered by C", fontsize=7.3, va="center")
            ax.text(ft - d["W"] + 0.1, rows_y["cls"] + 2.75, "hold window", fontsize=6.8, color=BLUE)
        else:
            ax.text(0.99, 0.06, "A→B fires, but no C follows within the window: no class node fires (correct)",
                    transform=ax.transAxes, ha="right", fontsize=7.3)
        for lab, y in (("input spikes", rows_y["ch"] + 0.4), ("part detectors", rows_y["part"]),
                       ("order detectors", rows_y["comp"]), ("class node", rows_y["cls"])):
            ax.text(-0.012, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.3, color=MUTED)
        n_ev = int(np.isfinite(t).sum()) + len(a["units"]) + (1 if ft is not None else 0)
        ax.set_title(names[key], fontsize=8.2)
        ax.text(1.0, 1.02, f"{n_ev} events", transform=ax.transAxes, ha="right", fontsize=7.3, color=MUTED)
    axs[1].set_xlabel("time (arbitrary units)", fontsize=7.5)
    lo = min(min(v for v in d["anatomy"][k]["spikes"] if v < 1e9) for k in ("positive", "decoy")) - 0.6
    hi = max(max(v for v in d["anatomy"][k]["spikes"] if v < 1e9) for k in ("positive", "decoy")) + 0.6
    axs[1].set_xlim(lo, hi)
    return fig


# ── 4. Why the credit rule decides whether depth can be learned (§84) ─────────────────────────────────────
def fig_credit():
    d = _load(MECH); th = d["theta"]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    titles = {"every candidate (union)": "Credit every candidate:\nweight split, never fires",
              "greedy instant": "Credit the closest instant:\ntrapped on the shared prefix",
              "instant + cooled exploration": "Explore instants (cooled):\nfinds the order, stays"}
    for ax, (name, rows) in zip(axs, d["dynamics"].items()):
        a = np.array(rows)
        ax.plot(a[:, 0], a[:, 1], color=ORANGE, lw=1.4, label="prefix instant (A, B): wrong")
        ax.plot(a[:, 0], a[:, 2], color=BLUE, lw=1.6, label="valid instant (A, B, C): right")
        ax.axhline(th, color=MUTED, lw=0.8)
        ax.set_title(titles[name], fontsize=8.2); ax.set_xlabel("training examples", fontsize=7.5)
        ax.tick_params(labelsize=7); ax.set_ylim(-0.02, 1.02); ax.set_xlim(0, a[-1, 0])
        ax.set_xticks([0, 4000, 8000]); ax.set_xticklabels(["0", "4k", "8k"])
    axs[0].text(80, th + 0.03, "firing threshold", fontsize=6.8, color=MUTED)
    axs[0].set_ylabel("class node's drive\non a fixed example", fontsize=7.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    return fig


# ── 5. The two proved thresholds (§83(ii), §86) ───────────────────────────────────────────────────────────
def fig_theory():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.32})
    th = np.linspace(0.5, 1.0, 200); bs = (2 * th - 1) / th ** 2
    a.fill_between(th, 0, np.minimum(bs, 1), color=LIGHT_BLUE, lw=0)
    a.plot(th, np.minimum(bs, 1), color=BLUE, lw=1.6)
    a.text(0.83, 0.28, "every false fire moves\nweight toward the target", fontsize=7.2, color=INK, ha="center")
    for x, y, lab, dx, dy in ((0.5, 0.0, "θ = ½: no safe step", 6, 8), (0.6, 0.3, "E34w", 6, -4), (0.6, 0.5, "E53", 6, 2)):
        a.plot(x, y, "o", color=EVENT, ms=5, zorder=4, clip_on=False)
        a.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7)
    a.set_xlim(0.45, 1.0); a.set_ylim(0, 1.0)
    a.set_xlabel("firing threshold θ (fraction of the weight budget)", fontsize=7.5)
    a.set_ylabel("demotion step β", fontsize=7.5)
    a.set_title("AND credit is safe iff θ > ½ (§83)", fontsize=8.6); a.tick_params(labelsize=7)
    beta, t0 = 0.5, 0.6
    f = lambda w: w * (1 - beta) / (1 - beta * w)                      # noqa: E731
    for w0, col in ((0.65, ORANGE), (0.9, BLUE)):
        ws = [w0]
        for _ in range(3):
            ws.append(f(ws[-1]))
        b.plot(range(4), ws, "-o", color=col, ms=4.5, lw=1.5)
        b.annotate(f"starts at {w0}", (0, w0), xytext=(8, 5), textcoords="offset points", fontsize=7, color=col)
    b.axhline(t0, color=MUTED, lw=0.8); b.text(-0.5, t0 + 0.012, "threshold", fontsize=6.8, color=MUTED)
    b.set_xticks(range(4)); b.set_xlim(-0.6, 3.2); b.set_ylim(0.3, 1.0)
    b.set_xlabel("demotions in a row", fontsize=7.5); b.set_ylabel("weight on the correct route", fontsize=7.5)
    b.set_title("A margin survives demotions (§86)", fontsize=8.6); b.tick_params(labelsize=7)
    return fig


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "figures")
    for name, fn in (("concept", fig_concept), ("supremacy_map", fig_supremacy_map), ("anatomy", fig_anatomy),
                     ("credit_dynamics", fig_credit), ("theory_thresholds", fig_theory)):
        fig = fn(); fig.savefig(os.path.join(out, name + ".png"), bbox_inches="tight", facecolor="white"); plt.close(fig)
        print("wrote", name)
