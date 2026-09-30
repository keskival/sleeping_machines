"""Reader-first MD/PDF report, assembled from the same editorial blocks.

The previous chronological narrative is retained in report/archive/. Raw
experiment histories remain in FINDINGS.md and the numbered theory notes.
"""
from datetime import date
import hashlib
import html
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/"experiments/results"
FIG = ROOT/"report/figures"


def read(path):
    row = json.loads((RES/path).read_text())
    if row.get("status", "completed") != "completed":
        raise ValueError(f"Report requires completed result: {path}")
    return row


def results():
    tasks = {task: read(f"e120/{task}_d8_20260929.json")
             for task in ("language", "market", "temporal", "mnist", "modular", "dvs")}
    tasks["recall"] = read("e120/recall_d8_supported_20260929.json")
    tasks["phase_fixed"] = read("e121/shared_d2_phase_s6_200.json")
    tasks["phase"] = read("e121/shared_d2_guard_s6_200.json")
    tasks["plain"] = read("e121/shared_d2_plain_s6_200.json")
    tasks["shd_control"] = read("e122/d8_n2048_control_s6.json")
    tasks["shd_invariance"] = read("e122/d8_n2048_invariance_s6.json")
    tasks["phase_only"] = read("e124/modular_phase_only_s6_200.json")
    tasks["work_audit"] = read("e124/consolidated_work_20260929.json")
    tasks["shd_scaled"] = read("e122/d8_n4096_invariance_continue_s6_e2.json")
    tasks["shd_pool_mean"] = read("e122/d8_n4096_pool_control_s6.json")
    tasks["shd_pool_weighted"] = read("e122/d8_n4096_pool_weighted_s6.json")
    tasks["temporal_shallow"] = read("e120/temporal_d2_20260929.json")
    tasks["recall_shallow"] = read("e120/recall_d2_20260929.json")
    tasks["breadth_work"] = read("e124/breadth_work_counted_20260929.json")
    tasks["shd_bridge"] = read("e122/d8_n4096_bridge_s6.json")
    tasks["shd_bridge_frozen"] = read("e122/d8_n4096_bridge_frozen_s6.json")
    tasks["shd_key_value"] = read("e131/key_value_comparison_20260929.json")
    tasks["generic_language"] = {depth: read(f"e133/generic_language_d{depth}_s6_20260929.json") for depth in (1,8)}
    tasks["generic_language_audit"] = read("e133/generic_language_audit_20260929.json")
    tasks["complete_work"] = read("e172/complete_work_v2_20260930.json")
    tasks["training_work"] = read("training_work/local_total_training_work_v4_20260930T123846Z.json")
    tasks["event_language_work"] = read("event_language_work/local_event_language_work_20260930T124646Z.json")
    tasks["online_language"] = read("online_language/local_online_language_20260930T121741Z.json")
    tasks["stream_contract"] = read("e175/stream_language_contract_20260930.json")
    tasks["stream_training"] = read("e176/stream_language_d8_20260930.json")
    tasks["token_training"] = read("e178/prefix_language_d8_20260930.json")
    tasks["language_scaling"] = [read(str(path.relative_to(RES)))
        for path in sorted((RES/"parallel_language").glob("local_scale_capacity_*_20260930T153653Z.json"))]
    tasks["language_memory"] = [read(str(path.relative_to(RES)))
        for path in sorted((RES/"parallel_language").glob("local_memory_w128_D131072_*_20260930T155000Z.json"))]
    tasks["language_selective"] = [read(str(path.relative_to(RES)))
        for path in sorted((RES/"parallel_language").glob("local_selective_w128_D131072_*_20260930T161050Z.json"))]
    tasks["language_scaleup"] = [read(str(path.relative_to(RES)))
        for path in sorted((RES/"parallel_language").glob("local_staged_language_*Z.json"))]
    audit_path="parallel_language/local_language_representation_20260930T162337Z.json"
    tasks['language_representation'] = read(audit_path) if (RES/audit_path).exists() else None
    tasks["parallel_contract"] = read("parallel_language/local_parallel_language_contract_v3_20260930T153300Z.json")
    tasks["mechanisms"] = {
        "timing": read("e35/free.json"),
        "motifs": read("e34/d2_K15_ph1.5_sum_t0.6_a1_b0.5_latest_T0_W4.2_iv3.json"),
        "deep_order": read("e54/D4_L2_S5R4_T0.3.json"),
        "order_curve": read("e53/d3_S5R4_t0.6_curve_latest_T0_b0.5_m0.9_g0.9.json"),
        "order_references": {n: read(f"e36/transformer_e53_e53_n{n//1000}k_wd0.json")
                             for n in (2000, 5000, 10000, 20000, 40000)},
        "deep_reference": read("e36/transformer_e54_e54_n40k_wd0.json"),
        "timing_reference": read("e36/transformer_e27_rel.json"),
        "motif_reference": read("e36/transformer_e28_rel.json"),
    }
    tasks["shd_full_values"] = read("e134/full_value_comparison_20260929.json")
    content_path = "e135/content_comparison_20260929.json"
    tasks["shd_content"] = read(content_path) if (RES/content_path).exists() else None
    exchange_paths = {name: f"e136/scattering_{name}_d12_n1024_s6_e3_20260929.json"
                      for name in ("state", "packets")}
    exchange_paths["ablation"] = "e136/scattering_angle_ablation_20260929.json"
    tasks["shd_exchange"] = ({name: read(path) for name,path in exchange_paths.items()}
                              if all((RES/path).exists() for path in exchange_paths.values()) else None)
    compact_path = "e137/compact_comparison_20260930.json"
    tasks["shd_compact"] = read(compact_path) if (RES/compact_path).exists() else None
    for key,path in (
        ("shd_warm", "e122/d8_n6144_best_warm_s6_e1_20260930.json"),
        ("shd_fine", "e139/d8_fine_source_n6144_warm_s6_e1_20260930.json"),
        ("shd_phase", "e140/d8_phase_source_n6144_warm_s6_e1_20260930.json"),
        ("shd_local", "e141/d8_new_only_n6144_warm_s6_e1_20260930.json"),
        ("shd_state_residual", "e143/d8_parent_d6_state_residual_n6144_s6_e3_20260930.json"),
        ("shd_state_ablation", "e145/state_residual_ablation_20260930.json"),
        ("shd_state_summary", "e146/event_state_summary_20260930.json"),
        ("shd_single_clean", "e150/single_state_n6144_s6_e3_20260930.json"),
        ("shd_single_paired", "e152/nuisance_state_n6144_s6_e2_20260930.json"),
        ("shd_calibrated_d6", "e159/calibrated_d6_n6144_s6_e1_20260930.json"),
        ("shd_calibrated_d12", "e159/calibrated_d12_n6144_s6_e1_20260930.json"),
        ("shd_single_audit", "e154/single_encoder_audit_20260930.json"),
        ("shd_observer_depth", "e163/observer_depth_n6144_s6_e1_20260930.json"),
        ("shd_observer_audit", "e164/observer_depth_audit_20260930.json"),
        ("shd_selected_prefix", "e165/selected_prefix_20260930.json")):
        tasks[key] = (read(path) if (RES/path).exists() and
            json.loads((RES/path).read_text()).get("status") == "completed" else None)
    return tasks


def aws_e64_reference(model, data_size):
    """Read completed AWS reference evidence, including its cost and source path."""
    for provenance_path in sorted((RES / "aws_20260929").glob("*/provenance.json")):
        try:
            meta = json.loads(provenance_path.read_text())
            args = meta.get("arguments", [])
            if (meta.get("status") != "completed" or
                    meta.get("script") != "experiments/e64_lm_baselines.py" or
                    "--model" not in args or args[args.index("--model") + 1] != model or
                    "--D" not in args or int(args[args.index("--D") + 1]) != data_size):
                continue
            for result_path in sorted(provenance_path.parent.glob("*.json")):
                if result_path.name == "provenance.json":
                    continue
                row = json.loads(result_path.read_text())
                value = row.get("test_bpc")
                if (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value > 0):
                    return {"result": row, "path": str(result_path.relative_to(ROOT))}
        except (OSError, ValueError, TypeError, IndexError):
            continue
    return None


def evidence(M):
    causal = read("e173/causal_language_10m_20260930.json")
    aws_references = {model: aws_e64_reference(model, 90_000_000) for model in ("lstm", "tf")}
    def aws_e64_bpc(model):
        reference = aws_references[model]
        return reference["result"]["test_bpc"] if reference else None
    return {"native10": causal["arms"]["without_word"]["test_bpc"],
            "native_word10": causal["arms"]["with_causal_word"]["test_bpc"],
            "lstm1": read("e64/lstm_D1000000_s256_p20_dr0.2_v.json")["test_bpc"],
            "lstm10": read("e174/aligned_lstm_10m_20260930.json")["test_bpc"],
            "lstm90": aws_e64_bpc("lstm"),
            "tf1": read("e64/tf_D1000000_s256_p20_dr0.2_v.json")["test_bpc"],
            "tf10": read("e174/aligned_tf_10m_20260930.json")["test_bpc"],
            "tf90": aws_e64_bpc("tf"),
            "aws_references": aws_references,
            "recall_tf": max(p["n32"] for f in (RES/"e61").glob("tf_K32_n8*.json")
                             for row in json.loads(f.read_text())["rows"] for p in row["curve"])}


def language_90m_reference_text(ev):
    """Publish each completed control without treating validation logs as test evidence."""
    scores = []
    pending = []
    for key, label in (("lstm90", "LSTM"), ("tf90", "four-layer Transformer")):
        if ev[key] is not None:
            scores.append(f"{ev[key]:.3f} for the {label}")
        else:
            pending.append(label)
    text = ("At 90M training characters, reference test scores are " + ", ".join(scores) + ". "
            if scores else "")
    costs = []
    for provenance in sorted((RES / 'aws_20260929').glob('aws_e64_*/provenance.json')):
        meta = json.loads(provenance.read_text())
        if meta.get('status') != 'completed':
            continue
        for path in provenance.parent.glob('*.json'):
            result = json.loads(path.read_text())
            if result.get('args', {}).get('D') != 90_000_000:
                continue
            estimate = result.get('training_flops_estimate', {})
            if estimate.get('total_training_flops'):
                costs.append(f"{result['args']['model'].upper()}: {estimate['total_training_flops'] / 1e15:.2f} PFLOP")
    if costs:
        text += ('Estimated training work (forward, backward, Adam and gradient clipping): '
                 + ', '.join(costs) + '. Shape-based estimates count multiply-add as two operations; '
                 'backward is approximated as twice forward. Validation/test inference is excluded. ')
    if pending:
        text += " and ".join(pending) + " reference results are pending. "
    text += "These are single-seed comparisons; capacities and fitting budgets are not matched. "
    return text


def accomplishments_figure(M, ev, tasks):
    import matplotlib.pyplot as plt
    import numpy as np
    blue, orange, gray = M["BLUE"], M["ORANGE"], M["GRAY"]
    f, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    event = [100*next(point["test"] for point in row["curve"] if point["step"] == 2000)
             for row in tasks["mechanisms"]["order_curve"]["rows"]]
    reference = [100*row["acc"] for row in tasks["mechanisms"]["order_references"][2000]["rows"]]
    for i, (values, color) in enumerate(((event, blue), (reference, orange))):
        mean = np.mean(values)
        ax[0].bar(i, mean, width=.55, color=color)
        ax[0].errorbar(i, mean, yerr=[[mean-min(values)], [max(values)-mean]],
                       color="#172431", capsize=4, linewidth=1.2)
        ax[0].text(i, max(values)+3, f"{min(values):.2f}–{max(values):.2f}%",
                   ha="center", fontsize=9)
    ax[0].set(xticks=[0,1], xticklabels=["Ours: event chains\n5 runs; one pass",
                                      "Transformer\n2 runs; repeated fitting"],
              ylim=(0,116), ylabel="Held-out accuracy (%) ↑")
    ax[0].set_title("Learning an order rule\nSame 2,000 distinct fitting examples", fontsize=10)
    ax[1].bar(range(2), [100, 100*ev["recall_tf"]], color=[blue, orange], width=.55)
    ax[1].set_xticks([0, 1], ["Ours: race retrieval\n5 runs", "Best recorded\nTransformer result"])
    ax[1].set_ylim(0, 116)
    ax[1].set_ylabel("Accuracy at 4× context (%) ↑")
    for i, value in enumerate([100, 100*ev["recall_tf"]]):
        ax[1].text(i, value+2, f"{value:.1f}%", ha="center", fontsize=10)
    ax[1].set_title("Retrieval beyond training length\nFour times the training context", fontsize=10)
    for a in ax:
        a.grid(axis="x", visible=False)
        a.set_yticks([0,25,50,75,100])
        a.tick_params(axis="x", labelsize=7.3)
    f.tight_layout(w_pad=2.5)
    return f


def figures(M, tasks, ev):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    blue, orange, gray = M["BLUE"], M["ORANGE"], M["GRAY"]
    FIG.mkdir(exist_ok=True)
    def save(fig, name):
        if name != "accomplishments":
            fig.text(.01, 1.015, "Ours = Sleeping Machines", color=blue, fontsize=8,
                     fontweight="bold", ha="left")
        fig.savefig(FIG/(name+".png"), dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    f, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    for depth, color in ((1, gray), (8, blue)):
        row = tasks["generic_language"][depth]
        curve = [row["initial"]["dev"]["bpc"]] + [e["dev"]["bpc"] for e in row["curve"]]
        axes[0].plot(range(len(curve)), curve, "o-", color=color, label=f"Ours: {depth} layer" + ("s" if depth > 1 else ""))
        work = tasks["generic_language_audit"]["rows"][str(depth)]["wall_s"]
        final = curve[-1]
        axes[1].scatter(work, final, color=color, s=65)
        axes[1].annotate(f"{depth} layer" + ("s" if depth > 1 else "") + f"\n{final:.3f} bpc", (work,final),
                         xytext=(0,12), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set(xlabel="Passes over 8,192 training characters", ylabel="Validation bpc (lower is better)",
                title="Learned prediction through event layers", xticks=range(5))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Total CPU wall time (s; fitting + evaluation)",
                ylabel="Validation bpc (lower is better)", title="Quality / observed time tradeoff",
                ylim=(3.32,3.58))
    f.tight_layout()
    save(f, "e133_generic_language")
    stream = tasks["stream_training"]
    f, a = plt.subplots(figsize=(7.2, 2.65))
    a.plot(range(5), [stream["initial"]["bpc"]]+[row["dev"]["bpc"] for row in stream["curve"]],
           "o-", color=blue, label="Ours: validation")
    a.plot(range(1,5), [row["fit"]["bpc"] for row in stream["curve"]],
           "s--", color=gray, label="Ours: fitting, frozen evaluation")
    token = tasks["token_training"]
    a.plot(range(5), [token["initial"]["bpc"]]+[row["dev"]["bpc"] for row in token["curve"]],
           "^-", color=orange, label="Ours: prefix tokens, validation")
    a.set(xlabel="Passes over 8,192 training characters", ylabel="Bits per character ↓",
          title="Eight-layer persistent language stream", xticks=range(5))
    a.legend(fontsize=8)
    f.tight_layout()
    save(f, "e176_stream_language_learning")
    f = accomplishments_figure(M, ev, tasks)
    save(f, "accomplishments")

    f, a = plt.subplots(figsize=(7.2, 2.65))
    a.set_xlim(0, 10); a.set_ylim(0, 4); a.axis("off")
    boxes = [(0.1, 1.55, 1.8, .9, "Input events\nvector + time"),
             (2.45, 1.55, 2.4, .9, "Mix input + memory\nGated residual vector"),
             (5.5, 1.55, 1.8, .9, "Clock race\nwinning arrival"),
             (7.95, 1.55, 1.9, .9, "Next layer\nthen a query"),
             (2.45, .05, 2.4, .85, "Alternatives + loss\nvector + clock credit")]
    for x, y, w, h, label in boxes:
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04",facecolor="#edf3fb",edgecolor=blue))
        a.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=9)
    for x1, x2 in ((1.95, 2.4), (4.9, 5.45), (7.35, 7.9)):
        a.add_patch(FancyArrowPatch((x1,2),(x2,2),arrowstyle="-|>",mutation_scale=14,color=blue))
    a.annotate("", (3.65,.94), (3.65,1.5), arrowprops={"arrowstyle":"->","linestyle":"--","color":orange})
    a.text(5.65,.47,"One message is emitted per retained arrival.\nLocal state survives between events.",fontsize=9,va="center")
    a.text(.1,3.3,"Ours: a learned event-state block",fontsize=11,fontweight="bold")
    a.text(.1,2.85,"Source identity, content and elapsed time determine the next vector and arrival.",fontsize=9)
    save(f,"shared_architecture")

    f, a = plt.subplots(figsize=(7.2,2.85))
    values=[100*tasks['shd_full_values']['rows']['parent']['held_accuracy'],
            100*tasks['shd_key_value']['rows']['separate']['held_accuracy'],
            100*tasks['shd_full_values']['rows']['fixed']['held_accuracy']]
    labels=["Starting\ncheckpoint","New context maps\n6,336 taught parameters","All value layers\n58,048 taught parameters"]
    if tasks['shd_content'] is not None:
        values.append(100*tasks['shd_content']['rows']['content']['held_accuracy'])
        labels.append("Content retrieval\n60,096 taught parameters")
    a.bar(range(len(values)),values,color=["#c5ced8",gray,orange,blue][:len(values)],width=.55)
    a.set(xticks=range(len(values)),xticklabels=labels,
          ylim=(0,100),ylabel="Held-speaker accuracy (%) — higher is better",title="Eight-layer event classifier: same parent, one-pass continuations")
    a.tick_params(axis='x',labelsize=8)
    for i,v in enumerate(values):a.text(i,v+2,f"{v:.1f}%",ha="center",fontsize=11)
    f.tight_layout();save(f,"e122_speech")

    f, axes = plt.subplots(1,2,figsize=(7.2,2.85))
    def held(row):
        parts=[row["final"][k] for k in ("dev_original","dev_additional")]
        return 100*sum(p["correct"] for p in parts)/sum(p["n"] for p in parts)
    parent=held(tasks["shd_scaled"])
    entries=[("Original\nparent",tasks["shd_scaled"],gray)]
    for key,label,color in (("shd_warm","Warm\ncontinuation",orange),
                            ("shd_fine","Fine source\nmessages",blue),
                            ("shd_phase","Fine source +\ntemporal phase","#1baf7a"),
                            ("shd_local","New learner;\nparent frozen","#8766b6")):
        if tasks[key] is not None:entries.append((label,tasks[key],color))
    axes[0].bar(range(len(entries)),[held(r) for _,r,_ in entries],
                color=[c for _,_,c in entries],width=.55)
    for i,(_,r,_) in enumerate(entries):
        axes[0].text(i,held(r)+2,f"{held(r):.2f}%",ha="center",fontsize=8.5)
    axes[0].set(xticks=range(len(entries)),xticklabels=[n for n,_,_ in entries],
        ylim=(0,100),ylabel="Private held-speaker accuracy (%)",title="Recognition quality — higher is better")
    timed=entries[1:]
    for i,(label,r,color) in enumerate(timed):
        axes[1].scatter(r["wall_s"],held(r),s=65,color=color,label=label.replace("\n"," "))
    axes[1].axhline(parent,color=gray,linestyle="--",linewidth=1)
    axes[1].text(.98,.96,f"Original parent: {parent:.2f}%",transform=axes[1].transAxes,
        fontsize=7,ha="right",va="top")
    axes[1].set(xlabel="Recorded continuation wall time (s)",ylabel="Accuracy (%) — higher is better",
        title="One pass; same 6,144 fitting examples",ylim=(65,80))
    axes[1].legend(fontsize=6.1,loc="lower center",ncol=2)
    axes[0].tick_params(axis="x",labelsize=6.1)
    f.tight_layout();save(f,"e139_source_information")

    if tasks["shd_state_residual"] is not None:
        row=tasks["shd_state_residual"]
        f,axes=plt.subplots(1,2,figsize=(7.2,2.7))
        epochs=[0]+[r["epoch"] for r in row["curve"]]
        dev=[row["initial"]["dev"]["accuracy"]]+[r["dev"]["accuracy"] for r in row["curve"]]
        axes[0].plot(epochs,[100*x for x in dev],"o-",color=blue,label="Temporal residual + frozen parent")
        axes[0].axhline(100*dev[0],color=gray,linestyle="--",label="Original parent")
        axes[0].set(xlabel="Passes over 6,144 fitting utterances",ylabel="Private accuracy (%) — higher is better",
            title="Transfer to held speakers",xticks=epochs,ylim=(65,100))
        axes[0].legend(fontsize=6.5,loc="upper left")
        for split,label,color in (("fit","Fitting speakers",orange),("dev","Held speakers",blue)):
            axes[1].plot([r["epoch"] for r in row["curve"]],[r[split]["nll"] for r in row["curve"]],
                "o-",label=label,color=color)
        axes[1].axhline(row["initial"]["dev"]["nll"],color=gray,linestyle="--",label="Parent held NLL")
        axes[1].set(xlabel="Passes over fitting utterances",ylabel="NLL — lower is better",
            title="Prediction quality",xticks=epochs[1:])
        axes[1].legend(fontsize=6.5,frameon=True,facecolor="white",edgecolor="white",framealpha=1.)
        f.tight_layout();save(f,"e143_temporal_residual_learning")

    if tasks["shd_state_ablation"] is not None and tasks["shd_state_summary"] is not None:
        f,axes=plt.subplots(1,2,figsize=(7.2,2.7))
        rows=tasks["shd_state_ablation"]["rows"]
        entries=[("trained","Learned model"),("reset_clocks","Initial hidden clocks"),
            ("reset_source_embedding","Initial source marks"),("reset_state_stack","Initial state stack"),
            ("reset_modal_dynamics","Initial modal poles")]
        values=[100*rows[key]["held"]["accuracy"] for key,_ in entries]
        axes[0].barh(range(len(entries)),values,color=[blue,orange,gray,gray,gray])
        axes[0].set(yticks=range(len(entries)),yticklabels=[name for _,name in entries],
            xlim=(60,87),xlabel="Accuracy (%) — higher is better",title="Reset one learned subsystem")
        axes[0].invert_yaxis();axes[0].tick_params(axis="y",labelsize=7)
        for i,v in enumerate(values):axes[0].text(v+.4,i,f"{v:.1f}%",va="center",fontsize=7)
        summary=tasks["shd_state_summary"]
        audit=read("e147/disjoint_speaker_audit_20260930.json")
        for i,(n,gain,lost) in enumerate(((512,summary["new_only_correct"],summary["parent_only_correct"]),
            (audit["n"],audit["new_only_correct"],audit["parent_only_correct"]))):
            axes[1].bar(i-.16,gain,width=.3,color=blue,label="Newly correct" if i==0 else None)
            axes[1].bar(i+.16,-lost,width=.3,color=orange,label="Previously correct, now wrong" if i==0 else None)
            axes[1].text(i,gain+3,f"Net +{gain-lost}",ha="center",fontsize=8)
        axes[1].axhline(0,color=gray,linewidth=.8)
        axes[1].set(xticks=[0,1],xticklabels=["Development\n512 utterances",f"Disjoint audit\n{audit['n']} utterances"],
            ylabel="Changed correct answers",title="Matched gains and regressions",ylim=(-25,82))
        axes[1].tick_params(axis="x",labelsize=7)
        axes[1].legend(fontsize=6,loc="upper left")
        f.tight_layout();save(f,"e145_learned_timing_and_transfer")

    if tasks["shd_single_clean"] is not None and tasks["shd_single_paired"] is not None:
        f,axes=plt.subplots(1,2,figsize=(7.2,2.7))
        for key,label,color in (("shd_single_clean","Clean head initialization",orange),
                               ("shd_single_paired","Paired-view initialization",blue)):
            row=tasks[key]
            curve=[r for r in row["curve"] if r["epoch"]>0]
            points=[row["conditioned_initial"]["dev"]]+[r["dev"] for r in curve]
            axes[0].plot([0]+[r["epoch"] for r in curve],[100*p["accuracy"] for p in points],
                "o-",label=label,color=color)
        reference=max(tasks["shd_state_residual"]["curve"],key=lambda r:r["dev"]["correct"])
        axes[0].axhline(100*reference["dev"]["accuracy"],color=gray,linestyle="--",label="Combined model")
        axes[0].set(xlabel="Single-encoder continuation passes",ylabel="Private accuracy (%) — higher is better",
            title="Head initialization and encoder updates",xticks=[0,1,2,3],ylim=(60,100))
        axes[0].legend(fontsize=5.6,loc="upper left",ncol=2)
        row=tasks["shd_single_paired"]
        for i,(key,label) in enumerate((("matched_clean_head","Clean-fit head"),("conditioned_initial","Paired-view head"))):
            part=row[key]
            clean=part.get("clean_fit",part.get("fit"))["nll"]
            aug=part["augmented_fit"]["nll"]
            axes[1].bar(i-.16,clean,width=.3,color=blue,label="Clean fitting speech" if i==0 else None)
            axes[1].bar(i+.16,aug,width=.3,color=orange,label="Augmented fitting speech" if i==0 else None)
        axes[1].set(xticks=[0,1],xticklabels=["Clean-fit head","Paired-view head"],
            ylabel="NLL — lower is better",title="Same frozen temporal features")
        axes[1].legend(fontsize=6.1,loc="upper right")
        f.tight_layout();save(f,"e152_single_encoder_learning")

    if tasks["shd_single_audit"] is not None and tasks["shd_observer_audit"] is not None:
        audit=tasks["shd_single_audit"];directional=tasks["shd_observer_audit"]
        f,axes=plt.subplots(1,2,figsize=(7.2,2.8))
        points=[(audit['rows'][name],label,color) for name,label,color in (
            ('combined','Combined model',gray),('single_paired','Single paired head',blue),
            ('matched_d6','Trained six blocks','#1baf7a'),('grown_d12','Bounded twelve blocks',orange))]
        points.append((directional['rows']['directional_d12'],'Directional twelve blocks','#9154c3'))
        points.append((directional['rows']['directional_prefix'],'Selected trained prefix','#144da1'))
        for row,label,color in points:
            axes[0].scatter(row['forward_wall_s'],100*row['audit']['accuracy'],color=color,s=30,label=label)
        axes[0].set(xlabel='CPU forward seconds / 657 utterances — lower is better',
            ylabel='Reused audit accuracy (%) — higher is better',title='Quality and observed CPU work',ylim=(72,85))
        axes[0].legend(fontsize=5.8,loc='upper right')
        for i,(full,prefix) in enumerate(((audit['rows']['grown_d12']['audit'],audit['rows']['grown_d12_prefix']['audit']),
            (directional['rows']['directional_d12']['audit'],directional['rows']['directional_prefix']['audit']))):
            for offset,row,color,label in ((-.16,full,blue,'Full twelve blocks'),(.16,prefix,gray,'Six appended blocks removed')):
                axes[1].bar(i+offset,100*row['accuracy'],width=.3,color=color,label=label if i==0 else None)
                axes[1].text(i+offset,100*row['accuracy']+.4,str(row['correct']),ha='center',fontsize=7)
        axes[1].set(xticks=[0,1],xticklabels=['Bounded outputs','Directional units'],
            ylabel='Reused audit accuracy (%) — higher is better',title='Fitted contribution of added depth',ylim=(70,86))
        axes[1].legend(fontsize=5.8,loc='upper right')
        f.tight_layout();save(f,'e164_depth_use_and_work')

    if tasks["shd_exchange"] is not None:
        f, axes = plt.subplots(1,2,figsize=(7.2,2.95))
        for name,label,color,style in (("state","Full state query",blue,"-"),
                                       ("packets","Packet query",gray,":")):
            row=tasks["shd_exchange"][name]
            for axis,split in zip(axes,("fit","held")):
                points=[row["initial"][split]["accuracy"]]+[x[split]["accuracy"] for x in row["curve"]]
                axis.plot(range(len(points)),[100*x for x in points],marker="o",linestyle=style,color=color,label=label)
        if tasks["shd_compact"] is not None:
            for mode,label,color,style in (("learned","Compact, learned angles","#1baf7a","-"),
                                          ("frozen","Compact, fixed angles",orange,"--")):
                row=tasks["shd_compact"]["rows"][mode]
                for axis,split in zip(axes,("fit","held")):
                    axis.plot([x["epoch"] for x in row["curve"]],
                              [100*x[split+"_accuracy"] for x in row["curve"]],
                              marker="o",linestyle=style,color=color,label=label)
        for axis,title in zip(axes,("Learning the fitting utterances","Transfer to held speakers")):
            axis.set(xlabel="Passes over 1,024 utterances",ylabel="Accuracy (%) — higher is better",
                     title=title,xticks=range(4),ylim=(0,105))
        axes[0].legend(fontsize=6.8,loc="lower right")
        f.tight_layout();save(f,"e136_scattering_learning")

    f, axes = plt.subplots(1,2,figsize=(7.2,3.15))
    entries=tasks["work_audit"]["rows"]
    names={"shared_phase_only":"Ours: phase rule (69 scalars)","shared_d2_phase":"Ours: encoder + phase rule",
           "shared_d2_recall":"Ours: encoder + pointer","lstm":"LSTM, width 32 / 2 layers",
           "transformer":"Transformer, width 32 / 2 layers"}
    colors={"shared_phase_only":blue,"shared_d2_phase":"#1baf7a","shared_d2_recall":blue,
            "lstm":gray,"transformer":orange}
    for axis,task,split,title in zip(axes,("modular","recall"),("unseen_all","context4x"),
                                    ("Arithmetic: every unseen triple","Recall: four times the context")):
        for row in entries:
            if row["task"]!=task or row["split"]!=split:continue
            label=names[row["model"]];x=row["work"]["estimated_operations"];y=100*row["accuracy"]
            axis.scatter([x],[y],s=55,marker="D" if row["model"].startswith("shared") else
                         "s" if row["model"]=="transformer" else "o",color=colors[row["model"]],label=label,zorder=4)
            offset=(-12,10) if task=="modular" and row["model"]=="lstm" else \
                   (12,10) if task=="modular" and row["model"]=="transformer" else (0,6 if y<90 else -14)
            axis.annotate(f"{y:.1f}%",(x,y),xytext=offset,textcoords="offset points",ha="center",fontsize=8)
        axis.set(xscale="log",ylim=(-3,112),xlabel="Estimated operations per query (log)",title=title)
        if task=="recall":
            from matplotlib.ticker import NullFormatter
            axis.set_xticks([700_000,1_000_000,2_000_000,4_000_000],["0.7M","1M","2M","4M"])
            axis.xaxis.set_minor_formatter(NullFormatter())
        axis.legend(fontsize=6.4,loc="center left",bbox_to_anchor=(-.02,.6))
    axes[0].set_ylabel("Development accuracy (%)")
    f.tight_layout();save(f,"consolidated_work_frontiers")
    mechanism = tasks["mechanisms"]
    f, axes = plt.subplots(1,2,figsize=(7.2,3.05))
    groups = [("Timing\npatterns",mechanism["timing"]),
              ("Shared-motif\ncomposition",mechanism["motifs"]),
              ("Depth-four\norder",mechanism["deep_order"])]
    for i,(_,row) in enumerate(groups):
        values = [100*r["final"]["test"] for r in row["rows"]]
        axes[0].scatter([i]*len(values),values,color=blue,s=28,zorder=4)
        axes[0].annotate(f"{min(values):.2f}–{max(values):.2f}%",(i,min(values)),
                         xytext=(0,-15),textcoords="offset points",ha="center",fontsize=7.1)
    axes[0].set(xticks=range(3),xticklabels=[n for n,_ in groups],ylim=(98.5,100.35),
                ylabel="Synthetic evaluation accuracy (%) ↑",title="Five runs per event mechanism")
    axes[0].tick_params(axis='x',labelsize=7)
    curves=mechanism["order_curve"]["rows"]
    steps=[r["step"] for r in curves[0]["curve"]]
    quality=np.array([[100*r["test"] for r in row["curve"]] for row in curves])
    axes[1].fill_between(steps,quality.min(0),quality.max(0),color=blue,alpha=.18)
    axes[1].plot(steps,np.median(quality,0),"o-",color=blue,ms=3,label="Ours: event chain, one pass")
    for n,row in mechanism["order_references"].items():
        axes[1].scatter([n]*len(row["rows"]),[100*r["acc"] for r in row["rows"]],
                        color=orange,s=23,marker="s",label="Transformer; repeated fitting" if n==2000 else None)
    axes[1].set(xscale="log",ylim=(25,103),xlabel="Distinct fitting examples (log)",
                ylabel="Synthetic evaluation accuracy (%) ↑",title="Depth-three sample efficiency")
    axes[1].legend(fontsize=6.4,loc="lower right")
    f.tight_layout(w_pad=2)
    save(f,"supremacy_map")

    f, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    stages = ("forward_and_loss", "backward", "gradient_clipping", "optimizer")
    names = ("Forward + loss", "Backward", "Clip", "Adam")
    colors = (blue, orange, gray, "#8c73aa")
    rows = tasks["training_work"]["rows"]
    for a, model, title in zip(axes, ("common", "transformer"), ("Ours: event query encoder, 8 layers", "Transformer: 2 layers")):
        bottom = np.zeros(len(rows))
        for stage, label, color in zip(stages, names, colors):
            key = "common_stages" if model == "common" else "reference_stages"
            values = np.array([row[key][stage]/1e9 for row in rows])
            a.barh(range(len(rows)), values, left=bottom, label=label, color=color)
            bottom += values
        if model == "common":
            values = np.array([row["common_setup"]["total_arithmetic_flops"]/1e9 for row in rows])
            a.barh(range(len(rows)), values, left=bottom, label="Evidence + calibration", color="#64a68c")
        a.set_yticks(range(len(rows)), ["Text", "Market", "Composition", "MNIST", "Gestures"])
        a.set_xlim(0, max(max(row["common_total_training_flops"], row["reference_total_training_flops"])
                        for row in rows)/1e9*1.1)
        a.invert_yaxis()
        a.set_xlabel("Whole fitting budget (estimated GFLOPs) ↓", fontsize=8)
        a.set_title(title, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    f.legend(handles, labels, fontsize=6.5, loc="lower center", ncol=3)
    f.tight_layout(w_pad=2, rect=(0, .15, 1, 1))
    save(f, "e172_complete_training_work")

    rows = sorted(tasks["language_scaling"], key=lambda row:row["parameters"])
    f, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for row, color in zip(rows, (gray, orange, blue, "#64a68c")):
        axes[0].plot([point["epoch"] for point in row["curve"]],
                     [point["dev"]["bpc"] for point in row["curve"]], "o-", color=color,
                     label=f"Ours: width {row['args']['width']}")
        total = row["work"]["total_training_arithmetic_flops"]/1e9
        score = row["final"]["dev"]["bpc"]
        axes[1].scatter(total, score, color=color, s=50)
        axes[1].annotate(f"Ours: {row['parameters']/1000:.1f}K parameters",(total,score),
            xytext=(0,10),textcoords="offset points",ha="center",fontsize=7)
    axes[0].set(xlabel="Passes over the same 131,072 characters",ylabel="Development bpc ↓",
                title="Completed language learning curves",xticks=[1,2,3,4])
    axes[0].legend(fontsize=7)
    axes[1].set(xscale="log",xlabel="Total fitting arithmetic (GFLOPs; log) ↓",
                ylabel="Development bpc ↓",title="Capacity costs work and improves quality")
    axes[1].margins(x=.25,y=.3)
    f.tight_layout(w_pad=2)
    save(f,"language_capacity_scaling")

    capacity=[row for row in tasks['language_scaleup']
              if row['args']['fit']==131072 and row['args']['width']==256]
    if capacity and tasks['language_selective']:
        constant=next(row for row in tasks['language_memory'] if row['args']['memory_profile']=='inherited')
        selective=next(row for row in tasks['language_selective'] if row['args']['memory_profile']=='inherited')
        f,a=plt.subplots(figsize=(7.2,2.65))
        points=[constant,selective,capacity[-1]]
        labels=['Ours: constant memory, width 128','Ours: input gates, width 128','Ours: input gates, width 256']
        for row,label,color,offset in zip(points,labels,(gray,blue,orange),((16,13),(16,-16),(-16,12))):
            x=row['work']['total_training_arithmetic_flops']/1e12;y=row['final']['dev']['bpc']
            a.scatter(x,y,color=color,s=55)
            a.annotate(label+f"\n{y:.3f} bpc; {x:.3f} TFLOPs",(x,y),xytext=offset,
                textcoords='offset points',ha='right' if offset[0]<0 else 'left',fontsize=8,
                arrowprops=dict(arrowstyle='-',color=color))
        a.set(xlim=(.75,4.5),ylim=(2.53,2.68),xlabel='Total fitting arithmetic (TFLOPs) ↓',
              ylabel='Development bpc ↓',title='Where additional work helped this fit')
        f.tight_layout();save(f,'language_compute_choices')

    rows=tasks["breadth_work"]["rows"]
    costs={row["task"]:row for row in tasks["training_work"]["rows"]}
    f, axes=plt.subplots(1,2,figsize=(7.2,2.2))
    selected=[next(row for row in rows if row["task"]==task)
              for task in ("language","market","temporal","mnist","dvs")]
    ratios=[row["common_forward_map_scan_flops"]/row["reference_forward_map_attention_flops"]
            for row in selected]
    training=[costs[row["task"]]["common_to_reference_ratio"] for row in selected]
    for axis,values,title in zip(axes,(ratios,training),
        ("Ours / Transformer: inference core FLOPs","Ours / Transformer: total fitting FLOPs")):
        axis.barh(range(5),values,color=blue,height=.65)
        axis.axvline(1,color=orange,linestyle="--",linewidth=1)
        axis.set_yticks(range(5),["Text","Market","Composition","MNIST","Gestures"])
        axis.invert_yaxis();axis.set_xlim(0,2.45)
        axis.set_title(title,fontsize=9)
        axis.set_xlabel("Ratio ↓  (Transformer = 1)",fontsize=8)
        for i,value in enumerate(values):axis.text(value+.04,i,f"{value:.2f}×",va="center",fontsize=8)
    f.tight_layout(w_pad=2)
    save(f,"breadth_work_ratios")

    if tasks['language_representation']:
        audit=tasks['language_representation']
        names=('full','reset_history_every_token','zero_incoming_embeddings','remove_all_memory_corrections')
        f,a=plt.subplots(figsize=(7.2,2.8))
        for j,(row,color,label) in enumerate(zip(audit['rows'],(gray,blue),
            ('Ours: constant memory','Ours: input-gated memory'))):
            values=[row['interventions'][name]['bpc'] for name in names]
            positions=np.arange(4)+(j-.5)*.35
            a.bar(positions,values,width=.35,color=color,label=label)
            for x,value in zip(positions,values):a.text(x,value+.08,f"{value:.3f}",ha='center',fontsize=7)
        a.set_xticks(range(4),['Full content\nand history','Reset history\nevery character','Zero incoming\nembeddings','Remove all\nmemory corrections'])
        a.set(ylabel='Development bpc ↓',ylim=(0,8.5),title='Fitted predictions depend on content and history')
        a.legend(fontsize=7,loc='upper left');a.grid(axis='x',visible=False)
        f.tight_layout();save(f,'language_content_memory_audit')


def blocks(M, tasks, ev):
    """Project entry point: capabilities, evidence, principles and applications."""
    def compact_work(value):
        if value >= 1e15:
            return f"{value/1e15:.2f}P"
        if value >= 1e12:
            return f"{value/1e12:.2f}T"
        return f"{value/1e9:.2f}G" if value>=1e9 else f"{value/1e6:.2f}M"
    phase=tasks["phase_only"]["final"]["dev"]
    work=tasks["work_audit"]["rows"]
    phase_work=next(r for r in work if r["model"]=="shared_phase_only")["work"]["estimated_operations"]
    recall_work=next(r for r in work if r["model"]=="shared_d2_recall" and r["split"]=="context4x")["work"]["estimated_operations"]
    pool_mean,pool_weighted=tasks["shd_pool_mean"],tasks["shd_pool_weighted"]
    def pooled(row,endpoint="final"):
        if "dev" in row[endpoint]:
            part=row[endpoint]["dev"]
            return part["correct"]/part["n"]
        parts=[row[endpoint][k] for k in ("dev_original","dev_additional")]
        return sum(p["correct"] for p in parts)/sum(p["n"] for p in parts)
    pages=[]
    completed_stage = tasks['language_selective']+[row for row in tasks['language_scaleup']
                                                 if row['args']['fit']==131072]
    stage_bpc=min((row['final']['dev']['bpc'] for row in completed_stage),default=None)
    lm_costs={row["model"]:row for row in tasks["training_work"]["language_rows"]}
    pages.append([
        ("title","Sleeping Machines"),
        ("sub","Deep learning that computes with time"),
        ("small","Tero Keski-Valkama and Karoliina Salminen · Research report · 30 September 2026"),
        ("p","Messages carry content and an arrival time. Nodes mix incoming vectors with persistent memory, "
         "gate their updates and compete through learned delays. Arrival order and winning races determine the computation. "
         "The goal is useful intelligence with much less active work."),
        ("h1","The differentiators at a glance"),
        ("bullets",[
         "<b>Time performs computation.</b> Delays, races and phase transformations implement useful functions.",
         "<b>Hard routes can learn.</b> Winning messages execute; unrealized alternatives receive counterfactual credit.",
         "<b>Deep, persistent event representations.</b> Vector messages and local memory carry information and credit through layers; retrieval and temporal primitives share the model family.",
         "<b>Capacity beyond activity.</b> The scaling goal is more useful dormant capacity, selectively recruited and judged by prediction quality at a given total work budget."]),
        ("h1","The strongest demonstrated results"),
        ("bullets",[
         "<b>Generalization.</b> Race retrieval reaches <b>100% at four times the training context</b> "
         "within 4,000 examples in all five runs. A learned phase rule solves <b>all 3,440 unseen "
         "modular triples</b>, using the supplied period 17.",
         "<b>Learning from fewer examples.</b> Depth-three event chains reach <b>99.73–99.93%</b> "
         "after 2,000 examples seen once; saved Transformer controls reach <b>33.25–40.80%</b> "
         "with the same number of distinct examples and repeated fitting. Depth-four chains reach 99.9–100%.",
         f"<b>Learned representations.</b> The persistent language model reaches "
         f"<b>{tasks['stream_training']['final']['dev']['bpc']:.3f} development bpc</b> in the 8K-character pilot. "
         +(f"The new 131K-character screen reaches <b>{stage_bpc:.3f}</b>. " if stage_bpc is not None else "")+
         "A learned speech encoder reaches <b>79.69%</b> on 512 private development utterances. "
         "Embeddings, temporal state and vector maps learn."]),
        ("figure",("accomplishments",174)),
        ("small","Left: means and recorded ranges, five event runs and two Transformer runs; "
         "2,000 distinct examples, seen once / presented 400,000 times. Right: all five event runs "
         "reach 100% within 4,000 examples; the control is the best saved result across seven "
         "Transformer configurations and their learning curves. These synthetic tasks use different "
         "architectures and structural priors. Sources: E53/E36 and E61."),
        ("small","<b>Language scale-up:</b> precision-checked, staged fitting is underway. The comparable "
         "10M-character test remains pending; completed neural controls and costs are in Appendix B.")])

    reference_rows=[
        ["Ours: learned event-state model (planned)", "10M / four passes", "Pending", "Pending"],
        ["LSTM; width 512, one recurrent layer", "10M / six passes", f"{ev['lstm10']:.3f}",
         compact_work(lm_costs['lstm']['total_training_flops'])],
        ["Transformer; width 256, four layers", "10M / four passes", f"{ev['tf10']:.3f}",
         compact_work(lm_costs['tf']['total_training_flops'])],
    ]
    reference_sources=[
        '<a href="experiments/results/e174/aligned_lstm_10m_20260930.json">10M LSTM aligned result</a>',
        '<a href="experiments/results/e174/aligned_tf_10m_20260930.json">10M Transformer aligned result</a>',
    ]
    official=[row for row in tasks['language_scaleup'] if row['args'].get('official_test')]
    if official:
        reference_rows.pop(0)
        for row in official:
            protocol=row['protocol']
            if (protocol['fitting']!=[0,10_000_000] or protocol['development']!=[90_000_000,90_200_000]
                    or protocol['test']!=[95_000_000,96_000_000] or not protocol['official_test_read']
                    or not protocol['weights_frozen_on_test'] or protocol['statistical_experts']
                    or row['final']['official_test']['n']!=999_999):
                raise ValueError('Completed learned language result does not match the comparison protocol')
            reference_rows.insert(0,[f"Ours: input-gated event state; width {row['args']['width']}",
                '10M / four passes',f"{row['final']['official_test']['bpc']:.3f}",
                compact_work(row['work']['total_training_unit_special_flops'])])
    for model, label in (("lstm", "LSTM; width 512, one recurrent layer"),
                         ("tf", "Transformer; width 256, four layers")):
        reference = ev["aws_references"][model]
        if reference:
            row = reference["result"]
            cost = row.get("training_flops_estimate", {}).get("total_training_flops")
            reference_rows.append([label, f"90M / {row['args']['passes']:g} passes",
                                   f"{row['test_bpc']:.3f}", compact_work(cost) if cost else "Not audited"])
            reference_sources.append(f'<a href="{reference["path"]}">90M {model.upper()} saved result</a>')
    language_reference_page=[
        ("h1","Appendix B (continued). Completed language references"),
        ("p","The Transformer and LSTM benchmarks have already been run. Their completed result files "
         "remain in the repository and are reused as reference targets for the full learned-event benchmark. "
         "Lower bits per character (bpc) means better prediction."),
        ("table",(["Model","Fitting characters / passes","Test bpc ↓","Full training FLOPs ↓"],
                  reference_rows,[60,46,24,44])),
        ("p","The 10M references score exactly the same 999,999 text8 targets in [95M,96M), "
         "with frozen validation-selected weights and cold initial context. The 90M LSTM uses the same "
         "test interval and its saved recurrent scoring protocol. All use the historical 27-character alphabet "
         "and 200,000-character validation selection; data budgets, capacities and fitting passes differ."),
        ("p","Training estimates include every fitting step, forward/loss, backpropagation, gradient clipping "
         "and Adam. Backward is approximated as twice forward; multiply-add counts as two FLOPs. "
         "G/T/P mean billion/trillion/quadrillion. Validation/test evaluation, memory traffic and runtime "
         "are outside these arithmetic totals."),
        ("h2","Ours: learned-language benchmark status"),
        ("p",("The completed learned-model row above scores the same cold-context character targets as "
         "the saved controls, with weights frozen. Its fitting arithmetic and additional special functions "
         "are recorded separately; the table includes unit-weight specials for comparison with the older "
         "neural estimates. Architecture/capacity and optimization differ. " if official else
         "The planned comparison uses 10M fitting characters, four passes, 200,000 validation characters "
         "and the same 1M test interval. Completed development stages select width 128 or 256. The "
         "six-layer content-gated candidates have 309,561 or 1,208,889 parameters. The earlier sequential "
         "run was paused after a clock-precision error; precise-clock staged fitting precedes promotion. "
         "The full test score and training work remain pending. ")+
         "The preserved 28,403-parameter pilot's 3.351 development bpc uses a smaller fitting budget "
         "and different split; it is not a comparable test result."),
        ("p",f"Earlier 1M-character references also remain saved: LSTM <b>{ev['lstm1']:.3f}</b> "
         f"and Transformer <b>{ev['tf1']:.3f} test bpc</b>, each with twenty fitting passes. "
         "The separate count/copy baseline and the cross-task Transformer/retrieval LSTM comparisons "
         "remain in their labeled sections and Appendix B."),
        ("small","Saved evidence: "+"; ".join(reference_sources)+". An earlier 90M Transformer attempt was "
         "interrupted by its RSS watchdog before producing a completed test result; its provenance is "
         "preserved.")]

    pages.append([
        ("h1","Why this research matters"),
        ("p","Sparse neural computation promises to spend work only where information changes. The hard "
         "part is teaching useful deep representations when routes can be silent, discrete or absent. "
         "A cheap forward pass is insufficient if discovering those routes consumes the savings."),
        ("table",(["Research lineage","What it established","The question we pursue"],[
         ["Neuromorphic and spiking networks","Event-driven signals and trained spike timing; EventProp differentiates at events.","How do inactive alternatives receive useful credit without exhaustive replay?"],
         ["Temporal logic and learned delays","Race/delay algebra computes with time; delay learning already improves SNN recognition.","Can temporal computation coexist with rich vector content and deep learned state?"],
         ["Sparse conditional models / MoE","Selected experts allow capacity to grow faster than active work; Switch trains at scale.","Can message timing, communication and correction work also become selective?"],
         ["Asynchronous state-space models","EventSSM learns asynchronous streams with parallel scans; this is a strong precedent.","Can hard races and counterfactual alternatives add quality per unit of total work?"],
        ],[39,66,69])),
        ("h2","What is distinctive here"),
        ("p","We combine computation through trainable time, content-bearing messages, persistent local state "
         "and credit to unrealized alternatives. Optionality asks whether distinct, reachable future corrections "
         "remain available under a work budget. The theory connects temporal algebra, key/value separation, "
         "credit transport and supervision that includes silence. The contribution is this construction and "
         "its tested consequences; learned delays and sparse capacity are established ideas."),
        ("p","Our earlier deep sparse-routing pilots often lost activity and useful credit before the final "
         "layers. Counterfactual proposals alone did not reliably fix that. The subsequent vector-state and "
         "persistent-memory work addresses those observed obstacles. Completed structured-task gains motivate "
         "the larger learned-model tests; broad quality, training efficiency and energy must still be measured together."),
        ("small",'Primary precedents: <a href="https://www.nature.com/articles/s41598-021-91786-z">EventProp</a>; '
         '<a href="https://arxiv.org/abs/2001.04242">Space-Time Algebra</a>; '
         '<a href="https://arxiv.org/abs/2306.17670">Learning Delays in SNNs</a>; '
         '<a href="https://www.jmlr.org/papers/v23/21-0998.html">Switch Transformers</a>; '
         '<a href="https://arxiv.org/abs/2404.18508">EventSSM</a>. '
         'Our routing failures and revised interpretations remain in the theory index and findings.')])

    pages.append([
        ("h1","One architecture, several learned computations"),
        ("p","The common idea is local computation triggered by an arrival: retain memory, combine the "
         "incoming vector with that memory, and choose an outgoing time. A delay changes which messages "
         "meet and which race finishes first. This makes timing part of the learned function. "
         "The model family implements this idea at several levels of generality."),
        ("figure",("shared_architecture",154)),
        ("table",(["Model","Mechanism","Evidence","What it establishes"],[
         ["Ours: learned event-state encoders","Source embeddings, temporal modes, nonlinear vector maps and competing clocks","Language E176; speech E165","Learned representations and persistent state"],
         ["Ours: routed event query encoders","Candidate payloads, receiver memory and hard value/time races","Breadth E120; language E133","Trainable event depth across tasks; text/market breadth variants add statistical evidence"],
         ["Ours: structured event mechanisms","Temporal chains, relative pointers and phase composition","E34/E53/E54, E61, E124","Sample efficiency and generalization with declared structural priors"],
         ["Ours: statistical controls","Conditional counts, backoff and copy probabilities","E173; online pilot","Separate baselines; no event backbone"],
        ],[38,57,33,46])),
        ("p","<b>Incoming content is retained.</b> In the event-state encoder, an incoming vector is "
         "projected into rotating, decaying memory. A learned memory read is mixed with a direct input path, "
         "normalized and gated. The outgoing vector adds that correction to the incoming vector. The payload "
         "therefore depends on both current content and history; timing supplies an additional control."),
        ("small","Each task has separately fitted weights. Current learned encoders use fixed depth and locally "
         "dense vector maps; the language scheduler retains state and delayed messages across chunks. "
         "The routed query encoder instead rebuilds a supplied context per query. Clock learning uses "
         "declared surrogate credit through a hard schedule; it is not an exact derivative of every order change.")])

    pages.append([
        ("h1","How the model computes and learns"),
        ("p","<b>Time performs computation.</b> An arrival time is a computational value. A delay adds to "
         "that value; a first-arrival race computes a minimum and selects a payload; coincidence detects "
         "the latest required arrival. Inhibition can veto a path. With a declared periodic reference, "
         "phase transformations compose reusable modular relations. Learned timing therefore changes "
         "the function and its causal paths, beyond deciding when a fixed dense calculation runs."),
        ("bullets",[
         "<b>Local temporal memory</b> accumulates observed content and elapsed time without evaluating empty time ticks.",
         "<b>Computation through delays</b> uses waiting times, arrival order and clock races to transform information and select outcomes.",
         "<b>Hard races and counterfactual credit</b> choose the emitted vector and delay. Losing alternatives teach better routes while remaining distinct from the winning forward message.",
         "<b>Separate keys and values</b> let content-dependent keys set routes and clocks while value learning preserves the selected schedule.",
         "<b>Trainable depth</b> carries representations and learning credit through a hierarchy. The reversible construction preserves conditional norms under its stated assumptions; readout visibility and routing remain necessary.",
         "<b>Structured memories</b> include relative pointers, learned phase transformations and conditional evidence. Their task gains retain their declared priors; statistical experts are labeled separately.",
         "<b>Appropriate supervision</b> teaches a completed class decision or the next event's type and waiting time, including information carried by silence."]),
        ("p","For a temporal pattern such as A followed by B, a learned delay can bring A's trace into "
         "coincidence with B. A competing path can veto the match when C intervenes. The timing-pattern "
         "and compositional experiments test these mechanisms; the language and speech encoders learn "
         "richer vector messages and temporal state."),
        ("small",'The primitives and their symmetry limits are developed in '
         '<a href="experiments/theory/05_temporal_computation_and_scaling.md">temporal computation theory, §56</a>; '
         '<a href="experiments/theory/01_foundations_and_counterfactual_credit.md">counterfactual learning</a>, '
         '<a href="experiments/theory/22_key_value_separation_and_race_boundaries.md">key/value separation</a> '
         'and <a href="experiments/theory/25_reversible_event_memory_and_depth.md">reversible depth</a> '
         "give the learning contracts. Clockless delay/race networks compute relative timing relations; "
         "phase arithmetic requires its reference. Each implemented model uses a declared subset. "
         "Candidate discovery and training alternatives are charged to the work ledger.")])

    pages.append([
        ("h1","A demonstrated advantage: generalization with less work"),
        ("p","The retrieval and phase computations preserve useful rules when the evaluation extends beyond "
         "the fitting examples. The saved compact Transformer and LSTM controls use the same synthetic "
         "evaluation targets. <b>Higher and further left is better:</b> more accurate answers from less "
         "estimated inference work. All points use one logical operation ledger."),
        ("figure",("consolidated_work_frontiers",174)),
        ("table",(["Ours: event computation","Held-out capability","Estimated work per query"],[
         ["Periodic path; 69 learned scalars",f"{phase['correct']:,}/{phase['n']:,} unseen triples",f"{phase_work:,.0f} logical operations"],
         ["Two-layer carrier + hard pointer","100% at four times context",f"{recall_work:,.0f} logical operations"],
        ],[62,57,55])),
        ("p","The phase rule costs 188 logical operations per triple; the saved LSTM and Transformer "
         "cost 104,518 and 155,592 and score 1.95% and 3.60%. At four times the recall context, the event "
         "encoder plus learned pointer scores 100% at 728,602 operations; both compact controls score "
         "7.42% at 2.24M and 4.46M operations. The phase model receives a periodic representation with "
         "period 17, and retrieval has a pointer mechanism. These useful priors explain the task advantage "
         "and are part of what must transfer to harder tasks."),
        ("p","<b>Learning work is also selective.</b> The periodic teacher makes 29,003 mistaken-example updates "
         "and 145,015 learned-scalar update visits. The dense arithmetic controls make 4,800 Adam steps: "
         "92.2M parameter visits for the LSTM and 132.5M for the Transformer. These count parameter updates, "
         "excluding optimizer state and backward arithmetic; they are not training FLOPs or joules."),
        ("small",f"Arithmetic: 1,473 fitting triples, a 200-epoch budget, all 3,440 unseen triples; supplied period 17. "
         f"The phase-only path stops after {tasks['phase_only']['actual_epochs']} passes, when an entire fitting pass makes no updates. Recall: "
         "4,000 pointer-fitting examples plus 512 neural-fitting examples; the dense controls receive all 4,512 "
         "examples for eight epochs. Width 32 and two generic layers where present, seed 6, one small dense setting. "
         "Work is an analytic logical-operation estimate, including configured vector maps, routers, scans, "
         "normalization, clock candidates and pointer search. These inference counts are not measured joules or "
         "backward/optimizer counts. Full work definitions appear in the evidence appendix.")])

    pages.append([
        ("h1","A demonstrated advantage: learning temporal structure"),
        ("p","Event chains learn timing patterns and compose recognizable parts into ordered structures. "
         "The preserved five-run results show accurate recognition and strong sample efficiency. The right "
         "panel compares distinct examples rather than incompatible activity counters."),
        ("figure",("supremacy_map",174)),
        ("table",(["Preserved comparison","Ours","Transformer","Fitting protocol"],[
         ["Timing patterns","99.95–100%; five runs","99.60–99.80%; two runs","200k examples once / 1M with relative-time bias"],
         ["Shared-motif composition","99.00–99.93%; five runs","99.60–99.85%; two runs","40k examples once / 1M with relative-time bias"],
         ["Depth-four order","99.90–100%; five runs","98.95–99.05%; two runs","Same 40k distinct examples; one pass / 50 passes"],
         ["Depth-three order at 2k examples","99.73–99.93%; five runs","33.25–40.80%; two runs","One pass / repeated fitting on the same 2k examples"],
        ],[45,41,41,47])),
        ("p","The strongest depth-four comparison has approximately ten times fewer classification errors "
         "despite the event learner seeing each fitting example once. The sample-efficiency curve makes the "
         "next question concrete: can learned embeddings and broader event representations retain that advantage "
         "when the inputs no longer supply known temporal parts?"),
        ("small","Completed E35, E34, E53/E54 and E36 files. Ranges describe the recorded runs, not confidence "
         "intervals. Architectures and optimization differ; synthetic evaluation sets were reused during research. "
         "Event deliveries remain activity measurements. Total arithmetic, backward and optimizer work require "
         "a declared ledger; activity divided by dense MACs is not a training-cost or power ratio.")])

    pages.append([
        ("h1","Why asynchronous, sparse computation matters"),
        ("p","Persistent local state can retain experience without repeatedly reconstructing a whole history. "
         "An asynchronous node updates when useful information arrives. A hard race emits one chosen message. "
         "These mechanisms create a path to spending less computation and moving less data per useful answer. "
         "On an event-oriented processor, inactive nodes and communication links could remain idle."),
        ("h2","Spend the next unit of work where it helps"),
        ("p","A larger budget can buy a longer memory, better retrieval, a deeper representation for difficult "
         "inputs or more local adaptation. The development rule is to measure the held-out improvement from "
         "each choice and allocate work to the most useful one. The phase and pointer results show why an "
         "appropriate computation can be much cheaper than a generic dense approximation. The scaling "
         "program tests how much of this flexibility survives when that computation must itself be learned."),
        ("table",(["Architecture","Fitting known sequences","Generating or processing a stream"],[
         ["LSTM","Gates depend on prior hidden state; recurrent work is sequential","Retained hidden state; dense gate updates per token"],
         ["Transformer","Causal attention permits sequence-parallel fitting","Sequential token generation with a key/value cache; adaptation is possible"],
         ["Ours: Sleeping Machines","Affine event memory supports parallel scans once incoming values/times are known","Retained local state and delayed-message queue; only due arrivals execute"],
        ],[38,68,68])),
        ("p",f"Parallel fitting and sequential generation are compatible. The precise-clock language scan "
         f"gives a {tasks['parallel_contract']['measured_step_speedup']:.2f}× complete-step CPU speedup "
         "against serial execution of the same width-256 model, including backward, clipping and Adam. "
         "Predictions, states, gradients, chunk boundaries and causality are checked. This follows a "
         "bounded-delay schedule; arbitrary reordering networks require their own contract."),
        ("p",'Parallel recurrent computation also appears in '
         '<a href="https://proceedings.mlr.press/v119/katharopoulos20a.html">linear attention</a> and '
         '<a href="https://arxiv.org/abs/2312.00752">selective state-space models</a>. '
         'EventSSM already processes asynchronous events with scans. These are important controls. '
         'The distinctive hypothesis here is the combination of learned timing, sparse communication, '
         'credit to alternatives and independent compute budgets; it must earn its advantage empirically.'),
        ("small","Primary FLOPs describe the declared event algorithm, including required vector maps, "
         "candidate computation, scans, backward and optimizer updates. Simulator padding and dispatch are "
         "separate implementation overhead. Physical power also depends on memory, queues, communication "
         "and hardware utilization. Demonstrated work savings and projected power savings are distinguished; "
         "total device joules have not yet been measured.")])

    pages.append([
        ("h1","A mathematical foundation for trainable computation"),
        ("table",(["Principle","What it enables"],[
         ["Stable transport through depth","The reversible packet/memory construction preserves conditional value and credit norms under its stated operator and boundary assumptions. Readout visibility, routing and optimization remain separate requirements."],
         ["Active communication support","Inputs need causal paths through which to interact. A context channel supplies joint information when sparse packets leave local groups disconnected."],
         ["Credit to unrealized alternatives","A losing payload or timing choice can show how a different route would change the outcome, while forward computation remains a hard race."],
         ["Periodic state as an isometry","Learned rotations/reflections have unit-magnitude occurrence derivatives. Their composition supports reusable arithmetic instead of a table of observed tuples."],
         ["Certified composition","Target-constrained min/max composition of phase errors certifies the fitted modular rule across all 4,913 possible tuples; exhaustive checking confirms it."],
         ["Natural supervised credit","Categorical and event likelihoods both credit predicted sufficient statistics minus observations. Silence enters through integrated exposure."],
         ["Useful optionality","Reserve consists of distinct, attainable future corrections under a causal work budget. Reachability and transferable learning matter alongside immediate loss."],
         ["Statistically useful credit","Expected improvement must overcome the curvature cost of fitting noise. Cross-example teacher agreement separates reproducible correction from raw gradient magnitude."],
        ],[57,117])),
        ("h2","From mathematics to an engineering discipline"),
        ("p","The theory connects representation, topology, clocks and optimization. Expressivity describes what "
         "a network can compute; transport describes whether information and credit survive; the objective describes "
         "what the teacher asks it to learn. These pieces must agree. The periodic certificate is one concrete case "
         "where the formal model explains and verifies a learned computation."),
        ("p","A common implementation makes these principles reusable across tasks. Efficient primitives can own "
         "a computation when its structure is known; a deep carrier can learn representations when it is not. "
         "The research objective is to combine this flexibility with affordable route discovery and increasingly "
         "capable models."),
        ("small","Formal derivations and their assumptions are indexed in the project's theory notes. Conditional "
         "stability and a certificate for a fitted rule do not establish global optimizer convergence or a scaling law. "
         "The program builds on established deep spiking and sparse conditional computation; its focus is the "
         "combination of useful temporal operators, hard causal routing and counterfactual learning. Related survey: "
         '<a href="https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full">'
         'Direct training of deep spiking networks</a>.')])

    pages.append([
        ("h1","Potential grounded in the completed evidence"),
        ("p","The objective is capable models that spend computation where it improves an answer. The saved "
         "structured-task results and learned stream pilots provide specific starting points. Moving from "
         "those mechanisms to frontier prediction requires useful representations, longer context and "
         "measured quality at a fixed total resource budget."),
        ("table",(["Opportunity","Present foundation","What would establish the larger case"],[
         ["Compact prediction and memory","Persistent learned state; pointers generalize to longer contexts.","Competitive held-out language quality, retention and complete fitting/inference cost."],
         ["Continuous perception","Learned temporal speech representations and composition.","Full speech/vision tests and confidence-based early decisions at measured latency."],
         ["More useful training per budget","Structured-task sample efficiency; exact causal parallel scans.","Quality improvements at equal total fitting work, including route discovery."],
         ["Dormant skills and selective depth","Hard race routing and compute-allocation theory.","Learned marginal work allocation that outperforms fixed allocation on real tasks."],
         ["Mobile and industrial autonomy","Local state and event-triggered updates.","Device joules, memory traffic and task quality measured on deployable implementations."],
        ],[40,62,72])),
        ("h2","Why the hardware and economic implications could be large"),
        ("p","If comparable quality needs less total training and inference energy, a fixed power and capital "
         "budget can support more capable models, more research or continuous adaptation on robots, phones "
         "and instruments. Persistent local state and selective communication would favor hardware that "
         "handles message delivery, queues and memory efficiently. These are conditional consequences of "
         "measured savings, rather than savings inferred from event counts."),
        ("p","The near-term experiment asks where the next unit of computation helps: longer memory, "
         "richer content transformations, retrieval, depth or local learning. Held-out gains and complete "
         "work determine promotion. A successful scaling result must show that those gains continue across "
         "independently fitted sizes and budgets.")])

    scaling=sorted(tasks["language_scaling"],key=lambda row:row["parameters"])
    memory=sorted(tasks["language_memory"],key=lambda row:row["args"]["memory_profile"])
    memory+=sorted(tasks["language_selective"],key=lambda row:row["args"]["memory_profile"])
    memory_blocks = ([
        ("h2","From longer memory to selective content"),
        ("table",(["Ours: memory variant","Parameters","Development bpc ↓","Fitting GFLOPs ↓"],[
         [("Input gates / " if row['args'].get('content_memory') else "Constant / ")+
          row['args']['memory_profile'].replace('_',' '),f"{row['parameters']:,}",
          f"{row['final']['dev']['bpc']:.3f}",f"{row['work']['total_training_arithmetic_flops']/1e9:,.2f}"]
         for row in memory],[64,32,37,41])),
        ("small","Identical width, seed, data, four passes and 64-character credit horizon. Constant-memory "
         "arms vary initial timescales/frequencies. Input-gated arms add content-dependent write/forget controls "
         "(0.50% more parameters, 1.41% more fitting arithmetic). Longer decay alone worsens this fit. "
         "These are single-seed development comparisons; complete numerical contracts precede training.")
    ] if memory else [("p","The inherited event initialization leaves individual modal timescales at only "
         "a few character intervals after fitting. A matched small ablation now tests longer decay times and "
         "resolved temporal periods before committing to the large run. Modal decay is a diagnostic, not a "
         "hard bound on the complete stack's context.")])
    pages.append([
        ("h1","Ours: language learning and staged scale-up"),
        ("p","These learned models use character embeddings, gated residual content transformations and "
         "persistent rotating/decaying state. Temporal modes encode relative token distance. No statistical "
         "count, copy or word experts provide their predictions. The goal is competitive quality from a "
         "learned backbone before claiming a language compute advantage."),
        ("figure",("language_capacity_scaling",166)),
        ("table",(["Ours: width","Parameters","Development bpc ↓","Total fitting GFLOPs ↓"],[
         [str(row['args']['width']),f"{row['parameters']:,}",f"{row['final']['dev']['bpc']:.3f}",
          f"{row['work']['total_training_arithmetic_flops']/1e9:.2f}"] for row in scaling],[35,42,45,52])),
        ("small","All curves use six layers, seed 6, 131,072 fitting characters, four passes and the same "
         "8,191 cold-context development targets. Credit is truncated every 64 characters; memory persists. "
         "FLOPs include forward/loss, backward, clipping and Adam; special functions are reported separately "
         "in each result. These runs vary capacity at equal data/passes, rather than equal compute, and "
         "establish neither a scaling law nor official-test superiority."),
    ]+memory_blocks)

    if tasks['language_representation']:
        audit=tasks['language_representation'];gated=audit['rows'][-1]['interventions']
        pages.append([
            ('h1','Ours: content and memory in fitted language models'),
            ('p','An event carries information about its input. The learned vector is a transformation of '
             'incoming content and persistent state, with a residual path and an output gate. The new '
             'candidate also gates memory writing and forgetting from incoming content. These checks '
             'measure whether the fitted predictions use those paths.'),
            ('figure',('language_content_memory_audit',170)),
            ('p',f"Resetting all history before each character preserves its current embedding and learned "
             f"content transformations, but raises the gated model's development loss from "
             f"{gated['full']['bpc']:.3f} to {gated['reset_history_every_token']['bpc']:.3f} bpc. "
             f"Zeroing incoming embeddings raises it to {gated['zero_incoming_embeddings']['bpc']:.3f}. "
             'Both present input and earlier messages contribute to prediction.'),
            ('h2','Selective retention adds a useful control'),
            ('p','The two input gates start at one, preserving the constant-memory model exactly at '
             'initialization. During fitting they learn different write strengths and forgetting factors '
             'for different incoming vectors. A factor below one slows decay; above one accelerates it. '
             'The controls are known from the causal previous layer, so serial execution and parallel '
             'affine scans retain their checked outputs and teachers.'),
            ('p','In the matched small fit, gates improve 2.643 to 2.587 bpc for 0.50% more parameters '
             'and 1.41% more fitting arithmetic. This is a local quality/work improvement, with one seed. '
             'All six layers still execute for every character; dormant-unit scaling remains a separate target.'),
            ('small','Frozen selected checkpoints, identical 8,191 cold development targets, no training '
             'or official-test reads. These interventions disrupt a trained model; they establish fitted '
             'dependence, not the quality of retrained ablated architectures or lossless storage. Source: '
             'parallel_language/local_language_representation_20260930T162337Z.json.')])

    if tasks['language_scaleup']:
        rows=sorted(tasks['language_scaleup'],key=lambda row:(row['args']['fit'],row['args']['width']))
        larger_blocks=[
            ('h1','Ours: larger language development stages'),
            ('p','Each stage fits independently from initialization. Data, capacity and memory controls are '
             'declared below. Development selects weights within the fixed four-pass budget; ongoing '
             'training logs are never substituted for a completed result.'),
            ('table',(['Ours: memory / width','Fit characters','Parameters','Development bpc ↓','Fitting TFLOPs ↓'],[
             [('Input gates' if row['args'].get('content_memory') else 'Constant')+f" / {row['args']['width']}",
              f"{row['args']['fit']:,}",f"{row['parameters']:,}",f"{row['final']['dev']['bpc']:.3f}",
              f"{row['work']['total_training_arithmetic_flops']/1e12:.3f}"] for row in rows],[46,34,29,34,31]))]
        capacity=[row for row in rows if row['args']['fit']==131072 and row['args']['width']==256]
        if capacity:
            smaller=next(row for row in tasks['language_selective'] if row['args']['memory_profile']=='inherited')
            larger=capacity[-1]
            extra=smaller['final']['dev']['bpc']-larger['final']['dev']['bpc']
            ratio=larger['work']['total_training_arithmetic_flops']/smaller['work']['total_training_arithmetic_flops']
            larger_blocks += [
                ('figure',('language_compute_choices',170)),
                ('p',f"On the identical 131K-character/four-pass screen, widening the gated model buys "
                 f"{extra:.3f} bpc for {ratio:.2f}× the total fitting arithmetic. Adding input gates at "
                 'width 128 instead improves 0.057 bpc for 1.41% more arithmetic. This makes the '
                 'allocation question quantitative: measure useful correction before spending broadly '
                 'on width. These are finite, single-seed interventions, rather than a scaling law.')]
        larger_blocks += [
            ('p','The fixed-capacity data comparison and fixed-data capacity comparison answer different '
             'questions. Equal passes and data do not imply equal compute. The pipeline checks finite '
             'learning, trained value blocks, complete work and source provenance before promotion. '
             'A development gain is not an official-test or frontier claim.'),
            ('small','Precise clocks; float32 payloads; causal persistent state; 64-character credit horizon. '
             'Special functions, evaluation passes and physical traffic are separate from the arithmetic '
             'ledger. One seed, no statistical experts. Source: experiments/results/parallel_language.')]
        pages.append(larger_blocks)

    pages.append([
        ("h1","What establishes the larger advantage"),
        ("p","The ambition is a common model family whose strongest mechanisms remain useful as tasks, data and "
         "capacity grow. Arithmetic and retrieval retain their demonstrated strengths in the consolidated "
         "implementation. The strongest language mixture is specialized; the generic backbone still needs to "
         "demonstrate competitive learned representations. Character and subword-token budgets must be distinguished."),
        ("table",(["Objective","Decisive evidence"],[
         ["Generic language scaling","Train a learned event backbone that owns the prediction. Scale through declared data budgets with matched Transformer, recurrent and state-space references; record loss, capacity, complete training work, memory traffic, time and joules."],
         ["Preserve capabilities","Repeat established generalization and sample-efficiency results within the common model family, with task-appropriate depth and explicit resource accounting."],
         ["Strong real-event recognition","Accurate speech and event-camera decisions on complete held-out benchmarks; calibrated confidence and time-to-answer."],
         ["Learn routes and representations at scale","Reliable deep credit and useful counterfactual alternatives as width, depth, memory and data increase."],
         ["Efficient persistent operation","Maintain local state and pending messages across queries, preserving causal predictions while reducing repeated work."],
         ["Lower total energy at useful quality","Measure training and inference joules, memory traffic, latency and communication under declared hardware and quality targets."],
        ],[57,117])),
        ("p","Success on these dimensions would turn the current task-level advantages into a broader foundation "
         "for frontier models. The project's distinctive resources—timing, local memory, hard selection and credit "
         "to alternatives—remain the guide for architecture and learning.")])

    speech_runs=[("Original eight-layer parent",tasks["shd_scaled"])]
    for key,label in (("shd_warm","Warm larger-data continuation"),
                      ("shd_fine","Fine source messages"),
                      ("shd_phase","Fine sources + temporal phase"),
                      ("shd_local","Fine/phase learner; parent frozen")):
        if tasks[key] is not None:speech_runs.append((label,tasks[key]))
    state_residual=tasks["shd_state_residual"]
    audit_sentence=""
    if tasks["shd_state_summary"] is not None:
        audit=tasks["shd_state_summary"]["disjoint_audit"]
        audit_sentence=(f" On {audit['n']} disjoint utterances from the same held speakers, accuracy improves "
            f"from {100*audit['parent_correct']/audit['n']:.2f}% to {100*audit['residual_correct']/audit['n']:.2f}%.")
    if state_residual is not None:
        best_row=max(state_residual["curve"],key=lambda r:r["dev"]["correct"])
        speech_runs.append((f"Parent + parallel six-block residual; pass {best_row['epoch']}",{"final":best_row}))
    single_paired=tasks["shd_single_paired"]
    if single_paired is not None:
        selected=min(single_paired["curve"],key=lambda r:(-r["dev"]["correct"],r["dev"]["nll"]))
        speech_runs.append((f"Single six-block temporal encoder; pass {selected['epoch']}",{"final":selected}))
    for key,label in (("shd_calibrated_d6","Calibrated six-block continuation"),
                      ("shd_calibrated_d12","Bounded twelve-block encoder"),
                      ("shd_observer_depth","Directional twelve-block encoder")):
        if tasks[key] is not None:speech_runs.append((label,tasks[key]))
    if tasks['shd_selected_prefix'] is not None:
        speech_runs.append(('Selected single six-block prefix; trained with twelve blocks',tasks['shd_selected_prefix']))
    best_label,best_speech=max(speech_runs,key=lambda entry:(pooled(entry[1]),
        -entry[1]['final'].get('dev',{}).get('nll',float('inf'))))
    speech_rows=[]
    display_runs=speech_runs
    if state_residual is not None:
        display_runs=[speech_runs[0]]+[(f"Parent + six-block temporal residual; pass {r['epoch']}",{"final":r})
            for r in state_residual["curve"]]
        if single_paired is not None:
            display_runs=[speech_runs[0],(f"Parent + six-block residual; pass {best_row['epoch']}",{"final":best_row})]
            for label,run in speech_runs:
                if label.startswith(("Single six","Selected single","Calibrated","Bounded","Directional")):display_runs.append((label,run))
    for label,row in display_runs:
        parts=[row["final"]["dev"]] if "dev" in row["final"] else [row["final"][k] for k in ("dev_original","dev_additional")]
        correct=sum(p["correct"] for p in parts)
        speech_rows.append([label,f"{correct}/512",f"{100*pooled(row):.2f}%"])
    pages.append([
        ("h1","Appendix A. Deep event recognition"),
        ("p",f"The strongest completed speech result in the model family is <b>{100*pooled(best_speech):.2f}%</b> "
         f"on 512 private held-speaker utterances ({best_label.lower()}). "
         "Published official-test results below use a different partition; they are reference targets."),
        ("table",(["Ours: private development configuration","Correct","Accuracy ↑"],speech_rows,[108,32,34])),
        ("figure",("e143_temporal_residual_learning" if state_residual is not None else "e139_source_information",152)),
        ("p",("A six-block width-128 temporal encoder learns corrections while the inherited eight-layer parent "
         "stays frozen. It adds 395,814 parameters to the parent's 53,296. Signed modal states, nonlinear gates "
         "and residual vectors learn from all source identities and original event times before causal pooling. "
         "This is a larger parallel model, not fourteen sequential layers; completed-utterance supervision "
         "does not yet teach calibrated early answers."+audit_sentence if state_residual is not None else
         "Raw channel identities and original source times now enter learned vector messages before "
         "coalescing. Signed temporal rotations extend the receiver memory, with the old mean as its zero-phase "
         "case. Initial predictions, hard winners and clocks match the trained parent exactly. Each packet "
         "still emits one winning vector and delay; source and temporal transformations add measured work.")),
        ("table",(["Published reference; official-test protocol","Reported accuracy ↑"],[
         ["Ours: full official SHD comparison","Pending"],
         ["EventSSM: asynchronous learned state-space layers","95.9%"],
         ["S7: input-dependent temporal state","96.3%"],
        ],[131,43])),
        ("small","Our private sample uses training-file speakers 3/6; official test accuracy is unmeasured. "
         + ("The temporal residual uses three passes and a fresh optimizer; its larger capacity and budget "
            "are not a matched single-factor comparison. Its listed score selects the best private-development "
            "epoch; the curve shows all three. " if state_residual is not None else "") +
         "The original parent was fitted on 4,096 examples. Continuation timers exclude loading; the residual "
         "timer includes it. Both include evaluation and are not energy measurements. One seed. References: "
         '<a href="https://arxiv.org/html/2404.18508v2">EventSSM</a>, '
         '<a href="https://arxiv.org/html/2410.03464v1">S7</a>. '
         "These published scores were checked against the original papers; they are targets for a full "
         "official-test comparison, not scores on our private split.")])

    if tasks["shd_state_ablation"] is not None and tasks["shd_state_summary"] is not None:
        summary=tasks["shd_state_summary"]
        reset=tasks["shd_state_ablation"]["rows"]
        reset_rows=[]
        for key,label in (("trained","All learned parameters retained"),("reset_clocks","Only hidden clocks reset"),
            ("reset_source_embedding","Only source vectors reset"),("reset_state_stack","Only state stack reset"),
            ("reset_modal_dynamics","Only decay/frequency parameters reset")):
            part=reset[key]["held"]
            reset_rows.append([label,f"{part['correct']}/512",f"{100*part['accuracy']:.2f}%"])
        pages.append([
            ("h1","Appendix A (continued). Learned timing and transfer"),
            ("p","<b>Learned delays contribute to the answer.</b> Restoring the hidden clocks to their initial "
             "values, with source vectors, state/value maps and the trained classifier retained, loses 13 correct "
             "answers. Timing changes the temporal interactions used by the representation; it performs computation."),
            ("figure",("e145_learned_timing_and_transfer",174)),
            ("table",(["Ours: same trained readout; subsystem reset","Correct","Accuracy ↑"],reset_rows,[108,32,34])),
            ("p","The temporal stack and source vectors also learn useful coordinated representations. Resetting "
             "the stack loses 50 correct answers; resetting sources loses 35. Decay/frequency resets change one "
             "decision. These changes depend on the fitted solution's coordination; their effects cannot be added "
             "or treated as a matched comparison of retrained architectures."),
            ("p","The 657-utterance audit is disjoint from fitting and the development sample and uses the selected "
             "checkpoint unchanged. It gains 62 correct answers and loses 15, for a net improvement of 47. "
             "Both samples use the same two held training speakers. Official-test and additional-speaker "
             "generalization are the next evaluation targets."),
            ("h2","A stronger mathematical account of routing"),
            ("p","An affine packet summary can preserve both the final state and the average of raw temporal "
             "states, including their teachers. This permits richer pooling before expensive nonlinear maps. "
             "A second derivation shows why many almost-equal delays may offer little usable choice: their "
             "effects point in nearly the same direction. Diverse payloads and temporal modes, sufficient delay "
             "spread and downstream visibility determine useful route reserve."),
            ("small","One exploratory run: 449,110 total parameters, inherited parent plus a trained six-block "
             "encoder. Three passes use about 33 minutes including preparation/evaluation and 1.79 GiB peak RSS; "
             "the guarded host retains at least 10,361 MiB sampled available memory. Complete physical work and "
             "joules remain unmeasured. Formal statements and numerical contracts are in THEORY §§226–236.")])
    elif tasks["shd_exchange"] is not None:
        state=tasks["shd_exchange"]["state"]["final"]
        packets=tasks["shd_exchange"]["packets"]["final"]
        ablation=tasks["shd_exchange"]["ablation"]
        query_rows=[
            ["Full retained-state query","140,428",f"{100*state['fit']['accuracy']:.1f}%",f"{100*state['held']['accuracy']:.1f}%"],
            ["Emitted-packet query","2,188 active",f"{100*packets['fit']['accuracy']:.1f}%",f"{100*packets['held']['accuracy']:.1f}%"],
        ]
        compact_blocks=[]
        if tasks["shd_compact"] is not None:
            rows=tasks["shd_compact"]["rows"]
            learned,frozen=rows["learned"],rows["frozen"]
            for mode,label in (("learned","Compact; learned angles"),("frozen","Compact; fixed angles")):
                last=rows[mode]["curve"][-1]
                query_rows.append([label,f"{rows[mode]['train_parameters']:,}",
                                   f"{100*last['fit_accuracy']:.1f}%",f"{100*last['held_accuracy']:.1f}%"])
            compact_blocks=[("p",f"A rank-16 bank/channel/class query uses 4,968 decoder parameters. Learned exchanges "
                f"reach {100*learned['curve'][-1]['held_accuracy']:.1f}% held accuracy versus "
                f"{100*frozen['curve'][-1]['held_accuracy']:.1f}% with fixed angles. These compact arms share initial "
                "predictions, calibration, keys, query capacity, examples and optimizer budget. Both embeddings and "
                "queries learn; only angle adaptation is disabled. This isolates useful exchange adaptation under constrained supervision."),
                ("small","Compact/full terminal queries estimate 115,028/138,900 forward MACs, excluding key/exchange work, "
                 "normalization, backward, optimizer and traffic. Parameter compression is much larger than this query-work saving; energy is unmeasured.")]
        pages.append([
            ("h1","Appendix A (continued). Trainable event memory at twelve layers"),
            ("p","A winning key selects one memory bank. Its orthogonal exchange stores and emits vector information "
             "with the winning delay. Conditional packet/state norms survive depth; the completed query reads retained "
             "memory. All twelve exchange layers receive credit."),
            ("figure",("e136_scattering_learning",174)),
            ("table",(["Ours: completed three-pass query","Trainable parameters","Fit: higher is better","Held: higher is better"],query_rows,[63,35,38,38])),
            ("p",f"Resetting learned angles preserves all 1,024 full-query fitting decisions, "
             f"while held accuracy changes from {100*ablation['trained']['held']['accuracy']:.1f}% to "
             f"{100*ablation['all_angles_reset_same_decoder']['held']['accuracy']:.1f}%. This frozen-checkpoint probe shows "
             "angle contribution/coadaptation, not a retrained control. The full-state and packet queries differ in active decoder capacity."),
            *compact_blocks,
            ("small","Exploratory seed 6: 1,024 unaugmented fitting utterances; 512 held training-file speakers; "
             "three passes, width 32. Each model retains 53,296 frozen key parameters from the eight-layer/4,096-fit checkpoint. "
             "Query-fitting examples are a subset of its fitting data. These are not from-scratch or matched continuations. All-layer state queries provide "
             "direct supervision. Official SHD test data and calibrated early decisions remain untested.")])

    if tasks["shd_single_clean"] is not None and single_paired is not None:
        clean_run=tasks["shd_single_clean"]
        selected=min(single_paired["curve"],key=lambda r:(-r["dev"]["correct"],r["dev"]["nll"]))
        model_rows=[]
        for label,part in (("Combined model: parent + temporal correction",best_row["dev"]),
            ("Single encoder: clean head, before continuation",clean_run["conditioned_initial"]["dev"]),
            ("Single encoder: paired head, selected pass zero",single_paired["conditioned_initial"]["dev"])):
            model_rows.append([label,f"{part['correct']}/512",f"{100*part['accuracy']:.2f}%",f"{part['nll']:.3f}"])
        if tasks['shd_selected_prefix'] is not None:
            part=tasks['shd_selected_prefix']['final']['dev']
            model_rows.append(['Single encoder: selected trained prefix',f"{part['correct']}/512",
                f"{100*part['accuracy']:.2f}%",f"{part['nll']:.3f}"])
        transfer_text=""
        timing_text=""
        if tasks["shd_single_audit"] is not None:
            audit=tasks["shd_single_audit"]
            original= audit["rows"]["combined"]
            single= audit["rows"]["single_paired"]
            transfer_text=(f" On the reused 657-utterance disjoint audit, the single encoder reaches "
                f"{100*single['audit']['accuracy']:.2f}% versus {100*original['audit']['accuracy']:.2f}% "
                "for the combined model. This audit is excluded from updates and checkpoint selection.")
            timing_text=(f" One CPU forward evaluation of those utterances takes {single['forward_wall_s']:.2f} s "
                f"for the single encoder and {original['forward_wall_s']:.2f} s for the combined model. "
                "This includes packing/query work and excludes loading; it is one timing observation, not joules.")
        if tasks['shd_selected_prefix'] is not None:
            selected_prefix=tasks['shd_selected_prefix']
            transfer_text=(f" The selected trained prefix reaches {selected_prefix['reused_audit']['correct']}/657 "
                f"({100*selected_prefix['reused_audit']['accuracy']:.2f}%) versus 510/657 (77.63%) for the combined model "
                "on the reused disjoint audit. Audit labels do not choose the checkpoint.")
            timing_text=(f" One CPU forward evaluation takes {selected_prefix['observed_forward_wall_s']:.2f} s for "
                "the selected prefix versus 23.71 s for the combined model. Packing/query included, loading excluded; "
                "one timing observation, not joules.")
        pages.append([
            ("h1","Appendix A (continued). One temporal encoder"),
            ("p","A six-block temporal encoder retains nearly all the combined model's development accuracy "
             "through one ordinary query head. Deployment removes the frozen parent: 395,814 parameters "
             "replace 449,110. Modal states, gated vector messages and winning delays remain. Its weights "
             "inherit earlier encoder training; combined teacher predictions are used only to initialize the head."),
            ("table",(["Ours: private development configuration","Correct","Accuracy ↑","NLL ↓"],model_rows,[98,27,27,22])),
            ("figure",("e152_single_encoder_learning",174)),
            ("p","Fitting the head on clean and transformed speech improves held accuracy by 32 answers "
             "with the temporal features frozen. Its covariance penalty suppresses class-visible nuisance "
             "variation. The right panel compares the heads on the same features; the left shows all subsequent "
             "unrestricted encoder passes."+transfer_text),
            ("p","Training the directional twelve-block extension, then selecting its six-block prefix on development, "
             "retains the combined model's 408 correct answers with lower NLL and 395,814 deployed parameters. "
             "The extra training blocks are removed after their learned contributions reduce held accuracy. "
             "This improves deployment quality/work; it does not establish a positive deep-block accuracy gain."),
            ("small","Seed 6, private train-file speakers 3/6; official-test parity remains unmeasured. "
             "Head fitting uses 6,144 unique fitting utterances: one clean view for the first arm, clean plus "
             "one transformed view for the paired arm. The arms also change regularization and use three/two "
             "encoder passes respectively, so total budgets are not matched. All continuation epochs are plotted; "
             "selection uses development accuracy, then NLL. Extra teacher/cache/head work and inherited fitting "
             "must be charged. Modal/vector maps are locally dense; no empty ticks or event-pair attention are added. "
             "The selected prefix additionally inherits the full twelve-block fitting pass; pruning does not erase "
             "that training cost. Prefix/full choice is post-hoc private-development selection. "
             "Formulae and numerical checks: THEORY §§237–264."+timing_text)])

    if tasks['shd_observer_depth'] is not None and tasks['shd_observer_audit'] is not None:
        depth_rows=[]
        for key,label in (('shd_calibrated_d6','Six-block control'),('shd_calibrated_d12','Twelve blocks: bounded outputs'),
                          ('shd_observer_depth','Twelve blocks: directional units')):
            run=tasks[key];final=run['final']
            depth_rows.append([label,f"{run['deployed_parameters']:,}",f"{100*final['fit']['accuracy']:.2f}%",
                f"{100*final['dev']['accuracy']:.2f}%",f"{final['dev']['nll']:.3f}"])
        audit=tasks['shd_observer_audit'];paired=audit['paired']['its_trained_prefix']
        pages.append([
            ('h1','Appendix A (continued). Making depth useful'),
            ('p','Identity growth preserves the classifier and old teachers while added output maps receive '
             'label credit. They must also change useful features. A tightly bounded twelve-block extension '
             'learns weights but changes no audit decisions when its six appended blocks are removed.'),
            ('table',(['Ours: one matched fitting pass','Parameters','Fit accuracy ↑','Private accuracy ↑','NLL ↓'],depth_rows,[68,29,26,29,22])),
            ('figure',('e164_depth_use_and_work',174)),
            ('p','Directional conditioning normalizes temporal-state features before their output map. An '
             'invertible coordinate change rescales classifier-sensitive directions and preserves hidden null '
             'directions for later computation. The fixed transform folds into an ordinary map at deployment. '
             'Initial outputs and old teachers remain exact; fitting replays verify the actual proposed update.'),
            ('p',f"The directional model reaches {audit['rows']['directional_d12']['audit']['correct']}/657 on the reused audit. "
             f"Removing its six appended blocks changes {paired['changed_predictions']} predictions: "
             f"{paired['full_only_correct']} are correct only with the blocks and {paired['reference_only_correct']} only without them. "
             'This measures fitted contribution with the trained prefix/head retained; it is not a retrained architecture comparison.'),
            ('small','All arms inherit the paired-head checkpoint and use 6,144 fitting utterances, the same order, '
             'channel/time transformations and one encoder pass. Old-group LR is 0.0000203125; directional new groups '
             'use 0.0001953125 from fitting-only replay. Changed normalization and update coordinates form one '
             'intervention. New blocks initially add 36 ms latency; labels supervise completed untimed utterances. '
             'Audit reuse is explicit; official-test parity remains unmeasured. CPU points are one warmed observation '
             'per model, packing/query included and loading excluded; energy is unmeasured. Weight-coordinate folding, '
             'all source work and added depth must be charged during training. Local maps remain dense, with no empty '
             'ticks or event-pair attention. Theory §§249–264.')])

    coverage=[]
    breadth={row["task"]:row for row in tasks["breadth_work"]["rows"]}
    total_work={row["task"]:row for row in tasks["training_work"]["rows"]}
    for task,label in (("language","Text8"),("market","Market event prediction"),("temporal","Temporal composition"),
                       ("mnist","MNIST"),("dvs","Event-camera gestures")):
        row=breadth[task]
        def score(metric):
            if task=="language":return f"{metric['nll']/math.log(2):.3f} bpc"
            if "accuracy" in metric:return f"{100*metric['accuracy']:.1f}%"
            return f"{metric['nll']:.3f} nats/event"
        direction = "bpc ↓" if task=="language" else "nats/event ↓" if task=="market" else "accuracy ↑"
        coverage.append([label+"<br/>"+direction,score(row["common_metric"]),score(row["reference_metric"]),
             f"{row['common_forward_map_scan_flops']/1e6:.2f}",
             f"{row['reference_forward_map_attention_flops']/1e6:.2f}"])
    pages.append([
        ("h1","Appendix B. Breadth of the common implementation"),
        ("p","These small development screens test one implementation across tasks. Text and market variants "
         "include separately fitted statistical evidence. Ours denotes Sleeping Machines; TF is the saved "
         "Transformer. Accuracy improves upward; prediction loss and FLOPs improve downward."),
        ("table",(["Task / quality direction","Ours: quality","TF: quality","Ours: inference MFLOPs ↓","TF: inference MFLOPs ↓"],coverage,[42,29,29,37,37])),
        ("figure",("breadth_work_ratios",158)),
        ("p","Ours uses eight layers and TF two, both width 32, with the same neural-fitting examples, "
         "encoding, objective and eight epochs. Inference counts maps/scans (ours) and maps/attention (TF): "
         "two FLOPs per multiply-add, excluding padding, scalar nonlinearities and expert preparation. "
         "Training includes backward, clipping, Adam and our evidence/calibration; its ledger follows."),
        ("p","<b>Lower core inference work in every screen.</b> Gestures use <b>5.18× less</b> with "
         "59.1% versus 15.9% accuracy. That 44-query screen has an underfitting TF control; a general "
         "vision claim requires complete benchmarks and stronger references."),
        ("p","<b>Why training can cost more:</b> the race core evaluates all three candidate vector payloads "
         "during training, versus only the winner during inference. Its eight layers also exceed the reference's two. "
         "With short contexts, that work outweighs the saved attention cost; these rows do not show a training "
         "efficiency advantage. On the longer event-camera prefixes, the common model's estimated total uses "
         f"{100*total_work['dvs']['common_to_reference_ratio']:.1f}% of the reference training arithmetic, including calibration."),
        ("small","Seed 6; neural fit/development counts: text 2,048/256, market 512/256, temporal 1,024/256, "
         "MNIST 1,024/256, gestures 88/44. The common text model also has a separately fitted 32,768-character "
         "evidence bank; market evidence is fitted on a prior day. The references have no such bank. "
         "MNIST uses pooled training-set images; gestures use first-second prefixes and disjoint users. "
         "Batch 16, or four for gestures. No official real-data test. Market fixed evidence: 3.670 nats/event.")])

    stage_rows=[]
    for row in tasks["training_work"]["rows"]:
        for key,label in (("common","Ours"),("reference","TF")):
            part=row[key+"_stages"]
            stage_rows.append([row["task"].capitalize()+": "+label]+
                [f"{part[stage]/1e9:.3f}"
                 for stage in ("forward_and_loss","backward","gradient_clipping","optimizer")]+
                [f"{row['common_setup']['total_arithmetic_flops']/1e9:.3f}" if key=="common" else "0",
                 f"{row[key+'_total_training_flops']/1e9:.2f}"])
    pages.append([
        ("h1","Appendix B (continued). Total training cost"),
        ("p","The ledger estimates the entire completed fitting budget for each reported model. It includes "
         "prediction and loss, backpropagation, gradient clipping and Adam across all eight epochs. "
         "The common model also pays for its evidence bank and initial readout calibration. "
         "These models were trained from initialization; there is no inherited neural fitting to omit."),
        ("figure",("e172_complete_training_work",150)),
        ("table",(["Model/task","Forward + loss","Backward","Clip","Adam","Evidence + calibration","Total"],stage_rows,[43,25,24,16,19,23,24])),
        ("small","All table values are estimated GFLOPs for the whole fitting run, not per query. "
         "A multiply-add counts as two operations. Forward/loss and backward use the saved E172 four-query "
         "operator trace scaled by recorded map/scan work. Ours uses its logged candidate-map/scan "
         "counts. Reference padding is reconstructed from all fitting prefix lengths, the original shuffle seed "
         "and batch sizes, including fused attention products. Clipping and Adam are charged once per original step. "
         "Other arithmetic and the small calibration eigensolver are estimates."),
        ("small","Event-target arithmetic excludes padding and simulator dispatch/allocation; required candidate "
         "maps, losing-value teaching, scans and learning remain charged. References use the same FLOP convention. "
         "Evaluation, search, encoding, special functions, integer/index work, comparisons and memory traffic "
         "are outside these totals. These single-seed screens have different depths/quality and do not measure "
         "event hardware, matched-quality cost or energy. Ledger: "
         '<a href="experiments/estimate_training_work.py">estimate_training_work.py</a>.')])

    pages.append(language_reference_page)
    language_rows=[
        ["Ours: separate statistical count/copy baseline",f"{ev['native10']:.3f}","10M count fitting + three 1M mixing-rate trials",compact_work(lm_costs['native_without_word']['total_training_flops'])+" + integer count construction"],
        ["Ours: count/copy plus causal word context",f"{ev['native_word10']:.3f}","10M count fitting + three 1M mixing-rate trials",compact_work(lm_costs['native_with_causal_word']['total_training_flops'])+" + integer count construction"],
    ]
    pages.append([
        ("h1","Appendix B (continued). Separate statistical language baseline"),
        ("p","This count/copy predictor does not use the learned Sleeping Machines event backbone. "
         "It is a separate statistical system: order-0 through order-6 counts, backoff probabilities and "
         "a bounded causal copy cache, combined by learned mixing weights. Its quality/work results "
         "must not be attributed to the event architecture."),
        ("p","The statistical predictor and saved neural references score the same 999,999 character targets, "
         "starting from a cold context. Parameters are frozen during testing. Earlier observed test characters "
         "can supply causal context, including the mixture's bounded 256-character copy cache. Lower bits per "
         "character means better prediction."),
        ("table",(["Predictor","Test bpc ↓","Fitting and selection budget","Estimated training FLOPs ↓"],language_rows,[51,21,59,43])),
        ("small","The neural totals charge all original optimizer steps that produced the inherited E174 "
         "checkpoints: forward, estimated 2×-forward backward, clipping and Adam. Alignment evaluation is excluded. "
         "The statistical floating estimate charges expert probability preparation and all three mixing-rate trials, "
         "including local gradients and weight updates. Exp/log/root evaluations count as one operation in these "
         "language estimates. Count construction additionally uses about 80M integer count presentations, "
         "plus sorting, lookup and hashing; that work is not quantified as FLOPs. "
         "The floating totals alone cannot establish total compute, runtime or energy savings."),
        ("p",f"The count/copy mixture improves on LSTM by {ev['lstm10']-ev['native10']:.3f} bpc "
         f"and Transformer by {ev['tf10']-ev['native10']:.3f} bpc. These results establish useful specialized prediction; "
         "generic learned representations are assessed in the separate language screen."),
        ("p","The statistical baseline's count arrays occupy 66.55 MB. "
         "Vocabulary, capacities, optimization and fitting budgets differ from the neural references. "
         "The comparison does not measure total training energy or a matched-capacity advantage."),
        ("p","All three use the same historical 27-character alphabet. Modern shared subword tokenization "
         "is a separate comparison gate for the learned event architecture, described later in this appendix."),
        ("small","One exploratory seed. Text8 offsets: count fitting [0,10M), mixing-weight validation "
         "[90M,91M), test [95M,96M); test index zero is excluded for all three predictors. The mixture "
         "selects its update rate on validation. Saved neural weights are unchanged. "
         "Results: E173/E174; stream contract: E175. "+language_90m_reference_text(ev))])

    generic = tasks["generic_language_audit"]["rows"]
    pages.append([
        ("h1","Appendix B (continued). Learned language and depth"),
        ("p","A bounded screen trains the common event backbone to predict the next character. "
         "Both configurations use width 32, the same 8,192 training characters, "
         "four passes, 32-character contexts and 1,024 validation predictions. All eight layers' value, route "
         "and memory-time parameters update. Lower bits per character means better prediction."),
        ("figure",("e133_generic_language",174)),
        ("table",(["Ours: depth","Learned parameters","Validation bpc: lower is better","Total CPU wall time"],[
         [str(depth),f"{generic[str(depth)]['parameters']:,}",f"{generic[str(depth)]['final_dev_bpc']:.3f}",f"{generic[str(depth)]['wall_s']:.1f} s"]
         for depth in (1,8)],[24,40,60,50])),
        ("p","Eight layers improve validation loss from 4.752 to 3.395 bpc, versus 3.464 with one layer. "
         "Shuffling preceding characters while preserving the last character, count and timestamps increases "
         "the deeper model's loss to 3.805; replacing preceding context raises it to 3.777. These frozen input "
         "probes show context sensitivity, not a retrained baseline comparison."),
        ("p","The deeper model has more parameters and takes more CPU time. The quality/time panel includes "
         "fitting and evaluation, with backpropagation and optimizer updates executed during fitting. These "
         "bounded-query models replay preceding context. The following persistent implementation consumes "
         "each character once. Physical memory traffic and joules remain unmeasured."),
        ("small","One seed; different parameter counts. This establishes a generic learned-language foothold "
         "and a small depth gain, not competitive large-scale representation, a matched tuned dense-model "
         "advantage or a scaling law. The statistical 10M-character mixture remains a separate result. Official test "
         "data are untouched. E133 preserves commands, source/data hashes, layer diagnostics and work coverage.")])

    pages.append([
        ("h1","Appendix B (continued). Persistent learned language"),
        ("p","The event-state language model consumes each character once and retains local modal "
         "memories and its delayed-message queue. Chunk boundaries "
         "truncate learning credit without discarding the observed history. Only actual event arrivals evaluate "
         "layers; text time is measured in token intervals."),
        ("figure",("e176_stream_language_learning",174)),
        ("table",(["Ours: representation","Parameters","Fitting bpc ↓","Validation bpc ↓","Deliveries/pass"],[
            ["Characters",f"{tasks['stream_training']['parameters']:,}",
             f"{tasks['stream_training']['final']['fit']['bpc']:.3f}",
             f"{tasks['stream_training']['final']['dev']['bpc']:.3f}",
             f"{tasks['stream_training']['final']['training_event_deliveries']:,}"],
            ["Causal prefix tokens",f"{tasks['token_training']['parameters']:,}",
             f"{tasks['token_training']['final']['fit']['bpc']:.3f}",
             f"{tasks['token_training']['final']['dev']['bpc']:.3f}",
             f"{tasks['token_training']['final']['training_layer_deliveries']:,}"],
        ],[46,31,30,31,36])),
        ("p",f"Validation loss falls from {tasks['stream_training']['initial']['bpc']:.3f} to "
         f"{tasks['stream_training']['final']['dev']['bpc']:.3f} bpc. All eight layer teachers are nonzero in "
         "every fitting pass. Each pass consumes 8,223 characters including warmup and makes 65,784 block "
         "deliveries. The result establishes learning with persistent causal state and no prefix replay."),
        ("p","A train-only 131-token prefix dictionary reduces layer deliveries by 30.4% and observed CPU time "
         "by 26.2%, while validation bpc is 3.517. Exact partial-token marginalization scores identical raw "
         "targets. Its larger vocabulary fits better but generalizes less well in this small screen: compression "
         "alone does not explain or resolve the quality gap."),
        ("small",f"One seed; 28,403 parameters, width 32, sixteen temporal modes per block. Four passes, "
         "128 Adam steps/pass, credit truncated every 64 characters, 31 warm characters and 1,024 validation "
         "targets. Target offsets match the bounded E133 screen; topology, capacity, history and update counts "
         "differ, so this is not a matched intervention. Total CPU wall time "
         f"{tasks['stream_training']['wall_s']:.1f} s including fitting/evaluation; peak RSS "
         f"{tasks['stream_training']['max_rss_kb']/1024:.1f} MiB. These are event counts and observed resources, "
         "not total arithmetic, physical memory traffic or energy. No official test or large-corpus claim. E176.")])

    event_work=tasks["event_language_work"]
    pages.append([
        ("h1","Appendix B (continued). Learned event language: training and inference work"),
        ("p","These counts belong to the learned eight-layer persistent Sleeping Machines event model "
         "that reaches 3.351 validation bpc. It has no count/copy/word experts. Primary counts describe "
         "the logical event algorithm; simulator dispatch/allocation is excluded."),
        ("table",(["Ours: fitting stage","Estimated arithmetic FLOPs"],[
          [stage.replace('_',' ').capitalize(),compact_work(value)]
          for stage,value in event_work['training_stages'].items()
        ]+[["Total",compact_work(event_work['total_training_arithmetic_flops'])]],[95,79])),
        ("p",f"The entire completed budget includes 32,768 fitting targets, 512 Adam/clipping steps and "
         f"four stream warmups. Special functions add {event_work['training_special_function_evaluations']/1e6:.3f}M "
         "evaluations, reported separately from arithmetic FLOPs."),
        ("table",(["Ours: inference boundary","Per character"],[
          ["Prediction plus NLL-scoring arithmetic",f"{event_work['inference_arithmetic_flops_per_character']:,.0f} FLOPs"],
          ["Additional special functions",f"{event_work['inference_special_functions_per_character']:,.0f} evaluations"],
        ],[112,62])),
        ("p","A 64-character saved-checkpoint trace measures forward/loss, backward, clipping and warm Adam; "
         "the fitting ledger scales those stages by the original 512 steps and separately charges warmup. "
         "This is a representative arithmetic estimate, not a whole-run trace. The traced chunk has complete "
         "floating-operator formula coverage. Index/queue work, comparisons, memory traffic and evaluation "
         "passes are outside the fitting arithmetic boundary."),
        ("p","The 10M-character LSTM/Transformer checkpoints have different data, capacity and achieved "
         "quality. Dividing their full fitting budgets by this small run would not establish a fair training "
         "advantage. Full learned-event runs must complete before a larger aligned quality/work comparison."),
        ("small","Evidence: event_language_work/local_event_language_work_20260930T124646Z.json; "
         "quality: E176. Two arithmetic FLOPs per multiply-add. Additional unit-weight special functions "
         "would give a different logical-operation total; no event-device runtime or energy has been measured.")])

    online=tasks["online_language"]
    pages.append([
        ("h1","Appendix B (continued). Statistical online-learning pilot"),
        ("p","The official language comparison freezes parameters during evaluation. A new development-only "
         "pilot asks whether causal local learning improves prediction: score each character first, reveal it, "
         "then update only the expert mixing weights. Count experts stay frozen and both arms use identical "
         "causal copy-cache behavior."),
        ("table",(["Ours: statistical expert set","Frozen bpc ↓","Online bpc ↓","Extra update FLOPs"],[
          ["Without word" if r['arm']=='without_word' else "With causal word",
           f"{r['frozen_bpc']:.3f}",f"{r['online_bpc']:.3f}",compact_work(r['extra_update_arithmetic_flops'])]
          for r in online['rows']],[56,37,37,44])),
        ("p","This pilot reuses a 100,000-character count checkpoint and its learning rate selected on an "
         "earlier 2,048-character validation window. It scores 8,191 targets in a fresh 8,192-character "
         "development window [90,032,768,90,040,960), excluding the first target. Each adaptive arm performs "
         "8,192 updates, including first-position warmup. Official test data are untouched."),
        ("small","One checkpoint and one window; a specialized adaptive readout, not deep learned TTT or a "
         "frontier result. The extra arithmetic column counts only local gradient/weight updates, on top of "
         "shared prediction and inherited fitting costs. Total pilot CPU wall time is 0.587 s. "
         "Evidence: online_language/local_online_language_20260930T121741Z.json."),
        ("h2","Compute allocation is the next architectural hypothesis"),
        ("p","Independent budgets for active width/depth, dormant capacity, temporal memory, retrieval, "
         "credit and adaptation may let extra work buy more prediction quality. The new theory derives "
         "conditional marginal-value allocation and retrieval-error bounds, and identifies exposure, "
         "routing overhead and hardware utilization as possible limits. A small-scale win does not prove "
         "a better scaling exponent or a widening frontier advantage."),
        ("p","Next learned-language comparisons need shared modern subword tokenization, Unicode/byte "
         "coverage, suitable rotary/relative position, packed optimized Transformer kernels, sparse MoE "
         "and modern recurrent/attention-hybrid controls. Existing event memory already uses relative-time "
         "rotations. Token coordinates must remain distinct from learned scheduling delays."),
        ("p","Known sequences can be fitted with input-known affine scans and then generated sequentially "
         "with persistent state. Sequence-parallel fitting also permits Transformer adaptation between "
         "tokens/chunks. Compare frozen and online arms at identical observations and adaptation budgets; "
         "charge cache consistency and update work."),
        ("small",'<a href="experiments/theory/43_compute_allocation_and_frontier_scaling.md">Theory §§280–286</a> '
         'and the <a href="experiments/FRONTIER_COMPUTE_PROTOCOL.md">frontier compute protocol</a> '
         "specify these tests. Modern architecture and larger-scale comparison arms remain proposed, not completed.")])

    historical_language=[]
    for size,k in ((1_000_000,5),(10_000_000,6),(90_000_000,7)):
        path=f"e79/race_mixer_D{size}_K{k}_e77none.json"
        old=read(path)['copy_window_256']
        historical_language.append([f"{size/1e6:g}M",f"{old['race_frozen_test_bpc']:.3f}",
                                    f"{old['race_online_test_bpc']:.3f}"])
    historical_market=[]
    for path,label in (("e57/regime_m5.json","Ours: event hazard + rate/flow state"),
                       ("e57/regime_m5_pt.json","Ours: event hazard + per-type state"),
                       ("e57/regime_m5_fine_pt.json","Ours: event hazard + per-type state; finer gap bank")):
        old=max(json.loads((RES/path).read_text()),key=lambda r:r['val_day5'])
        historical_market.append([label,f"{old['test_day6']:.3f}",f"{old['test_day7']:.3f}"])
    old_tf=read("e52/thp_test_d64_L32_f0_e11.json")['epochs'][-1]
    historical_market.append(["Saved Transformer Hawkes reference",f"{old_tf['day6']:.3f}",f"{old_tf['day7']:.3f}"])
    pages.append([
        ("h1","Appendix C. Preserved historical results and revisions"),
        ("p","Earlier result files remain part of the research record. The following numbers explain "
         "older report headlines and why their interpretation changed. They are preserved here with the "
         "identified protocol errors; they are not current valid benchmark comparisons."),
        ("h2","Earlier statistical language results"),
        ("table",(["Ours: fitting characters","Frozen historical bpc","Online historical bpc"],
                  historical_language,[48,63,63])),
        ("p","These E79 mixtures use counts, a partial-word expert and a 256-character copy window. "
         "The partial-word key depended on whether the target character was a space: changing the unseen "
         "target changed the predicted distribution. The old 1.613/1.504 headlines therefore cannot establish "
         "a causal language advantage. The corrected E173 10M results are 1.727 without word context and "
         "1.719 with causal word context; a corrected 90M mixture comparison remains open."),
        ("h2","Earlier event world-model results"),
        ("table",(["Historical predictor","Day 6 log-likelihood ↑","Day 7 log-likelihood ↑"],
                  historical_market,[94,40,40])),
        ("p","The event hazard models use sparse conditional memories and local rate/flow state. Their "
         "frozen test parameters and causal state updates are useful mechanisms. However, the event-size "
         "threshold was fitted across all seven pilot days, including the evaluation days, for both the "
         "event and neural references. Day resets and warmup exclusions also differ. The recorded gap "
         "needs fitting-only preprocessing and aligned rescoring before it supports a held-day advantage."),
        ("small",'<a href="experiments/EXPERIMENTAL_REVIEW.md">Source and numerical review</a>; '
         '<a href="experiments/results/e79/">E79 language records</a>; '
         '<a href="experiments/results/e57/">E57 world-model records</a>. '
         "Preserved timing, composition, retrieval and modular results remain in the main report. "
         "New learned-model benchmarks add evidence; they do not erase these earlier runs.")])

    pages.append([
        ("h1","Appendix D. Evidence and metric definitions"),
        ("table",(["Metric","Interpretation"],[
         ["Bits per character","Held-out negative log probability in base two; lower is better next-character prediction."],
         ["Accuracy","Fraction of correct class decisions on the declared development or test protocol."],
         ["Event likelihood","Scores both the next event type and waiting time, including the observed silence."],
         ["Arithmetic FLOPs","Multiply-add counts as two operations. Tables state whether special functions are separate or assigned unit cost; these conventions must be aligned before forming ratios."],
         ["Logical operations","The structured-task ledger assigns 2 units per MAC and 1 per other scalar arithmetic/nonlinear operation or estimated sort comparison."],
         ["Resource boundary","Event-target arithmetic includes required active algorithm work. Memory traffic, queues, indexing and simulator overhead are reported separately where measured. Activity counts alone do not determine total FLOPs."],
         ["Energy","Measured total joules over an explicit boundary. Operation estimates and CPU timings support work comparisons, but are not joule measurements."],
        ],[45,129])),
        ("p","The evidence is preserved in versioned result summaries with configurations, split identities, "
         "learning curves and source hashes. E173/E174 support the language comparison; E61 supports retrieval; "
         "E34/E53/E54 support native composition; E41 supports the original periodic computation. E121/E124 "
         "establish consolidated arithmetic and its certificate; E123 supplies the new dense controls and E124 "
         "the operation ledger. E118/E119/E122/E125/E126 support deep speech, readout and causal-context comparisons; "
         "E127–E131 audit credit geometry, hard race boundaries and separate key/value learning; E132 checks "
         "a joint race-credit formalism, E133 supplies the language/depth screen, and E134–E135 test "
         "whole-value credit and content-selective temporal memory. E136 audits reversible augmented transport "
         "and its supervised memory boundary, including twelve-layer query/learning interventions. E137 tests "
         "compact memory queries and class-visible credit geometry. E138–E141 examine richer source messages "
         "and trainable signed temporal memory, with exact local teacher and initial-nesting contracts. "
         "E142 establishes signed-state and first-coalescing identities; E143 tests a larger nonlinear temporal "
         "residual learner, and E144 audits simultaneous state/query pooling. E171 reproduces the consolidated "
         "screens and selected speech answers, and checks causal input boundaries. E172 records complete "
         "training-step arithmetic; E175 checks the generic persistent language stream."),
        ("p","The project theory index contains formal assumptions and proofs. Research findings retain detailed "
         "analyses and the full experimental record. The model documentation describes reproducible configurations "
         "and operational procedures. This report presents the project, its evidence and its potential.")])
    return pages


def markdown(pages):
    def convert(text):
        text = re.sub(r"<b>(.*?)</b>", r"**\1**", text)
        text = re.sub(r"<i>(.*?)</i>", r"*\1*", text)
        text = re.sub(r'<a href="([^"]+)">(.*?)</a>', r"[\2](\1)", text)
        return html.unescape(text)
    out = []
    for page in pages:
        for kind, value in page:
            if kind in ("title","h1","h2"):
                out.append({"title":"# ","h1":"## ","h2":"### "}[kind]+convert(value))
            elif kind == "bullets":
                out.append("\n".join("- "+convert(t) for t in value))
            elif kind == "table":
                header, rows, widths = value
                out.append("\n".join(["| "+" | ".join(header)+" |", "| "+" | ".join("---" for _ in header)+" |"]+
                                    ["| "+" | ".join(convert(t) for t in row)+" |" for row in rows]))
            elif kind == "figure":
                name, width = value
                out.append(f"![{name.replace('_', ' ')}](report/figures/{name}.png)")
            else:
                out.append(convert(value))
    # Editorial blocks can end with a space; generated Markdown must stay clean
    # for git diff --check.
    rendered = "\n\n".join(out)
    return "\n".join(line.rstrip(" \t") for line in rendered.splitlines())+"\n"


def build(M):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Table, TableStyle
    tasks, ev = results(), evidence(M)
    figures(M, tasks, ev)
    pages = blocks(M, tasks, ev)
    archive = ROOT/"report/archive/20260929_before_shared_model.md"
    if not archive.exists():
        archive.parent.mkdir(exist_ok=True)
        archive.write_text((ROOT/"REPORT.md").read_text())
    (ROOT/"REPORT.md").write_text(markdown(pages))
    st = M["styles"]()
    st["body"].fontSize = 9.7; st["body"].leading = 14.1; st["body"].spaceAfter = 7
    st["bullet"].fontSize = 9.5; st["bullet"].leading = 13.4; st["bullet"].spaceAfter = 7
    st["small"].fontSize = 8.0; st["small"].leading = 11.2
    st["h1"].fontSize = 14; st["h1"].leading = 19
    st["cell"].fontSize = 8.7; st["cell"].leading = 12.1
    flow = []
    for index, page in enumerate(pages):
        if index:
            flow.append(PageBreak())
        for kind, value in page:
            if kind == "figure":
                name, width = value
                flow.append(M["png"](name, width))
            elif kind == "bullets":
                flow.extend(M["bullets"](value, st))
            elif kind == "table":
                header, rows, widths = value
                # Escape data text while preserving our explicit table line breaks.
                data = [[Paragraph(html.escape(t).replace("&lt;br/&gt;", "<br/>"), st["cell"])
                         for t in row] for row in [header]+rows]
                table = Table(data,colWidths=[w*mm for w in widths],repeatRows=1,hAlign="LEFT")
                table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#edf3fb")),
                    ("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor(M["BLUE"])),
                    ("BOTTOMPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),7),
                    ("LINEBELOW",(0,1),(-1,-1),.25,colors.HexColor("#dbe1e8"))]))
                flow.append(table)
            else:
                flow.append(Paragraph(value,st["body" if kind == "p" else kind]))
    pdf = ROOT/"report/sleeping_machines_status.pdf"
    temporary = pdf.with_suffix(".building.pdf")
    doc = SimpleDocTemplate(str(temporary),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=15*mm,
                            bottomMargin=16*mm,title="Sleeping Machines — accomplishments and the shared model",
                            author="Sleeping Machines project")
    doc.build(flow,onFirstPage=M["footer"],onLaterPages=M["footer"])
    # Publish only a complete PDF at the project's single canonical path.
    temporary.replace(pdf)
    print("wrote",pdf,"and REPORT.md")
