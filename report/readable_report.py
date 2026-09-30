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


def evidence(M):
    e79 = {n: read(f"e79/race_mixer_D{n}_K{k}_e77none.json")["copy_window_256"]["race_frozen_test_bpc"]
           for n, k in ((1_000_000, 5), (10_000_000, 6), (90_000_000, 7))}
    def aws_e64_bpc(model, data_size):
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
                    value = json.loads(result_path.read_text()).get("test_bpc")
                    if (isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) and value > 0):
                        return float(value)
            except (OSError, ValueError, TypeError, IndexError):
                continue
        return None
    return {"e79": e79,
            "lstm1": read("e64/lstm_D1000000_s256_p20_dr0.2_v.json")["test_bpc"],
            "lstm10": read("e64/lstm_D10000000_s512_p6_dr0.1_v.json")["test_bpc"],
            "lstm90": aws_e64_bpc("lstm", 90_000_000),
            "tf1": read("e64/tf_D1000000_s256_p20_dr0.2_v.json")["test_bpc"],
            "tf10": M["load_tf_10m_final"]()["test_bpc"],
            "tf90": aws_e64_bpc("tf", 90_000_000),
            "recall_tf": max(p["n32"] for f in (RES/"e61").glob("tf_K32_n8*.json")
                             for row in json.loads(f.read_text())["rows"] for p in row["curve"])}


def language_90m_reference_text(ev):
    """Publish each completed control without treating validation logs as test evidence."""
    scores = [f"{ev['e79'][90_000_000]:.3f} for the native mixture"]
    pending = []
    for key, label in (("lstm90", "LSTM"), ("tf90", "four-layer Transformer")):
        if ev[key] is None:
            pending.append(label)
        else:
            scores.append(f"{ev[key]:.3f} for the {label}")
    text = "At 90M, completed held-out test scores are " + ", ".join(scores) + ". "
    if pending:
        text += "The 90M " + " and ".join(pending) + " reference results are pending. "
    text += ("These are exploratory single-seed comparisons on the same text8 split; "
             "model sizes, training passes and computation are not matched. ")
    return text


def accomplishments_figure(M, ev):
    import matplotlib.pyplot as plt
    import numpy as np
    blue, orange, gray = M["BLUE"], M["ORANGE"], M["GRAY"]
    f, ax = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [1.2, 1]})
    labels = ["Sleeping\nMachines", "LSTM", "Transformer"]
    ten_m = [ev["e79"][10_000_000], ev["lstm10"], ev["tf10"]]
    ninety_m = [ev["e79"][90_000_000], ev["lstm90"], ev["tf90"]]
    x = np.arange(3)
    width = .34
    colors = [blue, gray, orange]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax[0].bar(x[i] - width/2, ten_m[i], width, color=color,
                  label="10M training" if i == 0 else None)
        ax[0].text(x[i] - width/2, ten_m[i] + .035, f"{ten_m[i]:.3f}",
                   ha="center", fontsize=8)
        if ninety_m[i] is not None:
            ax[0].bar(x[i] + width/2, ninety_m[i], width, color=color, hatch="//",
                      label="90M training" if i == 0 else None)
            ax[0].text(x[i] + width/2, ninety_m[i] + .035, f"{ninety_m[i]:.3f}",
                       ha="center", fontsize=8)
    ax[0].set_xticks(x, labels)
    ax[0].set_ylim(0, 2.5)
    ax[0].set_ylabel("Test bits per character ↓")
    ax[0].set_title("Better real-text prediction\ntext8 test score by training scale", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper left")
    ax[1].bar(range(2), [100, 100*ev["recall_tf"]], color=[blue, orange], width=.55)
    ax[1].set_xticks([0, 1], ["Local race\nretrieval", "Best of 7\nTransformers"])
    ax[1].set_ylim(0, 118)
    ax[1].set_ylabel("Accuracy at 4× context (%) ↑")
    for i, value in enumerate([100, 100*ev["recall_tf"]]):
        ax[1].text(i, value+2, f"{value:.1f}%", ha="center", fontsize=10)
    ax[1].set_title("Retrieval that generalizes\nSynthetic key/value task", fontsize=10)
    for a in ax:
        a.grid(axis="x", visible=False)
    f.tight_layout(w_pad=2.5)
    return f


def figures(M, tasks, ev):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    blue, orange, gray = M["BLUE"], M["ORANGE"], M["GRAY"]
    FIG.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(FIG/(name+".png"), dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    f, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    for depth, color in ((1, gray), (8, blue)):
        row = tasks["generic_language"][depth]
        curve = [row["initial"]["dev"]["bpc"]] + [e["dev"]["bpc"] for e in row["curve"]]
        axes[0].plot(range(len(curve)), curve, "o-", color=color, label=f"{depth} layer" + ("s" if depth > 1 else ""))
        work = tasks["generic_language_audit"]["rows"][str(depth)]["training_forward_map_scan_flops"]/1e9
        final = curve[-1]
        axes[1].scatter(work, final, color=color, s=65)
        axes[1].annotate(f"{depth} layer" + ("s" if depth > 1 else "") + f"\n{final:.3f} bpc", (work,final),
                         xytext=(0,12), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set(xlabel="Passes over 8,192 training characters", ylabel="Validation bpc (lower is better)",
                title="Learned prediction; no explicit experts", xticks=range(5))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Training-forward contractions (GFLOPs; estimate)",
                ylabel="Validation bpc (lower is better)", title="Quality / computation tradeoff",
                xlim=(0,135), ylim=(3.32,3.58))
    f.tight_layout()
    save(f, "e133_generic_language")
    f = accomplishments_figure(M, ev)
    save(f, "accomplishments")

    f, a = plt.subplots(figsize=(7.2, 2.65))
    a.set_xlim(0, 10); a.set_ylim(0, 4); a.axis("off")
    boxes = [(0.1, 1.55, 1.8, .9, "Input events\nvector + time"),
             (2.45, 1.55, 2.4, .9, "Local memory + context\n3 delayed choices"),
             (5.5, 1.55, 1.8, .9, "Winning vector\nand delay"),
             (7.95, 1.55, 1.9, .9, "Next layer\nthen a query"),
             (2.45, .05, 2.4, .85, "Losing alternatives\ntraining credit only")]
    for x, y, w, h, label in boxes:
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04",facecolor="#edf3fb",edgecolor=blue))
        a.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=9)
    for x1, x2 in ((1.95, 2.4), (4.9, 5.45), (7.35, 7.9)):
        a.add_patch(FancyArrowPatch((x1,2),(x2,2),arrowstyle="-|>",mutation_scale=14,color=blue))
    a.annotate("", (3.65,.94), (3.65,1.5), arrowprops={"arrowstyle":"->","linestyle":"--","color":orange})
    a.text(5.65,.47,"Only the winner is emitted.\nLabels teach the completed query.",fontsize=9,va="center")
    a.text(.1,3.3,"Shared layers; depth and weights configured per task",fontsize=11,fontweight="bold")
    a.text(.1,2.85,"Conditional evidence, hard pointers and phase memory inform query readouts.",fontsize=9)
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
    names={"shared_phase_only":"Common phase path (69 scalars)","shared_d2_phase":"Common two-layer + phase",
           "shared_d2_recall":"Common two-layer + pointer","lstm":"LSTM, width 32 / 2 layers",
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
    import figures_mech as historic
    historic.EVENT, historic.DENSE_T = blue, orange
    save(historic.fig_supremacy_map(), "supremacy_map")


def blocks(M, tasks, ev):
    """Project entry point: capabilities, evidence, principles and applications."""
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
    pages.append([
        ("title","Sleeping Machines"),
        ("sub","Computing with time: learned delays, vector messages and local memory"),
        ("p","<b>Time is part of the computation.</b> Messages carry both a vector and an arrival time. Learned "
         "delays change arrival order, shape temporal memory and decide which competing message wins. Nodes "
         "can wait, accumulate evidence and transform a message's content and timing. A clock race can select "
         "a class; learned phase transformations can compose an arithmetic rule. Unrealized alternatives "
         "teach better choices. Dormant capacity and sparse activation reduce cost, while the organizing idea "
         "is to make timing itself a trainable computational medium."),
        ("h1","The strongest demonstrated results"),
        ("bullets",[
         f"<b>Better real-language prediction.</b> With 10M training characters, the native predictive mixture reaches "
         f"<b>{ev['e79'][10_000_000]:.3f} test bits per character</b>, ahead of the completed LSTM ({ev['lstm10']:.3f}) "
         f"and four-layer Transformer ({ev['tf10']:.3f}) on the same text8 split. "
         + language_90m_reference_text(ev)
         + "Lower bits per character means better prediction.",
         "<b>Accurate retrieval with far fewer examples.</b> Local race retrieval learns perfect recall at four "
         "times the training context within 4,000 examples in all five runs. The consolidated model preserves "
         "100% on its standard and longer contexts.",
         "<b>Rule learning and deep composition.</b> The consolidated periodic path reaches <b>100% across all "
         "3,440 unseen modular triples</b>. Native depth-four order models reach 99.9–100%; shared-motif composition "
         "reaches about 99.65% from one pass at roughly 10,000× lower counted work than its Transformer reference."]),
        ("figure",("accomplishments",174)),
        ("small","Language scores are held-out test results from the named predictive mixture and gradient baselines, "
         "with different model sizes and schedules. That specialized mixture is not yet reproduced by a generic "
         "deep language model. Retrieval, composition and arithmetic are controlled synthetic tasks. "
         "The following pages distinguish the consolidated implementation from the original native components.")])

    pages.append([
        ("h1","Consolidated models: accuracy versus computation"),
        ("p","The common implementation now retains perfect modular generalization and longer-context retrieval. "
         "New Transformer and LSTM controls use the same synthetic examples. <b>Higher and further left is better:</b> "
         "more accurate answers from less counted work. The logarithmic axis makes large cost differences visible."),
        ("figure",("consolidated_work_frontiers",174)),
        ("table",(["Common configuration","Held-out capability","Estimated work per query"],[
         ["Periodic path; 69 learned scalars",f"{phase['correct']:,}/{phase['n']:,} unseen triples",f"{phase_work:,.0f} logical operations"],
         ["Two-layer carrier + hard pointer","100% at four times context",f"{recall_work:,.0f} logical operations"],
        ],[62,57,55])),
        ("p","The periodic path executes directly through the shared model interface. A two-layer carrier variant "
         "is also shown: its phase state supplies the same answers while the carrier adds cost. Speech and other "
         "representation tasks use deeper carrier configurations. The architecture chooses the required primitives "
         "and depth per task; each task has separately trained weights."),
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
        ("h1","Native components: quality, work and sample efficiency"),
        ("p","The native components establish why selective temporal computation is promising. Timing patterns, "
         "composition and deep order reach high accuracy with much less counted work. Retrieval and rule learning "
         "also show that a reusable computation can generalize beyond the observed examples."),
        ("figure",("supremacy_map",174)),
        ("small","Work panels count event operations or dense multiply-adds per example; training budgets are labeled. "
         "Data efficiency and arithmetic use different horizontal axes. Market quality is held-out log-likelihood, "
         "with an online GRU and Transformer reference. The largest work advantages here belong to the named native "
         "components; the preceding page measures consolidated configurations directly. Operation counts do not "
         "assign equal hardware energy to different operations.")])

    pages.append([
        ("h1","How the model computes and learns"),
        ("p","The architecture computes through local memory, vector payloads and timing. A learned delay "
         "changes which arrivals interact and which route wins; it participates in the function being learned. "
         "Activity sparsity controls how much computation occurs. A message can carry a learned "
         "representation, a pointer, evidence or a periodic state. Layers need not all perform the same operation. "
         "Their job is to preserve useful information and recruit the computation needed for the task."),
        ("figure",("shared_architecture",174)),
        ("bullets",[
         "<b>Local temporal memory</b> accumulates observed content and elapsed time without evaluating empty time ticks.",
         "<b>Computation through delays</b> uses waiting times, arrival order and clock races to transform information and select outcomes.",
         "<b>Hard races</b> choose the emitted vector and delay. Losing alternatives provide training credit without becoming identical forward messages.",
         "<b>Reusable memories</b> include conditional outcome statistics, relative pointers and learned phase transformations.",
         "<b>Trainable depth</b> uses bounded carrier updates to preserve representations and credit through a hierarchy.",
         "<b>Appropriate supervision</b> teaches a completed class decision or the next event's type and waiting time, including silence."]),
        ("p","The consolidated model is trained independently on each task. Input queries contain only observations "
         "already available at their cutoff. The label and next event remain targets. Its current query interface "
         "processes event packets; persistent scheduling across overlapping queries is an additional systems capability."),
        ("small","Small local vector maps and query readouts can remain dense. The intended efficiency comes from "
         "selective event/state computation and appropriate primitives; total memory access and communication must "
         "also be measured when assessing an implementation.")])

    pages.append([
        ("h1","A mathematical foundation for trainable computation"),
        ("table",(["Principle","What it enables"],[
         ["Stable transport through depth","A reversible packet/memory program preserves conditional value and credit norms at any depth. Twelve-layer speech prototypes learn with observable memory queries. Readout alignment, route support and transfer remain separate requirements."],
         ["Active communication support","Inputs need causal paths through which to interact. A context channel supplies joint information when sparse packets leave local groups disconnected."],
         ["Credit to unrealized alternatives","A losing payload or timing choice can show how a different route would change the outcome, while forward computation remains a hard race."],
         ["Periodic state as an isometry","Learned rotations/reflections have unit-magnitude occurrence derivatives. Their composition supports reusable arithmetic instead of a table of observed tuples."],
         ["Certified composition","Target-constrained min/max composition of phase errors certifies the fitted modular rule across all 4,913 possible tuples; exhaustive checking confirms it."],
         ["Natural supervised credit","Categorical and event likelihoods both credit predicted sufficient statistics minus observations. Silence enters through integrated exposure."],
         ["Useful optionality","Reserve consists of distinct, attainable future corrections under a causal work budget. Reachability and transferable learning matter alongside immediate loss."],
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
         "stability and a certificate for a fitted rule do not establish global optimizer convergence or a scaling law.")])

    pages.append([
        ("h1","Potential: what the demonstrated capabilities put within reach"),
        ("p","The opportunity is intelligence that learns reusable structure and spends computation in proportion "
         "to useful activity. Strong predictive mixtures, reliable retrieval, efficient temporal composition and "
         "certified periodic computation already provide working foundations. The shared implementation gives "
         "those mechanisms a common place to develop."),
        ("h2","Compact prediction and memory"),
        ("p","Local predictive memories can support compression, stream forecasting and adaptation to recurring "
         "patterns. Learned pointer rules can keep their meaning as context grows. These are useful ingredients "
         "for models that retain experience without recomputing an entire dense history at each query."),
        ("h2","A reusable event-to-decision module"),
        ("p","Sound, event-camera vision, touch and telemetry all arrive as evolving evidence. A capable recognizer "
         "could maintain context, identify meaningful patterns and answer as soon as confidence is sufficient. "
         "Deep speech learning and temporal composition establish parts of this capability. "
         + (f"The new temporal encoder improves private speech accuracy by "
            f"{tasks['shd_state_summary']['selected_gain_percentage_points']:.2f} points; its "
            "gain also holds on disjoint utterances, and its learned clocks contribute to classification. "
            if tasks['shd_state_summary'] is not None else "") +
         "Reliable early decisions and broader generalization are central development goals."),
        ("h2","Training efficiency creates capability"),
        ("p","Cheaper updates can buy more data, depth and experimentation from the same budget. Better sample "
         "efficiency makes each experience more useful. The exact event-memory scan already reduces audited "
         "forward/backward CPU time by 1.68×, preserving the checked predictions and gradients. Local timing "
         "and routing teachers provide additional routes to efficient learning."),
        ("h2","Industrial and scientific applications"),
        ("p","Machines and instruments could maintain local predictive models, recognize changes and adapt from "
         "new operating conditions. Laboratories could use event histories to select informative measurements "
         "and control experiments. These applications connect fast local responses with longer-term memory, "
         "close to the source of the observations.")])

    pages.append([
        ("h1","Potential: a new foundation for frontier models"),
        ("p","If the architecture combines frontier predictive quality, reliable deep learning and lower total "
         "training and inference cost, it would change the practical recipe for building frontier models. "
         "Useful capacity, active computation and learning cost could become more independently controllable."),
        ("h2","Useful quality at a lower energy cost"),
        ("p","A broad advantage can begin with comparable quality at substantially lower measured training or "
         "inference energy. A modest quality tradeoff with a large energy saving can also unlock new applications. "
         "Tenfold savings, or larger, would transform feasible deployments and research budgets; these are "
         "conditional scenarios, not current measured energy ratios."),
        ("h2","The economics of creating intelligence"),
        ("p","A fixed power and capital budget could produce a more capable model, more specialized models or "
         "more research. Teams constrained by compute could enter new scales and applications. Efficient learning "
         "would expand what is feasible, including the size and sophistication of frontier training runs."),
        ("h2","Capacity that can remain dormant"),
        ("p","Large stores of memories and skills could stay available while only relevant portions participate "
         "in a decision. Straightforward situations could resolve with little work; difficult ones could recruit "
         "more computation. Cheap search, communication and selective credit would make this a different "
         "scaling regime from repeatedly activating an entire dense model."),
        ("h2","Autonomy in mobile platforms"),
        ("p","Robots, vehicles, phones, wearables and remote instruments could sustain perception, memory and "
         "adaptation within a mobile power budget. Local intelligence could retain context continuously, react "
         "quickly and learn from experience while reducing dependence on continuous connectivity."),
        ("h2","Hardware and infrastructure"),
        ("p","If sparse communication and persistent local state determine cost, processors and data centers "
         "would increasingly optimize those operations: message delivery, delay queues, memory access, candidate "
         "lookup and selective updates. Investment would follow measured useful learning per watt and per unit "
         "of capital. The architecture could change which accelerators are most valuable and where capable "
         "models can operate."),
        ("small","These larger outcomes depend on demonstrating quality, scaling, retention and total resource "
         "cost together. The existing results provide concrete footholds for that research program.")])

    pages.append([
        ("h1","What establishes the larger advantage"),
        ("p","The ambition is a common model family whose strongest mechanisms remain useful as tasks, data and "
         "capacity grow. Arithmetic and retrieval retain their demonstrated strengths in the consolidated "
         "implementation. The strongest language mixture is specialized; the generic backbone still needs to "
         "demonstrate competitive learned representations. Character and subword-token budgets must be distinguished."),
        ("table",(["Objective","Decisive evidence"],[
         ["Generic language scaling","Train a learned event backbone without explicit n-gram/pointer experts. Scale through declared data budgets with matched Transformer, recurrent and state-space references; record loss, capacity, forward/backward work, memory traffic, time and joules."],
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
         "Accurate general recognition remains an open capability; published official-test results below "
         "are reference targets, evaluated on a different partition."),
        ("table",(["Private development configuration","Correct","Accuracy ↑"],speech_rows,[108,32,34])),
        ("figure",("e143_temporal_residual_learning" if state_residual is not None else "e139_source_information",174)),
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
         ["EventSSM: asynchronous learned state-space layers","95.9%"],
         ["S7: input-dependent temporal state","96.3%"],
         ["2026 multiscale residual encoder; publisher abstract","96.44%"],
        ],[131,43])),
        ("small","Our private sample uses training-file speakers 3/6; official test accuracy is unmeasured. "
         + ("The temporal residual uses three passes and a fresh optimizer; its larger capacity and budget "
            "are not a matched single-factor comparison. Its listed score selects the best private-development "
            "epoch; the curve shows all three. " if state_residual is not None else "") +
         "The original parent was fitted on 4,096 examples. Continuation timers exclude loading; the residual "
         "timer includes it. Both include evaluation and are not energy measurements. One seed. References: "
         '<a href="https://arxiv.org/html/2404.18508v2">EventSSM</a>, '
         '<a href="https://arxiv.org/html/2410.03464v1">S7</a>, '
         '<a href="https://www.sciencedirect.com/science/article/abs/pii/S0893608026003345">multiscale encoding</a>. '
         "The last reference's full training/selection protocol has not yet been inspected.")])

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
            ("table",(["Same trained readout; one subsystem reset","Correct","Accuracy ↑"],reset_rows,[108,32,34])),
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
            ("table",(["Completed three-pass query","Trainable parameters","Fit: higher is better","Held: higher is better"],query_rows,[63,35,38,38])),
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
            ("table",(["Private development configuration","Correct","Accuracy ↑","NLL ↓"],model_rows,[98,27,27,22])),
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
            ('table',(['One matched fitting pass','Parameters','Fit accuracy ↑','Private accuracy ↑','NLL ↓'],depth_rows,[68,29,26,29,22])),
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
    def compact_work(value):
        return f"{value/1e9:.2f}G" if value>=1e9 else f"{value/1e6:.2f}M"
    breadth={row["task"]:row for row in tasks["breadth_work"]["rows"]}
    for task,label in (("language","Text8"),("market","Market event prediction"),("temporal","Temporal composition"),
                       ("mnist","MNIST"),("dvs","Event-camera gestures")):
        row=breadth[task]
        def score(metric):
            if task=="language":return f"{metric['nll']/math.log(2):.3f} bpc"
            if "accuracy" in metric:return f"{100*metric['accuracy']:.1f}%"
            return f"{metric['nll']:.3f} nats/event"
        direction = "Prediction error: lower is better" if task in ("language", "market") else "Accuracy: higher is better"
        coverage.append([label+"<br/>"+direction,score(row["common_metric"]),score(row["reference_metric"]),
             compact_work(row["common_forward_map_scan_flops"])+" / "+compact_work(row["reference_forward_map_attention_flops"]),
             compact_work(row["common_training_forward_map_scan_flops"])+" / "+compact_work(row["reference_training_forward_map_attention_flops"])])
    pages.append([
        ("h1","Appendix B. Breadth of the common implementation"),
        ("p","The common event backbone has independently trained development screens across language, event "
         "prediction, temporal composition, images and event cameras, in addition to speech, retrieval and arithmetic. "
         "These bounded screens establish implementation breadth; the stronger native comparison results use their "
         "own complete protocols."),
        ("p","<b>How to read the comparison:</b> accuracy is the percentage of correct answers, so <b>higher is better</b>. "
         "Bits per character (bpc) and nats/event measure prediction error, so <b>lower is better</b>. "
         "FLOPs estimate arithmetic work: <b>lower means less computation</b>. Each work pair lists the common model "
         "first and the Transformer (TF) second."),
        ("table",(["Task and quality direction","Common quality","Transformer quality","Forward FLOPs per query: common / TF; lower is better","Training-forward FLOPs: common / TF; lower is better"],coverage,[40,24,26,43,41])),
        ("p","The common screens use eight layers and the Transformer references two, both at width 32 for "
         "eight epochs. They share neural-fitting examples, held-out examples, input encoding, objective and "
         "learning-rate schedule. These are one small reference setting per task. Two-layer follow-ups retain 100% recall at both "
         "context lengths and reach 96.1% temporal composition versus 97.3% with eight layers, using four times "
         "fewer hidden carrier emissions. The model's depth is chosen to suit the computation."),
        ("p","<b>Why training can cost more:</b> this older race core evaluates all three candidate vector payloads "
         "during training, versus only the winner during inference. Its eight layers also exceed the reference's two. "
         "With short contexts, that work outweighs the saved attention cost; these rows do not show a training "
         "efficiency advantage. On the longer event-camera prefixes, counted training-forward work is lower."),
        ("small","Seed 6; neural fit/development counts: text 2,048/256, market 512/256, temporal 1,024/256, "
         "MNIST 1,024/256, gestures 88/44. The common text model also has a separately fitted 32,768-character "
         "evidence bank; market evidence is fitted on a prior day. The references have no such bank. "
         "MNIST uses pooled training-set images; gestures use first-second prefixes and disjoint users. "
         "No official real-data test sets are used here. The market fixed-evidence reference is 3.670 nats/event."),
        ("small","FLOPs count 2 per map, attention or memory-scan MAC; M = million, G = billion. Forward "
         "counts are per unpadded prefix. Training-forward sums the declared fitting budget and includes the "
         "common model's losing-value evaluations. These are contraction estimates, excluding nonlinearities, "
         "sorting, normalization arithmetic, evidence fitting/lookup, backward and optimizer updates; they "
         "are not total training FLOPs or measured energy.")])

    generic = tasks["generic_language_audit"]["rows"]
    pages.append([
        ("h1","Appendix B (continued). Learned language without experts"),
        ("p","A new bounded screen trains the common event backbone without explicit n-gram, pointer, copy "
         "or periodic prediction experts. Both configurations use width 32, the same 8,192 training characters, "
         "four passes, 32-character contexts and 1,024 validation predictions. All eight layers' value, route "
         "and memory-time parameters update. Lower bits per character means better prediction."),
        ("figure",("e133_generic_language",174)),
        ("table",(["Depth","Learned parameters","Validation bpc: lower is better","Total CPU wall time"],[
         [str(depth),f"{generic[str(depth)]['parameters']:,}",f"{generic[str(depth)]['final_dev_bpc']:.3f}",f"{generic[str(depth)]['wall_s']:.1f} s"]
         for depth in (1,8)],[24,40,60,50])),
        ("p","Eight layers improve validation loss from 4.752 to 3.395 bpc, versus 3.464 with one layer. "
         "Shuffling preceding characters while preserving the last character, count and timestamps increases "
         "the deeper model's loss to 3.805; replacing preceding context raises it to 3.777. These frozen input "
         "probes show context sensitivity, not a retrained baseline comparison."),
        ("p","The deeper model costs more: recorded training-forward map/scan contractions are 111.38G "
         "versus 14.04G FLOPs. One instrumented 16-query batch estimates 108.13M versus 13.61M backward "
         "contraction FLOPs. These partial arithmetic measures exclude unsupported operations and optimizer "
         "work. Physical memory traffic and joules are unmeasured; contexts are still replayed."),
        ("small","One seed; different parameter counts. This establishes a generic learned-language foothold "
         "and a small depth gain, not competitive large-scale representation, a matched tuned dense-model "
         "advantage or a scaling law. The native 10M-character mixture remains a separate result. Official test "
         "data are untouched. E133 preserves commands, source/data hashes, layer diagnostics and work coverage.")])

    pages.append([
        ("h1","Appendix C. Evidence and metric definitions"),
        ("table",(["Metric","Interpretation"],[
         ["Bits per character","Held-out negative log probability in base two; lower is better next-character prediction."],
         ["Accuracy","Fraction of correct class decisions on the declared development or test protocol."],
         ["Event likelihood","Scores both the next event type and waiting time, including the observed silence."],
         ["Logical operations","The consolidated ledger assigns 2 units per MAC and 1 per other scalar arithmetic/nonlinear operation or estimated sort comparison."],
         ["Resource boundary","Logical memory reads are reported separately. Transfers, allocations, kernel launch and instrumentation are outside the arithmetic ledger. Division, exponential and remainder costs have unit weights."],
         ["Energy","Measured total joules over an explicit boundary. Operation estimates and CPU timings support work comparisons, but are not joule measurements."],
        ],[45,129])),
        ("p","The evidence is preserved in versioned result summaries with configurations, split identities, "
         "learning curves and source hashes. E79/E64 support the language comparison; E61 supports retrieval; "
         "E34/E53/E54 support native composition; E41 supports the original periodic computation. E121/E124 "
         "establish consolidated arithmetic and its certificate; E123 supplies the new dense controls and E124 "
         "the operation ledger. E118/E119/E122/E125/E126 support deep speech, readout and causal-context comparisons; "
         "E127–E131 audit credit geometry, hard race boundaries and separate key/value learning; E132 checks "
         "a joint race-credit formalism, E133 supplies the expert-free language screen, and E134–E135 test "
         "whole-value credit and content-selective temporal memory. E136 audits reversible augmented transport "
         "and its supervised memory boundary, including twelve-layer query/learning interventions. E137 tests "
         "compact memory queries and class-visible credit geometry. E138–E141 examine richer source messages "
         "and trainable signed temporal memory, with exact local teacher and initial-nesting contracts. "
         "E142 establishes signed-state and first-coalescing identities; E143 tests a larger nonlinear temporal "
         "residual learner, and E144 audits simultaneous state/query pooling."),
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
    return "\n\n".join(out)+"\n"


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
    # Publish only a complete PDF. Keep an explicitly named edition as well:
    # another host may publish its generated status PDF during a rebase.
    snapshot = ROOT/"report/sleeping_machines_shared_20260929.pdf"
    snapshot.write_bytes(temporary.read_bytes())
    temporary.replace(pdf)
    print("wrote",pdf,"and REPORT.md")
