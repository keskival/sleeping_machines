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
                    if isinstance(value, (int, float)):
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
    rows=[tasks["shd_scaled"],tasks["shd_pool_mean"],tasks["shd_pool_weighted"],tasks["shd_bridge"],tasks["shd_bridge_frozen"]]
    values=[100*sum(r["final"][s]["correct"] for s in ("dev_original","dev_additional"))/512 for r in rows]
    a.bar(range(5),values,color=["#c5ced8",gray,"#5f9be3",orange,blue],width=.55)
    a.set(xticks=range(5),xticklabels=["Starting\ncheckpoint","Mean\npool","Learned\npool","Context:\nfull updates","Context:\nnew columns"],
          ylim=(0,100),ylabel="Held-out accuracy (%)",title="Deep speech: one-epoch continuation comparisons")
    for i,v in enumerate(values):a.text(i,v+2,f"{v:.1f}%",ha="center",fontsize=11)
    f.tight_layout();save(f,"e122_speech")

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
         + (f"At 90M, the held-out scores are {ev['e79'][90_000_000]:.3f} for the native mixture, "
            f"{ev['lstm90']:.3f} for the LSTM, and {ev['tf90']:.3f} for the four-layer Transformer. "
            if ev["lstm90"] is not None and ev["tf90"] is not None else
            "The matched 90M LSTM and Transformer controls are queued. ")
         + "Lower bits per character means better prediction.",
         "<b>Accurate retrieval with far fewer examples.</b> Local race retrieval learns perfect recall at four "
         "times the training context within 4,000 examples in all five runs. The consolidated model preserves "
         "100% on its standard and longer contexts.",
         "<b>Rule learning and deep composition.</b> The consolidated periodic path reaches <b>100% across all "
         "3,440 unseen modular triples</b>. Native depth-four order models reach 99.9–100%; shared-motif composition "
         "reaches about 99.65% from one pass at roughly 10,000× lower counted work than its Transformer reference."]),
        ("figure",("accomplishments",174)),
        ("small","Language scores are held-out test results from the named predictive mixture and gradient baselines, "
         "with different model sizes and schedules. Retrieval, composition and arithmetic are controlled synthetic tasks. "
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
         ["Stable transport through depth","Bounded residual carriers preserve payload and credit under stated fixed-route conditions. Route changes and readout geometry are analyzed separately."],
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
         "Deep speech learning and temporal composition establish parts of this capability; reliable early "
         "decisions and broader generalization are central development goals."),
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
         "capacity grow. Arithmetic and retrieval now retain their demonstrated strengths in the consolidated "
         "implementation. Real language mixtures and native composition provide additional reference capabilities."),
        ("table",(["Objective","Decisive evidence"],[
         ["Preserve capabilities","Repeat established generalization and sample-efficiency results within the common model family, with task-appropriate depth and explicit resource accounting."],
         ["Strong real-event recognition","Accurate speech and event-camera decisions on complete held-out benchmarks; calibrated confidence and time-to-answer."],
         ["Learn routes and representations at scale","Reliable deep credit and useful counterfactual alternatives as width, depth, memory and data increase."],
         ["Efficient persistent operation","Maintain local state and pending messages across queries, preserving causal predictions while reducing repeated work."],
         ["Lower total energy at useful quality","Measure training and inference joules, memory traffic, latency and communication under declared hardware and quality targets."],
        ],[57,117])),
        ("p","Success on these dimensions would turn the current task-level advantages into a broader foundation "
         "for frontier models. The project's distinctive resources—timing, local memory, hard selection and credit "
         "to alternatives—remain the guide for architecture and learning.")])

    pages.append([
        ("h1","Appendix A. Deep event recognition"),
        ("p",f"The eight-layer speech checkpoint reaches <b>{100*pooled(tasks['shd_scaled']):.1f}%</b> across 512 held-out "
         "utterances. Both readout continuations start from that checkpoint. At the matched "
         f"readout comparison below, count pooling reaches <b>{100*pooled(pool_mean):.1f}%</b> and learned event "
         f"pooling reaches <b>{100*pooled(pool_weighted):.1f}%</b> across 512 held-out utterances. Each model has "
         "4,096 fitting utterances and begins from the same checkpoint. Hidden messages remain winning vectors "
         "and delays; the learned pool adds 32 scalar parameters."),
        ("figure",("e122_speech",174)),
        ("p",f"Two causal context channels allow distant packets to interact through accumulated state. "
         f"Their zero-initialized columns preserve the starting predictions exactly. A matched full-update "
         f"continuation reaches {100*pooled(tasks['shd_bridge']):.1f}%; training only those columns reaches "
         f"{100*pooled(tasks['shd_bridge_frozen']):.1f}%. The added state has linear event work and 6,534 learned parameters."),
        ("p","The learned pool scores each observed winning payload, accumulates a weighted numerator and mass, "
         "and reads their ratio at the query. It has linear work in the number of active packets and a local "
         "supervised score gradient. Zero initialization exactly recovers count pooling. This tests whether "
         "informative parts of an utterance should contribute more strongly to the decision."),
        ("p","The exact linear-work memory scan preserves audited predictions and gradients while reducing scan "
         "combines 5.49×. Median one-thread CPU inference improves 1.52× and forward/backward computation 1.68× "
         "at the audited checkpoint, excluding optimizer updates."),
        ("small","Speech scores are development evidence from training speakers 3/6; the official SHD test set "
         "is untouched. They are not directly comparable to published official-test scores. One seed, width 32, "
         "eight layers. The matched readout arms share checkpoint, examples, augmentation and update budget. "
         "Sparse event packets avoid a hidden time grid; calibrated early output remains a further capability.")])

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
        coverage.append([label,score(row["common_metric"]),score(row["reference_metric"]),
             compact_work(row["common_forward_map_scan_flops"])+" / "+compact_work(row["reference_forward_map_attention_flops"]),
             compact_work(row["common_training_forward_map_scan_flops"])+" / "+compact_work(row["reference_training_forward_map_attention_flops"])])
    pages.append([
        ("h1","Appendix B. Breadth of the common implementation"),
        ("p","The common event backbone has independently trained development screens across language, event "
         "prediction, temporal composition, images and event cameras, in addition to speech, retrieval and arithmetic. "
         "These bounded screens establish implementation breadth; the stronger native comparison results use their "
         "own complete protocols."),
        ("table",(["Task","Common model","Transformer reference","Forward FLOPs per query: common / TF","Training-forward FLOPs: common / TF"],coverage,[35,26,29,43,41])),
        ("p","The common screens use eight layers and the Transformer references two, both at width 32 for "
         "eight epochs. They share neural-fitting examples, held-out examples, input encoding, objective and "
         "learning-rate schedule. These are one small reference setting per task. Two-layer follow-ups retain 100% recall at both "
         "context lengths and reach 96.1% temporal composition versus 97.3% with eight layers, using four times "
         "fewer hidden carrier emissions. The model's depth is chosen to suit the computation."),
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
         "the operation ledger. E118/E119/E122/E125 support deep speech and its readout comparisons."),
        ("p","The project theory index contains formal assumptions and proofs. Research findings retain detailed "
         "analyses and the full experimental record. The model documentation describes reproducible configurations "
         "and operational procedures. This report presents the project, its evidence and its potential.")])
    return pages


def markdown(pages):
    def convert(text):
        text = re.sub(r"<b>(.*?)</b>", r"**\1**", text)
        text = re.sub(r"<i>(.*?)</i>", r"*\1*", text)
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
                data = [[Paragraph(html.escape(t), st["cell"]) for t in row] for row in [header]+rows]
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
