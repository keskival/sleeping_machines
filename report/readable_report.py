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
    return tasks


def evidence(M):
    e79 = {n: read(f"e79/race_mixer_D{n}_K{k}_e77none.json")["copy_window_256"]["race_frozen_test_bpc"]
           for n, k in ((1_000_000, 5), (10_000_000, 6), (90_000_000, 7))}
    return {"e79": e79,
            "lstm1": read("e64/lstm_D1000000_s256_p20_dr0.2_v.json")["test_bpc"],
            "lstm10": read("e64/lstm_D10000000_s512_p6_dr0.1_v.json")["test_bpc"],
            "tf1": read("e64/tf_D1000000_s256_p20_dr0.2_v.json")["test_bpc"],
            "tf10": M["load_tf_10m_final"]()["test_bpc"],
            "recall_tf": max(p["n32"] for f in (RES/"e61").glob("tf_K32_n8*.json")
                             for row in json.loads(f.read_text())["rows"] for p in row["curve"])}


def figures(M, tasks, ev):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    blue, orange, gray = M["BLUE"], M["ORANGE"], M["GRAY"]
    FIG.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(FIG/(name+".png"), dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    f, ax = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [1.2, 1]})
    labels = ["Sleeping\nMachines", "LSTM", "Transformer"]
    values = [ev["e79"][10_000_000], ev["lstm10"], ev["tf10"]]
    ax[0].bar(range(3), values, color=[blue, gray, orange], width=.6)
    for i, value in enumerate(values):
        ax[0].text(i, value+.035, f"{value:.3f}", ha="center", fontsize=10)
    ax[0].set_xticks(range(3), labels)
    ax[0].set_ylim(0, 2.2)
    ax[0].set_ylabel("Test bits per character ↓")
    ax[0].set_title("Better real-text prediction\n10M training characters", fontsize=10)
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
    save(f, "accomplishments")

    f, a = plt.subplots(figsize=(7.2, 2.65))
    a.set_xlim(0, 10); a.set_ylim(0, 4); a.axis("off")
    boxes = [(0.1, 1.55, 1.8, .9, "Input events\nvector + time"),
             (2.45, 1.55, 2.4, .9, "Local memory\n3 delayed choices"),
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
    a.text(.1,3.3,"Eight shared layers; separate weights for each task",fontsize=11,fontweight="bold")
    a.text(.1,2.85,"Conditional evidence and hard pointer memory can inform the query readout.",fontsize=9)
    save(f,"shared_architecture")

    f, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for a, task, title in zip(axes.flat[:2], ("temporal", "mnist"), ("Temporal composition", "MNIST")):
        x = tasks[task]
        for split, label, color in (("fit","Training",gray),("dev","Development",blue)):
            a.plot([r["epoch"] for r in x["curve"]], [100*r[split]["accuracy"] for r in x["curve"]], "o-", label=label,color=color)
        a.set_title(title); a.set_xlabel("Epoch"); a.set_ylabel("Accuracy (%)")
        a.set_ylim(0, 103); a.legend(fontsize=8)
    for a, task, title in zip(axes.flat[2:], ("language", "market"), ("Text: development loss improves", "Market: fitting does not transfer")):
        x = tasks[task]
        for split, label, color in (("fit","Training",gray),("dev","Development",blue)):
            a.plot([0]+[r["epoch"] for r in x["curve"]], [x["initial"][split]["nll"]]+[r[split]["nll"] for r in x["curve"]], "o-",label=label,color=color)
        a.set_title(title,fontsize=9); a.set_xlabel("Epoch"); a.set_ylabel("NLL (nats) ↓")
        a.legend(fontsize=8)
    f.tight_layout()
    save(f,"e120_shared_learning")

    audit = read("e120/readout_count_audit_v2_20260929.json")["splits"]["context4x"]
    f, a = plt.subplots(figsize=(6.6, 2.55))
    values = [100,100*audit["before"]["accuracy"],100*audit["count_projected"]["accuracy"]]
    a.barh([2,1,0],values,color=[gray,orange,blue],height=.6)
    a.set_yticks([2,1,0],["Pointer alone", "Initial combined readout", "One-coordinate correction"])
    a.set_xlim(0,114); a.set_xlabel("Accuracy at four times the context length (%)")
    a.set_title("An exact integration failure, then a frozen repair",fontsize=10)
    for y,v in zip([2,1,0],values):
        a.text(v+1,y,f"{v:.1f}%",va="center",fontsize=10)
    a.grid(axis="y",visible=False)
    f.tight_layout()
    save(f,"e120_count_repair")
    previous = M["fig_e119_work_and_learning"]()
    if previous is not None:
        plt.close(previous)  # Existing helper writes its standalone figure.


def blocks(M, tasks, ev):
    """Pages of (kind, payload) blocks shared by the PDF and Markdown."""
    pages = []
    pages.append([
        ("title", "Sleeping Machines"),
        ("sub", "Learning and computing through timed messages · 29 September 2026"),
        ("p", "A Sleeping Machine is a network of nodes with local memory. Messages carry a vector and an arrival time. "
         "Nodes use that content and timing to remember, compare, wait, and send a new message. Competing messages race; "
         "the winner performs the computation, while losing alternatives can teach the network during training."),
        ("h1", "The most significant accomplishments"),
        ("bullets", [
         f"<b>Better prediction on real language.</b> At 10M training characters, the native predictive mixture reaches "
         f"<b>{ev['e79'][10_000_000]:.3f} test bits per character</b>, ahead of the completed LSTM ({ev['lstm10']:.3f}) "
         f"and four-layer Transformer ({ev['tf10']:.3f}) on the same text8 split. Lower means better prediction.",
         "<b>Learning to retrieve from far fewer examples.</b> Local race retrieval reaches <b>100% accuracy on contexts "
         "four times longer</b> after at most 4,000 examples in all five runs. The strongest of seven Transformer "
         f"settings reaches {100*ev['recall_tf']:.1f}% there after up to one million examples.",
         "<b>Deep composition with little data and work.</b> Native depth-four order models reach <b>99.9–100%</b> "
         "after 10–15k examples in five runs. A separate shared-motif task averages <b>99.65% from one pass</b>, "
         "at roughly <b>10,000× lower counted work</b> than its Transformer reference."]),
        ("figure", ("accomplishments", 174)),
        ("small", "Language: completed held-out test scores, one seed, shared data splits but different model sizes, "
         "schedules and expert resources. Retrieval and composition: controlled synthetic tasks. Counted event "
         "operations and dense multiply-adds are different work units; these are not measured energy ratios."),
        ("p", "<b>Why this matters:</b> the evidence supports a route to intelligence that learns useful structure from "
         "less experience and spends computation on selected events. The next challenge is to combine those strengths "
         "in one scalable architecture. That synthesis now has a working eight-layer implementation and cross-task results.")])

    pages.append([
        ("h1", "1. What the results make possible"),
        ("p", "The opportunity has three parts: learning from fewer examples, avoiding unnecessary arithmetic, and "
         "retaining useful computation as context and depth grow. Each has a concrete experimental foothold."),
        ("table", (["Capability", "Completed evidence", "Practical significance"], [
         ["Prediction", f"Text8: E79 {ev['e79'][1_000_000]:.3f} / {ev['e79'][10_000_000]:.3f} test bpc at 1M / 10M training characters.",
          "Local predictive memories can compete with much larger learned dense models in these comparisons."],
         ["Retrieval", "E61: perfect 4×-context recall in 5/5 runs by 4k examples.",
          "A learned query-to-memory rule can preserve its meaning beyond training lengths."],
         ["Composition", "E54: 99.9–100% depth-four order recognition after 10–15k examples.",
          "Learned parts can be reused through a hierarchy rather than relearned for every combination."],
         ["Periodic computation", "E41: 99.4–99.9% unseen modular triples, training on 30% of tuples.",
          "An appropriate temporal representation can discover a reusable rule instead of storing examples."],
         ["Event world models", "E57: within 0.08–0.19 nats/event of its Transformer reference at about 1/3000 counted work.",
          "Predictive state can be maintained cheaply on real streams; the Transformer is more accurate."],
        ], [29, 70, 75])),
        ("h2", "The common idea behind these results"),
        ("p", "Information often arrives in bursts, with long stretches when little changes. A useful pattern may "
         "depend on which signal came first, what followed it, and how long the gap was. Local state and learned "
         "delays let the network express those relationships directly. A successful route can remain reusable "
         "across many examples instead of being reconstructed through a full dense calculation each time."),
        ("h2", "How to read the claims"),
        ("bullets", ["<b>Established comparisons:</b> the language, retrieval and synthetic-composition results above belong to their named prototypes and protocols.",
         "<b>New architecture evidence:</b> the shared model has small development runs across eight task families. Its results are described separately below.",
         "<b>Potential:</b> frontier-scale quality, a better scaling law, and lower total training energy are the larger objectives. They require further measurements."]),
        ("small", "The arithmetic prototype includes a provided rhythm resource. Its success does not imply that every "
         "event architecture will learn arithmetic. New AWS relative-time Transformer controls also reach near-perfect "
         "accuracy on timing/composition tasks; the large counted-work distinction remains the relevant result there.")])

    pages.append([
        ("h1", "2. One architecture, separately trained for each task"),
        ("p", "The tasks do not share a training objective or a set of weights. They share an implementation: local "
         "temporal memory, vector messages, delayed hard races, stable depth, and a query readout. Each task has its "
         "own encoder, outcome space, memory contents and fitted parameters."),
        ("figure", ("shared_architecture", 174)),
        ("h2", "What has been brought together"),
        ("bullets", [
         "<b>Deep event representations:</b> eight bounded residual layers carry information and learning credit. A layer evaluates three delayed choices and emits only the winning vector and delay.",
         "<b>Conditional evidence:</b> sparse context tables learn local outcome counts or event counts per exposure time. A learned feature-dependent readout combines their evidence.",
         "<b>Learned retrieval:</b> a hard pointer chooses a source and relative destination. Losing alternatives receive local mistake credit; the forward answer uses the winning pointer.",
         "<b>Task-appropriate supervision:</b> categorical labels supervise a completed query; event prediction uses the likelihood of the next mark and waiting time, including the observed silence."]),
        ("h2", "Causality is part of the interface"),
        ("p", "Language and market queries receive only an explicitly observed prefix. Future tokens and future "
         "gaps are targets, never inputs. This matters because learned delays can reorder arrivals: processing a whole "
         "sequence and attaching a next-token loss to an older delayed carrier can otherwise leak future information."),
        ("p", "This first implementation replays each prefix independently. It processes sparse event packets but "
         "does not yet retain a persistent online queue across queries. Small vector maps and query output scores "
         "remain dense. Native hold/veto detectors, periodic phase computation and learned event cancellation still "
         "need to join this shared core."),
        ("small", "Implementation: sleeping_machines/shared_event.py, event_memory.py, event_query.py, evidence_memory.py, "
         "objectives.py and readout_calibration.py. Theory §§176–180. E118/E119 remain compatible entry points.")])

    rows = []
    for task, label in (("language","Text8"),("market","Market events"),("recall","Associative recall"),
                        ("temporal","Temporal composition"),("mnist","MNIST"),("dvs","Event-camera gestures"),("modular","Modular arithmetic")):
        x = tasks[task]; end = x["final"]["dev"]
        value = f"{end['correct']}/{end['n']} = {100*end['accuracy']:.1f}%" if "accuracy" in end else f"{end['nll']:.3f} nats/event"
        if task == "language": value = f"{end['nll']/math.log(2):.3f} bpc; {100*end['accuracy']:.1f}% next-character"
        if task == "recall": value = "100% at standard and 4× context"
        rows.append([label,f"{len(x['fit_ids']):,} / {len(x['dev_ids']):,}",value])
    rows.insert(0,["SHD speech (preserved)","1,024 / 256","151/256 = 59.0% final; 63.7% best epoch"])
    pages.append([
        ("h1", "3. What the shared model does today"),
        ("p", "All new runs use the same eight-layer, width-32 core and independent task weights. The new tasks run "
         "for eight epochs. Speech uses the existing trained checkpoint after an exact implementation-extraction audit. "
         "These are development screens, not new full-scale comparisons against Transformers."),
        ("table", (["Task", "Neural fit / dev", "Final development result"],rows,[42,32,100])),
        ("h2", "What transfers"),
        ("p", "The common core learns temporal composition, image recognition and event-camera prefix recognition. "
         "Its speech checkpoint preserves all 256 checked predictions. Learned retrieval survives the synthesis "
         "after repairing an unsupported count feature. Text prediction improves from 3.022 to 2.915 development "
         "bits per character in the small run; the full E79 language mixture remains a stronger, separate result."),
        ("h2", "What the screen reveals"),
        ("p", "Market fitting improves while development likelihood worsens. Modular arithmetic does not generalize. "
         "Those results identify work for the common architecture: control the interaction between evidence and "
         "neural corrections, and incorporate the earlier periodic computation mechanism. Additional depth alone "
         "has not supplied every specialist's useful representation."),
        ("small", "One seed per screen. Text memory: 32,768 characters fitted separately, then 2,048 neural queries; dev "
         "is a 256-character slice of the text8 validation region. Market: bounded trade prefixes on disjoint days. "
         "Recall: an additional 4,000 examples fit the pointer memory. MNIST: 2×2 pooled training-set images. Gestures: "
         "only the first second, 88 fit / 44 held-out-user examples. SHD holds out training speakers 3 and 6. "
         "No official real-data test set is used by E120. Exact protocols and IDs are saved in each result JSON.")])

    pages.append([
        ("h1", "4. Learning curves expose the remaining gaps"),
        ("figure", ("e120_shared_learning",174)),
        ("p", "Temporal composition reaches 99.7% fitting accuracy and 97.3% development accuracy. MNIST reaches "
         "91.9% and 75.8%, respectively. Useful features and credit reach the eight-layer core in both tasks. "
         "The difference between fitting and development performance still matters."),
        ("p", "The market curve separates optimization from generalization: fitting NLL falls to 1.546 nats/event, "
         "but development NLL ends at 3.823. Its fixed evidence baseline scores 3.670. Removing the additive neural "
         "head after training improves development NLL to 3.549 while retaining the deep-feature evidence gate. "
         "On text, that same frozen deletion improves NLL from 2.020 to 1.942."),
        ("p", "These deletions are diagnostic interventions, not retrained controls. They motivate a matched test "
         "of how much freedom the evidence gate and additive head should have. The arithmetic screen instead "
         "points toward missing structure: 11.9% fit and 2.3% unseen-tuple accuracy after eight epochs, versus "
         "5.9% chance. The earlier rhythm model's success has not yet transferred."),
        ("small", "Evaluation losses use the same task objective throughout each curve. A reduction in training loss "
         "alone is not evidence of better prediction on new data. These small runs do not estimate scaling laws.")])

    pages.append([
        ("h1", "5. A concrete failure found and repaired"),
        ("p", "A perfect retrieval component initially became unreliable when placed inside the shared model. "
         "At four times the context length, accuracy collapsed from 100% for the pointer alone to 9.0% for the "
         "combined model. The failure was in how the new readout handled a feature it had never learned to use."),
        ("figure", ("e120_count_repair",174)),
        ("h2", "The cause, in plain terms"),
        ("p", "Every training example had the same length, so the count feature never varied. The standardizer "
         "divided by a tiny variance floor anyway. A longer input then became a 1,299-unit normalized feature, "
         "which produced random, untrained score contributions as large as 79. Those scores overwhelmed the "
         "correct answer from memory."),
        ("h2", "The decisive check"),
        ("p", "With every learned weight frozen, removing only that unsupported count contribution restores "
         "256/256 correct longer-context answers. Standard-context accuracy remains 100%. A fresh training run "
         "with the corrected calibration also reaches 100% on both lengths. The failed run is retained."),
        ("h2", "What we learned about synthesis"),
        ("p", "Combining successful components requires preserving the conditions under which each generalizes. "
         "A new branch can override a correct answer through an unidentified parameter direction. The new rule "
         "uses fitting data only: static count metadata that never varies cannot acquire an arbitrary extrapolation "
         "effect. Variable-count calibration remains unchanged."),
        ("small", "E120 frozen readout audit; theory §179. This repair preserves the demonstrated retrieval rule. "
         "It does not show that the deep core independently learned that rule.")])

    pages.append([
        ("h1", "6. The theory is becoming an engineering guide"),
        ("table", (["Question", "Current understanding", "Design consequence"],[
         ["Can information survive depth?", "Conditional bounds control the whole event sequence and its payload credit when routing is fixed.",
          "Use bounded residual carriers; check route changes and readout conditioning separately. §§166–170."],
         ["How do losing routes learn?", "Detached alternative payloads and delays can provide score credit without being emitted.",
          "Keep hard forward computation; account for the extra alternatives evaluated in training. §169."],
         ["How is silence supervised?", "Event likelihood includes the integrated hazard over the observed waiting time.",
          "Credit predicted event counts minus observed counts; no label at every time tick. §178."],
         ["What is optionality?", "Useful reserve consists of distinct, attainable future corrections under a causal work budget.",
          "Measure reachability and transfer; entropy or noisy same-sample improvement is insufficient. §§159–163, 171–172."],
         ["Why did recall break?", "Constant training metadata leaves a readout direction unidentifiable.",
          "Project unsupported static count dependence; retain a longer-context contract. §179."],
         ["Why does arithmetic need more?", "For three uniform modular operands, any two are statistically independent of the label.",
          "Test higher-order features and restore trainable periodic state. Depth alone is no guarantee. §180."],
        ],[34,72,68])),
        ("h2", "What is proved, and what experiments must establish"),
        ("p", "The formal work supplies expressivity results, local credit identities, conditional stability bounds, "
         "and counterexamples that expose implementation errors. Attention-like retrieval is expressible through "
         "the event formalism under its stated assumptions. This provides a design space; it does not prove that "
         "an optimizer will discover every useful computation efficiently."),
        ("p", "The next proofs and measurements should meet: identify a missing representation or credit direction, "
         "derive an intervention, then test its predicted effect. The exact scan audit and count-feature repair "
         "are examples of that process. The full derivations remain in experiments/THEORY.md and its thematic notes.")])

    pages.append([
        ("h1", "7. Potential: the capabilities now coming into reach"),
        ("p", "The ambition is a model whose useful capacity can grow without waking all of that capacity for every "
         "observation. The results already show several parts of that possibility: strong predictive mixtures, "
         "reliable selective retrieval, efficient temporal composition, and a trainable deep event core."),
        ("h2", "Components that can become useful first"),
        ("p", "Predictive memories can support compact stream models and compression. Learned retrieval can provide "
         "an addressable store whose rules work beyond training context lengths. Temporal composition can recognize "
         "sparse patterns in sensor and industrial streams. The shared implementation creates a practical way to "
         "improve these mechanisms together while fitting a separate model for each application."),
        ("h2", "A reusable event-to-decision module"),
        ("p", "A strong asynchronous recognizer would accumulate evidence from sound, event cameras, touch or "
         "telemetry and answer when confidence is sufficient. Such a module could become a common building block "
         "for mobile perception and monitoring. The new speech and gesture results establish learning in small "
         "deep models; the next capability is reliable, calibrated early decisions on complete benchmarks."),
        ("h2", "Training efficiency creates capability, too"),
        ("p", "Less work per update can buy more data, more depth and more useful experiments from the same budget. "
         "The exact scan improvement already reduces audited forward/backward CPU time by 1.68× for the speech "
         "core. The larger prospect combines cheap updates with better sample efficiency: each joule and each "
         "experience would produce more learning. Training, inference, search and memory maintenance all belong "
         "in that accounting."),
        ("h2", "The immediate research payoff"),
        ("p", "The project now has a common place to test whether a mechanism improves multiple kinds of "
         "computation. A correction to causal querying, memory, depth or credit can be exercised across domains "
         "without rebuilding each model from scratch. The value of unification is both scientific and practical: "
         "it exposes which advantages are general and which need an additional primitive.")])

    pages.append([
        ("h1", "8. If the full architecture succeeds"),
        ("p", "If the architecture combines frontier predictive quality, reliable deep learning and lower total "
         "training and inference cost, it would provide a new foundation for frontier models. The potential "
         "transformation is in how intelligence uses computation, memory and experience."),
        ("h2", "Frontier models and the economics of training"),
        ("p", "A fixed power budget could train a more capable model, explore more architectures or sustain more "
         "specialized models. Large training facilities could produce more capability per unit of electricity "
         "and capital. Smaller teams could enter areas previously constrained by compute budgets. If sparse "
         "communication and persistent state determine cost, hardware and infrastructure design would increasingly "
         "optimize those operations. The best use of GPUs, other accelerators and new event hardware would follow "
         "the measured workload."),
        ("h2", "Autonomy within a mobile power budget"),
        ("p", "Robots, vehicles, phones, wearables and remote instruments could maintain context continuously, "
         "react quickly to important changes and learn locally from experience. Large memories could remain "
         "available while only the relevant portions participate in a decision. The resulting capability is "
         "sustained perception, memory and adaptation where battery life, heat and connectivity constrain today's systems."),
        ("h2", "Industry and science"),
        ("p", "Factories and infrastructure could host many local predictive models that detect changes, diagnose "
         "faults and update from new operating conditions. Instruments could choose informative measurements, "
         "track rare events and control experiments with short feedback loops. Rich temporal models could work "
         "close to the source of data, making intelligence a routine part of equipment and processes."),
        ("h2", "A different scaling regime"),
        ("p", "The most consequential outcome would separate total useful capacity from the work needed for each "
         "decision. A growing repertoire of memories and skills could stay mostly dormant, with difficult "
         "situations recruiting more computation and straightforward ones resolving quickly. Establishing that "
         "regime requires jointly strong quality, affordable search, stable learning and measured resource savings. "
         "The existing accomplishments make this a concrete research program with several working foundations.")])

    pages.append([
        ("h1", "9. The next decisive work"),
        ("table", (["Priority", "Experiment or implementation", "What would count as progress"],[
         ["Preserve specialist strengths", "Integrate trainable phase state and native hold/veto composition; retain retrieval length contracts.",
          "Shared model inherits the earlier arithmetic and composition generalization under matched protocols."],
         ["Improve evidence coupling", "Train matched evidence-gate and additive-head controls for language and market; use separate selection and evaluation data.",
          "A gain on unseen data over the fixed evidence bank, not merely a lower fitting loss."],
         ["Advance real event recognition", "Scale SHD and gesture training with fixed speaker/user splits; calibrate confidence and time-to-answer.",
          "Repeatable accuracy/latency gains; one final evaluation on the official test split after selection."],
         ["Make prefixes incremental", "Maintain pending messages and local state across queries; enforce query closure without future leakage.",
          "Equivalent predictions with less repeated prefix work, including scheduler and memory costs."],
         ["Establish scaling and energy", "Use AWS for larger matched runs; profile both training and inference on declared hardware.",
          "Quality versus total joules, memory, latency, data and capacity, including candidate search and credit."],
        ],[35,76,63])),
        ("h2", "How experiments are kept safe and independent"),
        ("p", "Local runs use one guarded job at a time, a memory watchdog, a timeout and at least 8 GiB host "
         "memory reserve. The E120 runs peaked below 0.5 GiB process RSS. The AWS sibling retains ownership of "
         "its existing larger benchmark queues; the shared-model handoff uses separate tags and outputs."),
        ("h2", "What a strong final demonstration would contain"),
        ("p", "One architecture, trained independently on each task, should preserve the specialist wins and "
         "improve real-stream prediction and recognition. It should show how quality changes with depth, data "
         "and capacity, and how total resource cost changes with them. That would connect the current task-level "
         "advantages to the larger frontier-model claim.")])

    pages.append([
        ("h1", "Appendix A. Deep speech recognition and execution"),
        ("p", "The eight-layer speech model reaches 151/256 (59.0%) at the final epoch and 163/256 (63.7%) at "
         "the best development checkpoint after fitting 1,024 utterances for eight epochs. Fit accuracy is 80.8%. "
         "The official SHD test set was not used. An earlier controlled-width comparison reached 40.6% with "
         "eight layers versus 25.4% with one layer; the deeper model had more parameters and used more work."),
        ("figure", ("e119_work_and_learning",174)),
        ("p", "The linear-work memory scan preserves the audited predictions and gradients. At the earlier frozen "
         "checkpoint, scan combines fall 5.49×; median one-thread CPU inference improves 1.52× and forward/backward "
         "computation 1.68×, excluding optimizer updates. Sorting and small dense vector maps remain."),
        ("small", "E119: one seed, training speakers 3/6 reserved for development. Best and final checkpoints are "
         "different endpoints. The 20 ms coalescing point saves 35.5% of input packets and loses 3.52 accuracy "
         "points on the frozen final model. Timing and counted work are not measured joules. E120 confirms exact "
         "logit/gradient/winner equivalence on an audited batch and preserves 151/256 after extraction.")])

    pages.append([
        ("h1", "Appendix B. Evidence and reproduction"),
        ("table", (["Report claim", "Primary repository evidence"],[
         ["Language comparisons", "results/e79/race_mixer_D… JSON; results/e64 LSTM and Transformer test JSON. Completed AWS controls remain under results/aws_20260929."],
         ["Retrieval and context transfer", "results/e61/event_K32_n8.json and tf_K32_n8*.json; E120 recall and frozen count audit."],
         ["Native depth / composition / arithmetic", "results/e53, e54, e34 and e41; Transformer controls in e36 and AWS results."],
         ["Market world model", "results/e48, e52 and e57. These full-day protocols are separate from E120's bounded development prefixes."],
         ["Shared model screens", "results/e120/*.json: configuration, split identities, curves, component deletions, work and memory counts, hardware and source hashes."],
         ["Speech / exact scan", "results/e118 and e119; e120/shared_contracts_20260929.json."],
        ],[56,118])),
        ("h2", "Metric definitions"),
        ("bullets", ["<b>Bits per character (bpc):</b> average negative log probability in base two. Lower means better next-character prediction.",
         "<b>Negative log-likelihood (NLL):</b> prediction loss. For event streams it scores both the next event type and waiting time, including the absence of events before arrival.",
         "<b>Counted work:</b> a declared count of event operations, candidates, scans or multiply-adds. Different operations can have different hardware costs.",
         "<b>Energy:</b> total measured joules over a stated boundary. It is not interchangeable with a work count or CPU time; E120 has no joule measurements."]),
        ("h2", "Where the detail lives"),
        ("p", "experiments/SHARED_MODEL.md describes the common model, benchmark coverage, safe commands and AWS "
         "handoff. experiments/THEORY.md indexes the derivations; experiments/FINDINGS.md retains the chronological "
         "research record. report/archive/20260929_before_shared_model.md preserves the previous long narrative."),
        ("p", "Run report/make_pdf.py through the guarded queue to rebuild this PDF and REPORT.md from the same "
         "editorial source and completed result files. Run ./commit_done.sh from the host checkout to stage the "
         "completed source, JSON evidence, figures and report; checkpoints and logs stay excluded."),
        ("small", "Scope: one common backbone and optional evidence/retrieval modules, independently trained per task. "
         "The synthesis is partial: it does not yet incorporate every historical primitive or rerun every historical "
         "configuration. Official full-scale cross-domain comparisons and continual-learning evaluations remain outstanding.")])
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
