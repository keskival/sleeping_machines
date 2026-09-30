"""Estimate whole fitting budgets from completed runs; never train a model.

The E172 operator trace supplies forward/loss and backward arithmetic. Scale
each by the full run's contraction count, replaying the reference's original
padding and batching. Charge clipping and Adam per actual optimizer step, and
charge the common model's evidence fitting and initial readout calibration.
Non-contraction arithmetic is an extrapolation, not a new operator trace.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e120_shared_tasks import temporal, mnist
from e120_dvs_adapter import dvs
from lm_training_flops import estimate_training_flops

RESULTS = ROOT / "experiments/results"


def read(path):
    record = json.loads(path.read_text())
    if record.get("status") != "completed":
        raise ValueError(f"Completed result required: {path}")
    return record


def contractions(trace):
    return sum(op["arithmetic_flops"] for op in trace["operators"].values()
               if op["classification"].startswith("registered contraction"))


def transformer_contractions(lengths, batch_size, epochs, seed, classes, counted=False):
    """Original shuffled batches, including padding and the final short batch."""
    rng = np.random.default_rng(seed + 10)
    total, steps, tokens, pairs = 0, 0, 0, 0
    for _ in range(epochs):
        order = rng.permutation(len(lengths))
        for start in range(0, len(order), batch_size):
            part = order[start:start + batch_size]
            size, width = len(part), max(lengths[i] for i in part)
            tokens += size * width
            pairs += size * width * width
            # Two width-32 layers: QKV/output, FFN, QK and attention-value.
            total += 2 * (2 * (12 * size * width * 32**2 +
                              2 * size * width**2 * 32) + size * 32 * classes)
            if counted:
                total += 2 * size * width * 32
            steps += 1
    return dict(flops=total, optimizer_steps=steps,
                padded_tokens=tokens, padded_token_pairs=pairs)


def scaled_stages(sample, full_contractions, steps, sample_contractions=None):
    if not sample["formula_coverage_complete"]:
        raise ValueError("The source step has unsupported floating operators")
    denominator = (contractions(sample["stages"]["forward_and_loss"])
                   if sample_contractions is None else sample_contractions)
    factor = full_contractions / denominator
    return {
        name: round(stage["arithmetic_flops"] *
                    (steps if name in ("gradient_clipping", "optimizer") else factor))
        for name, stage in sample["stages"].items()
    }


def common_setup(record, sample, fitting_queries):
    """Fit-only evidence and calibration; all are charged to the common model."""
    config = record["config"]
    dim, depth, classes = config["dim"], config["depth"], config["classes"]
    features, options, evidence = 2 * dim + 1, 3, config.get("evidence_count", 0)
    initial = record["initial"]["fit"]
    events, scans = initial["prefix_packets_processed"], initial["scan_compositions"]
    inference_contractions = 2 * (
        depth * events * (options * features + dim * features) +
        options * (dim + 1) * scans +
        fitting_queries * ((dim + 1)**2 + (dim + 1) * classes +
                          evidence * (dim + 1 + classes)))
    calibration_forward = round(sample["stages"]["forward_and_loss"]["arithmetic_flops"] *
                                inference_contractions /
                                contractions(sample["stages"]["forward_and_loss"]))
    # Mean, variance and standardization, covariance, eigensolver and whitening.
    width = dim + 1
    whitening = 2 * fitting_queries * width**2 + 6 * fitting_queries * width + 12 * width**3
    evidence_updates, evidence_scores, logs = 0, 0, 0
    for memory in record.get("evidence_work", []):
        if record["args"]["task"] == "market":
            bins, outcomes = 6, 4
            evidence_updates += memory["updates"] * (1 + bins)
            evidence_scores += fitting_queries * (2 * bins * outcomes + bins)
            logs += fitting_queries * bins * outcomes
        else:
            evidence_updates += memory["updates"]
            evidence_scores += fitting_queries * (3 * classes + 1)
            logs += fitting_queries * classes
    return dict(calibration_forward_flops=calibration_forward,
                calibration_whitening_flops=whitening,
                evidence_fitting_flops=evidence_updates,
                fitting_evidence_precomputation_flops=evidence_scores,
                evidence_log_evaluations=logs,
                total_arithmetic_flops=calibration_forward + whitening +
                                       evidence_updates + evidence_scores)


def estimate_breadth():
    step_path = RESULTS / "e172/complete_work_v2_20260930.json"
    breadth_path = RESULTS / "e124/breadth_work_counted_20260929.json"
    step = {r["task"]: r for r in read(step_path)["rows"]}
    breadth = read(breadth_path)
    source_paths = [step_path, breadth_path]
    rows = []
    for row in breadth["rows"]:
        task = row["task"]
        common_path, reference_path = ROOT / row["common_result"], ROOT / row["reference_result"]
        common, reference = read(common_path), read(reference_path)
        source_paths.extend((common_path, reference_path))
        if common["fit_ids"] != reference["fit_ids"] or common["dev_ids"] != reference["dev_ids"]:
            raise ValueError(f"Different fitting/development examples: {task}")
        fit, dev, seed = len(common["fit_ids"]), len(common["dev_ids"]), common["args"]["seed"]
        if task in ("language", "market"):
            lengths = [common["protocol"]["context"]] * fit
        else:
            data = {"temporal": temporal, "mnist": mnist, "dvs": dvs}[task](fit, dev, seed)
            if [r.identity for r in data.fit] != common["fit_ids"]:
                raise ValueError(f"Reconstructed examples differ: {task}")
            lengths = [len(r.prefix.channels) for r in data.fit]
        if lengths[:4] != step[task]["query_events"]:
            raise ValueError(f"Source trace uses different prefixes: {task}")
        common_epochs = common["args"]["epochs"]
        if len(common["curve"]) != common_epochs:
            raise ValueError(f"Incomplete epoch history: {task}")
        common_steps = common_epochs * math.ceil(fit / common["args"]["bs"])
        reference_shape = transformer_contractions(lengths, reference["args"]["bs"],
            reference["args"]["epochs"], reference["args"]["seed"], common["config"]["classes"], task == "dvs")
        common_stages = scaled_stages(step[task]["common"],
            row["common_training_forward_map_scan_flops"], common_steps)
        reference_stages = scaled_stages(step[task]["transformer"],
            reference_shape["flops"], reference_shape["optimizer_steps"],
            transformer_contractions(lengths[:4], 4, 1, seed,
                                     common["config"]["classes"], task == "dvs")["flops"])
        setup = common_setup(common, step[task]["common"], fit)
        common_total = sum(common_stages.values()) + setup["total_arithmetic_flops"]
        reference_total = sum(reference_stages.values())
        rows.append(dict(task=task, common_result=row["common_result"],
            reference_result=row["reference_result"], fitting_queries=fit,
            common_epochs=common_epochs, reference_epochs=reference["args"]["epochs"],
            common_batch_size=common["args"]["bs"], reference_batch_size=reference["args"]["bs"],
            common_optimizer_steps=common_steps, reference_optimizer_steps=reference_shape["optimizer_steps"],
            common_stages=common_stages, reference_stages=reference_stages,
            common_setup=setup, common_total_training_flops=common_total,
            reference_total_training_flops=reference_total,
            common_to_reference_ratio=common_total / reference_total,
            fitting_prefix_events=sum(lengths), reference_padding=reference_shape))
    return dict(rows=rows, source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in source_paths})


def estimate_language():
    """Charge inherited neural fitting and all three native mixing-rate trials."""
    native_path = RESULTS / "e173/causal_language_10m_20260930.json"
    native = read(native_path)
    alphabet, orders = 27, native["args"]["K"] + 1
    validation = native["args"]["validation"]
    # KT distributions, Witten-Bell backoff and the word distribution are
    # constructed once and shared across rate trials and both reported arms.
    preparation = validation * (orders * (5 * alphabet + 2) + 2 * alphabet + 1)
    preparation_special = validation * (orders + 2) * alphabet
    rows = []
    for arm, record in native["arms"].items():
        experts, trials = len(record["experts"]), len(record["validation_grid"])
        forward = validation * trials * (2 * experts * alphabet + 2 * alphabet - 1)
        gradient_and_update = validation * trials * (2 * experts * alphabet + 2 * experts)
        special = preparation_special + validation * trials * (alphabet + 1)
        rows.append(dict(model="native_" + arm, total_training_flops=preparation + forward +
            gradient_and_update + special, forward_flops=preparation + forward,
            backward_and_update_flops=gradient_and_update, special_function_evaluations=special,
            rate_trials=trials, fitting_labels=validation,
            integer_count_presentations=native["args"]["D"] * (orders + 1),
            convention="Arithmetic plus one unit per exp/log; no Adam in this local gradient learner",
            scope="Includes all rate trials and fully charges shared expert preparation to each standalone arm; count construction uses integer/sort/hash work, which is additional and unquantified"))
    source_paths = [native_path, ROOT / "experiments/lm_training_flops.py",
                    ROOT / "experiments/e173_causal_language.py", ROOT / "experiments/e79_race_mixer.py"]
    for model, file in (("lstm", "lstm_D10000000_s512_p6_dr0.1_v.json"),
                        ("tf", "tf_D10000000_s256_L4_p4_dr0.1_v.json")):
        path = RESULTS / "e64" / file
        record = json.loads(path.read_text())
        aligned_path = RESULTS / "e174" / f"aligned_{model}_10m_20260930.json"
        aligned = read(aligned_path)
        if Path(aligned["checkpoint"]).with_suffix(".json") != path.relative_to(ROOT):
            raise ValueError(f"Inherited training source differs: {model}")
        estimate = estimate_training_flops(record["args"], record["params"], record["steps"], alphabet)
        rows.append(dict(model=model, inherited_fitting_result=str(path.relative_to(ROOT)), **estimate))
        source_paths.extend((path, aligned_path))
    return rows, {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag:
        raise ValueError("A plain unique tag is required")
    output = RESULTS / "training_work" / (args.tag + ".json")
    if output.exists():
        raise FileExistsError(output)
    result = estimate_breadth()
    result["language_rows"], language_sources = estimate_language()
    result["source_sha256"].update(language_sources)
    result.update(status="completed", method="trace_calibrated_whole_fitting_budget_v4_event_target",
        multiply_add_flops=2,
        event_target_accounting=dict(
            boundary="Declared logical event algorithm; simulator dispatch/allocation overhead is excluded",
            implementation="Unpadded event candidate maps and linear-work affine scans, with current full learning credit",
            includes="All executed winning/losing training values, required scan arithmetic, backward, clipping, Adam and fitting setup",
            caveat="Trace-calibrated event arithmetic estimate; not an event-hardware runtime/energy measurement or a claim of ideal winner-only training",
            separate_costs="Ordering/index/integer work, special functions and memory traffic are outside breadth arithmetic FLOPs"),
        scope="All completed fitting epochs, backward, clipping, Adam, evidence fitting and readout calibration. No new training.",
        assumptions=["Scale traced forward/loss and backward arithmetic by whole-run forward contractions",
                     "Non-contraction arithmetic extrapolates the four-query saved-parameter trace",
                     "Reference padding is reconstructed from original seed, prefix lengths and batch size",
                     "Reference contraction scaling includes fused attention matrix products in both numerator and denominator",
                     "Clipping and warm Adam costs are charged once per original optimizer step",
                     "Calibration forward uses inference contraction scaling; whitening includes a 10*d^3 eigensolver estimate"],
        excludes=["Development/test inference and hyperparameter search", "Data loading, encoding, sorting and indexing",
                  "Memory traffic and allocations", "Special functions, integer arithmetic and comparisons (not arithmetic FLOPs)"],
        evidence="Exploratory single-seed screens; total arithmetic estimates, not hardware instructions, elapsed time or joules")
    result["source_sha256"][str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result.update(wall_s=time.perf_counter() - started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(platform=platform.platform(), device="cpu", threads=1))
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(json.dumps({k: row[k] for k in ("task", "common_total_training_flops",
                                             "reference_total_training_flops", "common_to_reference_ratio")}), flush=True)


if __name__ == "__main__":
    main()
