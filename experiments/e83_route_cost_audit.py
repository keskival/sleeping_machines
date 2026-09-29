"""Measure class-loss and sparse-work deltas for matched E83 route-pair shadows.

This is a frozen-checkpoint diagnostic. It does not update model parameters.
For each held-out minibatch it samples one near-boundary receiver bundle per
layer when eligible, replays 10/01/11, and records the resulting event, message,
candidate-score, readout, and fixed-grid state-scan counts. The purpose is to
calibrate a later cost-constrained credit experiment from observed deltas,
not to claim an energy estimate.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import e51_shd_world as S  # noqa: E402
from e71_event_cde import events  # noqa: E402
from e83_deep_shd import (  # noqa: E402
    DeepSHD,
    batch_to_events,
    nearby_closed_route_pairs,
    sampled_prefix_times,
    sparse_event_objective,
    stratified_limit,
)

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e83")


def work_snapshot(info, depth, batch_size):
    """Return per-example primitive work counts for the current trace."""
    candidates = np.asarray(info["candidates"], dtype=np.float64)
    messages = np.asarray(info["msgs"], dtype=np.float64)
    spikes = np.asarray(info["spikes"], dtype=np.float64)
    return {
        "candidate_scores_by_layer": (candidates[:depth] / batch_size).tolist(),
        "accepted_messages_by_layer": (messages[:depth] / batch_size).tolist(),
        "hidden_spikes_by_layer": spikes.tolist(),
        "readout_edge_updates_by_layer": (messages[depth:2 * depth] / batch_size).tolist(),
        "hidden_state_vector_scans_per_example": int(info["state_vector_updates_per_utt"]),
    }


def delta_arrays(after, before, depth):
    keys = (
        "candidate_scores_by_layer",
        "accepted_messages_by_layer",
        "hidden_spikes_by_layer",
        "readout_edge_updates_by_layer",
    )
    return {
        key: [round(float(a - b), 5) for a, b in zip(after[key], before[key])]
        for key in keys
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_limit", type=int, default=0,
                        help="override checkpoint eval_limit; zero uses the saved value")
    parser.add_argument("--pairs_per_layer_batch", type=int, default=1,
                        help="uniformly sampled eligible pair shadows per layer per minibatch")
    parser.add_argument("--run_tag", default="route_cost_audit")
    args_cli = parser.parse_args()
    if not all(ch.isalnum() or ch in "-_" for ch in args_cli.run_tag):
        raise ValueError("run_tag may contain only letters, digits, '-' and '_'")
    checkpoint = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    a = SimpleNamespace(**checkpoint["args"])
    if not getattr(a, "event_readout", a.objective == "event_prefix"):
        raise ValueError("route cost audit requires the sparse event-prefix readout")
    if a.objective != "event_prefix":
        raise ValueError("checkpoint objective must be event_prefix")
    if args_cli.pairs_per_layer_batch < 1:
        raise ValueError("pairs_per_layer_batch must be positive")
    eval_limit = args_cli.eval_limit or int(a.eval_limit)
    if eval_limit < a.bs:
        raise ValueError("eval_limit must be at least one minibatch")

    torch.manual_seed(a.seed)
    eval_subset_rng = np.random.default_rng(a.seed + 300_007)
    pair_rng = np.random.default_rng(a.seed + 700_031)

    def load(split, part):
        return [(*events(t, u, a.merge), y)
                for t, u, y in S.utterances(split, a.bands, part) if len(t) > 1]

    evaluation = stratified_limit(load("train", "val_spk"), eval_limit,
                                  eval_subset_rng)
    widths_sd = [float(v) for v in a.w_sd.split(",")]
    net = DeepSHD(
        a.bands, a.d, a.n, a.M1, a.M, a.depth, a.window, a.fan2,
        a.readout_fan, a.dmax, widths_sd, a.seed,
        event_readout=True,
        readout_fusion=a.readout_fusion,
        input_count_payload=a.input_count_payload == "additive",
        early_event_skip=a.early_event_skip,
        route_topk=int(getattr(a, "route_topk", 0)),
    )
    net.load_state_dict(checkpoint["model_state_dict"])
    for layer in net.layers:
        layer.spike_reconstruction = getattr(a, "spike_reconstruction", "legacy")
    net.eval()

    pair_records = []
    pairs_by_layer = np.zeros(a.depth, dtype=np.int64)
    eligible_by_layer = np.zeros(a.depth, dtype=np.int64)
    work_deltas_by_layer = [[] for _ in range(a.depth)]
    loss_deltas_by_layer = [[] for _ in range(a.depth)]
    gamma_by_layer = [[] for _ in range(a.depth)]
    batch_records = []

    for start in range(0, len(evaluation), a.bs):
        items = evaluation[start:start + a.bs]
        eb, ei, et, input_counts, labels, tmax, seq_end = batch_to_events(
            items, a.bands, 0, np.random.default_rng(0), 0.0)
        batch_size = len(items)
        grid = tmax + (a.depth + 1) * int(np.ceil(a.dmax)) + 60
        prefix_times = sampled_prefix_times(
            batch_size, a.prefix_samples, a.prefix_horizon_ms,
            a.prefix_window_start, et.device, et.dtype)

        with torch.no_grad():
            _, base_info = net(
                eb, ei, et, batch_size, grid, return_taps=True,
                collect_routes=True, input_counts=input_counts)
            base_main, base_aux, _ = sparse_event_objective(
                net.event_heads, base_info["layer_events"], labels, seq_end,
                a.dmax, prefix_times, a.readout_fusion)
            base_loss = float((base_main + a.aux_weight * base_aux).item())
        base_work = work_snapshot(base_info, a.depth, batch_size)
        eligible_counts = []
        selected_count = 0

        for layer_id, route_info in enumerate(base_info["route_candidates"]):
            # Restrict the existing sparse adjacent-candidate sampler to one
            # layer. Uniform sampling within that layer makes each selected
            # pair's inclusion propensity exactly 1 / candidate_count.
            count, _, selected = nearby_closed_route_pairs(
                [route_info], a.cf_band, a.cf_pair_window_ms, pair_rng,
                args_cli.pairs_per_layer_batch, "global")
            eligible_by_layer[layer_id] += count
            eligible_counts.append(count)
            if not selected:
                continue

            for _, route_a, route_b, inclusion_probability in selected:
                route_ids = base_info["route_candidates"][layer_id]["event_index"]
                receivers = base_info["route_candidates"][layer_id]["receiver"]
                scores = base_info["route_candidates"][layer_id]["score"]
                event_a, event_b = int(route_ids[route_a]), int(route_ids[route_b])
                receiver = int(receivers[route_a])
                if receiver != int(receivers[route_b]):
                    raise RuntimeError("pair sampler returned different receivers")

                def replay(overrides):
                    with torch.no_grad():
                        _, shadow_info = net(
                            eb, ei, et, batch_size, grid, return_taps=True,
                            route_overrides=overrides, input_counts=input_counts)
                        main, aux, _ = sparse_event_objective(
                            net.event_heads, shadow_info["layer_events"], labels,
                            seq_end, a.dmax, prefix_times, a.readout_fusion)
                        loss_value = float((main + a.aux_weight * aux).item())
                    return loss_value, work_snapshot(shadow_info, a.depth, batch_size)

                L10, work10 = replay([(layer_id, event_a, receiver, True)])
                L01, work01 = replay([(layer_id, event_b, receiver, True)])
                L11, work11 = replay([
                    (layer_id, event_a, receiver, True),
                    (layer_id, event_b, receiver, True),
                ])
                delta_class = L11 - base_loss
                interaction = L11 - L10 - L01 + base_loss
                joint_work_delta = delta_arrays(work11, base_work, a.depth)
                singleton_a_work_delta = delta_arrays(work10, base_work, a.depth)
                singleton_b_work_delta = delta_arrays(work01, base_work, a.depth)
                work_deltas_by_layer[layer_id].append(joint_work_delta)
                loss_deltas_by_layer[layer_id].append(delta_class)
                gamma_by_layer[layer_id].append(interaction)
                pairs_by_layer[layer_id] += 1
                selected_count += 1
                pair_records.append({
                    "batch_start": start,
                    "layer": layer_id + 1,
                    "receiver": receiver,
                    "source_events": [event_a, event_b],
                    "source_times_ms": [
                        round(float(base_info["route_candidates"][layer_id]["source_time"][route_a]), 4),
                        round(float(base_info["route_candidates"][layer_id]["source_time"][route_b]), 4),
                    ],
                    "closed_scores": [round(float(scores[route_a]), 6),
                                      round(float(scores[route_b]), 6)],
                    "candidate_pair_count": count,
                    "inclusion_probability_within_layer": inclusion_probability,
                    "loss_00_10_01_11": [base_loss, L10, L01, L11],
                    "joint_class_loss_delta": delta_class,
                    "interaction_gamma": interaction,
                    "work_00": base_work,
                    "work_10_minus_00": singleton_a_work_delta,
                    "work_01_minus_00": singleton_b_work_delta,
                    "work_11_minus_00": joint_work_delta,
                })

        batch_records.append({
            "batch_start": start,
            "batch_size": batch_size,
            "base_class_loss": base_loss,
            "eligible_pairs_by_layer": eligible_counts,
            "selected_pairs": selected_count,
            "base_work": base_work,
        })

    summary = []
    for k in range(a.depth):
        losses = np.asarray(loss_deltas_by_layer[k], dtype=np.float64)
        gammas = np.asarray(gamma_by_layer[k], dtype=np.float64)
        work_rows = work_deltas_by_layer[k]
        fields = (
            "candidate_scores_by_layer",
            "accepted_messages_by_layer",
            "hidden_spikes_by_layer",
            "readout_edge_updates_by_layer",
        )
        mean_work = {}
        for field in fields:
            if work_rows:
                matrix = np.asarray([row[field] for row in work_rows], dtype=np.float64)
                mean_work[field] = matrix.mean(0).round(5).tolist()
            else:
                mean_work[field] = [None] * a.depth
        summary.append({
            "layer": k + 1,
            "eligible_pairs": int(eligible_by_layer[k]),
            "sampled_pairs": int(pairs_by_layer[k]),
            "mean_joint_class_loss_delta": round(float(losses.mean()), 6) if len(losses) else None,
            "fraction_joint_opening_improves_class_loss": round(float((losses < 0).mean()), 6) if len(losses) else None,
            "mean_interaction_gamma": round(float(gammas.mean()), 6) if len(gammas) else None,
            "fraction_gamma_negative": round(float((gammas < 0).mean()), 6) if len(gammas) else None,
            "mean_joint_work_delta_per_example": mean_work,
        })

    result = {
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "run_tag": args_cli.run_tag,
        "split": "speaker-held-out validation",
        "examples": len(evaluation),
        "batch_size": a.bs,
        "depth": a.depth,
        "seed": a.seed,
        "pair_policy": "one uniform within-layer pair per eligible layer and minibatch",
        "pairs_per_layer_batch": args_cli.pairs_per_layer_batch,
        "objective": "frozen checkpoint; no parameter updates",
        "work_count_note": "Primitive counts only; current state-vector scans are grid-based and operation counts are not energy measurements.",
        "eligible_pairs_by_layer": eligible_by_layer.tolist(),
        "sampled_pairs_by_layer": pairs_by_layer.tolist(),
        "summary_by_layer": summary,
        "batches": batch_records,
        "pairs": pair_records,
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"route_cost_audit_{args_cli.run_tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({
        "output": path,
        "examples": len(evaluation),
        "eligible_pairs_by_layer": result["eligible_pairs_by_layer"],
        "sampled_pairs_by_layer": result["sampled_pairs_by_layer"],
        "summary_by_layer": summary,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
