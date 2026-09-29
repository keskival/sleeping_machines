"""Bounded counterfactual route-tree audit for frozen E83 checkpoints.

Each state branches into retaining its current route policy or taking one
error/entropy-conditioned receiver swap. The swap is replayed through the
remaining network, so its descendants and route candidates are recomputed.
Leaves compare current label loss and one finite suffix-SGD step to a factual
branch with the same suffix. No update persists.
"""
import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import e51_shd_world as S  # noqa: E402
from e71_event_cde import events  # noqa: E402
from e83_deep_shd import (  # noqa: E402
    DeepSHD,
    batch_to_events,
    route_replacement_candidates,
    sampled_prefix_times,
    stratified_limit,
)

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e83")


def deepest_losses(net, info, labels, seq_end, a, prefix_times):
    rows = net.event_heads[-1].prefix_logits(
        info["layer_events"][-1], seq_end, a.depth, a.dmax, prefix_times)
    losses = torch.stack([
        F.cross_entropy(row, labels[b].expand(len(row)))
        for b, row in enumerate(rows)
    ])
    return losses, torch.stack([row[-1] for row in rows])


def suffix_parameters(net, last_swap_layer):
    # Freezing through the last forced layer keeps its source event IDs stable.
    hidden = [p for layer in net.layers[last_swap_layer + 1:]
              for p in layer.parameters() if p.requires_grad]
    return hidden + [p for p in net.event_heads[-1].parameters() if p.requires_grad]


def route_birth_candidates(route_candidates, margin_band, route_topk):
    """Closed gate alternatives that can fill an unused per-event top-k slot."""
    by_layer = [[] for _ in route_candidates]
    for layer_id, route_info in enumerate(route_candidates):
        scores = route_info["score"].detach().cpu().numpy()
        event_ids = route_info["event_index"].detach().cpu().numpy()
        receivers = route_info["receiver"].detach().cpu().numpy()
        active = route_info.get("active")
        if active is None:
            continue
        active = active.detach().cpu().numpy().astype(bool)
        groups = {}
        for route_idx, event_id in enumerate(event_ids):
            groups.setdefault(int(event_id), []).append(route_idx)
        for event_id, route_ids in groups.items():
            if sum(bool(active[idx]) for idx in route_ids) >= route_topk:
                continue
            for idx in route_ids:
                score = float(scores[idx])
                if not active[idx] and -margin_band <= score <= 0.0:
                    by_layer[layer_id].append((layer_id, event_id,
                                               int(receivers[idx]), score))
    return by_layer


def virtual_progress(net, loss, params, replay_loss, lr, clip_norm, retain_graph):
    grads = torch.autograd.grad(loss, params, allow_unused=True,
                                retain_graph=retain_graph)
    norm_sq = sum(float(g.detach().square().sum()) for g in grads if g is not None)
    grad_norm = math.sqrt(norm_sq)
    clip_scale = min(1.0, clip_norm / max(grad_norm, 1e-12))
    originals = [p.detach().clone() for p in params]
    try:
        with torch.no_grad():
            for param, grad in zip(params, grads):
                if grad is not None:
                    param.add_(grad, alpha=-lr * clip_scale)
            after = float(replay_loss())
    finally:
        with torch.no_grad():
            for param, original in zip(params, originals):
                param.copy_(original)
    before = float(loss.detach())
    return {
        "loss_before": before,
        "loss_after_virtual_step": after,
        "progress": before - after,
        "gradient_norm": grad_norm,
        "clip_scale": clip_scale,
    }


def choose_option(info, layer_id, target_batch, target_class, logits,
                  query_horizon, args, rng):
    options = []
    route_modes = {
        "replacement": (True, False),
        "birth": (False, True),
        "mixed": (True, True),
        "spike": (False, False),
        "all": (True, True),
    }
    use_replacement, use_birth = route_modes[args.route_mode]
    if use_replacement:
        rows = route_replacement_candidates(
            info["route_candidates"], args.route_band, args.route_topk)[layer_id]
        source_batch = info["route_inputs"][layer_id][0].detach().cpu().numpy()
        options.extend(("replacement", row, float(row[-1])) for row in rows
                       if int(source_batch[row[1]]) == target_batch)
    if use_birth:
        rows = route_birth_candidates(
            info["route_candidates"], args.route_band, args.route_topk)[layer_id]
        source_batch = info["route_inputs"][layer_id][0].detach().cpu().numpy()
        options.extend(("birth", row, -float(row[-1])) for row in rows
                       if int(source_batch[row[1]]) == target_batch)
    if args.route_mode in ("spike", "all"):
        diagnostic = info["spike_diagnostics"][layer_id]
        margins = diagnostic["spike_margin_trace"]
        fired = diagnostic["spike_fire_mask"]
        refractory = diagnostic["spike_refractory_trace"]
        time_ids = torch.arange(margins.shape[0], device=margins.device)
        valid_time = ((time_ids > 0)
                      & (time_ids.to(margins.dtype) <= query_horizon))
        valid = (
            valid_time[:, None]
            & (margins[:, target_batch] <= 0.0)
            & (margins[:, target_batch] >= -args.spike_band)
            & ~fired[:, target_batch]
            & (refractory[:, target_batch] < args.max_refractory_state)
        )
        coordinates = torch.nonzero(valid, as_tuple=False).detach().cpu().numpy()
        for time_id, unit_id in coordinates:
            margin = float(margins[time_id, target_batch, unit_id])
            row = (layer_id, int(time_id), target_batch, int(unit_id), margin)
            options.append(("spike_birth", row, abs(margin)))
    probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
    p_true = float(probs[target_class])
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12)))
                    / math.log(probs.shape[-1]))
    error = 1.0 - p_true
    uncertainty = 0.5 * (entropy + error)
    if not options:
        return None, {
            "candidate_count": 0, "p_true": p_true, "entropy": entropy,
            "error": error, "uncertainty": uncertainty, "temperature": None,
            "swap_probability": 0.0,
        }
    temperature = args.base_temperature + args.temperature_span * uncertainty
    gaps = np.asarray([item[2] for item in options], dtype=np.float64)
    weights = np.exp(-gaps / max(temperature, 1e-12))
    route_probs = weights / weights.sum()
    selected = int(rng.choice(len(options), p=route_probs))
    mechanism, row, margin = options[selected]
    rho = min(args.max_swap_probability,
              args.swap_floor + args.swap_uncertainty_span * uncertainty)
    action = {"mechanism": mechanism, "layer": layer_id + 1,
              "candidate_count": len(options),
              "sampled_route_probability": float(route_probs[selected])}
    if mechanism == "spike_birth":
        action.update({"spike_time": int(row[1]), "spike_unit": int(row[3]),
                       "threshold_margin": float(row[4])})
    else:
        action.update({
            "event_id": int(row[1]),
            "receiver": int(row[2]) if mechanism == "birth" else int(row[3]),
            "winner_receiver": int(row[2]) if mechanism == "replacement" else None,
            "score_margin": margin,
        })
    return action, {
        "candidate_count": len(options), "p_true": p_true,
        "entropy": entropy, "error": error, "uncertainty": uncertainty,
        "temperature": temperature, "swap_probability": rho,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--run_tag", default="option_tree")
    parser.add_argument("--route_band", type=float, default=0.5)
    parser.add_argument("--route_mode", choices=("replacement", "birth", "mixed", "spike", "all"),
                        default="replacement")
    parser.add_argument("--spike_band", type=float, default=0.5)
    parser.add_argument("--max_refractory_state", type=float, default=1e-3)
    parser.add_argument("--base_temperature", type=float, default=0.05)
    parser.add_argument("--temperature_span", type=float, default=0.45)
    parser.add_argument("--swap_floor", type=float, default=0.05)
    parser.add_argument("--swap_uncertainty_span", type=float, default=0.45)
    parser.add_argument("--max_swap_probability", type=float, default=0.5)
    parser.add_argument("--virtual_lr", type=float, default=0.001)
    parser.add_argument("--virtual_clip_norm", type=float, default=1.0)
    parser.add_argument("--backup_temperature", type=float, default=0.1)
    parser.add_argument("--learning_weights", default="0,1,10,100")
    args = parser.parse_args()
    if (args.examples < 1 or args.batch_size < 1 or args.route_band <= 0
            or args.spike_band <= 0 or args.max_refractory_state < 0
            or args.base_temperature <= 0 or args.temperature_span < 0
            or args.swap_floor < 0 or args.swap_uncertainty_span < 0
            or not 0 < args.max_swap_probability < 1
            or args.virtual_lr <= 0 or args.virtual_clip_norm <= 0
            or args.backup_temperature <= 0):
        raise ValueError("invalid positive audit parameter")
    if not all(ch.isalnum() or ch in "-_" for ch in args.run_tag):
        raise ValueError("run_tag may contain only letters, digits, '-' and '_'")
    learning_weights = [float(x) for x in args.learning_weights.split(",")]
    if not learning_weights or any(x < 0 for x in learning_weights):
        raise ValueError("learning weights must be nonnegative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    a = SimpleNamespace(**checkpoint["args"])
    if a.objective != "event_prefix" or not int(getattr(a, "route_topk", 0)):
        raise ValueError("route-tree audit requires an event-prefix top-k checkpoint")
    args.route_topk = int(a.route_topk)
    torch.manual_seed(int(getattr(a, "seed", args.seed)))
    data_rng = np.random.default_rng(int(getattr(a, "seed", args.seed)) + 300_007)
    data = [(*events(t, u, a.merge), y)
            for t, u, y in S.utterances("train", a.bands, "val_spk") if len(t) > 1]
    examples = stratified_limit(data, args.examples, data_rng)
    rng = np.random.default_rng(args.seed + 911_071)
    target_rng = np.random.default_rng(args.seed + 701_009)
    widths_sd = [float(x) for x in str(a.w_sd).split(",")]
    net = DeepSHD(
        a.bands, a.d, a.n, a.M1, a.M, a.depth, a.window, a.fan2,
        a.readout_fan, a.dmax, widths_sd, a.seed, event_readout=True,
        readout_fusion="deepest",
        input_count_payload=a.input_count_payload != "off",
        input_count_route_neutral=a.input_count_payload == "address_neutral",
        early_event_skip=bool(getattr(a, "early_event_skip", False)),
        route_topk=int(a.route_topk))
    net.load_state_dict(checkpoint["model_state_dict"])
    for layer in net.layers:
        layer.spike_reconstruction = getattr(a, "spike_reconstruction", "legacy")
    net.eval()

    tree_rows = []
    nodes_by_layer = np.zeros(a.depth, dtype=np.int64)
    candidates_by_layer = np.zeros(a.depth, dtype=np.int64)
    sampled_actions = []

    for start in range(0, len(examples), args.batch_size):
        items = examples[start:start + args.batch_size]
        eb, ei, et, input_counts, labels, tmax, seq_end = batch_to_events(
            items, a.bands, 0, np.random.default_rng(0), 0.0)
        batch_size = len(items)
        grid = tmax + (a.depth + 1) * math.ceil(a.dmax) + 60
        prefix_times = sampled_prefix_times(
            batch_size, int(getattr(a, "prefix_samples", 2)),
            float(getattr(a, "prefix_horizon_ms", 1000.0)),
            float(getattr(a, "prefix_window_start", 0.0)), et.device, et.dtype)
        with torch.no_grad():
            _, root_info = net(eb, ei, et, batch_size, grid, return_taps=True,
                               collect_routes=True, return_spike_diagnostics=True,
                               input_counts=input_counts)
            root_losses, root_logits = deepest_losses(
                net, root_info, labels, seq_end, a, prefix_times)
        root_probs = torch.softmax(root_logits, dim=-1).cpu().numpy()
        label_np = labels.cpu().numpy()
        factual_predictions = root_probs.argmax(-1)
        wrong = np.flatnonzero(factual_predictions != label_np)
        if not len(wrong):
            continue
        p_true_all = root_probs[np.arange(batch_size), label_np]
        entropy_all = -(root_probs * np.log(
            np.maximum(root_probs, 1e-12))).sum(-1) / math.log(root_probs.shape[-1])
        selection_weights = ((1.0 - p_true_all[wrong])
                             * (0.5 + 0.5 * entropy_all[wrong]))
        if selection_weights.sum() > 0:
            selection_weights /= selection_weights.sum()
        else:
            selection_weights[:] = 1.0 / len(wrong)
        target_batch = int(target_rng.choice(wrong, p=selection_weights))
        target_class = int(label_np[target_batch])
        root_loss = float(root_losses[target_batch])
        root_hidden_counts = [
            int(np.count_nonzero(layer_events[0].detach().cpu().numpy() == target_batch))
            for layer_events in root_info["layer_events"]
        ]
        leaves = []
        decision_summaries = []
        local_nodes = np.zeros(a.depth, dtype=np.int64)
        suffixes = {layer_id: suffix_parameters(net, layer_id)
                    for layer_id in range(a.depth)}
        factual_progress = {}

        # Matched factual update progress for each possible last swapped layer.
        with torch.enable_grad():
            _, root_grad_info = net(eb, ei, et, batch_size, grid, return_taps=True,
                                    input_counts=input_counts)
            root_grad_losses, _ = deepest_losses(
                net, root_grad_info, labels, seq_end, a, prefix_times)
            root_grad_loss = root_grad_losses[target_batch]
            for last_layer, params in suffixes.items():
                def factual_replay():
                    with torch.no_grad():
                        _, replay_info = net(
                            eb, ei, et, batch_size, grid, return_taps=True,
                            input_counts=input_counts)
                        replay_losses, _ = deepest_losses(
                            net, replay_info, labels, seq_end, a, prefix_times)
                    return float(replay_losses[target_batch])

                factual_progress[last_layer] = virtual_progress(
                    net, root_grad_loss, params, factual_replay,
                    args.virtual_lr, args.virtual_clip_norm, retain_graph=True)

        query_horizon = float(getattr(a, "prefix_horizon_ms", 1000.0))

        def visit(layer_id, route_overrides, spike_overrides, info, path):
            if layer_id == a.depth:
                with torch.enable_grad():
                    _, leaf_info = net(
                        eb, ei, et, batch_size, grid, return_taps=True,
                        collect_routes=True, return_spike_diagnostics=True,
                        route_overrides=route_overrides,
                        spike_overrides=spike_overrides,
                        input_counts=input_counts)
                    leaf_losses, leaf_logits = deepest_losses(
                        net, leaf_info, labels, seq_end, a, prefix_times)
                    leaf_loss = leaf_losses[target_batch]
                    prediction = int(leaf_logits[target_batch].detach().argmax())
                    last_layer = max((row["layer"] - 1 for row in path), default=-1)
                    leaf_hidden_counts = [
                        int(np.count_nonzero(layer_events[0].detach().cpu().numpy()
                                             == target_batch))
                        for layer_events in leaf_info["layer_events"]
                    ]
                    applied_interventions = []
                    for action in path:
                        if action["mechanism"] == "spike_birth":
                            fired = leaf_info["spike_diagnostics"][action["layer"] - 1][
                                "spike_fire_mask"]
                            applied = bool(fired[action["spike_time"], target_batch,
                                                 action["spike_unit"]])
                        else:
                            route_info = leaf_info["route_candidates"][action["layer"] - 1]
                            event_ids = route_info["event_index"].detach().cpu().numpy()
                            receivers = route_info["receiver"].detach().cpu().numpy()
                            active = route_info["active"].detach().cpu().numpy().astype(bool)
                            active_pairs = {
                                (int(event_ids[i]), int(receivers[i])): bool(active[i])
                                for i in range(len(event_ids))
                            }
                            if action["mechanism"] == "birth":
                                applied = active_pairs.get(
                                    (action["event_id"], action["receiver"]), False)
                            else:
                                applied = (
                                    not active_pairs.get((action["event_id"],
                                                          action["winner_receiver"]), True)
                                    and active_pairs.get((action["event_id"],
                                                         action["receiver"]), False))
                        applied_interventions.append({
                            "layer": action["layer"],
                            "mechanism": action["mechanism"],
                            "forced_state_matches_intervention": bool(applied),
                        })
                    if last_layer >= 0:
                        params = suffixes[last_layer]

                        def branch_replay():
                            with torch.no_grad():
                                _, replay_info = net(
                                    eb, ei, et, batch_size, grid, return_taps=True,
                                    route_overrides=route_overrides,
                                    spike_overrides=spike_overrides,
                                    input_counts=input_counts)
                                replay_losses, _ = deepest_losses(
                                    net, replay_info, labels, seq_end, a, prefix_times)
                            return float(replay_losses[target_batch])

                        branch_progress = virtual_progress(
                            net, leaf_loss, params, branch_replay,
                            args.virtual_lr, args.virtual_clip_norm,
                            retain_graph=False)
                        learning_advantage = (
                            branch_progress["progress"]
                            - factual_progress[last_layer]["progress"])
                    else:
                        branch_progress = None
                        learning_advantage = 0.0
                    record = {
                        "leaf_index": len(leaves),
                        "path": path,
                        "swap_count": len(path),
                        "last_swapped_layer": last_layer + 1 if last_layer >= 0 else None,
                        "prediction": prediction,
                        "correct": prediction == target_class,
                        "factual_hidden_events_by_layer": root_hidden_counts,
                        "counterfactual_hidden_events_by_layer": leaf_hidden_counts,
                        "hidden_event_delta_by_layer": [
                            int(after - before) for before, after
                            in zip(root_hidden_counts, leaf_hidden_counts)],
                        "forced_interventions_verified": applied_interventions,
                        "label_loss": float(leaf_loss.detach()),
                        "immediate_advantage_vs_factual": (
                            root_loss - float(leaf_loss.detach())),
                        "factual_suffix_progress": (
                            factual_progress[last_layer]["progress"]
                            if last_layer >= 0 else None),
                        "counterfactual_suffix_progress": (
                            branch_progress["progress"] if branch_progress else None),
                        "learning_option_advantage": learning_advantage,
                        "factual_suffix_gradient_norm": (
                            factual_progress[last_layer]["gradient_norm"]
                            if last_layer >= 0 else None),
                        "counterfactual_suffix_gradient_norm": (
                            branch_progress["gradient_norm"]
                            if branch_progress else None),
                        "option_only": bool(
                            root_loss - float(leaf_loss.detach()) <= 0.0
                            and learning_advantage > 0.0),
                    }
                    leaves.append(record)
                return {
                    str(weight): record["immediate_advantage_vs_factual"]
                    + weight * record["learning_option_advantage"]
                    for weight in learning_weights
                }

            local_nodes[layer_id] += 1
            nodes_by_layer[layer_id] += 1
            with torch.no_grad():
                state_losses, state_logits = deepest_losses(
                    net, info, labels, seq_end, a, prefix_times)
                action, meta = choose_option(
                    info, layer_id, target_batch, target_class,
                    state_logits[target_batch], query_horizon, args, rng)
            candidates_by_layer[layer_id] += meta["candidate_count"]

            no_values = visit(layer_id + 1, route_overrides,
                              spike_overrides, info, path)
            if action is None:
                return no_values
            sampled_actions.append(action)
            swapped_route_overrides = list(route_overrides or [])
            swapped_spike_overrides = list(spike_overrides or [])
            if action["mechanism"] == "spike_birth":
                swapped_spike_overrides.append((
                    layer_id, action["spike_time"], target_batch,
                    action["spike_unit"], True))
            elif action["mechanism"] == "birth":
                swapped_route_overrides.append(
                    (layer_id, action["event_id"], action["receiver"], True))
            else:
                swapped_route_overrides.extend([
                    (layer_id, action["event_id"], action["winner_receiver"], False),
                    (layer_id, action["event_id"], action["receiver"], True),
                ])
            with torch.no_grad():
                _, swapped_info = net(
                    eb, ei, et, batch_size, grid, return_taps=True,
                    collect_routes=True, return_spike_diagnostics=True,
                    route_overrides=swapped_route_overrides or None,
                    spike_overrides=swapped_spike_overrides or None,
                    input_counts=input_counts)
            swapped_path = path + [action]
            swap_values = visit(layer_id + 1, swapped_route_overrides or None,
                                swapped_spike_overrides or None,
                                swapped_info, swapped_path)
            rho = meta["swap_probability"]
            tau = args.backup_temperature
            backed_values = {
                str(weight): tau * float(np.logaddexp(
                    math.log1p(-rho) + no_values[str(weight)] / tau,
                    math.log(rho) + swap_values[str(weight)] / tau))
                for weight in learning_weights
            }
            decision_summaries.append({
                "layer": layer_id + 1,
                "action": action,
                "candidate_count": meta["candidate_count"],
                "p_true": meta["p_true"],
                "normalized_entropy": meta["entropy"],
                "error": meta["error"],
                "swap_probability": rho,
                "proposal_temperature": meta["temperature"],
                "no_swap_value_by_learning_weight": no_values,
                "swap_value_by_learning_weight": swap_values,
                "backed_value_by_learning_weight": backed_values,
            })
            return backed_values

        backup_values = visit(0, None, None, root_info, [])
        tree_rows.append({
            "subset_index": start + target_batch,
            "batch_index": start // args.batch_size,
            "label": target_class,
            "factual_prediction": int(factual_predictions[target_batch]),
            "factual_true_class_probability": float(p_true_all[target_batch]),
            "factual_normalized_entropy": float(entropy_all[target_batch]),
            "factual_loss": root_loss,
            "tree_node_count": int(local_nodes.sum()),
            "tree_leaf_count": len(leaves),
            "backup_value_by_learning_weight": backup_values,
            "best_immediate_leaf": max(
                leaves, key=lambda row: row["immediate_advantage_vs_factual"]),
            "best_learning_option_leaf": max(
                leaves, key=lambda row: row["learning_option_advantage"]),
            "best_combined_leaf_index_by_weight": {
                str(weight): max(
                    leaves, key=lambda row: row["immediate_advantage_vs_factual"]
                    + weight * row["learning_option_advantage"])["leaf_index"]
                for weight in learning_weights},
            "option_only_leaf_count": sum(row["option_only"] for row in leaves),
            "multi_swap_leaf_count": sum(row["swap_count"] >= 2 for row in leaves),
            "leaves": leaves,
            "recursive_decisions": decision_summaries,
        })

    os.makedirs(OUT, exist_ok=True)
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "run_tag": args.run_tag,
        "split": "train/val_spk",
        "examples_requested": args.examples,
        "examples_used": len(examples),
        "trees_with_factual_errors": len(tree_rows),
        "batch_size": args.batch_size,
        "depth": a.depth,
        "route_topk": int(a.route_topk),
        "proposal": {
            "condition": "target selected among factual errors by error and entropy",
            "route_candidate_weights": "exp(-route_margin / temperature)",
            "route_mode": args.route_mode,
            "route_birth": "activate a near-zero closed gate only when source event has unused top-k capacity",
            "spike_birth": "force one nonfiring, near-threshold, nonrefractory receiver at an exogenous prefix horizon",
            "temperature": "base_temperature + temperature_span * mean(normalized_entropy, 1-p_true)",
            "swap_probability": "swap_floor + swap_uncertainty_span * mean(normalized_entropy, 1-p_true)",
            "base_temperature": args.base_temperature,
            "temperature_span": args.temperature_span,
            "swap_floor": args.swap_floor,
            "swap_uncertainty_span": args.swap_uncertainty_span,
            "max_swap_probability": args.max_swap_probability,
            "sampled_route_probability_recorded": True,
        },
        "tree_backup": {
            "rule": "tau*log((1-rho)*exp(V_no/tau) + rho*exp(V_swap/tau))",
            "temperature": args.backup_temperature,
            "learning_weights": learning_weights,
            "interpretation": "one scalar is recursively propagated from each descendant; one route alternative is sampled from q at each state, not exhaustive over all candidates",
        },
        "virtual_update": {
            "optimizer": "one-step suffix-only clipped SGD diagnostic",
            "learning_rate": args.virtual_lr,
            "gradient_clip_norm": args.virtual_clip_norm,
            "suffix": "layers strictly after the last forced swap plus deepest event head",
            "weights_restored_after_each_replay": True,
            "forced_route_indices_stable": "all layers through the last forced swap are frozen during its suffix update",
        },
        "trees": tree_rows,
        "decision_nodes_by_layer": nodes_by_layer.tolist(),
        "near_boundary_candidates_seen_by_layer": candidates_by_layer.tolist(),
        "sampled_route_actions": sampled_actions,
        "interpretation_note": (
            "A sampled receiver swap, route birth, or spike birth is rerun through all deeper layers, "
            "and only scalar option values are propagated back; "
            "This is a frozen diagnostic, not trained accuracy, an AdamW update, or an exhaustive route expectation."
        ),
    }
    os.makedirs(OUT, exist_ok=True)
    output = os.path.join(OUT, f"route_option_value_{args.run_tag}.json")
    with open(output, "w") as stream:
        json.dump(result, stream, indent=2)
    summary = [{
        "subset_index": row["subset_index"],
        "factual_prediction": row["factual_prediction"],
        "label": row["label"],
        "tree_nodes": row["tree_node_count"],
        "leaves": row["tree_leaf_count"],
        "option_only_leaves": row["option_only_leaf_count"],
        "multi_swap_leaves": row["multi_swap_leaf_count"],
        "backup_values": row["backup_value_by_learning_weight"],
        "best_immediate_advantage": row["best_immediate_leaf"]["immediate_advantage_vs_factual"],
        "best_learning_option_advantage": row["best_learning_option_leaf"]["learning_option_advantage"],
    } for row in tree_rows]
    print(json.dumps({
        "output": output,
        "trees": summary,
        "decision_nodes_by_layer": nodes_by_layer.tolist(),
        "near_boundary_candidates_seen_by_layer": candidates_by_layer.tolist(),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
