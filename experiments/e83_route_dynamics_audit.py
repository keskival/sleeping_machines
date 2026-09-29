"""Frozen E83 audit separating router load from temporal event propagation.

For a saved sparse route policy, count the per-layer receiver load, source
fan-out, arrival coincidences, hidden spikes, support, and exact receiver
voltage margins on the same held-out-speaker subset used by training. This is
diagnostic only: it performs no updates and never constructs dense event by
receiver tensors.
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
from e83_deep_shd import DeepSHD, batch_to_events, stratified_limit  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e83")


def gini(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or values.sum() <= 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    return float((2.0 * np.dot(np.arange(1, n + 1), values)
                  / (n * values.sum())) - (n + 1.0) / n)


def close_arrival_pairs(groups, windows):
    """Count within-window arrival pairs using a linear sliding window."""
    totals = {window: 0 for window in windows}
    for times in groups.values():
        times = np.sort(np.asarray(times, dtype=np.float64))
        for window in windows:
            left = pairs = 0
            for right in range(len(times)):
                while times[right] - times[left] > window:
                    left += 1
                pairs += right - left
            totals[window] += pairs
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--run_tag", default="route_dynamics")
    args_cli = parser.parse_args()
    if args_cli.examples < 1 or args_cli.batch_size < 1:
        raise ValueError("examples and batch_size must be positive")
    if not all(ch.isalnum() or ch in "-_" for ch in args_cli.run_tag):
        raise ValueError("run_tag may contain only letters, digits, '-' and '_'")

    checkpoint = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    a = SimpleNamespace(**checkpoint["args"])
    if a.objective != "event_prefix":
        raise ValueError("route dynamics audit requires event_prefix checkpoint")
    torch.manual_seed(int(getattr(a, "seed", args_cli.seed)))
    eval_rng = np.random.default_rng(int(getattr(a, "seed", args_cli.seed)) + 300_007)
    data = [(*events(t, u, a.merge), y)
            for t, u, y in S.utterances("train", a.bands, "val_spk") if len(t) > 1]
    examples = stratified_limit(data, args_cli.examples, eval_rng)
    widths_sd = [float(v) for v in str(a.w_sd).split(",")]
    net = DeepSHD(
        a.bands, a.d, a.n, a.M1, a.M, a.depth, a.window, a.fan2,
        a.readout_fan, a.dmax, widths_sd, a.seed,
        event_readout=True,
        readout_fusion=a.readout_fusion,
        input_count_payload=a.input_count_payload == "additive",
        early_event_skip=bool(getattr(a, "early_event_skip", False)),
        route_topk=int(getattr(a, "route_topk", 0)),
    )
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    depth = a.depth
    widths = [layer.M for layer in net.layers]
    recv_load = [np.zeros(width, dtype=np.int64) for width in widths]
    recv_spikes = [np.zeros(width, dtype=np.int64) for width in widths]
    source_out_counts = [[] for _ in range(depth)]
    score_open = [0 for _ in range(depth)]
    score_count = [0 for _ in range(depth)]
    selected_scores = [[] for _ in range(depth)]
    max_margin_by_receiver_example = [[] for _ in range(depth)]
    pairs_by_window = [{10.0: 0, 50.0: 0} for _ in range(depth)]
    accepted = np.zeros(depth, dtype=np.int64)
    candidates = np.zeros(depth, dtype=np.int64)
    spikes_by_example = np.zeros((len(examples), depth), dtype=np.int64)
    support_by_layer = np.zeros(depth, dtype=np.int64)

    with torch.no_grad():
        for start in range(0, len(examples), args_cli.batch_size):
            items = examples[start:start + args_cli.batch_size]
            eb, ei, et, input_counts, _, tmax, _ = batch_to_events(
                items, a.bands, 0, np.random.default_rng(0), 0.0)
            grid = tmax + (depth + 1) * math.ceil(a.dmax) + 60
            _, info = net(eb, ei, et, len(items), grid, return_taps=True,
                          collect_routes=True, return_spike_diagnostics=True,
                          input_counts=input_counts)
            for layer_id, route_info in enumerate(info["route_candidates"]):
                scores = route_info["score"].cpu().numpy()
                receivers = route_info["receiver"].cpu().numpy()
                event_ids = route_info["event_index"].cpu().numpy()
                source_batch = route_info["source_batch"].cpu().numpy()
                source_times = route_info["source_time"].cpu().numpy()
                active = route_info["active"].cpu().numpy().astype(bool)
                candidates[layer_id] += len(scores)
                score_count[layer_id] += len(scores)
                score_open[layer_id] += int(np.count_nonzero(scores > 0.0))
                if active.any():
                    accepted[layer_id] += int(active.sum())
                    selected_scores[layer_id].extend(scores[active].tolist())
                    recv_load[layer_id] += np.bincount(
                        receivers[active], minlength=widths[layer_id])
                    unique_events, out_counts = np.unique(event_ids[active],
                                                          return_counts=True)
                    source_out_counts[layer_id].extend(out_counts.tolist())

                    layer = net.layers[layer_id]
                    rd = torch.as_tensor(scores[active], dtype=layer.log_rate.dtype)
                    if not layer.cdelay:
                        src_units = info["route_inputs"][layer_id][1].cpu().numpy()
                        rd = F.softplus(layer.c[src_units[event_ids[active]],
                                                torch.as_tensor(receivers[active])]).detach()
                    receiver_t = torch.as_tensor(receivers[active], dtype=torch.long)
                    delay = (torch.exp(layer.log_td.detach()[receiver_t])
                             * rd.clamp(min=0)).clamp(max=layer.dmax).cpu().numpy()
                    arrival = source_times[active] + delay
                    arrival_groups = {}
                    for bi, receiver, at in zip(source_batch[active],
                                                receivers[active], arrival):
                        arrival_groups.setdefault((int(bi), int(receiver)), []).append(float(at))
                    pair_counts = close_arrival_pairs(arrival_groups, (10.0, 50.0))
                    for window, count in pair_counts.items():
                        pairs_by_window[layer_id][window] += count

                margin_trace = route_info["spike_margin_trace"]
                fire_mask = route_info["spike_fire_mask"]
                receiver_margin = margin_trace.amax(dim=0).cpu().numpy()
                max_margin_by_receiver_example[layer_id].extend(receiver_margin.reshape(-1).tolist())
                recv_spikes[layer_id] += fire_mask.sum(dim=(0, 1)).cpu().numpy().astype(np.int64)
                out_batch = info["layer_events"][layer_id][0]
                if out_batch.numel():
                    counts = torch.bincount(out_batch, minlength=len(items)).cpu().numpy()
                    spikes_by_example[start:start + len(items), layer_id] = counts
                    support_by_layer[layer_id] += int(np.count_nonzero(counts))

    per_layer = []
    for k in range(depth):
        load = recv_load[k]
        total = int(load.sum())
        proportions = load / max(total, 1)
        nonzero = proportions[proportions > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum()) if len(nonzero) else 0.0
        counts = np.asarray(source_out_counts[k], dtype=np.float64)
        margins = np.asarray(max_margin_by_receiver_example[k], dtype=np.float64)
        pair10 = pairs_by_window[k][10.0]
        pair50 = pairs_by_window[k][50.0]
        per_layer.append({
            "layer": k + 1,
            "width": widths[k],
            "candidate_edges": int(candidates[k]),
            "positive_score_fraction": round(score_open[k] / max(score_count[k], 1), 6),
            "selected_messages": int(accepted[k]),
            "positive_gate_to_selected_fraction": round(accepted[k] / max(score_open[k], 1), 6),
            "receiver_fraction_used": round(float(np.count_nonzero(load)) / widths[k], 6),
            "receiver_load_entropy": round(entropy, 6),
            "receiver_load_entropy_fraction_of_max": round(entropy / math.log(widths[k]), 6)
                if widths[k] > 1 else 1.0,
            "receiver_load_max_share": round(float(proportions.max(initial=0.0)), 6),
            "receiver_load_gini": round(gini(load), 6),
            "per_receiver_selected_counts": load.tolist(),
            "mean_max_arrival_window_pairs_per_example_receiver_10ms": round(
                pair10 / len(examples), 4),
            "mean_max_arrival_window_pairs_per_example_receiver_50ms": round(
                pair50 / len(examples), 4),
            "mean_source_out_degree": round(float(counts.mean()), 6) if len(counts) else 0.0,
            "source_out_degree_histogram": {
                str(int(x)): int(np.count_nonzero(counts == x))
                for x in np.unique(counts).astype(int)},
            "hidden_spikes_per_example": round(float(spikes_by_example[:, k].mean()), 6),
            "hidden_support_fraction": round(float(support_by_layer[k]) / len(examples), 6),
            "spikes_per_receiver": recv_spikes[k].tolist(),
            "receiver_max_voltage_margin_quantiles": {
                str(q): round(float(np.quantile(margins, q)), 6)
                for q in (0.5, 0.9, 0.99, 1.0)} if len(margins) else {},
        })

    result = {
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "run_tag": args_cli.run_tag,
        "split": "train/val_spk",
        "examples": len(examples),
        "batch_size": args_cli.batch_size,
        "route_topk": int(getattr(a, "route_topk", 0)),
        "readout_fusion": a.readout_fusion,
        "per_layer": per_layer,
        "interpretation_note": (
            "Frozen route counts, receiver load, arrival coincidences, and actual voltage margins "
            "separate score-level route abundance from causal temporal integration. No training "
            "or dense event-by-receiver scoring is performed."
        ),
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"route_dynamics_{args_cli.run_tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"output": path, "examples": len(examples),
                      "per_layer": [{key: row[key] for key in (
                          "layer", "candidate_edges", "selected_messages",
                          "receiver_fraction_used", "receiver_load_gini",
                          "mean_max_arrival_window_pairs_per_example_receiver_10ms",
                          "mean_max_arrival_window_pairs_per_example_receiver_50ms",
                          "hidden_spikes_per_example", "hidden_support_fraction",
                          "receiver_max_voltage_margin_quantiles")}
                          for row in per_layer]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
