"""Frozen second-order hidden-spike audit for an E83 checkpoint.

Select one pair of distinct, in-band, nonrefractory threshold candidates from
one hidden layer per batch, then replay all four binary outcomes through the
ordinary suffix. Optional shared-receiver selection tests whether both source
events can reach a common next-layer unit. Diagnostic only.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import e51_shd_world as S  # noqa: E402
from e71_event_cde import events  # noqa: E402
from e83_deep_shd import (  # noqa: E402
    DeepSHD,
    batch_to_events,
    sampled_prefix_times,
    sparse_event_objective,
    stratified_limit,
)

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e83")


def loss_parts(net, info, labels, seq_end, dmax, prefix_times):
    deep, aux, _ = sparse_event_objective(
        net.event_heads, info["layer_events"], labels, seq_end, dmax,
        prefix_times, "deepest")
    fused, _, _ = sparse_event_objective(
        net.event_heads, info["layer_events"], labels, seq_end, dmax,
        prefix_times, "all_depths")
    return {"deepest_main": float(deep.item()),
            "all_depths_main": float(fused.item()),
            "auxiliary": float(aux.item())}


def event_work(net, info, batch_size):
    depth = net.depth
    hidden_spikes = np.zeros((depth, batch_size), dtype=np.int64)
    readout_edges = np.zeros((depth, batch_size), dtype=np.int64)
    for layer_id, (event_b, units, _times, _payload) in enumerate(info["layer_events"]):
        if len(event_b):
            counts = np.bincount(event_b.detach().cpu().numpy(), minlength=batch_size)
            hidden_spikes[layer_id] = counts
            edge_counts = (net.event_heads[layer_id].row_ptr[units + 1]
                           - net.event_heads[layer_id].row_ptr[units])
            readout_edges[layer_id] = np.bincount(
                event_b.detach().cpu().numpy(),
                weights=edge_counts.detach().cpu().numpy(), minlength=batch_size)
    return {"hidden_spikes_per_example": hidden_spikes,
            "readout_edges_per_example": readout_edges}


def work_delta(after, before):
    return {key: (after[key] - before[key]).tolist() for key in before}


def select_pair(route_info, band, window_ms, shared_receiver_mask=None):
    """Choose a near-boundary pair, optionally requiring a common next-layer child."""
    margins = route_info["spike_margin_trace"].cpu().numpy()
    fired = route_info["spike_fire_mask"].cpu().numpy()
    refractory = route_info["spike_refractory_trace"].cpu().numpy()
    valid = (np.abs(margins) <= band) & (refractory <= 1e-6)
    valid[0] = False  # TVLayer does not emit a crossing at simulation tick 0.
    candidates = []
    for t, b, u in np.argwhere(valid):
        candidates.append({"time": int(t), "batch": int(b), "unit": int(u),
                           "margin": float(margins[t, b, u]),
                           "naturally_firing": bool(fired[t, b, u])})
    pair_options = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if a["batch"] != b["batch"] or a["unit"] == b["unit"]:
                continue
            gap = abs(a["time"] - b["time"])
            if gap > window_ms:
                continue
            shared_receivers = []
            if shared_receiver_mask is not None:
                shared_receivers = np.flatnonzero(
                    shared_receiver_mask[a["unit"]]
                    & shared_receiver_mask[b["unit"]]).tolist()
                if not shared_receivers:
                    continue
            ma, mb = abs(a["margin"]), abs(b["margin"])
            pair_options.append(((ma + mb, max(ma, mb), gap,
                                  a["batch"], a["time"], b["time"],
                                  a["unit"], b["unit"]), a, b, shared_receivers))
    if not pair_options:
        return None, len(candidates)
    _, first, second, shared_receivers = min(pair_options, key=lambda item: item[0])
    return (first, second, shared_receivers), len(candidates)


def shared_receiver_delivery(net, info, source_layer, pair, shared_receivers, batch_id):
    """Read actual accepted common receivers for the all-events-on outcome."""
    if source_layer + 1 >= net.depth:
        return {"accepted_shared_receivers": [], "arrival_gaps_ms": []}
    out_b, out_units, out_times, _ = info["layer_events"][source_layer]
    _, input_units, _ = info["route_inputs"][source_layer + 1]
    route = info["route_candidates"][source_layer + 1]
    out_b = out_b.detach().cpu().numpy()
    out_units = out_units.detach().cpu().numpy()
    out_times = out_times.detach().cpu().numpy()
    input_units = input_units.detach().cpu().numpy()
    event_index = route["event_index"].detach().cpu().numpy()
    receiver = route["receiver"].detach().cpu().numpy()
    score = route["score"].detach().cpu().numpy()
    source_time = route["source_time"].detach().cpu().numpy()
    child_layer = net.layers[source_layer + 1]
    delay_scale = torch.exp(child_layer.log_td.detach()).cpu().numpy()
    accepted = []
    arrival_gaps = []
    for target in shared_receivers:
        arrivals = []
        for candidate in pair[:2]:
            possible = np.flatnonzero(
                (out_b == batch_id) & (out_units == candidate["unit"]) &
                (np.abs(out_times - candidate["time"]) <= 1.0001))
            if not len(possible):
                arrivals = []
                break
            source_event = int(possible[np.argmin(
                np.abs(out_times[possible] - candidate["time"]))])
            edge_rows = np.flatnonzero((event_index == source_event) & (receiver == target))
            if not len(edge_rows):
                arrivals = []
                break
            row = int(edge_rows[0])
            if score[row] <= 0:
                arrivals = []
                break
            if child_layer.cdelay:
                delay_score = max(score[row], 0.0)
            else:
                source_unit = int(input_units[source_event])
                static_gate = float(child_layer.c[source_unit, target].detach())
                delay_score = float(np.logaddexp(0.0, static_gate))
            delay = min(float(delay_scale[target] * delay_score), child_layer.dmax)
            arrivals.append(float(source_time[row] + delay))
        if len(arrivals) == 2:
            accepted.append(int(target))
            arrival_gaps.append(abs(arrivals[0] - arrivals[1]))
    return {"accepted_shared_receivers": accepted,
            "arrival_gaps_ms": arrival_gaps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--examples", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--spike_band", type=float, default=0.25)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--layer", type=int, default=1,
                    help="hidden layer to pair (one-indexed)")
    ap.add_argument("--pair_window_ms", type=int, default=50)
    ap.add_argument("--require_shared_receiver", action="store_true",
                    help="restrict candidate pairs to units with a common next-layer child")
    ap.add_argument("--prefix_samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--run_tag", default="")
    a = ap.parse_args()
    if (a.examples < 1 or a.batch_size < 1 or a.spike_band <= 0
            or a.sigma <= 0 or a.pair_window_ms < 0):
        raise ValueError("examples/batch_size/band/sigma must be positive and pair window nonnegative")
    if not all(ch.isalnum() or ch in "-_" for ch in a.run_tag):
        raise ValueError("run_tag may contain only letters, digits, '-' and '_'")

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["args"]
    get = lambda key, default: saved.get(key, default)
    bands = int(get("bands", 140)); d = int(get("d", 8)); n = int(get("n", 4))
    M1 = int(get("M1", 16)); M = int(get("M", 16)); depth = int(get("depth", 4))
    window = int(get("window", 30)); fan2 = float(get("fan2", 0.25))
    readout_fan = float(get("readout_fan", 0.5)); dmax = float(get("dmax", 50.0))
    if not 1 <= a.layer <= depth:
        raise ValueError(f"layer must be in [1, {depth}]")
    layer_id = a.layer - 1
    shared_receiver_mask = None
    if a.require_shared_receiver:
        if layer_id + 1 >= depth:
            raise ValueError("shared-receiver pairing requires a following hidden layer")
    w_sd = [float(x) for x in str(get("w_sd", "0.05,0.05")).split(",")]
    seed = int(get("seed", a.seed)); aux_weight = float(get("aux_weight", 0.2))
    fusion = get("readout_fusion", None) or "deepest"
    torch.manual_seed(seed)
    net = DeepSHD(bands, d, n, M1, M, depth, window, fan2, readout_fan,
                  dmax, w_sd, seed, event_readout=True, readout_fusion=fusion,
                  input_count_payload=get("input_count_payload", "off") == "additive",
                  early_event_skip=bool(get("early_event_skip", False)),
                  route_topk=int(get("route_topk", 0)))
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    if a.require_shared_receiver:
        full_mask = net.layers[layer_id + 1].mask
        if full_mask is None:
            raise ValueError("shared-receiver pairing requires explicit sparse connectivity")
        shared_receiver_mask = full_mask[:net.widths[layer_id]].detach().cpu().numpy().astype(bool)

    rng = np.random.default_rng(a.seed + 71_337)
    data = [(*events(t, u, float(get("merge", 0.002))), y)
            for t, u, y in S.utterances("train", bands, "val_spk") if len(t) > 1]
    samples = stratified_limit(data, a.examples, rng)
    records = []
    attempted = 0
    for start in range(0, len(samples), a.batch_size):
        items = samples[start:start + a.batch_size]
        eb, ei, et, input_counts, labels, tmax, seq_end = batch_to_events(
            items, bands, 0, rng, 0.0)
        grid = tmax + (depth + 1) * int(np.ceil(dmax)) + 60
        prefix_times = sampled_prefix_times(len(items), a.prefix_samples,
                                            1000.0, 0.0, et.device, et.dtype)
        with torch.no_grad():
            _, base_info = net(eb, ei, et, len(items), grid, return_taps=True,
                               collect_routes=a.require_shared_receiver,
                               return_spike_diagnostics=True,
                               input_counts=input_counts)
            base_loss = loss_parts(net, base_info, labels, seq_end, dmax, prefix_times)
            base_work = event_work(net, base_info, len(items))
        pair, candidate_count = select_pair(base_info["spike_diagnostics"][layer_id],
                                            a.spike_band, a.pair_window_ms,
                                            shared_receiver_mask)
        if pair is None:
            records.append({"batch_start": start, "valid_spike_candidates": candidate_count,
                            "pair_found": False})
            continue
        attempted += 1
        states = {}
        a_candidate, b_candidate = pair[:2]
        natural = (int(a_candidate["naturally_firing"]),
                   int(b_candidate["naturally_firing"]))
        all_on_info = base_info if natural == (1, 1) else None
        for toggles in ((), (0,), (1,), (0, 1)):
            if not toggles:
                states["factual"] = (base_loss, base_work)
                continue
            selected = [pair[i] for i in toggles]
            outcome = list(natural)
            for i in toggles:
                outcome[i] = 1 - outcome[i]
            overrides = [(layer_id, item["time"], item["batch"], item["unit"],
                          not item["naturally_firing"]) for item in selected]
            with torch.no_grad():
                _, shadow_info = net(eb, ei, et, len(items), grid, return_taps=True,
                                     collect_routes=a.require_shared_receiver,
                                     spike_overrides=overrides,
                                     input_counts=input_counts)
                states["toggle_" + "".join(str(i) for i in toggles)] = (
                    loss_parts(net, shadow_info, labels, seq_end, dmax, prefix_times),
                    event_work(net, shadow_info, len(items)))
            if tuple(outcome) == (1, 1):
                all_on_info = shadow_info

        if all_on_info is None:
            raise RuntimeError("failed to replay the both-spikes-on outcome")
        corner_losses, corner_work = {}, {}
        corner_losses[natural] = states["factual"][0]
        corner_work[natural] = states["factual"][1]
        for key, changed in (("toggle_0", (0,)), ("toggle_1", (1,)),
                             ("toggle_01", (0, 1))):
            outcome = list(natural)
            for idx in changed:
                outcome[idx] = 1 - outcome[idx]
            corner_losses[tuple(outcome)] = states[key][0]
            corner_work[tuple(outcome)] = states[key][1]
        L00, L10 = corner_losses[(0, 0)], corner_losses[(1, 0)]
        L01, L11 = corner_losses[(0, 1)], corner_losses[(1, 1)]
        m_a, m_b = a_candidate["margin"], b_candidate["margin"]
        p_a = 1.0 / (1.0 + np.exp(-np.clip(m_a / a.sigma, -60, 60)))
        p_b = 1.0 / (1.0 + np.exp(-np.clip(m_b / a.sigma, -60, 60)))

        def interaction(metric):
            gamma = (L11[metric] - L10[metric] - L01[metric] + L00[metric])
            grad_a = (p_a * (1.0 - p_a) / a.sigma) * (
                (1.0 - p_b) * (L10[metric] - L00[metric])
                + p_b * (L11[metric] - L01[metric]))
            grad_b = (p_b * (1.0 - p_b) / a.sigma) * (
                (1.0 - p_a) * (L01[metric] - L00[metric])
                + p_a * (L11[metric] - L10[metric]))
            return {"L00": L00[metric], "L10": L10[metric],
                    "L01": L01[metric], "L11": L11[metric],
                    "interaction_gamma": gamma,
                    "expected_margin_gradients": [float(grad_a), float(grad_b)]}

        example_id = a_candidate["batch"]
        pair_flip_work = states["toggle_01"][1]
        factual_spikes = base_work["hidden_spikes_per_example"][:, example_id]
        pair_flip_spikes = pair_flip_work["hidden_spikes_per_example"][:, example_id]
        downstream_slice = slice(layer_id + 1, depth)
        pair_flip_downstream_birth = bool(
            np.any(pair_flip_spikes[downstream_slice] > 0)
            and not np.any(factual_spikes[downstream_slice] > 0))
        layer_work = {}
        for outcome, work in corner_work.items():
            layer_work["".join(map(str, outcome))] = {
                "hidden_spikes": work["hidden_spikes_per_example"][:, example_id].tolist(),
                "readout_edges": work["readout_edges_per_example"][:, example_id].tolist()}
        records.append({
            "batch_start": start,
            "valid_spike_candidates": candidate_count,
            "pair_found": True,
            "selected_example": int(example_id),
            "candidates": pair[:2],
            "natural_state": list(natural),
            "layer": a.layer,
            "pair_window_ms": a.pair_window_ms,
            "pair_flip_downstream_spike_birth": pair_flip_downstream_birth,
            "pair_flip_downstream_spike_delta": (
                pair_flip_spikes[downstream_slice] - factual_spikes[downstream_slice]).tolist(),
            "deepest": interaction("deepest_main"),
            "all_depths": interaction("all_depths_main"),
            "auxiliary_corner_losses": {"factual": states["factual"][0]["auxiliary"],
                                         "toggle_0": states["toggle_0"][0]["auxiliary"],
                                         "toggle_1": states["toggle_1"][0]["auxiliary"],
                                         "toggle_01": states["toggle_01"][0]["auxiliary"]},
            "selected_example_work_by_corner": layer_work,
            "pair_flip_delta_deepest": states["toggle_01"][0]["deepest_main"] - base_loss["deepest_main"],
            "pair_flip_delta_all_depths": states["toggle_01"][0]["all_depths_main"] - base_loss["all_depths_main"],
        })
        if a.require_shared_receiver:
            shared = shared_receiver_delivery(net, all_on_info, layer_id, pair[:2],
                                              pair[2], int(example_id))
            records[-1]["shared_topology_receivers"] = pair[2]
            records[-1].update(shared)

    flat = [r for r in records if r.get("pair_found")]
    summary = {}
    for metric in ("deepest", "all_depths"):
        gammas = [r[metric]["interaction_gamma"] for r in flat]
        summary[metric] = {
            "pairs": len(flat),
            "nonzero_interactions": sum(abs(x) > 1e-12 for x in gammas),
            "mean_interaction_gamma": float(np.mean(gammas)) if gammas else None,
            "mean_abs_interaction_gamma": float(np.mean(np.abs(gammas))) if gammas else None,
            "pairs_with_downstream_spike_birth": sum(
                r["pair_flip_downstream_spike_birth"] for r in flat),
            "pairs_with_helpful_joint_flip": sum(r[f"pair_flip_delta_{metric}"] < 0 for r in flat),
        }
    if a.require_shared_receiver:
        summary["shared_receiver"] = {
            "pairs_with_accepted_common_receiver": sum(
                bool(r.get("accepted_shared_receivers")) for r in flat),
            "accepted_common_receiver_count": sum(
                len(r.get("accepted_shared_receivers", [])) for r in flat),
            "median_arrival_gap_ms": float(np.median([
                gap for r in flat for gap in r.get("arrival_gaps_ms", [])]))
                if any(r.get("arrival_gaps_ms") for r in flat) else None,
        }
    result = {"checkpoint": a.checkpoint, "checkpoint_args": saved,
              "audit_args": vars(a), "split": "train/val_spk",
              "examples": len(samples), "batches": len(records),
              "pairs_attempted": attempted, "summary": summary,
              "per_batch": records}
    os.makedirs(OUT, exist_ok=True)
    tag = f"_{a.run_tag}" if a.run_tag else ""
    layer_tag = "" if a.layer == 1 else f"_L{a.layer}"
    path = os.path.join(OUT, f"spike_pair_audit{layer_tag}{tag}_s{a.seed}_n{a.examples}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"result": path, "pairs": attempted, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
