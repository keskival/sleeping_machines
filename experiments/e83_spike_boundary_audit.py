"""Paired hidden-spike birth/death audit for an E83 checkpoint.

For each hidden layer and batch, choose the closest firing-threshold margin,
force that one spike to the opposite state, replay the same causal prefix loss,
and record L_on - L_off. The intervention includes the changed refractory
trace and all downstream event consequences.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

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


def event_work(info, depth):
    """Per-example event/message/readout counts for matched spike worlds."""
    messages = np.asarray(info["msgs"], dtype=np.float64)
    return {
        "hidden_spikes_per_example": [float(x) for x in info["spikes"][:depth]],
        "hidden_messages_per_example": (messages[:depth]).tolist(),
        "readout_edge_updates_per_example": (messages[depth:2 * depth]).tolist(),
    }


def work_delta(on, off):
    return {
        key: [round(float(a - b), 6) for a, b in zip(on[key], off[key])]
        for key in on
    }


def delivered_message_mask(net, info, layer_id, batch_size, grid):
    """(time,batch,receiver) mask for a selected upstream message already delivered."""
    layer = net.layers[layer_id]
    width = layer.M
    earliest = np.full((batch_size, width), np.inf, dtype=np.float64)
    route = info["route_candidates"][layer_id]
    active = route["active"].cpu().numpy().astype(bool)
    if active.any():
        receivers = route["receiver"].cpu().numpy()[active]
        event_ids = route["event_index"].cpu().numpy()[active]
        source_batch = route["source_batch"].cpu().numpy()[active]
        source_times = route["source_time"].cpu().numpy()[active]
        scores = route["score"].cpu().numpy()[active]
        source_units = info["route_inputs"][layer_id][1].cpu().numpy()
        receiver_t = torch.as_tensor(receivers, dtype=torch.long)
        if layer.cdelay:
            delay_score = torch.as_tensor(scores, dtype=layer.log_rate.dtype)
        else:
            source_unit_t = torch.as_tensor(source_units[event_ids], dtype=torch.long)
            delay_score = F.softplus(layer.c[source_unit_t, receiver_t]).detach()
        delays = (torch.exp(layer.log_td.detach()[receiver_t])
                  * delay_score.clamp(min=0)).clamp(max=layer.dmax).cpu().numpy()
        arrivals = np.ceil(source_times + delays).astype(np.int64)
        for batch, receiver, arrival in zip(source_batch, receivers, arrivals):
            if 0 <= arrival < grid:
                earliest[int(batch), int(receiver)] = min(
                    earliest[int(batch), int(receiver)], float(arrival))
    time = torch.arange(grid, dtype=torch.float32)[:, None, None]
    arrived = torch.as_tensor(earliest, dtype=torch.float32)[None, :, :] <= time
    return arrived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--examples", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--spike_band", type=float, default=0.25)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--prefix_samples", type=int, default=2)
    ap.add_argument("--layers", default="all",
                    help="comma-separated one-indexed hidden layers to audit, or 'all'")
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--run_tag", default="")
    ap.add_argument("--fusion", choices=("checkpoint", "deepest", "all_depths"),
                    default="checkpoint",
                    help="override the checkpoint's event-readout fusion for the frozen audit")
    ap.add_argument("--require_nonrefractory", action="store_true",
                    help="exclude candidate coordinates with positive refractory state")
    ap.add_argument("--require_input_arrival", action="store_true",
                    help="require a selected upstream message to have arrived by the candidate time")
    a = ap.parse_args()
    if a.examples < 1 or a.batch_size < 1 or a.spike_band <= 0 or a.sigma <= 0:
        raise ValueError("examples, batch size, spike band, and sigma must be positive")
    if not all(ch.isalnum() or ch in "-_" for ch in a.run_tag):
        raise ValueError("run_tag may contain only letters, digits, '-' and '_'")

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["args"]
    get = lambda key, default: saved.get(key, default)
    bands = int(get("bands", 140)); d = int(get("d", 8)); n = int(get("n", 4))
    M1 = int(get("M1", 16)); M = int(get("M", 16)); depth = int(get("depth", 4))
    window = int(get("window", 30)); fan2 = float(get("fan2", 0.25))
    readout_fan = float(get("readout_fan", 0.5)); dmax = float(get("dmax", 50.0))
    w_sd = [float(x) for x in str(get("w_sd", "0.05,0.05")).split(",")]
    seed = int(get("seed", a.seed)); aux_weight = float(get("aux_weight", 0.2))
    fusion = get("readout_fusion", None) or "deepest"
    if a.fusion != "checkpoint":
        fusion = a.fusion
    if a.layers == "all":
        layers_to_audit = set(range(depth))
    else:
        try:
            layers_to_audit = {int(value) - 1 for value in a.layers.split(",")}
        except ValueError as exc:
            raise ValueError("layers must be 'all' or comma-separated layer numbers") from exc
        if not layers_to_audit or any(k < 0 or k >= depth for k in layers_to_audit):
            raise ValueError(f"layers must be in [1, {depth}]")

    torch.manual_seed(seed)
    net = DeepSHD(bands, d, n, M1, M, depth, window, fan2, readout_fan,
                  dmax, w_sd, seed, event_readout=True, readout_fusion=fusion,
                  input_count_payload=get("input_count_payload", "off") == "additive",
                  early_event_skip=bool(get("early_event_skip", False)),
                  route_topk=int(get("route_topk", 0)))
    net.load_state_dict(ckpt["model_state_dict"])
    for layer in net.layers:
        layer.spike_reconstruction = get("spike_reconstruction", "legacy")
    net.eval()

    rng = np.random.default_rng(a.seed + 71_337)
    data = [(*events(t, u, float(get("merge", 0.002))), y)
            for t, u, y in S.utterances("train", bands, "val_spk") if len(t) > 1]
    samples = stratified_limit(data, a.examples, rng)
    per_layer = [[] for _ in range(depth)]
    eligible_batches = np.zeros(depth, dtype=np.int64)
    batches = 0
    for start in range(0, len(samples), a.batch_size):
        items = samples[start:start + a.batch_size]
        eb, ei, et, input_counts, labels, tmax, seq_end = batch_to_events(
            items, bands, 0, rng, 0.0)
        grid = tmax + (depth + 1) * int(np.ceil(dmax)) + 60
        prefix_times = sampled_prefix_times(len(items), a.prefix_samples,
                                            1000.0, 0.0, et.device, et.dtype)
        with torch.no_grad():
            _, base_info = net(eb, ei, et, len(items), grid, return_taps=True,
                               collect_routes=a.require_input_arrival,
                               return_spike_diagnostics=True,
                               input_counts=input_counts)
            base_main, base_aux, _ = sparse_event_objective(
                net.event_heads, base_info["layer_events"], labels, seq_end,
                dmax, prefix_times, fusion)
            base_loss = float((base_main + aux_weight * base_aux).item())
            base_main_value = float(base_main.item())
            base_aux_value = float(base_aux.item())

        for layer_id, diagnostic in enumerate(base_info["spike_diagnostics"]):
            if layer_id not in layers_to_audit:
                continue
            margins = diagnostic["spike_margin_trace"]
            fired = diagnostic["spike_fire_mask"]
            refractory = diagnostic["spike_refractory_trace"]
            valid = torch.ones_like(margins, dtype=torch.bool)
            valid[0] = False  # TVLayer intentionally cannot emit at grid index 0.
            if a.require_nonrefractory:
                valid &= refractory < 1e-3
            if a.require_input_arrival:
                valid &= delivered_message_mask(
                    net, base_info, layer_id, len(items), grid)
            if not valid.any():
                continue
            eligible_batches[layer_id] += 1
            near = valid & (margins.abs() <= a.spike_band)
            eligible = int(near.sum().item())
            if eligible:
                scores = margins.abs().masked_fill(~near, float("inf"))
                used_band = True
            else:
                scores = margins.abs().masked_fill(~valid, float("inf"))
                used_band = False
            flat = int(scores.reshape(-1).argmin().item())
            time_id, batch_id, unit_id = np.unravel_index(flat, tuple(scores.shape))
            margin = float(margins[time_id, batch_id, unit_id].item())
            was_firing = bool(fired[time_id, batch_id, unit_id].item())
            refractory_state = float(diagnostic["spike_refractory_trace"][
                time_id, batch_id, unit_id].item())
            with torch.no_grad():
                _, shadow_info = net(
                    eb, ei, et, len(items), grid, return_taps=True,
                    spike_override=(layer_id, int(time_id), int(batch_id),
                                    int(unit_id), not was_firing),
                    input_counts=input_counts)
                shadow_main, shadow_aux, _ = sparse_event_objective(
                    net.event_heads, shadow_info["layer_events"], labels,
                    seq_end, dmax, prefix_times, fusion)
                shadow_loss = float((shadow_main + aux_weight * shadow_aux).item())
                shadow_main_value = float(shadow_main.item())
                shadow_aux_value = float(shadow_aux.item())

            # Report the loss difference for a spike-on versus spike-off world.
            l_on_minus_l_off = (base_loss - shadow_loss if was_firing
                                else shadow_loss - base_loss)
            main_l_on_minus_l_off = (base_main_value - shadow_main_value if was_firing
                                     else shadow_main_value - base_main_value)
            aux_l_on_minus_l_off = (base_aux_value - shadow_aux_value if was_firing
                                    else shadow_aux_value - base_aux_value)
            base_work = event_work(base_info, depth)
            shadow_work = event_work(shadow_info, depth)
            spike_on_work = base_work if was_firing else shadow_work
            spike_off_work = shadow_work if was_firing else base_work
            p = 1.0 / (1.0 + np.exp(-np.clip(margin / a.sigma, -60.0, 60.0)))
            boundary_coefficient = p * (1.0 - p) * l_on_minus_l_off / a.sigma
            per_layer[layer_id].append({
                "batch": batches,
                "time_ms": int(time_id),
                "item_in_batch": int(batch_id),
                "unit": int(unit_id),
                "margin": margin,
                "naturally_firing": was_firing,
                "refractory_state": refractory_state,
                "refractory_blocked": refractory_state > 1e-6,
                "require_input_arrival": bool(a.require_input_arrival),
                "require_nonrefractory": bool(a.require_nonrefractory),
                "within_spike_band": used_band,
                "eligible_margins_in_batch": eligible,
                "base_loss": base_loss,
                "toggled_loss": shadow_loss,
                "L_on_minus_L_off": l_on_minus_l_off,
                "main_L_on_minus_L_off": main_l_on_minus_l_off,
                "aux_L_on_minus_L_off": aux_l_on_minus_l_off,
                "spike_on_minus_off_work": work_delta(spike_on_work, spike_off_work),
                "single_utterance_L_on_minus_L_off": len(items) * l_on_minus_l_off,
                "logistic_boundary_coefficient": float(boundary_coefficient),
            })
        batches += 1

    summary = []
    for layer_id, rows in enumerate(per_layer):
        deltas = np.asarray([row["L_on_minus_L_off"] for row in rows], dtype=float)
        main_deltas = np.asarray([row["main_L_on_minus_L_off"] for row in rows], dtype=float)
        aux_deltas = np.asarray([row["aux_L_on_minus_L_off"] for row in rows], dtype=float)
        coeffs = np.asarray([row["logistic_boundary_coefficient"] for row in rows], dtype=float)
        margins = np.asarray([row["margin"] for row in rows], dtype=float)
        summary.append({
            "layer": layer_id + 1,
            "shadows": len(rows),
            "eligible_batches": int(eligible_batches[layer_id]),
            "batches": batches,
            "fraction_batches_with_in_band_candidate": float(sum(
                row["within_spike_band"] for row in rows) / batches) if batches else None,
            "fraction_within_band": float(np.mean([row["within_spike_band"] for row in rows])) if rows else None,
            "fraction_naturally_firing": float(np.mean([row["naturally_firing"] for row in rows])) if rows else None,
            "fraction_refractory_state_positive": float(np.mean([row["refractory_blocked"] for row in rows])) if rows else None,
            "fraction_in_band_and_not_refractory": float(np.mean([
                row["within_spike_band"] and not row["refractory_blocked"] for row in rows
            ])) if rows else None,
            "mean_abs_margin": float(np.mean(np.abs(margins))) if len(margins) else None,
            "mean_L_on_minus_L_off": float(np.mean(deltas)) if len(deltas) else None,
            "std_L_on_minus_L_off": float(np.std(deltas)) if len(deltas) else None,
            "fraction_spike_on_improves_loss": float(np.mean(deltas < 0)) if len(deltas) else None,
            "mean_main_L_on_minus_L_off": float(np.mean(main_deltas)) if len(main_deltas) else None,
            "std_main_L_on_minus_L_off": float(np.std(main_deltas)) if len(main_deltas) else None,
            "fraction_spike_on_improves_main_loss": float(np.mean(main_deltas < 0)) if len(main_deltas) else None,
            "mean_aux_L_on_minus_L_off": float(np.mean(aux_deltas)) if len(aux_deltas) else None,
            "std_aux_L_on_minus_L_off": float(np.std(aux_deltas)) if len(aux_deltas) else None,
            "fraction_spike_on_improves_aux_loss": float(np.mean(aux_deltas < 0)) if len(aux_deltas) else None,
            "mean_abs_logistic_boundary_coefficient": float(np.mean(np.abs(coeffs))) if len(coeffs) else None,
        })
    result = {
        "checkpoint": a.checkpoint,
        "checkpoint_args": saved,
        "audit_args": vars(a),
        "run_tag": a.run_tag,
        "split": "train/val_spk",
        "examples": len(samples),
        "batches": batches,
        "prefix_loss_fusion": fusion,
        "per_layer": summary,
        "paired_shadows": per_layer,
    }
    os.makedirs(OUT, exist_ok=True)
    tag = f"_{a.run_tag}" if a.run_tag else ""
    path = os.path.join(OUT, f"spike_boundary_audit{tag}_s{a.seed}_n{a.examples}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"result": path, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
