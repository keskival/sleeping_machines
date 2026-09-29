"""Measure how deep route-pair proposal availability changes with time window.

Frozen-checkpoint diagnostic only: it does not shadow routes or update weights.
It reports single near-closed route counts and adjacent same-receiver pair
counts by layer for several windows, along with how many minibatches have at
least one pair. This distinguishes temporal-window scarcity from missing
deep co-occupancy before another training run is designed.
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
    stratified_limit,
)

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e83")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--windows_ms", default="25,50,100,250,500,1000")
    parser.add_argument("--run_tag", default="pair_occupancy")
    args_cli = parser.parse_args()
    if not all(ch.isalnum() or ch in "-_" for ch in args_cli.run_tag):
        raise ValueError("run_tag may contain only letters, digits, '-' and '_'")
    windows = sorted({float(v) for v in args_cli.windows_ms.split(",")})
    if not windows or any(v <= 0 for v in windows):
        raise ValueError("windows_ms must contain positive values")

    checkpoint = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    a = SimpleNamespace(**checkpoint["args"])
    if a.objective != "event_prefix":
        raise ValueError("checkpoint objective must be event_prefix")
    torch.manual_seed(a.seed)
    subset_rng = np.random.default_rng(a.seed)
    augment_rng = np.random.default_rng(a.seed + 500_009)

    data = [(*events(t, u, a.merge), y)
            for t, u, y in S.utterances("train", a.bands, "fit_spk") if len(t) > 1]
    examples = stratified_limit(data, a.limit, subset_rng)
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
    net.eval()

    pair_counts = {w: np.zeros(a.depth, dtype=np.int64) for w in windows}
    batches_with_pairs = {w: np.zeros(a.depth, dtype=np.int64) for w in windows}
    near_closed_routes = np.zeros(a.depth, dtype=np.int64)
    active_source_events = np.zeros(a.depth, dtype=np.int64)
    batch_count = 0

    with torch.no_grad():
        for start in range(0, len(examples), a.bs):
            items = examples[start:start + a.bs]
            eb, ei, et, input_counts, _, tmax, _ = batch_to_events(
                items, a.bands, a.shift, augment_rng, a.drop)
            batch_size = len(items)
            grid = tmax + (a.depth + 1) * int(np.ceil(a.dmax)) + 60
            _, info = net(eb, ei, et, batch_size, grid, return_taps=True,
                          collect_routes=True, input_counts=input_counts)
            batch_count += 1
            for layer, route_info in enumerate(info["route_candidates"]):
                scores = route_info["score"].cpu().numpy()
                near_closed_routes[layer] += int(np.count_nonzero(
                    (scores < 0.0) & (scores >= -a.cf_band)))
                active_source_events[layer] += len(np.unique(
                    route_info["event_index"].cpu().numpy()))
                for window in windows:
                    n_pairs, _, _ = nearby_closed_route_pairs(
                        [route_info], a.cf_band, window,
                        np.random.default_rng(0), 1, "global")
                    pair_counts[window][layer] += n_pairs
                    batches_with_pairs[window][layer] += int(n_pairs > 0)

    result = {
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "run_tag": args_cli.run_tag,
        "data": "same stratified fit-speaker subset used by training; frozen weights",
        "examples": len(examples),
        "batch_size": a.bs,
        "batches": batch_count,
        "seed": a.seed,
        "closed_route_band": float(a.cf_band),
        "windows_ms": windows,
        "near_closed_routes_by_layer": near_closed_routes.tolist(),
        "source_events_with_candidate_routes_by_layer": active_source_events.tolist(),
        "pair_candidates_by_window_layer": {
            str(w): pair_counts[w].tolist() for w in windows
        },
        "batches_with_pair_by_window_layer": {
            str(w): batches_with_pairs[w].tolist() for w in windows
        },
        "interpretation_note": (
            "Candidate counts only; widening W increases proposal availability but does not measure pair utility. "
            "A wide window may combine arrivals that do not interact under the receiver's actual integration kernel."
        ),
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"route_pair_occupancy_{args_cli.run_tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({
        "output": path,
        "examples": len(examples),
        "near_closed_routes_by_layer": result["near_closed_routes_by_layer"],
        "pair_candidates_by_window_layer": result["pair_candidates_by_window_layer"],
        "batches_with_pair_by_window_layer": result["batches_with_pair_by_window_layer"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
