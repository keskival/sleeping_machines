"""Check jump-triggered emission semantics, then quantify them on frozen SHD.

This is a kernel diagnostic, not a recognition benchmark. Run via run_safe.sh.
"""
import argparse
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from e74_time_vector_net import TVLayer
from e83_deep_shd import DeepSHD, batch_to_events, stratified_limit
from e71_event_cde import events
import e51_shd_world as S


def impulse_witness(mode):
    layer = TVLayer(1, 1, 1, 1, 1, 50, gate=0, spike_reconstruction=mode)
    with torch.no_grad():
        layer.log_rate.fill_(math.log(1e-6))
        layer.freq.zero_(); layer.Bre.fill_(1); layer.Bim.zero_()
        layer.wre.fill_(1); layer.wim.zero_()
        layer.Cre.fill_(1); layer.Cim.zero_(); layer.emb.zero_()
        layer.q.zero_(); layer.c.zero_()
    value = torch.tensor([[2.0]], requires_grad=True)
    out, _, info = layer(torch.tensor([0]), torch.tensor([0]), torch.tensor([0.5]),
                         value, 1, 2, return_spike_diagnostics=True)
    grad, = torch.autograd.grad(out[3].sum(), value)
    ref = torch.tensor(2.0 * math.exp(-1e-6 * 0.5), requires_grad=True)
    reference = F.gelu(ref)
    reference_grad, = torch.autograd.grad(reference, ref)
    return {"mode": mode, "events": len(out[2]), "time": out[2].tolist(),
            "payload": out[3].tolist(), "payload_derivative": grad.tolist(),
            "post_arrival_grid_payload": float(reference.detach()),
            "post_arrival_grid_derivative": float(reference_grad * math.exp(-1e-6 * 0.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--examples", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    witness = [impulse_witness(mode) for mode in ("legacy", "grid")]
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    a = SimpleNamespace(**ckpt["args"])
    torch.manual_seed(a.seed)
    net = DeepSHD(a.bands, a.d, a.n, a.M1, a.M, a.depth, a.window, a.fan2,
                  a.readout_fan, a.dmax, [float(v) for v in a.w_sd.split(",")],
                  a.seed, event_readout=True, readout_fusion=a.readout_fusion,
                  input_count_payload=a.input_count_payload == "additive",
                  early_event_skip=getattr(a, "early_event_skip", False),
                  route_topk=getattr(a, "route_topk", 0),
                  spike_reconstruction=getattr(a, "spike_reconstruction", "legacy"))
    net.load_state_dict(ckpt["model_state_dict"]); net.eval()
    rng = np.random.default_rng(a.seed + 300_007)
    data = [(*events(t, u, a.merge), y)
            for t, u, y in S.utterances("train", a.bands, "val_spk") if len(t) > 1]
    samples = stratified_limit(data, args.examples, rng)
    rows = [{"layer": k + 1, "events": 0, "same_bin_arrival": 0,
             "zero_prior_state": 0, "payload_changed": 0,
             "outside_detection_interval": 0, "payload_difference_sum": 0.0}
            for k in range(a.depth)]
    with torch.no_grad():
        for start in range(0, len(samples), args.batch_size):
            items = samples[start:start + args.batch_size]
            eb, ei, et, counts, labels, tmax, end = batch_to_events(items, a.bands, 0, rng, 0.0)
            grid = tmax + (a.depth + 1) * math.ceil(a.dmax) + 60
            _, info = net(eb, ei, et, len(items), grid, return_taps=True,
                          return_spike_diagnostics=True, input_counts=counts)
            for row, diag in zip(rows, info["spike_diagnostics"]):
                d = diag["emission_audit"]
                row["events"] += len(d["prior_state_norm"])
                row["same_bin_arrival"] += int((d["arrival_jump_norm"] > 1e-8).sum())
                row["zero_prior_state"] += int((d["prior_state_norm"] == 0).sum())
                row["payload_changed"] += int((d["payload_grid_difference_norm"] > 1e-6).sum())
                delta = d["time_minus_detection_grid"]
                row["outside_detection_interval"] += int(((delta > 1e-6) | (delta < -1)).sum())
                row["payload_difference_sum"] += float(d["payload_grid_difference_norm"].sum())
    result = {"checkpoint": args.checkpoint, "examples": len(samples), "split": "train/val_spk",
              "witness": witness, "per_layer": rows,
              "scope": "Frozen legacy checkpoint; grid comparison includes time and reset discretization changes. No accuracy claim."}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
