"""Mechanistic audit of closed final-readout routes in E83.

For a small held-out-speaker batch, score the readout's masked event-to-class
candidates, choose closed edges nearest the hard gate, and shadow-open each
one at the gate boundary (zero added delay). Measure the exact max-potential
cross-entropy change and the logistic-noise boundary-gradient contribution.
The ordinary gradient is measured on the same fixed-support forward pass.

This isolates the final readout gate only. It does not establish that hidden
near-miss spikes carry useful credit, nor does it train the model. Run through
the guarded queue because each shadow performs another readout pass.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import e51_shd_world as S
from e71_event_cde import events
from e83_deep_shd import DeepSHD, batch_to_events, pool_readout, stratified_limit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--near-misses", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--merge", type=float, default=0.002)
    ap.add_argument("--bands", type=int, default=140)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--M1", type=int, default=16)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--fan2", type=float, default=0.25)
    ap.add_argument("--readout-fan", type=float, default=0.5)
    ap.add_argument("--dmax", type=float, default=50.0)
    ap.add_argument("--w-sd", default="0.05,0.05")
    a = ap.parse_args()
    if a.samples < 1 or a.near_misses < 1 or a.sigma <= 0:
        raise ValueError("samples, near-misses, and sigma must be positive")

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = [(*events(t, u, a.merge), y)
            for t, u, y in S.utterances("train", a.bands, "val_spk")
            if len(t) > 1]
    items = stratified_limit(data, a.samples, rng)
    eb, ei, et, labels, tmax, seq_end = batch_to_events(
        items, a.bands, 0, rng, 0.0)
    grid = tmax + (a.depth + 1) * math.ceil(a.dmax) + 60
    w_sd = [float(x) for x in a.w_sd.split(",")]
    if len(w_sd) != 2:
        raise ValueError("--w-sd requires two comma-separated values")
    net = DeepSHD(a.bands, a.d, a.n, a.M1, a.M, a.depth, a.window,
                  a.fan2, a.readout_fan, a.dmax, w_sd, a.seed)
    net.eval()

    raw_v = net.emb(ei)
    out = None
    for layer_index, layer in enumerate(net.layers):
        if layer_index == 0:
            ib, ij, it, iv = eb, ei, et, raw_v
        else:
            ib, ij, it, iv = out
        out, _ = layer(ib, ij, it, iv, len(items), grid)
    final_b, final_j, final_t, final_v = out

    V, _, routes = net.ro(final_b, final_j, final_t, final_v,
                          len(items), grid, return_routes=True)
    scores = pool_readout(V, seq_end, a.depth + 1, a.dmax, "max")
    per_item_loss = F.cross_entropy(scores, labels, reduction="none")
    path_q, path_c = torch.autograd.grad(per_item_loss.sum(), (net.ro.q, net.ro.c),
                                         allow_unused=False)
    path_norm = math.sqrt(float(path_q.square().sum() + path_c.square().sum()))

    by_item = [[] for _ in items]
    for e, c, r in zip(routes["event_index"].tolist(), routes["receiver"].tolist(),
                       routes["score"].tolist()):
        if r < 0.0:
            b = int(final_b[e])
            by_item[b].append((float(r), int(e), int(c)))

    grad_q = torch.zeros_like(net.ro.q)
    grad_c = torch.zeros_like(net.ro.c)
    details = []
    total_closed = sum(len(xs) for xs in by_item)
    total_selected = 0
    total_near = 0
    for b, candidates in enumerate(by_item):
        candidates.sort(key=lambda row: row[0], reverse=True)
        total_near += sum(r >= -5.0 * a.sigma for r, _, _ in candidates)
        selected = [row for row in candidates if row[0] >= -5.0 * a.sigma][:a.near_misses]
        total_selected += len(selected)
        for r, event_index, class_id in selected:
            with torch.no_grad():
                cfV, _ = net.ro(final_b, final_j, final_t, final_v,
                                len(items), grid,
                                force_route=(event_index, class_id))
                cf_scores = pool_readout(cfV, seq_end, a.depth + 1,
                                         a.dmax, "max")
                cf_loss = F.cross_entropy(cf_scores, labels, reduction="none")
                delta = float(cf_loss[b] - per_item_loss[b].detach())

            p = 1.0 / (1.0 + math.exp(-r / a.sigma))
            boundary_density = p * (1.0 - p) / a.sigma
            route_gradient = boundary_density * delta
            unit_id = int(final_j[event_index])
            grad_q[class_id] += route_gradient * final_v[event_index].detach()
            grad_c[unit_id, class_id] += route_gradient
            details.append({
                "item": b,
                "label": int(labels[b]),
                "event_index": event_index,
                "sender_unit": unit_id,
                "class_candidate": class_id,
                "route_score": r,
                "noise_on_probability": p,
                "boundary_density": boundary_density,
                "shadow_loss_delta_on_minus_off": delta,
                "boundary_dloss_droute_score": route_gradient,
            })

    boundary_norm = math.sqrt(float(grad_q.square().sum() + grad_c.square().sum()))
    dot = float((grad_q * path_q).sum() + (grad_c * path_c).sum())
    cosine = dot / (boundary_norm * path_norm) if boundary_norm and path_norm else None
    result = {
        "args": vars(a),
        "split": "train/val_spk",
        "sample_count": len(items),
        "params": sum(p.numel() for p in net.parameters()),
        "base_accuracy": float((scores.argmax(-1) == labels).float().mean()),
        "base_per_item_loss": [float(x) for x in per_item_loss.detach()],
        "closed_readout_route_count": total_closed,
        "closed_routes_within_5sigma": total_near,
        "shadowed_near_misses": total_selected,
        "helpful_shadow_fraction": (sum(x["shadow_loss_delta_on_minus_off"] < 0
                                         for x in details) / len(details)
                                    if details else None),
        "pathwise_q_c_gradient_norm": path_norm,
        "counterfactual_q_c_gradient_norm": boundary_norm,
        "counterfactual_to_pathwise_norm_ratio": (boundary_norm / path_norm
                                                  if path_norm else None),
        "counterfactual_vs_pathwise_cosine": cosine,
        "candidates": details,
        "scope": "readout-gate insertions only; no hidden spike shadowing or training",
    }
    out_dir = os.path.join(os.path.dirname(__file__), "results", "e83")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"counterfactual_readout_audit_s{a.seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    summary = {k: result[k] for k in (
        "sample_count", "base_accuracy", "closed_readout_route_count",
        "closed_routes_within_5sigma", "shadowed_near_misses",
        "helpful_shadow_fraction", "pathwise_q_c_gradient_norm",
        "counterfactual_q_c_gradient_norm", "counterfactual_to_pathwise_norm_ratio",
        "counterfactual_vs_pathwise_cosine", "scope")}
    summary["output"] = out_path
    print(json.dumps(summary), flush=True)
    print(out_path, flush=True)


if __name__ == "__main__":
    main()
