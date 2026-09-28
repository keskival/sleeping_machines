"""E83: depth and gradient-flow study for Sleeping Machines on spoken digits.

All layers are event layers. Layer 1 uses local tonotopic wiring; each later
layer receives sparse messages only from the immediately preceding layer. A
shared auxiliary readout is trained at each depth, while inference uses only
the deepest readout. This supplies local learning signals without allowing the
prediction to bypass the event hierarchy.

This isolates the fixed-topology depth question from the separate speaker
equivariance question in E75. It reports per-layer gradient norms, candidate
score pairs, accepted messages, state scans, and spikes; the small pilot is not
a benchmark claim.

The classifier is utterance-to-class with an anytime output race. Before any
class is sufficiently likely it emits nothing; the first class whose softmax
probability crosses a confidence threshold wins. Training maximizes the
probability that the correct class wins a differentiable competing-hazard
relaxation of that race. Integral-potential and max-over-time readouts remain
available as sequence-classification controls.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
import e51_shd_world as S  # noqa: E402
from e71_event_cde import events  # noqa: E402
from e74_time_vector_net import TVLayer, to_events  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e83")


def pool_readout(V, seq_end, delay_stages, dmax, mode):
    """Turn a (time, batch, class) potential trace into one class-score vector.

    Each item is pooled only through its own last-input-time + maximum path
    delay + settling interval. This avoids letting a longer co-batched example
    alter a shorter item's prediction. `integral` is cross-entropy on summed
    potentials; `max` is cross-entropy on each class's maximum potential.
    """
    times = torch.arange(V.shape[0], device=V.device, dtype=seq_end.dtype)
    ends = seq_end.to(device=V.device)
    cutoff = ends + delay_stages * dmax + 60.0
    valid = times[:, None] <= cutoff[None, :]
    if mode == "integral":
        # V is sampled every millisecond. Use the 1 ms Riemann-sum factor so
        # changing grid resolution does not arbitrarily rescale CE logits.
        return (V * valid[:, :, None]).sum(0) * 1e-3
    if mode == "max":
        return V.masked_fill(~valid[:, :, None], -torch.inf).amax(0)
    raise ValueError(f"unknown readout pooling mode: {mode}")


def race_nll(V, seq_end, delay_stages, dmax, labels, threshold, gate_temperature,
             logit_temperature, rate_scale, latency_discount):
    """Negative log discounted probability that `labels` wins the confidence race.

    Class-specific hazards rise smoothly as that class's instantaneous
    probability passes `threshold`. The survival term means no class has
    emitted yet; the cause term scores the first emitted class. Thus the label
    is attached to the eventual winner, not copied onto every input prefix.
    Time bins are 1 ms, matching the SHD simulation grid.
    """
    times = torch.arange(V.shape[0], device=V.device, dtype=seq_end.dtype)
    cutoff = seq_end.to(V.device) + delay_stages * dmax + 60.0
    valid = times[:, None] <= cutoff[None, :]
    probs = torch.softmax(V / logit_temperature, dim=-1)
    confidence = probs.amax(-1)
    total_rate = (gate_temperature * torch.nn.functional.softplus(
        (confidence - threshold) / gate_temperature) * rate_scale)
    hazards = total_rate[:, :, None] * probs * valid[:, :, None]
    total = hazards.sum(-1)
    survival_before = torch.cat((total.new_zeros((1, total.shape[1])), -total.cumsum(0)[:-1]), 0)
    event_mass = (-torch.expm1(-total)).clamp_min(1e-30)
    correct_hazard = hazards.gather(2, labels[None, :, None].expand(V.shape[0], -1, 1)).squeeze(-1)
    cause_fraction = correct_hazard / total.clamp_min(1e-30)
    log_discount = -latency_discount * (times[:, None] / 1000.0)
    log_terms = (survival_before + event_mass.log()
                 + cause_fraction.clamp_min(1e-30).log() + log_discount)
    log_terms = log_terms.masked_fill(~valid, -torch.inf)
    log_win_probability = torch.logsumexp(log_terms, dim=0)
    return -log_win_probability.mean()


def race_decision(V, seq_end, delay_stages, dmax, threshold, logit_temperature):
    """Return first threshold-crossing class, emission flag, latency, and peak confidence."""
    times = torch.arange(V.shape[0], device=V.device, dtype=seq_end.dtype)
    cutoff = seq_end.to(V.device) + delay_stages * dmax + 60.0
    valid = times[:, None] <= cutoff[None, :]
    probs = torch.softmax(V / logit_temperature, dim=-1)
    confidence, classes = probs.max(-1)
    crossed = (confidence >= threshold) & valid
    emitted = crossed.any(0)
    first = crossed.to(torch.int64).argmax(0)
    batch = torch.arange(V.shape[1], device=V.device)
    prediction = classes[first, batch]
    prediction = torch.where(emitted, prediction, torch.full_like(prediction, -1))
    latency = first.to(V.dtype)
    peak_conf = confidence.masked_fill(~valid, -torch.inf).amax(0)
    return prediction, emitted, latency, peak_conf


def batch_to_events(items, bands, shift, rng, drop):
    """Pack events while retaining each utterance's own endpoint for causal pooling."""
    eb, ei, et, labels, _ = to_events(items, bands, shift, rng, drop)
    seq_end = torch.tensor([float(t[-1]) * 1000.0 for _, t, _, _ in items], dtype=torch.float32)
    max_time = int(math.ceil(float(seq_end.max()))) + 1
    return eb, ei, et, labels, max_time, seq_end


class DeepSHD(nn.Module):
    def __init__(self, bands, d, n, M1, M, depth, window, fan2, readout_fan,
                 dmax, w_sd, seed=0):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one")
        self.depth = int(depth)
        gen = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(bands, d)
        self.widths = [M1] + [M] * (depth - 1)
        self.layers = nn.ModuleList()
        centers = (torch.arange(M1) + 0.5) * bands / M1
        local = (torch.arange(bands)[:, None] - centers[None]).abs() <= window / 2
        self.layers.append(TVLayer(bands, M1, d, d, n, dmax, local, True, w_sd[0]))
        for i in range(1, depth):
            previous_width = self.widths[i - 1]
            mask = torch.rand(previous_width, M, generator=gen) < fan2
            for j in range(M):
                if not mask[:, j].any():
                    mask[torch.randint(previous_width, (), generator=gen), j] = True
            for i0 in range(previous_width):
                if not mask[i0].any():
                    mask[i0, torch.randint(M, (), generator=gen)] = True
            self.layers.append(TVLayer(previous_width, M, d, d, n, dmax, mask, True, w_sd[1]))
        readout_width = max(self.widths)
        readout_mask = torch.rand(readout_width, 20, generator=gen) < readout_fan
        for j in range(20):
            if not readout_mask[:, j].any():
                readout_mask[torch.randint(readout_width, (), generator=gen), j] = True
        for i0 in range(readout_width):
            if not readout_mask[i0].any():
                readout_mask[i0, torch.randint(20, (), generator=gen)] = True
        self.ro = TVLayer(readout_width, 20, d, d, n, dmax, readout_mask, False, 0.3,
                          gate_bias=1.0, normalize=True)

    @staticmethod
    def scored_pairs(layer, indices):
        if layer.mask is None:
            return len(indices) * layer.M
        return int(layer.mask[indices].sum())

    def forward(self, eb, ei, et, B, G, return_taps=False):
        raw_v = self.emb(ei)
        emitted = []
        messages, spikes, candidates = [], [], []
        for i, layer in enumerate(self.layers):
            if i == 0:
                ib, ij, it, iv = eb, ei, et, raw_v
            else:
                ib, ij, it, iv = emitted[-1]
            candidates.append(self.scored_pairs(layer, ij))
            out, msg = layer(ib, ij, it, iv, B, G)
            emitted.append(out); messages.append(msg); spikes.append(len(out[2]) / B)

        taps = []
        readout_inputs = emitted if return_taps else emitted[-1:]
        for out in readout_inputs:
            candidates.append(self.scored_pairs(self.ro, out[1]))
            V, msg = self.ro(out[0], out[1], out[2], out[3], B, G)
            messages.append(msg)
            taps.append(V)
        scan_updates = G * self.layers[0].n * (sum(self.widths) + len(readout_inputs) * self.ro.M)
        deep_scan_updates = G * self.layers[0].n * (sum(self.widths) + self.ro.M)
        deep_candidates = sum(candidates[:self.depth]) + candidates[-1]
        deep_messages = sum(messages[:self.depth]) + messages[-1]
        return taps[-1], {"msgs": messages, "candidates": candidates,
                          "spikes": spikes, "tap_traces": taps,
                          "state_vector_updates_per_utt": scan_updates,
                          "deep_msgs_per_utt": deep_messages,
                          "deep_candidate_scores_per_utt": deep_candidates,
                          "deep_state_vector_updates_per_utt": deep_scan_updates}


def grad_norm(module):
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in module.parameters()
                         if p.grad is not None))


def route_gradient_probe(final_loss, aux_loss, layers):
    groups = [list(layer.parameters()) for layer in layers]
    params = [p for group in groups for p in group]
    final_grads = torch.autograd.grad(final_loss, params, retain_graph=True, allow_unused=True)
    aux_grads = torch.autograd.grad(aux_loss, params, retain_graph=True, allow_unused=True)
    out = {"final_norms": [], "aux_norms": [], "cosines": []}
    start = 0
    for group in groups:
        stop = start + len(group)
        gf, ga = final_grads[start:stop], aux_grads[start:stop]
        final_sq = sum(float(g.detach().square().sum()) for g in gf if g is not None)
        aux_sq = sum(float(g.detach().square().sum()) for g in ga if g is not None)
        dot = sum(float((f.detach() * a.detach()).sum()) for f, a in zip(gf, ga)
                  if f is not None and a is not None)
        fn, an = math.sqrt(final_sq), math.sqrt(aux_sq)
        out["final_norms"].append(round(fn, 6))
        out["aux_norms"].append(round(an, 6))
        out["cosines"].append(round(dot / (fn * an), 6) if fn and an else None)
        start = stop
    return out


def stratified_limit(data, limit, rng):
    if not limit or limit >= len(data):
        return data
    by_class = {}
    for i, item in enumerate(data):
        by_class.setdefault(int(item[-1]), []).append(i)
    each, extra = divmod(limit, len(by_class))
    chosen = []
    leftovers = []
    for key in sorted(by_class):
        ids = np.asarray(by_class[key])
        rng.shuffle(ids)
        chosen.extend(ids[:each].tolist())
        leftovers.extend(ids[each:].tolist())
    if extra:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:extra])
    rng.shuffle(chosen)
    return [data[i] for i in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=140)
    ap.add_argument("--merge", type=float, default=0.002)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--M1", type=int, default=64)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--fan2", type=float, default=0.25)
    ap.add_argument("--readout_fan", type=float, default=0.5)
    ap.add_argument("--aux_weight", type=float, default=0.2,
                    help="weight per intermediate supervised readout; zero is the no-auxiliary control")
    ap.add_argument("--objective", choices=("race", "integral", "max"), default="race",
                    help="race: first confident class wins; integral/max are sequence-level controls")
    ap.add_argument("--race_threshold", type=float, default=0.6,
                    help="minimum class softmax probability that triggers an output event")
    ap.add_argument("--race_temperature", type=float, default=0.03,
                    help="softness of the train-time output gate")
    ap.add_argument("--race_logit_temperature", type=float, default=2.0,
                    help="softmax temperature shared by train-time race and inference decisions")
    ap.add_argument("--race_rate", type=float, default=5.0,
                    help="per-ms scale of the train-time competing hazards")
    ap.add_argument("--race_latency_discount", type=float, default=1.0,
                    help="per-second discount on correct race outcomes, rewarding earlier answers")
    ap.add_argument("--dmax", type=float, default=50.0)
    ap.add_argument("--w_sd", default="0.05,0.05")
    ap.add_argument("--shift", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--eval_limit", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); t0 = time.time()

    def load(split, part):
        return [(*events(t, u, a.merge), y) for t, u, y in S.utterances(split, a.bands, part) if len(t) > 1]

    tr = stratified_limit(load("train", "fit_spk"), a.limit, rng)
    ev = stratified_limit(load("train", "val_spk"), a.eval_limit, rng)
    widths_sd = [float(v) for v in a.w_sd.split(",")]
    if len(widths_sd) != 2:
        raise ValueError("--w_sd requires two comma-separated values")
    net = DeepSHD(a.bands, a.d, a.n, a.M1, a.M, a.depth, a.window, a.fan2,
                  a.readout_fan, a.dmax, widths_sd, a.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    nb = math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb, pct_start=0.1)
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()),
           "layer_params": [sum(p.numel() for p in l.parameters()) for l in net.layers],
           "train": len(tr), "eval": len(ev), "curve": []}
    print(json.dumps({"train": len(tr), "eval": len(ev), "params": res["params"],
                      "layer_params": res["layer_params"], "load_s": round(time.time() - t0)}), flush=True)
    for ep in range(a.epochs):
        net.train(); tl = 0.0; aux_tl = 0.0; route_probe = None
        perm = rng.permutation(len(tr)); gsum = np.zeros(a.depth)
        for i0 in range(0, len(tr), a.bs):
            items = [tr[j] for j in perm[i0:i0 + a.bs]]
            eb, ei, et, y, tmax, seq_end = batch_to_events(items, a.bands, a.shift, rng, a.drop)
            grid = tmax + (a.depth + 1) * math.ceil(a.dmax) + 60
            _, info = net(eb, ei, et, len(items), grid, return_taps=True)
            traces = info["tap_traces"]
            if a.objective == "race":
                main_loss = race_nll(traces[-1], seq_end, a.depth + 1, a.dmax, y,
                                     a.race_threshold, a.race_temperature,
                                     a.race_logit_temperature, a.race_rate,
                                     a.race_latency_discount)
                aux_losses = [race_nll(trace, seq_end, k + 2, a.dmax, y,
                                       a.race_threshold, a.race_temperature,
                                       a.race_logit_temperature, a.race_rate,
                                       a.race_latency_discount)
                              for k, trace in enumerate(traces[:-1])]
            else:
                main_scores = pool_readout(traces[-1], seq_end, a.depth + 1, a.dmax, a.objective)
                main_loss = nn.functional.cross_entropy(main_scores, y)
                aux_losses = [nn.functional.cross_entropy(
                                  pool_readout(trace, seq_end, k + 2, a.dmax, a.objective), y)
                              for k, trace in enumerate(traces[:-1])]
            aux_loss = sum(aux_losses) if aux_losses else main_loss.new_zeros(())
            if i0 == 0:
                route_probe = route_gradient_probe(main_loss, aux_loss, net.layers)
            loss = main_loss + a.aux_weight * aux_loss
            opt.zero_grad(); loss.backward()
            gsum += np.asarray([grad_norm(l) for l in net.layers])
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tl += float(main_loss.detach()); aux_tl += float(aux_loss.detach())

        net.eval(); max_ok = 0; tap_ok = np.zeros(a.depth)
        race_emitted = race_correct = 0; race_latency_sum = race_peak_confidence_sum = 0.0
        threshold_grid = np.asarray((0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
        grid_emitted = np.zeros(len(threshold_grid), dtype=np.int64)
        grid_correct = np.zeros(len(threshold_grid), dtype=np.int64)
        grid_latency = np.zeros(len(threshold_grid), dtype=np.float64)
        st = {"msgs": np.zeros(a.depth * 2), "candidates": np.zeros(a.depth * 2),
              "spikes": np.zeros(a.depth), "state_vector_updates_per_utt": np.zeros(1),
              "deep_msgs_per_utt": np.zeros(1), "deep_candidate_scores_per_utt": np.zeros(1),
              "deep_state_vector_updates_per_utt": np.zeros(1)}
        for layer in net.layers:
            layer.sent.zero_()
        net.ro.sent.zero_()
        with torch.no_grad():
            for i0 in range(0, len(ev), a.bs):
                items = ev[i0:i0 + a.bs]
                eb, ei, et, y, tmax, seq_end = batch_to_events(items, a.bands, 0, rng, 0.0)
                grid = tmax + (a.depth + 1) * math.ceil(a.dmax) + 60
                _, info = net(eb, ei, et, len(items), grid, return_taps=True)
                traces = info["tap_traces"]
                final_max = pool_readout(traces[-1], seq_end, a.depth + 1, a.dmax, "max")
                max_ok += int((final_max.argmax(1) == y).sum())
                for k, trace in enumerate(traces):
                    tap_scores = pool_readout(trace, seq_end, k + 2, a.dmax, "max")
                    tap_ok[k] += int((tap_scores.argmax(1) == y).sum())
                pred, emitted_mask, latency, peak_confidence = race_decision(
                    traces[-1], seq_end, a.depth + 1, a.dmax,
                    a.race_threshold, a.race_logit_temperature)
                race_emitted += int(emitted_mask.sum())
                race_correct += int(((pred == y) & emitted_mask).sum())
                race_latency_sum += float(latency[emitted_mask].sum())
                race_peak_confidence_sum += float(peak_confidence.sum())
                # Validation-only threshold frontier. The readout trace is
                # computed once; changing the threshold only changes when
                # its first class decision is emitted.
                for ti, threshold in enumerate(threshold_grid):
                    gp, ge, gl, _ = race_decision(
                        traces[-1], seq_end, a.depth + 1, a.dmax,
                        float(threshold), a.race_logit_temperature)
                    grid_emitted[ti] += int(ge.sum())
                    grid_correct[ti] += int(((gp == y) & ge).sum())
                    grid_latency[ti] += float(gl[ge].sum())
                for key in st:
                    st[key] += np.asarray(info[key]) * len(items)
        send = [round(float(l.sent[l.mask].float().mean()), 3) for l in net.layers]
        row = {"epoch": ep + 1, "train_loss": round(tl / nb, 4),
               "shd_max_over_time_acc": round(max_ok / len(ev), 4),
               "race_coverage": round(race_emitted / len(ev), 4),
               "race_acc_when_emitted": round(race_correct / race_emitted, 4) if race_emitted else None,
               "race_mean_latency_ms": round(race_latency_sum / race_emitted, 2) if race_emitted else None,
               "race_mean_peak_confidence": round(race_peak_confidence_sum / len(ev), 4),
               "race_threshold_frontier": [
                   {"threshold": float(threshold),
                    "coverage": round(int(n_emit) / len(ev), 4),
                    "accuracy_when_emitted": round(int(n_ok) / int(n_emit), 4) if n_emit else None,
                    "mean_latency_ms": round(float(lat) / int(n_emit), 2) if n_emit else None}
                   for threshold, n_emit, n_ok, lat in zip(
                       threshold_grid, grid_emitted, grid_correct, grid_latency)],
               "tap_acc": (tap_ok / len(ev)).round(4).tolist(),
               "aux_train_loss": round(aux_tl / nb, 4),
               "first_batch_gradient_routes": route_probe,
               "msgs_per_utt": (st["msgs"] / len(ev)).round(0).tolist(),
               "candidate_scores_per_utt": (st["candidates"] / len(ev)).round(0).tolist(),
               "state_vector_updates_per_utt": int(st["state_vector_updates_per_utt"][0] / len(ev)),
               "deep_msgs_per_utt": round(float(st["deep_msgs_per_utt"][0] / len(ev)), 1),
               "deep_candidate_scores_per_utt": int(st["deep_candidate_scores_per_utt"][0] / len(ev)),
               "deep_state_vector_updates_per_utt": int(st["deep_state_vector_updates_per_utt"][0] / len(ev)),
               "spikes_per_utt": (st["spikes"] / len(ev)).round(0).tolist(),
               "synapses_sending": send,
               "layer_grad_norms": (gsum / nb).round(5).tolist(),
               "wall_s": round(time.time() - t0)}
        res["curve"].append(row); print(json.dumps(row), flush=True)
    path = os.path.join(OUT, f"deep_d{a.d}_n{a.n}_M{a.M1}-{a.M}_depth{a.depth}_aux{a.aux_weight:g}_obj{a.objective}_spk_s{a.seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
