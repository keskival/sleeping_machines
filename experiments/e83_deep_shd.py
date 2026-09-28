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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import e51_shd_world as S  # noqa: E402
from e71_event_cde import events  # noqa: E402
from e74_time_vector_net import TVLayer, to_events  # noqa: E402
from sparse_anytime_readout import SparseAnytimeReadout  # noqa: E402

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
    # The shared gate makes sum_c lambda_c exactly Gamma; avoid recovering
    # this by summing possibly saturated float32 softmax probabilities.
    total = total_rate * valid
    survival_before = torch.cat((total.new_zeros((1, total.shape[1])), -total.cumsum(0)[:-1]), 0)
    event_mass = (-torch.expm1(-total)).clamp_min(1e-30)
    # Since lambda_y / Gamma = p_y, use log_softmax directly. Dividing the
    # hazards and then log(clamp(p_y)) gives zero gradient after underflow.
    log_cause = torch.log_softmax(V / logit_temperature, dim=-1).gather(
        2, labels[None, :, None].expand(V.shape[0], -1, 1)).squeeze(-1)
    log_discount = -latency_discount * (times[:, None] / 1000.0)
    log_terms = survival_before + event_mass.log() + log_cause + log_discount
    log_terms = log_terms.masked_fill(~valid, -torch.inf)
    log_win_probability = torch.logsumexp(log_terms, dim=0)
    return -log_win_probability.mean()


def anytime_nll(V, seq_end, delay_stages, dmax, labels, threshold, gate_temperature,
                logit_temperature, rate_scale, latency_discount):
    """Race likelihood with a forced max-potential answer at the deadline.

    The output is correct if either the right class wins the early race, or no
    race event occurs and the terminal classifier is correct. At zero latency
    discount these disjoint outcomes form a normalized probability of a
    correct final answer, so silence is explicitly penalized without requiring
    an arbitrary extra coverage coefficient.
    """
    times = torch.arange(V.shape[0], device=V.device, dtype=seq_end.dtype)
    ends = seq_end.to(V.device)
    cutoff = ends + delay_stages * dmax + 60.0
    valid = times[:, None] <= cutoff[None, :]
    probs = torch.softmax(V / logit_temperature, dim=-1)
    confidence = probs.amax(-1)
    total_rate = gate_temperature * torch.nn.functional.softplus(
        (confidence - threshold) / gate_temperature) * rate_scale
    hazards = total_rate[:, :, None] * probs * valid[:, :, None]
    total = total_rate * valid
    survival_before = torch.cat((total.new_zeros((1, total.shape[1])), -total.cumsum(0)[:-1]), 0)
    event_mass = (-torch.expm1(-total)).clamp_min(1e-30)
    log_cause = torch.log_softmax(V / logit_temperature, dim=-1).gather(
        2, labels[None, :, None].expand(V.shape[0], -1, 1)).squeeze(-1)
    log_discount = -latency_discount * (times[:, None] / 1000.0)
    log_terms = survival_before + event_mass.log() + log_cause + log_discount
    log_terms = log_terms.masked_fill(~valid, -torch.inf)
    log_early_correct = torch.logsumexp(log_terms, dim=0)

    log_no_early_event = -total.sum(0)
    terminal_scores = pool_readout(V, ends, delay_stages, dmax, "max")
    log_terminal_correct = torch.nn.functional.log_softmax(terminal_scores, dim=-1).gather(
        1, labels[:, None]).squeeze(1)
    log_deadline_discount = -latency_discount * (cutoff / 1000.0)
    log_fallback_correct = log_no_early_event + log_terminal_correct + log_deadline_discount
    log_success = torch.logaddexp(log_early_correct, log_fallback_correct)
    return -log_success.mean()


def race_decision(V, seq_end, delay_stages, dmax, threshold, logit_temperature):
    """Return the first emitted class and its vector-valued probability payload.

    The payload is the complete class-probability vector at the crossing; its
    argmax decodes the emitted class and its remaining coordinates preserve
    uncertainty for any downstream message consumer.
    """
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
    payload = probs[first, batch]
    payload = torch.where(emitted[:, None], payload, torch.zeros_like(payload))
    latency = first.to(V.dtype)
    peak_conf = confidence.masked_fill(~valid, -torch.inf).amax(0)
    return prediction, emitted, latency, peak_conf, payload


def sampled_prefix_times(eb, et, batch_size, count=4, window_start=0.2, rng=None):
    """Sample a finite set of causal prefixes from a normalized time window.

    One time is drawn from each equal-width stratum of [window_start, 1] in
    normalized utterance time. Evaluation uses stratum midpoints; training
    uses fresh stratified samples. The label supervises the window-average
    proper score, not a single arbitrarily chosen instant.
    """
    out = []
    for b in range(batch_size):
        times = et[eb == b].sort().values
        if not times.numel():
            out.append(times)
            continue
        lo, hi = float(times[0]), float(times[-1])
        start = lo + window_start * (hi - lo)
        edges = np.linspace(start, hi, count + 1)
        if rng is None:
            samples = (edges[:-1] + edges[1:]) * 0.5
        else:
            samples = np.asarray([rng.uniform(edges[i], edges[i + 1]) for i in range(count)])
        out.append(torch.as_tensor(samples, dtype=et.dtype, device=et.device))
    return out


class SparseEventReadout(nn.Module):
    """Sparse vector-to-class evidence updated only when a hidden event arrives.

    The same additive event updates train the posterior at sampled prefixes
    and drive the indexed-tree first-crossing rule at inference. Training
    computes the class log-normalizer only at a few sampled prefixes; it does
    not scan classes at every event or simulation tick. An explicit EOS event
    carries the final silent interval.
    """

    def __init__(self, n_units, payload_dim, n_classes, fanout, seed):
        super().__init__()
        self.n_units, self.payload_dim, self.n_classes = n_units, payload_dim, n_classes
        gen = torch.Generator().manual_seed(seed)
        mask = torch.rand(n_units + 1, n_classes, generator=gen) < fanout
        for j in range(n_units):
            if not bool(mask[j].any()):
                mask[j, torch.randint(n_classes, (), generator=gen)] = True
        for c in range(n_classes):
            if not bool(mask[:n_units, c].any()):
                mask[torch.randint(n_units, (), generator=gen), c] = True
        # One scheduled EOS update may touch all classes to encode trailing silence.
        mask[n_units] = True
        row_ptr = torch.cat((torch.zeros(1, dtype=torch.long),
                             mask.sum(1).to(torch.long).cumsum(0)))
        edge_class = mask.nonzero(as_tuple=True)[1]
        self.register_buffer("row_ptr", row_ptr)
        self.register_buffer("edge_class", edge_class)
        self.weight = nn.Parameter(torch.randn(len(edge_class), payload_dim + 3) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def _edge_values(self, units, times, payload, deadline):
        """Return sparse (time, class, logit-increment) records."""
        device = payload.device
        if times.numel():
            times, order = times.sort(stable=True)
            units, payload = units[order], payload[order]
            unique_t, inverse = torch.unique_consecutive(times, return_inverse=True)
            prev_t = torch.cat((unique_t.new_zeros(1), unique_t[:-1]))
            gaps = (unique_t - prev_t).clamp_min(0)[inverse]
            features = torch.cat((payload, torch.log1p(gaps[:, None] / 10.0),
                                  torch.log1p(times[:, None] / 100.0),
                                  payload.new_zeros((len(times), 1))), dim=1)
            last_t = times[-1]
        else:
            times = payload.new_empty((0,))
            units = torch.empty((0,), dtype=torch.long, device=device)
            features = payload.new_empty((0, self.payload_dim + 3))
            last_t = payload.new_zeros(())

        end_t = torch.as_tensor(deadline, dtype=payload.dtype, device=device).reshape(())
        end_t = torch.maximum(end_t, last_t)
        end_gap = (end_t - last_t).clamp_min(0)
        eos_feature = torch.cat((payload.new_zeros((1, self.payload_dim)),
                                 torch.log1p(end_gap.reshape(1, 1) / 10.0),
                                 torch.log1p(end_t.reshape(1, 1) / 100.0),
                                 payload.new_ones((1, 1))), dim=1)
        all_units = torch.cat((units, torch.tensor([self.n_units], dtype=torch.long, device=device)))
        all_times = torch.cat((times, end_t.reshape(1)))
        all_features = torch.cat((features, eos_feature), dim=0)

        counts = self.row_ptr[all_units + 1] - self.row_ptr[all_units]
        event_ids = torch.repeat_interleave(torch.arange(len(all_units), device=device), counts)
        starts = torch.repeat_interleave(self.row_ptr[all_units], counts)
        offsets = torch.arange(len(event_ids), device=device) - torch.repeat_interleave(
            counts.cumsum(0) - counts, counts)
        edge_ids = starts + offsets
        classes = self.edge_class[edge_ids]
        values = (all_features[event_ids] * self.weight[edge_ids]).sum(-1)
        return all_times[event_ids], classes, values

    def logits_at(self, units, times, payload, query_times, deadline):
        """Accumulate the identical sparse event updates at selected prefixes."""
        edge_times, classes, values = self._edge_values(units, times, payload, deadline)
        qs = torch.cat((query_times, torch.as_tensor(deadline, dtype=times.dtype,
                                                       device=times.device).reshape(1)))
        rows = []
        for q in qs:
            active = edge_times <= q
            rows.append(self.bias.index_add(0, classes[active], values[active]))
        return torch.stack(rows)

    def infer(self, units, times, payload, deadline, threshold, temperature):
        """Use the same sparse updates with indexed trees at every event time."""
        edge_times, classes, values = self._edge_values(units, times, payload, deadline)
        order = edge_times.argsort(stable=True)
        edge_times, classes, values = edge_times[order], classes[order], values[order]
        state = SparseAnytimeReadout(
            self.n_classes, threshold, temperature,
            initial_logits=self.bias.detach().cpu().tolist())
        peak = state.confidence()
        if edge_times.numel():
            group_times, group_counts = torch.unique_consecutive(edge_times, return_counts=True)
            t_cpu = group_times.detach().cpu().tolist()
            counts_cpu = group_counts.detach().cpu().tolist()
            class_cpu = classes.detach().cpu().tolist()
            value_cpu = values.detach().cpu().tolist()
            cursor = 0
            for event_time, count in zip(t_cpu, counts_cpu):
                stop = cursor + count
                updates = [(class_cpu[j], value_cpu[j]) for j in range(cursor, stop)]
                decision = state.update(event_time, updates)
                peak = max(peak, state.confidence())
                if decision is not None:
                    return decision, peak
                cursor = stop
        decision = state.finish(float(deadline))
        return decision, max(peak, max(decision.posterior))

    def prefix_loss(self, layer_events, eb, et, labels, seq_end, delay_stages, dmax,
                    prefix_samples=4, prefix_window_start=0.2, rng=None):
        event_b, units, times, payload = layer_events
        prefixes = sampled_prefix_times(eb, et, len(labels), prefix_samples,
                                        prefix_window_start, rng)
        per_item = []
        for b, query_times in enumerate(prefixes):
            select = event_b == b
            deadline = seq_end[b] + delay_stages * dmax + 60.0
            logits = self.logits_at(units[select], times[select], payload[select],
                                    query_times, deadline)
            per_item.append(F.cross_entropy(logits, labels[b].expand(len(logits))))
        return torch.stack(per_item).mean()


def batch_to_events(items, bands, shift, rng, drop):
    """Pack events while retaining each utterance's own endpoint for causal pooling."""
    eb, ei, et, labels, _ = to_events(items, bands, shift, rng, drop)
    seq_end = torch.tensor([float(t[-1]) * 1000.0 for _, t, _, _ in items], dtype=torch.float32)
    max_time = int(math.ceil(float(seq_end.max()))) + 1
    return eb, ei, et, labels, max_time, seq_end


class DeepSHD(nn.Module):
    def __init__(self, bands, d, n, M1, M, depth, window, fan2, readout_fan,
                 dmax, w_sd, seed=0, event_readout=False):
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
        self.event_readout = bool(event_readout)
        if self.event_readout:
            self.event_heads = nn.ModuleList([
                SparseEventReadout(width, d, 20, readout_fan, seed + 1000 + i)
                for i, width in enumerate(self.widths)])
            self.ro = None
        else:
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
            self.event_heads = None

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
        if self.event_readout:
            heads = self.event_heads if return_taps else self.event_heads[-1:]
            for out, head in zip(readout_inputs, heads):
                updates = int((head.row_ptr[out[1] + 1] - head.row_ptr[out[1]]).sum())
                candidates.append(updates); messages.append(updates / B)
            scan_updates = G * self.layers[0].n * sum(self.widths)
            deep_scan_updates = scan_updates
        else:
            for out in readout_inputs:
                candidates.append(self.scored_pairs(self.ro, out[1]))
                V, msg = self.ro(out[0], out[1], out[2], out[3], B, G)
                messages.append(msg)
                taps.append(V)
            scan_updates = G * self.layers[0].n * (sum(self.widths) + len(readout_inputs) * self.ro.M)
            deep_scan_updates = G * self.layers[0].n * (sum(self.widths) + self.ro.M)
        deep_candidates = sum(candidates[:self.depth]) + candidates[-1]
        deep_messages = sum(messages[:self.depth]) + messages[-1]
        layer_events = [(out[0], out[1], out[2], out[3]) for out in emitted]
        return (taps[-1] if taps else None), {"msgs": messages, "candidates": candidates,
                          "spikes": spikes, "tap_traces": taps,
                          "layer_events": layer_events,
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
    ap.add_argument("--objective", choices=("race", "anytime", "integral", "max", "event_prefix"), default="race",
                    help="event_prefix: sparse event-updated logits trained with sampled-prefix proper log loss")
    ap.add_argument("--prefix_samples", type=int, default=4,
                    help="stratified prefix samples used to estimate the window-averaged proper score")
    ap.add_argument("--prefix_window_start", type=float, default=0.2,
                    help="start of the normalized supervision window, in [0, 1)")
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
    ap.add_argument("--save_checkpoint", action="store_true",
                    help="save final model weights beside the result JSON")
    a = ap.parse_args()
    if a.prefix_samples < 1 or not (0.0 <= a.prefix_window_start < 1.0):
        raise ValueError("prefix_samples must be positive and prefix_window_start must be in [0, 1)")
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); t0 = time.time()

    def load(split, part):
        return [(*events(t, u, a.merge), y) for t, u, y in S.utterances(split, a.bands, part) if len(t) > 1]

    tr = stratified_limit(load("train", "fit_spk"), a.limit, rng)
    ev = stratified_limit(load("train", "val_spk"), a.eval_limit, rng)
    widths_sd = [float(v) for v in a.w_sd.split(",")]
    if len(widths_sd) != 2:
        raise ValueError("--w_sd requires two comma-separated values")
    event_readout = a.objective == "event_prefix"
    net = DeepSHD(a.bands, a.d, a.n, a.M1, a.M, a.depth, a.window, a.fan2,
                  a.readout_fan, a.dmax, widths_sd, a.seed, event_readout=event_readout)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    nb = math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb, pct_start=0.1)
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()),
           "layer_params": [sum(p.numel() for p in l.parameters()) for l in net.layers],
           "readout_params": sum(p.numel() for head in net.event_heads for p in head.parameters()) if event_readout else
                             sum(p.numel() for p in net.ro.parameters()),
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
            if event_readout:
                head_losses = [head.prefix_loss(info["layer_events"][k], eb, et, y,
                                                seq_end, k + 1, a.dmax,
                                                a.prefix_samples, a.prefix_window_start, rng)
                               for k, head in enumerate(net.event_heads)]
                main_loss = head_losses[-1]
                aux_losses = head_losses[:-1]
            elif a.objective in ("race", "anytime"):
                objective_fn = race_nll if a.objective == "race" else anytime_nll
                main_loss = objective_fn(traces[-1], seq_end, a.depth + 1, a.dmax, y,
                                         a.race_threshold, a.race_temperature,
                                         a.race_logit_temperature, a.race_rate,
                                         a.race_latency_discount)
                aux_losses = [objective_fn(trace, seq_end, k + 2, a.dmax, y,
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
        race_payload_decode_correct = 0
        anytime_correct = fallback_count = anytime_payload_decode_correct = 0
        anytime_latency_sum = 0.0
        threshold_grid = np.asarray((0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
        if event_readout:
            threshold_grid = np.unique(np.append(threshold_grid, a.race_threshold))
        grid_emitted = np.zeros(len(threshold_grid), dtype=np.int64)
        grid_correct = np.zeros(len(threshold_grid), dtype=np.int64)
        grid_latency = np.zeros(len(threshold_grid), dtype=np.float64)
        grid_anytime_correct = np.zeros(len(threshold_grid), dtype=np.int64)
        grid_fallback = np.zeros(len(threshold_grid), dtype=np.int64)
        grid_anytime_latency = np.zeros(len(threshold_grid), dtype=np.float64)
        epoch_output_payloads = []
        epoch_firing_rasters = []
        prefix_nll_sum = np.zeros(a.prefix_samples, dtype=np.float64)
        prefix_correct = np.zeros(a.prefix_samples, dtype=np.int64)
        prefix_count = 0
        st = {"msgs": np.zeros(a.depth * 2), "candidates": np.zeros(a.depth * 2),
              "spikes": np.zeros(a.depth), "state_vector_updates_per_utt": np.zeros(1),
              "deep_msgs_per_utt": np.zeros(1), "deep_candidate_scores_per_utt": np.zeros(1),
              "deep_state_vector_updates_per_utt": np.zeros(1)}
        for layer in net.layers:
            layer.sent.zero_()
        if not event_readout:
            net.ro.sent.zero_()
        with torch.no_grad():
            for i0 in range(0, len(ev), a.bs):
                items = ev[i0:i0 + a.bs]
                eb, ei, et, y, tmax, seq_end = batch_to_events(items, a.bands, 0, rng, 0.0)
                grid = tmax + (a.depth + 1) * math.ceil(a.dmax) + 60
                _, info = net(eb, ei, et, len(items), grid, return_taps=True)
                if event_readout:
                    queries = sampled_prefix_times(eb, et, len(items),
                                                   a.prefix_samples, a.prefix_window_start)
                    head_logits = []
                    for k, head in enumerate(net.event_heads):
                        layer_rows = []
                        event_b, units, times, payload = info["layer_events"][k]
                        for bi, query_times in enumerate(queries):
                            select = event_b == bi
                            deadline_k = seq_end[bi] + (k + 1) * a.dmax + 60.0
                            layer_rows.append(head.logits_at(
                                units[select], times[select], payload[select],
                                query_times, deadline_k))
                        head_logits.append(layer_rows)

                    final_rows = head_logits[-1]
                    final_scores = torch.stack([rows[-1] for rows in final_rows])
                    terminal_pred = final_scores.argmax(-1)
                    max_ok += int((terminal_pred == y).sum())
                    for k, layer_rows in enumerate(head_logits):
                        tap_ok[k] += sum(int((rows[-1].argmax() == y[bi]))
                                         for bi, rows in enumerate(layer_rows))
                    for bi, rows in enumerate(final_rows):
                        prefix_scores = rows[:-1]
                        prefix_targets = y[bi].expand(len(prefix_scores))
                        prefix_losses = F.cross_entropy(prefix_scores, prefix_targets, reduction="none")
                        prefix_nll_sum += prefix_losses.cpu().numpy()
                        prefix_correct += (prefix_scores.argmax(-1) == prefix_targets).cpu().numpy()
                        prefix_count += 1

                    final_events = info["layer_events"][-1]
                    event_b, final_units, final_times, final_vectors = final_events
                    final_head = net.event_heads[-1]
                    for bi in range(len(items)):
                        select = event_b == bi
                        deadline = float(seq_end[bi] + (a.depth + 1) * a.dmax + 60.0)
                        for ti, threshold in enumerate(threshold_grid):
                            decision, peak = final_head.infer(
                                final_units[select], final_times[select], final_vectors[select],
                                deadline, float(threshold), a.race_logit_temperature)
                            early = decision.reason == "threshold" and decision.time < deadline
                            correct = int(decision.class_id == int(y[bi]))
                            grid_emitted[ti] += int(early)
                            grid_correct[ti] += int(early and correct)
                            grid_latency[ti] += float(decision.time) if early else 0.0
                            grid_anytime_correct[ti] += correct
                            grid_fallback[ti] += int(not early)
                            grid_anytime_latency[ti] += float(decision.time)
                            if abs(float(threshold) - a.race_threshold) < 1e-9:
                                race_emitted += int(early)
                                race_correct += int(early and correct)
                                race_latency_sum += float(decision.time) if early else 0.0
                                race_peak_confidence_sum += peak
                                fallback_count += int(not early)
                                anytime_correct += correct
                                anytime_latency_sum += float(decision.time)
                                race_payload_decode_correct += int(
                                    max(range(len(decision.posterior)), key=decision.posterior.__getitem__)
                                    == decision.class_id)
                                anytime_payload_decode_correct += int(
                                    max(range(len(decision.posterior)), key=decision.posterior.__getitem__)
                                    == decision.class_id)
                                epoch_output_payloads.append({
                                    "true_class": int(y[bi]),
                                    "predicted_class": int(decision.class_id),
                                    "emission": "event_race" if early else "terminal_eos",
                                    "latency_ms": round(float(decision.time), 3),
                                    "value_vector": [round(float(v), 6) for v in decision.posterior],
                                })
                    for bi in range(min(4, len(items))):
                        local = event_b == bi
                        epoch_firing_rasters.append({
                            "eval_index": i0 + bi,
                            "true_class": int(y[bi]),
                            "predicted_class": int(final_rows[bi][-1].argmax()),
                            "spikes": [[round(float(t), 3), int(u)]
                                       for t, u in zip(final_times[local], final_units[local])],
                        })
                    for key in st:
                        st[key] += np.asarray(info[key]) * len(items)
                    continue
                traces = info["tap_traces"]
                final_max = pool_readout(traces[-1], seq_end, a.depth + 1, a.dmax, "max")
                max_ok += int((final_max.argmax(1) == y).sum())
                fallback_pred = final_max.argmax(1)
                terminal_payload = torch.softmax(final_max / a.race_logit_temperature, dim=-1)
                deadline = seq_end.to(final_max.device) + (a.depth + 1) * a.dmax + 60.0
                for k, trace in enumerate(traces):
                    tap_scores = pool_readout(trace, seq_end, k + 2, a.dmax, "max")
                    tap_ok[k] += int((tap_scores.argmax(1) == y).sum())
                pred, emitted_mask, latency, peak_confidence, payload = race_decision(
                    traces[-1], seq_end, a.depth + 1, a.dmax,
                    a.race_threshold, a.race_logit_temperature)
                race_emitted += int(emitted_mask.sum())
                race_correct += int(((pred == y) & emitted_mask).sum())
                race_payload_decode_correct += int(((payload.argmax(-1) == pred) & emitted_mask).sum())
                race_latency_sum += float(latency[emitted_mask].sum())
                race_peak_confidence_sum += float(peak_confidence.sum())
                fallback_mask = ~emitted_mask
                anytime_pred = torch.where(emitted_mask, pred, fallback_pred)
                anytime_payload = torch.where(emitted_mask[:, None], payload, terminal_payload)
                anytime_correct += int((anytime_pred == y).sum())
                fallback_count += int(fallback_mask.sum())
                anytime_payload_decode_correct += int((anytime_payload.argmax(-1) == anytime_pred).sum())
                anytime_latency_sum += float(torch.where(emitted_mask, latency, deadline).sum())
                for bi in range(len(items)):
                    epoch_output_payloads.append({
                        "true_class": int(y[bi]),
                        "predicted_class": int(anytime_pred[bi]),
                        "emission": "race" if bool(emitted_mask[bi]) else "terminal_fallback",
                        "latency_ms": round(float(latency[bi] if emitted_mask[bi] else deadline[bi]), 3),
                        "value_vector": [round(float(v), 6) for v in anytime_payload[bi]],
                    })
                final_batch, final_units, final_times, _ = info["layer_events"][-1]
                for bi in range(min(4, len(items))):
                    local = final_batch == bi
                    epoch_firing_rasters.append({
                        "eval_index": i0 + bi,
                        "true_class": int(y[bi]),
                        "predicted_class": int(anytime_pred[bi]),
                        "spikes": [[round(float(t), 3), int(u)]
                                   for t, u in zip(final_times[local], final_units[local])],
                    })
                # Validation-only threshold frontier. The readout trace is
                # computed once; changing the threshold only changes when
                # its first class decision is emitted.
                for ti, threshold in enumerate(threshold_grid):
                    gp, ge, gl, _, _ = race_decision(
                        traces[-1], seq_end, a.depth + 1, a.dmax,
                        float(threshold), a.race_logit_temperature)
                    grid_emitted[ti] += int(ge.sum())
                    grid_correct[ti] += int(((gp == y) & ge).sum())
                    grid_latency[ti] += float(gl[ge].sum())
                    gfallback = ~ge
                    gpred = torch.where(ge, gp, fallback_pred)
                    grid_anytime_correct[ti] += int((gpred == y).sum())
                    grid_fallback[ti] += int(gfallback.sum())
                    grid_anytime_latency[ti] += float(torch.where(ge, gl, deadline).sum())
                for key in st:
                    st[key] += np.asarray(info[key]) * len(items)
        send = [round(float(l.sent[l.mask].float().mean()), 3) for l in net.layers]
        row = {"epoch": ep + 1, "train_loss": round(tl / nb, 4),
               "race_coverage": round(race_emitted / len(ev), 4),
               "race_acc_when_emitted": round(race_correct / race_emitted, 4) if race_emitted else None,
               "race_payload_dim": 20 if event_readout else int(traces[-1].shape[-1]),
               "race_payload_decode_agreement": round(
                   race_payload_decode_correct / race_emitted, 4) if race_emitted else None,
               "race_mean_latency_ms": round(race_latency_sum / race_emitted, 2) if race_emitted else None,
               "race_mean_peak_confidence": round(race_peak_confidence_sum / len(ev), 4),
               "anytime_acc_with_terminal_fallback": round(anytime_correct / len(ev), 4),
               "anytime_fallback_rate": round(fallback_count / len(ev), 4),
               "anytime_mean_response_latency_ms": round(anytime_latency_sum / len(ev), 2),
               "anytime_payload_dim": 20 if event_readout else int(traces[-1].shape[-1]),
               "anytime_payload_decode_agreement": round(anytime_payload_decode_correct / len(ev), 4),
               "race_threshold_frontier": [
                   {"threshold": float(threshold),
                    "coverage": round(int(n_emit) / len(ev), 4),
                    "accuracy_when_emitted": round(int(n_ok) / int(n_emit), 4) if n_emit else None,
                    "mean_latency_ms": round(float(lat) / int(n_emit), 2) if n_emit else None,
                    "anytime_accuracy_with_fallback": round(int(n_any_ok) / len(ev), 4),
                    "fallback_rate": round(int(n_fallback) / len(ev), 4),
                    "mean_response_latency_ms": round(float(n_any_latency) / len(ev), 2)}
                   for threshold, n_emit, n_ok, lat, n_any_ok, n_fallback, n_any_latency in zip(
                       threshold_grid, grid_emitted, grid_correct, grid_latency,
                       grid_anytime_correct, grid_fallback, grid_anytime_latency)],
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
        if event_readout:
            row["event_terminal_accuracy"] = round(max_ok / len(ev), 4)
            row["prefix_window_nll_by_stratum"] = (prefix_nll_sum / max(prefix_count, 1)).round(4).tolist()
            row["prefix_window_accuracy_by_stratum"] = (
                prefix_correct / max(prefix_count, 1)).round(4).tolist()
            row["prefix_window_start"] = a.prefix_window_start
            row["prefix_samples_per_utterance"] = a.prefix_samples
            row["readout_updates_per_utterance"] = int(
                st["deep_msgs_per_utt"][0] / len(ev))
        else:
            row["shd_max_over_time_acc"] = round(max_ok / len(ev), 4)
        res.setdefault("eval_output_payloads", []).append(epoch_output_payloads)
        res.setdefault("final_layer_firing", []).append({
            "epoch": ep + 1, "unit_count": net.widths[-1], "samples": epoch_firing_rasters})
        res["curve"].append(row); print(json.dumps(row), flush=True)
    path = os.path.join(OUT, f"deep_d{a.d}_n{a.n}_M{a.M1}-{a.M}_depth{a.depth}_aux{a.aux_weight:g}_obj{a.objective}_spk_s{a.seed}.json")
    if a.save_checkpoint:
        checkpoint_path = os.path.splitext(path)[0] + ".pt"
        torch.save({"args": vars(a), "model_state_dict": net.state_dict()}, checkpoint_path)
        res["checkpoint"] = checkpoint_path
    with open(path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
