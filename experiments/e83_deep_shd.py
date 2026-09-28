"""E83: depth and gradient-flow study for Sleeping Machines on spoken digits.

All hidden layers are event layers. Layer 1 uses local tonotopic wiring; each
later layer receives sparse messages only from the immediately preceding
layer. A shared auxiliary readout is trained at each depth, while inference
uses only the deepest readout. The event-prefix objective applies sampled
causal label losses and can add a bounded counterfactual boundary signal for
route gates that lost their content races.

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


def sampled_prefix_times(batch_size, count=4, horizon_ms=1000.0, window_start=0.0,
                         device=None, dtype=torch.float32, rng=None):
    """Sample causal query times from a fixed, exogenous physical-time window.

    The horizon is fixed across examples; it must not be derived from the
    utterance's last event, which is future information at a prefix. One time
    is drawn from each equal-width stratum. Evaluation uses stratum midpoints;
    training uses fresh stratified samples.
    """
    start = float(window_start) * float(horizon_ms)
    edges = np.linspace(start, float(horizon_ms), count + 1)
    if rng is None:
        samples = (edges[:-1] + edges[1:]) * 0.5
    else:
        samples = np.asarray([rng.uniform(edges[i], edges[i + 1]) for i in range(count)])
    shared = torch.as_tensor(samples, dtype=dtype, device=device)
    return [shared for _ in range(batch_size)]


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
            # Group identity is discrete, but the representative group time
            # stays differentiable. `unique_consecutive` has no time-gradient
            # implementation in PyTorch.
            group_start = torch.ones(len(times), dtype=torch.bool, device=device)
            group_start[1:] = times.detach()[1:] != times.detach()[:-1]
            group_id = group_start.to(torch.long).cumsum(0) - 1
            n_groups = int(group_id[-1]) + 1
            group_sum = times.new_zeros(n_groups).index_add(0, group_id, times)
            group_count = torch.bincount(group_id, minlength=n_groups).to(times.dtype)
            group_time = group_sum / group_count
            group_prev = torch.cat((group_time.new_zeros(1), group_time[:-1]))
            gaps = (group_time - group_prev).clamp_min(0)[group_id]
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

    def prefix_loss(self, layer_events, labels, seq_end, delay_stages, dmax,
                    prefix_times):
        event_b, units, times, payload = layer_events
        per_item = []
        for b, query_times in enumerate(prefix_times):
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

    def forward(self, eb, ei, et, B, G, return_taps=False, collect_routes=False,
                route_override=None):
        raw_v = self.emb(ei)
        emitted = []
        route_candidates, route_inputs = [], []
        messages, spikes, candidates = [], [], []
        for i, layer in enumerate(self.layers):
            if i == 0:
                ib, ij, it, iv = eb, ei, et, raw_v
            else:
                ib, ij, it, iv = emitted[-1]
            candidates.append(self.scored_pairs(layer, ij))
            force_route = drop_route = None
            if route_override is not None and route_override[0] == i:
                _, event_index, receiver, active = route_override
                if active:
                    force_route = (event_index, receiver)
                else:
                    drop_route = (event_index, receiver)
            result = layer(ib, ij, it, iv, B, G, force_route=force_route,
                           drop_route=drop_route, return_routes=collect_routes)
            if collect_routes:
                out, msg, route_info = result
                route_candidates.append(route_info)
                route_inputs.append((ib, ij, iv))
            else:
                out, msg = result
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
        info = {"msgs": messages, "candidates": candidates,
                          "spikes": spikes, "tap_traces": taps,
                          "layer_events": layer_events,
                          "state_vector_updates_per_utt": scan_updates,
                          "deep_msgs_per_utt": deep_messages,
                          "deep_candidate_scores_per_utt": deep_candidates,
                          "deep_state_vector_updates_per_utt": deep_scan_updates}
        if collect_routes:
            info["route_candidates"] = route_candidates
            info["route_inputs"] = route_inputs
        return (taps[-1] if taps else None), info


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
    ap.add_argument("--prefix_horizon_ms", type=float, default=1000.0,
                    help="fixed physical-time horizon shared by all utterances")
    ap.add_argument("--prefix_window_start", type=float, default=0.0,
                    help="start fraction of the fixed prefix horizon, in [0, 1)")
    ap.add_argument("--cf_shadows_per_layer", type=int, default=1,
                    help="near-boundary route toggles shadowed per hidden layer and batch; 0 disables counterfactual credit")
    ap.add_argument("--cf_band", type=float, default=0.5,
                    help="route-score band around zero eligible for counterfactual shadows")
    ap.add_argument("--cf_sigma", type=float, default=0.25,
                    help="logistic gate-noise scale used by the counterfactual boundary derivative")
    ap.add_argument("--cf_weight", type=float, default=1.0,
                    help="scale on the sampled, normalized boundary-gradient signal")
    ap.add_argument("--cf_lr", type=float, default=1e-3,
                    help="separate SGD step size for the clipped local counterfactual update")
    ap.add_argument("--cf_grad_clip", type=float, default=1.0,
                    help="global norm cap for each local counterfactual gradient")
    ap.add_argument("--cf_delta_clip", type=float, default=5.0,
                    help="absolute cap on the shadow loss difference in the local boundary signal")
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
    if (a.prefix_samples < 1 or not (0.0 <= a.prefix_window_start < 1.0)
            or not math.isfinite(a.prefix_horizon_ms) or a.prefix_horizon_ms <= 0.0):
        raise ValueError("prefix_samples and prefix_horizon_ms must be positive; prefix_window_start must be in [0, 1)")
    if (a.cf_shadows_per_layer < 0 or not math.isfinite(a.cf_band) or a.cf_band <= 0.0
            or not math.isfinite(a.cf_sigma) or a.cf_sigma <= 0.0
            or not math.isfinite(a.cf_lr) or a.cf_lr < 0.0
            or not math.isfinite(a.cf_grad_clip) or a.cf_grad_clip <= 0.0
            or not math.isfinite(a.cf_delta_clip) or a.cf_delta_clip <= 0.0):
        raise ValueError("counterfactual count must be nonnegative; band, sigma, loss-difference clip, and gradient clip must be positive; cf_lr must be nonnegative")
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    prefix_rng = np.random.default_rng(a.seed + 100_003)
    counterfactual_rng = np.random.default_rng(a.seed + 200_003)
    t0 = time.time()

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
        cf_eligible_ep = cf_shadow_ep = 0
        cf_abs_delta_ep = cf_signed_delta_ep = 0.0
        cf_delta_sq_ep = cf_boundary_sensitivity_ep = cf_boundary_coeff_abs_ep = 0.0
        cf_helpful_ep = 0
        cf_clipped_ep = cf_local_update_norm_ep = cf_local_update_batches_ep = 0
        cf_raw_grad_norm_ep = cf_path_grad_norm_ep = cf_grad_ratio_ep = cf_grad_cosine_ep = 0.0
        cf_grad_stats_batches_ep = 0
        cf_layer_grad_norm_ep = np.zeros(a.depth)
        cf_layer_path_norm_ep = np.zeros(a.depth)
        cf_layer_grad_ratio_ep = np.zeros(a.depth)
        cf_layer_grad_cosine_ep = np.zeros(a.depth)
        cf_layer_stats_count_ep = np.zeros(a.depth)
        cf_layer_shadow_count_ep = np.zeros(a.depth)
        cf_layer_delta_sum_ep = np.zeros(a.depth)
        cf_layer_delta_sq_ep = np.zeros(a.depth)
        cf_layer_helpful_ep = np.zeros(a.depth)
        cf_layer_boundary_coeff_ep = np.zeros(a.depth)
        perm = rng.permutation(len(tr)); gsum = np.zeros(a.depth)
        for i0 in range(0, len(tr), a.bs):
            items = [tr[j] for j in perm[i0:i0 + a.bs]]
            eb, ei, et, y, tmax, seq_end = batch_to_events(items, a.bands, a.shift, rng, a.drop)
            grid = tmax + (a.depth + 1) * math.ceil(a.dmax) + 60
            prefix_times = None
            if event_readout:
                prefix_times = sampled_prefix_times(len(items), a.prefix_samples,
                                                    a.prefix_horizon_ms, a.prefix_window_start,
                                                    et.device, et.dtype, prefix_rng)
            _, info = net(eb, ei, et, len(items), grid, return_taps=True,
                          collect_routes=event_readout and a.cf_shadows_per_layer > 0)
            traces = info["tap_traces"]
            if event_readout:
                head_losses = [head.prefix_loss(info["layer_events"][k], y,
                                                seq_end, k + 1, a.dmax, prefix_times)
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
            cf_eligible = cf_shadow_count = 0
            cf_abs_delta_sum = 0.0
            cf_clipped_count = 0
            cf_grads = None
            if event_readout and a.cf_shadows_per_layer:
                base_total = (main_loss + a.aux_weight * aux_loss).detach()
                cf_proxy = main_loss.new_zeros(())
                for k, route_info in enumerate(info["route_candidates"]):
                    scores = route_info["score"].cpu().numpy()
                    near = np.flatnonzero(np.abs(scores) <= a.cf_band)
                    cf_eligible += len(near)
                    if not len(near):
                        continue
                    take = min(a.cf_shadows_per_layer, len(near))
                    selected = counterfactual_rng.choice(near, size=take, replace=False)
                    batch_ids, source_units, source_payload = info["route_inputs"][k]
                    route_event_ids = route_info["event_index"]
                    route_receivers = route_info["receiver"]
                    for route_idx in np.atleast_1d(selected):
                        event_index = int(route_event_ids[route_idx])
                        receiver = int(route_receivers[route_idx])
                        score_value = float(scores[route_idx])
                        active = score_value > 0.0
                        with torch.no_grad():
                            _, shadow_info = net(
                                eb, ei, et, len(items), grid, return_taps=True,
                                route_override=(k, event_index, receiver, not active))
                            shadow_heads = [head.prefix_loss(
                                shadow_info["layer_events"][j], y, seq_end, j + 1,
                                a.dmax, prefix_times)
                                for j, head in enumerate(net.event_heads)]
                            shadow_main = shadow_heads[-1]
                            shadow_aux = sum(shadow_heads[:-1]) if len(shadow_heads) > 1 else shadow_main.new_zeros(())
                            shadow_total = shadow_main + a.aux_weight * shadow_aux
                        # L1-L0 is measured by toggling this route through every
                        # downstream layer. This includes changes in later spikes.
                        delta = (base_total - shadow_total) if active else (shadow_total - base_total)
                        delta_value = float(delta)
                        cf_abs_delta_sum += abs(delta_value)
                        cf_delta_sq_ep += delta_value * delta_value
                        cf_signed_delta_ep += delta_value
                        cf_helpful_ep += int(delta_value < 0.0)
                        cf_shadow_count += 1
                        cf_clipped_count += int(abs(delta_value) > a.cf_delta_clip)
                        cf_layer_shadow_count_ep[k] += 1
                        cf_layer_delta_sum_ep[k] += delta_value
                        cf_layer_delta_sq_ep[k] += delta_value * delta_value
                        cf_layer_helpful_ep[k] += int(delta_value < 0.0)

                        source_unit = source_units[event_index]
                        live_score = (net.layers[k].q[receiver] * source_payload[event_index]).sum()
                        live_score = live_score + net.layers[k].c[source_unit, receiver]
                        p_open = torch.sigmoid(live_score.detach() / a.cf_sigma)
                        clipped_delta = float(np.clip(delta_value, -a.cf_delta_clip, a.cf_delta_clip))
                        boundary_derivative = (p_open * (1.0 - p_open) / a.cf_sigma) * clipped_delta
                        cf_boundary_sensitivity_ep += float(
                            p_open * (1.0 - p_open) / a.cf_sigma)
                        cf_boundary_coeff_abs_ep += abs(float(boundary_derivative))
                        cf_layer_boundary_coeff_ep[k] += abs(float(boundary_derivative))
                        # Average sampled boundary terms within each layer.
                        # This deliberately estimates a normalized local
                        # update, not the sum over every candidate in the band.
                        cf_proxy = cf_proxy + (a.cf_weight * boundary_derivative
                                               * (live_score - live_score.detach()) / take)
                cf_grads = None
                if cf_shadow_count and a.cf_lr:
                    model_params = [p for p in net.parameters() if p.requires_grad]
                    cf_grads = torch.autograd.grad(cf_proxy, model_params, retain_graph=True,
                                                   allow_unused=True)
                    cf_sq = sum(float(g.detach().square().sum()) for g in cf_grads if g is not None)
                    cf_norm = math.sqrt(cf_sq)
                    cf_scale = min(1.0, a.cf_grad_clip / max(cf_norm, 1e-12))
                    cf_clipped_ep += cf_clipped_count
                else:
                    model_params = []
                    cf_scale = 1.0
                cf_clipped_ep += 0 if cf_grads is not None else cf_clipped_count
            cf_eligible_ep += cf_eligible
            cf_shadow_ep += cf_shadow_count
            cf_abs_delta_ep += cf_abs_delta_sum
            opt.zero_grad(); loss.backward()
            if cf_grads is not None:
                path_sq = dot = 0.0
                for param, cf_grad in zip(model_params, cf_grads):
                    path_grad = param.grad
                    if path_grad is None:
                        continue
                    path_sq += float(path_grad.detach().square().sum())
                    if cf_grad is not None:
                        dot += float((cf_grad.detach() * path_grad.detach()).sum())
                path_norm = math.sqrt(path_sq)
                cf_raw_grad_norm_ep += cf_norm
                cf_path_grad_norm_ep += path_norm
                cf_grad_ratio_ep += cf_norm / path_norm if path_norm else 0.0
                cf_grad_cosine_ep += dot / (cf_norm * path_norm) if cf_norm and path_norm else 0.0
                cf_grad_stats_batches_ep += 1
                cf_grad_by_id = {id(p): g for p, g in zip(model_params, cf_grads)}
                for li, layer in enumerate(net.layers):
                    layer_cf_sq = layer_path_sq = layer_dot = 0.0
                    for param in layer.parameters():
                        cf_grad = cf_grad_by_id.get(id(param))
                        path_grad = param.grad
                        if path_grad is not None:
                            layer_path_sq += float(path_grad.detach().square().sum())
                        if cf_grad is not None:
                            layer_cf_sq += float(cf_grad.detach().square().sum())
                            if path_grad is not None:
                                layer_dot += float((cf_grad.detach() * path_grad.detach()).sum())
                    layer_cf_norm = math.sqrt(layer_cf_sq)
                    layer_path_norm = math.sqrt(layer_path_sq)
                    cf_layer_grad_norm_ep[li] += layer_cf_norm
                    cf_layer_path_norm_ep[li] += layer_path_norm
                    if layer_path_norm:
                        cf_layer_grad_ratio_ep[li] += layer_cf_norm / layer_path_norm
                    if layer_cf_norm and layer_path_norm:
                        cf_layer_grad_cosine_ep[li] += layer_dot / (layer_cf_norm * layer_path_norm)
                    cf_layer_stats_count_ep[li] += 1
            gsum += np.asarray([grad_norm(l) for l in net.layers])
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            if cf_grads is not None:
                with torch.no_grad():
                    for param, grad in zip(model_params, cf_grads):
                        if grad is not None:
                            param.add_(grad, alpha=-a.cf_lr * cf_scale)
                cf_local_update_norm_ep += a.cf_lr * min(cf_norm, a.cf_grad_clip)
                cf_local_update_batches_ep += 1
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
                    queries = sampled_prefix_times(len(items), a.prefix_samples,
                                                   a.prefix_horizon_ms, a.prefix_window_start,
                                                   et.device, et.dtype)
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
                                race_payload_decode_correct += int(early and
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
            row["prefix_window_start_fraction"] = a.prefix_window_start
            row["prefix_horizon_ms"] = a.prefix_horizon_ms
            row["prefix_samples_per_utterance"] = a.prefix_samples
            row["sparse_readout_edge_updates_per_utterance"] = round(
                float(st["msgs"][-1] / len(ev)), 1)
            row["counterfactual_near_routes_per_epoch"] = int(cf_eligible_ep)
            row["counterfactual_route_shadows_per_epoch"] = int(cf_shadow_ep)
            row["counterfactual_mean_abs_loss_delta"] = round(
                cf_abs_delta_ep / max(cf_shadow_ep, 1), 6)
            row["counterfactual_mean_signed_open_minus_closed_loss"] = round(
                cf_signed_delta_ep / max(cf_shadow_ep, 1), 6)
            row["counterfactual_fraction_opening_improves_loss"] = round(
                cf_helpful_ep / max(cf_shadow_ep, 1), 4)
            row["counterfactual_shadow_delta_std"] = round(math.sqrt(
                max(cf_delta_sq_ep / max(cf_shadow_ep, 1)
                    - (cf_signed_delta_ep / max(cf_shadow_ep, 1)) ** 2, 0.0)), 6)
            row["counterfactual_shadow_delta_signal_to_noise"] = round(
                (cf_signed_delta_ep / max(cf_shadow_ep, 1)) /
                max(math.sqrt(max(cf_delta_sq_ep / max(cf_shadow_ep, 1)
                                  - (cf_signed_delta_ep / max(cf_shadow_ep, 1)) ** 2, 0.0)), 1e-12), 6)
            row["counterfactual_layer_shadow_counts"] = cf_layer_shadow_count_ep.astype(int).tolist()
            row["counterfactual_layer_mean_open_minus_closed_loss"] = [
                round(float(cf_layer_delta_sum_ep[k] / cf_layer_shadow_count_ep[k]), 7)
                if cf_layer_shadow_count_ep[k] else None for k in range(a.depth)]
            row["counterfactual_layer_shadow_delta_std"] = [
                round(float(math.sqrt(max(cf_layer_delta_sq_ep[k] / cf_layer_shadow_count_ep[k]
                                          - (cf_layer_delta_sum_ep[k] / cf_layer_shadow_count_ep[k]) ** 2,
                                          0.0))), 7)
                if cf_layer_shadow_count_ep[k] else None for k in range(a.depth)]
            row["counterfactual_layer_fraction_opening_improves"] = [
                round(float(cf_layer_helpful_ep[k] / cf_layer_shadow_count_ep[k]), 4)
                if cf_layer_shadow_count_ep[k] else None for k in range(a.depth)]
            row["counterfactual_layer_mean_abs_boundary_coefficient"] = [
                round(float(cf_layer_boundary_coeff_ep[k] / cf_layer_shadow_count_ep[k]), 7)
                if cf_layer_shadow_count_ep[k] else None for k in range(a.depth)]
            row["counterfactual_mean_gate_boundary_sensitivity"] = round(
                cf_boundary_sensitivity_ep / max(cf_shadow_ep, 1), 6)
            row["counterfactual_mean_abs_clipped_boundary_coefficient"] = round(
                cf_boundary_coeff_abs_ep / max(cf_shadow_ep, 1), 6)
            row["counterfactual_clipped_delta_fraction"] = round(
                cf_clipped_ep / max(cf_shadow_ep, 1), 4)
            row["counterfactual_local_update_norm_per_batch"] = round(
                cf_local_update_norm_ep / max(cf_local_update_batches_ep, 1), 7)
            row["counterfactual_local_update_batches"] = int(cf_local_update_batches_ep)
            row["counterfactual_raw_grad_norm_per_batch"] = round(
                cf_raw_grad_norm_ep / max(cf_grad_stats_batches_ep, 1), 7)
            row["pathwise_grad_norm_per_counterfactual_batch"] = round(
                cf_path_grad_norm_ep / max(cf_grad_stats_batches_ep, 1), 7)
            row["counterfactual_to_pathwise_grad_norm_ratio"] = round(
                cf_grad_ratio_ep / max(cf_grad_stats_batches_ep, 1), 7)
            row["counterfactual_pathwise_grad_cosine"] = round(
                cf_grad_cosine_ep / max(cf_grad_stats_batches_ep, 1), 6)
            row["counterfactual_layer_grad_norms"] = np.divide(
                cf_layer_grad_norm_ep, np.maximum(cf_layer_stats_count_ep, 1)).round(7).tolist()
            row["pathwise_layer_grad_norms_on_counterfactual_batches"] = np.divide(
                cf_layer_path_norm_ep, np.maximum(cf_layer_stats_count_ep, 1)).round(7).tolist()
            row["counterfactual_layer_to_pathwise_norm_ratios"] = [
                round(float(cf_layer_grad_ratio_ep[k] / cf_layer_stats_count_ep[k]), 7)
                if cf_layer_stats_count_ep[k] else None for k in range(a.depth)]
            row["counterfactual_layer_pathwise_cosines"] = [
                round(float(cf_layer_grad_cosine_ep[k] / cf_layer_stats_count_ep[k]), 6)
                if cf_layer_stats_count_ep[k] else None for k in range(a.depth)]
        else:
            row["shd_max_over_time_acc"] = round(max_ok / len(ev), 4)
        res.setdefault("eval_output_payloads", []).append(epoch_output_payloads)
        res.setdefault("final_layer_firing", []).append({
            "epoch": ep + 1, "unit_count": net.widths[-1], "samples": epoch_firing_rasters})
        res["curve"].append(row); print(json.dumps(row), flush=True)
    cf_tag = (f"_cfnorm{a.cf_shadows_per_layer}_b{a.cf_band:g}_sg{a.cf_sigma:g}"
              f"_w{a.cf_weight:g}_dl{a.cf_delta_clip:g}_lr{a.cf_lr:g}_gc{a.cf_grad_clip:g}"
              if event_readout else "")
    path = os.path.join(OUT, f"deep_d{a.d}_n{a.n}_M{a.M1}-{a.M}_depth{a.depth}_aux{a.aux_weight:g}_obj{a.objective}{cf_tag}_spk_s{a.seed}.json")
    if a.save_checkpoint:
        checkpoint_path = os.path.splitext(path)[0] + ".pt"
        torch.save({"args": vars(a), "model_state_dict": net.state_dict()}, checkpoint_path)
        res["checkpoint"] = checkpoint_path
    with open(path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
