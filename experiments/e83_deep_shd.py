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
from e74_time_vector_net import TVLayer  # noqa: E402
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
        rows = self.prefix_logits(layer_events, seq_end, delay_stages, dmax,
                                  prefix_times)
        losses = [F.cross_entropy(logits, labels[b].expand(len(logits)))
                  for b, logits in enumerate(rows)]
        return torch.stack(losses).mean()

    def prefix_logits(self, layer_events, seq_end, delay_stages, dmax,
                      prefix_times):
        """Return each item's causal prefix logits plus its EOS logits."""
        event_b, units, times, payload = layer_events
        per_item = []
        for b, query_times in enumerate(prefix_times):
            select = event_b == b
            deadline = seq_end[b] + delay_stages * dmax + 60.0
            logits = self.logits_at(units[select], times[select], payload[select],
                                    query_times, deadline)
            per_item.append(logits)
        return per_item


def batch_to_events(items, bands, shift, rng, drop):
    """Pack events, retaining merged spike count and each item's endpoint."""
    eb, ei, et, ec, labels = [], [], [], [], []
    seq_ends = []
    for item, (bands_i, times_i, log_count_i, label) in enumerate(items):
        keep = rng.random(len(bands_i)) >= drop if drop else np.ones(len(bands_i), dtype=bool)
        shift_i = int(rng.integers(-shift, shift + 1)) if shift else 0
        eb.append(np.full(int(keep.sum()), item, dtype=np.int64))
        ei.append(np.clip(bands_i[keep].astype(np.int64) + shift_i, 0, bands - 1))
        et.append(times_i[keep] * 1000.0)
        # Singleton groups map to zero; positive values encode repeated spikes.
        ec.append(log_count_i[keep] - np.float32(math.log(2.0)))
        labels.append(int(label))
        seq_ends.append(float(times_i[-1]) * 1000.0)
    eb = np.concatenate(eb); ei = np.concatenate(ei); et = np.concatenate(et); ec = np.concatenate(ec)
    eb = torch.from_numpy(eb).long(); ei = torch.from_numpy(ei).long()
    et = torch.from_numpy(et).float(); ec = torch.from_numpy(ec).float()
    labels = torch.tensor(labels, dtype=torch.long)
    seq_end = torch.tensor(seq_ends, dtype=torch.float32)
    max_time = int(et.max().item()) + 1
    return eb, ei, et, ec, labels, max_time, seq_end


class DeepSHD(nn.Module):
    def __init__(self, bands, d, n, M1, M, depth, window, fan2, readout_fan,
                 dmax, w_sd, seed=0, event_readout=False,
                 readout_fusion="deepest", input_count_payload=False,
                 early_event_skip=False, route_topk=0,
                 trainable_thresholds=False):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one")
        self.depth = int(depth)
        self.early_event_skip = bool(early_event_skip)
        self.route_topk = int(route_topk)
        self.trainable_thresholds = bool(trainable_thresholds)
        if self.route_topk < 0:
            raise ValueError("route_topk must be nonnegative")
        gen = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(bands, d)
        self.widths = [M1] + [M] * (depth - 1)
        self.layers = nn.ModuleList()
        centers = (torch.arange(M1) + 0.5) * bands / M1
        local = (torch.arange(bands)[:, None] - centers[None]).abs() <= window / 2
        self.layers.append(TVLayer(bands, M1, d, d, n, dmax, local, True, w_sd[0],
                                   route_topk=self.route_topk,
                                   trainable_thresholds=self.trainable_thresholds))
        adjacent_masks = []
        for i in range(1, depth):
            previous_width = self.widths[i - 1]
            mask = torch.rand(previous_width, M, generator=gen) < fan2
            for j in range(M):
                if not mask[:, j].any():
                    mask[torch.randint(mask.shape[0], (), generator=gen), j] = True
            for i0 in range(mask.shape[0]):
                if not mask[i0].any():
                    mask[i0, torch.randint(M, (), generator=gen)] = True
            adjacent_masks.append(mask)

        # Keep the control's adjacent topology identical when adding skips.
        # A separate generator prevents skip draws or empty-row repairs from
        # shifting the random masks of later adjacent layers.
        skip_gen = torch.Generator().manual_seed(seed + 50_000_003)
        for i, adjacent_mask in enumerate(adjacent_masks, start=1):
            mask = adjacent_mask
            if self.early_event_skip and i >= 2:
                skip_mask = torch.rand(M1, M, generator=skip_gen) < fan2
                for i0 in range(skip_mask.shape[0]):
                    if not skip_mask[i0].any():
                        skip_mask[i0, torch.randint(M, (), generator=skip_gen)] = True
                mask = torch.cat((adjacent_mask, skip_mask), dim=0)
            n_in = mask.shape[0]
            self.layers.append(TVLayer(n_in, M, d, d, n, dmax, mask, True, w_sd[1],
                                       route_topk=self.route_topk,
                                       trainable_thresholds=self.trainable_thresholds))
        self.event_readout = bool(event_readout)
        self.readout_fusion = readout_fusion
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
        self.count_proj = nn.Linear(1, d, bias=False) if input_count_payload else None
        if self.count_proj is not None:
            # Keep the count-aware model identical to the baseline at step 0.
            nn.init.zeros_(self.count_proj.weight)

    @staticmethod
    def scored_pairs(layer, indices):
        if layer.mask is None:
            return len(indices) * layer.M
        return int(layer.mask[indices].sum())

    def forward(self, eb, ei, et, B, G, return_taps=False, collect_routes=False,
                route_override=None, spike_override=None,
                return_spike_diagnostics=False, input_counts=None,
                route_overrides=None, spike_overrides=None):
        raw_v = self.emb(ei)
        if self.count_proj is not None:
            if input_counts is None or input_counts.shape != et.shape:
                raise ValueError("count-aware E83 requires one merged-count mark per input event")
            raw_v = raw_v + self.count_proj(input_counts.reshape(-1, 1))
        emitted = []
        route_candidates, route_inputs, spike_diagnostics = [], [], []
        messages, spikes, candidates = [], [], []
        for i, layer in enumerate(self.layers):
            if i == 0:
                ib, ij, it, iv = eb, ei, et, raw_v
            elif self.early_event_skip and i >= 2:
                prev_b, prev_j, prev_t, prev_v = emitted[-1]
                first_b, first_j, first_t, first_v = emitted[0]
                ib = torch.cat((prev_b, first_b))
                ij = torch.cat((prev_j, first_j + self.widths[i - 1]))
                it = torch.cat((prev_t, first_t))
                iv = torch.cat((prev_v, first_v))
            else:
                ib, ij, it, iv = emitted[-1]
            candidates.append(self.scored_pairs(layer, ij))
            force_route = drop_route = None
            local_route_overrides = []
            if route_override is not None and route_override[0] == i:
                _, event_index, receiver, active = route_override
                if active:
                    force_route = (event_index, receiver)
                else:
                    drop_route = (event_index, receiver)
            if route_overrides is not None:
                local_route_overrides = [
                    (event_index, receiver, active)
                    for layer_id, event_index, receiver, active in route_overrides
                    if layer_id == i]
            local_spike_override = None
            if spike_override is not None and spike_override[0] == i:
                _, spike_time, spike_batch, spike_unit, spike_active = spike_override
                local_spike_override = (spike_time, spike_batch, spike_unit, spike_active)
            local_spike_overrides = [
                (spike_time, spike_batch, spike_unit, spike_active)
                for layer_id, spike_time, spike_batch, spike_unit, spike_active
                in (spike_overrides or []) if layer_id == i]
            result = layer(ib, ij, it, iv, B, G, force_route=force_route,
                           drop_route=drop_route,
                           route_overrides=local_route_overrides or None,
                           return_routes=collect_routes or return_spike_diagnostics,
                           spike_override=local_spike_override,
                           spike_overrides=local_spike_overrides or None,
                           return_spike_diagnostics=return_spike_diagnostics)
            if collect_routes or return_spike_diagnostics:
                out, msg, route_info = result
                if collect_routes:
                    route_candidates.append(route_info)
                    route_inputs.append((ib, ij, iv))
                if return_spike_diagnostics:
                    spike_diagnostics.append(route_info)
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
        if self.event_readout and self.readout_fusion == "all_depths":
            # The final classifier consumes the sparse evidence stream from
            # every stage, so count every active readout edge in its work.
            deep_candidates = sum(candidates)
            deep_messages = sum(messages)
        else:
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
        if return_spike_diagnostics:
            info["spike_diagnostics"] = spike_diagnostics
        return (taps[-1] if taps else None), info

    def infer_event_readout(self, layer_events, item, seq_end, dmax,
                            threshold, temperature):
        """Run the selected sparse readout, including exact multi-depth fusion."""
        if self.readout_fusion != "all_depths":
            event_b, units, times, payload = layer_events[-1]
            select = event_b == item
            deadline = float(seq_end + (self.depth + 1) * dmax + 60.0)
            return self.event_heads[-1].infer(
                units[select], times[select], payload[select], deadline,
                threshold, temperature)

        deadline = float(seq_end + (self.depth + 1) * dmax + 60.0)
        edge_times, edge_classes, edge_values = [], [], []
        initial_logits = torch.zeros(self.event_heads[0].n_classes)
        for head, layer in zip(self.event_heads, layer_events):
            event_b, units, times, payload = layer
            select = event_b == item
            et, ec, ev = head._edge_values(units[select], times[select],
                                           payload[select], deadline)
            edge_times.append(et); edge_classes.append(ec); edge_values.append(ev)
            initial_logits = initial_logits + head.bias.detach().cpu()
        edge_times = torch.cat(edge_times)
        edge_classes = torch.cat(edge_classes)
        edge_values = torch.cat(edge_values)
        order = edge_times.argsort(stable=True)
        edge_times, edge_classes, edge_values = (
            edge_times[order], edge_classes[order], edge_values[order])
        state = SparseAnytimeReadout(
            self.event_heads[0].n_classes, threshold, temperature,
            initial_logits=initial_logits.tolist())
        peak = state.confidence()
        group_times, group_counts = torch.unique_consecutive(
            edge_times, return_counts=True)
        t_cpu = group_times.detach().cpu().tolist()
        counts_cpu = group_counts.detach().cpu().tolist()
        class_cpu = edge_classes.detach().cpu().tolist()
        value_cpu = edge_values.detach().cpu().tolist()
        cursor = 0
        for event_time, count in zip(t_cpu, counts_cpu):
            stop = cursor + count
            updates = [(class_cpu[j], value_cpu[j])
                       for j in range(cursor, stop)]
            decision = state.update(event_time, updates)
            peak = max(peak, state.confidence())
            if decision is not None:
                return decision, peak
            cursor = stop
        decision = state.finish(deadline)
        return decision, max(peak, max(decision.posterior))


def sparse_event_objective(heads, layer_events, labels, seq_end, dmax,
                           prefix_times, fusion="deepest"):
    """Proper prefix CE, optionally summing evidence from every depth.

    In the fused mode each sparse readout contributes additive class log
    evidence at the same causal prefix. The terminal EOS evidence is aligned
    at the full-stack deadline. Intermediate heads retain their own local
    prefix losses as deep supervision.
    """
    local_losses = [head.prefix_loss(layer_events[k], labels, seq_end, k + 1,
                                     dmax, prefix_times)
                    for k, head in enumerate(heads)]
    if fusion == "deepest":
        main = local_losses[-1]
    else:
        full_delay = len(heads) + 1
        all_rows = [head.prefix_logits(layer_events[k], seq_end, full_delay,
                                       dmax, prefix_times)
                    for k, head in enumerate(heads)]
        fused_rows = [sum(all_rows[k][b] for k in range(len(heads)))
                      for b in range(len(labels))]
        main = torch.stack([
            F.cross_entropy(rows, labels[b].expand(len(rows)))
            for b, rows in enumerate(fused_rows)
        ]).mean()
    auxiliary = (sum(local_losses[:-1]) if len(local_losses) > 1
                 else main.new_zeros(()))
    return main, auxiliary, local_losses


def per_item_deepest_loss(net, info, labels, seq_end, depth, dmax, prefix_times):
    """Deepest-head prefix loss and terminal logits, one row per utterance."""
    rows = net.event_heads[-1].prefix_logits(
        info["layer_events"][-1], seq_end, depth, dmax, prefix_times)
    losses = torch.stack([
        F.cross_entropy(row, labels[b].expand(len(row)))
        for b, row in enumerate(rows)
    ])
    return losses, torch.stack([row[-1] for row in rows])


def sample_spike_option_path(net, eb, ei, et, input_counts, batch_size, grid,
                             initial_info, target_batch, args, rng):
    """Sample near-threshold births and retain each causal prefix of the path."""
    overrides = []
    prefix_overrides = []
    actions = []
    state_info = initial_info
    horizon = min(grid - 1, int(getattr(args, "prefix_horizon_ms", 1000.0)))
    for layer_id in range(net.depth):
        if rng.random() >= args.cf_spike_option_probability:
            continue
        diagnostic = state_info["spike_diagnostics"][layer_id]
        margins = diagnostic["spike_margin_trace"]
        fired = diagnostic["spike_fire_mask"]
        refractory = diagnostic["spike_refractory_trace"]
        time_ids = torch.arange(margins.shape[0], device=margins.device)
        valid_time = (time_ids > 0) & (time_ids <= horizon)
        valid = (
            valid_time[:, None]
            & (margins[:, target_batch] <= 0.0)
            & (margins[:, target_batch] >= -args.cf_spike_option_band)
            & ~fired[:, target_batch]
            & (refractory[:, target_batch] < 1e-3)
        )
        coordinates = torch.nonzero(valid, as_tuple=False).detach().cpu().numpy()
        if not len(coordinates):
            continue
        candidate_margins = np.asarray([
            float(margins[t, target_batch, unit]) for t, unit in coordinates
        ])
        weights = np.exp(-np.abs(candidate_margins)
                         / max(args.cf_spike_option_temperature, 1e-8))
        probabilities = weights / weights.sum()
        selected = int(rng.choice(len(coordinates), p=probabilities))
        time_id, unit_id = map(int, coordinates[selected])
        margin = float(candidate_margins[selected])
        action = {
            "layer": layer_id,
            "time": time_id,
            "batch": target_batch,
            "unit": unit_id,
            "margin": margin,
            "candidate_count": int(len(coordinates)),
            "proposal_probability": float(probabilities[selected]),
        }
        actions.append(action)
        overrides.append((layer_id, time_id, target_batch, unit_id, True))
        prefix_overrides.append(list(overrides))
        with torch.no_grad():
            _, state_info = net(
                eb, ei, et, batch_size, grid, return_taps=True,
                return_spike_diagnostics=True, spike_overrides=overrides,
                input_counts=input_counts)
    return overrides, actions, prefix_overrides


def continue_spike_option_rollout(net, eb, ei, et, input_counts, batch_size,
                                 grid, labels, seq_end, depth, dmax,
                                 prefix_times, initial_info, initial_overrides,
                                 start_layer, target_batch, args, rng):
    """Sample one sparse future route continuation from a counterfactual state."""
    overrides = list(initial_overrides)
    state_info = initial_info
    actions = []
    horizon = min(grid - 1, int(getattr(args, "prefix_horizon_ms", 1000.0)))
    for layer_id in range(start_layer, net.depth):
        # Consume the same pair of random numbers at every layer so paired
        # parent/child rollouts share proposal randomness even when one arm
        # has no eligible candidates.
        include_draw = rng.random()
        proposal_draw = rng.random()
        if include_draw >= args.cf_spike_option_probability:
            continue
        diagnostic = state_info["spike_diagnostics"][layer_id]
        margins = diagnostic["spike_margin_trace"]
        fired = diagnostic["spike_fire_mask"]
        refractory = diagnostic["spike_refractory_trace"]
        time_ids = torch.arange(margins.shape[0], device=margins.device)
        valid_time = (time_ids > 0) & (time_ids <= horizon)
        valid = (
            valid_time[:, None]
            & (margins[:, target_batch] <= 0.0)
            & (margins[:, target_batch] >= -args.cf_spike_option_band)
            & ~fired[:, target_batch]
            & (refractory[:, target_batch] < 1e-3)
        )
        coordinates = torch.nonzero(valid, as_tuple=False).detach().cpu().numpy()
        if not len(coordinates):
            continue
        candidate_margins = np.asarray([
            float(margins[t, target_batch, unit]) for t, unit in coordinates
        ])
        weights = np.exp(-np.abs(candidate_margins)
                         / max(args.cf_spike_option_temperature, 1e-8))
        probabilities = weights / weights.sum()
        selected = min(int(np.searchsorted(
            np.cumsum(probabilities), proposal_draw, side="right")),
            len(coordinates) - 1)
        time_id, unit_id = map(int, coordinates[selected])
        overrides.append((layer_id, time_id, target_batch, unit_id, True))
        actions.append({
            "layer": layer_id,
            "time": time_id,
            "unit": unit_id,
            "candidate_count": int(len(coordinates)),
            "proposal_probability": float(probabilities[selected]),
        })
        with torch.no_grad():
            _, state_info = net(
                eb, ei, et, batch_size, grid, return_taps=True,
                return_spike_diagnostics=True, spike_overrides=overrides,
                input_counts=input_counts)
    with torch.no_grad():
        terminal_losses, _ = per_item_deepest_loss(
            net, state_info, labels, seq_end, depth, dmax, prefix_times)
    return float(terminal_losses[target_batch]), state_info, actions


def soft_minimum(values, temperature):
    """Normalized entropic continuation value for loss minimization."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("soft minimum requires at least one continuation")
    if temperature <= 0:
        return float(values.min())
    minimum = float(values.min())
    return minimum - temperature * math.log(float(np.exp(
        -(values - minimum) / temperature).mean()))


def option_suffix_parameters(net, last_layer):
    hidden = [p for layer in net.layers[last_layer + 1:]
              for p in layer.parameters() if p.requires_grad]
    return hidden + [p for p in net.event_heads[-1].parameters()
                     if p.requires_grad]


def finite_progress_from_grads(loss, params, grads, replay_loss, lr, clip_norm):
    """Measure finite one-step suffix-SGD progress after its graph was consumed."""
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
    return before - after, grad_norm


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


def nearby_closed_route_pairs(route_candidates, near_band, time_window, rng, limit,
                              sampling="global"):
    """Sample sparse pairs of closed routes to one receiver with nearby arrivals.

    Grouping by batch/receiver and taking adjacent time-ordered candidates
    keeps proposal construction linear in the candidate count; it never forms
    the all-pairs Cartesian product. The returned route indices refer to each
    layer's route_candidates record.
    """
    candidates_by_layer = [[] for _ in route_candidates]
    for layer_id, route_info in enumerate(route_candidates):
        scores = route_info["score"].cpu().numpy()
        source_batch = route_info["source_batch"].cpu().numpy()
        source_time = route_info["source_time"].cpu().numpy()
        receivers = route_info["receiver"].cpu().numpy()
        event_ids = route_info["event_index"].cpu().numpy()
        eligible = np.flatnonzero((scores < 0.0) & (scores >= -near_band))
        groups = {}
        for route_idx in eligible:
            key = (int(source_batch[route_idx]), int(receivers[route_idx]))
            groups.setdefault(key, []).append(int(route_idx))
        for route_ids in groups.values():
            route_ids.sort(key=lambda rid: float(source_time[rid]))
            for left, right in zip(route_ids[:-1], route_ids[1:]):
                if event_ids[left] == event_ids[right]:
                    continue
                gap = float(source_time[right] - source_time[left])
                if 0.0 <= gap <= time_window:
                    candidates_by_layer[layer_id].append((layer_id, left, right))
    per_layer_count = np.asarray([len(rows) for rows in candidates_by_layer], dtype=np.int64)
    candidates = [row for layer_rows in candidates_by_layer for row in layer_rows]
    if sampling == "global":
        take = min(limit, len(candidates))
        if len(candidates) > take:
            chosen = rng.choice(len(candidates), size=take, replace=False)
            selected_rows = [candidates[int(i)] for i in np.atleast_1d(chosen)]
        else:
            selected_rows = candidates
        inclusion_probability = (take / len(candidates)) if candidates else 0.0
        selected = [(*row, inclusion_probability) for row in selected_rows]
    elif sampling in ("layer_balanced", "late_balanced"):
        if limit != 1:
            raise ValueError(f"{sampling} pair sampling requires --cf_pairs_per_batch 1")
        eligible_layers = np.flatnonzero(per_layer_count)
        if sampling == "late_balanced":
            # D4 selects L3/L4; for a general D, use the last half of the
            # stack. Do not fall back to early layers when none are eligible.
            first_late_layer = len(candidates_by_layer) // 2
            eligible_layers = eligible_layers[eligible_layers >= first_late_layer]
        if len(eligible_layers):
            layer_id = int(rng.choice(eligible_layers))
            layer_rows = candidates_by_layer[layer_id]
            row = layer_rows[int(rng.integers(len(layer_rows)))]
            inclusion_probability = 1.0 / (len(eligible_layers) * len(layer_rows))
            selected = [(*row, inclusion_probability)]
        else:
            selected = []
    else:
        raise ValueError(f"unknown pair-sampling mode: {sampling}")
    return len(candidates), per_layer_count, selected


def route_replacement_candidates(route_candidates, score_gap, route_topk):
    """Find top-k boundary swaps: weakest selected route vs strongest loser.

    Candidate construction is linear in the sparse edge list. A swap keeps
    the source event's outgoing message count fixed and changes only its
    receiver, making the counterfactual comparable to a categorical MoE choice.
    """
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
            on = [idx for idx in route_ids if active[idx]]
            if len(on) != route_topk:
                continue
            off = [idx for idx in route_ids if not active[idx]]
            if not off:
                continue
            weakest = min(on, key=lambda idx: scores[idx])
            # Top-k is applied only to positive content gates. A negative
            # loser is the null/no-message alternative, not a competing route;
            # keep this experiment specifically about receiver replacement.
            positive_off = [idx for idx in off if scores[idx] > 0.0]
            if not positive_off:
                continue
            strongest_loser = max(positive_off, key=lambda idx: scores[idx])
            gap = float(scores[weakest] - scores[strongest_loser])
            if 0.0 <= gap <= score_gap:
                by_layer[layer_id].append((
                    layer_id, event_id, int(receivers[weakest]),
                    int(receivers[strongest_loser]), weakest,
                    strongest_loser, gap))
    return by_layer


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
    ap.add_argument("--readout_fusion", choices=("deepest", "all_depths"), default="all_depths",
                    help="event_prefix only: use the deepest event head or add sparse class evidence from every depth")
    ap.add_argument("--input_count_payload", choices=("off", "additive"), default="off",
                    help="preserve the merged raw-spike count as an additive sparse vector mark")
    ap.add_argument("--early_event_skip", action="store_true",
                    help="let layers 3+ receive a sparse skip stream from layer 1 as well as the adjacent layer")
    ap.add_argument("--run_tag", default="",
                    help="optional filename tag for otherwise identical runs with different evaluation protocols")
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
    ap.add_argument("--cf_pairs_per_batch", type=int, default=0,
                    help="maximum receiver-bundle route pairs shadowed per batch; each pair uses 3 matched replays")
    ap.add_argument("--cf_pair_window_ms", type=float, default=25.0,
                    help="maximum arrival-time gap for a candidate pair targeting the same receiver")
    ap.add_argument("--cf_pair_sampling", choices=("global", "layer_balanced", "late_balanced"), default="global",
                    help="sample globally, balance across eligible layers, or balance across the last half of layers")
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
    ap.add_argument("--route_topk", type=int, default=0,
                    help="cap each event to its top-K positive-score sparse receivers; 0 keeps independent gates")
    ap.add_argument("--cf_route_swaps_per_layer", type=int, default=0,
                    help="score-preserving-count route replacement shadows per layer and batch (requires --route_topk)")
    ap.add_argument("--cf_route_swap_band", type=float, default=0.5,
                    help="maximum winner-minus-loser score gap for route replacement shadows")
    ap.add_argument("--trainable_thresholds", action="store_true",
                    help="add zero-initialized per-unit threshold offsets (needed for the spike-option update)")
    ap.add_argument("--cf_spike_option_updates", action="store_true",
                    help="train per-unit thresholds from a sampled multi-layer spike-option path")
    ap.add_argument("--cf_spike_option_weight", type=float, default=10.0,
                    help="lambda multiplying finite suffix-learning progress in scalar option utility")
    ap.add_argument("--cf_spike_option_probability", type=float, default=0.5,
                    help="per-layer probability of proposing one spike birth along the sampled path")
    ap.add_argument("--cf_spike_option_band", type=float, default=0.5,
                    help="voltage-margin band below threshold for candidate spike births")
    ap.add_argument("--cf_spike_option_sigma", type=float, default=0.25,
                    help="logistic voltage scale used to convert scalar utility into threshold credit")
    ap.add_argument("--cf_spike_option_temperature", type=float, default=0.15,
                    help="proposal temperature over absolute subthreshold margins")
    ap.add_argument("--cf_spike_option_lr", type=float, default=0.05,
                    help="separate threshold-credit step multiplier")
    ap.add_argument("--cf_spike_option_step_clip", type=float, default=0.01,
                    help="maximum absolute per-action threshold offset change per batch")
    ap.add_argument("--cf_spike_option_bound", type=float, default=0.1,
                    help="trust-region bound for learned threshold offsets")
    ap.add_argument("--cf_spike_virtual_lr", type=float, default=0.001,
                    help="learning rate for the temporary suffix-progress measurement")
    ap.add_argument("--cf_spike_virtual_clip", type=float, default=1.0,
                    help="gradient-norm cap for the temporary suffix-progress measurement")
    ap.add_argument("--cf_spike_suffix_weight", type=float, default=0.0,
                    help="replace one sampled example's factual suffix gradient with its forced-spike branch gradient")
    ap.add_argument("--cf_spike_suffix_causal", action="store_true",
                    help="choose each hidden layer's counterfactual gradient from the latest forced prefix strictly before that layer")
    ap.add_argument("--cf_spike_option_local_utility", action="store_true",
                    help="credit each sampled spike by its conditional parent-to-child loss and suffix-learning change")
    ap.add_argument("--cf_spike_option_transfer_utility", action="store_true",
                    help="measure option learning value on other minibatch examples, not self-fit on the forced example")
    ap.add_argument("--cf_spike_option_rollout_value", action="store_true",
                    help="score optionality by proposal-conditioned future route continuations")
    ap.add_argument("--cf_spike_option_rollouts", type=int, default=2,
                    help="matched future route continuations per parent/child state")
    ap.add_argument("--cf_spike_option_backup_temperature", type=float, default=0.1,
                    help="temperature for the normalized entropic soft-min continuation backup")
    ap.add_argument("--cf_spike_option_improvement_epsilon", type=float, default=0.05,
                    help="loss improvement cutoff for the separate continuation-breadth diagnostic")
    ap.add_argument("--shift", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--eval_limit", type=int, default=128)
    ap.add_argument("--rng_protocol", choices=("legacy_shared", "split"), default="split",
                    help="split evaluation selection from train sampling/order/augmentation (default); legacy_shared reproduces older coupled runs")
    ap.add_argument("--save_checkpoint", action="store_true",
                    help="save final model weights beside the result JSON")
    a = ap.parse_args()
    if a.run_tag and not all(ch.isalnum() or ch in "-_" for ch in a.run_tag):
        raise ValueError("--run_tag may contain only letters, digits, '-' and '_'")
    if (a.prefix_samples < 1 or not (0.0 <= a.prefix_window_start < 1.0)
            or not math.isfinite(a.prefix_horizon_ms) or a.prefix_horizon_ms <= 0.0):
        raise ValueError("prefix_samples and prefix_horizon_ms must be positive; prefix_window_start must be in [0, 1)")
    if (a.cf_shadows_per_layer < 0 or a.cf_pairs_per_batch < 0
            or a.cf_route_swaps_per_layer < 0 or a.route_topk < 0
            or not math.isfinite(a.cf_route_swap_band) or a.cf_route_swap_band <= 0.0
            or not math.isfinite(a.cf_pair_window_ms) or a.cf_pair_window_ms <= 0.0
            or not math.isfinite(a.cf_band) or a.cf_band <= 0.0
            or not math.isfinite(a.cf_sigma) or a.cf_sigma <= 0.0
            or not math.isfinite(a.cf_lr) or a.cf_lr < 0.0
            or not math.isfinite(a.cf_grad_clip) or a.cf_grad_clip <= 0.0
            or not math.isfinite(a.cf_delta_clip) or a.cf_delta_clip <= 0.0):
        raise ValueError("counterfactual count must be nonnegative; band, sigma, loss-difference clip, and gradient clip must be positive; cf_lr must be nonnegative")
    if (not math.isfinite(a.cf_spike_option_weight) or a.cf_spike_option_weight < 0
            or not 0.0 <= a.cf_spike_option_probability <= 1.0
            or not math.isfinite(a.cf_spike_option_band) or a.cf_spike_option_band <= 0
            or not math.isfinite(a.cf_spike_option_sigma) or a.cf_spike_option_sigma <= 0
            or not math.isfinite(a.cf_spike_option_temperature) or a.cf_spike_option_temperature <= 0
            or not math.isfinite(a.cf_spike_option_lr) or a.cf_spike_option_lr < 0
            or not math.isfinite(a.cf_spike_option_step_clip) or a.cf_spike_option_step_clip <= 0
            or not math.isfinite(a.cf_spike_option_bound) or a.cf_spike_option_bound <= 0
            or not math.isfinite(a.cf_spike_virtual_lr) or a.cf_spike_virtual_lr <= 0
            or not math.isfinite(a.cf_spike_virtual_clip) or a.cf_spike_virtual_clip <= 0
            or not math.isfinite(a.cf_spike_suffix_weight) or a.cf_spike_suffix_weight < 0):
        raise ValueError("invalid spike-option update, proposal, or virtual-step parameter")
    if a.cf_spike_option_updates and (not a.trainable_thresholds
                                      or a.objective != "event_prefix"):
        raise ValueError("spike-option updates require --trainable_thresholds and --objective event_prefix")
    if a.cf_spike_suffix_weight > 0 and not a.cf_spike_option_updates:
        raise ValueError("counterfactual suffix-gradient substitution requires --cf_spike_option_updates")
    if a.cf_spike_suffix_causal and a.cf_spike_suffix_weight <= 0:
        raise ValueError("causal suffix credit requires --cf_spike_suffix_weight > 0")
    if a.cf_spike_option_local_utility and not a.cf_spike_option_updates:
        raise ValueError("local option utility requires --cf_spike_option_updates")
    if a.cf_spike_option_transfer_utility and not a.cf_spike_option_local_utility:
        raise ValueError("transfer utility requires --cf_spike_option_local_utility")
    if a.cf_spike_option_rollout_value and not a.cf_spike_option_local_utility:
        raise ValueError("continuation optionality requires --cf_spike_option_local_utility")
    if (a.cf_spike_option_rollouts < 1
            or not math.isfinite(a.cf_spike_option_backup_temperature)
            or a.cf_spike_option_backup_temperature < 0
            or not math.isfinite(a.cf_spike_option_improvement_epsilon)
            or a.cf_spike_option_improvement_epsilon < 0):
        raise ValueError("continuation rollouts must be positive; backup temperature and improvement cutoff must be nonnegative")
    if a.cf_route_swaps_per_layer and not a.route_topk:
        raise ValueError("--cf_route_swaps_per_layer requires --route_topk > 0")
    if a.cf_pair_sampling in ("layer_balanced", "late_balanced") and a.cf_pairs_per_batch != 1:
        raise ValueError(f"{a.cf_pair_sampling} pair sampling requires --cf_pairs_per_batch 1")
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed)
    if a.rng_protocol == "legacy_shared":
        shared_rng = np.random.default_rng(a.seed)
        train_subset_rng = eval_subset_rng = train_order_rng = augmentation_rng = shared_rng
    else:
        # Each stochastic stage owns a stream. In particular, evaluation-set
        # size cannot advance the streams that select or train on examples.
        train_subset_rng = np.random.default_rng(a.seed)
        eval_subset_rng = np.random.default_rng(a.seed + 300_007)
        train_order_rng = np.random.default_rng(a.seed + 400_009)
        augmentation_rng = np.random.default_rng(a.seed + 500_009)
    prefix_rng = np.random.default_rng(a.seed + 100_003)
    counterfactual_rng = np.random.default_rng(a.seed + 200_003)
    pair_rng = np.random.default_rng(a.seed + 700_031)
    route_swap_rng = np.random.default_rng(a.seed + 900_017)
    spike_option_rng = np.random.default_rng(a.seed + 1_100_021)
    spike_option_rollout_rng = np.random.default_rng(a.seed + 1_100_023)
    spike_option_target_rng = np.random.default_rng(a.seed + 1_300_027)
    t0 = time.time()

    def load(split, part):
        return [(*events(t, u, a.merge), y) for t, u, y in S.utterances(split, a.bands, part) if len(t) > 1]

    tr = stratified_limit(load("train", "fit_spk"), a.limit, train_subset_rng)
    ev = stratified_limit(load("train", "val_spk"), a.eval_limit, eval_subset_rng)
    widths_sd = [float(v) for v in a.w_sd.split(",")]
    if len(widths_sd) != 2:
        raise ValueError("--w_sd requires two comma-separated values")
    event_readout = a.objective == "event_prefix"
    net = DeepSHD(a.bands, a.d, a.n, a.M1, a.M, a.depth, a.window, a.fan2,
                  a.readout_fan, a.dmax, widths_sd, a.seed, event_readout=event_readout,
                  readout_fusion=a.readout_fusion,
                  input_count_payload=a.input_count_payload == "additive",
                  early_event_skip=a.early_event_skip,
                  route_topk=a.route_topk,
                  trainable_thresholds=a.trainable_thresholds)
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
        option_paths_ep = option_actions_ep = option_candidates_ep = 0
        option_immediate_ep = option_learning_ep = option_utility_ep = 0.0
        option_local_utility_sum_ep = 0.0
        option_local_immediate_sum_ep = option_local_learning_sum_ep = 0.0
        option_local_optionality_sum_ep = 0.0
        option_local_continuation_gain_sum_ep = 0.0
        option_local_beneficial_mass_delta_sum_ep = 0.0
        option_rollout_count_ep = option_rollout_actions_ep = 0
        option_local_utility_actions_ep = 0
        option_local_utility_by_layer_ep = np.zeros(a.depth, dtype=np.float64)
        option_local_optionality_by_layer_ep = np.zeros(a.depth, dtype=np.float64)
        option_local_beneficial_mass_delta_by_layer_ep = np.zeros(a.depth, dtype=np.float64)
        option_local_actions_by_layer_ep = np.zeros(a.depth, dtype=np.int64)
        option_grad_norm_ep = option_threshold_step_ep = 0.0
        option_suffix_batches_ep = 0
        option_suffix_correction_norm_ep = 0.0
        option_suffix_correction_by_layer_ep = np.zeros(a.depth + 1)
        option_hidden_delta_ep = np.zeros(a.depth, dtype=np.float64)
        option_action_by_layer_ep = np.zeros(a.depth, dtype=np.int64)
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
        cf_pair_eligible_ep = cf_pair_shadow_ep = 0
        cf_pair_delta_sum_ep = cf_pair_delta_sq_ep = 0.0
        cf_pair_interaction_sum_ep = cf_pair_interaction_sq_ep = 0.0
        cf_pair_helpful_ep = cf_pair_synergy_ep = 0
        cf_pair_shadow_by_layer_ep = np.zeros(a.depth, dtype=np.int64)
        cf_pair_candidate_by_layer_ep = np.zeros(a.depth, dtype=np.int64)
        cf_pair_sample_prob_sum_ep = 0.0
        cf_pair_sample_prob_min_ep = float("inf")
        cf_pair_sample_prob_max_ep = 0.0
        cf_route_swap_eligible_ep = np.zeros(a.depth, dtype=np.int64)
        cf_route_swap_shadow_ep = np.zeros(a.depth, dtype=np.int64)
        cf_route_swap_delta_sum_ep = np.zeros(a.depth, dtype=np.float64)
        cf_route_swap_gap_sum_ep = np.zeros(a.depth, dtype=np.float64)
        cf_route_swap_helpful_ep = np.zeros(a.depth, dtype=np.int64)
        perm = train_order_rng.permutation(len(tr)); gsum = np.zeros(a.depth)
        gsum_with_spike_cf = np.zeros(a.depth)
        for i0 in range(0, len(tr), a.bs):
            items = [tr[j] for j in perm[i0:i0 + a.bs]]
            eb, ei, et, input_counts, y, tmax, seq_end = batch_to_events(
                items, a.bands, a.shift, augmentation_rng, a.drop)
            grid = tmax + (a.depth + 1) * math.ceil(a.dmax) + 60
            prefix_times = None
            if event_readout:
                prefix_times = sampled_prefix_times(len(items), a.prefix_samples,
                                                    a.prefix_horizon_ms, a.prefix_window_start,
                                                    et.device, et.dtype, prefix_rng)
            _, info = net(eb, ei, et, len(items), grid, return_taps=True,
                          collect_routes=event_readout and (
                              a.cf_shadows_per_layer > 0 or a.cf_pairs_per_batch > 0
                              or a.cf_route_swaps_per_layer > 0),
                          return_spike_diagnostics=a.cf_spike_option_updates,
                          input_counts=input_counts)
            traces = info["tap_traces"]
            if event_readout:
                main_loss, aux_loss, head_losses = sparse_event_objective(
                    net.event_heads, info["layer_events"], y, seq_end,
                    a.dmax, prefix_times, a.readout_fusion)
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
            if not event_readout:
                aux_loss = sum(aux_losses) if aux_losses else main_loss.new_zeros(())
            if i0 == 0:
                route_probe = route_gradient_probe(main_loss, aux_loss, net.layers)
            loss = main_loss + a.aux_weight * aux_loss
            option_threshold_deltas = []
            option_pending = None
            if a.cf_spike_option_updates:
                item_losses, item_logits = per_item_deepest_loss(
                    net, info, y, seq_end, a.depth, a.dmax, prefix_times)
                probabilities = torch.softmax(item_logits.detach(), dim=-1)
                labels_np = y.detach().cpu().numpy()
                predictions = probabilities.argmax(-1).cpu().numpy()
                wrong = np.flatnonzero(predictions != labels_np)
                if len(wrong):
                    p_true = probabilities.detach().cpu().numpy()[
                        np.arange(len(items)), labels_np]
                    entropy = (-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
                               / math.log(probabilities.shape[-1])).cpu().numpy()
                    target_weights = ((1.0 - p_true[wrong])
                                      * (0.5 + 0.5 * entropy[wrong]))
                    target_weights = (target_weights / target_weights.sum()
                                      if target_weights.sum() > 0
                                      else np.full(len(wrong), 1.0 / len(wrong)))
                    target_batch = int(spike_option_target_rng.choice(
                        wrong, p=target_weights))
                else:
                    target_batch = int(item_losses.detach().argmax())
                path_overrides, option_actions, prefix_overrides = sample_spike_option_path(
                    net, eb, ei, et, input_counts, len(items), grid, info,
                    target_batch, a, spike_option_rng)
                option_candidates_ep += sum(row["candidate_count"]
                                            for row in option_actions)
                if option_actions:
                    peer_indices = [index for index in range(len(items))
                                    if index != target_batch]
                    peer_loss = (item_losses[peer_indices].mean()
                                 if peer_indices else None)

                    def peer_replay_loss(indices=tuple(peer_indices)):
                        if not indices:
                            return 0.0
                        with torch.no_grad():
                            _, replay_info = net(
                                eb, ei, et, len(items), grid, return_taps=True,
                                input_counts=input_counts)
                            replay_losses, _ = per_item_deepest_loss(
                                net, replay_info, y, seq_end, a.depth, a.dmax,
                                prefix_times)
                        return float(replay_losses[list(indices)].mean())

                    with torch.enable_grad():
                        root_target_loss = item_losses[target_batch]
                        last_layer = max(action["layer"] for action in option_actions)
                        suffix = option_suffix_parameters(net, last_layer)
                        causal_contexts = None
                        if a.cf_spike_suffix_causal or a.cf_spike_option_local_utility:
                            first_layer = min(action["layer"] for action in option_actions)
                            root_suffix = option_suffix_parameters(net, first_layer)
                            root_grad_values = torch.autograd.grad(
                                root_target_loss, root_suffix, allow_unused=True,
                                retain_graph=True)
                            root_grad_by_id = {
                                id(param): grad for param, grad
                                in zip(root_suffix, root_grad_values)}
                            causal_contexts = []
                            for action, branch_overrides in zip(
                                    option_actions, prefix_overrides):
                                _, prefix_info = net(
                                    eb, ei, et, len(items), grid, return_taps=True,
                                    return_spike_diagnostics=True,
                                    spike_overrides=branch_overrides,
                                    input_counts=input_counts)
                                prefix_losses, _ = per_item_deepest_loss(
                                    net, prefix_info, y, seq_end, a.depth, a.dmax,
                                    prefix_times)
                                prefix_loss = prefix_losses[target_batch]
                                prefix_layer = action["layer"]
                                prefix_suffix = option_suffix_parameters(
                                    net, prefix_layer)
                                prefix_grad_values = torch.autograd.grad(
                                    prefix_loss, prefix_suffix, allow_unused=True,
                                    retain_graph=False)
                                prefix_grad_by_id = {
                                    id(param): grad for param, grad
                                    in zip(prefix_suffix, prefix_grad_values)}
                                frozen_overrides = tuple(branch_overrides)

                                def prefix_replay_loss(overrides=frozen_overrides):
                                    with torch.no_grad():
                                        _, replay_info = net(
                                            eb, ei, et, len(items), grid,
                                            return_taps=True,
                                            spike_overrides=list(overrides) or None,
                                            input_counts=input_counts)
                                        replay_losses, _ = per_item_deepest_loss(
                                            net, replay_info, y, seq_end, a.depth,
                                            a.dmax, prefix_times)
                                    return float(replay_losses[target_batch])

                                causal_contexts.append({
                                    "last_layer": prefix_layer,
                                    "loss": prefix_loss,
                                    "suffix": prefix_suffix,
                                    "grads": prefix_grad_values,
                                    "grad_by_id": prefix_grad_by_id,
                                    "replay": prefix_replay_loss,
                                    "overrides": frozen_overrides,
                                    "info": prefix_info,
                                })
                            branch_info = causal_contexts[-1]["info"]
                            branch_target_loss = causal_contexts[-1]["loss"]
                            branch_grads = tuple(
                                causal_contexts[-1]["grad_by_id"].get(id(param))
                                for param in suffix)
                            factual_grads = tuple(
                                root_grad_by_id.get(id(param)) for param in suffix)
                        else:
                            _, branch_info = net(
                                eb, ei, et, len(items), grid, return_taps=True,
                                return_spike_diagnostics=True,
                                spike_overrides=path_overrides,
                                input_counts=input_counts)
                            branch_losses, _ = per_item_deepest_loss(
                                net, branch_info, y, seq_end, a.depth, a.dmax,
                                prefix_times)
                            branch_target_loss = branch_losses[target_batch]
                            factual_grads = torch.autograd.grad(
                                root_target_loss, suffix, allow_unused=True,
                                retain_graph=True)
                            branch_grads = torch.autograd.grad(
                                branch_target_loss, suffix, allow_unused=True,
                                retain_graph=False)

                        def root_replay_loss():
                            with torch.no_grad():
                                _, replay_info = net(
                                    eb, ei, et, len(items), grid, return_taps=True,
                                    input_counts=input_counts)
                                replay_losses, _ = per_item_deepest_loss(
                                    net, replay_info, y, seq_end, a.depth,
                                    a.dmax, prefix_times)
                            return float(replay_losses[target_batch])

                        def branch_replay_loss():
                            with torch.no_grad():
                                _, replay_info = net(
                                    eb, ei, et, len(items), grid, return_taps=True,
                                    spike_overrides=path_overrides,
                                    input_counts=input_counts)
                                replay_losses, _ = per_item_deepest_loss(
                                    net, replay_info, y, seq_end, a.depth,
                                    a.dmax, prefix_times)
                            return float(replay_losses[target_batch])
                    option_pending = {
                        "target_batch": target_batch,
                        "actions": option_actions,
                        "last_layer": last_layer,
                        "suffix": suffix,
                        "root_loss": root_target_loss,
                        "branch_loss": branch_target_loss,
                        "root_grads": factual_grads,
                        "branch_grads": branch_grads,
                        "root_grad_by_id": (root_grad_by_id
                                            if causal_contexts is not None else None),
                        "causal_contexts": causal_contexts,
                        "peer_loss": peer_loss,
                        "peer_replay": peer_replay_loss,
                        "root_replay": root_replay_loss,
                        "branch_replay": branch_replay_loss,
                        "root_events": info["layer_events"],
                        "branch_events": branch_info["layer_events"],
                    }
            cf_eligible = cf_shadow_count = 0
            cf_abs_delta_sum = 0.0
            cf_clipped_count = 0
            cf_pair_shadow_count = 0
            cf_route_swap_count = 0
            cf_grads = None
            if event_readout and (a.cf_shadows_per_layer or a.cf_pairs_per_batch
                                  or a.cf_route_swaps_per_layer):
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
                                route_override=(k, event_index, receiver, not active),
                                input_counts=input_counts)
                            shadow_main, shadow_aux, _ = sparse_event_objective(
                                net.event_heads, shadow_info["layer_events"], y,
                                seq_end, a.dmax, prefix_times, a.readout_fusion)
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
                if a.cf_route_swaps_per_layer:
                    swap_candidates = route_replacement_candidates(
                        info["route_candidates"], a.cf_route_swap_band, a.route_topk)
                    for k, candidates_k in enumerate(swap_candidates):
                        cf_route_swap_eligible_ep[k] += len(candidates_k)
                        take = min(a.cf_route_swaps_per_layer, len(candidates_k))
                        if not take:
                            continue
                        chosen = route_swap_rng.choice(len(candidates_k), size=take,
                                                       replace=False)
                        batch_ids, source_units, source_payload = info["route_inputs"][k]
                        for chosen_idx in np.atleast_1d(chosen):
                            (layer_id, event_id, winner_receiver, loser_receiver,
                             winner_idx, loser_idx, score_gap) = candidates_k[int(chosen_idx)]
                            overrides = [
                                (layer_id, event_id, winner_receiver, False),
                                (layer_id, event_id, loser_receiver, True)]
                            with torch.no_grad():
                                _, swap_info = net(
                                    eb, ei, et, len(items), grid, return_taps=True,
                                    route_overrides=overrides, input_counts=input_counts)
                                swap_main, swap_aux, _ = sparse_event_objective(
                                    net.event_heads, swap_info["layer_events"], y,
                                    seq_end, a.dmax, prefix_times, a.readout_fusion)
                                alt_loss = swap_main + a.aux_weight * swap_aux
                            event_unit = source_units[event_id]

                            def live_score(receiver_id):
                                return ((net.layers[layer_id].q[receiver_id]
                                         * source_payload[event_id]).sum()
                                        + net.layers[layer_id].c[event_unit, receiver_id])

                            score_win = live_score(winner_receiver)
                            score_loser = live_score(loser_receiver)
                            score_difference = score_win - score_loser
                            p_win = torch.sigmoid(score_difference.detach() / a.cf_sigma)
                            delta_win_vs_loser = float(base_total) - float(alt_loss.item())
                            dloss_dgap = (p_win * (1.0 - p_win) / a.cf_sigma) * delta_win_vs_loser
                            cf_proxy = cf_proxy + (a.cf_weight * dloss_dgap
                                                   * (score_difference - score_difference.detach())
                                                   / take)
                            alt_delta = float(alt_loss.item()) - float(base_total)
                            cf_route_swap_delta_sum_ep[layer_id] += alt_delta
                            cf_route_swap_gap_sum_ep[layer_id] += score_gap
                            cf_route_swap_helpful_ep[layer_id] += int(alt_delta < 0.0)
                            cf_route_swap_shadow_ep[layer_id] += 1
                            cf_route_swap_count += 1

                if a.cf_pairs_per_batch:
                    pair_candidate_count, pair_candidates_by_layer, eligible_pairs = nearby_closed_route_pairs(
                        info["route_candidates"], a.cf_band, a.cf_pair_window_ms,
                        pair_rng, a.cf_pairs_per_batch, a.cf_pair_sampling)
                    cf_pair_eligible_ep += pair_candidate_count
                    cf_pair_candidate_by_layer_ep += pair_candidates_by_layer
                    pair_proxy = main_loss.new_zeros(())
                    for k, route_a, route_b, sample_probability in eligible_pairs:
                        cf_pair_sample_prob_sum_ep += sample_probability
                        cf_pair_sample_prob_min_ep = min(cf_pair_sample_prob_min_ep,
                                                         sample_probability)
                        cf_pair_sample_prob_max_ep = max(cf_pair_sample_prob_max_ep,
                                                         sample_probability)
                        route_info = info["route_candidates"][k]
                        event_ids = route_info["event_index"]
                        receivers = route_info["receiver"]
                        event_a, event_b = int(event_ids[route_a]), int(event_ids[route_b])
                        receiver_a, receiver_b = int(receivers[route_a]), int(receivers[route_b])
                        if receiver_a != receiver_b:
                            raise RuntimeError("receiver-bundle proposal has mismatched targets")

                        def shadow_route_loss(overrides):
                            with torch.no_grad():
                                _, pair_info = net(
                                    eb, ei, et, len(items), grid, return_taps=True,
                                    route_overrides=overrides, input_counts=input_counts)
                                pair_main, pair_aux, _ = sparse_event_objective(
                                    net.event_heads, pair_info["layer_events"], y,
                                    seq_end, a.dmax, prefix_times, a.readout_fusion)
                                return float((pair_main + a.aux_weight * pair_aux).item())

                        # The factual trace is L00. Three matched downstream
                        # replays give L10, L01, L11 for the same utterance.
                        L00 = float(base_total)
                        L10 = shadow_route_loss([(k, event_a, receiver_a, True)])
                        L01 = shadow_route_loss([(k, event_b, receiver_b, True)])
                        L11 = shadow_route_loss([
                            (k, event_a, receiver_a, True),
                            (k, event_b, receiver_b, True)])
                        delta_a0 = L10 - L00
                        delta_b0 = L01 - L00
                        delta_a1 = L11 - L01
                        delta_b1 = L11 - L10
                        delta_a0 = float(np.clip(delta_a0, -a.cf_delta_clip, a.cf_delta_clip))
                        delta_b0 = float(np.clip(delta_b0, -a.cf_delta_clip, a.cf_delta_clip))
                        delta_a1 = float(np.clip(delta_a1, -a.cf_delta_clip, a.cf_delta_clip))
                        delta_b1 = float(np.clip(delta_b1, -a.cf_delta_clip, a.cf_delta_clip))

                        batch_ids, source_units, source_payload = info["route_inputs"][k]
                        def live_route_score(route_idx):
                            event_id = int(event_ids[route_idx])
                            receiver_id = int(receivers[route_idx])
                            source_unit = source_units[event_id]
                            score = (net.layers[k].q[receiver_id] * source_payload[event_id]).sum()
                            return score + net.layers[k].c[source_unit, receiver_id]

                        score_a, score_b = live_route_score(route_a), live_route_score(route_b)
                        q_a = torch.sigmoid(score_a.detach() / a.cf_sigma)
                        q_b = torch.sigmoid(score_b.detach() / a.cf_sigma)
                        dL_dscore_a = (q_a * (1.0 - q_a) / a.cf_sigma) * (
                            (1.0 - q_b) * delta_a0 + q_b * delta_a1)
                        dL_dscore_b = (q_b * (1.0 - q_b) / a.cf_sigma) * (
                            (1.0 - q_a) * delta_b0 + q_a * delta_b1)
                        pair_proxy = pair_proxy + a.cf_weight * (
                            dL_dscore_a * (score_a - score_a.detach())
                            + dL_dscore_b * (score_b - score_b.detach()))

                        pair_delta = L11 - L00
                        interaction = L11 - L10 - L01 + L00
                        cf_pair_delta_sum_ep += pair_delta
                        cf_pair_delta_sq_ep += pair_delta * pair_delta
                        cf_pair_interaction_sum_ep += interaction
                        cf_pair_interaction_sq_ep += interaction * interaction
                        cf_pair_helpful_ep += int(pair_delta < 0.0)
                        cf_pair_synergy_ep += int(interaction < 0.0)
                        cf_pair_shadow_count += 1
                        cf_pair_shadow_ep += 1
                        cf_pair_shadow_by_layer_ep[k] += 1
                    if eligible_pairs:
                        cf_proxy = cf_proxy + pair_proxy / len(eligible_pairs)
                cf_grads = None
                if (cf_shadow_count or cf_pair_shadow_count or cf_route_swap_count) and a.cf_lr:
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
            if option_pending is not None:
                suffix = option_pending["suffix"]
                root_rng = torch.random.get_rng_state()
                root_cuda_rng = (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None)
                measure_suffix_progress = (
                    not a.cf_spike_option_rollout_value
                    and not (a.cf_spike_option_local_utility
                             and a.cf_spike_option_weight == 0.0))
                if measure_suffix_progress:
                    factual_progress, factual_grad_norm = finite_progress_from_grads(
                        option_pending["root_loss"], suffix, option_pending["root_grads"],
                        option_pending["root_replay"], a.cf_spike_virtual_lr,
                        a.cf_spike_virtual_clip)
                    after_root_rng = torch.random.get_rng_state()
                    after_root_cuda_rng = (torch.cuda.get_rng_state_all()
                                           if torch.cuda.is_available() else None)
                    torch.random.set_rng_state(root_rng)
                    if root_cuda_rng is not None:
                        torch.cuda.set_rng_state_all(root_cuda_rng)
                    branch_progress, branch_grad_norm = finite_progress_from_grads(
                        option_pending["branch_loss"], suffix, option_pending["branch_grads"],
                        option_pending["branch_replay"], a.cf_spike_virtual_lr,
                        a.cf_spike_virtual_clip)
                    torch.random.set_rng_state(after_root_rng)
                    if after_root_cuda_rng is not None:
                        torch.cuda.set_rng_state_all(after_root_cuda_rng)
                else:
                    factual_progress = 0.0
                    factual_grad_norm = math.sqrt(sum(
                        float(g.detach().square().sum())
                        for g in option_pending["root_grads"] if g is not None))
                    branch_progress = 0.0
                    branch_grad_norm = math.sqrt(sum(
                        float(g.detach().square().sum())
                        for g in option_pending["branch_grads"] if g is not None))
                immediate_advantage = float(
                    option_pending["root_loss"].detach()
                    - option_pending["branch_loss"].detach())
                learning_advantage = branch_progress - factual_progress
                scalar_utility = (immediate_advantage
                                  + a.cf_spike_option_weight * learning_advantage)
                actions = option_pending["actions"]
                local_action_utilities = None
                if a.cf_spike_option_local_utility:
                    contexts = option_pending["causal_contexts"]
                    root_grad_by_id = option_pending["root_grad_by_id"]
                    root_replay = option_pending["root_replay"]
                    local_action_utilities = []
                    for action_index, (action, child) in enumerate(
                            zip(actions, contexts)):
                        if action_index == 0:
                            parent_loss = option_pending["root_loss"]
                            parent_grad_by_id = root_grad_by_id
                            parent_replay = root_replay
                        else:
                            parent = contexts[action_index - 1]
                            parent_loss = parent["loss"]
                            parent_grad_by_id = parent["grad_by_id"]
                            parent_replay = parent["replay"]
                        suffix_params = child["suffix"]
                        parent_grads = tuple(
                            parent_grad_by_id.get(id(param))
                            for param in suffix_params)
                        local_immediate = float(
                            parent_loss.detach() - child["loss"].detach())
                        if a.cf_spike_option_rollout_value:
                            parent_info = info if action_index == 0 else parent["info"]
                            parent_overrides = (() if action_index == 0
                                                else parent["overrides"])
                            parent_continuations = []
                            child_continuations = []
                            for _ in range(a.cf_spike_option_rollouts):
                                common_seed = int(spike_option_rollout_rng.integers(
                                    0, 2**32 - 1))
                                parent_loss_rollout, _, parent_future_actions = (
                                    continue_spike_option_rollout(
                                        net, eb, ei, et, input_counts, len(items),
                                        grid, y, seq_end, a.depth, a.dmax,
                                        prefix_times, parent_info, parent_overrides,
                                        action["layer"] + 1, target_batch, a,
                                        np.random.default_rng(common_seed)))
                                child_loss_rollout, _, child_future_actions = (
                                    continue_spike_option_rollout(
                                        net, eb, ei, et, input_counts, len(items),
                                        grid, y, seq_end, a.depth, a.dmax,
                                        prefix_times, child["info"], child["overrides"],
                                        action["layer"] + 1, target_batch, a,
                                        np.random.default_rng(common_seed)))
                                parent_continuations.append(parent_loss_rollout)
                                child_continuations.append(child_loss_rollout)
                                option_rollout_actions_ep += (
                                    len(parent_future_actions) + len(child_future_actions))
                                option_rollout_count_ep += 2
                            parent_default = float(parent_loss.detach())
                            child_default = float(child["loss"].detach())
                            temperature = a.cf_spike_option_backup_temperature
                            parent_value = min(parent_default, soft_minimum(
                                [parent_default, *parent_continuations], temperature))
                            child_value = min(child_default, soft_minimum(
                                [child_default, *child_continuations], temperature))
                            parent_optionality = parent_default - parent_value
                            child_optionality = child_default - child_value
                            optionality_gain = child_optionality - parent_optionality
                            continuation_gain = parent_value - child_value
                            epsilon = a.cf_spike_option_improvement_epsilon
                            parent_beneficial_mass = float(np.mean(
                                np.asarray(parent_continuations)
                                <= parent_default - epsilon))
                            child_beneficial_mass = float(np.mean(
                                np.asarray(child_continuations)
                                <= child_default - epsilon))
                            beneficial_mass_delta = (child_beneficial_mass
                                                     - parent_beneficial_mass)
                            local_learning = 0.0
                            local_utility = (local_immediate
                                             + a.cf_spike_option_weight * optionality_gain)
                            option_local_optionality_sum_ep += optionality_gain
                            option_local_continuation_gain_sum_ep += continuation_gain
                            option_local_beneficial_mass_delta_sum_ep += beneficial_mass_delta
                            option_local_optionality_by_layer_ep[action["layer"]] += optionality_gain
                            option_local_beneficial_mass_delta_by_layer_ep[
                                action["layer"]] += beneficial_mass_delta
                        else:
                            if a.cf_spike_option_weight == 0.0:
                                parent_progress = child_progress = 0.0
                            else:
                                parent_progress, _ = finite_progress_from_grads(
                                    parent_loss, suffix_params, parent_grads,
                                    parent_replay, a.cf_spike_virtual_lr,
                                    a.cf_spike_virtual_clip)
                                child_progress, _ = finite_progress_from_grads(
                                    child["loss"], suffix_params, child["grads"],
                                    child["replay"], a.cf_spike_virtual_lr,
                                    a.cf_spike_virtual_clip)
                            local_learning = child_progress - parent_progress
                            local_utility = (local_immediate
                                             + a.cf_spike_option_weight * local_learning)
                        local_action_utilities.append(local_utility)
                        layer_id = action["layer"]
                        option_local_utility_by_layer_ep[layer_id] += local_utility
                        option_local_actions_by_layer_ep[layer_id] += 1
                        option_local_immediate_sum_ep += local_immediate
                        option_local_learning_sum_ep += local_learning
                    option_local_utility_sum_ep += sum(local_action_utilities)
                    option_local_utility_actions_ep += len(local_action_utilities)
                option_paths_ep += 1
                option_actions_ep += len(actions)
                option_immediate_ep += immediate_advantage
                option_learning_ep += learning_advantage
                option_utility_ep += (sum(local_action_utilities)
                                      if local_action_utilities is not None
                                      else scalar_utility)
                option_grad_norm_ep += 0.5 * (factual_grad_norm + branch_grad_norm)
                target_batch = option_pending["target_batch"]
                root_event_counts = [
                    int((layer_events[0] == target_batch).sum())
                    for layer_events in option_pending["root_events"]]
                branch_event_counts = [
                    int((layer_events[0] == target_batch).sum())
                    for layer_events in option_pending["branch_events"]]
                option_hidden_delta_ep += (
                    np.asarray(branch_event_counts) - np.asarray(root_event_counts))
                for action_index, action in enumerate(actions):
                    layer_id, unit_id = action["layer"], action["unit"]
                    margin = action["margin"]
                    fire_probability = 1.0 / (1.0 + math.exp(
                        -margin / a.cf_spike_option_sigma))
                    sensitivity = (fire_probability * (1.0 - fire_probability)
                                   / a.cf_spike_option_sigma)
                    action_utility = (
                        local_action_utilities[action_index]
                        if local_action_utilities is not None
                        else scalar_utility / len(actions))
                    raw_delta = (-a.cf_spike_option_lr * sensitivity
                                 * action_utility)
                    delta = float(np.clip(
                        raw_delta, -a.cf_spike_option_step_clip,
                        a.cf_spike_option_step_clip))
                    option_threshold_deltas.append((layer_id, unit_id, delta))
                    option_action_by_layer_ep[layer_id] += 1
            gsum += np.asarray([grad_norm(l) for l in net.layers])
            if option_pending is not None and a.cf_spike_suffix_weight > 0:
                correction_scale = a.cf_spike_suffix_weight / max(len(items), 1)
                correction_by_id = {}
                correction_sq = 0.0
                substitutions = []
                if a.cf_spike_suffix_causal:
                    contexts = option_pending["causal_contexts"]
                    for layer_index, layer in enumerate(net.layers):
                        eligible = [context for context in contexts
                                    if context["last_layer"] < layer_index]
                        if eligible:
                            chosen = max(eligible,
                                         key=lambda context: context["last_layer"])
                            for param in layer.parameters():
                                substitutions.append((
                                    param,
                                    option_pending["root_grad_by_id"].get(id(param)),
                                    chosen["grad_by_id"].get(id(param))))
                    chosen = max(contexts,
                                 key=lambda context: context["last_layer"])
                    for param in net.event_heads[-1].parameters():
                        substitutions.append((
                            param,
                            option_pending["root_grad_by_id"].get(id(param)),
                            chosen["grad_by_id"].get(id(param))))
                else:
                    substitutions = list(zip(
                        option_pending["suffix"], option_pending["root_grads"],
                        option_pending["branch_grads"]))
                for param, factual_grad, branch_grad in substitutions:
                    if factual_grad is None and branch_grad is None:
                        continue
                    if factual_grad is None:
                        correction = branch_grad.detach().clone()
                    elif branch_grad is None:
                        correction = -factual_grad.detach().clone()
                    else:
                        correction = branch_grad.detach() - factual_grad.detach()
                    correction.mul_(correction_scale)
                    correction_by_id[id(param)] = correction
                    correction_sq += float(correction.square().sum())
                    if param.grad is None:
                        param.grad = correction
                    else:
                        param.grad.add_(correction)
                option_suffix_batches_ep += 1
                option_suffix_correction_norm_ep += math.sqrt(correction_sq)
                for layer_id, layer in enumerate(net.layers):
                    layer_sq = sum(
                        float(correction_by_id[id(param)].square().sum())
                        for param in layer.parameters()
                        if id(param) in correction_by_id)
                    option_suffix_correction_by_layer_ep[layer_id] += math.sqrt(layer_sq)
                head_sq = sum(
                    float(correction_by_id[id(param)].square().sum())
                    for param in net.event_heads[-1].parameters()
                    if id(param) in correction_by_id)
                option_suffix_correction_by_layer_ep[a.depth] += math.sqrt(head_sq)
            gsum_with_spike_cf += np.asarray([grad_norm(l) for l in net.layers])
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            if option_threshold_deltas:
                with torch.no_grad():
                    for layer_id, unit_id, delta in option_threshold_deltas:
                        offsets = net.layers[layer_id].theta_offsets
                        old_value = offsets[unit_id].clone()
                        offsets[unit_id].add_(delta).clamp_(
                            -a.cf_spike_option_bound, a.cf_spike_option_bound)
                        option_threshold_step_ep += float(
                            (offsets[unit_id] - old_value).abs())
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
        fusion_ablation_ok = np.zeros(a.depth, dtype=np.int64)
        fusion_single_ok = np.zeros(a.depth, dtype=np.int64)
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
        active_utterances = np.zeros(a.depth, dtype=np.int64)
        support_nesting_violations = 0

        def update_support_counts(layer_events, batch_size):
            nonlocal support_nesting_violations
            active_masks = []
            for event_b, _, _, _ in layer_events:
                active = torch.zeros(batch_size, dtype=torch.bool, device=event_b.device)
                if event_b.numel():
                    active[event_b.unique()] = True
                active_masks.append(active)
            active_utterances[:] += np.asarray(
                [int(active.sum()) for active in active_masks], dtype=np.int64)
            support_nesting_violations += sum(
                int((active_masks[k + 1] & ~active_masks[k]).sum())
                for k in range(len(active_masks) - 1))
        for layer in net.layers:
            layer.sent.zero_()
        if not event_readout:
            net.ro.sent.zero_()
        with torch.no_grad():
            for i0 in range(0, len(ev), a.bs):
                items = ev[i0:i0 + a.bs]
                eb, ei, et, input_counts, y, tmax, seq_end = batch_to_events(
                    items, a.bands, 0, eval_subset_rng, 0.0)
                grid = tmax + (a.depth + 1) * math.ceil(a.dmax) + 60
                _, info = net(eb, ei, et, len(items), grid, return_taps=True,
                              input_counts=input_counts)
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

                    if a.readout_fusion == "all_depths":
                        full_logits = [
                            head.prefix_logits(info["layer_events"][k], seq_end,
                                               a.depth + 1, a.dmax, queries)
                            for k, head in enumerate(net.event_heads)]
                        final_rows = [sum(full_logits[k][bi]
                                          for k in range(len(net.event_heads)))
                                      for bi in range(len(items))]
                    else:
                        final_rows = head_logits[-1]
                    final_scores = torch.stack([rows[-1] for rows in final_rows])
                    terminal_pred = final_scores.argmax(-1)
                    max_ok += int((terminal_pred == y).sum())
                    if a.readout_fusion == "all_depths":
                        for k in range(a.depth):
                            ablated = [final_rows[bi] - full_logits[k][bi]
                                       for bi in range(len(items))]
                            single = [full_logits[k][bi]
                                      for bi in range(len(items))]
                            fusion_ablation_ok[k] += sum(
                                int(rows[-1].argmax() == y[bi])
                                for bi, rows in enumerate(ablated))
                            fusion_single_ok[k] += sum(
                                int(rows[-1].argmax() == y[bi])
                                for bi, rows in enumerate(single))
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
                    for bi in range(len(items)):
                        deadline = float(seq_end[bi] + (a.depth + 1) * a.dmax + 60.0)
                        for ti, threshold in enumerate(threshold_grid):
                            decision, peak = net.infer_event_readout(
                                info["layer_events"], bi, float(seq_end[bi]),
                                a.dmax, float(threshold), a.race_logit_temperature)
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
                    update_support_counts(info["layer_events"], len(items))
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
                update_support_counts(info["layer_events"], len(items))
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
               "event_support_coverage": (active_utterances / len(ev)).round(4).tolist(),
               "spikes_per_active_utterance": (
                   st["spikes"] / np.maximum(active_utterances, 1)).round(2).tolist(),
               "support_nesting_violations": int(support_nesting_violations),
               "input_count_projection_norm": (
                   round(float(net.count_proj.weight.detach().norm()), 8)
                   if net.count_proj is not None else None),
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
            if a.readout_fusion == "all_depths":
                readout_updates = float(st["msgs"][a.depth:2 * a.depth].sum())
            else:
                readout_updates = float(st["msgs"][-1])
            row["readout_fusion"] = a.readout_fusion
            row["sparse_readout_edge_updates_per_utterance"] = round(
                readout_updates / len(ev), 1)
            row["readout_edge_updates_by_layer_per_utterance"] = (
                st["msgs"][a.depth:2 * a.depth] / len(ev)).round(1).tolist()
            if a.trainable_thresholds:
                row["threshold_offset_mean_by_layer"] = [
                    round(float(layer.theta_offsets.detach().mean()), 7)
                    for layer in net.layers]
                row["threshold_offset_max_abs_by_layer"] = [
                    round(float(layer.theta_offsets.detach().abs().max()), 7)
                    for layer in net.layers]
            if a.cf_spike_option_updates:
                row["spike_option_paths_per_epoch"] = int(option_paths_ep)
                row["spike_option_actions_per_epoch"] = int(option_actions_ep)
                row["spike_option_actions_by_layer"] = option_action_by_layer_ep.tolist()
                row["spike_option_proposal_candidates_sum"] = int(option_candidates_ep)
                row["spike_option_mean_immediate_advantage"] = round(
                    option_immediate_ep / max(option_paths_ep, 1), 7)
                row["spike_option_mean_suffix_learning_advantage"] = round(
                    option_learning_ep / max(option_paths_ep, 1), 7)
                row["spike_option_mean_scalar_utility"] = round(
                    option_utility_ep / max(option_paths_ep, 1), 7)
                row["spike_option_local_utility_actions"] = int(
                    option_local_utility_actions_ep)
                if a.cf_spike_option_local_utility:
                    row["spike_option_mean_local_scalar_utility"] = round(
                        option_local_utility_sum_ep
                        / max(option_local_utility_actions_ep, 1), 7)
                    row["spike_option_local_immediate_sum"] = round(
                        option_local_immediate_sum_ep, 7)
                    row["spike_option_local_learning_sum"] = round(
                        option_local_learning_sum_ep, 7)
                    row["spike_option_local_utility_sum_by_layer"] = (
                        option_local_utility_by_layer_ep.round(7).tolist())
                    if a.cf_spike_option_rollout_value:
                        row["spike_option_metric"] = "proposal_conditioned_continuation_reserve"
                        row["spike_option_rollout_count"] = int(option_rollout_count_ep)
                        row["spike_option_rollout_mean_future_actions"] = round(
                            option_rollout_actions_ep / max(option_rollout_count_ep, 1), 4)
                        row["spike_option_mean_delta_reserve"] = round(
                            option_local_optionality_sum_ep
                            / max(option_local_utility_actions_ep, 1), 7)
                        row["spike_option_mean_continuation_value_gain"] = round(
                            option_local_continuation_gain_sum_ep
                            / max(option_local_utility_actions_ep, 1), 7)
                        row["spike_option_mean_delta_beneficial_continuation_mass"] = round(
                            option_local_beneficial_mass_delta_sum_ep
                            / max(option_local_utility_actions_ep, 1), 5)
                        row["spike_option_improvement_epsilon"] = (
                            a.cf_spike_option_improvement_epsilon)
                        row["spike_option_delta_reserve_sum_by_layer"] = (
                            option_local_optionality_by_layer_ep.round(7).tolist())
                        row["spike_option_delta_beneficial_mass_sum_by_layer"] = (
                            option_local_beneficial_mass_delta_by_layer_ep.round(5).tolist())
                        row["spike_option_backup_temperature"] = (
                            a.cf_spike_option_backup_temperature)
                    row["spike_option_local_actions_by_layer"] = (
                        option_local_actions_by_layer_ep.tolist())
                row["spike_option_mean_suffix_gradient_norm"] = round(
                    option_grad_norm_ep / max(option_paths_ep, 1), 7)
                row["spike_option_threshold_update_l1"] = round(
                    option_threshold_step_ep, 7)
                row["spike_option_hidden_event_delta_sum_by_layer"] = (
                    option_hidden_delta_ep.astype(int).tolist())
                row["spike_option_learning_weight"] = a.cf_spike_option_weight
                if a.cf_spike_suffix_weight > 0:
                    row["spike_option_suffix_weight"] = a.cf_spike_suffix_weight
                    row["spike_option_suffix_substitution_batches"] = int(
                        option_suffix_batches_ep)
                    row["spike_option_suffix_correction_norm_per_batch"] = round(
                        option_suffix_correction_norm_ep / max(option_suffix_batches_ep, 1), 7)
                    row["spike_option_suffix_correction_norm_by_layer"] = (
                        option_suffix_correction_by_layer_ep
                        / max(option_suffix_batches_ep, 1)).round(7).tolist()
                    row["layer_grad_norms_with_spike_suffix_cf"] = (
                        gsum_with_spike_cf / nb).round(7).tolist()
            if a.readout_fusion == "all_depths":
                row["readout_branch_ablation_accuracy"] = (
                    fusion_ablation_ok / len(ev)).round(4).tolist()
                row["readout_branch_standalone_accuracy"] = (
                    fusion_single_ok / len(ev)).round(4).tolist()
            row["counterfactual_near_routes_per_epoch"] = int(cf_eligible_ep)
            row["counterfactual_route_shadows_per_epoch"] = int(cf_shadow_ep)
            row["counterfactual_route_pairs_eligible_per_epoch"] = int(cf_pair_eligible_ep)
            row["counterfactual_route_pairs_shadowed_per_epoch"] = int(cf_pair_shadow_ep)
            row["counterfactual_route_pair_replays_per_epoch"] = int(3 * cf_pair_shadow_ep)
            row["counterfactual_route_pair_sampler"] = a.cf_pair_sampling
            row["counterfactual_route_pair_candidates_by_layer"] = cf_pair_candidate_by_layer_ep.tolist()
            row["counterfactual_route_pair_mean_inclusion_probability"] = round(
                cf_pair_sample_prob_sum_ep / max(cf_pair_shadow_ep, 1), 8)
            row["counterfactual_route_pair_min_inclusion_probability"] = round(
                cf_pair_sample_prob_min_ep, 8) if cf_pair_shadow_ep else None
            row["counterfactual_route_pair_max_inclusion_probability"] = round(
                cf_pair_sample_prob_max_ep, 8) if cf_pair_shadow_ep else None
            row["counterfactual_pair_mean_L11_minus_L00"] = round(
                cf_pair_delta_sum_ep / max(cf_pair_shadow_ep, 1), 6)
            row["counterfactual_pair_delta_std"] = round(math.sqrt(max(
                cf_pair_delta_sq_ep / max(cf_pair_shadow_ep, 1)
                - (cf_pair_delta_sum_ep / max(cf_pair_shadow_ep, 1)) ** 2, 0.0)), 6)
            row["counterfactual_pair_fraction_joint_opening_improves"] = round(
                cf_pair_helpful_ep / max(cf_pair_shadow_ep, 1), 4)
            row["counterfactual_pair_mean_interaction_gamma"] = round(
                cf_pair_interaction_sum_ep / max(cf_pair_shadow_ep, 1), 6)
            row["counterfactual_pair_interaction_std"] = round(math.sqrt(max(
                cf_pair_interaction_sq_ep / max(cf_pair_shadow_ep, 1)
                - (cf_pair_interaction_sum_ep / max(cf_pair_shadow_ep, 1)) ** 2, 0.0)), 6)
            row["counterfactual_pair_fraction_synergistic_gamma_negative"] = round(
                cf_pair_synergy_ep / max(cf_pair_shadow_ep, 1), 4)
            row["counterfactual_pair_shadow_counts_by_layer"] = cf_pair_shadow_by_layer_ep.tolist()
            row["route_topk"] = a.route_topk
            row["counterfactual_route_swap_eligible_by_layer"] = cf_route_swap_eligible_ep.tolist()
            row["counterfactual_route_swap_shadowed_by_layer"] = cf_route_swap_shadow_ep.tolist()
            row["counterfactual_route_swap_replays"] = int(cf_route_swap_shadow_ep.sum())
            row["counterfactual_route_swap_mean_alternative_minus_current_loss_by_layer"] = [
                round(float(cf_route_swap_delta_sum_ep[k] / cf_route_swap_shadow_ep[k]), 7)
                if cf_route_swap_shadow_ep[k] else None for k in range(a.depth)]
            row["counterfactual_route_swap_fraction_alternative_helpful_by_layer"] = [
                round(float(cf_route_swap_helpful_ep[k] / cf_route_swap_shadow_ep[k]), 4)
                if cf_route_swap_shadow_ep[k] else None for k in range(a.depth)]
            row["counterfactual_route_swap_mean_score_gap_by_layer"] = [
                round(float(cf_route_swap_gap_sum_ep[k] / cf_route_swap_shadow_ep[k]), 6)
                if cf_route_swap_shadow_ep[k] else None for k in range(a.depth)]
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
    pair_tag = (f"_cpairs{a.cf_pairs_per_batch}_tw{a.cf_pair_window_ms:g}"
                if event_readout and a.cf_pairs_per_batch else "")
    pair_sampling_tag = {
        "global": "",
        "layer_balanced": "_pslayerbalanced",
        "late_balanced": "_pslatebalanced",
    }[a.cf_pair_sampling] if event_readout else ""
    route_tag = (f"_topk{a.route_topk}_swaps{a.cf_route_swaps_per_layer}"
                 f"_sb{a.cf_route_swap_band:g}" if event_readout and a.route_topk else "")
    fusion_tag = f"_rf{a.readout_fusion}" if event_readout else ""
    count_tag = "_cntadd" if a.input_count_payload == "additive" else ""
    skip_tag = "_skfirst" if a.early_event_skip else ""
    rng_tag = "_rngsplit" if a.rng_protocol == "split" else ""
    run_tag = f"_{a.run_tag}" if a.run_tag else ""
    path = os.path.join(OUT, f"deep_d{a.d}_n{a.n}_M{a.M1}-{a.M}_depth{a.depth}_aux{a.aux_weight:g}_obj{a.objective}{fusion_tag}{count_tag}{skip_tag}{rng_tag}{run_tag}{cf_tag}{pair_tag}{pair_sampling_tag}{route_tag}_spk_s{a.seed}.json")
    if a.save_checkpoint:
        checkpoint_path = os.path.splitext(path)[0] + ".pt"
        torch.save({"args": vars(a), "model_state_dict": net.state_dict()}, checkpoint_path)
        res["checkpoint"] = checkpoint_path
    with open(path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
