"""E77: a time-vector character language model (THEORY §105-§107) on text8.

Characters are events at half-integer times (character j at j + 0.5, so the state read at integer time k + 1 has seen
characters 0..k and nothing later). Architecture:
  layer 1   spiking time-vector units (§105): content-gated, content-delayed messages from the characters; threshold
            firing (never earlier than the detecting step: causal); snapshot payloads
  deeper    spiking time-vector units with sparse prior-event and raw-input routes
  event KV  causal modern-Hopfield update over emitted events; retrieved payloads pass to later event/readout layers
  state     non-spiking time-vector units read at every character boundary: h_k (the recurrent memory of §107(a))
  token retrieval delay-coded attention (§107(c)): query from (h_k, x_k); keys and values from earlier positions j < k, values
            carry the character that followed (x_{j+1}); exact softmax over the keys within Delta of the best (the race opens
            a window of length kappa * Delta); simulated as that masked softmax, which the lemma makes identical; work = keys
            inside the window
  readout   softmax over 27 characters from (h_k, e(x_k), retrieval output)
Trained by gradients through spike times, payloads and delays (as E74); early stopping on 200k validation characters;
test on the same 1M characters as E62-E66 and E64 (windows of L with L/2 context). Reports bits per character, and messages,
spikes and retrieved keys per character.
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
import e62_charlm as S1  # noqa: E402
from e74_time_vector_net import TAU_R, TVLayer  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e77")
A = 27


class AdaptiveMemoryLayer(nn.Module):
    """Causal retrieval plus a learned recurrent-memory alternative at the retrieval interface.

    The attention mode is exactly recoverable (gate -> 1, interaction factors -> 0). The
    state mode is a content-gated diagonal recurrence and costs linear work in sequence
    length. A low-rank query/key feature can modify both retrieval timing and its value.
    """
    def __init__(self, heads, dh, delta, rank=0, gate_temperature=0.25):
        super().__init__()
        self.heads, self.dh, self.delta = heads, dh, delta
        self.rank, self.gate_temperature = rank, gate_temperature
        if rank:
            self.q_factor = nn.Parameter(torch.randn(heads, dh, rank) / math.sqrt(dh))
            self.k_factor = nn.Parameter(torch.randn(heads, dh, rank) / math.sqrt(dh))
            # Zero initialization preserves ordinary dot-product attention at start.
            self.score_mix = nn.Parameter(torch.zeros(heads, rank))
            self.value_mix = nn.Parameter(torch.zeros(heads, rank, dh))
        self.write = nn.Linear(dh, dh)
        self.forget = nn.Linear(dh, dh)
        nn.init.zeros_(self.write.bias)
        nn.init.constant_(self.forget.bias, 1.5)
        self.state_read = nn.Linear(dh, dh, bias=False)
        self.mode_gate = nn.Sequential(nn.Linear(2 * dh, dh), nn.GELU(), nn.Linear(dh, 1))
        nn.init.zeros_(self.mode_gate[-1].weight)
        nn.init.zeros_(self.mode_gate[-1].bias)  # both routes get gradient initially

    def forward(self, q, k, v):
        # q, k, v: [batch, head, position, feature]. State at i sees keys < i.
        B, H, L, D = q.shape
        states = []
        state = v.new_zeros((B, H, D))
        for t in range(L):
            states.append(state)
            kt, vt = k[:, :, t], v[:, :, t]
            decay = torch.sigmoid(self.forget(kt))
            write = torch.sigmoid(self.write(kt))
            state = decay * state + write * vt
        state = torch.stack(states, dim=2)  # [B,H,L,D], strictly causal

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(D)
        qf = kf = None
        if self.rank:
            qf = torch.einsum("bhld,hdr->bhlr", q, self.q_factor)
            kf = torch.einsum("bhld,hdr->bhlr", k, self.k_factor)
            # A learned low-rank bilinear term co-trains query/key compatibility and delay.
            weighted_k = kf * self.score_mix[None, :, None, :]
            scores = scores + (qf @ weighted_k.transpose(-1, -2)) / math.sqrt(self.rank)

        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device), diagonal=-1)
        causal = causal[None, None]
        scores = scores.masked_fill(~causal, float("-inf"))
        has_keys = causal.any(-1, keepdim=True)
        top = scores.amax(-1, keepdim=True).detach()
        top = torch.where(has_keys, top, torch.zeros_like(top))
        # Delay δ = κ(top-score); its exponential arrival weight is softmax(score).
        # During training, a smooth near-miss gate gives keys just outside the
        # hard window a gradient. Evaluation uses a hard window for aggregation.
        if math.isinf(self.delta):
            near = causal.expand_as(scores).to(scores.dtype)
        elif self.training:
            near = torch.sigmoid((scores - top + self.delta) / self.gate_temperature)
            near = torch.where(causal, near, torch.zeros_like(near))
        else:
            near = ((scores >= top - self.delta) & causal).to(scores.dtype)
        weights = torch.exp(scores - top) * near
        denom = weights.sum(-1, keepdim=True)
        attn = weights / denom.clamp_min(torch.finfo(weights.dtype).tiny)
        attn = torch.where(has_keys, attn, torch.zeros_like(attn))
        retrieved = attn @ v
        if self.rank:
            # Expected query-key interaction changes the transmitted value without
            # materializing a [B,H,L,L,R] tensor.
            expected_k = attn @ kf
            pair = qf * expected_k
            retrieved = retrieved + torch.einsum("bhlr,hrd->bhld", pair, self.value_mix)

        memory = self.state_read(state)
        alpha = torch.sigmoid(self.mode_gate(torch.cat([q, state], -1)))
        out = alpha * retrieved + (1.0 - alpha) * memory
        active = (near > 0.5).sum(-1).float().mean()
        return out, {"keys_per_query": float(active.detach()), "attention_mix": float(alpha.detach().mean())}


class EventHopfieldLayer(nn.Module):
    """Causal key/value Hopfield updates over emitted events, independently per sequence.

    Only existing event payloads participate. Exact mode is the trainability reference;
    optional top-k sparsifies aggregation after scoring and therefore is not claimed to
    reduce candidate-search cost.
    """
    def __init__(self, d, depth, candidate_k=0):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(2 * d, 1)
        self.log_delay_gain = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        # Residual scaling by 1/L gives a depth-uniform Jacobian bound when the
        # per-layer Hopfield correction has bounded gain (THEORY §107(g)).
        self.scale = 1.0 / depth
        self.candidate_k = candidate_k

    def forward(self, batch, times, payload):
        S, d = payload.shape
        if S < 2:
            return payload, {"hopfield_pairs": 0.0, "hopfield_active": 0.0,
                             "hopfield_gate": 0.0, "hopfield_update_norm": 0.0,
                             "hopfield_entropy": 0.0, "hopfield_score_std": 0.0,
                             "hopfield_key_diam_upper": 0.0, "hopfield_value_diam_upper": 0.0,
                             "hopfield_query_jacobian_bound": 0.0,
                             "hopfield_payload_jacobian_bound": 0.0,
                             "hopfield_scaled_jacobian_bound": 0.0,
                             "hopfield_max_key_fanout": 0.0, "hopfield_delay_gain": 1.0}
        retrieved = torch.zeros_like(payload)
        scored = active = 0
        entropies, score_spreads, key_fanouts = [], [], []
        q_all, k_all, v_all = self.q(payload), self.k(payload), self.v(payload)
        delay_gain = self.log_delay_gain.clamp(-6.9, 6.9).exp()
        for b in batch.unique(sorted=True):
            ids = torch.nonzero(batch == b, as_tuple=False).flatten()
            if ids.numel() < 2:
                continue
            tb = times[ids]
            # Rows are query events; columns are strictly earlier memory events.
            past = tb[None, :] < tb[:, None]
            rows = torch.nonzero(past.any(-1), as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            scores = (q_all[ids[rows]] @ k_all[ids].T) / math.sqrt(d)
            mask = past[rows]
            scored += int(mask.numel())
            row_spreads = [scores[r, mask[r]].std(unbiased=False)
                           for r in range(scores.shape[0]) if int(mask[r].sum()) > 1]
            if row_spreads:
                score_spreads.append(torch.stack(row_spreads).mean())
            if self.candidate_k > 0 and self.candidate_k < ids.numel():
                scores = scores.masked_fill(~mask, float("-inf"))
                topn = min(self.candidate_k, ids.numel())
                ix = scores.topk(topn, dim=-1).indices
                select = torch.zeros_like(mask).scatter_(1, ix, True) & mask
                mask = select
                scores = scores.masked_fill(~mask, float("-inf"))
            # A score is an arrival-time advantage: δ_ij = γ(s_i,max - s_ij).
            # Unit-time-constant decay exp(-δ) gives softmax inverse temperature γ.
            top = scores.masked_fill(~mask, float("-inf")).amax(-1, keepdim=True).detach()
            delay = delay_gain * (top - scores).clamp_min(0)
            weights = torch.exp(-delay).masked_fill(~mask, 0.0)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
            got = weights @ v_all[ids]
            retrieved[ids[rows]] = got
            active += int(mask.sum().item())
            entropies.append((-(weights * weights.clamp_min(1e-12).log()).sum(-1)).mean())
            key_fanouts.append(weights.sum(0).max())
        gate = torch.sigmoid(self.gate(torch.cat([payload, retrieved], -1)))
        delta = self.out(retrieved)
        updated = payload + self.scale * gate * delta
        with torch.no_grad():
            k_diam_upper = 2.0 * float(k_all.detach().norm(dim=-1).max()) if S else 0.0
            v_diam_upper = 2.0 * float(v_all.detach().norm(dim=-1).max()) if S else 0.0
            # Sequence-level local Jacobian certificate, conditional on fixed event
            # and candidate masks. A key can affect many later queries, so include
            # its maximum incoming attention mass (fan-out), not only D_q y.
            beta = float(delay_gain.detach()) / math.sqrt(d)
            wq = float(torch.linalg.matrix_norm(self.q.weight.detach(), ord=2))
            wk = float(torch.linalg.matrix_norm(self.k.weight.detach(), ord=2))
            wv = float(torch.linalg.matrix_norm(self.v.weight.detach(), ord=2))
            wo = float(torch.linalg.matrix_norm(self.out.weight.detach(), ord=2))
            wg = self.gate.weight.detach().reshape(-1)
            wg_payload = float(wg[:d].norm())
            wg_retrieval = float(wg[d:].norm())
            qmax = float(q_all.detach().norm(dim=-1).max())
            key_diam = k_diam_upper
            value_diam = v_diam_upper
            # For R_i = sum_j p_ij v_j, off-diagonal block norms are bounded by
            # p_ij (||W_v|| + beta D_v ||W_k|| max||q||). Row mass is one;
            # column mass is the max-key-fanout statistic above.
            q_bound = beta * key_diam * value_diam * wq / 4.0
            source_bound = wv + beta * value_diam * wk * qmax
            fanout = max((float(v.detach()) for v in key_fanouts), default=0.0)
            row_bound = q_bound + source_bound
            col_bound = q_bound + source_bound * fanout
            retrieval_jacobian_bound = math.sqrt(max(row_bound, 0.0) * max(col_bound, 0.0))
            value_radius = float(v_all.detach().norm(dim=-1).max())
            correction_r_gain = wo * (1.0 + value_radius * wg_retrieval / 4.0)
            correction_x_gain = wo * value_radius * wg_payload / 4.0
            payload_jacobian_bound = (correction_r_gain * retrieval_jacobian_bound
                                      + correction_x_gain)
            scaled_jacobian_bound = self.scale * payload_jacobian_bound
        return updated, {
            "hopfield_pairs": float(scored), "hopfield_active": float(active),
            "hopfield_gate": float(gate.detach().mean()),
            "hopfield_update_norm": float(delta.detach().norm() / math.sqrt(max(S, 1))),
            "hopfield_entropy": float(torch.stack(entropies).mean().detach()) if entropies else 0.0,
            "hopfield_score_std": float(torch.stack(score_spreads).mean().detach()) if score_spreads else 0.0,
            "hopfield_key_diam_upper": k_diam_upper,
            "hopfield_value_diam_upper": v_diam_upper,
            "hopfield_query_jacobian_bound": beta * key_diam * value_diam / 4.0,
            "hopfield_payload_jacobian_bound": payload_jacobian_bound,
            "hopfield_scaled_jacobian_bound": scaled_jacobian_bound,
            "hopfield_max_key_fanout": fanout,
            "hopfield_delay_gain": float(delay_gain.detach()),
        }


class TVLM(nn.Module):
    def __init__(self, d, n, M1, M2, Mr, fan2, dmax, w_sd, retrieval, heads, dh, delta, seed=0,
                 coupling_rank=0, depth=2, skip_fan=0.125, event_hopfield=True, event_candidate_k=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(A, d)
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth
        self.M1 = M1
        self.M2 = M2
        self.event_hopfield = event_hopfield
        kw = dict(causal=True, tau_range=(1.0, 200.0), td0=1.0)
        layers = [TVLayer(A, M1, d, d, n, dmax, None, True, w_sd[0], **kw)]
        for i in range(1, depth):
            # Each stage retains every earlier event and appends its own new events.
            # It also receives a sparse raw-input skip, so input channels grow with depth.
            prev_width = M1 + (i - 1) * M2
            mask = torch.zeros(prev_width + A, M2, dtype=torch.bool)
            mask[:prev_width] = torch.rand(prev_width, M2, generator=gen) < fan2
            mask[prev_width:] = torch.rand(A, M2, generator=gen) < skip_fan
            layers.append(TVLayer(prev_width + A, M2, d, d, n, dmax, mask, True, w_sd[-1], **kw))
        self.layers = nn.ModuleList(layers)
        self.event_memories = nn.ModuleList(
            [EventHopfieldLayer(d, depth, event_candidate_k) for _ in range(depth)] if event_hopfield else [])
        readout_in = M1 + (depth - 1) * M2
        self.st = TVLayer(readout_in, Mr, d, d, n, dmax, None, False, 0.3, gate_bias=1.0, normalize=True, **kw)
        self.retrieval, self.heads, self.dh, self.delta = retrieval, heads, dh, delta
        f = Mr + d
        if retrieval:
            self.Wq = nn.Linear(f, heads * dh); self.Wk = nn.Linear(f, heads * dh); self.Wv = nn.Linear(f, heads * dh)
            self.Ev = nn.Embedding(A, heads * dh)
            self.memory = AdaptiveMemoryLayer(heads, dh, delta, coupling_rank)
        self.head = nn.Sequential(nn.Linear(f + (heads * dh if retrieval else 0), 256), nn.GELU(), nn.Linear(256, A))

    def forward(self, x, collect_routes=False, collect_voltage=False, route_override=None):
        """x: (B, L) characters -> logits (B, L, A) for x[:, 1:] (position k predicts character k + 1), and work."""
        B, L = x.shape; G = L + 1
        eb = torch.arange(B).repeat_interleave(L); ei = x.reshape(-1); et = torch.arange(L).float().repeat(B) + 0.5
        ev = self.emb(ei)
        messages, layer_spikes, hm_stats = [], [], []
        route_candidates, voltage_samples = [], []
        be = je = te = ye = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                ib, ij, it, iy = eb, ei, et, ev
            else:
                previous_units = self.M1 + (i - 1) * self.M2
                ib = torch.cat([be, eb]); ij = torch.cat([je, ei + previous_units])
                it = torch.cat([te, et]); iy = torch.cat([ye, ev])
            force_route = drop_route = None
            if route_override is not None and route_override[0] == i:
                _, event_index, receiver, active = route_override
                if active:
                    force_route = (event_index, receiver)
                else:
                    drop_route = (event_index, receiver)
            needs_route_info = collect_routes or collect_voltage
            result = layer(ib, ij, it, iy, B, G, force_route=force_route,
                           drop_route=drop_route, return_routes=needs_route_info,
                           return_route_graph=collect_routes,
                           return_voltage_samples=collect_voltage)
            if needs_route_info:
                (new_b, new_j, new_t, new_y), msg, route_info = result
            else:
                (new_b, new_j, new_t, new_y), msg = result
            if collect_routes:
                route_candidates.append(route_info)
            if collect_voltage:
                voltage_samples.append(route_info["voltage_samples"])
            previous_units = 0 if i == 0 else self.M1 + (i - 1) * self.M2
            new_j = new_j + previous_units
            if be is None:
                be, je, te, ye = new_b, new_j, new_t, new_y
            else:
                be = torch.cat([be, new_b]); je = torch.cat([je, new_j])
                te = torch.cat([te, new_t]); ye = torch.cat([ye, new_y])
            messages.append(msg); layer_spikes.append(len(new_t) / (B * L))
            if self.event_hopfield:
                ye, hs = self.event_memories[i](be, te, ye)
                hm_stats.append(hs)
        V, read_msg = self.st(be, je, te, ye, B, G)
        h = nn.functional.layer_norm(V[1:L + 1].permute(1, 0, 2), (V.shape[-1],))
        f = torch.cat([h, self.emb(x)], -1)
        work = {"msgs_per_char": [m / L for m in messages] + [read_msg / L],
                "spikes_per_char": layer_spikes}
        if collect_routes:
            # Transient training-only tensors used by the optional shadow probe.
            work["route_candidates"] = route_candidates
        if collect_voltage:
            work["voltage_samples"] = voltage_samples
        if hm_stats:
            work["hopfield_pairs_per_char"] = sum(s["hopfield_pairs"] for s in hm_stats) / (B * L)
            work["hopfield_active_per_char"] = sum(s["hopfield_active"] for s in hm_stats) / (B * L)
            work["hopfield_gate"] = float(np.mean([s["hopfield_gate"] for s in hm_stats]))
            work["hopfield_update_norm"] = float(np.mean([s["hopfield_update_norm"] for s in hm_stats]))
            work["hopfield_entropy"] = float(np.mean([s["hopfield_entropy"] for s in hm_stats]))
            work["hopfield_score_std"] = float(np.mean([s["hopfield_score_std"] for s in hm_stats]))
            work["hopfield_key_diam_upper"] = float(np.mean([s["hopfield_key_diam_upper"] for s in hm_stats]))
            work["hopfield_value_diam_upper"] = float(np.mean([s["hopfield_value_diam_upper"] for s in hm_stats]))
            work["hopfield_query_jacobian_bound"] = float(np.mean([s["hopfield_query_jacobian_bound"] for s in hm_stats]))
            work["hopfield_payload_jacobian_bound_max"] = float(max(s["hopfield_payload_jacobian_bound"] for s in hm_stats))
            work["hopfield_scaled_jacobian_bound_max"] = float(max(s["hopfield_scaled_jacobian_bound"] for s in hm_stats))
            work["hopfield_scaled_jacobian_bound_sum"] = float(sum(s["hopfield_scaled_jacobian_bound"] for s in hm_stats))
            work["hopfield_max_key_fanout"] = float(max(s["hopfield_max_key_fanout"] for s in hm_stats))
            work["hopfield_delay_gain"] = float(np.mean([s["hopfield_delay_gain"] for s in hm_stats]))
        if self.retrieval:
            H, dh = self.heads, self.dh
            q = self.Wq(f).view(B, L, H, dh).transpose(1, 2); k = self.Wk(f).view(B, L, H, dh).transpose(1, 2)
            nxt = torch.cat([x[:, 1:], x[:, -1:]], 1)                          # value j carries x_{j+1}
            v = (self.Wv(f) + self.Ev(nxt)).view(B, L, H, dh).transpose(1, 2)
            o, am = self.memory(q, k, v)
            o = o.transpose(1, 2).reshape(B, L, H * dh)
            f = torch.cat([f, o], -1)
            work.update(am)
            work["keys_visible"] = float((L - 1) / 2)
        return self.head(f), work


def score(net, data, L, bs, dump=False):
    """bits per character: windows of L, each scoring its second half (first window scores all), as E64. dump: also the
    probability of each true character, indexed by its position in `data` (for mixing with the native experts, E78)."""
    net.eval(); tot = 0.0; n = 0; half = L // 2; starts = list(range(0, len(data) - L - 1, half)); wk = []
    ptrue = np.full(len(data), np.nan, np.float32) if dump else None
    logp = np.full((len(data), A), np.nan, np.float16) if dump else None
    with torch.no_grad():
        for i in range(0, len(starts), bs):
            ss = starts[i:i + bs]
            X = torch.stack([data[s:s + L] for s in ss]); Y = torch.stack([data[s + 1:s + L + 1] for s in ss])
            logits, w = net(X); wk.append(w)
            ce = nn.functional.cross_entropy(logits.reshape(-1, A), Y.reshape(-1), reduction="none").view(len(ss), L)
            for r, s0 in enumerate(ss):
                lo = 0 if s0 == 0 else half
                tot += float(ce[r, lo:].sum()); n += L - lo
                if dump:
                    ptrue[s0 + 1 + lo:s0 + 1 + L] = torch.exp(-ce[r, lo:]).numpy()
                    logp[s0 + 1 + lo:s0 + 1 + L] = torch.log_softmax(logits[r, lo:], -1).numpy().astype(np.float16)
    net.train()
    agg = {k: (np.mean([w[k] for w in wk], 0).round(3).tolist() if isinstance(wk[0][k], list) else round(float(np.mean([w[k] for w in wk])), 2))
           for k in wk[0]}
    return (tot / n / math.log(2), agg, ptrue, logp) if dump else (tot / n / math.log(2), agg)


def calibrate_spike_thresholds(net, train, a):
    """Calibrate initial event rates from training inputs without labels.

    The voltage quantile supplies a scale estimate. A local replay then applies
    the exact threshold/reset recurrence to the saved voltage traces and finds
    a threshold matching the realized spike rate. This avoids rerunning the
    entire model for every search step. Calibration proceeds in depth order,
    so each deeper layer sees the already-calibrated event stream below it.
    """
    if a.target_spikes_per_char <= 0:
        return {"thresholds": [], "rates": [], "forward_batches": 0}
    if any(a.target_spikes_per_char >= layer.M for layer in net.layers):
        raise ValueError("target_spikes_per_char must be smaller than every hidden layer width")
    rng = np.random.default_rng(a.seed + 77_101)
    batches = []
    for _ in range(a.threshold_calibration_batches):
        idx = rng.integers(0, a.D - a.L - 1, a.bs)
        batches.append(torch.stack([train[i:i + a.L] for i in idx]))
    thresholds, rates, forward_batches = [], [], 0

    def replay_rate(traces, threshold):
        # Match TVLayer's reset and fractional-crossing recurrence exactly.
        G, total_batch, M = traces.shape
        R = torch.zeros(total_batch, M, dtype=traces.dtype, device=traces.device)
        Vp = torch.zeros_like(R)
        eR = math.exp(-1 / TAU_R)
        spike_count = 0
        with torch.no_grad():
            for k in range(G):
                R.mul_(eR)
                Vd = traces[k] - threshold * R
                fire = Vd >= threshold
                frac = ((threshold - Vp) / (Vd - Vp).clamp(min=1e-6)).clamp(0, 1)
                jump = fire.to(traces.dtype) * torch.exp(-(1 - frac) / TAU_R)
                if k >= 1:
                    spike_count += int(fire.sum())
                Vp = Vd - threshold * jump
                R.add_(jump)
        return spike_count / (len(batches) * a.bs * a.L)

    for layer_index, layer in enumerate(net.layers):
        collected = []
        for X in batches:
            with torch.no_grad():
                _, work = net(X, collect_voltage=True)
            collected.append(work["voltage_samples"][layer_index])
            forward_batches += 1
        traces = torch.cat(collected, dim=1)
        samples = traces.reshape(-1)
        tail_probability = a.target_spikes_per_char / layer.M
        quantile = torch.tensor(1.0 - tail_probability, dtype=samples.dtype, device=samples.device)
        guess = float(torch.quantile(samples, quantile).clamp_min(1e-6))
        best_threshold = guess
        best_rate = replay_rate(traces, guess)
        tolerance = max(0.25 * a.target_spikes_per_char, 1.0 / (a.bs * a.L))
        if abs(best_rate - a.target_spikes_per_char) > tolerance:
            low = 1e-6
            low_rate = replay_rate(traces, low)
            if low_rate >= a.target_spikes_per_char:
                high = max(float(samples.max()), guess, 1.0) + 1e-6
                high_rate = replay_rate(traces, high)
                for _ in range(8):
                    if high_rate <= a.target_spikes_per_char:
                        break
                    high *= 2.0
                    high_rate = replay_rate(traces, high)
                if high_rate <= a.target_spikes_per_char:
                    if abs(low_rate - a.target_spikes_per_char) < abs(best_rate - a.target_spikes_per_char):
                        best_threshold, best_rate = low, low_rate
                    if abs(high_rate - a.target_spikes_per_char) < abs(best_rate - a.target_spikes_per_char):
                        best_threshold, best_rate = high, high_rate
                    for _ in range(a.threshold_calibration_steps):
                        mid = 0.5 * (low + high)
                        mid_rate = replay_rate(traces, mid)
                        if abs(mid_rate - a.target_spikes_per_char) < abs(best_rate - a.target_spikes_per_char):
                            best_threshold, best_rate = mid, mid_rate
                        if abs(mid_rate - a.target_spikes_per_char) <= tolerance:
                            break
                        if mid_rate > a.target_spikes_per_char:
                            low = mid
                        else:
                            high = mid
            elif abs(low_rate - a.target_spikes_per_char) < abs(best_rate - a.target_spikes_per_char):
                best_threshold, best_rate = low, low_rate
        layer.theta = best_threshold
        thresholds.append(layer.theta)
        rates.append(best_rate)
    return {"thresholds": thresholds, "rates": rates, "forward_batches": forward_batches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=1_000_000)
    ap.add_argument("--passes", type=float, default=5.0)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--M1", type=int, default=128)
    ap.add_argument("--M2", type=int, default=128)
    ap.add_argument("--Mr", type=int, default=64)
    ap.add_argument("--fan2", type=float, default=0.25)
    ap.add_argument("--dmax", type=float, default=8.0)
    ap.add_argument("--w_sd", default="0.05,0.05")
    ap.add_argument("--retrieval", type=int, default=1)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--dh", type=int, default=32)
    ap.add_argument("--delta", type=float, default=8.0)
    ap.add_argument("--coupling_rank", type=int, default=0,
                    help="rank of query-key score/value interaction; 0 gives plain dot-product retrieval")
    ap.add_argument("--depth", type=int, default=2, help="number of event-state layers (each deeper layer has a raw-event skip)")
    ap.add_argument("--skip_fan", type=float, default=0.125, help="fraction of units receiving each raw-event skip")
    ap.add_argument("--event_hopfield", type=int, default=1, help="causal associative updates over emitted event payloads")
    ap.add_argument("--event_candidate_k", type=int, default=0,
                    help="optional top-k event aggregation after score computation; 0 keeps all past events")
    ap.add_argument("--cf_shadows_per_layer", type=int, default=0,
                    help="near-boundary hard-route toggles per hidden layer and minibatch; 0 disables the probe")
    ap.add_argument("--cf_band", type=float, default=0.5,
                    help="score distance from zero defining candidate routes for shadow evaluation")
    ap.add_argument("--cf_sigma", type=float, default=0.25,
                    help="logistic route-noise scale used by the boundary derivative")
    ap.add_argument("--cf_weight", type=float, default=1.0,
                    help="scale of the normalized counterfactual boundary-gradient diagnostic")
    ap.add_argument("--cf_delta_clip", type=float, default=5.0,
                    help="clip on batch-mean open-minus-closed token loss difference")
    ap.add_argument("--cf_lr", type=float, default=0.0,
                    help="separate post-Adam counterfactual step; leave at 0 until shadow statistics justify it")
    ap.add_argument("--cf_grad_clip", type=float, default=1.0,
                    help="global norm cap for an enabled counterfactual step")
    ap.add_argument("--target_spikes_per_char", type=float, default=0.1,
                    help="label-free initial hidden-event budget per character; 0 disables activity calibration")
    ap.add_argument("--threshold_calibration_batches", type=int, default=1,
                    help="fixed minibatches used to estimate and match each layer's initial event rate")
    ap.add_argument("--threshold_calibration_steps", type=int, default=6,
                    help="binary-search iterations after the voltage-quantile initialization")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--valid", type=int, default=200_000)
    ap.add_argument("--test", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if (a.cf_shadows_per_layer < 0 or a.cf_band <= 0 or a.cf_sigma <= 0
            or a.cf_delta_clip <= 0 or a.cf_lr < 0 or a.cf_grad_clip <= 0):
        raise ValueError("counterfactual count must be nonnegative; band, sigma, delta clip, and grad clip must be positive; cf_lr must be nonnegative")
    if (a.target_spikes_per_char < 0 or a.threshold_calibration_batches < 1
            or a.threshold_calibration_steps < 1):
        raise ValueError("target_spikes_per_char must be nonnegative; calibration_batches and calibration_steps must be positive")
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    x = S1.load(); train = torch.tensor(x[:a.D]).long()
    valid = torch.tensor(x[90_000_000:90_000_000 + a.valid]).long(); test = torch.tensor(x[95_000_000:95_000_000 + a.test]).long()
    net = TVLM(a.d, a.n, a.M1, a.M2, a.Mr, a.fan2, a.dmax, [float(v) for v in a.w_sd.split(",")], a.retrieval, a.heads, a.dh,
               a.delta, a.seed, a.coupling_rank, a.depth, a.skip_fan, bool(a.event_hopfield), a.event_candidate_k)
    calibration = calibrate_spike_thresholds(net, train, a)
    calibrated_thresholds = calibration["thresholds"]
    calibrated_rates = calibration["rates"]
    calibration_token_visits = calibration["forward_batches"] * net.depth * a.bs * a.L
    if calibrated_thresholds:
        print(json.dumps({"threshold_calibration_target_spikes_per_char": a.target_spikes_per_char,
                          "threshold_calibration_batches": a.threshold_calibration_batches,
                          "threshold_calibration_steps": a.threshold_calibration_steps,
                          "threshold_calibration_forwards": calibration["forward_batches"],
                          "threshold_calibration_token_visits": calibration_token_visits,
                          "initial_layer_thresholds": calibrated_thresholds,
                          "initial_layer_spikes_per_char": calibrated_rates}), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    steps = int(a.passes * a.D / (a.bs * a.L)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1))
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()), "steps": steps,
           "initial_layer_thresholds": calibrated_thresholds or [float(layer.theta) for layer in net.layers],
           "initial_layer_calibration_spikes_per_char": calibrated_rates,
           "threshold_calibration_forwards": calibration["forward_batches"],
           "threshold_calibration_token_visits": calibration_token_visits,
           "valid_curve": []}
    print(json.dumps({"params": res["params"], "steps": steps,
                      "threshold_calibration_token_visits": calibration_token_visits}), flush=True)
    best = (float("inf"), None, -1)
    cf_rng = np.random.default_rng(a.seed + 77_003)
    cf_interval = None
    def new_cf_interval():
        return {"eligible": 0, "shadows": 0, "sum": 0.0, "sq": 0.0,
                "helpful": 0, "layer_n": np.zeros(a.depth, dtype=np.int64),
                "layer_sum": np.zeros(a.depth), "layer_sq": np.zeros(a.depth),
                "layer_helpful": np.zeros(a.depth), "cf_norm": 0.0,
                "path_norm": 0.0, "cosine": 0.0, "grad_batches": 0,
                "applied_norm": 0.0, "applied_batches": 0, "clipped": 0}
    cf_interval = new_cf_interval()
    for step in range(steps):
        idx = rng.integers(0, a.D - a.L - 1, a.bs)
        X = torch.stack([train[i:i + a.L] for i in idx]); Y = torch.stack([train[i + 1:i + a.L + 1] for i in idx])
        collect_routes = a.cf_shadows_per_layer > 0
        logits, work = net(X, collect_routes=collect_routes)
        loss = nn.functional.cross_entropy(logits.reshape(-1, A), Y.reshape(-1))
        params = [p for p in net.parameters() if p.requires_grad]
        cf_grads = None
        if collect_routes:
            base_loss = loss.detach()
            cf_proxy = loss.new_zeros(())
            step_shadows = 0
            for layer_index, route_info in enumerate(work["route_candidates"]):
                scores = route_info["score"].cpu().numpy()
                near = np.flatnonzero(np.abs(scores) <= a.cf_band)
                cf_interval["eligible"] += len(near)
                if not len(near):
                    continue
                take = min(a.cf_shadows_per_layer, len(near))
                selected = cf_rng.choice(near, size=take, replace=False)
                for route_index in np.atleast_1d(selected):
                    event_index = int(route_info["event_index"][route_index])
                    receiver = int(route_info["receiver"][route_index])
                    score_value = float(scores[route_index])
                    active = score_value > 0.0
                    with torch.no_grad():
                        shadow_logits, _ = net(
                            X, route_override=(layer_index, event_index, receiver, not active))
                        shadow_loss = nn.functional.cross_entropy(
                            shadow_logits.reshape(-1, A), Y.reshape(-1))
                    # Signed effect is L(open)-L(closed), through every later
                    # event layer, the readout state, and all causal token losses.
                    delta = float(base_loss - shadow_loss) if active else float(shadow_loss - base_loss)
                    clipped_delta = float(np.clip(delta, -a.cf_delta_clip, a.cf_delta_clip))
                    cf_interval["shadows"] += 1
                    step_shadows += 1
                    cf_interval["sum"] += delta
                    cf_interval["sq"] += delta * delta
                    cf_interval["helpful"] += int(delta < 0.0)
                    cf_interval["layer_n"][layer_index] += 1
                    cf_interval["layer_sum"][layer_index] += delta
                    cf_interval["layer_sq"][layer_index] += delta * delta
                    cf_interval["layer_helpful"][layer_index] += int(delta < 0.0)
                    cf_interval["clipped"] += int(abs(delta) > a.cf_delta_clip)
                    live_score = route_info["score_live"][route_index]
                    p_open = torch.sigmoid(live_score.detach() / a.cf_sigma)
                    boundary = p_open * (1.0 - p_open) / a.cf_sigma * clipped_delta
                    # Per-layer sample mean avoids an inverse-probability factor
                    # over the enormous candidate set.
                    cf_proxy = cf_proxy + a.cf_weight * boundary * (
                        live_score - live_score.detach()) / take
            if step_shadows:
                cf_grads = torch.autograd.grad(cf_proxy, params, retain_graph=True,
                                               allow_unused=True)
        opt.zero_grad(); loss.backward()
        layer_grad = [math.sqrt(sum(float(p.grad.detach().square().sum()) for p in layer.parameters() if p.grad is not None))
                      for layer in net.layers]
        retrieval_grad = (math.sqrt(sum(float(p.grad.detach().square().sum()) for p in net.memory.parameters() if p.grad is not None))
                          if a.retrieval else 0.0)
        hopfield_grad = [{name: math.sqrt(sum(float(p.grad.detach().square().sum()) for p in module.parameters()
                                             if p.grad is not None))
                          for name, module in (("query", layer.q), ("key", layer.k), ("value", layer.v),
                                               ("output", layer.out), ("gate", layer.gate))}
                         for layer in net.event_memories]
        for i, layer in enumerate(net.event_memories):
            hopfield_grad[i]["delay"] = (float(layer.log_delay_gain.grad.detach().abs())
                                         if layer.log_delay_gain.grad is not None else 0.0)
        if cf_grads is not None:
            cf_norm = math.sqrt(sum(float(g.detach().square().sum()) for g in cf_grads if g is not None))
            path_norm = math.sqrt(sum(float(p.grad.detach().square().sum())
                                       for p in params if p.grad is not None))
            dot = sum(float((g.detach() * p.grad.detach()).sum())
                      for p, g in zip(params, cf_grads) if g is not None and p.grad is not None)
            cf_interval["cf_norm"] += cf_norm
            cf_interval["path_norm"] += path_norm
            cf_interval["cosine"] += dot / (cf_norm * path_norm) if cf_norm and path_norm else 0.0
            cf_interval["grad_batches"] += 1
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if cf_grads is not None and a.cf_lr:
            cf_norm = math.sqrt(sum(float(g.detach().square().sum()) for g in cf_grads if g is not None))
            cf_scale = min(1.0, a.cf_grad_clip / max(cf_norm, 1e-12))
            with torch.no_grad():
                for param, grad in zip(params, cf_grads):
                    if grad is not None:
                        param.add_(grad, alpha=-a.cf_lr * cf_scale)
            cf_interval["applied_norm"] += a.cf_lr * min(cf_norm, a.cf_grad_clip)
            cf_interval["applied_batches"] += 1
        if (step + 1) % max(steps // 10, 1) == 0 or step == steps - 1:
            vb, wk = score(net, valid, a.L, a.bs)
            row = {"step": step + 1, "train_bpc": loss.detach().item() / math.log(2), "valid_bpc": vb, **wk,
                   "event_layer_grad_norms": layer_grad, "retrieval_grad_norm": retrieval_grad,
                   "event_hopfield_grad_norms": hopfield_grad,
                   "wall_s": round(time.time() - t0)}
            if collect_routes:
                count = max(cf_interval["shadows"], 1)
                layer_means, layer_stds, layer_ses = [], [], []
                for i in range(a.depth):
                    ni = int(cf_interval["layer_n"][i])
                    if ni:
                        layer_mean = cf_interval["layer_sum"][i] / ni
                        layer_var = max(cf_interval["layer_sq"][i] / ni - layer_mean ** 2, 0.0)
                        layer_means.append(round(float(layer_mean), 8))
                        layer_stds.append(round(math.sqrt(layer_var), 8))
                        layer_ses.append(round(math.sqrt(layer_var / ni), 8))
                    else:
                        layer_means.append(None)
                        layer_stds.append(None)
                        layer_ses.append(None)
                row.update({
                    "event_layer_peak_voltage": [round(info["peak_voltage"], 6)
                                                 for info in work["route_candidates"]],
                    "event_layer_peak_threshold_margin": [round(info["peak_threshold_margin"], 6)
                                                           for info in work["route_candidates"]],
                    "event_layer_firing_fraction": [round(info["firing_fraction"], 8)
                                                    for info in work["route_candidates"]],
                    "counterfactual_near_routes": int(cf_interval["eligible"]),
                    "counterfactual_route_shadows": int(cf_interval["shadows"]),
                    "counterfactual_mean_open_minus_closed_loss": round(cf_interval["sum"] / count, 8),
                    "counterfactual_shadow_delta_std": round(math.sqrt(max(
                        cf_interval["sq"] / count - (cf_interval["sum"] / count) ** 2, 0.0)), 8),
                    "counterfactual_fraction_opening_improves": round(cf_interval["helpful"] / count, 4),
                    "counterfactual_layer_shadow_counts": cf_interval["layer_n"].tolist(),
                    "counterfactual_layer_mean_open_minus_closed_loss": layer_means,
                    "counterfactual_layer_shadow_delta_std": layer_stds,
                    "counterfactual_layer_delta_standard_error": layer_ses,
                    "counterfactual_layer_fraction_opening_improves": [
                        round(float(cf_interval["layer_helpful"][i] / cf_interval["layer_n"][i]), 4)
                        if cf_interval["layer_n"][i] else None for i in range(a.depth)],
                    "counterfactual_grad_norm": round(cf_interval["cf_norm"] / max(cf_interval["grad_batches"], 1), 8),
                    "pathwise_grad_norm_on_shadow_batches": round(cf_interval["path_norm"] / max(cf_interval["grad_batches"], 1), 8),
                    "counterfactual_to_pathwise_norm_ratio": round(cf_interval["cf_norm"] / max(cf_interval["path_norm"], 1e-12), 8),
                    "counterfactual_pathwise_cosine": round(cf_interval["cosine"] / max(cf_interval["grad_batches"], 1), 6),
                    "counterfactual_clipped_delta_fraction": round(cf_interval["clipped"] / count, 4),
                    "counterfactual_applied_update_norm_per_batch": round(
                        cf_interval["applied_norm"] / max(cf_interval["applied_batches"], 1), 8),
                })
            res["valid_curve"].append(row); print(json.dumps(row), flush=True)
            cf_interval = new_cf_interval()
            if vb < best[0]:
                best = (vb, {k: v.clone() for k, v in net.state_dict().items()}, step + 1)
    net.load_state_dict(best[1])
    tb, wk, pt, lt = score(net, test, a.L, a.bs, dump=True)
    _, _, pv, lv = score(net, valid, a.L, a.bs, dump=True)
    tag = (f"tvlm_D{a.D}_p{a.passes:g}_r{a.retrieval}_M{a.M1}-{a.M2}-{a.Mr}"
           f"_depth{a.depth}_c{a.coupling_rank}_eh{a.event_hopfield}_ek{a.event_candidate_k}_s{a.seed}")
    if a.cf_shadows_per_layer:
        tag += (f"_cf{a.cf_shadows_per_layer}_b{a.cf_band:g}_sg{a.cf_sigma:g}"
                f"_w{a.cf_weight:g}_dl{a.cf_delta_clip:g}_lr{a.cf_lr:g}_gc{a.cf_grad_clip:g}")
    if a.target_spikes_per_char:
        tag += (f"_boot{a.target_spikes_per_char:g}_cb{a.threshold_calibration_batches}"
                f"_cs{a.threshold_calibration_steps}")
    np.save(os.path.join(OUT, tag + "_ptrue_test.npy"), pt); np.save(os.path.join(OUT, tag + "_ptrue_valid.npy"), pv)
    np.save(os.path.join(OUT, tag + "_logp_test.npy"), lt); np.save(os.path.join(OUT, tag + "_logp_valid.npy"), lv)
    res.update({"best_step": best[2], "best_valid_bpc": best[0], "test_bpc": tb, "test_work": wk, "wall_s": round(time.time() - t0)})
    print(json.dumps({"test_bpc": tb, "best_step": best[2], **wk}), flush=True)
    with open(os.path.join(OUT, tag + ".json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
