"""Compiled episode-batched native core: batched_episodes' arithmetic with each layer step fused (THEORY §412).

At language sizes the batched path is dispatch-bound: one 64-lane window costs about as much as a 128-lane one, and
backward is about 60% of the time.  This module evaluates one (event, layer) step as a single torch.compile'd
function, so its many small operations become a few fused kernels in forward and backward.  The step loop, state
bookkeeping and the race-noise draw order are batched_episodes'.

The factorized race is written without a custom autograd.Function so that it can be traced:
  value  = proposals[lane, winner]                       (winner-only payload credit)
  first  = min_i noise_i / exp(s_i), held fixed, plus the zero-valued term -first * (lse(s) - lse(s).detach())
whose score gradient is -first * pi_i, the common first-time clock credit dT/ds_i = -T pi_i of LaneRace.  Forward
values are unchanged (the added term is exactly zero).  Forces and race recording are not supported here; shadow
lanes stay on the eager path.

Contract (tests/test_compiled_episodes.py): logits and every parameter gradient agree with batched_logits in float64
within 1e-9 relative; winners are identical.
"""
import math

import torch
from torch.nn import functional as F

from .parallel_stream_language import precise_rotate


def _rotate(state, angles):
    pairs = state.reshape(*state.shape[:-1], -1, 2)
    a, b = pairs.unbind(-1)
    cos, sin = angles.cos().to(state.dtype), angles.sin().to(state.dtype)
    return torch.stack((cos * a - sin * b, sin * a + cos * b), -1).flatten(-2)


def _transport(value, age, rate_raw, frequency):
    age = age.clamp_min(0)
    rate = F.softplus(rate_raw) + 1e-6
    decayed = value * torch.exp(-age.to(value.dtype)[:, None] * rate).repeat_interleave(2, -1)
    return _rotate(decayed, age[:, None] * frequency.to(torch.float64))


def layer_step(x, arrival, m, arr_d, seen_d, active, noise, mix_w, mix_b, query, key, key_read, clock_bias, control_w,
               control_b, rate, frequency, input_w, output_w, gate_w, gate_b, gain, transport_rate, transport_frequency):
    """One layer of one event for all lanes.  noise (H, U) float64 in draw order.  Returns the next x, arrival,
    the layer's new memories/arrival stamps/written masks, the head values and their arrivals."""
    n = x.shape[0]
    H, U, P = m.shape[1], m.shape[2], m.shape[3]
    total = H * P
    mixed = F.linear(x, mix_w, mix_b)
    all_features = F.layer_norm(mixed, (total,))
    incoming = mixed.view(n, H, P)
    q = torch.einsum('hpd,ld->lhp', query, all_features)
    prev = torch.where(seen_d, arr_d, arrival[:, None, None])
    x_u = incoming[:, :, None, :].expand(n, H, U, P)
    q_u = q[:, :, None, :].expand(n, H, U, P)
    read = key.view(H, U, P) + torch.einsum('hupq,lhuq->lhup', key_read.view(H, U, P, P), m)
    scores = ((q_u * read).sum(-1) / math.sqrt(P) + clock_bias.view(H, U)).clamp(-12, 12)
    controls = torch.einsum('hucp,lhup->lhuc', control_w.view(H, U, 2, P), F.layer_norm(x_u, (P,))) + control_b.view(H, U, 2)
    forget = F.softplus(controls[..., 0]) / math.log(2)
    write = 2 * torch.sigmoid(controls[..., 1])
    age = (arrival[:, None, None] - prev).clamp_min(0)
    decay = torch.exp(-age.to(m.dtype)[..., None] * rate.view(H, U, P // 2) * forget[..., None]).repeat_interleave(2, -1)
    m_new = _rotate(m * decay, age[..., None] * frequency.view(H, U, P // 2))
    m_new = m_new + write[..., None] * torch.einsum('hupq,lhuq->lhup', input_w.view(H, U, P, P), x_u)
    y = F.layer_norm(torch.einsum('hupq,lhuq->lhup', output_w.view(H, U, P, P), m_new) + x_u, (P,))
    gate = torch.einsum('hupq,lhuq->lhup', gate_w.view(H, U, P, P), F.gelu(y)) + gate_b.view(H, U, P)
    proposals = x_u + gain * y * torch.sigmoid(gate)
    # factorized races, all heads at once (each head's noise row is its own draw)
    s64 = scores.to(torch.float64)                                             # (n, H, U)
    times = noise[None] / s64.exp()
    first, winner = times.min(-1)                                              # (n, H)
    first = first.detach()
    lse = torch.logsumexp(s64, -1)
    first_s = first - first * (lse - lse.detach())                            # value first; d/ds_i = -first * pi_i
    delay = .001 + .010 * first_s / (1 + first_s)
    onehot = F.one_hot(winner, U).to(torch.bool) & active[:, None, None]      # (n, H, U); writes only active lanes
    values = torch.gather(proposals, 2, winner[:, :, None, None].expand(n, H, 1, P)).squeeze(2)   # (n, H, P)
    new_mem = torch.where(onehot[..., None], m_new, m)
    new_arr = torch.where(onehot, arrival[:, None, None], arr_d)
    new_seen = seen_d | onehot
    arrivals = arrival[:, None] + delay                                        # (n, H)
    arrival_out = arrivals.max(-1).values
    age_h = arrival_out[:, None] - arrivals
    x_out = torch.cat([_transport(values[:, h], age_h[:, h], transport_rate[h], transport_frequency[h]) for h in range(H)], -1)
    return x_out, arrival_out, new_mem, new_arr, new_seen, values, arrivals


_COMPILED = {}


def compiled_step():
    if 'step' not in _COMPILED:
        _COMPILED['step'] = torch.compile(layer_step, dynamic=False, fullgraph=True)
    return _COMPILED['step']


def compiled_logits(model, rows, seed, all_logits=False, step=None):
    """Drop-in for batched_logits(model, rows, seed, all_logits=...) without forces/record.  step: the layer function
    (default compiled; pass layer_step for the eager reference of this formulation)."""
    step = step or compiled_step()
    source = 0
    layers = model._stacked(source)
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    n = len(rows); dtype = model.embedding.weight.dtype
    lengths = torch.tensor([len(r['events']) for r in rows])
    T = int(lengths.max())
    mem = [torch.zeros(n, H, U, P, dtype=dtype) for _ in range(D)]
    arr = [torch.zeros(n, H, U, dtype=torch.float64) for _ in range(D)]
    seen = [torch.zeros(n, H, U, dtype=torch.bool) for _ in range(D)]
    stamp_rows = torch.zeros(n, T, dtype=torch.float64)
    mark_rows = []
    for i, r in enumerate(rows):
        events = r['events']
        stamp_rows[i, :len(events)] = torch.tensor([float(e[0]) for e in events], dtype=torch.float64)
        marks_i = [torch.as_tensor(e[1], dtype=dtype) for e in events]
        mark_rows.append(torch.stack(marks_i + [marks_i[0]] * (T - len(events))))
    mark_rows = torch.stack(mark_rows)
    ctx_vals = torch.zeros(n, H * P, dtype=dtype); ctx_arr = torch.zeros(n, H, dtype=torch.float64)
    has_ctx = torch.zeros(n, dtype=torch.bool)
    out = torch.zeros(n, model.head.out_features, dtype=dtype)
    every = []

    def transport(value, age, depth, head):
        age = age.clamp_min(0)
        rate = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype)[:, None] * rate).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for k in range(T):
            active = lengths > k
            x = model.embedding.weight[source][None] + model.content(mark_rows[:, k])
            arrival = stamp_rows[:, k]
            read_time = torch.where(has_ctx, torch.maximum(arrival, ctx_arr.max(-1).values), arrival)
            arrival = read_time
            context = torch.cat([transport(ctx_vals[:, h * P:(h + 1) * P], read_time - ctx_arr[:, h], D - 1, h) for h in range(H)], -1)
            x = torch.where(has_ctx[:, None], F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context, (model.total_payload,)), x)
            for depth in range(D):
                Lp = layers[depth]
                noise = torch.stack([torch.empty(U, dtype=torch.float64).exponential_() for _ in range(H)])
                mix = model.channel_mix[depth]
                x, arrival, mem[depth], arr[depth], seen[depth], values, arrivals = step(
                    x, arrival, mem[depth], arr[depth], seen[depth], active, noise, mix.weight, mix.bias, Lp['query'],
                    Lp['key'], Lp['key_read'], Lp['clock_bias'], Lp['control_w'], Lp['control_b'], Lp['rate'],
                    Lp['frequency'], Lp['input'], Lp['output'], Lp['gate_w'], Lp['gate_b'], Lp['gain'],
                    model.transport_rate[depth], model.transport_frequency[depth])
            ctx_vals = torch.where(active[:, None], values.reshape(n, H * P), ctx_vals)
            ctx_arr = torch.where(active[:, None], arrivals, ctx_arr)
            has_ctx = has_ctx | active
            last = lengths == k + 1
            logits_k = model.head(x)
            if all_logits:
                every.append(torch.where(active[:, None], logits_k, torch.zeros_like(logits_k)))
            out = torch.where(last[:, None], logits_k, out)
    return torch.stack(every, 1) if all_logits else out
