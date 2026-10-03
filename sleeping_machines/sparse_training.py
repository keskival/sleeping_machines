"""Winner-plus-sampled-alternative training for the batched native core (THEORY §416).

The batched training path computes every unit's proposal at every race (forward and a dense backward), although only the
winner's value is forwarded and written.  Here each race computes the winner's proposal and, for route credit, one
alternative j sampled among the losers with probability q_j = pi_j / (1 - pi_w) (an independent exponential race on a
separate generator, so the main race noise is unchanged).  Key reads key + key_read . m of the stored memories are cached
per slot and refreshed only on that slot's write, exactly as at inference (§414).

Credit.  The linearized local-expectation credit is sum_k d pi_k / d s . (v_k - v_w) . g (the winner as baseline; equal to
pi_i g.(v_i - v_bar) because sum_k d pi_k = 0).  'sampled' estimates it without bias by d pi_j / d s . (v_j - v_w) . g / q_j,
written as the zero-valued surrogate (pi_j - sg pi_j) / sg q_j . sg(v_j - v_w).  Clock credit and value credit are those of
the factorized race.

Contracts (tests/test_sparse_training.py): with route credit off, logits and every parameter gradient equal batched_logits;
with 'sampled', forward values are unchanged and the average gradient over alternative draws equals the 'linear' gradient.
Per race the proposal work is 2 units (1 without credit) instead of U; scoring is U P multiply-adds.
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


def _unit(sel, m, arr_d, seen_d, arrival, xh, U, P, control_w, control_b, rate, frequency, input_w, output_w, gate_w,
          gate_b, gain):
    """proposal, new memory and new stamp of the selected unit per (lane, head): sel (n, H)."""
    n, H = sel.shape
    idx = torch.arange(H)[None, :] * U + sel                                    # stacked unit index (n, H)
    pick = sel[:, :, None, None].expand(n, H, 1, P)
    m_s = torch.gather(m, 2, pick).squeeze(2)                                  # (n, H, P)
    prev = torch.where(torch.gather(seen_d, 2, sel[:, :, None]).squeeze(2),
                       torch.gather(arr_d, 2, sel[:, :, None]).squeeze(2), arrival[:, None])
    controls = torch.einsum('nhcp,nhp->nhc', control_w.view(H * U, 2, P)[idx], F.layer_norm(xh, (P,))) \
        + control_b.view(H * U, 2)[idx]
    forget = F.softplus(controls[..., 0]) / math.log(2)
    write = 2 * torch.sigmoid(controls[..., 1])
    age = (arrival[:, None] - prev).clamp_min(0)
    decay = torch.exp(-age.to(m.dtype)[..., None] * rate.view(H * U, P // 2)[idx] * forget[..., None]).repeat_interleave(2, -1)
    m_new = _rotate(m_s * decay, age[..., None] * frequency.view(H * U, P // 2)[idx])
    m_new = m_new + write[..., None] * torch.einsum('nhpq,nhq->nhp', input_w.view(H * U, P, P)[idx], xh)
    y = F.layer_norm(torch.einsum('nhpq,nhq->nhp', output_w.view(H * U, P, P)[idx], m_new) + xh, (P,))
    gate = torch.einsum('nhpq,nhq->nhp', gate_w.view(H * U, P, P)[idx], F.gelu(y)) + gate_b.view(H * U, P)[idx]
    return xh + gain * y * torch.sigmoid(gate), m_new, idx


def sparse_layer_step(x, arrival, m, arr_d, seen_d, reads_d, active, noise, alt_noise, mix_w, mix_b, query, key, key_read,
                      clock_bias, control_w, control_b, rate, frequency, input_w, output_w, gate_w, gate_b, gain,
                      transport_rate, transport_frequency, sampled_credit=False):
    n = x.shape[0]
    H, U, P = m.shape[1], m.shape[2], m.shape[3]
    mixed = F.linear(x, mix_w, mix_b)
    all_features = F.layer_norm(mixed, (H * P,))
    incoming = mixed.view(n, H, P)
    q = torch.einsum('hpd,ld->lhp', query, all_features)
    scores = ((q[:, :, None, :] * reads_d).sum(-1) / math.sqrt(P) + clock_bias.view(H, U)).clamp(-12, 12)
    s64 = scores.to(torch.float64)
    times = noise[None] / s64.exp()
    first, winner = times.min(-1)
    first = first.detach()
    lse = s64.exp().sum(-1).log()                                              # scores are clamped to +-12
    first_s = first - first * (lse - lse.detach())                            # clock credit -T pi_i
    delay = .001 + .010 * first_s / (1 + first_s)
    unit_args = (U, P, control_w, control_b, rate, frequency, input_w, output_w, gate_w, gate_b, gain)
    value_w, m_new_w, idx_w = _unit(winner, m, arr_d, seen_d, arrival, incoming, *unit_args)
    values = value_w
    if sampled_credit and U > 1:
        pi = torch.softmax(s64, -1)                                            # (n, H, U) float64
        alt_times = alt_noise[None] / pi.detach()
        alt_times = alt_times.masked_fill(F.one_hot(winner, U).to(torch.bool), float('inf'))
        alt = alt_times.argmin(-1)                                             # j ~ pi_j / (1 - pi_w) among the losers
        value_j, _, _ = _unit(alt, m, arr_d, seen_d, arrival, incoming, *unit_args)
        pi_j = torch.gather(pi, 2, alt[:, :, None]).squeeze(2)
        pi_w = torch.gather(pi, 2, winner[:, :, None]).squeeze(2)
        q_j = (pi_j / (pi.sum(-1) - pi_w)).detach()                            # pi_j / (1 - pi_w)
        coef = ((pi_j - pi_j.detach()) / q_j).to(value_w.dtype)
        values = values + coef[..., None] * (value_j - value_w).detach()
    onehot = F.one_hot(winner, U).to(torch.bool) & active[:, None, None]
    new_mem = torch.where(onehot[..., None], m_new_w[:, :, None, :], m)
    new_arr = torch.where(onehot, arrival[:, None, None], arr_d)
    new_seen = seen_d | onehot
    refreshed = key.view(H * U, P)[idx_w] + torch.einsum('nhpq,nhq->nhp', key_read.view(H * U, P, P)[idx_w], m_new_w)
    new_reads = torch.where(onehot[..., None], refreshed[:, :, None, :], reads_d)
    arrivals = arrival[:, None] + delay
    arrival_out = arrivals.max(-1).values
    age_h = arrival_out[:, None] - arrivals
    x_out = torch.cat([_transport(values[:, h], age_h[:, h], transport_rate[h], transport_frequency[h]) for h in range(H)], -1)
    return x_out, arrival_out, new_mem, new_arr, new_seen, new_reads, values, arrivals


_COMPILED = {}


def compiled_sparse_step():
    if 'step' not in _COMPILED:
        from torch._dynamo import config as dynamo_config
        for name in ('cache_size_limit', 'recompile_limit'):
            if hasattr(dynamo_config, name):
                setattr(dynamo_config, name, max(getattr(dynamo_config, name), 64))
        _COMPILED['step'] = torch.compile(sparse_layer_step, dynamic=False, fullgraph=True)
    return _COMPILED['step']


def sparse_train_logits(model, rows, seed, all_logits=False, route_credit=None, step=None, alt_seed=None):
    """Drop-in for compiled_logits(model, rows, seed, all_logits, route_credit=None|'sampled').  step: the layer function
    (default compiled; pass sparse_layer_step for eager).  alt_seed seeds the alternative draws (default seed + 7)."""
    step = step or compiled_sparse_step()
    sampled = route_credit == 'sampled'
    layers = model._stacked(0)
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    n = len(rows); dtype = model.embedding.weight.dtype
    lengths = torch.tensor([len(r['events']) for r in rows])
    T = int(lengths.max())
    mem = [torch.zeros(n, H, U, P, dtype=dtype) for _ in range(D)]
    arr = [torch.zeros(n, H, U, dtype=torch.float64) for _ in range(D)]
    seen = [torch.zeros(n, H, U, dtype=torch.bool) for _ in range(D)]
    reads = [L['key'].view(H, U, P).expand(n, H, U, P) for L in layers]
    stamp_rows = torch.zeros(n, T, dtype=torch.float64); mark_rows = []
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
    alt_gen = torch.Generator().manual_seed(seed + 7 if alt_seed is None else alt_seed)

    def transport(value, age, depth, head):
        age = age.clamp_min(0)
        rate = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype)[:, None] * rate).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for k in range(T):
            active = lengths > k
            x = model.embedding.weight[0][None] + model.content(mark_rows[:, k])
            arrival = stamp_rows[:, k]
            read_time = torch.where(has_ctx, torch.maximum(arrival, ctx_arr.max(-1).values), arrival)
            arrival = read_time
            context = torch.cat([transport(ctx_vals[:, h * P:(h + 1) * P], read_time - ctx_arr[:, h], D - 1, h) for h in range(H)], -1)
            x = torch.where(has_ctx[:, None], F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context, (model.total_payload,)), x)
            for depth in range(D):
                Lp = layers[depth]
                noise = torch.stack([torch.empty(U, dtype=torch.float64).exponential_() for _ in range(H)])
                alt_noise = torch.empty(H, U, dtype=torch.float64).exponential_(generator=alt_gen) if sampled else noise
                mix = model.channel_mix[depth]
                x, arrival, mem[depth], arr[depth], seen[depth], reads[depth], values, arrivals = step(
                    x, arrival, mem[depth], arr[depth], seen[depth], reads[depth], active, noise, alt_noise, mix.weight,
                    mix.bias, Lp['query'], Lp['key'], Lp['key_read'], Lp['clock_bias'], Lp['control_w'], Lp['control_b'],
                    Lp['rate'], Lp['frequency'], Lp['input'], Lp['output'], Lp['gate_w'], Lp['gate_b'], Lp['gain'],
                    model.transport_rate[depth], model.transport_frequency[depth], sampled)
            ctx_vals = torch.where(active[:, None], values.reshape(n, H * P), ctx_vals)
            ctx_arr = torch.where(active[:, None], arrivals, ctx_arr)
            has_ctx = has_ctx | active
            logits_k = model.head(x)
            if all_logits:
                every.append(torch.where(active[:, None], logits_k, torch.zeros_like(logits_k)))
            out = torch.where((lengths == k + 1)[:, None], logits_k, out)
    return torch.stack(every, 1) if all_logits else out
