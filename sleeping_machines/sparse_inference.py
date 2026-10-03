"""Winner-only native inference with cached key reads (THEORY §414).

The batched path (batched_episodes.batched_logits) evaluates every unit's proposal and then selects one per race.  At
inference only the winner's proposal is needed: the forwarded value, the memory write and the control gates are the
winner's.  The race scores read key + key_read · m from the *stored* memory m (lazy decay is applied only when a slot is
written), so each slot's key read changes only when that slot is written.  Caching it per slot leaves, per head and
layer and event: U score dot products of length P, and the winner's maps (control, input, output, gate) plus one key_read
refresh of the written slot.  Pool size then costs U·P per head instead of about 4·U·P² multiply-adds.

Contract (tests/test_sparse_inference.py): logits equal batched_logits (no forces) within float64 rounding, with identical
race winners, on variable-length episodes, so the same seeds give the same routes.
"""
import math

import torch
from torch.nn import functional as F

from .parallel_stream_language import precise_rotate


@torch.no_grad()
def sparse_logits(model, rows, seed, all_logits=False):
    source = 0
    layers = model._stacked(source)
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    n = len(rows); dtype = model.embedding.weight.dtype
    lengths = torch.tensor([len(r['events']) for r in rows])
    T = int(lengths.max())
    mem = [torch.zeros(n, H, U, P, dtype=dtype) for _ in range(D)]
    arr = [torch.zeros(n, H, U, dtype=torch.float64) for _ in range(D)]
    seen = [torch.zeros(n, H, U, dtype=torch.bool) for _ in range(D)]
    # cached key reads of the stored memories: key + key_read @ m (m = 0 initially)
    reads = [L['key'].view(H, U, P).expand(n, H, U, P).clone() for L in layers]
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
    lanes = torch.arange(n)

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
                mixed = model.channel_mix[depth](x)
                all_features = F.layer_norm(mixed, (model.total_payload,))
                incoming = mixed.view(n, H, P)
                query = torch.einsum('hpd,ld->lhp', Lp['query'], all_features)
                scores = ((query[:, :, None, :] * reads[depth]).sum(-1) / math.sqrt(P) + Lp['clock_bias'].view(H, U)).clamp(-12, 12)
                values, arrivals = [], []
                for head in range(H):
                    noise = torch.empty(U, dtype=torch.float64).exponential_()          # same draw order as batched
                    times = noise[None, :] / scores[:, head].to(torch.float64).exp()
                    first, w = times.min(-1)                                              # (n,)
                    idx = head * U + w                                                    # stacked unit index
                    xh = incoming[:, head]                                                # (n, P)
                    m = mem[depth][lanes, head, w]                                        # stored winner memory
                    prev = torch.where(seen[depth][lanes, head, w], arr[depth][lanes, head, w], arrival)
                    controls = torch.einsum('lcp,lp->lc', Lp['control_w'].view(H * U, 2, P)[idx], F.layer_norm(xh, (P,))) \
                        + Lp['control_b'].view(H * U, 2)[idx]
                    forget = F.softplus(controls[:, 0]) / math.log(2)
                    write = 2 * torch.sigmoid(controls[:, 1])
                    age = (arrival - prev).clamp_min(0)
                    decay = torch.exp(-age.to(dtype)[:, None] * Lp['rate'].view(H * U, P // 2)[idx] * forget[:, None]).repeat_interleave(2, -1)
                    m_new = precise_rotate(m * decay, age[:, None] * Lp['frequency'].view(H * U, P // 2)[idx])
                    m_new = m_new + write[:, None] * torch.einsum('lpq,lq->lp', Lp['input'].view(H * U, P, P)[idx], xh)
                    y = F.layer_norm(torch.einsum('lpq,lq->lp', Lp['output'].view(H * U, P, P)[idx], m_new) + xh, (P,))
                    gate = torch.einsum('lpq,lq->lp', Lp['gate_w'].view(H * U, P, P)[idx], F.gelu(y)) + Lp['gate_b'].view(H * U, P)[idx]
                    value = xh + Lp['gain'] * y * torch.sigmoid(gate)
                    # winner-only state update (active lanes) and its cached key read
                    write_lane = active
                    mem[depth][lanes[write_lane], head, w[write_lane]] = m_new[write_lane]
                    arr[depth][lanes[write_lane], head, w[write_lane]] = arrival[write_lane]
                    seen[depth][lanes[write_lane], head, w[write_lane]] = True
                    refreshed = Lp['key'].view(H * U, P)[idx] + torch.einsum('lpq,lq->lp', Lp['key_read'].view(H * U, P, P)[idx], m_new)
                    reads[depth][lanes[write_lane], head, w[write_lane]] = refreshed[write_lane]
                    values.append(value); arrivals.append(arrival + .001 + .010 * first / (1 + first))
                arrivals = torch.stack(arrivals, -1)
                arrival = arrivals.max(-1).values
                x = torch.cat([transport(values[h], arrival - arrivals[:, h], depth, h) for h in range(H)], -1)
            ctx_vals = torch.where(active[:, None], torch.cat(values, -1), ctx_vals)
            ctx_arr = torch.where(active[:, None], arrivals, ctx_arr)
            has_ctx = has_ctx | active
            last = lengths == k + 1
            logits_k = model.head(x)
            if all_logits:
                every.append(torch.where(active[:, None], logits_k, torch.zeros_like(logits_k)))
            out = torch.where(last[:, None], logits_k, out)
    return torch.stack(every, 1) if all_logits else out
