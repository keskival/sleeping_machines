"""Unadmitted CPU-only compact suffix prototype; no active driver imports this."""
import math
import torch
from torch.nn import functional as F
from .parallel_stream_language import precise_rotate
from .causal_language_shadow_cached_prefix import LaneRace


@torch.no_grad()
def compact_suffix_returns(model, observed, targets, snapshots, forces, winners=None):
    """Sorted losing forces; one absolute-token loop, growing active lane prefix."""
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    if model.embedding.weight.device.type != 'cpu' or model.sources != 1:
        raise ValueError('Diagnostic prototype requires CPU and single source')
    T = len(observed['events'])
    if not forces or len(targets) != T or len(snapshots) != T:
        raise ValueError('Nonempty forces and matching complete token snapshots required')
    starts = [r // (D * H) for r, i in forces]
    if starts != sorted(starts) or starts[0] != 0 or starts[-1] >= T:
        raise ValueError('Forces must be sorted, cover initial token, and lie inside chunk')
    if len(set(forces)) != len(forces) or any(not 0 <= i < U for _, i in forces):
        raise ValueError('Unique valid alternative identities required')
    initial_states = [snapshots[k]['state'] for k in starts]
    for state in initial_states:
        tensors = list(state.memories.values()) + list(state.arrivals.values())
        tensors += [v for pair in state.contexts.values() for v in pair]
        if any(v.requires_grad for v in tensors):
            raise ValueError('Detached causal snapshots required')
    source = 0
    layers = model._stacked(source)
    total_lanes = len(forces)
    n = total_lanes
    dtype = model.embedding.weight.dtype
    force_race = torch.tensor([r for r, i in forces])
    force_alt = torch.tensor([i for r, i in forces])
    seed = snapshots[0]['rng']
    mem = torch.stack([torch.stack([st.memories.get((d,h,0,u), torch.zeros(P,dtype=dtype))
            for d in range(D) for h in range(H) for u in range(U)]).reshape(D,H,U,P)
            for st in initial_states])
    arr = torch.stack([torch.stack([st.arrivals.get((d,h,0,u), torch.zeros((),dtype=torch.float64))
            for d in range(D) for h in range(H) for u in range(U)]).reshape(D,H,U)
            for st in initial_states])
    seen = torch.tensor([[(d,h,0,u) in st.memories for d in range(D) for h in range(H)
            for u in range(U)] for st in initial_states]).reshape(n,D,H,U)
    ctx_vals = torch.stack([st.contexts.get(0,(torch.zeros(H*P,dtype=dtype),None))[0] for st in initial_states])
    ctx_arr = torch.stack([st.contexts.get(0,(None,torch.zeros(H,dtype=torch.float64)))[1] for st in initial_states])
    has_ctx = torch.tensor([0 in st.contexts for st in initial_states])
    queue_wait = torch.zeros(n,dtype=torch.float64)
    returns = torch.zeros(total_lanes, dtype=dtype)
    race = 0

    def transport(value, age, depth, head):
        age = age.clamp_min(0)
        rate = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype)[:, None] * rate).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    with torch.random.fork_rng():
        torch.set_rng_state(seed) if isinstance(seed, torch.Tensor) else torch.manual_seed(seed)
        for k in range(T):
            # Only activated lanes enter any model arithmetic.
            n = sum(start <= k for start in starts)
            active = torch.ones(n, dtype=torch.bool)
            full = (mem, arr, seen, ctx_vals, ctx_arr, has_ctx, queue_wait)
            mem, arr, seen, ctx_vals, ctx_arr, has_ctx, queue_wait = [v[:n] for v in full]
            stamps = torch.full((n,), float(observed['events'][k][0]), dtype=torch.float64)
            marks = torch.as_tensor(observed['events'][k][1], dtype=dtype)[None].expand(n, -1)
            x = model.embedding.weight[source][None] + model.content(marks)
            arrival = stamps.clone()
            read_time = torch.where(has_ctx, torch.maximum(arrival, ctx_arr.max(-1).values), arrival)
            queue_wait = queue_wait + torch.where(active, read_time - arrival, torch.zeros_like(arrival))
            arrival = read_time
            context = torch.cat([transport(ctx_vals[:, h * P:(h + 1) * P], read_time - ctx_arr[:, h], D - 1, h) for h in range(H)], -1)
            x = torch.where(has_ctx[:, None], F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context, (model.total_payload,)), x)
            for depth in range(D):
                Lp = layers[depth]
                mixed = model.channel_mix[depth](x)
                all_features = F.layer_norm(mixed, (model.total_payload,))
                incoming = mixed.view(n, H, P)
                query = torch.einsum('hpd,ld->lhp', Lp['query'], all_features)
                m = mem[:, depth]
                prev = torch.where(seen[:, depth], arr[:, depth], arrival[:, None, None])
                x_u = incoming[:, :, None, :].expand(n, H, U, P)
                q_u = query[:, :, None, :].expand(n, H, U, P)
                read = Lp['key'].view(H, U, P) + torch.einsum('hupq,lhuq->lhup', Lp['key_read'].view(H, U, P, P), m)
                scores = ((q_u * read).sum(-1) / math.sqrt(P) + Lp['clock_bias'].view(H, U)).clamp(-12, 12)
                controls = torch.einsum('hucp,lhup->lhuc', Lp['control_w'].view(H, U, 2, P),
                                        F.layer_norm(x_u, (P,))) + Lp['control_b'].view(H, U, 2)
                forget = F.softplus(controls[..., 0]) / math.log(2)
                write = 2 * torch.sigmoid(controls[..., 1])
                age = (arrival[:, None, None] - prev).clamp_min(0)
                decay = torch.exp(-age.to(dtype)[..., None] * Lp['rate'].view(H, U, P // 2) * forget[..., None]).repeat_interleave(2, -1)
                m_new = precise_rotate(m * decay, age[..., None] * Lp['frequency'].view(H, U, P // 2))
                m_new = m_new + write[..., None] * torch.einsum('hupq,lhuq->lhup', Lp['input'].view(H, U, P, P), x_u)
                y = F.layer_norm(torch.einsum('hupq,lhuq->lhup', Lp['output'].view(H, U, P, P), m_new) + x_u, (P,))
                gate = torch.einsum('hupq,lhuq->lhup', Lp['gate_w'].view(H, U, P, P), F.gelu(y)) + Lp['gate_b'].view(H, U, P)
                proposals = x_u + Lp['gain'] * y * torch.sigmoid(gate)
                values, arrivals = [], []
                new_mem, new_arr, new_seen = mem[:, depth], arr[:, depth], seen[:, depth]
                for head in range(H):
                    noise = torch.empty(U, dtype=torch.float64).exponential_()     # shared across lanes
                    alt = torch.where(force_race[:n] == race, force_alt[:n], torch.full_like(force_alt[:n], -1))
                    value, delay, winner = LaneRace.apply(scores[:, head], proposals[:, head], noise, alt)
                    if winners is not None:
                        winners.append(winner.detach().clone())
                    race += 1
                    onehot = F.one_hot(winner, U).to(torch.bool) & active[:, None]          # write only active lanes
                    sel = onehot[..., None]
                    head_mem = torch.where(sel, m_new[:, head], new_mem[:, head])
                    new_mem = torch.cat([new_mem[:, :head], head_mem[:, None], new_mem[:, head + 1:]], 1)
                    new_arr = new_arr.clone(); new_arr[:, head] = torch.where(onehot, arrival[:, None], new_arr[:, head])
                    new_seen = new_seen.clone(); new_seen[:, head] = new_seen[:, head] | onehot
                    values.append(value); arrivals.append(arrival + delay)
                mem = torch.cat([mem[:, :depth], new_mem[:, None], mem[:, depth + 1:]], 1)
                arr = arr.clone(); arr[:, depth] = new_arr
                seen = seen.clone(); seen[:, depth] = new_seen
                arrivals = torch.stack(arrivals, -1)
                arrival = arrivals.max(-1).values
                x = torch.cat([transport(values[h], arrival - arrivals[:, h], depth, h) for h in range(H)], -1)
            ctx_vals = torch.where(active[:, None], torch.cat(values, -1), ctx_vals)
            ctx_arr = torch.where(active[:, None], arrivals, ctx_arr)
            has_ctx = has_ctx | active
            target = targets[k].expand(n)
            returns[:n] += F.cross_entropy(model.head(x), target, reduction='none')
            for destination, current in zip(full, (mem, arr, seen, ctx_vals, ctx_arr, has_ctx, queue_wait)):
                destination[:n].copy_(current)
            mem, arr, seen, ctx_vals, ctx_arr, has_ctx, queue_wait = full
        end_rng = torch.get_rng_state().clone()
    return returns, dict(shadow_lanes=total_lanes,
        shadow_events=sum(T-k for k in starts), token_loop_iterations=T,
        factual_end_rng=end_rng)
