"""Shadow lanes: every counterfactual replay of an episode in one lane-batched, gradient-free pass (THEORY §404.2).

Lane l carries its own dense copy of the persistent state (memories, arrivals, written-slot masks, carried context).  All
lanes see the same events.  Each race draws its noise once and shares it across lanes, in the factual run's exact
draw order, so every lane is coupled to the factual run by common random numbers.  A lane with force (r, i) takes
alternative i at race r at its own first-arrival time (the conditional race law, note 92) and then evolves on its
own.  A lane without a force reproduces the factual run.  The arithmetic is the fast training path's (FastNativeCoreMixin)
with a leading lane dimension.  Per-lane losses equal the sequential forked replays (contract-tested).

Single observed source (index 0), as in the DVS and joint-task drivers.
"""
import math

import torch
from torch.nn import functional as F

from .parallel_stream_language import precise_rotate


@torch.no_grad()
def shadow_losses(model, row, seed, forces):
    """forces: list of (race, alternative) or None per lane; returns per-lane episode losses (float64)."""
    source = 0
    layers = model._stacked(source)
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    n = len(forces); dtype = model.embedding.weight.dtype
    mem = torch.zeros(n, D, H, U, P, dtype=dtype)
    arr = torch.zeros(n, D, H, U, dtype=torch.float64)
    seen = torch.zeros(n, D, H, U, dtype=torch.bool)
    ctx_vals = ctx_arr = None
    race = 0
    force_race = torch.tensor([f[0] if f is not None else -1 for f in forces])
    force_alt = torch.tensor([f[1] if f is not None else 0 for f in forces])
    lanes = torch.arange(n)

    def transport(value, age, depth, head):          # value (n, P), age (n,)
        age = age.clamp_min(0)
        rate = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype)[:, None] * rate).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for timestamp, content in row['events']:
            mark = torch.as_tensor(content, dtype=dtype)
            x = (model.embedding.weight[source] + model.content(mark)).expand(n, -1)
            arrival = torch.full((n,), float(timestamp), dtype=torch.float64)
            if ctx_vals is not None:
                read_time = torch.maximum(arrival, ctx_arr.max(-1).values); arrival = read_time
                context = torch.cat([transport(ctx_vals[:, h * P:(h + 1) * P], read_time - ctx_arr[:, h], D - 1, h)
                                     for h in range(H)], -1)
                x = F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context, (model.total_payload,))
            for depth in range(D):
                Lp = layers[depth]
                mixed = model.channel_mix[depth](x)
                all_features = F.layer_norm(mixed, (model.total_payload,))
                incoming = mixed.view(n, H, P)
                query = torch.einsum('hpd,ld->lhp', Lp['query'], all_features)
                m = mem[:, depth]                                                   # (n, H, U, P)
                prev = torch.where(seen[:, depth], arr[:, depth], arrival[:, None, None])
                x_u = incoming[:, :, None, :].expand(n, H, U, P)
                q_u = query[:, :, None, :].expand(n, H, U, P)
                key = Lp['key'].view(H, U, P); key_read = Lp['key_read'].view(H, U, P, P)
                read = key + torch.einsum('hupq,lhuq->lhup', key_read, m)
                scores = ((q_u * read).sum(-1) / math.sqrt(P) + Lp['clock_bias'].view(H, U)).clamp(-12, 12)
                controls = torch.einsum('hucp,lhup->lhuc', Lp['control_w'].view(H, U, 2, P),
                                        F.layer_norm(x_u, (P,))) + Lp['control_b'].view(H, U, 2)
                forget = F.softplus(controls[..., 0]) / math.log(2)
                write = 2 * torch.sigmoid(controls[..., 1])
                age = (arrival[:, None, None] - prev).clamp_min(0)
                rate = Lp['rate'].view(H, U, P // 2); freq = Lp['frequency'].view(H, U, P // 2)
                decay = torch.exp(-age.to(dtype)[..., None] * rate * forget[..., None]).repeat_interleave(2, -1)
                m = precise_rotate(m * decay, age[..., None] * freq)
                m = m + write[..., None] * torch.einsum('hupq,lhuq->lhup', Lp['input'].view(H, U, P, P), x_u)
                y = F.layer_norm(torch.einsum('hupq,lhuq->lhup', Lp['output'].view(H, U, P, P), m) + x_u, (P,))
                gate = torch.einsum('hupq,lhuq->lhup', Lp['gate_w'].view(H, U, P, P), F.gelu(y)) + Lp['gate_b'].view(H, U, P)
                proposals = x_u + Lp['gain'] * y * torch.sigmoid(gate)
                values, arrivals = [], []
                for head in range(H):
                    noise = torch.empty(U, dtype=torch.float64).exponential_()      # shared across lanes
                    times = noise[None, :] / scores[:, head].to(torch.float64).exp()
                    first, winner = times.min(-1)
                    forced = force_race == race
                    winner = torch.where(forced, force_alt, winner)
                    race += 1
                    values.append(proposals[lanes, head, winner])
                    mem[lanes, depth, head, winner] = m[lanes, head, winner]
                    arr[lanes, depth, head, winner] = arrival
                    seen[lanes, depth, head, winner] = True
                    arrivals.append(arrival + .001 + .010 * first / (1 + first))
                arrivals = torch.stack(arrivals, -1)                               # (n, H)
                arrival = arrivals.max(-1).values
                x = torch.cat([transport(values[h], arrival - arrivals[:, h], depth, h) for h in range(H)], -1)
            ctx_vals, ctx_arr = torch.cat(values, -1), arrivals
    logits = model.head(x)
    return F.cross_entropy(logits, torch.full((n,), row['target']), reduction='none').double()
