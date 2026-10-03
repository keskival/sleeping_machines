"""Episode-batched native core with gradients: many independent episodes as lanes of one pass (THEORY §405).

The DVS drivers give every episode of a pass the same race-noise seed (common per-pass draws; identity never enters
randomness).  Episodes can therefore run as lanes that share each race's noise draw, indexed by event position, and
reproduce the sequential per-episode runs exactly, including episodes of different lengths (a lane stops after its last
event).  The arithmetic is the fast training path's with a leading lane dimension; races use the factorized law
(winner payload credit; common first-time clock credit dT/ds_i = -T pi_i), i.e. the race of dvs_local_expectation's
realized branch.  Optional forces (race, alternative) per lane give shadow lanes in the same pass (no gradient use).

Contract (tests/test_batched_episodes.py): per-lane logits and the summed parameter gradients equal sequential runs.
"""
import math

import torch
from torch.nn import functional as F

from .parallel_stream_language import precise_rotate


class LaneRace(torch.autograd.Function):
    """per-lane factorized race over U candidates with shared noise; forced lanes switch identity at the first time."""

    @staticmethod
    def forward(ctx, scores, proposals, noise, forced_alt):
        rates = scores.to(torch.float64).exp()                              # (n, U)
        times = noise[None, :] / rates
        first, winner = times.min(-1)
        winner = torch.where(forced_alt >= 0, forced_alt, winner)
        ctx.save_for_backward(rates, first, winner)
        ctx.shape = proposals.shape
        lanes = torch.arange(len(winner))
        return proposals[lanes, winner], .001 + .010 * first / (1 + first), winner

    @staticmethod
    def backward(ctx, error_value, error_delay, error_winner):
        rates, first, winner = ctx.saved_tensors
        n, U, P = ctx.shape
        lanes = torch.arange(n)
        value_credit = None
        if error_value is not None:
            value_credit = error_value.new_zeros(n, U, P)
            value_credit[lanes, winner] = error_value
        credit = torch.zeros_like(rates)
        if error_delay is not None:
            pi = rates / rates.sum(-1, keepdim=True)
            credit = (error_delay * .010 / (1 + first).square())[:, None] * (-first[:, None] * pi)
        return credit.to(error_value.dtype if error_value is not None else rates.dtype), value_credit, None, None


def linear_route_credit(scores, proposals):
    """Zero-valued surrogate whose score gradient is the linearized local-expectation route credit
    d E[L] / d s_i ~= pi_i g.(v_i - v_bar) (g: the loss gradient at the realized value).  Values get no extra credit.
    scores (n, U), proposals (n, U, P); returns (n, P) exact zeros (THEORY §413)."""
    pi = torch.softmax(scores.to(torch.float64), -1).to(proposals.dtype)
    return ((pi - pi.detach())[..., None] * proposals.detach()).sum(-2)


def batched_logits(model, rows, seed, forces=None, record=None, all_logits=False, route_credit=None):
    """rows: episodes (dict with 'events'); forces: per-lane (race, alt) or None; record: list receiving each race's
    (n, U) scores in race order.  Returns (n, classes) final logits, or (n, T, classes) logits after every event when
    all_logits (positions beyond an episode's length are zero).  route_credit='linear' adds the linearized value
    credit to the race scores (forward values unchanged)."""
    source = 0
    layers = model._stacked(source)
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    n = len(rows); dtype = model.embedding.weight.dtype
    forces = forces or [None] * n
    force_race = torch.tensor([f[0] if f is not None else -1 for f in forces])
    force_alt = torch.tensor([f[1] if f is not None else 0 for f in forces])
    lengths = torch.tensor([len(r['events']) for r in rows])
    T = int(lengths.max())
    mem = torch.zeros(n, D, H, U, P, dtype=dtype)
    arr = torch.zeros(n, D, H, U, dtype=torch.float64)
    seen = torch.zeros(n, D, H, U, dtype=torch.bool)
    ctx_vals = torch.zeros(n, H * P, dtype=dtype); ctx_arr = torch.zeros(n, H, dtype=torch.float64)
    has_ctx = torch.zeros(n, dtype=torch.bool)
    out = torch.zeros(n, model.head.out_features, dtype=dtype)
    every = []
    race = 0
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
            stamps = torch.tensor([float(r['events'][k][0]) if len(r['events']) > k else 0. for r in rows], dtype=torch.float64)
            marks = torch.stack([torch.as_tensor(r['events'][k][1] if len(r['events']) > k else r['events'][0][1], dtype=dtype)
                                 for r in rows])
            x = model.embedding.weight[source][None] + model.content(marks)
            arrival = stamps.clone()
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
                    alt = torch.where(force_race == race, force_alt, torch.full_like(force_alt, -1))
                    if record is not None:
                        record.append(scores[:, head])
                    value, delay, winner = LaneRace.apply(scores[:, head], proposals[:, head], noise, alt)
                    if route_credit == 'linear':
                        value = value + linear_route_credit(scores[:, head], proposals[:, head])
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
            last = lengths == k + 1
            logits_k = model.head(x)
            if all_logits:
                every.append(torch.where(active[:, None], logits_k, torch.zeros_like(logits_k)))
            out = torch.where(last[:, None], logits_k, out)
    return torch.stack(every, 1) if all_logits else out
