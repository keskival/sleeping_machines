"""Unchanged native races with detached causal token-boundary snapshots (118)."""
import math

import torch
from torch.nn import functional as F

from .parallel_stream_language import precise_rotate
from .native_stream_language import NativeLanguageState


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


def batched_chunks(model, rows, seed, initial_states=None, forces=None, record=None, winners=None, snapshots=None):
    """rows: episodes (dict with 'events'); forces: per-lane (race, alt) or None; record: list receiving each race's
    (n, U) scores in race order.  Returns (n, classes) final logits."""
    source = 0
    layers = model._stacked(source)
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    n = len(rows); dtype = model.embedding.weight.dtype
    if snapshots is not None and n!=1:
        raise ValueError('Snapshots are only for the one factual lane')
    forces = forces or [None] * n
    force_race = torch.tensor([f[0] if f is not None else -1 for f in forces])
    force_alt = torch.tensor([f[1] if f is not None else 0 for f in forces])
    lengths = torch.tensor([len(r['events']) for r in rows])
    T = int(lengths.max())
    if model.sources != 1 or not rows or any(not r['events'] for r in rows):
        raise ValueError('Single observed source and nonempty causal chunks required')
    initial_states = initial_states or [model.new_state() for _ in rows]
    if len(initial_states) != n:
        raise ValueError('One detached entering state per lane required')
    for st in initial_states:
        tensors = list(st.memories.values()) + list(st.arrivals.values())
        tensors += [v for pair in st.contexts.values() for v in pair]
        if any(v.requires_grad for v in tensors):
            raise ValueError('Credit-boundary entering state must be explicitly detached')
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
    outputs = []
    race = 0
    lanes = torch.arange(n)

    def transport(value, age, depth, head):
        age = age.clamp_min(0)
        rate = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype)[:, None] * rate).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    def snapshot(k):
        # Copy only actual pre-token state. No future token/target or graph enters.
        with torch.no_grad():
            old=initial_states[0];st=NativeLanguageState()
            st.memories={(d,h,0,u):mem[0,d,h,u].detach().clone() for d in range(D)
                         for h in range(H) for u in range(U) if bool(seen[0,d,h,u])}
            st.arrivals={key:arr[0,key[0],key[1],key[3]].detach().clone() for key in st.memories}
            st.contexts={0:(ctx_vals[0].detach().clone(),ctx_arr[0].detach().clone())} if bool(has_ctx[0]) else {}
            st.visited_units=set(old.visited_units)|set(st.memories)
            st.events=old.events+k;st.candidate_scores=old.candidate_scores+k*D*H*U
            st.selected_updates=old.selected_updates+k*D*H
            st.counterfactual_values=old.counterfactual_values+k*D*H*U
            st.last_input_time=float(rows[0]['events'][k-1][0]) if k else old.last_input_time
            st.queue_wait_sum=old.queue_wait_sum+float(queue_wait[0].detach())
            return dict(state=st,rng=torch.get_rng_state().clone(),token=k)

    with torch.random.fork_rng():
        torch.set_rng_state(seed) if isinstance(seed, torch.Tensor) else torch.manual_seed(seed)
        for k in range(T):
            if snapshots is not None:
                snapshots.append(snapshot(k))
            active = lengths > k
            stamps = torch.tensor([float(r['events'][k][0]) if len(r['events']) > k else 0. for r in rows], dtype=torch.float64)
            marks = torch.stack([torch.as_tensor(r['events'][k][1] if len(r['events']) > k else r['events'][0][1], dtype=dtype)
                                 for r in rows])
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
                    alt = torch.where(force_race == race, force_alt, torch.full_like(force_alt, -1))
                    if record is not None:
                        record.append(scores[:, head])
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
            outputs.append(torch.where(active[:,None], model.head(x), torch.zeros(n,model.classes,dtype=dtype)))
        end_rng = torch.get_rng_state().clone()
    states = []
    for lane, (row, old) in enumerate(zip(rows, initial_states)):
        st = NativeLanguageState()
        st.memories = {(d,h,0,u):mem[lane,d,h,u] for d in range(D) for h in range(H)
                       for u in range(U) if bool(seen[lane,d,h,u])}
        st.arrivals = {key:arr[lane,key[0],key[1],key[3]] for key in st.memories}
        st.contexts = {0:(ctx_vals[lane],ctx_arr[lane])}
        st.visited_units = set(old.visited_units) | set(st.memories)
        st.events = old.events + len(row['events'])
        st.candidate_scores = old.candidate_scores + len(row['events'])*D*H*U
        st.selected_updates = old.selected_updates + len(row['events'])*D*H
        st.counterfactual_values = old.counterfactual_values + len(row['events'])*D*H*U
        st.last_input_time = float(row['events'][-1][0])
        st.queue_wait_sum = old.queue_wait_sum + float(queue_wait[lane].detach())
        states.append(st)
    return torch.stack(outputs,1), states, end_rng
