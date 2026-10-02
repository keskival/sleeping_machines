"""Frozen native batched kernel with reception-phase offset only (theory90).

Copied from batched_addressed_fit.forward; physical ages/delays/state unchanged.
"""
import math
import torch
from torch.nn import functional as F
from .fast_native_core import FastNativeCoreMixin
from .parallel_stream_language import precise_rotate
from .batched_addressed_fit import BatchedTemporalRoute


def forward(model, rows, seed, terminal_pairs=False):
    """Same observed event schedule per clip; independent addressed memory."""
    if model.sources != 1 or model.credit != 'counterfactual' or not rows:
        raise ValueError('Nonempty one-source native counterfactual clips required')
    schedule = [t for t, _ in rows[0]['events']]
    if any([t for t, _ in row['events']] != schedule for row in rows):
        raise ValueError('Matched dense packet schedules required; no padding/reordering')
    if terminal_pairs and model.heads != 2:
        raise ValueError('Bounded two-head terminal pair mode required')
    if terminal_pairs: raise ValueError("Offset admission uses native local credit only")
    B, H, U, P = len(rows), model.heads, model.pool, model.payload; nodes = H*U
    dtype = model.embedding.weight.dtype; device = model.embedding.weight.device
    content = torch.tensor([[list(x) for _, x in row['events']] for row in rows], dtype=dtype, device=device)
    if terminal_pairs and not bool((content[:, -1, -1] == 1).all()):
        raise ValueError('Observed terminal query flags required')
    layers = FastNativeCoreMixin._stacked(model, 0)
    memories = [content.new_zeros(B, nodes, P) for _ in range(model.depth)]
    previous = [torch.zeros(B, nodes, dtype=torch.float64, device=device) for _ in layers]
    seen = [torch.zeros(B, nodes, dtype=torch.bool, device=device) for _ in layers]
    context = context_times = None; winners = []; terminal = None
    tr = F.softplus(model.transport_rate)+1e-6; tf = model.transport_frequency.to(torch.float64)
    def align(values, times, arrival, depth):
        ages = (arrival[:, None]-times).clamp_min(0)
        decay = torch.exp(-ages.to(dtype)[:, :, None]*tr[depth][None]).repeat_interleave(2, -1)
        angles = ages[:, :, None]*tf[depth][None]+model.evolution_phase(depth)[None]
        rotated = precise_rotate(values*decay, angles)
        return rotated.reshape(B, H*P)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for event, timestamp in enumerate(schedule):
            arrival = torch.full((B,), timestamp, dtype=torch.float64, device=device)
            x = model.embedding.weight[0]+model.content(content[:, event])
            if context is not None:
                arrival = torch.maximum(arrival, context_times.max(-1).values)
                old = align(context, context_times, arrival, model.depth-1)
                x = F.layer_norm(x+torch.sigmoid(model.source_gate(x))*old, (H*P,))
            for depth, L in enumerate(layers):
                mixed = model.channel_mix[depth](x)
                features = F.layer_norm(mixed, (H*P,))
                incoming = mixed.reshape(B, H, P)
                query = torch.einsum('hpd,bd->bhp', L['query'], features)
                xu = incoming.repeat_interleave(U, 1); qu = query.repeat_interleave(U, 1)
                memory = memories[depth]
                read = L['key'][None]+torch.einsum('npq,bnq->bnp', L['key_read'], memory)
                scores = ((qu*read).sum(-1)/math.sqrt(P)+L['clock_bias'][None]).reshape(B,H,U).clamp(-12,12)
                controls = torch.einsum('ncp,bnp->bnc', L['control_w'], F.layer_norm(xu,(P,)))+L['control_b'][None]
                forget = F.softplus(controls[..., 0])/math.log(2.); write = 2*torch.sigmoid(controls[..., 1])
                age = torch.where(seen[depth], (arrival[:,None]-previous[depth]).clamp_min(0), 0.)
                decay = torch.exp(-age.to(dtype)[...,None]*L['rate'][None]*forget[...,None]).repeat_interleave(2,-1)
                proposed = precise_rotate(memory*decay, age[...,None]*L['frequency'][None])
                proposed = proposed+write[...,None]*torch.einsum('npq,bnq->bnp',L['input'],xu)
                y = F.layer_norm(torch.einsum('npq,bnq->bnp',L['output'],proposed)+xu,(P,))
                gate = torch.einsum('npq,bnq->bnp',L['gate_w'],F.gelu(y))+L['gate_b'][None]
                values = (xu+L['gain']*y*torch.sigmoid(gate)).reshape(B,H,U,P)
                # Keep one draw call per head, the original generator contract.
                noise = torch.stack([torch.empty(U,dtype=torch.float64,device=device).exponential_() for _ in range(H)])
                chosen, delay, winner = BatchedTemporalRoute.apply(scores, values, noise)
                winners.append(winner.detach())
                mask = F.one_hot(winner,U).reshape(B,nodes).bool()
                memories[depth] = torch.where(mask[...,None], proposed, memory)
                previous[depth] = torch.where(mask, arrival[:,None], previous[depth]); seen[depth] = seen[depth] | mask
                times = arrival[:,None]+delay; arrival = times.max(-1).values
                x = align(chosen,times,arrival,depth)
                if terminal_pairs and event == len(schedule)-1 and depth == model.depth-1:
                    terminal = (scores,values,times,arrival,depth)
            context,context_times = chosen,times
    logits = model.head(x); pairs = None
    if terminal is not None:
        scores,values,times,arrival,depth = terminal
        pair_logits=[]; weights=[]; probabilities=scores.softmax(-1)
        for first in range(U):
            for second in range(U):
                both=torch.stack((values[:,0,first],values[:,1,second]),1)
                pair_logits.append(model.head(align(both,times,arrival,depth)))
                weights.append(probabilities[:,0,first]*probabilities[:,1,second])
        pairs=(torch.stack(pair_logits,1),torch.stack(weights,1))
    return logits,dict(memories=memories,arrivals=previous,seen=seen,context=context,
        context_times=context_times,winners=winners,events=len(schedule)*B),pairs
