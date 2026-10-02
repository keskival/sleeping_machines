"""Batched training path for the native temporal core: same arithmetic, fewer operations (THEORY §393 scale).

In training every receiver unit of every head proposes (losing routes need counterfactual values), so one event
costs depth x heads x pool separate `TemporalUnit.propose` calls of ~20 small operations each, and backward walks
the same fragmented graph.  This subclass stacks each layer's unit parameters once per chunk and evaluates all
heads' and units' proposals of a layer as batched tensor operations.  The race for each head is unchanged
(TemporalRoute, called per head in the original order, so race noise is drawn identically), as are state
bookkeeping, timing, transport and the read path.  Units without a previous arrival use age 0, which leaves the
memory unchanged and contributes zero derivative to rate/frequency/forget, matching the reference's skipped
branch.  Evaluation (one proposal per head, winner only) uses the reference path.  As a mixin placed first in the
bases (fast_class), it accelerates any count-carrying, gated or pooled composition built on the native core.

Equivalence with NativeStreamLanguageModel (outputs, every parameter gradient, persistent state) is
contract-tested in double precision.  The reference sources are untouched.
"""
import math

import torch
from torch.nn import functional as F

from .native_stream_language import NativeStreamLanguageModel
from .parallel_stream_language import precise_rotate


class FastNativeCoreMixin:
    """Place first in the bases: wraps any subclass's forward_chunk and replaces the core's training consume_event."""
    _fast_layers = None

    def _stacked(self, source):
        """Per layer: unit parameters stacked over (head, unit); built once per chunk, gradients flow to originals."""
        layers = []
        for depth in range(self.depth):
            units = [u for h in range(self.heads) for u in self.units[depth][h][source]]
            layers.append(dict(
                key=torch.stack([u.key for u in units]),
                key_read=torch.stack([u.key_read.weight for u in units]),
                clock_bias=torch.stack([u.clock_bias for u in units]),
                input=torch.stack([u.input.weight for u in units]),
                output=torch.stack([u.output.weight for u in units]),
                gate_w=torch.stack([u.gate.weight for u in units]),
                gate_b=torch.stack([u.gate.bias for u in units]),
                control_w=torch.stack([u.control.weight for u in units]),
                control_b=torch.stack([u.control.bias for u in units]),
                rate=F.softplus(torch.stack([u.raw_rate for u in units])) + 1e-6,
                frequency=torch.stack([u.frequency for u in units]).to(torch.float64),
                query=torch.stack([self.queries[depth][h].weight for h in range(self.heads)]),
                gain=units[0].gain))
        return layers

    def forward_chunk(self, tokens, state=None):
        if not self.training:
            return super().forward_chunk(tokens, state)
        self._fast_layers = {}
        try:
            return super().forward_chunk(tokens, state)
        finally:
            self._fast_layers = None

    def consume_event(self, source, timestamp, content, state):
        if not self.training:
            return super().consume_event(source, timestamp, content, state)
        source, timestamp = int(source), float(timestamp)
        if not 0 <= source < self.sources or not math.isfinite(timestamp) or timestamp < state.last_input_time:
            raise ValueError('Known observed address and finite nondecreasing timestamps required')
        if source not in self._fast_layers:
            self._fast_layers[source] = self._stacked(source)
        layers = self._fast_layers[source]
        mark = torch.as_tensor(content, dtype=self.embedding.weight.dtype, device=self.embedding.weight.device)
        if mark.shape != (self.content_dim,) or not bool(torch.isfinite(mark).all()):
            raise ValueError('Finite fixed-width observed content required')
        x = self.embedding.weight[source] + self.content(mark)
        arrival = torch.tensor(timestamp, dtype=torch.float64, device=x.device)
        if source in state.contexts:
            previous, times = state.contexts[source]
            ready = times.max()
            read_time = torch.maximum(arrival, ready)
            state.queue_wait_sum += float((read_time - arrival).detach())
            arrival = read_time
            context = self.align(list(previous.split(self.payload)), times, arrival, self.depth - 1)
            x = F.layer_norm(x + torch.sigmoid(self.source_gate(x)) * context, (self.total_payload,))
        H, U, P = self.heads, self.pool, self.payload
        for depth in range(self.depth):
            L = layers[depth]
            mixed = self.channel_mix[depth](x)
            all_features = F.layer_norm(mixed, (self.total_payload,))
            incoming = mixed.view(H, P)                                            # (H, P)
            query = torch.einsum('hpd,d->hp', L['query'], all_features)            # (H, P)
            keys = [(depth, h, source, i) for h in range(H) for i in range(U)]
            zero = incoming.new_zeros(P)
            memory = torch.stack([state.memories.get(k, zero) for k in keys])      # (H*U, P)
            seen = [k in state.arrivals for k in keys]
            prev = torch.stack([state.arrivals[k] if s else arrival for k, s in zip(keys, seen)])
            x_u = incoming.repeat_interleave(U, 0)                                 # (H*U, P)
            q_u = query.repeat_interleave(U, 0)
            read = L['key'] + torch.einsum('npq,nq->np', L['key_read'], memory)
            scores = ((q_u * read).sum(-1) / math.sqrt(P) + L['clock_bias']).view(H, U).clamp(-12, 12)
            controls = torch.einsum('ncp,np->nc', L['control_w'], F.layer_norm(x_u, (P,))) + L['control_b']
            forget = F.softplus(controls[:, 0]) / math.log(2)
            write = 2 * torch.sigmoid(controls[:, 1])
            age = (arrival - prev).clamp_min(0)                                    # 0 for units never written
            decay = torch.exp(-age.to(x.dtype)[:, None] * L['rate'] * forget[:, None]).repeat_interleave(2, -1)
            memory = precise_rotate(memory * decay, age[:, None] * L['frequency'])
            memory = memory + write[:, None] * torch.einsum('npq,nq->np', L['input'], x_u)
            y = F.layer_norm(torch.einsum('npq,nq->np', L['output'], memory) + x_u, (P,))
            gate = torch.einsum('npq,nq->np', L['gate_w'], F.gelu(y)) + L['gate_b']
            proposals = (x_u + L['gain'] * y * torch.sigmoid(gate)).view(H, U, P)
            memory = memory.view(H, U, P)
            values, arrivals = [], []
            for head in range(H):
                value, delay, winner = self.race(scores[head], proposals[head])
                index = int(winner)
                state.counterfactual_values += self.pool
                address = (depth, head, source, index)
                state.memories[address] = memory[head, index]; state.arrivals[address] = arrival
                state.visited_units.add(address)
                state.candidate_scores += self.pool; state.selected_updates += 1
                values.append(value); arrivals.append(arrival + delay)
            arrival = torch.stack(arrivals).max()
            x = self.align(values, arrivals, arrival, depth)
        state.contexts[source] = (torch.cat(values), torch.stack(arrivals))
        state.events += 1; state.last_input_time = timestamp
        return self.head(x), arrival


class FastNativeStreamLanguageModel(FastNativeCoreMixin, NativeStreamLanguageModel):
    pass


def fast_class(cls):
    """the same model class with the batched training path; identical parameters and checkpoints."""
    return type('Fast' + cls.__name__, (FastNativeCoreMixin, cls), {})
