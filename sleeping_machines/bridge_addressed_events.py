"""Bounded-score native reference; ordinary sparse inference and actual state.
The base consume_event is retained with only its bounded score map replaced.
"""
import math
import torch
from torch.nn import functional as F
from .addressed_event_heads import AddressedEventHeads
from .fast_native_core import fast_class
from .bounded_score_sensitivity import bounded_score_bridge
from .factorized_race import factorized_race


class BridgeAddressedEvents(fast_class(AddressedEventHeads)):
    def __init__(self, *args, score_bridge=.1, **kwargs):
        if not 0. <= score_bridge <= 1.:
            raise ValueError('Bounded bridge strength required')
        super().__init__(*args, **kwargs)
        self.score_bridge = float(score_bridge)

    def race(self, scores, values=None):
        return factorized_race(scores, values)

    def consume_event(self, source, timestamp, content, state):
        source, timestamp = int(source), float(timestamp)
        if not 0 <= source < self.sources or not math.isfinite(timestamp) or timestamp < state.last_input_time:
            raise ValueError('Known observed address and finite nondecreasing timestamps required')
        mark = torch.as_tensor(content, dtype=self.embedding.weight.dtype,
                               device=self.embedding.weight.device)
        if mark.shape != (self.content_dim,) or not bool(torch.isfinite(mark).all()):
            raise ValueError('Finite fixed-width observed content required')
        x = self.embedding.weight[source]+self.content(mark)
        arrival = torch.tensor(timestamp, dtype=torch.float64, device=x.device)
        if source in state.contexts:
            previous, times = state.contexts[source]
            # Admission is local to this source. Other sources keep independent
            # progress; no global periodic clock or global next-event barrier.
            ready = times.max()
            read_time = torch.maximum(arrival, ready)
            state.queue_wait_sum += float((read_time-arrival).detach())
            arrival = read_time
            context = self.align(list(previous.split(self.payload)), times, arrival, self.depth-1)
            x = F.layer_norm(x+torch.sigmoid(self.source_gate(x))*context, (self.total_payload,))
        for depth in range(self.depth):
            mixed = self.channel_mix[depth](x)
            all_features = F.layer_norm(mixed, (self.total_payload,))
            values, arrivals = [], []
            for head in range(self.heads):
                incoming = mixed[head*self.payload:(head+1)*self.payload]
                units = self.units[depth][head][source]
                query = self.queries[depth][head](all_features)
                raw_scores = torch.stack([(query @ (unit.key+unit.key_read(
                    state.memories.get((depth,head,source,i), incoming.new_zeros(self.payload))))) /
                    math.sqrt(self.payload)+unit.clock_bias for i,unit in enumerate(units)])
                scores = bounded_score_bridge(raw_scores, self.score_bridge)
                if self.training:
                    proposals = [unit.propose(incoming,state.memories.get((depth,head,source,i),
                        incoming.new_zeros(self.payload)),state.arrivals.get((depth,head,source,i)),arrival)
                        for i,unit in enumerate(units)]
                    value,delay,winner = self.race(scores,torch.stack([p[0] for p in proposals]))
                    index = int(winner); memory = proposals[index][1]
                    state.counterfactual_values += self.pool
                else:
                    _,delay,winner = self.race(scores)
                    index = int(winner)
                    value,memory = units[index].propose(incoming,state.memories.get((depth,head,source,index),
                        incoming.new_zeros(self.payload)),state.arrivals.get((depth,head,source,index)),arrival)
                address = (depth,head,source,index)
                state.memories[address] = memory; state.arrivals[address] = arrival
                state.visited_units.add(address)
                state.candidate_scores += self.pool; state.selected_updates += 1
                values.append(value); arrivals.append(arrival+delay)
            arrival = torch.stack(arrivals).max()
            x = self.align(values,arrivals,arrival,depth)
        state.contexts[source] = (torch.cat(values),torch.stack(arrivals))
        state.events += 1; state.last_input_time = timestamp
        return self.head(x), arrival
