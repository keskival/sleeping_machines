"""Native addressed content/time events with persistent independent race heads.

No per-position KV bank. Observed source IDs address receiver pools and deep
contexts; an ID is input information, never a label or a discovered oracle.
Heads evolve and mix exactly as in the integrated receiver construction.
"""
from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F

from .parallel_head_race_language import ParallelHeadRaceLanguageModel
from .sparse_race_language import TemporalUnit


@dataclass
class AddressedEventState:
    memories: dict = field(default_factory=dict)
    arrivals: dict = field(default_factory=dict)
    contexts: dict = field(default_factory=dict)
    visited_units: set = field(default_factory=set)
    events: int = 0
    candidate_scores: int = 0
    selected_updates: int = 0
    counterfactual_values: int = 0
    last_input_time: float = -math.inf
    queue_wait_sum: float = 0.

    def detach(self):
        self.memories = {k: v.detach() for k, v in self.memories.items()}
        self.arrivals = {k: v.detach() for k, v in self.arrivals.items()}
        self.contexts = {k: (x.detach(), t.detach()) for k, (x, t) in self.contexts.items()}
        return self

    def storage(self):
        tensors = list(self.memories.values())+list(self.arrivals.values())
        tensors += [t for pair in self.contexts.values() for t in pair]
        return dict(occupied_sources=len(self.contexts), occupied_receivers=len(self.memories),
                    persistent_tensor_bytes=sum(t.numel()*t.element_size() for t in tensors),
                    scope='Addressed live state tensors; Python metadata, parameters and graphs separate')


class AddressedEventHeads(nn.Module):
    transport = ParallelHeadRaceLanguageModel.transport
    align = ParallelHeadRaceLanguageModel.align

    def __init__(self, sources=4, content_dim=2, classes=4, payload=16, depth=8,
                 pool=2, heads=2, credit='counterfactual'):
        super().__init__()
        if min(sources, content_dim, classes, payload, depth, pool, heads) < 1 or payload % 2:
            raise ValueError('Positive dimensions and even per-head payload required')
        self.sources, self.content_dim, self.classes = sources, content_dim, classes
        self.payload, self.depth, self.pool, self.heads = payload, depth, pool, heads
        if credit not in ('counterfactual','pathwise'):
            raise ValueError('Declared counterfactual or diagnostic pathwise credit required')
        self.credit = credit
        self.total_payload = payload*heads
        self.embedding = nn.Embedding(sources, self.total_payload)
        self.content = nn.Linear(content_dim, self.total_payload, bias=False)
        self.source_gate = nn.Linear(self.total_payload, self.total_payload)
        self.channel_mix = nn.ModuleList([nn.Linear(self.total_payload, self.total_payload, bias=False)
                                          for _ in range(depth)])
        for mix in self.channel_mix:
            nn.init.eye_(mix.weight)
        self.queries = nn.ModuleList([nn.ModuleList([
            nn.Linear(self.total_payload, payload, bias=False) for _ in range(heads)])
            for _ in range(depth)])
        self.units = nn.ModuleList([nn.ModuleList([nn.ModuleList([
            nn.ModuleList([TemporalUnit(payload, .5/math.sqrt(depth)) for _ in range(pool)])
            for _ in range(sources)]) for _ in range(heads)]) for _ in range(depth)])
        tau = torch.logspace(0, 2, payload//2)
        self.transport_rate = nn.Parameter(torch.expm1(1/tau).log().repeat(depth, heads, 1))
        self.transport_frequency = nn.Parameter(torch.linspace(-math.pi/2, math.pi/2,
            payload//2).repeat(depth, heads, 1))
        self.head = nn.Linear(self.total_payload, classes)

    def new_state(self):
        return AddressedEventState()

    def race(self, scores, values=None):
        if self.credit=='counterfactual' or values is None:
            return ParallelHeadRaceLanguageModel.race(scores,values)
        # Diagnostic control: identical forward clocks and winner content, but
        # only the interior clock/value derivatives; no losing-route teacher.
        rates = scores.to(torch.float64).exp()
        time,winner = (torch.empty_like(rates).exponential_()/rates).min(0)
        return values[winner],.001+.010*time/(1+time),winner

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
                scores = torch.stack([(query @ (unit.key+unit.key_read(
                    state.memories.get((depth,head,source,i), incoming.new_zeros(self.payload))))) /
                    math.sqrt(self.payload)+unit.clock_bias for i,unit in enumerate(units)]).clamp(-12,12)
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
