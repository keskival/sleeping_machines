"""Parallel content-bearing race heads with timestamped separate channels.

Each head has sparse receivers and its own episodic keys/values. Heads race
independently and retain distinct output channels. Timestamped messages evolve
while waiting; the next block reads them at their latest arrival and applies
a learned channel projection. No exact arrival coincidence is required.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .parallel_stream_language import precise_rotate
from .sparse_race_language import TemporalRoute, TemporalUnit
from .packed_episodic_race_language import PackedEpisodicState, PackedKVBank, PackedBucketMap
from .indexed_episodic_race_language import IndexedEpisodicRaceLanguageModel


@dataclass
class ParallelHeadState(PackedEpisodicState):
    context_arrivals: torch.Tensor | None = None
    kv_head_groups: int = 0
    kv_distinct_positions: int = 0

    def detach(self):
        super().detach()
        if self.context_arrivals is not None:
            self.context_arrivals = self.context_arrivals.detach()
        return self


class ParallelHeadRaceLanguageModel(nn.Module):
    # Same known content index and candidate ordering; each head owns its bank.
    bucket = IndexedEpisodicRaceLanguageModel.bucket
    candidates = IndexedEpisodicRaceLanguageModel.candidates

    def __init__(self, payload=32, depth=8, pool=2, vocabulary=27,
                 matching=8, recent=4, heads=2, block_size=256):
        super().__init__()
        if (min(payload, depth, pool, heads, block_size) < 1 or payload % 2 or
                min(matching, recent) < 0 or matching + recent < 1 or depth * .022 >= .5):
            raise ValueError('Positive dimensions, even head width and causal delay bound required')
        self.payload, self.depth, self.pool, self.heads = payload, depth, pool, heads
        self.total_payload = payload * heads
        self.matching, self.recent, self.block_size = matching, recent, block_size
        self.embedding = nn.Embedding(vocabulary, self.total_payload)
        self.source_gate = nn.Linear(self.total_payload, self.total_payload)
        self.channel_mix = nn.ModuleList([nn.Linear(self.total_payload, self.total_payload,
                                                   bias=False) for _ in range(depth)])
        for mix in self.channel_mix:
            nn.init.eye_(mix.weight)
        self.queries = nn.ModuleList([nn.ModuleList([
            nn.Linear(self.total_payload, payload, bias=False) for _ in range(heads)])
            for _ in range(depth)])
        self.units = nn.ModuleList([nn.ModuleList([nn.ModuleList([
            nn.ModuleList([TemporalUnit(payload, .5/math.sqrt(depth)) for _ in range(pool)])
            for _ in range(vocabulary)]) for _ in range(heads)]) for _ in range(depth)])
        channels = depth * heads
        self.kv_query = nn.ModuleList([nn.Linear(payload, payload, bias=False) for _ in range(channels)])
        self.kv_key = nn.ModuleList([nn.Linear(payload, payload, bias=False) for _ in range(channels)])
        self.kv_value = nn.ModuleList([nn.Linear(payload, payload, bias=False) for _ in range(channels)])
        self.kv_gate = nn.ModuleList([nn.Linear(payload, payload) for _ in range(channels)])
        for gate in self.kv_gate:
            nn.init.zeros_(gate.weight); nn.init.constant_(gate.bias, -2.)
        self.kv_recency = nn.Parameter(torch.full((channels,), -4.))
        tau = torch.logspace(0, 2, payload//2)
        self.transport_rate = nn.Parameter(torch.expm1(1/tau).log().repeat(depth, heads, 1))
        self.transport_frequency = nn.Parameter(torch.linspace(-math.pi/2, math.pi/2,
            payload//2).repeat(depth, heads, 1))
        self.head = nn.Linear(self.total_payload, vocabulary)
        self.register_buffer('index_planes', torch.randn(channels, 3, payload))

    def new_state(self):
        channels = self.depth * self.heads
        return ParallelHeadState(banks=[PackedKVBank(self.block_size) for _ in range(channels)],
            buckets=[{} for _ in range(channels)],
            semantic_buckets=[PackedBucketMap() for _ in range(channels)])

    def transport(self, value, age, depth, head):
        """Closed-form evolution of a stored channel to a later read time."""
        age = age.clamp_min(0)
        rate = F.softplus(self.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype) * rate).repeat_interleave(2)
        return precise_rotate(decayed, age * self.transport_frequency[depth, head].to(torch.float64))

    def align(self, values, arrivals, read_time, depth):
        return torch.cat([self.transport(value, read_time-arrival, depth, head)
                          for head, (value, arrival) in enumerate(zip(values, arrivals))])

    @staticmethod
    def race(scores, values=None):
        if values is not None:
            return TemporalRoute.apply(scores, values)
        rates = scores.to(torch.float64).exp()
        time, winner = (torch.empty_like(rates).exponential_()/rates).min(0)
        return None, .001 + .010*time/(1+time), winner

    def consume(self, token, state):
        token = int(token)
        x = self.embedding.weight[token]
        arrival = torch.tensor(float(state.position), dtype=torch.float64, device=x.device)
        if state.context is not None:
            previous = list(state.context.split(self.payload))
            context = self.align(previous, state.context_arrivals, arrival, self.depth-1)
            x = F.layer_norm(x + torch.sigmoid(self.source_gate(x))*context, (self.total_payload,))
        for depth in range(self.depth):
            # Keep H separate channels; a square learned map permits cross-head
            # interactions while preserving H*d dimensions for the next races.
            mixed = self.channel_mix[depth](x)
            all_features = F.layer_norm(mixed, (self.total_payload,))
            values, arrivals, selected_positions = [], [], []
            for head in range(self.heads):
                incoming = mixed[head*self.payload:(head+1)*self.payload]
                units = self.units[depth][head][token]
                query = self.queries[depth][head](all_features)
                scores = torch.stack([(query @ (unit.key + unit.key_read(
                    state.memories.get((depth, head, token, i), incoming.new_zeros(self.payload))))) /
                    math.sqrt(self.payload) + unit.clock_bias for i,unit in enumerate(units)]).clamp(-12,12)
                if self.training:
                    proposals = [unit.propose(incoming, state.memories.get((depth,head,token,i),
                        incoming.new_zeros(self.payload)), state.arrivals.get((depth,head,token,i)), arrival)
                        for i,unit in enumerate(units)]
                    value, delay, winner = self.race(scores, torch.stack([p[0] for p in proposals]))
                    index = int(winner); memory = proposals[index][1]
                    state.counterfactual_values += self.pool
                else:
                    _, delay, winner = self.race(scores)
                    index = int(winner)
                    value, memory = units[index].propose(incoming,
                        state.memories.get((depth,head,token,index), incoming.new_zeros(self.payload)),
                        state.arrivals.get((depth,head,token,index)), arrival)
                address = (depth, head, token, index)
                state.memories[address] = memory; state.arrivals[address] = arrival
                state.visited_units.add(address); state.candidate_scores += self.pool
                state.selected_updates += 1; state.deliveries += 1
                head_arrival = arrival + delay
                bank_id = depth*self.heads + head
                bank = state.banks[bank_id]
                features = F.layer_norm(value, (self.payload,))
                q = self.kv_query[bank_id](features)
                ids = self.candidates(state, bank_id, token, q)
                if ids:
                    keys = torch.stack([bank[i][0] for i in ids])
                    ages = torch.tensor([state.position-bank[i][2] for i in ids],
                                        dtype=value.dtype, device=value.device)
                    scores = (keys @ q/math.sqrt(self.payload) -
                        F.softplus(self.kv_recency[bank_id])*torch.log1p(ages)).clamp(-12,12)
                    if self.training:
                        retrieved, delay, winner = self.race(scores, torch.stack([bank[i][1] for i in ids]))
                        state.kv_teacher_reads += len(ids)
                    else:
                        _, delay, winner = self.race(scores)
                        retrieved = bank[ids[int(winner)]][1]
                    selected_position = bank[ids[int(winner)]][2]
                    age = state.position-selected_position
                    state.kv_winner_age_sum += age; state.kv_winner_age_max = max(state.kv_winner_age_max, age)
                    # The already received channel evolves during the KV race;
                    # the arriving historical message is then added to it.
                    value = self.transport(value, delay, depth, head) + torch.sigmoid(
                        self.kv_gate[bank_id](features))*retrieved
                    head_arrival = head_arrival + delay
                    state.kv_queries += 1; state.kv_scores += len(ids); state.kv_values += 1
                    selected_positions.append(selected_position)
                cache_features = F.layer_norm(value, (self.payload,))
                bank.append((self.kv_key[bank_id](cache_features),
                             self.kv_value[bank_id](cache_features), state.position))
                bucket = self.bucket(bank[-1][0], bank_id)
                state.semantic_buckets[bank_id].setdefault(bucket, []).append(len(bank)-1)
                values.append(value); arrivals.append(head_arrival)
            if selected_positions:
                state.kv_head_groups += 1
                state.kv_distinct_positions += len(set(selected_positions))
            # This is a read/join of retained state, not simultaneous spikes.
            # Parallel branch bounds depend on max arrival, never their sum.
            arrival = torch.stack(arrivals).max()
            x = self.align(values, arrivals, arrival, depth)
        state.context = torch.cat(values)
        state.context_arrivals = torch.stack(arrivals)
        state.position += 1
        return self.head(x)

    def forward_chunk(self, tokens, state=None):
        state = self.new_state() if state is None else state
        indices = torch.as_tensor(tokens, dtype=torch.long, device=self.embedding.weight.device)
        if indices.ndim != 1 or not len(indices):
            raise ValueError('Nonempty character stream required')
        return torch.stack([self.consume(token,state) for token in indices.tolist()]), state
