"""Per-position KV race retrieval inside the sparse temporal backbone.

All historical entries survive until stream reset. A declared character index
plus recent positions bounds candidate discovery; it is not arbitrary semantic
retrieval. Inference reads only the winner's value. Training charges all admitted
counterfactual values. No dense language carrier is introduced.
"""
from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F

from .sparse_race_language import SparseLanguageState, SparseRaceLanguageModel, TemporalRoute


@dataclass
class EpisodicState(SparseLanguageState):
    banks: list = field(default_factory=list)
    buckets: list = field(default_factory=list)
    detached_until: int = 0
    kv_queries: int = 0
    kv_scores: int = 0
    kv_values: int = 0
    kv_teacher_reads: int = 0
    kv_winner_age_sum: int = 0
    kv_winner_age_max: int = 0

    def detach(self):
        # Detach just newly written entries. Rewalking the entire cache every
        # credit chunk would introduce quadratic Python work, not neural work.
        for bank in self.banks:
            for i in range(self.detached_until, len(bank)):
                k, v, position = bank[i]
                bank[i] = (k.detach(), v.detach(), position)
        self.detached_until = len(self.banks[0]) if self.banks else 0
        self.memories = {k: v.detach() for k, v in self.memories.items()}
        self.arrivals = {k: v.detach() for k, v in self.arrivals.items()}
        if self.context is not None:
            self.context = self.context.detach()
        return self


class EpisodicRaceLanguageModel(SparseRaceLanguageModel):
    def __init__(self, payload=32, depth=6, pool=2, vocabulary=27,
                 matching=8, recent=4):
        super().__init__(payload, depth, pool, vocabulary)
        if min(matching, recent) < 0 or matching + recent < 1 or depth * .022 >= .5:
            raise ValueError('Positive candidate budget and doubled-delay bound required')
        self.matching, self.recent = matching, recent
        self.kv_query = nn.ModuleList([nn.Linear(payload, payload, bias=False) for _ in range(depth)])
        self.kv_key = nn.ModuleList([nn.Linear(payload, payload, bias=False) for _ in range(depth)])
        self.kv_value = nn.ModuleList([nn.Linear(payload, payload, bias=False) for _ in range(depth)])
        self.kv_gate = nn.ModuleList([nn.Linear(payload, payload) for _ in range(depth)])
        for gate in self.kv_gate:
            nn.init.zeros_(gate.weight); nn.init.constant_(gate.bias, -2.)
        self.kv_recency = nn.Parameter(torch.full((depth,), -4.))

    def new_state(self):
        return EpisodicState(banks=[[] for _ in range(self.depth)],
                             buckets=[{} for _ in range(self.depth)])

    def candidates(self, state, depth, token):
        bank = state.banks[depth]
        matching = state.buckets[depth].get(token, [])
        # Zero budgets must not become [-0:] (which means the entire bucket).
        ids = matching[-self.matching:] if self.matching else []
        recent = range(max(0, len(bank) - self.recent), len(bank))
        return sorted(set(ids).union(recent))

    def consume(self, token, state):
        token = int(token); x = self.embedding.weight[token]
        if state.context is not None:
            x = F.layer_norm(x + torch.sigmoid(self.source_gate(x)) * state.context,
                             (self.payload,))
        arrival = torch.tensor(float(state.position), dtype=torch.float64, device=x.device)
        for depth, pools in enumerate(self.units):
            units = pools[token]
            query = self.queries[depth](F.layer_norm(x, (self.payload,)))
            scores = torch.stack([(query @ (unit.key + unit.key_read(
                state.memories.get((depth, token, i), x.new_zeros(self.payload))))) /
                math.sqrt(self.payload) + unit.clock_bias for i, unit in enumerate(units)]).clamp(-12, 12)
            if self.training:
                proposals = [unit.propose(x, state.memories.get((depth, token, i), x.new_zeros(self.payload)),
                    state.arrivals.get((depth, token, i)), arrival) for i, unit in enumerate(units)]
                value, delay, winner = TemporalRoute.apply(scores, torch.stack([p[0] for p in proposals]))
                index = int(winner); memory = proposals[index][1]
                state.counterfactual_values += self.pool
            else:
                rates = scores.to(torch.float64).exp()
                time, winner = (torch.empty_like(rates).exponential_() / rates).min(0)
                index = int(winner); delay = .001 + .010 * time / (1 + time)
                value, memory = units[index].propose(x,
                    state.memories.get((depth, token, index), x.new_zeros(self.payload)),
                    state.arrivals.get((depth, token, index)), arrival)
            address = (depth, token, index)
            state.memories[address] = memory; state.arrivals[address] = arrival
            state.visited_units.add(address); state.candidate_scores += self.pool
            state.selected_updates += 1; state.deliveries += 1
            x, arrival = value, arrival + delay

            bank = state.banks[depth]; ids = self.candidates(state, depth, token)
            features = F.layer_norm(x, (self.payload,))
            if ids:
                q = self.kv_query[depth](features)
                keys = torch.stack([bank[i][0] for i in ids])
                # Exact integer ages, including at origins beyond 10M.
                ages = torch.tensor([state.position - bank[i][2] for i in ids],
                                    dtype=x.dtype, device=x.device)
                scores = (keys @ q / math.sqrt(self.payload) -
                          F.softplus(self.kv_recency[depth]) * torch.log1p(ages)).clamp(-12, 12)
                if self.training:
                    values = torch.stack([bank[i][1] for i in ids])
                    retrieved, delay, winner = TemporalRoute.apply(scores, values)
                    state.kv_teacher_reads += len(ids)
                else:
                    rates = scores.to(torch.float64).exp()
                    time, winner = (torch.empty_like(rates).exponential_() / rates).min(0)
                    delay = .001 + .010 * time / (1 + time)
                    # No admitted-value stack or weighted sum at inference.
                    retrieved = bank[ids[int(winner)]][1]
                age = state.position - bank[ids[int(winner)]][2]
                state.kv_winner_age_sum += age; state.kv_winner_age_max = max(state.kv_winner_age_max, age)
                x = x + torch.sigmoid(self.kv_gate[depth](features)) * retrieved
                arrival = arrival + delay
                state.kv_queries += 1; state.kv_scores += len(ids); state.kv_values += 1

            # Cache only the current observed input representation. No target or
            # unknown successor is stored. Keys and values stay separate.
            cache_features = F.layer_norm(x, (self.payload,))
            bank.append((self.kv_key[depth](cache_features),
                         self.kv_value[depth](cache_features), state.position))
            state.buckets[depth].setdefault(token, []).append(len(bank) - 1)
        state.context = x; state.position += 1
        return self.head(x)
