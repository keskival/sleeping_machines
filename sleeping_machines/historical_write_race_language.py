"""Parallel sparse race heads with compact historical write eligibility.

Copied consume ordering intentionally preserves the frozen parent primal and
RNG. Only sealed write maps acquire extra fixed-feature local adjoints.
"""
from dataclasses import dataclass, field
import math

import torch
from torch.nn import functional as F

from .parallel_head_race_language import ParallelHeadRaceLanguageModel, ParallelHeadState
from .packed_episodic_race_language import PackedKVBank, PackedBucketMap
from .historical_write_credit import PackedFeatureBank, HistoricalKeyMatch, HistoricalValueCredit


@dataclass
class HistoricalWriteState(ParallelHeadState):
    write_features: list = field(default_factory=list)
    historical_key_queries: int = 0
    historical_key_candidates: int = 0
    historical_value_deliveries: int = 0
    historical_feature_reads: int = 0
    historical_feature_writes: int = 0

    def packed_storage(self):
        stats = super().packed_storage()
        features = [bank.storage() for bank in self.write_features]
        stats.update(eligibility_entries=sum(s['entries'] for s in features),
            allocated_eligibility_bytes=sum(s['allocated_feature_bytes'] for s in features),
            scope=stats['scope']+'; eligibility is one additional detached vector per write')
        return stats


class HistoricalWriteRaceLanguageModel(ParallelHeadRaceLanguageModel):
    def __init__(self, *args, historical_credit=1., **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(historical_credit) or not 0 <= historical_credit <= 1:
            raise ValueError('Historical credit must lie in [0,1]')
        self.historical_credit = historical_credit

    def new_state(self):
        channels = self.depth*self.heads
        return HistoricalWriteState(banks=[PackedKVBank(self.block_size) for _ in range(channels)],
            buckets=[{} for _ in range(channels)],
            semantic_buckets=[PackedBucketMap() for _ in range(channels)],
            write_features=[PackedFeatureBank(self.block_size) for _ in range(channels)])

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
                    old = [j for j, i in enumerate(ids) if i < state.detached_until]
                    eligible = self.training and torch.is_grad_enabled() and self.historical_credit > 0 and bool(old)
                    feature_rows = None
                    if eligible:
                        feature_rows = torch.stack([state.write_features[bank_id][ids[j]] for j in old])
                        old_indices = torch.tensor(old, dtype=torch.long, device=q.device)
                        matches = HistoricalKeyMatch.apply(q, keys, feature_rows,
                            self.kv_key[bank_id].weight, old_indices, self.historical_credit)
                        state.historical_key_queries += 1
                        state.historical_key_candidates += len(old)
                        state.historical_feature_reads += len(old)
                    else:
                        matches = keys @ q
                    scores = (matches/math.sqrt(self.payload) -
                        F.softplus(self.kv_recency[bank_id])*torch.log1p(ages)).clamp(-12,12)
                    if self.training:
                        retrieved, delay, winner = self.race(scores, torch.stack([bank[i][1] for i in ids]))
                        state.kv_teacher_reads += len(ids)
                    else:
                        _, delay, winner = self.race(scores)
                        retrieved = bank[ids[int(winner)]][1]
                    if eligible and int(winner) in old:
                        retrieved = HistoricalValueCredit.apply(retrieved,
                            feature_rows[old.index(int(winner))], self.kv_value[bank_id].weight,
                            self.historical_credit)
                        state.historical_value_deliveries += 1
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
                state.write_features[bank_id].append(cache_features)
                state.historical_feature_writes += 1
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

