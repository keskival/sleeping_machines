"""Independent sparse heads with shared-match repeated temporal arrivals.

Prepared experimental extension; active independent-head sources stay unchanged.
Receivers still choose one winner. Historical retrieval admits m Poisson marks
per head and mixes messages evolved to their final read time.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .parallel_head_race_language import ParallelHeadRaceLanguageModel, ParallelHeadState
from .packed_episodic_race_language import PackedKVBank, PackedBucketMap
from .repeated_temporal_race import poisson_clocks, temporal_arrivals


@dataclass
class RepeatedArrivalState(ParallelHeadState):
    kv_emitter_renewals: int = 0
    kv_minimum_comparisons: int = 0


class RepeatedArrivalRaceLanguageModel(ParallelHeadRaceLanguageModel):
    def __init__(self, *args, arrivals_per_query=1, **kwargs):
        if arrivals_per_query not in (1,2,4,8):
            raise ValueError('Declared arrival ladder: 1,2,4,8')
        super().__init__(*args, **kwargs)
        self.arrivals_per_query = arrivals_per_query

    def new_state(self):
        channels = self.depth*self.heads
        return RepeatedArrivalState(banks=[PackedKVBank(self.block_size) for _ in range(channels)],
            buckets=[{} for _ in range(channels)],
            semantic_buckets=[PackedBucketMap() for _ in range(channels)])

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
                    if self.arrivals_per_query == 1:
                        # Bitwise reference nesting, including RNG and teacher.
                        if self.training:
                            retrieved, delay, winner = self.race(scores, torch.stack([bank[i][1] for i in ids]))
                            state.kv_teacher_reads += len(ids)
                        else:
                            _, delay, winner = self.race(scores)
                            retrieved = bank[ids[int(winner)]][1]
                        winners = winner[None]
                        value = self.transport(value, delay, depth, head) + torch.sigmoid(
                            self.kv_gate[bank_id](features))*retrieved
                    else:
                        if self.training:
                            messages, delays, winners = temporal_arrivals(scores,
                                torch.stack([bank[i][1] for i in ids]), self.arrivals_per_query)
                            state.kv_teacher_reads += len(ids)
                        else:
                            _, times, winners = poisson_clocks(scores, self.arrivals_per_query)
                            delays = .001 + .010*times/(1+times)
                            messages = torch.stack([bank[ids[int(w)]][1] for w in winners])
                        delay = delays[-1]
                        # The receiver and each arrival evolve to the last read.
                        # Spatial heads remain independent; temporal arrivals
                        # add no tied or extra Q/K/V projection matrices.
                        accumulated = torch.stack([self.transport(message, delay-t,
                            depth, head) for message,t in zip(messages,delays)]).mean(0)
                        value = self.transport(value, delay, depth, head) + torch.sigmoid(
                            self.kv_gate[bank_id](features))*accumulated
                    for winner in winners:
                        selected_position = bank[ids[int(winner)]][2]
                        age = state.position-selected_position
                        state.kv_winner_age_sum += age
                        state.kv_winner_age_max = max(state.kv_winner_age_max, age)
                        selected_positions.append(selected_position)
                    head_arrival = head_arrival + delay
                    state.kv_queries += 1; state.kv_scores += len(ids)
                    state.kv_values += self.arrivals_per_query
                    state.kv_emitter_renewals += self.arrivals_per_query-1
                    state.kv_minimum_comparisons += self.arrivals_per_query*(len(ids)-1)
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

