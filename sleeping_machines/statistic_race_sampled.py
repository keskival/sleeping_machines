"""Sampled race writes for statistic-valued race memory (THEORY §392 collapse repair).

The base StatisticRaceNativeModel writes each outcome to the zero-temperature race winner (argmax pi), which
can concentrate writes on a few receivers (smoke: 2 of 256 occupied).  Here the writer is the first of M
exponential clocks with rates pi, the substrate's actual race: receiver a wins with probability pi_a, which
spreads assignments in proportion to the router's uncertainty.  Clock noise comes from a generator seeded by
(race_seed, event index): deterministic, recoverable from the stream position, and independent of the core's
race noise.  Delivery (the exact race expectation) and its credit are unchanged.
"""
import torch
from torch.nn import functional as F

from .count_carrying_language import compose

from .statistic_race_memory import StatisticRaceNativeModel


class SampledStatisticRaceNativeModel(StatisticRaceNativeModel):
    def __init__(self, *args, race_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.race_seed = race_seed

    def race_winner(self, pi, event):
        g = torch.Generator().manual_seed(self.race_seed * 1_000_003 + int(event))
        clocks = torch.empty(pi.shape, dtype=torch.float64).exponential_(generator=g)
        return int((clocks / pi.to(torch.float64).clamp_min(1e-300)).argmin())

    def forward_chunk(self, tokens, state=None):
        state = self.new_state() if state is None else state
        start = state.events
        indices = torch.as_tensor(tokens, dtype=torch.long, device=self.embedding.weight.device)
        if indices.ndim != 1 or not len(indices):
            raise ValueError('Nonempty observed token stream required')
        logits, pis, snapshots = [], [], []
        for token in indices:
            if self.writes and state.pool_previous is not None:  # outcome write for the previous context
                state.pool_counts = state.pool_counts.clone()
                state.pool_counts[state.pool_previous, int(token)] += 1
                state.pool_writes += 1
            mark = F.one_hot(token, num_classes=self.vocabulary).to(self.embedding.weight.dtype)
            z, _ = self.consume_event(0, float(state.events), mark, state)
            pi = self.race_distribution(state.contexts[0][0])
            state.pool_previous = self.race_winner(pi.detach(), state.events)  # sampled race clock
            state.pool_reads += self.addresses
            logits.append(z); pis.append(pi); snapshots.append(state.pool_counts)
        logits = torch.stack(logits)
        counts = self.streams[self.active][:, start:start + logits.shape[0]]
        if counts.shape[1] != logits.shape[0]:
            raise IndexError('stream position beyond registered counts')
        log_q = F.log_softmax(self.count_message(logits, counts) if self.count_message is not None else logits, -1)
        if getattr(self, 'base_only', False):
            return log_q, state
        rows = [self.pooled(log_q[i:i + 1], pis[i][None], snapshots[i]) for i in range(len(pis))]
        log_base = torch.log(torch.cat(rows).clamp_min(1e-30))
        if self.escape_gate is not None:
            D, th = self.escape_gate(log_base, counts, self.raw_discount, self.raw_theta)
        else:
            D, th = self.escape_parameters()
        return compose(log_base, counts, D, th), state
