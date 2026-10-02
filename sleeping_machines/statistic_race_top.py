"""Top-placed statistic-valued race memory (THEORY §392 placement correction).

In StatisticRaceNativeModel the pooled level sits below the exact suffix orders, so its delivery is multiplied
by every order's escape mass: the same starvation that limits the residual base.  A context routed by the core's
state, which summarizes the whole history, is more specific than an order-K suffix, so here the pooled level
sits on top: each receiver backs off to the full exact cascade,

    p(y) = sum_a pi_a [max(c_ay - D, 0) + (theta + D T_a) p_exact(y)] / (n_a + theta),

and where a receiver has evidence it claims the prediction directly, giving the router first-order credit.
Writes use the sampled race (statistic_race_sampled).  With empty receivers the model equals the gated count
model exactly.
"""
import torch
from torch.nn import functional as F

from .count_carrying_language import compose
from .statistic_race_sampled import SampledStatisticRaceNativeModel


class TopStatisticRaceNativeModel(SampledStatisticRaceNativeModel):
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
        if self.escape_gate is not None:
            D, th = self.escape_gate(log_q, counts, self.raw_discount, self.raw_theta)
        else:
            D, th = self.escape_parameters()
        exact = compose(log_q, counts, D, th)  # exact suffix cascade over the learned base
        rows = [self.pooled(exact[i:i + 1], pis[i][None], snapshots[i]) for i in range(len(pis))]
        return torch.log(torch.cat(rows).clamp_min(1e-30)), state
