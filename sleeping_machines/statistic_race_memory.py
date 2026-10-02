"""Statistic-valued race memory over learned keys (THEORY §§382-383, §392).

The native core's top-level state h_t forms a query u_t = Q h_t.  M receivers have learned keys k_a and values
that are sufficient statistics (next-symbol counts c_a, total n_a), not learned vectors.  Race rates are
exp(u_t . k_a / tau); pi_t = softmax over the M candidates.  The pooled level of the escape cascade is the exact
race expectation

    p_pool(y) = sum_a pi_a [max(c_ay - D, 0) + (theta + D T_a) q_y] / (n_a + theta)      (q = learned base),

with p_a = q for empty receivers, so it costs M lookups and needs no sampled teacher: the counterfactual
delivery credit to every candidate route is exact at fixed counts (§383.1).  The exact context-suffix orders
(count_carrying_language) then cascade on top of p_pool, with the per-position escape gate.  The zero-temperature race
winner (argmax pi; no extra RNG draws, so the core's race noise and recovery are unchanged) receives the outcome: when token x_{t+1} arrives, c_{w_t, x_{t+1}} += 1.

The pooled level is the generalization path: contexts that never matched as exact strings share evidence when
the core maps them to the same key, and the core receives credit wherever the pooled level carries
responsibility (low-evidence contexts), instead of only through the residual base (median responsibility
.0045, theory 62).  With every receiver empty the model equals GatedCountCarryingNativeModel exactly.

Counts are causal state: they accumulate within a fitting pass and start empty at each pass.  For development
scoring they start from the counts of the latest completed fitting pass (fit data only), kept in the
`pool_seed` buffer and saved with the selected weights.  Writes are not differentiated (write credit, §383.2,
is not used here).
"""
import torch
from torch import nn
from torch.nn import functional as F

from .count_carrying_language import compose
from .count_escape_gate import GatedCountCarryingNativeModel
from .native_stream_language import NativeLanguageState

A = 27


class StatisticRaceState(NativeLanguageState):
    def __init__(self, *args, pool_counts=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_counts = pool_counts
        self.pool_previous = None
        self.pool_reads = self.pool_writes = 0

    def detach(self):
        super().detach()
        return self

    def storage(self):
        stats = super().storage()
        extra = self.pool_counts.numel() * self.pool_counts.element_size() if self.pool_counts is not None else 0
        stats.update(pool_count_bytes=extra, persistent_tensor_bytes=stats['persistent_tensor_bytes'] + extra)
        return stats

    def packed_storage(self):
        return super().packed_storage()


class StatisticRaceNativeModel(GatedCountCarryingNativeModel):
    def __init__(self, payload=16, depth=8, pool=2, heads=2, orders=4, vocabulary=27,
                 addresses=256, key_dim=16, escape_gate=True, count_message=False, writes=True):
        super().__init__(payload, depth, pool, heads, orders=orders, vocabulary=vocabulary,
                         escape_gate=escape_gate, count_message=count_message)
        self.addresses, self.writes = addresses, writes
        self.pool_query = nn.Linear(self.total_payload, key_dim, bias=False)
        self.pool_keys = nn.Parameter(torch.randn(addresses, key_dim) / key_dim ** .5)
        self.pool_log_temperature = nn.Parameter(torch.zeros(()))
        self.raw_pool_discount = nn.Parameter(torch.tensor(1.0986))  # sigmoid -> .75
        self.raw_pool_theta = nn.Parameter(torch.tensor(.5413))      # softplus -> 1
        self.register_buffer('pool_seed', torch.zeros(addresses, A))
        self.seed_pool = False

    def new_state(self):
        counts = self.pool_seed.clone() if self.seed_pool else torch.zeros_like(self.pool_seed)
        return StatisticRaceState(pool_counts=counts)

    def race_distribution(self, h):
        scores = self.pool_keys @ self.pool_query(h) / self.pool_keys.shape[1] ** .5
        return torch.softmax(scores * torch.exp(-self.pool_log_temperature), -1)

    def pooled(self, log_q, pi, counts):
        """exact race expectation of the per-receiver escape predictive; rows of log_q/pi are positions."""
        q = log_q.exp()                                   # (L, A)
        n = counts.sum(-1, keepdim=True)                  # (M, 1)
        T = (counts > 0).to(q.dtype).sum(-1, keepdim=True)
        D, th = torch.sigmoid(self.raw_pool_discount), F.softplus(self.raw_pool_theta)
        own = torch.clamp(counts - D, min=0.) / (n + th)  # (M, A)
        esc = (th + D * T) / (n + th)                     # (M, 1)
        seen, empty = (n > 0).to(q.dtype), (n == 0).to(q.dtype)
        own, esc = own * seen, esc * seen + empty         # empty receivers deliver q exactly
        return pi @ own + (pi @ esc) * q                  # (L, A)

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
            state.pool_previous = int(pi.detach().argmax())  # zero-temperature race winner gets the next outcome
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
