"""Count-conditioned base and state-dependent escape for count receivers (THEORY §379 caveat, §387.3, §389).

Under the scalar-escape cascade a learned base never sees the count state whose residual (§387.1) it must
supply, so it learns only the average residual: composed carrier w32 and w128 both reach 2.313 bpc at 131K, and
their bases alone score 8.17 / 11.34 bpc (§389).  Two zero-initialized, exactly nested repairs:

* CountMessage: the count receivers deliver their statistics into the base logits (information path, §389.1);
* EscapeGate: per-position D_k, theta_k from base-predictive and evidence features (override path, §387.3).

The cascade, causal count streams, persistent addressed state and responsibility-gated credit are unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .count_carrying_language import CountCarryingNativeModel, compose
from .count_composed_stream import CountComposedModel, PositionedState

FEATURES = 5


def gate_features(log_q, c):
    """(L, FEATURES) for one order: log_q (L, A) base log-probs, c (L, A) counts."""
    n = c.sum(-1, keepdim=True)
    seen = c > 0
    q = log_q.exp()
    mass = torch.log((q * seen).sum(-1, keepdim=True).clamp_min(1e-12))
    entropy = -(q * log_q).sum(-1, keepdim=True)
    agreement = (c * log_q).sum(-1, keepdim=True) / n.clamp_min(1.)
    return torch.cat([torch.log1p(n), torch.log1p(seen.to(c.dtype).sum(-1, keepdim=True)),
                      mass * (n > 0), entropy, agreement], -1)


class CountMessage(nn.Module):
    """§389: count receivers deliver their statistics into the base logits, z' = z + sum_k log1p(c_k) W_k + 1[c_k>0] V_k.

    Without it the base cannot see the count state whose residual it must supply.  Zero init nests exactly."""

    def __init__(self, orders, symbols=27):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(orders, 2, symbols, symbols))

    def forward(self, logits, counts):
        out = logits
        for k in range(counts.shape[0]):
            c = counts[k]
            out = out + torch.log1p(c) @ self.weight[k, 0] + (c > 0).to(c.dtype) @ self.weight[k, 1]
        return out


class EscapeGate(nn.Module):
    def __init__(self, orders):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(orders, FEATURES, 2))

    def forward(self, log_q, counts, raw_discount, raw_theta):
        """per-position D, theta of shape (K, L, 1)."""
        shift = torch.stack([gate_features(log_q, counts[k]) @ self.weight[k] for k in range(counts.shape[0])])
        D = torch.sigmoid(raw_discount[:, None, None] + shift[..., :1])
        th = F.softplus(raw_theta[:, None, None] + shift[..., 1:])
        return D, th


def _compose(model, logits, counts):
    if getattr(model, 'base_only', False):  # diagnostic (P389b): the learned base's standalone predictive
        return F.log_softmax(logits, -1)
    if model.count_message is not None:
        logits = model.count_message(logits, counts)
    log_q = F.log_softmax(logits, -1)
    if model.escape_gate is not None:
        D, th = model.escape_gate(log_q, counts, model.raw_discount, model.raw_theta)
    else:
        D, th = model.escape_parameters()
    return compose(log_q, counts, D, th)


class GatedCountCarryingNativeModel(CountCarryingNativeModel):
    def __init__(self, *args, escape_gate=True, count_message=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.escape_gate = EscapeGate(self.orders) if escape_gate else None
        self.count_message = CountMessage(self.orders) if count_message else None

    def forward_chunk(self, tokens, state=None):
        state = self.new_state() if state is None else state
        start = state.events
        logits, state = super(CountCarryingNativeModel, self).forward_chunk(tokens, state)
        counts = self.streams[self.active][:, start:start + logits.shape[0]]
        if counts.shape[1] != logits.shape[0]:
            raise IndexError('stream position beyond registered counts')
        return _compose(self, logits, counts), state


class GatedCountComposedModel(CountComposedModel):
    def __init__(self, base, orders, escape_gate=True, count_message=True):
        super().__init__(base, orders)
        self.escape_gate = EscapeGate(orders) if escape_gate else None
        self.count_message = CountMessage(orders) if count_message else None

    def forward_chunk(self, tokens, state=None):
        state = self.new_state() if state is None else state
        logits, inner = self.base.forward_chunk(tokens, state.inner)
        start = state.position
        counts = self.streams[self.active][:, start:start + logits.shape[0]]
        if counts.shape[1] != logits.shape[0]:
            raise IndexError('stream position beyond registered counts')
        return _compose(self, logits, counts), PositionedState(inner, start + logits.shape[0])
