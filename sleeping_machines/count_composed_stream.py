"""Generic count-receiver composition over any stream model (labelled diagnostics; THEORY §§386-387).

The integrated native composition is sleeping_machines.count_carrying_language.CountCarryingNativeModel; this
wrapper reuses its escape-race cascade (compose) for other bases such as the temporal carrier.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .count_carrying_language import A, CountCarryingNativeModel, compose


class PositionedState:
    """Wraps any stream state with its stream position; attribute access falls through to the inner state."""

    def __init__(self, inner, position=0):
        self.inner, self.position = inner, position

    def detach(self):
        self.inner = self.inner.detach()
        return self

    def __getattr__(self, name):
        if name in ('inner', 'position'):
            raise AttributeError(name)
        return getattr(self.inner, name)


class CountComposedModel(nn.Module):
    """Count receivers of orders 1..K composed (escape-race cascade) over any stream model's predictive.

    Used for labelled diagnostics with non-native bases (e.g. the temporal carrier); the integrated
    native composition is CountCarryingNativeModel.
    """

    def __init__(self, base, orders):
        super().__init__()
        if orders < 1:
            raise ValueError('at least one count order required')
        self.base, self.orders = base, orders
        self.raw_discount = nn.Parameter(torch.full((orders,), float(np.log(.75 / .25))))
        self.raw_theta = nn.Parameter(torch.full((orders,), float(np.log(np.expm1(1.0)))))
        self.streams, self.active = {}, None

    use_stream = CountCarryingNativeModel.use_stream
    escape_parameters = CountCarryingNativeModel.escape_parameters

    def register_stream(self, name, counts):
        counts = torch.as_tensor(counts, dtype=next(self.base.parameters()).dtype)
        if counts.shape[0] != self.orders or counts.shape[2] != A:
            raise ValueError('counts must be (orders, positions, 27)')
        self.streams[name] = counts

    def new_state(self):
        return PositionedState(self.base.new_state(), 0)

    def forward_chunk(self, tokens, state=None):
        state = self.new_state() if state is None else state
        logits, inner = self.base.forward_chunk(tokens, state.inner)
        start = state.position
        counts = self.streams[self.active][:, start:start + logits.shape[0]]
        if counts.shape[1] != logits.shape[0]:
            raise IndexError('stream position beyond registered counts')
        D, th = self.escape_parameters()
        return compose(F.log_softmax(logits, -1), counts, D, th), PositionedState(inner, start + logits.shape[0])
