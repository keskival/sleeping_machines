"""Positive content keys inside an addressed, decaying event memory.

Research adapter: the event backbone and hard winning schedule are unchanged.
Production retrieval uses a segmented linear-work scan, never an event-pair
matrix. Expanded state and projections must be included in the work ledger.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from sleeping_machines.event_memory import linear_memory


class ContentMemory(nn.Module):
    def __init__(self, dim, pairs=4, strength=.5, row_bound=.5):
        super().__init__()
        if dim < 1 or pairs < 1 or not 0 < strength < 1 or row_bound <= 0:
            raise ValueError("Invalid content feature bounds")
        self.dim, self.pairs = dim, pairs
        self.strength, self.row_bound = strength, row_bound
        self.query = nn.Parameter(torch.zeros(pairs, dim))
        key = torch.randn(pairs, dim)
        key = row_bound * key / key.abs().sum(-1, keepdim=True)
        self.key = nn.Parameter(key)
        self.calls = []

    def bounded(self, weight):
        return weight * self.row_bound / weight.abs().sum(-1, keepdim=True).clamp_min(self.row_bound)

    def features(self, x, weight):
        h = self.strength * torch.tanh(F.linear(x, self.bounded(weight)))
        # Each pair has a constant sum. Zero queries recover a constant kernel
        # with diverse key states and an available query-learning direction.
        return torch.stack((1+h, 1-h), -1).flatten(1) / math.sqrt(2*self.pairs)

    def forward(self, x, times, counts, keys, taus, sequential=False):
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError("Unexpected payload width")
        pk = torch.tanh(F.linear(x, self.bounded(self.key)))
        pq = (self.strength**2/self.pairs)*torch.tanh(F.linear(x, self.bounded(self.query)))
        marked = torch.cat((x, x.new_ones((len(x), 1))), -1)
        # Algebraically compress paired positive features to a constant mean
        # and P signed content moments. The resulting kernel remains positive.
        expanded = torch.cat((x, (pk[:, :, None]*marked[:, None, :]).flatten(1)), -1)
        state, mass, work = linear_memory(expanded, times, counts, keys, taus, sequential)
        moments = state[:, :, self.dim:].reshape(len(x), len(taus), self.pairs, self.dim+1)
        numerator = state[:, :, :self.dim] + torch.einsum('er,ekrd->ekd', pq, moments[:, :, :, :-1])
        denominator = 1 + torch.einsum('er,ekr->ek', pq, moments[:, :, :, -1])
        # The constant term includes C+epsilon, so zero queries recover the
        # exact old regularized mean. No division by a zero signed moment.
        memory = numerator / denominator[:, :, None]
        e, k, r, d = len(x), len(taus), self.pairs+1, self.dim
        self.calls.append({
            'events':e, 'banks':k, 'features':r,
            'receivers':int(torch.unique(keys).numel()),
            'positive_feature_width':2*self.pairs,
            'scan_compositions':work, 'state_width':r*(d+1),
            'scan_scalar_multiply_adds':work*k*r*(d+1),
            'plain_scan_scalar_multiply_adds':work*k*(d+1),
            'projection_multiply_adds':2*e*self.pairs*d,
            'retrieval_multiply_adds':e*k*self.pairs*(d+1),
            'persistent_state_scalars':int(torch.unique(keys).numel())*k*r*(d+1),
            'materialized_state_scalars':e*k*r*(d+1),
            'scope':'Partial forward ledger; nonlinearities, marking, sorting, backward and optimizer additional',
        })
        return memory, mass, work

    def input_lipschitz_bound(self, payload_bound):
        """Max-norm bound with fixed times/receivers and bounded payloads."""
        gamma = self.strength**2*self.row_bound/(1-self.strength**2)
        return 1 + 4*payload_bound*gamma

    def pop_calls(self):
        calls, self.calls = self.calls, []
        return calls


def install(model, **kwargs):
    """Replace value-stream memories only; immutable key program stays intact."""
    for layer in model.layers:
        layer.memory = ContentMemory(layer.dim, **kwargs)
    return model
