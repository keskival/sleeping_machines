"""Experimental protected/temporal state with private or shared addressed rules.

The frozen AddressedEventHeads transition is inherited. Sharing rules never
shares addressed state; identity silent modes do not remove the temporal suffix.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .addressed_event_heads import AddressedEventHeads
from .parallel_stream_language import precise_rotate
from .sparse_race_language import TemporalUnit


def split_evolve(value, age, raw_rate, frequency, protected_pairs, forget=1.):
    """Identity prefix plus ordinary decaying/rotating pairs; no idle updates."""
    width = 2 * protected_pairs
    age = age.clamp_min(0)
    dynamic = value[width:]
    rate = F.softplus(raw_rate) + 1e-6
    dynamic = dynamic * torch.exp(-age.to(value.dtype) * rate * forget).repeat_interleave(2)
    dynamic = precise_rotate(dynamic, age * frequency.to(torch.float64))
    return torch.cat((value[:width], dynamic))


class SplitTemporalUnit(TemporalUnit):
    def __init__(self, payload, gain, protected_pairs=0):
        super().__init__(payload, gain)
        self.protected_pairs = protected_pairs
        if not 0 <= protected_pairs < payload // 2:
            raise ValueError('Retain at least one temporal pair')
        if protected_pairs:
            self.raw_rate = nn.Parameter(self.raw_rate[protected_pairs:].detach().clone())
            self.frequency = nn.Parameter(self.frequency[protected_pairs:].detach().clone())

    def propose(self, x, memory, previous, arrival):
        if not self.protected_pairs:
            return super().propose(x, memory, previous, arrival)
        controls = self.control(F.layer_norm(x, (self.payload,)))
        forget = F.softplus(controls[0]) / math.log(2)
        write = 2 * torch.sigmoid(controls[1])
        if previous is not None:
            memory = split_evolve(memory, arrival-previous, self.raw_rate,
                                  self.frequency, self.protected_pairs, forget)
        memory = memory + write * self.input(x)
        y = F.layer_norm(self.output(memory) + x, (self.payload,))
        value = x + self.gain * y * torch.sigmoid(self.gate(F.gelu(y)))
        return value, memory


class SharedSourcePools(nn.Module):
    """One registered pool; source indexing selects rules, not private state."""
    def __init__(self, pool, sources):
        super().__init__()
        self.pool, self.sources = pool, sources

    def __getitem__(self, source):
        if not 0 <= source < self.sources:
            raise IndexError(source)
        return self.pool


class CommonSourceSeed(nn.Module):
    def __init__(self, sources, width):
        super().__init__()
        self.sources = sources
        self.seed = nn.Parameter(torch.randn(1, width))

    @property
    def weight(self):
        return self.seed.expand(self.sources, -1)


class SplitEventHeads(AddressedEventHeads):
    def __init__(self, sources=4, content_dim=2, classes=4, payload=8, depth=8,
                 pool=2, heads=2, credit='counterfactual', shared_maps=False,
                 protected_pairs=0):
        # Constructor ordering deliberately nests the private zero-split parent.
        nn.Module.__init__(self)
        if min(sources, content_dim, classes, payload, depth, pool, heads) < 1 or payload % 2:
            raise ValueError('Positive dimensions and even payload required')
        if not 0 <= protected_pairs < payload // 2:
            raise ValueError('Retain at least one temporal pair')
        if credit not in ('counterfactual', 'pathwise'):
            raise ValueError('Declared credit required')
        self.sources, self.content_dim, self.classes = sources, content_dim, classes
        self.payload, self.depth, self.pool, self.heads = payload, depth, pool, heads
        self.credit, self.shared_maps, self.protected_pairs = credit, shared_maps, protected_pairs
        self.total_payload = payload * heads
        self.embedding = (CommonSourceSeed(sources, self.total_payload) if shared_maps else
                          nn.Embedding(sources, self.total_payload))
        self.content = nn.Linear(content_dim, self.total_payload, bias=False)
        self.source_gate = nn.Linear(self.total_payload, self.total_payload)
        self.channel_mix = nn.ModuleList([nn.Linear(self.total_payload, self.total_payload, bias=False)
                                          for _ in range(depth)])
        for mix in self.channel_mix:
            nn.init.eye_(mix.weight)
        self.queries = nn.ModuleList([nn.ModuleList([
            nn.Linear(self.total_payload, payload, bias=False) for _ in range(heads)])
            for _ in range(depth)])
        def make_pool():
            return nn.ModuleList([SplitTemporalUnit(payload, .5/math.sqrt(depth), protected_pairs)
                                  for _ in range(pool)])
        self.units = nn.ModuleList([nn.ModuleList([
            SharedSourcePools(make_pool(), sources) if shared_maps else
            nn.ModuleList([make_pool() for _ in range(sources)])
            for _ in range(heads)]) for _ in range(depth)])
        tau = torch.logspace(0, 2, payload // 2)
        raw = torch.expm1(1/tau).log().repeat(depth, heads, 1)
        frequency = torch.linspace(-math.pi/2, math.pi/2, payload//2).repeat(depth, heads, 1)
        self.transport_rate = nn.Parameter(raw[..., protected_pairs:].clone())
        self.transport_frequency = nn.Parameter(frequency[..., protected_pairs:].clone())
        self.head = nn.Linear(self.total_payload, classes)

    def transport(self, value, age, depth, head):
        if not self.protected_pairs:
            return super().transport(value, age, depth, head)
        return split_evolve(value, age, self.transport_rate[depth, head],
                            self.transport_frequency[depth, head], self.protected_pairs)
