"""Factorized terminal memory query, preserving E136's winning event program."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from e136_scattering_shd_model import ScatteringClassifier


class CompactMemoryHead(nn.Module):
    def __init__(self, packet_features, banks, dim, classes, rank):
        super().__init__()
        if rank < 1:
            raise ValueError('Rank must be positive')
        self.packet_features, self.banks, self.dim = packet_features, banks, dim
        self.classes, self.rank = classes, rank
        self.packet = nn.Linear(packet_features, classes)
        nn.init.normal_(self.packet.weight, std=.001)
        nn.init.zeros_(self.packet.bias)
        self.bank = nn.Parameter(torch.randn(rank, banks) / math.sqrt(banks))
        self.channel = nn.Parameter(torch.randn(rank, dim) / math.sqrt(dim))
        self.class_weight = nn.Parameter(torch.randn(classes, rank) * .01)

    def forward(self, z):
        states = z[:, self.packet_features:].reshape(-1, self.banks, self.dim)
        projections = F.linear(states, self.channel)
        latent = (projections * self.bank.T[None]).sum(1)
        return self.packet(z[:, :self.packet_features]) + F.linear(latent, self.class_weight)

    def forward_macs_per_query(self):
        return (self.banks * self.dim * self.rank + self.banks * self.rank
                + self.rank * self.classes + self.packet_features * self.classes)


class CompactScatteringClassifier(ScatteringClassifier):
    def __init__(self, depth=12, rank=16, angles='learned'):
        if angles not in ('learned', 'frozen'):
            raise ValueError('Invalid angle learning mode')
        super().__init__(depth, 'state')
        # Identical initialization in both interventions, independently of
        # whether angle parameters subsequently receive optimizer updates.
        torch.manual_seed(607)
        self.head = CompactMemoryHead(self.dim + 1, depth * self.receivers * self.options,
                                      self.dim, self.key.classes, rank)
        self.angle_weight.requires_grad_(angles == 'learned')
        self.angle_bias.requires_grad_(angles == 'learned')
        self.angles, self.rank = angles, rank
