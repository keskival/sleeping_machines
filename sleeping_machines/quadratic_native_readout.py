"""Zero-nested standard degree-2 readout of the current native vector.

No prefix carrier or retrieval is added. The original affine logits remain and
upper-triangular products expose local interactions to a learned residual.
"""
import math
import torch
from torch import nn


class QuadraticNativeReadout(nn.Module):
    def __init__(self, linear):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError('An original affine native readout is required')
        self.linear = linear
        self.dimension = linear.in_features
        indices = torch.triu_indices(self.dimension, self.dimension, device=linear.weight.device)
        self.register_buffer('first', indices[0])
        self.register_buffer('second', indices[1])
        # Preserve the factory's RNG state and all existing initialized weights.
        with torch.random.fork_rng():
            self.quadratic = nn.Linear(len(self.first), linear.out_features, bias=False,
                device=linear.weight.device, dtype=linear.weight.dtype)
            nn.init.zeros_(self.quadratic.weight)

    def forward(self, value):
        products = value[..., self.first] * value[..., self.second] / math.sqrt(self.dimension)
        return self.linear(value) + self.quadratic(products)
