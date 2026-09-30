"""Token-unit initialization ablations for the existing event-language blocks.

This changes starting poles, not the model's computation, capacity or teacher.
The inherited event initialization is preserved as an explicit comparison arm.
All rates and frequencies remain learned. No test data enter this choice.
"""
import math

import torch


PROFILES = ('inherited', 'long_decay', 'long_spectrum')


@torch.no_grad()
def initialize_language_memory(model, profile):
    if profile not in PROFILES:
        raise ValueError('Unknown language memory profile')
    if profile == 'inherited':
        return
    for layer in model.layers:
        # A token is one time unit. Cover character, word and longer spans;
        # softplus inversion preserves stable positive rates without clipping.
        tau = torch.logspace(0, math.log10(1024), layer.modes,
                             dtype=layer.raw_rate.dtype, device=layer.raw_rate.device)
        layer.raw_rate.copy_(torch.expm1(1/tau).log())
        if profile == 'long_spectrum':
            # Resolved periods 4..2048 tokens; alternating signs retain complex
            # modes without spending most initial frequencies above Nyquist.
            frequency = torch.logspace(math.log10(2*math.pi/2048),
                math.log10(2*math.pi/4), layer.modes,
                dtype=layer.frequency.dtype, device=layer.frequency.device)
            frequency[1::2].neg_()
            layer.frequency.copy_(frequency)
