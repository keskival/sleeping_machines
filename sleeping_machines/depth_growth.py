"""Grow an event encoder by identity residual blocks with live output teachers.

Unlike zeroing every map in a new serial branch, only its output and direct
maps start at zero. Random input/state maps supply nonzero sufficient signals
for output-map credit. Existing blocks, head, sources and gains are preserved.
"""
import copy
import math
import torch
from .event_state import EventStateBlock


def grow_event_encoder(encoder, depth):
    if depth <= len(encoder.layers):
        raise ValueError("Growth requires strictly greater depth")
    grown=copy.deepcopy(encoder)
    first=encoder.layers[0]
    for _ in range(depth-len(encoder.layers)):
        layer=EventStateBlock(first.width,first.modes,first.options,gain=first.gain)
        layer.to(device=first.direct.device,dtype=first.direct.dtype)
        with torch.no_grad():
            layer.output.weight.zero_();layer.direct.zero_();layer.norm.bias.zero_()
            # Cancel the otherwise large 1/sqrt(eps) LayerNorm derivative at zero.
            layer.norm.weight.fill_(math.sqrt(layer.norm.eps))
            layer.gate.bias.zero_()
        grown.layers.append(layer)
    grown.train(encoder.training)
    return grown
