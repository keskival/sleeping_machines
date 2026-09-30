"""Input-known write/forget gates on the precise event-language state.

Only the initialization of the new gates is an exact identity. After learning,
this is a different model from the constant-rate/write baseline. Controls depend
on incoming content, preserving an affine recurrence within each causal layer.
Both serial execution and parallel fitting implement that same recurrence.
"""
import heapq
import math

import torch
from torch import nn
from torch.nn import functional as F

from .event_memory import affine_prefix
from .parallel_stream_language import ParallelEventLanguageModel, precise_rotate


class SelectiveEventLanguageModel(ParallelEventLanguageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in self.layers:
            layer.memory_control = nn.Linear(self.width, 2)
            nn.init.zeros_(layer.memory_control.weight)
            nn.init.zeros_(layer.memory_control.bias)

    @staticmethod
    def controls(layer, x):
        features = F.layer_norm(x, (layer.width,))
        logits = layer.memory_control(features)
        # Both controls equal one initially. Stable positive retention rates
        # and bounded write gain avoid an unstable signed decay recurrence.
        forget = F.softplus(logits[..., 0])/math.log(2)
        write = 2*torch.sigmoid(logits[..., 1])
        return forget, write

    def advance(self, state, cutoff, readout=True):
        while state.pending and state.pending[0][0] <= cutoff:
            _, _, node, x, t = heapq.heappop(state.pending)
            if node == self.depth:
                state.output = x
                continue
            layer = self.layers[node]
            forget, write = self.controls(layer, x)
            previous, h = state.last_arrival[node], state.modes[node]
            if previous is not None:
                dt = (t-previous).clamp_min(0).to(x.dtype)
                rate = F.softplus(layer.raw_rate)+1e-6
                h = precise_rotate(h*torch.exp(-dt*rate*forget).repeat_interleave(2),
                    (t-previous).clamp_min(0)*layer.frequency.to(torch.float64))
            h = h+write*layer.input(x)
            state.modes[node], state.last_arrival[node] = h, t
            value, arrival, _ = layer.emit(x[None], h[None], t[None])
            state.deliveries += 1
            self._enqueue(state, node+1, value[0], arrival[0])
        return self.head(state.output) if readout else None

    def forward_chunk(self, tokens, state=None):
        state = self.new_state() if state is None else state
        if state.pending:
            raise ValueError('Selective scan requires an empty boundary queue')
        indices = torch.as_tensor(tokens, dtype=torch.long, device=self.embedding.weight.device)
        if indices.ndim != 1 or not len(indices):
            raise ValueError('A nonempty one-dimensional token chunk is required')
        x = self.embedding(indices)
        times = torch.arange(len(indices), device=x.device, dtype=torch.float64)+state.position
        for node, layer in enumerate(self.layers):
            incoming_times = times
            forget, write = self.controls(layer, x)
            rates = F.softplus(layer.raw_rate)+1e-6
            angles = (times-times[0])[:, None]*layer.frequency.to(torch.float64)[None]
            drive = precise_rotate(write[:, None]*layer.input(x), -angles).reshape(
                len(x), layer.modes, 2)
            elapsed = torch.diff(times, prepend=times[:1]).to(x.dtype)
            decay = torch.exp(-elapsed[:, None]*forget[:, None]*rates[None])
            previous, initial = state.last_arrival[node], state.modes[node]
            if previous is not None:
                gap = (times[0]-previous).clamp_min(0).to(x.dtype)
                initial = precise_rotate(initial*torch.exp(-gap*rates*forget[0]).repeat_interleave(2),
                    (times[0]-previous).clamp_min(0)*layer.frequency.to(torch.float64))
            drive = torch.cat(((drive[0]+initial.reshape(layer.modes, 2))[None], drive[1:]))
            modal, _ = affine_prefix(decay, drive)
            modal = precise_rotate(modal.flatten(1), angles)
            x, times, _ = layer.emit(x, modal, incoming_times)
            state.modes[node], state.last_arrival[node] = modal[-1], incoming_times[-1]
        state.output = x[-1]
        state.position += len(indices)
        state.serial += len(indices)*(self.depth+1)
        state.deliveries += len(indices)*self.depth
        return self.head(x), state
