"""Causal parallel fitting and precise clocks for the persistent language model.

The bounded-delay contract makes every token finish before its query and before
the next source token. Hence each layer's arrivals retain source order. An
input-known affine scan can evaluate that layer for a whole chunk, including
incoming memory and its exact teacher. This is the same learned block family;
it adds no experts, bypass, parameters or future observations.

Keep absolute clocks in float64, then cast elapsed differences to the payload
dtype. Float32 absolute token coordinates otherwise erase sub-token delays at
long positions. Chunk-relative rotation coordinates also avoid large angles.
"""
import heapq

import torch
from torch.nn import functional as F

from .event_memory import affine_prefix
from .rotating_memory import rotate_pairs
from .stream_language import StreamingEventLanguageModel


def precise_rotate(state, angles):
    """Evaluate angles accurately while retaining payload precision and credit."""
    pairs = state.reshape(*state.shape[:-1], -1, 2)
    a, b = pairs.unbind(-1)
    cos, sin = angles.cos().to(state.dtype), angles.sin().to(state.dtype)
    return torch.stack((cos*a-sin*b, sin*a+cos*b), -1).flatten(-2)


class ParallelEventLanguageModel(StreamingEventLanguageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.depth * .011 >= self.query_budget:
            raise ValueError("Parallel language fitting requires all messages before the query")

    def advance(self, state, cutoff, readout=True):
        """Serial reference with precise clocks and float32 elapsed-state arithmetic."""
        while state.pending and state.pending[0][0] <= cutoff:
            _, _, node, x, t = heapq.heappop(state.pending)
            if node == self.depth:
                state.output = x
                continue
            layer = self.layers[node]
            previous, h = state.last_arrival[node], state.modes[node]
            if previous is not None:
                dt = (t-previous).clamp_min(0).to(x.dtype)
                rate = F.softplus(layer.raw_rate)+1e-6
                angle = (t-previous).clamp_min(0)*layer.frequency.to(torch.float64)
                h = precise_rotate(h*torch.exp(-dt*rate).repeat_interleave(2), angle)
            h = h+layer.input(x)
            state.modes[node], state.last_arrival[node] = h, t
            value, arrival, _ = layer.emit(x[None], h[None], t[None])
            state.deliveries += 1
            self._enqueue(state, node+1, value[0], arrival[0])
        return self.head(state.output) if readout else None

    def consume(self, token, state):
        """Reference one-token execution; time and payload precision are distinct."""
        if not 0 <= int(token) < self.vocabulary:
            raise ValueError("Unknown token")
        arrival = state.position
        self.advance(state, arrival, readout=False)
        index = torch.tensor(int(token), device=self.embedding.weight.device)
        clock = torch.tensor(arrival, dtype=torch.float64, device=index.device)
        self._enqueue(state, 0, self.embedding(index), clock)
        prediction = self.advance(state, arrival+self.query_budget)
        state.position += 1
        return prediction

    def forward_chunk(self, tokens, state=None):
        """Evaluate causal affine scans layer by layer; retain persistent state."""
        state = self.new_state() if state is None else state
        if state.pending:
            raise ValueError("Parallel contract requires an empty queue at the token boundary")
        indices = torch.as_tensor(tokens, dtype=torch.long, device=self.embedding.weight.device)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("A nonempty one-dimensional token chunk is required")
        x = self.embedding(indices)
        times = torch.arange(len(indices), device=x.device, dtype=torch.float64)+state.position
        for node, layer in enumerate(self.layers):
            incoming_times = times
            rates = F.softplus(layer.raw_rate)+1e-6
            # Origin-independent coordinates keep phase arguments bounded by
            # the chunk length, rather than the corpus's absolute position.
            relative = times-times[0]
            angles = relative[:, None]*layer.frequency.to(torch.float64)[None]
            drive = precise_rotate(layer.input(x), -angles).reshape(len(x), layer.modes, 2)
            elapsed = torch.diff(times, prepend=times[:1]).to(x.dtype)
            decay = torch.exp(-elapsed[:, None]*rates[None])
            previous, initial = state.last_arrival[node], state.modes[node]
            if previous is not None:
                gap = (times[0]-previous).clamp_min(0).to(x.dtype)
                initial = precise_rotate(initial*torch.exp(-gap*rates).repeat_interleave(2),
                    (times[0]-previous).clamp_min(0)*layer.frequency.to(torch.float64))
            # The first affine drive owns incoming state; subsequent prefix
            # combinations transport it and preserve the cross-chunk teacher.
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
