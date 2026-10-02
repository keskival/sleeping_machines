"""Defer linear value projection until addressed-memory retrieval.

Store sums of native top-context features, then apply the current learned value
projection to their mean. With fixed weights this equals storing projected
values (linearity). After detach it preserves exact fixed-feature credit to the
projection and avoids mixing old projection versions. Historical feature
producers remain truncated; fixed hashed addresses are not learned pooling.
"""
import torch
from torch.nn import functional as F

from .context_addressed_memory import ContextAddressedNativeModel, ContextMemoryState


class LateProjectedMemoryState(ContextMemoryState):
    """Slots contain pre-projection feature sums, never projected values."""


class LateProjectedContextModel(ContextAddressedNativeModel):
    def new_state(self):
        return LateProjectedMemoryState()

    def read_value(self, state, address):
        n = state.counts.get(address, 0)
        if not n:
            return self.embedding.weight.new_zeros(self.total_payload)
        return self.memory_write_map(state.slots[address] / n)

    def forward_chunk(self, tokens, state=None):
        state = self.new_state() if state is None else state
        if not isinstance(state, LateProjectedMemoryState):
            raise TypeError('Late-projection slots require raw-feature state; projected-value state is incompatible')
        indices = torch.as_tensor(tokens, dtype=torch.long, device=self.embedding.weight.device)
        if indices.ndim != 1 or not len(indices):
            raise ValueError('Nonempty observed token stream required')
        outputs = []
        for token in indices:
            state.history = (state.history + [int(token)])[-self.order:]
            address = self.address(state.history)
            value = self.read_value(state, address)
            n = state.counts.get(address, 0)
            self._memory_read = torch.cat([value, value.new_tensor([float(n > 0)])])
            state.memory_reads += 1
            try:
                mark = F.one_hot(token, num_classes=self.vocabulary).to(self.embedding.weight.dtype)
                logits, _ = self.consume_event(0, float(state.events), mark, state)
            finally:
                self._memory_read = None
            outputs.append(logits)
            if state.previous_address is not None:
                previous = state.previous_address
                feature = state.contexts[0][0]
                state.slots[previous] = feature if previous not in state.slots else state.slots[previous] + feature
                state.counts[previous] = state.counts.get(previous, 0) + 1
                state.memory_writes += 1
            state.previous_address = address
        return torch.stack(outputs), state
