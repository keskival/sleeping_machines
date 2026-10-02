"""Context-addressed persistent memory for the native temporal core (THEORY §390 addendum).

The native language core has one source address, so all tokens share a few units (512 floats of state).
Count tables win with thousands of context addresses.  This module gives the native core addressed
capacity: address a_t = hash(last k tokens) mod B selects one memory slot, O(1) per event.

  outcome write  at event t, the core's top-layer state h_t is written into the slot of the *previous*
                 context a_{t-1}: the slot accumulates what followed that context (a learned-value
                 count table: running mean of written vectors, with its count);
  read           the slot of the current context a_t is read and delivered into the core's input through
                 R (zero-initialised), so the untrained model equals the native core exactly.

Capacity grows with distinct contexts while activity stays one read and one write per event.  Fixed hashed
keys are the first, nested step; learned keys that pool contexts by predictive similarity (§382) are the
next.  Writes keep their graph inside the credit chunk and are detached at chunk boundaries.
"""
import torch
from torch import nn

from .native_stream_language import NativeLanguageState, NativeStreamLanguageModel

_PRIME = 1_000_003


class ContextMemoryState(NativeLanguageState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slots, self.counts, self.history, self.previous_address = {}, {}, [], None
        self.memory_reads = self.memory_writes = 0

    def detach(self):
        super().detach()
        self.slots = {a: v.detach() for a, v in self.slots.items()}
        return self

    def storage(self):
        stats = super().storage()
        slot_bytes = sum(v.numel()*v.element_size() for v in self.slots.values())
        stats.update(context_slots=len(self.slots), context_slot_tensor_bytes=slot_bytes,
                     persistent_tensor_bytes=stats['persistent_tensor_bytes']+slot_bytes,
                     context_count_entries=len(self.counts))
        stats['scope'] += '; context-slot tensors included; integer counts and Python metadata separate'
        return stats

    def packed_storage(self):
        stats = super().packed_storage()
        stats['differentiable_entries'] += sum(int(v.requires_grad) for v in self.slots.values())
        return stats


class _ContentWithMemory(nn.Linear):
    def __init__(self, base, owner):
        super().__init__(base.in_features, base.out_features, bias=False)
        with torch.no_grad():
            self.weight.copy_(base.weight)
        self._owner = [owner]

    def forward(self, mark):
        out = super().forward(mark)
        read = self._owner[0]._memory_read
        return out if read is None else out + self._owner[0].memory_read_map(read)


class ContextAddressedNativeModel(NativeStreamLanguageModel):
    def __init__(self, payload=16, depth=8, pool=2, heads=2, vocabulary=27, order=3, buckets=4096):
        super().__init__(payload, depth, pool, heads, vocabulary)
        if order < 1 or buckets < 1:
            raise ValueError('positive context order and bucket count required')
        self.order, self.buckets = order, buckets
        d = self.total_payload
        self.content = _ContentWithMemory(self.content, self)
        self.memory_write_map = nn.Linear(d, d, bias=False)
        self.memory_read_map = nn.Linear(d + 1, d, bias=False)
        nn.init.zeros_(self.memory_read_map.weight)
        self._memory_read = None

    def new_state(self):
        return ContextMemoryState()

    def address(self, history):
        a = 0
        for token in history[-self.order:]:
            a = (a * _PRIME + int(token) + 1) % (2 ** 61 - 1)
        return a % self.buckets

    def forward_chunk(self, tokens, state=None):
        state = self.new_state() if state is None else state
        indices = torch.as_tensor(tokens, dtype=torch.long, device=self.embedding.weight.device)
        if indices.ndim != 1 or not len(indices):
            raise ValueError('Nonempty observed token stream required')
        d, outputs = self.total_payload, []
        for token in indices:
            state.history = (state.history + [int(token)])[-self.order:]
            a = self.address(state.history)
            slot, n = state.slots.get(a), state.counts.get(a, 0)
            mean = slot / n if n else self.embedding.weight.new_zeros(d)
            self._memory_read = torch.cat([mean, mean.new_tensor([float(n > 0)])])
            state.memory_reads += 1
            try:
                mark = torch.nn.functional.one_hot(token, num_classes=self.vocabulary).to(self.embedding.weight.dtype)
                logits, _ = self.consume_event(0, float(state.events), mark, state)
            finally:
                self._memory_read = None
            outputs.append(logits)
            if state.previous_address is not None:  # outcome write: what followed the previous context
                b, h = state.previous_address, state.contexts[0][0]
                v = self.memory_write_map(h)
                state.slots[b] = v if b not in state.slots else state.slots[b] + v
                state.counts[b] = state.counts.get(b, 0) + 1
                state.memory_writes += 1
            state.previous_address = a
        return torch.stack(outputs), state
