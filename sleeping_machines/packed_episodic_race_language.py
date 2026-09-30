"""Lossless storage for the content-indexed sparse temporal language model.

The inherited model still computes the same races and counterfactual teachers.
Only detached historical K/V storage and integer bucket lists are packed.
Recent differentiable entries keep their original tensors until credit ends.
No eviction, quantization, compression, index change or extra RNG is introduced.
"""
from array import array
from dataclasses import dataclass
import operator

import torch

from .indexed_episodic_race_language import (
    IndexedEpisodicRaceLanguageModel, IndexedEpisodicState,
)


class PackedKVBank:
    """Append-only K/V slabs with a small differentiable write buffer.

    Stored row views are immutable. New writes cannot alter historical rows.
    Races stack admitted keys/teaching values before using them in autograd;
    the inherited inference path reads the selected value directly.
    """

    def __init__(self, block_size=256):
        if block_size < 1:
            raise ValueError('Positive slab size required')
        self.block_size = block_size
        self.slabs = []
        self.positions = array('q')
        self.pending = {}

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index in self.pending:
            key, value = self.pending[index]
        else:
            slab, row = divmod(index, self.block_size)
            keys, values = self.slabs[slab]
            key, value = keys[row], values[row]
        return key, value, self.positions[index]

    def append(self, entry):
        key, value, position = entry
        if key.ndim != 1 or key.shape != value.shape:
            raise ValueError('Equal vector keys and values required')
        if key.dtype != value.dtype or key.device != value.device:
            raise ValueError('Key/value dtype and device must match')
        if self.slabs:
            prototype = self.slabs[0][0]
            if (key.numel() != prototype.shape[1] or key.dtype != prototype.dtype
                    or key.device != prototype.device):
                raise ValueError('A bank must retain a single storage format')
        index = len(self)
        slab, row = divmod(index, self.block_size)
        if slab == len(self.slabs):
            # Zero unused rows so checkpoints never serialize uninitialized data.
            shape = (self.block_size, key.numel())
            self.slabs.append((key.new_zeros(shape), value.new_zeros(shape)))
        self.positions.append(operator.index(position))
        if key.requires_grad or value.requires_grad:
            self.pending[index] = (key, value)
        else:
            with torch.no_grad():
                self.slabs[slab][0][row].copy_(key)
                self.slabs[slab][1][row].copy_(value)

    def seal(self):
        """Detach and copy only this credit segment, never rewalk old entries."""
        with torch.no_grad():
            groups = {}
            for index, pair in self.pending.items():
                slab, row = divmod(index, self.block_size)
                groups.setdefault(slab, []).append((row, pair))
            for slab, entries in groups.items():
                # Pending rows are consecutive under append-only execution.
                start = entries[0][0]
                assert [row for row, _ in entries] == list(range(start, start + len(entries)))
                stop = start + len(entries)
                keys = torch.stack([pair[0].detach() for _, pair in entries])
                values = torch.stack([pair[1].detach() for _, pair in entries])
                self.slabs[slab][0][start:stop].copy_(keys)
                self.slabs[slab][1][start:stop].copy_(values)
        self.pending.clear()

    def storage(self):
        return dict(entries=len(self), slabs=len(self.slabs),
            allocated_key_value_bytes=sum(t.numel() * t.element_size()
                for slab in self.slabs for t in slab),
            position_bytes=len(self.positions) * self.positions.itemsize,
            differentiable_entries=len(self.pending))


class PackedBucketMap(dict):
    """Same bucket order/indices as the reference, using signed 64-bit IDs."""

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = array('q', () if default is None else default)
        return self[key]


@dataclass
class PackedEpisodicState(IndexedEpisodicState):
    def detach(self):
        for bank in self.banks:
            bank.seal()
        self.detached_until = len(self.banks[0]) if self.banks else 0
        self.memories = {k: v.detach() for k, v in self.memories.items()}
        self.arrivals = {k: v.detach() for k, v in self.arrivals.items()}
        if self.context is not None:
            self.context = self.context.detach()
        return self

    def packed_storage(self):
        stats = [bank.storage() for bank in self.banks]
        return dict(entries=sum(s['entries'] for s in stats),
            slabs=sum(s['slabs'] for s in stats),
            allocated_key_value_bytes=sum(s['allocated_key_value_bytes'] for s in stats),
            position_bytes=sum(s['position_bytes'] for s in stats),
            bucket_id_bytes=sum(len(ids) * ids.itemsize
                for buckets in self.semantic_buckets for ids in buckets.values()),
            differentiable_entries=sum(s['differentiable_entries'] for s in stats),
            scope='Tensor and integer-array payload; excludes Python/allocator metadata and live autograd graphs')


class PackedEpisodicRaceLanguageModel(IndexedEpisodicRaceLanguageModel):
    """Identical initialization and inherited event computations; new storage."""

    def __init__(self, *args, block_size=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        if block_size < 1:
            raise ValueError('Positive slab size required')

    def new_state(self):
        return PackedEpisodicState(
            banks=[PackedKVBank(self.block_size) for _ in range(self.depth)],
            buckets=[{} for _ in range(self.depth)],
            semantic_buckets=[PackedBucketMap() for _ in range(self.depth)])
