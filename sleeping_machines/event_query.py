"""Explicit observed-prefix queries. Targets never enter the event batch.

This offline interface replays each supplied prefix independently. It is not a
persistent online scheduler. A query may be answered after its delayed carriers
finish; the observation cutoff and compute completion time are different.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class ObservedPrefix:
    channels: np.ndarray
    times: np.ndarray
    counts: np.ndarray
    cutoff: float
    marks: np.ndarray | None = None

    def validate(self):
        n = len(self.channels)
        if not n or self.times.shape != (n,) or self.counts.shape != (n,):
            raise ValueError("A query needs nonempty, aligned event arrays")
        if not np.issubdtype(self.channels.dtype, np.integer) or np.any(self.channels < 0):
            raise ValueError("Channels must be nonnegative integers")
        if not np.isfinite(self.cutoff) or not np.isfinite(self.times).all():
            raise ValueError("Query and event times must be finite")
        if np.any(np.diff(self.times) < 0) or self.times[0] < 0 or self.times[-1] > self.cutoff + 1e-7:
            raise ValueError("Only chronological events observed by the cutoff may enter")
        if not np.isfinite(self.counts).all() or np.any(self.counts <= 0):
            raise ValueError("Counts must be finite and positive")
        if self.marks is not None and (self.marks.ndim != 2 or len(self.marks) != n
                                      or not np.isfinite(self.marks).all()):
            raise ValueError("Continuous marks must be finite and aligned")


def pack_queries(prefixes):
    if not prefixes:
        raise ValueError("Empty query batch")
    for p in prefixes:
        p.validate()
    if len({p.marks is None for p in prefixes}) != 1:
        raise ValueError("All queries must use the same mark schema")
    result = {
        "b": torch.from_numpy(np.concatenate([p.channels for p in prefixes])).long(),
        "t": torch.from_numpy(np.concatenate([p.times for p in prefixes])).float(),
        "c": torch.from_numpy(np.concatenate([p.counts for p in prefixes])).float(),
        "ids": torch.repeat_interleave(torch.arange(len(prefixes)),
                                       torch.tensor([len(p.channels) for p in prefixes])),
        "size": len(prefixes),
    }
    if prefixes[0].marks is not None:
        result["continuous"] = torch.from_numpy(np.concatenate([p.marks for p in prefixes])).float()
    return result
