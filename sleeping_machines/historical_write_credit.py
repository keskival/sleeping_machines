"""Fixed-feature delayed write adjoints; these are declared local surrogates.

Cached primals never change. A historical producer perturbation is transported
to the current shared map, without reopening the old representation graph.
"""
import operator

import torch


class PackedFeatureBank:
    """One immutable detached feature vector per cached K/V write."""

    def __init__(self, block_size=256):
        if block_size < 1:
            raise ValueError('Positive slab size required')
        self.block_size, self.slabs, self.count = block_size, [], 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        slab, row = divmod(index, self.block_size)
        return self.slabs[slab][row]

    def append(self, feature):
        if feature.ndim != 1:
            raise ValueError('Vector write feature required')
        if self.slabs:
            prototype = self.slabs[0]
            if (feature.numel() != prototype.shape[1] or feature.dtype != prototype.dtype
                    or feature.device != prototype.device):
                raise ValueError('A bank must retain one storage format')
        slab, row = divmod(self.count, self.block_size)
        if slab == len(self.slabs):
            self.slabs.append(feature.new_zeros((self.block_size, feature.numel())))
        with torch.no_grad():
            self.slabs[slab][row].copy_(feature.detach())
        self.count += 1

    def storage(self):
        return dict(entries=self.count,
                    allocated_feature_bytes=sum(t.numel()*t.element_size() for t in self.slabs))


class HistoricalKeyMatch(torch.autograd.Function):
    """Ordinary cached key match, plus a factorized old-write map teacher.

    dW = alpha outer(q, sum_i error_i feature_i). No old feature adjoints.
    This is the derivative of a COMMON perturbation of historical write maps,
    conditional on saved features; it is not a derivative of the forward value
    with respect to the current map (which the cached primal does not use).
    """

    @staticmethod
    def forward(ctx, query, keys, features, weight, indices, alpha):
        ctx.save_for_backward(query, keys, features, indices)
        ctx.alpha = alpha
        return keys @ query

    @staticmethod
    def backward(ctx, error):
        query, keys, features, indices = ctx.saved_tensors
        gradient = torch.outer(query, features.T @ error[indices]) * ctx.alpha
        return keys.T @ error, error[:, None]*query[None, :], None, gradient, None, None


class HistoricalValueCredit(torch.autograd.Function):
    """Identity delivery plus the winner's fixed-feature historical map credit."""

    @staticmethod
    def forward(ctx, value, feature, weight, alpha):
        ctx.save_for_backward(feature)
        ctx.alpha = alpha
        return value

    @staticmethod
    def backward(ctx, error):
        (feature,) = ctx.saved_tensors
        return error, None, torch.outer(error, feature)*ctx.alpha, None
