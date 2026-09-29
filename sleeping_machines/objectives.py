"""Query-time objectives on common natural scores; no hidden time grid."""
import torch
from torch.nn import functional as F


def categorical_nll(scores, labels, reduction="mean"):
    return F.cross_entropy(scores, labels, reduction=reduction)


def hazard_nll(scores, labels, bucket, exposure, reduction="mean"):
    """Exact marked piecewise-constant hazard NLL, with an unbounded final bin.

    scores: [queries, bins, event types], exposure: observed duration in each
    bin until the *target* arrives. Exposure/bucket are labels, never features.
    Intensities are per physical second. No clipping changes this likelihood.
    """
    if scores.ndim != 3 or exposure.shape != scores.shape[:2]:
        raise ValueError("Invalid hazard shapes")
    if not torch.isfinite(exposure).all() or (exposure < 0).any():
        raise ValueError("Exposure must be finite and nonnegative")
    rows = torch.arange(len(scores), device=scores.device)
    loss = (scores.exp()*exposure[:, :, None]).sum((1, 2)) - scores[rows, bucket, labels]
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError(reduction)
    return loss.mean()
