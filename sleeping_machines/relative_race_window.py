"""Race-scaled reception support; physical transport remains a separate operator."""
import torch
from .race_window import bounded_delay


def relative_deadline(first_time,extension):
    if not torch.isfinite(first_time).all() or not torch.isfinite(extension).all():
        raise ValueError('Finite clock and extension required')
    if not (first_time>=0).all() or not (extension>=0).all():
        raise ValueError('Nonnegative first clock and extension required')
    return bounded_delay((1+extension)*first_time)


def expected_relative_receivers(scores,extension):
    """Exact expected count at fixed entering scores, integrating winner and time.

    Independent Exp(exp(scores)) clocks with common start, same delay map and
    reception A_j <= (1+c) min_i A_i. No dependence on a common rate multiplier.
    This count is not utility, inference FLOPs or a complete training objective.
    """
    if scores.ndim<1 or scores.shape[-1]<1 or not torch.isfinite(scores).all():
        raise ValueError('Finite nonempty candidate scores required')
    if extension.ndim or not torch.isfinite(extension) or not extension>=0:
        raise ValueError('Finite scalar nonnegative extension required')
    pi=scores.softmax(-1);extra=extension*pi/(1+extension*pi)
    return 1+(pi*(extra.sum(-1,keepdim=True)-extra)).sum(-1)


def maximum_added_physical_delay(extension):
    root=torch.sqrt(1+extension)
    return .010*(root-1)/(root+1)
