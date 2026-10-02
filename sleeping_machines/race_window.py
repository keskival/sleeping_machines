"""Exact conditional arrival law for a fixed deadline after an exponential race.

This is a numerical/theoretical primitive, not an installed native layer.
All clocks share an admission time and the same strictly increasing delay map.
"""
import torch


def bounded_delay(raw_time):
    return .001+.010*raw_time/(1+raw_time)


def residual_cutoff(first_time,width):
    """Raw residual allowed by g(T+X)<=g(T)+H; return finite cutoff and cap flag.

    Once the deadline covers the remaining delay span, every loser is heard.
    The finite placeholder on that branch avoids infinity-times-zero gradients;
    callers must use the returned cap flag. Float64 timing is required.
    """
    if first_time.dtype!=torch.float64 or width.dtype!=torch.float64:
        raise ValueError('Float64 physical timing required')
    if not torch.isfinite(first_time).all() or not torch.isfinite(width).all():
        raise ValueError('Finite clocks and widths required')
    if not (first_time>=0).all() or not (width>=0).all():
        raise ValueError('Nonnegative first time and width required')
    span=.010-width*(1+first_time)
    covered=span<=0
    cutoff=width*(1+first_time).square()/span.clamp_min(1e-30)
    return torch.where(covered,torch.zeros_like(cutoff),cutoff),covered


def residual_probability(log_rate,first_time,width):
    cutoff,covered=residual_cutoff(first_time,width)
    probability=-torch.expm1(-log_rate.exp()*cutoff)
    return torch.where(covered,torch.ones_like(probability),probability)


def conditional_residual(log_rate,first_time,width,uniform):
    """Residual of a heard loser, conditioned on its membership.

    q=1-exp(-lambda R), X=-log(1-U*q)/lambda. Covered-span branches
    use the untruncated exponential law. Differentiating this conditional draw
    alone is insufficient: membership-probability and winner credit are needed.
    """
    if not ((uniform>0)&(uniform<1)).all():
        raise ValueError('Uniform nodes strictly inside (0,1) required')
    q=residual_probability(log_rate,first_time,width)
    return -torch.log1p(-uniform*q)/log_rate.exp()
