"""Bounded monotone score bridge; alpha0 exactly retains the old hard clamp."""
import torch


def bounded_score_bridge(raw,alpha,bound=12.):
    if not 0.<=alpha<=1. or bound<=0:raise ValueError('Bridge in[0,1], positive bound required')
    hard=raw.clamp(-bound,bound)
    if alpha==0.:return hard
    smooth=raw/torch.sqrt(1+(raw/bound).square())
    return (1-alpha)*hard+alpha*smooth


def smooth_slope(raw,bound=12.):
    return (1+(raw/bound).square()).pow(-1.5)
