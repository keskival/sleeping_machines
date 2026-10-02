"""Finite paired choice credit; actual outcome loss differences, bounded proposal."""
import torch
from torch.nn import functional as F


def alternative_probabilities(probabilities,winner,exploration=.1):
    if probabilities.ndim!=2 or probabilities.shape[1]<2 or not 0<=exploration<1:
        raise ValueError('At least two candidates and exploration in[0,1) required')
    pi=probabilities.double();pool=pi.shape[1]
    nonwinner=torch.ones_like(pi)-F.one_hot(winner,pool).to(pi.dtype)
    conditional=pi*nonwinner
    return (1-exploration)*conditional/conditional.sum(-1,keepdim=True)+exploration*nonwinner/(pool-1)


def paired_choice_credit(probabilities,winner,alternative,proposal,factual_loss,alternative_loss):
    pi=probabilities.double()
    if bool((winner==alternative).any()):raise ValueError('Actual nonwinner alternative required')
    selected=alternative[:,None]
    propensity=proposal.gather(1,selected).squeeze(-1)
    if bool((propensity<=0).any()):raise ValueError('Positive alternative propensity required')
    importance=pi.gather(1,selected).squeeze(-1)/propensity
    difference=alternative_loss.double()-factual_loss.double()
    direction=F.one_hot(alternative,pi.shape[1]).to(pi.dtype)-pi
    return importance[:,None]*difference[:,None]*direction


def sample_alternative(proposal):
    # Zero propensity at the factual winner produces infinite proposal time.
    # Every eligible alternative has positive propensity; exactly one is drawn.
    noise=torch.empty_like(proposal).exponential_()
    return (noise/proposal).min(-1).indices
