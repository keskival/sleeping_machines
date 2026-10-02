"""Exact-pi linearized race credit (THEORY §400).

The counterfactual teacher of TemporalRoute scores race i with rate_i * T * g.(v_i - mean(v)) and conserves the total
at the winner.  Its expectation over the winner is the linearized expected-loss gradient pi_i g.(v_i - sum_j pi_j v_j),
but the factor rate_i * T is a one-sample estimate of pi_i, and pi is known exactly from the scores.  This Function keeps
the forward race (same noise, winner, value and bounded delay) and replaces the score credit by its exact expectation
over the winner:

    d/ds_i = pi_i * g.(v_i - sum_j pi_j v_j)       (+ the realized winner's exact interior delay derivative)

Value credit still reaches only the winner (its realized delivery).  Same cost as the original teacher: every
proposal value is already computed in training.  Unbiasedness is unchanged; only estimator noise is removed.
"""
import torch

from .sparse_race_language import TemporalRoute


class ExactPiRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, values):
        rates = scores.to(torch.float64).exp()
        noise = torch.empty_like(rates).exponential_()
        times = noise / rates
        time, winner = times.min(dim=0)
        ctx.save_for_backward(rates, time, winner, values)
        return values[winner], .001 + .010 * time / (1 + time), winner

    @staticmethod
    def backward(ctx, error_value, error_delay, error_winner):
        rates, time, winner, values = ctx.saved_tensors
        pi = rates / rates.sum()
        credit = torch.zeros_like(rates)
        value_credit = torch.zeros_like(values)
        if error_value is not None:
            direction = (values @ error_value).to(rates.dtype)
            credit = pi * (direction - (pi * direction).sum())
            value_credit = value_credit.index_add(0, winner[None], error_value[None])
        if error_delay is not None:
            timing = -error_delay * .010 * time / (1 + time).square()
            credit = credit.scatter_add(0, winner[None], timing[None])
        return credit.to(values.dtype), value_credit


def exact_pi_race(scores, values=None):
    """drop-in for AddressedEventHeads.race in training (values given); evaluation draws identically."""
    if values is None:
        rates = scores.to(torch.float64).exp()
        time, winner = (torch.empty_like(rates).exponential_() / rates).min(0)
        return None, .001 + .010 * time / (1 + time), winner
    return ExactPiRoute.apply(scores, values)


def use_exact_pi_credit(model):
    model.race = exact_pi_race
    return model


__all__ = ['ExactPiRoute', 'exact_pi_race', 'use_exact_pi_credit', 'TemporalRoute']
