"""Factorized race credit: categorical choice and common first-time clock (THEORY note 92 remedy 2; §402 correction).

Competing exponential clocks with rates lambda_i = exp(s_i) satisfy W ~ pi = lambda / Lambda and T | W ~ Exp(Lambda),
independently: the race is distributionally W ~ Categorical(pi), T = E / Lambda with E ~ Exp(1).  The forward pass
is the usual race (same noise, winner, payload, bounded delay).  The backward pass uses this factorization:

  value credit   to the realized winner's payload only;
  clock credit   pathwise through T = E / Lambda: dT/ds_i = -T * pi_i for every candidate (not only the winner);
  choice credit  none here: it comes from first-time-preserving counterfactual replays
                 (sum_i pi_i * stopgrad(F_i(T)), dvs_local_expectation_benchmark), whose gradient is
                 pi_i (F_i(T) - sum_j pi_j F_j(T)).

Forced replay of alternative i keeps the factual first time T and changes only identity (payload and memory write):
`force_at_first_time`.
"""
import torch


def delay_of(time):
    return .001 + .010 * time / (1 + time)


class FactorizedRace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, values):
        rates = scores.to(torch.float64).exp()
        times = torch.empty_like(rates).exponential_() / rates
        time, winner = times.min(dim=0)
        ctx.save_for_backward(rates, time, winner, values)
        return values[winner], delay_of(time), winner

    @staticmethod
    def backward(ctx, error_value, error_delay, error_winner):
        rates, time, winner, values = ctx.saved_tensors
        pi = rates / rates.sum()
        credit = torch.zeros_like(rates)
        value_credit = torch.zeros_like(values)
        if error_value is not None:
            value_credit = value_credit.index_add(0, winner[None], error_value[None])
        if error_delay is not None:
            dtime = error_delay * .010 / (1 + time).square()          # d delay / dT
            credit = dtime * (-time * pi)                              # dT/ds_i = -T pi_i
        return credit.to(values.dtype), value_credit


def factorized_race(scores, values=None):
    if values is None:
        rates = scores.to(torch.float64).exp()
        time, winner = (torch.empty_like(rates).exponential_() / rates).min(0)
        return None, delay_of(time), winner
    return FactorizedRace.apply(scores, values)


def force_at_first_time(scores, values, index):
    """counterfactual race outcome: identity `index`, factual first time (identical RNG consumption)."""
    rates = scores.to(torch.float64).exp()
    time = (torch.empty_like(rates).exponential_() / rates).min()
    return values[index], delay_of(time), torch.tensor(index)
