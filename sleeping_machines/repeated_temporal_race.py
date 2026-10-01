"""Shared-rate Poisson arrivals and a conserved local counterfactual teacher.

A fixed query sets rates once. Only a winning emitter renews its clock.
The bounded time encoding is CPU/protocol emulation, not demonstrated hardware.
"""
import torch

from .sparse_race_language import TemporalRoute


def poisson_clocks(scores, samples):
    if samples < 1 or samples > 32 or scores.ndim != 1 or not len(scores):
        raise ValueError('Nonempty scores and 1..32 arrivals required')
    rates = scores.to(torch.float64).exp()
    clocks = torch.empty_like(rates).exponential_() / rates
    times, winners = [], []
    for sample in range(samples):
        time, winner = clocks.min(0)
        times.append(time); winners.append(winner)
        if sample + 1 < samples:
            # Losing clocks retain their absolute next-arrival times.
            renewal = torch.empty((), dtype=rates.dtype, device=rates.device).exponential_()
            clocks[int(winner)] = time + renewal / rates[winner]
    return rates, torch.stack(times), torch.stack(winners)


class PoissonTemporalRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, values, samples):
        if values.ndim != 2 or len(values) != len(scores):
            raise ValueError('One content vector per scored key required')
        rates, times, winners = poisson_clocks(scores, samples)
        ctx.save_for_backward(rates, times, winners, values)
        ctx.mark_non_differentiable(winners)
        return values[winners], .001 + .010 * times / (1 + times), winners

    @staticmethod
    def backward(ctx, error_values, error_delays, error_winners):
        rates, times, winners, values = ctx.saved_tensors
        score_credit = torch.zeros_like(rates)
        value_credit = torch.zeros_like(values)
        if error_values is not None:
            # Sum the per-arrival centered teachers in O(C*d + m*d), rather
            # than m separate all-candidate contractions. This is a declared
            # surrogate, not an exact derivative of nonlinear route changes.
            gaps = torch.diff(times, prepend=times.new_zeros(1))
            centered = (values - values.mean(0)).to(rates.dtype)
            errors = error_values.to(rates.dtype)
            accumulated_error = (gaps[:, None] * errors).sum(0)
            score_credit = rates * (centered @ accumulated_error)
            weighted_center = (rates[:, None] * centered).sum(0)
            correction = -gaps * (errors @ weighted_center)
            score_credit.index_add_(0, winners, correction)
            value_credit.index_add_(0, winners, error_values.to(values.dtype))
        if error_delays is not None:
            # At fixed marks each emitter's absolute arrival is Z_j/rate_j;
            # its interior derivative is -time for that winning emitter.
            timing = -error_delays.to(rates.dtype) * .010 * times / (1 + times).square()
            score_credit.index_add_(0, winners, timing)
        return score_credit.to(values.dtype), value_credit, None


def temporal_arrivals(scores, values, samples):
    if samples == 1:
        # Preserve the original value/clock/teacher/RNG path exactly.
        value, delay, winner = TemporalRoute.apply(scores, values)
        return value[None], delay[None], winner[None]
    return PoissonTemporalRoute.apply(scores, values, samples)
