"""Shared matches drive a content race and independent clock-only policies.

Clock-only outputs are continuous minima: their pathwise clock derivatives
need no losing-value proposals. The content race retains its existing local
counterfactual teacher and exact interior arrival derivative.
"""
import torch


class ContentAndClockRaces(torch.autograd.Function):
    @staticmethod
    def forward(ctx, policy_scores, values):
        rates = policy_scores.to(torch.float64).exp()
        times, winners = (torch.empty_like(rates).exponential_()/rates).min(-1)
        ctx.save_for_backward(rates,times,winners,values)
        return values[winners[0]], .001+.010*times/(1+times), winners[0]

    @staticmethod
    def backward(ctx, error_value, error_delays, error_winner):
        rates,times,winners,values = ctx.saved_tensors
        credit = torch.zeros_like(rates)
        value_credit = torch.zeros_like(values)
        if error_value is not None and values.numel():
            direction = (values-values.mean(0)) @ error_value
            primary = rates[0]*times[0]*direction.to(rates.dtype)
            primary = primary.scatter_add(0,winners[:1],-primary.sum()[None])
            credit[0] = primary
            value_credit = value_credit.index_add(0,winners[:1],error_value[None])
        if error_delays is not None:
            timing = -error_delays*.010*times/(1+times).square()
            credit = credit.scatter_add(1,winners[:,None],timing[:,None])
        return credit.to(values.dtype), value_credit
