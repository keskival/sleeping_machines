"""Conditional allocation for measured separable quality/cost response curves.

This is a research planning calculator, not an empirical language scaling law.
Coefficients must be estimated on development data and confirmed independently.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Axis:
    name: str
    amplitude: float
    exponent: float
    cost: float
    shift: float = 1.0
    minimum: float = 0.0
    maximum: float = math.inf


def allocate(axes, budget, overhead=0.0):
    """Minimize sum(a*(b+s)**-alpha) under overhead + sum(c*b) <= budget.

    Assumes positive finite coefficients and independent continuous axes.
    Real kernels, interactions, discrete capacities and memory/latency limits
    require separately measured constraints. Return allocations, never a claim.
    """
    if not axes or not math.isfinite(budget) or not math.isfinite(overhead) or overhead < 0:
        raise ValueError("Finite budget, nonnegative overhead and axes required")
    if len({a.name for a in axes}) != len(axes):
        raise ValueError("Axis names must be unique")
    for a in axes:
        positive = (a.amplitude, a.exponent, a.cost, a.shift)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Positive finite response coefficients required")
        if not math.isfinite(a.minimum) or a.minimum < 0 or math.isnan(a.maximum) or a.maximum < a.minimum:
            raise ValueError("Valid nonnegative axis bounds required")
    available = budget - overhead
    minimum_cost = sum(a.cost * a.minimum for a in axes)
    if available < minimum_cost:
        raise ValueError("Budget does not cover overhead and minimum allocation")
    if available == minimum_cost:
        return {a.name: a.minimum for a in axes}
    if sum(a.cost * a.maximum for a in axes) <= available:
        return {a.name: a.maximum for a in axes}

    def proposal(log_price):
        return [min(a.maximum, max(a.minimum,
            math.exp(min(700.0, (math.log(a.amplitude) + math.log(a.exponent)
                     - log_price - math.log(a.cost)) / (a.exponent + 1))) - a.shift))
            for a in axes]

    low, high = -700.0, 700.0
    for _ in range(160):
        middle = (low + high) / 2
        values = proposal(middle)
        if sum(a.cost * b for a, b in zip(axes, values)) > available:
            low = middle
        else:
            high = middle
    values = proposal(high)
    return {a.name: b for a, b in zip(axes, values)}
