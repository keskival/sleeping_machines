import math

import pytest

from experiments.compute_allocation import Axis, allocate


def test_allocation_obeys_full_budget_and_equal_marginal_value():
    axes = [Axis("memory", 18, 0.7, 2), Axis("retrieval", 6, 0.4, 3)]
    result = allocate(axes, budget=100, overhead=10)
    assert sum(a.cost * result[a.name] for a in axes) == pytest.approx(90)
    prices = [a.amplitude * a.exponent * (result[a.name] + a.shift)
              ** (-a.exponent - 1) / a.cost for a in axes]
    assert prices[0] == pytest.approx(prices[1])
    # Compare the predicted loss with feasible fixed-cost perturbations.
    def loss(memory):
        values = [memory, (90 - 2 * memory) / 3]
        return sum(a.amplitude * (b + a.shift) ** -a.exponent
                   for a, b in zip(axes, values))
    optimum = loss(result["memory"])
    assert all(optimum <= loss(x) + 1e-12 for x in range(46))


def test_bounds_and_setup_are_binding_not_free():
    axes = [Axis("memory", 10, 1, 2, minimum=1, maximum=2),
            Axis("depth", 1, 1, 1, minimum=1, maximum=4)]
    assert allocate(axes, 6, overhead=3) == {"memory": 1, "depth": 1}
    assert allocate(axes, 100, overhead=3) == {"memory": 2, "depth": 4}
    with pytest.raises(ValueError):
        allocate(axes, 5, overhead=3)
    with pytest.raises(ValueError):
        allocate([Axis("invalid", 1, 1, 1, maximum=math.nan)], 10)


def test_attention_omission_bound_has_required_value_and_head_conditions():
    # A high-value omitted item can matter despite small omitted mass.
    import numpy as np
    retained, omitted = np.array([1., 0.]), np.array([-1., 0.])
    epsilon = 0.03
    full = (1 - epsilon) * retained + epsilon * omitted
    head = np.array([[2., 0.], [-2., 0.]])
    def nll(state):
        logits = head @ state
        return np.logaddexp.reduce(logits) - logits[1]
    assert abs(nll(full) - nll(retained)) <= 4 * 2 * epsilon
