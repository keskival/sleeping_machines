"""Read-only exact conditional race-score references; no fitted model changes."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def probabilities(scores):
    rates = np.exp(np.asarray(scores, dtype=float))
    return rates, rates / rates.sum()


def conditional_credit(scores, losses, time_derivatives, minimum_time):
    """Exact conditional categorical + total-rate credit for smooth branch costs.

    Branch losses include the actual route-specific persistent transition and
    suffix outcome. This is a reference, not an installed autograd estimator.
    Expectation is over tau~Exp(1), T=tau/sum(exp(scores)), and suffix randomness.
    """
    _, p = probabilities(scores)
    losses = np.asarray(losses, dtype=float)
    derivatives = np.asarray(time_derivatives, dtype=float)
    return p * (losses - p @ losses - minimum_time * (p @ derivatives))


def local_teacher(scores, values, error_value, minimum_time, winner):
    rates, _ = probabilities(scores)
    direction = (np.asarray(values) - np.mean(values)) * error_value
    credit = rates * minimum_time * direction
    credit[winner] -= credit.sum()
    return credit


def audit():
    nodes, weights = np.polynomial.laguerre.laggauss(64)
    scores = np.array([.3, -.4, .8])
    a, b, c = np.array([1., 4., -2.]), np.array([.2, -.3, .8]), np.array([.1, .2, .3])

    def objective(s):
        rates, p = probabilities(s); total = rates.sum()
        return p @ (a + b / total + 2 * c / total**2)

    rates, p = probabilities(scores)
    times = nodes / rates.sum()
    exact = sum(w * conditional_credit(scores, a+b*t+c*t*t, b+2*c*t, t)
                for t, w in zip(times, weights))
    finite = np.array([(objective(scores + np.eye(3)[i]*1e-5) -
                        objective(scores - np.eye(3)[i]*1e-5))/2e-5 for i in range(3)])
    np.testing.assert_allclose(exact, finite, rtol=1e-8, atol=1e-9)

    # Poisson NLL for target count4: exp(v)-4v+log(4!). It is strictly convex.
    scores2 = np.zeros(2); values = np.array([0., 2.]); _, p2 = probabilities(scores2)
    losses = np.exp(values) - 4 * values + math.lgamma(5)
    exact_poisson = conditional_credit(scores2, losses, np.zeros(2), .5)
    teacher = sum(p2[winner] * local_teacher(scores2, values, np.exp(values[winner])-4, .5, winner)
                  for winner in range(2))
    assert exact_poisson @ teacher < 0
    np.testing.assert_allclose(teacher.sum(), 0., atol=1e-15)

    # Time and route are independent, but winner-interior clock credit is not
    # a categorical/total-rate decomposition. The existing joint teacher can
    # nevertheless be exact for a linear payload loss: do not misdiagnose it.
    scores3 = np.array([.2, -.3]); values3 = np.array([-1., 2.])
    rates3, p3 = probabilities(scores3)
    exact_linear = np.zeros(2); existing_linear = np.zeros(2)
    for node, weight in zip(nodes, weights):
        t = node/rates3.sum(); delay = .001 + .010*t/(1+t); derivative = .010/(1+t)**2
        exact_linear += weight*conditional_credit(scores3, values3*delay, values3*derivative, t)
        for winner in range(2):
            contribution = local_teacher(scores3, values3, delay, t, winner)
            contribution[winner] -= values3[winner]*derivative*t
            existing_linear += weight*p3[winner]*contribution
    np.testing.assert_allclose(existing_linear, exact_linear, rtol=1e-9, atol=1e-11)

    # Enumerating winners Rao-Blackwellizes this reference estimator; variance
    # reduction is compared with the SAME hybrid estimator, not every estimator.
    losses0 = a+b*times[0]+c*times[0]**2
    deriv0 = b+2*c*times[0]
    hybrid = np.array([(np.eye(3)[i]-p)*losses0[i] - p*times[0]*deriv0[i] for i in range(3)])
    np.testing.assert_allclose(p @ hybrid, conditional_credit(scores, losses0, deriv0, times[0]))
    covariance = sum(p[i]*np.outer(hybrid[i]-p@hybrid, hybrid[i]-p@hybrid) for i in range(3))
    assert np.linalg.eigvalsh(covariance).min() > -1e-10

    return dict(status='completed', scope='Deterministic mathematical operator audit; no training, checkpoint quality or supremacy claim',
        polynomial_score_gradient=dict(reference=exact.tolist(), finite_difference=finite.tolist(),
            max_abs_error=float(np.max(np.abs(exact-finite)))),
        nonlinear_poisson_counterexample=dict(values=values.tolist(), count_target=4,
            true_score_gradient=exact_poisson.tolist(), expected_current_local_teacher=teacher.tolist(),
            inner_product=float(exact_poisson @ teacher), strictly_convex_loss=True,
            conclusion='Opposed expected route-score directions on this toy loss; no full-model diagnosis inferred'),
        linear_joint_clock_credit=dict(exact=exact_linear.tolist(), existing=existing_linear.tolist(),
            max_abs_error=float(np.max(np.abs(exact_linear-existing_linear))),
            conclusion='Current joint teacher passes linear-value/bounded-delay identity'),
        conditional_variance=dict(covariance_removed_trace=float(np.trace(covariance)),
            scope='Rao-Blackwellization of the stated hybrid reference, conditional on time and downstream noise'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    output = Path(parser.parse_args().output)
    if output.exists():
        raise ValueError('Never overwrite an audit result')
    result = audit(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
