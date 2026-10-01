import numpy as np

from experiments.race_gradient_reference import conditional_credit, local_teacher, audit


def test_pure_time_credit_has_common_rate_direction():
    scores = np.log([1., 3.]); losses = np.array([2., 2.])
    credit = conditional_credit(scores, losses, np.array([4., 4.]), .5)
    np.testing.assert_allclose(credit, [-.5, -1.5])
    assert credit.sum() == -2.


def test_nonlinear_teacher_can_reverse_true_descent_direction():
    values = np.array([0., 2.]); losses = np.exp(values)-4*values
    true = .5*(losses-losses.mean())
    teacher = sum(.5*local_teacher(np.zeros(2), values, np.exp(values[w])-4, .5, w)
                  for w in range(2))
    # The positive dot product with the NEGATIVE teacher means the expected
    # approximate descent update locally increases the exact objective.
    assert true @ (-teacher) > 0


def test_joint_clock_reference_and_finite_differences():
    result = audit()
    assert result['polynomial_score_gradient']['max_abs_error'] < 1e-9
    assert result['linear_joint_clock_credit']['max_abs_error'] < 1e-10
    assert result['conditional_variance']['covariance_removed_trace'] > 0
