"""Constructive information-loss, observability and race-statistic witnesses.

Read-only numerical checks, no fits or optimizer steps. See theory notes54–56.
"""
import numpy as np


def test_precision_messages_resolve_a_raw_mean_ambiguity():
    observations = np.array([-1., 1.])
    means = []
    for precision in (np.array([100., 1.]), np.array([1., 100.])):
        # Unit-precision, zero-mean prior and two independent Gaussian sensors.
        eta = (precision*observations).sum()
        total = 1.+precision.sum()
        means.append(eta/total)
    assert observations.mean() == 0
    assert means[0] < -.9 and means[1] > .9


def test_rotation_reception_identifies_state_but_aliased_samples_do_not():
    def design(times):
        return np.stack((np.cos(times), -np.sin(times)), axis=1)
    state = np.array([.8, -.4])
    alias = design(np.array([0., 2*np.pi, 4*np.pi]))
    useful = design(np.array([0., .7, 1.3]))
    assert np.linalg.matrix_rank(alias, tol=1e-10) == 1
    assert np.linalg.matrix_rank(useful, tol=1e-10) == 2
    estimate = np.linalg.solve(useful.T@useful, useful.T@(useful@state))
    np.testing.assert_allclose(estimate, state, atol=1e-12)


def test_race_time_carries_common_rate_information_absent_from_winner():
    rng = np.random.default_rng(621)
    clocks = rng.exponential(size=(100000, 3))/np.array([.5, 1., 1.5])
    minimum = clocks.min(1)
    winners = clocks.argmin(1)
    # Common scaling leaves every winner identical and changes every arrival.
    scaled = clocks/5
    np.testing.assert_array_equal(winners, scaled.argmin(1))
    np.testing.assert_allclose(scaled.min(1), minimum/5, rtol=1e-12)
    score = 1.-3.*minimum  # derivative w.r.t. log(total rate)
    assert abs(score.mean()) < .02
    assert abs(score.var()-1.) < .03


def test_coarse_to_fine_information_details_are_orthogonal():
    # Known coarse/fine independent causes, rather than neural depth as a clock.
    coarse = np.repeat(np.array([-1., 1.]), 2)
    fine = np.tile(np.array([-1., 1.]), 2)
    target = 2*coarse+.5*coarse*fine
    conditional_coarse = target.reshape(2, 2).mean(1).repeat(2)
    detail = target-conditional_coarse
    assert (conditional_coarse*detail).mean() == 0
    np.testing.assert_allclose((target**2).mean(),
        (conditional_coarse**2).mean()+(detail**2).mean(), atol=1e-12)


def test_individual_redundancy_can_hide_complementary_factors():
    first = np.repeat(np.array([0, 1]), 2)
    second = np.tile(np.array([0, 1]), 2)
    target = first ^ second
    for factor in (first, second):
        for value in (0, 1):
            assert target[factor == value].mean() == .5
    assert np.all((first ^ second) == target)
