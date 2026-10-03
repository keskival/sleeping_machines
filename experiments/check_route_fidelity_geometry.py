"""Scalar calibration/factorial identities, never a native model contract."""
import math
from pathlib import Path
import runpy
import sys


def main():
    ns = runpy.run_path(str(Path(__file__).with_name('route_fidelity_geometry.py')))
    measure, factorial, addition = ns['fidelity'], ns['factorial'], ns['addition_fidelity']
    p, q, a = [.2, .3, .5], [1., 4., -2.], [2., -3., 1.]
    result = measure(p, q, a)
    error = sum((x-y)**2/z for x, y, z in zip(result['exact_local_choice_gradient'],
                                             result['approximate_local_choice_gradient'], p))
    assert math.isclose(error, result['pi_whitened_error_squared'], abs_tol=1e-12)
    shifted = measure(p, [x+7 for x in q], [x-4 for x in a])
    for key in ('pi_whitened_error_squared', 'pi_metric_inner_product', 'oracle_nonnegative_scale'):
        assert math.isclose(result[key], shifted[key], abs_tol=1e-12)
    exact = measure(p, q, q)
    assert exact['pi_whitened_error_squared'] == 0 and exact['oracle_nonnegative_scale'] == 1
    scaled = measure(p, q, [2*x+5 for x in q])
    assert scaled['oracle_nonnegative_scale'] == .5 and scaled['oracle_scaled_error_squared'] == 0
    opposed = measure([.4, .6], [1., 0.], [0., 1.])
    assert opposed['pi_metric_cosine'] < -.999999 and opposed['oracle_nonnegative_scale'] == 0
    missing = measure([.3, .7], [math.log1p(math.exp(-1)), math.log(2)], [3., 3.])
    assert missing['exact_pi_whitened_norm_squared'] > 0
    assert missing['approximate_pi_whitened_norm_squared'] == 0 and missing['oracle_signed_scale'] is None
    # Value and write separately have zero effect, but their interaction matters.
    parts = factorial(3., 3., 3., 1., 0.)
    assert parts['delivered_value_effect'] == parts['persistent_commit_effect'] == 0
    assert parts['delivery_commit_interaction'] == parts['teacher_residual'] == -2
    parts = factorial(2., 1., 4., 5., -.5)
    assert parts['teacher_residual'] == parts['residual_reconstruction']
    # An extra term opposed to Q can fix an overestimated base. What matters
    # is its alignment with the unexplained residual, not Q alone.
    corrected = addition([.4, .6], [1., 0.], [2., 0.], [-1., 0.])
    assert corrected['oracle_nonnegative_alpha'] == 1 and corrected['oracle_best_error_squared'] == 0
    harmful = addition([.4, .6], [1., 0.], [2., 0.], [1., 0.])
    assert harmful['oracle_nonnegative_alpha'] == 0
    assert harmful['unit_addition_error_squared'] > harmful['base_error_squared']
    pooled = ns['pooled_addition_fidelity']([
        ([.5, .5], [1., 0.], [0., 0.], [1., 0.]),
        ([.5, .5], [2., 0.], [0., 0.], [1., 0.])])
    assert pooled['shared_oracle_nonnegative_alpha'] == 1.5
    assert pooled['separate_oracle_local_error_sum'] == 0
    assert pooled['shared_alpha_local_error_sum'] == .125
    for probabilities in ([], [.2, .2], [0., 1.]):
        try:
            measure(probabilities, [1.]*len(probabilities), [2.]*len(probabilities))
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid probabilities admitted')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(dict(scalar_contracts='passed', baseline_invariance=True,
               positive_scale_cannot_fix_opposition=True, interaction_is_separate=True,
               native_contracts='pending'))


if __name__ == '__main__':
    main()
