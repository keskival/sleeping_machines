"""Finite-difference/adjoint and stability witnesses; no native runtime."""
import math
import runpy
from pathlib import Path
import sys


def main():
    ns = runpy.run_path(str(Path(__file__).with_name('write_credit_feedback_geometry.py')))
    jvp, vjp, bounds = (ns[name] for name in ('auxiliary_jvp', 'auxiliary_vjp', 'norm_bounds'))
    p = [.2, .3, .5]
    d, s = [[1., -2.], [3., .5], [-.7, 1.2]], [[.4, 1.], [-2., .3], [.8, -.4]]
    v, g = [[.2, -.1], [-.3, .4], [.1, .7]], [[.6, -.2], [.8, .9], [-.3, .5]]
    alpha = .37
    h, ht = jvp(p, d, s, v, alpha), vjp(p, d, s, g, alpha)
    dot = lambda a, b: math.fsum(x*y for ar, br in zip(a, b) for x, y in zip(ar, br))
    assert math.isclose(dot(h, g), dot(v, ht), rel_tol=1e-13, abs_tol=1e-13)
    # FD of a frozen-anchor extension. Re-evaluating detach at each point
    # would give an identically zero function, not its custom derivative.
    scores = [math.log(x) for x in p]
    direction = [math.fsum(x*y for x, y in zip(row, vi)) for row, vi in zip(s, v)]
    def anchored(t):
        z = [x+t*y for x, y in zip(scores, direction)]
        exp = [math.exp(x-max(z)) for x in z]
        pi = [x/math.fsum(exp) for x in exp]
        return [[alpha*(pp-baseline)*x for x in row] for pp, baseline, row in zip(pi, p, d)]
    epsilon = 1e-5
    plus, minus = anchored(epsilon), anchored(-epsilon)
    error = max(abs((a-b)/(2*epsilon)-c) for pr, mr, hr in zip(plus, minus, h)
                for a, b, c in zip(pr, mr, hr))
    assert error < 1e-10
    # Dense columns agree and yield exactly the block-orthogonal Frobenius formula.
    columns = []
    for slot in range(3):
        for feature in range(2):
            basis = [[0., 0.] for _ in p]
            basis[slot][feature] = 1.
            columns.append(jvp(p, d, s, basis, alpha))
    b = bounds(p, d, s, alpha)
    dense_frobenius = math.sqrt(math.fsum(dot(col, col) for col in columns))
    assert math.isclose(dense_frobenius, b['auxiliary_frobenius_norm'], rel_tol=1e-13)
    assert math.sqrt(dot(h, h)) <= b['auxiliary_spectral_upper_bound']*math.sqrt(dot(v, v))+1e-13
    assert jvp(p, d, [[0., 0.] for _ in p], v) == [[0., 0.] for _ in p]
    assert jvp([1.], [[2., 3.]], [[4., 5.]], [[1., 1.]]) == [[0., 0.]]
    assert jvp(p, d, s, v, alpha=0.) == [[0., 0.] for _ in p]
    assert ns['choice_jvp'](p, [2., 2., 2.]) == [0., 0., 0.]
    witness = ns['witnesses']()
    w = witness['bounded_write_and_selected_decay']
    assert w['auxiliary']['auxiliary_spectral_norm_if_pool_at_most_two'] == .5
    assert b['auxiliary_spectral_norm_if_pool_at_most_two'] is None
    assert math.isclose(w['largest_eigenvalue'], 1.3535533905932737)
    assert math.isclose(w['repeated_fixed_matrix_16_gain'], w['largest_eigenvalue']**16, rel_tol=1e-13)
    # Every positive scalar has gain >1 in this family, even though writes are bounded.
    for c in (1e-8, .001, .1, 1.):
        largest = (.5+1+2*c+math.sqrt(.5**2+4*c*c))/2
        assert largest > 1.
    assert witness['mean_over_winners']['largest_eigenvalue'] == 1.25
    two = bounds([.3, .7], [[1., 2.], [-3., .5]], [[-.2, .7], [1.5, -.4]], alpha)
    expected = alpha*.3*.7*math.sqrt(1+4+9+.25)*math.sqrt(.04+.49+2.25+.16)
    assert math.isclose(two['auxiliary_spectral_norm_if_pool_at_most_two'], expected, rel_tol=1e-13)
    n = witness['nonnormal']['auxiliary']
    assert [[math.fsum(n[i][k]*n[k][j] for k in range(2)) for j in range(2)] for i in range(2)] == [[0., 0.], [0., 0.]]
    assert math.isclose(witness['nonnormal']['fixed_matrix_20_direction_gain'], math.sqrt(101))
    for bad in ([.2, .2], [-.1, 1.1], [float('nan'), 1.]):
        try:
            bounds(bad, [[1.], [1.]], [[1.], [1.]])
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid probabilities admitted')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(dict(standard_library_contracts='passed', anchored_fd_max_error=error,
               adjoint_identity=True, block_frobenius_identity=True,
               bounded_write_does_not_bound_effective_gain=True,
               native_measurements='pending'))


if __name__ == '__main__':
    main()
