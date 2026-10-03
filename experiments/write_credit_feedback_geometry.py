"""Local write-surrogate tangent algebra, using only the standard library.

These operators hold incoming content, clocks, gates, winner and detached
write coefficients fixed. They are not complete native recurrent Jacobians.
No model import, optimizer, training or benchmark runs here.
"""
import math


def _validate(probabilities, deltas, score_reads):
    p = list(probabilities)
    d, s = [list(row) for row in deltas], [list(row) for row in score_reads]
    if not p or len(d) != len(p) or len(s) != len(p):
        raise ValueError('One delta and score-read vector per probability required')
    width = len(d[0])
    if not width or any(len(row) != width for row in d+s):
        raise ValueError('Equal nonempty slot-vector dimensions required')
    if not all(math.isfinite(x) for x in p+[x for row in d+s for x in row]):
        raise ValueError('Finite represented coefficients required')
    if min(p) < 0 or abs(math.fsum(p)-1) > 1e-12:
        raise ValueError('Nonnegative normalized probabilities required')
    return p, d, s, width


def choice_jvp(probabilities, vector):
    """Jpi v = pi * (v - E_pi v); shared score shifts vanish."""
    mean = math.fsum(p*v for p, v in zip(probabilities, vector))
    return [p*(v-mean) for p, v in zip(probabilities, vector)]


def auxiliary_jvp(probabilities, deltas, score_reads, perturbation, alpha=1.):
    """Apply H = alpha B Jpi S without an (U*P)-squared matrix."""
    p, d, s, width = _validate(probabilities, deltas, score_reads)
    v = [list(row) for row in perturbation]
    if len(v) != len(p) or any(len(row) != width for row in v):
        raise ValueError('One matching perturbation vector per slot required')
    if not math.isfinite(alpha) or not all(math.isfinite(x) for row in v for x in row):
        raise ValueError('Finite alpha and perturbation required')
    scores = [math.fsum(a*b for a, b in zip(read, row)) for read, row in zip(s, v)]
    choice = choice_jvp(p, scores)
    return [[alpha*c*x for x in row] for c, row in zip(choice, d)]


def auxiliary_vjp(probabilities, deltas, score_reads, cotangent, alpha=1.):
    """Apply H^T = alpha S^T Jpi B^T; slot dimensions may differ from U."""
    p, d, s, width = _validate(probabilities, deltas, score_reads)
    g = [list(row) for row in cotangent]
    if len(g) != len(p) or any(len(row) != width for row in g):
        raise ValueError('One matching cotangent vector per slot required')
    if not math.isfinite(alpha) or not all(math.isfinite(x) for row in g for x in row):
        raise ValueError('Finite alpha and cotangent required')
    coefficients = [math.fsum(a*b for a, b in zip(delta, row)) for delta, row in zip(d, g)]
    choice = choice_jvp(p, coefficients)
    return [[alpha*c*x for x in row] for c, row in zip(choice, s)]


def norm_bounds(probabilities, deltas, score_reads, alpha=1.):
    """Analytic ideal-arithmetic upper bounds, not estimated native gains.

The exact Frobenius norm and a softmax bound bracket possible local
amplification from above. Neither is a stability certificate for a full
    history. Score reads must already include the sqrt(P) and clamp derivative.
    Python floats evaluate formulas; these are not rounding-certified intervals.
"""
    p, d, s, _ = _validate(probabilities, deltas, score_reads)
    if not math.isfinite(alpha):
        raise ValueError('Finite alpha required')
    dn = [math.sqrt(math.fsum(x*x for x in row)) for row in d]
    sn = [math.sqrt(math.fsum(x*x for x in row)) for row in s]
    small = [[alpha*dn[i]*p[i]*((1. if i == j else 0.)-p[j])*sn[j]
              for j in range(len(p))] for i in range(len(p))]
    frobenius = math.sqrt(math.fsum(x*x for row in small for x in row))
    # ||Jpi||_2 <= min(1/2, max_i 2*pi_i*(1-pi_i)) by symmetry/Gershgorin.
    softmax_bound = min(.5, max(2*x*(1-x) for x in p))
    product = abs(alpha)*max(dn)*softmax_bound*max(sn)
    # ||A||_2 <= sqrt(||A||_1 ||A||_infinity), here on the reduced U*U matrix.
    rows = max(math.fsum(abs(x) for x in row) for row in small)
    columns = max(math.fsum(abs(row[j]) for row in small) for j in range(len(p)))
    induced = math.sqrt(rows*columns)
    values = dn+sn+[frobenius, product, induced]
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Feedback bound arithmetic overflow')
    return dict(delta_norms=dn, score_read_norms=sn,
                reduced_operator=small, auxiliary_frobenius_norm=frobenius,
                auxiliary_spectral_norm_if_pool_at_most_two=frobenius if len(p) <= 2 else None,
                auxiliary_spectral_upper_bound=min(frobenius, product, induced),
                softmax_product_upper_bound=product,
                reduced_induced_upper_bound=induced,
                scope='Partial fixed-input/clock/winner ideal surrogate; excludes other recurrent paths')


def witnesses():
    """Constructed algebraic examples, never observed trained trajectories."""
    p, d = [.5, .5], [[1.], [1.]]
    k = [[1.], [1.]]
    c, a = .25, .5
    largest = (a+1+2*c+math.sqrt((1-a)**2+4*c*c))/2
    # Symmetric positive-definite fixed local matrix: transpose gain is same.
    j = [[a+c, -c], [-c, 1+c]]
    v = [-c, largest-(a+c)]
    length = math.hypot(*v)
    v = [x/length for x in v]
    out = list(v)
    for _ in range(16):
        out = [math.fsum(x*y for x, y in zip(row, out)) for row in j]
    # Opposite score reads make H nilpotent: H^2=0, J=I+H has eigenvalues 1.
    nonnormal = auxiliary_jvp(p, d, [[1.], [-1.]], [[1/math.sqrt(2)], [1/math.sqrt(2)]])
    hv = [row[0] for row in nonnormal]
    x = [1/math.sqrt(2)+20*h for h in hv]
    return dict(status='constructed standard-library identities; no native run',
                bounded_write_and_selected_decay=dict(selected_decay=a, written_norms=[1., 1.],
                    physical_partial_norm=1., auxiliary=norm_bounds(p, d, k),
                    effective_partial_matrix=j, largest_eigenvalue=largest,
                    repeated_fixed_matrix_16_gain=math.hypot(*out),
                    scope='Frozen local operator repeated as a witness; no native trajectory asserted'),
                mean_over_winners=dict(matrix=[[1., -.25], [-.25, 1.]], largest_eigenvalue=1.25,
                    scope='Equal conditional winner average with all other local coefficients fixed'),
                nonnormal=dict(auxiliary=[[.25, .25], [-.25, -.25]], eigenvalues_I_plus_H=[1., 1.],
                    fixed_matrix_20_direction_gain=math.hypot(*x),
                    scope='Nilpotent auxiliary, possible transient amplification; no trained trajectory'))


if __name__ == '__main__':
    import json
    print(json.dumps(witnesses(), indent=2, allow_nan=False))
