"""Conditional categorical credit diagnostics; standard library only."""
import math


def fidelity(probabilities, utilities, coefficients):
    """Oracle local diagnostics, never a fitted/global calibration guarantee.

    The pi-whitened score-gradient error is Var_pi(Q-a). The best
    nonnegative scalar minimizes that metric; emitter pullbacks and
    clipping/Adam can change which metric predicts an actual update.
    """
    p, q, a = map(list, (probabilities, utilities, coefficients))
    if not p or len(p) != len(q) or len(p) != len(a):
        raise ValueError('Matching nonempty vectors required')
    if not all(math.isfinite(x) for x in p+q+a) or min(p) <= 0 or abs(sum(p)-1) > 1e-9:
        raise ValueError('Finite coefficients and positive normalized probabilities required')
    # Remove an exact represented baseline first. Constant coefficients must
    # not acquire a tiny artificial variance from weighted-mean roundoff.
    qdiff, adiff = [x-q[0] for x in q], [x-a[0] for x in a]
    qbar, abar = sum(x*y for x, y in zip(p, qdiff)), sum(x*y for x, y in zip(p, adiff))
    qc, ac = [x-qbar for x in qdiff], [x-abar for x in adiff]
    varq = sum(x*y*y for x, y in zip(p, qc))
    vara = sum(x*y*y for x, y in zip(p, ac))
    covariance = sum(x*y*z for x, y, z in zip(p, qc, ac))
    raw_error = sum(x*(y-z)**2 for x, y, z in zip(p, qc, ac))
    signed_scale = covariance/vara if vara else None
    scale = max(0., signed_scale) if signed_scale is not None else 0.
    scaled_error = sum(x*(y-scale*z)**2 for x, y, z in zip(p, qc, ac))
    values = [varq, vara, covariance, raw_error, scaled_error, scale]
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Diagnostic arithmetic overflow')
    return dict(exact_local_choice_gradient=[x*y for x, y in zip(p, qc)],
                approximate_local_choice_gradient=[x*y for x, y in zip(p, ac)],
                exact_pi_whitened_norm_squared=varq,
                approximate_pi_whitened_norm_squared=vara,
                pi_whitened_error_squared=raw_error,
                pi_metric_inner_product=covariance,
                pi_metric_cosine=covariance/math.sqrt(varq*vara) if varq and vara else None,
                oracle_signed_scale=signed_scale, oracle_nonnegative_scale=scale,
                oracle_scaled_error_squared=scaled_error,
                scope='Fixed-prefix/time/future-noise local score coordinates; oracle diagnostics, no training prescription')


def factorial(f00, f10, f01, f11, linear_value):
    """Factual, value-only, commit-only, both; hybrids are diagnostics."""
    if not all(math.isfinite(x) for x in (f00, f10, f01, f11, linear_value)):
        raise ValueError('Finite factorial losses and coefficient required')
    value, write = f10-f00, f01-f00
    interaction = f11-f10-f01+f00
    residual = f11-f00-linear_value
    reconstructed = value-linear_value+write+interaction
    if not math.isclose(residual, reconstructed, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('Factorial residual reconstruction failed')
    return dict(total_branch_loss_difference=f11-f00, delivered_value_effect=value,
                persistent_commit_effect=write, delivery_commit_interaction=interaction,
                local_delivered_value_linearization=linear_value,
                value_linearization_residual=value-linear_value,
                teacher_residual=residual, residual_reconstruction=reconstructed)


def addition_fidelity(probabilities, utilities, base, extra):
    """Calibrate an optional term against the BASE RESIDUAL, keeping base fixed."""
    base, extra, utilities = map(list, (base, extra, utilities))
    original = fidelity(probabilities, utilities, base)
    if len(extra) != len(base):
        raise ValueError('Matching additional coefficients required')
    projection = fidelity(probabilities, [q-a for q, a in zip(utilities, base)], extra)
    unit = fidelity(probabilities, utilities, [a+b for a, b in zip(base, extra)])
    return dict(base_error_squared=original['pi_whitened_error_squared'],
                unit_addition_error_squared=unit['pi_whitened_error_squared'],
                residual_extra_inner_product=projection['pi_metric_inner_product'],
                extra_pi_variance=projection['approximate_pi_whitened_norm_squared'],
                oracle_signed_alpha=projection['oracle_signed_scale'],
                oracle_nonnegative_alpha=projection['oracle_nonnegative_scale'],
                oracle_best_error_squared=projection['oracle_scaled_error_squared'],
                oracle_possible_local_error_reduction=original['pi_whitened_error_squared']-projection['oracle_scaled_error_squared'],
                scope='Keep successful value coefficient fixed; oracle residual projection only, no optimizer/quality guarantee')


def pooled_addition_fidelity(cases):
    """One shared oracle alpha across equally weighted local case metrics."""
    cases = list(cases)
    if not cases:
        raise ValueError('At least one declared case required')
    stats = [addition_fidelity(*case) for case in cases]
    covariance = sum(row['residual_extra_inner_product'] for row in stats)
    variance = sum(row['extra_pi_variance'] for row in stats)
    signed = covariance/variance if variance else None
    alpha = max(0., signed) if signed is not None else 0.
    error = sum(fidelity(p, q, [a+alpha*d for a, d in zip(base, extra)])['pi_whitened_error_squared']
                for p, q, base, extra in cases)
    return dict(cases=len(cases), shared_oracle_signed_alpha=signed, shared_oracle_nonnegative_alpha=alpha,
                base_local_error_sum=sum(row['base_error_squared'] for row in stats),
                unit_addition_local_error_sum=sum(row['unit_addition_error_squared'] for row in stats),
                shared_alpha_local_error_sum=error,
                separate_oracle_local_error_sum=sum(row['oracle_best_error_squared'] for row in stats),
                opposed_residual_cases=sum(row['residual_extra_inner_product'] < 0 for row in stats),
                per_case_oracle_signed_alphas=[row['oracle_signed_alpha'] for row in stats],
                scope='Sum of equally weighted conditional score metrics; reused-case oracle, no cross-site parameter metric or population/quality claim')
