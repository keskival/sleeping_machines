# Joint race credit: correctness and variance are separate gates

The factorial audit found substantial persistent-write effects in twelve fixed
probes. Note57 supplies a local joint state/time likelihood identity that can
credit those effects using actual suffix losses. Before treating that identity
as a useful learning rule, examine variance as well as unbiasedness. This note
derives a small conditional case; no integrated model is changed.

Let p_i=lambda_i/Lambda, tau=Lambda*T~Exp(1), and W~Categorical(p), independent
of tau. The local joint score vector is z=e_W-p*tau. Conditional on the prefix,
E[z]=0. For a prefix-only scalar baseline b, E[z(F-b)]=E[zF]. This is the
standard score-function baseline principle, specialized below to a race.

Since E[tau]=1 and E[tau²]=2,

    E[||z||² | W=i] = a_i = 1 - 2*p_i + 2*sum_j p_j²
    E[||z||²] = sum_i p_i*a_i = 1.

The second identity holds for every normalized probability vector, including
highly uneven races. Concentrating winner probabilities does not by itself
make the joint-score noise vanish: clock-rate credit remains.

For time-independent branch losses f_i, the exact local gradient is

    G_i = p_i*(f_i - sum_j p_j*f_j).

The sampled estimator z*(f_W-b) has total variance (trace covariance)

    V(b) = sum_i p_i*a_i*(f_i-b)² - ||G||².
    b_star = sum_i p_i*a_i*f_i.
    V(b)-V(b_star) = (b-b_star)².

Thus the ordinary mean loss is not always the scalar baseline that minimizes
total vector variance. For p=(.9,.1), f=(0,1), a=(.84,2.44): b_star=.244,
ordinary mean=.1, ||G||²=.0162. V(0)=.2278, V(.1)=.1890,
V(.244)=.168264. These values are derived expectations, not measurements of
the trained architecture. The optimum requires branch information; learning a
critic or enumerating branch costs consumes work and can introduce estimation
error. A detached current-suffix adjoint is not a prefix-independent baseline.

There is a distinct common-rate direction: sum_i z_i=1-tau. With the same
time-independent losses its expected gradient is zero, but the sampled common
component has variance sum_i p_i*(f_i-b)², minimized by the ordinary mean.
The best scalar baseline for common urgency can therefore differ from the best
baseline for the full score vector. Projecting away that component would erase
real urgency learning when future losses depend on time; it is not a general
variance fix.

Conditional winner enumeration at a sampled tau gives

    E[z*f_W | tau] = p*f - p*tau*sum_i p_i*f_i.

It removes winner noise but retains clock noise. In this restricted
time-independent case, integrating tau analytically yields G with zero sampling
variance. Actual stateful suffix losses vary with selected time and subsequent
routes, so one cannot obtain that guarantee by pretending their sampled values
are time independent. Note57's affine analytic surrogate plus sampled residual
is a controlled route toward this reduction, with independence and replay costs
explicit. An approximate residual remains biased.

For time-dependent F, the same scalar optimum is
E[||z||²*F]/E[||z||²]=E[||z||²*F], provided moments exist and b is independent
of the current race draw. This characterizes a target for prefix-conditioned
calibration; it does not provide an inexpensive exact critic.

Admission implications: preserve the unchanged teacher control, start with
zero-added-credit nesting, instrument the correction's conditional mean,
variance and common-rate component, then test an accounted small integrated
fit. Charge prefix traces, addressed-state discovery, critic fitting and any
suffix replay. Full joint credit may repair local fidelity and still lose on
total fitting work or quality because of variance. No supremacy follows from
the identity alone.

## Executable local contract

`sleeping_machines/joint_race_credit.py` now implements the analytic affine
surrogate plus either enumerated or importance-sampled actual branch residual.
It returns detached credit vectors; no training model installs this kernel.
Four numerical tests check polynomial suffix expectations against finite
differences, sampled-proposal averaging against branch enumeration, exact
affine zero-residual variance, and invalid inputs/support. The polynomial
contract retains nonzero common-rate credit. Losing-value and addressed-write
effects belong in supplied branch losses, rather than being approximated by
delivered payload equality.

This kernel proves neither inexpensive replay nor an unbiased full-network
teacher. It assumes prefix-independent surrogate coefficients and proposals;
the API cannot certify that statistical condition. A model integration must
keep direct derivatives separate, avoid counting existing rate credit twice,
and define suffix credit at each stochastic node. Checkpoint replay at fixed
sampled time tests branch costs but cannot supply the missing expectation over
time merely by naming the resulting vector an exact gradient.
