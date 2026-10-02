# Terminal curvature, outcome credit and the next controlled fit

The native DVS encoder learns useful features, but its completed quality and
runtime trail strong full and compact kernels. Packet-scale clock initialization
does not close the gap. The admitted batching path preserves the integrated
architecture and passes state/every-parameter gradients, actual recovery and
full/partial-window accounting. The next intervention changes terminal learning
credit only; it keeps computation through time, hard key races, persistent sparse
state, separate values and optional losing-value learning. Inference is unchanged.

## A precise source of biased policy credit

Fix an entering prefix, candidate values and the downstream computation at fixed
arrival times. Scores s produce rates lambda_i=exp(s_i), total Lambda and
probabilities pi_i=lambda_i/Lambda. Independent exponential clocks give a winner
W independent of the first time T, with E[T]=1/Lambda. Let vbar be the unweighted
candidate mean, ell_i=ell(v_i), and a_i=gradient_v ell(v_i).

The implemented TemporalRoute value teacher, excluding the separate timing
derivative, is

    G_i = lambda_i T a_W.(v_i-vbar)
          - 1[W=i] T sum_j lambda_j a_W.(v_j-vbar).

Consequently its conditional mean is

    E[G_i] = pi_i sum_w pi_w
             [a_w.(v_i-vbar) - a_i.(v_w-vbar)].

The actual expected outcome loss R=sum_i pi_i ell_i instead has derivative

    dR/ds_i = pi_i (ell_i-R).

These agree for affine downstream loss. With two candidates, Delta=v_1-v_0,
the implemented mean reduces exactly to

    E[G_1] = pi_0 pi_1 Delta.(a_0+a_1)/2,
    dR/ds_1 = pi_0 pi_1 (ell_1-ell_0).

Thus the policy teacher substitutes a trapezoidal approximation to the integral
of the downstream gradient for the actual endpoint loss difference. This is a
property of the estimator, not an autograd implementation error. Gradient scale
normalization cannot restore the missing nonlinear loss differences.

## Convex cross-entropy can reverse the direction

Take one scalar delivered value x, three affine logits

    z(x) = [0, x-3, 1-3x], target class 0,
    ell(x) = log(1 + exp(x-3) + exp(1-3x)),
    v_0=0, v_1=5.

This is convex cross-entropy with an ordinary linear head. The loss at5 is
larger than at0, but ell'(0)+ell'(5) is negative. The true policy gradient
decreases the probability of5; the mean local teacher increases it. The
counterexample concerns a fixed terminal state and does not imply every native
update has the wrong direction, nor prove that this bias caused the observed
real-stream quality gap. A numerical contract will enumerate sufficient
statistics (winner, mean first time) through the actual reference and batched
backward implementations and check a small policy step against exact risk.

For f(t)=ell(v_0+t Delta), the trapezoidal error has magnitude at most
sup_t |f'''(t)|/12 on t in[0,1]. The corresponding score-gradient bias is bounded
by pi_0 pi_1 times that quantity. Small endpoint separation/curvature makes the
teacher useful; neither convexity alone nor a larger gradient guarantees it.
In logit coordinates cross-entropy has Hessian diag(p)-pp^T, operator norm at
most1/2 by its row absolute-sum bound2p_i(1-p_i)<=1/2. Its Taylor remainder lies
between0 and ||delta_z||^2/4. This logit bound is not a bound for a nonlinear
value-to-logit decoder without its Jacobian/curvature factors.

## What exact terminal content risk repairs

For the two final-query heads, conditional on prefix and sampled first times,

    R = sum_ij pi_1(i) pi_2(j) ell_ij,
    dR/ds_1(i) = pi_1(i) [sum_j pi_2(j) ell_ij - R].

Enumerating actual aligned endpoint losses retains their interactions and
curvature. Ordinary differentiation gives weighted losing-value and downstream
parameter derivatives. Winner-only payload gradients are themselves unbiased
Monte Carlo derivatives when policies/prefix/times are fixed; their sparsity is
not intrinsically a bug. Enumeration removes conditional outcome variance for
that component, by the law of total variance, while changing the biased local
policy teacher into the exact conditional content derivative.

Earlier routing remains a local surrogate; first-time derivatives still flow
through sampled winners and continuous causal joins. We do not claim minimum
variance timing credit, exact whole-sequence gradients, or counterfactual earlier
state commits. Existing common per-pass race draws are correlated across clips;
batching preserves them and does not guarantee a1/B gradient-variance decrease.
Pair head/loss evaluation, weights and backward are charged fitting work.

## Predeclared integrated comparison

Use the existing batched driver p16/L2/H2/pool2, first256 fitting gestures and all
192 subject-disjoint development gestures, seed6, four fixed passes/U16/lr.003.
Both arms have identical initial parameters, observed packets, per-pass noise,
order, optimizer and minimum-dev-NLL selection. Only terminal local versus pair
credit changes. Every pass and full fitting/resource estimate is retained.
Unique one-job queues and source hashes are in
`../queue/local_dvs_batched_credit_pilot_20261002T164700Z.json`.

Admission to an unchanged second fitted seed requires at least.02 lower dev NLL,
at most1 percentage point accuracy decline, at most1.10 times whole fitting
work and peak RSS below900000KiB. A failed gate stops this credit campaign;
do not extend passes or retune on dev. This is a mechanism comparison with less
data than the saved984-fit kernel controls, not practical superiority. Full-data
quality/resource confirmation and strong controls at any new storage budget are
required before a claim of advantage. No official-test labels are accessed.

## A conventional streaming control closes a tempting false advantage

An RBF prototype can accumulate squared distance causally:

    ||x-c_j||^2 = sum_t ||x_t||^2 + ||c_j||^2 - 2 sum_t x_t.c_jt.

The same fit-only per-coordinate log transform can run at packet closure. Keep
one common norm and one inner product per prototype; at the query compute the
RBF and existing decoder. A33-prototype control needs34 floating accumulators,
not a retained640-coordinate observation vector. Prototype/transform parameters,
packet work and final exponentials remain paid. This is an exact mathematical
streaming realization in real arithmetic, not yet an implementation/timing
result. It blocks a native storage or streaming-latency claim based solely on
comparing native state with a conventional concatenated-prefix buffer.

The resource opportunity must come from better learned sufficient state,
useful temporal dynamics, sparse value delivery or quality per total work.
Reusing already-paid candidate values for better credit is one testable route;
the present note supplies a reason to test it rather than a predicted win.
