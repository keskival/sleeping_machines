# Pay for residual counterfactual uncertainty, retaining the full-credit mean

## Failure, retained mechanisms, and proposed distinction

Corrected deep language replay shows a positive online learning signal but pays
for long losing-route suffixes. Linear message credit is cheap and scales, yet
can miss utility of a persistent write (140/142). The unstable write surrogate
also shows why simply adding a recurrent straight-through path is insufficient
(145). We want cheaper estimation of actual suffix utility, not a substitution
of dense content for races or a declaration that local message credit is exact.

Retain the native temporal race, factual clocks, private addressed memory,
separate keys/values, deep persistent updates and full alternative support.
Change only how a training estimator obtains losing-route returns. Inference
is unchanged; a predictor of returns is training-only. This note proposes no
active source modification or admitted training job.

## Exact conditional estimator

Condition on factual inputs, states, race history and common random numbers
used for forced suffixes. Write the existing full-credit raw parameter gradient
as F + sum_j v_j, where F contains the unchanged factual/pathwise terms and
j indexes existing counterfactual contributions, including their normalization.
For example v_j = Q_j * b_j, with detached actual return Q_j and the appropriate
categorical score VJP b_j. The construction applies to the existing estimator;
it does not repair an incorrectly specified causal return or clock derivative.

Compute a cheap anchor h_j for EVERY admitted term. Anchor returns and sampling
probabilities are detached in the route auxiliary. They may depend on the fixed
history, current weights or past training, provided sampling random bits remain
fresh conditional on these quantities. With independent I_j~Bernoulli(p_j),
0<p_j<=1, define r_j=v_j-h_j and

    Ghat = F + sum_j h_j + sum_j I_j*r_j/p_j.

Conditional expectation is EXACTLY G_full, whatever the anchor quality. The
conditional covariance is sum_j (1/p_j-1)*r_j*r_j^T. There is no approximation
to the mean return from a finite critic: only estimator noise changes. This is
ordinary control-variate plus Horvitz-Thompson algebra, not a new statistical
principle. The contribution is its proposed integration with real causal
persistent-write replay and complete cost contracts.

Use h_j=Qhat_j*b_j, evaluated through one weighted-score backward without
materializing every parameter Jacobian. Do not backpropagate Qhat_j in this
auxiliary: that creates a different estimator. A separate supervised predictor
can learn actual replay returns, charging its fitting and optimizer work.
A scalar factual-winner baseline is a restricted anchor; a useful state-aware
predictor must represent changes in written memory, clocks and future reads,
not just delivered message differences. Otherwise its residual may remain large.

All routes remain eligible. Setting p_j=0 for a nonzero residual changes the
mean; hard learned pruning is not this construction. Adaptive stopping after
seeing selected current returns needs actual inclusion-probability analysis.
Choosing p from cheap history/previous residual statistics before fresh draws
is valid. An oracle p computed from every actual Q is only an analysis ceiling
and costs full replay; it is not free adaptive selection.

## Allocation and work condition

For fixed residuals and a fixed linear diagnostic map A, define
 a_j=||A*r_j||^2. Under additional expected replay budget C, minimize
 sum_j (1/p_j-1)*a_j subject to sum_j p_j*c_j<=C. Interior Lagrange calculus gives

    p_j = min(1, sqrt(a_j/(lambda*c_j))).

Choose lambda to satisfy the budget; require a positive support floor when a_j
is estimated. This extends137's coverage calculus to residuals rather than raw
returns. It cannot use unavailable current exact residuals for free. Candidate
scores, all-anchor prediction and backward, replay selection, suffix work,
critic updates and the main optimizer belong in total C_fixed + sum p_j*c_j.
Correlated replay selections need covariance cross terms and inclusion rules;
the independent-Bernoulli expression does not apply unchanged to fixed-k draws.

A=I measures raw-gradient error. A frozen clip/Adam Jacobian from139 measures a
local update diagnostic, not an exact expected Adam step. Unbiased gradients do
not imply unbiased nonlinear clipped/warm-Adam updates or improved heldout risk.
Race-history variability remains a separate floor even if conditional residual
sampling variance vanishes. Pair finite actual optimizer forks before quality
promotion, and include the anchor's cost in any variance-times-work comparison.

## Horizon telescoping as a second possible anchor

At a fixed forced route, define returns on nested causal suffix horizons with
THE SAME full-target normalization and common random numbers. Then
 Q_H=Q_h0 + sum_l (Q_hl-Q_h(l-1)). The same residual estimator can sample horizon
increments with positive probabilities. Prefix-cached execution may share work;
its measured incremental cost must replace independent suffix costs in the
allocation objective. Existing cached-prefix prototypes can be slower despite
fewer events, so no wall-time conclusion follows from this algebra.

Changing the denominator to the length of each suffix destroys this telescope.
Returns must preserve the original forced action, downstream intervention,
target weights and physical first-time law. Truncating the last increments
introduces bias; estimating them with positive support preserves the original
finite full-horizon mean. This offers a way to retain occasional long delayed
write credit while avoiding full long suffixes on every alternative, if residual
variance and implementation costs permit it.

## Executed algebra and next numerical gates

Pure-stdlib exhaustive three-term Bernoulli enumeration verifies every mean and
covariance entry and expected replay cost (max error1.78e-15). A constructed
anchor with10% residual has100x lower variance at identical replay probabilities;
an inaccurate anchor yields81x HIGHER variance. An exact anchor has zero residual
variance, but constructing it is not assumed cheap. These are analytical fixtures,
not trained-state noise reduction or learning results. Artifacts:
analysis/aws_residual_credit_algebra.py and
results/diagnostics/aws_residual_credit_algebra_20261003.json.

Priority remains the assigned AWS90M arms and original corrected full-replay
controls. After a free guarded diagnostic slot exists, first contract one actual
trained private/shared chunk: full estimator against exact enumeration of sampled
residual estimators, every parameter, identical factual state/clocks/RNG, charged
anchor/backward/Adam work and exact next partial recovery. Estimate residual
variance from prespecified FIT-only held-out predictor targets, including deep
persistent-write utilities; compare zero/factual/local-message/state-aware anchors.
Use finite warm-Adam forks and disjoint FIT anchors to reject noisy harmful
updates before a bounded integrated DEV-selected fit. No new test-driven sweep,
no training queue admitted from the scalar fixture alone.

## Calibrating an anchor without asserting that a critic helps

Let cheap candidate gradient contributions u_j be fixed at the conditioned
history. Use h_j=alpha*u_j and w_j=1/p_j-1. Conditional trace in any fixed
linear diagnostic A is a quadratic:

    V(alpha)=sum_j w_j ||A*v_j-alpha*A*u_j||^2.
    alpha_star = sum_j w_j <A*v_j,A*u_j> / sum_j w_j ||A*u_j||^2.

If the denominator vanishes, choose alpha=0. Otherwise the optimal reduction
is numerator^2/denominator; restricting alpha to a prespecified interval
projects the optimum onto that interval. This is an oracle fixed-state identity,
not a free policy: computing v_j already pays actual replay. FIT-only past
samples can estimate the moments, with proper inclusion weighting and a
positive support floor, but their estimated optimum need not help the next
history. Learning alpha from the same selected current returns and treating it
as fixed would invalidate the simple conditional proof. Freeze it before fresh
draws, or derive the dependence explicitly.

For uniform probabilities, lowering p from p0 to p1 increases the noise factor
from (1/p0-1) to (1/p1-1). Equal or lower conditional raw variance requires

    sum ||v-h||^2 / sum ||v||^2 <= (1/p0-1)/(1/p1-1).

For example reducing probability1/2 to1/4 needs residual squared norm at most
one third of the original. This is a measurable threshold, not a generic claim
that any learned predictor saves work. Let total fixed overhead increase from
C0 to C0+Ca and full residual shadow cost be Cs. Work decreases only if
Ca < (p0-p1)*Cs. Both variance and total-work gates must pass; alpha0 may still
incur anchor overhead unless the implementation actually skips that computation.

A prepared trained-state protocol is now source-bound to the earliest immutable
private full-replay checkpoint above1M targets, chosen by target count rather
than quality. analysis/aws_residual_credit_protocol.py verifies the checkpoint
SHA and records every-gradient/factual-state/RNG/causality/warm-Adam/recovery and
complete-resource gates in results/diagnostics/aws_residual_credit_protocol_20261003.json.
It only prepares metadata; no native execution or new training is claimed.
