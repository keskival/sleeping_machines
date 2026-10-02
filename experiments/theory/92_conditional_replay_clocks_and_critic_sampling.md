# Conditional replay clocks and sampling-safe critic corrections

2 October2026. A constructive correction to shared note59 §§402–403, preserving
its actual-write/full-suffix replay motivation, sparse temporal inference and
historical empirical candidates. The newly added external driver
`dvs_local_expectation_benchmark.py` forces alternative i at its individual
exponential arrival T_i, rather than the actual first time T. Its current tests
verify the algebra of the resulting surrogate and factual-winner reproduction;
neither property verifies the claimed expected-risk gradient. No external
model/queue/result is modified by this note or its one-race audit.

## Correct conditional variables

For an entering prefix with rates lambda_i=exp(s_i), Lambda=sum_i lambda_i,
independent exponential races give

    p(W=i,T=t) = lambda_i*exp(-Lambda*t),
    pi_i=lambda_i/Lambda,
    T | W=i ~ Exp(Lambda).

Winner and first time are independent at fixed scores. The winner's individual
arrival conditioned on winning is the first time, not its unconditioned
Exp(lambda_i) arrival. Fixing all current race exponentials fixes W; W cannot
then also be integrated with categorical probabilities pi. The conditional
calculation must specify which stochastic variable is being integrated.

For fixed entering prefix, factual first time T and independent future random
numbers, legal full-suffix losses F_i(T) must change the selected payload AND
its actual memory write while preserving that first time. Conditional choice
credit is pi_i*(F_i(T)-sum_j pi_j*F_j(T)). Smooth physical-time and earlier-state
credit remain separate. Resampled or forced individual candidate times are
legitimate combined interventions, but this categorical formula does not make
them exact conditional choice credit.

## An explicit direction-reversal witness without messages

Take two rates1 and3 and the loss F=T, independent of winner/content. The true
risk is1/Lambda, with score derivative[-.0625,-.1875]. Correct conditional
choice credit is zero. Forcing each alternative at its own arrival gives
E[T_i]=[1,1/3]; applying detached categorical credit to these returns has
expectation[+.125,-.125]. The existing factual interior derivative has
expectation[-.0625,-.1875] in this simple race. Adding that incorrect replay
choice correction yields[+.0625,-.3125], reversing the first score's direction.
This is an analytic two-route counterexample to the general unbiased claim,
not a statement about the final quality of an empirical DVS fit.

A factual forced-winner replay still reproduces the factual loss exactly: its
individual arrival is already the minimum. Thus that existing identity test
cannot detect the error for a losing alternative. The new contract probes the
actual replay wrapper with one time-recording event and the actual pathwise
race method: four seeds reproduce factual minimum and forced individual times,
with every losing alternative later than the original first time. Model logits
are constant in this wrapper; the gradient witness above is separately analytic,
not inferred from those constant logits or called an integrated training test.

## Two coherent remedies, with explicit credit boundaries

1. Use first-time-preserving actual-write replays for choice utility, and a
   separately derived clock estimator. Merely preserving T does not prove that
   the old winner-only interior time derivative plus categorical correction is
   exact in a general winner-dependent model. Notes81–85 preserve their scopes
   and failed clock/choice confirmations; do not silently rebrand them exact.
2. Factor the race as W~pi, T=E/Lambda with E~Exp(1) independent of W. A smooth
   pathwise time derivative then has dT/ds_i=-T*pi_i. Combine it with correctly
   conditional choice credit and deterministic branch derivatives, accounting
   for every future discrete node and timing boundary as required. This is a
   distributionally equivalent forward construction, but changes conditional
   parameter derivatives and requires state/every-parameter/recovery contracts
   before any integrated fit. It is not already implemented by the external
   driver and is not an admitted long campaign here.

Alternatively the full joint score is d log p(W,T)/ds_i=1[W=i]-lambda_i*T.
Conditionally averaging W at factual T gives

    pi_i*(F_i-R) + pi_i*(1-Lambda*T)*R,  R=sum_j pi_j*F_j.

The first is choice, the second common-clock credit. This expectation is exact
for this stochastic node's likelihood term under integrability and legal
conditioning, including discontinuous downstream outcomes. Deterministic
parameter derivatives at fixed sampled variables and all other stochastic
nodes still have to be included. If using the joint time likelihood, do not
also add a reparameterized derivative through that same sampled time. Choose
and derive the estimator consistently; arbitrary mixtures double-count or
miss terms. Continuous-score credit can be noisy and no quality benefit follows
from unbiasedness alone. The time-only witness integrates E[T]=1/Lambda and
E[T^2]=2/Lambda^2 to recover the true derivative exactly.

Common random numbers reduce variance only when the coupled outcomes are
positively correlated; they do not guarantee low variance universally. Pool2
enumeration is exact conditional averaging when the conditional law is correct,
not automatically ARM's specifically constructed antithetic estimator. k/R
race subsampling preserves a declared sum of node terms with proper inclusion
probabilities; it cannot correct an already biased node term.

## Critic control variates: freeze before the correction sample

For fixed per-site vectors g_r, uniformly sampled k of R sites, and a critic
q_r fixed before that sample, the estimator

    sum_r q_r + (R/k)*sum_(r in S)(g_r-q_r)

has conditional expectation sum_r g_r for any fixed critic. Critic dependence
on the episode can be legitimate if the sampling is conditionally independent;
critic features/outputs used as loss returns must be detached for this direct
score term. Learned critics can use the expensive replay targets after the
current estimator, or use cross-fitting/independent sampling with the required
conditioning. Their training cost and all-site evaluation remain charged.

If q is trained on the SAME selected subset before applying the correction,
that cancellation is not automatic. Counterexample R2/k1, all true g_r=0,
and q_r=1[r selected]. Every estimator equals1-2=-1, although the true sum is
zero. The fixed-critic numerical contract recovers a nonzero true sum exactly;
the sample-adaptive counterexample retains its nonzero bias. In addition,
critic reweighting is unbiased for its BASE replay sum: if the base used the
wrong time law, no unbiased critic residual turns it into true expected-risk
credit. Correct law and critic sampling are separate contracts.

## Current decision

Record these corrections beside §§402–403 before interpreting their fits.
Preserve their empirical results as results of the declared combined-intervention
estimator. Other-host training ownership remains unchanged. Locally, no new
long replay/critic fit is admitted: first establish the conditional time law,
full legal memory-write semantics, clock/choice estimator consistency and
actual update/recovery/work accounting in an integrated model. The existing
offset/choice confirmation failures remain failures, not evidence that this
new mathematical correction alone guarantees improvement. This note advances
a necessary credit contract while preserving the research objective.
