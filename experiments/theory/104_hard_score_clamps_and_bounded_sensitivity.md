# Hard score saturation, available clock freedom and a bounded bridge

## Measured failure and scope

The completed theory103 conditional native Jacobian audit finds five of sixteen
selected trained-site cases with a score clamped to12. The winning preclamp
score derivative is zero; choice and clock sensitivities are collinear there.
The audit deliberately includes late sites and only two unused FIT examples,
so this is not a population failure rate or proof of the quality bottleneck.
All32 Jacobian pairs and initial controls remain preserved. In the broader
theory99 profile,4.77%/3.72% of candidate scalar scores reach a clamp in trained
seeds6/7, versus0 initially. Score occupancy alone does not measure lost utility.

This is an intended hard-bound derivative, not an incorrect autograd rule or
proof of a historical regression. Even correct deep counterfactual score credit
cannot change a locally constant clipped score directly. It can still change
an unclamped competitor and shared contents/history through other paths.
Near-certain choices also make categorical information small without any clamp.

## A rank obstruction and a retained bounded operator

If s0=clip(r0)=C with r0>C, and r1 is interior, then

    grad(s0-s1)=-grad(r1),
    grad(logsumexp(s))=pi1*grad(r1).

Their local conditional pullback has rank1. Separate preclamp clock biases do
not restore a blocked derivative. This does not establish loss interference:
the study has no target loss, utility or expected-risk gradient. Biases remain
tied over histories, and conditioning previous clocks differs from native
pathwise timing derivatives (notes85/91/102/103).

Candidate bridge, fixed alpha=.1 for diagnostics, C=12:

    f_alpha(r)=(1-alpha)*clip(r,-C,C)+alpha*r/sqrt(1+(r/C)^2).
    f'_alpha(r)=(1-alpha)*1[|r|<C]+alpha*(1+(r/C)^2)^(-3/2).

Alpha0 reproduces the original operator and every existing gradient. For
positive alpha, away from hard-bound kinks the derivative is strictly positive
at every finite mathematical input; the output stays in[-C,C], is monotone
and odd. No straight-through derivative or unbounded rate is substituted.
At r=20, positive bridge slope~.0136; a hard clamp has zero. Restored sensitivity
can still be small, and floating overflow/underflow at extreme inputs remains
a numerical issue. At the hard-bound kinks use the implementation convention;
no nonexistent two-sided derivative is claimed.

The emitter input remains actual incoming content/persistent memory; effective
scores continue to set both categorical races and common computational times.
Native bounded physical delays, one selected value/write, causal transport,
key/value separation and available capacity remain. Softening can change real
clock times and route identities; neither original predictions nor improved
quality is promised for positive alpha. This is distinct from101's frozen
temperature intervention, which retained the entering total rate.

## Bounded integrated diagnostic before any learning

Reuse exactly103's two disjoint producer-unseen FIT inputs, four fixed sites,
one noise history and four frozen encoder variants. Original cross-entropy is
used only for alpha0 gradient-nesting contracts; no quality/alpha selection,
optimizer or DEV/test. Verify primitive finite differences/gradcheck,
boundedness/oddness/monotonicity/alpha0 nesting and a saturated rank witness.

At native inference, read already-computed query/key-read outputs through
hooks and recompute the exact original preclamp dot product order. Compare
alpha0 original logits/ALLstate/end-RNG and training gradients before positive
mode is evaluated. The diagnostic recomputation is EXTRA scoring work, not
free candidate discovery or an optimized production implementation.

At each original conditioned history, replace only the entering score map for
Jacobian measurement and retain all previous physical times/winners. Report
hard versus bridge local metric spectra, score values, raw saturation and
actual clock-bias sensitivity. This local comparison is not a new full-prefix
quality result. Also run the positive bridge in the actual complete native
primal: original random-draw count, causal readiness and sparse writes must
remain; changed past clocks/routes/content are allowed. Targets cannot affect
forward computation. Positive training explicitly refuses until correct
native credit, optimizer/recovery/accounting contracts and a small comparison.

Every old producer fit, graph/VJP, duplicated score and bounded-map operation
remains paid. Total audit FLOPs/traffic/energy unknown, not zero. One safe
180s queue,1thread,3GiB virtual/1.25GiB RSS/8GiB host floor. No large fit or
training nomination solely from derivative restoration. Deeper matched replay
and growth-by-nesting stay with their existing owner queues.
