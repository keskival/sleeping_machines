# Race-noise coupling, effective batches and fresh update randomness

2 October2026. A bounded diagnostic-first follow-up after unchanged choice and
evolution-offset confirmation failures. Read notes87–92 for their exact scopes.
The concrete failure hypothesis is correlated training noise/exposure, rather
than a new architecture or another credit-rule substitution. Native DVS fitting
uses one identical race-noise stream for all clips and all optimizer windows in
an epoch. The batched kernel preserves that old serial-driver contract exactly;
this is not a vectorization regression. Four-pass fits therefore use four
whole-history draw streams repeatedly while weights change between windows.
All clips still have their own observed features, physical times and memory.

## Fixed-weight covariance: what the diagnostic can decide

Let g_j(Z) be the full native parameter-gradient estimator for fitting clip j at
fixed weights under the same whole-history draw stream Z. For B clips,

    Var_trace(mean_j g_j(Z))
      = (sum_j Var_trace(g_j) + 2 sum_(j<k) E[(g_j-Eg_j) dot (g_k-Eg_k)]) / B^2.

With independent streams Z_j and the same marginal laws, the cross term is
zero. Positive cross-covariance prevents ordinary batch noise averaging;
negative cross-covariance can make shared noise beneficial. Independence is
therefore not automatically a variance improvement. With identical clip
functions, shared-noise batch variance equals single-clip variance whereas
independent-noise variance is smaller by B. With an equicorrelation model it
scales as sigma^2*(1+(B-1)rho)/B, giving effective batch size
B/(1+(B-1)rho) when the scalar model applies. Gradients here include native
scoped route teachers, not a claimed exact whole expected-risk gradient.

Audit the first four FITTING prefixes of both saved native local pilot seeds6/7,
32 independent whole-history draws per model, no optimizer. Compute per-clip
vectors under common streams. Estimate independent-batch trace covariance as
sum of their marginal sample covariances/B^2 from the SAME samples, and compare
with measured common-batch covariance. Their difference is exactly the summed
sample cross-covariance contribution, checked independently. Report parameter
groups as well as all parameters; a large decoder norm can obscure route noise.
No development-label tuning. Saved weights were development-selected earlier,
so this is a frozen selected-model diagnostic, not a new fitting-only selection
protocol or a representative corpus theorem. Finite draw estimates have no
claimed population confidence interval. Independent candidate admission needs
route-map common/independent variance ratio>=1.20 in BOTH models. Otherwise
stop that proposed cross-row change; do not relax the gate after seeing results.

## Freshness across optimizer updates is a separate issue

Standard score-estimator conditional-mean arguments assume the noise for the
current update is independent of the prior optimizer history, conditional on
current weights and the observed batch. Reusing the same draw within an epoch
lets later weights depend on that draw. Even a one-sample estimator that was
unbiased at fixed externally given weights need not keep that conditional
property in the adaptive training trajectory. A toy witness is g(theta,Z)=Z at
fixed theta with E[Z]=0; choose theta=Z and g(theta,Z)=theta*Z instead. Its
fixed-theta mean is zero but the reused adaptive estimator has E[Z^2], not zero.
This does not prove bias of each existing native teacher or that fresh noise
improves validation. Those teachers already have separate conditional limits.
It identifies a genuine assumption lost by reuse and a testable trajectory
change. Deterministic seeded fresh pseudo-random streams are reproducibility
mechanisms; they do not create a mathematical proof of finite-generator
independence. Current noise must never use a label, identity or clip index as
a seed/channel. An update ordinal from saved optimizer state is acceptable.

If the audit admits a candidate, implement two explicit fitting policies:
fresh_shared uses one new common stream each optimizer window;
fresh_independent uses a new stream per window and independent clip draws within
it. The latter contains both freshness and cross-row decorrelation, so the
fresh_shared control is necessary for attribution. Keep unchanged model,
physical races/transport, sparse actual writes, losing-value credit, native
surrogate and initialization. Inference/development continue using the original
fixed seed and winner-only state commit. No offset, replay or new baseline is
mixed into these first comparisons. This is a stochastic fitting-protocol
change, not substitution of dense computation for temporal architecture.

Require native single-clip nesting, explicit-tape serial/batched all-state/
all-parameter agreement, label/index independence, actual seeded continuous/
interrupted Adam/model/RNG/accounting recovery with partial windows, and complete
operator accounting before any fit. Charge extra random generation separately
from unit-special arithmetic; a discarded legacy common stream, if used by a
wrapper to preserve the frozen kernel, is still counted. Original inference
operators/work stay unchanged. No inference-only speed claim follows.

After both24-fit/eight-dev/two-pass smokes learn with complete work, declare two
256-fit/192-dev/four-pass seed6 pilots versus the SAVED native local control.
Minimum devNLL among schedules passing >=.02NLL improvement, <=1pp accuracy
decline, <=1.50 whole fitting work ratio and RSS<900000KiB selects at most one
unchanged seed7 confirmation. Seed7 must pass those same gates against its
saved control before any full984-fit stage. Preserve all curves, failed
schedules and controls; no extra epochs, third seed or post-result retuning.
Two-policy selection is exploratory; it does not itself prove mechanism
attribution or practical advantage. Whole fitting and per-presentation/inference
units remain common across all rows, with RNG/traffic/energy limitations explicit.

## Practitioner precedent and scope

[Flipout (Wen et al., ICLR2018)](https://arxiv.org/abs/1803.04386) studies shared
weight perturbations across minibatches and constructs pseudo-independent
perturbations to reduce gradient correlation. This is an analogy to the
covariance problem, not an implementation transplant: our noise determines
stochastic event times and memory winners, not sampled neural weights. Exact
covariance algebra above is independent of that paper. No claim that its
quality effects transfer to this substrate is made. Other-host tied/depth/
local-expectation fits and AWS dense comparisons remain separately owned;
this bounded local diagnostic does not duplicate them or alter frozen sources.
