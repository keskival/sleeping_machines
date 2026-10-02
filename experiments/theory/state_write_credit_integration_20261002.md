# Addressed-state write credit: integrated comparison

This note complements §§373–375 and the AWS factorial audit. It introduces a
separately named **local surrogate**, retaining the frozen split-event forward.

## SW1. Failure and retained mechanisms

Twelve fixed-time/noise checkpoint probes found mean absolute commit effect
.01269 versus value-linearization residual.000288. This is not an additive
attribution or expected-gradient estimate. Two probes had opposed delivery-only
and full-write loss directions. Message credit alone can miss where a route
stores information that later events use. Equal delivered messages and equal
proposed state contents can still have unequal utility at distinct addresses.

StateCreditEventHeads retains independent query/key projections, computational
race delays, protected temporal state when selected, shared/private receiver
rules, incoming-content/state mixing, eight layers, hard selection and the
original counterfactual message/clock teacher. It adds an addressed memory and
arrival-time adjoint term during training. It removes no forward mechanism.

## SW2. Boundary utility and exactness boundary

For route i, write delta Δm_i=m'_i−m_i, arrival delta Δa_i=a−a_i, and future
adjoints g_i,h_i, define U_i=g_i·Δm_i+h_iΔa_i. With rate λ_i and sampled minimum
T, the extra conserved boundary teacher is

    b_i=λ_i T (U_i−mean(U));  g_scores=b−e_W sum(b).

For fixed prefix-measurable linear utilities, integrating E[T]=1/sum(λ)
and W~Categorical(p) gives E[g_scores]=p⊙(U−p·U), the exact categorical utility
gradient. The unit test integrates/enumerates this identity analytically.
Actual future adjoints depend on the realized winner and downstream dynamics.
They give a local Taylor surrogate, **not** the independent control variate
required by the exact joint-race residual kernel. For a smooth future loss with
Hessian norm ≤M, one branch Taylor remainder is at most M‖Δstate‖²/2;
hard downstream route, state birth and validity boundaries need separate credit.
No arbitrary whole-sequence unbiasedness follows.

The original clock interior derivative is retained once. This variant does not
add the joint likelihood estimator on top of it. Future adjoints include
addressed write times, but not exact replay of every downstream scheduling jump.

## SW3. Implementation and charged resources

A custom route returns the actual selected value/delay, plus the whole candidate
state stack with only the winning write applied. Training-only views retain
identity paths for losing addresses, allowing later events to supply their
adjoints. Physical memory and arrival dictionaries still commit only the winner.
Unwritten zero states get virtual clocks for this smooth training extension;
that is not a proof of correct discrete birth credit. Clearing a source clears
both physical and auxiliary state. Evaluation uses the original parent directly.

Extra local dot products, stack/clone operations, losing-address reads and
backward graph work are traced. Logical auxiliary bytes may overlap physical
storage; report them separately from whole-process RSS and graph storage.
Winner-only inference has no auxiliary dictionary. Candidate scoring and all
losing proposals still cost work. Full fitting charges forward/loss, backward,
gradient normalization/clipping and Adam. Numerical prerequisite steps are
separate from fitted-model work, as for the frozen parent driver.

## SW4. Frozen pilot, not promotion by assumption

Unique local plan `queue/local_state_credit_boundary_20261002T015000Z.json`:
S4, private rules, P0, H2, d8, L8, pool2, seed6;128 fit queries/pass, four
passes,256 dev queries; same observed streams/noise and U64/lr.003 as the
completed AWS split screen. A contemporaneous strength0 baseline and strength1
fit isolate added write credit. No extra strength sweep or larger run admitted.

Prerequisites: actual-task full optimizer/checkpoint recovery; zero-credit exact
parent gradients; train/eval bitwise forward/RNG equality; all head/query credit;
winner-only physical commits; shared/P2 paired-timing contracts; then separate
baseline/teacher accounting smokes. Any unsupported floating operator, changed
source, nonfinite result, failed contract or smoke RSS >85% cap stops admission.

Exploratory follow-up gate: ≥5pp development accuracy gain and ≥.02NLL gain,
whole fitting work ratio ≤2. Both fits complete before comparison. Same data,
selected activity and units; report all outcomes. Development populations were
used in prior architecture screens, so independent seeds/fresh confirmation
are required before a benchmark advantage. No long training authorized by this
pilot's gate alone. Compare against the frozen parent and the separate
statistic-assisted direction; these address distinct information/credit gaps.
