# Message credit, private writes and the new language diagnosis

## Scope and current decision

Concurrent e096ca6/305da08 and theory §413 add message-linearized routing
credit to the segment-batched native language recipe. Preserve their v4
queue and recorded predictions. The positive completed p32/D4 score around
2.507 bpc is a different width/depth/initialization/work configuration from
p16/D8 around 2.719. Neither this difference nor the new surrogate's
implementation establishes a main causal explanation for all depth failures.

The source observation is precise: the earlier recipe supplied payload
credit to realized messages and common first-time clock credit, without
an explicit alternative-value categorical term. Its clocks and score
parameters still learned, and clock gradients can change relative rates.
"No choice-utility term" is more accurate than "routing weights do not
learn." The new linear term repairs that omission for a particular local
approximation, without installing actual private-write suffix utilities.

No architectural replacement or long run is nominated here. The prioritized
integrated language programs remain independently owned. The prepared
note138 warm-Adam coverage comparison remains pending physical-host
admission. This note defines the next fidelity question if the new language
surrogate is insufficient, rather than crowding the queue with variants.

## What the installed surrogate differentiates

For one race, let scores be s, pi=softmax(s), candidate messages v_i and
the factual downstream message cotangent g_v. The zero-valued addition is

    z = sum_i (pi_i-stopgrad(pi_i))*stopgrad(v_i).

Its numerical forward is zero. Its score credit is

    dL/ds_i = pi_i * g_v^T*(v_i-vbar),
    vbar = sum_j pi_j*v_j.

It adds no losing-message content derivative. Its cotangent is evaluated
on the realized branch; notes78/89 already show that curvature and
winner-dependent errors can reverse a local linear route ranking. This
is a different issue from note139's optimizer-transformation bias.

Private write selection has an additional dependency. In the actual core,
winner i also determines which persistent slot receives m_new_i, which
timestamp changes, and which unseen bit becomes true. Later keys, values,
forgetting, clocks and decisions can depend on those changes. The installed
surrogate is added to the emitted value. It does not add an analogous
choice derivative at the private-memory/timestamp/seen updates. Existing
common-clock derivatives and factual value derivatives remain present.

## An exact addressed-memory counterexample

This is a constructed interface example, not a measured native fit. Both
legal routes emit the same message v. Slot0 starts at M0=0. Choosing route0
writes M0=1; choosing route1 leaves M0=0 and writes the other slot. A later
fixed read returns M0. Let the terminal loss be

    ell(M0)=log(1+exp(-M0)),
    Q0=log(1+exp(-1)), Q1=log(2).

With the first time fixed and pi0+pi1=1, exact conditional choice credit is

    d E[ell]/d s0 = pi0*pi1*(Q0-Q1) < 0.

The message-linearized term is identically zero because v0=v1. This failure
does not require any approximation about message-loss curvature: the two
messages really are identical. The difference is wholly in the retained
write. A useful learned address therefore needs credit for future state
utility as well as immediate delivered-value utility.

The construction does not prove that this gap dominates real language;
native messages and writes are often correlated, so message credit can
indirectly improve addresses. It does prove that the message-only operator
cannot generally represent the required alternative-state derivative.
`check_credit_geometry.py` verifies the nonzero exact score derivative by
finite differences while the identical-message surrogate is zero, and
checks the state-linearized sign at both factual outcomes. These are
static scalar identities; no integrated native gradient contract is claimed.

## A retained-mechanism local extension, with explicit limits

Before implementing any extension, distinguish a cheap state linearization
from actual forced-write replay. For a fixed history, fixed first time and
fixed future topology, define deltaM_i=m_new_i-M_before_i. If g_Mi is the
factual cotangent of the post-write slot, the linear alternative coefficient
can include

    a_i = g_v^T*v_i + g_Mi^T*deltaM_i + g_Ai*deltaA_i,
    categorical_score_credit_i = pi_i*(a_i-sum_j pi_j*a_j).

The memory part could be implemented with the zero-valued addition

    z_Mi=(pi_i-stopgrad(pi_i))*stopgrad(deltaM_i)

at the already hard-updated slot. The timestamp part has an analogous
fixed-topology expression. Hard boolean seen/readiness changes do not have
ordinary continuous derivatives; they must not be declared covered by this
formula. New future route decisions and finite loss curvature also remain
outside it. Exact suffix replay naturally observes those discrete changes.

In the constructed example, a state linearization supplies a nonzero
negative coefficient because the later read has nonzero memory sensitivity;
its magnitude still differs from the finite logistic-loss difference.
For a linear terminal loss it would recover the exact conditional gradient.
This separates missing state paths from the distinct local-curvature issue.

If tried, this operator must retain the same hard messages/writes/times,
private memory, key/value separation and physical transport in inference.
It would add training-only score VJPs, with no extra losing-content credit.
Its work includes all auxiliary reads/backward/optimizer effects. It is
an unimplemented candidate, not a substitute for the exact teacher or a
new core mechanism already validated by the v4 message-only contracts.

Required admission before a fit: exact zero-valued forward/state/RNG
nesting; every-parameter identity against explicit fixed-topology scalar
objectives; no unintended branch-content credit; causal timestamp/seen
semantics; real Adam recovery; complete work; and integrated tiny examples
with equal-message/different-write branches. Then compare message-only,
state-linearized and actual forced-write utilities on declared FIT contexts
of the SAME source-bound trained model. Signs, parameter-space alignment
and finite prediction changes must all be retained. New language states
need actual saved weights, not a reconstructed model inferred from scores.

## Interpretation and resource boundaries for the concurrent diagnostics

- Pool1 removes alternative-slot routing and dormant slot capacity. It is
  a useful diagnostic control, not proof of learned sparse memory access.
- The proposed argmax diagnostic sets all clock noise to1, changing first
  times to1/max(rate). Its own source labels this delay change. A score
  difference cannot isolate a route-policy effect; preserving the sampled
  first time is required for a separate identity-only intervention.
- A probability mixture has NLL no greater than the average NLL of its
  constituent seeds, by Jensen. It can be worse than the first individual
  seed. Save each component's NLL and the mean before claiming a Jensen gap.
  All extra inference passes must be charged.
- The new diagnostics' mean_pi_per_unit records expected probability mass,
  not observed winner occupancy. The docstring promises winner usage but
  the current body does not count winners. Actual winner counts are needed
  before attributing slot specialization or fragmentation to this statistic.
- Similar aggregate T128/T256 scores do not establish that context beyond
  64 characters is unused. Longer-context benefit can cancel with other
  window/position effects. A matched-target history intervention is needed.
- The eager/compiled implementation computes all candidate keys and
  proposals before gathering the winner. Selected writes and deliveries
  are sparse; current candidate-value computation is wider. Charge actual
  executed operations and traffic, rather than treating selected-weight
  counts as measured total work or declaring losing candidates free.

The random-write/reread probability 1/U under independent uniform choices is
a useful toy baseline, not a general information-capacity bound. Native
first-time rates depend on all scored memory keys; that clock path can
carry information even about nonselected slots. Neither a 1/U heuristic
nor a poor restricted-teacher fit proves a mathematical barrier to the
complete temporal/addressed substrate. Keep supported gains and equally
demanding negative controls, with the learning mechanism and cost explicit.

## 3 October, 09:20 UTC: shared implementation and measurement update

The historical "unimplemented" statement above applied before8bcea13.
That shared change implements the memory-content part as `linear_rw` in
both eager and compiled paths, adding `(pi-stopgrad(pi))*stopgrad(m_new-m)`
to the already hard-updated slot, masked by active lanes. It retains the
same forward writes and delays and extends the v4 contracts/queue. Timestamp
and boolean-seen changes, finite return curvature and future routing
topology remain outside this local surrogate; do not equate it with actual
suffix replay. The current v4 arms remain the integrated priority.

Completed50K diagnostics now show sharp trained probability distributions,
so near-uniform trained routing is no longer a supported description.
They still record probability mass rather than actual winner occupancy,
and their greedy policy changes clocks. [141](141_streaming_routing_and_horizon_measurements.md)
prepares a source-bound, bounded streaming measurement of actual decisions,
post-clamp common-clock sensitivity, identity-only and unit-clock argmax
policies and the correct mean-seed mixture Jensen gap. Its native contracts
and forward job remain pending physical-host admission. It also derives
why unit-forget base half-lives do not establish the actual context horizon.
