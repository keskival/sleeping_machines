# Protected state, shared processing and identifiable timing

Global §§354–357. Motivated by the completed small AWS pilots and local R2
fit in RESULTS_DESIGN_UPDATE_20261001.md. This is a separate experimental
branch; frozen native/reception models and their queues remain intact.

## 354. Separate invariance from sensitivity inside one receiver

The order label is invariant to gap stretching, but the fitted native state is
not. Conversely the timing label may change with elapsed time. One homogeneous
state geometry need not serve both objectives well. Extend §334's direct sum:

    m = (s,z),   E(delta)m = (s, exp(-Gamma delta) R(Omega delta) z).

The protected coordinates s are unchanged between events; temporal coordinates
z compute with delays. At an admitted arrival both can be written through
content-conditioned gates and learned mixing. A protected coordinate is not
permanent: event writes and subsequent learned maps can replace it.

For fixed nonnegative delta and diagonal nonnegative rates,
E(delta1)E(delta2)=E(delta1+delta2), ||E(delta)||<=1, and
d s_after/d s_before=I, d s_after/d delta=0. Temporal phase has the ordinary
rotation derivative, and rate has -delta times the evolved mode, within a
fixed route history. This certifies local silent retention, not invariance of
the whole predictor: temporal coordinates still affect scores, writes and
mixing. Hard route changes still need counterfactual boundary credit; the
implemented local teacher remains a surrogate, not an exact expected gradient.

Implement the same split in receiver memories AND stored head/context
transport. Protecting memory alone while rotating every delivered message
would leave an avoidable information-loss path. Keep payload width fixed;
reserve half its coordinate pairs. Remove unused protected rate/frequency
parameters rather than pretending they add learnable capacity. Zero protected
pairs must exactly nest the saved parent, including RNG, parameters and forward.

This is a candidate response to a measured gap failure, not a claim that all
content should be static. It retains time-based races, deep addressed state,
separate keys/values, content mixing, independent heads, losing-route teaching
and selected commits. Inference evolves only the temporal suffix; learning
still pays all gates, maps, candidate proposals, backward and Adam.

## 355. Separate private state from private learned rules

Current populations share a task, but each address has its own receiver maps.
At fixed T queries, a source has T/S supervised observations per pass. Growing
S therefore expands parameter capacity while reducing exposure per private
rule. It is unnecessary to duplicate every learned transition to keep streams
independent. Compare shared depth/head/pool maps with private addressed states.

For per-source receiver parameter count p, shared count q and embedding width D:

    private N = q + S*p + S*D;
    shared N = q + p + D.

Available addressed receiver-state slots remain S*L*H*P; selected commits L*H
and key scores L*H*P per event stay unchanged. Compute state updates from one
address's state only; sharing gradients across streams must not share their
contents. Optimizer work can shrink while persistent storage still scales
with occupied sources. Audit full fitting operators instead of assuming N
alone determines FLOPs. Shared parameters also couple optimization gradients;
they do not guarantee better quality or independent learning.

In the shared arm use one common learned source seed, avoiding an arbitrary
learned source-ID nuisance. With identical race noise, renaming addresses then
renaming stored state commutes with the forward transition. This is address
permutation equivariance, NOT invariance to reordering observations. Private
source embeddings are retained in the unmodified/private arm. Thus sharing
receiver maps and removing private embeddings change together; an embedding-only
control is required before attributing any improvement exclusively to map sharing.

Factorialize private/shared processing and zero/half protected channels first.
Prioritize S4 and S16; defer S64 until small contracts/smokes establish memory.
Use identical data, seeds, widths, depths and optimizer windows. Report changed
parameter count and full work; do not label different capacities equal models.

## 356. A timing benchmark must make timing identifiable

The initial narrow-gap screen allowed a refitted rank-time control to match
observed-time accuracy. Sensitivity under timestamp replacement does not prove
that timing is essential: an intervention can be out of distribution.

Construct pairs with identical source addresses, marks and event order, but
short versus long query ages whose analytic two-mode trace labels are opposite
for every source. Keep the same timestamp order in both rows. Reject mark
vectors without opposite labels, using only the analytic synthetic task rule;
use independent fixed fitting/development/confirmation data seeds.

For the rank-only observation X, each source-pair has exactly one label0 and
one label1. A classifier seeing only X has paired expected accuracy 1/2 and
minimum expected binary log loss log(2). This includes a stochastic classifier
whose randomness is independent of the target. There is no guarantee that a
finite sampled prediction set lands at exactly 50%. A model with clock times
can, in principle, distinguish the pair. This proves a task information gap,
not that the proposed architecture solves it efficiently or uniquely.

Fit/dev split whole independently generated mark populations; never split the
two members across folds. Bootstrap over complete pairs of populations, not
over their dependent members. At least several independent pairs are required;
a single-pair interval is unavailable, not precise. Evaluate all models on the
same frozen queries. Record that rejection sampling creates a controlled
synthetic distribution; real-stream validation remains a separate milestone.

## 357. Admission, measurements and rejection criteria

Read-only numerical tests check silent preservation/semigroup, temporal
gradients, exact zero-split nesting, address equivariance and no cross-source
state leakage. Guarded integrated contracts check all-head gradients,
counterfactual proposal counts and exact optimizer/checkpoint continuation.
Small full-accounting smokes precede fixed-budget fits. No source is changed
under an active frozen queue; no second local trainer is admitted.

Eight order fits (four variants x S4/S16), paired-timing observed/rank fits,
and protected/private paired timing are the first screen. Use 128 fitting
queries/pass, four passes, 256 dev queries for S4 and 256 for S16 so uncertainty
has several populations. Final confirmation uses independently seeded pairs
and paired model seeds, conditional on a useful quality/work tradeoff. All
gap probes, whole-fit/per-target/inference work, occupancy and parameter counts
remain beside the original results. No long run is warranted by local algebra
alone. Language integration follows only if event evidence identifies a gain.
