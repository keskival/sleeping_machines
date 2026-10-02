# Joint credit over protected addressed outcomes

## Failure and why this next change

The completed balanced native/tapped/full/shallow fit remains near one bit
and 50% on the distant joint target. A frozen readout followthrough also fails:
affine one-bit probes are near chance on new suffixes; degree-2 heads become
overconfident without generalization. These results do not justify increasing
readout size or assuming the available final context already contains usable
bits. They motivate an explicit evidence-access intervention.

This proposal retains the native temporal core, persistent vector receivers,
separate keys and values, sparse event updates, computational clocks and
admitted counterfactual learning. It adds a small addressed **outcome** bank:
every observed token writes itself as the latest successor of the previously
observed token. The address is that observed predecessor, never a target,
hand-selected bit position or marker-specific rule. All occupied addresses
are candidates. The bank has 27 available addresses and one causal outcome
write per event after the first. Raw symbols are protected until overwritten;
current learned value embeddings interpret them at read time, avoiding old
value-map versions or hidden producer-credit promises.

At the observed query cue, two learned key/query races each deliver one small
value. A generic bilinear query decoder mixes those values and the current
native context. No component extracts bits, computes XOR or filters candidates
to the two useful markers. The decoder's affine and bilinear residuals start
at zero, so the parent's forward predictions nest exactly. A learned query
may exploit the constant cue, and learned keys may identify useful fixed
addresses; this does not establish deep context abstraction. A same-width
shallow native control is required, and extra depth must earn its work.

The bank preserves old evidence on the test but lacks learned write selection
and long-context keys. It is a diagnostic construction, not a complete
language architecture. An algorithm explicitly inspecting other count-table
addresses could also access this evidence; the paired one-bit bound covers
only query suffix/count inputs. Do not claim all counting is defeated.

## Conditional joint credit

For occupied candidates i,j, let pi1 and pi2 be the categorical probabilities
of the two exponential races. With protected values v_i,v_j and current core
feature h, the binary query logit is

    z_ij = parent_logit(h) + b + w^T h + u1^T v_i + u2^T v_j + v_i^T A v_j.

The decoder is bilinear in the two delivered messages; this is a standard
local vector interaction. For label y and logistic loss ell(y,z), train the
conditional expected risk

    L = sum_ij pi1_i * pi2_j * ell(y,z_ij).

Then for the first race's score s1_i,

    dL/ds1_i = pi1_i * (sum_j pi2_j * ell(y,z_ij) - L).

The second race has the analogous derivative. Decoder/value/core derivatives
are also included through all pairs. This is exact for this terminal
categorical-content objective and protected candidate state. It is not a full
sequence gradient through earlier native hard routes or write alternatives.
The earlier native surrogate remains scoped as before.

Inference uses two actual exponential races and decodes only the delivered
pair. Their winning times determine delivery completion; the protected raw
symbol values do not change during this terminal wait and no later event is
scored. No differentiable downstream winner-time objective is claimed. Native
prefix timing computation remains intact. This separates exact content-choice
credit from unproved whole-timing credit.

Compare joint risk with the same model trained through the existing local
counterfactual race surrogate at a sampled delivered pair. Both have identical
inference mechanisms and initialize identically; credit/objective estimators
differ. Joint risk costs C squared terminal decoder/loss evaluations, C shared
candidate value interpretations and two C-key scores. Local training scores
the same keys and forms candidate values for its local teacher. Inference
scores two C-key sets and delivers two values. Candidate discovery, raw state,
native prefix work, losing values and optimizer work remain charged. This is
not zero-cost dormant capacity or free attention.

## Bounded protocol and claims

Use the same complement-balanced gap-8 episodes. Initial comparison: native
payload4/depth2 versus same-width depth1, two core heads/two receivers, four
query-key/value dimensions, 16 fitting groups/64 queries,16 passes,U4,Adam
.01,seed6. Every arm processes all 15 inputs and learns from query-only
binary labels. Full credit spans each episode; noise couples quartets and
variants. Cold state resets per episode. Development groups/seed72001 select
from fixed passes and remain exploratory. Reserve new seed74001/32 groups
for fixed-model evaluation after selection, without retuning on its labels.

Contracts must verify generic causal outcome writes, unchanged parent-core
parameters, zero-residual forward nesting, target-free predictions, exact
joint score/decoder derivatives, actual next-Adam and driver interruption
recovery, teacher/inference forward identity for local credit, and complete
operator coverage. Small smokes precede fits, one safe-run job at a time with
the established RSS/VMS/8GiB availability guards. No Transformer/LSTM training
is moved from AWS and no other host's language pooling run is duplicated.

A useful-dependency gate still requires at least75% and at most.8bits on
held-out suffixes. A joint-credit attribution additionally requires .05bits
over its local-credit control, retaining every result and all work. Fresh
evaluation is a fixed-model one-seed structured-task result, not natural-text
advantage, semantic depth, dense-control superiority or resource supremacy.
If shallow matches full, preserve that fact and avoid paying unnecessary
depth. Successful retrieval cannot by itself prove historical core-producer
learning or optional write-route credit.
