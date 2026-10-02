# Statistic credit contracts and the remaining integrated comparison

This implements local primitives from curie's note59 without changing a running
model. The concrete failure being addressed is the AWS factorial audit's
persistent-write effect missing from message-only local credit. Statistic
values make a restricted predictive/write effect analytic; learned queries and
race keys remain an intended path to generalization. No architectural
substitution or claim of full-model learning is made by these primitives.

`sleeping_machines/statistic_race_credit.py` provides symmetric Dirichlet
prediction, exact current expected hard-route loss credit, and exact expected
one-next-event log-predictive write gain under a supplied future law q. Four
tests compare actual predictive changes and autodiff, reject invalid states,
and separate equal delivery predictions from unequal future write utility.
Together with note59's original four tests and the joint-clock kernel's four
tests, all12 contracts pass.

Important scope alongside note59's 'free and exact' counterfactual wording:
candidate loss evaluation needs lookups and arithmetic. It avoids a neural
suffix replay in this restricted objective; it is not zero resource work.
Normalizing every full predictive costs O(A*B) for alphabet A,candidates B;
cached totals and target-count access can reduce target-loss evaluation to
O(B). Candidate discovery and scored keys must still be charged. Selected
increment is O(1) with a cached total. These are complexity statements, not
measured wall time/FLOPs in the full substrate.

For fixed counts and losses, delivery enumeration has no winner-sampling
variance. It is exact for E[negative log selected predictive], which differs
from negative log mixture predictive. Neither is by itself the full sequential
gradient: hard selection changes subsequent counts, routing and times. Joint
clock credit is unnecessary only if that local objective is genuinely time
independent. A time-sensitive suffix still needs the joint state/time terms.

Write credit is exact CONDITIONAL on q. Equal present counts do not identify
future q: the tests give equal delivery predictions but opposite write gains
under two future laws. Using a receiver's own predictive as q is a declared
approximation, not a proof of optimal future writes. Future-q learning,
next-visit sampling and storage/versioning are charged separately. A single
next-event gain does not certify credit over every subsequent event.

The exact sign threshold for a written symbol y is

    q_y > log1p(1/(n+A*alpha)) / log1p(1/(c_y+alpha)).

The simpler q_y > predictive_y rule in note59 is first order. Preserve both
interpretations; sparse counts are precisely where finite increments matter.

Next integrated test must retain deep temporal content processing, race keys,
sparse addressed updates, private persistent state and counterfactual credit.
Treat a pure count table as a diagnostic control. Compare the unchanged model
with statistic-assisted memory under identical information/online-write rules,
selected activity and fitting data. Freeze the mixture/expected-loss objective
before fitting. Verify zero statistic influence nests the old model, checkpoint
recovery, read-before-write/no label leakage, clock shifts and full resource
accounting before a matched small fit. No long run is admitted here.
