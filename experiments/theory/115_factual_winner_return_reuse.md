# Exact factual-winner reuse in causal full-write credit

The newly queued AWS depth8 teacher/factorized/full-replay10M comparisons own
the next quality evidence. Do not duplicate those fits or edit their frozen
drivers. Complement them by reducing the learning cost of their estimator,
with source-frozen sibling implementations and numerical contracts first.

Concrete inefficiency: every race enumerates U shadows, including its sampled
factual winner. If replay forces the SAME winner at the SAME original first
time, keeps entering state and all future noises fixed, and makes the actual
sparse write, induction over later events gives the factual trajectory. Its
downstream loss is therefore already available from the factual pass. Let
Q_ri be the detached suffix sum for receiver i at race r and w_r its factual
winner. Reuse Q_r,w_r=sum_{t>=event(r)}loss_t and run only U-1 different outcomes.

The local choice objective remains sum_r sum_i pi_ri Q_ri, so every score VJP
and earlier producer gradient remain identical in exact arithmetic. Factual
clock, payload, private-state and representation derivatives remain unchanged.
The branch-index and reused losses are detached: do NOT differentiate the
reused factual loss through the categorical teacher a second time. This is
an estimator implementation optimization, not a new credit rule or proof of
whole-stream exact expected-risk differentiation. Coupled timing/routes and
truncated all-target utility scopes from108/113 remain unchanged.

For T tokens and D*H races/token: old R=T*D*H yields R*U shadow lanes and
R*U*T shadow events; new yields R*(U-1) and R*(U-1)*T. For p16/L8/H2/pool2/T16,
512->256lanes and8192->4096shadowevents. Candidate discovery and factual
backward are unchanged, all available keys still score, and shadows still
recompute prefixes; full fit work is measured, not declared halved from counts.
At U1 no shadow is required and categorical choice credit is zero. At U4
only25%shadow-lane saving is expected. Inference is exactly the same selected
native program with no additional state/parameters/scoring/deliveries.

Admission: new winner-recording kernel must nest the frozen RNG kernel's
primal/state/end RNG and every gradient. Private/shared p4/L8 U1/U2/U4,
nonempty entering state, two/three targets, compare every-parameter replay
gradients against BOTH old batched and independent sequential full enumeration;
reuse all factual-winner outcomes, preserve future token/target causality and
caller RNG, and recover pending gradients/state/Adam/RNG/counters exactly.
Production float32 p16/L8/T16 must compare all gradients/objective and actual
target-weighted Adam updates, report measured formula-complete whole/per-target
fit and inference work for old/new under identical denominators. This is a
resource/estimator equivalence result, not text8 trained quality advantage.
One guarded serial CPU job, virtual3GB/RSS1.25GB/min8GiB/timeout180s initially;
reuse note109's production memory admission, retain all frozen sources/results.
