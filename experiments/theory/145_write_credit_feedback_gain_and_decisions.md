# Write-credit feedback gain and decisions toward practical advantage

3 October 2026. Read139–144 and §§398/413/414 first. This note derives
partial surrogate operators and checks constructed witnesses with the standard
library. It does not diagnose a trained trajectory or report a new model fit.

## The failure and the mechanism we retain

Value choice credit made the integrated hard-race language model substantially
better, including at depth8. Additional stored-delta write credit diverged;
written-only credit was stable but worse at pool2 and diverged at pool4.
These failures justify withdrawing those additions, not removing temporal
computation, addressed state, separate keys/values or counterfactual credit.
The current successful `linear` operator remains the default.

Note144 corrected a misleading distinction: value and write choice teachers
both differentiate memory-conditioned scores. Their auxiliaries are inserted
at different locations. Here we derive why bounded write coefficients do
not guarantee stability of the write insertion, and what to measure before
trying a smaller coefficient, a normalization or a stop-gradient repair.

## A partial memory operator, not the full recurrent Jacobian

Fix one active lane/head and its incoming vector, query, arrival, stored
timestamps, forget/write gates and realized winner. Inside an unsaturated
score clamp, write

`s_i(m_i) = q^T(key_i + K_i m_i)/sqrt(P) + bias_i`,

`k_i = ds_i/dm_i = K_i^T q/sqrt(P)`.

Strictly outside the clamp k_i is zero. At its boundary an actual-runtime
derivative convention is required; do not infer it from the clamped output.
Let S be the block-row map `(S v)_i = k_i^T v_i`. The factual memory
update has partial Jacobian A: identity on unselected slots and the gated
decay/rotation map on the selected slot. At fixed inputs/clocks the selected
map has norm at most1, so `||A||_2 <= 1`. A dormant identity block prevents
strict contraction in the full private-memory norm. This bound is not a
statement about the full native loop, whose content, query, clocks, source
context and later messages also depend on memory.

Let d_i be a detached candidate delta (stored change or newly written vector),
B map a scalar z_i to the vector d_i z_i in slot i, and
`Jpi = diag(pi)-pi pi^T`. The zero-valued write auxiliary is

`alpha * (pi_i - stopgrad(pi_i)) * stopgrad(d_i)`.

It changes no factual values but adds the ideal-arithmetic local operator

`H = alpha B Jpi S`, hence `J_partial = A + H`.

The added adjoint is

`(H^T G)_j = alpha k_j pi_j [G_j^T d_j - sum_i pi_i G_i^T d_i]`.

This is the installed write-coefficient score credit pulled back to memory.
It is distinct from the value auxiliary's emitted-message insertion, even
though both use S. There is no extra derivative of the detached d_i.
The effective Jacobian is a property of the surrogate backward program,
not the derivative of the identically zero forward auxiliary recomputed
with a new stop-gradient anchor at every point. It is also not an exact
expected-risk or route-boundary gradient identity.

## Cheap measurable bounds with slot geometry retained

B's columns occupy disjoint memory slots, as do S's rows. Put
`D_d = diag(||d_i||)` and `D_k = diag(||k_i||)`. On their nonzero directions,
the auxiliary has the same nonzero singular values as the U-by-U matrix

`M = alpha D_d Jpi D_k`.

Zero vectors contribute zero singular values. Thus one need not materialize
an `(U P)`-squared memory Jacobian to study this partial addition. In
particular,

`||H||_F^2 = alpha^2 sum_ij ||d_i||^2 Jpi_ij^2 ||k_j||^2`,

`||H||_2 <= min(||H||_F, sqrt(||M||_1 ||M||_infinity),`
`                 |alpha| max_i||d_i|| max_j||k_j|| r(pi))`,

where `r(pi) = min(1/2, max_i 2 pi_i(1-pi_i))`. The latter follows from
the symmetric softmax Jacobian's Gershgorin row bounds. These are local
ideal-arithmetic upper bounds evaluated in Python floats, not rounding-certified
interval enclosures or observed gains. For U2, H has rank at most1, so
the Frobenius bound is its exact ideal-arithmetic spectral norm:
`|alpha| pi_0 pi_1 sqrt(||d_0||^2+||d_1||^2) sqrt(||k_0||^2+||k_1||^2)`.
Written vectors are
not uniformly bounded in the actual model; unnormalized mixed inputs and
trainable input maps matter (142). Bounding d alone would still leave K/q,
and therefore k, unconstrained.

`write_credit_feedback_geometry.py` provides O(U P) JVP/VJP actions and
these U-by-U bounds without importing Torch. Its checks compare the adjoint
identity, dense-column Frobenius norm and finite differences of the correct
**fixed-anchor extension**, including U1, zero-score-read and alpha0 limits.
The scalar checks are not native automatic-differentiation contracts.

## Bounded-write amplification survives winner averaging

Take U2, scalar slots, pi=(1/2,1/2), d=(1,1), k=(1,1), alpha1 and selected
memory derivative1/2. With winner0,

`A = [[1/2, 0], [0, 1]]`,
`H = [[1/4, -1/4], [-1/4, 1/4]]`,
`A+H = [[3/4, -1/4], [-1/4, 5/4]]`.

The largest eigenvalue/singular value is `1+sqrt(1/8) = 1.3535533906`.
Each write has norm1 and the physical selected update decays, yet this
partial backward operator amplifies. Repeating this **frozen local matrix**
16 times gives126.9423 gain on its leading direction. This repetition is
a constructed operator witness, not a realized native state trajectory:
native coefficients, clocks, winners and cotangents change over time.

For any alpha>0 the dormant direction's Rayleigh quotient is `1+alpha/4`,
so simply making alpha small cannot guarantee contraction for this family.
With the other winner A swaps its diagonal. Equal conditional winner
averaging gives `E[A]+H = [[1,-1/4],[-1/4,1]]`, whose largest eigenvalue
is1.25. The fixed-coefficient conditional average also amplifies; this
is not a claim about products of expected matrices or expected native loss.
The sign and orientation of k/d matter. Negative feedback can damp instead;
the theorem does not say every write addition is unstable.

## Eigenvalues and coefficient magnitudes are insufficient alone

With d=(1,1), k=(1,-1), pi=(1/2,1/2), the addition is

`H = [[1/4,1/4],[-1/4,-1/4]]`, with `H^2=0`.

Take the physical partial map A=I, possible at zero age. All eigenvalues
of I+H are1, yet `(I+H)^n=I+n H`. On `(1,1)/sqrt(2)`, n20 gives
`sqrt(101)=10.04988` amplification. Nonnormal transient growth is therefore
possible without an eigenvalue above1. Again, this repeats a fixed local
operator and asserts no trained trajectory. For actual depth/time credit,
use singular-vector/adjoint actions and products at the actual history;
a single eigenvalue or an averaged local norm cannot determine stability.

## What normalization, clipping and repairs would actually address

Dividing the loss by a target count scales the downstream cotangent; it
does not change H or the relative amplification of longer feedback paths.
Global clipping happens after backward. It cannot repair an intermediate
nonfinite adjoint, and finite clipping can suppress useful unrelated content
updates when feedback dominates the norm. Warm Adam further changes the
update geometry (139). None of these identities establishes that clipping
caused the observed divergence.

Scaling alpha reduces a local auxiliary bound but does not supply strict
contraction where dormant coordinates already have identity transport.
Bounding/normalizing k or d changes the surrogate and requires a fidelity
comparison. Removing only the write auxiliary's score-memory dependency
would set this local S contribution to zero while retaining score-parameter
and incoming-query credit; it can also remove useful memory learning.
Other query/message/clock feedback paths survive, so it would not establish
full-loop stability. No such core change is installed here.

Independent questions must be answered together: (1) does the addition
explain the useful value teacher's residual utility (144), and (2) what
actual adjoint/update distortion does its feedback introduce? Good local
utility alignment alone does not answer (2). Low feedback gain alone does
not answer (1). A repair should pass both before another integrated fit.

## Decision order with limited hosts and compute

The latest completed p64/D4/U4 credited fit is2.179497 T256 bpc versus
2.183315 at U2, at44.0738 versus26.7875 whole-fit TFLOPs: only.003818 bpc
better for1.64532x work. The p32 U2->U4 gain was.026334 bpc. This is
consistent with capacity/exposure saturation (398), not causal identification
of dilution or a mathematical ceiling. The tied-map pool arms already
queued test that hypothesis while retaining private keys, clocks, timescales
and memories. Their discovery/losing-proposal work remains charged.

The exposure argument also needs parameter-group scope. With `linear`,
proposal vectors are detached in the categorical auxiliary, so unselected
input/output/gate/control maps receive no direct local proposal gradient.
But every unsaturated candidate's key-read matrix can receive score credit:
if gamma_i is the score cotangent, then
`dL/dK_i = gamma_i q m_i^T/sqrt(P)`, whether i won or lost. The shared
query also receives contributions from every admitted key. Thus selected
write frequency is not the exposure count for every parameter group, and
all-event score support is not uniformly informative exposure either: pi,
coefficient contrasts, query/state excitation and saturation affect it.
The c*parameter-count/effective-examples ansatz in398 is a conditional
estimation model, not a generalization theorem for endogenous recurrent
routing. Width-by-pool results alone cannot identify that assumed rate.

1. Preserve the active producer's integrated chain and sources. Its v7
   remaining width arm, v6 tied-pool arms and independent seeds precede
   the horizon arms in the current queue. Do not add a redundant width,
   clipping or write-coefficient sweep. Shared-map numerical contracts
   and completed results must precede any larger promotion.
2. Complete actual-trained winner/state/cache/RNG admission (143) before
   assigning completed language quality to the cached winner-only backend.
   FLOPs, per-call matrix copying, wall time and energy remain separate.
3. When the producer's checkpoint and physical reservation are available,
   run144's bounded same-cotangent delivery/commit factorial before another
   write-credit fit. For a subsequent gain audit, capture d_i, k_i,
   unclamped scores and selected decay/rotation at prespecified FIT sites.
   Verify H and H^T against isolated native surrogate VJPs, including
   inactive/clamped cases and original float32 versus promoted64. Then
   compare actual adjoint histories and paired finite clip/Adam forks.
   This follow-up is a protocol requirement, not an admitted job or result.
4. Preserve the assigned AWS90M models and saved dense controls. The
   owner executed checkpoint-preserving reallocation during this continuation:
   HANDOFF's14:32 update records15 passing contracts, a completed491520-
   presentation admission pilot and the first90M/p32/D4/U4/linear fit started.
   Private streaming replay/teacher fits resume in the other two guarded
   slots; shared teacher remains recoverable. This continuation changes
   neither scheduling nor those healthy fits. Pilot throughput and streaming
   replay's positive online trend remain separate from completed heldout
   quality and from the reset-segment protocol.
5. Decide advantage from completed heldout quality AND full resource
   boundaries. Counts remain strong in their established statistical region;
   beating one lightly trained Transformer there is insufficient. Scaling
   benefit, useful distant-state/depth interventions and sparse capacity
   at controlled learning/discovery cost are the next discriminating
   evidence. Keep all seeds and use the assigned DEV rule; these observed
   test results must not choose a new fitted coefficient or test-tuned gate.

This work adds measurement algebra and a decision rule, not a substantive
architecture substitution, a new queue or a superiority claim. Existing
integrated jobs remain prioritized. Dynamic routing and representation
learning still interact in the retained construction; stability and utility
are testable joint constraints rather than a proof of impossibility.
