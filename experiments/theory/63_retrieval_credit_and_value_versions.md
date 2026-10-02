# Retrieval credit and consistent value coordinates

## Failure addressed and retained mechanisms

The existing context-addressed prototype writes v_t = W_t h_t, accumulates
those values at a fixed context address, and detaches the slot at every credit
boundary. An old slot supplies neither a current gradient to W nor values in a
single current coordinate system: its mean contains multiple historical W_t.
This is a concrete credit/versioning limitation in addition to the previously
measured residual-responsibility and single-address hypotheses. It is not an
autograd regression or evidence of a complete explanation of language failure.

The proposed repair retains native temporal computation/races, separate
receiver keys/values, addressed persistent state, sparse writes and admitted
unrealized-route surrogate credit. Fixed hash addresses, running means and the
extra reader remain; it does not introduce learned pooling or KV race attention.
Only the placement of the linear value projection changes. Original variants
and completed results remain controls and historical evidence.

## Derivation and exact boundary

For fixed weights, linearity gives

\[
\frac1n\sum_i Wh_i=W\left(\frac1n\sum_i h_i\right)=W\bar h.
\]

Let the read delivery be u=R_v W\bar h + R_s 1[n>0], with \bar h treated
as a fixed stored feature. For a downstream delivery adjoint g,

\[
\nabla_W L=(R_v^T g)\bar h^T,\qquad
\nabla_{R_v}L=g(W\bar h)^T.
\]

Storing \sum h_i and applying current W on a read therefore contracts credit
to all those historical **fixed features** without restoring a history graph.
By contrast, a detached projected-value slot has zero derivative to the W
which produced it. This also avoids mixing old W coordinate systems. It is
accurate to call the recovered derivative exact for the fixed-feature linear
projection; a whole hard-route model still uses its declared local surrogate.
Future W updates can change old memories' interpreted values, as intended.

This does not give the historical h_i producers exact credit. Their features
were computed under older core parameters and routing decisions; detachment
still removes those paths. Nonlinear value transformations cannot generally
be moved through a sum without richer sufficient statistics. A learned reader
over frozen historical features is not proof that the core learns deep features.
The full/minimal integrated comparison is required for that attribution.

## Cost and prediction

Original memory uses one W product per outcome write plus one R read per
event. Late projection uses one W product per occupied read plus one R read
and a vector sum per outcome write. Same feature width/slot capacity, not a
zero-cost mechanism: charge stored sums/counts, lookup/hash, all projections,
core keys/selected deliveries and counterfactual/optimizer work. Available
addresses4096 and active address reads/writes remain distinct.

Prediction: after actual chunk detachment a repeated-context retrieval gives
W nonzero current credit in the late variant, where the original cannot credit
old W from that retrieval alone. Frozen whole-stream outputs/gradients should
match within numerical tolerance before detachment; zero-read nesting of the
native model is exact. Demonstrate these contracts, actual next-Adam recovery,
source/cursor persistence and operation coverage before fitting.

Predeclared small learning comparison: four passes over text8[0:1024], dev
text8[90M:90M+2048], seed6, c16/U64/warmup512/lr.002, H2/pool2;
full payload16/depth8 native, original addressed and late addressed, plus late
payload2/depth1 and **same-width payload16/depth1** controls. Width and depth
change together in the minimal arm, so it cannot isolate depth alone. Select
the minimum frozen **cold-state** dev bpc across fixed
passes. At that same selected checkpoint, evaluate frozen fit replay followed
by dev, retaining predictive state but resetting context-hash history and
previous-address to forbid invented fit/dev count transitions. Both evaluation
policies are causal; warm replay cost and extra history are explicit. Never
select using the warm score. Development is reused exploratory evidence,
not confirmation. Do not compare its2047-target scores directly to saved8191-
target references or fill report cells with pending estimates.

Follow-up nomination requires late full improve both native full and original
addressed full by .02bpc, beat late minimal and same-width shallow by .02bpc,
and use no more than2×
original full fitting work under the same estimate convention. Otherwise
preserve negative evidence and diagnose which information/credit contract
failed; no automatic larger fit. Whole-fit/per-target costs, capacity/activity,
replay and inference work belong in the completed appendix. The AWS unchanged
integrated capacity/exposure campaign remains independent and prioritized.

## Factorization is optimization, not extra reader expressivity

Both maps are square, unrestricted linear maps. With fixed weights the only
value-read map is A=R_v W; a single unrestricted A represents the same class.
Changing the placement restores fixed-feature credit and consistent coordinates,
but does not by itself enlarge the reader's inference function class. If useful
quality improves, attribute it to learning/coordinate consistency and the
resulting core/reader weights, not extra inference expressivity of two maps.

For ordinary SGD, writing G=∂L/∂A gives ∂L/∂R_v=G W^T and
∂L/∂W=R_v^T G. Ignoring the second-order product of the two updates,

\[
\Delta A=-\eta\left(GW^TW+R_vR_v^TG\right)+O(\eta^2).
\]

The factorized optimization thus supplies left/right Gram conditioning. This
is an algebraic SGD observation, not a derivation of the actual Adam trajectory
or a guarantee of faster learning. Historical raw features are fixed under the
truncated objective; their omitted producer derivatives remain omitted.

Frozen inference can compile A once and omit W thereafter, preserving raw
feature slot semantics and all native temporal dynamics. Under2FLOPs/MAC,
fusion costs2d³ and avoids2d² per occupied read, breaking even after d occupied
reads (32 for full/shallow,4 for minimal). Charge compiler initialization/copy
and hash/RNG/traffic separately; no latency/energy claim follows from this count.
Verify coupled frozen outputs, slots, removal of d² unused weights, actual
sampled operation counts and model fingerprint preservation. Compiled models
must refuse training: training A directly is a different optimizer experiment.

## A stronger feature diagnostic than equal n-gram loss

Construct inputs [24,b1,25,b2,noise...,26] and target b1 XOR b2, with all four
bit pairs equally represented for each identical noise suffix. Noise uses only
symbols2–23, so cue26 never occurs earlier in that example. For any local
order K≤noise_length+1, every paired query suffix is identical. In a causal
count cascade initialized from common fit counts, query-time counts are also
identical: prefix transitions never end in cue26. Validate those actual count
vectors independently for orders1–8 (avoiding integer-code overflow).

For this balanced distribution any predictor restricted to those query suffixes
and query count vectors has at least1bit target logloss and at most50% expected
accuracy; the full observed prefix determines the target exactly. The extra
useful conditional information is1bit. This bound applies to those predictor
inputs, not arbitrary models which recurrently process the whole prefix or
inspect unrelated count addresses. It is an explicit information-path test,
not a universal n-gram theorem. Sample matching must be built into the generator,
not approximated by a per-target minimum over count losses.

Frozen text-model sensitivity on these pairs can test whether prefix differences
survive to the query. It cannot establish parity learning: these checkpoints
were never fitted to this task. Fixed-context lookup cannot recall a unique
earlier cue automatically if the queried suffix has never been written. Useful
old information must arrive via core recurrence, learned context addressing,
an explicit delay, or another demonstrated path. That is why a fixed-hash gain
alone would not settle the deeper-feature question. A later integrated parity
fit would need matched full/shallow controls, prefix-data generalization,
target-only versus whole-stream supervision labelled as different objectives,
complete prefix computation/credit work and bounded admission prerequisites.
