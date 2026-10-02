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
