# Precision and equivalent counterfactual programs

115 proves winner reuse in exact arithmetic and tests every small double
parameter gradient independently.116 measures49.6%full fittingwork saving
at p16/L8/T16, but BOTHfloat32 coordinate gatesFAIL. Globalgradient errors
are~1e-6 and actualAdamupdate differences.026%/.059%, wellinside declared
global/update limits. Preserve allthese outcomes; do not weaken old gates or
edit activeAWS10M teacher/factorized/full-replay protocols.

Exact mathematical trajectory equality does not imply byte equality between
the same winner evaluated in one factual lane and one row of a512-lane GEMM.
Both matrix contractions and loss reductions can round differently. For a
fixed localscorevector, g_i=pi_i(Q_i-sum_j pi_j Q_j). If each Q_i incurs error
atmostdelta, scorecredit error is bounded by2pi_i*delta. Parametercredit then
pulls this through J_score^T, in addition to errors in that Jacobian. No bound
on its conditioning is assumed. Large common suffixloss can cause numerical
subtraction even when actual choiceadvantages are tiny. A detached common
baseline leaves exact categoricalcredit unchanged, but can reduce this
rounding term. This is a hypothesis about the observed tightcoordinate
failure, not a claim that precision explains underfitting or deep-feature gaps.

Frozen audit: EXACTsame float32 initial p16/L8/H2/pool2/private/shared model
weights and nonempty state as116, promote those represented weights/state to
double (no double reinitialization), synthetic16 observed/next-token targets,
same chronological RNG. Compare originalfull and winnerreuse EVERYparameter
gradients in float32 ANDdouble. Double old/reuse must meet established
rtol3e-7/atol3e-9. Capture all factual and forced-shadow winner histories in
bothprecisions, using the115bitwise-nesting winner-recording kernel; no route
mismatch may be attributed solely to arithmetic inside a fixed branch.

Report each float32 estimator's global and every-coordinate error against its
own corresponding double program, old/newdoubleagreement and actualbranch
mismatch counts. Preserve full gradientvectors and candidate-route histories
for reproducibility. Helpers fork callerRNG; explicit endRNG matches allarms.
No optimizer updates, DEV/test/text8 training, precision repair or fitted
benefit. Diagnostic campaign FLOPs/traffic/energy are unknown, notzero; no
new resourcecomparison beyond the completed counted116result. One guarded
serial CPUjob, one thread, virtual3GB/RSS1.25GB/min8GiB/timeout180s. Existing
production memory admission and116measured~400MiB provide the baseline.
