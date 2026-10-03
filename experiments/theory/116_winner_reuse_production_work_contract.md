# Production work comparison for exact winner-return reuse

Admission115 fixes the equivalent estimator and checks all double-precision
gradients independently. Now compare original versus winner-reuse full-write
credit at p16/L8/H2/pool2/T16, private AND depth-shared receiver maps, from
identical nonempty private state, parameters and chronological entering RNG.
Synthetic observed/next-token arrays only; no text8/DEV/test/quality claim.

Both arms take the SAME actual target-weighted Adam update:16targets,
warmup32/lr.002/clip1. Capture ALL accumulation (factual/shadow forward and
backward), gradient normalization, clipping and optimizer work. Native
inference is identical cold selected-value forward. Compare every preclip
parameter gradient (float32 rtol3e-4/atol3e-6), global gradient-relative error
<=3e-5, and actual update difference norm <=.005 of the old update norm.
These tolerate ordinary batched float32 reduction differences, not estimator
changes. Report measured errors as well as admission thresholds. Independent
double every-gradient equivalence remains the exact implementation contract.

Each optimized arm additionally checkpoints its NONEMPTYAdam/private state/
RNG/counters with a partly filled three-target gradient window, continues
one target, and verifies recovered loss/predictions/state/RNG/all gradients/
actual second warmup update bitwise. Recovery repetition paid and separately
labelled; comparison table describes only ONE actual sixteen-target first
update per arm, not the total diagnostic campaign.

Old512lanes/8192shadowevents versus new256/4096; factual key scoring32 and
selected writes16 pertarget remain. Require full-work reduction >40% under
the SAME formula-complete arithmetic2FLOPs/MAC plus unit-special convention.
Whole-stepGFLOPs, fittingMFLOPs/target and nativeinferenceMFLOPs/target appear
for BOTHarms with16target denominator. No conversion of shadow counts into
energy, physical-hardware cost or general model advantage. Record measured
traced wall time/RSS, but tracing overhead and ordinary CPUemulation make
this a correctness/work admission, not a benchmark throughput comparison.

No live AWS job source/queue/settings change; long10M comparison ownership
remains aws_language_credit_matrix_20261003T013500Z. Local one-job uniquequeue,
one thread, virtual3GB/RSS1.25GB/min8GiB/timeout180s based on note109's measured
86.859s/429912KiB six production updates plus independent sequential returns.
