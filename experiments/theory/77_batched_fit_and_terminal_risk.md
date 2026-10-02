# Independent-clip batching and conditional terminal risk

The original full native fit65.10%/.963161 trails strong raw-input and compact
controls. Frozen features give67.19% with a fresh fitting-selected head versus
57.81% for the initial reservoir, supporting useful feature learning but not
closing the control gap. Packet-scale initialization reaches66.15%/1.041987,
so this timescale change also fails to dominate either control or the original
in both quality measures. Preserve both fits, all passes and fixed selection.

## An implementation change before increasing capacity

The fitting driver currently dispatches the same small tensor graph separately
for every clip. All984 fitting and192 development clips contain twenty50ms
packets and an observed query at1s. Within an optimizer window, vectorize these
independent clips with independent state and the same observed schedule.
Share the per-pass random draws exactly as the old driver does; never mix
states or add label/index-dependent noise. Preserve the original local
TemporalRoute backward rule elementwise for every clip/head/race.

This changes kernel dispatch and graph organization, not the architecture or
credit estimator. Numerical contracts must compare every forward, winner,
persistent memory/time and parameter gradient against independent double
precision reference clips, then actual interrupted model/Adam/cursor recovery.
Measure a full U16 and partial U8 accounting/learning smoke before a full fit.
Peak live graphs can grow despite fewer graph nodes; retain the RSS watchdog,
one-thread/one-job lock and8GiB memory floor. No compilation jobs are needed.
Charge every candidate/optimizer operation; faster wall is not fewer FLOPs.

## Optional actual terminal outcome risk

The local score teacher linearizes downstream loss in delivered value and
gives payload gradients only to the realized winner. At the final observed
query, final layer, there is no supervised future state in this clip. For
two heads with U candidates each, let pi_h(i)=lambda_h(i)/sum(lambda_h).
Conditional on the entering prefix state and sampled first-arrival times,
the exponential winner is independent of its first time. Therefore

    risk = sum_ij pi_1(i) pi_2(j) CE(head(align(v_1i,v_2j)), target)

is the exact conditional terminal content risk. Keep the original sampled
first times and their bounded delays; align each candidate pair at the same
causal join. The minimum time is continuous across race exchanges, so its
interior pathwise derivative remains valid for this continuous conditional
risk. Standard derivatives of the categorical weights give terminal policy
credit, and each losing candidate receives its actual weighted outcome
derivative. Historical producer credit reaches retained prefix graphs.

Earlier prefix routes keep their original local surrogate. This is not exact
whole-core expected gradient or a proof of learned abstraction. Candidate
values were already computed during training; U² head/loss evaluations,
categorical weights and their backward work are additional and must be paid.
Inference remains the original two hard winners, value deliveries and native
state updates. No statistical expert, extra decoder or dense carrier enters.
The soft categorical arithmetic is learning work, not inference attention.

Verify the enumerated risk and all derivatives against a loop over candidate
outcomes, plus reference/batched whole-prefix gradients and actual recovery.
Begin with a matched bounded integrated local/pair fit before a larger
campaign. Hold data, width/depth/pool, random stream, updates and stopping
policy fixed. Small or seed-dependent gains do not establish superiority;
all strongest full/compact controls and research costs remain visible.

If batching passes and is materially faster, larger native capacity becomes
feasible within a measured host budget. Any later width/pool change requires
its own shape-specific contracts, learning smoke and matched control. Larger
pool capacity still charges all scored keys and losing training values; it
cannot be described as zero discovery/training work beyond active commits.
