# Language replay benchmark driver contract

Note113 validates the stateful target-weighted replay accumulator. The active
original-teacher language driver has a separate traced path that bypasses
accumulate; aliasing the new learner there would omit corrected credit from
traced windows. This is a concrete new-driver admission failure to prevent,
not a reported bug in the unchanged original-teacher control.

Create a new sibling benchmark driver that ALWAYS calls ReplayAccumulator,
with optional operation tracing around that same call. Keep chronological
native RNG, actual persistent state, original hard score map, private or shared
receiver maps, p16/L8/H2/pool2, original one-unit/token timing and delayed
target-weighted Adam/warmup/clipping. Full-write first-time-preserving shadows
sum all downstream losses within each detached credit chunk. Sparse selected
inference remains original native reference. No horizon/operator/data change.

Checkpoint AFTER ANY microchunk: actual model, Adam, pending gradients, private
stream state, RNG, learning/counterfactual counters, epoch/token/window cursor,
in-progress trace samples and DEV-selected best state. Source/settings/data
hashes must match recovery; preserve completed results. Whole fit arithmetic
comes from completed sampled first/last full/partial optimizer windows with
explicit actual target denominators. Forward AND backward shadows are charged;
no unsupported physical-event projection or omitted losing-clock simulation.

Numerical admission before a real fit: small p4/L8 private/shared, synthetic
observed/next-token arrays only,2-target chunks/U4,8targets/twoupdates. Compare
uninterrupted execution with saved pending2target/zero-update recovery and
completed-one-update recovery for EVERY model/Adam/gradient/private-state/RNG/
counter/cursor field, output curves and full work estimate. Verify traced and
untraced windows apply identical full-return gradients, reject changed source/
data/settings, never overwrite completed output and keep complete operator
coverage. One guarded job, RSS1.25GB/address3GB/timeout180s/min8GiB available.
Any later real quality comparison needs a fixed data/quality/work protocol;
active AWS10M original-teacher models remain unchanged and independently owned.
