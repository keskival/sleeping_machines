# Deep replay language: causal return and the scale bottleneck

The completed depth8 gesture nomination changes the empirical picture: both
private and depth-shared corrected replay improve heldout NLL against BOTH
teacher and factorized controls on seed7. Depth4's failed DEV gate cannot be
generalized to depth8. This is not resource advantage: whole-fit replay is
~52–55x the controls. Seed8 must decide replication, and10M-character language
must decide whether the family works beyond small-data character fits.

For a detached entering language state, a credit chunk has T observed symbols,
T next-symbol losses, depth L, H heads and U candidate receivers. A race r at
observed event t can affect that event's prediction and subsequent predictions
through actual writes and arrival times. The appropriate chunk return is

    Q_r(i) = sum_(k=t..T-1) loss_k(legal forced alternative i at r).

Earlier losses do not depend on this intervention and may be excluded exactly.
Hold r's factual FIRST time, not the alternative's individual exponential
time; preserve future common random numbers, actually write the alternative's
private memory, and replay every dependent downstream head and event. The
conditional choice contribution, with detached shadow returns, is

    g_r = J_(scores_r)^T [ pi_r * (Q_r - dot(pi_r,Q_r)) ].

This is conditional categorical credit, combined with the separately derived
factorized first-time pathwise clock/content credit. It is NOT a proof of exact
full-stream gradient: detaching at a chunk boundary omits credit through
later chunks; future crossing winners involve the estimator's established
scope. Do not put future target tokens in observed input before their event.
Do not infer loss improvements from larger gradient magnitude alone.

Naive all-race full-chunk replay uses R=T*L*H races, U*R shadow lanes, and
U*L*H*T^2 shadow events per chunk. At T16/L8/H2/U2 this is512 shadow lanes
and8192 shadow events (512 replay events per target), before additional deep
head/key work. Memory capacity does not remove these operations. Native
winner-only inference remains sparse, but the optimizer must pay replay cost.

Cache detached factual state just BEFORE each race (including intra-event
head values/arrivals) and replay only its dependent suffix. Event-accurate
causal suffixes reduce the quadratic event sum toward T(T+1)/2; this provides
roughly2x, not orders of magnitude. Snapshotting whole state separately for
every race adds traffic and must be charged. A snapshot only at token start
cannot skip its preceding heads while preserving causal time.

Uniform k-race sampling without replacement, with R/k weighting, gives an
unbiased estimator of the finite-chunk SUM of conditional choice terms
conditional on the factual rollout and replay randomness. It does not make
truncated whole-stream credit exact, nor establish low variance or learning
quality. Shadow batching and shared parameter stacks can lower Python/kernel
launch overhead but cannot remove losing-value arithmetic. Existing DVS
priority/oracle studies do not license a language quality claim.

Next implementation contract before a language replay fit: original versus
factorized forward/state parity; actual all-target sum-loss sequential replay
versus batched shadow gradients in double precision for EVERY parameter;
forced winner identity and FIRST-time preservation for losing alternatives;
shared-map gradients equal summed untied gradients under identical weights;
causal future-token intervention; actual Adam/cursor/RNG/private-state recovery;
all traced shadow/optimizer operations charged. Preserve the present10M
private/shared ORIGINAL-TEACHER language runs as controls. They are not
silently renamed full replay. An efficient language implementation should
pass these contracts before a new unique10M queue, never replace long-run
quality evidence with a small smoke or projected throughput.
