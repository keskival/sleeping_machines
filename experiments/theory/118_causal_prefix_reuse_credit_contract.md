# Exact conditional credit from cached causal prefixes

The priority quality comparison remains AWSwinner_matrix014100Z's10M-char
teacher/factorized/full-write replay.117 establishes production double
winner-return reuse equivalence and a shared float32 rounding floor; no active
fit is changed. Next remove repeated prefix work, rather than remove support,
timing/state mechanisms or downstream losses to make training cheaper.

Concrete paid inefficiency: for a race in observed token k, every shadow
recomputes tokens0..k-1 even though its forced alternative cannot affect them.
Let S_k and omega_k be factual persistent state and RNG just BEFORE token k.
They depend only on observed earlier tokens, entering memory and earlier race
draws, never future tokens/targets. Snapshot them detached. A conditional
alternative at layer/head r of token k can begin from(S_k,omega_k), force its
other receiver at the same first time, and replay tokens k..T-1. By causal
induction this is the same mathematical branch as the old complete-chunk
shadow; sum all k..T-1 target losses and preserve its factual score Jacobian.

This retains every original candidate, real sparse losing write, coupled
clocks/messages, native pathwise delay/payload derivative, private depth state
and full within-chunk return. Earlier token losses are invariant to the forced
write and already omitted from the old choice utility. Inference unchanged;
cached states exist only during learning. It does NOT extend credit across a
detach boundary or produce an exact whole-stream expected-risk derivative.

For B=D*H races/token,Ureceivers,Ttargets: winnerreuse shadows B*(U-1)*T lanes
and B*(U-1)*T^2 simulatedtokens. Group suffixes by token k, each beginning at
its factual snapshot: same total lanes, B*(U-1)*T*(T+1)/2 shadowtokens. For
L8/H2/U2/T16:256lanes retained,4096->2176shadowtokens (46.875%less versus
winnerreuse;73.4375%less versus original8192). All current-token earlier
layers still recompute, as do all scored keys/proposals for each suffix.
Snapshot/RNG copies, repeated parameter stacking and group launches are
additional work. FLOPs, CPUwall and memory must be measured before claiming
improvement; do not infer total resource benefit solely from token counts.

Numerical contract in new siblings: each factual snapshot/state/RNG matches
independent sequential prefix and entering-state storage. New snapshot kernel
nests original factual ALLlogits/state/endRNG/EVERYpathwise gradient bitwise;
all tokens/pools1/2/4 private/shared L8/p4 empty/nonempty entering states.
Cached suffixes match old full shadows and independent sequential actual-write
returns, EVERYparameter objective/gradient in double, future input/target
causality and callerRNG. Real target-weighted Adam with pendinggradient/state/
RNG/counter recovery and complete accounting. No new qualityfit. Guarded
onejob CPU, virtual3GB/RSS1.25GB/min8GiB/timeout180s; production p16/T16
comparison follows only after these contracts, in its own fixed protocol.
