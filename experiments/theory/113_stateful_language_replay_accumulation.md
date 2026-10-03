# Stateful language replay accumulation

The numerical language helper used an explicit per-chunk seed inside fork_rng.
A benchmark adapter must preserve the existing chronological stream RNG law:
factual native races consume the entering RNG state once; shadow alternatives
start at that SAMEstate and never advance the real stream. Do not change
randomness cadence as an accidental architectural/credit ablation.

Create new sibling RNG-state helper/kernel files, preserving the successfully
contracted note108 sources. Tensor RNG state is accepted in place of an integer
seed. Return factual end RNG in activity; the replay accumulator installs that
state once after the chunk's objective/backward. All shadows remain forked.
Existing target-weighted GradientAccumulator update, warmup, normalization,
clipping and Adam semantics are inherited. Actual persistent receiver state
crosses chunks/updates; its graph detaches each credit chunk. Track all shadow
lanes/events separately from selected actual updates and trace their operations.

Admission: private/shared p4/L8/H2/pool2 double models, nonempty entering state,
2+1 observed-target microchunks. Verify every factual prediction, ALLnative
state and end RNG against original teacher forward before delayed update;
full all-target accumulated gradients against independent sequential replays;
exact partly filled gradient/Adam/private-state/cursor/RNG/replay-counter
checkpoint continuation; target-label independence before optimizer update;
actual partial-window target normalization, warmup/clipping/Adam equality and
complete stage accounting. Only a correctness adapter, not a trained benchmark
or changed active AWS10M source. A new fitted driver/protocol remains necessary.
