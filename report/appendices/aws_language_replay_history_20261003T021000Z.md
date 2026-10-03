
## Production-p16 integrated fitting work: approximately halved

Both optimized actual-driver recovery contracts014000Z pass (98.695s/
497016KiB); both production-p16 learning smokes complete/learn under494MiB.
Same1024 fitting targets/onepass, p16/L8/H2/pool2, lr.002/U256/warmup4096,
seed7,129-char disjointDEV. Tiny fits are numerical/resource admission ONLY;
language quality comparisons require completed10M fits. Full fitting work
includes factual/loss/backward, all actual shadows, normalization/clip/Adam.
Same2FLOPs/MAC+unit-special conventions for allcolumns; inference percharacter.
No physicalprojection, traffic/RNG/energy separate.

| Family/implementation | Fitting targets | Whole-fit GFLOPs | Per-target MFLOPs | Inference MFLOPs/char | Shadow lanes | Observed smoke wall s |
|---|---:|---:|---:|---:|---:|---:|
| private/all-lane | 1024 | 69.349421 | 67.724044 | 0.097376 | 32768 | 554.12 |
| private/winner-reuse | 1024 | 34.913923 | 34.095628 | 0.097376 | 16384 | 379.20 |
| depth/all-lane | 1024 | 69.347356 | 67.722027 | 0.097376 | 32768 | 548.41 |
| depth/winner-reuse | 1024 | 34.911858 | 34.093611 | 0.097376 | 16384 | 374.33 |

Private full work drops49.6565%, shared49.6580%; lanes32768→16384 and
shadowevents524288→262144 perfit. FinalBPC differences between implementations
are <=3.44e-7, as expected from shape-dependent floating arithmetic. New
implementation recovery is bitwise exact; old/new optimizer trajectories are
mathematically equivalent within measured numerical tolerance, not declared
bitwise equal. Observedsmoke wall drops~32%, under differing slot occupancy;
this is not a paired hardware speed/energy benchmark. Reference fullreplay
rows remain available. No claim of superiority to dense language controls.

Both saved original10M teachers now resume exactly from73728targets/288steps,
and private optimized full replay starts10M training with1M finalDEV. Shared
replay and both factorized10M controls stay queued, max3guardedCPUslots.
Restart accounting for teachers: at most4095 extra uncheckpointed targets
may have been computed before the controlled interruption; exact discarded
work is unknown, notzero. Final successful-work estimates must be presented
with this <=0.041%-of10M extra-target bound, alongside raw observed lifecycle
wall. Original full1M initialDEV versus new1025-char initialdiagnostic also
makes startupwall unequal; final1MDEV quality/data remain matched.
