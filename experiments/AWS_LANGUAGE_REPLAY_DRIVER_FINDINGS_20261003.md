# Actual causal language replay driver admission

Six actual private/shared×teacher/factorized/full-replay driver contracts pass
in238.020s/514592KiB. Each executes three16-target chunks: first/warm traced,
third untraced. Interrupted after first update, all modes resume to exact
weights/Adam/cursor/RNG/replay counters and final work/curve equality.
Upstream double every-parameter sequential replay/state/RNG contracts remain
unchanged; the sibling fixes the traced-loop bypass identified by the other
host. This is numerical/resource admission only. Tiny-data BPC is not language
advantage evidence; quality comparisons require completed10M-char fits.

Allrows below have p4/L8/H2/pool2, same48 text8 fittingtargets/onepass,
16-target updates/lr.002/warmup32 and33-char disjointDEV (officialtest untouched).
Whole-fit and per-target use2FLOPs/MAC+unit-special conventions; all simulated
clocks, gradients, actual shadows and Adam charged. No hardware projection.
Inference is measured percharacter, not perprefix. Traffic/RNG/energy separate.

| Family/credit | Fitting targets | Whole-fit GFLOPs | Per-target MFLOPs | Inference MFLOPs/char | Actual shadow lanes |
|---|---:|---:|---:|---:|---:|
| private/teacher | 48 | 0.002514 | 0.052384 | 0.010616 | 0 |
| private/factorized | 48 | 0.002687 | 0.055975 | 0.010616 | 0 |
| private/replay | 48 | 0.364215 | 7.587809 | 0.010616 | 1536 |
| depth/teacher | 48 | 0.002400 | 0.050005 | 0.010616 | 0 |
| depth/factorized | 48 | 0.002573 | 0.053596 | 0.010616 | 0 |
| depth/replay | 48 | 0.364101 | 7.585430 | 0.010616 | 1536 |

First012500Z diagnostic stopped at state-object comparison ambiguity after
matched actual teacher full/recovery scores. Retry013300Z compares native
state fields explicitly, allsix pass. Failedscript/log and complete teacher
rows retained; no changed training implementation hidden behind a test fix.

Original10M teacher controls paused and archived at73728targets/288updates
for a language-scoped publisher recovery: upstream legitimately changed an
unrelated completed DVS-reg dependency in the mixed matrix. Original language
sources still match exactly. Immutable snapshots inresults/aws_depth8_language
have recovery_archive suffix; live checkpoint paths retained for exact resume.

Next admissible optimization reuses factual winning-route returns, preserving
all alternatives but halving pool2 shadow lanes/events. Independent gradient/
state/RNG/work and actual driver recovery plus production learning smokes are
required before long use. No winner-reuse measured gain yet.

## Factual winner return reuse: equivalent gradients, less work

Completed013700Z proof: all four private/shared ×1/3-token cases match EVERY
parameter gradient to at most3.25955e-15 absolute error in double precision.
Factual logits, private state and factual end RNG bitwise identical. All
categorical candidate returns remain present. At pool2 the factual winner's
detached return replaces its redundant shadow; only the losing shadow executes.
Lanes/events exactly halve in every case. No route sampling or architecture
substitution. Actual forced losing writes and firsttime preservation retained.

Same private p4/L8/H2/pool2 three-token diagnostic; complete operator coverage,
2FLOPs/MAC plus unit-special conventions for BOTH rows:

| Replay implementation | Targets | Forward+backward GFLOPs | Forward+backward MFLOPs/target | Shadow lanes | Shadow events |
|---|---:|---:|---:|---:|---:|
| Original all-lane replay | 3 | 0.004387707 | 1.462569 | 96 | 288 |
| Factual winner return reused | 3 | 0.002269611 | 0.756537 | 48 | 144 |

Measured work ratio0.517266:48.2734% less forward-plus-backward work.
These rows exclude normalization/clipping/Adam, inference and traffic/RNG/
energy; those costs are not zero. This is an isolated implementation-work
reduction with gradient/state equivalence, not completed language quality or
whole-fit supremacy. Actual optimized driver recovery and p16 full fitting
work/learning checks precede10M use. Original full replay evidence retained.

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
