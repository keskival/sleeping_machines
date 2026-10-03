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
