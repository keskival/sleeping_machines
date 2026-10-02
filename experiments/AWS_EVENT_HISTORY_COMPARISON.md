# Fresh occupied-capacity confirmation and learned history controls

Completed source64 development means are99.251% shared/common and97.396%
private/common. Whole fitting arithmetic is~1.034 versus~1.324GFLOPs, or
~.505 versus~.647MFLOPs/query; inference is~.107MFLOPs/query for both.
Preserve all three seeds and the equal-exposure/unequal-total-data limitation.
These are development results; this new protocol tests fresh data.

Score4096 new queries from generator seed4201 after fixed development selection.
Keep native source64 checkpoints frozen and reuse all six. Verify public/checkpoint
settings, source hashes, fitting/development hashes and exact selected dev score
before confirmation. Never silently refit. Charge the original whole fitting
work and wall time; new evaluation is separate. No weight tuning from confirmation.

Fit two diagnostic history decoders, widths32 and128, for seeds6/7/8. Each keeps
up to three observed marks/timestamps per source. At an observed query flag it
forms nine features: three marks, three elapsed ages divided by10, three validity
bits. A Linear/GELU/Linear shared decoder predicts four classes. It knows the
three-mark history bound but not the labeling rule or generator time constants.
This is a strong task-specific learned control, not an integrated sparse-race
architecture: no computational races, key discovery or counterfactual credit.
Do not promote it into the research architecture.

Same fit512 queries from seed1201, dev1024 from2201, four passes, lr.003, U64,
per-seed population shuffles, clipping1 and Adam as native. Exactly four dev
selection opportunities. Report BOTH widths and every seed; no best-width,
best-seed or ensemble selection. Query-only decoding is allowed by the explicit
input query flag; mark storage, elapsed-time features and full fitting are charged.
The native model computes through all events; that difference belongs in the
resource comparison rather than being hidden.

All checking stages finish before final comparisons. Six control optimizer/
next-update recovery contracts, six accounting smokes, six native restoration
checks precede twelve final results. Control target/time/source invariance tests
pass read-only; numeric optimizer checks run only under run_safe. Prerequisite
steps are counted separately from fitted-model work and included in whole-job
wall. Operator coverage must be complete. Three guarded CPU slots,4GiB RSS/
6GiB VMS each,8GiB available floor,1800s cap. Native frozen eval extends an
existing measured1024-query evaluation fourfold; this margin is conservative.
No new long native training or core source changes.

Prespecified contrasts: shared versus private native, shared native versus
history32, shared native versus history128. Cross training seeds and paired
whole confirmation populations; use98.33% descriptive accuracy intervals for
three contrasts. The intra-family resource gate requires shared/native private
accuracy lower difference bound≥−1pp and every shared/private whole fitting
arithmetic ratio≤.85. This1pp tolerance is declared before reading seed4201;
report NLL and actual accuracy differences alongside the gate, never call it
quality equality. This can establish only the scoped intra-family tradeoff.

The history controls test the broader advantage claim. Report their complete
quality and arithmetic/special/wall/state ledgers beside native. A cheaper
control reaching native quality prevents a general resource-supremacy claim
on this task. A native accuracy lead with higher work is a frontier point,
not automatically a resource advantage. No real-data or physical-energy claim.
Further architecture changes require a fresh confirmation protocol.
