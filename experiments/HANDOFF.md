# Session handoff — 2026-09-30

## Live transition — 1 October, 07:43 UTC

Repeated-arrival campaign completed: m4 3.770751 bpc / 20.913599 GFLOPs,
.007978 bpc better than nested m1, below the .02 promotion gate. Four historical
value arrivals cost4.57% extra whole fitting work; keep this positive mechanism
result beside its modest quality gain. m2 was worse. No larger repeated-arrival
fit was launched. Results auto-published in153ea8a.

The credit campaign started its guarded frozen H2/H4 diagnosis at07:42:30 UTC;
next full64-credit contracts, smoke and2K fit follow serially. Supervisor/tmux
`local_language_credit_campaign_20261001T074000Z` is active. Preserve all its
source/queue hashes; do not launch another trainer. Inspect status before edits
or work. Its short-credit/U64 8K uses the earlier unused benchmark definition.

## Current state — 1 October, 07:22 UTC

The prioritized head campaign completed both 8K fits, then stopped at its
predeclared quality gate before 32K/131K: H2 3.485055 bpc / 79.952694 whole-fit
GFLOPs; H4 3.542576 / 193.750653. Both are worse than the saved 8K single-head
indexed control 3.357342 and receiver 3.310618. H4 also increases total width.
The source/channel construction changes with the head model, so do not call
this an isolated head ablation. Preserve all historical evidence. Completed
stage results were published automatically and committed on main overnight.

The repeated-arrival supervisor is running in tmux
`local_repeated_arrivals_after_heads_20261001T015000Z`. Full m1/m2/m4 numerical,
optimizer/recovery and accounting smokes passed under the guard. Exact m1
nesting confirms reuse of its completed reference. m2 2K completes at 3.819845
bpc / 20.495032 GFLOPs versus m1 3.778729 / 19.999171; no gain. m4 2K is running;
its intermediate scores remain excluded from benchmark tables. The supervisor
will run one 8K follow-up only if a completed pilot gains at least .02 bpc;
otherwise it stops. Inspect live status/results before another launch.

Host is CPU-only, ~11GiB available; only one trainer, ~746MiB RSS. Active
repeated-arrival model/driver/helper and original head dependencies stay frozen.
Next priority: diagnose the multihead quality failure with saved checkpoints,
including candidate-score/credit fidelity, source/channel recurrence and state
conditioning. Any frozen intervention is a diagnosis, not a refitted benchmark
or a new superiority claim. Use a guarded serial queue after the current campaign.

## Active campaign update — 1 October, 01:21 UTC

All three accumulated-optimizer pilots completed and auto-published through
commits 7921ce6/a41a5b3. U64/lr.002 gives 3.732586 bpc / 22.753030 GFLOPs;
U64/lr.004 gives 3.800302 / 22.750429; U128/lr.004 gives 3.778729 / 19.999171.
The declared 0.05 bpc quality tolerance selected U128/lr.004: 27.9% lower whole
fitting work than the original U16 reference, 0.046143 bpc worse than the best
pilot. The selected H2 eight-block 8K fit started 01:20:55 UTC in the existing
supervisor/guard. ~11.3 GiB available, trainer RSS ~650 MiB, no GPU. Sources remain
frozen. H4 contracts/smoke and its pilot follow; inspect live status before any
launch. New repeated-arrival work is prepared separately and must not bypass the
single-trainer lock or displace the existing integrated head/data comparison.

The appendix now plots completed optimizer-stage costs and reports whole fitting,
per-training-target fitting and per-character inference work together for every
completed language variant/control. All update stages and losing-route credit
remain charged; LR/warmup differences and one-seed scope are stated.

## Report clarification — native learning, 1 October

The report/README now explicitly identify native on-substrate learning as a
research upside, beyond clockless inference. Theory §321 derives a conditional
readiness-scheduled credit graph, including dependency completion, local traces
and weight-version identity. Current autograd/global clipping/block-window Adam
are not a fully asynchronous hardware learner. The completed 3.191→3.096 bpc
online CPU experiment supports adaptation, not chip energy. Loihi 2 has prior
on-chip learning; novelty concerns the combined expressive temporal/sparse-credit
construction, not on-chip learning alone. All active hashed sources remain fixed. Theory §322 adds the noncommuting event-update/
waiting-flow construction and its order/time bias. Structured multi-seed order
results support it; temporal precedence alone is not causal identification.

Tabular prediction is an additional prospective sparse-routing test, with
feature-ID-preserving adapters, row-state reset, order-invariance and missingness
contracts. Boosted trees and modern tabular Transformers remain controls; no
tabular benchmark is launched or claimed. Theory §323 constructs ideal feature
thresholds using paired positive delays, then tree paths/ensembles; learning
good paths and accurate losing-subtree credit are separate open questions. Read TABULAR_RESEARCH_PROTOCOL.md.
The first optimizer pilot completed at 3.732586 bpc / 22.753030 whole-fit GFLOPs,
versus reference 3.786482 / 27.730791: a one-seed schedule result, not architecture
supremacy. It auto-published commit e9b39bf; the U64/lr.004 pilot began 00:17:47.

## Current priority — complete independent-head models, 23:46 UTC

The 21:15 single-head eight-block campaign completed. Matched 8K controls:
receiver 3.310618 bpc / 40.243259 unit-special whole-fit GFLOPs; indexed KV
3.357342 bpc / 50.006204 GFLOPs. KV regresses 0.046724 bpc; preserve this beside
the previous 2K evidence. All are one-seed development screens, not supremacy.

Running supervisor: `scripts/run_parallel_heads_overnight.py`; manifest/log
stem `local_parallel_heads_overnight_20260930T231500Z`. Started after commit
`bfbbcf8` in tmux `local_parallel_heads_overnight_20260930T234620Z`. It reused
the completed numerical/smoke/baseline jobs and began the first guarded
U64/lr.002 optimizer pilot at 23:46 UTC. Inspect tmux/processes
and its `.status.json` before launching. The baseline H2 2K pilot completed:
3.786482 bpc on 8,191 development targets, selected epoch 2; epoch 4 was
3.892307, indicating overfitting. It uses new independent spatial heads,
source-state evolution and learned channel mixing; it is not a head-only
comparison with the earlier single-head model. Baseline tag ends `225500Z`.

Frozen candidate sources: `sleeping_machines/parallel_head_race_language.py`,
`packed_episodic_race_language.py`, `experiments/parallel_head_race_language_screen.py`,
`parallel_head_accumulated_language.py`, `parallel_head_gradient_accumulation.py`
and their source-hashed dependencies. Never edit while jobs run. H2 has payload
32/head (total64), H4 total128, depth8, independent per-head Q/K/V/gates,
receiver pools and historical banks. Channels evolve until their read time;
next-layer mixing preserves separate channels. Numerical contracts and the
baseline full-gradient smoke passed. Accumulation contracts passed, including
partial-window normalization, summed-gradient comparison and exact resumed next
update. Its guarded full-gradient smoke passed before optimizer pilots (351.103155 million
unit-special fitting operations versus baseline smoke 447.656271 million; distinct
update schedules/learning rates, not a matched-quality advantage).

Campaign: compare U64/lr.002, U64/lr.004, U128/lr.004 with fixed16-character
credit; select work within declared quality tolerance, then H2/H4 8K, selected
32K, second seed8K and conditionally131K. Every stage runs via a unique one-job
queue, serial guard, VMS4,000,000KiB/groupRSS2,500,000KiB caps and minavailable
8192MiB. CPU host ~31.3GiB total/~11.8GiB available; no NVIDIA GPU. Stop on
failed contracts, source changes, poor quality or memory gate. No new dense
training locally. The superseded single-head packed ladder is prepared only;
its supervisor refuses to launch. AWS10M six-block definition remains separate.

The report's new pages 2–5 develop the general event interface, staged research upside, architectural hypothesis and full-bank
work/access scenario; scores for every key remain charged. Same context/depth
orders, constant attention arithmetic savings, winner-value logical access
savings, and counterfactual/optimizer costs are explicit. Temporal expressivity,
smaller models, event-camera suitability and useful dormant-capacity scaling
are hypotheses with stated milestones, not established frontier superiority.
Independent heads are implemented; additional within-head shared-match policies
and local scalar-credit traffic optimization remain proposed. Section320 additionally derives
fixed-query winner-local Poisson renewal: many independent softmax marks without
rescoring keys or globally resetting losing clocks. It is an unimplemented
proposal, with explicit variance, latency, traffic and gradient limitations. Read theory48,
§§313–323 and the updated root README/integrated guide.

## Current priority — eight-block content-indexed KV, 21:15 UTC

Current tmux: `local_indexed_episodic_depth8_resume_20260930T214000Z`.
Manifest/log stem: `local_indexed_episodic_depth8_20260930T211500Z`.
Supervisor: `scripts/run_indexed_episodic_language.py`. The first guarded 2K fit
started at 21:10:49 UTC; inspect its live status before any launch. Model/driver:
`indexed_episodic_race_language.py` (sleeping_machines / experiments respectively,
experiment filename ends `_screen.py`). Do not change their hashed sources.
They retain separate per-position keys/values, with learned-query content
indexing, eight sparse receiver blocks and temporal competition. A fixed
three-bit random-hyperplane index probes own/one-bit-neighbor buckets, including
uniform full-history samples; four recent positions also qualify, <=12 keys
scored per KV query. Inference reads one winner; teaching reads all admitted
values. All entries remain stored. Hash projections are charged; bucket/RNG
operations and traffic are separate, with no full-bank attention guarantee.

Guarded eight-depth content-index contracts and 129-character full-gradient /
accounting smoke completed. Causality, exact teacher/inference/chunk values,
10M clock origin, full-history eligibility, all-entry retention, bounded
candidates, query/key/value/gate gradients and exact next-update recovery pass.
Completed check/smoke tags: `local_indexed_episodic_{contracts,smoke}_D8_20260930T211000Z`.

Completed small controls, four passes / 2K fit / 2K dev / payload 32 / seed 6:
six receiver blocks 3.632968 bpc; eight receiver blocks 3.541515. Character-tail
KV gives 3.619986 (six) / 3.538588 (eight). This index stores entries it cannot
later address, so the new variant replaces that candidate prior without
discarding these small gains. The new bounded 8K promotion requires >=0.05
depth gain and <=0.10 regression of the new KV pilot versus eight-block
receiver; it does not require or assert KV supremacy. It runs a new matched
8K eight-block receiver/KV pair only if that predeclared gate passes.

The two earlier KV coordinators have exited; no coordinator remains suspended.
Their manifests record completed 2K pairs and supersession for the content
index. The old seed-7 six-block/32K checkpoint and exact recovery queue remain
preserved; it does not automatically restart after the new campaign. Longer KV
fits need packed storage and measured workload before promotion. Existing AWS
six-block receiver queues/sources are preserved as a separate comparison.

Root/experiment READMEs, roadmap, shared-model introduction and language,
parallel and online protocols now point to `experiments/INTEGRATED_LANGUAGE.md`
and distinguish the current combined model from earlier carrier diagnostics.
The report retains the strongest structured-task evidence and adds completed
depth/KV results with architectural versus emulator cost ledgers and value-read
comparisons. The content-index pilot completed at 3.553976 bpc versus receiver
3.541515: a 0.012461 regression, not a quality advantage. It scores a mean
10.07 keys per KV query and reads one winning value; the oldest selected entry
is 2,026 characters old. The 0.091453 depth gain and this small regression pass
the declared bounded 8K promotion gate. The supervisor stopped safely because
report documentation was being edited; after committing those edits, restart
the same supervisor in a uniquely named tmux session. This was done at 21:37
UTC after commit `5b6f147`: the guard skipped the successful pilot, the clean
report hook published `459b855`, and the 8K eight-block receiver started at
21:37:17. The matched content-indexed KV fit follows serially. At restart the
host was CPU-only, with 11.7 GiB available and ~404 MiB trainer RSS; the guard
retains its 8 GiB available-memory floor and 2,500,000 KiB group RSS watchdog.

The report now has separate accuracy-versus-total-fitting and accuracy-versus-
inference-work plots, sharing checkpoint IDs and a per-variant ledger. Inference
uses saved winner-only traces for ours and shape estimates for neural controls.
Solid Transformer points approximate its overlapping-window scorer (two forward
positions per scored character); hollow points are hypothetical cached decode
costs, not measured cached quality. Learned window-relative positions prevent
assuming score equivalence under cache reuse. Numerical-clock emulator costs
remain in the global graph; the KV appendix separately gives event projections.
The rebuilt PDF has 40 pages; inference accuracy/work is on page 27. All 22
completed quality/work pairs, campaign source hashes, PDF bounds/no-orphan checks
and nine report/promotion tests passed. The root README links both work graphs.

## Latest priority — per-position race KV experiment, 20:50 UTC

Depth steering: the user now requests eight event blocks. The six-depth KV
pair is preserved to completion, but its coordinator was suspended (only the
coordinator, never the active guard/watchdog) to prevent more six-depth jobs.
After its guarded child exits, terminate that superseded coordinator and launch
`scripts/run_episodic_depth8_pairs.py` in tmux
`local_episodic_depth8_pairs_20260930T210000Z`. Its separately named plan/queues
compare receiver/KV at depth eight on 2K, then gated 8K. Numeric contracts are
configuration-specific and run inside each guarded job. Eight receiver plus
eight KV races give sixteen selections; eight persistent receiver updates;
432 receiver units. The doubled delay bound is 8 × .022 = .176 < .5.
Depth six remains evidence, not the longer-run default for this KV comparison.
AWS six-depth prepared/running queues and sources remain preserved.

The user requested a more appropriate KV-cache analogue, with small data first,
and architectural FLOPs distinguished from CPU simulation. The new integrated
`sleeping_machines/episodic_race_language.py` preserves the sparse receiver
backbone and adds historical token-position keys/values at each of six depths.
All entries survive until stream reset. Up to eight matching-character and four
recent positions form a deduplicated shortlist; this is an explicit coverage
prior, not arbitrary semantic search. One historical value is read/delivered at
inference; training charges all admitted counterfactual values. Six receiver
races plus six KV races give twelve selections per character after warmup.
The extra delays satisfy the doubled causal bound. Gradients into recent cached
activations end at the sixteen-character boundary; old keys/values stay cached.

Guarded contracts `local_episodic_contracts_20260930T204500Z` and full-gradient /
accounting smoke `local_episodic_smoke_d32_20260930T204600Z` passed: causal and
chunk-identical forward, exact teacher/inference values, precise 10M clocks,
all-entry retention, candidate bounds, query/key/value/gate gradients, temporal
softmax frequencies, prediction-before-update and exact next-update recovery.
Smoke loss is not quality evidence. Do not edit these model/driver sources while
their queued jobs run; the new source hashes are separate from existing AWS.

Tmux/plan/log: `local_episodic_pairs_20260930T205000Z`. It fits receiver-only and
episodic KV variants at width 32, seed 6, 2,048 fitting / 2,048 development
characters, four passes. They share backbone initialization/data/learning rate;
extra races change RNG consumption. An exploratory >=0.03-bpc KV gain permits
a new matched 8K/8K pair. Every run uses its own one-job queue and guarded runner,
virtual/RSS caps 4,000,000/2,500,000 KiB and 8,192 MiB available-memory floor.
Completed pairs publish separate projected-event versus emulator FLOP ledgers
and inference value-read savings; physical clock/index/traffic costs are not
claimed free. Existing dense references are retained; no new dense local fit.

To prioritize this user-requested experiment, the seed-7 width-16/32K scaling
job was stopped through its guard. Its exact checkpoint at epoch 1 / 27,648
targets is preserved. No coordinator remains suspended. The original manifest
now names a unique recovery queue with --resume; the KV supervisor resumes
that scaling ladder only after the small pair and any justified 8K pair. It
stops for review on failures instead of blindly launching more training.

The separate online-backbone 8K experiment completed: frozen 3.190859 versus
online 3.095738 prequential bpc, 512 block-delayed updates. Its initial publishing
hook correctly preserved overlapping report edits; manual combined publication
with context/query explanations committed `85e308b`. The report has 35 pages
before completed KV evidence adds an appendix page. Current references to the
older follow-up manifest being active are historical; inspect the latest tmux.

## Priority update — larger messages and AWS 10M, 19:35 UTC

The user requests the larger integrated model for larger-data runs, conditional
on a useful small matched capacity test. Use payload 32/depth 6/pool 2, compare
at 32K against the completed payload-16 result (3.120653 development bpc), then
promote to 131K and 1M only after the larger model demonstrates benefit.
The AWS 10M/four-pass/200K-validation/1M-test run is now defined separately in
`experiments/AWS_INTEGRATED_10M.md` and its committed AWS one-job queue. The new
resumable driver is `experiments/integrated_language_benchmark.py`; never launch
it outside `experiments/queue/run_safe.sh`. Official scores are frozen and read
only after the full fixed fitting budget. Preserve its source snapshot.

The old pool-4/8K capacity arm completed at 3.426425 development bpc, 720,035
parameters and 1,452.33 seconds. At this small data budget, doubling addressed
alternatives did not improve the leading pool-2 score (3.398284). This is not
the message-width intervention. Its report/evidence commit is `a7422a4`.
The old payload-16/131K job had already begun when the user redirected larger
runs. It was stopped through its guarded runner after preserving the early
checkpoint/log/result progress; it has no completed score. The old coordinator
was held during this change, then released to record the stop and exit. No
coordinator remains suspended. The old manifest is superseded for large runs.

Width-32 guarded smoke `local_integrated_benchmark_smoke_d32_20260930T193500Z`
completed. Configuration-specific checks passed: causal forward, identical
teacher/inference values, precise 10M clocks, prediction-before-update and exact
next-update recovery from model/Adam/event state/RNG. This is smoke evidence,
not language quality. A separate online experiment is being queued with two
arms: inherited frozen weights versus all-neural-parameter adaptation, both
with persistent event memory and predictions before block-delayed updates.

The following older priority sections are historical. Inspect live queues and
the newer larger-message campaign before resuming them.

### New follow-up campaign and report ledger — 20:00 UTC

Tmux/manifest/suite log: `local_integrated_followups_20260930T200000Z`.
Its supervisor `scripts/run_integrated_followups.py` runs the separate guarded
online-backbone 8K development experiment, publishes its completed result if
layout/source checks pass, then invokes the existing guarded ladder controller
on the new manifest. AWS's separately committed seed-6 campaign is preserved.
Local capacity comparisons use seed 7: matched width-16 and width-32 at 32K,
then 131K requires an exploratory 0.02-bpc improvement over that same-seed
width-16 control. The larger 1M stage requires >=0.1 bpc gain over width-32/32K
and <=3.0 development bpc. All larger-data stages use width 32.
The superseded payload-16 long jobs do not resume automatically.

The supervisor actually started at 20:30:20 UTC. The online job holds the
host-local guarded training lock; initial RSS is about 432 MiB, with about
12 GiB MemAvailable before launch. There is no NVIDIA GPU. Inspect live state
before relaunching: completion automatically publishes online then starts the
seed-7 capacity ladder. Pending manifest status during online is expected.

The report now separates queries/race selection from memory organization.
The integrated primary model races over two compressed receiver states per
observed character/depth, not arbitrary historical token KV entries. It carries
forward state beyond the 16-character credit horizon. Width-32 raw persistent
tensors are 43.16 KiB versus a conceptual 2 MiB FP32 KV allocation for the saved
four-layer/width-256/256-character Transformer (~47×); this is not equal recall
capacity, measured RSS or a measured cache speedup. The reference recomputes
windows and does not implement a KV cache. Theory §§308–309 records this scope.

New width-32 model: 1,388,871 parameters; still 324 units / six selected state
updates. Theory §§303–307 explains the width intervention, the 16-versus-26
centered-logit rank restriction, causal online evaluation and conditional
scaling. These are reasons to test, not a prediction of frontier supremacy.
The index now links current theory notes 44–47.

The online protocol adapts all neural parameters on [90,065,536,90,073,728),
making each block's predictions before parameter updates; feedback delay/credit
is 16, fixed learning rate 0.0001, fresh Adam, paired race noise. Frozen and
online arms share the selected width-16/32K checkpoint and both retain event
memory. Online smoke v1 finished computation but failed at result assembly
because of a variable-name error; its progress/logs are preserved. The corrected
v2 smoke passes and is not used as quality evidence in the report.

Actual end-to-end recovery passed: the guarded width-32 probe stopped in epoch
two at 64 targets and resumed. Its complete curve, final score, parameter
changes, random counts and work ledger exactly equal its uninterrupted control.
Completed comparison JSON: `local_integrated_recovery_comparison_d32_20260930T194000Z.json`.
Do not edit the new protocol/benchmark/online drivers while their jobs run.

The PDF adds accuracy-versus-whole-fitting-FLOPs panels for integrated ours,
earlier carrier controls, LSTM and Transformer, plus every plotted variant's
capacity, data/passes, quality split and whole-fit work. Development/test panels
are separate; point numbers map to the detailed table to avoid overlapping
labels. Costs consistently use unit-weight specials, with arithmetic-only
integrated totals preserved in the earlier appendix. New online curves and
complete adaptation cost are included only from completed result files.
Incoming AWS setup commits `545b771`/`95e6b41` arrived through a host-side
autostash pull during report editing. The result-discovery conflict was resolved
by retaining explicit local/AWS globs plus the independent online-result loader;
AWS runner/queues/fingerprints remain preserved. Its model/driver source hashes
match this checkout. Avoid overlapping report edits/publishing across hosts.

## Priority update — integrated architecture, 17:54 UTC

The user explicitly prioritizes full architectural experiments over carrier-only
or hybrid half-measures. Active tmux/manifest/log:
`local_full_sparse_ladder_20260930T175400Z`. Inspect its live state first.
It runs one guarded job at a time: full sparse payload-16/depth-6/pool-2 fits at
8K then 32K characters; pool-4 capacity comparison at 8K; gated 131K then 1M
data stages. All use four passes, seed 6, 8,192 cold development characters,
16-character truncated credit and no official test. Gates stop weak learning
for diagnosis, rather than promoting automatically to a long benchmark.

`sleeping_machines/sparse_race_language.py` has no dense language carrier.
Content/state keys set exponential clocks; a winner mixes incoming content and
persistent state and emits a timed value at each depth. Exactly six receiver
states update per character from 324 available units (648 in the capacity arm).
Two/four keys per depth are scored. Training evaluates addressed losing values
for a conserved centered score teacher, and charges those reads. The fixed
character index, bounded-delay graph and local surrogate are declared limits.
No learned topology growth, complete sparse attention equivalence or measured
hardware energy is claimed. Theory §§299–302 defines the integrated contract.

Completed contracts: `parallel_language/local_sparse_contract_20260930T175000Z.json`.
Causal predictions, identical chunking and large-origin execution, equality of
training forward and winner-only inference, key/value/time/retention gradients,
sparse state updates and conserved teacher passed. Smoke:
`local_sparse_smoke_20260930T175000Z.json`, 1,024 fitting characters, one pass,
payload 8/depth 3, 5.482 to 5.140 development bpc; 35.872M estimated fitting
arithmetic. This is implementation verification, not benchmark quality.

The earlier carrier ladder is `paused_for_integrated_architecture`.
Width-128/1M completed at 2.210279 development bpc, 8.324920T fitting arithmetic,
2,151.37 seconds; report commit `1d9f72e`. Width-256/1M was paused in epoch 2
at the 245,760-target checkpoint, preserving exact sources/settings/checkpoint.
Its old coordinator was stopped and removed after its guarded child exited.
Do not mistake the old manifest's official queues for active priority.
The user requested that the dense control evidence remain preserved.

The hybrid indexed retrieval candidate and softmax counterpart remain deferred
diagnostics: `race_language.py`, `race_language_screen.py`; both pass contracts
and 4K-character smoke fits. The first smoke stopped safely on an unsupported
matrix-vector audit formula; corrected v2 adds explicit 2-FLOP/MAC matrix-vector,
dot and outer-product formulas. Original failed logs/checkpoint are retained.
No failed smoke score is a completed benchmark result.

Historical temporal softmax is an exact choice-probability identity, not proof
of free whole attention. §102's shared-clock covariance bound and clock-cutoff
scope were incorrect; explicit corrections sit beside the originals, and
§§294–298 derives the centered conserved teacher. Preserve all earlier valid
structured-task comparisons. The breadth table's eight-layer versus two-layer
training ratios do not describe the larger language comparison. FLOPs, seconds
and physical joules have distinct boundaries.

Do not edit any contract-locked model/driver sources during this ladder.
Report hooks commit completed stages on main only, after layout/source checks.
Keep report edits committed before a stage ends. The report now makes the full
ambition and mechanism coverage explicit, with a diagram of capacity versus
selected activity. Push commits from the authenticated host.

The sections below preserve the previous campaign/history; this priority update
supersedes their descriptions of what is active.

### Completed first integrated stage — 18:11 UTC

`local_full_sparse_language_D8192_p2_20260930T175400Z.json` completed:
361,367 parameters; 324 units; six selected states per character; development
5.340124 initially, then 3.728912, 3.535521, 3.448233, 3.398284 across four
epochs. Selected epoch 4. 8,191 cold development targets. Representative
fitting arithmetic: 7.492246G (forward/loss 1.233630G, backward 2.521473G,
clipping 0.748783G, Adam 2.988360G). Representative inference/scoring:
25,027.375 arithmetic FLOPs per character. Wall time 829.48 seconds.
These are completed development-stage results, not matched official supremacy.
The old 3.351 language pilot uses a different development interval/window,
warm context and credit horizon; do not call those quality/runtime comparisons
matched merely because both fit about 8K characters.

The report hook safely stopped because a diagram edit overlapped this completion.
The edits were committed and the completed stage was published manually from
the clean tree (`a99a403`), validating the 30-page PDF. The same ladder resumed
at its next job, full sparse pool-2/32K, at 18:10:55 UTC; one guarded trainer
is active. Source contracts are unchanged. Core setup/report commits:
`fe0a451`, `aaa2a8f`, `a99a403`. The user pushes from the authenticated host.
Future completion hooks should see clean report files; avoid overlapping edits.

### Continuity guidance and evolving work appendix — 18:30 UTC

`AGENTS.md` now explicitly guards architectural continuity, evidence-based
substitutions, integrated-experiment priority and clear presentation of strong
results. It requires the evolving completed-stage cost/quality appendix.
The PDF has 31 validated pages. The new appendix records full fitting work,
per-target fitting work, inference arithmetic and saved neural reference costs.
Graphs/ratios use arithmetic plus one unit per special function, matching the
historical neural estimate convention; the detailed ours table keeps specials
separate. The first completed stage's raw configuration gaps versus the larger
10M Transformer are ~291× forward and ~93× fitting work per target. Quality,
targets and data budgets differ; these are not matched-quality supremacy or
physical energy claims. The appendix updates only from completed result JSONs.
Full pool-2/32K remains active; after two completed epochs it scores 3.251387
development bpc. This is live epoch evidence, not a completed stage to publish.

Work on `main`. The user authorized committing and pushing all work, wants one
presentable PDF, and intends to start a fresh session. Preserve the established
architecture, theory and historical results; extend them with new evidence.

### Completed 32K stage and consistent cost columns — 18:58 UTC

Full pool-2/32K completed at 3.120653 development bpc after four passes;
29.883299G fitting arithmetic, 31.053267G including unit-weight specials,
2,815.51 seconds. Its automatic report commit is `09701d1`. Pool-4/8K capacity
comparison is now active; the 131K data stage follows, with its gate satisfied
by the 32K quality and gain. Inspect live manifest before changing the queue.

The user correctly flagged a misleading appendix layout: ours whole-fit GFLOPs
sat above reference per-target MFLOPs in separate tables. The evolving appendix
now places all completed integrated stages and both 10M controls in one table,
with identical units and denominators in each column: whole-fit GFLOPs, fitting
MFLOPs/target, forward MFLOPs/position. All use unit-weight specials consistently
with the chart; arithmetic-only totals remain explicit. The 32K stage is
0.236925 MFLOPs/target versus LSTM 7.210099 and Transformer 22.223084. Different
quality/data/model sizes still preclude a matched-quality supremacy claim.
`AGENTS.md` now requires this column consistency for future comparison tables.
The accompanying visual has three aligned panels: whole-fit GFLOPs, fitting
MFLOPs/target and forward MFLOPs/position, with fitting budgets in model labels.
Older language capacity evidence also exists: carrier widths 32/64/128/256;
E64 LSTM widths 256/512; Transformers width 112/depth 8 and width 256/depth 2/4.
These have different fitting budgets, passes and protocols. Do not call the
8K-to-32K integrated data ladder a capacity-scaling curve or treat the older
carrier sizes as integrated sparse/timed architecture results.

## Report and publishing

- Canonical PDF: `report/sleeping_machines_status.pdf`; Markdown: `REPORT.md`.
  Regenerate both with `.venv-docker/bin/python report/make_pdf.py`.
- The revision restores all four opening differentiators, explains computation
  through time and counterfactual learning, restores visual comparisons, and
  removes the redundant opening reference table. Complete neural language
  references remain in Appendix B; older revised claims remain in Appendix C.
- Saved text8 test scores: 10M LSTM 1.799, 10M Transformer 1.908, AWS 90M LSTM
  1.661 bpc. Estimated full training costs: 432.59T, 888.78T, 3.89P FLOPs.
- The completed persistent learned-language pilot is 3.351 development bpc,
  28,403 parameters, 8,192 fitting characters and four passes. Estimated event
  arithmetic: 6.962G total fitting FLOPs and 59.741K inference/scoring FLOPs per
  character, with special functions separate. It is not the full benchmark.
- E79's old 1.613/1.504 statistical language headlines had target leakage; E173
  corrects the 10M score to 1.727, or 1.719 with causal word context. Old market
  thresholds used evaluation days. Retain the raw records and stated errors.
- Publishing from this container has been unavailable: origin uses SSH, the
  SSH executable and credentials are absent, and the connected GitHub app
  rejected blob creation with 403. The user has pushed previous commits from
  the host. HTTPS reads work. Verify `git status` and `git log` for the new local
  report commit, then push from an authenticated host. Do not claim it is remote
  until the remote commit is verified.

## Running benchmarks — inspect the live state before resuming

Language has priority; speech remains paused from epoch 12, example 6,144.
Its latest completed epoch scored 1,020/1,169 development utterances (87.25%),
not an official test. Its original sources and checkpoint are preserved.

The original 10M sequential language run was also paused, preserving its
checkpoint. Float32 absolute token positions erased its sub-token delays as
positions grew: at 1,000,000 the spacing is 0.0625, larger than every delay.
The replacement keeps clocks in float64 and payloads in float32. A bounded-delay
contract permits causal affine scans across a chunk, preserving persistent
state and checked predictions/gradients. Original model files remain unchanged.

- Precision fix and driver commit: `970b854`.
- Width-256, warm-state numerical contracts:
  `experiments/results/parallel_language/local_parallel_language_contract_v3_20260930T153300Z.json`.
  Checks include positions 0 and 10M, chunk equality, future perturbation and
  parameter gradients. Measured complete-step CPU speedup is 12.30× against
  precise serial execution of the same model; this is not an energy advantage.
- Staged campaign: `experiments/queue/local_language_scaling_20260930T153653Z.json`.
  Tmux and log use the same stem. Capacity is varied at fixed 131,072 fitting
  characters; data is varied at fixed width 128. All use six layers, four passes,
  seed 6 and the same 8,192-character development window. Only the eventual
  fixed 10M/200K/1M run may read the official test.
- Completed small stages: width 32, 21,741 parameters, 2.858 development bpc;
  width 64, 80,301 parameters, 2.727. These are exploratory, not official tests.
- Width 128 also completed: 308,013 parameters, 2.643 development bpc. The old
  orchestrator was terminated after that job completed. Its remaining queues
  are superseded, since model-driver contract sources have since changed.
- Completed tmux: `local_language_memory_20260930T155000Z`; manifest and suite log
  are in `experiments/queue/` with the same stem. Three width-256 numerical
  contracts passed for inherited, long-decay and long-spectrum initialization.
  The campaign fits those three profiles serially at width 128, fixed 131,072
  characters/four passes and identical development targets. No official test.
  All three fits completed: inherited 2.643, long decay 2.752, long spectrum
  2.858 development bpc. Retain inherited as the leading matched result.
  Longer modal retention alone worsened this screen. Investigate content-aware
  write/forget selection in a small matched fit before larger promotion.
- Completed tmux: `local_language_selective_20260930T161050Z`; manifest/log use
  the same stem. Three new numerical contracts passed, including nonzero
  input-dependent controls and their identity initialization. The first gated
  width-128 fit completed at 2.586650 bpc, 309,561 parameters and 1,040.61G
  fitting arithmetic, versus 2.643410, 308,013 and 1,026.18G for constant memory.
  The longer-decay gated arm also completed at 2.626911; inherited spectrum
  remains the leading matched model. Frozen representation audit completed:
  `parallel_language/local_language_representation_20260930T162337Z.json`.
  Reset history/current content unchanged: 4.519 bpc; zero embeddings: 7.569;
  full gated model: 2.587. These are fitted-dependence interventions, not
  retrained architecture comparisons.
- Active staged campaign: `local_language_nextscale_20260930T163234Z`, launched
  at 16:38 UTC; manifest and tmux/log use that stem. It reuses completed gated width-128/131K evidence,
  fits width 256 at 131K, then both widths at 1M. Only after both primary stages
  finish does it select the smallest width within 0.03 bpc of the best. A <=2.25
  1M score and >=0.1 fixed-width data gain are required to run exactly one
  selected fresh 10M/200K/1M comparison. Unselected official queues are retained
  but never executed. Practical gates cannot guarantee superiority.
  Width-256/131K completed at 2.572493 bpc, 1,208,889 parameters and 4.009T
  fitting arithmetic; its report hook committed `d6c7c55`. Width-128/1M began
  at 16:44 UTC and is the current guarded job. Live monitors are not results.
- `scripts/update_language_report.py` rebuilds/validates the report after each
  completed training stage and commits completed results/artifacts on main.
  It never pushes remotely, never publishes live scores, and refuses to mix
  existing staged changes or overwrite report edits. The authenticated host
  can push each resulting commit. Failed guards or report hooks preserve
  results/checkpoints and stop the pipeline for review.
- The inherited event initialization has mostly sub-character modal timescales.
  `sleeping_machines/language_memory.py` adds explicit token-unit alternatives;
  `experiments/theory/44_precise_language_scans_and_content.md` derives the
  schedule, precision and content-mixing contracts. The input vector is
  retained, mixed with memory, gated and passed through a residual. The new
  `selective_stream_language.py` candidate additionally learns input-dependent
  scalar write and forget controls; the preceding model remains a distinct
  constant-memory baseline. Current language models still activate every layer
  for each character. No measured energy or general dormant-unit claim.
- All jobs use unique one-job queues and `run_safe.sh`; caps are 4,000,000 KiB
  virtual memory, 2,500,000 KiB group RSS and at least 8,192 MiB MemAvailable.
  The host is CPU-only. Do not train new Transformer/LSTM controls here.

The user pushes reviewed commits from the authenticated host. The front page
now leads with completed order-learning and retrieval comparisons. Keep broader
language superiority pending until completed, comparable test and work evidence.

### Original suite record (superseded)

- Former tmux: `proper_events_20260930T131028Z` (stopped).
- Manifest: `experiments/queue/local_full_proper_suite_20260930T131028Z.json`.
- Historical suite log: `experiments/queue/local_full_proper_suite_20260930T131028Z.out`.
- One guarded job at a time, through `experiments/queue/run_safe.sh`; never
  launch Python training directly. Current job: full SHD, 20 epochs, seed 6,
  from scratch, 395,814 learned parameters, 6,987 fit / 1,169 development
  utterances. At this handoff it had reached epoch 11. Official test runs once
  after development selection; ongoing development scores are not final tests.
- Next jobs, serially: DVS, MNIST, market, temporal composition, learned
  language. The language job has six layers, width 256, 128 temporal modes,
  1,205,805 parameters; 10M fit characters, four passes, 200K development and
  the existing 1M test interval. No count/copy/word experts.
- CPU-only host. Suite caps: virtual memory 4,000,000 KiB, RSS 2,500,000 KiB,
  minimum available memory 8,192 MiB, timeout 864,000 seconds. Available memory
  at handoff was about 11.2 GiB and only one training process was active.
- Progress/checkpoints are ignored local files under `experiments/results/`.
  Final result JSONs are versioned. Check the live state before deciding whether
  a job needs resumption; the queue and tmux process may still be running.
- Do not train Transformers or LSTMs here. The user reserves new dense controls
  for AWS; reuse existing reference results. Update the report from completed
  comparable results without deleting older evidence.

The original suite remains a historical record. New precise-clock and selective
memory sources are separately versioned with their numerical contracts. Never
resume an old queue after source changes without recovering its exact source
revision; never overwrite completed results or silently reinterpret old metrics.

Report/setup commit: `c5fb85d`, following `8675098` (context/visuals) and
`55b31e8` (selective memory). The report has 26 pages and a front-page 131K
development result alongside the preserved 8K pilot. Numerical contracts,
representation interventions, PDF bounds/no-orphan checks and nine focused
report/promotion tests passed. Frontier language quality and physical energy
remain unestablished; the active campaign is building the next evidence.

After the first larger stage the PDF has 27 pages. It now visualizes the
allocation comparison at identical 131K data/four passes: content gates improve
0.057 bpc for 1.41% more fitting arithmetic; widening the gated model improves
another 0.014 bpc for 3.85× total fitting arithmetic. This is a local finite
comparison, not a scaling law or dense-model superiority. Keep the report
working tree clean before its automatic post-stage hook. Bounds/no-orphan and
nine focused checks passed after the latest rebuild.

## Deferred integrated repeated-arrival trial — 1 October, 01:50 UTC

Completed optimizer evidence and consistent per-target fitting units were
committed in `57d8b65`. Current H2 8K stage remains guarded; first-pass frozen
8K development is 3.498321 bpc. This is an intermediate score, not the completed
fit or a new report benchmark. Inspect its status and log before any launch.

New sources `repeated_arrival_race_language.py` / `repeated_temporal_race.py`
add m historical arrivals per head with one set of key/query matches. Only the
winning emitter renews. Receiver races, independent spatial Q/K/V heads, depth8,
persistent state, sparse index and counterfactual credit are retained. Read
THEORY §324 for the failure addressed, temporal aggregation, conserved
O(Cd+md) teacher and cost/physical-clock limitations. m=1 exactly nests the
existing model. Seven read-only numerical/operation checks passed; with report
and scaling checks, fourteen tests passed. Full optimizer contracts and smoke
fits must run under the host guard before pilots; they have not completed yet.

Prepared manifest: `experiments/queue/local_repeated_arrivals_after_heads_20261001T015000Z.json`.
Supervisor: `scripts/run_repeated_arrivals_after_heads.py`. It waits for the
prioritized head campaign's terminal state, not merely an unlocked moment.
It proceeds after completion or the exact declared 8K head-quality stop; other
errors require review. It must not displace or edit the active head campaign.
Ten unique one-job queues cover m1/2/4 contracts/smokes, m2/4 2K pilots and two
alternative 8K promotion definitions (only the selected one may run). H2,
d32/head/depth8, U128/lr.004, four passes/seed6, disjoint 8K dev. Reuse the
completed m1 reference only after nesting checks. Minimum .02 bpc pilot gain
required for one 8K fit; no 32K/131K promotion without reviewing that evidence.
Caps remain VMS4,000,000KiB/groupRSS2,500,000KiB/minavailable8192MiB; CPU-only.

Report ingestion distinguishes `/m2` and `/m4`, keeps them out of optimizer-
schedule and independent-head-only comparisons, and adds a matched appendix
cost/quality table only after completed fit records exist. Pending cells contain
no predicted results. Winner deliveries, local renewals, candidate teacher
reads and numerical minimum comparisons have explicit counters. Clock costs,
RNG, discovery/traffic and physical energy are not erased by FLOP projection.

After commits `2101d70`/`735dc55`, the waiting supervisor was started in tmux
`local_repeated_arrivals_after_heads_20261001T015000Z` with the preserved command:

```bash
tmux new-session -d -s local_repeated_arrivals_after_heads_20261001T015000Z '.venv-docker/bin/python scripts/run_repeated_arrivals_after_heads.py >> experiments/queue/local_repeated_arrivals_after_heads_20261001T015000Z.out 2>&1'
```

Check its same-stem `.status.json`; `waiting_for_prioritized_campaign` means no
new training has started. Do not start another instance or change frozen sources.

## Prepared diagnosis and longer-credit ladder — 1 October, 07:40 UTC

Read theory §325 before continuing. The failed 8K head gate constrains this
implementation. A verified numerical limitation is that the 16-character
boundary detaches older KV producers even though their contents are retrieved;
forward history and learning history are different. Extending credit to64
retains temporal races, addressed receiver state, independent Q/K/V heads,
full indexed history and counterfactual learning. It changes graph lifetime and
training cost, not fixed-weight inference. It is truncated backpropagation,
not a new architecture/novelty claim or unlimited long-history credit.

Prepared manifest `experiments/queue/local_language_credit_campaign_20261001T074000Z.json`;
supervisor `scripts/run_language_credit_campaign.py`. Started after `a9a90cb`
in tmux with the same stem; currently waiting. Do not start another instance. It waits for completed repeated-arrival campaign,
then runs every stage through unique `run_safe.sh` one-job queues. Order:
frozen saved-checkpoint diagnosis; full64-credit gradient/update contracts;
129-character full-configuration smoke; matched2K64-credit/U64/lr.002 fit;
existing unused16-credit/U64/lr.002 H2 8K queue. A .02 bpc pilot gain admits
one64-credit8K run. Best completed8K must satisfy the existing indexed-control
+.10 gate before32K; secondseed8K follows.131K additionally requires32K gain
>=.05, projectedRSS<=2.2MKiB and timeout derived from measured32K wall time,
with48h maximum. VMS4MKiB/groupRSS2.5MKiB/minavailable8192MiB; CPU-only.
Reuse strongest completed U64 pilot, not assumed success of cheapest U128.

Two new read-only contracts pass: detached producer credit versus identical
forward content, and frozen replay/intervention parameter/hook integrity. With
previous checks, sixteen focused tests pass. Full optimizer64-credit contracts
and fitting remain pending until the serial guard admits them. The diagnostic
conditions on one race time/continuation seed for24 local replay probes and
uses a256-target window; it is not a full expected gradient or refitted model
benchmark. Publication adds only completed results and separates credit spans
from optimizer/head comparisons; source hashes/queues are frozen before launch.
