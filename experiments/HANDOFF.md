# Session handoff — 2026-09-30

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
