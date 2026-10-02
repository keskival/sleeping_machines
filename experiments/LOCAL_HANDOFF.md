# Local host: current research continuation

## curie: DVS capacity ladder claimed (§396), 2 October, 16:40 UTC

curie runs a capacity ladder on the other host's clock-calibrated DVS driver (same data, controls, seed and
protocol as local_dvs_clock_full_20261002T153000Z): pool 8, depth 4, payload 32, combined; tags curie_dvs_*;
queue curie_dvs_capacity_20261002T164500Z after the joint v2 observed fit (its rank control is withdrawn: the joint
task family is solved by stateful tables, so it is a capability diagnostic only). No change to the DVS sources;
the other host's DVS inference/port work is not duplicated.

## Full-window batching admission passed, 2 October16:35 UTC

Completed report refreshed and validated:121pages, publication
`results/diagnostics/local_dvs_completed_stages_report_20261002T163900Z.json`.
Both original and clock-scaled full fits and both bounded batching smokes appear
in the appendix with common work units, quality, data and activity; all old
negative evidence and strong controls retained.

Both unique guarded `local_dvs_batched_{local,pairs}_smoke_20261002T163400Z`
fits completed,24fit/8dev/two fixed passes. Each has two U16 and two partial
U8 updates,48 target presentations; every stage's operation accounting passes,
all frozen source SHA match and fitting NLL falls by more than.01. Admission
record `results/diagnostics/local_dvs_batched_smoke_admission_20261002T163600Z.json`
contains same-unit work, wall and RSS for both. Peak memory remains below
900000KiB. This is integrated learning/accounting admission, not quality evidence.

Next priority: bounded matched local/pair credit pilot at fixed data, seed,
capacity, optimizer windows/passes and selection rule before increasing native
capacity. Both smokes and all older failures remain preserved. Inference-only
NumPy timing audit is also ready to prepare from passed contracts; no speed
claim before repeated matched measurements. No job is left running at this
handoff. The rebase is complete,119-page combined report validated, all three
stashes preserved; work remains on main.

## Rebase recovered; real-stream evidence and next gate, 2 October16:32 UTC

Rebase onto `bda9f4a` completed on main. Conflicts were generated REPORT/PDF;
combined report source and all results retained. Guarded rebuild
`results/diagnostics/local_git_rebase_report_20261002T163200Z.json` passes
text-bound/orphan checks. Three stashes retained. Source SHA checks for completed
clock full fit, batched contracts and NumPy inference contracts all match.
Other-thread new count calibration and joint-event protocol work preserved.

Original integrated DVS p16/L2/H2/pool2 fit completes eight fixed passes:
65.1042%/.963161NLL,20.075193 whole-fit GFLOPs,2.550202 MFLOPs/target,
.591793 MFLOPs/inference prefix,1557.590s/356144KiB. Clock-scaled initialization
also completes:66.1458%/1.041987NLL,20.074713 GFLOPs,2.550141 MFLOPs/target,
.591779 MFLOPs/inference prefix,1565.128s/355488KiB. Both select pass8 by
minimum devNLL. Timescale change increases accuracy but worsens NLL; it does
not close the control gap. Same984 fit/192 dev users,21 observed events/clip,
7872 fit presentations/496 updates, no official-test read. Results live in
`results/dvs_native/local_dvs_{native_full,clock_full}_20261002T*.json`.

Strong selected full RBF:73.4375%/.706478NLL. Compact class-prototype33/g1/C10:
66.6667%/.902951NLL at97476 serialized bytes versus native106354. All72 compact
cells paid10.043s/188296KiB; solver FLOPs unknown. Original native sequential
inference55.076ms/prefix versus full RBF.671ms, with preprocessing included.
Compact .230ms is currently one pass only, pending repeated matched audit.
These controls defeat current practical quality/storage/runtime advantage.

Frozen native features improve fitting-selected fresh-head quality from
initial57.81%/1.131057 to selected67.19%/.999857. This supports useful feature
learning. Selected native's original head is65.10%/.963161; readout changes
alone do not close the gap. Diagnostic decoder CV is conditional on an encoder
already fitted on all fit labels, not unbiased whole-pipeline CV. Preserve
negative gates beside these positive representation findings.

Numerical inference port passes all384 original/clock dev prefixes with
identical hard routes, float32 states/logits plus double precision contracts;
causality and resets pass. No performance claim before a guarded repeated audit.
`results/diagnostics/local_dvs_numpy_contracts_20261002T160600Z.json`.
Independent-clip batched fit and optional conditional terminal pair risk pass
all-parameter/state equivalence, explicit finite-outcome derivatives and actual
local/pair driver Adam/cursor recovery:29.808s/393536KiB.
`results/diagnostics/local_dvs_batched_contracts_20261002T161500Z.json`.
Theory77 retains the integrated architecture; earlier routes still use local
surrogates, so no exact whole-core expected gradient is claimed.

Prioritized next integrated stage: batched local/pair full U16 plus partial U8
accounting and learning smokes,24fit/8dev/two fixed passes, serial unique queues.
Require complete operation coverage, lower fit NLL and RSS below900000KiB
before a matched bounded local/pair fit. No bigger campaign admitted yet.
One thread, RSS1250000KiB/VMS3000000KiB,8GiB available-memory floor. At handoff
host has11787MiB available, no GPU or active training. AWS still owns new dense
training; other local thread owns language/count/joint-event campaigns.

## Real-packet native full fit active, 2 October14:55 UTC

Combined relation/theory/confirmation report recovered successfully:113pages,
all text bounds/orphan checks pass, completed publication record
`results/diagnostics/local_joint_report_recovery_20261002T140500Z.json`.
Old failed publication logs and restored historical evidence remain preserved.

Strong DVS calibration completed on984 first-second fitting gestures/users1..19
and192 dev/users20..23,4x4/polarity/50ms closures. No official test access.
Fit-only calibrated class-count reference58.8542%/1.373369NLL; best linear
61.4583%; selected minimum-NLL RBF C10/g1:73.4375%/.706478NLL, maximum grid
accuracy74.4792%. Meaningful practical headroom:14.5833pp over counts and
all grid cells below95%. Raw preprocessing123.976s/264832KiB; entire strong
control grid15.926s/420976KiB. JSONs and checksummed data/model artifacts:
`results/dvs_calibration/local_dvs_calibration_20261002T141400Z*`.
Solver FLOPs unknown, not zero. All cells and warnings remain preserved.

Prioritized integrated candidate: unchanged AddressedEventHeads fast training,
one observed source,33 input coordinates,11 classes,p16/depth2/H2/pool2.
Temporal races, separate keys/values, sparse persistent state and unrealized
value credit retained. Counts are observed camera packets, not label-experts.
Common per-pass/fitting-seed race draws and fixed eval seed314159; identity,
index and labels do not enter randomness or event inputs. Same fitting-only
log-count normalization as controls, query flag at1s, reset each gesture.
Theory73 records admission; theory74 separates information/gradient/frontier
contracts. No exact whole-route gradient, useful-depth or dormant-capacity claim.

Numerical contracts completed: reference/fast forward and all parameter
gradients, causality/input metadata, trained inference, actual interrupted
driver/model/Adam/RNG/cursor recovery. Result
`results/diagnostics/local_dvs_native_contracts_20261002T143900Z.json`.
Full-shape smoke8fit/8dev,2epochs/U4 completes29.697s/344108KiB;
first8fit NLL2.704424→2.035225, learning gate and operation coverage pass.
Preserved `results/dvs_native/local_dvs_native_smoke_20261002T144500Z.json`.

Active frozen one-job plan `queue/local_dvs_native_full_20261002T145300Z.json`,
tmux `local-dvs-native-full-20261002T145300Z`, worker41833/trainer41852,
started14:50:41. Full984/192,8 fixed passes,U16,Adam.003,s6:7872 fitting
presentations/496 updates; min devNLL selection, no epoch extension. Timeout
7200s conservatively derives from smoke clip-call ratio. One thread/nice19,
RSS1250000KiB/VMS3000000KiB/min8192MiB guards. Observed~356000KiB groupRSS,
~11.5GiB MemAvailable. Epoch1 dev54.6875%/1.252066NLL is RUNNING evidence,
not completed benchmark or advantage. Online checkpoints align optimizer;
best-state selection saved separately. Numerical sources/theory73/calibration
and prerequisite JSON hashes remain frozen; do not edit or duplicate jobs.

Next: finish all fixed passes, run completed-only `dvs_native_analysis.py`
through a new one-job queue, compare against every strong cell in common
work units. Charge all preprocessing/validation and mark sampled native
arithmetic estimates. A development quality win needs frozen independent
confirmation and measured matching inference boundaries before practical
advantage. If it fails, diagnose information paths/conditioning/clock/credit
without weakening controls. AWS owns new large dense campaigns; other curie
thread owns language pooling/count strata/joint-recency fits. Main retained,
three stashes intact, no unmerged index at last check.

## Strong practical calibration changes priority, 2 October13:50 UTC

User explicitly requires advantage where existing solutions are not already
practically optimal. Guarded `queue/local_joint_stateful_table_20261002T134700Z.txt`
completed, result `results/diagnostics/local_joint_stateful_table_20261002T134700Z.json`.
Same planned joint-task fit512/seed1301,dev256/2301,confirmation1024/3301.
Question-string table learns a split over four causal mark ages, without
generator lexical mapping/threshold.20strings,one fitting pass,0.027026s
fit,67,940KiB RSS; dev99.61%/0.051632NLL; confirmation99.51%/0.055258NLL.
The20-point learned-control-versus-table gate cannot pass against this
reference. Keep the recency task as capability diagnosis; do not claim
advantage versus its time-blind table. Theory72 records admission priorities.

All relation confirmations completed. New seed75001/256targets: s7joint
99.21875%/0.0496041bits,local100%/0.00316024; s8joint100%/0.000147362,
local100%/0.00353496.0of2 .05-bit joint-over-local gates pass. Joint fits
remain~2.55% more arithmetic than local at unchanged inference architecture.
Value-zero accuracy98.828125/100/75/99.21875% respectively: three native
contexts learned useful nonlocal prediction without delivered values. This
does not isolate outcome-bank training benefit; no matched bare-core refit.
No generic gradient regression or general learning ceiling is established.

Rebase onto658932a completed at13:24, preserving both handoff sections;
all frozen sources verified unchanged,3stashes preserved. Theory70 alias/
conditional stationary-point/analytic-witness contracts passed0.556879s
and249,636KiB. Theory71 specializes existing full-support joint credit;
sampling remains unimplemented. Report publication first attempt stopped
on an orphan page2 after the concurrent cover expansion; oldPDF/REPORT
restored. Recovery will retain all completed negative and positive results.

Current priority after report recovery: calibrate real DVS Gesture under
subject-disjoint first-second observations against strong class-count,
time-aware linear and kernel controls. Old dense DVS controls predict the
majority class and cannot prove practical advantage. Reuse bounded raw
adapter; no official-test access during selection and no duplication of
other-host language or AWS dense control campaigns. No DVS fit launched yet.

## curie: regime calibration and milestone 3 (joint text + events), 2 October, 13:20 UTC

Small-data language is the home ground of counting. Stream-adaptive interpolated KN (continuation statistics) is
a near-optimal reference there (preliminary 2.521/2.414/2.271 bpc at 2K/8K/32K; official file queued as
curie_adaptive_kn_reference_20261002T131500Z). The report presents such references as calibration ceilings, not
competitors (§§393–394). New work follows the §394 calibration rule: strongest table, timestamp-aware learned
control, and ours. Milestone 3 task: joint text + irregular events on one native address, shortcuts removed by
construction (tests/test_joint_event_language_tasks.py). Native pilot queue: curie_joint_event_language_20261002T133000Z
(observed vs rank-trained; table bar and cleared-text control inside). Dense controls ready for AWS:
AWS_JOINT_EVENT_CONTROLS.md. Native AddressedEventHeads sources do not exchange state, so cross-address
sharing is a future architectural option; the pilot uses one address. Batched training path
(fast_native_core, contract-equal) is used by the joint driver. §392 top-placed pooled series finishes first.

## Completed relation advantage; fixed confirmation active, 2 October13:17 UTC

Plan `queue/local_joint_outcome_20261002T125700Z.json` is completed, with all
three fits and reserved fixed-model evaluation in
`results/diagnostics/local_joint_outcome_20261002T125700Z_analysis.json`.
Full joint/local:0.101897516/0.749780357bits,96.09375/78.125% on seed74001,
32suffix groups/128queries. Joint/local gain0.647883bits; conditional paired
suffix interval[0.398637,0.914683]. Full versus shallow gain0.017362bits,
not evidence that core depth earns its work. Shallow95.3125%/0.119259bits.

All arms fit64 distinct queries for16 passes,1,024 target presentations,
256updates. Full joint/local/shallow whole-fit estimates0.237875456/
0.231934208/0.153391616GFLOPs; per-target0.23230025/0.22649825/0.1497965MF;
inference0.053408/0.053368/0.036098MF per complete15-event query.
Joint costs2.56% more estimated fitting arithmetic than matched local.
Full native capacity8receivers,60commits/query; shallow4/30. All use27raw
addresses,mean10.5625occupied,14rawwrites/query,2Cterminal key scores,
2deliveries,Cshared losing candidate values during fitting,mean112.0625
pair losses for joint. Fixed observed predecessor addresses, no learned
write route, no whole-core exact-gradient/depth/language/supremacy claim.

Native/tap/full/shallow balanced stage and frozen readout failures remain
saved beside this positive result. Reporter source now ingests completed
stages; generated PDF/REPORT remain the previous publication until guarded
rendering succeeds. Both older report evidence and all old results persist.

Active confirmation: `queue/local_joint_confirmation_20261002T131700Z.json`,
tmux `local-joint-confirmation-20261002T131700Z`. Four unchanged full-depth
fits:joint/local s7,then joint/local s8. All selected fixed checkpoints score
new seed75001/64groups/256queries, plus delivered-value-zero intervention
with exact parameter restoration. No retuning or weak-seed extension.
Theory69 declares gates and scope. Frozen model/driver/analysis/plan source
hashes must remain unchanged. Main,one safe one-thread job,RSS1.25MKiB,
VMS3MKiB,8GiB MemAvailable floor; observed pilots~353MiB and>=11GiB free.
Next: finish every confirmation fit, report both seeds and any gate failure,
publish combined report through unique guarded queue, then final interpretation.
Other-host language pooling/evidence-strata and AWS dense/capacity work remain
unduplicated.3stashes intact; no rebase or unmerged index was present.

## Protected outcome/joint-credit test admitted, 2 October, 12:57 UTC

Balanced native/tapped/full/shallow cycle and frozen query-readout diagnostic
completed. None learns the joint label: native/tapped/shallow dev .999542/
1.000231/1.000243bits, accuracy .500/.484/.500. Whole fitting estimates
.560270/.618554/.199843GFLOPs,512targets/64updates. Frozen affine bit probes
are near chance on fresh suffixes, and degree2 heads fail badly (7.88–25.33
fresh bits,~50%). Preserve these negative results; neither mere retention
norm nor adding a larger readout established usable old evidence.

Theory68 states the next concrete information-path change. Generic causal
outcome bank writes the observed successor at the observed predecessor address,
without bit extraction or marker filtering. Every occupied address is scored;
two learned key races deliver values to a bilinear terminal decoder while the
native time/race/persistent-vector core remains. Joint training enumerates
terminal candidate pairs for exact conditional content-risk credit; earlier
native teacher and unlearned write-address scope stay explicit. Added raw
symbol storage, discovery, losing values and pair/optimizer work are charged.

Five numerical/driver contracts pass66.400s: results/diagnostics/
local_joint_outcome_contracts_20261002T125300Z.json. Zero parent nesting,
generic protected writes, exact joint-key derivatives/finite differences,
real interrupted model/Adam recovery for joint/local credit, trained local
teacher/inference and chunk identity. Guarded peak groupRSS594,540KiB;
host>=11GiB available. No frozen parent source changed.

Next serial plan queue/local_joint_outcome_20261002T125700Z.json:
joint full/local full/joint same-width shallow, p4,L2/L1,H2,pool2,key/value4,
16fitgroups/64queries,16passes,U4,lr.01,seed6,gap8. Three full-shape smokes
precede pilots. Held-out seed72001 dev selects across fixed passes; reserved
seed74001/32groups evaluates fixed selected models once after all fits.
All three arms reported, count bound limited to query suffix/count inputs,
no dense control duplication or automatic supremacy claim. Main, one safe
CPU job,1thread,1.25MKiB RSS/3MKiB VMS/8GiB availability guard. Other-host
language pooled-receiver and evidence-stratified studies remain unduplicated.

## Balanced joint-feature fitting admitted, 2 October, 12:13 UTC

Read theory/66_balanced_joint_learning.md. Counts are expected to lead in their
well-supported local regime; this local task instead holds query suffixes,
order-0..8 query count vectors and symbol marginals identical while opposite
labels depend on earlier ordered bits. New complement-balanced prefixes close
the old probe's unigram-total loophole. This bound excludes arbitrary inspection
of unrelated count addresses or the full prefix. No language/semantic claim.

Seven actual contracts pass in 122.318s, including full p8/L4 tap zero nesting
and parent gradients, full-shape Adam/next-update recovery, native/tapped real
driver interruption/recovery, target normalization and trained causality.
Result: results/diagnostics/local_balanced_joint_contracts_20261002T121000Z.json.
First contract attempt stopped on unsupported floor/rsub accounting; failed log
and unique queue preserved. Local audit now charges delay-index floor as a
unit special and reverse subtraction as arithmetic; no frozen parent edited.
Peak observed contract group RSS 619,824 KiB; host >=11 GiB available.

Plan queue/local_balanced_joint_20261002T121500Z.json: three one-pass two-group
smokes, then native full, tapped full and same-width tapped shallow fits, then
completed-only summary. Eight fitting suffix groups/32 queries,16 development
groups/64 queries,gap8,16 passes,p8,H2,pool2,L4/L1,Adam .003,U8,seed6.
Every arm has query-only binary supervision and complete 15-event credit;
all states reset per episode, noise coupled by quartet/pass, no target-derived
input or clock. Target alphabet restricted to two symbols for count controls
too. Prefix forward/backward and losing proposals are charged; fitting FLOPs
are declared first/last full-window estimates, actual activity separate.
Smokes/capacity guard precede fits; main, one thread,1,250,000KiB RSS cap,
3,000,000KiB VMS,8GiB availability floor. Reuse serial stage worker; no new
dense control or duplicate other-host sampled-pooling/evidence-strata run.
Exploratory quality gate75%/.8bits; taps additionally need .05bits against
native and same-width shallow to nominate. Fresh fixed-model confirmation
is required for claims beyond this one-seed diagnostic. No automatic scale-up.

## Rebase conflict repaired, 2 October, 11:49 UTC

Completed the main rebase onto 765ff48. The sole conflict was the generated
report PDF; both versions are preserved in Git history and copied under
`.git/conflict-backups/rebase-20261002T114824Z/`. Regenerated PDF and REPORT.md
from the merged source using the unique one-job guarded queue
`queue/local_rebase_report_20261002T114824Z.txt`. The combined report now has
101 validated pages and retains all headings from both prior reports.
Completed publication record:
`results/diagnostics/local_rebase_report_20261002T114824Z.json`, 20.423s,
65,108 KiB RSS. No model fitting or completed result was overwritten. The
three existing stashes remain intact; no rebase or unmerged index remains.

## Value-credit cycle fully closed, 2 October, 10:16 UTC

Parent and followthrough statuses are completed; no local training job or
tmux session remains. Five exploratory fits, frozen audit and guarded report
publication are committed on main. PDF: `../report/sleeping_machines_status.pdf`,
100 pages with bounds/text validation. Read the completed interpretation in
`theory/63_retrieval_credit_and_value_versions.md` before extending this stage.
The .02 bpc promotion gate fails because late projection loses to original
addressed memory; do not automatically scale it. Full versus same-width
shallow remains a positive .144148 bpc finding within this small protocol.

Frozen fitting loss also favors original addressed (3.659509) over late full
(3.727163), so development overfitting alone does not explain late's deficit.
Head gradients remain nonzero. Shallow has higher feature participation rank
yet worse loss: rank alone is not useful representation learning. All three
full models retain measured prefix dependence at gaps 8/32/64 with identical
query count vectors for orders 1–8; shallow dependence is much smaller and
minimal loses measured feature/output dependence entirely at gaps 32/64.
These checkpoints were not trained on parity. Frozen order-4 KN scores
3.751541 bpc on the exact window, still better than the best neural arm's
3.902970. One count pass/prefilled memory versus four gradient passes/cold
state differ; adaptive controls have yet another policy. All 20 are preserved.

Frozen reader fusion passes for all three late variants without optimizer
steps or source checkpoint changes. Full saves exactly 10,240 counted
operations across five occupied reads, removes 1,024 parameters and agrees
within 1.55e-6 logits over 256 inputs. Its 65,536-FLOP matrix fold amortizes
after 32 occupied reads. No full-dev quality, latency or energy claim follows.
Five followthrough tests pass; audit 181.181s / 348,956 KiB, publication
20.032s / 65,708 KiB. Host has about 12 GiB available, above the 8 GiB floor.

Prioritized integrated model/queue remains the unchanged common-seed S64 AWS
capacity/exposure comparison in
`gym/plans/aws_capacity_exposure_20261002T072141Z/manifest.json`; inspect its
actual state on the AWS host before continuing it. Local original addressed
full is the strongest neural arm here, a fixed-key diagnostic, not a new
leading comparable benchmark or complete learned-key race-attention model.
Missing mechanisms include learned context pooling/retrieval, confidence
delivery beyond an occupied bit, historical nonlinear producer credit and
demonstrated jointly useful deep features. No new local fit is queued.

Next theory: `theory/65_evidence_access_and_joint_credit.md` derives a precise
mean/occupancy confidence counterexample and explains why joint route
counterfactuals can have zero immediate utility with an unlearned decoder.
A proposed balanced-prefix integrated fit must first verify evidence access,
decoder learning, matched full/shallow controls and fully charged joint
candidate work. This is a proposed comparison, not a departure to dense
models or an authorization to bypass safe-run prerequisites. Native temporal
races, sparse state, key/value separation and unrealized-route credit remain.
Three old stashes are preserved; no unmerged index or rebase is present.

## Five value-credit fits complete; frozen followthrough, 2 October, 09:53 UTC

queue/local_value_credit_20261002T090800Z.status.json completed, all three
smokes/five pilots/analysis committed. Native/full-original/full-late/same-width
shallow-late/minimal-late cold dev bpc3.968133/3.902970/3.942092/4.086241/
4.539614 on2,047targets. Late full beats native by.026041, shallow by.144148,
minimal by.597522; original full beats late full by.039123. Gate FAILS: exact
fixed-feature credit restoration did not improve this fitting result over the
original model. Preserve positive full-versus-shallow quality too: uncomposed
standalone learning here benefits from the full core, unlike count-composition
minimal equivalence. Depth changes initialization of later parameters, so this
is one-seed architectural evidence, not isolated semantic abstraction proof.

CPU whole-fit estimates1.893/1.924/1.933/.348/.01785GFLOPs; common per-target
and inference/replay ledger saved in diagnostics/local_value_credit_analysis_
20261002T090800Z.json. Full pilots~522–526s/~423MB RSS, shallows~80s/~343MB,
caps all respected, host≥11GiB available. Fixed3-order hashes still omit n
confidence exceptseenbit and learned pooling; old nonlinear feature-producer
credit remains truncated. Do not scale late projection on this failed gate.

Theory63 now derives linear-reader expressivity equivalence, factorized SGD
Gram conditioning and frozen fusion (2d³ one-time versus2d² per occupied read,
break-even d reads). Fused reader refuses training. Theory64 derives joint
feature credit attenuation for balanced parity, a scoped information/credit
counterexample rather than a language explanation or universal ceiling.
Paired parity inputs validate identical actual query count vectors for orders
1–8 with opposite prefix-determined labels. No checkpoint was trained on parity.

Prepared frozen followthrough: scripts/run_value_credit_followthrough.py waits
completed parent, then single-job local_value_credit_frozen_20261002T090800Z
checks compiled equivalence/arithmetic, paired retention, frozen-fit/head-feature
geometry and exact same-window closed-form count controls, no optimizer/weight
change. Local publication queue renders completed report/learning plot and
validates PDF bounds/orphans. Both jobs guarded;600s audit/180s publication,
same1thread/RSS/VMS/8GiB floor. Preserve source files frozen in the parent;
new diagnostic/compiler/report files do not change fitted models. Next decisions
depend on these completed diagnostics, no automatic extra fit or superseded
host queue. AWS capacity/exposure remains separate.

## curie: statistic-valued race memory claimed (§392), 2 October, 09:00 UTC

The curie host implements and owns THEORY §§383/392 (learned keys, statistic values, pooled cascade level,
exact race-expectation delivery credit): sleeping_machines/statistic_race_memory.py,
tests/test_statistic_race_memory.py, and the driver flag --pool-addresses. Queue
curie_statistic_race_D8192_20261002T090000Z (tmux curie_statistic_race) runs after curie_count_chain
(32K scalar, 32K minimal, 8K 64-credit pair; the driver separates U64 from credit). The pooled minimal
core runs first as the control. The other host's ContextAddressed/tapped diagnostics are not duplicated here.

## Deferred value-projection comparison, 2 October, 08:58 UTC

Read theory63_retrieval_credit_and_value_versions.md. New concrete limitation:
the original addressed slots cache W_t h_t and detach; old retrieval gives no
current W credit and mixes historical W coordinates. LateProjectedContextModel
stores raw feature sums and applies current W on read. Fixed-weight linear
equivalence and exact fixed-feature W adjoint verified; historical core-feature
producers still truncated, hash keys still fixed (no learned pooling/KV race).
Native time/races/addressed receivers/key-values/unrealized-route credit retained.

FullH2/d16/L8 zero nesting, actual Adam/state/RNG recovery, trained causality,
plus8 real numerical/driver contracts pass150.618s/456692KiB:
results/diagnostics/local_late_projection_contracts_20261002T090500Z.json.
The first contract attempt failed only because its test requested unsupported
U32; corrected to actual U64 and preserved failed log/unique queue. All three
real driver interruption/resume variants recover selected parameters, Adam
moments, predictions, RNG, cursor, fitting traces/deltas bitwise. Warm replay
clears context history/previous address to prevent invented fit/dev outcomes.

Frozen plan queue/local_value_credit_20261002T090800Z.json: three guarded
late full/shallow/minimal192/129 smokes then five1024fit/2048dev/four-pass
pilots: full native, full original addressed, full late addressed, late
same-width shallow(p16/d1) and late minimal(p2/d1). H2/pool2/c16/U64/lr.002/
warm512/s6/order3/B4096. Same core initialization and training RNG reset;
selection minimum cold dev bpc, frozen fit replay at that checkpoint secondary
and separately charged. Depth and width both change in minimal, so the
same-width shallow control is mandatory. Gate: ≥.02bpc late-full gains against
all four controls, ≤2× original-full estimated fitting work. No automatic scale.

scripts/run_addressed_memory_pair.py serializes one-job queues, verifies frozen
sources/results and900000KiB prerequisite margin, commits each completed result
on main. Mem1250000KiB groupRSS/3000000KiB VMS/8192MiB floor,1thread/nice19;
fullsmoke600s/shallow300s/fullpilot1800s/shallow900s. CPU-only, no GPU;
host~11.6GiB available and no active jobs before launch. First/mature/partial
window fitting work is an explicitly labelled estimate, same convention every
arm; actual losing-value/optimizer arithmetic covered in samples, no energy
claim. Report only completed ledger. AWS capacity/exposure owns its independent
plan; local variants do not duplicate dense or AWS fits. Priority local model
is late-projection integrated addressed core for this discrimination, with
learned keys, old feature-producer credit and semantic generalization still open.

## Current local continuation supersedes the07:55 queue description, 08:13 UTC

Read LANGUAGE_LEARNING_DIAGNOSIS_20261002.md and theory62 before redesign.
Frozen full/minimal64-credit attribution completes12.850s/522736KiB:
full recurrent content helps one32-target dev slice by.048141bpc, principally
the source-context path; receiver erasure alone does not hurt. Full core still
loses the completed whole-development comparison. Direct residual responsibility
median.004530(full)/.003138(minimal); dynamic/base-only output-gradient norm
ratios.206793/.195473, all layers reached. Proper mixture gradients, not proven
autograd failure; Adam scaling and conditional usefulness must be distinguished.

Long-range protocol audit completes.982s/271400KiB. Lag targets include cues,
so uniform24/all-orders chance assertion is false. Tests now measure each causal
local order without target-dependent oracle selection. Driver now separates
credit from U64 optimizer windows (old8K×4 c16/c64 budgets2048/512updates,
new512/512). The old MULTI-JOB curie_long_range_core_20261002T072000Z queue
is retired before launch, commands preserved in archive/protocol_audits.
No waiter or numerical job was alive when admitted work began; prior07:55
tmux/wait description was stale. No actual completed long-range fit displaced.

Full8layer/H2/d16 addressed/tapped optimizer contracts pass31.250s/450832KiB:
exact zero nesting and parent gradients, target-weighted normalization, new
state/Adam/RNG serialization and next-update bitwise recovery, learned causality
and chunk invariance. Added state storage/detach is now counted. Numerical
prerequisites are not evidence of useful deep features. First guarded15-test
run passes11.71s; final driver/accounting changes require a new test tag.

Native whole-operation accounting smoke completes129.923s/385284KiB,
191targets/3updates, .088376GFLOPs including unit-weight specials, formula
coverage complete. Addressed smoke currently active in its own one-job queue
local_deep_addressed_smoke_20261002T080800Z.txt, tmux
local-deep-addressed-smoke-20261002T080800Z; no concurrent numerical job.
Same192fit/129dev/1pass/U64/c16/no-warmup/s6 and core initialization, training
race RNG reset after construction. All operations traced; whole fit and per
target CPU work kept together. Explicitly a prerequisite, not a quality pilot.
Caps1250000KiB groupRSS/3000000KiB VMS/8192MiB floor,1thread/nice19,
600s timeout, measured host availability~11.5GiB, no GPU. Source frozen.

Prioritized local integrated repair diagnostic: ContextAddressedNative,
unchanged native races/time evolution/addressed receivers/keys-values/
counterfactual surrogate + fixed-context outcome slots. It does not add learned
pooling or KV race attention. Native text memory remains cold at dev start,
unlike fit-prefilled count references. Full-driver checkpoint recovery and
quality comparison accounting/selection must precede a long fit. Taps are a
secondary path-length control; longer credit alone already failed its gate.
AWS unchanged common-seed capacity/exposure model and plan remain overall
priority; no dense/local control displacement or automatic scale-up. Pending
smoke/test/report results will be recorded below when completed.

08:18 completion: both accounting smokes and completed-only ledger pass.
Native/addressed:129.923/134.455s,385284/386652KiB RSS,191targets/3updates,
whole fit .088376/.089829GFLOPs, per-target fit .462702/.470307MFLOPs,
inference .097376/.101568MFLOPs (one target after127 warm tokens, a numerical
sample). Code-derived core activity16selected/32keys/32teacher values per
fitting target; context extra capacity4096/occupied150, one read/write except
first write, hashing/integer/traffic outside FLOPs. Added slot tensors19200B,
native/addressed listed state2448/21648B. Work ratio1.01644×, smoke dev gain
.004747bpc; no quality/promotion conclusion. Original empty text filler group
is NaN (undefined), recorded beside preserved original outputs; driver now
uses null and strict JSON for absent groups. Equal initial predictions/core
weights, training RNG and data/update budgets verified in
results/diagnostics/local_deep_feature_preflight_20261002T081500Z.json.

Final25 targeted tests pass11.79s under new unique guarded queue, including
real CLI partial-window optimizer-budget matching and empty-group serialization,
old publication/quarantine protocol checks and repair causality. Frozen full
contracts remain exact-source. Report publication is active in a separate
one-job queue local_deep_feature_publication_20261002T081800Z.txt; it validates
completed ledger hashes and PDF bounds/orphans, retaining historical evidence.
No research fit is running or automatically nominated. Next: whole-driver
checkpoint recovery and explicit cold-versus-prefilled comparison contracts,
then an accounted full/minimal/native/addressed small quality fit. Reserve
learned context pooling/KV-race and new-context generalization for deeper-feature
tests; fixed hashes alone cannot establish that claim.

08:19: guarded publication completed;95-page PDF passes bounds/orphan checks,
new frozen-state/credit and common-unit memory-smoke appendices retain previous
results. No local numerical job or tmux session remains. Corrected §390's
512-float statement: receiver contents only; top context32floats and34float64
timestamps make2448 listed tensor bytes when core fully occupied. Stashes
remain preserved, no unmerged paths or active rebase state. Commit on main.

## Deep-feature program (host curie), 2 October, 07:55 UTC

Attribution (Theory §389): every count-composition gain so far is learned count smoothing. Minimal cores
match or beat full ones (native 8K gate: minimal 2.588 vs full 2.601). Diagnosis (§390): sample
complexity, credit reach (depth·k path; 16-event truncation), a missing race-retrieval mechanism, and a
**single-address native language core (512 floats of persistent state)**. New nested, contract-tested
options: learned dilated delay taps (§391, sleeping_machines/dilated_delay_taps.py) and context-addressed
outcome memory (§390 addendum, sleeping_machines/context_addressed_memory.py). Queue
curie_long_range_core_20261002T072000Z (tmux `curie_long_range`) waits for `curie_count_chain` (32K scalar,
32K minimal, 8K 64-credit pair). Ladder: synthetic lag/induction, counts at chance by construction; native,
tapped and KV arms at 16/64 credit; text 8K native vs addressed in the same loop. Predictions P390a–d and
P391a–c are in theory 59. Next: combine taps + retrieval + addressed memory per the outcomes; sparse
forward credit contract; text at >= 1M characters against the minimal-core count composition.

## §389 ablation result (host curie), 2 October, 03:25 UTC

Native 2K: scalar 2.734, message 2.744, gate **2.695** (+0.5% work), both 2.703. Promoted candidate:
native receiver + escape gate. tmux `curie_count_message_gate` is running the 8K scalar reference;
tmux `curie_count_next` then runs curie_count_gate_D8192_20261002T042000Z (8K gate), then the 131K
carrier diagnostics (both repairs and gate-only, w32/w128). The next open problem is learning beyond
short local context: equivalent order above count-optimal at 131K+, credit horizon and retrieval.

## Count composition diagnosis and §389 repair (host curie), 2 October, 02:25 UTC

Composed carrier+counts 131K: 2.313/2.313/2.316 at w32/w128/w256. Bases alone are 8.17/11.34 bpc
(scratch check from the progress checkpoints), so the base is inert: it never sees the counts (§389).
New zero-init nested repairs are in sleeping_machines/count_escape_gate.py (CountMessage, EscapeGate)
with 6 contract tests. Running in tmux `curie_count_message_gate`, queue
curie_count_message_gate_20261002T032000Z: contracts, smoke, then integrated native 2K with both repairs,
message-only and gate-only, then the unchanged 8K scalar reference. The 031500Z 8K queue was superseded
before launch. Gates: both-repairs beats 2.734 by >= .02, base alone < 4.75, gate-only < .02. Next if they
pass: 8K with the repairs, then the credit horizon/retrieval work needed for equivalent order above count-optimal.

## Count calibration and sufficient-statistic theory (host curie), 2 October, 01:20 UTC

Guarded queue `curie_count_reference_language_20261002T012000Z` scored closed-form
count references on the exact shared language protocol (fit text8[0:N], 8,191
development targets, hash 65efabd8ceea). Frozen Kneser–Ney: 3.615 / 3.081 /
2.704 / 2.349 / 2.007 bpc at 2K / 8K / 32K / 131K / 1M; every completed fitted
model is .107–.402 bpc worse at the same N (native8K 3.557). Counting over the
development stream alone, with no fit data, gives 2.884. Theory note 58
(§§376–380) explains why: fixed learned gates under truncated credit cannot
be consistent per-address estimators. It derives escape-race cascades and
responsibility-gated learned base measures, and proposes **count-carrying
receivers**: native model + occupancy-gated addressed counts + escape races,
with the native predictive as base measure. Gate: beat both KN stream-adaptive
(2.699 at 8K) and native-alone by ≥.02 bpc at 2K/8K before any scale-up.
Contracts are specified; no model source changed, no trainer launched. All
future language tables should carry the KN reference row. curie has no text8
by default: it was downloaded to the ignored data/text8.

## Validated native strengths, live AWS follow-through, 2 October, 01:12 UTC

All11 split-event pilots validate source hashes, finite metrics and complete
operator accounting. New SPLIT_SCREEN_FINDINGS_20261002.md and machine-readable
analysis preserve the two main gains: paired timing95.3125% versus exact rank
ceiling50%, and S16 shared/P0 order75.3906% versus private44.1406%, with11.14x
fewer parameters and6.65% less unit-special whole fitting work. Exploratory
paired-population95% gain intervals are[42.97,47.66] and[25.39,37.11]pp, not
seed/confirmation uncertainty. Shared maps also remove private source embeddings.
P2 loses ordinary timing/accuracy while improving some long gaps; appendix
retains all variants and the initial-spectrum confound. Theory58 §§376–378
derives the timing information witness, exact two-state generator receptor,
qualified shared-rule exposure law and private-state/invariant-memory bounds.

REPORT/PDF puts the quantitative native mechanism figure on page3, with whole
fit/per-query/inference work together. Prior valid order/retrieval and native
language evidence is retained; R2 losses and repair plans stay in the appendix.
The new frozen write decomposition is completed: write effect meanabs.012694
versus delivered-value residual.000288, with both earlier opposed signs explained
by writes.48 forwards/12 backwards/12,288 races,7.107s/399708KiB; conditional
one-population evidence, no expected-gradient claim. Theory57/report appendix
now include it. No current teacher/source was modified.

AWS has independently reserved seed7/8 event replications in
aws_event_replication_20261002T005408Z after its active banknote confirmation.
Do not duplicate that chain. New AWS_NEXT_BATCH.md identifies the prepared
FOLLOW-UP after those replications:
gym/plans/aws_native_confirmation_20261002T010500Z/manifest.json.
Thirty stages, twelve final holdout scores; only TWO fresh private-S16 fits.
Ten selected checkpoint reuses include the six forthcoming timing/shared-P0
replication checkpoints. Original fitting work stays charged. Missing parents
stop, never refit. Two corrected crossed-seed/population accuracy gates are
fixed before seed3201/1024-query holdout access. The earlier010000Z draft is
preserved superseded/unlaunched because the AWS overlapping reservation arrived
during preparation. Only launch replacement after the existing worker exits.

Four new read-only tests pass for cluster dependence and selected checkpoint
restoration/data/protocol identity. Thirty queue/source fingerprints, unique
tags and dependency edges validate. No optimizer/test holdout was run locally.
Partial banknote confirmation already exists: ours seed6 test94.306%/.142593NLL
versus original trees93.950%/.206543; CatBoost seed6.120693NLL and seed7.107803
are stronger loss scores. Do not turn the incomplete comparison into a broad
tabular win. The AWS supervisor will publish full prespecified analysis after
all twelve scores; canonical appendix uses completed frozen-test JSONs only.

Local native timing512-query/eight-pass fit remains healthy in its original
guard/queue, about464MiB trainer RSS and11.7GiB available. Three recorded dev
epochs are83.98/79.30/86.72%; these are ongoing, not completed benchmark scores.
Its original serial order/pathwise/timing/capacity controls continue afterward.
No second local trainer was launched. Keep checking new AWS completions during
subsequent work; source/queue/publication lifecycles remain immutable.

## Full-state credit diagnostic and complete first wave, 2 October, 00:06 UTC

All17 first-wave pilots are now completed and source fingerprints validate.
Wine R2 improves native R0 RMSE8.0% (.757914 versus.823788), with12.6% more
whole fit work(.919098 versus.816008GF); trees remain better at.648887.
Preserve this positive reception effect beside banknote/language failures.
The report appendix and first-screen findings now state it explicitly.

AWS's frozen trained-route audit finds two opposed directions in12 probes
with actual alternative memory commits. New ROUTE_WRITE_DIAGNOSTIC.md and
route_write_decomposition.py prepare the next bounded factorial audit: same
checkpoint/population/address/seed/nodes, independently replace delivered
value and persistent commit, record their interaction and local-linearization
residual. Hybrid combinations are diagnostic, not proposed model routes.
Queue aws_route_write_decomposition_20261001T235000Z is prepared/unlaunched;
do not compete with the active split-event reservation. Two read-only tests
pass for the algebra and realized-path/time/noise/weight integrity.

Theory57 §§373–375 derives sparse full-state adjoints and joint likelihood
credit that handles downstream time jumps. Two analytical tests verify the
linear-time surrogate and hard-threshold witness. The score-function primitive
is prior art; the contribution under investigation is joint state/time credit
within this sparse substrate. Current-suffix adjoints cannot become independent
control-variate coefficients by detach alone. No frozen teacher is changed;
the actual factorial audit and integrated contracts/matched fit are prerequisites.

Native8K remains in its existing lifecycle and has recorded four epochs;
await final validation/publication before treating it as completed quality.
No second local trainer or optimizer diagnostic was launched. New banknote
confirmation manifest remains frozen; all its five read-only tests passed.

Subsequent completion: native8K is published in eac1fb6 at3.557380bpc /
15.115512 whole CPU fit GFLOPs,54,907 parameters. Same-data KV8K is3.490090 /
89.998971GF:5.95x less native work at.067290bpc worse, with different width/
history/capacity. Native2K→8K gains.207332bpc; no extrapolated supremacy claim.
The original native continuation has started its guarded full timing pilot,
local_native_event_timing_S4_s6_pilot_20261001T174000Z. Preserve that live source
and the serial host lock. Opening evidence now also states the completed native
data gain; repair plans and losing variants remain in the appendix.

## Autonomous continuation and banknote confirmation, 1 October, 23:48 UTC

Native8K is running under the existing delay recovery; the completed waiting
control gives3.763722bpc. Full reception misses its declared quality gate, so
no larger delay-feature fit is admitted. Native8K is the separate authorized
data-scaling test. Preserve the live source/queues; exactly one local trainer.

New bounded AWS follow-up committed in45b0ea1:
AWS_BANKNOTE_CONFIRMATION.md and
gym/plans/aws_banknote_confirmation_20261001T234000Z/manifest.json.
Twenty uniquely guarded stages: four contracts,four accounting smokes,twelve
comparisons across ours/original trees/CatBoost/logistic and seeds6/7/8.
Only two new native fits: seed6 reuses the first-screen selected checkpoint,
validates matching source/settings/data/dev score and retains the original full
fitting charge. A missing checkpoint stops, rather than silently retraining.
All281 reserved rows/270 feature groups are scored after dev selection; weights
stay frozen and no best seed/ensemble is selected. Paired seed/group analysis
keeps repeated rows dependent and adjusts primary NLL intervals across controls.

Do not displace the already reserved split-event battery or start a second AWS
worker. New confirmation source/queues match their manifest and are unlaunched.
Pinned CatBoost1.2.10 imports successfully locally; its additional dependencies
were installed without changing existing NumPy/PyTorch/sklearn versions.
Five read-only data/probability/pairing/evaluation tests pass. New guarded
optimizer-recovery and full fitting smokes remain AWS prerequisites; no local
training/optimizer diagnostic was added beside the native8K fit.

REPORT/PDF retains opening quality/work evidence and moves next-test repair
plans/losing variants into the appendix per user direction. Completed future
confirmation JSONs have a separate test-quality/consistent-work ledger, with
charged historical reuse explicit. No pending score fills it. Seventy-page PDF
bounds and opening-content checks pass. Stronger real-event adapters, physical
energy and a matched-quality language win remain open; no supremacy assumed.

## First-screen interpretation and next battery, 1 October, 23:14 UTC

Sixteen of17 first AWS pilots are published locally; wine R2 is pending.
All16 source fingerprints match. Banknote data/split fingerprints match ours
and trees:95.3125%/.155305 NLL versus92.96875%/.231868, a three-example
accuracy lead and33.0% lower loss. This is a single-seed dev quality signal;
tree work is uncounted and its CPU fitting is much faster. Wine R0 RMSE.823788
loses to tree.648887 and worsens after selected epoch2; do not blindly lengthen
that fit. FIRST_SCREEN_FINDINGS_20261001.md preserves every contrast and scope.

REPORT/PDF promotes banknote quality and the native/KV6.02x counted-work
tradeoff to the cover/page2, retaining all older strong synthetic evidence.
Editorial direction from the user: the opening banknote figure compares the
native model with trees. Keep the losing R2 reception variant and its cost/
quality interpretation in the appendix, rather than in the opening figure.
The user also requests that repair plans stay out of the opening chapter;
the next-experiment interpretation belongs beside the appendix mechanism screen.
New completed event_variants pilots will populate separate quality/work/activity
ledgers; contracts/smokes cannot populate them. Protected-prefix initialization
removes faster initial temporal modes as well as introducing retention;
note53 now states this confound without changing frozen model sources.

The AWS host has already prepared/reserved11 pilots/33 guarded stages:
AWS_SPLIT_EVENT_BATTERY.md and aws_split_event_20261001T230029Z. Prefer that
coordinated plan. Earlier local draft aws_split_screen_v1_20261001T225000Z is
retained as superseded/unlaunched; do not start it. The broad ADVANTAGE_BATTERY
links the admitted queue and conditional real-event/tabular/language/online
promotions. NONSTATIONARY_PROTOCOL defines causal pre-update scoring, drift
and forgetting; its new adapters are pending, not invented completed TTT.

Theory notes54–56 turn the generator/receptor intuition into task-weighted
tangent alignment, sufficient messages, predictive spectra, hierarchical
conditional moments and necessary operator/state witnesses. Eleven read-only
model/analytical tests pass; optimizer contracts stay guarded AWS prerequisites.

Local delay recovery remains healthy and serial. R2/R4/late2K are completed:
3.795698/3.794673/3.794863bpc, all worse than native3.764712. R4 waiting2K is
running; await its result and frozen quality gates. Sources are unchanged.
No extra trainer was launched. Core gaps remain fixed addresses/forced
activity, full producer graphs, surrogate loser credit and physical hardware.

Use this file for future local progress updates. HANDOFF.md preserves the shared
history and AWS-host notes; append here to avoid competing end-of-file commits.
Do not modify running model/driver/helper sources or reuse changed run tags.

## Current priority, 1 October 2026

Native order S4 pilot: 100% selected development accuracy. Native language2K:
3.764712 bpc / 3.778244 whole CPU fit GFLOPs / 54,907 parameters. Against the
saved KV2K construction: 6.02x less fit work at .032126 bpc worse, with differing
width/capacity/history construction. No frontier/energy claim.

Full-depth R0/R2/R4/uniform/late/waiting contracts passed. R2 and R4 accounting
smokes completed; R4-late follows. Guarded tabular classification/regression and
both tree-control smokes passed; full pilot results remain AWS work.

The first recovery waited safely through a second rebase, published R4 smoke,
then stopped on changed run_safe.sh fingerprint. This was the explicitly
approved AWS slot extension, not a model/source/optimizer change. Reviewed the
diff: local default still uses the ordinary host lock and identical caps;
AWS slot environment is absent locally. Preserve prior needs_review status/logs.

New manifest:
`experiments/queue/local_delay_feature_guard_recovery_20261001T215000Z.json`.
Run `scripts/run_delay_feature_recovery.py --manifest-stem local_delay_feature_guard_recovery_20261001T215000Z`
once in same-stem tmux. It revalidates/skips unchanged successful queues, finishes
R4-late smoke, fits matched R2/R4/R4-late 2K, refits the winner's waiting control,
uses declared gates for delay8K and the separate near-quality/work native8K test,
then hands to `local_native_research_guard_recovery_20261001T215000Z`.

The local native source64/payload16 extension is deferred rather than launched
under an unmeasured unsafe envelope: AWS's smaller payload8/source64 contract
already exceeded the local RSS cap (measured 2,585,864 KiB). Its original queue
and manifest remain unchanged; prioritize the existing AWS source64 pilot and
measure/provision before the wider extension. Source16, controls/replications and
all original quality gates remain in the bounded local continuation.

Local caps: VMS4,000,000 KiB, groupRSS2,500,000 KiB, minavailable8192MiB;
CPU-only ~31GiB host, idleavailable ~12GiB. One trainer. Read status/processes
before assuming later stages completed. Model/driver/helper/queue fingerprints
are unchanged; new manifests explicitly accept only the reviewed guard revision.

At 21:57 UTC the new continuation is healthy: R4-late accounting smoke was
published in 4663d51, and the full 2K/R2 reception pilot has been training since
21:51 UTC. Group RSS is about 447MiB, MemAvailable about 11.4GiB. No completed
quality score from that pilot yet. AWS recovery pilot results are now included
by the report loader across both original/recovery plans, deduplicated by tag;
contracts and smokes remain excluded from benchmark plots.

The design update in `theory/RESULTS_DESIGN_UPDATE_20261001.md` records useful
depth versus weak tested residual paths, the native 6.02x work/.032126bpc
tradeoff, proved but not yet fitted temporal functions, and training-memory
exposure. These are the current empirical constraints, not assumed supremacy.

22:18 UTC update: local R2 language2K completed at 3.795698bpc/4.032017GF,
versus native3.764712/3.778244; no promotion justified yet. Published c81f0e7;
full R4 started at22:18 under the same guard, followed by late allocation and
waiting control. Seven completed AWS small pilots are now in the PDF: timing
78.125% versus refitted rank79.6875%; order CF/pathwise both51.5625%; occupied
sources4/16/64 accuracy51.56/48.44/32.81% with commits16/matches32 per event.
State-clearing hurts, but invariant order labels degrade under stretched gaps.
See the theory update for hypotheses and the one-population sources64
uncertainty limitation. No full tabular pilot result is present locally yet.

REPORT/PDF now contains new AWS language and complete residual4 hierarchy results;
63-page PDF bounds checked. Plain depth3 85.06% remains best saved64K RHM point,
residual4 72.56%/ten passes still improving. Early AWS matrix pilots are to be
reported separately from 8,191-target language protocols. Numerical primitives
for trainable windows/trains are proved/tested, not a fitted full-model claim.
Robotics adapters remain pending; the task is supervised asynchronous streams,
not RL. Core gaps: observed fixed addresses, forced activity, local surrogate
loser credit, bounded ordinary credit and no validated physical clockless ASIC.

## Review and publication, 2 October, 01:49 UTC

User requests removal of target-leakage results so they cannot be reused. Ten
E63/E79/raw word-keyed result/provenance files are moved outside the active
results tree to `archive/invalid_protocol/target_leakage_20261002`, with original
paths, hashes and a manifest marking them ineligible. Historical reports and
withdrawn numerical findings are quarantined too. Report reader rejects stale
E63/E79 paths; old benchmark entry points and legacy report generation refuse
execution. Eight publication/quarantine regressions pass. Current PDF contains
none of the withdrawn mixture scores. Causal E173/E174 evidence remains valid.

Banknote:11/12 final cells; ours/tree/logistic each complete seeds6/7/8, last
CatBoost pending. Three-seed means: ours91.81%/NLL.2353, trees93.95%/.2065,
logistic94.66%/.0906. Competitive accuracy, no confirmed advantage; descriptive
accuracy intervals include zero but do not establish equivalence. Partial
paired analysis JSON records source hashes and98.33% NLL intervals. Original
development signal remains beside this revision in the appendix. Opening
figure now shows reserved-test means/seed points, not just the dev lead.
PDF rebuilt:78pages, no short orphan pages or off-page text. Repaired causal
10M count/copy comparison is now prominent alongside the native work tradeoff.

Other-host count-reference analysis is incorporated. Fixed-gate inconsistency
is not a theorem about every learned recurrent gate. Corrected pre-increment
count-update denominator n+1+Aα. Predictive responsibility bounds apply to
logit gradients/current prediction; state advance/future credit and parameter
Jacobians remain needed before claiming executed skip savings. Statistic
write formula is exact for one future visit under supplied q. Memorization-tax
numbers are conditional scenarios, not a dense-model lower bound. These review
corrections preserve measured positive gaps and frozen models. Root rule/state
note now uses named RS1–RS3 to avoid live global-number collisions.

Local native timing pilot remains guarded, single trainer, at epoch5 of8;
interim results are excluded. Original coordinator/guard/trainers and sources
are unchanged. Addressed-state write-credit implementation is separate and
uncommitted pending its own full contracts/accounting/small fit. AWS now has a
reserved fresh native confirmation supervisor after its event replication;
do not duplicate that chain. Curie has a separate count-carrying smoke and2K
pilot; avoid duplicate variants. No new local optimizer work ran beside trainer.

## Integrated write-credit boundary plan, 2 October, 01:55 UTC

Prepared `queue/local_state_credit_boundary_20261002T015000Z.json`: six
one-job queues, full-depth order and shared/P2 paired-timing optimizer/recovery
contracts, two accounting smokes, then matched strength0/strength1 order fits.
New modules/driver/contracts/tests are fingerprinted; original split/native
sources stay unchanged. Five new read-only model tests plus eight publication
regressions pass. No numerical prerequisite optimizer ran outside run_safe.

Supervisor `scripts/run_state_credit_boundary.py` verifies the original
coordinator231951/current guard232027 start identities and parent relation,
pauses only that coordinator, waits for the current timing pilot to complete,
then executes each job through run_safe. SIGTERM/HUP and failures resume the
original coordinator in finally. A hard-killed supervisor requires checking
its status file and manually resuming only the verified original coordinator.
No trainer/guard/watchdog is paused. Original campaign resumes after the
bounded insertion; no large fit is scheduled by this new pilot gate.

Caps:1,250,000KiB whole-job RSS/3,000,000KiB VMS,8GiB available-memory floor,
600s contracts/smokes,3600s pilots. S4/d8 parent pilots/checks used well below
1GiB; current host available~10.7GiB and only the preserved trainer~453MiB.
Each smoke must stay below85% RSS cap before pilots. Complete operator coverage,
finite results and source/queue fingerprints are mandatory. Auxiliary logical
views and RSS are separate; total learning includes all losing proposals,
backward and Adam. Teacher is a local full-state surrogate, not the exact
joint-race reference kernel. See theory/state_write_credit_integration_20261002.md.

Completed pilot results/paired comparison are committed automatically; the PDF
appendix ingests completed fits only and is rebuilt after the pair. Main report
code is outside the frozen training dependencies. User host pushes root commits.
Do not blindly restart the insertion or reuse its changed queues/tags.

New AWS update: CatBoost seed8 was manually stopped after3128s with zero
completed candidates; original failure51e9a38 is preserved. No score exists.
The report now says stopped/incomplete rather than still running. Independent
AWS recovery has unblocked the reserved event replication/confirmation chain;
contracts are arriving, not new completed quality scores yet. Banknote's valid
three-seed family summaries remain; the full four-family gate is withheld.

## Safe session recovery, 2 October, 02:13 UTC

This session starts on curie with no live trainer, guard or tmux server. Saved
native timing and state-credit insertion lifecycles remain nonterminal, but
their original PIDs/start identities are absent. Preserve these files and the
native timing progress checkpoint; do not signal stale PIDs, restart the old
insertion, or treat epoch5's score as completed evidence.

New recovery plan: `queue/local_state_credit_recovery_20261002T021222Z.json`;
interruption reconciliation is stored beside it. New supervisor
`scripts/run_local_frozen_recovery.py` runs the original six unchanged one-job
queues serially through run_safe, preserving all source/queue fingerprints.
Original models, settings, output tags and pilot gates are unchanged. It checks
absent predecessor identities, enforces each prerequisite and rejects changed
settings/data in the final comparison. Numerical contracts and accounting smokes
precede both fits; no automatic larger fit. Each completed result is committed
locally on main. Root publishing handles remote coordination separately.

User explicitly reiterates avoiding host hangs and otherwise continuing
autonomously. Retain1,250,000KiB groupRSS/3,000,000KiB VMS,8192MiB available
floor, one thread, nice19, watchdog,600s checks/smokes and3600s fits. Host starts
with about11.6GiB available, no GPU. The integrated addressed-write teacher
comparison takes priority over carrier-only scale queues. Native timing exact
checkpoint recovery follows the bounded comparison; AWS reserved replication/
confirmation and count-carrying work are not duplicated. Fixed observed addresses,
forced activity, full producer graphs, surrogate credit and hardware energy remain
limitations. This is lifecycle recovery, not an architectural departure.

## Completed learning audit and bounded follow-through, 2 October, 02:56 UTC

Rebase and subsequent report autostash conflicts are resolved on main. All three
stashes and archived conflict stages are retained. Frozen recovery sources and
queues are unchanged; the integrated teacher pilot remains the only trainer.
Baseline completes at54.296875% /0.978725879 NLL and reproduces the saved AWS
parent trajectory to floating tolerance. No teacher score is completed yet.

Completed guarded audit: results/diagnostics/
`local_language_learning_audit_20261002T023510Z.json` (33.212s,333336KiB).
All six saved carrier and eight native layers receive gradients. On32 fixed dev
positions, removing native history older than16 changes predictions by0.02252
nats KL; useful accuracy beyond16 is not established. Count-composed carrier
logit credit matches the escape-responsibility identity within1.86e-9. First
five clock gradients are present at initialization. Correction02:59 UTC: the
initial handoff misread fitted composed clock displacements as zero; actual
diagnostics show nonzero layers1–5 at all three widths, and only the unused
final clock is unchanged. There is no supported clock-learning discrepancy.
Their unavailable checkpoints still limit new base-replacement interventions. No optimizer
was changed by this audit. See LANGUAGE_LEARNING_DIAGNOSIS_20261002.md.

The report now distinguishes context dependence, gradient reach and predictive
benefit. The native2K count composition gains only0.0071bpc over its initialized
base+escape comparator; this measures whole-model fitting, not isolated deep
representation learning. The older learned carrier uses history and memory.
No global gradient-disconnection or strict bigram-only conclusion is supported.
Report rebuild:87 pages, bounds/orphan checks pass;10 publication regressions
pass. Existing eleven recovery/state-credit tests also pass.

Prepared follow-through: queue/local_credit_followthrough_20261002T025250Z.json.
It waits for exact predecessor identity and completed pair, then serially runs
two separate one-job guarded queues: common-unit report publication (180s) and
full-depth count-message/escape-gate optimizer/recovery contracts (300s). Frozen
source hashes, one CPU thread,1250000KiB RSS/3000000KiB VMS and8192MiB available
floor remain. Contracts test both repairs and each alone, exact zero nesting,
parent gradients, head gradients and serialized next optimizer update. These
contracts are prepared, not passed yet. No new fit or scale job is admitted.

Prioritized integrated model remains native addressed-write counterfactual
teacher, H2/d8/depth8, private four-source order task,128 fit queries/pass ×4.
Learning uses surrogate unrealized alternatives and actual producer graphs;
fixed observed addresses, forced activity and limited task population remain
gaps. Count-conditioned readout/escape repair is a language diagnostic, not a
replacement for deep persistent event representations. The other thread owns
its proposed fits; its combined queue must be split into one-job queues before
local admission. Deferred native timing checkpoint recovery and AWS large dense
comparisons remain separate. No new dense local training or duplicate AWS run.

Prepared continuation local_language_credit_horizon_20261002T030100Z.json waits for the full credit-followthrough
process to complete before a separate guarded300s/one-thread frozen-weight
native8K horizon audit. Three arms retain identical128-token context, identical
16-target loss suffix and per-position race noise; only graph reach16/32/64
changes.384 forward tokens,112 graph tokens,3 backwards, no optimizer. This
measures omitted surrogate credit, not proof a longer fit helps. Prepared, not
completed; no core substitution, new fit or scaling.

## Completed local chain, 2 October, 03:06 UTC

All six recovery jobs, common-unit report publication, all three full-depth
repair optimizer variants and the frozen horizon diagnostic complete under
serial guards. No current local trainer. Teacher pilot55.078125%/.977812493
versus parent54.296875%/.978725879; fitting.277294/.265264GFLOPs; ratio1.04535.
The admission gate fails (.78125point/.000913NLL versus required5points/.02);
no automatic larger write-teacher fit. Completed results/negative findings
are preserved and committed on main.

Repair optimizer contracts:13.678s/421616KiB, exact next-update recovery for
both/message-only/gate-only full native H2/d16/depth8. Horizon audit11.075s/
530424KiB: identical predictions for identical context/targets/noise;16vs64
gradient relative difference30.887% and cosine.957,32vs64 difference15.145%.
Zero optimizer steps in this audit; not improved fit-quality evidence. The
missing optimizer prerequisite is resolved. Counts/escape repair fits remain
owned by the other thread and require one-job queues plus measured accounting
smokes. Next informative credit experiment uses identical target/update
budgets and varying graph reach in the integrated model, after bounded
resource calibration; no dense substitution or new long fit is admitted here.
Native timing checkpoint recovery remains deferred; AWS owns large dense
comparisons and event confirmations. See diagnosis for mechanism gaps and
THEORY§389.1 corrections. Preserve stashes and existing results.

Final publication check:88-page report includes completed pair/gate decision
and the frozen horizon ledger;10 publication regressions pass, no orphan or
out-of-bounds PDF text. All three supervisor lifecycles are completed, tmux
exited normally, no local numerical job remains. Model sources stayed frozen
through execution; completed findings are committed locally on main. Remote
sync is handled by the external publisher; this session did not push via SSH.

## Integrated credit attribution, 2 October, 06:00 UTC

User renews autonomous work. Read latest shared/AWS handoff and theory59:
new count-gate minimal controls match or beat full cores at2K/8K. Thus gate
gains cannot establish deeper temporal learning. Existing32K count gate/scalar/
minimal queue belongs to the other thread; no local trainer/progress checkpoint
is present in this session. Do not duplicate its settings. AWS owns event
confirmations and provisioned large dense comparisons.

Prioritized new integrated comparison: queue/local_count_credit64_20261002T060000Z.json,
supervisor scripts/run_count_credit_pair.py. Reuses completed full/minimal
16-credit2K gate-only results; new64-credit full H2/d16/depth8 and minimal
H2/d2/depth1. Same data hashes, K4/prequential dev,4passes, seed6, lr.002,
U64/warm512,8188 fitting targets/128 optimizer updates. Only graph reach changes
within each core. Native races, persistent addressed state, separate keys/values,
channel mixing and unrealized-route credit remain. No model/driver substitution.
The mechanism gap is omitted producer credit across detach boundaries; the
frozen audit showed30.9% gradient difference, not a quality benefit.

Actual gated64-credit normalization/head-gradient/cursor/serialized-Adam
contracts precede full and minimal192/129-target accounting smokes (first64,
persistent64, partial63). Contracts and smokes must stay below900000KiB RSS
for admission; jobs retain1250000KiB groupRSS/3000000KiB VMS/8192MiB floor,
1thread/nice19/watchdog. Full fit3600s, minimal1200s, based on historical
1096s/164s16-credit fits with generous long-graph margin. One job per queue.
Model/source/queue hashes freeze before launch; every completed result committed
locally on main. Four-arm analysis charges all fitting work and counts separately.
Follow-up nomination requires full64 improve over full16 by.02bpc AND beat
matched minimal64 by.02bpc with≤2× full-core fitting work. No automatic scale-up.
Small single-seed/development evidence; semantic abstraction, available capacity
beyond activity, discovery/traffic and unbiased hard-route credit remain open.

64-credit full/minimal contracts pass22.699s/671304KiB: exact normalization,
head/escape gradients, persistent count cursor, next Adam recovery and64 vs16
partition forward equality. Full accounting smoke is active; no pilot admitted
yet. Native64 graphs fit inside900000KiB prerequisite margin. Report publication
waiter local_count_credit64_publication_20261002T060400Z waits exact worker
identity/completed analysis, renders completed common-unit ledger and validates
PDF bounds/orphans before committing. Reports keep old evidence while pending.

06:08 status: both accounting smokes complete (full157.127s/550596KiB;
minimal24.689s/361584KiB), formula coverage complete for first/persistent/
partial steps. Full64-credit2K pilot starts06:04:23, PID12404 under guard
PID12388, tmux local-count-credit64-20261002T060000Z; workerPID11662.
One trainer only, nice19/one thread; observed~671MiB RSS/~11.1GiB available.
Frozen hashes are unchanged. Minimal pilot, four-arm analysis and guarded
report publication are automatic after successful completion. New five
protocol tests reject pending output, mismatched data/seed/update budgets and
check consistent target denominators; with ten publication regressions,15pass.
No fit-quality score is completed or promoted yet.

## 64-credit comparison completed and interpretation, 2 October

All stages and guarded publication complete successfully; no automatic larger
fit admitted. Full16/64:2.695121823/2.694298280bpc,3.824211/3.842964GFLOPs.
Minimal16/64:2.693827662/2.693725141bpc,.0719065/.0719164GFLOPs. Credit gain
full.000823544, minimal.000102521; full64 remains.000573139bpc worse than
minimal64. Predeclared gate fails. Whole data/pass/update budgets match. Full
fit1164.472s/677168KiB; minimal176.753s/383768KiB. No broad long-credit or
semantic-feature conclusion from one2K seed. Full gradient changes in the
frozen audit did not translate into worthwhile held-out quality here.

Report auto-publication completes19.340s;91 pages pass PDF
bounds/orphan checks, completed four-arm ledger present. Preserve exact-source
new progress checkpoints for future frozen full/minimal attribution. The next
architectural hypothesis must address useful predictive information and
learning allocation beyond count smoothing, rather than increasing this
variant's data/credit automatically. Content-addressed retrieval, learned
context pooling and conditional interactions remain candidate mechanisms;
derive their information/credit contracts and resource costs first, retain
time/races/addressed state/key-value separation/unrealized-route credit, then
fit integrated controls. AWS reserved comparisons and the other thread's32K
count/minimal experiments remain independent; do not duplicate.
