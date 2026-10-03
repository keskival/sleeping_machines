# Local host: current research continuation

## Datacenter value and prepared serving — 3 October, 15:20 UTC

User clarified AI computing substrates/datacenters and prioritizes work that
strengthens valuation. New DATACENTER_VALUE_MILESTONES.md maps learned quality,
scaling, actual sparse-backend correctness, setup/residency, service resources
and hardware proof points. Financial estimates are not research evidence.
Retain core temporal races, addressed deep persistent state, separate keys/
values, counterfactual learning and silence-aware direction; no core replacement.

146 and sleeping_machines/prepacked_sparse_inference.py prepare a fixed-weight
CPU worker retaining one _stacked result across calls to UNCHANGED sparse_logits.
Private deep-copy isolates caller updates after preparation; source versions
checked across construction, private parameter/buffer/layer versions/gains/modes/
shape checked before calls. Snapshot starts at quiescent weight boundary,
creates versioned tensors inside inference_mode(False), remains no-grad.
One worker/process; global RNG fork and episode reset semantics retained.
Model copies, metadata traversal, gathers, all key scores, state/cache setup,
resident stack and per-version preparation remain charged. No GPU/concurrent
serving/persistent cross-request/online-learning implementation is claimed.

Standard-library fake lifecycle and payload-amortization contracts PASS,
including changed source during preparation and private mutation rejection.
No Torch/NumPy import or model runtime here. Shape-only p32/U4 ten-call
assembly ledger12,234,240 versus2,639,576bytes INCLUDING private model copy;
first positive payload count3calls. NOT measured DRAM, wall, energy or dollars.
This closes our repeated-stack gap; optimized controls can also prepack.

Native admission driver/frozen one-job queue:
queue/local_prepacked_sparse_contracts_20261003T150000Z.txt and .sources.json.
Uses actual AWS491520-presentation admission pilot .pt now present (878271B),
three original producer hashes match. Parent has NO completedDEV/test quality.
Twelve synthetic pool1/3/4 FP32/64+four actual-trained FIT cases check EVERY
winner/final-state/cache/RNG/output plus lifecycle isolation/version rejection.
80forward calls3540padded/2600active positions; backward/optimizer0.
Runtime is UNRUN. Changed source/arguments/parent/weights/queue reject BEFORE
Torch import. run_safe only, oneCPUthread, VMS3000000KiB/RSS1250000KiB,
minimum8192MiBavailable/420s, actual physical reservation after current chains.
Free Docker-local lock is NOT admission.143 completed-fit/144p64 weights still
absent; those prepared diagnostics remain pending, no benchmark score attached
to prepared backend. Current driver/helper sources are frozen by manifest.

Active prioritized integrated model stays credited native core. Owner curie
v8 p64/D4/U2/linear four-pass before v6 tied/width-LR/seeds and horizon, then
DVS chain; AWS90M/p32/D4/U4/linear running with recovered private full replay/
teacher. Shared controls recoverable; no new trainer/queue execution here.
New owner p96/D4/U2 completed2.1625T256 ahead of saved one-pass LSTM2.1706,
with more work, not a comparable-quality resource win. Report front refreshed
with that quality direction and explicitly pending serving milestone. Reviewed
208-page PDF published under status_datacenter_milestones_20261003T153500Z;
previous PDF retained in report/archive and provenance in results/publication.
Isolated render took31.19s at405840KiB peak RSS, one thread/nice19, enforced
8GiB available floor. All-page text/layout checks and source/evidence/canonical
hash guards passed; front, evidence summary and native tables visually reviewed.
No model runtime or new training job; active owner chains undisturbed.

## curie: supremacy ordering, 3 October 15:10 UTC

Prioritized model: native core with route credit (`--route-credit linear`, compiled). One pass at 10M: p96/d4 2.1626
(T = 256 2.1625) is the first native row ahead of the matched one-pass LSTM-256 (2.171), with more parameters and work;
p64/d4 2.184 / pool 4 2.180. Chain (tmux curie_reorder14; the old reorder12/chain13 were stopped at a job boundary):
v8 (p64/d4 + credit, 4 passes, the Transformer-256×4 step count; claim criteria in §415) → v6 (routing diagnostics,
tied pools p64/p32 pool 4 and p32 pool 8, p96 lr .002/.003, seed replicates s7/s8, horizon arms) → DVS large program
(resumes at coarse wd0.01 s8) → DVS credit twins. AWS 90M revision 2 (route credit, p64/d4 first) awaits a push from
the user's host.

## Status PDF reviewed and published — 3 October, 13:35 UTC

At the user's request, regenerated `report/sleeping_machines_status.pdf`
and REPORT.md from saved evidence: 208 validated PDF pages / 201 editorial
blocks. New front pages plot completed matched-T256 language quality and
whole fitting TFLOPs together, foreground p64/D4 2.183 and the credited
D8 2.326, preserve the stronger count calibration and explain capacity/work
tradeoffs. Readable tables retain all11 native rows/two dense controls,
whole-fit/per-input-position work, scored keys and emulator/winner proposals.
The p64 model still trails LSTM and consumes more estimated fitting work;
no comparable-quality supremacy claim. Both write-credit failures and their
withdrawal are explicit. AWS matched1M online evidence is separated from
completed heldout quality; all143/144 native diagnostics remain pending.

Rendered to an isolated copy, reviewed the cover/new front pages/native
appendix visually, checked every page for sparse/orphan content and text
bounds, and verified source/evidence/canonical hashes before replacement.
The first preview was deliberately not published after another host's source
and PDF update; the refreshed publication incorporates its changes.
The previous canonical PDF is preserved byte-for-byte in
`report/archive/status_refresh_20261003T133000Z_previous.pdf`.
Publication provenance/guards/hashes:
`results/publication/status_refresh_20261003T133000Z.json`.
One thread at nice19; ~396MiB peak renderer RSS, ~31s, VMS2000000KiB,
RSS750000KiB watchdog,420s timeout and8192MiB available-memory floor.
No Torch import, model runtime/forward/backward or training job. Existing
curie and AWS integrated queues stay reserved and unchanged.

Added a dated feedback-path scope correction beside the failed-write history:
both value/write score teachers can reach memories through keys. The write
auxiliary additionally alters the persistent-update Jacobian; value credit
alters the emitted-message Jacobian. This distinction is derived, not a
measured cause of either divergence. Keep value-only credit, current widths/
capacity comparisons, independent seeds, horizon arms and AWS90M priority.

## Actual write-fidelity audit prepared — 3 October, 12:55 UTC

Prioritized model stays the credited compiled integrated native core, current
completed p64/D4/H2/U2 test2.183315 at T256, with the v5/v7/DVS/v6 curie chain
unchanged. Existing AWS fits and revision2 contracts/pilots/90M allocation
retain their separate ownership. Shared87cfb1c preserves a positive matched
AWS full-replay online trend through1M targets (.052665/.092932bpc interval
lead over private/shared teachers); that is fitting evidence, not completed
heldout quality or a reason to overwrite any report comparison.

[144](theory/144_trained_write_factorial_and_residual_calibration.md) and
`language_route_fidelity.py` prepare the missing actual-trained write-utility
diagnosis from142. Four fixed-time branches separate delivery, commit and
interaction, with every future race-noise vector checked. An independent
zero memory probe measures G under unchanged message-only backward credit;
EVERY parameter gradient/logit/loss must match the ordinary pass. Compare
value, stored-memory and written-only coefficients using the SAME factual
cotangents, preserving gaps/forget/memory norms and actual loss differences.
Native tiny contracts also check causal prefixes and final-event writes.
No model/active source/optimizer/protocol was changed; no fitting occurred.

Formal calibration refinement: an extra write term should explain residual
Q-a_value. Its oracle nonnegative scale is
max(0,Cov_pi(Q-a_value,a_write)/Var_pi(a_write)), keeping useful value credit
fixed. A term opposed to Q can correct an overestimated base; a Q-aligned
term can worsen it. This is a local score-metric diagnostic, not an installed
scale, a full parameter-gradient claim or a clipping/warm-Adam repair.
Standard-library metric/baseline/interaction/residual witnesses and fake
callback inheritance/exception recovery pass. Native contracts are UNRUN.
With two candidates the centered score space is one-dimensional, so a
perfect per-site oracle scale can be automatic. The driver also reports
one shared alpha across eight local metrics and its residual error; these
are reused-case diagnostics, never an installed or validated training scale.

Unique pending one-job queue:
`queue/local_language_route_fidelity_20261003T125500Z.txt`.
Actual p64 producer checkpoint remains absent here; admission rejects before
runtime imports. After actual physical-host reservation and the existing
integrated chain: run_safe only, one thread, VMS3000000KiB/RSS1250000KiB,
minimum8192MiB available,timeout420s. Eight sites use FIT[0:33], two seeds,
events7/23, first/last depth,head0. Total actual-model34forward/10backward
passes,1088forward positions,8704forward races; tiny admission adds9/3.
No optimizer, DEV/test rescore or FLOP/energy claim. Original final weights
contain no historical Adam moments; none are invented. Other138/141/143
pending diagnostics retain their existing scope. No native runtime job,
forward, backward or profiler was launched from this Docker context.
Syntax/help, one-job manifest, actual producer hashes and missing-checkpoint
admission guards pass without NumPy/Torch. The diagnostic result/log remain
absent. Shared b648a5e subsequently completed the credited p32/D8 arm at
test2.326; shared68a89f4 prioritizes independent seeds before the v6 horizon
arms. Preserve those completed results and amended queue order. The user
has now requested a review/regeneration of the canonical status PDF; that
publication is the next authorized continuation, with old evidence retained.

## Sparse inference contracts prepared — 3 October, 12:30 UTC

**Current integrated priority remains the credited compiled native core.** The
shared p64/D4/H2/U2 fit completed at test2.184239(T128)/2.183315(T256),
DEV2.118444,422475parameters,1220updates,9994240sampled positions. The saved
one-pass LSTM is about2.171 at matchedT256; this narrows the deficit to.012.
The credited p32/U4 fit's2.345157(T256) improves p32/U2 by.026334 at the same
eight selected writes, with1.649917× total fitting work. These are positive
single-seed integrated results, not a claim of comparable-quality supremacy.
Preserve the v5/v7/DVS/v6 curie chain and AWSr2 p64-first90M priority below.

[143](theory/143_cached_inference_admission_and_resource_scope.md) proves the
fixed-weight cache invariant and separates winning proposal arithmetic from
all-key discovery, per-call all-parameter stacking and additional cached
state. Completed shape traces show p32/U2→U4 winner-only arithmetic.163227→
.164301MFLOPs per input position (.66% increase); pool4 emulator.393573.
Per-scored-target work differs because evaluation overlaps. Current U4 final
stack612736bytes/invocation and eight-lane unit/cache67840bytes are tensor
payload counts, not RSS/DRAM/energy measurements. No architecture, active
trainer, producer source or completed result was changed by this audit.

Prepared `cached_inference_contracts.py`: source/checkpoint admission before
runtime imports;12 synthetic float32/64 cases gate four actual-trained FIT
cases. Direct actual winners, final state, reconstructed cache, unchanged
parameters/gains/RNG and observer-output nesting. Failed route/state contracts
are preserved as failed results. It does not fit or rescore DEV/test, and
does not expose every intermediate payload. Original p64/pool2 and p32/pool4
final checkpoints are unavailable here. Two unique **pending one-job queues**:
`queue/local_cached_inference_{p64,pool4}_contracts_20261003T123000Z.txt`.
Admission requires the actual producer and physical-host reservation;
run_safe only, one thread, VMS3000000KiB/RSS1250000KiB,
minimum8192MiB available,timeout420s, after the existing integrated chain.
No additional model runtime/forward/profile/training was launched here.

Standard-library accounting/margin witnesses and fake-runtime observer
recovery pass; native Torch contracts are explicitly unexecuted. Added a
dated §414 scope addendum beside the original positive evidence and separate
emulator/winner proposal counts in the report appendix. Actual float32 full
DEV/test parity and total-resource timings remain subsequent steps, not
claims inferred from small float64 output fixtures. Older138/141 diagnostics
and archives remain intact. Rendered publication stays with its publisher.
Read-only full report assembly passes all199 editorial blocks, ten completed
native rows/two dense controls, seven mechanism-count columns and all30
historical136 source hashes. Both queue manifests contain exactly one job;
current producer hashes match and checkpoint-absence guards reject before
Torch/NumPy imports. No pending result or job log was created.

## curie: route credit closes the 10M language gap, 3 October 12:05 UTC

Prioritized integrated model: the native core with linearized local-expectation route credit
(`--route-credit linear`), compiled, segment-batched. One-pass 10M test bpc: p32/d4 pool 2 2.370, pool 4 2.343 (credited
capacity beyond activity), p64/d4 pool 2 **2.184** (T = 256 2.183; one-pass LSTM-256 2.171, Transformer-256×2 2.427).
Exact winner-only inference (§414) costs 0.16–0.60 MFLOPs per position, nearly flat in pool size. Running chain
(tmux curie_reorder12): v5 remaining arms (p32/d8 + credit, pool 4 + linear_rwn, routing diagnostics), then the DVS
large queue, which now starts with language v7 (p64/d4 pool 4, p96/d4 pool 2, both with credit), then the DVS credit
twins, then (tmux curie_chain13) v6 horizon arms. AWS: 90M revision 2 with route credit, p64/d4 first (needs a push
from the user's host). Open gaps: write-address credit (linear_rwn did not help), the memory horizon (v6), multiple
seeds, and 90M.

## Successful route credit and lazy-write audit — 3 October, 10:30 UTC

**Prioritized integrated reference:** completed compiled10M p32/D4/H2/pool2
with `--route-credit linear`, test2.370110(T128)/2.371491(T256), DEV2.313941,
108875parameters,1220updates,9994240sampled fitting positions. It improves
the same uncredited recipe2.506925 by.136815bpc. At matchedT256 it beats
the saved one-pass Transformer2.427 by.0554bpc; LSTM2.171 still leads.
Counted fitting work7.242661TF/.724684MF perposition, about.3% above the
uncredited native7.220754TF/.722492MF. Native tracing and dense shape
estimates retain different conventions; single-seed exploration is labelled.

**Existing queues retain priority:** curie
`queue/curie_language_batched_v5_20261003T101000Z.txt` carries value credit
to pool4/width64perhead/depth8 and tests written-only `linear_rwn`; shared
v6 horizon queue follows the current chain. AWSr2 native90M admission uses
the new103000Z route-credit arms after contracts/pilots; revision1 is
superseded without running. Preserve active sources, reservations and
their completed results. No native job/forward/backward/profiler was
launched from this Docker context; the physical lock remains inaccessible.
Its roughly10.4GiB MemAvailable does not establish physical-host admission.

New [142](theory/142_lazy_write_coordinates_and_credit_scope.md) and
`check_lazy_write_geometry.py` distinguish missing stamp cancellation
from real homogeneous-state utility. Six stdlib scalar/complex contract
classes pass, score finite-difference maxerror2.419e-12: pure coordinate
refresh has zero true credit but nonzero stored-only credit; raw stored-key
read has real credit omitted by written-only; changing forget produces a
true/stored-only sign reversal. Full finite endpoint memory+stamp Taylor
terms still need not cancel exactly. These witnesses do not attribute
the reported failed linear_rw fit or establish native gradients/quality.
Enumerating both factual outcomes preserves the constructed opposite sign
in expectation; merely attenuating that isolated component leaves ascent.
The newly written value is not uniformly bounded by normalized control
inputs; read/write operands use unnormalized mixed content. Conditional
norm/error bounds and a fixed-FIT actual-write suffix/real-Adam plan are
recorded. No replacement model or additional training queue introduced.

Added dated scope corrections beside the original §413/FINDINGS statements,
preserving the positive2.370fit and failed-write history. Report appendix
now counts available unit slots/value scalars, selected writes/deliveries,
all scored keys and computed candidate values for every completed native
row; same-unit whole-fit/perposition work and existing controls remain.
The matched-T256 advantage is explicit and the small actual work increase
is stated. Rendered artifacts remain the publisher's responsibility;
this turn prepares/verifies the report source, not a new PDF.
Verification passes all199 editorial blocks, seven completed native
mechanism-count rows/two dense controls, all30 historical136source hashes
and quarantine. Paired recipe checks confirm shared settings,1220updates
and9994240positions; actual work increase.3033946%, estimated whole-fit
Transformer/native gap15.36198×, matched-T256 bpc gain.05541813.
Existing139/141 stdlib checks pass too. Full assembly loads NumPy for
saved evidence summaries, no Torch; no additional model test is claimed.

Pending138/141 diagnostics remain behind the integrated queues. The141
producer checkpoint/physical runner logs are still unavailable here;
do not reconstruct weights or infer an idle host from the local lock.

## Routing measurement and horizon checkpoint — 3 October, 09:20 UTC

**Physical curie remains reserved for the existing queue.** New shared
results completed the original D8/pool4 arm (test2.498),50K routing diagnoses
and the D4/pool1 diagnostic control (test2.439). The other v4 credit arms
retain priority; this container cannot reserve their physical lock or
infer host idleness from its local process table. No model runtime,
training, forward diagnostic or profiler was launched here in this turn.
The last visible MemAvailable was about10.9GiB, not proof of admission.

Prepared `language_routing_measurements.py`, stdlib summaries/checks and
[141](theory/141_streaming_routing_and_horizon_measurements.md). Original
saved p32/D8/pool2 weights and verified83265f archived batched body; actual
winner counts separate from probability mass; depth/head buckets; local
post-clamp clock sensitivity/boundary hits; sampled-first-time argmax
identity versus the older unit-noise/clocks policy; every component NLL
and the true mean-seed Jensen gap. Streaming target log probabilities
replace retained all-span trajectories. No changes to the existing driver,
v4 queue, model core or saved results.
The loader also verifies the relative temporal dependency7237331f against
both archive producer anchors; the original language parent did not hash
that dependency separately, so this provenance check is explicit.

Unique **pending**, one-job queue:
`queue/local_language_routing_measurements_20261003T092000Z.txt`.
It uses2048DEV chars, T128, four sampled/two greedy production passes,
30windows/1984scored targets per arm. Planned capsVMS3000000KiB,
RSS1250000KiB, minimum8192MiB available, timeout420s, oneCPU thread via
run_safe, ONLY after a physical-host reservation. Native contracts are
first in that guarded job and remain unexecuted; no output/log exists.
`check_routing_measurement_summary.py`, static syntax and `--help` pass
without NumPy/Torch; these are measurement/math checks, not native tests.
Keep138's actual-warm-Adam comparison pending too, behind the integrated
v4 priority and physical-host admission.

The parent JSON points to its producer's untracked final `.pt`; it is not
available in this execution context. The driver rejects absence before
runtime imports. Its queue must execute on the producer with those actual
weights, or after a verified transfer. No reconstruction from reported
scores is allowed. The pending queue's source/argument shape verifies
locally; actual-checkpoint admission does not.

Full report assembly verifies199 editorial blocks and all30 historical136
source hashes; frozen archive collision/tamper guards and target-leakage
quarantine pass. This read-only check used NumPy for existing evidence
summaries, loaded no Torch and rendered no Markdown/PDF. Existing publication
artifacts and completed result bytes are preserved.

Updated140's dated status: the shared8bcea13 now implements `linear_rw`
memory-content credit, while timestamp/seen/topology/curvature remain
outside that surrogate. Preserve the original derivation and historical
"unimplemented" scope. Added a correction beside §413's original numbers:
unit-forget ln2/rate does not bound effective retention when forget is
input dependent; dormant stored memory also enters key scoring before
candidate decay. Sharp probabilities are measured, winner balance and
predictive specialization are different questions. The original .022/.019
mixture gains are against first seed, not a saved Jensen baseline; greedy
changes clocks. Do not silently erase these valid scoped results or treat
them as isolated noise/maximum-context proofs.

## curie: 10M language results, §413 diagnosis and v4 diagnostics, 3 October 07:45 UTC

Completed (one pass, compiled, test bpc T128): p16/d8 skip2 2.719, p32/d4 2.507, p32/d8 skip2 2.456 (one-pass E64:
LSTM-256 2.171, Transformer-256×2 2.427). FINDINGS, the report appendix (report/native_language_batched_appendix.py) and
THEORY §413 have the table and the diagnosis. In the fast language path the race address receives only first-time
clock credit, so pools fragment memory, and active width is small. Running: v3c p32/d8/pool4 (until about 08:35). Then
tmux curie_reorder11 stops the DVS contracts and runs queue curie_language_batched_v4_20261003T070000Z: contracts,
forward-only routing diagnostics of the p32/d8 pool-2/pool-4 weights, then p32/d4 pool1 / linear / linear_rw /
pool4 / pool4+linear / pool4+linear_rw / p64d4 (predictions (a)-(e) are in §413). After that the DVS program restarts.
AWS: 90M arms are queued (AWS_NATIVE_LANGUAGE_90M.md, revision pending on v4). Host memory: diagnostics beside a job
breach the guard (MemAvailable baseline about 9.4-10.3 GB). Put forward-only analyses in queues.

## Credit geometry checkpoint — 3 October, 06:52 UTC

**Physical-host admission is unresolved.** The curie handoff below reports
active compiled training in `curie_chain10`; this Docker execution context
cannot see that process, tmux socket or physical-host runner lock. Its free
`/tmp/experiments-runner.lock` is container-local and does not establish
host idleness. No new fit or profiler was launched from this context in this
continuation. The user has been asked for the missing host state; another
"continue" instruction does not establish that the reported fit ended.
Do not automatically start a waiting diagnostic on this local lock alone.

Prepared [138](theory/138_warm_adam_route_coverage.md), driver
`warm_adam_route_coverage.py`, and unique one-job queue
`queue/local_warm_adam_route_coverage_20261003T062500Z.txt`: original k1/k8/
full168, SAME132 actual trained weights/12-step Adam moments and all16 prior
race seeds.48 independently restored actual updates plus3 exact recovery
checks; raw/clipped/update vectors, masks, fixed FIT predictions, empirical
mean shifts and same-noise full-reference errors retained. No result exists;
the numerical driver remains unexecuted. Static syntax/source checks pass.
Its planned command, ONLY after confirmed physical idleness/separate host
or a coordinated host-global reservation, is:

```bash
MEM_CAP_KB=3000000 MEM_CAP_RSS_KB=1250000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=240 experiments/queue/run_safe.sh experiments/queue/local_warm_adam_route_coverage_20261003T062500Z.txt
```

[139](theory/139_clip_adam_geometry_and_host_admission.md) derives actual
clipping and historical Adam Jacobians, nonlinear conditional bias and
optimizer-conditioned coverage allocation. An exact convex example shows
that clipping/fresh Adam can turn an unbiased sparse gradient into ascent;
this is a constructed failure mode, not a measured native reversal.
`check_credit_geometry.py` passes standard-library rational/finite-difference
checks, maximum absolute error6.97e-12, without tensors or model fitting.
This strengthens the reason to measure actual updates, rather than promote
137's positive raw variance-work heuristic to a learning claim.

07eaaae evolved the shared compiled-capable BL driver. Preserve that active
source. Exact pre-compilation4b163a bytes are archived under
`archive/frozen_sources/4b163a25008ea8261acbee465770bad84dff3dbc1aa3c0ff74f39571fd83e8f8`.
`legacy_batched_driver_binding.py` loads that verified driver for138 and
provides a copied publication view resolving ONLY known historical source hashes.
Concurrent e096ca6 also changes batched_episodes: exact83265f bytes are
archived under the corresponding frozen_sources hash directory. The binding
resolves both known historical hashes, and138 uses that old batched function
for fitting AND fixed predictions, restoring its caller callback afterward.
Original result bytes/metrics/hashes and frozen producers remain unchanged.
Root readable_report uses the view; all30 route-cost source hashes verify
and its complete120-section loader passes. Scalar historical source hashes
and the target-leakage quarantine remain supported. REPORT/PDF retain the
completed203-page publication; no pending138 scores or rerender claimed.

Prior134/136 wall times cannot be treated as physically isolated speed
benchmarks: the later curie handoff reports compiled work active around
the same period, without visibility of a shared host lock. Their numerical
contracts, counts and positive FLOP/variance heuristic remain valid within
their recorded scope. See the annotation beside the original evidence below.

Other-host completed native10M p16/D8 compiled arm is test2.718723394 bpc
atT128 /2.719413474 atT256, DEV2.676420348,54,907 parameters and a nominal
one-pass9,994,240-character random-segment budget. Positive recipe progress
from2.899 is preserved, without isolated clipping/depth attribution. Source
inspection shows factual payload/common-clock credit here, with no added
losing-write categorical replay objective. This leaves the distinct original/
corrected-replay learning comparison open.
06:49–06:50 shared updates independently identify the omitted address credit,
add optional message-linearized eager/compiled credit and v4 diagnostics;
p32/D4 completes about2.507 testbpc. Preserve that progress and its different
width/work budget. V4 remains pending, not a completed exact-write teacher.

Priority remains source-owned integrated native temporal/private-state
language work and the user's AWS90M request. **The latest
`AWS_NATIVE_LANGUAGE_90M.md` holds full-arm admission pending v4 diagnostics;
honor that revision before starting the earlier90M arm list.** Retain the existing
private/shared original-teacher/corrected replay10M matrix
`queue/aws_language_winner_matrix_20261003T014100Z/manifest.json`. Preserve
curie's active compiled language/DVS sequence.138 cannot displace them.
Remaining gaps: actual-write categorical replay in the compiled reset-language recipe,
useful nonlinear depth attribution, horizon, generalization and total
discovery/losing-credit/optimizer work. No core architecture substitution.

[140](theory/140_message_credit_and_private_write_gap.md) audits the new
message-linearized operator: equal messages can leave different useful
private writes, making its message score credit zero despite nonzero exact
choice utility. A state/timestamp local extension is DERIVED, not installed;
boolean seen/topology/finite curvature remain outside it. Required contracts
and a matched trained-state utility comparison precede any fit. Also records
concurrent diagnostic limits: mean_pi is expected mass, not winner counts;
mixture Jensen compares against average constituent NLL, not the first seed;
argmax's clocks change; T128/T256 similarity is not a context-usage proof;
executed candidate proposals/keys must be charged. Preserve v4 predictions
and negative/positive outcomes; do not overwrite source-owned diagnostics.

## curie: compiled native training and the calibrated 10M language series, 3 October 05:45 UTC

Running (tmux curie_chain10): queue curie_language_batched_v2c_20261003T054000Z (compiled, §§411–412: 64 × 128
windows ≈ 1,220 updates per pass, lr .004, cosine, one pass, DEV text8[90M:91M], E64 windows at T = 128 and 256):
p16/d8 skip2 → p32/d4 → p32/d8 skip2 → declared capacity arm p32/d8/pool4 skip2 (346,331 parameters ≈ one-pass
LSTM-256's 338,395, same 16 selected writes per character). Then the large DVS program
curie_dvs_large_20261003T013500Z, all arms --compiled. First compiled arm: 5,800 characters/s (eager 1,424);
dev50k 2.894 at 3.3M characters (v1 3.210 at the same point; v1 final test 2.899).

Pending decision: the 90M run per §409 (DEV selection among the completed 10M arms, near-identity preferred within
.01 bpc, payload 32 if one pass fits about 20 h; compiled; 64-128 lanes). Matched one-pass references:
e64 lstm_D10000000_s256_p1 2.171 / tf_D10000000_s256_p1 2.427 test bpc.

Host notes: MemAvailable baseline is about 12 GB (other tenants), so the 8 GiB floor leaves about 4 GB. Never profile
beside a job: a torch.profiler pass stopped every queue at 04:44. python3.13-dev was installed (sudo apt) for inductor's
C++ kernels.

## Current checkpoint — 3 October01:08 UTC

Main is rebased with shared AWS changes and has no unresolved conflict.
Direct HTTPS push is unavailable (terminal credentials absent); completed
commits remain on main for the host/application sync. HTTPS fetch succeeds. Four
old autostashes intact. Latest generated REPORT/PDF180pages, guarded stateful-accumulator
publication012500Z completes25.769s with source/bounds/orphan checks. Completed sources
and numerical artifacts stay frozen. No local training/job active; host memory
remained~11GiB available, numerical peak470848KiB.

New numerical priority accomplished: corrected causal native language
all-target full-write replay port, private/shared L8,16EVERY-parameter/
causality/actual Adam recovery contracts and productionp16/T16 resource
admission.512fullshadowlanes/8192events at~67.8MF/target, actual next partial
recovery exact, peak429912KiB. This is numerical mechanism admission, not
a fitted language benchmark. Existing AWS10M ORIGINAL-TEACHER controls are
independently owned and unchanged; current long-run status is in HANDOFF.
Main fine-packet/deeper gesture queue curie_dvs_followup_20261002T183000Z.txt
remains independently owned. No new local dense fit.

Findings: hard-score clipping can erase native emitter gradients; bounded
bridge restores rank but the matched fresh smoke never clips and shows no
quality gain. Actual saturated-site utility is mostly unfavorable/tiny.
Credit horizon independently changes actual text8 downstream utility: first
frozen audit4/8site reversals, aggregate cosines-.6008/-.7926. Fresh-span
confirmation3/24 reversals, but mean cosines+.9994/+.4552: BOTH descriptive
aggregate-opposition gatesFAIL. No universal main-bottleneck claim or longer
horizon fit nomination. Preserve positive observations and failed confirmation.

Next useful work: complete the corrected replay BENCHMARK driver after
the now-passed stateful accumulator contracts. The helper/accumulator are
installed/numerically verified, but active10M language driver still uses original
teacher. Its traced path bypasses learner.accumulate and calls original forward
directly; a new driver must implement corrected replay in BOTH traced/untraced
windows, preserve stream RNG/replay counters and audit real Adam/cursor/state
recovery before a fixed language quality comparison. Full replay is expensive; broader frozen longer-suffix shared-parameter
utility/covariance/discovery accounting can inform horizon/support allocation.
Do not launch a large unchanged bridge, offset, reception or failed confirmation
variant, or infer any advantage from pending scores/projected work.

## Clock-preserving calibration and joint information contracts, 2 October23:23 UTC

23:30 continuation: safe state-scope publication232800Z passes161-page bounds/
orphan/source validation. Original161-page REPORT/PDF preserved in appendices;
explicit clarification on both calibration pages:720live-state bytes is maximum,
576/648/720 correspond to6/7/8 occupied units. No numerical/source mutation.
Concurrent automatic rebase onto a46e5d4 caused REPORT conflict; resolved by
preserving generated evidence plus upstream exact-query appendix, then all
remaining commits/autostash applied cleanly on main. Four earlier autostashes
unchanged. New query-only upstream text remains in canonical Markdown and a
versioned appendix; it is not yet a generated PDF page. Latest AWS depth4
teacher/factorized/replay pilot summary234200Z fails both quality/work gates;
confirmation null and no full-fit nomination. Preserve that independent result.

All local jobs complete under unique one-job safe queues; no local trainer
active. Main only, existing four autostashes preserved. New numerical sources
and completed notes101/102 are frozen. Preserve all original and new results.

local_clock_preserving_contracts_20261002T231200Z completes29.010s/354328KiB:
131072 sampled three-way clocks, conditional-uniform and winner/first-time
screens, common score-shift coupling, exact native tau1 output/state/end-RNG/
all-parameter gradients, positive-temperature sparse activity/target invariance
and training refusal. Conditional losing-residual uniform preserves correct
first-time independence; simply reweighting the original winner while retaining
its time would not. Positive-temperature training remains uninstalled/refused.

local_dvs_clock_preserving_calibration_20261002T231600Z completes115.746s/
356292KiB:16 prespecified unused FIT inputs DISJOINT from notes98..100 cohort,
four noise histories, two initial/fixed-pass4 seed6/7 pairs xfour settings,
1024 prefixes. Every prediction/readiness/work trace retained; weights fixed.
Temperature2/all NLL gains .011154728/.002256550, accuracy gains3.125/0 points:
BOTH .02 gates FAIL. No unchanged integrated fit or selected tau4 rescue.
Every prefix still scores168 keys/writes84 actual receivers;8 available units,
576..720live-state bytes (720 is maximum). Representative tau1 .591695..591849MF vs tau2/all .594131MF
includes actual calibrated transforms/cumulative sums; not whole-audit FLOPs.
Original trained fit2.285696GF/2.232125MF per1024 presentations each paid.
No DEV/test/optimizer. Current-state common rate retained; later altered state
legitimately changes times. Frozen post-fit softness does not test wider route
support during training or rule out useful learned confidence calibration.

Theory102 derives joint winner+first-time Fisher diag(pi), whereas winner-only
Fisher diag(pi)-pi*pi^T has a common-clock null direction. Explicit log-total
rate/categorical coordinates have orthogonal LOCAL information blocks without
removing actual shared-content/history coupling. Guarded9 exact-moment/
independent-quadrature/shared-Jacobian/noise contracts complete .530s/284704KiB;
max Fisher error4.77e-15, risk-gradient error6.40e-14. Inverse metric amplifies
rare-coordinate score noise by sum1/pi (1001 at pi=.001). Per-score division
is not shared-parameter natural gradient. These are local identities, not an
installed optimizer or empirical improvement. Logged scalar-conversion warning
retained; no numerical failure or model mutation. Full audit work unknown.

Guarded local_route_calibration_report_20261002T232000Z completes:161-page
REPORT/PDF pass every page bounds/orphan/source check. Eight new pages include
99/100/101, all AWS critics/calibration/allocation (including fresh score gains
and actual parameter failures), compact context plus raw4/raw20 controls, and
original/factorized/all-race seed7 common-unit fit/inference ledger. Verbatim
old manual AWS appendix preserved in report/appendices/aws_allocation_history_
20261002T231900Z.md. Failed gates do not erase learned-feature or score-allocation
positive evidence. Report source module and publisher now frozen too.

Prioritized integrated model: native p16/L2/H2/pool2 with computational times,
sparse memory writes, key/value separation and unrealized-route credit.
Independently owned replay confirmations queue:curie_dvs_lanes_20261002T224500Z;
generalization/AdamW/input-noise/coarse campaign:curie_dvs_followup_20261002T183000Z.
Do not duplicate or invoke these multi-job owner queues locally. AWS owns
parameter-targeted allocation and deferred larger Transformer comparisons.
Next local calibration requires measured shared-parameter noise/utility before
any inverse-metric optimizer; numerical identities alone admit no training.
Remaining mechanism gaps: learnable reception boundary credit/full native
gradients/recovery/accounting, irregular-stream silence-reset scheduler,
capacity beyond scored keys and practical matched-quality total-resource gains.
No proposed departure from the integrated temporal/sparse direction.

## Memory-key confidence decomposition complete, 2 October23:02 UTC

Unique guarded local_dvs_key_score_decomposition_20261002T230300Z completes:
5,376 race score PAIRS (10,752 candidate scalars), two fixed-pass4 producers
and initial reservoirs,16 unused FIT prefixes/one noise history. This clarifies
result field scalar_score_decompositions=5376, which counts race pairs, not
individual candidates. Eight explicit logit/state/end-RNG contracts include
an independent RNG check of the99 arrival collector. Every weight fixed;
no loss/optimizer/DEV/test. Static vs memory-read score algebra reconstructs
within float32 tolerance; full immutable arrays retained. See result for costs.

Trained mean |static gap|.110/.128 versus |memory-read gap|5.322/4.340;
memory gaps larger in93.75%/92.86% of races. Static-only entropy.996/.996;
factual .333/.402, dynamic-only .340/.404. First-event memory reads zero and
full routing nearly uniform. Memory-dependent confidence is demonstrated,
not harmful routing, causal generalization failure or a normalization fix.

Theory100 includes primary ST-MoE z-loss precedent and the reason blind router
normalization/penalties change this substrate's actual common clock. Next
independent frozen diagnostic: calibrated choice temperature with first-time
law/rate total retained, actual memory writes, disjoint producer-unseen FIT
inputs, no new decoder/encoder fit. Other-host exact all-race replay firstfit
fails quality; read current theory406/protocol before any learning escalation.

## Race-scaled reception support completed, 2 October22:52 UTC

New theory99 proves a relative deadline g((1+c)T), exact common-speed-shift
membership invariance, closed expected receiver count and unchanged native
11ms local delay bound. Six numerical contracts pass. Unique guarded
local_dvs_race_support_20261002T225200Z collects43,008 actual races across
all84sites, two fixed-pass4 native producers+initial reservoirs,16 unused FIT
inputs x8 whole-history draws. No hypothetical deliveries/writes, optimizer,
DEV/test evaluation or automatic training admission. Original logits/state
reproduce, all weights fixed;59.553s/335536KiB. Full score/arrival/site/noise
profile preserved with checksum; prior trained core fits2.285696GF each.
FLOPs/traffic/energy unknown, not zero. Positive prototype remains unfitted.

Trained normalized route entropy .335/.399 versus initial .750/.726;
99%-probability concentration44.47%/34.59% across allsites. Former event19/
L0/H0 site is more decisive (71.88%/57.03%). Relative c4 expected receiver
counts1.228/1.272 with actual meanextra waiting1.406/1.675ms; fixed3ms
expected1.341/1.394. Earlier1ms single-draw lack of extra reception is local,
not proof all alternative arrival support is absent. Relative timing fixes
common-shift dependence but does not solve categorical concentration or
establish useful alternatives. No unchanged failed one-site fit is admitted.

Next independent diagnostic: decompose native score into query/static-key/
clock-bias and query/key-read(persistent memory), preserving actual forward
and RNG. Test whether memory-conditioned routing rather than static keys
accounts for confidence; no key normalization fit before that evidence.
Other-host episode-batched corrected replay/shadow quality remains prioritized;
AWS owns critic calibration. Generated153-page report and all earlier results
remain valid while new profile/critic report pages are prepared.

## Git conflict repaired; producer-held check agrees, 2 October21:57 UTC

Autostash REPORT.md conflict after shared-main update is resolved by retaining
both the upstream AWS appendix and newly generated local diagnostic sections.
No active rebase state existed; index conflict cleared without reset/skip or
stash deletion. Four autostashes preserved. New PDF144pages and publication
214300Z complete with bounds/orphan checks; REPORT.md additionally retains the
upstream manually appended AWS evidence. Future renderer must retain those
appendices too. Existing frozen numerical/report publication source hashes
and all result files remain intact.

Producer-held affine head check215100Z completes both fixed-four-pass native
pilot encoders. Conditional decoder CV and producer-unseen728-FIT selection
BOTH choose nominal final C.1 in seeds6/7. There is no demonstrated selection
failure difference here. Final heads refit on all984 features: seed6 54.1667%/
1.222593, seed7 53.6458%/1.177975. These are unequal-readout-data comparisons
against256-only pilots, not native advantage or confirmation of a new joint
training repair. Mean L2 is kept invariant across folds/256/984; probabilities
port into the unchanged affine native head, eight winner-only serial checks
and all192 batched checks agree, all nonhead parameters preserved. Four
contracts include the constructed confidence-selection counterexample; it
is a possibility theorem, not evidence it caused this benchmark failure.
Both candidates/selections and all solver/core/verification costs are saved.

Theoretical/calibration evidence does not yet justify another guessed global
normalization or head regularizer fit. Next core-access candidate needs a
bounded integrated construction and controls for regularization versus extra
message information. Preserve sparse temporal mechanisms, actual writes and
correct first-time law; learned reception windows remain a concrete mechanism
candidate, with only primitive79 contracts presently established. Other-host
AWS coarse/readout/quadratic outcomes stay preserved and separately owned.

## State-access signal and regularization qualification, 2 October21:42 UTC

Persistent-state probe213300Z completes38.100s/459264KiB. Frozen initial query
57.8125%/1.131057 -> augmented63.5417%/.886238; selected query67.1875%/.999857
-> augmented71.3542%/.859084. Q32 plus all causal pre-query raw memories/age/
seen =176 features. Serial state, native and old-probe probability reproduction,
bitwise encoder preservation pass. Same984/192 data,9-cell3-fold FIT-only head
selection, no encoder updates. Complete sampled32/24 core replays1.735341GF
per encoder; selected prior encoder fit20.075193GF retained. Solver/grid/
feature-materialization/traffic/energy unmeasured, not zero. Dense probe control,
not a sparse inference architecture or practical advantage over RBF73.44%/.7065.

Partition probe214000Z completes12 fixed outcomes (four9-cell partitions and
two fixed decoder-setting swaps, both initial/selected). Cached features mean
zero EXTRA core replay, not zero total work. Selected payloads71.3542%/.860321,
clocks67.7083%/1.017809, layer0 70.8333%/.838315, layer1 69.7917%/.840329.
Crucial correction: selected query alone at the stronger state-selected C.1
reaches67.1875%/.889842. Most apparent full-state gain is regularization;
state adds only.030759 at C.1, and at weaker query-selected C1 full-state worsens
to68.2292%/1.278955. No uniquely established memory-access cause or depth
premium follows. Preserve original positive signals AND these qualifications.
Initial same-setting query/state comparison still improves1.131059->.886238.
Theory94/95 and all numerical dependencies are now frozen.

Next bounded direction: producer-held decoder selection on existing fixed-
four-pass native pilot checkpoints, using fitting examples never trained by
those encoders. Conditional decoder folds reuse label-trained features and
can prefer overconfident heads; strong regularization gain needs a selection
protocol check. Keep inference architecture unchanged if an affine head is
ported; no all-state probe is silently promoted. Existing covariance gate
fails and fresh-noise fits remain stopped. Other-host replay/depth/tied work
stays owned there; AWS remains the place for new dense controls. All queues
serial, one thread/nice19,3M KiB VMS/1.25M KiB RSS and8GiB memory floor.

Report source now includes completed noise/state/partition evidence and the
confounding controls, with common known-work units and unknown totals explicit.
Consolidated guarded publication is pending; canonical141-page report remains
until validation. No training job currently runs. Main and stashes preserved.

## Common-noise covariance does not support a fitting change, 2 October21:27 UTC

Frozen audit212700Z completes56.443s/405676KiB with four fitting prefixes per
saved native seed6/7 and32 independent whole-history draws each. Exact sample
covariance decomposition/direct-matrix and identical-clip scaling contracts pass.
Route-map shared/independent variance ratios1.001656/.991189; whole-gradient
ratios.997872/1.001780. Both miss preregistered1.20 route-map gate. No proposed
fresh_shared/fresh_independent fitting campaign is admitted. Time-map ratios
1.04860/.97407; no large covariance penalty is observed on these prefixes.
Native weights unchanged, zero optimizer updates, complete first-draw trace
coverage and .637545GFLOPs estimated diagnostic work. Finite four-prefix scope
excludes a causal explanation of seed failures or adaptive noise reuse.

Theory93 retains covariance/freshness derivations and Flipout's primary-paper
analogy with explicit limits. Next bounded diagnostic: compare the saved query
feature probe against that SAME feature augmented with causal pre-query
persistent memory/ages/seen flags. A stronger decoder can reveal retained
information inaccessible to the selected readout; it is a diagnostic dense
read of dormant state, not the main architecture or inference advantage.
Keep the learned-vs-initial comparison, fitting-only decoder selection, source
and probability reproduction contracts, full replay accounting and strong
controls. No new Transformer/LSTM, offset/choice refit or duplicate external
replay campaign. All numerical jobs remain serial under the safe host guard.

## Replay-law and critic-sampling correction, 2 October21:16 UTC

New theory92 audits the newly arrived external §402/403 proposal without changing
its model/queue/results. The actual local-expectation wrapper forces a losing
alternative at its individual arrival; its softmax-algebra and factual-winner
identity tests do NOT establish exact conditional choice credit. A time-only
rates1/3 witness has true score gradient[-.0625,-.1875], but adding that forced-
individual-time choice correction to factual pathwise credit yields[+.0625,
-.3125], reversing one direction. Correct first-time-preserving choice and
separately consistent clock estimators are required. Joint-score/factorized-law
alternatives are derived, not promoted as completed integrated repairs.

Four completed212000Z contracts verify analytic expectations, actual one-race
wrapper time identity at four seeds, fixed-critic subset cancellation and a
sample-adaptive critic counterexample. A critic must be fixed before its residual
correction sample (or use justified independent/cross-fitted conditioning).
Unbiased residual sampling cannot repair an already wrong base replay target.
Own script/note92 are frozen; audited external hashes record observed versions
and do not freeze another host's owned implementation. Qualifications now sit
beside the original §§402/403 proposals; preserve their empirical outcomes as
combined-intervention estimator evidence. No local new training is admitted.

Final publication212300Z completes21:15:47 UTC:141-page canonical report
retains theory92/contracts, both offset seeds and all older valid comparisons.
PDF text bounds/orphans and git diff checks pass;140-page stage is archived.
No training or numerical job remains active after serial guarded publication. Next integrated credit
candidate must first demonstrate correct time law, legal writes, estimator
consistency, parameter gradients, recovery and complete work on a small fit.
Other-host ownership and frozen source/results remain preserved. Do not duplicate
its existing long queues or interpret their present algebra tests as exactness.

## Coupled evolution offsets and formal geometry, 2 October21:10 UTC

Native integrated p16/L2/H2/pool2 remains the prioritized architecture. The
user-requested bounded reception-phase offset retains physical-age damping,
physical-time rotation derivatives, actual races/timestamps, sparse writes and
native counterfactual score credit. Beta=pi*tanh(raw_offset),32 extra parameters;
message-phase ownership is explicit, inactive Adam state frozen, shared maps
remain active. Four contracts205000Z pass native zero nesting, nonzero serial/
batched all-state/all-parameter gradients, physical-age/offset directions and
actual interrupted phase/Adam/RNG/work recovery. Both24-fit/eight-dev/two-pass
smokes205200Z/205300Z learn with complete accounting. Sources and theory90 frozen.

Manifest205400Z declared joint AND alternating seed6 pilots before results.
Matched256-fit/192-dev/four-pass comparisons: native54.1667%/1.305937,
joint57.2917%/1.258170, alternating53.6458%/1.263009. Full fitting work
2.285696/2.287775/2.281584GFLOPs; per-presentation2.232125/2.234156/2.228110MF;
inference.591737/.596711/.596655MF. Both gates pass; minimum NLL chooses JOINT.
Alternation does not demonstrate an independent benefit and changes private
block update counts/clipping. Unique result/queue names205500Z/205700Z and
completed common-unit comparison210100Z preserve all four passes.

UNCHANGED seed7 confirmation210400Z FAILS: native55.2083%/1.306508 versus
joint offset53.1250%/1.327347, improvement-.020839, decline2.0833pp;
work ratio1.000910. Comparison210800Z validates matching initialization/data/
noise/exposure/accounting. No full-data offset fit, extra epochs, third seed or
unselected alternating-seed fishing is admitted. Preserve first-seed gain and
failed replication together. Both pilot fits use~348MiB RSS/~101seconds.

Completed fitting-only branch audit203600Z (two prefixes per saved seed,
four draws, events9/19/both layers/head0) finds nearly unchanged sampled versus
enumerated full-gradient directions, mild route/content opposition. This small
scope does not establish a dominant exposure/interference bottleneck. Independent
203400Z gather/all-gradient contracts pass. Theory89 and204000Z contracts correct
the other-host exact-pi unchanged-expectation claim for winner-dependent errors:
convex quadratic direction reversal; earlier curie192000Z audit also changed
identity AND time law. Original numbers remain beside revised interpretation in
note59 section400. Other-host candidates/results/queues stay preserved/owned.

Theory91 gives an exact Schur-complement local improvement criterion and
phase/age determinant including saturation conditioning. Four independent
211000Z matrix contracts pass16 direct/eliminated solves, zero projection,
redundant-coordinate penalized benefit and determinant. This is formal progress,
not empirical validation of the offset cure. Notes90/91 and numerical dependencies
are source-frozen. No universal gradient break or deeper-feature impossibility
follows from these bounded failures.

Next: publish this completed stage with retained strong controls and both seeds;
then prioritize a fitting-only predictive-access/credit diagnostic that can
separate representation information from finite readout and route utility.
Do not start another unchanged offset/choice campaign. Other-host tied-map/depth/
exact-pi work and AWS large Transformer comparisons stay separately owned.
Host curie CPU-only,11.5GiB available; one-job lock, one thread/nice19,
3,000,000KiB VMS/1,250,000KiB RSS watchdog and8GiB available-memory floor.
No active training after confirmation. Main branch; no unresolved index conflict,
three stashes preserved. Publication211300Z completes21:09:24 UTC:140-page canonical PDF and REPORT.md
retain both seeds, diagnostics, corrected teacher scope, contracts and theory91.
PDF bounds/orphans and git diff checks pass; prior134-page evidence is archived.

## Bird's-eye decision and failed confirmation, 2 October20:20 UTC

Read theory87/88 for the current synthesis and primary-paper-inspired plan.
Objective remains useful trained capacity beyond sparse active temporal work;
no broad superiority is established. Prioritized inference model remains
native p16/L2/H2/pool2. The next justified research job is a small fitting-only
frozen joint-route/content/clock diagnostic, not unchanged full-data training.
Distinguish lost information, incorrect credit, noisy/rare content exposure,
shared-parameter interference and private-map dilution before selecting ONE
repair. Alternating route/message updates is conditional on measured utility
drift/interference, with coupled forward computation and explicit shared-map
ownership retained. Do not detach dependencies and call it just a schedule.

Choice correction seed6 passes but seed7 FAILS: local55.2083%/1.306508 versus
treatment55.2083%/1.336201, gain-.029693NLL, work ratio1.33867. Both fixed
256-fit/192-dev/four-pass comparisons193500Z/192100Z are completed and preserved.
No unchanged full984-fit stage, third seed or extra passes are admitted.
Seed6's+.111931NLL/+4.6875pp remains exploratory positive evidence.

Five calibration-coordinate contracts193300Z pass; note85 is source-frozen
with its original then-pending replication statement, superseded explicitly
here and in notes86/87. Five paired-credit contracts195800Z pass finite
2/3/8/64 expectation/proposal tests, exact2-route native state/parameter-gradient
nesting,8-route actual legal writes and actual2/8pool Adam/cursor/RNG recovery
with U3/partialU1 and complete operator coverage.59.332s/380176KiB. Failed
195200Z unsupported-op audit remains preserved; equivalent covered subtraction
passed in the fresh run. Numerical dependencies and notes84–86 stay frozen.

Paired pool8 readiness smoke201100Z completes27.533s/358024KiB:24fit/eightdev/
two passes/U16+partialU8,48presentations/four updates. FitNLL2.51701->2.17388;
tiny selected dev12.50%/2.29890, not a benchmark quality claim.42,091params/
32receivers,672inferencekeys/84commits/672candidatevalues,2376statebytes.
Fullfit.414378GFLOPs/8.632876MFLOPs per presentation/.872348MFinference.
One full alternative forward/window and epsilon.1 proposal; candidate scoring,
full shadows and optimizer charged;8auxiliary exponentials/target separate.
No direct losing-message derivative from no-grad shadows; winner sampling can
still have the correct branch derivative in expectation. Exposure/variance,
other local teachers and future timing-jump terms remain unresolved.

Other-host tied-map capacity and depth-fidelity/nested-growth work (note59
§§398–399) remain separately owned. AWS exact prefix-reuse readiness result
saves10.738% counted fitting work with identical curves/parameters/Adam/RNG;
scope24fit/eightdev/two passes, not strong-control superiority. Report source
now retains that evidence plus both credit seeds and new contracts/readiness.
Previous canonical129-page publication192900Z passed bounds/orphans after
the preserved192300Z orphan failure. Fresh consolidated publication202000Z
completes20:18:57; PDF bounds/orphans and git diff checks pass. It retains both
credit seeds, calibration/paired contracts, larger-pool readiness, primary-paper
inspiration and the other-host exact prefix-reuse saving with parent hashes.

No numerical job remains active after the smoke. One thread/nice19,3,000,000KiB
VMS/1,250,000KiB RSS watchdog and8GiB MemAvailable floor; host available~11.5GiB.
Main branch, three stashes preserved; no unresolved git index conflicts observed.
No local dense fit, official-test read or duplicated other-host campaign.

## Actual-write choice correction passes first pilot, 2 October19:20 UTC

Prioritized native integrated p16/L2/H2/pool2 remains unchanged in inference.
Theory84/new driver replaces one event9 head's value teacher by exact
conditional actual-write/full-suffix choice utility; original native timing
and factual content derivatives retained. Four contracts191100Z pass independent
choice/time derivatives, factual/state and legal-alternative replays, every
parameter-gradient residual VJP and actual Adam/cursor/RNG recovery/accounting.
Smoke191400Z passes24fit/8dev/two passes/U16+U8: fit2.64095->2.11061NLL,
23.041s/343548KiB, .143986GFLOPs/2.999703MFLOPs per target/.591695inferMFLOPs.

Matched256/192/four-pass seed6 pilot191900Z completes103.948s/347972KiB:
selected pass3,58.8542%/1.194006NLL versus saved local54.1667%/1.305937.
Gain.111931NLL/4.6875pp, full work3.059796GFLOPs/2.988082MFLOPs per target,
.591709inferMFLOPs; work ratio1.33867 passes<=1.50 gate. One-seed development
evidence, not practical advantage over full984-fit strong controls. All four
passes retained. Gate analysis192100Z validates matching data/initialization/
settings/work and admits seed7 local+treatment. No larger stage until the
same seed7 gate passes; queues192400Z/192600Z are separate one-job definitions.

Calibration audit191600Z uses frozen full/local selected weights and first16
fit prefixes, same event/head/draw, no optimizer. On full weights, joint-clock
raises gradient norm3.3901->6.4002, cosine.3940 and global clip scale.2950->.1562;
unchanged decoder gradient is attenuated too. Choice-only stays3.3915/cosine
.99998/clip.2949. All paid forward/replay/backward/normalization coverage passes.
31.254s/365148KiB. Single-draw balance diagnosis, not expected covariance,
Adam trajectory or causal proof. Positive normalization cannot fix a wrong
direction; preserve centered legal utilities and timing scope rather than
equalizing arbitrary gradient RMS. Theory85 calibration coordinates follow.

Other-host completed pool8/depth4 fits are negative versus clock pool2:
62.50%/1.08802 at57.2985GFLOPs and57.29%/1.22159 at37.8686GFLOPs;
clock reference66.15%/1.04199 at20.0747GFLOPs. Sources unchanged. Their tied-map
and exposure/replication hypotheses remain independently owned, not duplicated.
The other-host multi-job queue definitions must be split before local admission;
this session runs only unique one-job queues via run_safe.

Report source includes completed pilot/smoke/calibration and those external
capacity results, all controls retained; publication192300Z is prepared pending
guarded render. One thread/nice19, VMS3000000KiB/groupRSS1250000KiB/8192MiB
floor, no GPU or official-test read. Three stashes retained, main branch.

## Completed credit gate and variance diagnosis, 2 October17:50 UTC

State-clock pilot174300Z completes17:43:02:47.3958%/1.450687NLL versus local
54.1667%/1.305937; gain-.144750NLL, accuracy decline6.7708pp, full work
ratio1.33904. The quality gate fails; no unchanged seed7/scale-up. Completed
comparison174600Z and124-page validated publication174700Z preserve results.
Smokes and exact isolated-node contracts remain valid despite failed quality.

Variance audit175100Z passes23.205s/314672KiB: first four fitting prefixes/model,
both layers/heads at event9, both legal actual writes/full suffix at8/16 time
quadrature nodes. Common-clock RMS is1332x/local and904x/treatment choice RMS;
oracle prefix/future-specific baseline removes>99.9999% estimated score-space
variance.192 paid legal full forward shadows/model, no optimizer. Limited
fixed-future probes and finite quadrature are not independent confirmation,
batch covariance, causal explanation of regression or cheap baseline proof.
Theory82/83 derive variance decomposition and retain every negative result.

Next prioritized integrated model remains native p16/L2/H2/pool2 temporal
addressed memory. Isolate full actual-write *choice* credit, retaining the
original pathwise clock derivative and ordinary factual content gradients.
Next variant is specified, not fitted. Need separate component/commit/gradient/
recovery/accounting contracts, then24fit/8dev/two-pass smoke before fixed256/
192/four-pass/s6 comparison. Same.02NLL/1pp/1.50work/RSS900000 gates. Larger
memory support/windows and independent-noise/learned-baseline repairs remain
separate conditional hypotheses. Core mechanisms/inference remain; downstream
timing jumps and uncorrected route teachers remain explicit learning gaps.

No numerical job is left active after the completed diagnostic. Main branch,
three stashes preserved; one-thread guards and8GiB host floor retained. No
new local dense training or official-test read. Other-host capacity ladder and
AWS assignments retained. Variance report addition completed under the guard
at17:54:34; publication175400Z passes PDF text bounds/orphans and git diff
checks.125-page canonical report preserves all pilots and diagnostic scopes.

## Native DVS credit diagnosis and bounded correction, 2 October17:40 UTC

Read other-host capacity ladder and AWS completed dense calibration before
continuation; preserve their files and assignments. No concurrent local job.
New systematic decision: COUNTERFACTUAL_CREDIT_PLAN.md and theory80/81.
Original p16/L2/H2/pool2 native remains the integrated model. Correct actual
write/suffix and joint-clock utility at unchanged inference before widening
the counterfactual bank or adding integration operators. No general no-go or
promised cross-benchmark benefit follows from the current evidence.

Matched256-fit/192-dev/four-pass local/pair seed6 pilots completed. Local
54.1667%/1.305937NLL versus terminal pairs56.7708%/1.349807. Pair NLL is worse
by.043869; original .02 gain gate fails, so no unchanged second seed/extension.
All completed curves/results retained; comparison174000Z includes common units.

Frozen original route audit:35.8% of16,128 choices differ from initialization;
message/context/route/time/decoder groups all update. Full-suffix audit172000Z
replays32 sites/model on first four previously used development clips. Original
teacher opposes exact conditional choice utility at4/15 nonzero-comparable
sites. One legal alternative has value-only benefit-.01016 but write-only
harm+.10514 and full-route harm+.09193. All four original-model clips were
correctly classified. No attribution of the global quality gap or independent
confirmation. Failed original audit171500Z and its log remain preserved.

Theory78 actual-reference contracts prove a convex cross-entropy teacher
direction reversal. Theory79 silence-burst/popcorn primitives pass causal
deadline/EOF, fixed-partition gradients and exact finite timeout-bank risk;
not an integrated learned-window/popcorn fit. Need irregular timing adapter,
scheduler and merge/split credit before admitting that architectural change.

New bounded correction driver dvs_state_clock_credit_benchmark.py replaces
one earlier node's entire value/time score derivative with conditional-winner
joint winner/time likelihood credit, using actual alternative commits/full
suffix and a pre-draw detached baseline. Ordinary factual content derivatives
and all other local teachers remain. At event9, rotate head then layer across
fixed passes; full shadow forward paid, no extra inference work/parameters.
Contracts173000Z pass analytic smooth/jumping-time gradients, factual forward/
state equality and actual model/Adam/cursor/RNG recovery with complete work.
Smoke173700Z passes24fit/8dev/two passes/U16+U8: fitNLL2.64095->2.15847,
23.406s/343276KiB, .144025 whole-fitGFLOPs/3.000525MFLOPs per target/
.591695 inferenceMFLOPs per prefix. This is admission, not quality evidence.

Next: fixed256/192/four-pass seed6 correction versus saved local pilot.
Admit seed7 only for NLL gain>=.02, accuracy decline<=1pp, whole-fit work
ratio<=1.50, RSS<900000KiB. No dev-selected schedule/pass extension. Full-data
and practical repeated inference/control comparison require confirmation.
One unique one-job guarded queue, one thread/nice19, VMS3000000KiB,
groupRSS1250000KiB/8192MiB available floor; tmux for the pilot. Sources
frozen after contracts, no GPU or new local dense fits. Three stashes preserved.
Completed-stage report publication174100Z is prepared, not yet claimed passed.

## curie: DVS capacity ladder blocked on the data artifact, 2 October, 16:50 UTC

The ladder's smoke failed before training: curie lacks
experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.data.npz
(sha256 f1287286cd097dc8f13fd09c8dcf09131fd166dee8b2ae44c666d0769bc7e53f, git-ignored by
experiments/results/**/*.npz, about 3 MB) and the raw DVS128 Gesture AEDAT tree (data/dvsgesture/DvsGesture).
Request to the host that produced it: copy or force-add that npz (`git add -f`) so curie can run the ladder;
the driver verifies the checksum. The joint v2 observed fit was stopped after 2 of 8 passes (46.9% / 47.7%,
incomplete) to free the host, and its rank control was withdrawn.

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

## Reception-law and real-native controls, 2 October22:22 UTC

Theory97 seven numerical contracts complete: full residual arrival law,
physical deadline/cap, conditional expectation vs history+boundary+winner
gradients, independent finite differences, H0 birth and actual receiver writes.
Constructed width gradient sign reverses (-27.66249 vs+.65662 ordinary), not
quality evidence. Theory98's44 frozen native interventions complete104.222s/
396720KiB,12 contracts, no optimizer/DEV/test. Two fixed-pass4 seed6/7 producers
and initial reservoirs,16 unused FIT inputs, event19/L0/head0 only. 1ms receiver
activity is winner-only in both trained seeds; mean-vs-wait NLL gain0/0 FAILS
predeclared smoke gate. 3ms extra receivers4/16 and7/16 with onlysmall mean
NLL gains; no changed-gate promotion. Positive-width training refuses execution
until actual complete native gradient/recovery/accounting contracts exist.
Results/source notes now frozen. Preserve all44 outcomes and source lineage.

Rebase conflict HANDOFF resolved retaining AWS fixed-polynomial negative study
and local law contracts; main updated, four autostashes preserved. Generated
149-page report remains valid; latest AWS polynomial manual appendix plus local
reception results need the next guarded generated publication. Do not duplicate
shared-main replay/shadow-lane or AWS compact-head jobs. No unchanged one-site
window training admitted. Existing pure popcorn primitive is still not fitted.

The153-page generated report now includes theory97 contracts, both seeds'
all44 native outcomes/common resource scope and the AWS frozen polynomial
negative study. Guarded publication222500Z passes all layout/source checks.
Verbatim historical AWS appendices preserved. No local active training/job;
next integrated numerical priority is independently owned replay/shadow-lane,
not an unchanged failed reception smoke. Read HANDOFF before admitting work.

## Native score sensitivity, 3 October00:04 UTC

Completed notes103–105 identify a real local gradient obstruction without
claiming a quality repair. Conditional native parameter geometry (36.536s/
348112KiB) has five selected seed7 histories where one pre-clamp score exceeds
12 and clock/choice score pullbacks become collinear. The fixed alpha=.1
bounded smooth sensitivity bridge restores the missing local direction at
those same histories (37.961s/371992KiB); alpha0 nests all original gradients.
Positive training remains refused by that frozen diagnostic prototype.

Saved all-site cap occupancy is9.533/7.431% of trained races; a separate raw
score cohort confirms8.557/6.994% strictly outside the bound. No saved case
has BOTH candidates saturated. A constructed two-saturated-emitter exact-risk
example verifies a flat hard-clamp trap and a bridge escape, not a native
quality result. All exposure, scope and failed nomination gates are retained.

The169-page report publication retry000400Z passes source/layout checks
(26.434s/136916KiB); first235800Z attempt failed on an unsupported figure block
and rolled back. Report includes all completed fine-packet/deeper AWS replay,
priority allocation, tied-map and query-only evidence, with common work units.
Sources/results for completed diagnostics and publishers are frozen. Main has
no unresolved conflict; four old autostashes remain intact. No local job active.

Next: separate production/reference bridge implementation, alpha0 and positive
training/recovery/accounting contracts, then only a tiny matched integrated
learning smoke. Other owners retain full fine-packet/deeper replay and AWS
comparison priority; do not duplicate their queues. No native or practical
advantage established by local rank restoration.

## Integrated bounded bridge learning and utility, 3 October00:29 UTC

The production/reference and batched sibling bridge paths retain physical
races, key/value separation, sparse actual receiver writes and factorized
clock plus all-route first-time replay credit. Zero bridge exactly nests old
batched outputs and all gradients. Eight double training/reference/full-route
Adam/recovery/partial-window/accounting contracts pass (9.065s/344424KiB).
First attempt failed on absent-vs-explicit-zero unused gradients; corrected
comparison still rejects every unmatched nonzero gradient. Failed queue retained.

Matched restricted p8/L4/H2/pool2 seed7 smokes32FIT/16DEV/twopasses/64
presentations,U8/eightupdates,all-route: hard31.25%/2.281937DEV versus
bridge31.25%/2.281964; FIT NLL2.829119→2.037279 /2.829112→2.039687.
Hard9.769170GF/152.643283MF per presentation/.333083MF inference;
bridge9.827467GF/153.554179MF/.335771MF.16available receivers,16key
scores/eightwrites per event,720live bytes in audited prefixes. Both learn,
no superiority; bridge smoke does not improve.61.158/64.604s,peak373200/
374120KiB. Initial AND selected models have0cap exposures across5376
FIT races each; max raw2.59/2.30. Thus this smoke does not test saturation.

Fixed-pass4 seed7 FIT258/981 labels now revealed for diagnosis. Five
strictly saturated old sites: actual hard-map suffix alternative is better
in one case, worse in four. Forced original winners recover ALL original
logits/state/end RNG. All emitter choice sensitivities restore but are only
6.14e-11..5.36e-9 because alternate probabilities remain~1e-7..1e-5.
Ten forced plus five factual forwards; extra168raw dots/forward paid.
Conditional utility derivatives pass analytic/finite-difference checks, not
whole positive-bridge risk. Audit18.432s/328716KiB,full auditFLOPs unknown.
Theory107 derives positive-slope/raw-gradient/probability determinant;
rank repair alone does not establish meaningful predictive utility.

No larger bridge fit admitted; its main-obstruction hypothesis is unsupported
by these measured sites. Preserve this low-cost contracted architectural
option and its limits. Prioritized main integrated p16 deeper replay/growth
queue remains curie_dvs_followup_20261002T183000Z.txt, owned separately;
AWS dense comparisons remain reserved. Missing coverage: fitted multi-arrival/
popcorn, affordable larger associative support, comparable-quality total work
and deeper credit generalization. No departure to a dense carrier.

Guarded report publication003200Z completed with171 pages and passed all
source/layout checks. Its module/publisher and completed numerical sources
are frozen; report and common-unit ledgers are committed on main. Four old autostashes
preserved; no unresolved index entries. One guarded job at a time throughout,
minimum8GiB available; local MemAvailable stayed~11GiB.

## Shared-main conflict and depth8 update, 3 October00:36 UTC

Fetched30 AWS commits, rebased six local commits on main. REPORT additive
conflict resolved preserving both evidence sets; shared manual additions
archived verbatim at report/appendices/aws_depth8_history_20261003T003500Z.md.
No unresolved entries, four old autostashes intact. The174-page guarded
generated report now includes all12 depth8 pilot/confirmation rows and causal
language admission/common-unit costs. First004000Z publication failed on a
wide table header and rolled back; retry004100Z passes24.187s/layout/source checks.

Positive depth8 seed7 private replay NLL1.316790 versus1.369199/1.399592;
shared1.348356 versus1.410495/1.395239, both nominationgatesPASS. Seed8 BOTH
confirmationgatesFAIL: private1.282510 vs1.331247/1.270459; shared1.270943
vs1.283150/1.271398. Preserve positive learning and failed robustness together.
Replay~52–55x fittingwork; no full gesturepromotion/resourceadvantage.

AWS independently owns long10M-character original-teacher native L8 private
and shared runs, protocol and guards in HANDOFF; do not duplicate. New local
priority is a small all-target causal language replay numerical port: current
language models still lack corrected full-write replay credit. Read
theory/aws_20261002_depth8_language_replay_cost.md before implementation.
No changed long source, no additional dense fit or language qualityclaim.

## Corrected causal language credit port, 3 October00:42 UTC

New sibling causal_language_shadow.py and causal_language_replay_helpers.py
return every-token prediction from real nonempty detached entering state.
Full first-time-preserving alternative writes credit downstream token losses,
with factorized common-clock/winner-content derivatives. Original hard score
map remains; no bridge/calibration or changed active10M source.

Depth8/H2/p4/pool2 private/shared double models pass16 contracts,23.973s/
361084KiB. Every one of384/174 parameter gradients matches independent
sequential all-target full-write return; actual states/arrivals/context agree.
Original teacher/factorized primal state/output exactly nests. Variable2/3
token lanes from2event nonempty states, future input/target causality, losing
identity at factual FIRST time, global/end RNG, partial gradient accumulation/
Adam/private-state/cursor recovery exact, all operation coverage.96full shadow
lanes/288shadow events per3target case paid. Not a learned benchmark, not
full-stream gradient across detached boundaries. Sources/results frozen.
Next bounded production-width/T16 resource/recovery admission only; full
quality comparison needs its own fixed protocol and AWS resource allocation.

## Production language replay and horizon contracts, 3 October00:54 UTC

Private/shared p16/L8/H2/pool2 production T16 numerical admission PASSES,
86.859s/429912KiB in tmux/one-job guard.512shadow lanes/8192shadowevents;
eight prespecified suffix returns match independent sequential native replays
with factual first times/earlier losses/end RNG. Full primal state/logits match;
actual nonempty Adam/private-state/cursor/RNG next3target recovery bitwise
exact. Full16 step1.084409/1.083893GF,67.775587/67.743327MF pertarget;
partial3 .040100/.039584GF,13.366694/13.194644MF; nativeinference
.097256/.097247MF/target.32available/32scores/16writes pertoken,final2448
livebytes. Synthetic correctness strings, no DEV/test/BPC/fitclaim. Small
T3/p4 EVERY-parameter contracts remain distinct from selected fullT16 checks.

Theory110 seven exact delayed-label/sampling contracts pass.529s/282228KiB.
For constructed32delay/rhoexp(-.01),fullencodergradient-.567961 versus0
under16creditdetach; finite change improves loss. Delayedroutegradient-.114477
versus0withtruncatedutilitydespiteallcandidatecoverage. Nottext8diagnosis.
Longer64/k8 is16shadowevents/target vsfull16's512,butsingleinformative-route
variancefactor127 andlargerfactualgraph remain; counts notmeasuredadvantage.

177page publication010000Z PASSES26.000s, completedresourceandhorizon
appendicesandallold/sharedqualityevidence retained. Sources/results frozen.
No active local job afterpublication. Next: frozen actualtext8 horizon-return
audit using saved fixed-pass4 native8192 seed6 checkpoint on producer-unseen
FIT chars, preserving existing AWS10M runs and fullgestureownerqueue.

## Actual native text8 horizon evidence, 3 October01:08 UTC

Theory111 first frozen native8192 seed6 final-pass4 ONLINE checkpoint, double
weights, two producer-unseenFIT spans8192/8320,16observedwarmup/32targets,
(event0/8,layer0/7,head0):4/8 utility sign flips, sampled-site aggregate16/32
parameter cosines-.600832/-.792626. No chosen scorecap. Allfactualwinner
logits/private-state/end RNG, factualfirsttime/real alternatewrites, future
input/targetcausality/unchangedweights contracts pass.7.026s/426172KiB,
16forced32tokenforwards/eightVJPs; full54907coordinate arrays preserved.

Theory112 unchanged fresh confirmationFIT8448/8576,3race draws withfixed
warmup:3/24signflips, meanaggregatecosines+.999383/+.455223. Both preregistered
descriptiveopposition gatesFAIL.20.355s/470848KiB,48forced suffixes/24VJPs.
Omittedfutureutility is real but context/noisedependent, notuniversally
destructive16credit. No optimizer/DEV/test/longfitquality claim. Source/result
lineage and exact parent checkpoint are preserved for reproducibility; total
diagnosticFLOPs/traffic/energyunknown, notzero. Full arrays include direction,
16/32credit vectors and aggregates; savedparent original fittingwork retained.

Latest179page generated report includes both first observation and failed
confirmation. First011500Z publication failed wide lasttablecolumn, rolled
back before corrected unused retry011600Z succeeds. Failedqueues/logs retained,
allcompleted numerical/report modules/publishers frozen.

## Stateful replay training-loop adapter, 3 October01:15 UTC

Theory113 siblingRNGkernel/helpers and ReplayAccumulator pass10double
private/shared L8/p4 contracts,14.510s/345160KiB. Original chronological
teacher primal/private-state match and exact end RNG before delayedupdates;
no extra seed draw or replayadvancingrealstream. ALLaccumulated parameter
gradients agree with sequential full-write returns,2+1targetchunks. Pending
gradient/Adam/private-state/cursor/RNG/replay-counter continuation and actual
partial warmup/normalization/clip/Adam update bitwise recover. Labelchanges
leave bothpreupdatechunks predictions/state/RNG exact. Allshadow andupdate
workcovered;96lanes/160shadowevents across2+1chunks paid,notfull3's288.

Important next-driver pitfall observed: aws_depth8_language.py's TRACED loop
performs original forward/backward directly and bypasses its accumulator.
Simply aliasing ReplayAccumulator would therefore OMITreplay in audited
windows and invalidate the comparison. Build a new sibling driver with
consistent traced/untraced fullobjective and explicit replay activity/counters
before quality runs. Existing longoriginal-teacher source unaffected.

New reportmodule/publisher012500Z publication completed25.769s/
275940KiB with180pages; source/layout checks pass. Numerical and
publishing sources frozen. No local job active; no benchmark quality/10M
replayfit promoted.

## Restartable replay driver and coordinated AWS matrix, 3 October01:37 UTC

Theory114 new sibling native_language_replay_benchmark.py passes12actual
private/shared p4/L8 numerical driver contracts:83.791s/346016KiB. Synthetic
eight targets/twoupdates;256shadowlanes/512shadowevents. EVERYweight/Adam/
pendinggradient/state/RNG/counter/cursor, DEVselection/curve/work reproduces
bitwise from partlyfilledtwo-target gradients and completedoneupdate. Traced
anduntracedlearning identical; source/settings/data mismatches and completed
overwrite refuse. No real-data fitting result or benchmark advantage.

Report181pages publication retry2_013800Z completes24.038s. First013200Z and
retry013700Z fail sourcevalidation/rollback because otherhost legitimately
changed dvs_batched_reg_benchmark.py (configurableclip). Exact historicalbytes
ee3d8ada... recovered from bb7125e^ and preserved in archive/frozen_sources;
new publication wrapper resolves only this knownoriginalbinding to matching
archivedbytes. OriginalJSONsourcekeys/hashes/metrics and olderfrozenreport
modules unchanged. Bothfailedqueues/logs preserved; currentreport checks pass.

AWS has independently completed actualsix-mode driver contracts and queued
aws_language_credit_matrix_20261003T013500Z. PRIORITIZEDintegrated models:
native p16/L8/H2/pool2/private fullreplay10M alongside EXACToriginalteacher
continuations; factorized/private and depth-shared variants as slotsexit.
Source/contracts reused from108/109/113; longnewfits remain onAWS. No duplicate
local1025smoke or10M training is needed. Read sharedHANDOFF/theory AWScredit
protocol for currentcoordinator state. Remaining coverage gaps:16token
detachedcredit, allpoolkeyscoring, exhaustivewholechunkshadows and CPUemulation;
neither whole-stream exactgradient nor isoquality/resourceadvantage established.

Next complementary work: eliminate duplicate factual-winner shadow lanes by
reusing alreadycomputed factual downstreamreturn. Samefirsttime/write outcome,
so exact categoricalobjective/credit should be preserved; verify EVERYgradient,
causality/RNG/state/recovery and chargedproductionwork in new frozen siblings.
Existing AWS sources/queuedruns remain untouched. No corearchitecturedeparture.

## Exact winner-return reuse and paid cost, 3 October01:51 UTC

Theory115 winner-recording kernel + reuse helper + accumulator are new siblings;
28private/shared p4/L8 U1/U2/U4 double contracts pass25.026s/377000KiB. EVERY
gradient/objective matches oldbatched ANDindependentsequential fullenumeration;
allfactualwinnerforced outcomes match, newrecording primal/state/RNG/pathwise
gradients exact, causalinput/label and pendinggradient/Adammoment recovery pass.
Initial014300Z fails auditunsupported flip/cumsum; corrected reductions use
alreadyauditedsuffix sums, unique014400Zretry passes. Failedqueue/log retained.

Theory116 productionp16/L8/H2/pool2/T16 completewhole fittingwork falls
49.6174658%private/49.6410936%shared.512->256lanes,8192->4096shadowevents;
candidate scoring32 and selectedwrites16/target, nativeinference unchanged.
Original/reuse SAMEsynthetic16targets/nonempty state/parameters/RNG and
actualtargetnormalize/clip1/Adam/warmup16of32. Globalpreclipgradientrelative
errors9.960e-7/9.730e-7, actualAdamupdate relativeerrors.00025652/.00059413
(0.026%/0.059%); both global<=3e-5/update<=.005 limitsPASS. BUTthreeparameter
tensors perfamilymiss tightcoordinate rtol3e-4/atol3e-6: BOTHproductionnumerical
admissiongatesFAIL. DoNOTsilentlyloosen/adopt into currentqualityruns.

Strict014500Zrun abortedat privatekey_read mismatch; unique014800Zfullaudit
keeps originalthresholds, recordsfailedtensors plusALLstagework and actual
updates. Completed98.060s/399744KiB; optimizednonemptyAdam/state/pending3target
andnext1targetpartialupdate recoverbitwise. Extra recovery optimizersteps paid
outsidefirst-step comparison table. Firstpositivework saving preserved beside
numericlimitation; no trainedtext8quality/isoqualitysupremacy, no physical
projection/energy claim. Source/result paths nowfrozen. Newreportmodule adds
both proofs/work/gates without touching priorfrozenmodules.

Next: production double precision reference to distinguish differing float32
batch-shape/categorical cancellation from a changed route estimator. Verify
actualbranch agreement, EVERYgradient and unchangedRNG before proposing any
precision/centering repair. ExistingAWS10M fullreplay controls stay unchanged;
no duplicate fit is needed. Host peak~391MiB/available>11GiB thisstage.

Winner-reuse182page publication retry015500Z completes26.801s, source/layout/
orphanchecksPASS. First015100Z rendering failed a tableheader1.39pt beyond
pageedge and rolledback; originalqueue/log kept, newtable fits174mm content
width. Allsuccessfulmodules/publisher/sources frozen. Precision audit117 runs
in its ownunique guardedjob; no new quality-training arm launched locally.

## Production precision audit and remote reconciliation, 3 October02:00 UTC

Theory117 audit completes13.556s/374828KiB,4contracts. Production p16/L8/T16,
SAMErepresented initialfloat32 weights/private-state promotedtodouble (no
double reinitialization). EVERYoriginal/reuse doublegradient relativeerrors
1.439e-15private/2.735e-15shared; factualstate/logits/endRNG exact. ALLfactual
andshadow historiesmatch acrossfloat32/double:131328old and65792optimized
race decisions perfamily, no branchcrossing. Float32old/reuse relativeerrors
againstowndoubleprograms1.489e-6/1.595e-6private,2.120e-6/2.103e-6shared.
Bothimplementationshave thisprecisionfloor; no estimatorchangeidentified,
nor proofthat thissmallroundingexplains underfitting.116tightcoordinatefails
remainhistoricalfacts. Allgradientvectors androutehistories saved.vectors.npz.
Nooptimizer/quality/precisionrepair applied. Diagnosticworkunknown,notzero.

Read-onlyfetch finds13newremotecommits through2d31759, including independently
contractedAWSwinnerreuse and completedp16realtext8smokes; AWSprioritized
matrixnow aws_language_winner_matrix_20261003T014100Z. Preserve bothmodels'
originalfullreplay/controlrows alongsideoptimizedruns; no localduplicate.
Sharedmanualreport additions archived in report/appendices/aws_language_replay_
history_20261003T020000Z.md before rebase. Next reconcilemain and regenerate
report from completedJSONs including117precision andALLAWSadmission rows.

## Shared main conflict resolved and185page report complete, 3 October02:12 UTC

Integrated13remotecommits, then an automatic pull/rebase againstb147583
raised anotherREPORT conflict while publishing. BOTHtimes resolved by keeping
eachhost's additions, completed rebase onmain; currentautostash restoredcleanly,
fourolderautostashes leftuntouched. Source/result bytes remainexact. Additional
AWSproduction manualhistory archived at report/appendices/aws_language_replay_
history_20261003T021000Z.md. No unresolvedindex entries.

New frozen production_language_replay_evidence.py adds117precision, allsix
AWSactualdriver contracts/common-unit48target tables, independentAWSwinner
gradient/work proof and allEIGHTproduction-p16 real1025FIT/129DEV smokes.
Actualfull-fit1024targets old/reuse private69.349421GF/34.913923GF, shared
69.347356GF/34.911858GF;49.6565%/49.6580%saving includesAdam/norm/clip.
FinalBPCold/reuseprivate4.730271625/4.730271969, shared4.695410914/4.695410570,
differences<=3.44e-7. Alllearn/<1GB; tinyadmission, not10M qualityadvantage.
Newactualoptimizeddriverrecovery and independentdoubleproofpass; our116
strictcoordinatefails and117precisionfloors retainedbeside positive evidence.

Publicationretry021100Z completes26.484s/185pages withsource/layout/orphan
checksPASS. First020600Z failed becauseindependentAWSworkaudits include a
scalar total beside stagedicts; explicitschemafilter/sum-totalassert fixed
the renderer only, unusedretry kept. Allfailedqueues/logs preserved.

PRIORITYaws_language_winner_matrix_20261003T014100Z: exactoriginalprivate/
shared10Mteachercontinuations plusprivateoptimizedfullreplay10M active per
sharedHANDOFF; sharedreplay/factorizedcontrols queued. Teacherdiscardedwork
unknown<=4095extra targets each; original1MinitialDEV versusnew1025diagnostic
makesstartupwallunequal, final1MDEV matched. No pendingcellfilledwithonline
score. Next complementary cost reduction: cache causal factual token-boundary
state/RNG and replay only target suffixes, retainingfullconditionalwritecredit.
Deriveandcontract innew siblings before any proposal to change activeAWSruns.

## Systematic depth diagnosis complete — 3 October, 03:08 UTC

The user explicitly requested divide-and-conquer. Three agents audited completed
experiments/protocol, Adam/clipping, and architecture/transport. All numerical
work stayed serial through unique run_safe queues: one thread, 3GB virtual,
1.25GB RSS, 8GiB available floor, 180s timeout. Host remains healthy with over
11GiB available; no GPU. No unresolved Git conflicts; four old autostashes intact.

Read [Theory122](theory/122_deep_learning_systematic_diagnosis.md) for the full
synthesis, equations, experiment sources, primary papers and next decision order.
The 191-page report now includes every completed stage below, with same-unit
work columns, and retains all earlier positive/negative evidence. Publication
`local_deep_learning_diagnosis_report_retry_20261003T031200Z` completed in
24.582s; source/layout/orphan checks pass. First publication failed after the
shared batched kernel evolved; exact archived historical source bindings now
resolve on publication copies only. Original JSONs and earlier modules unchanged.

Clipping is not a demonstrated sufficient cure. D4 clip4 FIT32 worsens on both
seeds; completed D4 lr.006 and D2 clip4 also worsen FIT32. FIT32 is the first32
fitting gestures at DEV-selected weights, not whole-FIT/common-epoch loss.
D4 all-race seed8 improves to64.583%/1.020627, but seed7 fails, at roughly111x
factorized counted fitting cost. Preserve both findings. Small physical transport
decay does not exclude poor full recurrent Jacobian conditioning; DVS graphs
retain full episodes, unlike the detached language16-token horizon.

[Probe120](theory/120_deep_clipping_optimizer_probe.md) completed five checks,
100.552s/382576KiB, eight gradients/48 discarded Adam forks. Result:
`diagnostics/local_deep_clipping_optimizer_probe_current_inputs_20261003T025400Z.json`.
Fresh cap scaling nearly cancels in Adam (.019%/.190% step difference). Changing
only the current cap against stored moments changes direction and step1.36–1.47x
in the D4 probe and improves three FIT anchors; an entire cap4 training history
is different and fails the completed FIT32 comparison. Protocols differ between
D2/D4, so no causal cross-depth conclusion. D4 coarse transform byte hash differs
(original a0496fe9 versus current3fefa180); label this a controlled CURRENT-FIT
input probe at genuine online weights/moments, not exact historical input replay.
The original batched kernel8f93f5 is archived from155fbca^ and loaded explicitly.
D2 data metadata matches exactly. All four failed probe attempts/queues/logs
remain. Exact historical preprocessing is an artifact gap, not a proven bug.

IMPORTANT FOR THE OTHER HOST: legacy progressive growth resets non-tensor
`unit.gain`. D2->D4 has [.353553,.353553,.25,.25]; legacy D4->D6 resets every
old gain to.25, reducing the oldest residual amplitude29.29%. New
`dvs_grow_depth_lineage_benchmark.py` preserves the ACTUAL parent's vector and
records source-bound ancestry/gains in EVERY snapshot. [Theory121](theory/121_depth_growth_lineage_contract.md)
passes12 contracts in1.560s/261216KiB, including direct bitwise legacy nesting,
heterogeneous D6/D8 recovery and invalid-lineage rejection. The pending legacy
D6 arm in `curie_dvs_large_20261003T013500Z` is affected; do not describe it as
preserving parent gains. Original queue untouched. New
`local_corrected_progressive_depth6_deferred_20261003T030000Z` is reviewable but
NOT launched: its parent checkpoint is absent locally, and the owner must
coordinate the replacement and learning/accounting smoke first. Keep old evidence.

[Plasticity123](theory/123_depth_growth_plasticity_probe.md) completed6.142s/
370288KiB on fixed FIT0..15, no DEV/test arrays read. Four prespecified gate
biases -20/-8/-4/0, each one fresh normalized clip1 Adam.003 step;64 target
exposures, plus independent backward/instrumentation checks. ALL logits/EVERY
gradient bitwise nest plain computation; formulas/parent/source/RNG/kernel checks
pass. At-20, about99.7% of added candidate nonlinear messages round away in
float32 and ALL active gate/output gradients are below Adam epsilon after clip.
Gate steps receive about.007% of an unattenuated sign step. Input memory maps
remain live through key/timing paths (raw norms.046/.139, step~.096): added
layers are not entirely frozen. At-4, every measured candidate contribution is
visible and gate/output updates are near full sign steps. This is a gate-only
counterpart to owner §410, not its complete scratch protocol or a heldout gain.
Reuse that owner's queued initialization controls rather than duplicate fits.

[Cache118](theory/118_causal_prefix_reuse_credit_contract.md)/
[119](theory/119_cached_prefix_production_admission.md) pass36+6 contracts.
EVERY double gradient and pending Adam recovery agree. Same synthetic T4 total
fit arithmetic private full/winner/cache .069989/.036360/.023758GF; shared
.069473/.035844/.023242GF, all shadow/backward/norm/clip/Adam paid. T16 median
wall private1.654/1.221/2.579s, shared1.632/1.218/2.616s. Fewer shadow events
8192/4096/2176 but more kernel iterations16->136; grouped cache NOT promoted.
Keep negative wall beside positive operation savings. No T4-to-T16 projection.

Prioritized integrated quality model remains AWS native p16/L8/H2/pool2 private
corrected full replay10M, exact original private/shared teacher continuations,
and shared replay/factorized controls in
`aws_language_winner_matrix_20261003T014100Z`. Sources/protocols unchanged;
partial online scores are not completed quality. Complementary local priority:
useful live nonlinear depth, full-FIT/common-epoch and lesion/equal-pass shallow
continuation, actual functional update and exposure calibration. Retain all
computational clocks/races, sparse private memory, separate keys/values and
counterfactual/silence-aware principles. Outstanding gaps: language16-token
credit, all-pool key scoring, detached losing-content gradients, common-noise
covariance, replay fitting cost and CPU emulation. No isoquality/resource
supremacy established. Proposed content Rao-Blackwellization needs derivation,
contracts and charged losing-branch backward, not arbitrary stopgrad removal.

## Depth-sampling and actual-batch diagnosis complete — 3 October, 03:58 UTC

Three new numerical stages and two publications ran SERIAL under unique
run_safe queues, one thread, virtual3GB/RSS1.25GB, minavailable8GiB and
180/300s timeouts. No timeout/resource failure or Git conflict. Logged
watchdog snapshots show over9.5GiB available;8GiB floor never fired.
Largest diagnostic RSS866372KiB.
No local training job remains. Four old autostashes preserved; main only.

124 `local_depth_route_sampling_variance_20261003T033000Z` completes
five contract groups,292.571s/866372KiB. Exact initialized B2 conditional
k8 combined-gradient MSE D2/D4/D6 .017934/.084633/.305063; mean fresh
Adam cosine .702075/.561206/.567056 despite raw cosine .991/.961/.906.
All parameter vectors/full objectives/subset backward/actual Adam formulas
match. Scope is conditional initialization and fresh moments, not trained
cause or a complete risk gradient. k32 lowers variance but does not cure
step disagreement. Every result and64-draw subset index retained.

IMPORTANT ACCOUNTING CORRECTION beside immutable124 JSON/note: the bank
is evaluated AGAIN for original-driver equivalence. Main total4032 shadow
lanes/84672 events, tiny16/16. Original per-case lane/event fields count
only the first bank; original wording 'paid once' is not whole diagnostic
execution. Subset backward differentiates cachedQ, not independent sampled
driver. Input loader reads FIT0..15; only0/1 enter124. See125/126/report.

125 `local_sampled_credit_functional_forks_20261003T034000Z` completes
four groups,20.616s/576664KiB:54 actual discarded fresh clip1 Adam.003
forks,1824 prediction-target evaluations, no new gradients/shadows. FIRST
EIGHT saved draws each k8/k32, full/factorized controls, original+fresh
noise, all outcomes saved. Every fork improves FIT0/1 and worsens disjoint
FIT2..15. Gradient examples are classes0/1, anchors not IID/heldout.
D6 full-credit anchor NLL rises.46718/.46173, D4 .14452/.11285. Sampling
increases deep prediction disturbance, but k32 does not consistently
improve anchor loss. Do not nominate a sampling fit from this stage.

127 `local_projected_batch_credit_variance_20261003T035500Z` completes
four groups,56.096s/629624KiB. New artificial-cotangent mixed products
reproduce EVERY exact124 route projection for3 signs per depth, errors
<=1.38e-14. Actual B16 D4/D6 k8 MSE estimates.080239/.147045, descriptive
SE .003471/.006484; k32 .017051/.033145.32 signs, unbiased raw conditional
trace estimate, empirical relative SE~4.4%, worst-case relative RMS25%;
not exact trace, Adam variance, convergence or trained failure cause.
Full returns paid once:13440 lanes/282240 events,64 main+9 contract
pullbacks plus factual/full backprops. No optimizer step or quality fit.
Total diagnostic FLOPs/traffic/energy unknown, not zero. See127/128.

Both report stages pass all source/layout/orphan checks:196pages27.745s,
then197pages27.141s (`local_actual_batch_credit_report_20261003T040000Z`).
Final module actual_batch_credit_evidence.py chains frozen prior modules;
new general-purpose architectural framing from d9610ec retained.13 new
numerical contract groups total; all numerical/protocol/producer/report
sources now frozen. Mutable root generator and aggregate notes only.

Artifact policy:124 full vector bank is228MiB and stays LOCAL per.gitignore;
hash/reproducible source/queue/result JSON retained, no >100MiB Git blob.
Publisher checks its hash when present and explicitly reports absence on
other hosts.125 small predictions NPZ retained in Git. Exact archived
growth constructor6a6ab385 restored from302b82b^ for historical publication
bindings; original evidence/source modules unchanged. The active owner's
constructor now defaults-4 and recurses gains, so prior121/123 commands
are not historical reruns unless archived dependencies are restored.

Cross-host cautions in126: naive recursive gain inference is wrong for an
already legacy-grown D6 ancestor (actual .25 versus inferred .353553).
Current first D2->D4 and next D4->D6 owner paths are fine for that lineage;
do not generalize to arbitrary old ancestors. The deferred121-based
queue is still NOT launched and needs a new source-bound compatibility
contract plus its missing parent artifact.123 does not prove ALL prior
growth gain came from other live paths; lesion/continuation attribution
remains open. The250k AWS teacher audit rules against bias-20/second-
moment epsilon closure there, not every feature-learning issue.

Appended correction to owner's mutable §409 selects90M configuration by
DEV[90M:91M] bpc, keeping.01 preference and throughput gate; old written
TEST criterion preserved beside correction. Test[95M:96M] reporting-only.
No active training source/queue/result changed. Owner must bind DEV
selection before90M; precommitted test-based selection is still selection.

PRIORITY remains AWS streaming native p16/L8/H2/pool2 corrected private
full replay10M and matched original teacher/shared/factorized controls in
aws_language_winner_matrix_20261003T014100Z. Other-host live-gate/gain-
preserving DVS and reset-segment language comparisons are complementary;
await their completed quality before another policy. Main unresolved gaps:
useful nonlinear-depth attribution/full-FIT common-epoch continuation,
losing-content exposure,16-token language horizon, discovery/credit cost,
route stability and practical generalization. No architecture substitution
or superiority promise; retain clocks/races/private keys-values/sparse
writes/counterfactual/silence-aware direction and completed positives.

## Live counterfactual content admission — 3 October, 04:36 UTC

New training-only sibling sleeping_machines/conditional_branch_content_credit.py
keeps native coupled time/races/private addressed state/key-values/inference.
At ONE uniformly sampled legal race, pi-weighted LIVE branch gradients
REPLACE factual content gradients; unchanged detached-return choice credit
alone gets R scaling. Full prefixes remain live. No double-counting.
Theory129 passes8 groups2.644259s/343656KiB: every parameter against
explicit live branches/decomposition and independent factorized clock
Z=Lambda*T reference; early/late/unequal histories, factual winner/first
time, causal predictions, BLk1 choice, actual clip1 Adam/serialized next
update and operator coverage. Losing output gradient0 -> .01168205729,
pi .48176169428. Per-episode variance result, NOT shared-noise batch theorem.

Theory130 retry local_conditional_content_integrated_smoke_retry_20261003T042500Z
passes4 groups230.670498s/469892KiB. Same freshseed7 p4/D4/H2/pool2/clock.05,
float32 B4/clip1/Adam.003, FIT0..23/twopasses48presentations12updatesPERarm,
unusedFIT24..31 adjacent anchors (notIID or officialDEV/test). Exact same
initialization/order/sites/noise. Choice-only FIT2.674371->2.328692,
anchor2.604848->2.469267. Joint FIT->2.340674,anchor->2.470372: both fixed
learning gates pass, joint slightly worse for1.706935x counted fitting work.
Accuracy bothFIT8.33%->16.67%,anchor12.5%->0%; preserve this besideNLL.
Every update traced: wholeFIT .039905148/.068115492GF, per-presentation
.83135725/1.41907275MF, native inference .162127625MF/targetBOTH.
Available16private receivers,8selected/16scoredperevent,21events⇒168/336
per target,96fullshadowlanes/2016eventsFITeach; losing-backward paid.
No content policy promotion, larger fit, useful-depth or superiority claim.
Result data.fit_indices0..15 is inherited transform-verification metadata;
actual top-level fit_indices0..23/anchors24..31/update schedule govern.

Original042000Z attempt hit180s guard exit124 afterchoice-onlycompleted;
jointincomplete/noresult. Preservearchive/failed_runs/conditional_content_20261003T042000Z,
queue/log/runner. Failedtargetpresentations48..96, exactfailedworkunknown.
Newunique retry onlytimeout420s+runningexecutionaccounting; sameEVERYfit
setting/pass, safetyvirtual3GB/RSS1.25GB/8GiBavailable/onethread/hostlock.
No concurrent numerical training and host stayed healthy.

Report conditional_content_evidence.py/newpublish_conditional_content_report.py
chain frozenprior197pages, addlive-contenttheoryandcomplete quality/work
comparisons with savedp16references. local_conditional_content_report_20261003T043000Z
passes199pages27.167403s/273928KiB allsource/title/layout/orphan/diffchecks.
Sources/protocols/results/reportmodule/publisher nowfrozen, rootgenerator
andaggregate notes mutable. See131 for full scope/negative decisions.
Allfourautostashes and other-host changes retained, no Gitconflict present.

PRIORITY still AWS integratednativep16/L8/H2/pool2 correctedprivate full
replay10M/originalprivate/sharedteachers/sharedreplay/factorizedcontrols,
queue/aws_language_winner_matrix_20261003T014100Z/manifest.json. Otherhost
live-gate/gain-preserving DVS and reset-segment languagev2 batch/LR/cosine
runs remain separateprotocols. PreserveDEV selectioncorrection, await
completedquality; do notattributev2gainstoonechangedsettingor assume
update-starvationfrompartialplateaualone. Nextcontentdiagnostic is total
actualtrained-state batch/Adamvariance+cost, notanotherinitialnormclaim;
nextdepthattribution is frozen nonlinear message lesion/commonepoch
continuation retaining clocks/writes/routes. Neither launchedhere.
Mechanismgaps: usefulnonlineardepthattribution,16-tokenlanguagehorizon,
candidate/credit/optimizercost, sharednoise and generalization. No core
substitution or blanketfix: this sibling is numerically sound but fails
its tiny relative quality/work comparison, which is retained prominently.

## Trained content/route covariance — 3 October, 05:30 UTC

132local_trained_content_batch_variance_20261003T052200Z passes5groups,
65.156873s/430804KiB. Reproduces130choice-onlyEXACT12steps48presentations:
EVERYsavedfactual/conditionalstep loss, finalFIT/anchorpredictions and
parameter-groupmovement match. Saves small .state.pt actualweights+warm
12stepAdammoments and .vectors.npz allactualraw/clipped/updatevectors.
These are reconstructedtrainedp4D4, notnewbenchmark orseed. Double EVERY-
gradient contract, float32globalerror<=3.251e-7, actualwarmupdateerror
<=8.665e-7, byte-recoveredNEXTweights/moments,source/state/RNG andoperator
coveragepass. Allnew numerical/protocol/producerbytesfrozen.

16pairednoise/uniform-one-site draws onoriginalFIT0..3/B4, sharednoise
retained, SAMEchoiceA forC+A/Cbar+A;32discardedwarmclip1Adam.003updates.
Factual/jointrawtrace7.868724/7.852481 (ratio.997936), clipped.395247/.396003
(1.001913), actualAdam.000346709/.000345469 (.996424). Leave-one-pair-out
ranges descriptive, noguaranteedinterval. Contenttrace.320772/.300850
(-6.21%), choice7.571728 SAME (96.23%oftotalraw), signedcross-.023776/-.020097.
Exposure smoothsa smallterm; choicecreditalsoreachescontents/clocks/earlier
histories, notonlykeys. TWOactual isolated four-target steps cost
.003325429/.005835553GF, .83135725/1.45888825MFperpresentation, nativeinfer
.162245333MF/targetBOTH on12targets. Ratio1.754827; variance-times-work
raw/clipped/actualAdam1.751205/1.758184/1.748551: nocomputejustification.
Earlier48-targetwhole-fit1.706935 preserved asdifferentaccountingboundary.

All32forkpredictions savedFIT0..3/adjacentunusedFIT24..31 anchors (notIID,
DEV/test). MeanFITNLLdelta-.03524996/-.03524798, anchor-.00865158/-.00864847;
FIT16/16improve/anchors14/16BOTH, noselection. Reconstructedoriginalscore
exact;132baselineanchor2.38e-7differentonlynew12-targetbatchboundary.
Totalcampaignworkunknown/notzero: reconstructed48targets, ensemble64factual/
128fullshadowlanes2688events48componentpullbacks32discardedupdates,
2isolatedupdates+2recoveryforks/admission/evaluation. See132/133.

Reporttrained_content_variance_evidence.py/newpublisherchainspriorfrozen
199pages. local_trained_content_variance_report_20261003T052700Z completes
201pages35.990s, allsources/history/titles/bounds/orphans/diffchecks.
No newcontentlongrun. Nextdistinguish conditional SITE-sampling variance
from race-history variance using trained-state fullreturns and contracted
native parameterprojections;16draws132varies BOTH and cannotattribute
allnoise toonesite sampling. Keepdouble-promotion/traceuncertainty/costscope.
AWSoriginalprivate/sharedreplay10Mmatrixstillqualitypriority. Otherhost
reset-segmentv1completed2.899testbpc;v2calibration/v3capacityarmsuse distinct
protocols. Selectedwritecountisnotcandidate-scoring/value/optimizercost.

## Route-site coverage and measured budget — 3 October, 05:55 UTC

134local_trained_route_site_noise_20261003T054100Z completes5groups,
59.326s/390920KiB.132source-boundtrainedp4D4/B4/originalFIT0..3, represented
float32weights promotedDOUBLE, FIRSTFOUR132noisehistories. All168races
perrow fullreturns;32Rademacherparameter signs perhistory,128projections.
Trainedtiny EVERY-route VJP/projected errors<=4.58e-16; fullsum/originalBL
andexplicitk2mean/covarianceagree. All128mainprojectedsums matchfull
parametergradients. Float32factualCEexactly132, all2688factualwinnersagree
withDOUBLE, maxlogiterror2.85e-7. Forcedbranchprecisionequivalencenotproved.
Originalsources/state/RNG/helperseed/replaycounterretained; sourcefrozen.

Law-of-total-covariance estimates mean SITEtrace k1/8/32/168 =
7.046586/.843903/.179329/0; fullmeanbetweenHISTORYtrace .341525 SAME.
Total7.388111/1.185427/.520854/.341525; k1site share95.38%, k8totalratio
.160451. FOURhistories and32signs descriptive, notalltrainingcausality,
actualAdamvariance orguaranteedinterval; worstrelativeRMS25% tracebound.
Main5376fullshadowlanes112896events, tiny96lanes160events becausebank
ANDoriginalBLequivalence bothpaid;128+3projectedpullbacks24tinyrouteVJPs/
fullbackwards,16mainfactualtargets+32precisiontargets, nooptimizer.
SmallprojectionsNPZ retained; totalFLOPs/traffic/energyunknown/notzero.
See134/135;132sixteen-drawFLOAT32variance isdifferentprotocol,notreplaced.

136local_trained_route_coverage_work_20261003T054700Z passes6groups,
60.076s/362876KiB. FIVEactual EXISTINGBLwindows same132weights+12step
Adammoments, FIRST134noise, originalsampler: no-choice/k1/8/32/all168.

**3 October06:52 timing annotation:**134/136 numerical outcomes and counted
operations are retained. Physical-host exclusive wall timing is unestablished
because the container-local lock cannot observe the reported curie training.
Do not use59.326s/60.076s as isolated speed comparisons; see139. No reported
FLOP/variance heuristic claims wall-time or measured-energy advantage.

EVERYtraced/untraced nextweight ANDmoment EXACT, serializedfullrecovery,
causalidenticalfactualCE/RNG/sources/coveragepass. Elevenupdatesdiscarded.
STEPGF .002028565/.003327802/.012406626/.043533818/.219918762;
per-presentationMF .50714125/.8319505/3.1016565/10.8834545/54.9796905;
same12-targetnativeinfer .162245333MF/target. Capacity16private receivers,
168selected/336scoredpertarget unchanged. No-choice isdifferentteacher,
NOTk0onthisvariancecurve. Countcandidate/losing/normalization/optimizerwork.

k1/8/32/full rawVARratio1/.160451/.070499/.046226 paired withactualWORK
ratio1/3.728174/13.081853/66.085291 yieldsheuristic1/.598188/.922257/3.054878.
k8bestMEASUREDfixed-state variance-work point(~40%better), NOTquality,
actualAdam/convergence orsupremacy. AlllocalFIT/anchorpredictionssaved,
fullforkbestanchorwhileheuristicfavors8: noprediction-basednomination.
Traced1672shadowlanes35112events, verification3016lanes63336events,
extraevaluation72predictiontargets/inferencetraces/admission. Totalcampaign
FLOPs/traffic/energyunknown/notzero. No longfit oradaptiveschedulerlaunched.

137derives V(k)=a+b/k fromfinitepopulation+historyfloor andaffine C(k),
fixed-state averaging optimumsqrt(Cfixed*b/(c*a)) ifa>0, clipped1..R;
a<=0 favorsfullunderheuristic. Hereestimatedoptimum6.0895, NOTtestedk6.
Heterogeneousallocationk_j∝sqrt(b_j/c_j) withbounds; mean-preservinguniform
coveragechosenBEFOREsampling; stoppingbasedonsamesubsetreturnscanbias
R/kcredit. Independentpilot/properinclusionprobabilitiesneededandpaid.
No claimthecalculus proves actuallearningadvantage. NextACTUALwarm-Adam
coverageensemble beforepolicy, notanothercostlycontentfit orblindksweep.

Reporttrained_route_coverage_evidence.py/newpublisherchainsfrozen201pages.
055000Zfirstpublicationfailsorphanfooterpage204, previousreportrestored;
exactfailedproducer/publisherarchived, queue/log/runner retained, failed
PDFlocal.git/report-validation. Textonlyshortened; newunique055400Zretry
passes203pages27.498s/274068KiB, allsource/history/title/layout/orphan/diff
checks. Numericalresultsunchanged, producer/publishernowfrozen. Rootreport
andaggregatehandoff/indexmutable. Otherhost§412compiledlayer2.7x andv2c
queuespreserved; separatefromeageroriginalAWSprivate/sharedreplaymatrix.

PRIORITY still integratedAWSnativep16/L8/H2/pool2 correctedprivatefull
replay10M/exactoriginalprivate/sharedteachers/sharedreplay/factorizedcontrols
queue/aws_language_winner_matrix_20261003T014100Z/manifest.json. Preserve
otherhost live-gate/gain-preservingDVS andcalibrated/compiledresetlanguage
protocols. Maingapsusefulnonlineardepthattribution, horizon, totaldiscovery/
credit/optimizercost andgeneralization. No corearchitecturedeparture:
clocks/timecomputation/hardlearnedraces/sparseprivatepersistentvalues/
separatekeys-value/counterfactual/silence-aware direction retained.
## Write-feedback gain and current decisions — 3 October, 14:30 UTC

Read shared HANDOFF and latest results through3c71068. Producer integrated
chain remains reserved; no Torch/model job, profiling, training or new queue
was launched here. Actual p64/U2 and p32/U4 checkpoints are still unavailable
in this container; native143/144 and warm-Adam138 remain pending, not passed.

New145 derives the partial write-surrogate H=alpha*B*Jpi*S and its adjoint,
block-reduced U-by-U singular geometry/Frobenius/bounds, exact U2 norm and
bounded-write/nonnormal amplification witnesses. A physical selected decay.5
with unit writes can coexist with local surrogate gain1.353553; repeating the
fixed witness16times amplifies126.94, NOT an observed native trajectory.
Winner averaging also amplifies in the fixed-coefficient witness. A nilpotent
addition has unit eigenvalues but transient growth. These identify what to
measure; they do not attribute the failed fits or prescribe a new learner.

`write_credit_feedback_geometry.py` and its stdlib check pass adjoint,
fixed-anchor finite differences(maxerror1.81855e-12), dense-block Frobenius,
U2 exact norm, U1/alpha0/no-score-read/gauge and amplification contracts.
No NumPy/Torch imports. Partial fixed-input/clock/winner geometry excludes
full recurrent/query/message/time paths and rounding-certified enclosures.
Normalizing loss does not change H; end-of-backward clipping cannot fix an
intermediate nonfinite adjoint. Actual fidelity and adjoint/update effects
are separate admission requirements before another write-credit fit.

Corrected398 beside its original evidence: winner-only direct proposal-map
learning does NOT mean all parameter groups learn only on wins. Unsaturated
losing key-read maps receive score credit gamma_i*q*m_i^T/sqrt(P). All-event
support does not guarantee equal information, and the c*parameter-count/
effective-examples model is an assumption rather than a generalization
theorem for endogenous recurrent routing. Tied maps remain a useful test.

Latest completed p64/D4/U4/linear T2562.179497 improves U2 by.00381768 at
1.6453156x fitting work(44.0738 vs26.7875TF). It remains behind saved one-pass
LSTM2.1706. Native language mechanism gains are real and scoped; broader
advantage needs replication, complete resources and stronger scale controls.
The shared PDF already incorporates the new completed row and inference
traces. This continuation changes no report/benchmark/source evidence.

Priority: successful integrated value-credit p64/D4/U2 reference; finish
producer width arm, current v6 tied-pool and independent seeds, then horizon
arms. Queue `curie_language_batched_v6_20261003T111500Z.txt` stays untouched.
Run143 cached actual winners/state admission and144 actual write factorial
only after physical reservation/producer checkpoints. A gain follow-up must
capture actual d/k/unclamped-score/decay operands, contract isolated native
surrogate VJPs, then compare adjoint histories/finite warm-Adam forks; no new
gain queue admitted. Assigned AWS90M priority and original full-replay/control
fits remain owner-managed; no reallocation is executed by this container.
Core temporal races/private persistent state/separate keys-values/deep credit/
silence-aware direction retained; no architectural substitution.

14:35 concurrent owner update8141aff/5c9cba7: AWS reallocation is now EXECUTED
by owner, source-exact checkpoints archived and two private streaming fits
recovered. Fifteen compiled/driver contracts passed and491520-presentation
pool4 admission pilot completed. First assigned90M/p32/D4/U4/linear fit started
14:29UTC in bounded slot1, with recovered private replay/teacher in slots2/3.
Shared teacher deferred with recovery intact. No completed90MDEV/test exists;
pilot throughput is not a quality result. New owner manifest is
queue/aws_priority_language_allocation_20261003T143000Z. Preserve its sources,
locks/RSS reservations and serialized publishers.145 decision paragraph now
reflects that transition; earlier shared scheduling history remains intact.
