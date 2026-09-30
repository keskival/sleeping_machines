# Shared Sleeping Machines model

**One architecture and implementation; independently trained weights per task.**
No joint training is used. This is the first working synthesis of the deep event
carrier, conditional evidence, learned relative retrieval and periodic phase mechanisms. It is
not yet a union of every historical prototype's primitives.

## Implementation

| File | Responsibility |
|---|---|
| `sleeping_machines/shared_event.py` | Configurable bounded carrier depth; hard winning vector/delay; local training-only loser score credit; optional causal context receivers; evidence, periodic and weighted event readouts |
| `sleeping_machines/phase_memory.py` | Learned affine circle state, hard class clocks and local timing credit |
| `sleeping_machines/event_memory.py` | Reference and linear-work segmented numerator/mass scan |
| `sleeping_machines/event_query.py` | Explicit observed-prefix query, chronology/cutoff validation, separate examples |
| `sleeping_machines/evidence_memory.py` | Visited-context count/exposure memory and hard relative pointer routes |
| `sleeping_machines/objectives.py` | Categorical and exact piecewise-constant marked hazard likelihoods |
| `sleeping_machines/readout_calibration.py` | Fit-only conditioning; project unsupported static count dependence |
| `sleeping_machines/event_state.py` | Generic signed temporal-state blocks, nonlinear vector gates, winning clocks; full-source causal coalescing and a six-block candidate encoder |
| `sleeping_machines/affine_packets.py` | Simultaneous affine endpoint and weighted-state-query summaries, including packet/receiver resets |
| `sleeping_machines/readout_absorption.py` | Fitting-only affine branch absorption, categorical certificates and folded feature conditioning |
| `sleeping_machines/nuisance_readout.py` | Paired-view covariance and robust affine head fitting; no extra deployment transform |
| `sleeping_machines/depth_growth.py` | Identity event-block growth retaining old outputs/teachers with live new output-map credit |
| `sleeping_machines/head_conditioned_depth.py` | Existing class-head bound for new residual gains and their optimizer units |
| `sleeping_machines/observer_conditioned_depth.py` | Pre-normalized state/output growth, invertible observer units retaining hidden null directions, and physical deployment-map folding |
| `experiments/e120_shared_tasks.py` | Task encodings, disjoint evidence/neural fitting, target construction |
| `experiments/e120_dvs_adapter.py` | Bounded-packet AEDAT input, causal count coalescing |
| `experiments/e120_shared_bench.py` | The same optimizer/backbone training path for all new tasks |

E118's `RaceNet` is a compatibility subclass. E119 imports the shared scan.
Old checkpoint keys and default speech computation are preserved. The extraction
contract compares against committed revision `de6a45a`, not a second alias of
the new class. Its source is preserved in `reference/e118_pre_shared.py` with
an asserted SHA-256 so rebases and fresh clones do not lose the reference.
The check gives exact logits, parameter gradients and winners on the audited
batch, and 151/256 unchanged development answers on the existing speech model.

### Richer signed temporal states (E142–E145)

The new generic encoder uses real coordinate pairs for stable complex modes,
learned source vectors, input/output maps, nonlinear gates, LayerNorm and
residual vectors. It evaluates only supplied packets and their actual delayed
arrival order. Local projections are dense; event-pair attention and empty
time ticks are absent. Clock candidate credit is a declared local surrogate,
not an exact changed-order suffix replay. Its completed-query head observes
vectors and counts, so the last output clock receives no label teacher.

Raw source lookup occurs before causal coalescing. Transporting raw modal
drives to closure preserves first-layer affine endpoints and their smooth
teachers exactly, but nonlinear outputs are evaluated only at closures.
`affine_packets.py` additionally retains weighted raw-state queries; this
pooling extension is not in E143. Its streaming state/query storage is O(m)
per active receiver, while the current autograd scan materializes O(Nm).

E143 composes an immutable trained D8 width-32 parent with a **parallel** D6
width-128 encoder (64 state pairs). A zero correction head initially preserves
all parent predictions; the hidden encoder is initialized nonzero, allowing
its label teacher after the first head update. There are 53,296 inherited
parameters plus 395,814 new ones. This is not a fourteen-layer sequential model,
a from-scratch result, or a new primitive already rerun on every benchmark.
It is a common module for continuous/vector event representations, currently
evaluated on the SHD task class. Completed curves and reset diagnostics belong
in FINDINGS.md; official-test parity and early-confidence answers remain separate
acceptance gates.

Completed E143 improves the private sample from **370/512 to 408/512**, and
657 disjoint utterances from **463 to 510 correct**. Resetting learned hidden
clocks loses 13 decisions, confirming their use in this fitted computation;
resetting the trained stack loses 50. These probes retain the trained readout
and are not matched retrained controls. The new encoder has not yet replaced
the original carrier across the other task configurations.

### Semantics and resource boundaries

E152 deploys the temporal encoder alone: six blocks, 395,814 parameters and
406/512 private development answers, versus the combined model's 408/512.
Its affine head is fitted on clean/augmented fitting-only queries and combined
teacher predictions. Whitening folds into the existing head. The inherited
E143 training and teacher work remain part of its training lineage; this is
not a fresh standalone result or a complete synthesis with the language expert.

The same encoder grows to twelve sequential blocks with exact initial class
outputs and old teachers, nonzero new output teachers, and 36 ms additional
initial winning-clock latency. E159 verifies actual restored scheduler rates;
the corrected D12 pass reaches 400/512 with 696,888 parameters. It trains above
chance with useful fitting progress but does not beat the six-block starting
model. New depth costs more vector/state work. All local maps remain dense;
the packet schedule still has no empty ticks or dense event-pair attention.

The matched D6 yields identical development decisions; on the reused audit,
both get 513/657. Removing the appended D12 blocks changes zero answers.
E161–E163 implements a distinct growth variant: normalized state features
precede the zero output map, and its optimizer units condition visible class
directions while retaining unit scale in the observer nullspace. The full-rank
coordinate transform folds into ordinary output weights at deployment.
Its exact growth/local teacher/folding contracts and fitting-step replay pass.
The completed full D12 reaches 401/512 and 509/657; its trained prefix reaches
408/512 and 518/657. Added maps now change decisions, with negative net held
utility. E165 exports the development-selected six-block prefix with ordinary
weights and retained old optimizer/RNG state; E166 reproduces its development
score exactly. It matches the combined accuracy with lower NLL, fewer deployed
parameters and roughly half the observed CPU forward time, while inheriting
the full deeper training cost. It is not official-test or energy supremacy.

- Encoders supply only observations available at a query cutoff. The label,
  next character and next gap are separate targets.
- Learned arrival delays can reorder carriers; the shared core sorts arrivals
  within each receiver. All current benchmarks execute on CPU, one thread.
- Timing performs computation: delay changes alter the causal memory seen by
  later messages and the winning route. Dormant units alone do not explain the
  model's computational mechanism.
- Only the winning hidden vector/delay is emitted. Losing values are detached
  training counterfactuals, never averaged into the emitted hidden message.
- Evidence is fused at the output score. This geometric mixture is a separate
  primitive from the hidden hard race.
- Every supplied carrier continues through the stack. It is sparse in observed
  events and avoids a hidden time grid, but does not yet learn to delete carriers.
- Local vector maps and outcome readouts are small dense operations. The scan
  has linear combines, sorting remains O(E log E), and pointer search enumerates
  all present source/offset candidates. Count all of them.
- A language/market query replays its entire context. Adjacent queries do not
  yet share persistent scheduler state. No incremental throughput claim is made.
- The generic low-level `forward` accepts already grouped arrays. Public callers
  should use `pack_queries`, which enforces nonempty chronological prefixes.
- Query predictions are terminal readouts, not yet confidence-triggered early
  answers. Simulated message delays and CPU execution latency are different.

## Completed coverage (E120)

The initial E120 cores have depth 8, width 32, seed 6, eight epochs. Each has
its own initialization, optimizer, input alphabet, fitted weights and checkpoint.
The following development screens cover the eight principal data/task families;
they do not rerun every historical experiment. E123 adds bounded Transformer
references for five screens; their protocol and cost boundaries are below.

| Task | Protocol | Completed result |
|---|---|---|
| SHD | Frozen E119 checkpoint; 1,024 fit, 256 held-out training speakers | 151/256 (59.0%) preserved; original best checkpoint 63.7% |
| text8 | 32,768 evidence-fit chars, then 2,048 neural queries; 256 validation-region queries | 3.022 → 2.915 dev bpc; fixed evidence 3.024 bpc |
| BTC event world model | 200k-raw-trade prefixes on three disjoint days; 512 neural fit / 256 dev; exact next-type/time likelihood | Combined 3.823 dev nats/event vs fixed evidence 3.670; does not improve held-out likelihood |
| Associative recall | 4,000 memory-fit examples, 512 separate neural fit / 256 dev | Corrected core 256/256 at both standard and 4× context |
| Temporal composition | E28 generator, 24 channels, 6 motifs, 6 classes + none; 1,024 / 256 | 249/256 (97.3%) |
| MNIST | First 1,024 train images fit, next 256 dev; fixed 2×2 pooling, latency encoding | 194/256 (75.8%) |
| DVS Gesture | First 1 second; 4×4 spatial cells + polarity, 50 ms packets; 88 / 44, disjoint users | 26/44 (59.1%); not a full-gesture benchmark |
| Modular arithmetic | 30% of mod-17 triples; 1,473 / 256; no provided rhythm | 176/1473 fit, 6/256 dev; unsuccessful short screen |

No official real-data test split is read by these screens. Standard/long recall
queries are independently generated. Counts/exposure and pointer memories use
disjoint memory-fitting data before neural fitting. Development labels update
neither component. Evidence arrays are prepared once; evaluation wall time does
not include that preparation. End-to-end run time does include memory fitting
and loading. Numeric memory bytes exclude Python mapping overhead.

### The count-feature failure and repair

Initial recall: 100% at training length but 23/256 at 4× length, despite a
perfect pointer. Constant training count was standardized with a 1e-4 floor;
the longer context became a 1299.28-unit feature and produced a 78.97-point
untrained head contribution. Projecting only that input coordinate restores
256/256 with all weights frozen. A new run with the fitting-only support rule
also preserves 100% on both lengths. Variable-count conditioning is unchanged.

The failed result `recall_d8_20260929.json` is retained. The corrected run is
`recall_d8_supported_20260929.json`. The initial six queues predate the correction
and preserve their executed commands. To reproduce their former calibration
with current code, add `--legacy-count-calibration` and use a **new tag/queue**.
The default for new runs is the supported-count rule.

### Readout diagnostics

Removing the additive core head after training improves development NLL from
2.020 to 1.942 on text and 3.823 to 3.549 on the market. This retains the learned
deep-feature gate on the evidence. The `memory` diagnostic therefore is not an
independent memory-only model. `fixed_evidence` uses equal evidence weights and
no neural features. These are frozen deletions, not retrained controls.

## Shallow arithmetic follow-up (E121)

Depth is configurable by task. E121 compares two-layer, width-32 cores over
200 epochs with identical mod-17 fitting IDs, neural schedule and sample order.
The phase memory adds 69 local timing parameters to the 14,060 neural weights.
The plain core ends at 70/3,440 unseen tuples. Phase memory with a fixed bounded
neural correction ends at 3,391/3,440; its phase memory alone is 3,440/3,440.
The corrected `phase_margin_guard=True` configuration finishes at **3,440/3,440**
with all winners certified. Both fit and old development slice are perfect.

The guarded readout bounds each correction by one quarter of the clock lead.
It can calibrate class confidence but cannot overturn the phase winner. Phase
teaching must fix any wrong phase answer. Hidden carrier winners still execute
and learn; the arithmetic capability is supplied by the periodic branch. It
is not currently an internal hidden routing primitive, and cannot yet be
configured alongside conditional evidence in the same readout. Phase teaching
is CPU-only and expects unit-count occurrence events. Phase-clock units are
abstract phase units, not calibrated physical latency seconds.

Use `experiments/e121_shared_arithmetic.py --phase 1 --margin-guard 1 --depth 2`
through a new one-job safe queue. The exact completed commands and source hashes
are saved under `queue/e121*` and `results/e121`. The fixed-bound and plain runs
remain available. Three-input arithmetic does not require an eight-layer generic
stack; deeper speech continues to use eight. Full theory and scope: §181.

## Computational ownership and event readout (E124–E125)

`phase_only=True, depth=0` executes the same fitted periodic primitive directly
through the common model interface. It allocates no unused neural embedding,
carrier or head. The algebraic state has 69 learned scalars and emits a
two-coordinate unit-circle payload. The supplied period is 17; phase units
are abstract, not calibrated seconds. This configuration is suitable for the
three-symbol arithmetic task, not a substitute for a deep speech representation.

A complete fitting pass with zero mistaken-example updates certifies a fixed
point of this deterministic local teacher. E124 stops at that condition and
checks exact equality with E121's final coupled phase state: 47 passes,
69,231 fitting presentations, 29,003 mistaken-example updates, and all 3,440
unseen triples correct in 3.29 s. Its per-query
work ledger separately charges symbol reduction and class-clock selection.
The carrier-plus-phase configuration remains available and its additional work
is included in the comparison. Theory §§183–184 and §186 give the certificate,
ownership rule and stopping proof.

`readout="weighted"` adds a scalar content key to the final winning payloads.
It accumulates a count-times-bounded-gain numerator and mass with linear event
work. The key initializes to zero, exactly preserving the old mean. The label's
local score credit is its event weight times the teacher's projection onto
that payload minus the pooled payload. At initialization, key credit equals
event covariance times the teacher. E125 verifies this equation numerically,
with nonzero key credit and exact initial checkpoint predictions. It does not
turn losing hidden messages into winners or add event-pair dense attention.
Theory §185 distinguishes this terminal statistic from a hard hidden race.

## Deep speech continuation (E122)

The matched 2,048-example, four-epoch continuations share the E119 checkpoint,
Adam state, fitting order and evaluation IDs. Training-only time scaling and
small channel shifts improve pooled clean development accuracy from **350/512
(68.4%) to 369/512 (72.1%)**. Most of the paired gain is on speaker 3; speaker 6
changes little. Expanding to 4,096 fitting examples for two further passes
reaches **370/512 (72.3%)**. These are train-file held-out speakers, not official
test results or a published-score parity claim.

An additional mean-pooling continuation regresses to 353/512 (68.9%). Its
matched learned-pooling arm tests a specific terminal-statistic hypothesis;
it is not a new seed search. Both arms preserve the frozen fitting-only
calibration and use the same initial state, data, augmentation and update budget.
The retained results distinguish fitting progress from held-out improvement.
The learned pool finishes at 356/512 (69.5%), just three net answers above
the continuing mean and below the common starting checkpoint. Its key receives
nonzero credit but this intervention does not yet improve recognition.

## Safe reproduction

### Separate key/value learning phase (E131)

`freeze_keys_from_values()` snapshots a local key stream from the initialized
value carrier. Initialize added context columns to zero before this operation.
The keys run actual causal local hard races on each observed query. Values
share their winning choices and clocks, with independently learned global
content maps. A value update cannot change the key schedule. The method keeps
winner computation active; it does not use per-example stored winner labels.

Checkpoints use `separate_keys=True` in their model configuration to recreate
the modules before loading their states. The frozen key stream adds 52,616
parameters at width 32/depth eight; context value maps add 6,336 trainable
parameters. Key maps/scans are included in aggregate statistics and also
reported separately. Keys stay frozen in this learning phase. Joint key-policy
learning needs counterfactual utility for paired key/value/delay alternatives.
The original default shared-payload computation remains the compatibility path.

Inspect existing workloads and available memory first. Use a new output tag
and exactly one line per queue. For example:

```sh
# Create a uniquely named one-job queue containing:
# local_shared_recall_new experiments/e120_shared_bench.py --task recall --tag local_shared_recall_new --fit 512 --dev 256 --depth 8 --epochs 8 --bs 16
MEM_CAP_KB=3600000 MEM_CAP_RSS_KB=2600000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 \
  bash experiments/queue/run_safe.sh experiments/queue/local_shared_recall_new.txt
```

Never bypass the host lock. Completed tags refuse overwrite; successful queue
jobs are skipped on restart. E120 peak training RSS was below 0.5 GiB on this
host. Checkpoints contain configuration, model/optimizer/scheduler/RNG state,
frozen evidence memories and source hashes. The current CLI does not implement
mid-run resume. It persists result curves after every epoch and writes the
checkpoint at completion. Checkpoints and logs are excluded from the commit
helper; result JSON and source are included.

Validation queues:

- `e120_contracts_release_20260929.txt`: extraction, scan, gradients, winner-only
  computation, prefix cutoff, query isolation, exact pointer update.
- `e120_readout_audit_v2_20260929.txt`: frozen longer-context intervention and
  count-calibration contracts.
- `report_shared_final_20260929.txt`: readable PDF and Markdown build.

## AWS handoff and outstanding coverage

The AWS sibling retains its existing non-SHD queues. This local work did not
start those workloads or change their plan. Use a separate `aws/shared-*`
branch, host-specific tags and new one-job queues for shared-model work. The
handoff is a repository file, not a message sent to the sibling.

The report publisher now resolves conflicts automatically only for generated
report artifacts, preserving main until regeneration; editorial/builder conflicts
require resolution. A publisher process started before this code change must be
restarted on its host to adopt the revised policy. The named PDF edition
`report/sleeping_machines_shared_20260929.pdf` preserves this report independently
of the background status-PDF refresh.

1. Import the shared-core source and contracts; run the contracts on a host
   with the referenced SHD checkpoint, or obtain that untracked checkpoint
   explicitly. Do not silently substitute a randomly initialized model.
2. Preserve the recall 4×-context and exact-scan checks before scaling.
3. Compare **retrained** fixed-evidence, gated-evidence and additive-correction
   arms with identical examples and selection protocols. Existing frozen
   ablations justify this next comparison; they do not replace it.
4. Extend bounded text/market adapters to the existing full E79/E52 splits and
   larger evidence banks before making full-benchmark claims. The current
   market adapter deliberately caps raw input at 200k trades/day and rejects
   requests with insufficient target events.
5. Extend the integrated periodic state to hidden routes and other periods;
   port native hold/veto composition and preserve arithmetic/depth protocols. Add the continual-learning/retention suite;
   it is not covered by the present separate stationary fits.
6. Add a causal persistent scheduler and measure replay savings. GPU training
   requires explicit device support and profiling; this harness currently
   declares CPU execution and does not claim a CUDA implementation.

Do not use this exploratory suite as a new best-model selection on official
test sets. Freeze choices with validation, then measure held-out quality and
total training/inference resources. Theory §§176–186 explains the causal query,
natural-score credit, calibration support, periodic composition and event readout.

## Generic learned language screen (E133)

The shared backbone now has a bounded evidence-free language comparison:
`evidence_count=0`, no pointer/copy/phase readout, independently fitted one- and
eight-layer configurations. Width 32, 8,192 training targets, 32-character
contexts, four passes and 1,024 validation-region targets yield 3.464 and 3.395
bpc respectively. Every value, route and memory-time layer parameter set updates.
Count/time/last-character-preserving order perturbations raise the deeper loss
to 3.805. This supports learned context sensitivity, not competitive large-scale
language representation. The deeper configuration has more parameters and work.
`results/e133/generic_language_audit_20260929.json` verifies data/protocol/source
matching and records partial forward/backward operation estimates. Physical
memory traffic and energy remain unmeasured. Prefix replay is unchanged.
The generic scaling criteria are in
[LANGUAGE_SCALING_PROTOCOL.md](LANGUAGE_SCALING_PROTOCOL.md).

## Full value phases and content retrieval candidate

E134 teaches all eight value layers, embeddings, memory time constants and
readout under either fixed functional keys or coupled inputs. Both matched
58,048-parameter phases reach 349/512 held-speaker accuracy; the parent remains
370/512. Stable routes and nonzero layer gradients are supported mechanisms,
not a sufficient recipe for transfer.

The E135 research adapter replaces value-stream temporal means with a positive
bounded content kernel while keeping the immutable key program intact. Query
zero initialization reproduces the parent exactly. A mean plus four signed
key/value moments implements retrieval with 165 state scalars per receiver/
time bank, versus 33 for the plain memory. No event-pair matrix is used in the
candidate implementation. Only winning payloads propagate. New query/key
parameters and expanded scan/projection costs are explicitly recorded; the
adapter is not yet a default or an energy advantage. See theory §205 and
`experiments/e135_content_memory.py`.


## Persistent language stream and causal token adapters

`stream_language.py` reuses the signed `EventStateBlock` primitive with retained
modal state, actual delayed-message scheduling and chunk-preserving credit
truncation. Each source token is consumed once. E175 verifies causality and
all-layer teachers; E176 reaches 3.351 validation bpc in an eight-layer,
8,192-target/four-pass development screen. It has 28,403 parameters and uses
character intervals as text time. It is a new generic path rather than the
bounded-prefix classifier or native count/copy mixture.

`prefix_tokenizer.py` supplies a train-only complete prefix dictionary and a
raw-character-timed token adapter. Learned leaf probabilities induce exact
next-character probabilities by subtree marginalization; no ambiguous decoding
or dropped final partial phrase is needed. E177 checks normalization, chunking,
causality, raw/token probability identity and exact character-control equivalence.
Vocabulary capacity, tokenization work and buffering remain part of its cost.
Theory §§269–274 defines the filtration, scoring and local teachers.
