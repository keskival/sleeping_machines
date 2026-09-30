# Sleeping Machines

**Deep learning that computes with time.**

Sleeping Machines explores models in which an event carries content and an
arrival time. Nodes mix incoming vectors with persistent memory, gate their
updates, transform messages and compete
through learned delays. The winning message determines what happens next;
unrealized alternatives can teach the network to make better choices.

**Delays do computation.** Changing a delay changes arrival order, which memories
interact and which route wins. Sleeping units and sparse activity are part of the
resource model; temporal computation is the central architectural idea.

The goal is a broadly useful model family with competitive predictive quality and
substantially less physical work in both training and inference. This repository
contains the implementations, mathematical analysis, reproducible experiments and
completed results used to pursue that goal.

The current integrated language models combine these ideas: learned queries and
state-dependent keys set temporal races, selected receivers mix content with
memory, and counterfactual credit teaches hard choices. The new eight-block
candidate has independent parallel Q/K/V heads and timestamped channels whose
state evolves while other messages arrive. Each head retrieves a historical
winning value; the next layer can learn to mix the separate channels.
See the [integrated language guide](experiments/INTEGRATED_LANGUAGE.md).

Our hypothesis is **more capability per unit of active work**. Temporal races
can approximate attention while avoiding selected arithmetic and value
aggregation; evolving state and useful dormant modules may reduce the width,
depth or fitting needed for comparable quality. Full-bank key/query scoring
still costs work. One winner is not a deterministic attention average, and
compression and frontier language superiority remain experimental questions.
The report puts the [full-bank work scaling](report/figures/full_bank_temporal_scaling.png)
and [logical access comparison](report/figures/full_bank_temporal_traffic.png)
near the front. Both models retain the same context/depth order in this matched
attention substitution; further gains depend on learned temporal computation
and economical recruitment of dormant capacity. Irregular event streams are a
natural input interface, with event-camera benchmarks still to come.

The same event interface can carry language tokens or asynchronous sensor
observations. Reusable temporal primitives and training machinery are a practical
advantage; input adapters and objectives remain task-specific. Existing cross-task
results use separately trained models, so shared-weight multimodal learning is a
further milestone. The [event-stream advantage protocol](experiments/EVENT_STREAM_ADVANTAGE_PROTOCOL.md)
tests real-stream quality, silence, temporal information and dormant-capacity
scaling against controls that also accept timestamps. Instruction-conditioned
robotic control is a further application: language goals, persistent world state,
irregular observations and timed actions. Joint expressive reasoning and reliable
control need their own benchmarks; they do not follow from the current task wins.
The stronger integration target is instruction-conditioned event routing and
shared persistent state: language guides perception, events ground language, and
both inform actions. This joint model is a next milestone, not an existing result.

Distributed event-triggered hardware is another target: local state and
communication can reduce global coordination and avoid periodic work during
silence. FPGA prototypes can validate sparse execution; asynchronous ASICs can
test physical delay/race computation. Clock removal alone saves its measured
energy fraction, with control overhead deducted. The larger opportunity combines
that with sparse activity and reduced traffic. See the [hardware hypothesis and
measurement plan](experiments/HARDWARE_VALUE_PROPOSITION.md); hardware joules
are not yet measured for Sleeping Machines. Replicating Transformer-quality
training and inference at lower whole-system energy on clockless hardware would
already be a useful milestone; richer temporal state, dormant capacity and
cross-modal integration provide further directions.

The report's opening pages set out four routes to useful advantage: a
Transformer-relevant clockless workload, smaller models through richer temporal
computation, economical growth of dormant capacity, and integrated language/event
reasoning and control. Each has an explicit next test. Illustrative compression
and energy scenarios show the possible payoff without presenting assumptions as
completed results.

### The differentiators at a glance

- **Time performs computation:** learned delays, races and phase transformations.
- **Hard routes can learn:** winners execute; unrealized alternatives receive credit.
- **Deep, persistent representations:** vector messages, local memory and reusable temporal primitives.
- **Capacity beyond activity:** the scaling goal is useful dormant capacity with selective work and competitive quality.

**Start with the [project report](REPORT.md) or [PDF](report/sleeping_machines_status.pdf).**
The next language comparisons follow the [frontier compute protocol](experiments/FRONTIER_COMPUTE_PROTOCOL.md)
and [compute-allocation theory](experiments/theory/43_compute_allocation_and_frontier_scaling.md).
The original ideas are preserved in the
[historical motivation manifesto](HISTORICAL_MOTIVATION_MANIFESTO.md).

## Demonstrated capabilities

| Capability | Completed evidence | Scope |
| --- | --- | --- |
| Ours: integrated sparse temporal language | **3.121 development bpc**, 324 available units / six selected receiver updates per character | 32K fit, four passes, 8,191 cold development targets, payload 16; no dense carrier or statistical experts; [completed record](experiments/results/parallel_language/local_full_sparse_language_D32768_p2_20260930T175400Z.json) |
| Ours: integrated depth comparison | **3.633 → 3.542 development bpc**, six → eight receiver blocks | Same 2K fit/development, payload 32, four passes, seed 6; extra depth also increases capacity/work; [eight-block record](experiments/results/episodic_language/local_episodic_pair_receiver_D2048_d32_s6_depth8_20260930T210000Z.json) |
| Ours: online neural learning | **3.191 frozen → 3.096 adapting bpc** on a new stream | Inherited integrated checkpoint; score before each 16-character block update, 8K development stream, all-neural-parameter adaptation; [completed record](experiments/results/online_language/local_integrated_online_backbone_D8192_20260930T200000Z.json) |
| Ours: earlier temporal carrier | **2.210 development bpc** | Width 128 / 1M fit / four passes; every carrier layer executes. Retained diagnostic, distinct from sparse receiver and KV models |
| Ours: learned language context | **3.395 validation bits/character** with eight layers versus **3.464** with one; lower is better | Small language/depth development screen |
| Ours: temporal order learning | **99.73–99.93%** across five runs after one pass over 2,000 examples | Depth-three event chains; Transformer controls reach 33.25–40.80% after repeated fitting on those examples; declared structured task |
| Ours: rule generalization | **100% on all 3,440 unseen mod-17 triples**, with 69 learned phase scalars | Periodic primitive in the common model; supplied period 17; certified across all 4,913 possible triples |
| Ours: longer-context retrieval | **100% at four times the training context** | Common two-layer carrier plus learned relative pointer; controlled synthetic task |
| Ours: temporal composition | Native shared-motif models reach approximately **99.65%** | Task-specific native model; event activity and dense MACs are different work measures |

A matched width-128 pilot adds content-dependent memory write/forget gates and
improves **2.643 to 2.587 development bpc**, with 0.50% more parameters and
1.41% more fitting arithmetic. The input vector already enters persistent state
and a gated residual output; it is not replaced by a constant node vector.
Frozen interventions confirm that both input content and earlier messages affect
predictions. These are one-seed development results from the earlier carrier.
Its [historical staged campaign](experiments/queue/local_language_nextscale_20260930T163234Z.json)
is preserved. The current priority is the [parallel-head campaign](experiments/queue/local_parallel_heads_overnight_20260930T231500Z.json),
with independent heads, packed historical storage and measured optimizer
comparisons before larger-data promotion.

The completed language references remain available for the full learned-event
benchmark; lower test bits/character is better:

| Model | Fitting characters | Bits/character / split | Whole fitting GFLOPs |
| --- | --- | --- | --- |
| Ours: integrated sparse temporal, payload 16 / six blocks | 32K, four passes | **3.121 / development** | 31.053 |
| [LSTM, width 512](experiments/results/e174/aligned_lstm_10m_20260930.json) | 10M, six passes | **1.799 / test** | 432,593 |
| [Transformer, width 256, four layers](experiments/results/e174/aligned_tf_10m_20260930.json) | 10M, four passes | **1.908 / test** | 888,775 |
| [AWS LSTM, width 512](experiments/results/aws_20260929/aws_e64_lstm_D90M_baseline_20260929/lstm_D90000000_s512_p6_dr0.1_v.json) | 90M, six passes | **1.661 / test** | 3,893,396 |

These are different data budgets, capacities and scoring splits; this inventory
does not establish matched-quality or iso-FLOP superiority. Every cost uses the
same whole-fitting denominator, including backward, clipping and Adam, with
unit-weight special functions. The report plots [quality versus total fitting work](report/figures/language_quality_vs_work.png)
and [quality versus inference work per character](report/figures/language_quality_vs_inference.png),
with all completed variants in the appendix ledger. The inference plot distinguishes
the Transformer window scorer from a hypothetical KV-cache scenario; neither
measures joules. Arithmetic-only counts are preserved.
The [AWS integrated 10M definition](experiments/AWS_INTEGRATED_10M.md)
is committed; a completed integrated official test is still pending.

The separate statistical count/copy baseline reaches 1.727 text8 test
bits/character after 10M-character count fitting. It does not use the learned
Sleeping Machines event backbone and is not evidence of that backbone's
language quality or compute advantage. Its comparison is retained separately
in the report's language appendix.

[Accuracy versus computation](report/figures/consolidated_work_frontiers.png)
shows the consolidated arithmetic and retrieval comparisons. The report contains
protocols, reference models, training budgets and the definitions behind each
work estimate. Counted operations and logical memory visits are not measured
energy.

### The next integrated result

The current candidate combines sparse receivers, independent parallel temporal
heads and historical KV retrieval in **eight event blocks**. Completed
single-head 8K controls reach 3.311 bpc for receivers and 3.357 for content-indexed
KV; adding this KV did not improve quality. The new two-head 2K fit reaches
3.786 bpc on a larger 8K development stream, showing that more mechanisms alone
do not ensure better fitting. Numerical contracts pass for independent
projections, causal timing, evolving channel buffers, all-head gradients and
exact recovery. The next guarded stages test accumulated Adam updates, then
promote two-/four-head models only through declared quality and memory gates.
See [current state](experiments/HANDOFF.md), [the guide](experiments/INTEGRATED_LANGUAGE.md)
and [the theory](experiments/theory/48_parallel_heads_and_work_scaling.md).

We have explored only a small part of the architecture's design space. The next
evidence must show which combinations improve held-out quality for their complete
training and inference costs. The following earlier results remain useful
mechanism evidence and controls.

The common implementation currently
covers multiple independently trained tasks; sharing an implementation does not
by itself establish general representation learning.

A bounded **language/depth screen** now reaches **3.395 validation
bits/character with eight layers**, versus **3.464 with one layer**. It uses
8,192 training characters, four passes and 1,024 validation predictions; all
layers' value, route and memory-time parameters update. Shuffling preceding
characters while preserving the last character, count and timestamps increases
the deeper model's loss to 3.805. This is evidence of trainability and context
sensitivity, with higher computation cost for the deeper model. It is a small
development result, not a large-corpus result or a scaling claim.
[Results and work audit](experiments/results/e133/generic_language_audit_20260929.json).

The selected six-block temporal encoder reaches **79.69% on 512 held-out SHD
utterances**, and **78.84% on a reused disjoint 657-utterance audit**, from
reserved training-file speakers. It inherits a twelve-block fitting pass and
development-based prefix selection. This demonstrates learning and some
speaker transfer. The
published official-test protocols are different, and our official SHD test set
remains untouched.

The next scaling milestone is a learned language model whose event backbone owns
the prediction: increasing data budgets, matched Transformer/RNN/state-space
references, and explicit measurement of quality, parameters, training work,
memory traffic, time and energy. A large reduction in joules at useful predictive
quality would be a significant result even before an advantage in raw loss.
See the [language scaling protocol](experiments/LANGUAGE_SCALING_PROTOCOL.md).

## Architecture and learning

The [integrated language models](experiments/INTEGRATED_LANGUAGE.md) are the
current combined-mechanism experiments. At each depth a learned query races
state-dependent keys; one receiver updates and sends a gated content-bearing
message with an arrival time. Unselected receivers retain their state without
empty-tick evaluation. The KV variant adds indexed historical key/value races.
Content, keys, clocks, gates and retention all learn. Training evaluates admitted
losing alternatives for a conserved local route teacher and charges those reads.
Fixed receiver pools and a bounded causal schedule remain declared constraints;
arbitrary topology and unrestricted asynchronous overlap are not demonstrated.

The [shared model](experiments/SHARED_MODEL.md) combines configurable mechanisms:

- **Sparse event carriers:** vector messages pass through a configurable depth
  without evaluating a hidden grid of empty time steps.
- **Local temporal memory:** causal numerator/mass states retain and combine
  observations with linear scan work.
- **Hard races and learned delays:** a winning route emits its own payload and
  time. Losers remain distinct training alternatives.
- **Separate keys and values:** an optional key stream computes input-dependent
  routing and clocks while value learning preserves that schedule.
- **Structured memories:** optional conditional evidence, relative pointers and
  periodic transformations supply useful computations when appropriate.
- **Task-appropriate supervision:** categorical decisions or next-event type and
  waiting-time likelihoods, including the information in silence.

Tasks have separate fitted weights and may use different depth or primitives.
Small vector maps and output readouts use dense arithmetic. The earlier cross-task CPU
reference sorts arrivals and replays contexts. A separate generic streaming
path retains modal state and delayed messages across chunks; its eight-layer
contract checks prefix causality and all-layer credit without prefix replay.
The integrated language models retain state across chunks without prefix replay
and score bounded indexed candidates. Competitive larger-data prediction and
useful candidate coverage remain research objectives. All discovery, teaching
and optimizer work belongs in the resource accounting.

## Read the theory and evidence

| Document | Purpose |
| --- | --- |
| [Report and applications](REPORT.md) | Accessible overview, strongest results, potential and benchmark appendices |
| [Integrated language](experiments/INTEGRATED_LANGUAGE.md) | Current sparse receiver and episodic KV architecture, evidence and queues |
| [Shared model](experiments/SHARED_MODEL.md) | Earlier cross-task implementation, adapters, contracts and retained evidence |
| [Theory index](experiments/THEORY.md) | Formal derivations organized by theme, with assumptions and proof scope |
| [Mathematical program](experiments/MATHEMATICAL_PROGRAM.md) | Open analytic problems and their decisive measurements |
| [Research roadmap](experiments/ROADMAP.md) | Next experiments and architectural priorities |
| [Language scaling protocol](experiments/LANGUAGE_SCALING_PROTOCOL.md) | Generic prediction, baseline matching and physical work measurements |
| [Parallel training protocol](experiments/PARALLEL_TRAINING_PROTOCOL.md) | Sequence parallelism, persistent inference and proposed architecture controls |
| [Online language protocol](experiments/ONLINE_LANGUAGE_PROTOCOL.md) | Test-time adaptation, memory and frontier comparison experiments |
| [Findings](experiments/FINDINGS.md) | Completed experiment history and detailed observations |
| [Historical manifesto](HISTORICAL_MOTIVATION_MANIFESTO.md) | Original motivation, exploratory ideas and early references |

Source lives in [sleeping_machines/](sleeping_machines/); experiment drivers,
executed queue commands and result records live in [experiments/](experiments/).
The implementation is a research reference, with explicit contracts and source
hashes rather than a claim of a production event processor.

## Repository layout

| Path | Contents |
| --- | --- |
| [sleeping_machines/](sleeping_machines/) | Event engine, shared models and work accounting |
| [experiments/](experiments/README.md) | Experiment drivers, research documents, queue commands and results |
| [report/](report/) | Report generators, figures and published PDF |
| [tests/](tests/) | Unit checks for the event engine and foundational experiment claims |
| [scripts/](scripts/) | Host provisioning and development helpers |
| [legacy/](legacy/) | Earlier implementations and historical results |

Local datasets, virtual environments, caches, runner logs and model checkpoints
stay outside version control. Queue commands and result records remain tracked.
The [experiment guide](experiments/README.md) explains how to find the commands
and evidence for a result.

For the foundational unit checks in a Python environment with
[requirements.txt](requirements.txt) installed:

```bash
python -m pytest tests/ -q
```

## Run experiments safely

Read [AGENTS.md](AGENTS.md) before launching work. Every experiment runs through
[run_safe.sh](experiments/queue/run_safe.sh), which holds a host-local lock,
limits threads and monitors memory. Run **one job at a time per host**, preserve
at least **8 GiB of available host memory**, and give changed settings a new
queue name and result tag. Inspect existing jobs, completed results and GPU
occupancy first.

The current runner expects this checkout at `/workspace` and its interpreter at
`/workspace/.venv-docker/bin/python`. The CPU environment uses
[requirements.txt](requirements.txt), PyTorch and `h5py`; task datasets are
supplied separately. The [installation runbook](AWS_EXPERIMENT_RUNBOOK.md) and
[bootstrap script](scripts/bootstrap_aws_experiments.sh) document a provisioned
experiment host. Choose resource limits from the actual host capacity.

For a small shared-model contract check in an already provisioned environment:

```bash
RUN_TAG="shared_contracts_$(date -u +%Y%m%dT%H%M%SZ)"
QUEUE="experiments/queue/${RUN_TAG}.txt"
printf '%s experiments/e120_shared_contracts.py --tag %s\n' "$RUN_TAG" "$RUN_TAG" > "$QUEUE"
MEM_CAP_KB=3600000 MEM_CAP_RSS_KB=2600000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 \
  bash experiments/queue/run_safe.sh "$QUEUE"
```

These caps suit the bounded CPU check on a host with enough free memory. Training
runs need their own measured memory budget and timeout. Completed experiment
commands are preserved under `experiments/queue/`; use a new tag when reproducing
them. Logs and result records carry the executed settings and measured metrics.

## Citing

Sleeping Machines — Tero Keski-Valkama and Karoliina Salminen.

```bibtex
@article{keskival2021sleeping,
  title={Sleeping Machines},
  author={Keski-Valkama, Tero and Salminen, Karoliina},
  year={2021},
  doi={10.5281/zenodo.13207423}
}
```

[![DOI](https://zenodo.org/badge/342583401.svg)](https://zenodo.org/doi/10.5281/zenodo.13207423)
