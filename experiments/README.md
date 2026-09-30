# Experiments and evidence

Start with the [project report](../REPORT.md) for the current results and their
scope. This directory preserves the experiment implementations and the evidence
behind the report, including exploratory runs and failed approaches.

The current combined-mechanism experiments are the
[integrated sparse temporal language models](INTEGRATED_LANGUAGE.md): learned
content, sparse receiver updates, temporal query/key races and counterfactual
route credit. The eight-block parallel-head candidate retains historical token KV
entries, transfers per-head winning values and mixes evolving timestamped
channels through independent projections. Read [HANDOFF.md](HANDOFF.md) for live
queue state; a committed plan alone does not establish that a remote job is running.

## Research documents

| Document | Purpose |
| --- | --- |
| [INTEGRATED_LANGUAGE.md](INTEGRATED_LANGUAGE.md) | Current integrated receivers, episodic KV, evidence, work ledgers and queue |
| [EVENT_STREAM_ADVANTAGE_PROTOCOL.md](EVENT_STREAM_ADVANTAGE_PROTOCOL.md) | Prospective native-stream quality/work and dormant-capacity tests |
| [HARDWARE_VALUE_PROPOSITION.md](HARDWARE_VALUE_PROPOSITION.md) | Globally clockless target, FPGA/ASIC path and energy accounting |
| [SHARED_MODEL.md](SHARED_MODEL.md) | Earlier cross-task model, adapters and preserved implementation contracts |
| [THEORY.md](THEORY.md) | Index to mathematical derivations in [theory/](theory/) |
| [FINDINGS.md](FINDINGS.md) | Completed experiments, measured results and limitations |
| [ROADMAP.md](ROADMAP.md) | Research priorities and next experiments |
| [LANGUAGE_SCALING_PROTOCOL.md](LANGUAGE_SCALING_PROTOCOL.md) | Language model comparisons and measurement protocol |
| [PARALLEL_TRAINING_PROTOCOL.md](PARALLEL_TRAINING_PROTOCOL.md) | Sequence scans, persistent inference and architecture comparisons |
| [ONLINE_LANGUAGE_PROTOCOL.md](ONLINE_LANGUAGE_PROTOCOL.md) | Causal online adaptation and modern architecture controls |
| [FRONTIER_COMPUTE_PROTOCOL.md](FRONTIER_COMPUTE_PROTOCOL.md) | Modern tokenization/position, event-target work and adaptive budget experiments |
| [MATHEMATICAL_PROGRAM.md](MATHEMATICAL_PROGRAM.md) | Open analytic questions and proposed measurements |

## Locate a run

The [current parallel-head campaign](queue/local_parallel_heads_overnight_20260930T231500Z.json)
checks complete two-/four-head models, optimizer accumulation and gated data
scaling. The drivers are `parallel_head_race_language_screen.py` and
`parallel_head_accumulated_language.py`; completed records live in
[results/episodic_language/](results/episodic_language/). Theory note 48 derives
full-bank FLOPs/access comparisons and distinguishes hypotheses from results.
Receiver scaling uses `integrated_language_benchmark.py` and
[results/parallel_language/](results/parallel_language/). The separate
`integrated_online_language.py` experiment adapts the full backbone after causal
predictions; its results live in [results/online_language/](results/online_language/).
The [AWS 10M receiver definition](AWS_INTEGRATED_10M.md) has a distinct protocol.

The [older full learned-event suite](queue/local_full_proper_suite_20260930T131028Z.json)
preserves speech/vision/market/temporal and carrier-language configurations; it
is superseded as the local priority. Speech is paused, with checkpoints intact.
Every fit uses a separate one-job guarded queue. Live `.running.json` files,
logs and checkpoints stay local; only completed JSON is report evidence.
New dense Transformer/LSTM training is reserved for AWS.

Earlier experiment drivers use the `e<number>_*.py` naming scheme. The number connects
the driver to its section in the findings and its directory under
[results/](results/). Later runs also use dated tags to distinguish settings,
seeds, audits and repeated runs.

- [queue/](queue/) holds the commands used for runs and the guarded runner.
- [results/](results/) holds per-experiment metrics, configurations and audits.
- Local `queue/logs/` holds job logs; runner output uses `.out`.
- [reference/](reference/) contains reference implementations used in comparisons.

Read the result record and the corresponding findings together. A successful
single-seed development run supports an exploratory observation; broader
benchmark claims require the matching protocol, budgets and repeated evidence.
Checkpoints (`.pt`), per-example arrays (`.npy` and `.npz`), job logs (`.log`)
and runner output (`.out`) stay local and are ignored by Git. Queue commands and
result JSON remain tracked so the report's settings and evidence are reviewable.

## Reproduce safely

Follow [AGENTS.md](../AGENTS.md) and the
[AWS runbook](../AWS_EXPERIMENT_RUNBOOK.md) before launching an experiment.
Inspect existing jobs and results first. Use a uniquely named queue containing
one job and invoke [run_safe.sh](queue/run_safe.sh); run one training job at a
time on each host. Select memory caps and timeout from host capacity and the
measured workload, retain the RSS watchdog and at least 8 GiB of available host
memory, and inspect GPU occupancy for CUDA runs.

Historical queue files are records of earlier work. Reproduce a run with a new
queue name and output tag rather than overwriting its result. The
[root README](../README.md#run-experiments-safely) includes a bounded contract
check example and the expected experiment environment.
