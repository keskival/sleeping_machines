# Sleeping Machines

**Learning to compute with time, vector messages and local memory.**

Sleeping Machines explores models in which an event carries a learned vector and
an arrival time. Nodes accumulate local evidence, transform messages and compete
through learned delays. The winning message determines what happens next;
unrealized alternatives can teach the network to make better choices.

**Delays do computation.** Changing a delay changes arrival order, which memories
interact and which route wins. Sleeping units and sparse activity are part of the
resource model; temporal computation is the central architectural idea.

The goal is a broadly useful model family with competitive predictive quality and
substantially less physical work in both training and inference. This repository
contains the implementations, mathematical analysis, reproducible experiments and
completed results used to pursue that goal.

**Start with the [project report](REPORT.md) or [PDF](report/sleeping_machines_status.pdf).**
The original ideas are preserved in the
[historical motivation manifesto](HISTORICAL_MOTIVATION_MANIFESTO.md).

## Demonstrated capabilities

| Capability | Completed evidence | Scope |
| --- | --- | --- |
| Learned language context | **3.395 validation bits/character** with eight layers versus **3.464** with one; lower is better | Small expert-free development screen; no frontier language claim |
| Rule generalization | **100% on all 3,440 unseen mod-17 triples**, with 69 learned phase scalars | Periodic primitive in the common model; supplied period 17; certified across all 4,913 possible triples |
| Longer-context retrieval | **100% at four times the training context** | Common two-layer carrier plus learned relative pointer; controlled synthetic task |
| Temporal composition | Native shared-motif models reach approximately **99.65%** | Task-specific native model; event activity and dense MACs are different work measures |

[Accuracy versus computation](report/figures/consolidated_work_frontiers.png)
shows the consolidated arithmetic and retrieval comparisons. The report contains
protocols, reference models, training budgets and the definitions behind each
work estimate. Counted operations and logical memory visits are not measured
energy.

### The central next result

The earlier native-mixture headline scores are **withdrawn**: a partial-word
context used the character being predicted to detect a space. The experimental
review also quarantines an older market comparison whose trade-size threshold
used held-day observations. Original records are retained; corrected causal
language reruns compare mixtures with and without the word expert.
See the [experimental review](experiments/EXPERIMENTAL_REVIEW.md).

The common implementation currently
covers multiple independently trained tasks; sharing an implementation does not
by itself establish general representation learning.

A bounded **expert-free language screen** now reaches **3.395 validation
bits/character with eight layers**, versus **3.464 with one layer**. It uses
8,192 training characters, four passes and 1,024 validation predictions; all
layers' value, route and memory-time parameters update. Shuffling preceding
characters while preserving the last character, count and timestamps increases
the deeper model's loss to 3.805. This is evidence of trainability and context
sensitivity, with higher computation cost for the deeper model. It is a small
development result, not a large-corpus result or a scaling claim.
[Results and work audit](experiments/results/e133/generic_language_audit_20260929.json).

An eight-layer event/race model reaches **72.3% on 512 held-out SHD utterances**
from reserved training-file speakers. This demonstrates deep learning and some
speaker transfer. It does not establish competitive speech recognition; the
published official-test protocols are different, and our official SHD test set
remains untouched.

The next scaling milestone is a learned language model whose event backbone owns
the prediction: increasing data budgets, matched Transformer/RNN/state-space
references, and explicit measurement of quality, parameters, training work,
memory traffic, time and energy. A large reduction in joules at useful predictive
quality would be a significant result even before an advantage in raw loss.
See the [language scaling protocol](experiments/LANGUAGE_SCALING_PROTOCOL.md).

## Architecture and learning

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
Small vector maps and output readouts use dense arithmetic. The current CPU
reference also sorts arrivals and replays contexts; persistent incremental
execution and cheaper candidate discovery are important systems objectives.
All this work belongs in the resource accounting.

## Read the theory and evidence

| Document | Purpose |
| --- | --- |
| [Report and applications](REPORT.md) | Accessible overview, strongest results, potential and benchmark appendices |
| [Shared model](experiments/SHARED_MODEL.md) | Current implementation, task adapters, contracts and cost boundaries |
| [Theory index](experiments/THEORY.md) | Formal derivations organized by theme, with assumptions and proof scope |
| [Mathematical program](experiments/MATHEMATICAL_PROGRAM.md) | Open analytic problems and their decisive measurements |
| [Research roadmap](experiments/ROADMAP.md) | Next experiments and architectural priorities |
| [Language scaling protocol](experiments/LANGUAGE_SCALING_PROTOCOL.md) | Generic prediction, baseline matching and physical work measurements |
| [Findings](experiments/FINDINGS.md) | Completed experiment history and detailed observations |
| [Historical manifesto](HISTORICAL_MOTIVATION_MANIFESTO.md) | Original motivation, exploratory ideas and early references |

Source lives in [sleeping_machines/](sleeping_machines/); experiment drivers,
executed queue commands and result records live in [experiments/](experiments/).
The implementation is a research reference, with explicit contracts and source
hashes rather than a claim of a production event processor.

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
