# AWS agent: run this next

The committed next protocol is
[AWS_EARLY_INDICATION_MATRIX.md](AWS_EARLY_INDICATION_MATRIX.md).
The ready first-wave plan is
[gym/plans/aws_fast_matrix_v1_20261001T213000Z/manifest.json](gym/plans/aws_fast_matrix_v1_20261001T213000Z/manifest.json).
It has 17 small pilot cells across native temporal, language and tabular tasks,
with numerical contracts and accounting smokes before each. Two robotics
benchmarks are specified but blocked pending causal adapters. This is supervised
learning; do not start RL or a multi-day language run.

## Before running

1. Read HANDOFF.md and AGENTS.md; inspect active tmux sessions/controllers,
   runner locks, completed result files, memory and GPU occupancy. Finish or
   deliberately reserve the existing AWS coordinator after its current fit;
   keep the running depth-four result. Never kill it to make room. Stop its
   automatic advance cleanly before starting the new matrix. The new runner
   refuses to overlap the existing AWS coordinator.
2. Update main only while no report publisher or source-reading trainer can be
   disrupted by the checkout/rebase. Preserve all local changes and outputs.
   Do not force push. Install the pinned requirements; native tabular uses
   scikit-learn 1.9.1. The safe runner assumes /workspace/.venv-docker/bin/python.
3. Validate all source/queue hashes in the manifest. Its drivers are CPU-only.
   The observed AWS host had about 31 GiB RAM and no GPU; recheck actual capacity.
   Keep >=8 GiB available. Choose caps from a smoke; the following 2.5-GiB RSS /
   ~3.9-GiB VMS envelope mirrors measured local workloads, not an automatic
   instruction to apply those caps on an unknown host.

## Run serially, in tmux

After the host is clear and caps are suitable:

```bash
tmux new-session -d -s aws-fast-matrix-smokes-v1 '.venv-docker/bin/python scripts/run_research_gym.py --manifest experiments/gym/plans/aws_fast_matrix_v1_20261001T213000Z/manifest.json --rss-mib 2441 --vms-mib 3907 --smokes-only >> experiments/queue/aws_fast_matrix_v1_smokes.out 2>&1'
```

Inspect its worker status, result JSON, coverage, failure logs, actual wall and
RSS. If every selected smoke completed and budgets remain sensible, run the
pilots with a **new lifecycle attempt**, preserving the first:

```bash
tmux new-session -d -s aws-fast-matrix-pilots-v1 '.venv-docker/bin/python scripts/run_research_gym.py --manifest experiments/gym/plans/aws_fast_matrix_v1_20261001T213000Z/manifest.json --rss-mib 2441 --vms-mib 3907 --attempt 2 >> experiments/queue/aws_fast_matrix_v1_pilots.out 2>&1'
```

The second invocation revalidates and skips successful unchanged contract/smoke
jobs under run_safe, then fits pilots. Exactly one job runs at a time. Temporal
contrasts come first, followed by language and tabular/tree pairs. A smoke or
contract failure stops the worker; preserve it and diagnose, do not skip its
prerequisite. There is no automatic large-model promotion or remote provisioning.

For independent hosts use a distinct prefix/plan, optionally one domain:

```bash
.venv-docker/bin/python scripts/plan_research_gym.py --run-prefix aws_HOSTNAME_UNIQUEUTC --domains temporal
```

Run that new manifest with the same guarded worker after host inspection. Do not
run the same pilot tag on two hosts. New seeds/changed settings need a new plan;
`--seeds 7 8` is for paired confirmation of selected mechanisms, not a blanket
expansion before the screen. No concurrent same-host/GPU training under the
current rules.

## Return evidence

Preserve unique queues, result JSON, hashes, hardware, logs, checkpoints, wall
and activity/resource ledgers. Commit completed evidence on main and coordinate
report edits through one publisher; the gym worker intentionally does not push,
edit the PDF or pretend incomplete runs succeeded. Summarize paired comparisons
and negative results as well as wins. New Transformer/LSTM fits remain AWS work,
but must use their own declared protocols rather than unimplemented matrix cells.


## Measured AWS memory recovery — 1 October

The original attempt1 is preserved as needs_review: its64-source contract
was stopped at2,593,472KiB groupRSS against2,499,584KiB. The identical
contract passed under a uniquely named guarded memory probe, with peak
processRSS2,585,864KiB and39.067s wall. This identifies an inadequate cap,
not a failed numerical assertion or a reason to change the architecture.

Recovery plan:
`gym/plans/aws_fast_matrix_recovery_20261001T213409Z/manifest.json`.
It reuses eleven completed unchanged checks, including the memory probe;
all other stages have fresh tags/queues. Sources/models, data, passes,
activity and optimizer windows are unchanged.64-source stages use4GiB RSS
and6GiB VMS; other stages retain2441MiB RSS/3907MiB VMS. Keep the8GiB
available-memory floor and all guards. These caps apply to this measured
32GiB CPU host, not arbitrary workers.

`run_aws_matrix_recovery.py` runs every remaining contract/smoke before
pilots, validates finite numbers and complete operator coverage, checks
smoke RSS headroom, and commits/rebases/pushes each completed result before
starting the next job. No separate Git watcher is necessary. Failures are
preserved in a separate versioned failure record and stop the worker.
Previous lifecycle/status/logs remain intact. REPORT/PDF ownership stays
with the coordinated report publisher. No speculative larger promotion.

```bash
tmux new-session -d -s aws-fast-matrix-recovery-20261001T213409Z '.venv-docker/bin/python -u scripts/run_aws_matrix_recovery.py --manifest experiments/gym/plans/aws_fast_matrix_recovery_20261001T213409Z/manifest.json >> experiments/queue/aws_fast_matrix_recovery_20261001T213409Z.out 2>&1'
```

Worker status lives beside that manifest as `worker_recovery.status.json`.
Same-host parallelism remains disabled unless the user explicitly replaces
the one-job-per-host rule and a bounded scheduler/lock protocol is installed.


## Explicit AWS parallel exception — 1 October, 21:40 UTC

The user clarified that the serial host restriction applies to the limited
local host and authorized changing it on this AWS host. On
`ip-172-31-47-132`, the measured recovery worker may use `--jobs 3`.
This does not alter other hosts or permit concurrent GPU training.

The scheduler holds the ordinary exclusive host lock for its entire run;
legacy/default runners cannot enter. Each slot still invokes run_safe with
a uniquely named one-job queue and an inherited locked descriptor. run_safe
validates that descriptor and slot1..3, takes its separate slot lock, and
retains per-job RSS/VMS/timeout guards and the8GiB available-memory floor.
The scheduler reserves the sum of active RSS caps before admission (a
conservative check against current MemAvailable), pins math threads to one,
and limits active jobs to three. Complete all contracts/smokes before pilots.
Failures stop new admissions and allow active guarded jobs to finish and
publish. Git publication occurs on the coordinator thread between dispatches.

Launch the committed recovery command above with `--jobs 3` appended to
the Python arguments. Numerical/data/model settings are unchanged; wall
time now includes contention and should not be presented as isolated timing.
Three admission tests verify the inherited reservation requirement, maximum
slot count and exclusion of ordinary runners, without launching training.
