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
