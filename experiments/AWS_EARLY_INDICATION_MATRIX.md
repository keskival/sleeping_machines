# AWS: fast evidence across temporal, language, tabular and robot streams

The priority is a broad but small **hypothesis screen**, not a long language run
or RL training. Start with integrated eight-block, two-independent-head native
models. Compare the complete construction and matched mechanism controls;
scale only a completed, useful quality/resource result. The current local delay
campaign continues independently. AWS inventory/queue/GPU state has not been
verified from this workspace; this document does not claim remote jobs started.

## What should run first

| Priority | Question and experiment | First fit / dev budget | Promotion evidence |
|---|---|---|---|
| 1 | **Time is computation:** native elapsed-time kernel, observed timestamps vs rank-only times | 128 / 64 query targets, 4 passes, H2 d8 L8 pool2 | Observed-time held-out gain under equal content, plus gap/stretch diagnostics |
| 1 | **Counterfactual credit makes hard routes learn:** order task, full vs pathwise credit | Same 128 / 64 targets and 4 passes | Paired quality/whole learning-work result; independent seeds, not just route movement |
| 2 | **Dormant capacity beyond activity:** 4, 16, 64 active source addresses at the same total targets | 128 / 64 targets, same L/H/d/pool | Fixed selected updates/event; quality and measured storage; charge additional exposed parameters/Adam |
| 2 | **Useful temporal reception:** text8 native, R2 uniform, R4 late-half, R2 waiting-only | 512 / 1,024 characters, 4 passes, H2 d8 L8 pool2 | Gain over parent AND waiting control; full fit/inference work; R2-uniform vs R4-late matched added budget |
| 2 | **Conditional static interactions:** banknote classification, red-wine regression, native vs R2 vs boosted trees | 128 / 128 independent rows; 4 passes or 4 tree candidates | Train-only scaling, duplicate-feature-group isolation, dev log loss / RMSE and comparable wall/memory budgets |
| 3 | **Real unsynchronized robotics:** MIT planar pushing, preserve force/pusher/object timestamps | Predeclare 16 fit / 8 dev / 8 reserved-test whole trials; at most 64 queries/trial, 4 passes | Causal prediction beats persistence and dynamics controls; alignment and missing-observation ablations |
| 3 | **Tiny robot sequence contract:** UCI execution failures, one problem first then all five | Whole trials; 60/20/20 grouped split, 4 passes | Macro-F1/balanced accuracy; temporal order vs summary/tree control; replication |

The generated first-wave runnable matrix has **17 pilot cells**: seven temporal,
four language and six tabular. It also emits two blocked robotics specifications,
with their missing adapters explicitly named. Contracts/smokes are prerequisites,
not extra benchmark results. One architecture seed (6) is enough to reject a
broken design, not enough to establish an advantage. Keep all outcomes.

Initial wall-time budgets are scheduling limits, not measured runtime promises:
accounting smokes <=30 minutes; temporal/tabular pilots <=60 minutes; language
pilots <=90 minutes. Record actual time and RSS, then resize or stop a cell that
misses these budgets. Do not silently reduce its passes/dev set to get a win.
Small language results mostly diagnose learnability; 511 fitting targets are
not a plausible demonstration of frontier BPC.

## Robotics without reinforcement learning

**First real asynchronous target: MIT pushing.** The original data contains three
timestamped streams that are not synchronized: pusher pose, object pose and
interaction force/torque. Start with one surface/object slice and an immutable
filename list selected before fitting, using raw JSON/H5 rather than a resampled
export. This offers a direct asynchronous sensor-processing question on actual
robot recordings. [Official data and format](https://web.mit.edu/mcube/push-dataset/index.html),
[original paper](https://arxiv.org/abs/1604.04038).

Supervised tasks: (a) object-state estimation at a declared observation deadline
with the current object observation withheld; (b) predict object displacement
20/50 ms ahead from the available prefix and measured pusher/force history.
A causal readout uses only records timestamped <= its cutoff. Future object
samples supply labels only; interpolating between two **past** samples is allowed
for inputs, interpolation that reads a future sample is forbidden. State reset
between pushes, train-only scaling, relative pose convention and angular wrap
must be fixed. Several windows from one push never cross splits. Near-repeat
pushes are grouped; reserve new contact angles or speeds for a distinct shift
check, not substitute that harder split into the headline comparison.

Controls: hold-last-state, past-only constant velocity, ridge/boosted-tree
prediction from the same causal prefix, a small GRU, and a small Transformer
on AWS. The dynamics controls have the same sensors, history cutoff, prediction
horizon and training rows. A resampled dense control gets missing masks and
elapsed time, and its resampling/empty-bin costs are charged. Compare raw-stream
native vs rank-only time, frozen-clock reception, no receiver memory and no
counterfactual credit after a full-model pilot learns. Explicit asynchronous
sensor drops/staggering are controlled corruptions, not native dataset facts.

Report position RMSE in mm, angular MAE, error vs horizon/gap, held-out-trial
intervals, inference operations/query, events consumed, state bytes and latency.
Distinguish per-event latency from latency per prediction when native processes
many intervening sensor events. No RL, simulator rollouts or claims of closed-loop
robot safety/control capability follow from this supervised benchmark.

**Smallest quick robotics check: UCI Robot Execution Failures.** There are 463
trials across five problems, with 15 regularly sampled force/torque readings per
trial. It is post-detection failure classification, not asynchronous acquisition
or early warning. Keep each problem's class vocabulary and trial splits separate;
no pooling five label spaces. Start LP1, then run all five under a frozen plan.
Use indexed sample order unless precise timing can be established from metadata.
The dataset's regular sampling makes it a sequence-learning test, not evidence
of savings from native asynchronous clocks. [Official UCI dataset](https://archive.ics.uci.edu/dataset/138/robot+execution+failures).

Robomimic low-dimensional recordings are a later optional source for supervised
next-state prediction or action imitation; no RL is needed to use recorded
trajectories. Many small Lift recordings are simulator demonstrations and must
be labelled accordingly; real robot image sets impose a larger perception cost.
Pin dataset version and split entire demonstrations. Do not confuse predicting
recorded actions with verified closed-loop success.
[Official dataset/version documentation](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html).

## Make the screen enlightening rather than a grid search

Use single-factor contrasts first. The pool/source ladder changes capacity,
not just parameter count; measure exposure and the actual selected activity.
A clock branch also changes waiting and noise: a waiting-only control isolates
useful readout. R2 uniform versus R4 late-half tests depth allocation at the same
added clock/projection budget. A width ladder d8->d16 is separate from a data
ladder 512->2K->8K; change one first. L4 is a labelled shallow diagnostic, not
silently the main deep architecture. Multi-spike integration remains a proved
primitive; no current row should be labelled a fitted spike-train architecture.

Before promotion, check predictions/targets, gradient/route exposure, time-origin
invariance, row/episode resets, source provenance and accounting coverage. Repeat
promising matched pairs with seeds 7 and 8, keeping all three. Select an equal
number of hyperparameter configurations per family; final claims require a
held-out confirmation protocol, not whichever development point looks best.

Promotion for a language reception variant: >=.02 bpc gain vs native AND >=.01
vs waiting-only at <=1.25x projected fit work, then paired seeds. These are
screening thresholds, not proof of meaningful population improvement. A native
parent can separately advance under a predeclared near-quality/work envelope;
record the tolerated quality loss rather than say equivalent accuracy.
For event/tabular/robot screens, retain the completed Pareto frontier in quality,
whole fit work, inference work, wall time and memory. Promote a matched pair if
it Pareto-improves or supplies a reproducible mechanism effect worth studying;
set any numerical noninferiority margin before seeing its new scores. A plateau
or reversed ablation is useful negative evidence and changes the next allocation.

## AWS handoff and concurrency

Generate with `scripts/plan_research_gym.py --run-prefix aws_<host>_<UTC>`.
The planner never starts training or provisions capacity. The emitted JSON has
unique one-job queues, explicit prerequisites, immutable command/source hashes,
timeouts, hypotheses and blocked cells. Use `scripts/run_research_gym.py` after
host inspection and measured cap selection; it runs one cell at a time under
run_safe. It does not merge results into the report or bypass host locks.
Run distinct non-overlapping plans on different hosts; synchronize results via
one main-branch publisher. Do not reuse a tag after changing settings.

Several small fits could fit in RAM on one large instance. Current AGENTS.md
still requires one training job per host; the existing safe-runner enforces a
host-wide lock. Use multiple independent AWS instances for simultaneous screens,
or obtain an explicit policy change before implementing a measured bounded
same-host scheduler. Memory fit alone does not establish throughput: Python,
BLAS threads, cache/memory bandwidth and graph allocation can cause contention.
No concurrent GPU jobs. The new native drivers are CPU-only; a GPU instance
does not automatically accelerate them.

Immediate worker allocation: temporal hypotheses on CPU worker A; language
contrasts on CPU worker B; tabular/tree pairs on CPU worker C; deferred large
dense controls on the provisioned GPU/AWS host. With one instance, use the
priority order above serially, first-wave screen before big dense references.
The user has reserved new Transformer/LSTM fitting for AWS; local completed
references remain visible and unchanged.
