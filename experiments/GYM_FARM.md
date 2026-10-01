# A research gym for sparse temporal architectures

Current AWS exception: AGENTS.md authorizes up to three one-thread CPU jobs on
`ip-172-31-47-132` through the measured locked-slot scheduler. Other hosts
retain serial training, and GPU jobs never overlap. The original serial design
below is retained as the default; AWS_RUN_FAST_MATRIX.md documents recovery.


Goal: submit a declared architecture/variant, run a small cross-domain screen,
then spend resources only on hypotheses supported by completed comparisons.
This is an experiment farm, not a reinforcement-learning environment farm.
[The AWS matrix](AWS_EARLY_INDICATION_MATRIX.md) defines priorities and promotion.
[Procurement](RESEARCH_FARM_SHOPPING_LIST.md) separates equipment from cloud use.

## Worker and coordinator contract

One guarded training job per host, as required by AGENTS.md. Several independent
hosts can run different matrix cells concurrently. Containers on one host share
the same host lock; they are not independent hosts. An unusually large AWS
instance could support simultaneous small fits technically, but the current
policy does not permit a second trainer there. Do not bypass run_safe's lock.
Use measured memory/CPU throughput to choose between splitting future capacity
into smaller instances and revising the policy deliberately with the user.

A coordinator reserves a cell to a host with a unique AWS tag and a lease;
run_safe remains the final host-local admission authority. Claim a job once,
validate its immutable queue/source hashes, inspect existing results/jobs/GPU,
set resource caps from actual free memory (retain >=8 GiB), and run its unique
one-job queue in tmux. A failed lease is not evidence that the remote process
has stopped: inspect the host before retrying. Failed retries receive new tags,
except explicit exact-source recovery supported by that driver.

Workers keep logs, checkpoint/RNG/optimizer state where supported, hardware,
wall time, activity and full resource ledgers. They return completed JSON plus
its immutable sources and command. Only one publisher edits the main report and
commits on main. Do not let independently running hosts overwrite the report
or each other's queues. Give each host a non-overlapping tag prefix.

## Adapter registry and evidence stages

Every adapter declares observed fields, permitted labels, state reset boundaries,
time semantics, prediction target, train/dev/test split and accounting scope.
Architecture capabilities are explicit: content/state, independent heads,
computational delays, counterfactual credit, dormant capacity, optional activity,
window integration, repeated trains and local asynchronous learning. A method
cannot acquire an absent mechanism by changing its name in the registry.

| Domain | Existing usable adapter | Important restriction |
|---|---|---|
| Temporal order / elapsed-time kernels | native_event_benchmark.py | Synthetic; observed addresses and forced block/head activity |
| Character language | native_language_benchmark.py; clock_feature_language_benchmark.py | text8, ordinary credit16; no claim of frontier language quality |
| Tabular rows | native_tabular_benchmark.py | Independent rows, feature identities; static processing coordinates |
| Robot force/pose streams | Planned MIT pushing adapter | Must preserve unequal timestamps and whole-trial splits |
| Repeated trains / integration | Numerical primitive contracts | Not yet installed in an integrated fitted model |

Matrix planning is offline and does not provision AWS or start training. It
must emit unsupported cells with a reason rather than fabricated runnable
commands. Source fingerprints, data fingerprints and protocol IDs make results
comparable. A changed command requires a new tag even if the architecture name
is the same.

Stages: numerical contracts -> guarded accounting smoke -> single-seed small
screen -> paired independent seeds -> scale/data ladder -> frozen held-out
confirmation -> hardware energy/traffic. Promotion is a separate decision,
never a consequence of an attractive pilot training score. Do not run the full
Cartesian product of widths, depths, pools, seeds and tasks immediately.

## What the dashboard should show

Quality versus whole fitting work and inference work, with raw results available;
CPU emulator versus explicitly projected event arithmetic; fitting size/passes,
parameters, dormant/occupied state, scored keys, delivered values, selected
updates, losing-route credit, optimizer cost and wall time. Latency distributions
and memory should accompany FLOPs. Keep negative points and completed parents.
Tree comparisons need their own traversal/fitting ledgers, not invented FLOPs.

Use paired data splits/seeds for mechanism contrasts. Discovery development
scores are not independent confirmation. Tune an equal declared budget for every
family; freeze configuration before held-out test evaluation. Record blocked
adapters, timeouts and failed numerical contracts alongside successful cells.
The initial planner and JSON registry are deliberately small; distributed
leases, remote dispatch and universal model adapters remain engineering work.
