# Next native battery: replicate useful time and shared learning

Prepared, unlaunched plan:
`gym/plans/aws_native_confirmation_20261002T010500Z/manifest.json`.
This follows the already-reserved AWS seed7/8 event replications. It prioritizes the successful
**full eight-block native model** at a small budget, retaining independent H2,
temporal races, content/memory mixing, separate key/value roles, losing-route
credit and private addressed states. It introduces no architectural substitution.

## Fixed comparisons and admission

1. Paired timing S4/private/P0: observed elapsed time versus refitted rank time.
2. Order S16/P0: shared processing/common source seed versus private rules/seeds.

Seeds6/7/8, d8, depth8, H2, pool2,128 fitting queries/pass, four passes,
256 development queries,64-query Adam windows,lr.003. Data seeds1201/2201
stay fixed. Select the lowest development NLL checkpoint across exactly four
passes. Score1024 new queries from generator seed3201 only after selection.
Keep every seed. No hyperparameter changes, best-seed choice or ensemble.

There are **two new training jobs and ten frozen checkpoint evaluations**.
Seed6 reuses the completed screen's selected states; timing observed/rank and
shared/P0 seeds7/8 reuse the already-reserved AWS replication's checkpoints.
Only private S16 seeds7/8 need additional fits. Restoration must reproduce
the published development score within1e-7, matching source/data/settings,
parameter count and selected-epoch lineage. Missing checkpoints stop admission;
never silently refit. Its original full fitting work and wall time stay charged.
These are three fitted seeds but only two new seeds after screening, not three
independent architecture discoveries. The untouched data supply confirmation.

Thirty manifest stages comprise eight reused numerical contracts/smokes,
ten checkpoint-restoration checks without holdout access, and twelve final
scores. All checking stages finish before any final score. New seeds use the
same frozen driver and optimizer/configuration as the successful contracts.

Before launch: confirm the reserved event replication worker exited and its
completed summary exists, inspect live queues/processes,
memory and GPU occupancy, and verify the ordinary host lock is free. Do not
start beside another worker. On the approved host `ip-172-31-47-132`, the
existing worker may use up to three one-thread CPU slots. It reserves the host
lock and uses per-slot run_safe locks/watchdogs, summed RSS-cap admission and
an8GiB available-memory floor. Other hosts run one training job at a time.
The observed prior pilots took795–879s and stayed below976MiB RSS. Caps remain
2441MiB RSS/3907MiB VMS;3600s fresh-fit and600s checkpoint-evaluation timeouts.
No new GPU or dense-control training is launched locally.

```bash
tmux new-session -d -s aws-native-confirmation-20261002T010500Z '.venv-docker/bin/python -u scripts/run_aws_matrix_recovery.py --manifest experiments/gym/plans/aws_native_confirmation_20261002T010500Z/manifest.json --jobs 3 >> experiments/queue/aws_native_confirmation_20261002T010500Z.out 2>&1'
```

The existing worker publishes completed JSONs serially on main. Avoid external
rebase during publication; preserve a needs_review lifecycle before preparing
any recovery. Canonical report publication remains owned by the local host.

## Prespecified evidence and gates

Run the analysis only when all twelve final results exist:

```bash
.venv-docker/bin/python -m experiments.split_confirmation_analysis --manifest experiments/gym/plans/aws_native_confirmation_20261002T010500Z/manifest.json --output experiments/gym/plans/aws_native_confirmation_20261002T010500Z/confirmation_analysis.json
```

Primary outcomes are the two confirmation accuracy gains. Resample architecture
seeds and independent holdout populations in a crossed bootstrap; each short/
long timing pair remains one population. Bonferroni across the two comparisons
gives97.5% intervals. Three seeds on one fixed synthetic distribution provide
limited uncertainty; do not pool repeated examples across seeds as independent.

Timing qualifies for a separately frozen larger test only if its adjusted lower
gain is at least25pp and every observed-time seed reaches85%. Sharing qualifies
only if its adjusted lower gain is positive, each seed gains accuracy and each
shared/private whole fitting work ratio is at most1. Report NLL and all per-seed
numbers alongside the primary outcomes. No automatic scaling is admitted.

Historical and newly fitted work use the same denominator:512 fitting query
presentations, including all input events, forward/backward, losing proposals,
normalization, clipping and Adam. Show whole-fit GFLOPs, fit MFLOPs/query and
inference MFLOPs/query together. Frozen evaluation wall time is separate.
No zero-cost historical fitting or physical-energy claim.

The order comparison tests map sharing PLUS common source seeds. It does not
identify map-only attribution. Rank-time timing has an information ceiling,
not the expressivity of a time-aware recurrent or Transformer baseline. Passing
shows replicated mechanism advantage, not recognized-domain supremacy.

After this short battery, prioritize matched exposure/capacity and time-aware
controls, then recorded real events. Protected-spectrum/embedding controls and
the frozen delivery/write decomposition are separate diagnostics. The prepared
`AWS_BANKNOTE_CONFIRMATION.md` is already running before the reserved event
replications; let its original supervisor finish and publish its full analysis. Larger
language fits should wait for useful integrated credit/quality signals. Keep
the deferred large Transformer comparisons on the provisioned AWS host visible
and reserved; this CPU battery is not a replacement for those comparisons.

The first local draft `aws_native_confirmation_20261002T010000Z` is preserved
as superseded/unlaunched because AWS reserved overlapping fits during its
preparation. Use only the replacement above, which reuses that work.
