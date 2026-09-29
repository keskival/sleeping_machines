# AWS non-SHD benchmark batch — 2026-09-29

Authorized scope: all unfinished non-SHD queued benchmarks, serialized through
`experiments/queue/run_safe.sh`; commit and push completed work. SHD work belongs
to the workstation agent and is excluded. Branch: `aws/non-shd-benchmarks-20260929`.

Host: EC2 c7i.4xlarge, 16 vCPUs, Intel Xeon Platinum 8488C, approximately 30 GiB
visible RAM, CPU-only PyTorch 2.14.0. Package versions are recorded in
`queue/aws_packages_20260929.txt`. The initial checkout was c9a3e37; the latest
AWS instructions were fetched and fast-forwarded to bb89163 before preparing
this batch. The updated instructions require AWS-specific paths and a dedicated
branch; the first three E68 runs below predate that synchronization.

## Completed initial runs

E68 synthetic recall, seed 0, 8,000 updates / 512,000 sequences, unchanged queued
configuration. These are single-seed results, not an architecture ranking.

| Attention | Final accuracy | Wall time |
|---|---:|---:|
| Softmax (R=0) | 18.5% | 45.8 s |
| One race (R=1) | 18.4% | 79.0 s |
| Four races (R=4) | 18.2% | 171.1 s |

Results: `results/e68/recall_R{0,1,4}_s0.json`; exact commands and logs:
`queue/non_shd_e68_20260929.txt`, `queue/runner_non_shd_e68_20260929.out`, and
`queue/logs/e68_recall_R{0,1,4}_s0.log`.

The existing E77 depth-8 100k configuration failed during threshold calibration
under the original 6,000,000 KB address-space cap: the allocator could not obtain
1,124,730,368 bytes for a retrieval tensor. This is an execution failure, not a
quality result. See `queue/logs/e77_depth8_tv_D100k_p1_b2.log`.

The existing tiny E77 route-counterfactual smoke completed through the isolated
AWS wrapper in 1.802 s, peak process RSS 1,297,584 KB. Its result and provenance
are under `results/aws_20260929/aws_e77_route_cf_potential_d512_depth2_20260929/`.
This validates output isolation and execution only; it is not a scale result.

## Remaining batch

`queue/non_shd_audit_20260929.json` records the queue audit, completion evidence,
SHD exclusions, and unresolved dependencies. Exact duplicate commands and jobs
with saved matching configurations or successful lifecycle markers are excluded.
Some legacy queues lack full provenance; configurations without matching evidence
are rerun into isolated AWS paths, preserving historical results.

`queue/aws_plan_20260929.json` is the ordered runnable plan. Large Transformer
controls lead, followed by language, MNIST, synthetic, and deferred theory jobs.
The missing 1M E64 Transformer checkpoint is recreated for E76's attention audit;
the old E64 metric is preserved. E76 now accepts an explicit checkpoint directory
and fails on an empty checkpoint set rather than saving an empty success result.

`scripts/run_aws_non_shd.py` runs the plan under tmux, using a separate one-job
queue for every configuration. `experiments/aws_benchmark.py` changes only the
script's top-level OUT directory to an AWS-specific location and records the
source hash, command, revision, wall time, peak process RSS, and exit outcome.
Model settings, data splits, and seeds are unchanged. The global safe-runner lock
remains authoritative. The host memory floor is 8,192 MB. Each job has a six-hour
limit; failures are recorded before independent jobs continue. Host-wide memory
shortage stops the batch.

The E77 retry allows 24 GiB virtual address space and 20 GiB process-group RSS,
justified by the measured allocation failure and this host's capacity, while
retaining the 8 GiB floor. Large counting tables use 16 GiB virtual / 12 GiB RSS;
other jobs retain the original memory caps. These are ceilings, not reservations.

`queue/aws_progress_20260929.json` records outcomes as jobs finish. Each job's
JSON results, provenance, and logs are committed and pushed to the AWS branch.
Binary checkpoints and probability arrays remain on the host for dependent
analyses; they are not automatically added to Git. Preserve them before host
termination. DVS data, unavailable reference checkpoints, and dependent selectors
must be resolved before their blocked audit entries can be run.
