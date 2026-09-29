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
The missing E64 Transformer checkpoints (1M two-layer, 10M four-layer, and 1M eight-layer) are recreated for E76's attention audit;
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
termination. 

## Data and dependent follow-up

The expanded manifest contains 339 unique AWS job tags. The initial controller
loaded the earlier plan before data preparation completed; a second tmux session
waits for its normal completion and then runs the expanded plan, skipping every
recorded outcome. It refuses to continue if the first controller stops early.

All 21 Binance pilot archives (BTC spot, ETH spot, BTC perpetual; August 25–31)
were downloaded and verified against the publisher's SHA-256 checksums. Raw DVS
Gesture was downloaded from the mirror documented by snnTorch, verified against
MD5 8a5c71fb11e24e5ca5b11866ca6c00a1, and extracted; all 98 train and 24 test
recordings and labels are present. Dataset manifests are in
`queue/aws_market_data_20260929.json` and `queue/aws_dvs_data_20260929.json`.
The previous E60/E60b selected test results already exist and are not rerun.

The remaining DVS job explicitly selects `--data dvs`; no SHD jobs are scheduled.
E78 consumes the completed AWS E77 probability arrays, if available. E80's test
epochs and frozen policy come only from its completed validation results. New
optional reference-path arguments allow these analyses without copying over
legacy outputs. A failed prerequisite leaves its analysis blocked.

The output-isolation wrapper has three passing tests covering preservation of
legacy files, refusal to reuse an AWS output directory, exception provenance, and
non-finite accuracy/BPC rejection. The tiny E77 smoke also exercised it through
the real safe runner. Non-finite BPC/accuracy marks a job failed even if the
underlying script returned normally. Binary artifacts stay local.

Monitor with `tmux attach -t aws-non-shd`,
`tail -f /tmp/aws-non-shd-controller.log`, or the individual safe-runner logs.
The follow-up session is `aws-non-shd-followup`; its log is
`/tmp/aws-non-shd-followup.log`. Results are running/queued, not yet all complete.
