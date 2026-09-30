# Integrated 10M language benchmark on AWS

Use `main`. Read `AGENTS.md` and `experiments/HANDOFF.md` first. The user requests
the integrated architecture and a larger model for the large-data comparisons.
This run uses 32-dimensional messages, six event depths, two receivers per
character pool and 324 available units. It has no dense carrier or statistical
experts. Increasing message width changes local content capacity and arithmetic;
it does not increase selected state updates beyond six per character.

## Defined run

The committed one-job queue is
`experiments/queue/aws_full_sparse_language_D10000000_d32_p2_seed6_20260930T193500Z.txt`.
It calls `experiments/integrated_language_benchmark.py`, a separate resumable
driver; the earlier active/deferred drivers remain unchanged.

| Setting | Fixed value |
| --- | --- |
| Fitting | text8 characters [0, 10,000,000), four passes, from scratch |
| Validation | [90,000,000, 90,200,000), 199,999 cold-context targets |
| Selection | Lowest full-validation bpc at a completed epoch |
| Official test | [95,000,000, 96,000,000), exactly 999,999 targets |
| Test updates | None; selected weights frozen, causal event state persists |
| Alphabet | Existing 27-character alphabet; first target excluded |
| Architecture | payload 32, depth 6, pool 2; all integrated mechanisms retained |
| Learning | Adam 0.001; clipping 1; seed 6; credit horizon 16 |
| Checkpoint interval | Every 512 optimizer steps and every epoch boundary |
| Device | CPU, one thread; current sparse emulator is not a CUDA throughput implementation |

The saved 10M LSTM and Transformer aligned controls already score these same
999,999 test targets. Reuse them. Their sizes and optimizer histories differ;
compare quality together with complete fitting work. Official testing happens
only after the fixed fitting/validation budget; never adapt on those targets.
The separate online-adaptation experiment must not be substituted for this score.

## Before launch

Inspect the AWS host's existing queues, running processes, result files, actual
`MemAvailable`, storage, and GPU occupancy (`nvidia-smi` if present). Run exactly
one training job on the host. Let an existing guarded job finish. Do not start
CPU training concurrently with an existing GPU training job.

First inspect the local width-32/32K result, then its 131K and 1M development
stages. The new driver checks causal predictions, training/inference equality,
10M clock origins, prediction-before-update and exact next-update recovery on
the chosen model configuration before fitting. The long run should follow a
completed useful larger-model pilot, rather than assume capacity improvement.
All source hashes must match the completed core contracts and remain unchanged
for an active run/resume. Preserve the exact checkout used for a long run.

At the measured smaller-model CPU rate (~47 fitting targets/second), 10M/four
passes would take about ten days plus larger validation/test overhead. Width 32
can change that rate. Derive the timeout from its completed pilot's measured
throughput on this host, include validation/test and a recovery margin. A GPU
instance alone does not imply this serial emulator will train faster.

For a host with at least 11 GiB available, the following example retains the
8 GiB memory floor and uses the already measured small-model caps. Confirm the
width-32 smoke/pilot RSS fits; increase caps only from actual available memory
while preserving the floor. Set `AWS_JOB_TIMEOUT_S` from the measured estimate
before this command; there is deliberately no guessed timeout default.

```bash
AWS_JOB_TIMEOUT_S=<measured_total_seconds_with_margin>
tmux new-session -d -s aws_integrated_10m_d32 \
  "cd /workspace && MEM_CAP_KB=4000000 MEM_CAP_RSS_KB=2500000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=$AWS_JOB_TIMEOUT_S WAIT=1 WAIT_TIMEOUT_S=86400 bash experiments/queue/run_safe.sh experiments/queue/aws_full_sparse_language_D10000000_d32_p2_seed6_20260930T193500Z.txt"
```

The queue name is unused and AWS-specific. For another host or changed settings,
make a new timestamped AWS tag/one-job queue. Never overwrite a completed result
or reuse a successful job name for a changed configuration. Put the hardware,
chosen caps/timeout, exact command and commit in the host handoff. Preserve the
runner log and `experiments/queue/logs/<tag>.log` alongside result/checkpoint.

## Recovery and reporting

The driver saves `.progress.pt` and `.running.json` next to its result. Recovery
requires the same source hashes, arguments, fitting/development data and tag.
Create a uniquely named **recovery queue** containing the original job line
with `--resume` appended, then invoke `run_safe.sh` with the same resource guards.
This is recovery of an unfinished run, not a changed model. It restores model,
Adam moments, retained event state, selected checkpoint, pass/target pointer,
accumulated losses, initial parameters and RNG. Do not delete partial files.

On completion, the result includes selected validation/official-test bpc,
whole-stream deliveries, candidate/counterfactual counts, fitting targets,
complete-step representative arithmetic (forward, backward, clipping, Adam),
separate special-function counts and evaluation-work estimates. Runtime includes
validation and auditing; the fitting FLOP ledger excludes those and reports
evaluation separately. RNG, indexing and physical traffic remain additional.
No physical joules are inferred from FLOPs.

Update `REPORT.md`/the PDF from the completed JSON, keeping previous valid rows.
Use identical units and denominators across model rows and keep architecture,
data and quality visible. Commit evidence and report on `main`; the authenticated
host pushes. Do not advertise a pending/live score as completed supremacy.
