# Experiment host rules

- Read `experiments/HANDOFF.md` for the current report, publishing and running
  benchmark state before continuing work from a new session.
- Before launching training, inspect the existing queue, result files, report,
  host memory, running jobs, and GPU occupancy. Reuse an existing benchmark
  configuration when possible; prioritize the deferred large Transformer
  comparisons on a provisioned AWS host.
- Run exactly one training job at a time on each host. Put every run in a
  uniquely named one-job queue and invoke `experiments/queue/run_safe.sh`;
  never launch a benchmark directly with Python or bypass its lock. If the
  lock is held, wait or report it. The lock is host-local and does not
  coordinate separate machines.
- Choose memory caps and timeout from the actual host capacity and measured
  workload. Keep the RSS watchdog enabled and preserve at least 8 GiB of
  `MemAvailable`. For CUDA, check and monitor `nvidia-smi`, set the PyTorch
  memory fraction, and never run concurrent GPU jobs.
- Use unique AWS-specific run tags and output names. Never overwrite a prior
  result or reuse a successful queue job name for changed settings. Keep long
  runs in `tmux` and preserve their command, logs, hardware, wall time, and
  metrics.
- Work and commit on `main`, as requested by the user; do not create research
  branches. When another host is changing the repository, preserve its changes
  and coordinate overlapping files. Update findings and the report from completed result
  files, and distinguish exploratory single-seed evidence from benchmark
  claims.
- Preserve completed result files and older report evidence while new runs are
  pending. Replace a leading valid result only after a completed, comparable
  run improves it, retaining the previous result in the historical record.
  State protocol errors and revised interpretations beside the original
  numbers instead of silently deleting them. Keep existing dense controls;
  the user has reserved new Transformer/LSTM training for AWS.
- Preserve the established architectural case when editing the report: time
  performs computation, hard routes learn through counterfactual credit, deep
  persistent event representations, and capacity beyond activity. Retain the
  supporting temporal algebra, key/value separation, silence-aware supervision,
  depth theory and compute-allocation reasoning. Integrate new findings with
  their evidence and scope instead of replacing these principles ad hoc.
