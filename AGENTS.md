# Experiment host rules

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
- When another host is changing the repository, work on a separate branch and
  do not push to `main`. Update findings and the report from completed result
  files, and distinguish exploratory single-seed evidence from benchmark
  claims.
