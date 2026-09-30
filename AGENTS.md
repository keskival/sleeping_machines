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

# Research direction and architectural continuity

- The objective is a trainable, scalable substrate that computes through time
  and sparse asynchronous events, with useful capacity beyond active work.
  Optimize prediction quality, learning and total resource use together.
  Improving a conventional dense model alone does not establish this thesis.
- Evaluate unusual claims by derivation, implementation and reproducible
  evidence. Familiarity with published architectures is not a truth criterion.
  Preserve and clearly present surprising positive results when supported;
  investigate their limits with equally demanding controls. Neither assume
  supremacy nor impose an arbitrary ceiling on potential improvement.
- Keep the core mechanisms explicit: computational delays and temporal races,
  including race attention; sparse addressed state updates; small messages that
  mix incoming content with persistent memory; separate keys and values; deep
  credit to unrealized alternatives and optional routes; silence-aware
  supervision where applicable. Distinguish available capacity, scored keys,
  selected state updates, value deliveries and counterfactual learning work.
- Prioritize experiments combining the core mechanisms. Use numerical
  contracts and small integrated fits before scaling them. Carrier-only,
  dense, synchronous and single-mechanism models are diagnostic controls;
  keep them labelled and do not silently promote them into the main research
  architecture or let their queue displace the integrated experiments.
- Before a substantive architectural substitution, state the concrete failure
  being addressed, the theoretical reason for the change, which mechanisms it
  retains or removes, and its implications for inference and learning work.
  Record this reasoning and the required comparison in the theory/handoff.
  Demonstrate retained contracts and test the change in an integrated model
  before committing a long run. Routine work within the authorized scope does
  not require a new user-approval step.
- Established embeddings, gates, normalization and local vector operations
  may support the substrate. Do not substitute them for temporal computation,
  sparse selection or counterfactual learning merely because they are standard.
  Attribute known primitives accurately and explain the contribution of the
  complete construction and its tested consequences.
- A weak result from a restricted variant is evidence about that variant.
  Diagnose missing mechanisms, clock precision, information paths, credit and
  protocol before generalizing it to the full architecture. Preserve negative
  findings too; revise or abandon a hypothesis when the relevant evidence
  warrants it, with the reason stated beside the historical claim.
- Keep the report's architectural explanation and strongest valid comparisons
  prominent, quantitative and visual. Explain what each model actually uses.
  Dense-carrier accuracy is not evidence for sparse race attention; dormant
  state is not proof of zero key-scoring or training cost. Surprising results
  need clear scope, not automatic dilution into vague potential.
- Compare consistent resource boundaries and quality/data protocols. Charge
  candidate discovery, losing-value credit and optimizer work; distinguish
  FLOPs, wall time, traffic and measured energy. Report isolated or projected
  advantages as such. Fix accounting errors without treating familiar dense
  computation as the preferred architecture by default.
- After each completed integrated stage, update the report appendix with
  quality, data/passes, capacity/selected activity, full fitting FLOPs,
  per-target fitting work and inference work. Show supported raw work gaps
  against saved references while marking unequal quality/data and estimate
  conventions. Reserve comparable-quality or iso-FLOP supremacy claims for
  completed evidence under the relevant protocol; never fill a pending cell
  with a prediction or an ongoing training score.
- At handoff, name the prioritized integrated model and queue, the remaining
  gaps in mechanism coverage, and any proposed departure from this direction.
  Read `experiments/THEORY.md` and its relevant notes before redesigning a core
  mechanism. `HISTORICAL_MOTIVATION_MANIFESTO.md` records the original motivation;
  it is not a substitute for current proofs or measurements.
