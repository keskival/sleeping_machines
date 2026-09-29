# Generic language learning and physical scaling

## The claim this experiment must answer

Can a reasonably generic Sleeping Machines model learn useful deep language
representations while its total physical work grows more slowly than competitive
Transformer, recurrent and state-space references?

The native E79 text8 mixture is strong evidence for specialized prediction. It
is not the answer to this question. The E120 common language screen also fits
explicit conditional count memories before its neural stage. A shared interface
and a small development win do not establish generic learned representation.
The eight-layer SHD development result demonstrates learning and limited transfer,
but does not establish competitive speech representation. Both gaps require
successful held-out predictions from a learned event backbone.

## Primary model and controls

The primary model must use learned embeddings, event payload transformations,
causal local memory, hard route selection, delays and a learned output readout.
No explicit n-gram, dictionary, cached target distribution, pointer/copy expert,
phase solution or hand-written task solver supplies the output probabilities.
Keep these mechanisms available as separately labeled hybrid controls.

One generic model family is configured by capacity, depth and memory budget;
weights are trained separately on each task. For the initial language experiment,
use an evidence-free instance of `SharedEventModel`. Preserve observation/target
separation and compare a one-layer representation with genuinely trained deeper
representations. Frozen keys with learned values are a declared learning phase;
full model claims must also show how useful keys are learned.

Before increasing the data budget:

1. Demonstrate loss reduction and gradient/finite-update agreement without any
   expert contribution; verify all intended value layers actually learn.
2. Measure representation retention and transfer, not only fitting accuracy.
3. Establish usable route alternatives and their downstream credit. E130 shows
   why a correct interior derivative alone can miss a harmful hard winner change.
4. Implement persistent causal state across contiguous tokens. The existing
   common language query replays 32 characters, and its cost must be reported as
   prefix replay until that behavior is replaced.
5. Verify state reset, chunk boundaries, delayed messages, teacher alignment and
   truncated credit. Input character i must never depend on target i+1.

An evidence-free short screen can diagnose trainability now. It is a prerequisite,
not an extrapolated 10M-character or billion-token result. A mathematically exact
local race-credit option is developed in theory §§197–200; it is not yet the
current hard-race training implementation.

## Data and baseline matching

Use a fixed train/validation/test manifest with byte offsets or document IDs,
checksums, preprocessing, tokenizer and reset semantics. Choose hyperparameters
on validation only; test once for each declared completed comparison. Log unique
training tokens and total token presentations, including repeated epochs.

For continuity, retain the existing text8 character protocol and its 90M/5M/5M
split. A 10M-character result cannot be relabeled as 10M subword tokens. Moving
through 10M → 100M → 1B+ tokens requires a larger, predeclared corpus and a fixed
common tokenizer for every model. That is a new comparison with its own split;
do not train on the reserved text8 validation/test regions to extend its budget.

References should include a tuned causal Transformer, LSTM/GRU and a competent
state-space model. Use the same tokens, context information, loss, evaluation
positions and precision. Declare two distinct comparisons where feasible:

- comparable total model/storage capacity and data presentations;
- comparable measured training resource budget.

Equal parameter count does not imply equal compute, memory or optimization
quality. Give every family a declared tuning budget and preserve unsuccessful
configurations. Existing specialized mixtures remain an additional reference.
Heavy runs belong on the provisioned AWS host in a separate branch and unique
host-specific queues; coordinate ownership before launching them.

## Required measurement ledger

| Quantity | Required boundary |
| --- | --- |
| Quality | Validation/test NLL and bits/token; bits/character only for the character protocol; confidence intervals across evaluation blocks |
| Capacity | All learned parameters, frozen parameters, buffers, persistent event state, pending messages, optimizer state and peak resident memory |
| Work | Separate forward, backward, counterfactual replay and optimizer arithmetic; event deliveries, candidate evaluations, sorting, scan work and context replay |
| Memory | Logical reads/writes from the execution ledger **and** hardware-measured memory/cache traffic where counters are available; distinguish them |
| Time | Warmed throughput, time to specified loss, end-to-end training/evaluation time and preprocessing time; hardware, precision, threads and utilization |
| Energy | Metered joules to a prespecified validation loss and joules per useful prediction at declared quality; whole-device/host boundary, idle convention and instrumentation |

Profiler contraction FLOPs are incomplete when sorting, nonlinearities or sparse
bookkeeping lack counters. Label coverage and omissions. An event count does not
measure actual memory accesses. A cache miss counter is not automatically DRAM
bytes. GPU utilization is not energy. Report unavailable measurements explicitly.

Energy claims require measurements over the same hardware boundary. Record total
energy and any idle-adjusted number separately. Include both key/value streams,
loser-credit work, teacher signals and candidate discovery. Clock-delay simulation
units are not CPU latency seconds. Existing E127 contraction savings did not
produce a CPU speedup; operation ledgers are not an energy claim.

## Decision criterion

Plot held-out loss against data, parameters, physical work, elapsed time and,
when measured, joules. Show the quality/resource Pareto frontier. The practical
question is the work or energy needed to reach a useful quality target, with
uncertainty and a declared training budget.

Equivalent quality with much lower energy is a major win. A modest quality cost
with a large measured energy saving can also enable applications unavailable to
more expensive models. Ratios such as 10×, 20× or 100× are possible target
scenarios, not current project measurements. Higher quality and lower total
training/inference energy together would support the strongest architecture claim.

A favorable short-range fitted exponent is insufficient: context replay,
candidate search, delayed-credit storage or communication can change the curve
at larger scale. Publish the individual budget points and their measured work,
not only an extrapolation.

## Host-safe execution

Inspect queues, completed results, memory, running jobs and GPU occupancy before
training. Launch exactly one uniquely tagged job per host through
`experiments/queue/run_safe.sh`. Preserve at least 8 GiB of available memory,
keep the RSS watchdog enabled, and size timeout/caps from measured workload.
For CUDA, set and monitor a PyTorch memory fraction and GPU occupancy. Keep
long runs in tmux, preserve progress checkpoints, and never bypass a held lock.
