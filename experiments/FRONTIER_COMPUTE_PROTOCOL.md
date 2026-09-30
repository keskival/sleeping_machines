# Modern language models and adaptive compute budgets

This protocol turns [the compute-allocation theory](theory/43_compute_allocation_and_frontier_scaling.md)
into falsifiable architecture comparisons. It is a development plan, not a
completed frontier benchmark. Existing deferred AWS Transformer comparisons
retain priority; later runs require a provisioned GPU host and the guarded queue.

## Representation and baseline gate

The present 27-character text8 and 131-token causal-prefix experiments remain
historical controls. Before a modern architecture claim:

- Freeze a document-level train/development/test manifest with corpus revision,
  hashes, deduplication and preprocessing. Reserve evaluation documents; never
  fit vocabulary or routing statistics on them.
- Use one modern byte-level BPE or unigram tokenizer across all learned arms.
  Start with 32K vocabulary; compare 64K on development if the corpus supports
  it. Use maintained official tooling, record its version and tokenizer hash,
  preserve Unicode/byte coverage and declare normalization and special tokens.
  These sizes are proposed controls, not a discovered optimum.
- Train vocabulary on train-only documents or freeze an existing tokenizer at
  an immutable revision. Report fitting work, original bytes, emitted tokens,
  total presentations, release latency and vocabulary/head storage/work.
- Use fixed document segmentation and causal token masking. Report shared-token
  NLL and bits per original byte as a canonical-segmentation coding score with
  explicit EOS treatment. Do not call that exact per-byte conditional likelihood
  or compare different-tokenizer perplexities. A raw online-byte claim needs
  prefix-stable buffering and partial-token probability contracts.
- Give the dense Transformer a competent pre-normalized architecture, rotary
  positions, tuned gated feed-forward blocks, optimized SDPA/FlashAttention,
  packed/variable-length sequences, appropriate precision and equal tuning
  opportunity. Record actual implementation revisions and supported kernels.
- Include a sparse MoE and a modern selective-SSM/attention-recurrent hybrid,
  using official implementations where possible. These controls already exploit
  selective compute; the architecture claim must survive them.

Modern preprocessing cannot substitute for representative data, effective
optimization or trained representations. Retain the evidence-free learned
event backbone as the primary arm; count/copy/word experts are separate hybrids.

## Position, scheduling and execution gate

Use shared token coordinates for all primary arms. Store original byte offsets
for measurement; a byte-coordinate position arm is a separate intervention.
Keep linguistic coordinates distinct from learned event-delivery delays.

Compare no rotational phase, fixed multiscale phase and learned phase in the
event memory; use suitable RoPE on the Transformer. Existing event rotation is
already a relative-time mechanism, not absence of position. Verify offset shifts,
document resets, long gaps, aliasing and long-context precision. A learned
frequency change must receive useful credit and improve held-out predictions.

For each parallelizable event core, verify scan/sequential outputs and gradients,
causal future perturbations, and chunk/persistent decode equivalence. Benchmark
forward, backward and optimizer separately at fixed inputs. Generation remains
autoregressive after parallel sequence fitting. Charge candidate discovery,
sorting, scan buffers, state, cache consistency and communication.

## Budget interventions

Use paired development experiments and common initializations where meaningful.
Change one axis before combining mechanisms.

| Hypothesis | Intervention at controlled resources | Evidence needed | Failure signal |
| --- | --- | --- | --- |
| Useful dormant capacity | Increase stored experts/bank size at fixed active candidate/value budget | Better held-out NLL, sufficient useful exposure and acceptable storage/communication | Untrained experts, collapse, no quality gain or search growth |
| Better memory allocation | Vary number of modes and decay/phase ranges at fixed total active work | Long-context gain and ordinary text gain | Aliasing, old-state interference or loss of local quality |
| Bounded sparse retrieval | Compare indexed candidate budgets with full/audited retrieval | Task-visible recall, paired suffix quality, index maintenance cost | Hidden all-pairs discovery or valuable missed records |
| Useful credit allocation | Vary horizon and counterfactual teaching budget with full backward accounting | Reproducible downstream gain per additional training resource | Larger teachers without transfer; route changes hurt |
| Useful selective depth | Change active depth while preserving known baseline behavior where possible | Trained serial representation and better quality/resource frontier | Shallow bypass, route extinction or variance cost |
| Beneficial local adaptation | Frozen versus head/memory updates, scored before updates | Domain-drift gain, stable stationary behavior and forgetting controls | Leakage, cache inconsistency or adaptation cost outweighs gain |
| Better allocation than uniform growth | Fit marginal-value curves on one development subset; predict allocation; confirm elsewhere | Lower loss at the same complete resource caps | Controller overfits, interactions reverse predicted gains |

At a provisioned budget, first fit small architecture response curves, then
reserve an independent larger budget point to test their prediction. Sweep data
as well as capacity; equal raw bytes, tokens or parameters alone is not an equal
resource budget. Record all unsuccessful trials and distinguish final-run cost
from total research/tuning cost.

## Accounting on the event target

The primary architecture FLOP column counts the declared **logical event
algorithm**, not Python/PyTorch simulator overhead. Multiply-add is two floating
operations. Include event transitions, candidate scoring, losing alternatives
required for teaching, gradients, counterfactual/replay work, clipping, optimizer
updates and fit-only setup. Prefix replay required by a historical model remains
part of its algorithm until replaced by persistent state.

Existing complete-step traces calibrate the declared event arithmetic; they are
not measurements of a fabricated event processor. No common-model padding is
added. Arithmetic that the declared parallel scan requires remains charged.
Count event ordering, comparisons, integer/index work, special functions and
memory traffic in separate ledgers with coverage labels. Hardware event queues
can change their implementation, not make discovery or communication free.

Keep simulator overhead separate: host-language dispatch, allocation,
instrumentation and backend bookkeeping. Where its arithmetic cannot be
separated from algorithm arithmetic, label the estimate's uncertainty. A future
event hardware optimization must declare its schedule/state/learning algorithm
and verify equivalent outputs/updates before replacing the estimate. Do not
silently remove losing values, backward or dense optimizer work that current
learning still requires.

References get the same arithmetic convention and their optimized declared
algorithm. Report actual simulator/hardware runtime, utilization, transferred
bytes, resident memory and energy independently. Event-target logical FLOPs can
support architectural analysis; they do not establish measured device speed or
joules. The current breadth trace is an estimate of full logical fitting work,
with declared special-function/integer exclusions, not total physical cost.

## Promotion to larger scales

1. Close the deferred historical AWS references and validate modern tokenizer,
   position and execution contracts. Preserve the old and modern protocols as
   separate comparisons.
2. Train the modern matched arms at feasible GPU budgets. Use at least three
   seeds for the selected comparison and paired document/block uncertainty.
3. Promote only a confirmed quality/resource improvement. Refit data/capacity
   allocation at each scale and test one held-out larger budget prediction.
4. Measure resources to reach several fixed quality levels. Estimate exponents
   with uncertainty; report floors, crossovers and failed extrapolations.
5. For a frontier claim, train/evaluate modern large-scale models on representative
   held-out language and downstream tasks, with matched observations, declared
   tokenizer/adaptation policies and complete training/inference resources.

An initial win supports a local result. A growing relative advantage needs a
verified scaling trend; neither architecture flexibility nor a text8 win proves
frontier supremacy. The empirical objective is a widening quality/resource
frontier against strong contemporary controls.

## Transfer across task classes

Train separate weights with the same declared temporal/event primitives. A
language win does not establish vision, speech or forecasting superiority.

| Task class | Existing evidence boundary | Next fair comparison | Budget prediction to test |
| --- | --- | --- | --- |
| Language | Specialized text8 mixture; small learned stream and local adaptive pilot | Modern shared-token learned event, optimized Transformer, MoE and recurrent/attention hybrids | Spend on memory/retrieval/adaptation rather than uniform active width |
| Event speech | Strongest generic encoder is 408/512 on private held training speakers | Follow the frozen official [SHD protocol](SHD_FRONTIER_PROTOCOL.md), including EventSSM/S7 and declared selection | Preserve fine source information; improve useful state/query and credit before adding depth |
| Event vision | Small DVS gesture development screen with differing architectures; no official frontier result | Subject-disjoint official data, tuned raw-event and competent dense/frame controls with equal observations | Allocate event resolution and memory by task-visible information, charging coalescing and retrieval |
| Forecasting/world models | Exploratory market event likelihood and online learning | Walk-forward prequential evaluation against tuned temporal/SSM/attention controls; proper likelihood/calibration | Adapt local memory under drift at bounded update cost and test forgetting |
| Retrieval/composition | Controlled synthetic generalization and supplied periodic structure | Learned retrieval/composition with nuisance variation and withheld combinations | Larger useful dormant banks at fixed active work, with measured discovery/credit coverage |

Define per-task quality targets and costs before each comparison. Use latency
and state-memory budgets where continuous streams demand them. Shared primitives
must retain task information and useful teacher paths; neither a common interface
nor a supplied synthetic invariant establishes broad learned representation.
Publish a task-by-task Pareto table and only summarize a broad advantage when
independent completed comparisons support it. Do not average incomparable
accuracy, bits/byte and likelihood numbers into a supremacy score.
