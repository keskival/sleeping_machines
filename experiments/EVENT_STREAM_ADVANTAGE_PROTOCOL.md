# Constructing a decisive event-stream advantage

The target is useful predictive quality with lower total work across a declared
class of timestamped streams. Preserve the established temporal algebra,
counterfactual credit, persistent content and capacity/activity separation.
This is a prospective protocol; no new results or running event-camera jobs
are implied. Current local parallel-head language training retains priority;
new Transformer/LSTM training is reserved for AWS.

## A common interface, with distinct tasks

Represent an event as (timestamp, source/address, content). Characters are
regularly spaced observations; sensor events need not be. Reuse delay/race,
state evolution, mixing and route-credit primitives, with declared task adapters
and objectives. Current cross-task evidence uses separately trained models;
shared weights and joint multimodal representation learning are further tests.
Report adapter/projection and readout work as part of every full model.

## Evidence ladder

1. Reproduce the completed temporal-order, composition and longer-context
   retrieval results from their saved protocols and multi-seed records. Extend
   to held-out delays, long gaps, timestamp jitter and independent concurrent
   streams. Preserve original results beside each new protocol.
2. Use the existing SHD infrastructure as the first real-stream learning test.
   Freeze training-only preprocessing and speaker-disjoint development choices.
   Score the untouched official test only after model/protocol selection. Report
   quality at several observation deadlines, not just full-utterance accuracy.
3. Add DVS Gesture for event-camera classification, then automotive detection
   only after the spatial addressing and causal output path are validated.
   These are distinct claims: a speech result does not establish camera results.
4. Add marked-event prediction with both next-type and waiting-time likelihood.
   Include the survival term during silence and end-of-window censoring. Score
   before any online update; keep frozen and online comparisons separate.

Primary starting references: [IBM DVS Gesture publication](https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system)
and [Prophesee automotive dataset paper](https://arxiv.org/abs/2001.08499).
Native event hardware and learnable-delay SNNs have prior successes; do not
portray this field as uniformly failed. Our claim concerns the combined
trainable temporal/state/routing construction and its measured resource curve.

## Scaling axes that isolate the mechanisms

| Axis | Controlled change | Hypothesis and failure condition |
| --- | --- | --- |
| Silence/duration | Same event count and content; increase label-preserving gaps where task semantics permit | No periodic state scan; work follows arrivals and required deadlines. If the task changes with silence, score its time-dependent objective instead |
| Temporal information | Matched count/content, varied order or intervals | Learned time/state improves quality; timestamp permutation and frozen-time ablations identify reliance |
| Dormant capacity | Grow receiver/module pool, hold admitted routes and active depth fixed | Better quality without proportional activity; include discovery, cache footprint and losing-route teaching |
| Concurrent streams | Increase unrelated sources with fixed relevant signal | Selective recruitment avoids global updates; include contention, queue/selection and all observed input work |
| Relevant history | Increase retrieval distance while preserving useful memories | Persistent state or economical discovery retains quality; do not hide a shorter effective context |
| Depth | Increase actual event blocks under fixed data/work comparisons | Credit reaches useful deeper representations; depth alone does not prove efficient scaling |

Do not infer dormant-capacity savings from an oracle that supplies the useful
module. If a larger pool is searched exhaustively, charge every key score. Use
controlled retrieval coverage tests and unselected-module intervention checks.

## Controls and ablations

Compare an event-token Transformer with timestamp encodings, a tuned recurrent
or continuous-time state model, and a competitive event/SNN model where feasible.
Provide a tuned frame/bin representation as an application control, charging
binning and empty intervals; it must not be the only comparison. All controls
receive the same causal input information, output deadlines and data splits.
Use matched tuning budgets, quality/work curves and parameter/state inventories.

Ours ablations retain the same interfaces: fixed versus learned delays;
static versus evolving state; random/frozen versus learned routing; missing
versus full counterfactual credit; single versus independent parallel heads;
fixed versus increased available capacity at fixed admitted activity. Report
capacity/work changes caused by each intervention. A common implementation
family is useful; shared-weight transfer needs its own held-task experiment.

## A claim worth publishing

For each task publish accuracy or calibrated likelihood against complete
training FLOPs, inference work/event and work/stream, decision latency and peak
memory. Include candidate discovery, losing-route teaching, optimizer updates,
readouts, preprocessing and queue/index costs. Logical accesses and projected
physical-race work are separate from CPU emulator measurements. Call a joule
advantage only from a measured, specified hardware/system boundary.

Use at least three independent seeds for promoted claims, paired held-out
examples and confidence intervals appropriate to independent speakers/sessions.
Declare the primary quality margin, work budget and scaling sweep before fitting.
A defensible success is reproducible Pareto improvement at comparable quality,
plus a scaling sweep showing why the gap persists or grows. A single synthetic
win supports its task, not universal supremacy. Keep failures, checkpoints and
prior positive evidence visible; update the report only from completed files.

## Instruction-conditioned control and robotics

Robotics combines irregular observations, continuously evolving dynamics,
language instructions and timed actions. It motivates a common architecture
rather than assuming every part of control is an event-classification task.
Begin in reproducible simulation: a held-out instruction selects goals, sensor
streams reveal partial state, and action queries have explicit deadlines.
Test delayed feedback, long-lived goals, distractor events, observation gaps and
novel combinations of instructions/objects. Keep development environments and
test environments/seeds separate; prevent future observations from entering the
controller. Score task success, action latency/deadline misses, prediction
calibration, memory and complete training/inference resources.

Compare against strong instruction-conditioned recurrent and Transformer
controllers with the same observations, action opportunities and training budget.
Separate language understanding, persistent state, temporal prediction and motor
readout interventions. A richer world state need not reproduce attention exactly;
judge its learned usefulness. Sharing frozen or trained primitives differs from
jointly trained representations and must be labeled. Frontier language competence
and deployable robot reliability are not implied by present event-task results.
This remains a prospective simulation benchmark, not an authorized real robot
operation or a claim that a robotics experiment is running.

## Cross-modal integration: the stronger test

A shared computational family is a starting point. A joint model should allow
language to alter event routing/memory, events to ground language predictions,
and persistent state to inform actions. Use modality-specific learned input
projections into a common content/state space, source/modality identifiers and
explicit timestamps; preserve observation causality and action deadlines.
Language position and physical elapsed time are different quantities: document
their encoding/calibration rather than conflating token index with seconds.
The present parallel-head candidate is character-specific (27 symbol pools);
it is not already a joint sensor/language controller.

Require tasks that cannot be solved from either modality alone: an instruction
selects which of several visually/event-defined objects to follow; later sensor
evidence resolves a language query or changes the action required for a goal.
Hold out instruction/event combinations, measure single-modality baselines,
shuffle the instruction across examples, mask cross-modal memory access and
freeze instruction-conditioned routing. Report whether grounding improves
success/quality at a given complete work budget, including all adapters and
communication. A pair of independent networks merely sharing code does not
pass this integration milestone. Model size/compression gains are then measured
against joint Transformer/recurrent controls, not assumed from expressivity.
