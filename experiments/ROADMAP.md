# Current frontier priorities — 30 September 2026

## Current decision: consolidate quality before attributing a depth gain

E152 retains 406/512 (79.30%) with one 395,814-parameter six-block encoder,
versus E143's two-branch 408/512 (79.69%). Paired-view head fitting improves
the clean-only head by 32 held decisions with the encoder frozen. This is
inherited deployment consolidation, not a fresh generic-model scaling result.
The corrected twelve-block continuation fits 90.72% and reaches 400/512;
there is no demonstrated depth accuracy advantage yet. E155's apparent
calibrated failure was an optimizer scheduler restoration bug; preserve it
as implementation evidence and use E159's verified actual group rates.

1. The matched corrected D6/D12 comparison is complete: both give 400/512 and
   513/657 on the reused audit. Deleting the new D12 blocks changes no audit
   decisions. Test the analytically specified pre-normalized output maps with
   directional observer units (§§261–264), which retain hidden null directions.
   E161/E162 verify exact growth, local teachers, folding and the actual first
   fitting step. Read E163/E164's completed quality/deletion results next.
   Keep selection on private development and distinguish audit reuse from test.
2. Preserve the strongest encoder/head checkpoint. Use the exact affine
   weighted-state query already derived in §§231–233 to address the remaining
   reference pooling difference; compare it at declared training/data cost.
   Analyze fitting-speaker gradient transfer and augmentation covariance
   before changing capacity, seed or an optimizer heuristic.
3. A finite function-space update policy is implemented as E158. First run
   its resource calibration if needed. Its extra fitting/anchor forwards and
   moment semantics are part of training cost; a prepared policy is not an
   improvement. Do not confuse safe fitting descent with new-speaker parity.
4. Freeze an official SHD training/selection protocol for the leading named
   roughly 96% references. Compare quality, full forward/backward work,
   measured memory traffic and eventually joules at fixed quality.
5. Transfer the generic temporal primitive to independently trained language
   and other task models. A single-encoder speech result alone does not absorb
   the explicit count/pointer language expert or establish frontier scaling.

## Primary target: improve the strongest SHD model, then official-test parity

The new inherited parallel model reaches **408/512 (79.69%)**, improving the
original **370/512 (72.27%)** by 7.42 points. A disjoint same-speaker audit also
improves from 463/657 to 510/657. Its six-block width-128 temporal residual learns
source vectors, nonlinear state maps and useful hidden clocks around the
unchanged eight-layer parent. Resetting clocks alone loses 13 held decisions;
resetting the learned stack loses 50. These are fitted-coordinate diagnostics,
not matched retrained controls. See E143–E148 in FINDINGS.md for complete scope.

The immediate targets are to retain this improvement with a single generic
deep encoder; test exact pooled-state queries; measure task-visible clock-orbit
conditioning and bounded finite counterfactual utility; then freeze an official
training/selection protocol for comparison with the named approximately 96%
SHD references. All current scores are private training-speaker development.

E138's 6,144-example warm continuation at the parent terminal learning rate
finishes at 368/512, so it delivers no accuracy-record advance. The compact
12-layer controls below answer architectural questions but are not the primary
accuracy route. Do not substitute their lower scores for this target.

E139 tests the source information boundary: all 700 original channels have
distinct addresses, and source-time features enter learned vector messages
before packet coalescing. A zero-initialized fine contrast map exactly nests
the strongest checkpoint. Deep packet count, closure times, hard winners,
calibration, augmentation and old optimizer are preserved initially. Fine
updates can subsequently change races and are not guaranteed to improve.
The preliminary contract verifies matching extraction/augmentation, initial
logits/clocks/winners and a nonzero source label teacher. Evaluate the completed
run against 370/512; then use an official-test protocol for published parity.

The completed E139 source-only continuation reaches 369/512: one decision
above its matched plain control, still below the best. Fine weights learn but
this intervention is insufficient at the one-pass budget. E140 adds 384
dimensionless temporal phases in addressed coordinate-pair memories. At zero,
the parent logits/winners/clocks agree exactly; all eight phase blocks receive
credit. Its safe same-budget continuation tests useful temporal computation.
Theory §§220–223 derives the phase teacher and conditional depth bound, and
specifies a subsequent width expansion preserving the old function/optimizer.
E140 completes at 368/512, with all phase blocks changed and slightly better
fitting NLL. It does not improve the record. The next architecture must contain
content-dependent affine state updates and richer nonlinear query features,
with an explicit reference-block inclusion and local teacher; increasing
capacity must preserve the old function and optimizer rather than restart it.
Do not promote a phase/source variant or claim parity on these scores.

The frozen-parent E141 control also ends at 369/512, despite removing all
cross-block clipping of the new teachers (0/1,536 clipped updates). The next
architecture is now implemented: signed unnormalized modal states, nonlinear
gates and six width-128 residual event blocks. A zero correction head nests the
strong parent exactly. E142 verifies exact first-state coalescing and adjoints;
E143 completes the declared three-pass 6,144-example residual-learning experiment.
This adds capacity alongside the old model, not fourteen sequential layers.
It completes at 408/512; the improvement is measured, not inferred from teacher norms.
Theory §§226–230 proves the linear reference mapping and separates forward
inclusion from its optimizer geometry and nonlinear pooling semantics.

Theory §§215–219 derives the irrecoverable input quotient, exact affine
coalescing conditions and local source adjoint. It also defines a concrete
stable learned temporal-mode operator that can contain an asynchronous linear
SSM, with winning state/output/clock semantics. The current three-timescale
normalized receiver is not that complete operator. Richer content-conditioned
memory, suitable capacity and training budget remain implementation targets.

## Completed architectural investigation: compact, class-visible deep learning

**Completed:** compact learned exchanges reach 243/512 (47.46%) versus
181/512 (35.35%) for frozen angles, with identical compact query capacity.
The next gap is transferable fitting progress: the final-checkpoint angular
fit step improves fit but slightly harms the audited held-speaker loss, while
the query step improves both. Test fitting-speaker-only invariance objectives
and the continuing compact learning budget; preserve the matched frozen-angle
control. Avoid attributing held-label diagnostics to a deployable training rule.
Neither compact arm beats the full-state 57.8% prototype or common 72.3% result.

The twelve-layer full-state prototype learns 100% of its 1,024 fitting examples
and 57.8% of held speakers after three passes, while a packet-only query learns
46.0%/29.7%. Query capacity differs substantially. Resetting trained angles
preserves all fitting decisions but lowers held accuracy to 54.3%. The next
question is useful representation adaptation under a constrained query.

E137 declares a matched learned/frozen-angle intervention with a rank-16
bank/channel/class query: same initial logits, inherited immutable keys, fitting
and held examples, sample order, compact decoder capacity, and three-pass
optimizer budget. The primitive keeps one addressed winning exchange and one
emitted packet per event. Measure fitting/held curves and actual gradient
transfer before spending a larger fitting budget. A restored finite-update
probe distinguishes useful direction from nonzero gradient norm. Preserve
complete results and restart checkpoint provenance.

Theory §§210–213 connects supervised observability to reachable control,
separates error correction from directional future reserve, and shows why
packet/state cross-covariance matters for propagation. A trainable optionality
critic still needs a declared future adaptation budget and counterfactual
continuation targets. No reserve surrogate is promoted to the production model
without evidence. All-layer state queries also leave a deepest-only composition
question open. Official-test speech accuracy, calibrated early decisions,
persistent language execution and measured physical energy remain frontier gates.

**Usable credit and the memory query (E135–E136):** content retrieval also ends
at 349/512, and removing it preserves that score. It is only weakly exercised
under the present initialization/update budget and has a 5/3 selection-ratio
cap. Query/key gradient support is established; useful finite correction and
transfer are the next criteria. Reversible addressed packet/state exchange
now has an implementation and a twelve-layer augmented isometry contract.
Its classifier must query retained state: a complete exchange can emit zero
while preserving the entire input in memory. Integrate this observable query
and its exact local angle teacher, then measure fitting/held-speaker learning
before expanding depth or data. Keep the winning exchange distinct from its
counterfactual alternative, and charge both policy and state work.

**Full-value result and next structural intervention (E134–E135):** both matched
whole-value phases end at 349/512, below the 370/512 parent. Every value layer
learns, and functional key freezing does not improve this continuation. The
next representation change is content-selective temporal retrieval rather than
another seed or a depth-only sweep. A balanced kernel nests the old means
exactly, has an available query teacher, then unlocks key teaching. Compressed
state, direct retrieval/gradient contracts and the stricter conditional depth
bound are audited. Charge its additional feature state and projection work;
measure held-speaker utility independently of fitting gradient reach. Theory
§§202–205 also identifies pooled-label covariance/cancellation and the missing
persistent-prefix scheduling equivalence.

**Current hardest-gap intervention (E128–E131):** useful common interior
credit exists, but finite representation updates can change a hard winner and
defeat that predicted improvement. Fitting-only frozen-winner/order comparisons
isolate that effect. A separate immutable key stream now computes actual
choices/clocks while new value maps learn; its query, nesting, finite-credit
and checkpoint contracts pass. The two same-budget value-only SHD arms compare
shared and separate streams, with all extra key work counted. Completed scores
are 365/512 and 364/512 respectively, below the 370/512 parent despite improved
fitting accuracy. Preserve the verified isolation mechanism; improving general
representations requires more than teaching two new context maps. Develop
realized joint key/value/clock credit and measure independent value-layer credit.
Theory §§190–196. The [generic language scaling protocol](LANGUAGE_SCALING_PROTOCOL.md)
requires an evidence-free learned backbone, persistent execution and measured
physical work before a frontier scaling claim. Existing language mixture wins
and eight-layer speech learning do not close that gap.

**Generic representation priority (E132–E133):** distinguish existing specialized
language prediction from a learned, evidence-free event backbone. Check trained
value/key support and validation improvement in matched one-/eight-layer screens
before scaling data. The completed expert-free depth-eight screen reaches
3.395 dev bpc versus 3.464 at depth one; all layers update, but parameters and
work rise with depth. This establishes a small learned-language foothold, not
competitive scaling. Exact joint stochastic race credit now has numerical
contracts, including silence/Fisher geometry (§§197–201); it remains a candidate
for the key-policy implementation, with actual counterfactual suffix cost and
variance to be measured. Persistent execution and hardware measurements are
required by the language scaling protocol, not inferred from the short screen.

**Current consolidation:** arithmetic and longer-context recall are retained.
The common model supports a 69-scalar periodic path, two-layer synthetic
configurations and eight-layer speech. Its learned modular rule is certified
by constrained min/max composition. Compare configured operation ledgers with
the same-example Transformer/LSTM controls, including memory search and maps.

**Current SHD intervention:** the augmented deep model reaches 72.1% on 512
held-out utterances; 4,096-example continuation reaches 72.3%. The matched extra
epoch reaches 68.9% with mean pooling and 69.5% with learned event pooling;
neither improves that checkpoint. Inspect teacher alignment/cancellation across
fitting speakers and class-conditional retained information (§§185,187), then
speaker/class errors. Full benchmark quality,
confidence stopping and measured energy remain separate milestones.

**Completed structural test (E126):** two zero-initialized causal context
channels let distant active packets interact without needing intervening relay
packets. The disconnected-component obstruction and constructive remedy are
proved in §§188–189, with exact initial checkpoint/gradient contracts. This is
a more specific architecture change than adding depth or changing seeds.
Full updates reach 351/512 (68.6%); new-column-only teaching reaches 364/512
(71.1%) and 81.8% fitting accuracy, with all old state exactly fixed. Both
remain below the starting 370/512. The new joint-information subspace can
learn; its cross-speaker transfer is not established. Next measure class-wise
teacher alignment across fitting speakers and temporal information retained
by the context state, using fitting-only selection before further continuations.

**Breadth references:** five bounded two-layer Transformer controls now use
the common model's held-out prefixes, neural-fitting budget and input information.
They provide quality and contraction-FLOP references in report Appendix B.
The D8 common model has lower forward work on these screens, but higher
training-forward work on four short tasks; local arithmetic teaching and
event-camera gestures have distinct favorable training-work measures. Optimize
and measure backward work separately, preserving counterfactual support.

The manifesto remains sparse, asynchronous computation and learning driven by
messages, local state, delays, and credit to unrealized alternatives. The results
now include language-mixture and learned-retrieval leads plus deep synthetic
composition. These are different model families; integrating their capabilities
is the central architecture task.

The current decision sequence is in [MATHEMATICAL_PROGRAM.md](MATHEMATICAL_PROGRAM.md).
The analytic synthesis is [THEORY §§155–158](theory/10_trainability_and_frontier_synthesis.md),
and the revised optionality formalism is [§§159–163](theory/11_optionality_and_learning_reserve.md).
The earlier roadmap below is historical; its old queue order is superseded.

**Shared-model priority (E120):** one eight-layer implementation now has separately
trained cross-task screens. Preserve the exact speech extraction and corrected
recall length contracts. Next compare retrained evidence-gate/additive-head
controls, integrate periodic state and native hold/veto composition, and make
prefix queries incremental without future leakage. Full protocols and remaining
coverage are in [SHARED_MODEL.md](SHARED_MODEL.md); theory §§176–180. Existing
specialist wins remain the reference, not automatically properties of the core.

**Local SHD priority:** E118 establishes fit-set learning and a positive D8/D1
development comparison in the new winner-only carrier family. E119 removes
redundant event-memory work while preserving the audited predictions and credit.
Use that saving to increase the training/data budget and measure speaker transfer;
inspect temporal eligibility before changing delays. Measure causal packet
coalescing as an accuracy/work/latency tradeoff. The official test and measured
joules comparisons remain separate milestones. Compare optionality using
independent adaptation and evaluation to expose gradient-noise bonuses. Preserve
explicit search, replay, event, and memory budgets. See the concrete
[SHD frontier protocol](SHD_FRONTIER_PROTOCOL.md) and
[THEORY §§173–175](theory/16_event_work_and_temporal_credit.md).

**AWS ownership:** the sibling runs non-SHD benchmarks and heavier controls.
Do not duplicate those queues here. Run every local job through
`experiments/queue/run_safe.sh`, one workload at a time, with at least 8 GiB
host memory available and RSS/timeout guards enabled.

---

# Decision, 2026-09-26: the project's identity is local, sparse, error-gated learning

**Goal:** learning whose cost follows events and errors (E4, E5, M18), decisions that take as long as the
evidence needs (E2), and credit through what did not happen (THEORY P4, §21, §35). Not "match dense accuracy
on dense benchmarks by training spiking networks with backprop"; that is an established field in which we
would only be catching up.

**Consequences.**
- **Exact-gradient training (E20, E21) is a diagnostic ceiling, not the method.** It showed the race
  architecture itself is close to dense (0.9675 vs MLP 0.976 at depth 2, matched budget, no cancellation), so
  the remaining gap is mostly the local learning rule and cancellation.
- **Every result reports its energy side:** inference synaptic events and spikes per input, and training
  weight updates, next to accuracy. Mechanisms are judged on the accuracy-vs-energy frontier against dense
  baselines. A mechanism that costs events must buy accuracy (the residual stream did not: 2–4× events, no
  gain).
- **Locality audit for every mechanism:** per node (fine), per-layer broadcast (acceptable if slow and
  cheap), global backward pass (diagnostic only).
- **Benchmarks where asynchrony is native:** the Spiking Heidelberg Digits (E22) first, then event-camera data
  and E7's continual-learning streams, where sparse error-gated updates should have an edge.

# Roadmap after E6 — play to the strengths

Written 2026-09-25. The experiment designs this refers to are E7–E11 in this
directory; each one states its predictions before it runs.

## What the evidence says so far

| Holds up | Evidence |
|---|---|
| A cancelled node can be taught; credit reaches nodes that never fired | E4: 0.79 at K = 128 where reward-modulated rules are at chance |
| Decisions take as long as the evidence needs | E2: tracks the optimal MSPRT with additions and a threshold |
| Learning work tracks errors, not size | E5: ~300× fewer weight updates than a sparse softmax on the same connectivity |
| Sleeping capacity is nearly free | E5: work flat while capacity grows 64× |

| Does not (yet) | Evidence |
|---|---|
| Hidden layers are expensive | E6 r3: ~108k synaptic events per image, ~3.6× the energy of an equally accurate 32-unit MLP |
| Counterfactual hidden credit adds nothing over fired-only credit | E6 r3: 0.959 vs 0.961 (one seed) |
| Accuracy trails backprop | E6 r3: 0.96 vs 0.978 for a dense MLP |

The losses are in the parts that do dense work (every hidden node integrates
every input until its group decides) and in the fight we picked (IID MNIST, where
dense batched hardware is at home). The wins are about **work that follows
events and errors**, **capacity that sleeps**, and **time as a resource**.

## Strategy

1. **Remove the dense parts** (E9): wake only a few hidden groups per input, give
   hidden nodes sparse grown fan-in, and let the output decision cancel all pending
   work below it.
2. **Fight on our ground**: streams, continual learning, huge label spaces, batch 1,
   event sensors (E7, E10, E12).
3. **Make the differences from classical SNNs measurable** (E8): no clock,
   cancellation, near-miss credit, tiny learning state.
4. **Close the accuracy gap locally** (E11): exact spike-time credit, which the
   closed form gives almost for free, and learned delays.

## Experiments

| | Question | Teaches us / shows | Cost |
|---|---|---|---|
| **E7** | Can the race learn from a causal stream with scarce, late, self-requested labels? | the stream regime; queued | small |
| **E8** | What exactly do we gain over clocked SNNs, BPTT, e-prop and STDP? | supremacy claims that can fail | small–medium |
| **E9** | Can hidden layers be made cheaper than an equally accurate dense model? | **viability gate for depth** | small |
| **E10** | Does recruiting sleeping nodes and consolidating in sleep beat replay at forgetting? | continual-learning claim; the project's name | small |
| **E11** | Does exact spike-time credit make hidden learning beat the fired-only ablation? | **gate for local deep learning** | small |
| **E14** | Does counterfactual credit dominate with depth? (depth 1–3 × credit type) | whether depth is where the idea pays | small |
| **E13** | Races as policies: bandit feedback, reward rate, value as latency | whether RL is the natural home | small |
| E12 | Extreme classification (10⁴–10⁶ labels) against sparse softmax and hashing | the most industrially relevant claim | medium; needs a dataset and a memory check |

## Decision gates

- **After E9.** If no hidden configuration reaches dense ÷ event ≥ 1 at inference
  at matched accuracy (≥ 0.95), we stop claiming energy for deep race networks.
  Energy claims then rest on single layers, huge K (E5, E12) and learning cost.
- **After E11.** If neither Δ nor timing credit beats fired-only hidden credit
  (3 seeds, CI excluding zero), the hidden credit signal is not where accuracy
  comes from. We say so, and look for depth elsewhere (width, recruitment, routing).
- **After E10.** If recruitment + sleep does not beat an equal-memory replay buffer
  on forgetting, we drop the continual-learning superiority claim and keep the
  efficiency claim (the same forgetting at a fraction of the updates), if that holds.

## Order

E7 pilots (queued) → **theory checks M1–M3** (THEORY.md; small networks, cheap) →
E9 → E8 (time resolution, learning state, dead neurons) → E10 (with the M8 recruitment
switch) → E11 (terms chosen by M3) → E8 (accuracy per total work) → E12.

THEORY.md sets out what the learning rules are gradients *of*, which parts are known
(EventProp, perturbed argmax, surrogate gradients), and what is new: the residue
boundary term, boundaries shared by two competing nodes, shadow continuation, work
as a loss, and recruitment as the fallback when no boundary is within reach.
Every job now runs through `experiments/queue/run_safe.sh`, one at a time.
