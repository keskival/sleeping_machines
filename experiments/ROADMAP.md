# Current frontier priorities — 29 September 2026

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
