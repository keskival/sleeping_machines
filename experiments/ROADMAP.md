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
Every job runs through `experiments/run_queue.sh`, one at a time.
