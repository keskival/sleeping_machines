# E13 — Races as policies: reinforcement learning

Written 2026-09-25, before any E13 code. See THEORY.md §15.

## Question

Supervised labels say what was right. Rewards only say how good the chosen action was.
Races are action selections, residues grade the unchosen actions, and time is a cost the
race already manages. Is reinforcement learning where this architecture has its clearest
advantage?

## E13a — bandit feedback (cheapest bridge from our supervised results)

Contextual bandits: the E4 task (K = 3 … 128) and MNIST, where the learner sees only reward 1
if its chosen class is right and 0 otherwise, never the label.

- **Rules:** reward-modulated STDP (fired only); policy gradient from the pool (∇log softmax(−τ/σ),
  analytic, with a running reward baseline); near-miss credit with reward (the E4 rule with the
  teaching event replaced by reward × residue); softmax regression with REINFORCE (non-local
  reference).
- **Prediction:** the pool-based rules keep learning as K grows, where reward-modulated STDP
  drops to chance, which is E4's result carried over to evaluative feedback.
- **Fails if** pool policy gradient is no better than reward-modulated STDP at K = 32.

## E13b — reward rate with a cost for time

E2's Poisson evidence task, with reward per correct decision and time as the currency
(reward per second). Learn thresholds and weights to maximise reward rate.

- **Reference:** the reward-rate-optimal threshold of the drift-diffusion model, and MSPRT.
- **Prediction:** the race's learnt operating point approaches the optimal reward rate without
  being told the evidence statistics.

## E13c — value as latency

Gridworlds and shortest-path tasks. A race network in which each state's firing time encodes
its value (earlier is better) and actions race; the temporal-difference error is the gap
between predicted and realised time-to-reward.

- **Reference:** tabular TD / Q-learning; the exact min-plus solution.
- **Prediction:** convergence to the optimal policy, with work proportional to the states
  whose timing changed (event-driven value updates) rather than to all states.

## E13d — a world model trained by the world, and counterfactual actions through it

A world model is supervised learning whose labels the world supplies: the model races over
predicted next events, and the event that actually arrives is the teacher. All the supervised
machinery (near-miss credit, repair, tags, asking when unsure) applies, with no human labels;
the E7 stream learner is its training ground.

- **Counterfactual actions:** a shadow branch for an action not taken runs through the world
  model, which supplies the outcome the network could not otherwise know.
- **Sleep as planning:** replaying imagined branches through the model (Dyna-style) in E10's
  sleep phase consolidates model and policy together.
- **Test:** gridworlds first. Model-based races with shadow branches vs model-free E13a/c
  learners, at equal environment experience. Measures: experience needed to reach the optimal
  policy; work per decision including shadow branches.

## Protocol

Validation pilots, then 5 seeds. Small models, one job at a time through `run_queue.sh`.
E13a runs first; it reuses E4 and E6 code almost unchanged.
