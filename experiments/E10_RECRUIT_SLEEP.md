# E10 — Recruiting sleeping nodes, consolidating in sleep

Written 2026-09-25, before any E10 code. Builds on the E7 stream learner.

## Question

A network with sleeping capacity can do something a fixed dense network cannot do
cheaply: when something new appears, **wake a node for it** instead of overwriting
what other nodes know. And a system that is sometimes idle can use that time:
**sleep** to replay, consolidate and prune. Do these beat the standard continual-
learning remedies, and at what cost?

## Mechanisms

- **Recruitment.** Output nodes carry a class label; several nodes may share a class,
  and the prediction is the label of the race winner. A dormant node is recruited when
  a labelled frame shows *novelty*: the decision was forced by the deadline or wrong,
  **and** no node of the true class came within δ of threshold. The new node is imprinted
  from the frame (weights on its active inputs, scaled so it would just fire), in one
  step. Same for hidden groups under the E9 router: a group whose key nothing matches
  well is recruited. Pools are finite; when empty, the least-used node is recycled
  (counted, and reported).
- **Sleep.** Every N frames the learner sleeps for S steps with no input. It replays
  frames from a bounded store of *surprises* (frames that caused recruitment or a
  large error), consolidates (raises the consolidation strength of synapses that
  replay confirms), prunes synapses whose weight and use are both low, and rebalances
  thresholds. All sleep work is counted.

## Streams and baselines

E7 streams: class-blocked (forgetting), episodes with 10% labels, and a
**new-class** stream in which classes appear one by one and each new class is
labelled only a few times (few-shot). Baselines: E7 learner without recruitment or
sleep; MLP + SGD; MLP + reservoir replay with **the same memory** as the surprise
store; EWC (a standard regularisation method); and the control **"awake replay"**:
the same replay updates spread over the waking stream instead of in sleep.

## Measures

Forgetting (per-task peak minus final), final accuracy on all classes, accuracy on a
new class after 1, 5, 20 labels, nodes recruited over time, work per frame including
sleep, memory used by stores.

## Predictions

- **P1.** Recruitment cuts forgetting on class-blocked streams by at least half
  against the E7 learner, with no growth in work per frame (sleeping nodes cost nothing).
- **P2.** On the new-class stream, recruitment reaches useful accuracy (P) after
  1–5 labels, where SGD needs many more.
- **P3.** Sleep adds to recruitment: lower forgetting than awake replay with the same
  number of replay updates. *If this fails, sleep is just replay by another name*,
  and we report it that way.
- **P4.** Against an equal-memory replay buffer, recruitment + sleep matches or beats
  forgetting at a fraction of the weight updates. Losing on accuracy but winning on
  updates is still a result; losing on both closes the claim (ROADMAP gate).

## What could go wrong

Novelty that fires on every hard example (pools fill with duplicates): measured
by nodes recruited per class and the accuracy of recruited nodes. Imprinted nodes that
win too often: their initial scale is tuned on validation. Sleep that erases rather than
consolidates: measured directly by accuracy before and after each sleep.
