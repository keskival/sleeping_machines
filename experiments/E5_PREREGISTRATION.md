# E5 — Does work track activity rather than capacity?

Written 2026-09-24, before any E5 code was run.

## Claim under test

In an event race, dynamic work (synaptic events at inference, weight updates at
learning) depends on how many events are caused, not on how many nodes exist.
Adding dormant capacity should cost (almost) nothing per input.

## The honest framing

Most of that saving comes from **sparsity**, which is not unique to Sleeping
Machines: a sparse, masked softmax on sparse inputs also touches only the
synapses of active inputs. The fair comparison is therefore three-way on the
*same* sparse connectivity. The race's own contribution over sparse softmax is:

- **early termination:** input events after the decision are never processed;
- **selective learning:** only near-miss nodes are updated;
- **no normalisation over all K:** no partition function.

## Setup

- K classes, K ∈ {256, 1024, 4096, 16384}. M = max(256, K/4) input channels,
  so the fan-out per channel stays constant as K grows. This models a
  representation that grows with the world it represents.
- Each class has a 16-channel prototype. A sample activates each prototype
  channel with probability 0.7, plus 10 random distractor channels, each spiking
  once at a random time.
- Fixed sparse connectivity shared by all models: each class node receives
  F = 64 synapses, its 16 prototype channels plus 48 random channels. This is an
  assumption: the structure is given, and only the weights are learned.
- Training: 8·K teaching events (about 8 per class). Test: 2,000 samples.

Contenders:

| model | inference work | learning work |
|---|---|---|
| dense softmax | K × active inputs (counted, not trained beyond K = 4096) | K × active inputs |
| sparse softmax (masked) | synapses of active inputs | synapses of active inputs on nodes with non-zero gradient |
| race + `cf_margin` | synapses of inputs that arrive **before the decision** | synapses of **near-miss** nodes only |

## Predictions

P1. Race and sparse-softmax work per sample stay within a factor of 2 across
    K = 256 → 16384. Dense work grows about 64×.
P2. The race does less inference work than sparse softmax, by the fraction of
    input events that arrive after the decision (expected 1.5–3×).
P3. The race does at least 3× less learning work than sparse softmax at every K.
P4. Race accuracy stays within 10 points of sparse softmax at every K.
    *Falsified if* the gap exceeds 10 points at any K.

## What would count against the thesis

If the race's work is not below sparse softmax's (P2, P3), then the energy case
for Sleeping Machines reduces to "sparsity is cheap", which is not specific to it.

## Amendment 1 (2026-09-24, before any run)

The connectivity described above (each node pre-wired to its 16 prototype
channels) would partly encode the answer in the structure. Instead, all models
start from **random** fan-in (F = 64 random channels per node) and share one
local structural rule: when a teaching event names the target, the target
connects to the active inputs it lacks, replacing its weakest synapses.
Rewiring counts as learning work. Dense softmax work is counted analytically
(K × active inputs), not trained.

## Round 2 (2026-09-24, exploratory; after the preregistered evaluation)

A diagnosis (`e5_diagnose.py`) showed the race's accuracy gap was entirely in
its readout: the race-trained weights read out by argmax scored 0.953 at K =
1024 (sparse softmax 0.980), while the race itself scored 0.812, with 7.5% of
inputs timing out undecided. Round 2 adds a collapsing bound: at the end of the
input window the threshold collapses and the leading node fires. A decision
forced this way still counts as uncertain and triggers learning, like a close
call. β was re-chosen on tuning seed 100 (0.15; 0.892 vs 0.890 at 0.1).
A leader-margin stopping rule (fire when ahead of the runner-up by a margin)
was also tried on the tuning seed and did worse (0.59-0.68) at the settings
tuned for the threshold race; it was not pursued. Round 2 is not
preregistered and is reported as exploratory.
