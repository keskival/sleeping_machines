# E9 — Removing the dense parts: routing, sparse fan-in, cascade

Written 2026-09-25, before any E9 code. Thresholds marked (P) are set from
validation pilots and dated before confirmatory runs; everything else is fixed now.

## Question

The round-3 hidden network (0.96 on MNIST) spends ~108k synaptic events per image,
about 3.6× the energy of an equally accurate 32-unit MLP. Almost all of it is hidden
integration: 1000 nodes each integrate ~100 input events. Can the architecture's own
mechanisms (races that wake few, cancellation of pending work, sparse grown
connectivity) remove that cost without losing accuracy?

This is a **viability gate**: if no configuration makes the hidden network cheaper
than an equally accurate dense model, depth is not an energy story for this architecture.

## Mechanisms (each behind a flag, each ablated)

1. **Cascade.** When the output race decides at t_dec, every pending event below it
   is cancelled: hidden nodes that have not yet fired stop integrating, and input
   events after t_dec are never delivered. At inference this cannot change the
   decision (hidden spikes after t_dec reach the output too late), so any saving is
   free. During learning, hidden near-miss records are taken at min(own freeze, t_dec).
   *Measured first, because it is free: how much of today's work happens after the decision?*

2. **Sparse, grown fan-in.** Each hidden node has F input synapses (F ∈ {16, 32, 64},
   against 784 today), initially drawn at random from inputs that are ever active. The
   E5 growth rule: when a node receives positive credit, it connects to active inputs
   it lacks, replacing its weakest synapses so F stays fixed. Rewiring is counted as work.

3. **Routing race.** Each hidden group has a small key (a sparse vector over inputs).
   Keys race on the first m input events; the first G_on groups (of G) wake and the
   rest are cancelled before they integrate anything. Woken groups integrate either
   (a) only events after they wake (strictly causal), or (b) also the m events already
   seen, from a short delay-line buffer (counted). Keys learn by the same local rule
   as the nodes: the key of a group containing the output-credited winner is taught
   toward the input. G ∈ {100, 400} groups, G_on ∈ {4, 10, 25}.

## Baselines

- E6 round-3 network (dense fan-in, no cascade, no routing), same training budget.
- Dense MLPs of 16–256 hidden units (existing energy table).
- A top-k mixture of experts with the same number of groups and the same number
  awake: softmax gate, backprop. This shows whether "first to fire" routing is
  as good as a learned gate, and what it costs.

## Measures

Accuracy (test; validation for tuning); synaptic events, spikes, rewiring and
plasticity per image, split by layer and by before/after t_dec; energy on the
existing profiles; dense ÷ event at matched accuracy against the cheapest equally
accurate dense model, at batch 1 and batched.

## Predictions

- **P1 (cascade).** A substantial fraction of hidden work happens after the output
  decision; cascade removes it with identical inference decisions (tested: 100%
  agreement on the test set). Expected saving 1.3–3×.
- **P2 (fan-in).** F = 32 with growth stays within 1 point of dense fan-in, with
  ≥ 10× fewer hidden synaptic events.
- **P3 (routing).** With G = 400 and G_on = 10, accuracy within 1 point of the
  unrouted network with the same total width, with work set by G_on, not G
  (the E5 result inside a hidden layer).
- **P4 (gate).** Cascade + fan-in + routing together reach dense ÷ event ≥ 1 at
  inference against the cheapest dense model at least as accurate (≥ 0.95), at batch 1.

## Falsification and what we learn

- P1 fails (little work after t_dec): the cost sits before the decision, so routing
  and fan-in are the only levers.
- P2 fails: hidden features need global receptive fields. Then patch-structured
  fan-in (the E6 `patch` option, grown) is the next test before giving up on sparsity.
- P3 fails because routing picks poorly: compare with an oracle router (groups that
  would have won with full integration) to separate "routing is hard" from "routed
  capacity is not enough".
- P4 fails: see the gate in ROADMAP.md.

## Protocol

Pilots on the validation split, one seed, 3 epochs at the round-3 settings.
Confirmatory: test set, 10 epochs, 3 seeds for the chosen configuration and each
single-mechanism ablation. Run through `run_queue.sh`, one at a time.

## Pilot finding, 2026-09-25 (before any confirmatory run)

**P1 is falsified.** Cascade counter on a network trained on 3k images (crl_fa, round-3
settings), evaluated on 1000 validation images: 112,614 hidden+output synaptic events
per image without cascade, 112,532 with it (0.07% saving). Hidden groups finish their
races before the output decides, because the output needs their spikes. The cost sits
*before* the decision, so fan-in (P2) and routing (P3) are the only levers. Cascade stays
in the design (it is free and correct) but is no longer expected to contribute.
