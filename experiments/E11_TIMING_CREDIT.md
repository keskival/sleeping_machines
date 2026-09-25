# E11 — Exact spike-time credit and learned delays

Written 2026-09-25, before any E11 code.

## Question

With current-based (ramp) synapses, a node's potential after the inputs that have
arrived is v(t) = A·t − B, with A = Σ wᵢ and B = Σ wᵢ tᵢ over those inputs. It fires at

    t* = (θ + B) / A,   so   ∂t*/∂wᵢ = (tᵢ − t*) / A,   ∂t*/∂dᵢ = wᵢ / A

for an input that arrived before t* (dᵢ is a synaptic delay). Every quantity is local
to the node: its own firing time, its own slope, the arrival time of the input. This is
the exact gradient of *when* the node fires, and the closed form gives it for free.

Δ (distance to threshold) says how close a node came; ∂t*/∂w says how to make it fire
earlier or later. E6 round 3 found that Δ credit to hidden nodes adds nothing over
fired-only credit. Does timing credit?

## Update after THEORY.md

The round-3 rule is already the pathwise time gradient up to a per-node 1/A
(THEORY §3), and the exact pathwise part is EventProp. So E11 does not test
"timing credit" as such. It tests the terms THEORY.md identifies: 1/A and
extrapolated crossings (M1); the boundary term on the fired side and the rival's
side (§4, M6); a better local estimate of the jump than random feedback (M3). The
M3 decomposition on small networks runs first and decides which variants below
are worth a full run.

## Variants

- **Output layer.** Teach the target to fire earlier and near-miss competitors later,
  by ∂t*/∂w (for nodes that did not fire, use their extrapolated crossing time; they
  are the counterfactual). Compare with the E6 Δ rule.
- **Hidden layer.** The output error in time, sent back by fixed random feedback (as in
  crl_fa) and applied through each hidden node's ∂t*/∂w. Local except for the
  feedback, which is the same event as today.
- **Delays.** Learn dᵢ by ∂t*/∂dᵢ, bounded, as the timing counterfactual in the
  original design ("had this spike come earlier").
- **Ceiling.** Exact backprop through spike times (non-local: uses downstream
  weights). This is also E8's TTFS baseline.

## Predictions

- **P1.** Hidden timing credit beats fired-only hidden credit (3 seeds, CI excludes
  zero). This is the gate in ROADMAP.md.
- **P2.** Local timing credit gets within 1.5 points of the non-local exact-backprop
  ceiling on the same network.
- **P3.** Learned delays add accuracy on a task where timing carries the information
  (XOR-in-time; later SHD). On latency-coded MNIST they may not, and that is informative.

## What we learn either way

If P1 fails with both Δ and timing credit, a hidden layer here learns mostly through
its own competition and homeostasis (as fired-only suggests), and deeper credit is not
where accuracy comes from. If P1 holds, E9's cheap hidden layer and E11's credit
together are the local deep race network worth taking to event data.
