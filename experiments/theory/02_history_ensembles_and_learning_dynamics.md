# History ensembles and induced learning calculus

[Theory index](../THEORY.md) · Global sections 21–25; section numbers remain stable. · Next: [02b pivotal credit and learning dynamics](02b_pivotal_credit_and_learning_dynamics.md)

## 21. The unravelled pool: a dequantized tropical computation over a forest of histories

### 21.1 The object

A history is a sequence of events (node, time). At each collapse the pool forks: any pending
strand could have been the one to collapse. Branches share prefixes and **recombine** when a
branch's difference stops changing anything downstream (its light cone closes; M20). The full
unravelling is therefore a *packed forest of histories*: a directed hypergraph whose nodes are
events and whose hyperedges are collapses. It is not a tree.

### 21.2 The race is tropical

The primitive operation is first arrival, a minimum over projected times, while along a path
times add. So min plays "addition" and + plays "multiplication": the **min-plus (tropical)
semiring**. The deterministic forward pass of a race network is a tropical computation on the
event forest.

### 21.3 Temperature is dequantization

Replace min(a, b) by a ⊕_σ b = −σ log(e^{−a/σ} + e^{−b/σ}). As σ → 0 this is min again. In
idempotent analysis this family is the **Litvinov–Maslov dequantization**, the formal bridge
between sums over paths and their classical (tropical) limit. So the "wavefunction" of the
unravelled pool has a precise counterpart: at σ = 0, the machine (one history); at σ > 0,
every history weighted by e^{−cost/σ}. (A probability-like weighting, not complex amplitudes,
but the same structure: a sum over paths and its classical limit.)

### 21.4 Local form: Plackett–Luce races

With Gumbel noise on crossing times, which k members of a group win, and in what order, follows
the Plackett–Luce distribution with weights e^{−τ/σ}. The pool at temperature σ factorises into
a **layered product of Plackett–Luce races**, each conditioned on upstream spike times.

### 21.5 Forward weaving: inside values

Each event's soft arrival time is the ⊕_σ-sum over the paths that could produce it. Expanding in
the number of flips around the realised path gives a hierarchy: order 0 is the realised history
(the classical path); order 1 is single flips, exactly the **shadow spikes** of §20 and the
branches of M20; higher orders are interacting flips, which our rules ignore (§6).

### 21.6 Backward weaving: outside values

The derivative of ⊕_σ is a softmax: ∂(a ⊕_σ b)/∂a = e^{−a/σ}/(e^{−a/σ} + e^{−b/σ}). So credit
splits among competing paths in proportion to e^{−Δ/σ}: **the near-miss weighting is the
derivative of dequantized addition**, not a heuristic. Along the realised path the adjoint is
EventProp's time sensitivity; at each fork it is the branch's loss difference times the fork
probability's sensitivity (§14). Together this is the **inside–outside algorithm** run on the
network's own forest of event histories.

### 21.7 Inference and learning at different temperatures

Inference runs in the tropical limit: hard, sparse, cheap. Learning needs a neighbourhood of it
(σ > 0), because the tropical map is piecewise constant. Residues and shadow spikes are the
cheapest, first-order view of that neighbourhood; annealing σ is continuation from the
dequantized semiring down to the tropical one.

### 21.8 Credits and what is new

Known: tropical views of ReLU networks (tropical rational maps), Maslov dequantization and
idempotent analysis, Plackett–Luce, inside–outside, piecewise-deterministic Markov processes.
**Closest:** UltraLIF (arXiv 2602.11206) uses ultradiscretization and max-plus algebra for
spiking neurons, with log-sum-exp at a learnable temperature becoming hard thresholding as it
goes to zero: dequantization of *neuron dynamics* to make them differentiable. Here what is
dequantized is the race *between events*, over a recombining forest of histories. Not found: race inference as exactly tropical computation on events, learning as its
dequantization over a *recombining forest of event histories*, and shadow spikes as the
first-order term.

### 21.9 Weaving closes the past

The pool is always conditional on the inputs so far: each strand's projected time assumes no
further input, and future inputs will revise it. A collapse (weaving) is the operator that closes
the past: once an event is woven at t_c, nothing arriving later can influence it.

- **Collapse times are stopping times.** The decision to collapse at t_c uses only inputs up to
  t_c (a stopping time in the input filtration). Weaving turns part of the open pool into fixed
  history; the pool is the law of histories given the inputs so far and a prior over the rest.
- **The gradient respects it.** ∂T/∂w_i is nonzero only for inputs that arrived before the
  crossing; denying new inputs to the past is why timing gradients ignore later inputs.
- **Two kinds of counterfactual.** *Woven (truncated):* the residue Δ frozen at the collapse,
  seeing only evidence up to t_c. *Unwoven (continued):* what would have happened had the
  collapse not occurred, still receiving future inputs, which is the shadow channel of §20.
  This plausibly explains why the shadow neuron did best with a long window (0.4): it includes
  the evidence that weaving denied (M7 measures the trade-off).
- **When to weave is optimal stopping.** Collapsing early saves time and work but forgoes future
  evidence; E2's race tracks the optimal stopping rule (MSPRT). Thresholds are learnt stopping
  rules (§7, E13b); a world model (E13d) could supply the prior over future inputs that decides
  whether waiting is worth it.

### 21.10 Layers: one layer's outputs are the next layer's unknown future

For every layer but the first, the "future inputs" of §21.9 are the upstream layer's pending
strands, not yet woven themselves. The pool is therefore nested: layer l's pool is conditional on
layer l−1's weaving, which has not happened yet.

1. **Weaving spreads as a front.** A layer weaves only on upstream events already woven; the
   determined region grows through the network as a causal front, and the output decision is a
   stopping time composed through the layers.
2. **The network carries its own forecast.** A hidden layer's prior over inputs still to come
   is available locally: the upstream pool's projected times. Only the first layer faces an
   external unknown, so "is waiting worth it?" can be answered inside the network except at
   the input.
3. **Speculation across layers.** A downstream node can weave early on near-certain upstream
   strands (large lead, steep slope) before they fire, and roll back if they do not (§10.4),
   overlapping layers instead of paying depth × weaving delay.
4. **Unwoven uncertainty propagates as shadow spikes.** An upstream shadow spike is an input
   that might have arrived; feeding it into the downstream shadow compartment (as M23 does)
   propagates the upstream layer's unwoven futures through the downstream one. The two-channel
   neuron composes across layers into exactly this conditional structure.
5. **The backward pass respects weaving order.** A downstream event's adjoint reaches only
   upstream events woven before it through the factual channel; the "might have been" part
   returns through the shadow channel. Recombination happens where different upstream branches
   lead to the same downstream woven event.

### 21.11 Tests

- **M24a.** The near-miss weights used by the rules equal ⊕_σ derivatives on real samples (up
  to the first-order truncation).
- **M24b.** Inside–outside on the M20 beam reproduces M19's finite-difference gradient of the
  soft (σ > 0) objective, better as the beam grows.
- **M24c.** Annealed σ (dequantization continuation) trains at least as well as a fixed σ at depth.
## 22. Induced consequences

### 22.1 A race neuron computes a weighted mean of its input times

Within one piece (fixed set of arrived inputs), a ramp neuron fires at
T = θ/A + Σ_i (w_i/A) t_i with Σ w_i/A = 1: a weighted mean of input times plus an offset, and
the race then takes a minimum. The network alternates means and minimums: a *mean–min* network,
the timing analogue of max-affine (ReLU) networks.

- **Time-shift equivariance (exact).** Delay every input by c and every firing time moves by c;
  decisions are unchanged. Checked: shifts of 0.05 and 0.2 left all 300 test decisions and all
  hidden firing sets unchanged, firing times shifted by c to within 2·10⁻⁷ (the deadline is the
  only thing that breaks it).
- **Magnitude is urgency, direction is evidence.** Scaling a node's weights by α changes only
  θ/(αA): the norm sets how early a node tends to fire, the direction sets which evidence it
  averages. Checked: ×1.5 weights fire 0.084 earlier on average. Homeostasis acts on urgency,
  credit on direction; learning rules could treat them separately.

### 22.2 Is the network linear?

Piecewise linear in input times, like a ReLU network is piecewise linear in its inputs (and
nonlinear in the weights through w_i/A). The nonlinearities are built in: the minimum (first to
fire), k-winner cancellation (existence), causal truncation (inputs after the crossing are
ignored), and fire-or-not within the window. With only non-negative weights each firing time is a
convex average, so every output time is monotone in every input time; but the *decision* compares
outputs, and comparisons of monotone functions carve non-monotone regions. Checked: clamping all
weights non-negative costs almost nothing (depth 1: 0.848 vs 0.852; depth 3: 0.745 vs 0.746,
debug size). Excitation-only race networks suffice here: unsigned weights, Dale-compatible.

### 22.3 Credit is conserved at every collapse

∂(⊕_σ)/∂ is a softmax, so the credit a collapse sends to its competitors sums to 1: credit flows
through the forest like a current, split at races, neither created nor destroyed. The near-miss
rule as used violates this (competitor credit unnormalised). **Prediction confirmed (debug,
depth 3):** normalising competitor credit to −1 raises residue weighting 0.649 → 0.746 and the
shadow neuron 0.698 → 0.782, the largest single gain at depth so far.

### 22.4 The pool's revisions are a free learning signal

Under a correct model of future inputs, the pool's projected outcome probabilities form a
martingale during one inference. Systematic drift means miscalibration, so the difference between
the pool's prediction now and after more input is a label-free temporal-difference error inside a
trial. Training on it should teach the network to anticipate its own collapse: earlier decisions
at the same accuracy. (Derived form of §10.5.) **(test M25)**

### 22.5 Timing noise accumulates with depth

Jitter propagates and adds across layers, so the effective temperature at the output grows with
depth, while percolation (§17) wants σ large enough for credit to reach deep layers. That implies
a per-layer σ schedule rather than one global σ. **(test M26)**
## 23. Tools for aggregate behaviour (from statistical physics)

The system is classical; the fitting toolkit is the Wick-rotated side of quantum mechanics:
statistical mechanics and Euclidean path integrals, histories weighted by e^{−cost/σ} (Maslov
dequantization is the ħ → 0 limit of this picture).

1. **Laplace (semiclassical) expansion.** As σ → 0 the sum over histories is dominated by the
   minimal path, corrections by nearby paths: our hierarchy (realised path, single flips,
   interacting flips), with an error estimate for truncating at first order.
2. **Fluctuation–dissipation.** The response of an average to a parameter equals a covariance:
   ∇E[L] = Cov(L, ∂cost/∂w)/σ. The substrate's own timing noise can estimate gradients by
   observation: noise as a resource, suited to asynchronous analogue hardware. **(test M27)**
3. **Mean-field and cavity methods.** Macroscopic quantities of large random race networks
   (firing fractions, credit reach, the percolation threshold of §17, capacity); §17's recursion
   is already a mean-field equation.
4. **Renormalisation.** How the effective temperature flows from layer to layer (§22.5): a
   coarse-graining flow that would give the per-layer σ schedule.
## 24. Simplex coordinates: a true description, a wrong learning geometry

Write a node's non-negative weights as w = ρ·u, with urgency ρ = Σw and evidence mix u = w/ρ on
the simplex. Within a piece, T = θ/ρ + Σ u_i t_i: the neuron is exactly linear in u, and ρ acts
only through θ/ρ. This explains why non-negative networks lose nothing (§22.2): the simplex view
is exact for them.

It suggested mirror descent on the simplex (multiplicative, exponentiated-gradient updates of u,
additive updates of ρ). **Tested (M28, debug): it fails**, near chance at depth 1 and 3 for scales
1–20 (best 0.31 vs 0.85 additive). Diagnosis: multiplicative updates cannot grow a weight that is
near zero, and never one that is zero (~16% of initial weights are clamped to zero); but learning
must *recruit* evidence that currently contributes little: an output node must come to listen to
hidden nodes it barely hears. The additive rule places its update on the synapses whose inputs
actually arrived, whatever their current size, which is recruitment.

*Lesson:* the simplex is the right *description* and the wrong *learning geometry* wherever
the needed evidence is absent. Absent evidence is recruited additively or structurally (synapse
growth, E9; structure first, §11.3). A hybrid (multiplicative refinement of present evidence,
additive recruitment of absent evidence) is possible but not the bottleneck now.
## 25. Races are auctions: thresholds as prices, a layer as entropic optimal transport

Homeostasis has been an add-on. It follows from the same formalism.

1. **Thresholds are Lagrange multipliers.** Homeostasis enforces a constraint: each node wins
   with the same long-run frequency (k per group). Minimising loss under that constraint, the
   dual update is "raise the multiplier of an over-used node": θ ← θ + η(rate − target), exactly
   the homeostatic rule. Thresholds are dual variables; the target rate is a node's capacity.
2. **A race is an auction.** Nodes bid by arrival time, the earliest wins, and thresholds act as
   prices that rise on over-demanded nodes until demand balances: Bertsekas' auction algorithm
   for assignment.
3. **At σ > 0 it is entropic optimal transport.** Assigning inputs to nodes with capacities,
   minimising total crossing time, is optimal transport; its dequantized version is entropic OT,
   solved by Sinkhorn, with thresholds as dual potentials. A k-winner race layer with homeostasis
   performs an online, asynchronous, time-coded Sinkhorn. (Credits: Sinkhorn/balanced routing in
   MoE, e.g. BASE layers; the "conscience" mechanism of competitive learning.)

**Tested (M29, debug, depth 3):** Sinkhorn's log-ratio dual step θ ← θ + η log(rate/target)
balances usage better than the linear rule, as predicted (normalised usage entropy up to 0.98 vs
0.89–0.93), but accuracy falls as balance is enforced harder (0.741 / 0.612 / 0.562 at rates
0.001 / 0.005 / 0.02, vs 0.746 linear). *Lesson:* the structure is right (thresholds are prices on
a capacity constraint), but equal capacities are the wrong target: some nodes should win more
because they carry more useful evidence (as strict load balancing costs quality in MoE). The
capacities, the OT marginals, should be learnt, e.g. by maximising the information the code
carries about the label under an activity budget (a rate–distortion view of the hidden layer).
