# Race foundations and exact gradients

[Theory index](../THEORY.md) · Global sections 1–10; section numbers remain stable. · Next: [01b counterfactual credit and depth](01b_counterfactual_credit_and_depth.md)

## 1. The state is a pool of possible futures

Take ramp (current-based) synapses. After the inputs that have reached node n by
time t, its potential is linear in t:

    v_n(t) = A_n t − B_n,   A_n = Σ_i w_ni,   B_n = Σ_i w_ni t_i

so, if nothing else arrived, it would cross threshold at

    τ_n(t) = (θ + B_n) / A_n.

Call the pair (n, τ_n) a **strand**: a future that is pending, not yet real.
The machine's internal state is the pool of strands. Every input event *revises*
strands (moves τ earlier or later). A **collapse** is a first passage: the
earliest strand becomes a fact (a spike), and inhibition cancels the strands it
competes with. A forward pass is a sequence of collapses; the realised event
history is one path through a tree of possible histories.

What a collapse keeps of each cancelled strand:

- its **residue** Δ_n = (θ − v_n(t_c))/θ, the distance to threshold at the collapse time t_c;
- implicitly, its **charges** q_ni = t_c − t_i: how long each arrived input had been injecting current.

§4 shows these are exactly what a first-order gradient through the collapse needs.

The system is classical and deterministic. The "collapse" language is an analogy
about structure (many pending futures, one realised, the others leave a trace),
not a claim about quantum mechanics.
## 2. The map from weights to outcome is piecewise smooth

Fix the input. Divide weight space into **cells**: in each cell the combinatorial
history is fixed (which strands collapse, in which order, which are cancelled,
which inputs each node integrated). Inside a cell every event time is a smooth
(rational) function of the weights: τ = (θ + B)/A with A, B linear in the weights
and in upstream spike times, composed layer by layer.

Cells are separated by **branch surfaces** of three kinds:

| boundary | what changes | outcome time | winner |
|---|---|---|---|
| (a) **winner exchange**: T_a = T_b in one race | which strand collapses first | min T is **continuous** (a kink) | jumps |
| (b) **existence flip**: a hidden node goes from fired to cancelled, or back | a downstream input event appears or vanishes | **jumps** by a finite amount | may jump |
| (c) **piece change**: a crossing moves across an input arrival | which linear piece of v applies | continuous; derivative jumps | unchanged |

Type (c) is harmless: the time is continuous and piecewise smooth. Type (a) is
harmless for any loss defined on **times** rather than on the discrete winner. The
min is continuous, and a time margin such as T_y − min_{n≠y} T_n has a subgradient.
Type (b) is the real obstacle: the forward map is discontinuous there, so no
gradient exists, however the loss is written.
## 3. Within a cell: gradients flow along the realised history

Inside a cell the exact gradient exists and is local at every step. By the
implicit-function theorem at a crossing (v(T) = θ):

    ∂T/∂w_i = −(T − t_i)/A,     ∂T/∂t_i = w_i/A,     ∂T/∂d_i = w_i/A   (d_i: delay)

The chain rule runs backwards **along realised spikes only**. An output time
depends on a hidden spike time through ∂T_o/∂t_h = w_oh/A_o, and that hidden time
depends on its weights through ∂T_h/∂w. This is EventProp in our setting: its memory
and work scale with the number of realised spikes, not with neurons × time steps as
in backpropagation through time. A known limitation of EventProp is that it gives no
gradient for spikes appearing or disappearing. That is exactly the boundary term of §4. Cancelled strands contribute nothing here,
because they are not on the path.

So **yes, gradient flows through a collapse in the pathwise sense**: through the
winner's time into everything downstream of it. Through the cancelled strands it does
not, not in this term.

Relation to what we run: E6 round 3 gives each synapse the eligibility "charge
injected before the node froze", (t_freeze − t_i). For a node that fired, that is
−A·∂T/∂w_i. **The round-3 rule is already the pathwise time gradient**, up to a
per-node 1/A and the upstream factor (which feedback alignment replaces with a fixed
random matrix). **(test M1)**
## 4. Across a flip: the boundary term

A discontinuous map has no gradient, but its **expected** loss under noise does.
Jitter every crossing time by independent noise of scale σ. Then E_ε[L(w, ε)] is
smooth, and its gradient splits into two terms (the standard result for expectations
of piecewise-smooth functions: differentiate inside the cells, then add a flux across
the moving boundaries):

    ∇_w E[L] = E[ ∇_w L within the cell ]                       (pathwise, §3)
             + Σ_boundaries E[ ρ(distance) · ⟦L⟧ · ∇_w(distance) ]   (boundary)

where, for each branch surface near the realised history:

- ρ(distance) is the noise density at the history's distance to that surface:
  **how likely the other side was**;
- ⟦L⟧ is the jump in loss when crossing it: **what the other side would have cost**;
- ∇_w(distance) is how the weights move the surface: **which way to push**.

For an existence flip of a cancelled hidden node h, all three pieces are available at
the collapse:

- distance = its residue **Δ_h**, so ρ = ρ_σ(Δ_h) (≈ exp(−Δ_h/σ) for exponential tails);
- ∇_w Δ_h = −(t_c − t_i)/θ, its **charges**;
- ⟦L⟧ is the effect of its spike, had it happened at τ_h. That effect is downstream and
  finite: the output crossing times shift by about −w_oh(T_o − τ_h)/A_o, which changes
  the loss.

**(Δ, charges) is exactly the record needed for the first-order boundary term**, and
the only non-local ingredient is the jump ⟦L⟧. crl_fa estimates the jump with fixed
random feedback. E6's hidden eligibility exp(−Δ/σ) × charge is ρ × ∇distance.
So, read through this formula:

| rule | pathwise term | boundary term, cancelled side | boundary term, fired side |
|---|---|---|---|
| fired-only | yes (FA-approximated) | no | no |
| crl_fa (E6) | yes (FA) | yes: ρ(Δ) · charges · FA estimate of ⟦L⟧ | no (fired nodes use weight 1) |
| surrogate-gradient SNN | smoothed at every step | smoothed, for every neuron at every step | same |
| exact (target) | backprop through events | ρ(Δ) · charges · true ⟦L⟧ | ρ(margin) · charges · true ⟦L⟧ |

Three things follow.

1. **Fired nodes also sit near boundaries**: a node that fired only just before its
   group's cancel time could flip off. Its distance is its **fire margin** (cancel time
   minus fire time, times A). crl_fa gives it no boundary term.
2. **E6 round 3, fired-only ≈ crl_fa, is now a precise question**: is the boundary
   term small on MNIST, or is its FA estimate of ⟦L⟧ too noisy to help? **(test M3)**
3. **Surrogate gradients are this boundary term, smoothed and dense.** A surrogate
   derivative is the derivative of an expected spike under threshold noise, i.e. ρ at
   the neuron's current distance, applied at every neuron and every step. We evaluate
   the same ρ once, at the collapse, for strands that came close. **Same mathematics,
   at events instead of clock ticks**, which is the precise form of E8's claim.
## 5. The race is a softmax at zero temperature

With Gumbel noise of scale σ on crossing times, the probability that strand y
collapses first is the Gumbel-max identity (Luce's choice rule):

    P(y first) = exp(−T_y/σ) / Σ_n exp(−T_n/σ).

Taking L = −log P(y first):

    ∂L/∂T_n = (1/σ)(1[n=y] − p_n)

This pushes the target earlier and each competitor later, in proportion to its
probability of having won: the **near-miss weighting of E4/E6 is the gradient of a
cross-entropy over crossing times**, and σ is a temperature. For cancelled output
strands, T_n is not observed. The rule uses the frozen residue instead: the
*truncated* counterfactual, which uses only the evidence up to the collapse. The
*full* counterfactual would require integrating the inputs after the collapse, which
is exactly the work cancellation saved. The trade-off between those two is itself
measurable. **(test M2)**
## 6. First order only

The residues describe single flips. Two cancelled hidden nodes that both nearly
fired, and would change the output only together, are a second-order term that no
stored residue represents. Prediction: the rule degrades where near misses are dense
and interact. **(test M4)**
## 7. Time and work are differentiable objectives

Within a cell t_dec = T_winner, so ∂t_dec/∂w is the winner's pathwise gradient, and
the input events processed ≈ rate × t_dec. So

    L_total = L_race + λ_t · t_dec + λ_w · W

trains the speed–accuracy–work trade-off directly. A clocked network, whose cost is
fixed by its window, cannot express this. **(test M5)**
## 8. Streams

In a stream (E7), a teacher that arrives late credits earlier collapses through tags
that decay with age. A tag is an eligibility trace over **collapses**, not over clock
ticks, so its cost scales with the number of collapses.
## 9. What is new here

Exact event gradients handle the inside of a cell; surrogate gradients smooth
everything, densely. What this architecture adds is **cancellation**: a
competitor's collapse ends a strand, and the ending leaves a residue. Building on
that:

1. **Residues as sufficient statistics for the boundary term.** Claim: for
   existence flips caused by cancellation, (Δ, charges) at the collapse, plus an
   estimate of the downstream jump, are all a first-order boundary gradient needs.
   So the boundary term EventProp lacks can be added **sparsely, once per
   cancellation**, instead of densely as surrogates do. Operator: *EventProp +
   residue boundary term*. **(test M3)**

2. **Boundaries that belong to two nodes.** In a race with cancellation, whether
   strand h exists depends on h *and* on the rival whose collapse cancelled it: the
   boundary is T_h = t_c(rival). Its normal therefore has components in both nodes'
   weights (∇distance = ∇T_h − ∇t_c). The rival is pushed to fire later exactly as
   h is pushed earlier: a coupled, competitive update that plain SNN inhibition
   (a negative current) does not produce. Today's rules include only h's side.
   **(test M6)**

3. **The cost of the counterfactual is a design variable.** A residue frozen at the
   collapse is a *truncated* counterfactual: it saves exactly the work cancellation
   saves, at the price of bias. Integrating a strand for a while after it is
   cancelled (a "shadow" continuation, counted as work) trades work for gradient
   quality. This gives a measurable curve of gradient bias against extra work,
   and it probably has an optimum. **(test M7)**

4. **Work as a trainable objective, through collapse times.** Because cancellation
   ends work at collapse times, work is a piecewise-smooth function of the weights,
   so λ_w · W can be optimised directly (§7). Energy then becomes part of the loss,
   not something only reported. **(test M5)**

5. **When gradients cannot help, recruit.** The boundary term is weighted by
   ρ(Δ_min), the density at the nearest flip. When every residue is large, no
   first-order change can fix the error: the gradient vanishes because the
   nearest alternative history is too far away. That is the principled trigger for
   **recruitment** (E10): create a new strand rather than move an old one. Learning
   and growth become one operator with a switch: *move the nearest boundary if it is
   within reach, otherwise add a strand.* **(test M8)**

6. **Higher orders.** Pairs of near-flips interact (§6). Their count is observable
   from the residues, so the rule can tell when its own first-order expansion is
   unreliable, and fall back (e.g. to a smaller step, or to recruitment). **(test M4)**
## 10. Using the futures that never happen

The pool of pending strands is computed anyway: it is what the event scheduler needs.
At a collapse, every competing strand has a projected crossing time
τ_n = t_c + (θ − v_n)/A_n, obtained from quantities already present. Today we keep only
Δ. The pool is worth more than that.

1. **A full likelihood for the price of a race.** A cross-entropy loss normally
   requires every class's score. Here the pool gives every competitor a projected time
   at the moment of collapse, so the noisy-race likelihood of §5,
   −log softmax(−τ/σ)_y, is available **without running any loser to completion**.
   Its gradient is local (∂τ/∂w_i = (t_i − τ)/A). This replaces the hand-built
   near-miss rule with the gradient of a proper likelihood, with the same sparsity.
   **(test M9)**

2. **Anytime probabilistic output.** At any time t before the collapse, softmax(−τ(t)/σ)
   is a distribution over outcomes that sharpens as evidence arrives. If it is calibrated,
   the network gives a confidence with every decision, can be interrupted at any moment
   and still answer, and its "decide now" rule is a threshold on its own confidence.
   **(test M10)**

3. **Knowing what one doesn't know.** The shape of the pool is a diagnostic:
   - one strand far ahead: confident;
   - several close: ambiguous, so ask for a label (E7 "ask" uses a cruder version);
   - none near threshold: novel, so recruit (E10, M8).

   All three come from one object, instead of three separate heuristics.

4. **Speculation across layers.** A hidden strand that is almost certain to fire
   (large lead, steep slope) can send its spike downstream *before* it crosses:
   speculative execution, as in processors. If it is then cancelled, the downstream work is
   rolled back (counted). In deep event networks latency adds up layer by layer, and
   speculation trades a little rollback work for overlapping the layers. **(test M11)**

5. **Self-supervision from revisions.** Every input event revises strands. How far
   a strand moves when evidence arrives is a prediction error that needs no label: a
   local, unsupervised learning signal (predict your own crossing time better). It fits
   E7's continuity learning, and it can also say where to look next: sample the input
   that would move the leading strands most. **(test M12)**

6. **Exact expectations instead of sampling.** Stochastic spiking networks estimate
   expected gradients by sampling spikes. With the pool, the first-order expectation
   over alternatives is a sum over projected strands with known probabilities (a
   Rao-Blackwellised estimator): the boundary term of §4 is evaluated analytically, not
   sampled. That means lower-variance gradients at no extra forward work.
