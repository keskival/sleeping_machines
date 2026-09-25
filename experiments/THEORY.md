# Theory note — collapsing futures, and the gradient through a collapse

Written 2026-09-25. A working note: every claim marked **(test Mk)** is checked
numerically (table at the end) before later experiments rely on it.

## 0. What we build on (and do not re-derive)

Much of the machinery below exists. We use it and cite it (references to verify
before publication):

| piece | existing work |
|---|---|
| exact gradients through spike times, adjoint at events | SpikeProp (Bohte et al. 2002); non-leaky TTFS closed forms (Mostafa 2017; Göltz et al. 2021); **EventProp** (Wunderlich & Pehle 2021) |
| a race of linear accumulators as a choice model with a likelihood | race models; the linear ballistic accumulator (Brown & Heathcote 2008) |
| noisy argmax = softmax (Gumbel-max, Luce's choice rule); gradients of perturbed argmax | Luce 1959; Papandreou & Yuille 2011; perturbed optimizers (Berthet et al. 2020) |
| gradients of expectations of discontinuous functions = interior + boundary flux | Reynolds transport; non-differentiable reparameterisation (Lee, Yu & Yang 2018) |
| surrogate gradients as derivatives of expected spiking under escape noise | Neftci, Mostafa & Zenke 2019; Gygax & Zenke 2024 |
| online eligibility traces | e-prop (Bellec et al. 2020) |

Sections 1–5 restate these in our setting only as far as needed. Section 9 is what
is ours.

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

## 11. Beyond gradients: learning by rescheduling

Sections 2–10 translate the race into the language of losses and gradients, which
makes claims checkable, but it also pulls every idea back toward existing methods. The race
has structure that gradient language does not see. Three departures follow.

### 11.1 Timing constraints are half-spaces

For ramp synapses, "node n has fired by time c" means v_n(c) ≥ θ, i.e.

    Σ_i w_ni (c − t_i) ≥ θ     (sum over inputs with t_i < c)

**This is linear in the weights.** For each deadline c and each input history, the
weights that make n fire in time form a half-space, with normal given by the
*charges* (c − t_i). Whether an event happens before another is therefore not a soft
quantity to be pushed by a loss. It is membership in a convex set.

So the teaching event "the answer should have been y, not b" becomes a pair of
constraints at a meeting time c between the two crossings:

    y:  Σ w_yi (c − t_i) ≥ θ + μ        (fire before c, with margin)
    b:  Σ w_bi (c − t_i) ≤ θ − μ        (not before c)

and the update is the **exact projection** of each node's weights onto its half-space:
the smallest change that reorders the two events, in closed form, touching only the
two nodes involved. There is no loss, no learning rate (the step is set by
the violation), and no approximation within the history's cell. The meeting time c is
a free choice: halfway between the two crossings, or at the decision time.

Consequences to test:

- **One-shot correction.** One projection fixes the order for that input, exactly.
- **A perceptron-style theory in time.** If some weights order every training example
  correctly with margin μ, projections onto these half-spaces converge in a bounded number
  of mistakes (the classical argument for projections onto separable half-spaces). That
  would be a convergence guarantee for learning *event orders*, stated in time.
- **Hidden layers by target times.** A hidden node's contribution is also a deadline:
  "spike before c_h, so that the output race goes the right way." Output constraints turn
  into deadline constraints on hidden spikes (via the same half-spaces one layer up), and
  each hidden node projects onto its own. Credit becomes **deadlines passed down**, not
  gradients passed back.

**(test M13)**: projection learning vs the current rule, on E4 (single layer, K up to
128) and then E6. Measures: mistakes to convergence, accuracy, number of weight updates.

### 11.2 Dreaming the paths not taken

Each collapse leaves residues: the futures that almost happened. Over waking time these
form a **ledger of near misses**: input history, rival, distance. In sleep (E10), the
machine does not replay raw data. It **re-runs the near misses** from the ledger with its
current weights and resolves them: applies the reordering constraints of 11.1 wherever
the order is still wrong or the margin too thin, and drops ledger entries that are now
settled. Learning in sleep is about alternatives that never happened, not stored
experience. This is the literal meaning of the project's name.

**(test M14)**: sleep on the ledger vs sleep on replayed data vs no sleep, at equal
storage, on class-blocked streams.

### 11.3 Structure before weights

In a race machine the decisive facts are *which* events can trigger which, and in what
order. Weights only tune the timing. So the primary learning operators should be
structural:
- **create a strand** (recruit: a new node for a history no existing strand reaches);
- **link** (a synapse from an event that should have mattered);
- **cut** (a synapse whose events never change an order);
- **reschedule** (11.1).

Gradient methods have only the last kind, and only approximately. A learner that can
choose among all four decides for each mistake which one is cheapest. Rescheduling
suffices when the nearest boundary is within reach; linking or recruiting when it is not
(§9, item 5). **(test M15)**: on the E10 new-class stream, a learner that picks the
cheapest operator vs weight updates alone.

## Tests

| | Claim | Test |
|---|---|---|
| **M1** | round-3 rule = −A · pathwise gradient; 1/A and extrapolated times matter or not | update-vector cosine on real samples; accuracy with/without (E11) |
| **M2** | near-miss weighting = cross-entropy gradient over times; truncated vs full counterfactual | Monte-Carlo E_noise[L] on small nets, numerical gradient vs rule, as σ varies |
| **M3** | decomposition: pathwise vs boundary (cancelled side, fired side); FA jump estimate vs true jump | small nets: compute each term exactly (true jumps by re-running the flip), compare to the Monte-Carlo total, to fired-only and to crl_fa |
| **M4** | failures come from interacting near misses | per-sample count of strands within σ of flipping vs learning error |
| **M5** | decision time and work are trainable | λ_t > 0; accuracy vs input-events frontier against E2 |
| **M6** | the rival's side of a cancellation boundary adds useful signal | add the coupled rival term to crl_fa; small-net gradient check, then accuracy |
| **M7** | shadow continuation trades work for gradient bias | continue cancelled strands for s ∈ {0, 0.1, 0.3, ∞} of the window; bias vs M3's exact gradient, and extra work |
| **M8** | ρ(Δ_min) ≈ 0 predicts errors gradients cannot fix; recruiting there beats a fixed novelty threshold | E10 with the ρ-triggered switch vs the δ-threshold trigger |
| **M9** | cross-entropy over projected times trains at least as well as the near-miss rule, with the same sparsity | output layer first (E4 setting), then E6 |
| **M10** | softmax(−τ(t)/σ) is calibrated at the decision and before it | expected calibration error over time; accuracy of early-stopped decisions |
| **M11** | speculative spikes cut multi-layer latency for little rollback work | latency vs rollback work as the speculation threshold varies |
| **M12** | strand-revision error is a useful unsupervised signal | E7 with the revision loss vs continuity learning, at 10% labels |
| **M13** | half-space projection (rescheduling) learns event order in one shot, with a mistake bound | E4 single layer, then E6; mistakes, accuracy, updates vs the near-miss rule |
| **M14** | sleep that resolves the near-miss ledger beats data replay at equal storage | E10 streams |
| **M15** | choosing the cheapest structural operator beats weight updates alone | E10 new-class stream |

M3 is the most informative experiment in this list. It says which term carries the
learning signal, whether our estimator of it is good, and what fired-only is missing,
on a network small enough that every term can be computed exactly. It should run
before E11. E11 then becomes "replace each estimated term with the better local one
that M3 identifies".
