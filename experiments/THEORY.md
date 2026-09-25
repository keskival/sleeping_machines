# Theory note — collapsing futures, and the gradient through a collapse

Written 2026-09-25. A working note: every claim marked **(test Mk)** is checked
numerically (table at the end) before later experiments rely on it.

## Synthesis: the principles so far (read this first)

Twenty-odd sections reduce to five principles. Each result elsewhere in this note follows from
one of them; the evidence column says how far each is established.

| # | principle | what follows | evidence |
|---|---|---|---|
| P1 | **Inference is tropical; the unrealised futures are its dequantization.** A race computes minimums over event times (min-plus); at temperature σ the pool of possible histories is a recombining forest weighted by e^{−cost/σ} (§21). | near-miss weighting = derivative of the soft minimum (§21.6); credit is conserved at each collapse (§22.3); holistic backprop = inside–outside (§21, §14); shadow spikes = first-order term (§20) | conservation: large gains at depth (debug, full runs queued); M20: asynchronous branches exact |
| P2 | **Weaving closes the past.** Collapses are stopping times; later inputs cannot affect them (§21.9–21.10). | woven vs unwoven counterfactuals (residues vs shadows); when to collapse = optimal stopping (E2); layers forecast each other's inputs | shadow neuron best at depth (debug); E2 tracks MSPRT |
| P3 | **A race neuron is a weighted mean in time.** Within a piece T = θ/ρ + Σ u_i t_i (§22.1, §24). | exact time-shift equivariance; urgency (ρ) vs evidence (u); piecewise linear, monotone nets suffice (§22.2); but learning must recruit absent evidence additively (§24) | equivariance exact; non-negative nets lose nothing; multiplicative learning fails |
| P4 | **Credit must reach what did not happen.** Along the realised history, credit reaches only nodes that fired; the rest is a blind spot that compounds with depth (§14, §16, §17). | counterfactual credit matters from depth 2; percolation threshold for local feedback; routing networks have the same boundary term (§19) | E14: +1.2 / +1.5 at depths 2 / 3, 2 seeds; M3 blind spot 75–85% of hidden weights |
| P5 | **Thresholds are prices.** Homeostasis is the dual update of a capacity constraint; a race layer with homeostasis is an online entropic optimal-transport solver (§25). | the target rate is a capacity; log-ratio updates balance faster | balance confirmed; equal capacities hurt accuracy; learnt capacities not yet found |

Negative results that shaped these: the routing gradient is myopic when alternatives learn
(E16, §19); multiplicative simplex learning cannot recruit (§24); stricter balance hurts (§25);
label-free hidden learning hurts (M21); stopping hidden work at the decision saves nothing (E9).

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
is ours. See RELATED_WORK.md: race logic and space-time algebra (the forward
mechanism), spike discontinuity estimation (near-threshold causal credit), the
EventProp spike creation/deletion work, and Madaline Rule II are the closest prior
work, and §11.1's linearity in the weights is a known property that we exploit.

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

## 12. Knowing the whole unravelling: value messages over time

Each neuron knows its own pending future (its strand), but not what the rest of the
network would do under each of its alternatives. Suppose it did.

### 12.1 The value of one's own futures

For node n, let **V_n(τ)** be the outcome (the loss, or simply right or wrong with a
time margin) if n spikes at time τ, where τ = ∞ means "does not spike". Everything
else responds as the network's own dynamics dictate:
- in n's group, a new winner cancels the current last winner, and a removed winner lets
  the next strand fire at its own projected time;
- downstream, every node's crossing time moves.

The realised history is one point, V_n(τ₀); the counterfactuals are the rest of the
curve. V_n is piecewise constant or smooth, with jumps where the downstream order
changes. Those jumps are what gradients cannot see.

### 12.2 The optimal local update

Given V_n, the best move is to reach the best region of V_n at the smallest weight
change. "Fire by c" and "not before c" are half-spaces (§11.1), so the cost of reaching
a region is an exact distance d_n(c). The locally optimal update is a proximal step
over the node's own futures:

    τ* = argmin_τ  V_n(τ) + λ · d_n(τ),   then project w_n onto the half-space for τ*.

This can jump across a flip, which no gradient step can. With exact V_n per node,
updating one node at a time (the one with the best gain per unit distance: minimal
disturbance) is safe. Updating many at once needs care, because their effects interact.

### 12.3 Messages: deadlines with values, sent backwards

V_n can be computed without trial runs. A downstream node j that receives n's spike at
τ crosses at T_j(τ) = (θ + B_j + w_jn τ)/(A_j + w_jn) while τ < T_j, and is unaffected
otherwise: a linear-fractional, monotone map. The output race's result as a function
of τ therefore changes only at a few breakpoints, solvable in closed form from each
output node's own (A, B, w). So the message from the output layer to hidden node n is a
short list of

    (deadline c, before/after, value change δ)

and the backward pass is itself a set of events in reverse time: "fire before c and
the loss drops by δ". Deeper layers compose the messages through the same monotone
time maps, keeping the breakpoints nearest the realised point: a **beam of futures**.
Nodes too far from any breakpoint receive nothing. The backward pass is as sparse as
the near misses.

Compared with backprop:

| | backprop / EventProp | unravelled messages |
|---|---|---|
| message | a slope at the realised point | a function of one's own spike time (breakpoints + values) |
| sees flips | no | yes (they are the breakpoints) |
| update | small step along the slope | projection onto the best reachable half-space |
| sparsity | every node on the path | nodes with a breakpoint within reach |

What is exact and what is approximate: V_n is exact for one node varying with
everything else responding, i.e. exact in magnitude and first-order in the *number* of
nodes that change (§6). Coalitions of nodes that matter only together would need
messages to groups, which is where the k-winner groups are a natural unit.

### 12.4 Tests

- **M16 (value of the information).** With V_n computed exactly by re-running the
  downstream race for candidate spike times (an oracle, on small networks), learn by the
  proximal projection of 12.2. Compare with crl_fa, fired-only and the M13 output
  projection. If the oracle learner is not clearly better, messages are not worth
  building, and we learn that local information already suffices.
- **M17 (messages equal the oracle).** Compute V_n from output-node breakpoints (12.3)
  and check it equals the oracle on every sample. Then measure message size (breakpoints
  per node) and sparsity (nodes reached).

## 13. The deeper frame: learning as history repair

Sections 2–12 keep translating the race into existing vocabularies: losses, gradients,
likelihoods, value functions. Start instead from what the machine does.

**Inference produces a history.** The output is not a number but a causal record. Each
spike was *enabled* by the spikes that brought its node to threshold, and it *cancelled*
the strands it inhibited. Every event has causes, which are events.

**Every collapse is a branch point.** At each collapse the history could have gone
another way. The residues say, locally, how close it came.

**Teaching says the history went wrong.** Not "a smooth function missed by x", but "the
wrong future happened".

So learning is **repair of a causal history**: find the branch point where the history
should have gone the other way (the *pivotal collapse*), change that one tie by the
smallest change that flips it, and leave everything else alone. Credit assignment becomes
a question of actual causation ("but for this event, would the outcome differ?"), not of
sensitivity (how much does the loss move per unit of weight?).

### 13.1 The repair protocol

A repair request is an event travelling *backwards along the causal links* of the
history. A node receiving "you should have fired before c" (or "not before c") can:

1. **fix it itself**: reschedule its own strand, at the half-space distance d_n(c) (§11.1);
2. **delegate to a cause**: "I would reach threshold by c if input h came earlier, or if
   near-miss h had fired", which is a request to h with a new deadline;
3. **delegate to a canceller**: "I would have existed if my group's last winner m had
   fired later", which is a request to m.

Each option comes back with a cost; the cheapest wins and is applied. The whole unravelling
is never collected anywhere. It is **negotiated** along the history, through events that
happened or nearly did, and the search stops where repair becomes cheap. Nodes far from
every branch point never hear of it.

### 13.2 What changes

- **Minimal disturbance by construction.** One pivotal repair per mistake, not a small
  push to everything that contributed. Most of the network, and most of what it has
  learnt, is untouched, which bears directly on forgetting (E10).
- **Structure is part of the repair.** "No node can be moved cheaply enough" is an answer
  too; the request then becomes *recruit* or *link* (§11.3).
- **Messages carry deadlines, not slopes.** Deadlines compose along causal links (a later
  crossing needs an earlier cause), so the backward pass is itself a set of timed events.
- **Exact at the pivot.** The repair is exact for the chosen branch point (a
  projection), and verified: the machine can re-run the repaired history, because
  replaying one race is cheap.

### 13.3 First test

**M18**: on the small network (14×14 MNIST, 60 hidden nodes), a repair learner restricted to
single-event repairs (the output node itself, or one hidden node that is a cause or a
canceller), choosing the cheapest repair that is **verified** by re-running the race,
against crl_fa, fired-only and the M13 projection. Measures: accuracy, repairs per
mistake, weights touched per mistake, and (on a class-blocked stream) forgetting. If
single-event repair already rivals the gradient-like rules while touching far fewer
weights, the protocol is worth building out: multi-step delegation, recruitment as
a repair, messages instead of verification by re-running.

## 14. Backprop through the unravelled future

### 14.1 Why greedy backprop suffices in dense models, and not here

In a dense network the computation graph is fixed: the same units compute in the same
order for every input and every weight setting, and only values change. The tree of
possible histories has one branch, so the derivative along it is the whole story, and
greedy and holistic coincide.

In an asynchronous race **the graph depends on the weights**: which events happen, their
order, and who cancels whom. A small weight change can swap two events and send the
computation down another branch. The true dependence of the outcome on the weights
runs through the *tree of histories*. Backprop along the realised path sees one leaf and
ignores how the tree itself moves. M3 measures the cost: in about 80% of hidden-weight
coordinates the realised-path gradient is exactly zero, while the expected error
depends on them.

### 14.2 The holistic objective

Let each collapse be a fork whose branch probabilities come from the pool: the
competitors' projected times at temperature σ (§5, §10). A forward pass then defines a
distribution over histories h, P_w(h), and the objective is

    J(w) = Σ_h P_w(h) L(h),
    ∇J = Σ_h P_w(h) [ ∇_w L(h)  +  L(h) ∇_w log P_w(h) ].

The first term is backprop inside each branch. The second term, how the weights move
the forks, is exactly what greedy backprop drops. Because every fork's probabilities
are explicit functions of projected times (which are explicit in w), ∇ log P is analytic:
there is no sampling and no score-function variance.

### 14.3 Why it is affordable here

A fork changes only the events causally downstream of it (its **light cone**), and
cancellation cuts the cone short. Branches share everything else. In a dense network
every unit depends on every unit before it, so every fork would touch everything. In a
sparse race a beam of B branches costs about B × (events in the cone), not B × the network.
**The asynchrony that breaks greedy backprop is also what makes holistic backprop cheap.**

### 14.4 One tree, three ways to read it

| aggregation over the history tree | learning rule |
|---|---|
| one leaf (the realised path) | greedy backprop (EventProp) |
| sum over branches, weighted by probability (sum-product) | holistic backprop: ∇ of expected loss |
| minimum over branches, weighted by cost (min-sum) | history repair (§13): cheapest branch that is right |

Borrowed tool: semiring computation over trees (as in weighted automata and parsing). What is
new is the object: the learning rule is **a choice of semiring over the network's own
tree of possible event histories**, and the beam width is the knob between greedy and holistic.

### 14.5 Temperature as continuation

High σ gives a wide tree, a smooth landscape and a signal to every weight; σ → 0 recovers the
deterministic race. Training from wide to narrow is a principled schedule, replacing the
hand-set σ we use now.

### 14.6 Beams computed asynchronously: shadow events

The beam needs no separate pass and no barrier. Branches live in the same event queue as
the factual computation:

1. **Fork.** At a collapse whose runner-up came close, the runner-up's spike is also
   scheduled as a *shadow event*, tagged with a branch id and the branch probability.
2. **Propagate.** Shadow events travel at the same simulated times through the same
   nodes. A node keeps a per-branch delta on top of its factual state. Integration is
   linear (A and B add), so the branch potential is exactly factual + delta, at O(1) per
   shadow input. Shadow spikes can cancel within their branch.
3. **End or resolve.** A branch whose delta stops changing anything (cancelled or absorbed)
   ends and costs nothing further. One that reaches the output resolves at about the same
   time as the factual decision.
4. **Learn.** Each resolved branch sends its loss difference, addressed by tag, to the
   nodes that forked it. **Their collapse residue is exactly the record waiting for that
   message.** The teacher may arrive before or after, and learning completes when both have.

What makes this possible: linear integration (branches are additive deltas, not copies),
light cones (only nodes downstream of a fork see its shadows) and cancellation (branches
end early). A dense synchronous network has none of the three.

Limits: interacting forks multiply (first order treats them independently; tags cap the
count); shadow events cost like real ones, so forking needs a probability threshold and
branches with little possible effect are dropped; hardware needs a few tag bits per event
and per-branch deltas, only inside active light cones. Side benefit: at decision time the
machine knows what the close calls would have led to, which is a calibrated confidence and
a learning signal before any teacher arrives.

### 14.7 Test

**M19**: on the M3 network, forks at hidden groups (swap the last winner with the next strand
in line, probability from their time gap) and at the output race. Compare, at matched extra
events: greedy (B = 1), sum-product with B ∈ {4, 16}, min-sum repair on the same beam, and an
annealed σ schedule. Measures: accuracy, fraction of samples giving any signal, extra events
per sample, gradient alignment with M3's finite-difference estimate.

## 15. Reinforcement learning may be the natural home

A race is an action selection: the first strand to collapse is the chosen action, and its
time is the reaction time. Much of the formalism maps onto reinforcement learning directly.

1. **The race is a policy.** Under timing noise, P(action) = softmax(−τ/σ) over the pool (§5),
   so π and ∇log π are read off the pool analytically, with no sampling variance.
   σ is exploration.
2. **Evaluative feedback needs counterfactual credit.** Rewards say how good, not what was
   right. Reward-modulated STDP credits only what fired and collapses as the action count grows
   (E4); residues give graded credit to unchosen actions.
3. **Delayed reward.** Tags over collapses (E7) are eligibility traces, as in TD(λ), counted per
   event rather than per tick.
4. **Time costs are native.** Faster decisions give more decisions per unit time, so the race
   trades accuracy against reward rate: average-reward RL over actions with durations
   (semi-Markov decision processes). E2's frontier is already a reward-rate curve.
5. **Value as latency.** A race computes a minimum over arrival times. With delays along paths
   that is min-plus algebra, the Bellman optimality operator for shortest paths (race logic
   already solves dynamic programming this way). Values can be *earliness*: better states fire
   sooner, the race chooses, and a temporal-difference error is literally a timing error.

Limits: shadow branches can play out the network's internal alternatives, not the world's
response to an action not taken; that needs a model or a critic. History repair becomes "the
cheapest branch that raises expected reward", which needs a critic. Prior work to credit:
reinforcement-learning drift-diffusion models of choice (e.g. Pedersen, Frank & Biele 2017)
and basal-ganglia action selection as a race.

Design: E13_RL.md.

## 16. Depth: why counterfactual credit should matter more in deeper networks

With one hidden layer the output reads hidden spikes directly, so the nodes that matter most
are the ones that fired, and fired-only credit already reaches most of the signal. In deeper
networks a near-miss node in layer l affects the output only through nodes in layer l+1 that
its spike *would have* tipped over threshold: a path made of events that did not happen,
invisible to any credit that follows realised spikes. Along the realised history, credit must
pass through a fired node at every layer, so the share of weights reachable by pathwise credit
should shrink roughly multiplicatively with depth, while residues keep reaching near-miss nodes
at every layer. Prediction: the gap between counterfactual and fired-only credit grows with
depth, from nothing at one hidden layer (E6 r3, M18). **(test: E14)** A debug run supports it
(+19 points at depth 3); full runs are queued.

## 17. Credit percolation: a trainability phase transition for event networks

### 17.1 Setting

Suppose credit travels only along the network's own event graph: backwards across
synapses that exist, layer by layer, as event hardware without global wiring would do it
(and as exact event gradients do). A node in layer l+1 that holds credit can pass it to
the upstream nodes that fed it, and an upstream node can carry it further only if it is
**eligible**: it fired, or (with counterfactual credit) it came within reach of firing.

### 17.2 Branching ratio

Let layer l have N_l nodes, each layer-(l+1) node have fan-in F from layer l, and let p_l
be the eligible fraction of layer l. A credited node reaches about F·p_l eligible parents.
Each parent has fan-out about F·N_{l+1}/N_l, so the expected number of credited nodes in
layer l per credited node in layer l+1, the **branching ratio**, is about

    b_l ≈ F · p_l        (more precisely, the chance that an eligible parent is reached
                          saturates at 1 − exp(−F·ρ_{l+1}) for credited density ρ_{l+1}).

The credited density ρ_l obeys ρ_l = p_l · (1 − exp(−F · ρ_{l+1} · N_{l+1}/N_l)), a
percolation recursion. It has a nonzero fixed point only when F · p · N_{l+1}/N_l > 1.
Below that, ρ decays geometrically with depth and deep layers receive no credit;
above it, ρ settles at a depth-independent level.

### 17.3 What sets the knobs

| knob | effect on b | consequence |
|---|---|---|
| fired-only credit | p = p_f (≈ winners/group) | may sit below threshold |
| counterfactual credit | p = p_f + p_near(σ) | raises b; can cross the threshold |
| temperature σ | p_near grows with σ | a **critical σ\*** for deep trainability |
| sparse fan-in F (E9) | b ∝ F | energy saving and deep trainability trade off, with a computable boundary |
| direct feedback (DFA, E14) | bypasses the graph | no percolation decay; failures there are dead nodes, not reach |

### 17.4 Relation to known work

Dense networks have an analogous theory for gradient *magnitudes* (signal propagation,
edge of chaos, dynamical isometry). Dead units in sparse and spiking networks are known.
The new object here is a **reachability threshold on a sparse event graph**, with
counterfactual (near-miss) edges as the parameter that moves it, and σ as a control
parameter with a critical value. To be checked against the literature before claiming
novelty.

### 17.5 The same condition in sparse mixture-of-experts training

A top-1 mixture-of-experts router gets no gradient toward the experts it did not pick: the
router's blind spot. In race terms, the k winners of a group are the top-k experts and the
residues are the unselected experts' margins. Two consequences:

1. **The trainability condition is the same branching ratio.** Credit reaches the routing
   decision only if more than the best match is eligible (k > 1, or near misses counted). In
   stacked MoE layers the reachable set compounds with depth as in 17.2.
2. **Near-miss credit for routers.** The residue boundary term (§4) applied to routing
   credits unselected experts from their margins without running them, since unselected
   experts never execute. Known remedies (noisy top-k gating, top-2 routing, gradient
   estimators for sparse routing such as SparseMixer) run more experts or estimate the
   gradient differently; where the residue view stands relative to them is to be checked.
   E9's routing race is an MoE, and the natural test: first-to-fire top-1 routing, router
   trained with vs without near-miss credit.

### 17.6 Predictions (E15)

- **P1.** With layer-by-layer event feedback, credited density decays geometrically with
  depth when F·p < 1 and stays flat when F·p > 1.
- **P2.** At fixed F and depth, counterfactual credit crosses the threshold where fired-only
  credit does not; accuracy follows reach.
- **P3.** Sweeping σ reveals a critical σ\* below which deep layers stop learning, near where
  the measured F·(p_f + p_near(σ)) crosses 1.
- **P4.** Sparse fan-in that is safe with DFA (E9, E14) can become untrainable with local
  feedback; the boundary follows F·p ≈ 1.

## 18. The early-evidence monopoly: why sparsity helps races

A race is decided by order statistics. With dense fan-in every node sees the same earliest
input events, so all nodes are pushed by the same few spikes, fire in correlated ways, and
the layer carries little beyond those first events; deeper dense layers wait for all upstream
spikes before deciding. Sparse fan-in gives each node a different subset of early evidence,
decorrelating the code. Prediction: redundancy falls and accuracy rises as fan-in shrinks,
until the percolation threshold of §17 bites, so sparsity has an optimum set by two
competing effects.

Debug evidence (depth 3, 3k images, 1 epoch): dense hidden-to-hidden (fan-in 64): redundancy
0.17 in the first layer, deeper layers integrate ~100% of upstream spikes before freezing,
accuracy 0.30; fan-in 2: redundancy 0.06–0.08, evidence used 0.72–0.82, accuracy 0.73.
E15 records both measures across its sweep.

## 19. Every routing network has a counterfactual routing gradient

Any network whose computation path is *selected* (races, top-k MoE, hard attention, early
exit, retrieval, beam search, tree-structured nets) chooses among alternatives by an argmax
or argmin over scores s_i(w). Under noise of scale σ on the scores, the gradient of the
expected loss is

    ∇E[L] = E[∇L along the chosen route]
          + Σ_i ρ_σ(s_chosen − s_i) · (L_i − L_chosen) · ∇(s_i − s_chosen),

the pathwise term plus a boundary term per alternative (§4). Backprop keeps only the first:
the router's blind spot is the missing second term. What the race formalism contributes to
this general picture:

1. **Residues are sufficient statistics for the boundary term.** From an unchosen route
   one needs only its margin and the margin's gradient. The costly part, L_i − L_chosen, is
   needed only for alternatives inside the noise band ρ_σ, which is a small set.
2. **A cost–accuracy knob.** Evaluate L_i exactly for the top few near-miss alternatives
   (shadow execution, §14.6), approximate it for the rest (first order, or a critic), and
   ignore alternatives outside the band.
3. **Percolation for stacked routers (§17).** Credit reaches deep routing decisions only
   if the eligible branching exceeds one: k > 1, or near misses counted.
4. **Asynchronous shadows.** Near-miss routes can be executed as tagged shadow events,
   where hardware allows it.

**Correction after E16 (myopia).** The boundary term is exact for the *current* alternatives.
When the alternatives themselves learn from what is routed to them (experts, hidden nodes), an
unchosen alternative's present loss understates its potential, so the term reinforces existing
routes and suppresses exploration. In E16 it helped early and hurt later. A routing gradient for
learning systems must value an alternative's learning potential: for example its loss after one
hypothetical update on the input (lookahead), or an optimism bonus (as in bandits). This also
suggests why near-miss credit works in races where it *trains the near-miss node itself* (E6,
E14): there the counterfactual signal goes to the alternative, making it learn, instead of only
steering the router toward or away from it.

**E16 (run; see FINDINGS):** an ordinary, non-spiking MoE with top-1 routing on a small task; router
trained by (a) the standard gate-value gradient, (b) a straight-through estimator,
(c) the boundary term with the top-m near-miss experts executed in shadow (m = 1, 2) and the
rest approximated, (d) dense top-2 as a reference. Measures: accuracy, expert load balance,
router quality (share of inputs routed to the expert with the lowest loss), and compute per
step. The claim to test is that (c) gives better routers than (a–b) at far less compute than
(d).

## 20. The two-channel neuron: counterfactuals as a second kind of spike

The residue machinery (store Δ, compare, weight by exp(−Δ/σ)) can be replaced by ordinary
event mechanics that fit the substrate:

- **Factual channel.** The neuron races as before; inhibition stops its factual potential.
- **Shadow channel.** The same neuron keeps integrating, uninhibited, and emits a *shadow
  spike* when it would have crossed. Time orders the near misses for free (the race is a
  sleep sort); a window after the group's decision bounds the cost.
- **Downstream.** Real spikes drive the race; shadow spikes drive a separate compartment that
  never affects the decision, so the counterfactual forward pass of the losers propagates
  layer by layer *as spikes*, along exactly the paths fired-only credit cannot see.

Eligibility becomes binary and time-selected: fired, or shadow-fired within the window.

**Relation to MoE and straight-through.** Weighted (soft) MoE keeps losing experts in the
forward pass with small gates, so gradients reach them, but they perturb the output and all
experts run. The two-channel neuron separates the roles: losers contribute zero to the decision
and fully to learning. This is the straight-through pattern (hard forward, soft backward), except
the backward path is the real counterfactual continuation of the losers rather than a surrogate
derivative, and it is sparse (near misses only). Biological analogues to credit: burst
multiplexing (Payeur et al. 2021) and segregated-dendrite learning (Guerguiev et al. 2017).

**The analogy is for understanding, not a destination.** The choices follow from the
substrate (asynchronous, sparse), not from MoE practice:

| | sparse / weighted MoE (synchronous, dense hardware) | race substrate (asynchronous, sparse) |
|---|---|---|
| selection | compare scores, softmax normalisation | first to arrive wins; time orders |
| losers in the forward pass | small gate weights, every expert runs | contribute nothing to the decision |
| credit to losers | gradient through small weights, all of them | shadow spikes, near misses within a time window only |
| cost of the counterfactual | ∝ number of experts | ∝ near misses in the window; ends when the window closes |
| normalisation | global (softmax over all options) | none; each node sees only its own inputs and events |
| learning signal | dense, continuous | binary, event-triggered, local |

**Debug evidence (depth 3, 5k images, 1 epoch, one seed):** shadow neuron with window 0.4:
0.698; residue weighting: 0.649; shadow window 0.05/0.15: 0.62; hard Δ window: 0.584; sampled
binary eligibility: 0.548; fired-only: 0.525. **(test M23)**, queued at full size with 2 seeds.

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

## 26. Pivotal credit: computing the jump instead of projecting it

The boundary term (§4) weights an alternative by how likely its flip was, ρ(Δ), *and* by how
much the flip would change the outcome, the jump. Our hidden credit has the first and replaces
the second by a random feedback projection, which knows nothing about which way a flip would
push the decision (hence M3's weak alignment). The jump is computable locally:

    δ_n = Σ_o s_o · w_on · [n's spike, real or projected, arrives before the output decides]

(the real output weights carried back along the same synapses), weighted by ρ(Δ_n) for near
misses. Refinements forced by experiment:

1. **Arrival is itself a boundary.** Gating by arrival (the weaving constraint, §21.9) creates
   *dead-late* units: a node too late to matter gets no credit and never learns to be earlier.
   The fix is the boundary term on the arrival surface: a late spike gets its in-time credit
   weighted by its time residue, e^{−(T − t_dec)/σ_t}.
2. **Deeper paths fail.** Passing pivotal credit between hidden layers through their real
   weights collapses at depth 3 (0.10). Conserving credit within hidden groups (zero-sum per
   group) does not fix it and also breaks the working rule (0.87 → 0.10): deep layers need a net
   push to stay active. *Open:* why real-weight credit between hidden layers is unstable here
   (moving targets as weights change, correlated credit across nodes).
   *Tested and refuted:* using the exact race Jacobian ∂T/∂t_i = w_i/A (evidence shares, which
   make the backward pass a conservative flow) instead of raw weights: depth 1 0.918, depth 2
   0.865 (worse than pivot at the top), depth 3 still collapses (0.096). The formal fix is right
   but is not what is breaking.
3. **Pivot at the top works.** Pivotal credit for the top hidden layer (exact jump through the
   output) with random feedback deeper: debug (10k images, 2 epochs, output conservation on):
   depth 1 0.921 vs 0.895; depth 2 0.896 vs 0.880; depth 3 0.876 vs 0.870. A lead; full-length
   runs queued.

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
| **M16** | knowing the exact unravelling V_n(τ) makes local learning clearly better | oracle proximal-projection learner vs crl_fa, fired-only, M13 (small nets) |
| **M17** | breakpoint messages reproduce the oracle exactly, sparsely | agreement on every sample; breakpoints per node; nodes reached |
| **M18** | history repair (cheapest verified single-event repair) rivals gradient-like rules while touching far fewer weights | small nets; accuracy, weights touched, forgetting |
| **M24** | the pool is a dequantized tropical computation: near-miss weights = ⊕_σ derivatives; inside–outside on the beam = soft-objective gradient; annealing helps at depth | small nets, M19/M20 machinery |
| **M25** | intra-trial martingale (TD) consistency of the pool trains earlier decisions at equal accuracy | output race, then E14 |
| **M26** | a per-layer σ schedule beats a global σ at depth | E14, depth 3 |
| **M27** | gradients estimated from the substrate's own timing noise (fluctuation–dissipation) | small nets vs M3's finite differences |
| **M23** | the two-channel (shadow-spike) neuron trains deep race networks at least as well as residue weighting, with binary, sort-free eligibility | depth 1–3, windows, 2 seeds |
| **E15** | credit percolation: reach decays geometrically below F·p ≈ 1; counterfactual credit and σ move the threshold | local layer-wise feedback, depth × fan-in × σ × credit type; per-layer reach and accuracy |
| **M19** | backprop through a beam of histories (sum-product) beats greedy; min-sum on the same beam equals repair | small nets; accuracy, signal coverage, extra events, alignment with M3 |
| **M20** | shadow events in the event engine reproduce the batch beam exactly, asynchronously | equality with M19 per sample; extra events and per-node branch state vs beam width |

M3 is the most informative experiment in this list. It says which term carries the
learning signal, whether our estimator of it is good, and what fired-only is missing,
on a network small enough that every term can be computed exactly. It should run
before E11. E11 then becomes "replace each estimated term with the better local one
that M3 identifies".
