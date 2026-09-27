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

## 27. Common-mode credit: why deep real-weight credit becomes an activity signal

Split a layer's hidden credit (per sample, over its eligible nodes) into its mean and the rest,
δ = μ·1 + δ̃. The two parts do different jobs, travel backwards differently, and should be
owned by different mechanisms.

**What each part does.** A ramp node's update is Δw_ni = η δ_n x_i with x_i ≥ 0, so its urgency
ρ_n = Σ_i w_ni (§24) changes by η δ_n Σ_i x_i. The common part μ moves every urgency in the layer
together. Within a k-winner group that barely changes *who* wins, but it changes *whether* the
group fires before the deadline: μ is an **activity** signal. The differential part δ̃ changes the
order within groups: it is the **evidence** signal.

**Mean-field drift of activity.** Let a be the layer's firing fraction, a* its target, κ the price
(homeostasis) step, and a = F(θ/ρ̄) decreasing in the mean urgency ratio. Credit moves ρ̄ at rate
η μ̄ X̄ (X̄ the mean arrived drive), and the prices move θ at rate κ(a − a*). Setting d(θ/ρ̄)/dt = 0
gives the stationary deficit

    a* − a = η θ X̄ |μ̄| / (κ ρ̄)          (μ̄ < 0)

μ̄ is negative through selection bias: on an error, the nodes that fired are the culprits, so
eligible credit is mostly "fire later". **Activity dies when the deficit reaches a***, which gives a
critical price step κ_c = η θ X̄ |μ̄| / (ρ̄ a*). This is why faster prices (κ = 0.01 or 0.03 instead
of 0.001) rescued the collapsed deep pivotal path (§26.2) to about 70% in debug runs, and why
removing μ̄ itself helped more (82.6%).

**Why μ grows with depth under real weights.** Backwards through real weights,
δ^{l−1}_i = Σ_j W_ji g_j δ^l_j (g the eligibility gate). Write W = w̄·11ᵀ + W̃, with w̄ > 0
(non-negative weights, §22.2) and W̃ zero-mean with spread s. With fan-out n and eligible fraction
p, the common part passes with gain ≈ n p w̄ and the differential part with gain ≈ √(np)·s,
because its terms add incoherently. So the common-to-differential ratio multiplies by roughly

    γ = √(n p) · w̄ / s

per hidden-to-hidden hop. Weights start as N(μ, μ), so w̄/s = 1; with n = 400 and p ≈ 0.3,
γ ≈ √120 ≈ 11. One hop
below the top, the credit is already almost pure common mode: deep layers receive an activity
signal dressed up as evidence. This one mechanism accounts for all three observations of §26:

- **Random feedback is immune.** B has zero mean, so its common-mode gain n p B̄ = 0.
- **Pivot at the top works.** δ_n = Σ_o s_o w_on with Σ_o s_o = 0, so the common part is only
  Σ_o s_o w̄_o, which is small unless output nodes differ in mean weight.
- **The exact race Jacobian does not help.** Evidence shares w/A have rows summing to 1, so they
  conserve the mean exactly while averaging (and shrinking) the differential part. The ratio still
  grows, so §26.2's refutation is the predicted outcome, not a surprise.

Relation to §17: γ is the branching ratio of the common mode. Credit percolation needs the
*differential* branching ratio above 1 while the common mode is held down; they are two
thresholds, not one.

**Remedies that follow.** (a) Project the common mode out at every layer (weights get δ̃ only,
prices own the mean), i.e. `--center-credit`. (b) Carry credit back through centred weights W̃.
(c) Keep κ > κ_c. Remedy (a) removes the cause, and (c) only outruns it.

**Predictions (M31).** The measured common-mode share E[μ²]/E[δ²] (now logged per layer as
`common_mode_share`) is (i) near 1 in hidden layers below the top for crl_pivot, and rises with
distance from the output; (ii) small and roughly flat for crl_fa; (iii) centering *alone*, at the
default κ = 0.001, recovers at least as much of depth 3 as faster prices alone (about 70% debug).
*Caveats:* the gains assume the gate g is uncorrelated with W̃, and use initial-weight statistics.

## 28. The race's own temperature: extreme-value spacings

σ has been a hand-tuned constant (0.15). Extreme-value theory fixes its meaning and makes it
measurable from the network itself.

**The Gumbel race is exact.** If a group's crossing times are T_i = μ_i + σ ε_i with ε_i i.i.d.
Gumbel (minimum form), then P(i crosses first) = softmax(−μ/σ) exactly, and the whole crossing
order is Plackett–Luce (§21.4). Minima of many roughly independent delays tend to the Gumbel law,
so the dequantized race of §21 is the natural large-race limit, not only a convenient smoothing.

**Spacings are exponential with mean σ/k.** Take equal μ. Then E_i = exp((T_i − μ)/σ) are i.i.d.
Exp(1), T_(k) = μ + σ ln E_(k), and for a large group E_(k) ≈ S_k/G with S_k the partial sums of
i.i.d. Exp(1) variables. So the k-th spacing D_k = T_(k+1) − T_(k) = σ ln(S_{k+1}/S_k). Since
S_k/S_{k+1} ~ Beta(k, 1), P(D_k > d) = e^{−kd/σ}:

    D_k ~ Exp(mean σ/k),   and the rescaled spacings k·D_k are i.i.d. Exp(σ)

(a Rényi-type representation). The maximum-likelihood temperature from the first m spacings is
σ̂ = (1/m) Σ_k k·D_k.

**Applied to our residue.** At the freeze, the closest loser's residue (θ − v)/θ is its time gap
to the k-th winner in units of its own time scale θ/A, i.e. D_k with k = `winners` = 3. The §28
draft rule (σ_l ← mean closest-loser residue) therefore sets σ **k times too small**. The corrected
estimator is σ̂_l = k · mean residue (`--self-sigma 2`). **Prediction (M32):** if 0.15 is the race's
natural temperature, the mean closest-loser residue is about 0.05, and `--self-sigma 2` settles
near 0.15 and trains at least as well as mode 1.

**Near-miss eligibility is the Plackett–Luce flip probability.** Under fresh Gumbel noise, a loser
at gap Δ behind the k-th winner overtakes it with probability 1/(1 + e^{Δ/σ}). The eligibility
e^{−Δ/σ} we use is its tail (Δ ≫ σ). The exact form halves the weight of the closest near misses
(Δ → 0), a small, testable difference.

*Caveats.* (i) Signal inflates spacings: when a layer is decisive, the winners' margin adds to
D_k, so the estimate runs hot on confident layers. Spacings among losers only (D_{k+1}, D_{k+2}),
which carry less of the decision, are a cleaner estimator. (ii) The groups have G = 10 members, so
the large-G approximation is rough; the exact finite-G spacing law is E_(k) = Σ_{j≤k} X_j/(G−j+1).
(iii) Time units differ across nodes through θ/A; the estimate is in the residue's own units, which
is what the eligibility uses.

## 29. Perron credit: backward credit through excitatory weights is power iteration

§27 used a mean-field split into mean and remainder. The exact object is the backward operator's
Perron direction, and the exact statement is a contraction theorem.

**Setting.** One backward hop is δ^l = (δ^{l+1} ∘ e^{l+1}) W^{l+1} = δ^{l+1} M_l, with
M_l = E_{l+1} W^{l+1} (E the diagonal eligibility gate). Credit m hops below the top is
δ^{top} M_{top−1} ⋯ M_{top−m}: m steps of (non-stationary) **power iteration**. Whatever the
evidence at the top, repeated multiplication rotates it towards the product's dominant direction.

**Deterministic version (Birkhoff–Hopf).** For a matrix with strictly positive entries, the
projective diameter Δ(M) = max log(M_ij M_kl / M_il M_kj) is finite, and M contracts Hilbert's
projective metric by τ(M) = tanh(Δ(M)/4) < 1. A consequence is |λ₂|/λ₁ ≤ τ(M). So, **with no
randomness assumed:** at every hop through strictly excitatory weights, the credit's
non-Perron (evidence) part shrinks, relative to its Perron part, by at least τ. After m hops the
relative evidence is at most ∏ τ(M_i). Deep credit through excitatory synapses is driven into one
direction regardless of the data, unless that direction is removed at every hop. This is a
structural statement about Dale-compatible networks (§22.2), which run on positive weights.

**Random version (outlier spectrum).** For W_ji ~ N(w̄, s²) gated to a fraction p, M is a
rank-one mean w̄·e1ᵀ plus a random bulk. The rank-one part produces an outlier that separates
from the bulk by the factor γ = √(np)·w̄/s (the bounded-rank outlier theorems of random matrix
theory). This is §27's gain, now as a spectral gap: the non-Perron energy fraction after m hops
is ~γ^{−2m}.

**What the Perron direction is.** The left Perron direction of one hop is
v_i = Σ_j e_j W_ji, node i's total outgoing weight onto consumers that were eligible on this
sample. Moving along v changes the layer's total drive into the next layer, which is its
activity. So §27's "activity signal" is precisely the Perron mode of the backward operator, not
the uniform mean. The two coincide only if all nodes have equal gated out-weight.

**Mean centering leaks, by exactly the evidence scale.** cos²(1, v) = 1/(1 + CV²(v)), so plain
mean centering leaves a fraction CV²/(1 + CV²) of the Perron mode, which the next hop amplifies
by γ. At initialisation, v_i sums ≈ np terms of N(w̄, s²), so CV ≈ s/(w̄√(np)), and the leaked
amplitude is CV·γ = 1 (for w̄ = s): **mean-centered deep credit still carries an activity mode as
large as its evidence, independent of width.** Learning makes out-weights heterogeneous (hub
nodes), CV grows, and the leak grows with it. Removing the projection on v (per sample, per
layer) removes the mode exactly to first order: `--center-credit 2`. It is local: v_i is node
i's outgoing weight to consumers that were eligible, which the pivotal rule already reads
through reciprocal synapses.

**Predictions (M33).** (i) Below the top, uncentered pivotal credit has `perron_share` near 1,
at least as large as `common_mode_share`. (ii) After mean centering, the credit's Perron share
stays substantial (order 0.5, not 0). (iii) Perron centering ≥ mean centering at depth 3, with a
larger gap at the default κ = 0.001 and later in training; at depth 2 (one hidden-to-hidden hop)
the gap is small.

**What is borrowed and what is new.** Birkhoff–Hopf, Perron–Frobenius and outlier spectra are
classical. **Prior art on the core observation:** Clark, Abbott & Chung (NeurIPS 2021, "Credit
assignment through broadcasting a global error vector") note that in networks with non-negative
weights the backpropagated signal has a component along the outlier eigenvector, which makes the
backward pass grow; they avoid it with vectorised units and a broadcast error. §29's random-matrix
part is therefore a rediscovery. What I have not found elsewhere (a quick search, not a review):
(a) the deterministic, distribution-free bound via Birkhoff contraction, τ = tanh(Δ/4) per hop;
(b) the identification of the Perron mode with *activity*, to be owned by homeostatic prices,
and the per-sample local projection v_i = Σ_j e_j W_ji; (c) the result that mean centering leaks
the mode at exactly the evidence scale (CV·γ = w̄/s); (d) the application to event-driven race
networks, where it explains §26's failures.

## 30. Symmetries of the race and their Ward identities

§27 to §29 found the activity mode empirically and statistically. It follows exactly from the two
continuous symmetries of the race, with no statistics. Each symmetry of the loss gives an identity
on the gradient (a Ward identity, the static analogue of Noether's theorem). A symmetry that the
network only *almost* has leaves an anomaly term, and the anomaly is where the physics is.

### 30.1 Time translation: timing credit sums to the deadline's credit

Shift every firing time of layer l by c. By time-shift equivariance (§22.1), every downstream
crossing time shifts by c. Race orders are unchanged, and the loss, a function of time
differences (softmax(−τ/σ) is shift-invariant), is unchanged, *except* where a time is compared
with a fixed clock: the deadline. Shifting all times by c is the same as shifting the deadline by
−c. Differentiating at c = 0:

    Σ_i ∂L/∂T^l_i  =  ∂L/∂t_dl         (at every layer l, exactly)

(the sum runs over every node whose time enters: fired nodes and the projected crossings of near
misses). Consequences:

1. **The common mode is the deadline's credit, exactly.** The part of timing credit that does not
   sum to zero is the credit on the clock: "things should arrive sooner or later overall". That
   is §27's activity signal, now as an identity rather than a mean-field approximation, and it is
   rightly owned by the prices (thresholds), which are how a layer moves relative to the clock.
2. **Centering in time coordinates is the symmetry.** With the deadline's part given to the
   prices, the remaining credit sums to zero at every layer. Uniform centering of timing credit
   is not a heuristic: it enforces the exact identity. (The §22.3 conservation of credit at a
   collapse is the same identity for a single race.)
3. **The exact Jacobian is a Markov kernel.** ∂T_j/∂t_i = w_ji/A_j = P_ji ≥ 0 for non-negative
   weights, with Σ_i P_ji = 1 (row-stochastic): the symmetry, restricted to one node. Backward
   timing credit through the exact Jacobian is a Markov chain run backwards. Row-stochasticity
   gives Σ_i (δP)_i = Σ_j δ_j, so **the zero-sum subspace is invariant** and the sum is conserved
   hop by hop.

### 30.2 The exact gradient vanishes with depth at the Markov mixing rate

For zero-sum credit with roughly independent components across consumers,
E‖δP‖² ≈ ‖δ‖² · (1/F_eff)·(n_{l+1}/n_l), where F_eff = 1/Σ_i P_ji² is a node's effective fan-in
(the participation ratio of its evidence shares). In the worst case the Dobrushin coefficient of
P bounds the contraction of zero-sum vectors. So **even the exact timing gradient vanishes
geometrically with depth, at a rate set by how many inputs each node effectively averages.** A race
neuron is an averager (§22.1): averaging mixes, and mixing erases the differences that carry
evidence. Only the minimum (the race, k-winner cancellation) restores contrast.

This also explains the §26.2 refutation quantitatively. Any approximation that breaks the
identity (eligibility gating, the 0.05 eligibility cutoff, the lateness factor) injects a
nonzero sum. The kernel **conserves** that sum exactly while shrinking the zero-sum evidence by
√F_eff per hop, so the relative error grows by about √F_eff per hop, ×10–20 for dense layers.
The exact Jacobian failed not despite being exact but *because* exact kernels conserve errors in
the sum.

**Remedies that follow.** (a) Re-impose the identity after every hop: share Jacobian followed by
uniform centering (`--share-jac 1 --center-credit 1`). (b) Keep F_eff small where depth matters:
sparse, peaked evidence shares mix slowly. This gives E15's sparsity a mechanism, the §18
early-evidence monopoly a second role, and predicts that deep credit improves as learning
concentrates evidence shares (logged: `f_eff` per layer).

### 30.3 Scale: urgency and threshold are one degree of freedom (a gauge)

(θ_n, w_n) → (αθ_n, αw_n) leaves every firing time unchanged (T = θ/A + Σ (w/A) t, and the residue
(θ − v)/θ), so it is an exact symmetry of each node. Its Ward identity is

    Σ_i w_ni ∂L/∂w_ni  =  −θ_n ∂L/∂θ_n

and, consistently, ∂T/∂w_i = (t_i − T)/A, whose w-weighted sum is −θ/A = −θ ∂T/∂θ. Only the
urgency ratio θ/ρ is physical. **Credit rules that change ρ and homeostasis that changes θ are
two controllers on one degree of freedom.** §27's activity drift is their conflict: credit
pulls urgency down through selection bias while prices pull it back. The standard resolution of
a gauge redundancy is to fix the gauge: let credit change only the evidence mix w/ρ (restore
each node's ρ after every update) and let the prices alone set urgency (`--gauge 1`). This is
per node and across synapses, unlike centering, which is per layer and across nodes. The
exact gradient ∂T/∂w_i = (t_i − T)/A also says the natural input factor is how much earlier an
input arrived than the crossing, not its drive x_i as the rule uses. That is a further untested
refinement.

### 30.4 The conservation laws dual to these symmetries

A continuous symmetry of the loss gives two things: a Ward identity on the gradient (above) and,
under gradient flow on the parameters it acts on, a conserved quantity (Noether). Three cases, of
decreasing certainty.

1. **Credit current along depth (exact, already used).** Time translation acts on activities,
   not parameters, so its conserved quantity lives in the backward pass: the total timing credit
   Σ_i δ^l_i is conserved from layer to layer by the Markov kernel (§30.1), and created only by the
   deadline anomaly at each layer. Credit behaves like a current in Kirchhoff's sense: conserved at
   every node, with the clock as its only source. This is why errors in the sum persist (§30.2).
   The practical rule follows: *account for the source (give it to the prices), and keep the
   current divergence-free elsewhere.*
2. **Scale charge per node (exact for gradient flow, broken by our rules).** Under gradient flow on
   (θ_n, w_n), the scale Ward identity gives dQ_n/dt = 0 for Q_n = θ_n² + ‖w_n‖²: each node moves on
   a sphere, and only its angle (urgency θ/ρ and evidence mix w/ρ) is physical. Two consequences.
   (a) With finite steps, Q grows by η²‖∇‖² per step, so the effective step η/Q decays by itself: a
   built-in annealing (known for batch-norm networks). (b) Our rules are not gradient flows:
   homeostasis moves θ without the matching change in w, and credit has a selection bias. So the
   drift of Q_n measures how much of the learning rule's motion runs along the gauge orbit, where it
   changes nothing but the node's effective step size. Nodes whose Q shrinks (weights decaying
   under negative credit) get ever larger effective steps: a candidate mechanism for the
   instability of dying layers. Logged: `noether_q` (start, end) per layer. Gauge fixing (§30.3)
   removes the self-annealing, so it may need an explicit step schedule.
3. **Latency charge (exact at any step size, if delays are learnt).** If synaptic delays d_i are
   parameters, shifting all delays into a layer together with the deadline is an exact
   *translation* symmetry. Its charge is linear, Σ_i d_i + t_d (with a fixed deadline, Σ_i d_i
   moves only through the anomaly), and a linear charge is conserved
   exactly by gradient descent at any step size (⟨Δw, ξ⟩ = −η⟨∇L, ξ⟩ = 0), not just in the flow
   limit. So gradient descent can only **redistribute** latency among synapses. The mean latency,
   and with it how early the network decides, changes only through the anomaly: explicit time
   pressure from the clock or a time cost. **Consequence for the asynchronous programme:** a
   clock-free network cannot learn to be faster from accuracy alone. Speed must be priced
   explicitly (the λ_t of M5). Conversely, a pure accuracy loss cannot secretly trade latency
   for accuracy.

**Predictions (M34).** (i) Share Jacobian + centering trains the deep pivotal path (depth 3)
where the share Jacobian alone collapses (0.096). (ii) Gauge fixing alone prevents the activity
collapse of uncentered pivotal credit, and composes with centering. (iii) Gauge fixing costs the
working random-feedback rule little (crl_fa, depth 3). (iv) F_eff falls during learning in the
layers that train well.

**Borrowed and new.** Ward identities and gauge fixing are standard in physics. Conserved
quantities from symmetries of neural-network losses are known (e.g. scale symmetry and its
conserved norms, Kunin et al. 2021, "Neural mechanics"), and §30.3 is an instance of that. As far
as I know, these are new: the time-translation identity with the deadline as its anomaly, the
row-stochastic (Markov) structure of timing Jacobians, and the consequences for depth (mixing-rate
vanishing, exact conservation of sum errors). This is unverified against the literature.

## 31. Contraction: where contrast is lost, and the only places it is made

### 31.1 Every time-invariant neuron has a row-stochastic timing Jacobian

Let a neuron start at rest and follow time-invariant dynamics (no fixed clock inside the neuron;
leak, synaptic filters and refractoriness are all allowed). Then shifting all its input times by c
shifts its output spike by c: T(t + c1) = T(t) + c. Differentiating,

    Σ_i ∂T/∂t_i = 1        for any autonomous neuron model, not only the ramp

(for the ramp, ∂T/∂t_i = w_i/A; for Mostafa's exponential-synapse neuron, where
e^{T} = Σ w_i e^{t_i}/(Σ w − θ), it is w_i e^{t_i}/Σ_j w_j e^{t_j}). If every input is excitatory, delaying an
input cannot make the spike earlier, so the row is non-negative: **a Markov kernel**. The race
has the same structure: under Gumbel smoothing (§28) the soft minimum of a group has derivative
π_n = softmax(−T/σ)_n, also a stochastic row. A feedforward time-coded network of excitatory
neurons and races is, locally, **a product of Markov kernels**. Inhibition is the one exception:
its rows still sum to 1 but have negative entries, so ‖row‖₁ > 1 is possible (for example
T = 2t₁ − t₂).

### 31.2 Forward contrast and backward credit contract by the same coefficient

Within one piece (fixed arrival sets and winners), the layer map is linear: Δt^{l+1} = P Δt^l.
Two classical inequalities hold for any stochastic P, with the same Dobrushin coefficient
δ(P) = max_{j,k} TV(P_j, P_k):

- **forward:** osc(Pv) ≤ δ(P)·osc(v), where osc = max − min. The spread of times that carries
  the input's information shrinks.
- **backward:** ‖uᵀP‖₁ ≤ δ(P)·‖u‖₁ for zero-sum u. Credit on times (zero-sum by §30.1) shrinks.

These are dual statements, so the same number measures both how much a layer forgets about input
contrast and how much credit it loses. **Discriminability and trainability of a deep time-coded
network are one quantity.** δ is small when consumers average overlapping evidence (dense,
redundant rows) and near 1 when their evidence mixes are disjoint (sparse, specialised rows).
Logged: `dobrushin` [max, mean pairwise TV] per layer.

### 31.3 Contrast is made only at boundaries

If the linearised (pathwise) part of the network can only contract contrast, amplification must
come from the non-linear part: the switches between pieces. Those are which inputs arrived before
the crossing (causal truncation), which nodes won the race (k-winner cancellation), and
inhibition. In the gradient these are exactly the **boundary terms** (§4): the near-miss credit.
So near-miss credit is not a small correction to a pathwise gradient. **At depth, it is the only
channel that does not contract.** Prediction: the advantage of boundary credit over pathwise-only
credit grows with depth. *Existing evidence (E14, full length):* crl_fa minus crl_fired_only,
seed 0: −0.07, +0.64, +1.69, +1.87, +2.48 points at depths 1 to 5; seed 1: +0.40, +1.80, +1.24 at
depths 1 to 3. The two-seed mean at depths 1 to 3 is +0.17, +1.22, +1.47. It rises with depth,
but seed 1 is not monotone (depth 2 > depth 3), so this is consistent with the prediction, not
yet strong support. Seed 1 at depth 5 is queued. (Single seed; fired-only keeps some boundary
information through the winners it selects, so the test is conservative.)

### 31.4 Temperature trades credit reach against credit contrast

The race kernel π = softmax(−T/σ) mixes more as σ grows (δ → 0 as σ → ∞; a one-hot selection
with δ = 1 at σ → 0). Percolation (§17) wants σ large so that credit *reaches* deep nodes, while
contraction wants σ small so that credit keeps its *contrast*. Each layer has an optimum, and it
should move with depth, which gives M26 (a per-layer σ schedule) a derived objective: maximise
reach × ∏δ.

### 31.5 Design consequences

1. **Sparse, specialised fan-in keeps δ near 1.** This is a mechanism for E15's sparsity effects
   and for the "structure first" programme (§11.3).
2. **Inhibition is a contrast amplifier.** Excitatory-only time networks are contractive at every
   layer. Prediction: the cost of non-negative weights grows with depth. *Existing evidence:*
   depth 3, full length, seed 0: 0.9324 non-negative vs 0.9373 signed (−0.5 points, single
   seed; debug sizes showed no cost at depths 1 and 3). A depth sweep is needed.
3. **Selection restores contrast.** k-winner races re-sharpen what averaging blurs, so depth in
   race networks should interleave averaging (integration) with hard selection. Fewer winners
   means a sharper selection.

**Predictions (M35).** (i) The layers that learn well raise their mean pairwise TV (δ) during
training. (ii) The non-negative-weight penalty grows with depth (depths 1, 3, 5). (iii) The
boundary-credit advantage keeps growing with depth under multi-seed runs. (iv) At fixed
parameter count, sparse fan-in helps deep networks more than shallow ones.

## 32. Literature check (2026-09-26, web search; not a systematic review)

| Claim here | Status |
|---|---|
| Perron / outlier mode dominates backward credit through non-negative weights (§29) | **Prior art:** Clark, Abbott & Chung, NeurIPS 2021 (global error-vector broadcast): the backpropagated signal has an outlier-eigenvector component in non-negative networks. |
| Birkhoff contraction bound, Perron = activity mode owned by prices, mean-centering leak (§29) | Not found. |
| Gumbel race = softmax, Plackett–Luce (§28) | Classical (Gumbel-max trick, Luce, Plackett). |
| Residue = k-th Gumbel spacing, σ = k × residue (§28) | Rényi spacings are classical; the application was not found. |
| Time-shift equivariance of spiking neurons (§22.1) | Known and used implicitly in TTFS work (Mostafa 2017; Göltz et al. 2021; DelGrad notes that dendritic and axonal delays are interchangeable). |
| Row-stochastic timing Jacobian for any autonomous neuron; Ward identity with the deadline as anomaly; Markov-kernel view of depth (§30.1, §31.1) | Not found stated. The row-sum-1 property is implicit in Mostafa's formulas. |
| Vanishing gradients in deep TTFS networks (§30.2) | **Known phenomenon:** Stanojevic et al. 2024 (Nat. Commun., "0.3 spikes per neuron") attribute it to the neuron's slope at threshold and fix it by initialisation. The *mixing-rate* mechanism (1/F_eff, Dobrushin) and the exact conservation of sum errors were not found. |
| Scale and translation symmetries give conserved quantities under gradient flow (§30.3, §30.4) | **Known in general:** Kunin et al. 2021 ("Neural mechanics"); Tanaka & Kunin 2021 ("Noether's learning dynamics"): translation conserves an intercept, scale a norm. |
| Latency charge: delay learning cannot change mean latency without the clock (§30.4) | Not found. DelGrad trains delays with a spike-time margin loss (shift-invariant) and bounds delays with a sigmoid, but does not discuss total latency. |
| Forward-contrast / backward-credit duality through one Dobrushin coefficient; boundary terms as the only non-contracting channel (§31) | Not found. |
| Concave piecewise-linear firing time, one piece per causal set; expressivity (§34.1) | **Prior art:** "Polyhedral geometry of time-to-first-spike neural networks", arXiv 2609.11227 (2026). Time invariance and causality as axioms of temporal computing: Smith, space-time algebra (2017–18); race logic. |
| Topical-map view: sup-norm non-expansiveness, jitter certificate from race gaps, gaps = robustness margins, no chaos in recurrent excitatory race networks (§34.3–34.5) | Not found. The polyhedral paper's abstract does not treat robustness or recurrence. |

"Not found" means a few targeted searches found nothing; it is not a claim of priority.

## 33. What the theory does not yet explain (audit, 2026-09-26)

The theory so far is a set of local mechanisms (credit estimators, their failure modes, temperature,
contraction), not a theory of the model class. Open, in order of how much would follow from closing
them:

1. **What the networks compute.** No characterisation of the function class, its expressivity,
   or the role of k > 1 winners. *(Partly closed in §34; expressivity is largely prior art.)*
2. **Robustness to timing noise.** The defining vulnerability of a time code had no theory.
   *(§34: an exact certificate.)*
3. **Learning dynamics.** The rules are not gradients of a stated objective. There is no descent
   function and no convergence result. Prices plus credit is a primal–dual (Arrow–Hurwicz) system,
   whose stability was only sketched (κ_c, §27).
4. **Decision time.** First-crossing decisions are first-passage times. There is no optimal-stopping
   (sequential probability ratio test) account of when a race *should* decide, or of whether our
   thresholds implement one.
5. **Recurrent and asynchronous regimes (E7).** Essentially no theory. *(§34.5: stability.)*
6. **Generalisation.** None: no capacity or margin bound for race networks.
7. **Noise as model vs noise as smoothing.** σ is used both as the Gumbel noise of a stochastic model
   and as a smoothing temperature. The two roles coincide only if the substrate's actual timing noise
   has scale σ, which is unmeasured.

## 34. Excitatory race networks are topical maps

A map f: ℝⁿ → ℝᵐ is **topical** if it is monotone (x ≤ y ⇒ f(x) ≤ f(y)) and additively homogeneous
(f(x + c1) = f(x) + c1). Topical maps have a mature theory: non-linear Perron–Frobenius theory
(Crandall & Tartar 1980; Gunawardena & Keane 1995; Gaubert & Gunawardena 2004). The facts established
above (§22.1 shift equivariance, §31.1 monotonicity for excitatory inputs) say exactly that the
race network's timing maps are topical. The theory then transfers wholesale.

### 34.1 The neuron: a concave topical map, a minimum over causal sets

A non-negative ramp neuron crosses at the T solving Σ_i w_i (T − t_i)⁺ = θ. For any set S of inputs,
(x)⁺ ≥ x and (x)⁺ ≥ 0 give θ = Σ_i w_i (T − t_i)⁺ ≥ Σ_{i∈S} w_i (T − t_i), so T ≤ T_S = (θ + Σ_S w_i t_i)/W_S,
with equality for the causal set (the inputs that actually arrived). Hence

    T(t) = min over input sets S of  (θ + Σ_{i∈S} w_i t_i) / W_S

a tropical sum of weighted means. T is concave, piecewise affine with stochastic gradients, and
topical. *Prior art:* the concave piecewise-linear structure with one affine piece per causal set, and
the resulting expressivity (numbers of causal regions, richer than ReLU networks), is in
"Polyhedral geometry of time-to-first-spike neural networks" (arXiv 2609.11227, 2026). Race logic
and Smith's space-time algebra (2017–2018) take time invariance and causality as axioms and
build with min/max. §34.1 is a re-derivation; what follows builds on it.

### 34.2 A network is topical pieces glued along race boundaries

With the firing pattern fixed (which nodes win each group race), the network's map from input times
to output crossing times is a composition of concave topical maps, which is again **concave and
topical**. All non-concavity and all discontinuity come from the races: a node's output switches
between its own crossing time and "no spike" (+∞) when it swaps rank k and k + 1 in its group, or when a
crossing passes the horizon. This is the geometric version of §31.3: the smooth part is a contraction,
and everything else happens at race boundaries.

### 34.3 Non-expansiveness (Crandall–Tartar)

Every topical map is non-expansive in the sup norm: ‖f(x) − f(y)‖∞ ≤ ‖x − y‖∞. For a race network with
non-negative weights and a fixed firing pattern, **no timing perturbation is ever amplified**:
jitter of at most ε on every input moves every crossing time by at most ε, at every depth. Signed
weights break this. The sup-norm Lipschitz constant of a neuron is Σ_{i∈S} |w_i| / W_S ≥ 1, unbounded
as W_S → 0. Excitatory race networks are therefore intrinsically noise-stable in the timing
domain, and inhibition, the contrast amplifier of §31.5, is also the only amplifier of timing noise:
the same trade-off from the other side.

### 34.4 An exact timing-jitter certificate

Combine 34.2 and 34.3 by induction over layers. If every input moves by at most ε and the firing
pattern up to layer l is unchanged, every crossing time up to layer l moves by at most ε. A race at
layer l + 1 keeps its winners if the gap between its k-th and (k + 1)-th crossing exceeds 2ε, and if
no relevant crossing lies within ε of the horizon. The decision is unchanged if the output margin
(second-earliest minus earliest output crossing) exceeds 2ε. So:

    ε*(x) = min( ½·output margin,  min over races of ½·D_k,  min over boundary crossings of |T − H| )

is a **certified radius**: for non-negative weights no jitter with ‖Δt‖∞ < ε* can change the decision.
It costs one forward pass (`--certify 1`; checked against actual jitter).

Consequences:

1. **The near-miss residues are the robustness margins.** D_k at each race is exactly the quantity
   the near-miss rule reads (the closest loser's residue). A rule that widens the gaps of the
   races that matter is doing margin maximisation in the timing domain, as a hinge loss does for
   a linear classifier. This links the credit theory (§4, §20) to robustness.
2. **Extreme-value prediction for the null.** Under §28, each race's D_k ~ Exp(σ/k) when no race has
   an informative margin. With R races on the path, the minimum gap is ~ Exp(σ/(kR)), so the
   worst-case certificate of an *untrained* network shrinks like 1/R with size. Training must open
   the gaps that matter. The certificate is worst-case: most small gaps belong to races whose flip
   would not change the decision, so actual robustness should be much larger. The gap between
   certified and empirical robustness measures how much of the network is decision-irrelevant.
3. **Signed networks** carry no certificate. Measuring flips among "certified" samples for them
   shows how far they are from non-expansive in practice.

### 34.5 Recurrent excitatory race networks cannot be chaotic

Iterating a topical map (a recurrent race network running continuously, a candidate substrate for
E7's asynchronous stream) inherits non-linear Perron–Frobenius theory:

- **No timing chaos.** A non-expansive map has no positive Lyapunov exponent. Timing
  perturbations in recurrent excitatory race networks never grow, except through race-boundary
  switches.
- **A well-defined rhythm.** For piecewise-affine topical maps the cycle-time vector
  χ(f) = lim_k f^k(x)/k exists and is independent of x (Kohlberg 1980; Gaubert & Gunawardena). For the
  min-plus linear case it is the minimum cycle mean of the network graph (Karp's algorithm). So a
  recurrent race network has computable firing rates, set by its cycles, not by initial conditions.
- **Inhibition is necessary for rich dynamics.** Anything beyond fixed rhythms and contracting
  transients (memory by switching, chaotic exploration) needs either race-boundary switching or
  inhibition. This is a structural argument for E/I balance in asynchronous substrates, derived
  rather than borrowed from biology.

### 34.6 Tests (M36)

(i) Non-negative networks: **zero** decision flips among samples with ε* > ε, at every tested ε
(a failure means a bug or a wrong theorem). (ii) Certified radii of trained networks exceed the EVT
null σ/(kR). (iii) Empirical flips occur at ε well above ε*: the tightness gap. (iv) Signed networks:
flips among "certified" samples are possible; their rate measures non-expansiveness violations.

## 35. Weaving: commitment cost, free energy and certified early commitment

Results about the weaving operator itself (intuitive description: WEAVING.md). Notation: a race
among candidates n with projected times T_n; the pool's law at temperature σ is Plackett–Luce with
π_n = softmax(−T/σ)_n (§21.4, §28); the soft value of the race is its free energy
F = −σ log Σ_n e^{−T_n/σ} (the dequantized minimum, §21.3).

### 35.1 Committing to a branch costs temperature × surprisal (exact)

When a race weaves, the open value F is replaced by the value of the committed branch, T_w. From
the definition of π,

    T_w − F  =  −σ log π_w          (exact, for every race and every committed w)

Summed over the weaves of one input, by the Plackett–Luce chain rule,

    Σ_weaves (T_w − F)  =  −σ log P_σ(realised history)

**The value given up by weaving is the temperature times the surprisal of the history that was
woven.** Averaged over branches drawn from the pool's own law, the Gibbs identity F = E_π[T] − σH(π)
gives

    E_{w~π}[T_w − F]  =  σ · H(π)

The expected commitment cost of a race is σ times its entropy. This is the Landauer form: closing
off alternatives costs temperature × information. As σ → 0, a hard race always commits to its
minimum at zero cost, and all of the cost lives at σ > 0, in the pool.

### 35.2 Near-miss credit is the gradient of the commitment cost

Differentiate the commitment cost C_w = T_w − F with respect to the candidates' times:

    ∂C_w/∂T_w = 1 − π_w,     ∂C_w/∂T_n = −π_n   (n ≠ w)

These are the near-miss weights. Each loser gets credit ∝ e^{−Δ_n/σ} relative to the winner, and the
credits sum to zero, which is §22.3's conservation. So the three objects the theory treats
separately are one:

- the credit rule (§4, §21.6),
- credit conservation at a collapse (§22.3), and
- the cost of weaving,

**Near-miss credit is the gradient of what weaving throws away.** The supervised output loss is the same object: cross-entropy over output times,
−log π_y = (T_y − F)/σ, is the commitment cost of weaving the *teacher's* branch. Supervised learning
lowers the price of committing to the right answer. Minimising the summed commitment
cost minimises the surprisal of the network's own woven history. That is a label-free objective,
which makes races decisive, which widens the gaps D_k, which enlarges the certified jitter radius
(§34.4). With a label, the supervised loss's credit reaches each race only through these same
derivatives.

### 35.3 The soft value is a submartingale; weaving is its compensator

Condition on the woven past (the filtration F_t of §21.9). With M_t = E[e^{−C/σ} | F_t] (a Doob
martingale for any total cost C), the soft value F_t = −σ log M_t is a **submartingale** (Jensen,
since −log is convex): in expectation it can only rise as the past closes. Its Doob–Meyer
decomposition splits it into a martingale (news: the input revising forecasts) and an increasing
compensator (commitment: alternatives being closed off). By 35.1, the compensator's increment at
each race is, in expectation, σ H(π) of that race. (Exact for a single race whose branch values
are its candidates' times; across layers, where a branch's value is itself an open soft value,
this is a first-order identification, not yet a proof.) As σ → 0 the soft value becomes the tropical value
(the best cost still reachable given the woven past), which is **non-decreasing**: every weave can
only remove options. The step at a weave is the gap to the best alternative it cut off (the D_k of
§28 and §34.4).

Reading: **a network's decision process splits into learning from new evidence (martingale) and
paying for commitment (compensator)**. §22.4's temporal-difference signal is the martingale part.
The commitment part is what the near-miss credit differentiates.

### 35.4 Certified early commitment (bracketing the future)

With excitatory weights, more input only makes crossings earlier. At time t each candidate's
eventual crossing is therefore bracketed:

- **upper bound** T_n^+(t): its projection if no further input arrives (the current forecast);
- **lower bound** T_n^−(t): its crossing if every not-yet-arrived input to it arrived at t.

A group's outcome is **certified at t** if its k current leaders' upper bounds lie below every other
member's lower bound. Within a certified firing pattern the network is monotone (§34.2), so the
brackets propagate layer by layer exactly (topical maps preserve order: f(x⁻) ≤ f(x) ≤ f(x⁺)). The
decision is certified at the first t where the output race is certified given certified brackets
below. This can happen **before** the first output crossing: a downstream race can be woven
speculatively with **zero rollback probability**. It is the exact form of §21.10.3's speculation.

This is the dual of §34.4. The jitter certificate protects a decision against perturbations of
the *past*; the early-commitment certificate protects it against the unknown *future*. Both are
the same monotone (topical) bracketing, applied to different uncertainty sets.

### 35.5 Tests (M37)

(i) The self-supervised commitment objective (minimise Σ −log π_w of woven races) raises the median
certified radius (§34.4) and trains at least as well as unsupervised continuity learning in E7.
(ii) On trained networks, certified early commitment decides a substantial fraction of samples before
the first output crossing, with zero rollbacks, measured by the fraction and the time saved.
(iii) Measured commitment cost per race matches σ H(π) on samples drawn from the pool
(Gumbel-perturbed forward passes).

### 35.6 Borrowed and new

Borrowed: the log-sum-exp identity T_w − F = −σ log π_w (the Gibbs variational principle), Doob
martingales, Doob–Meyer, Landauer's reading of erasure cost, interval bracketing of monotone maps.
Not found (a few searches, §32): weaving read as paying σ × surprisal, near-miss credit as the
gradient of commitment cost, the martingale/compensator split of a spiking decision, and zero-rollback
speculative commitment certified by topical bracketing.

## 36. Learning dynamics: prices must run on the faster timescale

Thresholds are dual variables of an activity constraint (§25), and credit updates the primal
variables (weights). Learning is therefore a stochastic primal–dual (Arrow–Hurwicz) iteration with
two step sizes: η for weights and κ for prices.

**Two-timescale theory (Borkar 1997, stochastic approximation).** If κ/η → ∞, the fast variables
equilibrate for each slow configuration. The prices then track the constraint (a ≈ a*) and the
weights see a constraint that is always satisfied: the iteration follows the *constrained*
problem, and its limit points are that problem's stationary points (under the theorem's regularity
conditions). If instead κ ≪ η, the constraint lags. Any systematic push of the credit on activity
(its common mode, §27; the deadline anomaly, §30.1) moves the network off the constraint faster
than the prices can restore it. §27's stationary deficit, a* − a = ηθX̄|μ̄|/(κρ̄), is the linear
picture of this: it scales with η/κ.

**Our default is on the wrong side.** η = 0.01 and κ = 0.001, so the prices are *ten times slower*
than the weights. The theory makes three predictions:

1. Rules whose credit has little common mode (random feedback, centred credit) barely stress the
   constraint, so κ matters little for them. *Observed:* crl_fa trains well at κ = 0.001.
2. Rules with a strong common mode (uncentred pivotal credit) need κ ≳ η. *Observed (debug,
   depth 3):* collapse at κ = 0.001; about 70% at κ = 0.01 and κ = 0.03, both ≥ η.
3. Past κ ≈ η, raising κ helps a common-mode rule only while the prices remain stable (a price step
   too large makes the thresholds noisy, per-batch). There is a window, roughly η ≲ κ ≲ 10η, and
   centring shifts it downward.

This replaces "homeostasis rate" as a tuning knob by a derived ordering: **dual faster than
primal**, with the gap needed set by the size of the credit's common mode. (Borrowed: two-timescale
stochastic approximation, primal–dual methods; new here: the application and the reading of §27's
collapse as a timescale inversion.)

**Test (M38):** κ ∈ {0.001, 0.003, 0.01, 0.03, 0.1} for crl_fa and uncentred crl_pivot at depth 3
(debug). Prediction: crl_fa is flat across κ; crl_pivot has a threshold near κ ≈ η = 0.01 and a
plateau, then degrades at the largest κ.

## 37. Topology from hierarchical information: group size, winners, fan-in, depth and widths

Every size so far was chosen by hand (width 400, groups of 10, k = 3 winners, dense fan-in). Here
they are derived from four constraints: the code capacity of a race, credit percolation (§17),
contraction (§31), and the multi-scale entropy of the input. These are scaling arguments, not
theorems; each ends in a test.

### 37.1 What a race group can carry, and what k buys

A group of G members with k winners emits a k-subset (plus order and times). Its set code carries
log₂ C(G, k) bits, the order at most log₂ k! more, and the times at most k·log₂(H/ρ) at timing
resolution ρ. Per spike, the set code gives b(k) = log₂ C(G, k)/k. For G = 10 that is 3.32, 2.75,
2.30 and 1.59 bits for k = 1, 2, 3, 5. Against that:

- **Robustness** falls with k: the decisive gap D_k has mean σ/k (§28), so the certified radius
  (§34.4) shrinks as 1/k.
- **Credit reach** rises with k: percolation needs a branching ratio F·p ≥ 1 with firing fraction
  p = k/G (§17).

So the minimum number of winners is set by fan-in alone:

    k · F ≥ G          (the winner–fan-in coupling)

and any extra winner costs bits per spike and robustness, buying only credit reach and, per §41,
pattern stability (A ∝ 1/√k). With dense fan-in
(F = n ≫ G), **k = 1 is optimal for bits per spike and margin** (but maximises pattern chaos, §41), and our k = 3 at F = 400 is 120× over the coupling
bound. With sparse fan-in F < G, more winners become *necessary*: k ≥ ⌈G/F⌉.

### 37.2 Fan-in from contraction and coverage

- **Contraction wants sparse fan-in.** Two consumers with random fan-in F out of n_{l−1} share
  a fraction ≈ F/n_{l−1} of their inputs, so their rows' total-variation distance is
  ≈ 1 − F/n_{l−1}, close to 1 (little contraction) for F ≪ n_{l−1}. Dense rows differ only through
  weight heterogeneity: for N(μ, μ) weights the effective fan-in is F_eff ≈ n/2, so zero-sum credit
  energy falls by ≈ 2/n per hop (§30.2).
- **Coverage bounds how sparse.** Every upstream node must be read by some consumer:
  n_l·F ≳ n_{l−1}·ln n_{l−1} (coupon collector, random wiring).
- **Depth is set by fan-in.** Without overlap, a node's receptive field grows as s_l ≈ F^l, so
  seeing the whole input takes L ≈ ln n_in / ln F layers: about 2.4 for F = 16 and 3.2 for F = 8, on
  784 inputs. With dense fan-in, one layer already sees everything, and depth adds composition but
  no receptive field.

### 37.3 Widths from multi-scale entropy

A node at depth l summarises a field of s_l inputs. Suppose the information in a field grows as
h(s) ∝ s^α, with α < 1 for redundant signals and α = 1 for independent pixels. For the layer to
carry its fields' information at c bits per node,

    n_l ≳ (n_in / s_l) · h(s_l) / c  ∝  s_l^{α−1}  =  F^{−l(1−α)}

**Widths should shrink geometrically with depth, by a factor F^{1−α} per layer**, set by one
measurable exponent of the data (M39 estimates α from latency-code patches). This uses total
information, which is an upper bound. The information bottleneck (Tishby) says label-relevant
information is smaller and falls to log₂ 10 ≈ 3.3 bits at the top, so this is the generous end.

Coverage caps the shrink rate at n_l/n_{l−1} ≥ ln(n_{l−1})/F. Both hold only if
F^{α} ≳ ln n_{l−1}, a **minimum fan-in for a consistent pyramid**, F ≳ (ln n)^{1/α}. For n = 400 and
α = 0.5, F ≳ 36.

**Measured (M39, 10k MNIST latency codes, Gaussian-channel bound on patch information at noise sd
0.01 / 0.03 / 0.1):** h(s) for 1×1 to 28×28 patches gives **α ≈ 0.97 / 0.95 / 0.93**. Total information
grows almost in proportion to area: under this bound the latency-coded digits are nearly
incompressible at the pixel level. So the total-information argument predicts **almost constant
widths** (a shrink factor F^{0.05} ≈ 1 per layer), and constant width is *not* the worst case by this
criterion. The Gaussian bound is an upper bound on information at a given covariance, so the true α
may be lower, but not by enough to reverse this at these noise levels. Any case for a pyramid must
therefore come from **label-relevant** information (the information bottleneck), not from
redundancy in the input. With α ≈ 1 the minimum pyramid fan-in, (ln n)^{1/α} ≈ ln n ≈ 6, is easy to
satisfy. Revised prediction 4: the pyramid's advantage over constant width, if any, comes from
discarding irrelevant detail and should appear only in the upper layers.

### 37.4 The hierarchical signature in weaving

By §35.1 each race commits −log π_w nats (σ × this in value). Summed over a layer's races, this is
the information the layer commits per input. Along the Markov chain Y → X → L₁ → … → output, the data
processing inequality bounds what can reach the top. An efficient hierarchy commits many
fine-grained bits low down and few at the top (about log₂ 10 bits at the output). **Prediction:**
per-layer commitment entropy decreases with depth in well-trained networks, and more steeply in
better topologies. This links weaving (§35) to the information bottleneck.

### 37.5 What this says about the current design, and tests (M39)

The current network (dense, constant width, k = 3) is the worst case of this analysis. Its rows
overlap maximally (strong contraction), its width is constant where label-relevant information shrinks (total information does not, 37.3), and its
winners are over-provisioned. Predictions:

1. ~~α < 1 on MNIST latency codes (expected about 0.5 to 0.8).~~ **Refuted:** α ≈ 0.93–0.97 (see 37.3).
2. Sparse fan-in (16 to 64) at depth 3 trains at least as well as dense, with an advantage that
   grows with depth.
3. **The coupling k·F ≥ G:** at fan-in 8, k = 1 fails (credit does not percolate) while k = 2 and
   k = 3 train. At dense fan-in, k = 1 ≈ k = 3 in accuracy, with a larger certified radius for k = 1.
4. A pyramid (400, 200, 100) ≥ inverted (100, 200, 400) at a similar synapse budget; pyramid plus
   sparse fan-in is best.
5. Commitment entropy per layer decreases with depth (37.4).

*Borrowed:* information bottleneck and data processing inequality, rate–distortion, receptive-field
growth in convolutional hierarchies, coupon-collector coverage, percolation. *New here, as far as
checked:* the winner–fan-in coupling k·F ≥ G from credit percolation, widths from a race code's
capacity and the data's entropy exponent, and the minimum fan-in for a consistent pyramid.

## 38. When to weave: optimal stopping and the absence of evidence

### 38.1 The optimal weaving rule is a price on commitment cost

The output race decides at the first *absolute* crossing: the first class potential to reach its
threshold. For choosing among M hypotheses from accumulating evidence, the asymptotically optimal
rule (MSPRT, Baum & Veeravalli 1994) stops when the leader's posterior reaches 1 − ε, that is when

    −log π_lead ≤ −log(1 − ε)

By §35.1, −σ log π_lead is exactly the **commitment cost** of weaving the leader now. So the optimal
rule reads: **weave when committing to the leader costs less than a fixed price** (−σ log(1 − ε)),
and ε trades speed against accuracy (the price of time of §30.4, now explicit).

In potentials, π_lead ≥ 1 − ε means v_lead − σ log Σ_c e^{v_c/σ} ≥ log(1 − ε): each output's threshold
is raised by the **free energy of the output pool**, a single shared inhibitory signal. The
absolute race matches this only when the pool's free energy is constant across inputs and over
time, which it is not: easy inputs raise all potentials together.

- **Prediction (M40, evaluation only, `--speed 1`):** at matched mean decision time, the relative
  rule is at least as accurate as the absolute race across the speed–accuracy frontier, with the
  largest gain on inputs where several classes rise together.
- **Local implementation:** one inhibitory unit that tracks the log-sum-exp of the output potentials
  and feeds it back to every output as a moving threshold.
- *Prior art:* MSPRT as the target of neural decision circuits, including a log-sum-exp
  implementation in the basal ganglia (Bogacz & Gurney 2007). New here: the identification with the
  weaving commitment cost and the free energy of the pool.

### 38.2 Race neurons are blind to absence, and a clock-free network can only see it relatively

The Bayes-optimal accumulator for spike-time evidence has two parts. Arrived spikes contribute
log f_c(t_i), and inputs that **have not yet arrived** contribute log S_c(t), their survival
probability under class c. A dark pixel that should be bright for class c is evidence against c
that grows with time. An excitatory race neuron gets no drive from an input that has not arrived,
and by monotonicity (§34) its forecast can only move earlier. It is **structurally blind to
absence**.

Using absence needs a reference time: "pixel j has not spiked *by now*". A fixed clock breaks
time-shift symmetry (§30.1: the clock is an anomaly). But "pixel j has not spiked although the
input's first spike came Δ ago" is **relative** and shift-invariant. So a clock-free network can
use absence exactly when it references its own events. A local implementation is class-specific
inhibition that ramps from the input's onset event (the first input spike) through the weights of
expected-but-absent inputs. This breaks monotonicity, and with it the §34 guarantees, only through
inhibition, consistent with §31.5 (inhibition as the contrast amplifier) and §34.5.

**Prediction (M40b):** onset-referenced absence inhibition improves accuracy most on class pairs that
differ by *missing* strokes (for example 7 vs 9, 1 vs 7), and costs no shift equivariance.

## 39. What σ is: noise model or smoothing?

σ plays two roles: the scale of the timing noise in the pool's model (§28) and a smoothing
temperature for learning (§21.7). They coincide only when the substrate's actual timing noise has
Gumbel scale σ_s = σ. Then the soft objective *is* the expected loss of the noisy network, and the
near-miss weights are the true flip probabilities (§28's 1/(1 + e^{Δ/σ})). Consequences:

- **In a deterministic simulation (σ_s = 0)**, σ is purely a continuation parameter. It is needed
  for a gradient but biases the objective, so it should be annealed towards 0 (M24c).
- **On noisy hardware**, σ should track σ_s, which is measurable from the network's own spacings
  (k·D_k, §28): learning should then *follow the substrate's temperature*, not a schedule.
  σ < σ_s under-weights near misses that really flip (over-confident credit); σ > σ_s over-smooths.
- **Test (M41):** inject Gumbel timing noise of scale σ_s into forward passes. The best learning σ
  tracks σ_s, and the spacing estimator recovers σ_s.

## 40. Generalisation through timing robustness

Algorithmic robustness (Xu & Mannor 2012) bounds the generalisation gap by roughly √(K/n) when the
input space splits into K cells within which the loss barely changes. §34.4's certificate supplies
the cells: a decision is constant on a sup-norm ball of radius ε* in input-time space. The input
has about d_eff active spike times in a window H, so an ε-cover has K ≈ (H/ε)^{d_eff} cells. This
gives a gap of order √(d_eff · log(H/ε*) / n). The bound is numerically vacuous for d_eff in the
hundreds, but its **trend** is a prediction: across training runs and rules, the generalisation gap
falls as the median certified radius grows. Via §35.2 (near-miss credit is the gradient of
commitment cost, which widens race gaps), this links the credit rule itself to generalisation.
**Test (M42):** gap vs median ε* across the `--certify 1` runs.

## 41. Mean-field propagation: race networks are chaotic at the level of firing patterns

The mean-field signal-propagation analysis of deep networks (Poole et al. 2016; Schoenholz et al.
2017) tracks how the difference between two inputs evolves with depth, and predicts trainable depth
before any training. Here is its analogue for race networks. It is analytic; the constants are
order-of-magnitude, and the exponent is the prediction.

### 41.1 The map

Let ρ_l be the fraction of groups at layer l whose winner sets differ between inputs x and x′.

1. **Swapped inputs.** A group whose winners differ changes about one of its k spikes (for small
   ρ), so a consumer sees a fraction q ≈ ρ_l/k of its arrived inputs swapped: present for one
   input, absent for the other.
2. **Timing effect.** A ramp crossing is T = (θ + Σ_S w_i t_i)/W_S (§34.1). Adding or removing one input
   moves it by about (w_i/W_S)|t_i − T| ~ s/F, where s is the spread of input times. The sign varies
   across swaps, so qF swaps add like a random walk: |ΔT| ~ s·√q/√F.
3. **Spread within a group.** Group members read the same inputs with different weights, so their
   crossing times differ by about s·√(2/F). The normalised perturbation is z = |ΔT|/spread ~ √(q/2):
   **fan-in cancels.**
4. **Winner changes.** The gap between the k-th and (k+1)-th crossing has a finite density g at zero
   (§28), so the probability that the winners change is ≈ g·z for small z.

Together,

    ρ_{l+1}  ≈  A · √ρ_l ,     A ≈ g / √(2k)

### 41.2 Consequences

1. **There is no ordered phase.** Φ(ρ) = A√ρ has infinite slope at 0, so the "identical inputs"
   state is unstable for every weight scale. In tanh networks the ordered/chaotic phase depends on
   the weight variance. In race networks, **pattern-level chaos is structural**: it comes from the
   discreteness of selection. This is consistent with §34: firing *times* are non-expansive, and
   all amplification happens through the discontinuous winner swaps, which is also where the
   jitter certificate's boundaries lie.
2. **Doubly-exponential decorrelation.** Iterating gives
   log ρ_l ≈ (1 − 2^{−l})·log A² + 2^{−l}·log ρ_0. A tiny input difference (ρ_0 = 10⁻⁴) becomes a
   macroscopic pattern difference within 3–4 race layers, and then saturates at the fixed point
   ρ* ≈ min(A², ρ_max). **Beyond about log₂ log(A²/ρ_0) layers, the firing pattern no longer
   reflects input similarity.** This is a forward-pass limit on useful depth, independent of how
   good the credit is. It adds to the backward limits of §30.2 and §31.
3. **The certificate is the threshold of the cascade.** For jitter below ε* nothing flips (ρ = 0
   exactly, §34.4). The first flips occur with probability ∝ ε, and the √ law then amplifies them.
4. **More winners dampen the chaos.** A ∝ 1/√k: with more winners, a swap changes a smaller
   fraction of a group's output. **This revises §37.1.** k = 1 maximises bits per spike and the
   robustness margin, but it also maximises pattern chaos (A is √3 larger than at k = 3). The choice
   of k is a three-way trade-off: information per spike, margin, and pattern stability.
5. **The remedy is smooth codes.** What matters downstream is not *that* a winner was swapped for a
   near-loser, but how different their outgoing weights are. With r the correlation between the
   outgoing weight vectors of competitors that nearly tie, q_eff = q·(1 − r) and A_eff = A·√(1 − r).
   A network whose near-tied competitors project similarly (a **topographic** code, as in
   self-organising maps) suppresses the chaos and approaches an edge of chaos. Temperature σ > 0
   does not remove the √: it comes from the random walk over discrete swapped inputs, not from the
   sharpness of the race.

### 41.3 Predictions (M43)

Measured by `--certify 1` as `rho_per_layer` at jitter ε ∈ {0.001, 0.003, 0.01, 0.03}:

(i) Before saturation, **log ρ_{l+1} ≈ const + ½ log ρ_l**: slope ½ across layers and across ε.
(ii) Saturation at depth 5 (`cert_sg_d5`) to a common ρ* for all ε.
(iii) ρ_l is larger for k = 1 than for k = 3 at matched depth (`topo_dense_k1` vs `topo_dense_k3`),
with a ratio of A near √3 before saturation.
(iv) Trained networks have a smaller A than untrained ones if learning builds partially topographic
codes. A direct check is the outgoing-weight correlation r of frequently near-tied pairs.

*Borrowed:* the mean-field propagation method. *New here, as far as checked:* the √ map for
discrete race selection, the absence of an ordered phase, A ∝ 1/√k, and topographic codes as the
remedy.

## 42. The exactly solvable race: decision time carries no information about the winner

Take one race whose candidates' crossing times are T_n = μ_n + σ·G_n, with G_n i.i.d. standard Gumbel
(minimum form), the model under which the pool is exactly Plackett–Luce (§21.4, §28). Three facts
follow in closed form.

1. **The decision time is Gumbel, located at the free energy.** min_n T_n ~ Gumbel(F, σ) with
   F = −σ log Σ e^{−μ_n/σ} (§35). So E[decision time] = F − γσ (γ = 0.5772…, Euler–Mascheroni), and its
   variance is π²σ²/6, independent of the input.
2. **Who wins is independent of when.** The argmin and the min of independent Gumbels are
   independent (the same property as for races of exponential clocks). **Within one input, an early
   decision is exactly as likely to be right as a late one.** Across inputs, easy inputs (low F) are
   decided earlier, which is why E2 saw decision time fall with evidence strength. But at a fixed
   input, time is uninformative. So a speed–accuracy trade-off can only come from evidence that keeps
   *arriving* (the accumulation of E2 and §38), never from waiting within one noisy race. A measured
   dependence of accuracy on decision time at fixed input measures how far the substrate's race
   departs from Gumbel structure (test M44).
3. **Pricing time is a reweighting of near-miss credit.** With loss L = −log π_y + λ·E[T] (cross-entropy
   plus a price λ on decision time, §30.4 and M5), the exact gradient with respect to the candidates'
   locations is

       ∂L/∂μ_n = δ_{ny}/σ − π_n · (1/σ − λ)

   The target is pulled earlier as before. Competitors are pushed later by **(1 − λσ) times** their
   usual near-miss credit: at λσ = 1 they are left alone, and beyond it they are pulled earlier too,
   so everything speeds up. The credits now sum to λ, not 0: the time price is exactly the clock's
   anomaly term of §30.1, now as a controllable source. **The speed–accuracy trade-off is a single
   knob on the near-miss credit, (1 − λσ).**

*Borrowed:* Gumbel extreme-value facts and the min/argmin independence of Gumbel and exponential
races. *New here:* the reading for race networks (decision time as free energy, time pricing as a
(1 − λσ) factor on near-miss credit, and time as the clock anomaly's source).

## 43. Continual learning is tracking, and error-gated rules pay for noise

E17's continual race did no better than the same race frozen after a week (−0.2 points). This is
what adaptive-filter tracking theory (Widrow et al. 1976; Benveniste et al. 1990) predicts when
drift is slow relative to noise, and race rules make it worse in a specific way.

**Tracking trade-off.** Let the best weights drift like a random walk with variance q per update,
and let each label carry irreducible noise r. An online rule with step η settles at an excess loss of
about

    η·r/2  (noise the rule keeps chasing)   +   q/(2η)  (lag behind the drift)

minimised at **η\* ∝ √(q/r)**. A frozen model instead accumulates a lag of about q·t over time t. So
continual learning beats freezing only after t ≳ 2√(r/q). With slow drift and high noise, η\* is
tiny, and a fixed η = 0.01 mostly tracks noise.

**Why error-gated race rules are hit hardest.** Race rules learn from errors and near misses (the
source of their efficiency in E4 and E5). In a noise-dominated task (E17's error rate is about 41%,
nearly all irreducible), almost every update is driven by noise: the effective r scales with the
Bayes error, and the "learning work tracks errors" property turns from an asset into a liability.

**The prescription follows from the weaving theory.** Under a calibrated model, the pool's own
surprisal is expected: by §35.1 the mean commitment cost of the teacher's branch is σH(π). The
**excess surprisal**, −log π_y − H(π), has mean zero exactly when the model is calibrated, and turns
positive when the world has changed. So:

- **Change-gated plasticity:** scale η by a running test of excess surprisal (or, as a proxy, of the
  recent error rate above its long-run baseline). The rule then learns fast after a regime change and
  almost not at all under stationary noise. It is the "unexpected uncertainty" of Yu & Dayan (2005)
  and variable-step-size filtering, derived here from the martingale/compensator split of §35.3:
  learn from the compensator's *deviations*, not from its expected increments.

**Predictions (M45, E17 exploratory; not preregistered):**

(i) Accuracy rises as η falls from 0.03 towards η\*, and the best η is well below 0.01.
(ii) The change-gated rule (`--gate 1`) matches or beats the best fixed η without knowing it, with a
low mean gate on stationary days.
(iii) On streams with abrupt regime changes (E7's class-blocked streams), the gate beats both frozen
and fixed-η learning.

**First test (E17, exploratory; design flaw found).** Accuracy by step size: η = 0.001 0.583, 0.003
0.593, 0.01 0.593, 0.03 0.578; change-gated (error-rate proxy, mean gate 0.15) 0.578; frozen 0.595. Too
large a step hurts, as predicted, but no step size beats freezing, and the gate is 1.7 points worse
than frozen. **The test was confounded:** the step size and the gate also applied during the pilot week,
when the network learns from scratch, so small steps and a mostly closed gate starved initial learning.
§43 is about tracking after convergence. Coverage also varied from 68% to 95% with η, which makes
accuracy among decided episodes hard to compare. A corrected run (full-rate learning in the pilot week,
tracking variants only afterwards, gate statistics warmed up in the pilot) is queued. Prediction (ii)
stands untested; the evidence so far is that E17's three weeks show no drift that learning can use.

*Borrowed:* LMS tracking theory, variable step-size filters, expected vs unexpected uncertainty. *New
here:* the Bayes-error scaling of noise injection for error-gated race rules, and the gate derived
from excess commitment cost.

## 44. The output rule is preconditioned gradient descent (exact)

A partial answer to "which objective do the rules descend?" (§33, gap 3), for the output layer.

**The rule, as implemented.** A node's synapse update is Δw_i = η·c_n·x_i, where c_n is the node's
credit and x_i the charge that input i injected by the node's fire or cancel time: x_i = (τ_n − t_i)⁺
for ramp synapses (`RaceNet._elig`), with τ_n the node's crossing time if it fired and the
cancellation time if not.

**The exact derivative.** For a ramp node that fired, ∂T_n/∂w_i = −(T_n − t_i)/A_n over arrived inputs
(§30.3). So for a fired node

    Δw_i = η c_n (T_n − t_i)  =  −η · A_n · c_n · ∂T_n/∂w_i

and with c_n = −∂L/∂T_n (credit on "fire earlier") the update is **exactly −η A_n ∂L/∂w_i**: gradient
descent, preconditioned per node by its urgency A_n > 0. With conservation on, the credit on output times
is the Plackett–Luce gradient (δ_ny − π_n)/σ up to the residue-versus-time units of Δ (§35.2), so the
output layer does preconditioned gradient descent on the cross-entropy over its own race.

**Near misses get a truncated gradient, still a descent direction.** A cancelled node is updated with
charge up to its cancellation time t_k < T_n, which omits the inputs that would have arrived before its
projected crossing (the woven past, §21.9). Both vectors are non-negative on the same inputs and the
truncated one has smaller support, so their inner product is positive: **each node's update has positive
cosine with its exact gradient**, always.

**What follows.**

- M3's measurement, that the output rule is well aligned with the true gradient (+0.36 to +0.82) while
  hidden credit is not, is a theorem for the output layer up to units and truncation. The hidden layers
  have no such guarantee: their credit comes through random feedback, and the gap in understanding
  sits there.
- The preconditioner A_n equalises learning in time units: the induced change in a node's crossing time,
  ΔT_n = −η c_n Σ_i (T_n − t_i)²/A_n, does not shrink for high-drive nodes as a plain gradient step would.
- Standard results for preconditioned stochastic gradient descent (a positive diagonal preconditioner)
  then apply within each piece of the piecewise-smooth loss. Convergence across pieces, where the race
  order changes, is not covered; that is where §31–41's boundary phenomena live.

## 45. Locality of weaving, and what transfers from quantum physics

### 45.1 Weaving is local; there is no global collapse

A group weaves using only the spikes that have reached it: its causal past. There is no global "now".
The woven events at any moment form a **down-closed set of the causal order**: a consistent cut, in
the language of distributed systems (Lamport clocks, Chandy–Lamport snapshots). The network's
configuration space is the lattice of such cuts. Far-apart regions of a large network weave
independently, at their own times, and become correlated only when spikes join their causal cones.

From inside a node, everything not yet in its causal past is still a distribution: the pool
*conditioned on what that node has received*. What looks like "collapse" from its point of view is
conditioning. That is the classical counterpart of Everett's relative states. Nothing is objectively
indeterminate: the pool is classical probability, with no amplitudes, no interference and no
entanglement, and all correlations come from common causes.

### 45.2 Decoherence: robust outcomes are the pointer states

The analogy that does carry structure is decoherence and einselection. In quantum physics an outcome
becomes effectively classical when noise from the environment cannot flip it, and it becomes
objective when many independent records of it exist (Zurek's quantum Darwinism). Here:

- **Pointer states are certified outcomes.** A race whose gap D_k exceeds 2ε cannot be flipped by
  timing noise of size ε (§34.4). Under the extreme-value null (§28), P(D_k > 2ε) = e^{−2kε/σ}, so
  outcomes become noise-proof exponentially in gap/σ: a "decoherence rate" k/σ per unit of timing
  noise.
- **Objectivity is redundancy of records.** A race outcome is "objective" for the rest of the
  network when it is copied into many downstream events: its fan-out spikes. The mutual information
  between the outcome and small fragments of the downstream network (the quantum-Darwinism
  redundancy measure) is a well-defined information quantity for race networks. It should predict
  which hidden decisions the output can rely on. (Untested.)

### 45.3 A soft light cone (a Lieb–Robinson-type bound)

In local quantum systems, Lieb–Robinson bounds show influence outside a light cone is exponentially
suppressed. Race networks have a **strict** light cone for realised influence: spikes are the only
carriers. Counterfactual influence at σ > 0 (the pool's weight on histories that did not happen)
leaks outside the realised cone, but only through near misses, each costing a factor ≤ e^{−Δ/σ} (the
Plackett–Luce flip probability 1/(1 + e^{Δ/σ}), §28). Along a chain of m near misses with gaps
Δ₁…Δ_m,

    counterfactual weight  ≤  exp(−(Δ₁ + … + Δ_m)/σ)

The total gap along a path plays the role of a Euclidean **action**. Summed over paths with branching
F_eff near misses per hop, the counterfactual influence at hop distance m is bounded by b^m, where
b = F_eff · E[e^{−Δ/σ}]. **Credit percolation (§17) is this bound's threshold**: b < 1 means
counterfactual influence decays exponentially outside the realised cone, and b > 1 means it
percolates.

One closed form: under the extreme-value null, the closest loser's gap is Exp(σ/k), so
E[e^{−D/σ}] = k/(k + 1), independent of σ. The closest near miss alone contributes at most k/(k + 1)
per hop.

### 45.4 Semiclassics: history repair is the instanton

The least-action counterfactual path (the smallest total gap that changes the outcome) dominates the
sum over histories as σ → 0. That is the **instanton** of the Euclidean path integral (§23.1).
History repair (M18: the cheapest verified single-event change) is exactly the one-instanton
approximation; holistic credit (§14) is the full sum; and near-miss credit is the one-loop
(single-flip) term. This orders the three learning rules by semiclassical order and says when each
suffices: repair when one path dominates (gaps well separated relative to σ), the full sum when many
paths have similar action.

### 45.5 Tensor networks for the pool

The unravelled pool is a layered product of Plackett–Luce factors (§21.4): a tensor network whose
bond dimension is the number of plausible winner sets per group. Tensor-network contraction with
truncation (matrix-product methods from many-body physics) computes the pool's marginals with
controlled error, and the beam of M19/M20 is the lowest-bond-dimension truncation. This is a route
to exact holistic credit at controllable cost. (Untested.)

### 45.6 What does not transfer

Superposition with interference, entanglement and Bell-type correlations have no counterpart: the
pool is a classical measure over histories. Language such as "the wavefunction of the network" is a
metaphor for this measure (§21.3); dequantization is the precise statement.

**Tests (M46):** (i) the rate at which race outcomes become noise-proof follows e^{−2kε/σ} (from the
certificate runs' gap distributions); (ii) redundancy of records predicts which hidden decisions the
output depends on; (iii) counterfactual credit reach per hop is below 1 exactly when b < 1.

## 46. Flat supervision: why depth re-encodes instead of composing

With direct random feedback, hidden credit at layer l is δ^l = (e·B_l) ⊙ elig_l, where e is the output
error (with conservation: +1 on the target and −ν_c on competitors, Σν = 1) and B_l is a fixed random
matrix. Conditioned on the label y, E[δ^l_n | y] ≈ B_l[n, y] − E_ν B_l[n, ·]. A node is therefore pushed
to fire earlier for the classes its feedback row favours: its **template classes** (the DRTP picture,
Frenkel et al. 2021; M3's "template mechanism").

**The depth consequence.** Layer l's target is a function of the label and B_l only, not of anything
the layers above do with its output. Every layer independently fits random label templates from its
own input. Layer 1 already maps the input to (noisy) functions of y, so deeper layers see a code that is
close to a function of the label, and can only **denoise or re-encode** it, not add features that
layer 1 discarded. By the data processing inequality the label information cannot grow with depth, and
§31 (contraction) and §41 (pattern chaos) make it shrink. This is a mechanism for the observed decline
with depth (0.960 / 0.952 / 0.941 with conservation), separate from credit strength: better credit
(conservation, counterfactual) slows the decline but cannot turn it into composition.

**What composition would need.** Layer l's credit must depend on how layer l + 1 *uses* it:
- local feedback carried through the layer above (`--feedback local`, E15),
- pivotal credit through the real weights (§26; stabilised by §27–30), or
- targets defined by the layer above (target propagation).
Only these give deeper layers a reason to build features that are not label templates.

**Predictions (M47, linear probes per layer, `--probe 1`):**

(i) Random feedback: probe accuracy is flat or falls from layer 1 to layer 3.
(ii) Frozen random layers: it falls fastest (only contraction and chaos act).
(iii) Local layer-to-layer feedback and pivotal credit: probe accuracy rises from layer 1 to layer 3, or
falls less. That is the signature of composition.
(iv) Hidden-layer class selectivity under random feedback tracks the argmax of each node's feedback row
(template class), more closely in deeper layers.

*Prior art:* DRTP (Frenkel et al. 2021); direct feedback alignment's difficulty with hierarchical
features (e.g. Bartunov et al. 2018; Launay et al. 2019). *New here:* the race-network form, its
link to the depth decline alongside §31 and §41, and the probe predictions that separate composition
from re-encoding.

## 47. A residual event stream: identity paths for race networks

**Why race networks need identity paths more than transformers do.** A ReLU or attention layer can
learn an approximate identity (W ≈ I), so residual connections mainly ease optimisation there. A race
layer cannot: a k-of-G race re-decides every input and silences most nodes, an information bottleneck
by construction. Three mechanisms then compound with depth: contraction of timing contrast and
credit (§31), pattern-level chaos (§41), and flat supervision (§46).

**The construction.** Layer l reads the raw input and the spikes of every earlier hidden layer (a
growing event stream), and the output reads the whole stream (`--residual 1`). No layer replaces what
came before.

**What the theory predicts.**

1. **Forward:** the stream contains every earlier layer's spikes unchanged, so its Dobrushin factor
   is exactly 1 and no depth can destroy information already present. Accuracy should not fall with
   depth.
2. **Composition:** a deep layer only has to *add* useful features (the residual reading), which
   removes §46's objection that depth can only re-encode. Accuracy may rise with depth.
3. **Backward:** under random feedback every layer already receives credit directly, so identity paths
   do not change credit reach. With real-weight (pivotal) credit they would give undiminished paths,
   the analogue of gradient flow through transformer residuals. (Untested: the pivotal variants
   need the stream's column slicing.)
4. **Cost:** synaptic work grows with depth, since each layer reads the whole stream. Sparse fan-in
   (E18) bounds it.

**First result (full length, seed 0): plain skips made depth 3 worse,** 0.898 vs 0.937 for the plain stack,
at twice the synaptic events. **Why, specific to time codes:** an identity path is *faster* than a computed
path. Raw pixels (bright ones spike at t = 0) reach the output at once, hidden features only after each
layer integrates, so the output race commits on the shallow evidence before deep features arrive: E6
round 1's hasty decisions, reintroduced by the skip. In a race, **a skip connection is not neutral: it
gives the shallowest information a head start.**

**Fix derived from §22.1: delay-matched skips** (`--residual 2`). Delaying a path by a constant loses no
information (time-shift equivariance) but removes its head start. Each source's spikes enter the stream
delayed by the running mean latency of the layers they bypass. Queued: depth 1, 3, 10, 20 on MNIST and
E19 (Random Hierarchy Model).

**Result (depth 3, full length, seed 0):** delay-matched skips 0.915, plain skips 0.898, no skips 0.937.
Delay matching removes most of the head-start penalty, but skips do not rescue depth under random-feedback
learning, consistent with §46: the limit is the learning rule, not information loss. The exact-gradient
ceiling test (E20) decides whether the architecture can use depth at all.

**Test (M48, full length):** residual vs plain stacks at depths 1, 3, 5 (credit conservation on).
The plain stack declines 0.960 → 0.952 → 0.941 (depths 1–3). The prediction is that the residual stack
does not decline, and ideally improves.

## 48. The entropic k-winner race is a Fermi–Dirac distribution

**The problem it addresses.** Under exact spike-time gradients, hard k-of-G cancellation costs about 2
points (E20, depth 2: 0.930 at k = 3 vs 0.951 with no cancellation). A cancelled node's spike is absent, so
its gradient is zero except through a surrogate.

**The dequantized race.** Give each member of a group a soft membership m_n ∈ [0, 1] with Σ_n m_n = k,
chosen to maximise Σ m_n·(−T_n) + σ·Σ h(m_n), with h the binary entropy. The stationarity conditions give

    m_n = 1 / (1 + e^{(T_n − μ)/σ}),        μ fixed by Σ_n m_n = k

**the Fermi–Dirac distribution**. Winners behave as fermions: each node fires at most once, the analogue of
Pauli exclusion. The multiplier μ is the group's **chemical potential**, which is also its price in the
sense of §25 (a dual variable for the capacity k). As σ → 0, μ tends to the k-th crossing time, and m tends
to the hard top-k: the race is the zero-temperature limit.

**The Jacobian is closed form.** With f_n = m_n(1 − m_n) and ∂μ/∂T_j = f_j / Σ_l f_l,

    ∂m_n/∂T_j = (f_n/σ) · (f_j / Σ_l f_l − δ_nj)

so gradient reaches every member of the group, weighted by its occupation fluctuation f. Nodes deep inside
the winners (m ≈ 1) or far outside (m ≈ 0) get almost none; the gradient concentrates at the Fermi level,
the near misses. This is §4's boundary term in exact form.

**A training scheme follows.**
- **Train on the soft race:** every member spikes at its crossing time with amplitude m_n (graded
  spikes, in training only).
- **Anneal σ to zero** (continuation, §21.7), so training ends at the hard race.
- **Infer with the hard race:** the actual event network, at the actual cost.
- **Homeostasis sets μ's drift:** the thresholds are the slow part of the chemical potential.

**Prediction (M49):** Fermi–Dirac training recovers most of the ~2-point cancellation cost under exact
gradients. At depth 2 with k = 3, the hard evaluation should approach 0.951 (the no-cancellation network) from
0.930. Test: `e21_soft.py`, gradient-checked.

**Result (depth 2, k = 3, full data, 2 epochs, evaluated as the hard race):** 0.942, against 0.930 trained
hard and 0.951 with no cancellation: about 60% of the cancellation cost recovered. Partly confirmed. Note the
direction decision (ROADMAP, 2026-09-26): soft-race training is a training-time technique, not local learning.

*Prior art:* entropic (binary-entropy) relaxations of top-k give exactly this sigmoid-with-threshold form in
the differentiable sorting and top-k literature (e.g. soft top-k via optimal transport, Xie et al. 2020;
differentiable ranking, Blondel et al. 2020). *New here:* its identification with the race at temperature σ,
the chemical potential as the homeostatic price, and annealed soft-race training of an event network that runs
hard at inference.

## 49. The affine time gauge: temporal collapse and temporal normalisation

### 49.1 A second exact symmetry: dilation

Within a fixed causal set, a ramp neuron crosses at T = (θ + Σ w_i t_i)/A. Scaling every input time and the
threshold by c > 0 and shifting the times by b,

    T(c·t + b; c·θ) = (cθ + Σ w_i (c t_i + b)) / A = c·T(t; θ) + b

and since c > 0 preserves every order, the causal sets and race winners are unchanged. **A race layer is
equivariant under the affine group of time**, provided thresholds scale with the dilation. Shift was §22.1;
dilation is new. Consequence: a layer's timing can be re-centred and re-scaled, with the next layer's
thresholds rescaled to match, **without changing the function the network computes**. Normalising a layer's
timing is a gauge choice, the time-domain counterpart of the scale invariance behind batch and layer
normalisation. The fixed horizon breaks dilation, as the deadline breaks shift (§30.1): the clock is again the
only anomaly.

### 49.2 Temporal collapse

A race neuron outputs a weighted mean of its input times plus θ/A. Averaging shrinks spread: across a layer,
output times vary by about s_in/√F_eff plus the threshold heterogeneity (the forward contraction of §31).
Without renormalisation, the spread of spike times falls geometrically with depth, and this hurts twice:

- **Expressivity:** decisions ride on ever-smaller time differences, amplifying noise sensitivity and
  pattern chaos (§41).
- **Gradients:** the exact weight gradient is ∂T/∂w_i = (t_i − T)/A (§44), proportional to the spread of
  the node's input times. **Deep weight gradients shrink with the timing spread even under exact
  backpropagation.** Relative (Adam-style) steps restore the magnitude but not the signal-to-noise ratio.

This is our account of the vanishing and exploding gradients reported for deep first-spike networks
(Stanojevic et al. 2024). Their remedy, an initialisation that keeps each layer's timing scale matched
(the ReLU-equivalent mapping), fixes the dilation gauge at initialisation only.

### 49.3 Temporal normalisation

Fix the gauge throughout training: map each layer's output times affinely to a fixed mean and spread
(t' = 0.3 + 0.15·(t − m_l)/s_l, inside the horizon), with m_l and s_l **running** statistics, a slow per-layer
gain like homeostasis, not batch statistics (§ async design principle). By 49.1 this adds no restriction on
the function class; it only reconditions learning. In hardware it is a per-layer delay plus a time-scale
(ramp-slope) adjustment.

**Predictions (M50):** (i) without normalisation, the spread of fired times shrinks with depth, and so does the
weight-gradient norm of deep layers; (ii) with it, exact-gradient accuracy at depth 4 is at least that at
depth 2, and the depth-2 result (0.951, no cancellation) improves. Test: `e20_exact.py --tnorm 1`.

## 50. Why error-gated race learning forgets boundedly, and softmax SGD does not

The first theory aimed at the project's own niche (ROADMAP, 2026-09-26): continual learning.

### 50.1 The near-miss rule is ultraconservative

On a labelled sample, the output rule with credit conservation (§22.3) does nothing if the target wins with
margin; otherwise it promotes the target (+1) and demotes only competitors that came within the eligibility
window, with weights normalised to sum to −1. For the step-synapse race with simultaneous inputs (E4's
setting), a node's potential is v_c = ⟨w_c, x⟩ and the race picks the argmax. The update is then exactly an
**ultraconservative multiclass algorithm** (Crammer & Singer 2003): it touches only the target and the
competitors in the "error set", with competitor coefficients summing to −1. Their theorem gives a mistake
bound: on data separable with margin γ inside radius R, the number of updates is at most about 2(R/γ)²,
**independent of how long training continues**. For ramp synapses the same structure holds within a causal
set (§44: the update is urgency-preconditioned gradient), so the bound transfers to the linearised piece.

### 50.2 Bounded interference in class-incremental learning

Train on task A, then task B (new classes). An old class c's row changes during B only on updates where c is in
the error set (a near miss), so

    ‖Δw_c‖ during B  ≤  η·R·N_c(B),     N_c(B) ≤ M_B ≤ 2(R/γ_B)²

and the new classes' rows also change only on B's mistakes. **Forgetting caused by B is bounded by B's mistake
count, and stops growing once B is learned**, however long B continues.

### 50.3 Softmax cross-entropy with SGD keeps interfering

Cross-entropy SGD updates every competitor on every sample by η·p_c(x) > 0, and the target row on every sample.
On separable data the loss and p_c decay only like 1/t while weight norms grow like log t (the implicit-bias
dynamics of Soudry et al. 2018). The accumulated push on old classes is Σ_t η·p_c,t ~ η log T, and the new
classes' rows keep growing. **Forgetting keeps growing, logarithmically, with time on the new task.** Hidden
layers add representation drift, which is again error-gated in the race and not in SGD.

### 50.4 A sharp side prediction: homeostasis breaks the guarantee

Homeostasis updates thresholds on *every* frame, label or not, so it is not conservative. Its drift grows with
time on task, not with mistakes. **In race networks, homeostasis should be a leading source of forgetting.**
Turning it off (or gating it by error) should flatten forgetting further.

### 50.5 Predictions (M51, E23: class-incremental split-MNIST, 1k / 4k / 16k frames per task)

(i) Race forgetting is roughly flat in frames per task; MLP–SGD forgetting grows with it (roughly log).
(ii) The race without homeostasis forgets less than with it, with the gap growing with time on task.
(iii) The race's number of weight updates per task saturates; the MLP's grows linearly.

*Borrowed:* ultraconservative online algorithms and their mistake bounds (Crammer & Singer 2003); the
implicit-bias dynamics of cross-entropy (Soudry et al. 2018). *New here, as far as checked:* the
identification of the conserved near-miss race rule as ultraconservative, the resulting O(M_B) vs O(log T)
forgetting contrast, and homeostasis as the non-conservative leak.

## 51. The class prior belongs in the prices: why the race forgets old discriminations

*Written 2026-09-26 after E23's first readout, before the ablations below were run.*

### 51.1 What E23 showed first

With a single head, both the race and the MLP forget everything (forgetting 0.97 / 0.98 at 1k frames per task):
the newest classes win on every input. That is the known task-recency bias of class-incremental learning, and it
saturates the metric, so §50's predictions cannot be tested on it. The informative readout is **task-aware**
accuracy (only the task's own classes may win; for the race, the other outputs' thresholds are put out of reach
and the race is re-run). There the prediction reversed: race 0.17 forgetting, MLP 0.03.

### 51.2 The gap in §50.2

The mistake bound counts every update of B, including those where an *old* class wins or nearly wins on a B
input. In class-incremental learning that is most of B's early updates, and each one lowers an old class's
weights on B's features. §50.2 bounds ‖Δw_c‖, but not *where* the change goes: it goes onto shared features,
which are what separate c from its task-mate c′. So a small norm bound does not protect discrimination.

### 51.3 A prior channel absorbs the shift

When the label prior shifts (a new block), the loss-optimal response is mostly a per-class offset. SGD has one: the
bias gradient is p − y, whose mean over the block is exactly the prior mismatch, and a uniform offset across a
task's classes leaves their within-task ranking unchanged. The race's output thresholds are fixed (the bias column
is zeroed), so the whole prior shift is written into feature weights. **In time, a class bias is a price:** a
threshold, the chemical potential of §48. Let the teaching signal move it, θ_c ← θ_c·exp(−η_θ s_c). A few price
updates then give the old classes the margin that ends B's near misses (ultraconservative updates stop at margin),
so fewer updates reach old-class weights.

### 51.4 Predictions (M52, E23 at 1k frames per task)

(i) Race with learned prices (η_θ ∈ {0.003, 0.01, 0.03}): task-aware forgetting falls toward the MLP's; single-head
forgetting stays near 1 (a prior channel cannot fix recency; that needs a balanced prior or replay).
(ii) Output-only learning (frozen hidden) forgets less than full learning, and removing homeostasis helps only a
little: the leak of §51.2 is at the output, not in homeostasis.
(iii) Old-class weight updates during later blocks drop with prices.

*Borrowed:* task-recency bias and bias correction in class-incremental learning (Wu et al. 2019, BiC; Masana et
al. 2022 survey). *New here, as far as checked:* the reading of the class prior as a race price, and the gap
between a norm bound and discrimination in §50.

### 51.5 First results (E23, 1 seed, validation)

Task-aware forgetting, race vs MLP: 0.167 vs 0.030 at 1k frames per task, 0.158 vs 0.021 at 16k. Output-only
race (frozen hidden): 0.192; the same without homeostasis: 0.183.

- §50 (i), race forgetting flat in time on task: **holds** (0.167 → 0.158). MLP forgetting growing like log T:
  **refuted**; it falls (0.030 → 0.021). The implicit-bias argument of §50.3 is about norms, and, like §50.2, says
  nothing about where the change goes.
- §50 (ii), homeostasis as the leak: **not supported** with a frozen hidden layer (0.183 vs 0.192).
- §51 (ii), the leak sits at the output: **confirmed**. A purely ultraconservative linear race on fixed features
  forgets more than the full network. The race is flat in time, as the mistake bound says, but at a level 5–8×
  the MLP's, because the bounded number of updates lands on the wrong coordinates.

## 52. Grokking in the race: sleep turns an absorbing memorization into a phase transition

*Written 2026-09-26, before E24's race runs were read. Setting: E24, (a + b) mod p from two one-hot input
spikes, a fraction of the p² pairs for training. A dense MLP with AdamW groks there on this CPU (p = 31, half
the pairs: train 1.0 by step 1k, test 0.00 until ~3k, 0.87 at 20k, weight norm falling).*

### 52.1 Two circuits, two costs

A network that fits n training pairs can do it in two ways.

- **Memorization is memory indexing.** Each training pair gets its own hidden winner pattern, and the output
  row of its label is tuned to that pattern: a lookup table keyed by the pair. The table's norm grows with
  the number of entries: to give n patterns margin γ with near-orthogonal k-winner codes needs roughly
  ‖W‖²_mem ≈ c_mem · n/γ².
- **Generalization is a relation.** Modular addition is addition of phases (a ↦ e^{2πia/p}); the known
  generalizing circuit uses a few Fourier frequencies (Nanda et al. 2023), with a norm C_gen/γ² that does
  **not** grow with n.

The minimum-norm solution with margin γ is therefore the lookup table below a critical data size
n* ≈ C_gen/c_mem and the relation above it (the "circuit efficiency" account of Varma et al. 2023, in race
units). This is the manifesto's claim in miniature: a lookup is the spatial solution, a relation the
compressed one, and only a pressure towards small norm makes the system prefer the relation.

### 52.2 Without sleep the race cannot grok (a consequence of §50)

The output rule is ultraconservative (§50.1): it updates only on mistakes and near misses (margin window
`margin`). Once every training sample wins with margin, **the rule is inert**: the memorizing solution is an
absorbing state. Nothing moves the weights towards smaller norm, so the relation never takes over, whatever
the training time.

Softmax SGD differs: every sample always updates by p_c > 0, and the implicit bias of cross-entropy
(Soudry et al. 2018) drifts the weights towards the max-margin direction at rate ~1/log t. That is why dense
networks can grok slowly even without weight decay, and why weight decay speeds it up. **The same property
that bounds the race's forgetting (§50) forbids its grokking.**

### 52.3 Sleep is the leak that restores the drive

Sleep downscaling (the synaptic homeostasis hypothesis, Tononi & Cirelli 2003/2014): after each waking epoch,
every weight shrinks, w ← (1 − λ)w. Waking and sleeping together minimise

    λ/2 · ‖W‖²  +  Σ_samples hinge(margin − Δ(x, y))

by stochastic subgradient steps: the waking rule is the hinge subgradient (it fires only inside the margin
window, and §44 makes it exact up to a per-node positive preconditioner at the output), sleep is the L2 step.
That is **Pegasos** (Shalev-Shwartz et al. 2007), which converges to the regularized max-margin solution. By
§52.1 that solution is the relation when n > n*. So the race groks when it sleeps, and only then.

Sleep is non-conservative, like homeostasis (§50.4): it erodes margins everywhere, not only where there
were errors. **Sleep trades the forgetting guarantee for generalization.** The two E23/E24 axes are one
dial.

### 52.4 Timescales and the phase diagram

Control parameters: the data fraction n/p², the sleep strength λ, and the waking rate η.

- **Grokking time.** The memorizing part of W is defended only while it carries margin that the relation does
  not already provide. Once the relational component can carry the margin, the table decays geometrically
  during sleep: t_grok − t_fit ≈ (1/λ) · log(‖W_mem‖/‖W_gen‖). **Delay ∝ 1/λ**, and larger initial norms
  (higher `init_frac`) lengthen it logarithmically (Omnigrok's "LU mechanism": grokking needs the initial norm
  to be above the generalizing one).
- **Too much sleep underfits.** The waking rule restores at most η·(rate of margin violations)·R per epoch,
  while sleep removes λ‖W‖². When λ exceeds roughly η·R/‖W_gen‖ even the relation cannot hold its margin: the
  network neither memorizes nor generalizes.
- **Too little data never groks.** Below n* the minimum-norm solution *is* the table; sleep only makes it more
  efficient. Reducing the data after grokking should undo it ("ungrokking", Varma et al.).

So there are three phases: memorization (λ → 0, or n < n*), grokking (intermediate λ, n > n*), and
confusion (λ large).

### 52.5 Order parameters the race makes visible

The event substrate exposes quantities that dense networks hide:

- **Code sharing.** Memorization gives nearly one hidden winner pattern per pair; the relation gives patterns
  shared across the pairs with equal a + b (their fibre). The number of distinct hidden codes among training
  pairs, or the mutual information I(code; pair) − I(code; label), should drop sharply at grokking (a
  neural-collapse-like order parameter, measured in spikes).
- **Latent heat in plasticity.** The rule is error-gated, so plasticity events per epoch are a direct readout
  of how much margin is being re-carved. Prediction: after the fit, plasticity falls, then **peaks at the
  transition** as the table is dismantled and the relation built, then falls to a low floor. A peak in
  susceptibility at a phase transition, measurable for free.
- **Certified radius.** Max-margin solutions have larger margins in time units, so the median certified
  jitter radius ε* (§34.4) should jump at grokking, tying §40's generalisation-through-timing-robustness to
  this transition.

### 52.6 The hidden layer is the risk

The argument is exact for the output layer (§44). Hidden credit comes through random feedback, which by §46
fits label templates layer by layer rather than composing. If the hidden layer cannot form phase-like
features, the sleeping race is a max-margin readout on a slowly changing, nearly random code: a kernel
machine. Random k-winner codes of the pair then need a width that grows with p² to generalize, and the race
would generalize only through width, not grok. Frozen hidden vs random-feedback hidden separates the two.

### 52.7 Time coding could lower the critical data size

The manifesto's thesis is that relations are cheap in time. Modular addition is addition of phases, and a race
neuron computes weighted means of input times (P3). If operands arrive as delays on a cyclic clock (a at
phase a/p, b at phase b/p), the relation is a coincidence detector over summed delays, with a norm of a few
synapses per output, so C_gen, and with it n*, should fall well below the one-hot case. **Prediction:
phase-coded operands grok from smaller training fractions than one-hot operands.** The encoding needs a cyclic
(wrapping) readout, which the race does not yet have; to design.

### 52.8 Predictions (M53, E24)

(i) Race without sleep: fits the training set, test stays near chance for any training length; plasticity
decays towards zero after the fit.
(ii) Race with sleep, intermediate λ: delayed generalization, with a delay that scales roughly as 1/λ; large
λ underfits.
(iii) A plasticity peak and a drop in the number of distinct hidden codes at the transition.
(iv) Frozen hidden layer: generalization, if any, grows smoothly with width and not with training time.
(v) Below a critical training fraction no λ groks; the critical fraction is lower for phase-coded operands
(when built).
(vi) Sleep increases forgetting in E23 (§52.3's dial).

*Borrowed:* grokking (Power et al. 2022); circuit efficiency and ungrokking (Varma et al. 2023); the norm
account and initial-norm dependence (Liu et al. 2022, Omnigrok); Fourier circuits for modular addition (Nanda
et al. 2023); Pegasos (Shalev-Shwartz et al. 2007); the implicit bias of cross-entropy (Soudry et al. 2018);
sleep as synaptic downscaling (Tononi & Cirelli). Biologically motivated mechanisms that help grokking in MLPs,
including homeostasis and lateral inhibition, are studied by Leon (2026). *New here, as far as checked (web
search, 2026-09-26, no demonstration of grokking in spiking or event networks found):* that an
ultraconservative race rule makes memorization absorbing; that sleep downscaling turns it into a
Pegasos-driven phase transition; that one dial trades forgetting against generalization; and the event-level
order parameters (code sharing, plasticity peak, certified radius).

### 52.9 First test, and a correction: in a thresholded race, downscaling is price inflation

**Result (E24, p = 31, half the pairs, 5,000 epochs, no deadline, no homeostasis).** No λ grokked. λ ≤ 3e-4:
memorization, test 0.000–0.004 throughout. λ = 1e-3: slow collapse (train 0.97 → 0.51). λ ≥ 3e-3: the network
**dies** (weight norm → 0, train 0.00, plasticity 0). The grokking phase is missing, and the "confusion" phase
is not underfitting but silence. Without sleep, plasticity did not stop either (§52.2 assumed it would): random-
feedback hidden credit keeps the codes churning while the test error stays at the lookup's floor.

**What §52.3 got wrong.** A node fires when its potential reaches θ, so (W, θ) → (cW, cθ) is a gauge symmetry
of the race (the dilation of §49, applied to potentials). Shrinking W with θ fixed is not a norm penalty; it is
**raising every price**. Below threshold nothing fires, no error is registered (a silent output is not a
competitor), and the error-gated rule cannot recover: silence is a second absorbing state, next to
memorization. Pegasos needs two things the plain race lacks:

1. **The network must keep deciding while margins shrink.** A deadline (the leader fires at the horizon)
   makes the output an argmax again, so sleep shrinks margins measured in units of θ, and the waking rule
   defends only the margins that the data needs. That is Pegasos's absolute margin, in θ units.
2. **Hidden prices must follow the drive.** Homeostasis lowers hidden thresholds as downscaling lowers the
   drive, so hidden codes survive. The gauge-invariant quantity that sleep then shrinks is ‖w_n‖/θ_n only
   to the extent that homeostasis lags, so hidden-layer sleep is weak by design; the output layer carries
   the max-margin pressure.

So **sleep needs prices** (P5): downscaling regularizes only when thresholds are dual variables that re-balance
activity. This is the grokking counterpart of §51, where the missing prior channel was also a price.

**Revised prediction (M53, before the rerun):** with deadline and hidden homeostasis, the silent phase
disappears; an intermediate λ shows delayed generalization; large λ underfits instead of dying. If the
intermediate phase still does not appear, §52.6's risk (random-feedback hidden credit cannot build relational
features) is the leading explanation, and a frozen or wider hidden layer should behave the same.

## 53. Delays instead of lookup: modular arithmetic on a ring, learned by replay

*Written 2026-09-26 after E24's race runs (all test ≤ 0.008 against chance 0.032, feedback-alignment MLP 0.000:
only backprop groks there) and after E25 pilots. The E24 encoding put both operands at t = 0, so time did no work.
This section re-poses the task the way the README's Sleep Sort note suggests.*

### 53.1 The compiled ring

A ring of p relay nodes, each firing the next after one delay unit, is a cyclic clock: a spike injected at node a
sits at node (a + t) mod p at time t. Inject operand a as a position, let operand b set the read time (a delay of
b units on its line), and a coincidence detector per node reports (a + b) mod p. Addition is waiting; the modulus
is the cycle. 4p synapses, 2b + p + 1 synaptic events per query, all p² pairs correct, nothing learned
(`e25_delay_ring.py compiled`). The lookup table needs p² conjunction nodes and cannot answer an unseen pair.

### 53.2 What the substrate assumes: characters

Give every operand line and every class detector a learnable delay on a ring of any period; write it as a phasor,
z = e^{2πi·delay/period}. The prediction is the class whose detector phase is nearest to the sum of the two
operand delays:

  ŷ(a, b) = argmax_c Re( z̄_c · z_a · z_b ).

This fits a labelling exactly iff y = h(f(a) + g(b)) for some maps into a cyclic group (h injective on the used
classes): the labels factor through **one character** of an abelian group. It contains a + b, a − b, relabelled
sums, a² + b², and a·b on the nonzero residues (a cyclic group of order p − 1, so the delays must learn the
discrete logarithm). It excludes a² + ab + b² and random tables. 3p parameters. The substrate knows it is
composing phases; it is not told which operation, which frequency, or which encoding of the operands.

### 53.3 Learning is synchronization; replay is its power method

With the classes as labelled constraints z_a z_b z̄_y ≈ 1, learning is angular synchronization on the
tripartite hypergraph of training triples (a, b, y). The replay rule

  z_a ← unit( Σ_{samples with a} z_y z̄_b ),  and likewise for z_b and z_y,

is local (a delay moves to the circular mean of what the samples it took part in say it should be: in time,
the teacher's arrival minus the partner operand's arrival) and is the generalized power method for
synchronization. It needs all stored samples at once, so it is a sleep-phase computation. The online,
error-gated version of the same geometry (§35's weaving, in continuous delays) fails even on the training set
where replay succeeds. (Corrected in §54: the cause is the push on the wrong winner, not the loop.)

Replay is not required, but forgetting is. Keep one phasor trace per delay, add each sample's vote as it
arrives (S_a += z_y z̄_b, with the delays read from the current traces), and downscale all traces by (1 − λ)
in a sleep phase between epochs. Without downscaling the first, random-phase votes are never outweighed and
the traces freeze into an inconsistent state (pilot, p = 97, 20% of pairs: test 0.01). With λ = 0.5 the
same rule reaches test 1.000 from 20% of pairs. This is §52's sleep, now doing what §52 predicted it would do:
the online rule is a power method whose stale early iterates must be forgotten. In E24's race the same
downscaling only destroyed training accuracy; the difference is the hypothesis class, not the sleep.

### 53.4 Two thresholds: statistical and computational

- **Identifiability.** 3p phases with a gauge (a global phase per group and the frequency choice m ∈ Z_p^*)
  against n constraints of log p bits: a consistent fit is forced to be the relation once n exceeds about
  3p, i.e. frac ≳ 3/p. Below it, fits that memorize exist even in this tiny class (E25 pilots: p = 31,
  5% of pairs, train 1.0, test at chance). This is §52.1's lookup/relation dichotomy with parameter
  counting instead of norms.
- **Search.** The power method from random starts succeeds only well above that: pilots at p = 97 succeed from
  10% of pairs (8 restarts chosen on training error, test 1.000) and fail at 3–5% (train 0.1–0.4). As in
  sparse synchronization and planted problems, a gap between what the data determines and what local
  iteration finds is expected; its width is the measurement.

### 53.5 Relation to grokking

A grokking MLP ends in the circuit Σ_ω cos(ω(a + b − c)) (Nanda et al. 2023): it builds a phase representation
of the operands out of weights, slowly, under weight decay. The ring has that representation as physics:
delays add and cycles wrap. The prediction is that the delay substrate reaches the relation from a much smaller
fraction and with no slow memorize-then-generalize phase, because its hypothesis class holds nothing but
characters. The price is the class: one ring is one character, so a² + ab + b², which the MLP can grok, is out of
reach. A bank of K rings summing votes is a K-term character expansion, the MLP's grokked form, but it does not
rescue poly: e^{2πi m(a² + ab + b²)/p} = e^{2πi m a²/p} · e^{2πi m ab/p} · e^{2πi m b²/p}, and the middle
factor, as a p × p matrix in (a, b), is a DFT matrix, full rank. Each frequency needs K ≈ p separable rings, so
the bank grows to ~p² parameters, a table. Compression by delays exists exactly for separable compositions.

### 53.6 Predictions (M54, E25)

(i) Compiled ring: accuracy 1.0 at every p, events 2b + p + 1.
(ii) Replay generalizes (test ≥ 0.99) above a critical fraction f_c(p) that falls with p, while memorizing
    (train 1, test at chance) is possible below it; f_c between 3/p and ~10/p.
(iii) The online error-gated learner and the discrete near-miss learner fail where replay succeeds.
(iv) The class boundary: add, sub, perm, sq, mul learned; poly and rand not, at any fraction.
(v) The backprop MLP needs a larger fraction than replay at the same p, and thousands of steps.
(vi) A bank of rings learns poly only with K ≈ p rings per frequency (no compression): the delay substrate's
    advantage is confined to separable compositions, and the sample complexity for poly should look like a table's.

## 54. In a race, winning is positional: teach by pulling, never by pushing

*Written 2026-09-26 from E26 pilots (p = 31, 30% of pairs, 1–2 seeds). Full sweeps queued (`queue/e26.txt`,
`queue/e26b.txt`).*

§53's replay and trace learners work, but they are dense: every sample updates, in epochs, with global
normalisation. E26 asks the question in native terms. Passive delay ring; a query is two operand spikes; class
detectors race, and the first to coincide fires and cancels the rest; learning happens only on errors and touches
only the three delays involved plus, optionally, the wrong winner.

**Observation.** With the usual two-sided rule (pull the teacher earlier and push the wrong winner later), all p
detectors collapse onto a single phase (pilot: 30 of 31 inter-detector gaps < 0.1) and accuracy stays at
chance, even on the training set, with or without timing noise σ ∈ {1, 2, 3, 8}. Repelling crowded runners-up
does not fix it. With the push removed, the same sparse rule learns everything jointly from random delays:
test 0.86–0.91 on unseen pairs from 30% of pairs, about 37k updates in 100k samples, σ = 0.

**Why.** In a race, a class wins by being *earliest*, not by others being late: cancellation already implements
the competition. The teacher pull has a fixed point per class (its detector listens just after the phase its
samples produce), so pull-only learning is a set of independent contractions. The push has no fixed point of its
own: the pushed detector's position is set by *other* classes' errors, and each push hands the lead to the next
detector just behind it. The ring of detectors behaves like a queue, and pushes feed it back toward the read point
until it is one clump. A dense softmax needs the push because scores are not exclusive; a race does not, and
the push is actively harmful. (§53.3's "frustration" of online learning was this push, not the loop.)

**Consequence for the main architecture.** The race rule used everywhere since E6 has exactly this term: every
competitor gets −elig (a near-miss-weighted push later). If the argument holds beyond the ring, it is also a cause
of the race's weak results where competitors crowd (SHD, E23 forgetting, E24). Test: `--compete 0` on E22 (SHD)
and E24 (grokking), queued.

**Predictions (M55).** (Status: (i) confirmed at full length, 3 seeds; (iii) refuted for the weight-based race, see §60.) (i) E26 push = 0 generalizes above a critical fraction, push = 1 never does; (ii) annealed
timing noise changes sample efficiency but not the push result; (iii) `--compete 0` does not lower SHD accuracy,
and raises it if competitor crowding is a cause of the gap.

## 55. Where supremacy can and cannot be claimed

*Written 2026-09-26.*

The one formal separation found in the literature goes against spiking networks: indexing needs Ω(n/log²n)
spiking gates vs O(√n) sigmoid gates (Lynch, Musco & Parter 2017, Neuro-RAM). Indexing is the dense world's
native operation. The question is where the reverse holds.

**Not on operation counts for static functions.** The compiled ring (§53.1) adds mod p with O(1) events given
a shared pacemaker, but a dense network fed the operands as scalars computes cos(2π(a + b)/p) in O(1)
operations too. The apparent separation against a one-hot MLP is an encoding effect, not an effect of asynchrony.
Any static function has a clocked implementation whose op count matches the event count up to the encoding.

**Where the clockless system is different in kind.** A clocked system pays per tick × unit whether or not
anything happened; an event system pays per event. So a defensible separation needs:

1. **Streams whose information rate is far below any usable clock rate:** cost ∝ informative events vs ∝ T/dt.
   The clock can't be slowed without missing timing that matters (SHD-style precise timing inside long silence).
2. **Decisions whose latency is set by the evidence (E2):** the event system answers at the first sufficient
   event; a clocked pipeline answers after its fixed depth × tick.
3. **Learning whose cost ∝ errors (E26):** no epochs, no backward pass over time.

All three must hold on one task at matched accuracy, with the dense side allowed the same priors and input
encoding.

**A quantitative criterion for the input side.** A clocked system that must resolve timing δ pays at least
(channels × duration / δ) input samples; an event system pays one per spike. The separation factor is therefore
1 / ρ_δ, with ρ_δ = spikes per channel per δ-bin. Measured on SHD (test set, 227 utterances): 8,414 spikes per
0.71 s utterance on 700 channels, so ρ = 0.017 at δ = 1 ms (59×), 0.068 at 4 ms (15×), 0.17 at 10 ms (6×).
Dense SHD models do well with 10 ms bins, so SHD offers only about 6× on the input side: it is dense in time, a
poor supremacy benchmark. The benchmark must have ρ_δ ≪ 0.01 at the precision the task truly needs, e.g. rare
informative events in long silence, where the factor grows with the silence. That task family is the benchmark target: sparse event streams with rare, precisely timed informative
events (mostly-silent keyword spotting, event-camera onsets, anomaly onset), measured in events, latency, and
updates.

## 56. Computing with time: what clocklessness forbids, what a reference adds, and how credit flows

*Written 2026-09-26. Builds on the space-time algebra of J. E. Smith (ISCA 2018; "(Newtonian) Space-Time
Algebra", arXiv 2001.04242) and on race logic as tropical algebra (Madhavan, Sherwood & Strukov 2014; Madhavan
et al., "Temporal State Machines", 2021). Those works define the primitives and their algebra. This section adds
three things: what shift invariance forbids and the minimal fix (§56.2–56.3), the credit structure of
space-time networks (§56.4), and the resulting pull-only learning principle (§56.5).*

### 56.1 Primitives

Values are spike times in T = ℝ ∪ {∞} (∞ = no spike), each line spiking at most once per episode.

| primitive | output time | neural form | tropical form |
|---|---|---|---|
| delay δ_c | a + c | axon or dendrite | a ⊗ c (min-plus) |
| first-of (OR) | min(a, b) | either input fires the node | a ⊕ b |
| coincidence (AND) | max(a, b) | threshold 2 with long PSPs | max-plus ⊕ |
| veto (inhibit) | a if a < b, else ∞ | inhibitory synapse that arrives first | not tropical: breaks monotonicity |

The race of §1–§55 is min over a group, with cancellation of the rest: first-of with a label.

### 56.2 Clockless means shift-equivariant, and shift-equivariant networks cannot add times

With no clock, nothing in the network knows absolute time: shifting every input by c shifts every spike by c.
Every primitive above commutes with the shift, so every network built from them computes a function with
f(a₁ + c, …, a_n + c) = f(a₁, …, a_n) + c (Smith's invariance). Consequences:

- **a + b is not computable**, for two variable times: it would shift by 2c. More generally, only functions whose
  pieces have unit total slope. Differences can be computed (compare a against b), sums cannot.
- **Absolute magnitudes are not computable.** "Fire 3 ms after a" is fine; "fire at time 3" is not, because
  nothing marks time 0. §38's blindness to absence is the same fact: an absent spike can't be noticed without a
  reference that says when it was due.

So an asynchronous substrate of pure delays and races computes **relations among times, never their sums**.
Sleep Sort sorts because sorting is shift-equivariant. Modular addition is not, and needs more.

### 56.3 The minimal reference: one oscillator, and time becomes a group

Break the symmetry with the smallest possible reference: a free-running oscillator of period P, shared by
everything and amortized over all queries (a brain rhythm). Now a spike has a **phase** φ = t mod P, and a
unit can act at a phase set by one input and read at a time set by another (the E25 ring: operand a resets the
phase, operand b reads it). With the reference:

- The symmetry drops from all shifts ℝ to shifts by whole periods, Pℤ. Functions are equivariant only under
  those, so phase addition (a + b mod P) becomes computable: it is invariant under Pℤ shifts of both operands.
- **Races on a circle need an anchor.** "First" is not defined on a circle: every phase is after every other.
  Order exists only relative to a reference event (the read): the winner is the first detector phase *after*
  the read. Every cyclic race therefore has an anchor, and learning targets must be defined relative to it
  (E26: the teacher listens half a unit after the read).
- **What one reference buys is exactly one character.** With a single oscillator, the relations reachable at
  O(1) events per query are those factoring through one phase composition, y = h(f(a) + g(b) mod P): E25's
  learnable class, now derived from the symmetry instead of observed. Several oscillators with incommensurate
  periods give several characters; the DFT-rank argument of §53.5 bounds what they can compress.

The manifesto's "relative, causal, temporal referencing" is therefore not optional: a clockless system computes
relations; a system with one rhythm computes group operations; nothing in between computes sums.

### 56.4 Credit flows along one causal chain

A spike's time in a network of delays, first-ofs and coincidences is a tropical polynomial in the delays: the
sum of the delays along one path, the **critical path** (the argmin through first-ofs, the argmax through
coincidences). Its derivative with respect to a delay is 1 on the critical path and 0 elsewhere (almost
everywhere). So exact credit in a space-time network is:

- **Sparse by construction:** one path per spike, length = depth, not fan-in × depth as in a dense backward
  pass. The critical path is recorded for free during the forward race: each node remembers which input set its
  time (the E4 "which input arrived last before threshold" trace).
- **Local:** each node needs only whether it is on the path and the sign of the error at the end.
- **Discontinuous at ties:** when two paths tie, credit switches. This is where learning can oscillate, and
  where timing noise (§41, §48) smooths the switch into a probability.

(For integrate-to-threshold nodes the path generalizes to the causal set, §34.1; Mostafa 2018's exact TTFS
gradient has this structure.)

### 56.5 Teach the event that should have won; never touch the losers

Every error in a race network is a race lost by the right event: the teacher's spike came after the winner's
(or never came), or a spike that should have been vetoed was not. The native fix is to move **only the event
that should have won**, along its critical path, toward an **anchored target time** (the reference + margin),
and to leave the losers alone:

- Wrong class won: move the teacher's path so the teacher arrives at anchor + m.
- A spike fired that should not have: move the **veto's** path so the inhibitor arrives first. Veto is the
  only native way to make something later, and it makes suppression an act of winning too.
- Teacher never fired: move its path toward coincidence. The target of an arrival is its partner in the window,
  not "earlier": arrivals move toward each other. A one-sided "make it earlier" rule drifts, since shortening
  delays or loop periods makes everything earlier (E28 delays past the anchor, E29 periods to the floor).

A false positive does require acting on the loser, and here the distinction is how. Shifting its delays later
(a **push**) displaces it in time; that is what collapsed E26. Adding an inhibitory condition that blocks it on
the offending inputs (a **veto**) specializes it and leaves its timing alone. So the rule is: *losers are
specialized by inhibition, never displaced in time.*

This is regression to an anchored time on one causal chain, not a margin against competitors. §54 is the
evidence that it matters: the two-sided rule (pull the teacher, push the winner) collapses all detectors onto
one phase, while pull-only learns the relation. The push fails because a loser's time has no target of its own;
it is set by other classes' errors, and each push hands the lead to the next loser. In a race the losers are
beaten, not punished.

**Why this is sparse and asynchronous by construction:** updates happen only on errors (a lost race), touch
only one causal chain (depth-many delays), and need only a local anchor. No sums over samples, no backward pass
through all synapses, no clock except the one reference that §56.3 shows is needed anyway.

### 56.6 Predictions (M56, E27)

(i) A network with learnable delays on excitatory (coincidence) and veto synapses, trained by §56.5 only,
    learns temporal patterns of the form "B within Δ after A, unless C in between", generalizing to unseen
    timings.
(ii) Fixing false positives by pushing the loser's delays later, instead of by veto, breaks it (as in E26).
(iii) Without veto synapses, patterns that need "unless" are not learnable at any size (monotonicity).
(iv) Updates per sample fall to ≈ the error rate × depth; synaptic events per sample stay ≈ input spikes +
     O(1).
(v) Without a reference, a sum of two times is not computable (§56.2): a linear-time race network given the
    operands as spike times, with no oscillator, cannot learn a + b as an output time at any size, while it can
    learn comparisons (which of a, b is larger, by how much, within a window).

## 57. Routing needs counterfactuals, and cancellation supplies them without extra events

*Written 2026-09-27, prompted by the observation that a race network is a routing network: which node wins is a
discrete route choice, and critical-path credit (§56.4) only tunes timing along the route taken.*

### 57.1 The problem

§56.4's derivative is 1 along the critical path and 0 elsewhere. It says how to move the route that was taken,
never whether another route should have been taken. The same wall appears in sparse mixture-of-experts: the
gate's gradient exists only for experts that ran, which is why top-k routing with k ≥ 2 is used, to compare at
least two alternatives. Routing credit requires counterfactual information: what the untaken routes would have
produced.

### 57.2 Three sources of counterfactual information in a race

| source | what it tells | cost |
|---|---|---|
| k winners per group (k ≥ 2) | the runners-up's actual outputs | (k − 1) extra spikes per group, all downstream events they cause |
| cancelled near-misses | how close each cancelled node came (its frozen charge) and on which inputs (its best partial window) | none: the charge is state the node already has at cancellation |
| timing noise σ | which route would have won under a small perturbation | extra runs, or temperature over time |

The second is native and free. A cancelled node stops integrating, but the charge it has reached and the inputs
that produced it are its state at the moment of cancellation. They are exactly the counterfactual "had I been
allowed to continue, I would have fired on these inputs", ranked by how near the node came.

### 57.3 The rule

On an error, the teacher's missing input tells the hidden layer what route was needed (through the teacher's
synapses, w2). Pull the **cancelled node the teacher wants with the highest near-miss**, on **its best partial
window only**, so it wins its group next time; do not touch the route that was taken, and do not push the winner.
Strengthening all inputs that reached the cancelled node (distractors included) is destructive (E28 pilot: 0.18,
at the "none" floor); restricting the pull to the partial window turns it into a gain (0.41 vs 0.34 for top-k
fired credit, 0.35 at depth 1; one seed, 8k episodes; full runs queued).

### 57.4 Predictions (M57, E28)

*Status (full runs, 3 seeds): (i) confirmed against critical-path-only credit (0.31–0.33 vs 0.17); near-miss did not
beat fired credit at k = 2 (0.31 both), (ii) k = 1 near-miss matched k = 2 fired at 28% fewer events but with high
variance, (iii) not confirmed (depth 2 ≈ depth 1), (iv) confirmed (push 0.17). All arms far below ceiling; the
readout lacked §60's conservation and capacity. With §60 applied to every node (E28b, 3 seeds): depth 2 with routing
credit 0.72–0.74 vs depth 1 0.68 vs depth 2 path-only 0.17: (i) confirmed strongly, (ii) near-miss ≈ fired, (iii)
a modest lead for depth that did not hold with 15 classes from 6 shared motifs (E28c, 5 seeds: depth 1 0.51 vs
depth 2 0.44 at 60k): depth is learnable, not yet advantageous.*

(i) Near-miss routing credit beats critical-path-only credit at depth 2, and matches or beats top-k fired credit.
(ii) Near-miss credit with k = 1 approaches its k = 2 result: counterfactuals from cancellation replace
     counterfactuals from extra firing, at fewer events.
(iii) Depth 2 with near-miss credit beats depth 1 on hierarchical motifs.
(iv) Displacing false winners in time breaks it, as in E26 and E27.

## 58. What counts as generalization: restriction, forced generalization, and grokking

*Written 2026-09-27, after the objection that E25/E26 generalize because their structure restricts them to the
answer's form.*

A learner can reach the relation on unseen pairs for three different reasons, and only the last is grokking.

1. **Restriction.** The hypothesis class contains the relation and little else. E25/E26's single loop with a
   single-phase readout can only express one-character relations (§53.2); within that class, any good fit on
   enough data is the relation. This measures the prior, not the learner.
2. **Forced generalization.** The class does contain memorizers, but only below its capacity. E25 shows it
   exactly: with 3p parameters, fits that memorize exist below n ≈ 3p (p = 31, 5% of pairs: train 1.0, test at
   chance) and stop existing above it. Generalization above capacity is counting, not learning.
3. **Grokking.** At n well below capacity, memorizers consistent with the training set exist in the class, and
   the learner still reaches the relation. That is a property of the learning dynamics (its implicit preference),
   and it is what the dense MLP shows on E24 (≈ 17k parameters vs 480 training pairs at p = 31, frac 0.5).

**Criterion.** Report the capacity ratio ρ = n / (effective parameter count) with every generalization result.
Grokking claims need ρ ≪ 1, a class that provably contains memorizers at that n, and a demonstration that some
learner in the same class does memorize (a table-like baseline trained the same way).

**Consequence for this project.** A loop in the substrate is not itself the bias: a loop's period is a
learnable delay, and delays are scale-free (any period works with rescaled injection delays), so "a recurrent
delay loop exists" is as generic as recurrence. The bias in E26 is the **single-phase readout**, which removes
every non-cyclic hypothesis. The honest test (E29) is a general race network, the one that memorizes E24 at
test 0.00, given recurrent delay loops as a resource and native credit (pull-only, errors only, near-miss
routing). It passes only if it reaches the relation at ρ ≪ 1 while the same network without loops, or without
counterfactual credit, memorizes.

## 59. A two-counter machine wired from the basis, and restoration in time

*Written 2026-09-27 with E30 (`e30_minsky.py`, results in `results/e30/minsky.json`).*

**Construction.** A netlist of four node types (Delay; Or; And with a PSP window per input; Veto), plus a reference
oscillator, runs two-counter Minsky programs. Counter value n is the phase OFF + n·q of a spike in a hold loop of
period T. Per cycle, the spike takes one of three Delay paths: hold (T), increment (T + q), decrement (T − q);
path choice is an And with the instruction's enable line (long PSP) and a Veto on the hold path. The zero test is
an And of the counter with the reference (window q/2); branching is a Veto ("next unless zero") and an And ("next
if zero"); a decrement blocked at zero returns to hold through And(zero, gated counter). No node reads a clock,
stores a number, or branches in code.

**Result.** Exact on every test (add, double, parity; 10 inputs including zero edge cases): correct counters and
the exact cycle count, ~5k events per run. Two-counter machines are Turing-complete (Minsky 1967), so the basis
{delay, first-of, coincidence, veto, hold} + one reference is Turing-complete given unbounded phase precision
(prior art for spiking networks with exact delays: Maass 1996). Two wiring lessons are general: coincidence
windows must be per input (a PSP length per synapse), or stale spikes from earlier cycles pair with new ones;
and every veto that blocks a path must hand the spike to another path, or state is destroyed.

**Precision is the tape, and restoration makes length free.** With jitter σ on every hop, phases random-walk.
Without restoration, success falls with program length (q/σ = 20: 0.60 over 25 steps, 0.30 over 81). With one
restoring coincidence per counter per cycle (the loop delivers the spike q/2 early; a comb tick at the nominal
phase re-emits it), success is length-independent (q/σ = 20: 1.00 at both lengths; q/σ = 10: 0.975 at both). (A random walk fits the unrestored runs with one parameter: a counter phase after s steps is N(0, σ√(h s)) and
fails beyond q/2; q/σ = 20 with 60% success at 25 steps gives h ≈ 5.8 jittered hops per cycle, which predicts 35%
at 81 steps, observed 30%; the netlist has about six hops per counter per cycle.) This
is digital restoration done in time: the cost is one comb coincidence per counter per cycle, the capacity T/q
states per counter, and the error rate per step a function of q/σ only. The residual failures at q/σ ≤ 7 come from
the unrestored control margins inside a cycle.

**Consequence.** An asynchronous temporal substrate can compute arbitrarily long with noisy timing, and the
resource accounting is explicit: precision (T/q) replaces tape, cycles replace steps, events replace energy, and
restoration events are the price of reliability.

## 60. Pull-only on weights needs conservation; capacity sets the readout

*Written 2026-09-27 from E26b and the E29 readout diagnosis.*

**E26b refutes §54's transfer as stated.** Dropping the competitor push from the main weight-based race collapses
SHD from 0.35 to 0.061 (`--compete 0`, depth 1, 10 epochs). The positional principle holds for timing parameters
(delays and phases, where a pull cannot inflate anything: E26), but not for weights: in the weight race the push
is the only force bounding the weights, and without it every class node inflates and fires.

**The native counter-force is conservation.** E29's readout, isolated on a frozen hidden layer with unique codes,
shows the sequence: pull-only weights saturate (every class fires at once, the winner is timing noise); a
per-node conserved synaptic budget (heterosynaptic: a pull on the co-active synapses is paid by the node's
others) stops that, but only if each pull moves a fixed fraction of the budget (otherwise each pull rescales the
whole node and it remembers only its last sample); weakening a false winner must also conserve (otherwise it is
a one-way drain); per-node prices (thresholds raised by false wins, lowered by misses; §36, §51) break up the
remaining hub classes. Veto at the readout over-suppresses when codes overlap.

**Capacity, not the rule, set the plateau.** A first-to-threshold class node is a linear threshold unit on the
hidden spikes; with 96 hidden nodes it can separate only ~2·96 random patterns (Cover 1965), fewer than the 480
training pairs, and the readout plateaued near 0.45 for every rule variant. With 384 hidden nodes the same rule
memorizes (train 0.88 by epoch 7). Grokking tests need hidden layers above this bound, or the "memorizer" in
§58's criterion does not exist.

**Revised principle (§54, §56.5):** never displace in time; for weights, pull under conservation; remove false
winners by veto where codes are specific and by conserved weakening plus prices where they overlap.

## 61. Two kinds of temporal tolerance: aligning destroys the interval, holding keeps it

*Written 2026-09-27 from the E27 error analysis.*

A coincidence node can accept "B within Δ after A" in two ways.

- **Align:** delay A by about the typical gap so it meets B, with a narrow window. The interval [t_A, t_B] is
  compressed into the window; the node no longer knows what happened in between.
- **Hold:** give A a PSP of length Δ and fire when B arrives while it is still open. The node is armed exactly over
  the raw interval [t_A, t_B].

The two are equivalent for detecting the pair and not equivalent for anything that depends on the interval's
content. A veto for "unless C in between" must arrive while the node is armed; under alignment the armed span is
a narrow window near t_B + d_B, while C can fall anywhere in (t_A, t_B), so no single veto delay covers it. Under
holding, a veto with no delay does. E27 shows the symptom: with alignment the veto synapse on the right channel
reaches full strength for every pattern, yet vetoed near-misses remain the main error (192 of 318 errors after
100k episodes). With holding, veto adds +5 points over no veto (pilot, 2 seeds) versus +1.8 under alignment,
though the pilot's windows only grow, which caps its overall accuracy (0.71–0.73).

**Principle.** Delays are for *where in time* an event should act; PSP durations are for *how long* a node should
remember that it happened. Any operator whose meaning involves the interval between events (veto, ordering,
"no C since A") needs duration, not delay. Learning therefore has two timing parameters per synapse, delay and
duration, and they answer different credit questions: a coincidence missed by misalignment moves the delay; one
missed because the partner came too late for the PSP lengthens the duration; a false fire from a partner that came
too late shortens it.

## 62. Why depth does not pay yet (corrected: the hidden nodes are parts; the readout discards their order)

*Written 2026-09-27 from E28c and a hidden-selectivity diagnosis (15 classes built from 6 motifs, 192 hidden,
20k episodes; selectivity = max over conditions of P(fire | condition) − P(fire | not)).*

| hidden layer | median motif selectivity | median class selectivity | motif-selective nodes (> 0.3) | class-selective nodes (> 0.3) |
|---|---|---|---|---|
| untrained | 0.07 | 0.13 | 7 | 28 |
| trained, label-gated credit (fired) | 0.12 | 0.21 | 24 | 55 |
| trained, label-free (every winner pulls its window) | 0.11 | 0.19 | 22 | 55 |

Training makes hidden nodes more class-selective than motif-selective, with or without labels: they duplicate what
depth 1 does, so depth adds nothing (E28c: depth 1 0.51 vs depth 2 0.44). Label-free learning from recurrence
(the STDP route to repeated patterns) does not change this, although motifs are ~50× more frequent than chance
coincidences of the same channels.

**Correction (same day).** The selectivity metric conflates: a genuine motif detector also looks class-selective,
because it fires for every class containing its motif. Looking at receptive fields instead (the two strongest input
channels of each hidden node that some class relies on): after training, 86 of 124 such nodes take both inputs from
the same motif (parts), 16 from different motifs, 22 use a non-motif channel; untrained, 2 of 35 were parts. **The
hidden layer does learn parts.** What fails is the readout introduced by §60: it accumulates without a window,
which counts which parts occurred and discards their order, while E28's classes are ordered pairs (and the decoys
are reversed pairs). Non-leaky accumulation was right for a static task (E29) and is wrong for a temporal-order
task, where the readout must align one part's spike with the other's by a delay inside a window (§61). The original
diagnosis below is kept for the record.

**Second correction (error breakdown, same day).** Depth 2 makes no order errors at all (0% reversed-class answers,
vs 13% at depth 1); 56% of its answers are a class sharing one motif with the true class. The failure is
**conjunction**, not order: a single detected part fires a class. It survives capping single synapses below
threshold (~14 redundant detectors per motif still sum past it), windowed readouts, and global competition across
the hidden layer. The open problem is precise: a native readout that requires two *distinct* parts when parts are
represented redundantly. (Dense readouts get this from signed weights that learn to subtract single-part evidence;
the race readout has positive weights and one threshold.) Tried and negative: input consumption (the first hidden node to
fire uses up its input spikes, the weaving operator's self-cancellation), 0.16–0.35, because the earliest
coincidences are often distractors and they consume the motifs' spikes. Not yet tried: a readout node whose inputs
are vetoed by themselves after the first part (refractory per part group), so only a different part completes it.

**Original diagnosis (superseded).** A motif is "j within 0.3–1.5 after i"; the hidden window is 0.6. A pull moves only arrivals already
inside the firing window, so a node whose window catches i but not j can never align j onto it: the recurrence is
there, but the credit cannot reach it. This is §61's distinction again: to discover a motif a node must either hold
long enough to see both spikes (duration) and then align (delay), or receive credit from near-coincidences across a
span longer than its window. **Prediction (M62):** hidden nodes with learnable duration (hold), shortened as
their delays align, become motif-selective, and depth 2 then beats depth 1 on shared-motif tasks.

**Test (same day): not supported.** Hold-then-align (windows start at 2.0; each pull aligns the window's arrivals
and shrinks the window toward their spread) leaves accuracy unchanged (depth 2: 0.38–0.48, 2 seeds; depth 1 0.50)
and, label-free, raises class selectivity more than motif selectivity (median 0.28 vs 0.16; strongly
motif-selective nodes fall from 22 to 9). Longer windows let a node see more of an episode, and it uses that to
become a better class detector. Reach was not the missing piece; the missing piece is a pressure that makes a node
prefer a part over a whole. In dense networks that pressure is width-limited capacity shared across many
outputs; here every hidden node can afford to be a class template. Open.

## 63. A measured frontier: event learner vs clocked dense model on a timing task (E32)

*Written 2026-09-27. Task: E27's ("B within Δ after A unless C", 4 patterns + none, 12 channels, episodes of 10
time units). Dense: 1-D temporal conv on binned spikes (F filters, receptive field 4 time units, ReLU, max over
time, linear readout), backprop + Adam, 200k training episodes as for E27, 2 seeds; cost = multiply-adds (MACs) of
the forward pass. Event: E27's learner, 5 seeds; cost = synaptic events actually delivered (excitatory synapses of
the channels that spiked, existing veto synapses reached, detector spikes).*

| model | test accuracy | cost per episode |
|---|---|---|
| event learner (delays, windows, veto; errors-only updates) | 0.914 (0.887–0.934) | 10.2 synaptic events |
| dense, F = 16, δ = 0.05 | 0.995 | 3.07M MACs |
| dense, F = 16, δ = 0.25 | 0.984 | 123k |
| dense, F = 16, δ = 1.0 | 0.935 | 7.8k |
| dense, F = 4, δ = 1.0 | 0.895 (0.872, 0.918) | 1.9k |
| dense, F = 4, δ = 2.0 | 0.845 | 500 |
| dense, F = 2, δ = 1.0 | 0.686 | 970 |

**At matched accuracy (≈ 0.91)** the event learner uses ≈ 190× fewer operations per episode than the cheapest
clocked dense model found (1.9k MACs), ≈ 80× less energy at published per-operation costs (Loihi synaptic event
≈ 24 pJ vs a 45 nm MAC with its weight read ≈ 10 pJ). With the episode inside 99% silence (a sparse stream) the
clocked model's cost grows 100× and the event learner's does not: ≈ 19,000× in operations, ≈ 8,000× in energy.
Training: ≈ 0.09 updates per episode touching ≲ 10 parameters (≈ 1.8·10⁵ parameter changes in all) vs ≈ 1.2·10⁹
MACs of backprop, ≈ 6,000×.

**What this does not show.** (1) Accuracy: the dense model goes far higher (0.995) at far higher cost; the event
learner has no knob to buy accuracy with more compute yet. (2) Most of the separation is the clock, not the
learner: a dense conv evaluated only where spikes are (a sparse, event-driven digital implementation) would cost
≈ spikes × F × width ≈ 6.6 × 4 × 4 ≈ 106 MACs, only ≈ 10× more than the event learner. That is §55's point: the
advantage belongs to the event paradigm however it is implemented, and grows with silence. (3) One task, designed
around the primitives; the cheapest dense model was searched only over F ∈ {1, 2, 4, 16} and δ ∈ {0.05 … 2}.

## 64. Order is asymmetry of duration; one node computes an interval predicate; few mistakes suffice

*Written 2026-09-27, with E27's directional-hold result (0.9995 and 1.000 on 2 seeds, ~400 updates in 40k
episodes; without veto 0.81).*

### 64.1 The recurring construction

Three separate constructions needed the same thing. E30's Minsky machine worked only when coincidence windows were
set per input (a long PSP on the enable line, none on the counter spike); with symmetric windows, stale spikes paired
across cycles. E27's detector reached 1.0 only when A's input opened a long PSP and B's was instantaneous; its
symmetric version accepted reversed order (0.71) and its aligned version lost the interval the veto needs (0.91).
In each case **order is encoded by an asymmetry of PSP durations, and exclusion by a veto inside the held
interval.** A symmetric coincidence is order-blind by construction.

### 64.2 One node, one interval predicate

Give every synapse three timing parameters: delay d, duration (PSP length) w, and sign (excitatory or veto). A node
with an excitatory A-synapse (delay d_A, duration w), an instantaneous excitatory B-synapse (delay d_B) and veto
synapses (delays u_C) fires exactly when

  t_B + d_B − (t_A + d_A) ∈ [0, w]   and   no C with t_C + u_C in (t_A + d_A, t_B + d_B).

That is the predicate "B within [d_A − d_B, d_A − d_B + w] after A, and no C in between (shifted by u_C)", the
atom of the manifesto's causal logic ("emit A after x if no B") generalized to an interval. **Corrected claim, verified (E33).** A relation fixing one bounded difference with exclusions ("q within
[lo, hi] after p, no C in between") is one node. A relation fixing the order of m points is a chain of m − 1
nodes, and not one node: a node's hold condition ("each held input arrived within its window before the trigger")
is invariant under swapping the arrival order of two held inputs inside their windows, so it cannot fix their
mutual order; a chain can, because each node's output spike carries "so far in order" forward in time to the node
that checks the next point. **Temporal depth equals the length of the order chain.** *(Superseded by §71: depth 2 suffices for every conjunction of order constraints; one node orders at most three events.)* E33 wires all thirteen of
Allen's interval relations this way (before, meets and their inverses: 1 node; overlaps, starts, during,
finishes, equals and inverses: 3-node chains; causal, no negative delays) and checks them against their
definitions on ~19,900 random interval pairs with planted equalities (tolerance 0.05; 99 samples within a
tolerance band skipped as ambiguous): exact, every relation with 475–3,964 positive cases. (Allen's algebra is
prior art; the node-level construction and the depth statement are the claim.) First versions had three bugs,
each instructive: delaying the trigger instead of the held input weakens "after" into "not too long before"; a
chain cannot subtract a tolerance from a spike time (non-causal), it can only absorb it in later delays; and
planted equalities must keep intervals valid.

### 64.3 Why so few updates

The hypothesis class of one hold+veto node on N channels is small: an interval on one time difference (two
parameters) and a set of veto channels (a disjunction over N): VC dimension about 2 + N. Mistake-driven learning
of an interval and of a monotone disjunction needs O(VC · log(1/ε)) mistakes (Littlestone's bounds for
monotone disjunctions are logarithmic in N per relevant channel), independent of how many episodes stream past.
E27 hold: ~400 updates in 40k episodes (~1%), for K = 4 detectors on N = 12 channels, i.e. ~100 mistakes per
detector. The dense conv baseline has 16 × 12 × 16 + readout ≈ 3k parameters at δ = 0.25 and needs ~10⁵ episodes.
**The event learner's sample and update efficiency is an Occam effect of the primitive:** when the task is made
of interval predicates, a node that computes exactly one needs only as many mistakes as its few parameters.
The flip side is the same statement: tasks not made of such predicates need depth, which is where §62 stands.

## 65. Depth and routing, learned: hold/trigger chains (E34) and nothing-given detectors (E35)

*Written 2026-09-27.*

**E34: depth pays when composition is a chain.** E28's class nodes accumulated evidence, so any one part could fire a
class (§62). A hold/trigger class node cannot be satisfied by one part: one part's spike must be held and a different,
later part's spike must trigger. On E28's task (15 classes from 6 motifs; 5 seeds, 40k episodes): depth 2 0.97
(0.92–0.99) with part windows [0, 1.5], 0.86 with a generic bank of windows {1, 2, 4}, 0.79 with one wide window 4;
depth 1 with the same nodes 0.39; E28's accumulating readout 0.44–0.51. Routing (which part holds, which triggers)
is learned by counterfactual pulls under conserved budgets. Learning part windows from class routes failed
(0.08–0.13): a part shared across classes must not be shaped by one class's errors, the same lesson as §62.

**E35: nothing given.** Detectors with learned hold, trigger and veto weights over all channels and learned hold
durations solve E27's task at 1.000 on four seeds and 0.9995 on the fifth (7.5 synaptic events per episode, 443–1,530
updates in 200k episodes). Given E27 told its detectors their channels, this removes the structural prior that made
E32's comparison unfair; the headline comparison (§63's table) holds with nothing given.

**Why routing is cheap here.** Each detector's routing is a choice of one hold and one trigger channel: 2 · log₂N
bits. Counterfactual pulls find it with O(N) mistakes per detector (E35: ~100–400 per detector), consistent with
§64.3's Occam argument extended to routing.

## 66. What grokking requires of a timing network: shared intermediates, and a sum needs a rhythm

*Written 2026-09-27, before redesigning E29 with the machinery that works (hold/trigger nodes, routing credit).*

**Per-class parameters cannot generalize on (a + b) mod p.** Suppose each class detector c has its own delay on every
operand line. For a fixed c, operand a occurs in exactly one pair, (a, c − a). A training pair constrains only its own
two synapses on its own class node; an unseen pair's synapses are constrained by nothing. Such a network can memorize
every training pair and must be at chance on every test pair, whatever its learning rule. (The same holds for any
per-class lookup, which is why E24's local race memorized.) Generalization needs **intermediate nodes shared across
classes**, whose parameters every class's errors move; that is what an MLP's hidden layer provides.

**The time-native shared intermediate is a node whose output time is the sum.** If one node fires at a time that
encodes a + b, every class reads it by phase and all classes share the same operand parameters (this is E26, which
generalizes). But a + b as a time is a series composition, and a clockless network cannot add two times (§56.2); it
needs a reference: an oscillator whose phase is reset by one operand and sampled by the other (§56.3).

**So a true grokking test for this substrate is well posed as:** a network that can memorize (per-class delays on
operand lines, so memorizers exist at ρ ≪ 1), that also has access to generic shared resources (relay nodes and a
rhythm with random, learnable period), trained by the native rules that work (hold/trigger coincidence, routing by
counterfactual pulls under conserved budgets, errors only). It groks if it moves its solution from the per-class
route (memorization) to the shared route through the rhythm (the relation) while memorizers are still available. The
prediction is sharp: without the rhythm resource it must stay at chance on test pairs (per-class argument above);
with it, generalization requires the routing credit to prefer the shared route, which is what the routing rule must
be shown to do. §52's sleep (downscaling) is the candidate pressure: shared routes are reinforced by every class's
errors and survive decay, per-class routes by one pair each and fade.

## 68. The native rules are online learning on the simplex: why a few hundred mistakes suffice

*Written 2026-09-27, to explain E35's mistake counts and §60's need for conservation.*

**The update.** §60's conserved fractional pull on node c's synapse j is W_c ← (W_c + η·|W_c|·e_j) · |W_c| / |W_c|(1+η):
the pulled synapse gains, every other synapse is multiplied by 1/(1+η). On the normalized weights w = W_c/|W_c| this is
w ← (w + η e_j)/(1 + η), a step toward a vertex of the probability simplex; the conserved weakening of a false
winner is the mirror step away. This is the family of normalized online updates on the simplex (Winnow, Hedge,
exponentiated gradient: mirror descent with an entropic geometry), not the perceptron's Euclidean one.

**What that buys.** For a target that is a k-sparse choice among N inputs (a detector's hold channel and trigger
channel: k = 2; a class's parts: k = its number of parts), multiplicative/normalized online learners make
O(k log N) mistakes (Littlestone 1988; Kivinen & Warmuth 1997), where additive perceptron updates on the same
problem need O(k·N) in the worst case. Predicted E35 mistakes per detector: order k log N ≈ 2·log(12) ≈ 5 per
relevant choice times the number of patterns it must also reject; observed ~100–400 per detector including
duration and veto learning. The observations the bound explains:
- **E35**: routing among 12 channels found in a few hundred mistakes per detector, then learning stops.
- **§60**: unnormalized pull-only (additive, positive weights) saturates: every pulled synapse grows, nothing
  shrinks, all classes fire. The normalization is not a safeguard added to the rule; it is what makes the rule a
  learner with a logarithmic mistake bound.
- **§60's fractional steps**: a pull that moves a fixed fraction of the budget is the constant learning rate of
  the multiplicative family; an absolute step that rescales the whole row is a step size that grows with the row's
  sparsity, which is why nodes "remembered only their last sample".

**Durations and vetoes.** A hold duration learned by moving toward the observed gap when a partner came too late
(and below it on a late false fire) is online interval learning: O(log(range/resolution)) mistakes. Veto sets
learned by strengthening on false fires and weakening on false blocks are monotone disjunctions learned by
elimination: at most N mistakes in the realizable case. Together: a hold/trigger/veto node on N channels is learned
with O(log N + N_veto + log(range/resolution)) mistakes, independent of the number of episodes.

**Depth.** In a hold/trigger chain the credit for an error goes to one route (§56.4, §57), so the mistakes of a
chain of depth D add over its nodes rather than multiply: O(D·k log N). This is the theoretical reason E34's depth-2
network learned with ~1–2k updates while depth 1, whose nodes must each carry a whole class, did not converge.

## 69. Grokking is a flip of the absorbing state (E37)

*Written 2026-09-27, before E37's full runs were read.*

Error-gated learning with two routes has two candidate absorbing states. **Memorization** absorbs when there is no
decay: once each training pair's lookup entry answers correctly there are no errors, nothing learns, and the shared
route stays untrained (E37 pilot: train 1.0, test 0.031, 950 updates in all). **Decay flips which state absorbs.** A
lookup entry pulled to level L decays below threshold after n_d ≈ ln(L/θ)/λ epochs; then the pair errs, which
teaches both the lookup (re-memorizing) and the shared route. Once the shared route answers a pair correctly, the
decayed lookup entry falls silent, the rhythm answers, no error occurs, and **the lookup entry is never re-learned**.
The relation is now the absorbing state: every pair the shared route gets right is permanently handed over.

**Predictions (M69).**
(i) With λ = 0, test stays at chance for any number of epochs (absorbing memorization).
(ii) With λ > 0, test rises after train has saturated (delayed generalization), and the fraction of test answers
    given by the lookup falls to zero (the lookup empties).
(iii) The delay scales as 1/λ: the shared route learns only from errors, which arrive at a rate of about
    n·λ/ln(L/θ) per epoch; it needs a roughly fixed number of errors U (E26: ~10⁴–10⁵), so T_grok ≈ U·ln(L/θ)/(n·λ).
**Status after E37 (p = 31, half the pairs, 3 seeds, 400 epochs).** (i) confirmed: λ = 0 memorizes and stays at
chance in all seeds (test 0.029–0.044). (ii) confirmed in 2 of 3 seeds at every λ > 0 (test 0.93–0.97), with a
genuine delay at small λ (λ = 0.02, seed 0: train ≥ 0.95 at epoch 5, test ≥ 0.5 at epoch 70). Seed 1 fails at every
λ: its rhythm route never finds the relation, and once the lookup decays even training falls (0.40–0.48), which is
prediction (iv)'s mechanism. (iii) partly refuted: the delay shrinks with λ but far slower than 1/λ (test ≥ 0.5 at
epochs 70–100, 50–60, 45–50 for λ = 0.02, 0.05, 0.2): T_grok ≈ max(c/λ, T_rhythm), and the shared route's own learning
time dominates above λ ≈ 0.05.
(iv) *Confirmed (E37 op controls, 3 seeds, with cooled noise):* a − b and relabelled sums grok (0.95–0.98); a·b in 1
of 3 seeds (0.98; 0.51 and 0.08 in the others); a² + ab + b² and random tables stay at chance on test (0.01–0.04) while
training falls to 0.46–0.56 under sleep.
(iv, as predicted) Relations the rhythm cannot express (a² + ab + b², random tables) stay memorized at chance on test for any λ,
    and with strong λ lose training accuracy too, since the lookup keeps decaying and nothing can take over.

## 70. Order is native to a race and learned by a Transformer

*Written 2026-09-27, after E36's first runs (event-token Transformers 0.93–0.97 at 576k MACs on E27's task, 0.42–0.70
at 687k on E28's, vs 1.000 at 7.5 events and 0.97 at ~14 events), and before the long-training and relative-time
runs were read.*

Self-attention without position information is permutation-invariant: it cannot see order at all. With time
encodings, a head can compare two events' times only through dot products of sinusoidal features, i.e. it must
synthesize the predicate "t_B − t_A ∈ [lo, hi]" from learned projections, at O(n²·d) per layer whether or not the
comparison is relevant. A hold/trigger node computes exactly that predicate as its primitive, at the cost of the
events it receives (§64), and learns which one with O(log N) mistakes (§68). On tasks defined by order and bounded
differences (E27, E28), the prediction is: small Transformers fail, large ones improve slowly with training, and the
gap narrows but costs grow as n²·d.

**The fair strengthening.** Give each attention head the relational primitive directly: an additive bias on the
attention score that is a learned function of t_j − t_i (relative-time bias, in the spirit of relative position
encodings and ALiBi). This removes the "must synthesize comparisons" disadvantage; what remains is the dense cost.
**Predictions (M70):** (i) with 10× more training, event-token Transformers approach 1.0 on E27 but not at < 10⁴
MACs; (ii) a relative-time bias raises small Transformers substantially on E27 and E28; (iii) no Transformer
reaches the event network's accuracy at comparable cost, because its cost is quadratic in events while the event
network's is linear in the events a node receives.

## 71. A depth hierarchy for one-shot temporal networks (theorem, with proofs; corrects §64.2)

*Written 2026-09-27. The mapping of spiking depth to difference-constraint zones was not found in the searches made for
this project; that is not a claim of priority. Zones (difference-bound matrices) are standard in timed-automata
verification (Dill 1989; Alur & Dill 1994).*

**Setting.** Events are real times t_1, …, t_m (all finite: complete episodes). A *node* has one trigger input T with
delay d_T; hold inputs i with delay d_i and window w_i ∈ [0, ∞]; veto inputs j with delay d_j and window w_j ∈ [0, ∞];
all delays ≥ 0 (causality). It fires at τ = t_T + d_T iff every hold satisfies τ − w_i ≤ t_i + d_i ≤ τ and no veto
satisfies τ − w_j ≤ t_j + d_j ≤ τ. A veto with w_j = ∞ reads "has not arrived by τ". Min (first-of) and max (all-of)
nodes fire at the min / max of their inputs. A network's inputs are the events; a node's output is a spike at its
firing time, available to later nodes. Depth counts nodes on the longest path.

**Lemma 1 (one node = a product set in lag coordinates).** Write x_i = t_i − t_T. A hold constrains only x_i, to
[−w_i + (d_T − d_i), d_T − d_i] (bounded above by causality); a veto constrains only x_j, to the complement of an
interval; an input feeding several synapses gets the intersection. So the set of accepted configurations, in lag
coordinates, is S_1 × S_2 × … × S_m with each S_i ⊆ ℝ depending on input i alone. □

**Theorem 1 (total order).** One node computes P_m = {t_1 < t_2 < … < t_m} on all finite configurations iff m ≤ 3.
*If:* m = 2: trigger t_2, hold t_1 with w = ∞, d_1 = d_T = 0 (accepts t_1 ≤ t_2; strictness by a veto on t_1 with
w = 0). m = 3: additionally veto t_3 with w = ∞ and d_3 = 0: accepted iff t_3 has not arrived by t_2, i.e. t_3 > t_2.
*Only if* (on the unbounded domain ℝ^m, i.e. arbitrary spacings): suppose one node computes P_m, m ≥ 4, with trigger
k. At least two other inputs a < b lie on the same side of k. Before it: P_m's projection onto (x_a, x_b) contains
(−2, −1.5) and (−1, −0.5) (extend each to a full point of P_m by placing the other events suitably), so the node's
product set contains (−1, −1.5), which strictly reverses t_a < t_b, and the node accepts a configuration outside P_m.
After it: the points (1, 1.5) and (2, 3) give the product point (2, 1.5), again a strict reversal. (Causality matters
only in forcing the "after" inputs to be vetoes, since hold lags are bounded above, but the product argument does not
depend on the synapse type.) □ *Remark:* on a bounded domain (all lags within the node's delays and windows) the
theorem fails: bounded windows then partition the whole range and one node can fix larger orders. E39's first grid
search (times 0–4) found such nodes for m = 4; the theorem is about scale-free order, which is what a clockless system
must compute.

**Theorem 2 (hierarchy).** Let an *atom* be a constraint t_j − t_i ∈ [lo, hi] (hi possibly ∞) or its negation
restricted to one pair, and a *zone* a conjunction of atoms (a difference-bound region).
(a) Depth 1 computes exactly the product sets of Lemma 1 (for the best choice of trigger).
(b) Every zone is computed at depth 2: one node per atom (trigger the later event, hold the earlier one with the
    delays and window that give [lo, hi]; a negated atom as a veto), then one max node over the atom nodes.
(c) Every finite union of zones is computed at depth 3: a min node over the zones' depth-2 outputs.
(d) Depth 1 is strictly weaker than depth 2 (Theorem 1 with m = 4).
*Proof of (b).* An atom node fires iff its atom holds, at a finite time; the max node fires iff all atom nodes fire.
(c) likewise with min. (a) is Lemma 1. □

**Verification (E39, `e39_depth.py`).** Exhaustive search over single nodes (every trigger, every role in {none,
hold, veto} per input, delays {0, 1, 2}, windows {0, 1, 2, ∞}) against the total order on distinct integer times:
m = 2 and m = 3 are computed (grid 0–11; for m = 3 the search returns exactly the proof's node: trigger t_2, hold t_1
with w = ∞, veto t_3 with w = ∞); m = 4 is computed on the bounded grid 0–4 (the remark's exception) and by no node on
the grids 0–7 (1,680 configurations) and 0–11 (11,880). All thirteen Allen relations built as atom nodes plus one max
node (depth 2) are exact on ~19,900 random interval pairs. The search is exhaustive over its parameter grid; the proof
covers all real parameters.

**Consequences.** (i) §64.2's "temporal depth = order-chain length" is wrong: E33's 3-node chains have depth 3, but
every Allen relation is a zone, so depth 2 suffices, and the before/meets relations are depth 1. (ii) What a single
node cannot do is exactly to order two inputs on the same side of its trigger: the representational reason depth is
needed at all in these networks. (iii) The veto with an infinite window (absence until the trigger) is what lets one
node reach m = 3; without vetoes one node orders only m = 2. (iv) Grokking and depth connect here: E34's classes are
zones over part events, so depth 2 is the natural architecture; deeper structure (unions, sequences of zones) needs
depth 3 and beyond.

## 72. Sleep is a reuse filter: a phase diagram for grokking

*Written 2026-09-27, from §66, §68, §69 and E37 (grokking in 2 of 3 seeds at every λ > 0; memorization absorbing at
λ = 0; one seed losing even training accuracy).*

**Survival of a parameter.** Consider one synaptic parameter w (a lookup entry, an operand delay's attachment, a part's
routing weight) in a network trained by errors-only pulls of size η (on the normalized budget, §68) and decayed by a
fraction λ per epoch toward its baseline. Let m be the number of training examples whose errors pull w, and e the
fraction of epochs in which such an example errs. Per epoch, w gains about m·e·η and loses λ·w, so its equilibrium is

  w* ≈ m·e·η / λ,

and w stays above the firing threshold θ only if the parameter's **reuse** m exceeds

  m* = λ·θ / (e·η).

**Two kinds of parameters.** A memorizing parameter serves one training example (m = 1: E37's pair node for (a, b)).
A shared parameter serves every example it takes part in (m ≈ n/p for an operand delay in E37; m ≈ n/K·(classes
sharing it) for a part). Sleep therefore keeps a parameter iff it is reused at least m* times: **decay is a minimum-
reuse prior**, the event-network form of a description-length penalty, acting through errors instead of a loss term.

**Phase diagram (predicted).** With n training examples and p operand values:
- memorization phase: λ < e·η/θ (m* < 1): per-example parameters survive; with error gating, memorization is absorbing
  (§69). E37 at λ = 0.
- grokking phase: e·η/θ < λ and n/p ≥ m*: per-example parameters are pruned, shared ones survive; the relation becomes
  the absorbing state. E37 at λ ≥ 0.02 in 2 of 3 seeds.
- collapse phase: n/p < m* (too little reuse for the decay): everything is pruned, including the shared route before
  it is learned. Training accuracy falls (E37 seed 1, train 0.40–0.48; the shared route there never became correct,
  so its errors never stopped and its parameters kept being pulled in inconsistent directions).
The grokking boundary in data is **n* ≈ p·λ·θ/(e·η)**: more data widens the grokking phase linearly, and a stronger
sleep needs proportionally more data. (The dense-network analogue is the weight-decay/data trade-off of grokking;
here it has an explicit reuse form.)

**Delay.** Within the grokking phase the delay is max(time for memorized entries to decay below θ ≈ ln(w*/θ)/λ,
time for the shared route to learn from the recurring errors), which is §69's refined form and matches E37's weak
λ-dependence above λ ≈ 0.05.

**Status after the phase diagram (E37, p = 31, 4 fractions × 4 λ × 3 seeds, cooled noise on the shared route).**
Three regimes confirmed: at 20% of pairs no λ groks; weak sleep memorizes (train ≈ 0.94, test at chance) and strong
sleep erodes the memorizer with nothing to replace it (train 0.64 at λ = 0.5): data-limited collapse, data threshold
between 20% and 30% of pairs (n/p ≈ 6–9). The predicted *shape* was wrong: the grokking region is bounded below by a
minimum sleep λ_min(n) that falls with data (30%: reliable only at λ = 0.5, 0.89–0.96; 50%: from λ = 0.05, 0.96–0.99;
70%: already at λ = 0.01, 0.95–0.98), and no upper edge appears up to λ = 0.5 above the data threshold. Reading: sleep
must dismantle memorization faster than errors relearn it, and more data makes the shared route learn faster, so less
sleep suffices; the collapse edge of §72 lies above λ = 0.5 once n exceeds the threshold.
**Predictions (M72, as first stated).** In an E37 sweep over training fraction × λ: (i) test accuracy is high only in a band of λ
whose upper edge rises linearly with n; (ii) below the band, train 1 / test chance; above it, train falls too;
(iii) the band vanishes when n/p < ~m* at the smallest useful λ.

## 73. Compositionality from the same filter

A part used by c classes is pulled by c classes' errors; a class-specific whole by one's. By §72 sleep keeps parts
with reuse ≥ m* and prunes wholes. This is the missing pressure of §62: E28's hidden nodes became whole-class
detectors because nothing penalized a node for being used by one class only, and E28 ran without decay.
**Test (E34 with sleep, 5 seeds): refuted as stated.** Depth 2 with a window bank: no sleep 0.89 / 0.87 / 0.84 at 5k /
10k / 20k episodes; λ = 0.05: 0.63 / 0.66 / 0.63; λ = 0.2: collapse (0.06). The reuse filter acts on what is learned.
In E34 the parts are a fixed basis and only the class routes are learned; a class's routing weights are reused only by
that class's examples, so sleep erodes correct routes instead of selecting parts. The prediction presupposed a network
that learns both parts and wholes; it remains untested there and is wrong as a general "sleep helps depth" claim.
**Predictions (M73, as first stated).** With sleep on hidden routing weights: (i) part selectivity rises relative to class
selectivity (§62's receptive-field measure); (ii) on shared-motif tasks, depth 2 overtakes depth 1 at smaller
training budgets than without sleep; (iii) the effect grows with the number of classes sharing each part.

## 74. Scaling: widen the candidate basis at logarithmic learning cost and constant inference cost

In the networks of §64–§71 a node's inference cost is the events on its live synapses (routing weights above
threshold): O(depth × k) per decision, independent of the number N of candidate inputs, because candidates without a
live synapse receive nothing. Learning which k of N candidates to route costs O(k log N) mistakes (§68). So the
architecture can grow its candidate basis (more channels, more part types, more window scales, more rhythms) with
logarithmic learning cost and no inference cost; a dense model pays ∝ N at every step for every candidate.
**Prediction (M74, E35 N-sweep):** with N = 12, 24, 48, 96 channels (same spikes per episode), accuracy stays ≈ 1,
synaptic events per episode stay constant, and updates grow ∝ log N, not ∝ N.

## 75. Stateful nodes: order-XNOR in one node, and parity representable by one toggle

**Arm/disarm breaks the product-set lemma.** E27's hold node is stateful: A's spike arms it, C's spike disarms it, B's
spike fires it if armed. Its accept set (A before B, and C not between them) is not a product set in lag coordinates
relative to B (whether C vetoes depends on t_A), so §71's Lemma 1 covers stateless nodes only. In Boolean terms, with
A before B, "C not between A and B" is XNOR(C after A, C after B): not linearly separable in those two comparisons, so a
rate-based network needs a hidden layer for it, and one stateful node computes it. **Proposition (M75a):** one
arm/disarm node computes the interval-exclusion predicate; no stateless node does (proof: it is not a product set
relative to any trigger; the pair of configurations that swap C across A while keeping the lags to the trigger's
side fixed gives the counterexample, as in Theorem 1). *Checked (E39b):* the arm/disarm node is exact on all distinct-time
configurations of grids 0–4 and 0–11, and no stateless node of E39's family (every trigger, role, delay in {0, 1, 2},
window in {0, 1, 2, ∞}) computes the predicate on either grid.

**Parity.** A toggle node (every input spike flips its state; it reports its state at a readout event) computes the
parity of the number of events on its live synapses: k-sparse parity over N channels is represented by one node with
k synapses. Sparse parity is a canonical grokking task for gradient training on dense networks, which need
N^Ω(k) steps under statistical-query-type lower bounds. **Open (M75b):** whether native credit can find the k
channels. Parity has no per-channel correlation, so a correlation-driven rule (including the pulls of §68) gets no
signal from single channels; what the toggle representation changes is that each labelled example becomes one linear
equation over GF(2) in the unknown channel set, solvable from ~N examples by elimination, if a native, local form of
elimination exists. That is the question, not a claim.

## 76. Grokking with depth, and the two kinds of collapse

*Written 2026-09-27 from E37 with timing noise and E41's pilots; full runs queued.*

**Collapse has two causes.** §72's collapse phase lumps together (a) *data-limited* collapse: shared parameters are
reused fewer than m* = λθ/(eη) times, so sleep prunes them along with the memorizers, and no learning rule can avoid
it; and (b) *search-limited* collapse: reuse suffices, but the shared route's error-driven search sits in a frustrated
configuration whose errors pull it in inconsistent directions, so it never becomes correct and sleep then erodes the
memorizer with nothing to replace it. E37's failing seed is (b): with timing noise on the shared route, cooled with the
route's own update count (E26c's schedule), all three seeds grok (0.97–0.98; pilot, σ = 2 at p = 31), including the one
that collapsed in every earlier run, and accuracy on the other seeds rises (0.96 → 0.98). The grokking phase is bounded
above by data and below by search; fluctuations widen it from below. The noise must scale with the ring (σ ≈ 0.065 p
was sufficient at p = 31 and 97, not at p = 59 with σ = 2: scaling sweep queued). *5 seeds at p = 31 (λ = 0.05):* σ = 2
groks in 5/5 (0.95–0.99); σ = 0, 1 and 4 in 4/5 (the same hard seed fails): an optimal exploration temperature,
not a monotone effect, as expected of annealing.

**Depth-2 grokking (E41).** (a + b + c) mod p cannot be computed by one rhythm read (one read adds one time to one
phase). The shared route composes two stages: a resets rhythm 1 and b reads it, emitting a spike whose phase is
e₁(b) − d₁(a); that spike resets rhythm 2, which c reads. With a memorizer available (one node per triple,
ρ ≈ 0.3 p³ / p⁴ ≈ 0.018 at p = 17) and errors-only learning split along the one causal chain (§56.4): without sleep the
network memorizes and stays at chance on unseen triples (0.06); with sleep and cooled noise it generalizes to 0.998–0.999
(pilot, p = 17, 30% of triples, 2 seeds). The chain's credit reaches the first stage through the second without
contraction, as §56.4 and §68 predict: the error on one chain moves every delay on it by the same step.

**Full runs (E41, 3 seeds per cell):** 30% of triples: 0.994–0.999 (p = 17) and 0.992–1.000 (p = 31); 10%: 0.79–0.93
(p = 17) and 0.94, 0.98, 0.03 (p = 31, ρ ≈ 0.003); without sleep train 1.0 and test at chance; without the chain (lookup
only) collapse (train 0.11–0.23, test at chance). Depth-2 grokking is confirmed, and the fraction needed falls with p.

**What is and is not given.** The shared route's form (two rhythm stages) is a resource, as E37's single rhythm is: it
restricts the shared route to compositions of cyclic additions. What the experiment shows is that a network which can
also memorize discovers the composed relation, with depth, under sleep; whether it chooses the correct depth when
routes of several depths are available is the next test.

## 77. Learning cost is set by activity, not by the size of the candidate basis

*Written 2026-09-27 from E35's channel sweep: with the spikes per episode held fixed, accuracy ≈ 1.0 and the number of
updates did not grow from N = 12 to 48 channels (566–1,493; 194–535; 330–1,198 per 100k episodes, 3 seeds each), while
synaptic events per episode fell (7.2–8.2 → 3.9–4.8).*

§68 bounds mistakes by O(k log N) because normalized pulls behave like Winnow over all N candidates. The observed
curve is flatter, and the reason is structural: a pull only touches synapses that were active in the episode. On an
error the counterfactual candidates are the channel pairs that actually spiked, s² of them for s active channels, not
N². A normalized pull moves weight onto the chosen pair and away from the node's other synapses in proportion to their
weight; synapses of channels that were silent keep their (small, shrinking) share without ever competing. The relevant
competitors are the channels that co-occur with the target in erring episodes, so the mistake count behaves like
O(k log s_eff), where s_eff is the number of channels active in an episode, independent of N when activity is fixed.
Inference is likewise O(live synapses reached by active channels). **Scaling law:** at fixed activity, the candidate
basis (channels, part types, window scales, rhythms) can grow without bound at no cost in either learning or
inference; costs grow only with activity. This is the learning-side counterpart of E5 (inference work tracks activity).
**Prediction (M77):** at fixed N, raising the spikes per episode (s) raises the updates needed roughly as log s; at fixed
s, N has no effect (E35 N = 96 running).

## 78. Composition makes grokking exponentially cheaper than memorization

A depth-D rhythm chain over D + 1 operands (E41 at D = 2) has (D + 2)·p + D real delays. With delays resolved to timing
precision δ it is a finite class of size (p/δ)^{O(Dp)}, so an Occam/PAC bound gives a sample complexity of
O(D·p·log(p/δ)/ε) for error ε, linear in depth. Memorizing the relation needs every one of the p^{D+1} operand tuples.
The fraction of the tuples needed to grok is therefore about

  f* ≈ D·p·log(p/δ) / p^{D+1} ≈ D log p / p^D,

which falls exponentially with depth: deeper compositions should grok from exponentially smaller fractions of the data,
provided the learning rule finds the chain (§76's search condition) and sleep's reuse threshold is met (§72: each shared
delay is reused ≈ n/p times, which grows with n). For E41 at p = 17, D = 2 the bound suggests a few percent of the
4,913 triples; at D = 1 (E37, p = 31) about 10% of 961 pairs, consistent with E26's generalization from 20–30%.
*Pilot (same day, 200 epochs, 2 seeds):* at 2% and 4% of triples the network memorizes (train 0.92–0.98) and stays
at chance on unseen triples: at this training length the bound is not the binding constraint. The shared route learns
only from errors, at most n per epoch, and needs a roughly fixed number U of error events (E26: ~10⁴), so its learning
time is T ≈ U/(n·e) epochs; with n ≈ 100–200 triples, 200 epochs are too few. The refined claim separates a data
threshold (Occam, above) from a time threshold (∝ 1/n); long runs at 2% and 4% (1,500–2,000 epochs) test which binds.
**Prediction (M78), as first stated:** in E41 at p = 17, generalization appears at training fractions of a few percent, far below E37's
fractions at D = 1 for the same p, and the threshold fraction falls with p.
**Result (E41f, E41t; 3 seeds each): refuted as stated.** At 1%, 2%, 4% and 7% of triples (200 epochs) the network
memorizes (train 0.75–1.00) and stays at chance on unseen triples (0.05–0.07); at 2% for 2,000 epochs and 4% for 1,500
epochs, the same. The time threshold does not bind, and the Occam bound is not the learner's threshold: at p = 17 it lies
between 7% and 30% (at 10%: 0.79–0.93). The bound counts hypotheses; it says nothing about whether the error-driven search
with sleep reaches the chain before memorization absorbs the errors. The candidate constraint is §72's reuse filter: a
shared delay is reused ≈ n/p times per epoch, and below the reuse threshold sleep prunes it as fast as errors build it.

## 79. An asynchronous world model: the weaving operator as a learned temporal point process

*Written 2026-09-27, before the E43 world model was built beyond a direction classifier (which is not a world model:
it predicts one binary label, not what happens next, when, or with what probability).*

**What a world model of an event stream is.** A generative model of the process: given the history (and a latent
state carried between events), the probability of which event happens next and when. The standard formalism is the
temporal point process with conditional intensities λ_e(t | history): the next event is the first arrival among
competing risks, with density λ_e(t)·exp(−∫ Σ_e' λ_e'). Frontier instances: Hawkes processes (self-exciting
intensities), neural Hawkes processes (continuous-time LSTM, Mei & Eisner 2017), Transformer Hawkes processes (Zuo et
al. 2020), intensity-free models of inter-event times (Shchur et al. 2020). Latent-dynamics world models for control
(Dreamer's RSSM, Hafner et al.) add a deterministic recurrent state and a stochastic latent; JEPA-style models predict
in a learned representation rather than raw events.

**The correspondence to the manifesto.** The weaving operator W is exactly a competing-risks point process: the state
is a pool of pending future events with latency distributions; at each moment one fires, cancels itself from the pool,
and updates the others' latencies (§§1–5 of the manifesto's "initial thoughts"). A race of noisy latencies with Gumbel
timing noise is a softmax over event types (§44), so a race network is a native sampler of the competing-risks
distribution. A world model for this project is therefore W itself, built and learned.

**Design (E43).**
1. *Pool.* For each event type e (up-move, down-move, large aggressive buy, large aggressive sell, burst), a pending
   prediction whose latency distribution is set by the current state. Output: P(next type) and its timing
   distribution; sampling = one noisy race.
2. *Latent state in continuous time.* Fast state: exponentially decaying traces of recent events of each type (the
   Hawkes excitation, as leaky integrators between events). Slow state: a regime estimate (activity rate,
   volatility), updated at events. Persistent memory: hold loops (§59) when patterns must be carried longer.
3. *Learning = online likelihood at every event.* When type e occurs at t, each type's log-intensity parameters move
   along the competing-risks log-likelihood gradient: up for the type that happened (at its actual time), down in
   proportion to each type's integrated intensity since the last event. The native form is the positional rule (pull
   the prediction that should have won toward its time, §54) plus normalization over the pool (the "others were less
   likely" term arrives through the shared budget, §68), not a push of losers in time.
4. *Learning to learn.* (a) Complementary learning systems (McClelland et al. 1995; fast weights, Ba et al. 2016):
   fast parameters track the current regime, slow parameters are consolidated in sleep between sessions and keep only
   structure reused across days (§72). (b) Plasticity gated by surprise: a Bayesian online change-point estimate
   (Adams & MacKay 2007) of the stream's regime raises fast plasticity after a break and lowers it when the model is
   reliably right, which is §45's optimal tracking rule with the noise ratio estimated online.
5. *Decisions from the model.* Roll the race forward over the trading horizon to obtain the distribution of the price
   change; act only when the expected gain minus cost exceeds a profit-learned price (E42).

**Richer native state (same day):** hold/trigger pair parts over event types as extra state, started near zero:
neutral (−2.643/−2.854/−2.696 vs −2.641/−2.852/−2.695); a log-linear intensity with learned inhibition (to express
suppression, e.g. bid-ask bounce): worse (−3.26/−3.22/−3.08), because multiplicative steps on the log-intensity weights
saturate at every event and the weights random-walk. The gap to the neural point process stays open. With log-domain steps normalized per synapse (Adam on the
log-weights) the inhibitory model is stable but still below excitation-only (−2.80/−3.13/−2.94): suppression is not
what the GRU adds here, or not in this form.

**Stage 1 results (E44, 3 pilot days, nats per event):** Poisson −3.00/−3.42/−3.32; Hawkes (Adam) −2.62/−2.94/−2.84;
native −2.64/−2.85/−2.70; native with fast/slow surprise-gated plasticity −2.64/−2.85/−2.69; GRU neural point process
–/−2.61/−2.52. The first prediction holds (native ≈ Hawkes, both far above Poisson); the native model trails the neural
point process by ≈ 0.2 nats, consistent with its latent state being only fast traces; plasticity gating is neutral on
day averages (its test is the window after regime breaks).

**Evaluation.** Online (prequential) log-likelihood per event of type and timing, against (i) a Poisson baseline per
type, (ii) an online multivariate Hawkes process with exponential kernels, (iii) a GRU-based neural point process
trained online by backprop; adaptation after regime changes (likelihood in the hour after a volatility break); then
profit after costs through the decision layer. **Predictions (M79):** the native pool beats Poisson and matches the
online Hawkes process in likelihood (both have the same excitation structure); surprise-gated fast/slow plasticity
improves likelihood after regime breaks relative to fixed plasticity; whether the native model approaches the neural
point process is open.

## 80. Structure discovery as online model selection, with an implicit Occam razor

*Written 2026-09-27 from E45's pilot: given a menu of routes (single rhythms over each operand pair, a two-stage chain,
and a triple lookup), the network picks the (a, b) rhythm for (a + b) mod p (test 1.000, 2/2 seeds), the depth-2 chain
for (a + b + c) mod p (0.999, 1 of 2 seeds; the other did not converge in 120 epochs), and no shared route for a random
table (test at chance).*

**Mechanism.** Each route keeps a price, a running estimate of its error rate, updated on every example from its own
answer, including routes that did not answer: evaluating every route is cheap in an event network (each is a few
delays and one race), so route selection has full-information feedback. The cheapest route answers. This is online
model selection over M experts with full information, the setting of Hedge / multiplicative weights, with regret
O(√(T log M)): the network converges to the most reliable route on the menu, and a route that cannot express the
relation keeps a price near its chance error rate.

**Implicit Occam razor.** For a + b, the two-stage chain can also express the relation (it only needs to learn to ignore
c), yet the single (a, b) rhythm won in both seeds (its price reached 0.000, the chain's stayed ≈ 0.92). A route with
fewer parameters becomes reliable after fewer errors (§68: O(k log N) mistakes; §78: sample complexity ∝ its
parameter count), so its price falls first and it takes over before the larger route has learned. Racing on reliability
therefore prefers the simplest route that explains the data, without an explicit complexity penalty. **Predictions
(M80):** (i) with more seeds and epochs, a + b selects a single pair rhythm, a + b + c the chain, random tables no
route; (ii) the selected route is the one with the fewest parameters among those that can express the relation; (iii)
adding more candidate routes (more pairs, more chain orders) costs O(log M) extra errors, not O(M).

**Per-class menus (E46, pilot).** The same mechanism inside each class of E34's composition task: a class keeps a few
candidate (hold part, trigger part) routes, proposed from its misses and priced by their own reliability; it answers
through its cheapest route that fires. With a generic window bank this reaches 0.92 mean at 20k episodes (3 seeds;
0.86–0.98), against 0.86–0.87 for weighted routing even at 200k: a route that locked onto the wrong parts is priced out
and replaced instead of trapping the class. The network has **no synaptic weights**: learning is a discrete choice of
wiring by reliability, which is the manifesto's claim that plasticity serves computation (here it only selects which
temporal predicates to compose).

## 81. A mistake bound for route menus (proposition, with proof sketch)

**Setting (realizable, noise-free labels).** K classes; P part types (E34/E46's hold nodes on channel pairs at several
window scales). Class k is exactly one route (h_k*, g_k*): it holds iff part h_k* fires within W before part g_k*.
Distractor spikes are independent of the class. For a wrong pair (h, g) ≠ (h_k*, g_k*), let q ≥ its probability of
occurring (h before g within W) in an episode of class k; the true pair occurs in every episode of class k. Δ = 1 − q.

**Learner.** E46: on a miss of class k, every pair of fired parts (earlier → later within W) is counted for k; the
most frequent pair is proposed as a route; routes carry prices (running error rates); a class answers through its
cheapest firing route below price 1/2.

**Proposition (M81).** (a) After m misses of class k, the true pair is the most frequent candidate with probability at
least 1 − P²·exp(−2mΔ²) (Hoeffding over at most P² wrong pairs, each with mean ≤ q against the true pair's 1). So
m = O((log P + log(1/δ)) / Δ²) misses suffice with probability 1 − δ. (b) A proposed true route never errs on class k's
episodes; its price decays to 0. A wrong route for k fires on episodes of other classes or of none with probability
≥ some ε > 0 unless it is equivalent to the true one on the data; each such firing is an error that raises its price,
so it is priced out after O(1/ε) errors. (c) Total mistakes: O(K·(log P + log(1/δ)) / Δ² + K/ε), independent of the
number of episodes and logarithmic in the part basis. □ (sketch)

**What it says.** The cost of discovering structure is set by how often the wrong structure *looks* right (the gap Δ),
quadratically, and by the size of the candidate space only logarithmically: widening the part basis is cheap, noisy
data is expensive. **Predictions:** (i) updates grow with distractor density roughly as 1/(1 − q)²; (ii) updates grow
at most logarithmically with the number of window scales (part types).

## 82. What the market stream's world model needs: the time since the last event (E44, E48)

*Written 2026-09-27. Diagnostic path: decomposing each model's log-likelihood into a type part (log P(type | an event
now)) and a timing part (log of the total rate minus its integral) showed the GRU neural point process's 0.2-nat lead
over the native Hawkes-type model was entirely in the type part (−1.08 vs −1.27 nats on day 1; timing −1.31 vs −1.37).
Count baselines then located it: conditioning the next type on the last two types recovers little (−1.16), on the last
type and the time since it almost everything and more (−0.92, better than the GRU's −1.08).*

**The model.** A semi-Markov (Markov renewal) marked point process: the hazard of the next event and the distribution of
its type depend on the last event types and on the elapsed time, piecewise constant over a bank of windows (0.01, 0.05,
0.2, 1, 5 s). The model class is classical (Lévy 1954; Pyke 1961). As an event network: state nodes armed by the last
event and disarmed by the next (§75); window nodes opened in turn by expiry events of a delay line from the last event
(§61); hazard and type detectors whose rates are set by synapses from the armed state and the open window; learning is a
count on the synapse that predicted the event and exposure on the hazard synapses. E48 verifies that the event network
reproduces the table form's likelihood exactly (to 2·10⁻⁴ nats), at ≈ 4 network events and ≈ 19 synaptic operations per
market event.

**Why it beats the neural point process here.** (i) The predictive structure of this stream is a depth-1 temporal
predicate relative to the last event: the lag to the trigger, which §71 says one node computes. (ii) For a piecewise-
constant model, online counts are exact sufficient statistics: after every event the model is the maximum-likelihood fit,
with no step size; a recurrent network sees log Δt as an input and must learn its effect by stochastic gradient steps,
far less sample-efficient online. (iii) Non-exponential waiting times (the timing part improves from −1.31 to −0.78 on
day 1) come for free from the window bank; exponential Hawkes kernels cannot express them.

**Result (days 1–3, online, total nats per event):** GRU −2.39 / −2.61 / −2.52; semi-Markov event network (last two
types) −1.69 / −2.06 / −2.07. **Open:** whether it holds on held-out days with learning frozen (a count model with 96
contexts × 6 windows could track day-specific statistics), and whether a GRU given the same window-bank inputs closes the
gap (i.e. whether the advantage is the representation or the learning rule).

## 83. Learning an ordered conjunction: summed potentials, full-information credit, and why the threshold must exceed half the budget

*Written 2026-09-27 (before E34w).*

**Setting.** A class node has conserved hold weights h and trigger weights g over P part nodes (each sums to 1). It
fires at the first instant τ at which a part fires and both potentials exceed θ: the held input
H(τ) = Σ h_e over parts fired in [τ − W, τ) and the coincident trigger input G(τ) = Σ g_l over parts firing at τ.
Summation is what a membrane does; E34 instead required one synapse per role above θ = ½ (a hard route choice).
Target: a route (e*, l*) present (e* in the window before l*) in every positive example of the class and in no
negative one. Credit, on events only:
- miss (the class should have fired and did not): multiply by (1 + α) the hold weight of every part that fired within
  W before some later part, and the trigger weight of every part that fired within W after some earlier part; renormalize;
- false fire at τ: multiply by (1 − β) the hold contributors in [τ − W, τ) and the trigger contributors at τ; renormalize.

**(i) Ratio monotonicity (proof).** Renormalization multiplies every weight of a role by the same factor, so it cancels in
h_{e*}/h_j. On a miss e* is always promoted (it fires before l* within W), so log(h_{e*}/h_j) never decreases and grows by
log(1 + α) on every miss in which the distractor j is not promoted. If j is promotable in at most a fraction f of the
class's misses, its relative mass shrinks as (1 + α)^−(1−f)m after m misses, and the role concentrates past θ once
Σ_j h_j/h_{e*} < (1 − θ)/θ: O(log(P)/((1 − f)·log(1 + α))) misses per role, logarithmic in the candidate basis
(as §77, §81), unaffected by demotions of the kind in (ii) unless they hit e*.

**(ii) The AND-credit threshold (proof).** A false fire does not say which role was wrong. Let Φ = −log h_{e*} − log g_{l*}
(the relative entropy to the target). The example is negative, so the target route is not present at τ: at most one
target is among the demoted contributors. The demoted sets carry mass m > θ in each role (the node fired). A role whose
target is demoted changes log-weight by log(1 − β) − log(1 − βm) ≥ log(1 − β) − log(1 − βθ); a role whose target is not
demoted gains −log(1 − βm) > −log(1 − βθ). In the worst case

  ΔΦ < 2·log(1 − βθ) − log(1 − β),  which is negative iff (1 − βθ)² < 1 − β  ⟺  β < (2θ − 1)/θ².

So **conservation resolves the AND's credit ambiguity exactly when θ > ½ of the budget**: whichever role was wrong,
renormalization moves more mass onto its target than the demotion took from the other role's target. At θ = ½ the
worst-case drift is positive for every β (log((1 − β/2)²/(1 − β)) > 0), and below ½ it is worse. E34 ran at θ = ½.
Promotions never increase Φ (each role's promoted set contains its target: ΔΦ_role = −log(1 + α) + log(1 + α·m_U) ≤ 0).

**(iii) Copies cooperate only if potentials sum.** A window bank contains k copies of a part that co-fire. Full-information
promotion gives them equal shares; with a one-synapse threshold none reaches θ once 1/k < θ, which is the E34 bank failure
(classes that never form a route). With summation their shares add.

**Predictions (E34w, 5 seeds, E28/E34 task, 15 classes, θ = 0.6, α = 1, β = 0.3; β < (2θ − 1)/θ² = 0.56).**
P1: tuned part windows ≥ 0.99 test on at least 4 of 5 seeds (E34: 0.97; 0.954 after 200k episodes).
P2: window bank {1, 2, 4} within 0.02 of tuned (E34: 0.87 vs 0.954).
P3: θ = 0.4 and 0.5 worse than 0.6–0.7 (the drift of (ii)); at 10k episodes.
P4: N = 16 → 32 channels (P = 240 → 992 part nodes): updates grow by ≲ 1.5× (log P ratio 1.26), not 4×.
If P1 holds, the chains match the Transformer (0.998 after 2M episodes) with ≈ 10⁴× less inference compute and ≈ 50×
fewer training episodes.

**Results (E34w, 5 seeds).** P1 fails: tuned windows 0.977 / 0.911 / 0.990 / 0.983 / 0.981 at 40k episodes, with 890–1,531
updates (the plateau is reached by 8k episodes and then constant, apart from transient drops such as seed 1's 0.989 →
0.911). Part of the floor is the task, not the learner: the class window W = 3.5 inherited from E34 is shorter than the
longest span between the two part events (gap ≤ 2.5 plus the second motif ≤ 1.5), leaving 0.07–1.4% of test examples
unreachable per seed; the transient drops come from continued corrections on such irreducible errors. P2 fails
badly: the window bank collapses (0.05–0.10, ≈ 36k updates, all misses). This is the instant-splitting deadlock found
afterwards (§84, Proposition 2): with wide part windows, cross pairs between the two motifs fire in every positive at
instants other than the target's. P3 fails: θ = 0.4, 0.5 and 0.7 give the same accuracies seed by seed; the worst-case
drift of (ii) does not arise here (few false fires). P4 holds: N = 16 → 32 channels raises updates 1.3× (mean 1,150 →
1,495; predicted ≲ 1.5×, log-ratio 1.26). Follow-up (E34x): W = 4.2 and §84's instant credit, tuned and bank.

## 84. Depth 3: order among three parts, credit at an instant, and exploration toward the absorbing route

*Written 2026-09-27, after single-seed diagnostics of E53 (stated as such) and before its 5-seed runs.*

**Why depth 3.** With summed potentials (§83) a class node's held input is a sum: H > θ can say "A and B were both held"
but not "A before B". So classes that are different orders of the same motif set (E53: (A, B, C) vs (B, A, C)) need a
unit whose firing already encodes an order: a composite node (u → v) over two part nodes, firing at v's spike if u fired
within W₂ before. The class node then holds the composite A→B and is triggered by C.

**Proposition 1 (activity pays for the expansion).** Let a layer of candidate composites contain one node per ordered pair of
units below: P² nodes for P parts, P^(2^(d−1)) at depth d. Only composites whose inputs both fire within the window
ever produce an event, so the work per episode is the number of fired composites, at most (units fired per window)²,
independent of P², and a composite that never fires is never touched by learning. The mistake bound of full-information
Winnow grows with the log of the basis (§83(i)): 2·log P at depth 3, not P. Winnow over an exponential feature
expansion is intractable to simulate in general (Khardon, Roth & Servedio, JAIR 2005, for monomial kernels); **in an
event network temporal sparsity is what makes the expanded Winnow tractable**: an unfired composite has no cost, and
the fired ones are bounded by the event density per window. Dense streams (SHD, §55) are exactly where this breaks.

**Proposition 2 (deadlock of union credit at depth ≥ 3).** Under §83's union credit, every unit that fires in every
positive example with some unit before it (in the class window) is promoted at every miss, in the trigger role. At
depth 3 the intermediate units (part B and composite A→B, at B's end) fire in every positive, and so do the targets
(part C, composite B→C, at C's end). By §83(i) promotions never change the ratios among always-promoted units, so the
trigger mass stays split between two instants. If no single instant carries more than θ, the node never fires, no
false fire ever occurs, and no demotion (the only operation that separates them) happens: a fixed point without
learning. At depth 2 (E34) the first part has a predecessor only through noise (f < 1), so the split does not arise.
Observed (E53, one seed): trigger mass 0.26 / 0.26 / 0.25 / 0.23 over those four units, test 0.36, every error a miss.

**Instant credit.** On a miss, credit one instant τ̂ (a fired unit with an earlier unit in the window): promote the hold
units in [τ̂ − W, τ̂) if H(τ̂) ≤ θ and the trigger units at τ̂ if G(τ̂) ≤ θ (deficient roles only, full information within
the instant). The promoted mass is then ≤ θ, so a promotion that misses the target costs Φ at most log(1 + αθ), while
one that contains it gains at least log((1 + α)/(1 + αθ)).

**Proposition 3 (the valid route is absorbing; greedy can cycle).** Call a route valid if it is present in no negative
example. A valid route's units are demoted only as co-contributors of a false fire caused by other units, and then
§83(ii) still lowers Φ. So once a valid route carries θ in both roles it fires on every positive and is never removed:
it is an absorbing state (as the rule flip of §69). An invalid route (a prefix shared with another class) fires, is
demoted below θ, then misses its own examples; greedy selection (τ̂ = the instant closest to firing) credits the prefix
again because the prefix instant still holds most of the mass: a cycle that never visits the valid instant. Choosing
τ̂ at random with probability ∝ exp(min(H, G)/T) makes every candidate instant reachable, so the walk hits the absorbing
route with probability one; T trades search (large T) against wasted promotions (small T), as the cooled noise of §76.
Observed (one seed, 20k episodes): greedy 0.725 (the prefix cycle, seen in the weights: hold on part A 0.75, trigger at
B's end); T = 0.1 / 0.3 / 1: 0.998 with 1,022 / 1,142 / 1,423 updates; depth 2 with instant credit 0.44.

**Predictions (E53, 5 seeds, 40k episodes; 6 motifs, 5 sets × 4 orders = 20 classes, permuted decoys).**
P1: depth 3, instant credit, T = 0.3: ≥ 0.99 on ≥ 4/5 seeds.
P2: greedy (T = 0) fails on ≥ 2/5 seeds (< 0.95); union credit < 0.6 on all seeds (deadlock).
P3: depth 2 < 0.7 on all seeds (a summed hold is unordered).
P4: updates stay within 3× of depth 2 in E34w (log P² = 2 log P), and events per episode ≈ 40 (activity, not P² = 57,600).
P5: an event-token Transformer given a fixed 40k-episode training set (2M presentations, AdamW) stays below the chains;
given 2M fresh episodes it may match them, at ≈ 10⁴× the inference cost.

## 85. Synapses grown on credit: exact dense Winnow at the cost of activity, and the price of depth

*Written 2026-09-27 (before E54's 5-seed runs).*

**Representation.** A class node's role weights over a candidate basis of Q units (all chain units up to level L:
Q = P + P² + … + P^(L+1)) are stored as w_j = s·v_j for units with a grown synapse and w_j = s·u for all others (one
shared value u per node), with a scale s. A multiplicative update of a set S grows synapses for the units in S that lack
one (v_j := u), multiplies their v_j, and renormalizes by recomputing s from Σ v_j + (Q − n)·u (n grown synapses);
folding s into v and u is a change of representation.

**Proposition (exactness).** From the uniform start w_j = 1/Q, this lazy network produces the same weights, hence the same
decisions, as dense Winnow with a conserved budget. *Proof.* By induction over updates: in the dense algorithm every
unit never in an updated set has been multiplied only by the normalizers, so all such units share one value, which is
s·u; a grown unit has additionally been multiplied by its own update factors, as v_j. The normalizer computed from
Σ v_j + (Q − n)·u equals the dense total. ∎ *Verified (E54, D = 3, Q = 144,780):* identical decisions over 8,000 test
episodes (equal decision hash), identical update counts and accuracy, dense vs lazy.

**Consequences.** (a) The mistake bound keeps its log Q = (L + 1)·log P (§83(i)): from 1/Q a target needs log₂(θQ)
net doublings, linear in depth. (b) Memory is the number of distinct units ever in a credited set (8.5k grown of
145k in 4k episodes at D = 3); the basis itself (5.5·10⁷ at depth 4, 2·10¹⁰ at depth 5) is never stored. (c) Time per
episode is the number of fired units. A candidate that never fires, or fires but is never credited, costs nothing.

**The price of depth is activity.** A chain unit at level k fires for each part firing within W₂ after a firing level
k − 1 unit, so the fired units per episode grow as n·r^L, with r the parts firing per window: measured 44 / 155 / 454
events per episode at depth 3 / 4 / 5 (r ≈ 3). This is exponential in depth with base r (the stream's density), not
in P: cheap for sparse streams and moderate depth, and the reason dense streams (SHD, §55) are hard. A Transformer's
cost is O(layers·n²·d), polynomial in depth, so a crossover depth exists, L* ≈ log(layers·n²·d)/log r (≈ 10 here).
Demand-driven propagation (extend a unit only if it has grown synapses downstream) would cut n·r^L to the credited
paths; untested.

**Predictions (E54, 20 channels, 8 motifs, 5 sets × 4 orders = 20 classes; instant credit, T = 0.3).**
P1: depth 4 (L = 2): ≥ 0.99 on ≥ 4/5 seeds at 40k episodes. P2: one level short (L = 1): < 0.8 on all seeds.
P3: updates to plateau grow at most linearly with depth: depth 4 ≤ 2× depth 3 (log Q ratio 1.5).
P4: an event-token Transformer on the same task with a fixed 40k-episode training set stays below the chains.

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
| **M31** | real-weight hidden credit is dominated by its common mode, growing with depth; FA is immune; centering alone fixes the deep pivotal path | `common_mode_share` per layer for crl_fa, crl_pivot, crl_pivot + centering at κ = 0.001 (E14 depth 3) |
| **M32** | the closest-loser residue is the k-th Gumbel spacing (mean σ/k); σ = k × residue self-calibrates the temperature | measured residue vs 0.15/k; `--self-sigma 1` vs `2` vs fixed σ |
| **M33** | backward credit through positive weights collapses to the Perron direction (power iteration); mean centering leaks it at the evidence scale; Perron centering fixes it | `perron_share` vs `common_mode_share` per layer; `--center-credit 2` vs `1` at depth 2 and 3, κ = 0.001 and 0.03 |
| **M34** | time translation: timing credit sums to the deadline's credit; exact kernels conserve sum errors, so share Jacobian + centering trains depth 3; scale gauge: fixing ρ lets prices alone own urgency | `--share-jac 1` with and without `--center-credit 1`; `--gauge 1` on crl_pivot and crl_fa; `f_eff` over training |
| **M35** | timing Jacobians are Markov kernels: forward contrast and backward credit contract by one Dobrushin coefficient; boundary (near-miss) credit is the only non-contracting channel; inhibition and selection restore contrast | `dobrushin` per layer over training; non-negative penalty at depths 1, 3, 5; crl_fa − fired_only gap vs depth, multi-seed; sparse vs dense fan-in by depth |
| **M36** | excitatory race networks are topical maps: sup-norm non-expansive; certified jitter radius ε* = min(½ margin, ½ race gaps, horizon distances) is exact | `--certify 1`: zero flips among certified samples (non-negative); radii vs EVT null; signed networks for contrast |
| **M37** | weaving costs σ × surprisal; near-miss credit = its gradient; the soft value is a submartingale with commitment as compensator; topical bracketing certifies early commitment with zero rollback | self-supervised commitment objective vs certified radius and E7; fraction and time saved by certified early decisions; commitment cost vs σH(π) under Gumbel sampling |
| **M38** | prices (thresholds) must be the fast timescale: rules with common-mode credit need κ ≳ η; rules without it are insensitive to κ | κ sweep 0.001 to 0.1 for crl_fa and uncentred crl_pivot, depth 3 (debug) |
| **M39** | topology from information: k·F ≥ G; widths shrink by F^{1−α} per layer (α from patch entropy); sparse fan-in beats dense at depth; commitment entropy falls with depth | `m39_patch_entropy.py`; `--fanin` 8/16/64 × `--winners` 1/2/3; `--widths` pyramid vs inverted (depth 3, debug) |
| **M40** | the optimal weaving rule prices commitment cost (MSPRT: shared free-energy inhibition beats the absolute race); race neurons are blind to absence, usable clock-free only relative to the input's onset | `--speed 1`: absolute vs relative frontiers (depth 1, 3); onset-referenced absence inhibition (M40b, to build) |
| **M41** | σ should equal the substrate's timing-noise scale (anneal to 0 in deterministic simulation) | Gumbel timing noise injection σ_s; best learning σ vs σ_s; spacing estimator of σ_s |
| **M42** | generalisation gap falls with the certified jitter radius (algorithmic robustness) | train − test gap vs median ε* across `--certify 1` runs |
| **M43** | pattern-level chaos: ρ_{l+1} ≈ A√ρ_l (no ordered phase), doubly-exponential decorrelation, A ∝ 1/√k, topographic codes damp it | `rho_per_layer` from `--certify 1` at depths 3 and 5, k = 1 vs 3, trained vs untrained |
| **M44** | Gumbel race: decision time ~ Gumbel(F, σ), independent of the winner; time pricing scales competitor credit by (1 − λσ) | accuracy vs decision time at fixed input under injected timing noise; λ sweep on the output rule |
| **M45** | continual learning is tracking: η\* ∝ √(q/r); error-gated rules inject Bayes-error noise; excess-surprisal gating learns only after change | E17 η sweep and `--gate 1`; E7 class-blocked streams |
| **M46** | weaving is local (consistent cuts); certified outcomes are pointer states, noise-proof at rate e^{−2kε/σ}; counterfactual influence obeys a soft light cone b^m; repair = instanton | gap distributions from `--certify` runs; record redundancy vs output dependence; credit reach vs b |
| **M47** | flat supervision: random feedback makes every layer fit label templates independently, so depth re-encodes; composition needs credit that depends on the layer above | `--probe 1` per-layer linear readout: random feedback, frozen, local feedback, pivotal (depth 3) |
| **M48** | a residual event stream (identity paths) stops the accuracy decline with depth and lets deep layers add features | `--residual 1` vs plain, depths 1, 3, 5, full length |
| **M49** | the entropic k-winner race is Fermi–Dirac (chemical potential = price); annealed soft-race training recovers the cancellation cost under exact gradients | `e21_soft.py` depth 2, k = 3, hard evaluation vs E20 k = 3 (0.930) and k = 10 (0.951) |
| **M50** | race layers are equivariant under shift and dilation of time; temporal collapse shrinks deep weight gradients; running temporal normalisation (a gauge choice) restores depth | `e20_exact.py --tnorm 1` at depths 2 and 4 (no cancellation) vs without |
| **M51** | the conserved near-miss rule is ultraconservative: forgetting bounded by the new task's mistakes (flat in time on task), vs O(log T) for softmax SGD; homeostasis is the non-conservative leak | E23: race vs MLP forgetting at 1k/4k/16k frames per task; race without homeostasis |
| **M52** | the class prior belongs in the output prices; without them the prior shift is written into old classes' feature weights | E23 task-aware forgetting with `--price` 0.003/0.01/0.03; frozen hidden; homeo 0 |
| **M53** | without sleep the race's memorization is absorbing; sleep downscaling (Pegasos) gives grokking with delay ∝ 1/λ; plasticity peaks and hidden codes merge at the transition; sleep trades forgetting for generalization | E24: `e24_grok.py race --sleep` sweep, frozen hidden, training fraction sweep; E23 with sleep |
| **M54** | delays instead of lookup: a ring computes (a+b) mod p compiled; delays as phasors learn exactly the one-character relations; replay (power-method synchronization) generalizes above f_c(p), online error-gated learning is frustrated | E25: `e25_delay_ring.py` compiled / sync fraction sweeps p = 31, 59, 97 / ops / online / learn / table; E24 MLP at matched fractions |
| **M55** | in a race, winning is positional: pull-only error-driven learning converges, the wrong-winner push collapses the detectors; the main race rule's competitor push may be a cause of its weak results | E26 `e26_noisy_race.py --push 0/1 --sigma`; E26b `--compete 0` on E22 SHD and E24 |
| **M56** | clockless = shift-equivariant: no sums of times; one oscillator reference gives one cyclic character; credit flows along one critical path; teach only the event that should have won, toward an anchored time; veto is the native way to make something later | E27: learnable delays on coincidence + veto synapses, pull-only vs with loser push, with vs without veto, on 'B within Δ after A unless C' patterns |
| **M57** | routing needs counterfactuals; cancelled near-misses supply them at no extra events: pull the wanted cancelled node on its best partial window | E28 `e28_routing.py` arms path / fired / nearmiss (k = 1, 2) / push, depth 1 vs 2 |
| **M58** | generalization claims need the capacity ratio ρ = n / params; grokking = relation reached at ρ ≪ 1 while memorizers exist in the class | E29: general race network + recurrent delay loops as a resource, native credit, on E24; controls without loops and without counterfactual credit |
| **M59** | the operator basis + one reference is Turing-complete (two-counter machine); restoration by a comb coincidence per cycle makes reliability independent of program length | E30 `e30_minsky.py`: exact on all programs; success vs q/σ with and without restoration, 25 vs 81 steps |
| **M60** | pull-only on weights needs conserved per-node budgets (fractional steps), conserving weakening and prices; readout capacity (Cover) bounds memorization | E26b `--compete 0` on SHD (refutes transfer); E29 readout on frozen hidden, 96 vs 384 nodes |
| **M61** | aligning (delay) destroys the interval's content, holding (PSP duration) keeps it; veto and ordering need duration; delay and duration are separate learnable parameters with separate credit | E27 `--tol hold` vs `align`, veto vs no veto; with duration shrinking on late-partner false fires |
| **M62** | depth fails because hidden nodes become class detectors; motifs longer than the window are out of reach of pulls; hold-then-align hidden learning makes them motif-selective and lets depth pay | E28 depth 2 with learnable hidden durations; motif vs class selectivity; E28c task |
| **M63** | at matched accuracy an event learner beats the cheapest clocked dense model by ~10² in operations, growing linearly with silence; vs a sparse (event-driven) dense model only ~10× | E32 `e32_frontier.py`: dense conv over F × δ, silence padding; E27 synaptic-event counts |
| **M64** | order = asymmetric PSP durations; one node = one bounded difference with exclusions; fixing the order of m points needs an (m − 1)-node chain (temporal depth = order-chain length); O(VC ≈ 2 + N) mistakes per node | E27 `--tol hold` (directional) with and without veto; E33 `e33_allen.py`: all 13 Allen relations exact |
| **M65** | depth pays when composition is a hold/trigger chain; routing to channels and parts is learned with O(N) mistakes per node | E34 `e34_compose.py` depth 2 vs 1, window variants; E35 `e35_free.py` nothing given |
| **M66** | per-class parameters cannot generalize on (a + b) mod p (each (class, operand) pair is seen once); a shared intermediate is needed, and a sum-in-time needs a rhythm; grokking = routing moves from per-class to shared routes under decay | E37: memorizing network + relays + rhythm, native routing, with and without the rhythm, with and without decay |
| **M68** | conserved fractional pulls are normalized online learning on the simplex (Winnow/Hedge family): O(k log N) mistakes per node, additive over a chain's depth; durations O(log range), vetoes O(N) | mistake counts vs N (E35 with N = 12, 24, 48) and vs depth (E34); additive vs normalized pulls |
| **M69** | grokking = flip of the absorbing state: without decay memorization absorbs; with decay the relation absorbs; delay ∝ 1/λ; inexpressible relations stay memorized | E37 λ sweep curves; lookup share of test answers; op controls (poly, rand) |
| **M70** | order is native to hold/trigger nodes and synthesized by attention; relative-time attention bias is the fair Transformer baseline; no Transformer matches accuracy at comparable cost | E36 long training; E36 with `--reltime 1` on E27 and E28 |
| **M71** | depth hierarchy: depth 1 = product sets in lag coordinates, one node orders m events iff m ≤ 3; depth 2 = difference-bound zones; depth 3 = finite unions | E39: exhaustive single-node search on a grid (m = 3 found, m = 4 none); depth-2 Allen relations verified |
| **M72** | sleep keeps parameters reused ≥ m* = λθ/(eη) times: memorization / grokking / collapse phases; data boundary n* ∝ pλθ/(eη) | E37 phase diagram over training fraction × λ |
| **M73** | the same reuse filter makes hidden nodes parts rather than wholes; with sleep, depth pays at smaller budgets | E34/E28 with decay on hidden routing; §62 receptive-field measure |
| **M74** | candidate basis grows at O(k log N) learning cost and constant inference cost | E35 N = 12, 24, 48, 96 |
| **M75** | stateful arm/disarm nodes compute order-XNOR in one node (not a product set); toggle nodes represent k-sparse parity; native learnability open | exhaustive check of stateless vs stateful nodes on interval exclusion; toggle-node parity learning |
| **M76** | collapse is data-limited or search-limited; cooled noise removes the second; depth-2 grokking of (a + b + c) mod p through a composed rhythm chain | E37 noise sweep and p-scaling with σ ∝ p; E41 full runs (p = 17, 31; fractions; no-sleep and lookup-only controls); Transformer baseline on E41's task |
| **M77** | learning cost scales with activity (log of co-active channels), not basis size N | E35 N-sweep at fixed spikes/episode (done to N = 48: flat); spikes/episode sweep at fixed N |
| **M78** | grokking fraction f* ≈ D log p / p^D: composition makes grokking exponentially cheaper than memorization | E41 training-fraction sweep at p = 17 (and p = 31) |
| **M79** | the weaving operator W is a competing-risks temporal point process; a native pool learned by online likelihood with fast/slow, surprise-gated plasticity is an asynchronous world model | E43: online log-likelihood (type and timing) vs Poisson, online Hawkes, GRU neural point process; adaptation after regime breaks; decisions from the model |
| **M80** | structure discovery = full-information online model selection over routes (Hedge-like prices); racing on reliability selects the simplest route that fits (implicit Occam) | E45 5 seeds on a+b, a+b+c, random; menu size sweep |
| **M81** | route-menu mistake bound O(K (log P + log 1/δ)/Δ² + K/ε): logarithmic in the part basis, quadratic in the inverse distractor gap | E46 distractor-density sweep (q) and window-scale sweep |
| **M82** | the market stream's world model is dominated by the time since the last event; a semi-Markov event network (state + window-bank nodes, count learning) beats a neural point process | E44/E48 likelihood with type/timing split; held-out frozen days; GRU given window-bank inputs |
| **M23** | the two-channel (shadow-spike) neuron trains deep race networks at least as well as residue weighting, with binary, sort-free eligibility | depth 1–3, windows, 2 seeds |
| **E15** | credit percolation: reach decays geometrically below F·p ≈ 1; counterfactual credit and σ move the threshold | local layer-wise feedback, depth × fan-in × σ × credit type; per-layer reach and accuracy |
| **M19** | backprop through a beam of histories (sum-product) beats greedy; min-sum on the same beam equals repair | small nets; accuracy, signal coverage, extra events, alignment with M3 |
| **M20** | shadow events in the event engine reproduce the batch beam exactly, asynchronously | equality with M19 per sample; extra events and per-node branch state vs beam width |

M3 is the most informative experiment in this list. It says which term carries the
learning signal, whether our estimator of it is good, and what fired-only is missing,
on a network small enough that every term can be computed exactly. It should run
before E11. E11 then becomes "replace each estimated term with the better local one
that M3 identifies".
