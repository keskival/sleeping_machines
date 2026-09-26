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
| **M23** | the two-channel (shadow-spike) neuron trains deep race networks at least as well as residue weighting, with binary, sort-free eligibility | depth 1–3, windows, 2 seeds |
| **E15** | credit percolation: reach decays geometrically below F·p ≈ 1; counterfactual credit and σ move the threshold | local layer-wise feedback, depth × fan-in × σ × credit type; per-layer reach and accuracy |
| **M19** | backprop through a beam of histories (sum-product) beats greedy; min-sum on the same beam equals repair | small nets; accuracy, signal coverage, extra events, alignment with M3 |
| **M20** | shadow events in the event engine reproduce the batch beam exactly, asynchronously | equality with M19 per sample; extra events and per-node branch state vs beam width |

M3 is the most informative experiment in this list. It says which term carries the
learning signal, whether our estimator of it is good, and what fired-only is missing,
on a network small enough that every term can be computed exactly. It should run
before E11. E11 then becomes "replace each estimated term with the better local one
that M3 identifies".
