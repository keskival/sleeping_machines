# Counterfactual credit, routing, and depth

[Theory index](../THEORY.md) · Previous: [01 foundations and counterfactual credit](01_foundations_and_counterfactual_credit.md) · Global sections 11–20; section numbers remain stable.

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
