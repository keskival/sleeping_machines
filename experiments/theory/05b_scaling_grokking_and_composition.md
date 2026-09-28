# Scaling, grokking, and composition

[Theory index](../THEORY.md) · Previous: [05 temporal computation and scaling](05_temporal_computation_and_scaling.md) · Global sections 66–78; section numbers remain stable.

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
