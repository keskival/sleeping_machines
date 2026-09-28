# World models and learned topology

[Theory index](../THEORY.md) · Global sections 79–88; section numbers remain stable. · Next: [06b partial evidence state and speech](06b_partial_evidence_state_and_speech.md)

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

**Results (E53, 5 seeds, 40k episodes).** P1 fails as stated (final checkpoint ≥ 0.99 on 3/5: 0.998, 0.963, 0.999,
0.997, 0.964), but not for lack of learning: every seed is at 0.998–0.999 by 5k episodes (≈ 1,100 updates) and the misses
are transient dips after convergence (7 of 40 checkpoints fall to 0.96–0.98 and recover). Dips grow with the exploration
temperature (10 of 40 at T = 1), pointing at exploratory promotions on rare late misses. P2 holds: greedy 0.71–0.85 on
all seeds (4k–11k updates, the prefix cycle); union credit 0.33–0.36 on all seeds (≈ 26k updates, the class nodes never
fire: the deadlock). T = 0.1 searches more slowly (four seeds need 25–35k episodes), as the T trade-off predicts. P3
holds: depth 2 with the same credit 0.32–0.42. P4 holds: ≈ 1,100 updates to converge (depth 2 in E34w: 890–1,531), 41
events per episode against a basis of 57,840 units. P5 fails as stated: an event-token Transformer given the same kind
of data reaches the chains' accuracy: 0.997–0.999 with 2M fresh examples, 0.9975 / 0.996 given 40k examples 50 times
without weight decay (0.981 / 0.994 with weight decay 0.1). The chains (latest-instant credit, earned margin: 0.997–0.999)
match it from ≤ 5k examples seen once, at ≈ 41 events per example against ≈ 215k multiply-adds.
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

**Results (E54, 5 seeds).** P1 holds: depth 4 (L = 2) 1.000 / 1.000 / 1.000 / 0.999 / 1.000, reached by 10–15k episodes
and flat thereafter, 1,872–2,015 updates, 147–168 events per episode, 77k–84k synapses grown of 5.5·10⁷ candidates per
role. P2 holds: one level short (L = 1) 0.57–0.76 with 7.4k–8.7k updates. P3 holds: depth 4 needs ≈ 1.25× the updates of
depth 3 (E54 D3: 1,198–2,380; 0.981–1.000). P4 holds: an event-token Transformer given the same kind of data (a fixed
set of 40k examples, 50 passes) reaches 0.9895 / 0.9905 without weight decay (0.977 / 0.9705 with 0.1), against the
chains' 0.999–1.000 from 10–15k examples seen once: about ten times the error rate, at ≈ 265k multiply-adds per example
against ≈ 39 events (pruned, §93). (At depth 3 the same Transformer matches the chains; the gap opens with depth. The
Transformer with 2M fresh examples at depth 4 reaches 0.992 / 0.996: with ≈ 150× the data it still has 4–8× the
chains' error rate.)
## 86. Stability after convergence: a margin, maintained by near-miss credit

*Written 2026-09-27, after the E53 dip diagnostic, before the margin runs.*

**Observation (E53, E34x).** Converged networks dip: a class that answers all of its probe positives correctly drops to
0% within 200 episodes and is relearned later (E53: 15–17 late breaks per seed in 40k episodes). The diagnostic shows
that most breaks follow a single update, the class's demotion as a false winner (it fired first on another class's
example, or on a 'none' example), not exploration.

**Why (proposition).** Instant credit promotes only deficient roles (§84), so a node stops being promoted as soon as its
potential at the valid instant crosses θ: it converges to the edge of its firing region. A demotion that contains the
target unit (mass w) removes a factor (1 − β) from a set of mass m ≥ w and renormalizes, so the target becomes
w(1 − β)/(1 − βm) ≤ w(1 − β)/(1 − βw) =: f(w). A node at w slightly above θ falls below it after one such demotion.
It survives r successive demotions iff w > θ_r, where θ_0 = θ and θ_r = f⁻¹(θ_{r−1}) = θ_{r−1}/(1 − β(1 − θ_{r−1})).
At θ = 0.6, β = 0.5: θ_1 = 0.75, θ_2 = 0.857, θ_3 = 0.923. (The bound on m is tight when the demoted set is the
target alone; a demoted set that holds nearly all of the role's mass barely moves it, since renormalization restores it.)

**Near-miss credit.** When a class fires correctly, at the firing instant the node knows its own potentials. If a role's
contributing mass is below a margin θ_m, promote those contributors as on a miss. This is local (the node's own state at
its own firing event, plus the teacher's confirmation) and costs updates only until the mass exceeds θ_m. On positives the
target is present, so §83(i) applies to these promotions unchanged. With θ_m = θ_r, no r successive demotions can break a
converged node before the next correct fire repairs it.

**Predictions (5 seeds, 40k episodes).** P1: E53 (depth 3, T = 0.3) with θ_m = 0.9 ≈ θ_2: after 10k episodes every
checkpoint ≥ 0.99 on at least 4/5 seeds (without the margin: 7 of 40 checkpoints dip), and the final ≥ 0.99 on 5/5.
P2: θ_m = 0.75 = θ_1 removes fewer dips than 0.9. P3: E34's task with W = 4.2 and instant credit: θ_m = 0.9 keeps the
per-seed plateaus (0.988–0.995) without dips. P4: depth 4 (E54, θ_m = 0.9): ≥ 0.99 at the final checkpoint on 4/5 seeds.

**Results.** P3 holds exactly: on E34's task with θ_m = 0.9 every checkpoint of every seed is identical (0.988 / 0.989 /
0.995 / 0.994 / 0.991), with fewer updates than without the margin (591–939 vs 1,064–1,479). P2 holds (θ_m = 0.75 still
dips). P1 fails in an instructive way: on E53 the dips vanish, but 3 of 5 seeds freeze at 0.979–0.988, one class held
on an invalid shared prefix. The margin protects whatever route a node holds: an invalid prefix route is demoted on
every example of the class it collides with, and repaired on every one of its own correct fires, at the same rate, so
it persists and exploration can no longer replace it (without the margin all five seeds found the valid order by 5k
episodes). Depth 4 without a margin (E54) is stable anyway: 0.999–1.000 on 5/5.

**§86b: the margin must be earned.** A valid route is demoted rarely (noise); an invalid one systematically. The node
can tell them apart from its own record: the precision of its recent fires, correct / (correct + false), kept as
decaying counts at the node (the reliability price of §80). Near-miss credit only when that precision ≥ a gate: an
invalid prefix route (precision ≈ ½ when it collides with one other class) stays breakable, a valid one (precision ≈ 1)
is protected. *Prediction (E53g, 5 seeds):* with θ_m = 0.9 and gate 0.9, final ≥ 0.99 on ≥ 4/5 seeds and no checkpoint
after 10k episodes below 0.99 on those seeds.
**Result (E53g, 5 seeds): holds.** 0.998 / 0.999 / 0.999 / 0.997 / 0.999, identical at every checkpoint from 5k to 40k
episodes (no dips, no frozen prefixes), with 1,075–1,434 updates and 197–277 near-miss updates. The ungated margin at
depth 4 (E54m) froze the same way as at depth 3 (0.970–0.996 from the first checkpoint, vs 0.999–1.000 without a
margin). The gated rule is stable on every task tried: depth 4 1.000 / 1.000 / 1.000 / 0.999 / 1.000 (E54g); E34's task
0.988–0.995 per seed, identical at every checkpoint; the generic window bank 0.984–0.992 (E34g; union credit: 0.05–0.10,
E34's original rule: 0.86–0.87).
## 87. When is staying out the right policy? An executable-edge bound

*Written 2026-09-27, after the pilot edge audit (E55), before its confirmation.*

**Bound.** Let a policy observe a discrete state s (anything an event world model carries) at each decision point and
choose long, short or flat for a horizon H. Its expected net return per decision is at most
Σ_s P(s) · max(0, E[r_long | s] − f, E[r_short | s] − f), where r is the *executable* round trip (enter at the ask,
leave at the bid, or the reverse, read from the tape) and f the fee. So if no state's conditional mean executable
return exceeds f, flat is optimal for every policy that uses only that information, however clever. Trade prices
bounce between bid and ask, so the mid or last price would show spurious predictability (after a buy at the ask the
next print tends to be lower); executable returns remove it.

**Pilot (E55; fit days 1–5, score days 6–7).** Unconditional executable round trips are −0.2 to −0.3 bp. Selected
states give a genuine held-out edge before fees: +0.3 to +0.9 bp per trade from BTC spot's own state, up to +1.1 to
+1.5 bp over 30–120 s with BTC-perpetual or ETH lead–lag states: other markets do carry information about this one.
At f = 2 bp (a tenth of a realistic taker round trip) essentially no state clears the fee in-sample and held-out net
returns are −0.2 to −0.8 bp (one single-state exception, +1.9 bp at 63 trades/day, is within noise). Staying out is
correct for a taker; the ≈ 1 bp edge would need near-zero fees (market making, a different problem).
**Prediction (confirmation on the 21 untouched days, fit on all 7 pilot days):** with f = 0 the held-out edge stays
positive for the selected states and is larger with perp + ETH than with the own state alone; with f = 2 bp the net
return of the selected states is ≤ 0 (or the selection is empty).
**Result (21 untouched days, fit on the 7 pilot days): both predictions hold.** Before fees every information set keeps a
positive held-out edge at every horizon: BTC spot's own state +0.26 to +1.05 bp per trade (≈ 5–6k decisions a day),
own + perp + ETH +0.47 to +1.40 bp (≈ 1,100 a day). At f = 2 bp few states are selected and all lose (−0.11 to −1.35 bp);
at 5 bp none is selected. Other markets add 30–60% to the edge, which stays near 1 bp: for a taker, staying out is the
correct policy on this data.

**Across markets (E55b).** Each of BTC spot, ETH spot, SOL spot and the BTC perpetual as the traded market, the other three
as leaders; horizon and side chosen on the pilot days only, then read once on the 21 untouched days. Before fees: BTC
+1.0 to +1.1 bp per trade, the perpetual +0.6 bp, ETH ≈ 0, SOL ≈ 0 to +0.5 bp. At 2 bp every selection is negative or
empty; at 5 bp and above nothing qualifies. (Choosing the horizon on the untouched days instead would show up to +1 bp at
2 bp: selection on the test set, not an edge.) Even top-tier taker fees on the perpetual (≈ 3.4 bp round trip) exceed the
edge: across these four markets a taker should stay out.
## 88. Select structure, tune durations: what a candidate basis should and should not carry

*Written 2026-09-27, after single-seed diagnostics (stated), before E56's 5-seed runs.*

**Two ways to learn a duration.** (a) *Selection:* give each channel pair a bank of fixed-window copies [iδ, jδ] and let
§83–§86 credit choose among them. Exactly the machinery that learns routes and order, with O(log #intervals) mistakes,
but every lag fires all copies whose window contains it, about (T_max/δ)²/4 of them, so activity grows quadratically
with timing resolution (E34 with δ = 0.25, T_max = 4: > 60 s for 1,000 episodes, against 2 s without the bank).
Unions of coarser (dyadic) windows would be cheaper but cannot be expressed: under summation a lag fires only its own
piece, whose weight must then carry the whole threshold. (b) *Tuning:* one window per part, adjusted from the lags the
part itself observes when it contributes, at no extra activity. So: **select structure (routes, order), tune durations.**

**A native duration rule for shared parts (version space).** A part keeps the range of lags it had when it contributed to
a correct fire; the estimated support is that range widened by its minimum-variance unbiased extension
(max − min)/(n − 1) at each end; on a false fire a contributing part whose lag lies outside the estimated support moves
that window edge halfway toward it, and never excludes a lag seen in a positive. Shared parts are safe because only
positive evidence widens and only atypical negative evidence narrows. Two traps, both observed: *censoring* (once an edge
moves inward, positives beyond it never fire the part and are never observed, so an early, narrow range freezes a wrong
edge: motif lower edges stuck at 0.51–0.56 against a true 0.3) and *misattribution* (most contributors of a false fire had
legitimate lags); requiring ≥ 30 positive observations before any exclusion removes both in the diagnostic.

**Why the gap constraint appears without being asked for.** The label requires the second motif to start 0.5–2.5 after
the first ends, a difference of spike times, which one node cannot enforce (§71); the cross pair (end of A → start of B)
is a part that encodes it. It fires in every positive, so full-information credit splits the hold mass between the A part
and the cross part; with θ = 0.6 each alone is below threshold and the node needs both: the conjunction is learned, and
the cross part's window converges to the gap interval. Observed (one seed, 12k episodes): motif windows [0.26–0.30,
D + ≤ 0.12] against the true [0.3, D]; cross windows [0.38–0.50, 2.5–3.0]; test 0.993 with 548 updates (the tuned fixed
windows plateau at 0.988 on this seed).

**Predictions (E56, E34's task, 5 seeds, 40k episodes; broad initial windows [0, 3], instant credit T = 0.3, margin 0.9
earned at precision 0.9).** P1: ≥ 0.995 on at least 4/5 seeds (the Transformer: 0.998 after 2M examples). P2: every
seed at or above its fixed-window plateau (E34m: 0.988–0.995). P3: ≈ 20 events per episode, ≤ 1,000 updates.

**Results (E56, 5 seeds).** P1 fails (≥ 0.995 on 1/5: 0.993 / 0.991 / 0.999 / 0.990 / 0.985); P2 fails (3/5 at or above the
fixed-window plateau); P3 holds but for one seed (≈ 20 events; 377–1,358 updates). On average learned windows equal the
hand-tuned ones (0.991 vs 0.991): timing precision no longer has to be given, but it is not what limits accuracy. The
remaining errors (seed 4: 1.2% of the test set, all lost races, no misses) come from *shortcut routes*, §89.
