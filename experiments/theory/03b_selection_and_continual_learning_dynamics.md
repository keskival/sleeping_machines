# Selection dynamics, continual learning, and locality

[Theory index](../THEORY.md) · Previous: [03 stability symmetry and continual learning](03_stability_symmetry_and_continual_learning.md) · Global sections 41–45; section numbers remain stable.

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
