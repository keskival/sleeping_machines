# Stability, weaving, topology, and generalization

[Theory index](../THEORY.md) · Global sections 34–40; section numbers remain stable. · Next: [03b selection and continual learning dynamics](03b_selection_and_continual_learning_dynamics.md)

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
