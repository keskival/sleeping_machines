# Pivotal credit, symmetries, and learning dynamics

[Theory index](../THEORY.md) · Previous: [02 history ensembles and learning dynamics](02_history_ensembles_and_learning_dynamics.md) · Global sections 26–33; section numbers remain stable.

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
