# Residual depth and temporal gauges

[Theory index](../THEORY.md) · Global sections 46–49; section numbers remain stable. · Next: [04b forgetting grokking and compositional learning](04b_forgetting_grokking_and_compositional_learning.md)

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
