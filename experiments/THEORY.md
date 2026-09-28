# Theory note — collapsing futures, and the gradient through a collapse

Written 2026-09-25. A working note: every claim marked **(test Mk)** is checked
numerically (table at the end) before later experiments rely on it.

## Synthesis: the principles so far (read this first)

The foundational calculus in §§1–20 reduces to five principles. Later sections extend and
test them; the evidence column below summarizes that initial layer of the theory.

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

## Theme map

Section numbers remain global and unchanged, so references such as “THEORY §57” continue to work. The detailed derivations are split into shorter thematic notes:

- [Race foundations and exact gradients](theory/01_foundations_and_counterfactual_credit.md) — §§1–10
- [Counterfactual credit, routing, and depth](theory/01b_counterfactual_credit_and_depth.md) — §§11–20
- [History ensembles and learning dynamics](theory/02_history_ensembles_and_learning_dynamics.md) — §§21–25
- [Pivotal credit, symmetries, and learning dynamics](theory/02b_pivotal_credit_and_learning_dynamics.md) — §§26–33
- [Stability, weaving, topology, and generalization](theory/03_stability_symmetry_and_continual_learning.md) — §§34–40
- [Selection and continual-learning dynamics](theory/03b_selection_and_continual_learning_dynamics.md) — §§41–45
- [Residual depth and temporal gauges](theory/04_residual_depth_and_grokking.md) — §§46–49
- [Forgetting, grokking, and compositional learning](theory/04b_forgetting_grokking_and_compositional_learning.md) — §§50–55
- [Temporal computation and counterfactual routing](theory/05_temporal_computation_and_scaling.md) — §§56–65
- [Scaling, grokking, and composition](theory/05b_scaling_grokking_and_composition.md) — §§66–78
- [World models and learned topology](theory/06_world_models_and_sparse_topology.md) — §§79–88
- [Partial evidence, state, speech, and sparse inference](theory/06b_partial_evidence_state_and_speech.md) — §§89–93
- [Transformer attention and its training](theory/07_transformer_attention_and_training.md) — §§94–103
- [Vector dynamics and scaling](theory/08_vector_memory_and_deep_stacks.md) — §§104–106
- [Vector memory, retrieval, and deep local learning](theory/08b_memory_retrieval_and_deep_learning.md) — §§107–111
- [Sparse attention and depth bounds](theory/08c_sparse_attention_and_depth_bounds.md) — §§112–114
- [Sequential classification and output races](theory/09_sequence_classification_and_output_races.md) — §§115–133
- [Cross-cutting test matrix](theory/TEST_MATRIX.md)

### Canonical derivation map

- **Route changes and credit to unrealized alternatives:** §§19 and 57 are the canonical counterfactual routing derivations. Sparse experts and attention-key recruitment use this same mechanism.
- **Attention and associative retrieval:** §§96–107 develop race/softmax equivalence, key/value gradients, and event-sequence Jacobians. §112 specializes the existing route credit to missing attention keys; it does not re-derive the generic router gradient.
- **Depth and local learning:** §§97–98 give the ordered-pattern depth bound; §§107(f–h) treat fixed-topology payload propagation; §§110–111 analyze auxiliary gradient paths and interference.
- **Sparse approximation through depth:** §113 composes local approximation errors with residual-stack Jacobian propagation. §114 lifts fixed-support softmax truncation bounds to the full sequence Jacobian using row and key-fan-out column sums; E114's finite-difference diagnostic supports the bound on small synthetic cases. Uniform-region constants and support changes remain open.
- **Sequence labels and online output:** §§115–118 distinguish an utterance-level label from prefix labels, derive a competing-risk objective, define fallback/payload semantics, and catalog failure modes. §§119–121 generalize the task to event streams and separate connectivity/activity sparsity from sparse execution. §§122–124 derive a selective-risk guarantee under true prefix posteriors, explain the value of waiting, and specify requirements for a reusable sparse stream classifier. §125 derives exact sparse max-confidence maintenance with indexed max and log-sum-exp trees; §126 derives class evidence for marked point processes and shows when silence requires scheduled posterior updates. §127 audits E83's hard route and firing masks against the existing counterfactual boundary-gradient theory. §128 separates proper prefix-posterior estimation from decision-focused stopping. §129 proves a winner-gap dead zone for max-pooled route credit and records a small untrained final-readout shadow probe. §130 derives causal windowed prefix scoring, its event-time adjoint/eligibility signal, and the bias from choosing query time using future utterance duration. §131 applies the existing §19/§57 route-boundary estimator to E83's lost content routes, records the depth-4 instability of its inverse-probability total estimator, and motivates a bounded normalized local update. §132 derives its finite-population variance and scale tradeoff, interprets gradient alignment, and specifies per-layer diagnostics. §133 distinguishes E77's already-correct causal next-token loss from its hard hidden-route boundary gap, derives a layerwise token-loss shadow gradient, and separates Bayesian route-utility uncertainty from optimizer-metric step control. E83 pilots remain near chance and do not establish trainability.

Read §0 for the prior-work boundary and the synthesis above for the project's working principles. The test matrix is an inventory; it is not evidence that each listed prediction has been confirmed.
