# Theory note — collapsing futures, and the gradient through a collapse

Written 2026-09-25. A working note: every claim marked **(test Mk)** is checked
numerically (table at the end) before later experiments rely on it.

## Synthesis: the principles so far (read this first)

**Practical-region correction (2 Oct):** [stronger calibration](theory/72_practical_region_calibration.md)
learns a timestamp-aware question table on the proposed joint recency task:
99.51% reserved accuracy,0.05526NLL,0.027s fitting. The20-point advantage
gate is impossible against this control; time-blind tables are insufficient.
Joint-credit repeats also fail their .05-bit superiority gates, although
learned nonlocal prediction reaches99.2–100% and often survives delivered
value erasure. Preserve the first-seed gain as scoped evidence. Priority:
strong real-stream controls and practical headroom before further promotion.

**Completed conditional joint-learning result (2 Oct):** [protected outcome
races](theory/68_joint_addressed_outcome_races.md) learn the balanced distant
relation on reserved suffixes: full joint0.101898bits/96.09%, matched local
0.749780bits/78.13%; identical inference initialization and1,024 fitting
presentations. Full joint estimated fitting work0.237875GFLOPs versus
local0.231934GFLOPs (2.56% extra). Shallow joint0.119259bits/95.31% for
0.153392GFLOPs: useful extra core depth is not established here. The restricted
query-count bound is1bit, not a bound on all counting. Earlier native/tap and
frozen-readout failures remain preserved. [Fixed confirmation](theory/69_joint_credit_confirmation.md)
uses new fit seeds7/8 and new suffix seed75001 without retuning; pending fits
are not additional evidence. This is a protected-state/conditional-credit
diagnostic, with fixed observed addresses, not a natural-language superiority
or exact whole-core gradient claim.

[Uniform address aliasing](theory/70_uniform_address_aliasing.md) proves a
specific cold-start obstruction: uniform outcome-value reads erase the
second bit's distributional information, and a zero residual makes exact
terminal key credit initially zero. Distinct keys and a bilinear decoder
admit a near-perfect solution. A stationary terminal symmetry is therefore
an optimization issue, not an architectural impossibility theorem.

[Bounded terminal pair credit](theory/71_bounded_terminal_pair_credit.md)
specializes §199 to the two-read expected risk: full-support pair sampling
keeps conditional gradients unbiased and can bound decoder pair work, while
all key/proposal/optimizer work and variance remain paid. It does not remove
the cold-start alias; no sampled implementation or advantage is claimed.

**Calibration first (2 Oct, §§376–381):** [sufficient-statistic state, escape races and count references](theory/58_sufficient_statistic_state_and_count_references.md). Closed-form Kneser–Ney counts beat every completed fitted language model on the shared protocol at 2K–1M fitting characters, and counting over the development stream alone scores 2.884 bpc. It proves fixed learned gates cannot be consistent per-address estimators, shows hierarchical backoff is an exact cascade of escape races, derives responsibility-gated credit for a learned base measure, and proposes count-carrying receivers. [Note 59](theory/59_statistic_valued_race_memory.md) (§§382–392) extends it: learned race keys over statistic-valued receivers make counterfactual route and write credit exact and cheap.

Latest empirical design update: [what the completed October 1 results change](theory/RESULTS_DESIGN_UPDATE_20261001.md).
It connects depth, native language quality/work, temporal function contracts
and training-memory exposure to the next matched comparisons.

Next candidate: [protected state, shared processing and identifiable timing](theory/53_protected_state_and_shared_processing.md),
§§354–357. It retains the integrated sparse temporal mechanisms and tests
gap robustness, statistical exposure and timing information separately.

[Information matching and temporal statistics](theory/54_information_matching_and_temporal_statistics.md),
§§358–362, gives the antenna analogy a predictive-metric interpretation and
derives sufficient evidence pooling and aggregate information in race times.

[Predictive spectra and reception sampling](theory/55_predictive_spectra_and_reception_sampling.md),
§§363–368, separates generator realizability, explanatory multiresolution,
activation/weight roles and observability, with generator–recognizer precedents.

[Generator-to-receiver requirements](theory/56_generator_to_receiver_requirements.md),
§§369–372, gives necessary order/uncertainty/retention operations, a local
predictive-rank bound, and tests distinguishing exact redundancy from flatness.
Additional AWS audit: [joint race credit and useful occupied capacity](theory/aws_20261001_race_credit_and_useful_capacity.md). It verifies a strictly convex opposed-gradient witness for the local teacher, preserves its exact linear joint-clock case, and derives a conditional reference and charged residual-replay direction. No frozen model is changed.

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

- [Integrated sparse temporal language](theory/46_integrated_sparse_temporal_language.md) — §§299–302: contextual key races, persistent selected receivers, bounded arrival times, counterfactual credit and capacity/activity accounting. This is the prioritized combined-mechanism candidate.

- [Race attention and resource identity](theory/45_race_attention_and_resource_identity.md) — §§294–298: timing normalization, corrected shared-clock variance, conserved centered teachers, indexed winner retrieval and resource boundaries.

- [Precise language scans and content-bearing messages](theory/44_precise_language_scans_and_content.md) — §§287–293: bounded-delay causal scans, clock precision, gated vector mixing, token-unit memory initialization and input-known write/forget controls.

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
- [Sequential classification and output races](theory/09_sequence_classification_and_output_races.md) — §§115–154
- [Trainability and frontier synthesis](theory/10_trainability_and_frontier_synthesis.md) — §§155–158
- [Optionality and transferable learning reserve](theory/11_optionality_and_learning_reserve.md) — §§159–163
- [Marked events and stable deep learning](theory/12_marked_events_and_learning_stability.md) — §§164–165
- [Serial event transport and sequence credit](theory/13_serial_event_credit.md) — §§166–168: normalized local history, a whole-sequence payload/credit bound, and causal count packets.
- [Winning races and conditioning](theory/14_winning_races_and_conditioning.md) — §§169–170: winner-only delayed continuations, explicit loser-score credit, and the feature-covariance bottleneck.
- [Optionality as reachable correction geometry](theory/15_optional_control_geometry.md) — §§171–172: local correction ellipsoids, directional reserve, Gramian propagation and shared-control cancellation.
- [Event work and temporal credit](theory/16_event_work_and_temporal_credit.md) — §§173–175: linear-work memory and its adjoint; exact local timing eligibility; causal coalescing and its representation error.
- [Shared event models and causal queries](theory/17_shared_event_model.md) — §§176–180: separate task weights, natural-score credit, causal prefix closure, unsupported metadata and periodic representations.
- [Periodic memory and nuisance invariance](theory/18_periodic_memory_and_invariance.md) — §§181–184: affine circle state, occurrence credit, certified modular composition, computational ownership/work accounting and the objective changed by speech augmentation.
- [Event readout, fixed points and class credit](theory/19_event_readout_credit.md) — §§185–187: sparse numerator/mass pooling, exact local key/value credit, covariance support and transfer alignment; a stopping proof for mistake-only teaching; a convex certificate for class-wise local descent.
- [Active causal topology and context bridges](theory/20_causal_topology_and_bridges.md) — §§188–189: a disconnected-component expressivity obstruction; a causal linear-work remedy, exact checkpoint nesting, local teaching support and conditional depth bounds.
- [Credit geometry and sparse differentiation](theory/21_credit_geometry_and_sparse_differentiation.md) — §§190–193: retain complete counterfactual score credit with winner-only value differentiation; separate surrogate and realized descent; certify useful directions across classes and fitting-speaker groups.
- [Separate keys, values and race boundaries](theory/22_key_value_separation_and_race_boundaries.md) — §§194–196: isolate harmful winner changes; preserve actual routing during a smooth value-learning phase; derive joint key/value/clock counterfactual policy credit and its reachability requirements.
- [Joint race likelihood and generic scaling](theory/23_joint_race_likelihood_and_scaling.md) — §§197–201: separate route and clock scores; exact deep credit across discontinuous suffixes; budgeted counterfactual proposals; censored Fisher information and silence teaching; generic quality/physical-work criteria.
- [Full value credit and content retrieval](theory/24_content_retrieval_and_full_value_credit.md) — §§202–205: whole-value teaching, pooled-label dual norms, persistent-prefix scheduling and bounded content-key temporal state with a trainable mean-preserving initialization.
- [Reversible event memory and depth](theory/25_reversible_event_memory_and_depth.md) — §§206–209: addressed packet/state exchange, exact augmented isometry and local angle teachers; affine scans, phase computation, supervised memory observability and angular/Lie reachability with a scalar-reserve criterion.
- [Supervised observability and compact credit](theory/26_supervised_observability_and_compact_credit.md) — §§210–214: class-visible control geometry, compact memory queries, directional optionality, correlation-preserving reserves and a nonlinear expressivity/transport design boundary.
- [Information and asynchronous source payloads](theory/27_information_preserving_event_payloads.md) — §§215–219: input-quotient Bayes risk, exact affine event coalescing, nested fine-channel/time messages and their local label teacher, transferable new control and a constructive learned temporal-mode subset.
- [Rotating temporal memory and initialization](theory/28_rotating_temporal_memory.md) — §§220–223: linear-work signed temporal modes, a nonzero phase teacher at real-memory initialization, conditional depth bounds and an optimizer-preserving payload-width construction.
- [Joint clipping and local teacher statistics](theory/29_local_credit_and_joint_clipping.md) — §§224–225: cross-block sample weighting, a possible mean-teacher reversal and the completed frozen-parent intervention.
- [Event-state coalescing and reference inclusion](theory/30_event_state_coalescing_and_reference_inclusion.md) — §§226–230: precise linear-operator inclusion versus optimizer geometry, exact source/state adjoints, a nested deep residual learner, clock supervision nullspaces and selective affine composition.
- [Affine pooling with observable state](theory/31_affine_pooling_with_observable_state.md) — §§231–233: exact simultaneous final-state and weighted-state-query descriptors, query adjoints, linear-work selective composition and explicit inference/training memory boundaries.
- [Temporal orbits and route optionality](theory/32_temporal_orbits_and_route_optionality.md) — §§234–236: exact interior clock counterfactuals, adjacent-interval credit cancellation, Krylov route geometry, near-tie conditioning and task-visible future reserve.
- [Absorption and credit ownership](theory/33_absorption_and_credit_ownership.md) — §§237–240: fitting-only branch projection, categorical probability/decision certificates, offline conditioning folded into an ordinary head, and the representation/routing teachers changed by functional absorption.
- [Identity depth and live teachers](theory/34_identity_depth_and_live_teachers.md) — §§241–244: exact completed-query and old-teacher preservation at depth growth, local output-map credit and LayerNorm scale, a conditional transport neighborhood, and optimizer/clipping ownership.
- [Nuisance geometry and initialization](theory/35_nuisance_geometry_and_initialization.md) — §§245–248: exact finite-view CE/KL penalty, paired covariance decomposition, robust folded readout fitting, and class-visible nuisance support versus useful future reserve.
- [Optimizer steps and probability geometry](theory/36_optimizer_steps_and_event_probability_geometry.md) — §§249–252: fresh Adam's clipping cancellation, complete-query probability-space curvature, fitting-only finite step calibration for matched depth arms, and the local scope of its guarantees.
- [Class-observer-conditioned depth](theory/37_class_observer_conditioned_depth.md) — §§253–256: finite normalized-residual/class bounds, nonzero gain initialization in head units, compatible normalization steps and matched real depth replays.
- [Finite function-trusted credit](theory/38_finite_function_trusted_credit.md) — §§257–260: bounded actual parameter counterfactuals, fitting-only paired descent/probability acceptance, consistent moment ownership and explicit extra event work.
- [Directional depth and normalization](theory/39_directional_depth_and_normalization.md) — §§261–264: fitted depth deletion, normalization radial credit, pre-normalized live output maps and full-rank observer conditioning that retains hidden null directions.
- [Serial corrections and credit ownership](theory/40_serial_corrections_and_credit_ownership.md) — §§265–268: exact finite correction-risk decomposition, a nonzero serial suffix with its own zero correction head, paired supervision and explicit progressive-training boundaries.
- [Causal streams and tokenization](theory/41_causal_streams_and_tokenization.md) — §§269–274: preprocessing filtrations, persistent state versus truncated credit, complete prefix dictionaries, exact per-character marginal likelihood and local vocabulary teachers, with a quality/work experiment.
- [Information and statistical credit](theory/42_information_and_statistical_credit.md) — §§275–279: a stopped-token Fisher identity, control frequency versus statistical capacity, the noise cost of deep corrections, fitting-only cross-sample teachers and a class-calibration quotient.
- [Compute allocation and frontier scaling](theory/43_compute_allocation_and_frontier_scaling.md) — §§280–286: feasible resource sets, marginal learning value, dormant-capacity exposure, retrieval-error bounds, modern token/position controls, online adaptation and conditional scaling predictions.
- [Precise language scans and content](theory/44_precise_language_scans_and_content.md) — §§287–293: precise clocks, bounded causal scans and input-dependent content/memory learning.
- [Race attention and resource identity](theory/45_race_attention_and_resource_identity.md) — §§294–298: corrected shared-clock covariance, centered conserved counterfactual credit and complete resource boundaries.
- [Integrated sparse temporal language](theory/46_integrated_sparse_temporal_language.md) — §§299–302: content-bearing sparse timed events, key/value/state separation and capacity versus teaching work.
- [Language capacity and online learning](theory/47_language_capacity_and_online_learning.md) — §§303–312: centered-logit rank, integrated width/data tests, iso-FLOP risk, online adaptation, reproducible benchmarks, queries, compressed versus token KV memory, addressable history and the eight-depth content-index experiment.
- [Parallel heads and work scaling](theory/48_parallel_heads_and_work_scaling.md) — §§313–325: independent projections, timestamped mixing, full-bank attention work, counterfactual/Adam costs, approximate attention containment, shared-match temporal policies the dormant-capacity hypothesis winner-local Poisson renewal event-triggered on-substrate learning order/elapsed-time inductive biases delay-based feature thresholds the integrated shared-match repeated-arrival trial and longer-credit/failed-head diagnosis.
- [Cross-cutting test matrix](theory/TEST_MATRIX.md)

### Canonical derivation map

- **Current synthesis:** §§155–158 connect the manifesto to the remaining gaps. §155 distinguishes smooth threshold crossings from arrival-triggered jumps and documents a reproducible payload/gradient defect in legacy TVLayer. §§156–157 distinguish event support, class information, and label-aligned descent, and specify a bounded sparse continuation architecture.
- **Optionality refinement:** §§159–163 distinguish attainable correction sets from route entropy, derive the value of retaining a choice until information arrives, show why an isolated option premium cannot generally be backed up as one scalar, and derive the gradient-variance term that contaminates same-sample virtual learning progress. Independent adaptation/evaluation measures transferable reserve; the finite-budget gain curve measures useful breadth under an explicit proposal. §163 derives a control-normalized margin and distinguishes realizable parameter changes from incompatible forced-event bundles.

- **Route changes and credit to unrealized alternatives:** §§19 and 57 are the canonical counterfactual routing derivations. Sparse experts and attention-key recruitment use this same mechanism.
- **Attention and associative retrieval:** §§96–107 develop race/softmax equivalence, key/value gradients, and event-sequence Jacobians. §112 specializes the existing route credit to missing attention keys; it does not re-derive the generic router gradient.
- **Depth and local learning:** §§97–98 give the ordered-pattern depth bound; §§107(f–h) treat fixed-topology payload propagation; §§110–111 analyze auxiliary gradient paths and interference.
- **Sparse approximation through depth:** §113 composes local approximation errors with residual-stack Jacobian propagation. §114 lifts fixed-support softmax truncation bounds to the full sequence Jacobian using row and key-fan-out column sums; E114's finite-difference diagnostic supports the bound on small synthetic cases. Uniform-region constants and support changes remain open.
- **Sequence labels and online output:** §§115–118 distinguish an utterance-level label from prefix labels, derive a competing-risk objective, define fallback/payload semantics, and catalog failure modes. §§119–121 generalize the task to event streams and separate connectivity/activity sparsity from sparse execution. §§122–124 derive a selective-risk guarantee under true prefix posteriors, explain the value of waiting, and specify requirements for a reusable sparse stream classifier. §125 derives exact sparse max-confidence maintenance with indexed max and log-sum-exp trees; §126 derives class evidence for marked point processes and shows when silence requires scheduled posterior updates. §127 audits E83's hard route and firing masks against the existing counterfactual boundary-gradient theory. §128 separates proper prefix-posterior estimation from decision-focused stopping. §129 proves a winner-gap dead zone for max-pooled route credit and records a small untrained final-readout shadow probe. §130 derives causal windowed prefix scoring, its event-time adjoint/eligibility signal, and the bias from choosing query time using future utterance duration. §131 applies the existing §19/§57 route-boundary estimator to E83's lost content routes, records the depth-4 instability of its inverse-probability total estimator, and motivates a bounded normalized local update. §132 derives its finite-population variance and scale tradeoff, interprets gradient alignment, and specifies per-layer diagnostics. §133 distinguishes E77's already-correct causal next-token loss from its hard hidden-route boundary gap, derives a layerwise token-loss shadow gradient, and separates Bayesian route-utility uncertainty from optimizer-metric step control. §134 derives local Bayesian gain from prior precision, observation noise, and conditional-state novelty. §135 formulates the TV-quiz task as causal prefix-posterior estimation followed by a stopping policy, derives the event-plus-survival likelihood and its silence credit, and identifies that E83's sparse output is piecewise constant during silent gaps. E83 pilots remain near chance; E77's depth-8 smoke establishes gradient reach, not language-model quality or scaling.
- **Deep SHD routing and depth:** §136 derives sparse evidence fusion across depth; §137 proves the strict-chain support invariant; §138 derives the support-masked gradient statistics and gate-margin stability condition. The data-diversity screen at fixed 120 updates reverses direction across seeds, with layer-4 coverage 1.56–7.03% and late NLL above uniform. §139 audits candidate-graph reachability: in two exact E83 masks, 90.2% / 98.0% of first-to-fourth hidden-node pairs are structurally connected, and all 140 input bands can reach layer 4, yet realized layer-4 support is only 1.56–7.03%. It derives why MoE-style paired route outcomes are required and how single-edge shadows miss cooperative threshold crossings. §140 derives the combinatorial growth in candidate paths with depth alongside multiplicative per-layer survival; keeping half of examples active at depth 8 or 16 requires average conditional survival at least 0.906 or 0.955. The layer-1 skip raised layer-4 support to 99–100% without paired accuracy gain. More depth therefore creates potential expert compositions, not automatically trainable expert choices; route credit and survival must be established first.

- **Readout shortcuts versus deep credit:** §145 distinguishes direct class evidence from a shallow readout head from counterfactual utility that propagates through the stack. In the seed-6 E83 boundary audit, L1/L2 toggles changed all-depth loss but every matched deepest-only L1/L2 delta was exactly zero; L1 work changed its own readout edges without causing downstream hidden spikes. This is checkpoint-specific evidence that the apparent L1 utility was a shallow bypass, not serial depth credit. A deep trainability claim requires a deepest-only primary loss or an explicit sparse suffix replay with measured downstream event changes.

- **Event propagation versus pair-specific credit:** §146 derives the four-corner expected loss and shows that its mixed margin derivative is proportional to $\Gamma=L_{11}-L_{10}-L_{01}+L_{00}$. In 1,024-example L2/L3 spike-pair replays, double-open events sometimes changed deepest-only loss, but none of 113 sampled natural-off pairs needed both singleton events to help; $|\Gamma|>0.01$ occurred only in 2/42 L3 pairs and 0/210 L2 pairs. L2 openings often created downstream spikes without a useful class-loss change. This is a spike-event audit, not a same-receiver route-pair test; the next mechanism test must condition on shared topology and arrival-time overlap.

- **Route option value through counterfactual descendants:** §149 separates immediate route utility from finite suffix-learning progress, then defines a scalar soft value backed up through each route's recomputed descendants. Error and predictive entropy control proposal temperature and swap probability; raw activity is not rewarded. §150 tests receiver replacement, route birth, spike birth, and a combined proposal pool on eight error-conditioned examples. The single-action families yielded zero value, but the combined pool produced five verified leaves with added L4 events; three of 31 leaves improved both deepest-head loss and matched suffix-step progress. The scalar root value was positive on one of eight examples for learning weights 0, 1, and 10, and on two only at weight 100. This is a narrow frozen-checkpoint signal, not trained gain; it motivates a bounded scalar update with calibrated learning-progress weight. The audit stores branch traces for diagnosis, while its recursion only passes a scalar. §150 also reports top-2 route/receiver-load results: 13/128 vs 8/128 remains inconclusive (McNemar $p=0.359$), with receiver concentration and low L2/L3/L4 support.
- **Finite proposal support and deep-route extinction:** §153 gives the exact probability $1-(1-p_\epsilon)^K$ of seeing at least one useful continuation under a proposal, and proves that no rollout count or inverse-propensity weight can recover a route with zero proposal support. It factors strict-chain depth support into conditional transition rates and identifies the zero-path-gradient condition induced by detached hard gates. The six-epoch SHD extension stays at 6/128 in both matched arms; continuation-aware credit briefly expands deep event support but ends with zero L3/L4 support and no accuracy gain. The next measurement separates candidate absence, proposal-band censoring, event survival, and downstream task utility.

- **State-conditioned optionality and uncertainty:** §152 defines optionality as proposal- and horizon-conditioned reserve in expected best-of-$K$ continuation loss, with beneficial-route mass as a separate breadth diagnostic. The per-example score estimates a function of the current event state and label; it is not optimizer momentum. Predictive entropy, observed-label NLL/surprise, route entropy, epistemic uncertainty, and future option reserve have different meanings. The matched $\lambda=1$ two-rollout arm and immediate-only control both stayed at 6/128 accuracy; reserve gain was tiny and no future route exceeded the 0.05 loss-improvement cutoff. Event support remained under 1% at L4 and correction still mostly reached the readout head. §152 also notes that summed potential differences telescope, so optionality needs to enter a continuation/value backup or action-selection target; adding it as an unexamined per-step reward does not by itself create a new learning objective.
- **Training scalar option value at a firing boundary:** §151 derives the detached-utility threshold update $\Delta\theta=-\eta q(1-q)U/\tau$ and identifies the sampler's uncorrected inclusion probabilities as an implicit proposal-weighted objective. A three-arm depth-4 pilot (same seed, 120 train / 128 held-out, two epochs) compared pathwise, immediate-only, and $\lambda=10$ updates. All ended at 6/128 accuracy. The scalar arm raised held-out L2 support from 8.6% to 27.3%, but L4 support stayed 0.78% and its epoch-2 suffix-learning advantage was slightly negative. Epoch-2 L3/L4 pathwise gradient norms were at most $6\times10^{-5}/0$ across arms. Local option credit moved intermediate activity without producing durable deep task utility; candidate support, downstream survival, and gradient reach remain separate bottlenecks.

- **Dormant event counterfactuals:** §141 generalizes the lost-route shadow in §131 by recording *why* an event did not happen (closed content gate, missed voltage threshold, race loss, or refractory block). Each candidate gets a signed local margin; a paired replay measures its full downstream loss difference, and that utility updates the margin's local eligibility. A richer pool can improve discovery only if useful alternatives are actually sampled and their comparisons have adequate signal-to-noise. Keep proposal support sparse, stratify by failure cause and margin, preserve an exploration floor, log sampling propensities, and bound shadow work. The common-utility update is derived; its effect on E83 recognition and stability is unmeasured.
- **Counterfactual-rich sparse routing and depth activity:** §142 factors route alternatives by topology, gate, payload, time, integration, firing, race, state, continuation, and readout; it derives the exact four-outcome gradient, state-dependent message policy, and threshold/vector-projection proposal score. The global E83 sampler put all 120 pair shadows in L1; only 7/120 openings helped, and its 14/128 vs 8/128 validation gain was inconclusive (McNemar p=0.180). Layer-balanced sampling selected 43/34/15/28 pairs across L1–L4, maintained 100% L4 support, but ended at 6/128 vs 8/128 control (p=0.791), 1,647 L4 spikes/utterance, and only two predicted classes. §143 separates support from event multiplicity, derives a cost-constrained paired-route utility, and reports a frozen validation audit: late-layer pairs were more often helpful, while early pairs caused larger downstream event growth. The matched late-only sampler then produced 9 L3/L4 pair shadows in epoch 1 and none in epochs 2–4; accuracy finished at 6/128 versus 8/128 control. §144 derives the quadratic low-occupancy upper envelope for pair availability and why inverse-probability weighting cannot repair an empty candidate set. A frozen window sweep from 25 to 1,000 ms still found zero L3 pairs and one L4 pair across 30 minibatches, so the time cutoff is not the main bottleneck; deep source-route co-occupancy is. A refractory-aware matched spike audit leaves 21 valid L1 and 19 valid L2 candidates in the late-only checkpoint, but only 2/1 at L3/L4. L1 has the clearest local-utility direction; L2's mean remains positive after valid-candidate filtering despite a majority of individual improvements. The proposed sparse prefix-expansion estimator and any trained boundary update remain unimplemented. Current E83 hidden-state training still scans a 1 ms grid; asynchronous training efficiency, attention-equivalent expressivity, and repeatable deep SHD accuracy remain unestablished.

Read §0 for the prior-work boundary and the synthesis above for the project's working principles. The test matrix is an inventory; it is not evidence that each listed prediction has been confirmed.

- [Historical write eligibility](theory/49_historical_write_eligibility.md) — §§326–329: compact sealed-write producer credit, factorized key-map adjoints, exact fixed-feature perturbation scope, 50% K/V tensor-storage overhead and gated integrated comparisons, and the independent common clock mode. Forward history is preserved; full-history deep gradients and hardware efficiency remain unproved.

- [Native addressed event learning](theory/50_native_addressed_event_learning.md) — §§330–336: user-directed receiver/state construction without per-position KV attention, local causal schedules, occupied-capacity versus activity, full producer-credit boundaries, matched order/time/credit controls, map sharing versus state sharing, long-silence information loss, and complete optimizer/exposure scaling, and an exact ordered-state kernel/temporal eligibility identity.

- [Useful delay computation, temporal windows and spike trains](theory/51_temporal_windows_and_spike_trains.md) — §§337–341: shared-match fast clock features, exact smooth-window membership gradients, hybrid repeated-spike sensitivities, sparse event recurrences and explicit birth/routing credit limits.
- [Useful capacity and local mixing](theory/52_useful_capacity_and_local_mixing.md) — §§345–348: task-aligned reachable capacity, affine fusion versus nonlinear interaction rank, content selection in learned bases, and marginal quality/work allocation between mixing stages.
- Later additions: §349 derives rotating message-space projection/composition; §§350–351 analyze persistent reception banks, cached local projection cost, and frozen versus refitted utility ablations.

- [Full-state and joint-clock credit](theory/57_full_state_and_joint_clock_credit.md) — §§373–375: factorial delivery/write diagnosis; joint likelihood credit through downstream timing jumps; exact local-score residual correction and its conditional-independence/resource requirements. No frozen teacher is changed.

- [Reusable rules and private predictive state](theory/61_reusable_rules_and_private_predictive_state.md) — RS1–RS3: exact paired-timing information ceiling, two-state generator receptor, qualified shared-rule estimation/exposure law, private-state resource boundaries, and invariant-memory versus useful-time decomposition. Motivated by completed native mechanism gains; no frozen model is changed.

- [Sufficient-statistic state and count references](theory/58_sufficient_statistic_state_and_count_references.md) — §§376–381: count references on the shared protocol and at 10M/90M (dense controls sit near count level; causal E173 supplies the valid mixture comparison), count-gated writes as exact Dirichlet posteriors, inconsistency of fixed gates, hierarchical backoff as escape-race cascades, responsibility-gated learned base measure, and the count-carrying receiver test.
- [Statistic-valued race memory](theory/59_statistic_valued_race_memory.md) — §§382–392: exact Bayesian pooling criterion (generalization priced by evidence), learned keys with sufficient-statistic values giving zero-variance counterfactual route credit and closed-form write credit, in-context statistics without attention, data-dependent learning-work scaling, the memorization tax of dense models (with falsifiable predictions), the residual distribution a learned base converges to, and quality-calibrated equivalent count order, the proposed count-conditioned repair, and §389.1 review corrections separating context dependence, nonlinear conditional residual optimization, gradient reach and useful depth (preserving measured width failure and reported 8–11 bpc standalone scores); §§390–398: obstacles to deep temporal-core features (sample complexity, credit reach, single-address native language state, missing race retrieval), learned dilated delay taps, and learning-allocation starvation of residual bases with statistic-valued race memory over learned keys as the repair; and where counting runs out (evidence sparsity, range, storage, novelty), with evaluation stratified by count evidence (§393); the calibration rule (compare only where learning matters) and the joint text/event task with its leak and phase-rule revision (§394); where incumbents pay structural costs (§395); the real-data DVS capacity ladder (§396); Transformer import as race streams, deprioritized (§397); and the capacity–exposure law: private state scales freely, untied parameters dilute learning by P|θ|/N (§398).

- [Deep feature bottlenecks](theory/62_deep_feature_bottlenecks.md) — trained-state attribution, actual responsibility/dynamic-gate credit, Adam scaling caveat, conditional-information criterion, retired confounded long-range protocol, and retained-mechanism/resource contracts for addressed-memory and delay-tap diagnostics.

- [Retrieval credit and value versions](theory/63_retrieval_credit_and_value_versions.md) — moving a linear value map from outcome writes to addressed reads preserves fixed-weight computation and restores exact credit to that map from detached historical features; producer credit remains truncated. Matched integrated full/minimal comparison and separately charged frozen fit replay.

- [Joint feature credit](theory/64_joint_feature_credit.md) — exact balanced parity counterexample: individual features have zero target information/mean interaction credit, jointly delivered features have predictive credit, and independent unstructured hard selection attenuates it by the product of delivery probabilities. A scoped motivation for joint credit/candidate-discovery tests, not a language no-go.

- [Evidence access, precision and joint credit](theory/65_evidence_access_and_joint_credit.md) — completed projection repair fails its quality gate despite correct derivatives; restricted mean/occupancy reads can hide evidence precision, and joint route utility also depends on decoder learning. Bounded integrated next comparisons and a scoped producer-Jacobian resource calculation; no new fit or supremacy claim.

- [Balanced joint learning](theory/66_balanced_joint_learning.md) — a paired-prefix fitting protocol balances root symbol totals as well as query suffix/count vectors. Tests useful learned interaction beyond those restricted counts in native/tapped/full/shallow integrated cores with full-prefix credit and charged candidate/optimizer work; no broad counting or supremacy theorem.

- [Retention and interaction readout](theory/67_retention_and_interaction_readout.md) — a generic degree-2 query feature map exposes joint information that an affine readout can miss. Prespecified frozen temporal-encoder/initial-reservoir controls, separate supervised bit-retention probes and a fresh suffix evaluation; full prefix replay and original encoder fitting remain charged.

- [Joint addressed outcome races](theory/68_joint_addressed_outcome_races.md) — after failed retention/interaction diagnostics, protect causal predecessor–successor evidence and retrieve through two learned key/value races. Exact terminal expected-risk pair credit versus the local surrogate, with full candidate/pair work, shallow control and reserved suffix evaluation; earlier timing/write credit remains scoped.

- [Real gesture integrated admission](theory/73_real_gesture_integrated_admission.md) — strong time-aware class-count/linear/kernel calibration identifies measured headroom on causal subject-disjoint DVS packets. Unchanged native sparse temporal fit requires numerical contracts, a measured learning/accounting smoke and complete practical resource boundaries; no official test or pending advantage claim.

- [Predictive features and practical frontiers](theory/74_predictive_features_and_practical_frontiers.md) — conditional-information bottleneck versus failed finite readout, interaction stationary points, mixture responsibility and Adam's limits. Repeated relation-credit superiority fails while useful native context learning survives. Practical advancement requires completed quality/resource comparisons against preserved strong controls rather than a generic gradient diagnosis or promised win.

- [Frozen real-feature diagnosis](theory/75_frozen_real_feature_diagnosis.md) — fitting-selected readouts distinguish useful learned native features from an initial reservoir, with frozen inference probability reproduction and charged encoder/replay/decoder work. Conditional decoder CV does not establish whole-pipeline validation or practical advantage.

- [Packet-scale temporal initialization](theory/76_packet_scale_temporal_initialization.md) — match trainable decay/rotation initialization to observed packet units while retaining the unchanged integrated core. Completed matched fit raises accuracy but worsens NLL and still trails both strong controls; the change is insufficient on this protocol.

- [Batched fit and conditional terminal risk](theory/77_batched_fit_and_terminal_risk.md) — independent-clip vectorization retains reference state and every-parameter gradients. Optional final-query pair risk gives exact conditional terminal content derivatives; earlier routes retain local surrogates. Actual recovery and full/partial-window learning/accounting smokes pass before a bounded matched pilot.

- [Terminal curvature and credit](theory/78_terminal_curvature_and_credit.md) — an explicit convex cross-entropy counterexample reverses the reference local route teacher; actual-reference contracts, conditional scope and curvature bounds.
- [Learned windows and silence bursts](theory/79_learned_windows_and_silence_bursts.md) — causal deadline/EOF semantics, fixed-partition sensitivities, finite timeout-bank risk and starvation/work boundaries. Primitive contracts do not establish an integrated fit.
- [Joint clock, memory and support](theory/80_joint_clock_memory_credit_and_support.md) — completed terminal pilot fails its NLL gate; frozen legal-route/hybrid replay distinguishes payload and persistent-write utility; joint clock likelihood, counterfactual support and uncertainty scopes.
- [Bounded state and clock credit](theory/81_bounded_state_clock_credit.md) — one actual-state alternative replay, conditional-winner joint likelihood replacement, prefix-only baseline and honest full-forward shadow cost at unchanged sparse inference. Contracts and a smoke precede the fixed gated quality pilot; no full-model exact-gradient claim.
- [Joint-credit variance and failure](theory/82_joint_credit_variance_and_failure.md) — completed joint-clock pilot fails quality; exact choice/common-clock decomposition, baseline-noise identity, fixed-prefix oracle and shared-noise training scope.
- [Variance audit and next credit contract](theory/83_variance_audit_and_next_credit_contract.md) — small frozen actual-suffix quadrature finds large removable score-space variance; next isolate full-write choice utility while retaining native timing credit, with contracts before any new fit.
- [Actual-write choice with timing](theory/84_actual_write_choice_with_timing.md) — corrected categorical full-suffix write utility, separate retained native delay derivative, deep producer VJP/recovery/accounting contracts and fixed quality/work gates at unchanged sparse temporal inference.
