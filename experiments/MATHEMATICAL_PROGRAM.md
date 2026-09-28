# Mathematical program for event networks at language scale

**Purpose.** This note connects the project's existing results to mathematical tools that can explain when an event network should learn or compute better than a Transformer. It is a program of derivations and falsifiable tests, not a claim that frontier superiority has been established. The strongest language result in the current report is still behind Transformer quality, and several proposed mechanisms have not been tested at language scale.

## The target is a quality–resource frontier

“Better” must be made operational. For a fixed data distribution, report the attainable validation/test loss as a function of at least four budgets:

\[
  (\text{training examples},\;\text{training work},\;\text{inference work},\;\text{memory}).
\]

For hardware claims, replace abstract work with joules, wall time, peak memory, and throughput on named systems. A lower operation count does not itself imply lower joules: event generation, indexing, synchronization, and irregular memory access all have costs. The main target is the Pareto frontier: no other tested model has both lower loss and lower resource use. For training compute, the relevant precedent is empirical iso-compute scaling, not parameter count alone ([Kaplan et al.](https://arxiv.org/abs/2001.08361), [Hoffmann et al.](https://arxiv.org/abs/2203.15556)).

## One object, three topologies

Represent a model by an eligibility graph \(G=(V,E)\), node state \(z_v\), event times \(t_e\), payloads \(u_e\), and thresholds \(\theta_v\). For a ramp unit, a useful local form is

\[
  T_v=\inf\{t:\sum_{e\to v}w_{ve}(t-t_e)_+\geq\theta_v\}.
\]

This makes the project’s “topology” three different mathematical objects:

1. **Candidate topology:** the large graph of possible sender–receiver links or temporal motifs.
2. **Realized causal topology:** the event DAG on one example, containing only arrivals and firings that occurred.
3. **Counterfactual topology:** nearby events or routes that almost won, were vetoed, or were cancelled.

These must not be collapsed into one sparsity number. Candidate topology controls capacity and storage; realized topology controls forward work; counterfactual topology controls whether a sparse learner can discover a better route. Growing edges on credit and pruning unused edges can make a huge candidate graph affordable only if the search/gating mechanism is also cheap and the counterfactual learner does not lose useful paths.

Timed automata provide a formal-language lens: states and transitions are decorated by real-valued event times and constraints on clock differences ([Alur & Dill](https://doi.org/10.1016/0304-3975(94)90010-8)). The project's order detectors are a learnable, weighted extension of temporal predicates. The useful question is not merely whether a class is expressible, but the smallest graph, timing precision, and number of updates needed to express and learn it.

### Tropical geometry and stability

With a fixed winner pattern, many firing-time maps are piecewise affine and built from minima, maxima, and weighted means. That puts the forward computation near min-plus/max-plus algebra and tropical geometry. The project derives time-shift equivariance and, for excitatory race maps, non-expansion in the sup norm. These imply concrete robustness certificates: if all relevant winner gaps exceed twice the input jitter, the decision cannot change. They also suggest a key architecture tradeoff: inhibition can improve discrimination while amplifying timing noise.

Test the certificate against empirical flips as width, depth, temperature, and timing jitter vary. A shrinking certified margin with scale would expose a precision tax that operation counts miss. This connects event networks to robust optimization and numerical conditioning, not just neural architecture design.

## Representations: content, order, duration, and state

An event representation is a stream \((c_i,t_i,u_i)\), not a raster. It can carry:

- identity in the active channel or payload \(u_i\);
- rank information in the order of arrivals;
- metric information in relative delays and held intervals;
- history in decaying state or explicit key–value memory.

The rank channel over \(n\) distinct events has at most \(\log_2(n!)\) bits, but this upper bound is useful only if the ordering survives jitter and the learner can use it. Measure effective information by perturbing times, shuffling ranks, and comparing to payload-only and interval-only ablations.

There is no reason to force every payload into one vector format. Treat representation as a design space crossed with topology and time:

- **Dense learned embeddings** preserve distributed similarity and are a natural baseline when the task needs broad semantic transfer; they cost dense arithmetic and memory traffic.
- **Low-rank or factorized payloads** trade representational rank against storage and multiply cost. Sweep rank and measure reconstruction or task loss, rather than assume one-hot or full-width vectors are best.
- **Sparse feature vectors and learned codebooks** can share statistical strength while reducing active work, provided code lookup does not become the hidden all-pairs search.
- **Structured or hyperdimensional codes** make binding and approximate unbinding cheap in some tasks, but collisions and noise accumulate with load; estimate capacity and retrieval error at the model's actual code dimension.
- **Scalar symbols plus continuous event times** preserve identity/order/duration channels separately and can minimize payload work on symbolic tasks, while risking a semantic bottleneck on language.
- **Persistent state-space or key–value memory** gives long-lived context without requiring every past item to be reactivated. Compare update, read, and eviction costs with recomputing over history.

These are complementary choices, not mutually exclusive architectures: an event can carry a discrete code, a low-rank feature, and timing metadata, while slow state stores a compressed summary. A controlled comparison should hold data, parameter/storage budget, and training work fixed; cross payload family with topology (dense, sparse, learned index), temporal channel (rank, delay, duration), and memory (none, recurrent state, retrieval). Report task loss, robustness to time/payload corruption, bytes moved, candidates searched, and end-to-end energy. Universality or a favorable toy-task code is not evidence of a favorable language representation.

The model should be heterogeneous by design. Event-state layers build temporal features and update memory on arrivals; only a retrieval interface needs query/key/value comparisons; readout layers can combine event features and expert predictions. Attention's trainability and exact softmax limit belong at the retrieval interface, not as a requirement for every layer. A scalable model should allocate dense pairwise interactions only where their gradient signal or retrieval value justifies the work, while state updates and feature composition use sparse event paths.

There are exact symmetry opportunities. If a task changes by a global time shift, time-shift equivariance removes the need to relearn that nuisance variable. If it changes by tempo scaling, a scale lattice can make dilation an index shift. Group-equivariant learning predicts improved sample efficiency proportional to how much of the data variation is explained by the symmetry. The held-out-speaker gap in speech is a direct test: separate shift/tempo variation from speaker-specific variation, then compare ordinary and equivariant networks.

Controlled differential equations and rough-path signatures offer a second representation lens. For event streams, iterated integrals encode ordered interactions; sparse order detectors can be viewed as selected, thresholded temporal features. Universality of a feature family does not imply economical approximation. The discriminating measurement is approximation error versus number of active detectors, event work, and jitter tolerance, compared with a dense baseline.

## Attention as a stochastic race—and its hidden search cost

For scores \(s_i\), assign independent exponential clocks with rates \(\lambda_i=e^{s_i/\tau}\). Then

\[
  P(I=i)=\frac{\lambda_i}{\sum_j\lambda_j}=\operatorname{softmax}(s/\tau)_i,
  \qquad \mathbb E[u_I]=\sum_iP(I=i)u_i.
\]

Thus a race can sample exact softmax attention without explicitly normalizing the scores. Averaging \(R\) independent races reduces output variance by \(1/R\) (standard error by \(1/\sqrt R\)); the constants depend on the value covariance. This is a real route to sparse computation, but it does **not** make score search free: if every key must be compared to the query, work remains \(O(Nd)\). The current theory correctly distinguishes scoring candidates from aggregating the few keys near the best score.

The pivotal topology problem is therefore an index: can shared sparse codes or a learned routing graph return a candidate set of size \(C(N)\ll N\) while preserving recall and quality? Measure candidate recall, total candidates scored, value aggregation, and loss against exact search as context grows. Any claim of sublinear retrieval needs all four quantities. “Few keys win” alone is not a sublinear algorithm.

## Learning: pathwise gradients, boundary credit, and online guarantees

Inside a region where the event order and winners stay fixed, the model is piecewise smooth. The gradient follows the realized causal path. At a winner swap, spike creation, or cancellation, the objective has a boundary term: a nearby alternative changes the discrete history. The project's residue/near-miss credit is intended to estimate that term. This connects to event-based adjoints, which compute exact gradients through spike events for specified neuron models ([Wunderlich & Pehle](https://arxiv.org/abs/2009.08378)); it also makes explicit what ordinary pathwise credit omits at a discrete boundary.

Three learning regimes should be separated:

1. **Exact pathwise regime:** compare local event credit with autodiff on fixed event order, then with finite differences across event-order changes.
2. **Noisy race regime:** check that expected local updates equal the desired softmax gradient and report variance versus races per head.
3. **Structural regime:** when the correct route has no active edge, gradient estimates can be zero. Test near-miss exploration, edge recruitment, and pruning as online structure learning.

The project's depth mistake bound, \(O(d\log P/\delta)\), is valuable within its assumptions: realizable ordered patterns, a route already present in the candidate basis, and bounded distractor/trailing-noise rates. It is not a language-model sample-complexity theorem. Language requires learned shared codes, routing, and retrieval; those representation variables are the missing bridge.

Race mixtures and gated linear networks give local convex prediction updates with regret guarantees. This can make the *combination* of existing predictors reliable, but does not create useful features by itself. The architecture should deliberately split jobs: temporal layers discover reusable features; local convex mixers combine predictive experts; counterfactual credit grows or repairs routes.

### Histories, free energy, and counterfactual credit

The event tree can be treated as a statistical-mechanics ensemble. For histories \(h\) with cost \(C(h)\), define

\[
  Z_\sigma(x)=\sum_{h\in\mathcal H(x)}e^{-C(h)/\sigma},\qquad
  F_\sigma(x)=-\sigma\log Z_\sigma(x).
\]

As \(\sigma\to0\), \(F_\sigma\) approaches the minimum-cost history; at finite temperature, near-optimal alternatives retain mass. Forward/backward inside–outside messages on this forest give each branch's posterior contribution. This is a precise bridge between tropical inference, probabilistic inference, and credit assignment. The test is to compare branch posterior credit with sampled near-miss credit as a function of temperature and beam width, then measure whether the added credit improves deep learning enough to pay for its computation.

### Topology learning as online model selection

Growing a synapse only when credited can be interpreted as online structure selection with a code-length cost. If a learned topology uses \(|E|\) edges from a candidate family of size \(Q\), describing the selected structure takes roughly \(\log_2 {Q\choose |E|}\) bits, before encoding weights and delays. A good sparse learner should lower held-out log loss by more than the extra description length. This gives an MDL test for “the model discovered structure” versus “it memorized a task-specific route.”

Depth also has a propagation threshold. If a useful credit signal reaches an average of \(b\) credited ancestors per layer, a crude branching model predicts signal mass proportional to \(b^\ell\) at depth \(\ell\): decay for \(b<1\), persistence near \(b=1\), and amplification for \(b>1\). The theory note relates this to sparse fan-in, event activity, and spectral concentration of backward credit. Measure the full credit distribution by depth, not only average accuracy; compare branching predictions with spectral estimates and with the no-credit / near-miss controls. This can reveal whether deeper models fail because of expressivity, vanishing credit, or unstable amplification.

### Thresholds as dual variables; decisions as stopping times

If a layer is asked to keep a target firing rate or activity budget, its threshold can be treated as a dual price for a capacity constraint. Homeostasis then resembles a primal–dual update; race probabilities provide a smooth allocation over candidates, analogous to entropic optimal transport. The prediction is that learned prices should satisfy both the activity budget and the quality optimum. Equal fixed capacities are not enough: the project has already found that strict balancing can hurt accuracy.

Separately, a race's decision time is a stopping time. Sequential-testing theory suggests choosing the threshold by the value of another observation: stop when expected error reduction is below the cost of latency/energy. This reframes speed–accuracy curves as a decision problem and gives a principled target for adaptive thresholds. Compare fixed thresholds with calibrated sequential likelihood ratios on held-out streams; report accuracy, coverage, latency, and joules together.

## Grokking as a dynamical phase change

Treat training as a stochastic dynamical system over weight distributions and route occupancy. Memorization and a compact compositional route are competing attractors. Sleep, replay, or decay changes their relative stability. This invites statistical mechanics and dynamical-systems analyses:

- define order parameters for memorized-route mass, compositional-route mass, margin, and active synapse count;
- estimate transition boundaries over data fraction, noise, replay rate, and network size;
- test finite-size scaling across seeds and sizes rather than labeling one delayed accuracy jump “grokking”;
- test whether the selected route compresses the training set under MDL or improves held-out likelihood.

The payoff would be a mechanistic account of when temporal topology discovers algorithms rather than memorizing examples. The risk is that a phase transition on modular arithmetic does not transfer to language; the same order parameters must predict held-out transitions on non-synthetic tasks.

## Scaling law to fit

Fit quality and resource curves jointly, rather than infer scaling from a few endpoints. A starting model is

\[
L(D,S,A,T)=L_\infty + aD^{-\alpha}+bS^{-\beta}+cA^{-\gamma}+r(T),
\]

where \(D\) is data, \(S\) stored/learned structure, \(A\) active work per token, and \(T\) context length. This is only a candidate form; interactions and crossover regimes may dominate. At a fixed energy budget \(E\), optimize over topology, data, precision, race repetitions, and model size. A true advantage is a better quality–energy frontier, not a better exponent on one axis with a worse loss floor on another.

For each run log: loss by checkpoint, examples/tokens, active events, candidate edges touched, candidate keys scored, synapses stored, peak process-group RSS, wall time, and estimated/physical joules. Fit uncertainty intervals across seeds. Report both quality at matched compute and compute to matched quality. The standard Transformer scaling results are empirical baselines, not universal architectural laws ([Kaplan et al.](https://arxiv.org/abs/2001.08361); [Hoffmann et al.](https://arxiv.org/abs/2203.15556)).

### Amortized crossover model

Separate the cost of *using* a topology from the cost of *finding and learning* it. For context length $N$, width $d$, and $L$ layers, write the Transformer cost per sequence as

\[
C_{\rm dense}(N)=aLdN^2+bLd^2N,
\]

where the first term is attention and the second is projections/MLPs (constants depend on the block and hardware). For an event model let $E(N)$ be realized events, $Q(N)$ the candidates scored while routing, $U(N)$ the training updates/credit operations, and $M(N)$ bytes moved. A useful first-order model is

\[
C_{\rm event}(N)=c_1E(N)+c_2Q(N)+c_3U(N)+c_4M(N)+c_5S(N),
\]

where $S$ captures synchronization and irregular dispatch. This is intentionally a hardware-calibrated model, not a FLOP identity: event count is not equivalent to dense multiply-add count. At inference, an advantage requires measured $C_{\rm event}<C_{\rm dense}$ at matched loss and context. During training it additionally requires the topology search and credit terms to stay favorable over the full run.

If a learned index costs $C_{\rm build}$ and each query costs $C_{\rm query}$, the index only amortizes after

\[
R > \frac{C_{\rm build}}{C_{\rm exact}-C_{\rm query}},
\]

queries, assuming $C_{\rm query}<C_{\rm exact}$. This break-even count belongs in any retrieval claim; report it for both a fixed trained index and one updated online. For a dynamic graph, include rebuild/maintenance cost and degradation between updates. The same accounting applies to training: a sparse route with cheap execution may still lose overall if exploration and topology maintenance cost more than the dense model it replaces.

This model gives discriminating regimes rather than a single slogan. If $Q(N)=\Theta(N)$, attention-like routing still has a linear scan and may win only at long contexts or with hardware that favors the operation. If $Q(N)=\Theta(N^\gamma)$ for $\gamma<1$, subquadratic routing is possible, but only if candidate recall stays high enough that loss does not rise. If the learned graph has constant out-degree but hidden all-pairs construction, the apparent asymptotic win is an accounting artifact. Measure $Q(N)$, recall, quality, build cost, update cost, and bytes moved over several orders of magnitude in $N$; fit the exponent with uncertainty rather than infer it from one endpoint.

At a fixed training/inference energy budget, the right comparison is the constrained optimum

\[
L^*(E_{\rm train},E_{\rm infer},N)=\min_{\text{architecture, size, data, schedule}} L
\quad\text{s.t.}\quad
E_{\rm train}\leq B_{\rm train},\;E_{\rm infer}\leq B_{\rm infer}.
\]

The architecture has an advantage only if its attainable loss is lower for the same budgets, or its required energy is lower for the same loss, with confidence intervals across seeds. This separates four otherwise-confounded claims: better representation (lower loss at equal optimization), better optimization (faster loss reduction), better asymptotic scaling (better slope/crossover), and better systems efficiency (less measured energy per token). Each can be true or false independently.

## Highest-value falsification sequence

1. **Finish fair small language comparisons.** On identical text splits and tokenization, train converged LSTM, 2/4-layer Transformer, time-vector model, and event/native mixture at several data and compute budgets. Include frozen and online-adaptive results separately. The report says the current 90M-character race mixture reaches 1.50 bpc and remains behind published Transformer results near 1.1; E77 has not established parity.
2. **Test deep Hopfield event message passing.** Compare E77 at depth 2 and 4 against same-depth Transformer baselines, with event-Hopfield on/off, token retrieval on/off, and state-only. Record layerwise query/key/value/gate gradients, full fixed-topology Jacobian gain, payload diameters, route coverage, score entropy, event count, and validation loss. The theory predicts stable state credit only while per-layer correction gain is bounded; route discovery and event-boundary credit remain separate hypotheses.
3. **Separate associative recall from search cost.** Start with all-past-event softmax as the trainability/recall reference. Then train a candidate index against its softmax mass and task gradient, retaining near-miss credit; vary candidate budget and context. Measure candidate recall, omitted softmax mass, output error, gradient bias, index maintenance, bytes moved, and actual score operations. Top-k after dense scoring tests aggregation only.
4. **Test race attention against softmax directly.** Match parameters and training budget, vary races per head and depth, measure output/gradient bias and variance, convergence, loss, and actual event work. This isolates whether stochastic local credit preserves the Hopfield address/value learning signal.
5. **Only then scale tokens and hardware.** Move successful variants to larger text and longer contexts; compare energy and throughput on event-capable and GPU hardware. Include search/index construction, communication, synchronization, and training overhead.

## Decision rule

Advance a mechanism when its predicted intermediate statistic changes in the predicted direction and improves the equal-budget quality–resource frontier across seeds. Retire or revise it when the intermediate statistic fails, even if a single benchmark score improves. This keeps the mathematical story connected to causal evidence and directs effort toward the actual frontier: learned sparse representations, trainable deep routing, and sublinear retrieval without a quality loss.

## Cross-domain anchors

- Timed automata and timed-word languages: [Alur & Dill (1994)](https://doi.org/10.1016/0304-3975(94)90010-8).
- Temporal coding and computational expressivity of spiking neurons: [Maass (1997)](https://doi.org/10.1016/S0893-6080(97)00011-7).
- Exact event-time adjoints for spiking models: [Wunderlich & Pehle (2021)](https://arxiv.org/abs/2009.08378).
- Local convex prediction and gated linear networks: [Veness et al. (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17202).
- Empirical language-model scaling and compute-optimal allocation: [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361), [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556).
- The project's detailed derivations and tests remain in [THEORY.md](THEORY.md), especially §§34, 71, 83–89, 94–109; current measurements and caveats are in [REPORT.md](../REPORT.md).
- Modern Hopfield retrieval and its attention update: [Ramsauer et al. (2021)](https://arxiv.org/abs/2008.02217); E77's additional result is the fixed-memory query Jacobian bound and conditional depth composition in THEORY §107(g).
