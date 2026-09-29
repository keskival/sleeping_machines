# Mathematical program for event networks at language scale

## Current advances: preserved computation and local readout credit (E121–E125)

The modular phase state now certifies its complete mod-17 rule. A simple global
error bound fails, but target-constrained cyclic min/max composition proves
every clock cell with positive slack (§183). The common class can execute this
69-scalar primitive without an unused generic carrier (§184). A zero-update
fitting pass certifies a fixed point of its local teacher (§186).

For speech, input nuisance variation improves the matched pooled development
score from 68.4% to 72.1%, mostly on one speaker. The larger 4,096-example
continuation reaches 72.3%. The next analytic intervention targets the count
mean: a causal scalar key weights winning event payloads in an associative
numerator/mass state. At mean initialization its gradient is covariance times
the teacher direction, which is nonzero on the measured batch (§185). A matched
continuation finishes at 69.5% versus 68.9% with the old mean, both below the
72.3% starting checkpoint. Supported local credit alone does not establish
transfer: §185 derives the fitting/new-speaker gradient-alignment condition.

The consolidated work audit counts actual configured maps, router choices,
memory scans, normalization and pointer/clock candidates. It must distinguish
the sparse primitive from a carrier-plus-primitive configuration. New small
Transformer/LSTM controls use the same arithmetic/recall examples. Inference
estimates cannot establish total training-energy superiority.

## Current synthesis: causal queries, evidence geometry and invariances (E120)

A common architecture is trained separately for each task. §§176–180 provide
its implemented formalism. The exact count-feature intervention establishes one
root cause of failed integration: an unidentifiable static readout direction
can overwhelm a correctly generalizing pointer. Future synthesis must preserve
specialist invariances, not merely place their outputs beside a deep branch.

The next matched coupling experiment is motivated by frozen deletions: removing
the additive head improves development likelihood on both text and market,
while retaining the deep-feature evidence gate. The next representation task is
periodic state: pairwise features have zero population correlation with a uniform
three-operand modular label, while group-character products compose correctly.
Measure the actual higher-order feature/credit statistics and port the trainable
rhythm primitive before claiming the shared core inherits native grokking.

The implemented prefix interface is causal but replays context. Persistent event
state and query closure are required for incremental prediction costs. All claims
must include pointer candidate search, evidence preparation, local dense maps,
training alternatives, sorting, optimizer work and memory.


**Purpose.** Connect the manifesto, existing results, and derivations to a deep event architecture with a better quality–resource frontier. E79 leads the completed same-split 1M/10M language baselines; its expert mixture is a different model family from the deep vector stack. Learned retrieval and synthetic compositional depth also have measured advantages. The missing bridge is a shared representation learner that combines those capabilities with sparse execution and efficient credit.

## Current priorities, 29 September 2026

The latest analytic synthesis is in [§§155–158](theory/10_trainability_and_frontier_synthesis.md);
[§§159–163](theory/11_optionality_and_learning_reserve.md) refine optionality. These supersede
older next-run suggestions below. The AWS sibling owns non-SHD benchmarks; this host owns SHD.

**Completed bridge, E117–E119:** frozen-feature experiments isolate a readout-conditioning
bottleneck; bounded winner-only carriers establish a positive D8/D1 development comparison.
The larger D8 run reaches 151/256 (59.0%) at its final epoch and 163/256 (63.7%) at its best
development checkpoint, with 827/1024 fitting examples correct. Exact linear-work event
memory preserves the audited predictions/gradients and speeds forward/backward by 1.68x
on this CPU. We can now investigate generalization and temporal credit in a model that
learns. [§§173–175](theory/16_event_work_and_temporal_credit.md) derive its event adjoint,
timing eligibility and causal coalescing error. The
[frontier protocol](SHD_FRONTIER_PROTOCOL.md) separates development accuracy, official
test targets, CPU time and measured energy. The remaining numbered items describe
mechanisms to retain or develop, rather than claiming the new carrier is still at chance.

1. **Repair the hybrid event contract.** Legacy TVLayer detects crossings after arrivals but reconstructs
   payloads from before those arrivals. A minimal witness gives zero payload and zero payload gradient;
   the consistent grid reference restores both. Audit and compare this correction before more router tuning.
2. **Separate information from optimization.** Probe input and hidden representations, establish small
   fit-set learning with the deepest terminal head, then measure speaker transfer and useful depth.
   Event activity and nonzero auxiliary gradients alone have already failed as success criteria.
3. **Measure transferable optionality.** Same-sample virtual progress contains a gradient-variance bonus
   tr(MΣ). Adapt on one training subset and evaluate on another; retain future route choices, explicitly
   accounting for when their selection information is available. Do not reward duplicate choices or noise.
4. **Build bounded serial event continuations.** Preserve a payload/credit path through each learned
   transformation, bound event and shadow work, and use arrival-driven state updates. A sparse skip
   around deep layers is insufficient evidence that the deep transformations learn.
5. **Complete the frontier evidence.** AWS completes the non-SHD scale/baseline work. Compare matched
   quality, training cost, inference latency, memory, and measured energy; include indexing and credit.

SHD gates are: causal kernel correctness → fit-set learning → held-out-speaker generalization → useful
2/4/8-layer composition → calibrated early decisions → matched quality/work advantage. Optionality and
new depth are mechanism changes to cross those gates, not independent reasons to claim superiority.

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

1. **Finish fair small language comparisons.** On identical text splits and tokenization, train converged LSTM, 2/4-layer Transformer, time-vector model, and event/native mixture at several data and compute budgets. Include frozen and online-adaptive results separately. The 90M-character E79 race mixture reaches 1.50 bpc and remains behind published Transformer results near 1.1; E77 has not established parity. E64b's 1M, 20-pass LSTM and 2-layer Transformer score 2.179 and 2.367 held-out test bpc; the 10M four-layer Transformer scores 1.9083 held-out test bpc after validation-based checkpoint selection. It is 0.1090 above the 10M LSTM's test bpc. These are same-data, single-seed results, not capacity- or training-budget-matched comparisons.
2. **Test deep Hopfield event message passing.** E79's successful text8 result comes from a race mixture of independent predictive experts plus copy memory; its learned expert allocation is not evidence of gradient training through deep hidden layers. E77 is the separate deep time-vector character model: content-gated/delayed vector messages, prior-event routes, event-level key/value retrieval, recurrent state, and token retrieval. A 4,096-character, 16-update depth-4 pilot found that fixed $\theta=1$ silenced deeper layers. A raw voltage quantile then failed at default width: test rates were $[0,0.007,0.009,0.004]$, with gradients at only $[1,9,12,6]$ of 13 validation points. Exact replay of reset dynamics matched the initial rates to $[0.096,0.082,0.094,0.100]$ and reached all four layers at every point. A width-8 depth-8 pilot reached all eight layers at all 16 points, with test activity $[0.115,0.125,0.217,0.075,0.081,0.138,0.124,0.121]$. These are gradient-reach results, not quality or scaling evidence: 512-character test BPCs were 4.254 at depth 4 and 4.319 at depth 8. Fresh route shadows had a small, uncertain layer-2 signal; counterfactual/pathwise norm ratio was 0.77% and cosine 0.0040, so the correction remains disabled. The 1M-character, five-pass, parameter-matched 8-layer Transformer control completed at 2.352 test BPC in 1,617 seconds; its width-128 E77 partner exceeded the 3.5 GB RSS cap at 3.83 GB before training. Do not retry that configuration. Prioritize the 100k-character, one-pass E77 depth-8 run at batch 2 and 128-character windows; defer another Transformer control until its result is reviewed, and attempt depth 16 only after measuring this run's memory and time. Isolate event-Hopfield and token-retrieval ablations one at a time. Record layerwise query/key/value/gate gradients, pathwise/counterfactual norm ratios and cosines, counterfactual shadow-delta variance and coverage by layer, the full fixed-topology sequence-Jacobian certificate including maximum key fan-out, payload diameters, score entropy, retained/new event counts, pair-scoring work, and test loss. E77 has next-character labels at each causal position; apply the SHD lesson about lost routes by shadowing near-boundary message alternatives against the affected downstream token losses, not by copying the SHD sequence-prefix target. Begin with a weak prior on route utility, update it from fresh layer/score-stratified shadows, and cap corrections in the optimizer metric. THEORY §§133–134 distinguish route-utility uncertainty from parameter uncertainty and derive a local evidence-conditioned gain; neither posterior-driven gain nor online homeostasis is implemented. §§19/57 derive route discovery credit; measure its transfer to key/value support selection and its work as depth grows. THEORY §113 composes local fixed-support errors through residual depth, and §114 supplies a full-sequence Jacobian bound $\sqrt{RC}$ whose column sum captures shared-key fan-out. E114 completed first: 144 central finite-difference cases over sequence length, retained mass, temperature, and diffuse/reused-key patterns showed no absolute violation above 1e-8; among nontrivial bounds, the maximum Jacobian and forward ratios were 0.897 and 0.687. E77's token query/key retrieval still uses dense causal all-pairs scoring, so the hidden-route experiment makes no full-model sparsity claim.
3. **Separate associative recall from search cost.** Start with all-past-event softmax as the trainability/recall reference. For a candidate set, measure omitted mass and the exact query-gradient residual from §112: omitted within-group covariance plus the between-group key/advantage term. The new H-smooth extension also bounds the *full-loss* query, key, and value VJP errors, including the change in upstream gradient caused by the output error; its query bound is $\beta\epsilon D_K[(3/2-\epsilon)D_A^C+HD_V^2/4]$. Check these bounds against dense autodiff while varying support mass and downstream curvature. Mass recall alone does not guarantee gradient alignment. Use the existing near-miss route credit; its key-insertion loss estimate $rA$ has a curvature-bounded error, so allocate exact shadow execution by the resulting per-candidate remainder bound. Vary candidate budget and context, and measure route recall, output and gradient error, index maintenance, bytes moved, shadow work, and actual score operations. Top-k after dense scoring tests aggregation only.
4. **Test race attention against softmax directly.** Match parameters and training budget, vary races per head and depth, measure output/gradient bias and variance, convergence, loss, and actual event work. This isolates whether stochastic local credit preserves the Hopfield address/value learning signal.
5. **Only then scale tokens and hardware.** Move successful variants to larger text and longer contexts; compare energy and throughput on event-capable and GPU hardware. Include search/index construction, communication, synchronization, and training overhead.

## Decision rule

Advance a mechanism when its predicted intermediate statistic changes in the predicted direction and improves the equal-budget quality–resource frontier across seeds. Retire or revise it when the intermediate statistic fails, even if a single benchmark score improves. This keeps the mathematical story connected to causal evidence and directs effort toward the actual frontier: learned sparse representations, trainable deep routing, and sublinear retrieval without a quality loss.

## Deep event-stream models on real data

E83 is sequence classification: SHD gives one digit label for a complete
utterance. A code audit found that the previous implementation averaged
per-time softmax probabilities across a shared padded simulation grid. Silent
bins contributed uniform guesses, and a short utterance's score depended on
the longest utterance in its batch. The old depth-2 runs are retained as
debugging records but are excluded as depth or trainability evidence; the
guarded queue was stopped before depth 4.

The corrected E83 now separates posterior learning from the stopping rule.
Its `event_prefix` objective trains a class posterior with proper log loss at
a small set of causal query times, then evaluates threshold stopping on that
posterior. Query times are stratified in a fixed, shared 0–1000 ms physical
window; the earlier sampler used each example's last event to set its window,
which leaked future duration into prefix weighting. This objective is distinct
from the earlier race-only screening experiments:

- **Output race:** no answer before any class clears a confidence threshold;
  the first threshold-crossing class is emitted. THEORY §115 derives a
  competing-risk objective for the probability that the labeled class wins,
  including the survival probability of “no output yet.” A time discount
  rewards earlier correct answers. Validation must report emitted-answer
  accuracy, coverage, and latency, since either premature errors or no-output
  cases can make the race unusable.
- **Max-over-time potential:** a full-coverage SNN control used in the original
  SHD benchmark study.
- **Integrated potential:** cross-entropy on the time integral of readout
  potentials, another published sequence-classification scheme.
- **Causal prefix posterior:** cross-entropy on the labeled class at sampled
  prefixes. In expectation, this is proper for $P(Y\mid\mathcal F_t)$ even
  though only the complete utterance label is supplied. It does not force an
  early answer; the threshold remains a separate decision.

Each sequence now has its own end time and pooling deadline, so co-batched
padding is excluded. A tiny 80-train / 40-validation, one-epoch smoke remained
near chance: the revised race reached 92.5% coverage but only 10.8% accuracy
among emissions; max-over-time accuracy was 2.5%. This is a failed, low-power
pilot, not positive evidence. In the guarded 512-train / 128 held-out-speaker,
two-epoch controls, integral pooling ended at 4.69% max-over-time accuracy and
5.47% with terminal fallback; its race coverage was 17.97%, with 4.35%
accuracy among emitted answers. Max pooling ended at 4.69% max-over-time
accuracy and 3.91% with the default-threshold fallback. Both are at or near
20-way chance (5%), so they have not shown recognition. The race-plus-fallback
`anytime` control reached 6.25% max-over-time accuracy (8/128; chance-tail
probability 0.31), 5.47% emitted accuracy, and 100% coverage at threshold 0.6;
thresholds 0.3–0.9 all emitted every item, with confidence saturating at 1.0.
Layer-4 activity grew from 932 to 1,024 spikes per utterance. This identifies
false confidence and event amplification, not reliable evidence above chance.
The matched stable-cause race-only control ended at 3.12% max-potential
accuracy, 97.66% coverage, 4.8% emitted accuracy, and 0.981 peak confidence.
These screening runs use two epochs and a small subset, not a competitive SHD
benchmark.

For a true prefix posterior $\pi_t(c)=P(Y=c\mid\mathcal F_t)$, the first time
$\max_c\pi_t(c)\ge1-\epsilon$ gives conditional error at most $\epsilon$ at
the output. This bound applies at a stopping time and needs no prefix label,
but it requires calibration on the selected first-crossing prefixes. A
marked-point-process posterior accumulates both event log-likelihood jumps
and no-event survival terms; class-dependent silence can change confidence
between arrivals and may require scheduled clock updates. If an event updates
only $r$ class logits, indexed max and log-sum-exp trees give an exact
$O(r\log C)$ confidence-threshold check; generating the full $C$-class
probability vector still costs $O(C)$ on emission. The reference
`experiments/sparse_anytime_readout.py` implements the sparse additive-update
case, but is not yet integrated with E83. §§122–126 derive the guarantees and
the limits of this event-driven readout.

A second code audit found that `TVLayer` computes its content gate as
`r.detach() > 0` and its spike identity mask under `no_grad`. On a fixed batch,
a closed route has exactly zero pathwise derivative with respect to its gate
score; an open route gets only its conditional delay/payload derivative, and a
silent unit gets no end-to-end output derivative. The candidate route masks
remain fixed random/tonotopic buffers, so only content-gate choices within
that candidate graph can be recruited. The new event-prefix objective adds
sampled §19/§57 boundary credit for those gate choices; hard silent-unit
births and edges outside the fixed candidate graph remain uncredited.

The first exact shadow probe was deliberately narrow: four held-out-speaker
utterances, 32 near-gate insertions into the final readout, and a fresh seed-2
initialization. 31/32 interventions changed max-pooled terminal loss by
exactly zero; one improved it by 0.045. The summed counterfactual gradient was
2.2% of pathwise norm with cosine 0.012. Section 129 explains the max-pooling
winner-gap dead zone. E83 now also shadows hard content routes through the
full downstream network under the prefix objective. The unbiased
candidate-count/shadow-count estimator destabilized the matched depth-4 run:
about 480k near routes and 128 shadows per epoch preceded train-loss and
activity explosion. A bounded normalized local rule avoided that runaway but
did not improve recognition. In its matched seed-6 run, epoch-2 terminal
accuracy remained 6.25% (2/32), prefix NLL was [2.996, 19.735], only 11.6% of
112 sampled openings helped, and counterfactual/pathwise cosine was 0.0038.
The matched pathwise-only control had 6.25% accuracy and prefix NLL
[2.995, 3.202]. The local rule's raw counterfactual/pathwise gradient norm
ratio was 1.64%; its applied update was separately capped. This establishes
that estimator scale and layerwise route utility need attention, not that
counterfactual learning works on SHD.

Per-layer mean route effects varied, but were small relative to the sampled
shadow spread. A weak prior at initialization should not enforce untrained
route preferences; however, a small, noisy sample still produces a broad
posterior, not a license to magnify its mean. §132 proposes robust
layer/score-stratified posteriors for route utility and treats step-size as a
separate optimizer-metric trust-region problem. More shadows or useful
stratification are needed before increasing the gain. Hard silent-neuron
firing still has no counterfactual boundary term.

A paired hidden-spike audit now measures that separate boundary. On 128
held-out-speaker examples, near-threshold margin candidates (within ±0.25)
averaged 349 per batch in layer 1, 67 in layer 2, 7.9 in layer 3, and 6.6 in
layer 4; layer 4 had no in-band margin in 24/32 batches. The closest
spike-on/off intervention improved loss in only 16/32, 17/32, 14/32, and
14/32 batches across the layers. Thus deep spike boundaries are scarce, and
the single-event shadow is not class-useful consistently at this checkpoint.
Separately, E83 merges repeated same-band events and then discards their
returned log-count mark. Multi-spike groups are 55.6% of fitting events and
43.5% of held-out-speaker events. The current event map therefore applies a
many-to-one quotient to a potentially task-relevant mark before the first
vector layer. The sufficiency condition $I(Y;C\mid B,T)=0$ has not been
measured.

The strict chain has an additional exact support invariant. If layer $\ell$
receives no events, its event-driven state remains $z(t)=0$. There is no
baseline drive, and its threshold $\theta>0$, so it cannot emit an event:

$$
E_\ell(x)=\varnothing\Rightarrow E_{\ell+1}(x)=\varnothing,
\quad\text{hence}\quad
\mathcal A_{\ell+1}\subseteq\mathcal A_\ell,
\qquad \mathcal A_\ell=\{x:E_\ell(x)\ne\varnothing\}.
$$

This nesting is about the fraction of utterances with any event, not the
mean event count. Writing $c_\ell=P(x\in\mathcal A_\ell)$ and
$m_\ell=\mathbb E[|E_\ell|\mid x\in\mathcal A_\ell]$ gives
$\mathbb E|E_\ell|=c_\ell m_\ell$: coverage may fall monotonically while
multiplicity among active utterances grows sharply. In the original matched
32-example four-epoch depth-four `all_depths` trajectory, accuracy stayed at
3.1–6.25% and ended with mean spikes `[10, 1, 0, 0]`; on the first training minibatches of epochs
3 and 4, main-loss gradient norms for layers 3–4 were zero. Thus direct sparse
readout fusion did not reopen deep support in that trajectory. The implementation
now records per-layer active-example coverage, events per active example, and
support-nesting violations in future evaluations.

The 32-example deepest-only versus `all_depths` control and the one-factor
sparse count-mark arm are complete. Deepest-only reached 5/32 on the fixed D4
endpoint, but the paired head comparison was inconclusive (McNemar $p=0.453$)
and its prefix NLL was worse than uniform. The count mark kept layer-4 support
at 56–94% but ended at 0/32 after epoch 1, with epoch-4 prefix NLL
$[8.4786,27.28]$.

The matched 128-example seed-6 comparison is now complete. `all_depths` ended
at 17/128 terminal accuracy, versus 7/128 for `deepest`; terminal layer-4
coverage was 61.7% versus 4.7%. On thresholded predictions with terminal
fallback, the paired correct counts were 20 versus 7 (19 discordant cases
favoring fusion and 6 favoring deepest-only; exact McNemar $p=0.0146$). This
supports an effect of the multi-depth objective/readout package on this fixed
two-speaker evaluation, not a generalization claim. The late-prefix NLLs were
17.57 and 4.66, respectively, both worse than uniform log loss 2.996; the
all-depth model's terminal accuracy also did not change when its layer-4
branch was removed. The objective preserved deep support and improved
thresholded decisions in this seed, while additive evidence remained poorly
calibrated and the deepest branch's task contribution remains unproved.

The matched seed-7 `all_depths` replication ended at 9/128 terminal accuracy
versus 7/128 for `deepest`; race-plus-fallback predictions were 9 versus 6
(exact McNemar $p=0.607$). Layer-4 coverage was 22.7% versus 16.4%, and
late-prefix NLL was 5.74 versus 4.09. The all-depth race emitted eight times
at mean confidence 63.1%, with no correct emission. Thus the seed-6 paired
accuracy result did not replicate, while the overconfident race behavior did.

A second strict-chain seed is complete: deepest-only reached 7/128, with
16.4% layer-4 coverage and late-prefix NLL 4.09; it emitted four times at the
fixed threshold and none were correct. Both matched sparse
layer-1-event-skip runs are complete. At seed 6, layer-4 coverage rose from
4.7% to 99.2%, with 1.8% more candidate scores; terminal accuracy was 9/128
versus 7/128, while paired race-plus-fallback output accuracy was 9 versus 7
(McNemar $p=0.791$), and no fixed-threshold emissions were correct. At seed
7, layer-4 support reached 100%, terminal accuracy was 5/128, late-prefix
NLL 48.80, and candidate-score work was 22.3% above strict-chain; paired
output accuracy was 5 versus 6 ($p=1.0$), with no correct emissions. The
added path restores support in both seeds but not useful class credit, and
its activity scale is seed-sensitive. The skip topology now draws from its
own RNG, leaving strict adjacent-layer
masks identical to the control. This skip can
break intermediate-layer absorbing silence while preserving event-driven
fan-out, but it can also make the deeper hierarchy redundant, so compare
branch-ablation accuracy, support, and synaptic work. E83 now defaults to the
implemented `--rng_protocol split`: evaluation selection, training subset,
training order, augmentation, prefix sampling, and route shadows use separate
random streams. Existing artifacts used the old shared stream; request
`--rng_protocol legacy_shared` for that protocol. The split implementation
still needs paired cross-evaluation-limit validation before claiming that the
earlier 32-versus-128 difference has been isolated. Only after these
representation/support tests should a trained hidden-spike boundary
estimator be considered: its paired deltas are currently noisy and not
consistently class-useful. Threshold homeostasis is a separate control for
event-rate impedance, not evidence of task credit.

Section 138 makes the support penalty quantitative. If an example supplies a
pathwise gradient $X$ only when its layer is active, then $G=A X$ with
$A\sim\mathrm{Bernoulli}(c_\ell)$ has mean $c_\ell\mu_\ell$ and covariance
$c_\ell\Sigma_\ell+c_\ell(1-c_\ell)\mu_\ell\mu_\ell^\top$. In an IID batch
of size $B$, no active example occurs with probability $(1-c_\ell)^B$; when
conditional gradient noise dominates, task-direction gradient SNR scales
approximately as $\sqrt{B c_\ell}$. This establishes support as a direct
credit bottleneck under the stated sampling assumptions, while leaving the
conditional direction and class information in $\mu_\ell$ open.

The same section separates the hidden-spike boundary term from pathwise
credit. Under logistic threshold perturbation, it is
$p(1-p)(L_1-L_0)\nabla g/\sigma$ for the explicitly smoothed gate objective,
where the on/off replay includes the refractory effect. A deterministic hard
gate has zero derivative almost everywhere, so the replay estimate is a
surrogate learning rule whose variance and usefulness still need evidence.
It also derives a gate-space sufficient condition for stable firing decisions
under an AdamW step. Raw global gradient clipping does not generally bound
AdamW's parameter displacement; the relevant quantity is the predicted
margin movement $\nabla g_i^\top\Delta\theta$ relative to $|g_i|$ and local
curvature. E83's epoch-level cascade/extinction sequence is consistent with
large gate flips but cannot identify them without per-update margin traces.

The new compute-matched data-diversity screen used 120 optimizer updates in
each arm, with nested 120-example/four-epoch and 480-example/one-epoch
subsets. Accuracy favored the larger subset in seed 6 (20/128 vs 8/128,
paired McNemar $p=0.0227$) and the smaller subset in seed 7 (20/128 vs 8/128,
$p=0.0357$). In all four arms, layer-4 support remained 1.56–7.03% and
late-prefix NLL 3.02–6.58, above uniform $\log20$. This small two-speaker
screen finds no repeatable data-diversity gain; the seed reversal is
diagnostic of instability, not a prompt for more unstructured seed runs.

Sections 139–140 refine the diagnosis with explicit route reachability and
depth arithmetic. Reconstructing the exact E83 D4 masks shows 90.2% and
98.0% of first-layer/fourth-layer unit pairs have a static candidate path in
seeds 6 and 7; all 140 input bands reach at least one layer-4 unit. Yet
realized layer-4 support in the equal-update arms is 1.56–7.03%. The sparse
layer-1 skip controls raise it to 99–100% without paired accuracy improvement.
Static graph disconnection and missing active examples are therefore
separated from label usefulness; neither path existence nor path survival
alone establishes a trainable route.

For two competing routes, the relaxed router's loss gradient is proportional
to the paired loss difference between both route outcomes. E83's shadow
procedure replays a masked edge open/closed, but only for event-conditioned
scores inside its near-boundary band and only one edge per replay. Far-closed
instances get no direct shadow probability. In addition, vector contributions
can sum across a hard firing threshold: two individually subthreshold routes
may jointly create the class-useful event. The exact two-gate gradient shows
that when only the pair helps, one route's gradient is proportional to the
other route's opening probability and is zero if that partner is never
present. A small sparse set of joint 00/10/01/11 replays, selected for
overlapping arrivals near a receiver threshold, would test this synergy
without executing every candidate pair.

Section 142 now turns that calculation into a bounded receiver-bundle pilot.
For two closed gates, it uses the exact four-outcome independent-gate
gradient and logs the interaction $\Gamma=L_{11}-L_{10}-L_{01}+L_{00}$.
Only same-receiver near-boundary pairs with close source times are proposed;
each selected pair costs three downstream replays. This is ordinary sparse
router counterfactual credit specialized to temporal vector messages, not a
claim that sparse-route alternatives are generically novel. The pilot tests
whether receiver-level cooperation supplies useful SHD credit. Even if it
does, the exact relaxed derivative remains attenuated by the partner's
opening probability; longer path shadows or a better justified joint
exploration law may still be needed.

One additional design risk is that E83's current content score controls both
the hard gate and the positive delay: $a=t+\tau[s]_+$. Route credit that
pushes a useful gate open consequently retimes it. The shadow update measures
the presence contrast at the fixed current delay/payload; it is not a
counterfactual timing gradient. A follow-up ablation should factor a sparse
admission score from a conditional delay/value head, then separately test
gate utility, delay calibration, and pair utility stratified by effective
arrival-time difference and projected vector contribution. This preserves
sparse execution while removing a forced monotonic coupling between
admission and latency.

The same analysis answers whether to add many layers immediately. Under
independent random masks, expected static paths between endpoints grow as
$M^{D-2}p^{D-1}$ with hidden width $M$, per-edge probability $p$, and $D$
hidden layers. However, strict-chain example support is exactly
$c_D=c_1\prod_{\ell<D}s_\ell$, where $s_\ell$ is conditional example
survival at each transition. Keeping half the examples active at depth 8 or
16 from full initial support requires average per-transition survival of at
least 0.906 or 0.955. More layers add combinatorial expert compositions but
multiply every unprotected survival bottleneck. The current evidence does
not justify indiscriminately deepening E83; first determine why candidate
routes are not simultaneously realized and credited, then scale depth while
measuring both path survival and route-credit coverage.

Do not then train the stop threshold as a proxy for posterior quality. For a
prefix sampled from a declared latency distribution $t\sim\nu$, log loss
satisfies
$\mathbb E[-\log q_t(Y)\mid E_{\le t}]=H(\pi_t)+D_{KL}(\pi_t\Vert q_t)$,
so its population optimum is the actual prefix posterior even though the only
target is the complete-stream class. Sample a limited number of prefixes per
utterance (including its empty/prior prefix and its end), train this posterior
with proper scoring, then calibrate and select the stopping boundary on
held-out speakers. The race success objective can be added later as
decision-focused fine tuning; by itself it constrains integrated win mass,
not the probability trajectory at the first crossing. This gives a staged,
mechanism-based plan: first verify route credit, then verify prefix posterior
quality, then optimize early stopping.

After validating the event marks and operating rates, compare iso-width
depths and auxiliary weights on held-out speakers. The SHD source has no
phoneme alignment in this pipeline, so those depths are not yet matched to
annotated phonetic stages. Per-layer dmax also expands the maximum path delay
with depth; a subsequent scaling comparison must control the total delay
horizon or treat that expansion as an explicit experimental factor.

E84 remains a separate day-5 market likelihood experiment. It uses a strict
adjacent-layer event chain and deepest-only prediction, with sampled windows
matched across depths, and logs gradient alignment, candidate scores, messages,
state scans, and held-out per-event log likelihood. It has not run since the
guarded queue was stopped; the confirmatory market days remain untouched.

Read the diagnostics jointly. Nonzero deep gradients with no validation gain
point to representation or optimization quality; vanished deep gradients with
healthy early-layer gradients point to chain conditioning. For SHD, a race with
high coverage and poor emitted accuracy is overconfident or poorly
discriminative; low coverage indicates its threshold or evidence dynamics do
not trigger. Candidate score pairs expose work spent on rejected messages,
while state-vector scan counts expose the reference implementation's
grid/event-step updates. These models remain sparse in message topology but
still scan state cells at each step, so neither active messages nor sparse
masks alone establish low energy. Any winning objective or auxiliary scheme
must repeat across seeds before depth claims.

## xLSTM transfer: topology and scale protocol

xLSTM supplies a concrete deep-recurrent precedent, not an equivalence theorem for E77. The original family distinguishes
scalar gated memory (sLSTM) from query-addressed matrix memory (mLSTM); its 7B model and later scaling study show what a
serious non-Transformer scale program looks like ([xLSTM](https://arxiv.org/abs/2405.04517), [xLSTM 7B](https://arxiv.org/abs/2503.13427),
[xLSTM scaling laws](https://arxiv.org/abs/2510.02228)). Transfer the experimental discipline: define a repeatable macroblock,
stabilize gates and residual gains, measure model and kernel throughput together, and sweep depth, width, data, and context
under matched compute budgets. Preserve E77's event-state and sparse-route structure; do not infer that xLSTM's dense
position-wise blocks or its measured scaling result transfers automatically.

The algebraic opportunity specific to our state update is THEORY §107(i). For fixed event bins, represent each recurrence
step by its diagonal affine map $(A_k,x_k)$ and use the associative composition
$(A_j,x_j)\circ(A_i,x_i)=(A_jA_i,A_jx_i+x_j)$. The proposed verification sequence is:

1. On a fixed schedule, compare the sequential recurrence with a prefix scan at every time bin, including complex state and count normalizer.
2. Compare loss gradients for all input payloads and decay parameters; they should agree up to floating-point reduction error because both compute the same affine map.
3. Measure peak memory, wall time, state bytes, and scan work at increasing grid length and depth. The scan has linear work and logarithmic parallel depth, but still emits dense prefix states.
4. Only after that passes, test a small event-sparse segmented scan that skips empty intervals. Count arrival sorting, dispatch, hidden dense buffers, and emitted-state storage in the cost.

This path targets parallel training of the affine memory core. Spike threshold/reset logic and event-topology changes remain separately measured; scan speed alone would not establish an end-to-end or energy advantage.

Section 141 generalizes the same lost-route comparison to all dormant event
proposals. It assigns each candidate a signed failure margin for one cause
(content gate, receiver threshold, timing race, or refractory availability),
defines contextual utility $U=L_{off}-L_{on}$ from a paired replay through
the full downstream state, and derives the smoothed margin update
$-p(1-p)U\nabla m/\tau$. This says exactly how a counterfactual can tune the
knob that prevented an otherwise useful event. A richer candidate pool can
increase the chance of observing useful alternatives, but only with proposal
coverage and repeated low-noise utility estimates. Unstratified sampling can
dilute a fixed shadow budget; exhaustive replay can be costly and its
inverse-probability sum high-variance. The theory therefore proposes a
bounded, cause-stratified audit with an exploration floor and propensity
logging. No experiment has yet shown that this update improves SHD accuracy.

Section 143 uses the matched layer-balanced follow-up to separate support
from useful computation. Per-utterance event count is $n_\ell=c_\ell\mu_\ell$,
where $c_\ell$ is the active-example fraction and $\mu_\ell$ is conditional
event multiplicity. Support can remain 100% while event count grows by
factors above four across one layer; this is a distinct failure from
vanishing route support. Because E83's readout adds per-event class evidence,
activity amplification can also amplify misaligned logits. The balanced
pair run kept all examples active at layer 4 but ended at 4.69% accuracy,
with 1,647 layer-4 events per utterance and only two predicted classes.

The immediate mathematical extension is a constrained expected-work
objective $\min \mathbb E[L_{class}]$ subject to
$\mathbb E[C_\ell]\le B_\ell$. Its Lagrangian adds
$\lambda_\ell(C_\ell-B_\ell)$; a matched shadow should replay class loss
and per-layer event/message work together, then use their weighted utility
for gate credit. The next experiment first logs those joint deltas so the
budgets/multipliers can be calibrated rather than guessed. Coverage and
conditional activity must both be monitored to avoid the all-silent solution.
This analysis also records that current E83 hidden-state training scans a
1 ms grid and is not evidence for asynchronous training efficiency.

The frozen validation audit in §143 finds a layer-dependent utility/work
tradeoff on the collapsed layer-balanced checkpoint. Only 15.6% of sampled
L1 pairs and 31.3% of L2 pairs improve the matched prefix loss, versus 57.9%
of the 19 sampled L3 pairs and 50.0% of L4 pairs. A layer-2 pair adds 4.24
L4 spikes and 10.76 L4 readout updates per example on average; corresponding
L3 deltas are 1.79 and 4.04. Reconstructing the clipped two-gate derivative
gives mean $\partial L/\partial s$ of +0.291/+0.517/-0.013/-0.114 across
L1–L4, so the exact local rule tends to close early alternatives and weakly
open late ones. This motivated a matched sampler restricted to the last half
of layers. It was validation-informed development work, not a final test.

The late-balanced follow-up generated only 9 L3/L4 pair shadows in epoch 1
and zero in epochs 2–4, even though L4 validation support remained nonzero in
epochs 2–3. The exact bottleneck is pair co-occupancy: two distinct,
near-time, closed routes must converge on one receiver. For each
minibatch/receiver group $g$ with $n_g$ eligible routes, the sampler filters
adjacent sorted arrivals by gap and source-event identity, so its pair count
is at most $P_\ell=\sum_g(n_g-1)_+$. If $n_g$ is approximately Poisson with
low mean $\lambda_g$, this bound has expectation
$\lambda_g-1+e^{-\lambda_g}\approx\lambda_g^2/2$, compared with
$\Pr[n_g\ge1]\approx\lambda_g$ for a single-route shadow. Pair opportunities
therefore have a quadratic sparse-flux upper envelope. Importance weighting
cannot recover a gradient when the candidate set is empty. Preserve
first-order shadows when pair support is absent; for deep credit before
upstream activity exists, a sparse prefix-expansion replay must explicitly
create and propagate a candidate event, with its proposal probability and
marginal event work logged. This is a derived design constraint, not a
validated training method.

A no-update occupancy audit then swept the pair window from 25 to 1,000 ms on
the same 120-example fit subset. At 25 ms, per-layer pair counts were
$[181577,1,0,0]$; at 1,000 ms they were $[189987,6,0,1]$. No L3 pair appeared
even with a full-second window, while L2 and L4 pairs appeared in only 6/30
and 1/30 batches. The near-closed route records were
$[191787,111,3,13]$ across L1–L4. This rules out the current 25 ms cutoff as
the main source of late-layer pair starvation. Extending a window beyond the
receiver's integration times could add implausible pairs, so any later window
change must measure both utility and receiver-state interaction. The more
direct missing quantity is deep source-route co-occupancy.

The refractory-aware spike audit narrows a candidate training mechanism. It
compared the matched no-pair and late-only checkpoints on identical
held-out-speaker batches, retaining only toggles with $|m|\le0.25$ and no
refractory block. The valid L1/L2/L3/L4 counts were $[22,21,8,0]$ for control
and $[21,19,2,1]$ for late-only. The audit separates the fused main-posterior
loss from weighted auxiliary losses. In L1, spike-on helped 16/21 late-only
main-loss candidates (mean $-0.0094$, median $-0.0020$), versus 12/22 control
candidates (mean $+0.0266$); the auxiliary term has the same direction. Only
13 batches had valid L1 candidates in both arms; the mean difference between
the arm-specific selected spike-on utilities was $-0.037$ (SE $0.035$), and
the selected unit/time can differ between checkpoints. This is a single
validation-conditioned signal, not a reliable treatment effect. The L2
late-only main-loss mean was $+0.0040$ despite 12/19 individual improvements;
this does not support opening that layer on average. L3/L4 do not have enough
valid samples. A paired deepest-only replay changes the interpretation: every
valid L1/L2 delta is exactly zero in both arms. Late-only L1 has all-depth mean
$-0.00936$, but its toggles add no downstream hidden spikes; hidden-spike
changes per example are $[+0.1429,0,0,0]$, and only L1 readout edges change
($+1.2381$ updates/example). Because all-depth fusion sums evidence from every
layer, its main loss can reward a shallow readout branch without any serial
credit reaching the deepest head. This is a measured shortcut on these
checkpoints, not evidence that deep composition is impossible. A depth test
must train against the deepest-only primary objective or use an explicit,
work-capped sparse suffix-expansion replay and show a downstream event change
that improves that objective. The paired replay itself is not a training
result; L3/L4 sample sizes remain too small to estimate their utility.

Section 146 expands this audit to 1,024 held-out-speaker utterances. It finds
that event propagation and task utility are separate: among natural-off L2
pairs, opening both increases suffix spike count in 38/64 control and 19/34
late-only cases, but improves deepest-only loss in only 8/64 and 4/34. L3
counts are just 10 and 5 natural-off pairs; 5/10 and 4/5 double openings
improve deepest loss, but the late-only median is −0.032 while one harmful
outlier is +1.40. These are frozen interventions, not a treatment effect.

For independent logistic spike risks, the expected loss over the four binary
corners has mixed derivative
$\partial^2\widetilde L/(\partial m_i\partial m_j)=
p_i(1-p_i)p_j(1-p_j)\Gamma_{ij}/\tau^2$, where
$\Gamma_{ij}=L_{11}-L_{10}-L_{01}+L_{00}$. This isolates pair-specific credit
from the two singleton utilities. None of 113 sampled natural-off pairs had a
beneficial joint opening when both singleton openings were non-beneficial;
only 2/42 L3 and 0/210 L2 pairs had $|\Gamma|>0.01$ (descriptive threshold).
The sample shows rare terminal leverage but little measured pair-only
synergy. It also corrects a mechanism distinction: this spike-pair audit picks
two hidden-unit events within 50 ms, not two incoming message routes that
share a receiver. A route-pair claim cannot be inferred from it.

The next discriminating experiment should replay topology-conditioned pairs:
two candidate source events must have a common next-layer receiver, and their
candidate arrivals must overlap on its integration timescale. Compare them
with time-matched nonshared pairs on the same checkpoint. Log accepted route
gates, receiver-state and event-time/payload changes, all four deepest-loss
corners, $\Gamma$, and sparse replay work. This separates structural
coincidence from actual content-gate acceptance and class utility. Keep
single-event boundary shadows as the baseline; train with pair credit only if
the shared-receiver arm produces repeatable interaction signal and then
improves a matched deepest-only control.

## Cross-domain anchors

- Timed automata and timed-word languages: [Alur & Dill (1994)](https://doi.org/10.1016/0304-3975(94)90010-8).
- Temporal coding and computational expressivity of spiking neurons: [Maass (1997)](https://doi.org/10.1016/S0893-6080(97)00011-7).
- Exact event-time adjoints for spiking models: [Wunderlich & Pehle (2021)](https://arxiv.org/abs/2009.08378).
- Local convex prediction and gated linear networks: [Veness et al. (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17202).
- Empirical language-model scaling and compute-optimal allocation: [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361), [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556).
- The project's detailed derivations and tests remain in [THEORY.md](THEORY.md), especially §§34, 71, 83–89, 94–109; current measurements and caveats are in [REPORT.md](../REPORT.md).
- Modern Hopfield retrieval and its attention update: [Ramsauer et al. (2021)](https://arxiv.org/abs/2008.02217); E77's additional results are the fixed-memory query Jacobian bound, sequence-level key-fan-out certificate, and conditional retained-event depth composition in THEORY §107(f–h).
- Exponential-gated recurrent and matrix memory architectures: [Beck et al. (2024), xLSTM](https://arxiv.org/abs/2405.04517); [xLSTM 7B](https://arxiv.org/abs/2503.13427); [xLSTM scaling laws](https://arxiv.org/abs/2510.02228). These are inspiration and comparison targets, not evidence for E77's own scaling.
