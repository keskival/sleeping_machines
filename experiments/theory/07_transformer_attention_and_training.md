# Transformer attention and its training

[Theory index](../THEORY.md) · Global sections 94–103. Section numbers are stable across these files.

## 94. Language as an event stream: what it takes to compete with Transformers (design, before any run)

*Written 2026-09-27.*

**Baseline mapping.** With each token (or byte) an event, the counted semi-Markov world model with hierarchical backoff
(§90) is a variable-order n-gram model, and leaky regime counters are a cache or topic model: n-gram quality, not
Transformer quality. The market result (§90) names the missing capacities: retrieval by content from long context and
statistics shared across similar symbols.

**Four ingredients, each from the calculus.**
1. *Tokenization as synaptogenesis.* Byte or character events; the composite units of §84–§85 over them are candidate
   n-grams; a synapse is grown only on predictive credit (BPE learned by credit, not frequency); pruning by credit (§93)
   bounds the per-character activity whatever the candidate space.
2. *Sparse distributed token codes.* Per-class parameters cannot generalize (§66); a token fires k of N shared feature
   channels, learned from co-occurrence, so routes learned for one token transfer to similar ones. Time adds rank-order
   coding: a token's most informative features fire first, and first-to-fire readouts use them first.
3. *Induction traces.* Each token type keeps a trace of its recent successors, updated only when it occurs: "A then B
   earlier; A again → B", content-addressed by identity at O(1) per event, with cost independent of context length.
4. *Mixing by conserved multiplicative credit.* Many count-based predictors (context detectors, induction traces, regime
   counters) combined by Hedge-type weights under a conserved budget (§83): the structure of context-mixing compressors,
   which reach roughly LSTM-level bits per character on enwik8 without deep backpropagation.

**Proposition (sampling is a race).** Give every candidate next token an exponential clock with rate λ_i ∝ p_i; the first to
fire is token i with probability λ_i / Σ_j λ_j (competing risks, §79; the Gumbel-max trick in time). The race readout is
exact sampling from the model's distribution, with no normalization step.

**Predictions (character-level language modeling, text8 / enwik8, bits per character).** Counts with backoff ≈ n-gram
level (≈ 1.7–2.0); + composite units and induction traces better than PPM-style models; + mixing ≈ context-mixing or LSTM
level (≈ 1.2–1.4), still above large Transformers (≈ 1.0); per-character work proportional to active units, constant in
context length. Staged, so each ingredient's contribution is measured.
(These numbers describe the counting-and-mixing design only; with race attention and learned codes (§96) the aim is
Transformer-level or better, and local credit is not a handicap in principle (§98).)
## 95. Topology and scaling laws of an event language model (before any run)

*Written 2026-09-27; companion to §94.*

**Topology.** (1) Symbol events → a sparse feature encoder (k of N shared channels per symbol). (2) L levels of order units
("u then v within a window": skip-grams), all chains as candidates, synapses grown only on predictive credit (§85),
extensions pruned by credit (§93): a context tree grown by credit (the structure of PPM and context-tree weighting,
extended with gaps and shared codes). (3) Slow state: leaky counters (cache, topic, regime; §90) and induction traces per
unit. (4) Every active unit predicts the next symbol by counts backed off along its parent chain. (5) A mixer with
conserved multiplicative weights (Hedge, §83) over the active predictors; readout by a race of exponential clocks with
rates ∝ the mixed probabilities (exact sampling, §94). No dense layer anywhere; a symbol touches only the units it
activates.

**Scaling laws (predicted).**
(a) *Compute per token* c ≈ L·r (r units firing per level after pruning): independent of the number of grown synapses S and
of the context length T (traces and counters update in O(1) per event). Training compute ≈ c·D, linear in data, with no
parameter factor (a Transformer: ≈ 6·N·D, plus attention over T).
(b) *Memory* S(D) ∝ D^h, h the growth exponent of useful contexts (Heaps-like; plausibly 0.5–0.7 for text): memory, not
compute, is the binding resource.
(c) *Loss.* For count-based context models the excess over the best predictor in the class is ≈ (effective parameters ×
log D)/D per symbol (minimum description length; context-tree weighting); with S ∝ D^h,
      L(D) ≈ L_∞ + A · D^−(1−h) · log D,
a steeper approach than a Transformer's data exponent (≈ 0.1) toward a higher floor L_∞, the entropy the context class can
capture; shared sparse codes and induction traces exist to lower L_∞. (This floor belongs to the *counting* design; it
is not a limit of event networks: §96 shows the Transformer layer itself has an event form.)
(d) *Mixer.* Hedge's regret over S experts is O(log S) in total log-loss: adding candidate detectors costs ≈ log S and no
per-token compute.
(e) *Depth* multiplies candidates by P per level but active units only by r, and pruning keeps activity close to linear in
depth (§93: −75% at depth 4).

**Tests.** On text8 subsets of 10⁶, 10⁷, 10⁸ characters: measure S(D) (h), L(D) (the exponent 1 − h and the floor L_∞) and
the constant per-token cost, against an LSTM and a small Transformer at the same data.
## 96. A Transformer layer in event form: race attention (proposition; learnability open)

*Written 2026-09-27, prompted by the question whether event networks must have a higher loss floor than Transformers.*

**Claim.** There is no expressiveness gap in principle: every operation of a Transformer layer has an event equivalent,
and stacking layers is composition.

1. *Similarity.* With sparse binary codes, q·k is the overlap of active channels. A key token arms a hold node with its
code; the query's channels arrive as triggers; the summed coincident input is the overlap (§83).
2. *Softmax as a race.* (a) Hard attention: with integrate-to-threshold latency decreasing in the drive, the first key node to
fire has the largest overlap (argmax by first arrival). (b) Soft attention: let key i fire as an exponential clock with rate
λ_i = exp(q·k_i/τ) (an exponential integrate-and-fire nonlinearity); the first to fire is i with probability
λ_i / Σ_j λ_j = softmax(q·k/τ)_i (competing risks, §79, §94). The winner's value code is an unbiased sample of the
attention output Σ_i softmax_i v_i; R parallel races (or R repetitions in time) average to it with error O(1/√R).
3. *Relative position* is native: delays and hold windows compute time differences directly (§61, §71), which dense
attention needs added position encodings or learned biases for (E36: the relative-time bias).
4. *The feed-forward block* is summed thresholds over sparse codes (perceptron layers; two layers are universal for Boolean
features) or composite units (§84).
5. *Residual stream and heads.* A shared spike bus to which layers add codes; heads are independent races.
Together with the substrate's universality (§59: an exact two-counter machine given timing precision), expressiveness is not
the question.

**What is open.** (i) *Learnability with local credit at depth.* Backpropagation trains queries, keys, values and the
feed-forward block jointly through softmax gradients. The race supplies local credit: when the wrong key wins, the right key
is pulled earlier (positional winning, §54) using near-miss counterfactuals (§57), with conserved multiplicative updates on
which channels to hold and which to trigger on (§83). Whether this trains deep stacks as well as gradients do is the
research question; §84–§89 (credit at an instant, latest-instant credit, margins, pruning by credit) are the progress so far.
(ii) *Precision.* Timing noise plays the role numeric precision plays in a Transformer; restoration (§59) bounds its
accumulation. (iii) *Cost.* With sparse codes a query reaches only keys that share an active channel (an inverted index),
so retrieval is sublinear in the context length rather than O(T), and work per token still follows activity.

**A lower bound, and room above it.** The emulation is a thought experiment, not a blueprint: it shows an event network
can do at least as well as a Transformer, whether or not it is built layer by layer. The event form also has degrees of
freedom a dense layer lacks, which is where it could do better: (a) *time as a coding axis*: the order in which n channels
fire carries up to log₂(n!) bits beyond which channels fire (rank-order codes); (b) *conditional computation by
construction*: only active units work, so capacity and per-token cost are decoupled; (c) *structure grown by credit*:
candidates cost nothing until they earn a synapse (§85, §93), and on deep order this finds structure from ≈ 10× less data
than a gradient-trained Transformer (E94, measured); (d) *exact sampling by races and sublinear retrieval by shared
channels*. Which of these pays off at language scale is the empirical question.

**The thesis, stated in three strengths.** (1) *Function class: established in principle.* By §59 (universality) and the
event form of a Transformer layer above, every Transformer is a special case of a Sleeping Machines network. (2) *Learning:
trivially subsumed, non-trivially open.* A universal substrate can simulate backpropagation, but an event network emulating
dense gradient descent loses its advantages (it becomes dense and synchronous). The substantive claim is that *native* local
credit, with cost proportional to activity, reaches the same quality: supported so far by learned depth (§84–§89), ≈ 10× data
efficiency on deep order (E94) and parity or better at equal data on composition and deep order; untested at language scale.
(3) *Power: in proportion to sparsity.* One synaptic event costs roughly what one multiply-add does, so emulating a
computation that genuinely needs every input to meet every weight at every step saves nothing; the savings come from sparsity
in the data (silence is free), the representation (sparse codes) and the computation (only firing units work), which is where
10³–10⁵× was measured. For language, per-token sparsity is plausible, so large savings are plausible; that is a prediction.

**Resource clarification, 30 September 2026 (§298).** Historical large factors
compare particular event counts and clocked-reference work or projected
hardware costs, not measured whole-language energy ratios. Preserve the task,
silence fraction, quality and resource boundary; §63 already records how an
event-driven digital control narrows the separation.

**Consequence for §95.** The higher floor predicted there is a property of a counting design. With race attention over
learned sparse codes the floor could be Transformer-like at a fraction of the work; whether local credit can learn it is
the test.

**Prediction (to be tested).** A single race-attention layer trained by local credit learns associative recall (copy the
value that followed a key earlier in the stream) and the induction pattern at accuracy comparable to a one-layer Transformer
trained by gradients, with retrieval cost sublinear in the context length.
## 97. Trainability of depth: a mistake bound for depth-d order detectors (theorem, proof sketch)

*Written 2026-09-27; assembles §83, §85, §91 and the absorbing-route argument of §84 into one statement.*

**Concept class.** Events on N channels; P part units (ordered channel pairs with windows); chain units up to level
L = d − 2 (a unit at level j is a level-(j − 1) unit followed by a part within W₂), so the candidate basis has
Q = Σ_{j=0}^{L} P^(j+1) ≈ P^(d−1) units. Each class k is an ordered pattern of d parts with bounded gaps.
*Realizability (R):* for each class there is a route (hold unit u_k at level L, trigger part l_k) that is present, with u_k
in the window before l_k, in every positive example at the pattern's last event, and complete in no negative example.

**Learner.** Class nodes with summed hold and trigger potentials over units (§83), conserved budgets, threshold θ > ½;
synapses grown on credit (§85: exactly dense Winnow); on a miss, the deficient roles at the latest candidate instant are
promoted by (1 + α); on a false fire, the contributors at the firing instant are demoted by (1 − β) with
β < (2θ − 1)/θ² (§83(ii)).

**Noise parameters.** q: probability that an example's latest candidate instant lies after the pattern (trailing noise);
f: maximal probability that a given distractor unit is in a credited set on a miss; ρ: false fires per miss in which the
target contributes (validity makes these fires due to other units).

**Theorem.** Let δ = (1 − q − 2f)·log(1 + α) − ρ·log(1/(1 − β)) > 0. For every class, with probability ≥ 1 − ε, the number
of misses before its route carries more than θ in both roles is

  M = O( (log Q + log(θ/(1 − θ)) + log(1/ε)) / δ² ) = O( d·log P / δ² )   (in expectation O(d·log P / δ)),

after which the route fires on every positive example (it is absorbing: validity means no false fire removes it, §84),
and every false fire caused by other units lowers the relative-entropy potential by at least
γ = log(1 − β) − 2·log(1 − βθ) > 0 (§83(ii)), so false fires are bounded by the potential as well. Memory is the number of
units ever credited (§85), and work per example the number of fired units, n·r^L before pruning and fewer after (§93).

*Proof sketch.* For each distractor j and role, Λ_j = log(w_target / w_j) is invariant under renormalization (§83(i)); by §91
its per-miss increments are bounded and have mean ≥ δ, so by Azuma's inequality Λ_j(M) ≥ δM − O(√(M·log(Q/ε))) for all j
simultaneously with probability ≥ 1 − ε. The role's target carries more than θ once Σ_j e^(−Λ_j) < (1 − θ)/θ, which holds when
every Λ_j exceeds log(Q·θ/(1 − θ)); solving for M gives the bound. Absorption and the false-fire bound are §84 and §83(ii). ∎

*Remark (why the ratio potential).* Tracking only −log w_target gives a valid but conservative condition: a promotion
that misses the target costs at most log(1 + αθ) and one that hits it gains at least log((1 + α)/(1 + αθ)), which
guarantees progress only for q < log((1 + α)/(1 + αθ)) / log(1 + α) ≈ 0.32 at α = 1, θ = 0.6. The ratio potential
counts only distractors actually co-promoted; the measured sweep (§91) learned up to q ≈ 0.62, as the ratio form predicts.

**Evidence.** Updates grow linearly in depth, not in the basis: depth 3 → 4 multiplies Q by ≈ 380 but the updates by ≈ 1.25
(E54); the dependence on trailing noise follows 1/(1 − q − 2f) (E91); one level too few cannot express the class
(E54 L = 1: 0.57–0.76), as realizability requires.

**What the theorem does not cover, and where depth goes next.** It is a theorem about *selecting* depth from an exhaustive,
activity-paid candidate basis. It does not cover *learning* intermediate representations outside such a basis: shared
codes, and the keys and queries of race attention (§96). That is the trainability question language needs; its first test
is E61 (a race-attention layer trained by local credit on associative recall and induction).
## 98. Scalability of training, and whether local credit is as good as backpropagation

*Written 2026-09-27.*

**(a) The mistake bound is optimal in the basis size (Littlestone).** Selecting one route among Q candidates is a class of
Littlestone dimension ≥ ⌊log₂ Q⌋ (a mistake tree of that depth exists: each example can split the remaining candidates in
half), so every online learner, local or not, can be forced to make Ω(log Q) = Ω(d·log P) mistakes on depth-d order
detectors. §97's O(log Q / δ) is therefore optimal up to the noise-dependent factor δ; depth costs mistakes linearly, and
no learner does better in the worst case.

**(b) Samples.** By online-to-batch conversion, a run with M mistakes over T examples yields a hypothesis with expected
error ≤ M/T, so reaching error ε needs T = O(d·log P / (δ·ε)) examples: linear in depth, logarithmic in the candidate basis.
(Measured: ≈ 10× fewer examples than a gradient-trained Transformer at depth 3; E94.)

**(c) Compute.** Training work = Σ over examples of (units fired) + Σ over mistakes of (units credited) ≈ T·c + M·s, with c the
activity per example (n·r^L before pruning, less after, §93) and s the size of a credited set. It is linear in the data
and independent of the candidate basis Q. Backpropagation costs ≈ 6·N per example for N parameters, all of which are
touched on every example.

**(d) Memory.** Synapses exist only for units ever credited (§85), at most M·s: *in a realizable task memory stops growing
once learning converges*, whatever the data size. In non-realizable, open-ended data (language) new useful contexts keep
appearing and memory grows as D^h (§95).

**(e) Parallel training.** Credit at a node reads and writes only that node's weights, and the multiplicative updates of a
node commute (products commute; renormalization is a common scale), so the log-weight changes computed by different workers
on different examples add exactly. The only coupling is that which routes get credited depends on the current weights: with
updates merged after a staleness of τ examples, the standard analysis of delayed online learning adds O(τ) to the mistake
bound. Training therefore parallelizes across nodes with no global barrier and across data with bounded staleness; it
needs no backward pass through the whole network, no stored activations, and no synchronized weight broadcast.

**(f) Is local credit as good as backpropagation?**
*Proposition (exact races: the gradient is the causal chain).* A race network computes a max-plus (tropical) function of
its delays: arrival times are minima of sums. For y = max_i (a_i + x_i), ∂y/∂a_j = 1[j = argmax] (a subgradient). Hence the
exact backpropagated (sub)gradient of the output with respect to any delay or weight is non-zero only along the critical
path, the chain of spikes that caused the output, and each node can tag which input triggered it. Credit that follows the
causal chain backwards *is* backpropagation for this algebra; nothing is lost by locality.
*The catch: losers get no gradient.* Exact backpropagation through a max gives no signal to the paths that lost, which is
why counterfactual credit from near misses (§57) is needed: it is the local counterpart of replacing the max by a softmax.
*Proposition (soft races: local exact gradients).* For a race of exponential clocks with rates λ_i, the winner w has
probability λ_w / Λ (Λ = Σ λ), and ∂ log P(w) / ∂ log λ_i = 1[i = w] − λ_i / Λ: each clock needs only its own rate and the
total, which the race itself provides. This is the likelihood-ratio gradient of one race, exact and local.
*What is open.* (i) Through deep stacks of stochastic races the likelihood-ratio estimator's variance grows with depth, so
local learning could need more examples than backpropagation through a continuous, reparameterized network; conserved
multiplicative updates have mistake bounds instead of variance, near-miss information acts as a variance reducer, and the
measured data efficiency points the other way (E94), but this is not yet a theorem for learned hidden representations.
(ii) For hidden units inside additive distributed codes, error must be delivered to units whose effect is spread across many
sums; in continuous time, local schemes provably match backpropagation (equilibrium propagation; predictive coding), which
suggests the obstacle is not locality itself. E61 (race attention learned by local credit) is the first test with learned
internal routing.
## 99. Scaling laws from representations: the approximation–estimation tradeoff, order codes, and merging by a race

*Written 2026-09-27, after E62's first scaling points (1M / 10M / 90M characters: 2.05 / 1.81 / 1.66 bpc with copy; a
three-point fit gives an exponent ≈ 0.19 and a floor ≈ 1.39 for counting and copying).*

**(a) Where the data exponent comes from.** Let a predictor map contexts to m internal states (a representation φ with m
values) and predict the next symbol from counts per state. Its excess log-loss over the source's entropy rate splits into
  approximation  a(m) = H(X | φ*(context)) − H∞  (the best m-state representation's lost information), and
  estimation     ≈ m·(|A| − 1)·log₂ D / (2D) bits per symbol  (minimum description length; context-tree weighting
                 attains it for tree sources).
*Proposition.* If a(m) ≈ c·m^−γ, choosing m per D gives m* ∝ (D / log D)^(1/(1+γ)) and
  L(D) − H∞ ∝ (D / log D)^(−γ/(1+γ)).
The data exponent β = γ/(1 + γ) is set by how fast representations of growing size approach the entropy rate. Exact
contexts of bounded order approach it slowly and have a floor at the order-K conditional entropy; a representation that
merges contexts with similar futures has a larger γ (fewer states lose less information), hence a better exponent and a
lower floor. Memory follows m*: S(D) ∝ D^(1/(1+γ)), so h = 1/(1 + γ) and β = 1 − h, the relation stated in §95(c), now
derived with h tied to the representation. (This is the event-model counterpart of explanations of neural scaling laws by
the intrinsic dimension of the data: there the exponent is set by how well a model of size N resolves the data manifold.)

**(b) Order codes.** k spikes on N channels, read as a set, carry log₂ C(N, k) bits; read as an ordered sequence,
log₂(N! / (N − k)!) ≈ k·log₂ N bits, about k·log₂ k more. Rank-order codes therefore offer more distinguishable states per
active unit, a larger m at equal activity, which by (a) is worth a better exponent if the extra states carry predictive
information.

**(c) Merging contexts by a race (native representation learning).** Code units u = 1..m each hold a predictive
distribution q_u over the next symbol. A context c is assigned to the code unit whose distribution best predicts c's
observed continuations: every code unit's clock runs at a rate increasing in its log-likelihood on c's counts, and the
first to fire claims c (a race; competitive learning). Assignment and re-estimation alternate (the likelihood form of
k-means, i.e. clustering in Kullback–Leibler divergence); each step does not increase the training loss of the merged
predictor, which converges to a local optimum. The code of a context is then used as a context itself (alone or joined
with the most recent characters), generalizing across histories that the counts show to be predictively alike.

**Predictions (E65).** Adding merged-context experts to the stage-2 mixture lowers test loss at every data size, most at
small data (where estimation dominates), and raises the fitted data exponent relative to counting alone.

**Results (E65, E65b; 1M training characters, 100k test): the prediction fails, and the failures locate what a
representation must do.** (1) Merging order-3 contexts by their continuations adds nothing (2.0295 vs 2.0282 bpc): there
the counts are not data-starved. Merging order-5 contexts gains 0.0025: an unseen context gets no code, because the code
was computed from continuations it does not have; generalization to unseen contexts needs codes computed from the context's
content. (2) A learned recursive state (automaton: state + character → state, one table lookup per event) computed from
content, learned by merging (state, character) pairs with similar next-character distributions, collapses toward bigram
quality (3.4 bpc; the number of states in use falls, e.g. 622 → 17): next-symbol merging discards every distinction that
matters only later, and hard likelihood clustering lets broad clusters absorb low-count items. Merging by the next two
characters does no better (3.3). Starting from the exact order-2 automaton (2.95 bpc), merging only loses information.
**Consequence.** A representation that beats exact contexts must also *split* states where earlier history matters and
merge only statistically indistinguishable ones (the split-merge reconstruction of causal states, with tests), and, for
text, must know that different symbols behave alike: distributional classes of words, which no operation on exact strings
produces. These are the next representation experiments; a(m) in (a) is only improved by them.
**First step that works (E66, one run at 1M characters, 100k test): keys at the unit where text repeats, with backoff inside
each expert.** Counts and copy memories keyed on (previous word, partial word) and (two previous words, partial word) are
sparse (112k distinct keys in 200k characters); alone they fall back to a uniform guess where unseen (3.7–4.5 bpc), and a
Hedge mixture cannot use them, because its weights are global per selector, not per position. With Witten–Bell backoff
inside the expert (two words → one word → partial word) they are informative everywhere (2.26 / 2.28 bpc alone) and the
mixture improves from 2.028 to 2.005 bpc. Principle: a sparse, specific representation must carry its own backoff (or the
mixer must weight by per-position confidence); scaling runs queued.
**Sleeping experts (E66, same run).** The principled per-position mixer is the sleeping-experts algorithm (Freund, Schapire,
Singer, Warmuth 1997): only awake experts (a copy memory that found a match; a context whose key was seen) vote, and after
each event awake experts are multiplied by (p_i / p_mix)^η, which conserves the awake experts' total weight; with fixed share
(tracking a switching best expert) it reaches 1.963 bpc against 2.005 for the windowed Hedge mixture and 2.028 for the base.
This is the native semantics of an event network: a detector that does not fire is asleep, neither votes nor learns.
## 100. Signals that carry representations: payload events, race attention on vectors, local updates

*Written 2026-09-27.*

**What an event can carry.** (i) Its *time*: a latency relative to a reference encodes a real number; delays add, first
arrival is a minimum, clock rates encode probabilities (races already compute with this). (ii) A *payload*: an event is a
message (source, time, vector) rather than a pulse; the receiver works only when a message arrives, but the message carries
a representation (an embedding). (iii) *Change-triggered payloads*: a unit emits its new vector only when it has changed by
more than a threshold (delta networks, event-driven recurrent units): dense in value, sparse in traffic.

**Race attention on vectors.** A query event carrying q and stored events carrying keys k_i race with rates exp(q·k_i / τ):
the first to fire is i with probability softmax(q·k/τ)_i (§96), now over learned real-valued representations. For the
winner w, ∂ log P(w) / ∂k_i = (1[i = w] − P(i)) · q/τ, and ∂ log P(w)/∂q = (k_w − Σ_i P(i) k_i)/τ, which needs only the
query, the competing keys' rates and the winner: a local update, computable at the race. Credit to the winner's value
payload is local as well. So embeddings (the representations that make "cat" and "dog" interchangeable, which exact-string
counting and merging cannot learn, §99) become learnable without a global backward pass.

**Cost accounting.** A payload of dimension d costs ≈ d operations per receiving synapse. The advantage then comes from
temporal and structural sparsity (which units emit, and when), not from signals being one bit; for language, a few context
units are active per character and each emits only on its events.

**Next experiment.** Word keys with learned payload vectors (instead of exact identity) for the copy and counting experts:
retrieval by vector similarity via the race, keys and queries updated by the local rule above.
## 101. Coupling time and representation: the identity–time decomposition of a race, local softmax gradients, and backpropagation in the time domain

*Written 2026-09-27; theorem with proof, consequences, and a test (E67).*

**Setting.** A race node receives candidates i with scores s_i computed from representations (payloads, §100; e.g.
s_i = q·k_i/τ) and fires exponential clocks with rates λ_i = exp(s_i). W is the winner, T the first-arrival time.

**Theorem (identity–time decomposition).** (1) P(W = i) = λ_i / Λ = softmax(s)_i, Λ = Σ_j λ_j. (2) T ~ Exponential(Λ), so
E[T] = 1/Λ and E[log(1/T)] = log Λ + γ = logsumexp(s) + γ (γ: Euler's constant). (3) W and T are independent.
*Proof.* P(W = i, T > t) = ∫_t^∞ λ_i e^(−λ_i u) Π_{j≠i} e^(−λ_j u) du = (λ_i/Λ)·e^(−Λt), which factorizes; the rest is the
exponential law's moments. ∎
The race thus splits its output into two independent channels: *which* (a sample of the softmax: a representation choice)
and *when* (the log-partition function: confidence). Downstream nodes receive the normalizer that a dense softmax computes
explicitly, for free, in the arrival time.

**Corollary 1 (each competitor knows its own probability).** E[λ_j T] = λ_j / Λ = p_j: the product of a node's own rate and
the observed decision time is an unbiased estimate of its softmax probability (variance p_j², relative variance 1 per race;
R races reduce it by R).

**Corollary 2 (exact cross-entropy learning from local quantities).** For target y, ∂ log p_y / ∂s_j = 1[j = y] − p_j, estimated
without bias by 1[j = y] − λ_j T: each node needs its own rate, the one decision time every node observes, and whether it is
the target. No normalization circuit, no access to competitors.

**Corollary 3 (time teaches representations).** With s_j = q·k_j/τ: Δk_j ∝ (1[j = y] − λ_j T)·q (local to key j: it holds k_j,
receives q), Δq ∝ Σ_j (1[j = y] − λ_j T)·k_j (sent back only by keys with non-negligible λ_j T: sparse). The timing of the race
is the learning signal for the payload vectors.

**Corollary 4 (backpropagation in the time domain).** dE[T]/ds_i = −λ_i/Λ² = −p_i·E[T], estimated locally by −λ_i T²/2
(E[λ_i T²] = 2λ_i/Λ²). A node told "fire δ earlier" (a timing error from downstream) converts it into score changes for its
inputs, Δs_i ∝ δ·λ_i T²/2, and scores are functions of payloads, so timing errors become representation updates and
representation errors become timing errors further down: learning propagates between the two domains through each race.
For deterministic time-to-first-spike nodes the same holds with the causal set: t_out = (Σ_{i∈C} w_i t_i + θ)/Σ_{i∈C} w_i
for non-leaky integrate-and-fire nodes, so ∂t_out/∂w_i = (t_i − t_out)/Σ_C w and ∂t_out/∂t_i = w_i/Σ_C w: local quantities
(the exact spike-time gradient of time-to-first-spike networks).

**Delivering error to hidden units.** The error originates locally (Corollary 2); reaching hidden units requires sending it
back along the forward synapses (their transpose) or, without weight symmetry, along fixed random feedback (feedback
alignment), which trains hidden layers in practice.

**Predictions (E67, MNIST).** (P1) One-layer race classifier trained by Corollary 2 (one race per example) matches exact
softmax gradient descent in test accuracy at equal examples (within 0.5 points), and beats a winner-only race perceptron.
(P2) Two layers, hidden payloads trained through the race error: close to backpropagation with symmetric feedback, and
still above one layer with random feedback.
## 102. Time-normalized race attention: soft attention in representation space, natively

*Written 2026-09-27, from §101.*

**Construction.** Stored events carry key and value payloads (k_j, v_j); a query event carries q. Each armed key runs an
exponential clock with rate λ_j = exp(q·k_j/τ + b_j), where b_j is a native relative-position term (a delay shifts the key's
arrival; a learned bias per lag). The first arrival fixes the decision time T, observed by every key. Each key emits its value
payload scaled by λ_j·T; the query node sums what arrives: o = Σ_j λ_j T v_j.

**Proposition.** E[o] = Σ_j softmax(q·k/τ + b)_j v_j, soft attention (by §101: E[λ_j T] = λ_j/Λ). With R independent races
(heads with shared weights, or replicas) the covariance of o falls as 1/R; per coordinate, Var(o_d) = Σ_j p_j² v_{jd}² + …
≤ max_j |v_j|² Σ_j p_j². Hard attention is the identity channel alone (the winner's value, §96); time-normalized attention
uses both channels: which keys, and when.

**Correction, 30 September 2026 (§295).** The expectation is valid, but the
variance bound above is not: all terms share T. Exactly o=(Lambda T) mu and
Cov(o)=mu mu^T, or mu mu^T/R after independent averaging. Equal unit values
have variance one even with many keys. Centering reduces noise (§296).
Winner-only delivery is a separate estimator.

**Sparse retrieval with a bounded bias.** Keys whose clocks have not fired by a cutoff c·T can be skipped (their λ_j T is
small): the bias of the truncated output is bounded by (skipped softmax mass) × max |v|, the error of top-k attention;
with sparse codes, a query reaches only keys sharing an active channel (§96), so the armed set is small to begin with.

**Scope correction (§295).** Missing a clock cutoff does not certify a small
rate or omitted mass. The mass-based bound needs an independently verified
candidate/omission bound, as in §112.

**Learning (local).** Keys and queries from the race's timing (§101 Corollary 3); values from the output error
(Δv_j ∝ λ_j T · ∂L/∂o, local to key j); multi-head = several races; a layer = attention races + threshold feed-forward
nodes on payloads; composition = stacking, with timing errors propagated by §101 Corollary 4.

**Why this makes subsumption direct.** Every Transformer head maps to a race whose output is a time-weighted sum of value
payloads, with the same expected output as softmax attention; the remaining differences are variance (reduced by R), the
native relative-time term, and the option of sparse retrieval. The open question is the same as before: training deep stacks
by these local rules as well as backpropagation trains Transformers.

**Design constraint: small payloads.** Inside a unit a payload is dense, synchronous arithmetic: a d-dimensional payload costs
≈ d operations per receiving synapse per event, so total work ≈ events × fan-out × d. The paradigm's advantage survives only
if sparsity and asynchrony hold *between* units while d stays small (≈ 8–32; also the scale of graded spikes in neuromorphic
hardware). Representational capacity must then come from the event structure: which units fire (log₂ C(M, k) bits for k of
M), in what order (≈ k·log₂ k more, §99b), plus k·d·b payload bits at b bits of precision; with many units firing sparsely,
which-and-when dominates. A cheap dense core per asynchronous unit, combinatorial capacity across units.
*E67 (MNIST, one seed, iterate-averaged over the last epoch): exact softmax learning 0.927, race-time learning with 4 races
per example 0.923 (Corollary 2 in practice); full grid queued.*

**Test (E68).** Associative recall with learned vector representations: queries and keys as learned embeddings (d = 16) of
token identities, a query–key map learned from the race timing, values read by time-normalized attention; against the
Transformer baseline of E61 and the discrete-route race attention of E61.
## 103. Race Transformers subsume Transformer training: the pathwise race gradient (theorem, proofs)

*Written 2026-09-27; completes §96 (expressiveness), §100–§102 (representations, time-normalized attention) with trainability.*

**Lemma (race attention is reparameterizable).** Fix exponential noise E_j ~ Exp(1) and let λ_j = exp(s_j), T = min_j E_j/λ_j with
winner w, and o = Σ_j λ_j T v_j (§102). For fixed noise, w is locally constant in s and T = E_w e^(−s_w), so almost everywhere
  ∂o/∂s_i = λ_i T v_i − 1[i = w]·o,     ∂o/∂v_i = λ_i T.
*Unbiasedness.* E[λ_i T v_i] = p_i v_i (E[T] = 1/Λ). By the independence of W and T (§101), o = TΛ·ō with ō = Σ_j p_j v_j, so
E[1[w = i]·o] = P(w = i)·E[TΛ]·ō = p_i ō. Hence E[∂o/∂s_i] = p_i (v_i − ō), the Jacobian of softmax attention, and
E[∂o/∂v_i] = p_i. ∎  Each key's term is local: its own rate, the decision time, whether it won, and the output o returned with
the error; the time channel carries the normalization in the backward pass too (the winner's −o).

**Theorem (subsumption of training).** Build a race Transformer: every softmax (attention heads, output layer) replaced by a
time-normalized race with R independent races; feed-forward blocks, normalization and the residual stream as small dense cores
inside units (§102). Then
(i) forward: each head's expected output is softmax attention, and each output race samples the softmax (§101);
(ii) backward: event message passing, error payloads sent back only along connections active in the forward pass, with the
     local pathwise rule of the Lemma at every race and the ordinary chain rule inside dense cores, computes an unbiased
     estimate of ∇ E[L_R], the gradient of the race network's expected loss (the chain rule composes; noise is fixed per step);
(iii) E[L_R] − L_Transformer = ½·tr(H·Cov(o)) + O(E|o − ō|³) = O(1/R) for a loss with bounded second and third derivatives,
     and likewise for gradients: training a race Transformer is stochastic gradient descent on the Transformer objective up to
     an O(1/R) bias and additional gradient noise of variance O(1/R);
(iv) backward traffic mirrors the active forward paths (keys with non-negligible λ_j T, units that fired), so it is as sparse
     as the forward pass; truncating keys that have not fired by a cutoff biases outputs and gradients by at most the skipped
     softmax mass times the payload bound;
(v) relative position is native (learnable delays or lag biases b_j on the scores, with ∂L/∂b_j = ∂L/∂s_j); heads are
     independent races; the output layer's cross-entropy gradient is the local estimate of §101 Corollary 2.
Together with §96 (every Transformer is expressible) this makes the Transformer family, including its training by gradient
descent, a limit (R → ∞) of race networks trained by local event-driven message passing.

**What the theorem leaves open.** The O(1/R) terms, and whether the sparse regime where the efficiency lives (few races, hard
or truncated retrieval, discrete routes as in §83–§89) keeps training close enough; the asynchronous overlap of forward and
backward passes across tokens adds staleness, bounded as in §98(e).

**Training scope clarification, 30 September 2026 (§295–296).** Unbiased head
outputs and Jacobians do not make the sampled nonlinear-loss gradient unbiased
for the deterministic Transformer objective. Claim (ii) concerns the continuous
race network's own expected loss, with regularity permitting differentiation
under expectation. Claims (iii)–(iv) additionally require smoothness throughout
composition and a verified omitted-mass bound; firing late alone is not such
a bound. Sparse hard-winner delivery requires a separate teacher, whose loss
gradient scope is stated explicitly in §296.

**Test (E68).** The same small Transformer (payload dimension 32, 2 layers, 4 heads) trained with softmax attention and with
race attention at R = 1, 4, 16 (pathwise gradients as in the Lemma), on associative recall with learned embeddings and on
character-level text: prediction, curves approach the softmax Transformer as R grows, with R = 4 already close.
