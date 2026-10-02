# Statistic-valued race memory: pooling, exact counterfactual credit and in-context statistics

[Theory index](../THEORY.md) · Global sections 382–387. Written 2026-10-02 on host `curie`.
Continues §§376–380 ([note 58](58_sufficient_statistic_state_and_count_references.md)).

Note 58 established that small-N language here is limited by estimation, and that
addressed sufficient statistics solve estimation exactly at `O(K)` work per
event. Counts alone cannot generalize across contexts. This note works out how
the substrate's own mechanisms (races, sparse addressed writes, key/value
separation and counterfactual credit) supply that generalization. It also
explains why counterfactual credit becomes *exact and cheap* in this
construction, where with neural receivers it is costly and noisy.

## 382. When pooling contexts helps: an exact code-length criterion

Two contexts with next-symbol laws `q₁, q₂` are visited `n₁, n₂` times. With
Dirichlet(α) values, the sequential code length of a count vector `c` is
`L(c) = −log[Γ(Aα)/Γ(n+Aα) · Πᵧ Γ(c_y+α)/Γ(α)]`, which does not depend on the
order of events. The **exact** gain from pooling the two addresses into one is
`G = L(c₁) + L(c₂) − L(c₁+c₂)`.

**Proposition 382.1.** For regular interior empirical multinomials with common alphabet size A,
nonvanishing sample proportions and fixed positive α, as `n₁, n₂ → ∞`,
`G = ((A−1)/2)·log(n₁n₂/(n₁+n₂)) − (n₁+n₂)·JS_π(q₁,q₂) + O(1)`, where
`JS_π` is the Jensen–Shannon divergence weighted by `π_i = n_i/(n₁+n₂)`. For comparable visit counts this gives the leading-order threshold
`JS_π ≲ (A−1)·log n/(2n)`. Near that threshold, the exact finite-count G
determines the sign; sparse/boundary supports need different prior-dependent
corrections rather than silently replacing A by A_eff.

*Sketch.* Laplace expansion of each `L` gives `nH(q̂) + ((A−1)/2) log n + O(1)`.
Then use `(n₁+n₂)H(q̄) − n₁H(q₁) − n₂H(q₂) = (n₁+n₂)JS_π`. ∎

*Check* (equal visits, `A = 27`, 60 replicates; exact `G` against the
asymptotic form): the predicted sign change falls within one decade of the
observed one for `JS = .002, .018` and `.041`. Example: at `JS = .018`, `G` is
+18 nats at n = 200 and −23 nats at n = 2,000. The fixed offset of tens of nats
comes from rare symbols, an O(A) Laplace correction.

Consequences:

1. **Generalization has a price, set by evidence.** Pooling is worth exactly the
   estimation cost it saves, so its value concentrates in rarely visited
   contexts.
2. **Capacity should grow with data at constant activity.** The optimal
   partition refines as counts grow. The number of addresses rises with `D`,
   while each event still touches one address per level. This is the
   thesis's "capacity beyond activity" with an explicit optimality criterion
   instead of a hope.
3. **The escape mass marks where generalization pays.** Events with high
   escape mass `e_h` sit in low-evidence contexts, and those are exactly the
   events where §379's learned base receives credit. Suffix backoff is one fixed
   pooling rule, sharing statistics between contexts with a common suffix. A
   learned similarity can pool better: that is part 2 of the supremacy map in
   §380.

## 383. Learned keys, statistic values: exact counterfactual route credit

**Construction (statistic-valued race memory).** A learned stream model (the
native eight-block model) emits a query `u_t`. Receivers `a` have learned keys
`k_a` and **values that are sufficient statistics** `(c_a, n_a)`, not learned
vectors. A sparse candidate set `B_t` comes from addressed shortlists or
product keys. Race clocks with rates `exp(u_t·k_a/τ)` select a winner. The
winner's predictive `p_a(y)` comes from its counts through the escape cascade
(§378), backing off to suffix counts and finally to the learned base `q_θ`
(§379). After scoring, the winner's counts receive the event. This sparse
write also happens during evaluation, as persistent state with frozen weights.

**Proposition 383.1 (zero-variance delivery credit).** For fixed statistics,
`ℓ_a = −log p_a(y)` is a lookup for **every** candidate `a ∈ B_t`, not only the
winner. The expected race loss `E_{a∼π}[ℓ_a]` then has the exact gradient
`Σ_{a∈B_t} π_a(ℓ_a − ℓ̄)∇log π_a`, which costs `|B_t|` lookups and needs no
sampling. A distinct objective, the mixture objective `−log Σ_a π_a p_a(y)` gives
posterior-responsibility weights.

With vector-valued neural receivers, evaluating a losing route means running
that route's computation forward (and backward for deep credit), so the
project's signature mechanism costs a forward pass per alternative
(§§19, 57, 373). With statistic values, the **current delivery** counterfactual is exact and costs candidate lookups.
This removes winner-sampling noise at fixed counts. Future writes change
subsequent state and routes, so this is not exact full-sequence credit.
It makes local hard-routing credit inexpensive when shortlisted values are cheap, and
minimizing the summed sequential code length over routes is online MDL
clustering of contexts (§382).

**Proposition 383.2 (closed-form write credit).** Let receiver `a` hold counts
`c` (total `n`), and let its future events follow `q`. The expected change in
log predictive for the next event, after writing symbol `y` to `a`, is
exactly `q_y·log(1 + 1/(c_y+α)) − log(1 + 1/(n+Aα))`. To first order it is
positive if and only if `q_y > (c_y+α)/(n+Aα) = s_a(y)`: **A write helps a
receiver to first order when it underpredicts the written symbol relative to
its future law.** The exact threshold is
`q_y > log(1+1/(n+Aα))/log(1+1/(c_y+α))`; it generally differs from s_a(y).

*Check.* Over 2,000 random cases the identity holds to 3×10⁻¹⁶. The first-order
sign rule agrees in 99.9% of 1,148 cases with `|q_y − s_y| > .01`.

This gives the "persistent commit" component that the AWS factorial audit
(§373, ROUTE_WRITE_DIAGNOSTIC) found dominant (mean |commit effect| .0127 versus
.0003 value-linearization residual). For statistic-valued receivers it is
analytic for **one next visit with a supplied fixed future law**. It does
not equal the factorial audit's full suffix intervention. Estimating the future
law, crediting later visits and accounting for changed routing remain work. The unknown
`q` is estimated by the receiver's next-visit predictive, by held-out next
visits, or by the router itself.

**Prior work.** Learned keys with empirical next-token values resemble kNN-LM
(Khandelwal et al. 2020) and Memorizing Transformers (Wu et al. 2022). Large
sparse key memories follow product-key memory (Lample et al. 2019). Pooling
contexts by likelihood is Brown / exchange clustering (Brown et al. 1992;
Kneser & Ney 1993). Hierarchical Dirichlet/Pitman–Yor values come from Teh
(2006). What is new here is the composition: hard race routing whose
counterfactual credit is exact because values are closed-form; escape cascades
as the delivery mechanism; analytic write credit; and streaming writes at
inference, all inside the event substrate.

## 384. In-context statistics without attention

Counting over the development stream alone gives 2.884 bpc (§376), so in-context
statistics carry substantial information. Transformers acquire in-context
n-gram estimation by learning dedicated heads, with per-token work that grows
with the attended context, `O(T·d)` per head. Addressed counts acquire the same
statistic with `O(K)` updates per event and memory proportional to the number
of distinct observed contexts, with no learning. Time-decayed counts
(Corollary377.3) add recency through touched-address time differences and
decay arithmetic, which must also be charged. A trained Transformer can
learn the same function, so this is not an expressivity separation. It is a
**work separation for a specific, ubiquitous computation**: Bayes-optimal
in-context k-gram prediction.

A clean demonstration task, scoped as a mechanism test rather than a general
benchmark: sequences drawn from fresh random order-k Markov chains, where the
Bayes predictor is in-context Dirichlet counting. The count-carrying receiver is
Bayes-optimal without training. Controls must learn it. Report quality against
both training and per-token inference work. The supremacy-relevant version is
text, where these in-context statistics combine with learned generalization
(§383).

## 385. Learning-work scaling: the quantitative supremacy argument

A common dense matrix-work approximation is `≈6·P·D` for `P` parameters
and `D` tokens; attention, embedding sparsity, sequence state and
batch-amortized optimizer work need separate accounting. In the statistic-valued construction the per-event
work splits three ways:

- **counts:** `O(K + |B|)` lookups and increments; no backward pass;
- **learned forward pass:** needed for every event, unless the top escape mass
  `e_h < ε`, in which case skipping it changes the prediction by at most `ε` in
  total variation (379.2b);
- **learned backward pass:** weighted by the responsibility `r_y`, and skippable
  when `r_y < ε` with current-logit gradient error at most `ε√2` (379.2a); parameter Jacobians
  and future state credit are additional terms.

With a learned base of `P` parameters, the training work is therefore about
`D·[c_K + 2P·(1 − f_e(ε)) + 4P·(1 − f_r(ε))]`. Here `f_e` and `f_r` are the
fractions of events below the escape and responsibility thresholds. Both are
measured as functions of `D` (§§379, 381). They are the substrate's
data-dependent sparsity. A dense model has `f = 0` at every scale. Dense
models learn shared maps for frequent contexts; counters pay table writes and
storage instead. No universal dense-model lower bound follows. This is a **projected predictive-only cost model**, requiring a base whose
state advance and future credit can remain correct when local prediction work
is skipped. Add recurrence, candidate discovery, storage/traffic, all losing
lookups and batch-amortized optimizer work. Small responsibility alone does
not remove a recurrent backward pass. The open question is whether the learned base, freed from that
work, closes parts 2 and 3 of the gap at a smaller `P·D`. That needs the
integrated fit of §380, then AWS-matched dense controls. It is not assumed
here.

## 386. The memorization tax of dense models

At 90M fitting characters, the context tables of orders ≤ 7 hold 17,144,249
distinct (context, symbol) entries; at 10M they hold 4,820,474 (§381 result
file, `table_pairs_by_order`). A dense predictor may compress or share the predictive structure of those
entries rather than store each entry. Their raw count is not an information
lower bound on the predictor. Language models have been measured to
store about 2 bits of knowledge per parameter (Allen-Zhu & Li 2024,
[*Physics of Language Models3.3*](https://arxiv.org/abs/2404.05405)). That is an empirical rate for factual
knowledge, used here only as an order-of-magnitude assumption.

**Estimate 386.1.** Suppose each table entry carries between 1 and 4 useful
bits (presence plus a coarse count). Storing the order-≤7 statistic of 90M
characters then needs about 17–69 Mbit, or about 9–34M parameters at 2
bits/parameter. The90M Transformer control has3.24M parameters and the LSTM1.20M. The
9–34M scenario is3–10× the Transformer size, **conditional on independent
useful entry bits and transfer of the factual-capacity rate**. Neither
assumption is measured for these text8 contexts. Control/count proximity
(1.604/1.661 versus1.653) therefore motivates a capacity sweep; it does not
establish that these controls are memorization-bound or insufficiently large.

**Consequence.** Training work for a dense model is `≈ 6PD` per pass, and `P`
may grow with irreducible predictive information. Raw table cardinality alone
does not determine that information. Addressed counters hold the same
statistic with `O(K)` integer increments per event and no gradient. The
**memorization-tax hypothesis** is extra learned parameter work spent representing statistics that counters hold with charged integer operations
and memory. In the composition of
§§379, 383 the learned base is relieved of the tax and needs capacity only
for generalization and retrieval. This is the substrate's most concrete
route to quality per unit of training work. It is an architectural argument
about a dominant term, not a claim that counts generalize.

**Predictions, falsifiable with saved or cheap fits:**

1. *Frozen composition.* Composing a frozen small dense control (as base
   `q`) with count receivers by the §378 cascade should improve its test
   bpc by much more than the same composition improves a larger dense model
   of equal data. The gain should shrink as `P` grows past roughly
   `17–69 Mbit / 2 bits`. This needs saved AWS checkpoints, scored on the E64
   test segment with fit counts and prequential test counts; no training.
2. *Iso-quality parameter ratio.* For a target bpc between 1.5 and 1.3, the
   count-carrying model should reach it with a learned base several times
   smaller than a dense-only model trained on the same data. Report the full
   charged FLOPs and the count increments separately.
3. *Learned work shrinks with data.* In a fitted count-carrying model, the
   measured responsibility-weighted backward work per target should fall
   with `D` roughly as in the §381 schedule. A dense model's per-target work is
   constant.

If prediction 1 fails, meaning the gains do not depend on `P`, then this particular
capacity explanation of the proximity in §381 is not supported, and this route to an advantage is
weaker than argued here.

## 387. The learned base learns the residual distribution

Fix one context with true law `P`, count part `a` (where `Σa = 1 − e`) and escape
mass `e`. Minimize `−Σ_y P_y log(a_y + e q_y)` over the simplex.

**Proposition 387.1.** The optimum is the water-filled residual
`q*_y = max(P_y/λ − a_y/e, 0)`, with `λ` fixed by `Σq* = 1`. When `P_y ≥ a_y` for
every `y`, there is no clipping and `q* = (P − a)/e` exactly: the base learns
precisely the probability mass that the discounted counts fail to explain.

*Proof.* The KKT stationarity condition is `e P_y/(a_y + e q_y) = λ` on the
support of `q`; summing gives `λ = e` when nothing is clipped. ∎ Checked with
L-BFGS on softmax logits for five random `(P, a, e)` triples. Agreement with the
water-filling solution is within optimizer tolerance (≤ 9×10⁻⁴; the largest
deviations are at coordinates driven to zero).

Consequences:

1. **Division of labor is automatic.** Combined with responsibility-gated
   credit (379.1), the learned model receives gradient only where counts fall
   short, and its target is that shortfall. It is not asked to relearn the
   frequent statistics. This is the mechanism behind the memorization-tax
   relief of §386.
2. **The target moves with the counts.** The residual depends on the count
   state. Training on leave-one-out counts of the fitting stream and evaluating
   with fit-plus-prefix counts is consistent in expectation. But the
   residual target shifts as tables grow, so a base fitted at one `N`
   transfers imperfectly to another. Refit or condition the base on evidence
   features (`n_h`, `T_h`) when tables change scale.
3. **Escape modulation is the right learned gate.** Clipping occurs where the
   counts overpredict (`a_y > P_y`). The learned model can fix that only by
   raising the escape mass, that is, through a learned discount or
   concentration that depends on its state. This motivates making `θ_k` and
   `D_k` functions of the native state, not only scalars, with the work bounds
   of §379 applied to the learned escape.


**Review scope, 2 October.** Delivery enumeration is exact at fixed counts;
write gain is exact for one next visit with a supplied future law. The
whole-sequence gradient and executed skipping savings require further
contracts. The memorization estimate is a conditional scenario rather than a
parameter lower bound. Measured quality gaps and the proposed integrated test
are retained. See the state/credit prerequisites in the companion AWS note.

## 388. Equivalent count order: how much local context a learned model uses

Map a fitted model's development bpc onto the frozen-KN curve over context
order at the same `N`, interpolating linearly between neighbouring orders.
The result is an *equivalent count order*: the suffix length a counting model
needs to match the fitted model.

| Fitted model | Fit chars | Dev bpc | Equivalent KN order | Count-optimal order |
|---|---:|---:|---:|---:|
| native H2/d16/depth8 | 2,048 | 3.765 | 0.9 | 4 |
| best episodic (write credit) | 2,048 | 3.722 | 1.0 | 4 |
| native H2/d16/depth8 | 8,192 | 3.557 | 1.0 | 4 |
| episodic receiver, semantic | 8,192 | 3.311 | 2.3 | 4 |
| full sparse p32 | 32,768 | 3.106 | 2.2 | 5 |
| input-gated carrier w256 | 131,072 | 2.572 | 3.1 | 5 |
| input-gated carrier w128 | 1,048,576 | 2.210 | 3.6 | 6 |

**Readings.**

1. The native model at 2K and 8K is a **bigram-level predictor**: it has learned
   previous-character statistics and essentially nothing longer. Every learned
   model uses less local context than counts of the same data, and the shortfall
   persists to 1M. The fitted models are not yet limited by long-range
   modelling. They are not exploiting 3–6-character contexts that a table
   captures exactly.
2. This is consistent with the 2K outcome (§380): the cascade already contains order-1
   counts, so a bigram-level base supplies nothing beyond them. By §387 the
   base's target is the residual beyond order-4 counts, and a 2K stream holds
   almost no learnable residual.
3. **Prediction** for the admitted 8K count-carrying fit: the native base
   is still equivalent to order 1.0, so training it should add at most about
   .01 bpc over the untrained composition. The composition's advantage over
   native-alone comes from the counts.
4. **Where learning must contribute.** In a count-carrying model, a learned
   base helps only where its standalone predictive beats the count backoff on
   low-evidence contexts. That requires equivalent orders well above those
   that counts estimate reliably at that `N`, or information counts cannot
   represent (generalization, retrieval). Equivalent order is therefore a
   cheap acceptance metric for any learned component. It should rise with `N`
   faster than the count-optimal order before composition can pay off.

## 389. Why a composed learned base adds nothing: it cannot see the counts it must complement

**Measured failure (131K, seed 6, development only).** Count receivers (K5) over
the input-gated carrier reach 2.313 dev bpc at both w32 and w128, identical to
three decimals at every epoch, with identical learned escape parameters, at
12.7× different fitting work. Alone, the same carriers differ by .26 bpc (2.848
vs 2.587). An untrained random base already composes to 2.371. Scored alone
from their saved composed checkpoints, the trained bases give **8.17 and 11.34
bpc**, worse than uniform (4.75). The width-insensitivity is therefore not a
saturation of useful learning. The base has stopped being a predictor at all.

**Derivation.** By §387.1 the optimal base at one context is the water-filled
residual `q*_y ∝ max(P_y/λ − a_y/e, 0)`. It depends on the count part `a` (which
symbols the tables already cover, and how much) and on the escape mass `e`. The
base network receives only the token stream, never `a` or `e`. Its best achievable
output is therefore the conditional expectation of the residual given the
stream features it can compute. In practice it learns the *average*
residual: mass on symbols that are novel continuations of their contexts,
anti-correlated with the ordinary predictive. Alone this is a terrible
predictor, which explains the 8–11 bpc. It cannot become context-specific
without re-deriving the counts internally, which is exactly the memorization
work the composition was meant to remove (§386). Wider bases fit the same
average residual, so width does not matter. Responsibility-gated credit
(§379.1) is correct but does not fix this. It routes gradient to the escape
branch without supplying the information the residual depends on.

**What it says about learned features.** The learned models were already
below the count-optimal order alone (§388: equivalent order 0.9–3.6 against
4–6). In composition they are not even asked to learn longer-range features.
Their loss is dominated by an unknowable residual of the local tables. Two
distinct deficits follow, and they need different repairs:

1. *Information path (this section).* The base must condition on the count
   state it complements. Make the count receivers deliver into the base: base
   logits `z' = z + Σ_k [log1p(c_k) W_k + 1[c_k>0] V_k]`, per-order `A×A` maps,
   zero-initialized so the untrained model nests the scalar cascade exactly.
   This is the thesis's "small messages mixing incoming content with persistent
   memory": the addressed statistic is a message to the learned receiver. It
   adds 2KA² parameters and O(KA²) work per target, with no dense history.
   It retains races, sparse addressed state, counts and credit, and removes
   nothing.
2. *Escape gate (§387.3).* Even a correct residual cannot override a
   confidently wrong table, where `a_y > P_y`, without raising the escape.
   Make `D_k, θ_k` per-position functions of base-predictive and evidence
   features (zero-initialized, nested).
3. *Learning horizon (open).* Information beyond `K` characters (words, topic,
   copy) is what a learned component must add once (1)–(2) hold. Its credit is
   truncated at the chunk length. Equivalent order rising above count-optimal
   remains the acceptance metric (§388.4).

**Predictions and test.** (P389a) With the count-conditioned base, composed
bpc improves with width: w128 beats w32 by ≥ .02 at 131K. Under the scalar
cascade the gap is 0. (P389b) The trained base's standalone bpc stays below
uniform. (P389c) The escape gate alone, without (1), gives < .02. Run order:
contracts, then the 2K native integrated fit (compare 2.734), then the carrier
w32/w128 diagnostic. Labelled diagnostics stay labelled; the integrated native
receiver is the promoted target.


## 389.1. Review corrections and discriminating tests, 2 October, 03:02 UTC

The measurements above are retained. Several interpretations need narrower
claims; these corrections govern their use in the report and new experiments.

1. **Equivalent count order is a quality calibration.** The §388 heading and
   reading1 inferred context dependence from a scalar loss. Equal mean cross
   entropy does not imply equal conditional predictions or a bigram-only
   representation. The completed frozen native8K audit changes predictions by
   .02252nats meanKL when removing history older than16 on32 positions; hence
   strict bigram-only dependence is false in that audit. This slice does not
   establish a useful benefit beyond16 or semantic/hierarchical abstraction.
   The score table and its quality shortfall remain valid.
2. **Standalone q need not beat count backoff to improve a composition.**
   Reading388.4's necessity claim is too strong: §387.1 itself supplies residual
   distributions with poor standalone cross entropy but improved combined
   predictive probability. Equivalent order may be a diagnostic, not an
   acceptance contract for a residual-trained base. Keep fixed-escape q
   replacement and combined quality/work as the actual tests. The reported
   8.17/11.34 base-only scores remain observations with unreplicated saved
   checkpoint provenance here; they do not prove the base stopped learning.
3. **Hidden count state changes the optimization problem.** §389's assertion
   that the optimal base is the arithmetic conditional expectation of each
   water-filled residual does not generally follow. For base features X, the
   active-coordinate condition instead is

       E[P_y * e / (a_y + e*q_y) | X] = lambda.

   Conditional expectation and nonlinear minimization do not commute. Missing
   count conditioning is a plausible information limitation, not an identified
   unique cause of the measured width failure. A stream model can retain or
   reconstruct some evidence. CountMessage and EscapeGate remain reasonable
   nested interventions; their fits must test this hypothesis independently.
4. **Gradient reach and usefulness differ.** The guarded initialization audit
   verifies responsibility-scaled logit gradients (max error1.86e-9) and
   nonzero gradients through every content layer. Saved composed w32/w128/w256
   diagnostics show nonzero clock-weight changes in layers1–5 (norm ranges
   .968–1.211/1.741–1.898/2.067–2.221). Only the unused final clock is unchanged.
   An earlier local diagnosis misread these diagnostics as all-zero clocks;
   that claim is withdrawn. No general disconnected-gradient or clock-learning
   regression is established. Parameter movement alone does not prove deeper
   useful features.
5. **Repair output is not recurrent depth evidence.** CountMessage delivers
   statistics additively into output logits; a gain there can arise at the
   readout without richer recurrent state. P389b must distinguish the bare
   backbone q from q conditioned by CountMessage. The current `base_only`
   driver measures the bare backbone before CountMessage. Both definitions
   should be reported before using standalone quality to accept the repair.

The existing scope remains: native temporal/sparse addressed state and
counterfactual producer credit are the core; carrier composition is a labelled
diagnostic. None of these corrections replaces temporal computation with a
dense model. The pending full-depth repair optimizer contracts retain zero
parent nesting and exact recovery before any new integrated fit. A separate
frozen native16/32/64 credit-horizon audit holds context, targets and race noise
fixed, so extra memory is separated from extra graph reach. It measures missing
surrogate credit, not fitting quality or unbiasedness. Resource guards and all
losing-value/optimizer costs remain in force. See
[the local evidence and planned comparisons](../LANGUAGE_LEARNING_DIAGNOSIS_20261002.md).

**Result and revision, 2 October 03:20 UTC (native 2K, seed 6, dev only; scalar cascade 2.734 bpc / 3.753
whole-fit GFLOPs, untrained 2.741).** Both repairs 2.703 / 4.053; count message only 2.744 / 4.032; escape
gate only **2.695 / 3.824**. The gate alone gains .039 for 1.9% more counted fitting work, so P389c (gate
alone < .02) is **falsified**. The message alone does not learn (2.746→2.744 over four passes) and slightly
hurts in combination. P389b also failed: the bases alone score 5.05 (both), 4.61 (message) and 4.92 (gate)
against uniform 4.75. As posed it was mis-specified. §387.1 makes the optimal base a residual, which alone may
score worse than uniform, so this is not evidence of collapse.

Revised reading: at 2K the binding constraint is the *override path* (§387.3), not the information path. A
scalar escape forces the same trust in a table regardless of what the native state predicts. Letting the
state and the evidence features set `D_k, θ_k` per position lets learning act where tables are confidently
wrong. The information-path argument may still matter where residuals are context-specific at larger N.
The 131K carrier test therefore runs gate-only and both-repairs side by side. The integrated promotion
candidate is the gate-only native receiver, next at 8K.

**Attribution, 2 October 05:10 UTC: the gate gains are learned count smoothing, not temporal-core
learning.** Minimal-core controls with identical cascades (seed 6, dev only; unit-special whole-fit GFLOPs):

| Setting | Full core | Minimal core | Work ratio |
|---|---:|---:|---:|
| Native + gate, 2K | 2.695 (3.824) | 2.694 (0.072; p2/d1, 467 params) | 53× |
| Native + gate, 8K | 2.601 (15.31) | **2.588** (0.288) | 53× |
| Carrier + gate, 131K | 2.136 w32 / 2.130 w128 | 2.178 (3.6; w2/d1, 232 params) | 23× / 290× |
| Carrier + gate + message, 131K | 2.127 w32 / 2.129 w128 | **2.124** (20.5; 7.5K params) | 5× / 52× |

A near-empty base with a per-position learned escape and per-order `A×A` count maps beats KN (2.349)
and stream-adaptive counts (2.346) by .22 bpc at 131K. That is a useful, cheap, causal learned smoother,
and it belongs to the sufficient-statistic memory (§§377–379). It is not evidence for the temporal core.
At 2K–8K the native core adds nothing over a minimal core, and at 131K the carrier adds at most what
the count maps already supply. With the gate, the base predictive works only as a feature of local
smoothing.

**Consequence for the research direction.** The binding problem is the one §388 named: learned components
do not yet capture information beyond the count orders (equivalent order below count-optimal). The
count tables now set a high and cheap floor, so any temporal, retrieval or word-level mechanism must
beat the *minimal-core* composition, not the scalar one. That is the acceptance control from now on. The
full-core 32K run, now in progress, has its matched minimal-core control queued. A core contribution
must appear as full < minimal by ≥ .02. Mechanisms aimed at the gap, all to be tested against this
control: longer credit horizons; content retrieval (copy/induction over persistent addressed state,
§384); and learned keys pooling contexts by evidence (§382), so the core supplies generalization the
counts cannot.


**Next integrated credit test, 2 October06:00 UTC.** The full/minimal2K64-credit
pair retains the current native/count-gate construction exactly.64 rather than
16 targets of graph reach allows later losses to teach producer states within
the same64-target Adam window; forward persistent capacity and optimizer
exposure are held fixed. The four-arm full/minimal ×16/64 comparison measures
whether this helps the core beyond local count smoothing. Complete-step work
and actual64-credit optimizer/cursor recovery precede the fits. A≥.02bpc
full-core gain must also beat the matching minimal64 arm by≥.02bpc before
nomination; improvement over scalar counts alone is insufficient. This remains
an exploratory single-seed2K fit, with no core substitution or automatic scale-up.

## 390. Can the temporal core learn deep features? Obstacles, a mechanism gap, and the test

**No impossibility.** Delays, races and persistent addressed state can implement finite automata, gated linear
recurrences and (via §96) exact softmax retrieval. Counterfactual route credit is unbiased for one hard
choice. Nothing forbids learning features beyond local counts. The attribution result (§389) therefore
says something about the *configuration and protocol*, and three separable obstacles explain it:

1. **Sample complexity.** Character models overtake well-smoothed counts only at millions of characters.
   At 2K–32K little learnable structure lies beyond local statistics, so minimal ≈ full is expected for
   any architecture. Real text is a weak test below the crossover.
2. **Memory timescale and credit reach.** History crosses events by two paths in the native core.
   (a) The previous event's top-layer output re-enters the next event through gate + layer norm and all
   `depth` layers again: a gradient k events back crosses ≈ depth·k nonlinear layers. (b) Per-unit
   addressed memories `m ← rotate(m·e^{−age·rate·forget}) + write·W x` form a linear rotating recurrence
   with initial timescales 1–100 events, a good gradient path. Credit is truncated at 16 events, and the
   frozen 8K audit finds the 16-event gradient 31% (norm) away from the 64-event one.
3. **Storage that is addressable by content.** The native core's `pool = 2` units per head and its disabled
   episodic index (`matching = recent = 0`) give a handful of 16-dim superposed memories per layer. Induction
   over W stored pairs needs W separately retrievable items; a superposed memory of dimension d recovers
   O(d) of them at best, with interference. **The thesis's race attention over stored keys/values is
   absent from the native core.** It exists in ParallelHeadRaceLanguageModel: a per-position KV bank, bounded
   hashed and recent candidates, a hard race with counterfactual credit. This is a mechanism-coverage gap,
   not a limit of the family.

**Sparse exact credit is a structural advantage, still untested.** Exact forward-mode credit (RTRL)
costs O(n⁴) per step for a dense n-unit recurrence. With addressed updates, the sensitivity of a
memory entry is nonzero only for parameters of units that wrote it. Sparse forward credit (SnAp,
Menick et al. 2020) then scales with active writes rather than total state. That is what "capacity
beyond activity" means for learning work. It is a candidate repair for obstacle 2 that dense recurrent
models cannot match at equal cost. A cost/error contract must come before any fit.

**Test (queued: curie_long_range_core_20261002T072000Z).** Synthetic streams with i.i.d. fillers, where
count tables of every order are at chance on targets (4.585 bits; contract-tested): lag-L copy (timing
memory) and induction over W (content-addressed recall). The unchanged native core runs at L = 12/48 and
W = 64, with 16- versus 64-event credit; the KV arm adds race retrieval at equal budget.

Predictions: (P390a) native learns lag-12 but not lag-48 at 16-event credit; (P390b) 64-event credit
improves lag-48 target bpc by ≥ .2 bits; (P390c) native fails induction-64 (within .3 bits of chance)
at either credit; (P390d) the KV arm reduces induction-64 target bpc by ≥ 1 bit. If P390c and P390d hold,
race retrieval is restored to the integrated native receiver as a nested option. That keeps addressed
units, races and counterfactual credit, adds KV candidate scoring, value-delivery and teacher work
(charged), and is tested against the minimal-core count composition on text at ≥ 1M characters. If the
KV arm also fails, credit through retrieval (variance, reach) is the next diagnosis, not a dense substitute.

## 391. Dilated delay taps: WaveNet's receptive field as literal event delay

**Failure addressed.** §390 obstacle 2: in the native core a dependency k events back crosses about
depth·k nonlinear layers through the per-event context path. Credit and signal both decay along it.

**Construction.** At layer d the channel mix also reads that layer's own input from τ_d events earlier:
`mixed_d(t) = W_d x_d(t) + U_d x_d(t − τ_d)`, with `τ_d = softplus(r_d)` initialized to 2^d and read by
linear interpolation between buffered events, so τ_d learns. With depth L, a dependency up to 2^L − 1
events back is reachable through at most L taps (binary decomposition of the lag), as in WaveNet's dilated
causal convolutions (van den Oord et al. 2016). Here the dilation is a learned delay in event time, not a
fixed index offset, and it runs in the asynchronous stream.

**Retained / added / removed.** Races, addressed units, persistent rotating memories, counterfactual route
credit and the context path are all retained; nothing is removed. Added: one d×d product per layer per event,
and a bounded buffer of recent layer inputs (capacity 2^L + 2 vectors per layer; 258 × 32 floats per layer at
depth 8). Inference work stays O(L) per event, independent of history length. `U_d = 0` at
initialization, so the model equals the native core exactly (contract-tested, bitwise).

**Limits.** Taps are position-addressed. They cover lag, periodic and hierarchical timing, not content
recall at arbitrary distance (induction), which needs race retrieval (§390.3). Credit through a tap reaches
only events inside the current credit chunk, because buffered entries are detached at chunk boundaries. The
benefit therefore needs a 64-event chunk, or sparse forward credit (§390), for lags beyond 16.

**Predictions.** (P391a) Tapped lag-48 at 64-event credit improves target bpc by ≥ 1 bit over the native
core at the same credit. (P391b) At 16-event credit the gain is < .3 bits: there is no credit across the
boundary. (P391c) Taps do not solve induction-64 (within .3 bits of the native core). If P391a holds and
P390d (KV solves induction) holds, the integrated receiver combines taps and race retrieval. Next tests are
then text at ≥ 1M characters against the minimal-core count composition, and sparse forward credit to remove
the chunk limit.

**§390 addendum: the native language core has almost no addressed capacity.** For language the native
receiver has one source address (one conversation stream). Every token therefore races the same
`pool = 2` units per head per layer, and total persistent state is depth × heads × pool × payload =
8 × 2 × 2 × 16 = 512 floats. The thesis's capacity beyond activity, many addressed states of which
few are touched per event, is not realized in this configuration. Count tables realize it trivially:
thousands of context addresses, five touched per target. That explains why counts dominate and why a
minimal core loses nothing (§389 attribution). The earlier receiver/KV models address unit pools by
token identity (`vocabulary` pools per head and layer); the native adapter dropped that to keep source
IDs non-semantic.

Repair, consistent with §§382–383: address persistent units by a **learned key of recent context**
(a hashed or race-selected key over the suffix), so state grows with distinct contexts while each event
updates O(1) addresses. Values stay learned vectors, or carry sufficient statistics where exact (§383). This
is the learned generalization of the count table, and the place where deep features could beat it: keys
that pool contexts by predictive similarity (§382) give generalization that exact-string tables cannot.
It is a substantive change to the native receiver. It needs a nested contract (one address reproduces the
current model) and an integrated fit against the minimal-core count composition before any long run.

**2 October review of §§390–391, preserving their hypotheses.** The512-float
count covers receiver content only: a fully occupied native H2/d16/L8 state
also carries32 top-context floats and34 float64 arrival timestamps,2448 tensor
bytes total before Python metadata. This is still a single conversation address,
but "512 floats total persistent state" is not complete accounting. The fixed
context-hash prototype does not implement learned keys or predictive-similarity
pooling. Added slots/buffers and their graphs now appear in state diagnostics.
The synthetic all-orders chance claim lacks its stated proof (lag copies cues,
induction samples observed keys); the old queue was retired before launch.
P391b remains a falsifiable prediction: detached old tap values can train a
current reader even when their producer credit stops at a chunk boundary, so
absence of that credit does not mathematically require a small gain. Longer
credit alone failed the completed2K quality gate; full recurrent content helps
one frozen slice but not yet whole-development quality versus minimal.
Read [deep feature bottlenecks](62_deep_feature_bottlenecks.md) for measured
responsibilities, the Adam scaling caveat, retained mechanisms and required
matched comparisons. These qualifications do not retract the core principles
or replace completed earlier positive/negative evidence.

## 392. Learning allocation: why residual bases starve, and statistic-valued race memory as the repair

**Failure addressed.** Theory 62 measures direct base responsibility on trained gated models: mean .072,
median .0045 (full core). In a mixture cascade the base's logit gradient per target is `r_y(q − e_y)`. Its
learning signal on a fit of N targets is therefore weighted by `r`. A useful summary is the Kish effective
sample size `N_eff = N·(E r)²/E[r²]`. Because `r ≤ 1`, `E[r²] ≤ E r`, so `N_eff ≥ N·E r`; this is a lower
bound, and with heavy-tailed `r` (median 16× below the mean) `N_eff` sits close to it. At the measured mean,
the base learns from at most an order of magnitude fewer effective targets than the fit contains.

These are also the least structured targets: those where every count order failed, which is where novel
continuations sit. If learnable structure beyond counts needs data N* to show (§390.1), the composition
moves that crossover to roughly N*/E r for the base. Adam rescaling (theory 62) does not change which
examples the signal comes from. This is a quantitative reason, consistent with all completed fits, why
minimal and full cores tie in composition. It is not a proof that deep features cannot help.

**Repair: give the core a job with full credit where counts are weak.** The core's top-level state forms a
query `u_t`. M receivers have learned keys and **sufficient-statistic values**. The pooled level of the cascade
is the exact race expectation `Σ_a π_a p_a(y)` (§383.1): zero-variance delivery credit over all candidates at
M lookups. The core receives credit through `π` wherever the pooled level carries responsibility. Because the
pooled level sits below the exact orders, that is precisely the low-evidence contexts. There the core must
generalize, mapping an unseen or rare context to an address whose statistics fit it. This is learned context
pooling: online MDL clustering of contexts (§382), driven by the core's features rather than exact strings.
It is the first rung where the core can supply `I(Y; Z | C) > 0` (theory 62) at small data.

Retained: temporal core, races over learned keys, sparse addressed statistic state, key/value separation,
escape cascade, per-position escape gate, exact counterfactual delivery credit. Added: M×A count state
(M = 256: 6,912 counts), a d×16 query map, M keys, M·A lookups per target and one integer write per event.
Simplifications: the zero-temperature race selects the writer (argmax π; no extra RNG); writes are not
differentiated (write credit, §383.2, deferred); counts restart each fitting pass, and development starts
from the last completed pass's counts (fit data only, causal). With all receivers empty the model equals the
gated count model exactly (contract-tested).

**Predictions (8K, seed 6, K4 + gate; gate-only references: full 2.601, minimal 2.588).**
(P392a) Pooled memory improves the minimal-core composition by ≥ .02. Pooled statistics help even with a
weak query. (P392b) With pooled memory, the full core beats both the minimal core (p2/d1) and the same-width shallow core (p16/d1) by ≥ .02. Minimal alone changes width and depth together. This is the first
test where core features earn their work by routing. (P392c) Mean pooled-level responsibility exceeds the
residual base's (.072). If P392a holds and P392b fails, the query features, not the memory, are the bottleneck.
The next steps are then a richer query (taps, longer credit) and write credit (§383.2). If both fail, pooling at
M = 256 adds nothing at 8K: test at ≥ 131K before concluding.

*§392 versioning note (cf. theory 63).* Statistic values have no value-version drift. Counts live in the fixed
symbol coordinates, so a read never mixes historical projection maps W_t, and the delivery credit to the query,
keys and temperature is exact at every read, even after chunk detachment. Staleness moves to the
*assignment*: counts at address a were written under older keys and queries. Restarting counts at every
fitting pass bounds that drift to one pass. Measure it as the pooled-level loss gap between end-of-pass counts
and a frozen re-assignment replay of the same pass, before scaling M or the data.

**§392 first result: dead-receiver deadlock, not a test of pooling (2 October, 11:50 UTC).** Pooled minimal 8K
(M = 256, argmax writes) scores 2.5878, identical to gate-only minimal (2.5878), at 6.0 versus 0.29 whole-fit
GFLOPs. The frozen router audit (results/diagnostics/curie_statistic_race_router_audit_20261002T115500Z.json)
finds π ≈ uniform: mean entropy 5.529 nats (uniform 5.545), mean max share .0074 (1/M = .0039), and the
same argmax receiver at all 1,024 audited positions. Two receivers were ever written.

The mechanism is self-reinforcing. A flat untrained router sends every argmax write to one receiver, the
other M − 1 stay empty and deliver exactly q, and with identical candidates the delivery gradient
`Σ_a ∇π_a (p_a(y) − p̄(y))` vanishes, so the router never learns. This is the dead-unit failure of hard
k-means/EM. P392a/b are therefore untested, not falsified; the shallow and full runs were withdrawn as
uninformative.

Repair: sampled race writes (the substrate's actual race; rates π, a generator seeded by event index), so
early writes spread in proportion to π. Receivers then differ, the delivery gradient becomes nonzero, and
the router can specialize (soft-to-hard EM). Use M = 64 at 8K: about 128 counts per receiver instead of 32,
and a quarter of the lookup work. Same predictions and controls.

## 393. Where counting runs out: the regime in which advantage must be shown

Counting predictors are near-optimal estimators of local conditionals wherever contexts carry ample evidence.
A learner should not be expected to beat them there; that it does not is not evidence against it. Their
limits are structural:

1. **Evidence sparsity.** Distinct order-K contexts grow roughly as A^{H_K}, while the evidence per context
   falls. At any N, a fraction of positions sit in unseen or low-count top contexts, where counts can only back
   off to shorter suffixes and discard the longer context. A learner can generalize from contexts that are
   similar but not identical. That is the I(Y; Z | C) of theory 62, concentrated in these positions.
2. **Range.** Information beyond K characters (words, topic, copies, long agreements) is invisible to an
   order-K table at any N. Raising K multiplies the sparsity in (1).
3. **Storage.** Table size grows with distinct contexts, close to linear in N at high order. A bounded
   learned state does not.
4. **Novelty and shift.** New words and changing statistics need generalization or fast adaptation,
   not accumulated exact matches.

**Evaluation consequence.** Average bpc on a window dominated by high-evidence positions mostly measures how
close a learner gets to counts. Every count-composed comparison should also report bpc **stratified by the
counts' own evidence**: prequential top-order context count n_K ∈ {0, 1–2, 3–9, ≥ 10}. The advantage claim
for the integrated model is: no worse than counts in high-evidence strata, where the cascade defers to them;
strictly better in low-evidence strata, where the learner owns the prediction; and a gap that persists or
grows with N, since the low-evidence strata never vanish at higher K. The full-versus-minimal core
comparison belongs in the low-evidence strata above all. A minimal core can match counts where they are
strong. Only learned features can help where they are weak.

**Queued audit** (curie_count_limit_audit_20261002T121500Z): saved 8K/32K gate full and minimal checkpoints,
frozen, with per-position losses and stream-adaptive Witten–Bell o4 and frozen KN o5 on the same positions.
Prediction (P393): if the full core contributes anything, it does so in the unseen and 1–2 strata. If full ≤
minimal there too, the current core supplies no generalization beyond counts at these N, whatever the
average says.

**§392 placement correction (2 October, 12:10 UTC).** The pooled level was placed below the exact suffix orders,
q → pooled → order 1 … K. Its delivery is then multiplied by the escape mass of every exact order. That is
the same product that starves the residual base (median responsibility .0045). The §392 claim that it "carries
responsibility on low-evidence contexts" was wrong for that placement: it carries responsibility only where all
exact orders escape. Pooled minimal 8K with sampled writes (64/64 receivers occupied) scores 2.593 against
gate-only 2.588. P392a fails, consistent with this.

A context routed by the core's state, which summarizes the whole history, is *more* specific than an order-K
suffix. By the hierarchical-Pitman–Yor ordering (more specific contexts back off to less specific ones) it
belongs on **top**: `p(y) = Σ_a π_a [max(c_ay − D, 0) + (θ + D T_a) p_exact(y)]/(n_a + θ)`. A receiver with
evidence claims mass directly, and the router gets first-order credit. Implemented as
TopStatisticRaceNativeModel (sleeping_machines/statistic_race_top.py; nesting, causality, chunk invariance,
router gradients and direct-claim contracts pass). The bottom-placed series (minimal, shallow, full) completes
as recorded evidence about that placement. The top-placed series is next, with the same controls.
Predictions (P392a′/b′): top pooled minimal ≤ 2.568; top pooled full beats top pooled minimal and shallow by
≥ .02, concentrated in the unseen and low-evidence strata (§393).

**§393 result (2 October, 12:52 UTC; results/diagnostics/curie_count_limit_stratified_20261002T121500Z.json).**
Strata by prequential order-4 context evidence. Saved means are reproduced exactly.

| bpc | unseen | 1–2 | 3–9 | ≥ 10 |
|---|---:|---:|---:|---:|
| 8K counts WB o4 adaptive / KN o5 frozen | 3.749 / **3.464** | 2.567 / 3.317 | 2.335 / 2.944 | 2.141 / 2.568 |
| 8K ours gate full / minimal | 3.513 / 3.490 | 2.484 / 2.467 | 2.240 / 2.232 | 2.125 / 2.124 |
| 32K counts WB o4 adaptive / KN o5 frozen | 4.017 / **3.477** | 2.612 / 3.087 | 2.300 / 2.640 | 2.206 / 2.306 |
| 32K ours gate full / minimal | 3.632 / 3.577 | 2.475 / **2.384** | 2.183 / **2.127** | 2.160 / **2.140** |

(1) The composition beats both count references in every stratum with evidence. (2) In the **unseen** stratum,
where counting runs out, frozen Kneser–Ney beats it by .03–.16 bpc. KN's lower orders use continuation counts
(the number of distinct preceding contexts a symbol completes), which estimate the law of novel continuations;
our cascade backs off through raw occurrence counts, which overweight frequent symbols exactly where the top
context is new. (3) **P393 fails:** the full core is worse than the minimal core in every stratum, unseen
included. At 8K–32K the current core supplies no generalization beyond counts.

Consequences. The unseen stratum is the decisive arena, and its floor is set by the backoff statistic, not the
learner. Lower cascade orders should carry continuation statistics (a cheap, exact change to the count
receivers; the top order keeps occurrence counts, as in interpolated KN). Any core contribution must then beat
that stronger floor in the unseen and 1–2 strata. The top-placed pooled memory, the next queued series, is the
mechanism aimed at this stratum; its result must be read there, not in the mean.

## 394. Calibration: compare only where learning matters; a joint language–event task

**Calibration rule.** A comparison is informative about learning only if a strong conventional learner beats
the strongest table or count method on the task by a clear margin. On character text the repository's own
controls fail that test below about 90M characters: frozen modified-KN o7 scores 1.788 at 10M (LSTM 1.799,
Transformer 1.908), and the Transformer leads by only .05 at 90M (§381). Stream-adaptive interpolated KN is
stronger still (§393). Small-N language therefore measures closeness to counts for *every* learner, ours and
Transformers alike. Every new task reports three bars: the strongest table, a tuned conventional learned
control with the same input information, and ours, with quality, whole-fit and per-query work. A task where
the learned control does not beat tables is reported as non-informative for learning.

**Milestone task (joint text + irregular events, one persistent address).** Each episode is one stream on a
single native source address. The content vector carries 27 characters, 4 event marks and a query flag, and
every event has a physical timestamp. Phases: background marked events; a spelled question ("is <word>
recent", with optional filler words, characters at irregular intervals, some background events interleaved);
then the decisive phase, the last event of the named mark followed by k ∈ {0..3} distractor events of other
marks; then a query. Label: 1 iff the elapsed time from the last event of the named mark to the query is
below Δ. Offsets are drawn from [.3Δ, .95Δ] for label 1 and [1.05Δ, 2.5Δ] for label 0, with balanced labels.

Shortcuts removed by construction: (i) time-blind tables (marks, words, orders, counts) are at chance
because the label depends only on the offset; (ii) rank-only models are at chance because k, the
number of events after the named mark, and the text placement are drawn independently of the label;
(iii) a text-blind model cannot tell which mark is asked about; (iv) the named mark's last event sits after
the text, so the word must persist in state while events arrive. A solution must read the word from
characters, bind it to the mark stream, keep the binding across events, and compare an exact elapsed time
with Δ. That is language, irregular time and persistent state in one computation.

Arms: native (unchanged AddressedEventHeads, one address, observed time); the native rank-time control;
cleared-text control (text events removed, an information ablation); a table bar (best discrete lookup on
time-blind features, fitted on the fit set); and, on AWS, a Δt-input GRU and a time-encoded Transformer with
the same events. Calibration passes if a learned control beats the table bar by ≥ 20 points. Ours is then
judged on accuracy, whole-fit and per-query work against those controls.

## 395. Where incumbents are not optimal: streaming elapsed-time binding over long histories

Pure retrieval and small-data character prediction already have near-optimal conventional solutions: nearest-
neighbour indexes and smoothed counts. Learning-based advantage claims belong where conventional solutions pay
a structural price. The joint task of §394 has a history-length knob: the number of background events before
the decisive phase. Per *event* of history, the arms behave as follows.

- **Causal Transformer with time encodings:** attention over all retained events. Per-query inference work
  grows ∝ T and fitting ∝ T² per episode, but retrieval of the named mark's last event does not degrade with T.
- **Δt-GRU (continuous-time decay):** O(1) work per event. The word and the binding must survive T distractors
  in one dense state under learned decay; interference grows with T.
- **Native core:** O(1) work per event (depth·heads selected updates, depth·heads·pool scored keys in
  training). Persistent addressed state evolves analytically in physical time, so silence costs nothing. The
  question is whether quality holds as T grows.

**Claim form.** At history lengths where the Transformer's per-query work exceeds ours by a large factor, ours
must match its accuracy within a few points, or beat the Δt-GRU at equal or lower work. Either is a point on
the quality/work frontier that incumbents do not occupy. Ladder: background ∈ {[2,6), [30,40), [120,140)}
events, same rule and balance (contract-tested), all three arms plus the table bar. Per-query and whole-fit
work go in the same units. The ladder runs only after the base pilot shows learnability: native observed time
above the table bar by ≥ 20 points.

**Predictions.** (P395a) The Transformer's per-query inference work grows ≥ 10× from the shortest to the
longest rung, while ours stays within 1.5×. (P395b) The Δt-GRU's accuracy falls by ≥ 10 points across the ladder.
(P395c) Ours stays within 5 points of its base accuracy. P395c is the risky one: it requires the addressed
persistent state and race selection to protect the binding from distractors.

**§392 top-placement result (2 October, 13:52 UTC): falsified as implemented.** Top-placed pooled memory
(M = 64, sampled writes, 8K, seed 6) scores 3.471 (minimal) / 3.492 (shallow) / 3.474 (full), against 2.59
for bottom placement and gate-only. Every core size is badly worse, so P392a′/b′ fail. Diagnosis: the
specificity ordering assumes the more specific level holds fewer counts. With 64 receivers pooling about 128
counts from unrelated contexts each, an untrained pooled receiver is *less* specific than an order-4 suffix
(nearly a unigram), yet with n_a large it claims nearly all mass on top of the exact cascade. Four passes did
not separate the routing. The construction can only work once routing creates receivers that really are more
specific than the exact suffixes. Before that, a learned evidence gate must decide between the pooled level and
the exact cascade (a parallel expert with a per-position responsibility, as in the escape gate), not a fixed
ordering. Both placements are recorded. Following the §394 calibration rule, further pooled-memory work moves
to tasks where pooled generalization is needed, not small-data language, where counts are near-optimal.

**§394 task revision (2 October, 14:30 UTC): the v1 task leaked; v2 replaces it.** The v1 smoke reached 90.6%
after 32 episodes. The time since the last character predicts the v1 label with 97.4% accuracy text-blind,
because the text ended a nearly fixed gap before the decisive event. The time since the last event gives 72%.
The v1 pilot was stopped at start and its smoke result is retained as a record of the leak. Any *monotone* elapsed
rule ("within Δ") also lets recency rank stand in for time: a time-blind table then reaches 90%. v2 therefore
uses a non-monotone exact-time rule. Label 1 iff the named mark's elapsed time lies in an even band of width Δ
(six bands, 0.1Δ edge margins), with exactly two marks in even bands and two in odd bands, and the question
ending 6Δ + U(.3, 2) before the query. Measured bars on independent sets: text-blind time-aware gradient-boosted
trees 48.8% (chance); time-blind table (word, last mark, rank) 60.4%; exact rank + text Bayes bound 62.0%; full
information 100%. Contracts: tests/test_joint_event_language_tasks.py (v1 leak kept as a regression check).
The phase rule is literally a periodic elapsed-time computation, which the core's rotating memories can
represent. Whether it is learned is the test.
