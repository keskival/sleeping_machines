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

## 396. Real-data capacity ladder: DVS gestures (where practical headroom exists)

Calibration (other host, theory 73/74): on subject-disjoint first-second DVS Gesture packets (984 fit / 192 dev),
class counts score 58.85%, linear 61.46%, selected RBF 73.44% (grid max 74.48%), and a compact
kernel/prototype control 66.67%. That leaves 14.6 points of practical headroom over counts, with every control
below 75%. The native model (p16/depth2/H2/pool2, 15.5K parameters, 8 receivers) reaches 65.10%, or
66.15% with packet-calibrated clocks, and learns real features (frozen-feature probe 57.8% → 67.2%). It is
small relative to the task.

Ladder, on the same driver, data, seed and protocol: pool 8 (4× receivers, the same depth·heads selected
updates per event: capacity beyond activity), depth 4, payload 32, and all three combined. Predictions:
(P396a) pool 8 gains ≥ 2 points over pool 2 at an unchanged selected-update count; (P396b) the combined shape
reaches ≥ 70%, approaching RBF. Fitting work grows mainly through scored keys and losing-value credit,
which are charged. Per-query inference work and selected updates are reported beside quality. A gain from
pool alone would be the first real-data evidence that available capacity beyond activity pays.

## 397. Importing Transformer language into the asynchronous substrate

**Correspondence, stated exactly.** With clock rates exp(q·k/√d_h), a race's first arrival is key i with
probability softmax_i (§96), so one race is an unbiased one-sample estimator of an attention head. Its
*expected* delivery Σπ_i v_i is the head exactly. Pre-norm feed-forward blocks, layer norms, residuals and the
KV cache map to local vector maps and persistent addressed state. The substrate therefore contains Transformer
computation **in expectation**, and exactly under expected delivery. It is not identical step for step: sampling
variance, bounded candidate sets and truncated credit are the differences.

**Strategy.** Small-data language measures closeness to counts for every learner (§394), so learning language from
scratch here is not where the substrate can show value. A trained Transformer's weights can instead be run as a
race-attention event stream (sleeping_machines/race_transformer.py). Expected delivery reproduces the Transformer
exactly (contract: 1e-10 in double). Sampled races cut value reads from n to at most S per head, and shortlists
restrict both. A per-head elapsed-time clock term exp(−λ_h (t_now − t_i)), zero at conversion, lets physical time
modulate attention on asynchronous streams while the imported language weights stay fixed.

**Tests.** (1) Exactness on the trained E64 Transformer (AWS checkpoint; experiments/race_transformer_eval.py):
expected-delivery bpc equals the original's. (2) Sparsity frontier: bpc against value reads and key scores per
token for S ∈ {1,4,16} and m ∈ {8,32,64}. Prediction (P397a): m = 32 stays within .05 bpc of the original at a
≥ 4× value-read reduction with ctx 256; single-sample races (S = 1) lose more than .2 bpc, so variance reduction
is needed. Key scoring stays O(n) without a candidate index, so the honest saving is in value reads until
discovery is learned. (3) Asynchrony: the converted model on timestamped text interleaved with irregular events,
fitting only λ and a small event adapter, against the same Transformer given time encodings. This is the
regime where an advantage is possible.

*§397 status (2 October, 18:25 UTC): deprioritized by user decision.* A previous Transformer conversion attempt
needed more compute than is available to do well. The conversion module and contracts remain as a documented
option, and no runs are queued. The program pursues comparable expressive power within the native lean
structure: addressed persistent state, races, time and counterfactual credit, scaled by capacity beyond
activity (§396).

## 398. Capacity–exposure law: private state, shared parameters

**Setting.** A head routes each event to one of P receiver units by a race. A unit carries *state* (its persistent
memory and arrival time, rewritten only when it wins) and *parameters* θ_u (input, output, gate, control and
key-read maps). The counterfactual teacher gives every unit score credit, but value credit reaches only the
winner (TemporalRoute.backward). So θ_u learns only from events unit u wins.

**Proposition 398.1 (parameter dilution).** Let unit u win a fraction w_u of N fitting events, and suppose the
estimation error of a parametric map behaves as c·|θ_u|/n for n effective examples (the usual parametric rate).
The expected excess risk from estimating every unit's map is

    Σ_u w_u · c|θ_u| / (N w_u) = c Σ_u |θ_u| / N = c P |θ| / N   (untied, |θ_u| = |θ|),

independent of how routing is balanced and linear in P. With maps shared across the pool (θ_u = θ for all u),
every routed event updates the same θ and the estimation term is c|θ|/N, independent of P.
*Proof.* Substitute n_u = N w_u; the w_u cancel. ∎ Approximation error falls with P in both cases,
because more distinct states can specialize. Only the untied construction pays for it with estimation error
that grows linearly in P at fixed N.

**Consequences.**
1. *Capacity beyond activity pays in learning only for state, or for shared parameters.* Adding private
   parametric units at fixed data trades approximation for estimation one-for-one. Adding private *state*
   (memory slots, arrival clocks, keys) with shared maps does not.
2. *Exact sufficient statistics are the limiting case.* Count receivers have no parameters: every exposure is
   used exactly, with no dilution beyond the data per context (§§377, 383). That is why they scale with capacity so cheaply.
3. *Transformers are already on the right side.* A KV cache is private state with shared projection
   weights, which is why context length (state capacity) does not dilute learning.

**Evidence consistent with 398.1** (not a proof of the mechanism):
- DVS, pool 2 → 8 untied at fixed selected updates: 66.15% → 62.50%, with the curve still rising at the
  last pass and 2.85× the fitting work.
- AWS source-64 capacity: shared maps reach 99.25% at 1.03 GFLOPs, private maps 97.40% at 1.32 GFLOPs.
- Language: minimal cores match full ones while counts carry the capacity (§389).

**Design rule.** Grow addressed state; share maps; keep routing identity (keys, clock biases) and timescales
private, since they are cheap and few. **Test (P398):** DVS pool 8 with maps tied across each pool, keys, clock
biases and timescales private. Prediction: it is no worse than untied pool 8 at equal passes, it beats the pool 2
baseline mean across seeds 6/7/8, and its parameter count stays near pool 2 while available receivers are 4×.

## 399. DVS scale-up failures separate into generalization (width) and optimization (depth)

Seed-6 results, clock-calibrated driver; fit-subset numbers are on 32 fitting gestures, so they are indicative:

| Shape | Params | Fit acc / NLL | Dev acc / NLL | Whole-fit GFLOPs |
|---|---:|---|---|---:|
| p16 d2 pool 2 (baseline) | 15.5K | .844 / .622 | .661 / 1.042 | 20.1 |
| p16 d2 pool 8 | 42.1K | .781 / .586 | .625 / 1.088 | 57.3 |
| p16 d4 pool 2 | 28.5K | .781 / .650 | .573 / 1.222 | 37.9 |
| p32 d2 pool 2 | 57.6K | .906 / .387 | .594 / 1.170 | 70.9 |

*Width* (payload, untied pool) fits the training gestures better and generalizes worse: a generalization failure on
984 gestures with a subject-disjoint development split and no regularizer beyond epoch selection. The RBF control
(73.4%) is regularized through C. *Depth* fits worse and generalizes worse: an optimization or credit failure.
The baseline's 66.15% is the best epoch of a curve swinging ±5 points, so seed replicates are required before ranking.

**Plan.** Width: tied pools (§398) remove dilution and cut parameters (queued, seeds 6–8); a weight-decay
sweep follows if tied pools still overfit. Depth, in order: (1) frozen single-race credit-fidelity audit by
depth (forced-winner exact gradients; queued); (2) depth-4 controls (pathwise credit, which tests the surrogate;
16 passes, which tests under-training); (3) **growth by nesting**: append layers to a trained depth-2 model with
near-identity initialization (new units' gates closed, identity channel mixes), so the grown model starts at the
parent's quality and can only use the extra depth if credit can move it. This separates "can depth add anything" from
"can depth be trained from scratch".

## 400. Race credit fidelity on trained DVS models is near chance; exact-π linearized credit

**Protocol/expectation correction (2 October, [note89](89_winner_dependent_teachers_and_time_law.md)):**
The original numbers below are retained, but the forced replay uses the individual
candidate time, so it changes identity AND timing rather than measuring exact
fixed-time categorical utility. The near-chance interpretation is not established
by that protocol. Exact-π replacement preserves the old teacher's expectation
for a fixed downstream error; the error generally depends on the winner in a
nonlinear model. Four completed contracts204000Z include a convex quadratic
where replacement reverses the expected direction. It remains an empirical
candidate, with no general unchanged-expectation or zero-variance guarantee.
The fixed-time actual-write audit and new fitting-only branch audit have separate
scopes; do not merge their numbers. Source/results below stay preserved.

**Measured** (frozen audit, results/diagnostics/curie_dvs_credit_audit_20261002T192000Z.json; 6 development
gestures, 24 sampled races each; the exact single-race gradient comes from forced-winner replays with all other
race noise fixed). Sign agreement of the counterfactual surrogate with the exact gradient is .714 at depth 0 of
the depth-4 model and .55–.64 elsewhere. Mean cosine is .08–.30. The surrogate's magnitude is 15–33% of exact
(pool 2) and 2–5% (pool 8). Routers, keys and clocks learn from a signal barely better than random in
direction, at every depth. That is consistent with depth hurting, pool 8 not using its receivers, and full ≈
minimal cores on language.

**Two separable degradations.** (a) *Estimator noise.* The teacher rate_i·T·g·(v_i − v̄), with conservation at
the winner, uses the realized race time as a one-sample estimate of π_i. Its expectation over the winner is
π_i g·(v_i − Σπ_j v_j) (Monte-Carlo-verified in tests/test_exact_pi_race.py), and π is known exactly from the
scores. **Exact-π linearized credit** (sleeping_machines/exact_pi_race.py) has the same expectation, zero
estimator variance and identical cost. The forward race is unchanged. (b) *Linearization.* Both ignore how a
different winner changes later races. Correcting that needs lookahead or replay credit, which costs more.

**Tests.** The extended audit (v2) reports the exact-π estimator's fidelity beside the surrogate's on the same
races. If it is much higher, (a) dominates. Then exact-π training of the baseline shape and depth 4 on seed 7
(queue curie_dvs_exact_pi_20261002T201500Z), against the seed-7 baseline and the seed-7 depth controls.
Prediction (P400): exact-π raises sign agreement above .8. If it does, exact-π depth 4 closes at least half the
gap to depth 2. If agreement stays near chance, linearization dominates, and multi-step credit (replay or
lookahead over the small pools) is the next construction.

**§400 audit result (2 October, 20:32 UTC; curie_dvs_credit_audit_v2_20261002T201500Z.json): P400 fails.**
Exact-π linearized credit is no more faithful than the surrogate on the same races. Sign agreement is .48–.64
against .53–.71, and cosine .01–.28 against .04–.30. Estimator noise is not the bottleneck; **linearization is.**
The seed-7 baseline scores 57.8% against 66.15% for seed 6, an 8-point seed spread, so single-seed ladder
rankings (§399) are confounded by seed noise and need the seed replicates.

## 401. Commit-aware race credit: the write effect the linearization misses

A race outcome changes the loss through two paths: (i) the delivered value at this event, the only path the
surrogate and the exact-π estimator linearize; (ii) the **commit**. The winner's proposed memory is written to its
slot and carried into every later read, while each loser's slot keeps its old memory. The AWS factorial write
audit (§373, ROUTE_WRITE_DIAGNOSTIC) measured mean |commit effect| .0127 against .0003 for the
delivered-value residual. The commit path dominates.

**Linearized commit credit.** Let G_u be the gradient of the loss with respect to slot u's stored content after
event t (from future reads only), and Δ_u = m_u^new − m_u^old the change a write to u makes (the decayed old
content plus write_u·W_in,u x). To first order, replacing winner w by alternative i changes the loss by

    L(i) − L(w) ≈ g·(v_i − v_w) + G_i·Δ_i − G_w·Δ_w,

and the expected-loss score credit is π_i(ℓ_i − Σ_j π_j ℓ_j) with ℓ_i = g·v_i + G_i·Δ_i (the terms involving only
w are constant across i). G_w is the gradient of the winner's stored memory. G_i for losers is the gradient
of their unchanged slot from reads after t, which is isolated by aliasing each slot after every event. All of
these exist after one backward pass. Adding the correction to the score gradients costs a second backward through the
retained graph, about 2× backward. Approximations: slot gradients are taken on the realized trajectory, and the
arrival-time representation of a rewritten slot is linearized.

**Test before training.** An audit estimator computes ℓ_i with commit terms on the same races as §400. Prediction
(P401): sign agreement with the exact single-race gradient rises above .75. If it does, commit-aware credit
goes into training (depth 4 and the baseline, seeds 6–8). If it does not, the remaining gap is the
nonlinear routing cascade, and multi-step replay credit over the small pools is the construction to cost.

## 402. Races as stochastic computation graphs: local-expectation counterfactual credit with common random numbers

**Conditional-law correction (2 October, [note92](92_conditional_replay_clocks_and_critic_sampling.md)):**
The implementation below forces an alternative at its individual arrival time,
changing identity AND clock. That is not the first-time-preserving categorical
counterfactual in the claimed exact formula. A time-only two-route witness
reverses one expected score direction when that correction is added to factual
pathwise credit. Existing algebra/factual-winner tests do not establish unbiased
expected-risk credit. Preserve this as an empirical combined-intervention
candidate; require corrected law/clock contracts before an exactness claim.


**Framing.** Each race is a discrete stochastic node of a stochastic computation graph (Schulman et al. 2015).
The winner changes the delivered value, the committed memory, later scores and later winners: a change of
event topology, analogous to the jump terms that event-based adjoints (EventProp; Wunderlich & Pehle 2021)
must add. The measurements in §§400–401 show that local linearizations of this dependence are near chance on
trained DVS models. The unbiased and well-conditioned alternative uses the *actual* downstream return.

**Estimator (local expectation gradient; Titsias & Lázaro-Gredilla 2015, here with common random numbers).**
Fix all race noise ξ of an episode. For race r with probabilities π_r = softmax(s_r), let L_r(i) be the episode
loss when race r is forced to alternative i and every other race keeps its noise; this is exactly the audit's
"exact" quantity. Then

    ∇_θ E_{w_r}[L | ξ_{-r}] = Σ_i ∇_θ π_{r,i} · L_r(i) + E_{w_r}[∇_θ L |_{realized branch}],

and summing over races gives an unbiased estimate of ∇E[L]. The first term is the route credit, obtained with
the surrogate loss Σ_i π_{r,i} · stopgrad(L_r(i)). Its gradient is π_{r,i}(L_r(i) − Σ_j π_{r,j} L_r(j))∇s_{r,i},
which a baseline does not change. The second term is the pathwise derivative inside the realized branch: the
winner's exact value and interior delay derivatives, the core's `credit='pathwise'` mode. Sampling k of the R races
uniformly and scaling by R/k keeps the estimate unbiased.

**Why it should work where linearizations failed.** L_r(i) contains the commit, the later races and every topology
change. With common random numbers the only difference between alternatives is the forced winner, so variance
is low; with pool 2 each race is an antithetic pair, as in ARM (Yin & Zhou 2019). DiCE (Foerster et al. 2018)
warns against differentiating such surrogates twice. Only first-order gradients are used here.

**Cost.** k·P forward replays per episode without gradients (each replays the episode from the start; replaying
only the suffix from a saved state is an optimization) plus one softmax term per sampled race. For DVS gestures
(about 21 events per episode, 4–8 races per event) with k = 4 and P = 2, the replays add roughly 2–3× the
fitting work. That is charged, and it scales with the sampled races, not with all alternatives of all races.

**Test.** DVS depth 2 and depth 4, pool 2, seed 7 (references: baseline 57.8%, the depth controls, exact-π).
Prediction (P402): local-expectation credit beats the counterfactual-surrogate baseline at equal passes and
narrows the depth-4 gap, with route credit faithful by construction on the sampled races. If it fails at matched
work, the bottleneck is not route credit.

## 403. Scaling counterfactual credit: learned critics as control variates, synthetic gradients for truncation

**Sampling qualification (2 October, [note92](92_conditional_replay_clocks_and_critic_sampling.md)):**
The critic cancellation below holds for a critic fixed before the correction's
race subset is sampled (or justified independent/cross-fitted conditioning).
Training it on that same subset before the estimator can introduce bias. The
correction also preserves the BASE replay target, so it cannot remove the
conditional-time-law bias noted beside §402. Original proposal remains below.


**Route credit at scale (user proposal; RUDDER, Arjona-Medina et al. 2019; COMA's critic, Foerster et al. 2018;
REBAR/RELAX control variates).** §402 costs P forward replays per sampled race. Train a small local critic
Q_φ(r, i) ≈ L_r(i), predicting the counterfactual episode loss of alternative i at race r from local features (the
unit's state, its proposed value and memory, the scores, the query). Regress it on the exact targets the sampled
common-random-number replays already produce. Use it as a control variate:

    ĝ = Σ_{all r} Σ_i ∇π_{r,i} Q_φ(r, i)  +  (R/k) Σ_{sampled r} Σ_i ∇π_{r,i} (L_r(i) − Q_φ(r, i)).

The expectation of the second term over the sampled races cancels the critic's bias exactly, so ĝ is **unbiased
for any critic**. Its variance falls as Q_φ approaches L, letting k (replays) shrink while every race still receives
credit. The cost is one small critic evaluation per race plus k·P replays. Critic training and replays are charged
as fitting work.

**Truncation (Decoupled Neural Interfaces; Jaderberg et al. 2017).** Credit is truncated at chunk boundaries
(16-event windows; the 16- and 64-event gradients differ by 31% in norm, §390). A small synthetic-gradient model
predicts ∂L_future/∂(carried memory, context) at the boundary and is trained on the true gradients available
within longer windows. Unlike the control variate, it is biased unless accurate. It needs a fidelity contract
(cosine with the true long-window gradient) before use.

**Order.** Only after P402 shows that exact replay credit improves training: (1) the critic as a control variate,
measuring quality against k at fixed passes; (2) synthetic gradients for truncation as a separate, contract-gated test.

**§§400–402 correction (2 October, 21:40 UTC, after theory note 92).** The forced replays in the §400/§401 audits
and in the §402 driver forced each alternative at its own exponential arrival. That changes identity *and*
clock, and it is not the conditional counterfactual: W and T are independent, and T | W=i ~ Exp(Λ). The audits'
"exact" targets were therefore biased, and their near-chance fidelity numbers (v1, v2) are measurements against a
biased target. They are retained, labelled, and must be re-measured before drawing conclusions. Fixed in the code:
forced replays keep the factual first time and change only identity, payload and memory write
(force_at_first_time). The §402 realized branch now uses the factorized race (sleeping_machines/factorized_race.py):
winner payload credit plus common first-time clock credit dT/ds_i = −T·π_i for every candidate, and choice
credit only from the first-time-preserving replays. Contracts (tests/test_factorized_race.py): note 92's
time-only witness (rates 1 and 3) recovers the true expected derivative [−1/16, −3/16] by Monte Carlo, with zero
choice credit; forced alternatives preserve the first time and RNG consumption; choice credit is exact for
winner-dependent losses. The queued commit audit (which also reports surrogate and exact-π columns) and the
§402 fits run with the corrected law. Per note 92, a §403 critic must be fixed before (or cross-fitted against) the
races it corrects.

**§401 result against the corrected target (2 October, 21:49 UTC; curie_dvs_commit_audit_20261002T204500Z.json): P401
fails.** With first-time-preserving forced replays as the exact single-race target (note 92), on 3 models
and 6 gestures: surrogate sign agreement .45–.67 (cosine −.09 to .29), exact-π .51–.62, commit-aware .51–.64
(cosine .02–.22). Commit-aware credit corrects the *magnitude* (median ratio .39–1.24 against .02–.74
for the others), and the commit term carries 17–41% of the linearized credit. It does not correct the *direction*.
The near-chance fidelity of every linearized estimator survives the corrected target. A winner change propagates
through later races, writes and timing in a way no first-order local expansion captures. Training credit must use
actual counterfactual returns (§402 with the corrected law), possibly made cheap with a pre-fixed or cross-fitted
critic (§403). Exact-π training: depth 2 58.3% / 1.054, depth 4 57.3% / 1.193 (seed 7; baseline depth 2 57.8% /
1.105), no gain, consistent with the audit.

## 404. Making replay credit cheap: forks, shadow lanes, bounded horizons

A counterfactual replay agrees with the factual run up to the forced race, and only the suffix differs (user
suggestion: run shadow channels alongside the main signal). From exact to approximate:

1. **Forked replays (exact, implemented: experiments/dvs_fork_replay.py).** The factual run snapshots the detached
   persistent state, RNG state and race counter at every event boundary. A replay resumes at the snapshot of the
   forced race's event. Its loss equals the full replay's exactly (contract, reference and fast paths), and training
   gradients are identical (tests/test_dvs_critic_le.py). Cost falls from the episode to the suffix: about half on
   average, and less for late races.
2. **Shadow lanes (exact; wall time ≈ 1×).** Fork P−1 counterfactual lanes at each sampled race and run them in
   parallel with the factual lane, with the same events and random numbers and a different winner at the fork, vectorized
   over a lane dimension. Sequential event steps do not grow. That matters most where per-event overhead dominates
   (our CPU core), and it matches a substrate with parallel shadow channels that never touch the factual output. It
   needs a lane dimension in the core.
3. **Bounded-horizon shadows plus a critic (controlled bias).** Run each counterfactual lane H events past the fork
   and value the resulting state with a critic (n-step / TD(λ)). Cost k·P·H instead of k·P·(T−t). Keeping a few full
   replays as a correction preserves unbiasedness, as in §403.
4. **Shadow lookahead inside an event (approximate).** Losing units' proposals already exist in training.
   Propagating them one or two layers in a shadow channel gives local lookahead credit. That alone is a
   linearization-like approximation (§401 shows its limits), so it needs the critic and occasional full replays.

Use order: forks now (exact, free accuracy-wise); the critic control variate with forks to cut k; shadow lanes when
a lane-batched core is written; bounded horizons only with a correction term.

**§404.2 implemented: shadow lanes (sleeping_machines/shadow_lanes.py).** One lane-batched, gradient-free pass carries
every counterfactual of an episode. Lanes have dense per-lane state, races share one noise draw in the factual
order (common random numbers by construction), and forced lanes take their alternative at the lane's first time.
Contracts: every lane equals its sequential forked replay (1e-10 in double; depth 2/pool 2 and depth 3/pool 3), and
training gradients equal the forked driver's (tests/test_dvs_critic_le.py). Timing at the DVS shape (p16 d2 H2 pool 2,
21 events, k = 4, P = 2), eight counterfactual losses: full replays 0.381 s, forked 0.135 s, **shadow lanes 0.055 s**,
against 0.069 s for one factual forward. All counterfactuals of an episode cost less than one factual forward, which
cuts the replay-credit overhead from about 3.6× to roughly 1.25× of baseline training.

## 405. Episode-batched native training: about 10× wall time at identical gradients

The DVS drivers give every episode of a pass the same race-noise seed (common per-pass draws). Episodes therefore run
as lanes of one batched pass that shares each race's noise draw, indexed by event position. That reproduces the
sequential runs exactly, including episodes of different lengths (sleeping_machines/batched_episodes.py; the factorized
race law per lane). Contracts: per-episode losses to 1e-10 and summed parameter gradients to 1e-8 against sequential
runs (tests/test_batched_episodes.py); batched evaluation equals the sequential evaluation; batched all-race replay credit
equals the sequential all-race replay credit (tests/test_dvs_batched_le.py). Speed at the DVS shape, one 16-episode
window forward plus backward: depth 2, 2.25 s → 0.23 s; depth 4, 4.39 s → 0.45 s (9.8×). Replay credit for every race
of every episode runs as one batched, gradient-free shadow pass, so exact local-expectation credit without race
sampling is affordable at depth 2. `--route-races k` samples races, unbiased with scaling, for deeper models.

## 406. The DVS native fits are generalization-limited

First P402 result (batched, seed 7, depth 2): all-race exact replay credit 58.3% / 1.142, factorized clock-only credit
56.8% / 1.116, original teacher 57.8% / 1.105. No gain, so exact route credit does not move this model. Every native
fit shows a 20–25 point gap between fitting and development gestures (fit-subset 81–84% against dev 57–66%,
subject-disjoint users). Better credit improves fitting and cannot close a gap set by generalization from 984
gestures. The strong controls are explicitly regularized (RBF C; the 4-bin kernel control reaches 77.6%), and coarser
packets helped the native model on two seeds, consistent with less overfitting. The native fits had only epoch
selection. With batched training (§405) a fit takes about 2.5 minutes, so the next step is a regularization sweep
(experiments/dvs_batched_reg_benchmark.py): decoupled weight decay {1e-3, 1e-2} × training-only input noise {0, .3},
plus coarse packets with weight decay, on seeds 6–8. Prediction (P406): some arm reduces the fit/dev gap and raises
mean dev accuracy across seeds by ≥ 3 points over the unregularized batched baseline. Remaining replay-credit seeds and
the depth-4 runs continue first, since credit may matter more at depth 4.

## 407. Depth is the test of route credit

At depth 2 a winner change reaches the readout through at most one further layer of races, so a local linear
teacher already carries most of the useful route signal. Null results there (§406) do not test the credit
hypothesis. With depth L, a winner change at layer d changes the inputs of every race at layers > d in the same event,
and their commits into later events. The topology change compounds, which is where linearized teachers lose fidelity
(§§400–401; sign agreement fell from .71 at depth 0 to .45–.59 deeper in the depth-4 audit) and where depth 4 underfit
(fit .650 NLL against .622 at depth 2, §399).

**Decision rule.** Batched, matched pairs on seeds 7 and 8: factorized clock-only against all-race replay credit at depth
4, and factorized against replay credit with k = 8 at depth 6 (seed 7). Route credit is the depth bottleneck if replay
credit (a) lowers the depth-4 fit-subset NLL below its factorized control on both seeds, and (b) brings depth-4 dev
quality to at least the depth-2 level of the same seed, with the depth-6 gap behaving likewise. If replay credit fits
deeper models better without generalizing better, the bottleneck is again generalization (§406). If it fits no better,
depth is limited by something other than route credit (state conditioning, clock initialization, truncation within the
episode), and growth by nesting (§399) is the next control.

## 408. Depth-4 underfitting is not route credit; a candidate is the fixed gradient clip

Batched results, seed 7: depth 4 factorized 55.2% dev (fit .688 / .718 NLL) and depth 4 with k = 8 replay credit 55.7%
(fit .719 / .715), against depth 2 at 56.8–58.3%. Sampled replay credit does not improve depth-4 fitting. A frozen
diagnostic on the trained depth-2 and depth-4 checkpoints rules out inter-layer transport: each layer's transport
costs about 3% decay and .11 rad rotation over a typical delay, identical at both depths. It finds larger, not vanishing,
gradients at depth 4 (per-layer norms 3.0–4.1 against 1.3–1.5). With a global clip of 1, the total norm (about 6.6
against about 2) scales each depth-4 update about 3× more than at depth 2: slower learning per step, consistent with
underfitting. Controls (queue curie_dvs_depth_opt_20261003T004500Z): clip 4 at depth 4 (seeds 7 and 8), double the
learning rate at depth 4, and clip 4 at depth 2 as a control. Prediction (P408): a looser clip lowers the depth-4 fit NLL
and closes at least half the dev gap to depth 2. The all-race replay-credit depth-4 runs continue as the matched
credit test.

**§407 interim (3 October, 02:20 UTC; batched, 8 passes).** Depth 4 factorized against all-race replay credit: seed 7
55.2% / 1.130 against 55.2% / 1.153; seed 8 59.9% / 1.093 against **64.6% / 1.021** (fit NLL .588 → .553). Mixed: a
gain on one seed, none on the other. **Growth by nesting** (seed-7 depth-2 parent → depth 4, factorized) reaches
**60.9%** / 1.109, the best seed-7 depth-4 result (scratch 55.2%; depth 2 at 56.8–57.8%). Depth adds something when it
starts from a trained shallow model, which supports an optimization or initialization explanation over a pure credit
one (one seed). Depth 6 factorized scores 59.4% / 1.086; depth 6 with k = 8 sampled replay credit collapses to 47.9% /
1.336, so the race-sampling variance (scale R/k with many races) is harmful. Sampled replay credit needs the critic
(§403) or all-race lanes. Seed replicates are queued (growth on seeds 6 and 8; depth 4 with and without all-race credit
on seed 6), along with the §408 clip and learning-rate controls.

## 409. The native language model at the scale where learning matters

Small-data language only measures closeness to counts (§394). At 10–90M characters the repository's dense controls
overtake or match counts (§381: LSTM 1.799 / 1.661, Transformer 1.908 / 1.604, frozen modified-KN o7 1.788 / 1.653 bpc at
10M / 90M on text8[95M:96M]). The native core never reached that regime: the sequential core trains at about 33
characters/s (about 84 h per 10M pass). Segment batching makes it reachable. Training uses independent 128-character segments with
the state reset per segment and shared per-step race noise, run as lanes of one exact batched pass (§405, contract:
per-event logits equal sequential runs). Credit spans each whole segment. Evaluation uses the E64 window protocol
(windows of 128 at stride 64). Measured training throughput on one CPU thread: depth 8 / payload 16 / 128 lanes 2,130
chars/s (1.6 GB); depth 4 / payload 32 / 256 lanes 5,340 chars/s (2.1 GB). One pass takes about 1.3 h / 31 min at 10M and
about 12 h / 4.7 h at 90M. Queue curie_language_batched_20261003T023000Z runs both 10M configurations first. Window
evaluation with a 64-character context is weaker than streaming state; the dense controls used the same windows (T = 256
for theirs). The protocol differs in segment length, and this is stated beside the numbers.

**§408 result (3 October, 02:33 UTC): P408 fails; step size is not the depth bottleneck.** Fit-subset NLL, batched, 8
passes. Depth 4 with clip 4: .945 (seed 7) and .711 (seed 8), against .718 and .588 with clip 1. Depth 4 with double the
learning rate: .960 (seed 7). Depth 2 with clip 4: .683 against .576. Larger effective steps make fitting *worse* at both
depths. The deep native model is not under-stepped; larger updates destabilize the hard-routed stack. Inter-layer
transport (§408), the global clip and, so far, route credit (§407) are excluded as the depth-4 bottleneck. The one depth
intervention that helped is growth by nesting from a trained depth-2 model (seed 7: 60.9%). That suggests a
loss-landscape or initialization difficulty for hard-routed deep stacks trained from scratch, which starting from a working
shallow solution avoids. Growth replicates (seeds 6 and 8) and progressive growth to depth 6 are queued in the large
program. The language runs keep clip 1.

## 410. Near-identity initialization for deep native stacks (ReZero/SkipInit idea)

Growth by nesting gave the best depth-4 result (§408) by starting new layers near identity. A unit's value is
x + gain·y·sigmoid(gate), so a nearly closed gate passes the input through. The from-scratch equivalent
(experiments/dvs_batched_large_benchmark.skip_init): layers ≥ k start with unit gate bias −4 (sigmoid ≈ .018: near
identity but learnable; growth's −20 effectively froze the new units, so its gains came from the channel mixes and
queries of the new layers), and intermediate layers transport with zero frequency and negligible decay. The top layer
keeps its transport for context alignment. Tests: DVS depth 4 on seeds 6–8 and depth 6 on seed 7 from scratch with
near-identity init from layer 2, against scratch and growth; and 10M language at depth 8 with the same init beside the
default-init depth-8 run. Prediction (P410): near-identity depth 4 matches growth (≥ the depth-2 level of the same
seed) without a trained parent, and the depth-8 language model with near-identity init beats its default-init twin.

**§406 result (3 October, 03:17 UTC): P406 holds.** Batched, depth 2 / payload 16 / pool 2, 8 passes, means over seeds
6–8 (dev accuracy / NLL; fit-subset accuracy; gap): unregularized 58.9% / 1.114 (fit 78.1%, gap 19.3 points); weight decay
1e-3 60.2% / 1.109; weight decay 1e-2 61.3% / 1.097; weight decay 1e-3 + input noise .3 **62.8% / 1.039** (gap 15.3); weight
decay 1e-2 + input noise .3 60.4% / 1.091; **coarse 4-bin packets + weight decay 1e-2 64.4% / .987** (gap 12.7 points, 60
s per fit). The native DVS model was generalization-limited, and regularization plus coarser packets recover 3.9–5.5 points
on the three-seed mean. The large program's capacity arms now use weight decay 1e-3 + noise .3, and its coarse arms add a
noise variant. Strong controls on the same full data remain higher (4-bin control 77.6%, RBF 73.4%).

**Growth construction update (3 October, 03:35 UTC; theory 122/123 by the other host).** Their plasticity probe shows
that at appended gate bias −20 about 99.7% of the new branches' nonlinear contributions round away in float32, and gate and
output gradients fall below Adam's ε. The new message branches are effectively silent, while input maps, keys and time stay
live. At −4 every contribution is visible and the gates learn. They also found that unit.gain is a Python float absent from
the state_dict, so progressive growth from a grown parent reset the inherited gains. Changes for newly starting
experiments: appended gates default to −4 (`--appended-gate-bias`, −20 reproduces the legacy construction), and per-layer
gains are reconstructed from the growth lineage (dvs_grow_depth_benchmark.lineage_gains; contract-tested). The large
program's growth arms run −4 and legacy −20 on seeds 6 and 8, and progressive growth 4 → 6 runs with live gates and
preserved gains. The seed-7 legacy result (60.9%) is therefore not evidence about the new message branches; its gain came
from the other live paths.

**§409 90M selection rule (fixed before the 10M results, 3 October 03:40 UTC).** The 90M one-pass run uses the best 10M arm
by test bpc on the E64 windows: depth 8 default init, payload 32 / depth 4, and depth 8 with near-identity init from layer
2 (gates −4, live per theory 123; intermediate transport closed). If the near-identity depth-8 arm is within .01 bpc of the
best or better, the 90M run uses near-identity init, at payload 32 when throughput allows a single pass in about 20 h.
Clip stays at 1 (P408 failed), with no weight decay or input noise, since one pass over 90M characters is not
the small-data regime. Growth is DVS-only. The 90M run is scored with both the segment protocol (T = 128) and E64-matched
windows (T = 256, stride 128), and the table states the protocol of every row.

**§409 selection correction, 3 October, before completed 10M outcomes.** The written test-bpc criterion above is superseded
by DEV bpc on text8[90M:91M], under the same declared window protocol across all candidate arms. Retain the fixed .01-bpc
near-identity preference and throughput constraint. text8[95M:96M] remains reporting-only after configuration selection;
precommitting a rule does not make selection by test scores an independent test. No active fitting source, result or
queue setting changes; the 90M selection record must bind completed DEV metrics and the fixed chosen configuration.

## 411. Optimizer-step calibration of the 10M native language arms

**Failure.** The first 10M arm (payload 16, depth 8, 128 lanes × 128 characters) takes 16,384 characters per Adam update,
about 610 updates per pass. At 3.3M characters it scores 3.21 bpc on the first 50K development characters, and its training
loss is flat near 3.06 bits between 4.9M and 5.8M characters. Earlier trained native models reached about 2.4 bpc at 32K
characters with thousands of 16-target updates. The 10M arm is therefore update-starved, not data-starved.

**Calibration.** The saved E64 controls (experiments/e64_lm_baselines.py) use batch 32 × context 256 = 8,192 characters
per step, which gives about 1,220 steps per 10M pass, and they anneal the learning rate with a cosine schedule over all
steps (Transformer lr 1e-3, LSTM 2e-3), for four or six passes. The v2 arms take 64 lanes × 128 = 8,192 characters per
update (the same step count per pass), lr .004 and the controls' cosine schedule (`--cosine`), for one pass. They are
selected by DEV bpc on text8[90M:91M] (`--dev 1000000`, per the §409 correction), and scored with E64 windows at both the
training length T = 128 and the controls' T = 256 (`--eval-segment 256`, same weights). Mechanisms are unchanged: the
integrated core, factorized races, sparse addressed writes, near-identity init (gates −4) on the depth-8 arms.

**Scope.** One pass against the controls' four to six passes is still fewer total updates; a completed v2 arm is a
statement about one-pass learning at this size, not an iso-update comparison. If the best v2 arm is still update-limited
(training loss falling at the end), the 90M run inherits the step-size calibration, not the v1 one.

**v1 outcome (3 October 04:34 UTC).** The update-starved arm finished at dev 2.921 / test 2.899 bpc (T = 128; 54,907
parameters; 4.04 TFLOPs whole fit, estimate), still improving at the end of the pass. The matched one-pass E64 controls
score 2.171 (LSTM 256, 338K parameters, 20.3 TFLOPs) and 2.427 (Transformer 256×2, 1.66M, 111 TFLOPs) at T = 256 with
1,220 cosine steps. Capacity and update count both differ, so the v2 arms fix the step count and the v3 arm
p32/d8/pool4 (346,331 parameters) matches the LSTM's parameter count. The v3 arm keeps 16 selected writes per character,
so it isolates stored capacity from selected activity.

## 412. Compiled layer steps: the batched native core is dispatch-bound

**Failure.** At language sizes the exact batched path (§405) costs about the same per window at 64 and 128 lanes
(6.1 s versus 6.8 s at p16/d8). Backward is about 60% of the time. The arithmetic per operation is tiny (payload 16, pool 2),
so the time goes to dispatching hundreds of small operators per event and layer in forward and in autograd. Fewer
characters per update (§411) therefore cost throughput almost one-for-one: 1,424 versus 2,426 characters/s.

**Change.** sleeping_machines/compiled_episodes.py evaluates one (event, layer) step for all lanes as a single
torch.compile'd function (inductor CPU kernels, one compile thread). The model, state bookkeeping and race-noise draw
order are unchanged. The factorized race is written so it can be traced: the payload is gathered at the winner (winner-only
value credit), and the first time T is held fixed plus the exactly-zero term −T(lse(s) − stopgrad lse(s)). Its score
gradient is −Tπ_i, the common first-time clock credit dT/ds_i of LaneRace. No mechanism is removed. Sparse addressed
writes, temporal races, transport and the factorized credit are the same functions. Forced replays (shadow lanes) and
recorded race scores remain on the batched path.

**Contracts.** tests/test_compiled_episodes.py: the traceable formulation equals batched_logits within 1e-10 in float64
(including near-identity init), the compiled step within 1e-9 in float64 and 2e-4 in float32, on variable-length
episodes. tests/test_dvs_batched_le.py: a compiled DVS training window's gradients and evaluation equal the batched path
within 1e-9, and traced windows stay bitwise on the batched path. Work tracing sees only dispatched operators, so traced
windows always run the batched path, and FLOP estimates keep their convention.

**Measured.** Language p16/d8, 64 lanes × 128: 5,800 characters/s in the running arm against 1,424 eager (4.1×). The
training loss at window 50 agrees with the eager run to 7e-5 bits. DVS depth 4, 16 episodes × 21 events: 0.15 s versus 0.52 s
per window (3.4×). Compiling once costs about 20–50 s per shape and grad mode. A 90M pass at p16/d8 is now about 4.3 h, so
the §409 payload-32 condition (a single pass in about 20 h) is within reach. This is wall time on one CPU thread. FLOP
counts are unchanged by construction, and energy is not measured.

## 413. Why the 10M native language model lags LSTM/Transformer: an untrained address, then width

**Completed evidence (one pass, 10M, E64 windows; test bpc).** Native p16/d8 2.719 (54,907 parameters, 0.405 MFLOPs
fitting per character), p32/d4 2.507 (108,875, 0.722). The one-pass E64 controls with the same update count are
LSTM-256 2.171 (338,395, 2.03 MFLOPs/char) and Transformer-256×2 2.427 (1.66M, 11.1). Scoring at T = 256 changes ours by
<.001 bpc, so context beyond about 64 characters is not used yet.

**1. The address is not trained (the main structural cause).** In the segment-batched driver the race is the factorized
law: payload credit to the winner, and the common first-time clock credit dT/ds_i = −Tπ_i into the arrival delay
(.001–.011 per event step). The scores (query, key, key_read, clock bias) reach the loss only through that delay.
Delay enters transport and memory ages, which are tiny shifts of a per-character time step of 1. Measured at
initialization on a real window (p16/d8, skip2): gradient-to-weight ratios are about 1e-4 for queries, keys and
key_read, about 3e-3 for unit value maps, and 0.2–0.6 for channel mix and head. Under Adam the magnitude is normalized,
so the routing parameters do move, but only along the timing direction. **No term tells a race which alternative
would have predicted better.** That is the counterfactual route credit the architecture is built on (hard routes
learn through counterfactual credit). The fast language path omitted it, and the AWS streaming runs (teacher/replay)
keep it. At initialization routing is near-uniform (mean max-π .61 at pool 2; .02% of races above .9).

*Consequence: fragmentation.* With an uninformative address, a pool of U units with winner-only writes partitions the
stream at random. Each unit's memory integrates a random 1/U of the history, and each read sees only the winner's memory.
The expected information carried about a given past character falls roughly as 1/U, and sampled routing adds noise at
evaluation. Pools then cost capacity instead of adding it. Capacity beyond activity requires a learned or at least
consistent content address. With that address, the pool becomes a content-addressed slot memory, and more units add
state without adding selected writes.

**2. Active width is small (the capacity cause).** Per character, the forward function uses the channel mix and query
(2·W² for width W = H·P), the winner unit's four P×P maps per head, and the head. That is about 33K active weights at
p16/d8 and about 25K at p32/d4 per layer stack, against 338K for LSTM-256, and the native fit spends 3–5× less arithmetic per
character. The first two results show width dominates depth at this budget: doubling P with half the depth cut 0.21 bpc
at 1.8× the work.

**3. Each unit is a small linear recurrence.** A unit's memory is a decayed, rotated sum of input projections (a
complex diagonal linear recurrence with input-dependent scalar forget and write gates), read through one nonlinear
map. This is the selective state-space family, which is competitive when its state is wide. Here a selected unit holds
P = 16–32 numbers, and the race chooses which of U such states to read.

**Predictions, recorded before the v4 results** (p32/d4, same protocol, one seed; diagnostics are labelled):
(a) pool 1 (no routing; diagnostic control) is not worse than pool 2 without route credit (2.507);
(b) pool 4 without route credit is worse than pool 2;
(c) linearized route credit, a zero-valued surrogate with score gradient π_i g·(v_i − v̄) (THEORY §403/§407 linearized
local expectation; forward values bitwise unchanged; batched_episodes.linear_route_credit), improves pool 2, and
improves pool 4 more if a content address can be learned;
(d) width 128 (p64/d4, about 422K parameters) improves substantially over p32/d4, the capacity cause.
If (c) fails, a linearized credit is insufficient (the DVS fidelity audits found linearized estimators near chance).
The next step is then exact local-expectation credit restricted to a short horizon, or a consistent non-learned content
address, before abandoning large pools for language.

**§413 addendum (07:25 UTC, before any v4 result): the write address.** g·(v_i − v̄) credits only the value forwarded
in this step. It ignores where the write goes, so it cannot teach a content-addressed slot memory. Writing slot j changes
only that slot, Δ_j = m_new,j − m_j, so the slot's expected content is m_j + π_j Δ_j. The linearized write-address
credit is ∂E[L]/∂s_i ≈ π_i (G_i·Δ_i − Σ_j π_j G_j·Δ_j), where G_j is the BPTT gradient on slot j's stored memory. It sees
the future reads of that slot within the segment. The implementation is the same kind of zero-valued surrogate
(batched_episodes.linear_write_credit; `--route-credit linear_rw` adds it to the value credit), with forward values
bitwise unchanged and the compiled path within 1e-9 of eager in float64. **Prediction (e):** linear_rw ≤ linear in bpc, with
the larger gain at pool 4, where addressing matters more.

**§413 first outcome (08:41 UTC): more units without address credit hurt.** The declared capacity arm p32/d8/pool4
(346,331 parameters, 16 selected writes per character, 2.36 MFLOPs fitting per character) scored DEV 2.444 / test 2.498
bpc. The pool-2 arm at the same width and depth (210,043 parameters, 1.41 MFLOPs) scored DEV 2.404 / test 2.456. Adding
stored capacity at unchanged selected activity made the model worse when the address has no value credit, as the
fragmentation argument predicts. Single seed, depth 8. The v4 queue tests the same at depth 4 with and without route
credit.

**§413 revision from the routing diagnostics (08:50 UTC; experiments/language_route_diagnostics.py, forward only, DEV
first 50K).** The statement "routing is near-uniform" holds at initialization only. After the one-pass fit, the p32/d8 pool-2
races are sharp and balanced: mean max π .82–.96 per layer, 52–87% of races above .9, unit usage 35/65–54/46. At pool 4,
mean max π is .69–.84, usage 15–35% per unit, entropy .47–1.0 of 2 bits. The timing credit sharpens races (shorter delays),
so the address is decisive but trained by timing, not by predictive value. Fragmentation needs an address that is not
aligned with prediction, not a random one, so the pool-4 loss stays consistent with it. Routing noise is a small part of
the gap: argmax routing is .009 (pool 2) / .006 (pool 4) bpc better than sampled, and a four-seed probability mixture
.022 / .019 better. These are diagnostics, not reported scores. The decisive test remains whether value credit (linear,
linear_rw) changes what the address encodes and improves quality.

**§413 memory horizon (saved p32/d8 pool-2 weights; state-dict only).** Unit memory half-lives ln2/rate at forget = 1
(the input-dependent forget multiplies this) are p10 1.0–1.4, median 5.8–7.5 and maximum 16–27 characters at every layer.
Initialization spans time constants 1–100 (half-lives 0.7–69), so one pass shortened the longest memories. Transport
half-lives span .5–62. The native model is therefore a short-horizon model (about 6 characters typical, under 30
maximum), which explains why T = 256 scoring changes nothing. The LSTM uses hundreds of characters. A slot written by an
address that is not aligned with prediction accumulates a mixture of unrelated contexts. Under that interference the
optimizer prefers fast forgetting, so the short horizon is consistent with the address diagnosis. The test is again the
v4 route-credit arms: if value credit aligns the address, longer half-lives should survive training, and the route
diagnostics plus a horizon readout on their saved weights will show it.

**§413 prediction (a) holds (09:15 UTC).** p32/d4 pool 1 (diagnostic control: no selection, one unit per head; 74,803
parameters, 0.49 MFLOPs fitting per character, 9,130 characters/s) scored DEV 2.386 / test 2.439 bpc. Pool 2 scored
2.449 / 2.507 (108,875 parameters, 0.72 MFLOPs). Pool 1 also beats the depth-8 pool-2 arm (2.456). With the address
trained by timing alone, the sparse pool costs about .07 bpc at depth 4 and .04 at depth 8 (pool 4 against pool 2), with
more parameters and work. This is evidence about the current credit, not about addressed memory: the architecture's
claim needs the address to learn from value. The remaining v4 arms test that. Pool 1 stays a labelled control, and it
is not promoted.

**§413 measurement scope correction (09:20 UTC; original numbers retained).** The routing diagnostic's
`mean_pi_per_unit` above is expected probability mass, not counted winner occupancy; sharpness remains supported.
Its greedy improvement changes both identity and delay, and the .022/.019 mixture improvements are against the
first seed, not the Jensen baseline (mean constituent loss). Those differences remain valid within their stated
policies, without isolating a pure routing-noise contribution. Likewise, ln2/rate is a base half-life **at unit forget**:
the actual candidate uses rate×input-dependent forget, and stored memory enters keys before candidate decay.
The reported base-rate maximum16–27 therefore does not bound actual retention/context use; T128/T256 similarity
also does not prove that distant history is unused. Interference and shortened effective memory remain testable
hypotheses. [141](141_streaming_routing_and_horizon_measurements.md) gives the derivation and a pending bounded
streaming diagnostic with actual winners, separate head buckets, clock-preserving identity interventions and the
correct mixture baseline. Existing completed results and the prioritized v4 credit comparisons remain unchanged.

**§413 prediction (c) holds (09:45 UTC): value credit to the address turns the pool from a cost into a gain.** p32/d4
pool 2 with the linearized local-expectation score credit π_i g·(v_i − v̄) (forward values unchanged; 108,875
parameters; 0.725 MFLOPs fitting per character against 0.722 without it) scored **DEV 2.314 / test 2.370 bpc** (T = 256:
2.371). That is .137 below the same model without route credit (2.507) and .069 below the no-selection pool-1 control
(2.439). Development on the first 50K was 2.610 / 2.435 / 2.384 at windows 400/800/1200, against 2.720 / 2.558 / 2.516
uncredited. At the same one-pass, 1,220-update protocol it is .057 below the E64 Transformer-256×2 (2.427; 1.66M
parameters, 11.1 MFLOPs/char estimated) and .199 above LSTM-256 (2.171; 338K, 2.03). Single seed. The core claim that hard
routes learn through counterfactual credit is now supported in language at 10M: removing that credit (the fast path's
factorized law) cost .137 bpc. The deficit to the LSTM remains, and width, the write address (linear_rw) and memory
horizon are the next levers.

**§413 write credit, first attempt failed (09:52 UTC).** p32/d4 pool 2 with linear_rw diverged: training loss went from
2.99 to 3.99 bits between windows 200 and 250, then gradients became non-finite (result absent; log and runner line
retained). Cause by derivation: memories decay lazily. A slot not selected stores (m_j, t_j), and its decay over the
gap is applied at its next selection with that moment's forget gate. Δ_j = m_new,j − m_j therefore counts the pending
decay and rotation as a content change, while the stamp reset that accompanies it is ignored. The fictitious component
grows with the gap and feeds back through key_read on memory. The corrected variant linear_rwn uses Δ_j = the newly
written content write_j·(Input x), which is bounded through layer-normed inputs. At trained weights it adds about 10% to the
route-gradient norm over the value credit, against 46% for linear_rw. Queued in v5.
