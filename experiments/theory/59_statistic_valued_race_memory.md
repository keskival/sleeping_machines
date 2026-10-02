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
