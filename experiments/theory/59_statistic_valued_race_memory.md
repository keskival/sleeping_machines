# Statistic-valued race memory: pooling, exact counterfactual credit and in-context statistics

[Theory index](../THEORY.md) · Global sections 382–385. Written 2026-10-02 on host `curie`.
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

**Proposition 382.1.** As `n₁, n₂ → ∞`,
`G = ((A_eff−1)/2)·log(n₁n₂/(n₁+n₂)) − (n₁+n₂)·JS_π(q₁,q₂) + O(1)`, where
`JS_π` is the Jensen–Shannon divergence weighted by `π_i = n_i/(n₁+n₂)`. Pooling
helps if and only if `JS_π ≲ (A_eff−1)·log n/(2n)`.

*Sketch.* Laplace expansion of each `L` gives `nH(q̂) + ((A_eff−1)/2) log n + O(1)`.
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
sampling. Equivalently, the mixture objective `−log Σ_a π_a p_a(y)` gives
posterior-responsibility weights.

With vector-valued neural receivers, evaluating a losing route means running
that route's computation forward (and backward for deep credit), so the
project's signature mechanism costs a forward pass per alternative
(§§19, 57, 373). With statistic values, the counterfactual is free and exact.
That makes hard routing with full counterfactual credit scalable, and
minimizing the summed sequential code length over routes is online MDL
clustering of contexts (§382).

**Proposition 383.2 (closed-form write credit).** Let receiver `a` hold counts
`c` (total `n`), and let its future events follow `q`. The expected change in
log predictive for the next event, after writing symbol `y` to `a`, is
exactly `q_y·log(1 + 1/(c_y+α)) − log(1 + 1/(n+Aα))`. To first order it is
positive if and only if `q_y > (c_y+α)/(n+Aα) = s_a(y)`: **a write helps a
receiver exactly when the receiver underpredicts the written symbol relative to
its future law.**

*Check.* Over 2,000 random cases the identity holds to 3×10⁻¹⁶. The first-order
sign rule agrees in 99.9% of 1,148 cases with `|q_y − s_y| > .01`.

This gives the "persistent commit" component that the AWS factorial audit
(§373, ROUTE_WRITE_DIAGNOSTIC) found dominant (mean |commit effect| .0127 versus
.0003 value-linearization residual). For statistic-valued receivers it is
analytic: no replay is needed to credit where a write should go. The unknown
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
(Corollary 377.3) add recency at no extra cost. A trained Transformer can
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

Dense training work is `≈ 6·P·D` for `P` parameters and `D` tokens: every token
updates every parameter. In the statistic-valued construction the per-event
work splits three ways:

- **counts:** `O(K + |B|)` lookups and increments; no backward pass;
- **learned forward pass:** needed for every event, unless the top escape mass
  `e_h < ε`, in which case skipping it changes the prediction by at most `ε` in
  total variation (379.2b);
- **learned backward pass:** weighted by the responsibility `r_y`, and skippable
  when `r_y < ε` with gradient error at most `ε√2` (379.2a).

With a learned base of `P` parameters, the training work is therefore about
`D·[c_K + 2P·(1 − f_e(ε)) + 4P·(1 − f_r(ε))]`. Here `f_e` and `f_r` are the
fractions of events below the escape and responsibility thresholds. Both are
measured as functions of `D` (§§379, 381). They are the substrate's
data-dependent sparsity. A dense model has `f = 0` at every scale. Dense
models spend parameter updates to memorize frequent contexts that counters
store for free. The open question is whether the learned base, freed from that
work, closes parts 2 and 3 of the gap at a smaller `P·D`. That needs the
integrated fit of §380, then AWS-matched dense controls. It is not assumed
here.
