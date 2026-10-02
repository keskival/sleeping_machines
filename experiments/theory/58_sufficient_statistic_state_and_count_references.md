# Sufficient-statistic state, escape races and count references

[Theory index](../THEORY.md) · Global sections 376–381. Written 2026-10-02 on host `curie`.

This note starts from a measurement that changes how every completed language
result in this repository should be read. Then it derives why it happens and
which substrate-native mechanism it calls for. Read alongside §§90, 94–95
(the original count-and-mixing design), §108(e) (E79's geometric race mixer),
§§299–302 (integrated sparse temporal language) and §§330–336 (native addressed
event learning).

## 376. Count references on the shared language protocol (measurement)

**Protocol.** The integrated, episodic, native and carrier language results
fit on `text8[0:N]` and score all 8,191 targets of
`text8[90,000,000:90,008,192]` from an empty state (development hash
`65efabd8ceea…`). `experiments/count_reference_language.py` scores closed-form
smoothed context counts on exactly these windows. It ran under the guarded
queue `curie_count_reference_language_20261002T012000Z` (90 s, 149 MB RSS). Result:
`experiments/results/count_reference/curie_count_reference_language_20261002T012000Z.json`.
The counts are deterministic, so they have no seed variance. The order is a
hyperparameter, and the minimum over orders is dev-selected. The neural rows
also select an epoch on this development window. Every row in the table uses the
same units and the same 8,191-target denominator.

| Fit chars | KN counts, frozen (order) | Best completed fitted model | Native model |
|---:|---:|---:|---:|
| 2,048 | **3.615** (4) | 3.722 (episodic write-credit a.25) | 3.765 |
| 8,192 | **3.081** (4) | 3.311 (episodic receiver, semantic) | 3.557 |
| 32,768 | **2.704** (5) | 3.106 (AWS full sparse, p32) | — |
| 131,072 | **2.349** (5) | 2.572 (input-gated carrier, w256) | — |
| 1,048,576 | **2.007** (6) | 2.210 (input-gated carrier, w128) | — |

Interpolated Kneser–Ney with discount .75 beats every completed fitted model at
every fitting size. The margin ranges from .107 to .402 bpc. KN's order sweep is
flat near its optimum: fixed order 5 is within .01 of the minimum from 8K
upward. The KN fit performs `N·(K+1)` table increments and no passes. The
neural fits charge from 3.8 GFLOPs (native 2K) to 8,325 GFLOPs (1M carrier).

**Stream statistics.** The same counting rule, *with no fitting data at all*,
updated causally over the development stream, scores **2.884 bpc** at order 3.
Our native model also carries persistent state through this stream with frozen
weights. A fixed state-update rule (counting) therefore extracts more from the
8,191-character stream than every fitted model with up to 32K fitting
characters. The adaptive variant, which initializes from the fit counts and
continues counting, reaches 2.699 at 8K and 2.036 at 1M.

**Scope.** This covers one development window and character text8. These are
reference predictors, not proposed models and not neural controls. Small-N
character modelling is a regime where count estimators are known to be strong.
This does **not** show that counts beat neural models at scale: large
Transformers reach far lower bpc on text8 with 90M characters. It shows that, on
our own protocol, *none of our completed language fits yet surpasses
closed-form statistics of the same fitting data*. The previously reported
2.572 / 2.210 bpc "learned representations" headline and the native/KV 6.02×
work comparison both sit above this bar. E79 (§108(e)), which mixed count
experts, was the one line that included this information channel. The
integrated substrate dropped it.

## 377. Count-gated addressed state is the exact Bayesian estimator

Take one persistent address `a` (a context, receiver or stream key) that
receives symbol events `y ∈ {1..A}`. Its state is a predictive distribution
`s_a` and an occupancy counter `n_a`.

**Proposition 377.1 (count-gated write).** With `s_a ← s_a + (e_y − s_a)/(n_a + 1 + αA)`
followed by `n_a ← n_a + 1`, starting from `s_a = 1/A` and `n_a = 0`, `s_a`
equals the Dirichlet(α) posterior predictive `(c_a + α)/(n_a + αA)` after every
event.

*Proof.* By induction: `(c+α)/(n+αA) + (e_y − (c+α)/(n+αA))/(n+αA+1)` equals
`(c+e_y+α)/(n+1+αA)` coordinatewise. ∎

This is the core substrate operation: a small message `e_y` mixed into
persistent memory through a gate, touching only the addressed state. The gate is
not free, though. It must be `1/(n_a + 1 + αA)`, a function of an **occupancy
statistic** that the state has to carry.

**Proposition 377.2 (a fixed gate is inconsistent).** Let `s ← (1−g)s + g e_y` with
fixed `0 < g < 1` on an iid stationary source `q`. Then `E s → q` and `Var s_c → g q_c(1−q_c)/(2−g)`.
For symbols with `q_c ≫ g`, the expected excess log-loss per visit tends to the
constant `≈ g(A_eff − 1)/(2(2−g))` nats. The count-gated estimator's excess is
`≈ (A_eff − 1)/(2n)`, so its cumulative regret is `((A_eff−1)/2) log n`, against
linear regret for every fixed `g`. Symbols with `q_c ≲ g` are worse still,
because their estimate decays geometrically toward zero between occurrences
("support collapse").

*Check.* Monte Carlo, `A = 27` with a Dirichlet(.3) source:

| visits n | count excess KL | (A_eff−1)/2n | fixed g=.05 | fixed g=.01 |
|---:|---:|---:|---:|---:|
| 100 | .104 | .105 | .364 | .161 |
| 1,000 | .0125 | .0105 | .518 | .076 |
| 4,000 | .0032 | .0026 | .518 | .076 |

The fixed-gate floors exceed the variance formula because of support collapse.

**Corollary 377.3 (gates must see occupancy, or time).** A content-dependent
gate `g(x, s)` cannot implement `1/n` unless `n` is recoverable from `(x, s)`.
Learning such a dependence by gradient requires credit that spans all `n`
previous visits to the address. That is far beyond the 16- to 64-character
truncated credit windows of the completed language fits (§§308–310). The
schedule therefore has to be **structural**, not learned. Nonstationary streams
call for a time-decayed count `n_a(t) = Σ_i exp(−(t − t_i)/τ_a)`. This count is
exactly the leaky elapsed-time state the substrate already computes, with a
learnable time constant τ_a that trades estimation variance against drift.
Elapsed time computes the forgetting, which is the "time performs computation"
principle applied to sufficient statistics.

## 378. Hierarchical backoff is a cascade of escape races (exact)

Let the context suffixes of lengths `K, K−1, …, 0` be `K+1` addresses touched
by each event. At address `h`, run exponential clocks with rate `max(c_hy − D, 0)`
for every stored symbol `y`, plus one **escape clock** with rate `D·T_h`, where
`T_h` is the number of distinct stored symbols. If a symbol clock fires first,
emit that symbol. If the escape clock fires first, defer to the race at the next
shorter address. After order 0, use the uniform distribution.

**Proposition 378.1.** The first-arrival distribution of this cascade equals
interpolated absolute discounting `p_h(y) = max(c_hy − D, 0)/n_h + (D T_h/n_h) p_{h'}(y)`.
With Kneser–Ney continuation counts at the lower addresses, it is interpolated
Kneser–Ney. With rates `c_hy` and escape `T_h`, it is Witten–Bell / PPM-C.

*Proof.* The total rate at `h` is `Σ_y max(c_hy − D, 0) + D T_h = n_h`, because
every stored count is at least 1 > D. The competing-risks formula (§94) gives
symbol `y` probability `max(c_hy − D, 0)/n_h` and escape probability `D T_h/n_h`.
Recurse. ∎ Tested by sampling in `tests/test_count_reference_language.py`.

Consequences for the substrate:

1. **Time encodes evidence.** The expected decision latency at `h` is `1/n_h`.
   Well-attested contexts answer fast. Novel ones wait, escape and consult
   coarser addresses. Arrival time therefore carries the posterior's
   confidence, which is the speed/accuracy reading of P1–P3 in a closed-form
   setting.
2. **Capacity beyond activity, exactly.** Each event touches `K+1` addresses,
   independent of table size. The table grows with distinct contexts (§95(b))
   while per-event work stays `O(K)`.
3. **Escape is an unrealized alternative.** Learning the discount `D`, the escape
   rates or the base measure needs credit through the losing branch whenever a
   symbol clock wins. This is §§19/57 counterfactual credit with a closed-form
   likelihood, not a surrogate.

*Attribution.* Escape probabilities come from PPM (Cleary & Witten 1984),
interpolated smoothing from Witten–Bell (1991), Kneser–Ney (1995) and
Chen & Goodman (1999), and the Bayesian reading of KN from the hierarchical
Pitman–Yor process (Teh 2006; the Sequence Memoizer, Wood et al. 2009). The
race representation follows from competing risks. What is new here is placing
these estimators inside the sparse addressed temporal substrate, together with
the consequences in §§379–380.

## 379. A learned base measure receives responsibility-gated credit

Replace the uniform floor, or any coarse level, by a learned predictive
`q = softmax(z)`. An example is the native stream model's output. At the top
address, `p(y) = a_y + e_h q_y`, where `a_y = max(c_hy − D, 0)/n_h` and
`e_h = D T_h/n_h`.

**Proposition 379.1.** `∂ log p(y)/∂z = r_y (e_y − q)` with
`r_y = e_h q_y / p(y) ∈ [0, 1]`, the posterior probability that `y` arrived
through the escape branch. The learned model receives the ordinary
cross-entropy gradient scaled by its responsibility. (Autograd identity
test in `tests/test_count_reference_language.py`.)

**Corollary 379.2 (bounded sparse learning and inference).**
(a) Skipping backward work for events with `r_y < ε` changes each
example's **logit** gradient by at most `ε‖e_y − q‖₂ ≤ ε√2`.
For parameters θ this becomes `ε√2‖∂z/∂θ‖_op`; the logit bound alone
does not bound the parameter update or accumulated sequence gradient.
(b) Skipping the learned forward pass entirely, by replacing `q` with any
fallback, changes the prediction by at most `e_h` in total variation.
These bounds concern one predictive distribution at fixed state. A recurrent
base may still need to process this event and receive future adjoints even
when its current predictive contribution is small. Skipping the whole
transition requires a separately proved state/credit contract.

**Measured on the shared development stream** (KN tables, top-level
responsibilities, with lower-order counts standing in for `q`; a learned `q`
that predicts better would raise `r` somewhat):

| Fit chars (order) | mean r | events r < .05 | mean e_h | events e_h < .05 | unseen top context |
|---:|---:|---:|---:|---:|---:|
| 8,192 (4) | .745 | 3.1% | .685 | 2.6% | 43.6% |
| 131,072 (5) | .506 | 16.4% | .449 | 14.9% | 22.6% |
| 1,048,576 (6) | .372 | 33.2% | .328 | 32.1% | 16.7% |

The fraction of events that need learned credit or inference **falls as data
grows**, because the tables absorb frequent contexts. This scaling of learning
sparsity comes from data rather than from forced dormancy. It is a candidate
mechanism for the thesis's "capacity beyond activity" in learning work as well
as inference work. It is a prediction for a learned base, not yet a measured
training saving.

*Order caveat.* These fractions depend strongly on the table order `K`. A
small `e_h` means the composition *ignores* the learned base on that event.
It does not mean a base using information beyond `K` characters could not
help there. The table therefore uses each size's count-optimal order. If the
learned model modulates the escape rate (a learned per-context discount, so
it can override confident tables when longer context disagrees), the skip
bounds apply to the learned escape instead. Quality and work then trade
through `K` and that gate.

## 380. What this changes, and the integrated test it calls for

**Diagnosis.** The persistent state of the integrated and native models is
updated by learned gates under bounded credit. Section377 proves inconsistency
for a **fixed-gate iid estimator**, not for every learned recurrent gate: a
learned state can encode occupancy implicitly. The2.884bpc dev-only counting
result shows useful stream statistics that the completed fits do not exploit
as well. Missing reliable accumulation is therefore a concrete estimation
hypothesis, alongside optimization, credit and representation limits. The
completed mechanism variants (heads, repeated arrivals, write credit, reception
clocks) did not explicitly enforce the count-estimation identity. This is consistent with each of those variants missing
its .02 bpc gate.

**Proposed integrated model: count-carrying receivers.** The concrete failure
it addresses is that every completed fit loses to closed-form counts of its own
fitting data, and that frozen-weight persistent state underuses the stream.

- *Retains* everything: the native eight-block stream model, its races, sparse
  addressed receivers, key/value separation, counterfactual credit and
  persistent memory.
- *Adds* per-address occupancy (or time-decayed occupancy) and a cascade of
  escape races over observed context-suffix addresses (§378). The native
  model's predictive serves as the learned base measure (§379). Count state is
  updated during fitting **and** during evaluation, as persistent state with
  frozen weights. Learnable quantities are `D`, the time constants τ, and the
  native weights.
- *Removes* nothing. It is not a substitution of a dense primitive. Counters,
  escape races and addressed sparse writes are substrate operations.
- *Inference work* is the native forward pass plus `O(K)` lookups, with an
  optional skip under the §379.2(b) bound. *Learning work* is
  responsibility-weighted, with an optional skip under the (a) bound. Table
  memory is reported separately as capacity, not activity.

**Contracts before any fit:** Proposition 377.1 exactness; race-cascade
equality (378.1); the responsibility gradient (379.1); normalization; and a
causality check that development counts update only after scoring each target.

**Required comparisons**, all on 2K/8K first with the existing protocol and
seed 6, and the same units and denominator: KN-alone (frozen and stream-adaptive),
native-alone (completed), and count-carrying native. Promote only if the
integrated model beats **both** KN-alone stream-adaptive (2.699 at 8K, order 3
WB) and native-alone by ≥ .02 bpc. Charge table increments and lookups beside
FLOPs. Larger-N and AWS dense controls follow only after that gate.

**Outcome, 2K (2 October; `curie_count_carrying_D2048_K4_H2_d16_depth8_s6_20261002T020000Z`).**
The count-carrying native model completed at **2.734 bpc** with **3.753** whole-fit
GFLOPs. Native alone scored 3.765 bpc with 3.778 GFLOPs, so the composition is
1.031 bpc better at equal counted fitting work. It is .042 better than the
stream-adaptive count reference (2.776), and it clears the declared gate. The
same composition with the *untrained* native base scores 2.741, so training the
native base adds only **.007 bpc**: at 2K almost all of the gain comes from the
count receivers and their persistent stream state. Learned escape parameters
moved little (`D` .73–.75, `θ` .94–1.13). One seed. §388 explains the small
learned contribution.

**Supremacy map.** On text8, the gap from closed-form counts (≈2.0 bpc at 1M)
to strong Transformers (≈1.1 at 90M) has three parts:

1. estimation for frequent contexts, which counts solve exactly at `O(K)` work;
2. generalization across contexts (shared representations), which is what a
   learned base must deliver;
3. long-range content retrieval (copy/induction; E79's copy expert, E61 race
   retrieval).

A credible advantage claim for this substrate has to win parts 2 and 3 at
lower total work while keeping part 1 exact and cheap. Section379 derives shrinking immediate predictive responsibility; turning
that into shrinking executed learning work remains a state/credit hypothesis. That is the
specific, testable route by which sparse addressed temporal state could beat
dense models on quality per unit of training work. Dense models typically train shared parameter maps to represent frequent
contexts; counters instead pay integer writes, lookup and table storage.
This is an alternative resource boundary, not free work or a dense-model
lower bound. Every
language comparison table should carry the KN reference row from now on.

## 381. Large-data calibration: the dense controls sit at count level

`experiments/count_reference_scale.py` (guarded queue
`curie_count_reference_scale_20261002T014500Z`, 315 s, under 1 GB RSS) builds
interpolated KN tables (fixed `D = .75`, and modified KN with count-of-count
discounts, untuned) from `text8[0:N]`. It scores them on the E64 control test
segment `text8[95M:96M]` (999,999 targets), cold, as the AWS LSTM/Transformer
controls were scored. Result:
`experiments/results/count_reference/curie_count_reference_scale_20261002T014500Z.json`.
A first queue (`…013500Z`) was stopped because of a table bug and wrote no
result. Its queue file records why.

All rows below score the same 999,999 test targets, in bits per character.

| Fit chars | mKN counts, order7 | AWS LSTM | AWS Transformer |
|---:|---:|---:|---:|
|10M|1.788|1.799|1.908|
|90M|1.653|1.661|1.604|

The target-dependent historical E79 mixtures are quarantined and excluded
from this comparison. Corrected E173 at10M scores1.727 without word context
and1.719 with a causal word key. Corrected90M comparison is open.

The controls are a 1.2M-parameter LSTM and a 3.2M-parameter, four-layer
Transformer, each one seed. The count order is not tuned on test: order 7 is
the largest order built, and the curve is still falling at 90M.

**Readings.**

1. The repository's dense controls are within ±.05 bpc of untuned
   closed-form counts. They are useful matched-protocol controls, but they are
   **not frontier bars**. Published large Transformers report about 1.1 bpc on
   the standard text8 test split (Transformer-XL 1.08; Dai et al. 2019). The
   E64 segment is the first million characters of that split, so the numbers
   are approximately, not exactly, comparable.
2. The historical E79 route is motivation, not valid90M evidence. Its
   causally corrected10M successor E173 preserves a positive count/copy
   mixture comparison against saved controls. Counts are therefore a useful
   information path to retain, but the magnitude of a90M mixture advantage
   must be measured under the repaired protocol. Generalization and retrieval
   remain the missing learned contributions.
3. **Learning sparsity grows with data.** Mean top-level responsibility of the
   next-coarser predictive on the shared development window, at the
   count-optimal or largest order: .75 (8K) → .51 (131K) → .39 (1M, mKN) →
   .28 (10M, mKN-7) → .12 (90M, mKN-7). The fraction of events with
   `r < .05` grows .03 → .16 → .33 → .46 → .73. At 90M, under the composition of
   §379, about three-quarters of events would give a learned base less than 5%
   of its ordinary gradient. The order caveat of §379 applies, and higher orders
   at 90M would raise `r` somewhat.

**Consequence for claims.** Beating the current AWS controls is necessary but
not sufficient evidence of an architectural advantage. Language claims should
be stated against three bars: closed-form counts (now measured at every
`N`), the AWS controls, and published frontier values marked as
non-matched references.


**Review correction, 2 October.** The count-write denominator above now uses
the pre-increment occupancy plus one, consistent with its original induction
proof. Fixed-gate inconsistency and predictive skip bounds have been narrowed
to their actual assumptions; none invalidates the measured count-reference
gaps. These corrections do not change frozen drivers or completed scores.
