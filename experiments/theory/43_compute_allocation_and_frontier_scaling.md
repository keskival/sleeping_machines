# Compute allocation, tokenization and frontier scaling

2026-09-30. This note develops the project's resource-allocation hypothesis.
Its propositions are conditional mathematical statements; the proposed scaling
advantage remains an empirical question. Sections continue the global theory
numbering. No frontier model has been trained in this investigation.

## 280. The architectural opportunity is a larger useful budget space

Let a configuration be

\[
 b=(d,\ell,P,m,k,h,u,V,r),
\]

where these coordinates denote active width, active depth, stored capacity,
memory modes, retrieved candidates, credit horizon, online update budget,
vocabulary size and released tokens per original byte. They interact; they are
not independent knobs by definition. Include data presentations \(D\), precision,
reset policy and implementation in the configuration. Write held-out risk as
\(L(b,D)\), and account for the complete resource vector

\[
 R(b,D)=(C,\text{time},\text{resident bytes},\text{communication},\text{joules}).
\]

For the measured resource caps \(B\), define

\[
 L^*(B)=\inf_{(b,D):R(b,D)\le B}L(b,D). \tag{280.1}
\]

An event architecture's opportunity is to enlarge this feasible set: spend on
useful memory or retrieval without forcing every token through every stored
parameter; spend teaching work where alternatives could improve predictions.
Dense Transformers already vary width, depth, context, data and vocabulary.
Sparse MoE and modern recurrent/attention hybrids also decouple some capacity
from activity. The comparative hypothesis concerns **how much useful control our
implementation adds, at its actual cost**, rather than exclusivity of flexibility.

**Conditional inclusion proposition.** If every reference configuration has an
event implementation with identical predictions and training behavior and no
greater resource use, its feasible set is included and the event optimum is no
worse. This follows by taking an infimum over a larger set. Strict improvement
requires an additional configuration with lower risk at the same caps.

Exact representation alone proves neither the training condition nor the cost
condition. Current affine event memory is not established to contain arbitrary
state-dependent LSTM gates or arbitrary softmax attention at equal cost. Nor can
a more flexible but poorly optimized kernel inherit this proposition.

## 281. Marginal learning value determines where to spend

For differentiable local response surfaces, minimize \(L(b)\) subject to complete
compute \(C(b)\le B_C\), memory \(M(b)\le B_M\) and latency \(T(b)\le B_T\).
At an interior constrained optimum,

\[
 -\partial_i L=\lambda\partial_i C+
                  \mu\partial_i M+\nu\partial_i T. \tag{281.1}
\]

Thus the next increment should go to the highest reproducible reduction in
held-out loss per scarce resource, adjusted for memory/latency constraints.
Raw event counts, gradient magnitude, attention entropy and parameter count do
not themselves estimate this value. Interactions require paired experiments:
more retrieval may help only when the representation can use the result; more
depth may help only when route survival and downstream credit remain adequate.

A simple *fitted surrogate*, not a language scaling law, is

\[
 L=L_0+\sum_i a_i(b_i+s_i)^{-\alpha_i},\quad
 C=C_0+\sum_i c_i b_i,
\]

with positive \(a_i,\alpha_i,c_i,s_i\). Under these assumptions risk is decreasing
and convex. The unique compute-constrained solution obeys

\[
 b_i=\operatorname{clip}\!\left[
 (a_i\alpha_i/(\lambda c_i))^{1/(\alpha_i+1)}-s_i,
 \ b_{i,\min},b_{i,\max}\right]. \tag{281.2}
\]

Choose \(\lambda\) so the full budget, including \(C_0\), is respected. The
[allocation calculator](../compute_allocation.py) implements this conditional
solution. Coefficients must come from measured development interventions, with
uncertainty and held-out predictions. Separable curves omit interactions and
can recommend the wrong budget; remeasure each accepted increment.

For discrete choices, use the paired change in actual loss divided by the change
in measured cost, including setup/search and any different data presentations.
Keep tuning cost in the project ledger even when reporting final-run cost
separately. Fitting a controller and selecting its actions on the same noisy
validation examples can manufacture apparent gain; reserve confirmation data.

## 282. Dormant capacity, padding and discovery have real prices

For an event model, schematic training work is

\[
 C_E=C_{\rm data}+C_{\rm embedding/head}+C_{\rm active}
      +C_{\rm search}+C_{\rm sort/scan}+C_{\rm credit}
      +C_{\rm optimizer}+C_{\rm maintenance}.
\]

Stored parameters and optimizer moments also consume memory. Winner-only
execution can save value computation while candidate scoring, counterfactual
credit or dense optimizer updates still touch many losing parameters.
Unchanged parameters save optimizer work only if the implementation truly
updates the relevant sparse subset and preserves its declared moment semantics.

If an expert is selected independently with probability \(p_i\), then after
\(N\) training arrivals its probability of receiving no direct exposure is
\((1-p_i)^N\). For uniform selection from \(P\) experts this is approximately
\(e^{-N/P}\). Increasing dormant capacity at fixed exposure can increase
untrained capacity rather than predictive power. Real routing is correlated,
so record actual exposure, usable credit, collapse and specialization instead
of treating this approximation as a guarantee.

Dense baselines can pack documents, bucket lengths and use variable-length
attention. Attention padding waste can shrink substantially; dense token maps
remain regular but need not evaluate padded tokens. Compare with these controls
and measure useful-token throughput. Removing padding is an engineering gain
whose scaling depends on the baseline, not a universal dense-model lower bound.

A useful hardware approximation is

\[
 T\gtrsim\max(C/F_{\rm sustained},\; Q/B_{\rm sustained})
          +T_{\rm scheduling/communication}, \tag{282.1}
\]

where \(Q\) is transferred bytes. Lower arithmetic can lose to regular matrix
kernels through poor utilization or irregular memory access. The approximation
does not model all overlap; use measured time/traffic rather than nominal peak
hardware rates. [FlashAttention](https://arxiv.org/abs/2205.14135) is an important
IO-aware dense control.

## 283. Sparse retrieval needs a task-visible error budget

For one normalized attention-like readout, let omitted probability mass be
\(\epsilon\), and let retained and omitted weighted value means be
\(h_K,h_O\). If each value has norm at most \(R\),

\[
 h=(1-\epsilon)h_K+\epsilon h_O,\qquad
 \|h-h_K\|\le2R\epsilon. \tag{283.1}
\]

If the downstream map has Lipschitz constant \(K\) and every final linear
softmax-head row has norm at most \(G\), cross-entropy is \(2G\)-Lipschitz in
its head input. Therefore a single omission intervention has the conditional
bound

\[
 |\ell(h)-\ell(h_K)|\le4GKR\epsilon. \tag{283.2}
\]

This follows from the convex combination and the softmax gradient
\(\sum_j p_jw_j-w_y\). It does not guarantee useful index recall, bounded
learned norms or unchanged future routes. Hard races can change discontinuously;
the downstream Lipschitz assumption may fail at those boundaries. Test actual
paired suffix predictions when this happens. Small attention mass alone is not
a guarantee of a small relative target-probability error.

The design implication is a bounded retrieval arm with measured recall, paired
prediction impact and charged index maintenance. Discovering the top candidates
by evaluating every query/key pair would preserve the original search cost.
Occasional full/retrieval audits belong in training cost. Candidate selection
uses observed prefixes, never the target whose loss determines its utility.

## 284. Modern tokenization and position belong in the architecture budget

The 27-character benchmark and 131-token causal prefix control are useful local
contracts. They are not a frontier tokenizer. Introduce one fixed modern
byte-level BPE or unigram tokenizer for all learned architecture arms, fitted
on train-only documents or obtained at a recorded immutable revision. Preserve
Unicode, byte fallback, document boundaries, vocabulary and preprocessing hashes.
[Subword units](https://arxiv.org/abs/1508.07909) motivate this control;
[Tokenizers](https://github.com/huggingface/tokenizers) supplies maintained tooling.

For \(B\) original bytes and token rate \(r(V)\), a schematic vocabulary tradeoff is

\[
 C/B\approx r(V)[c_{\rm body}+c_{\rm head}(d,V)]
             +C_{\rm tokenizer}/B. \tag{284.1}
\]

A larger vocabulary can reduce deep event arrivals while enlarging embeddings,
the output projection/normalization and rare-token estimation burden. Charge
backward and optimizer work too. Tied embeddings save storage, not automatically
output arithmetic. Report original bytes, tokens, presentations and end-to-end
quality/cost; perplexities across different tokenizers are not comparable.

Offline BPE segmentation need not remain unchanged when a string is extended.
Causal masking at token boundaries does not prove per-byte stopping-time
causality. Standard token NLL divided by original bytes is a coding score for
the canonical segmentation; it need not be the marginalized probability of the
raw string over every possible tokenization. Declare that interpretation and
EOS treatment. Exact per-character scores in §§269–275 retain their own causal
prefix protocol. A raw streaming tokenizer needs a verified release/buffering
rule, partial-prefix handling and measured latency.

Position is already present in event memory: damped coordinate-pair rotations
produce relative-time phase. In complex coordinates,

\[
 z_t=e^{(-\rho+i\omega)\Delta p_t}z_{t-1}+u_t. \tag{284.2}
\]

Unrolling yields a sum weighted by relative distances \(p_t-p_j\). This shares
the relative-rotation idea with [RoPE](https://arxiv.org/abs/2104.09864), but does
not implement arbitrary content-dependent softmax attention. The repository's
`rotating_memory.py` and `stream_language.py` already use temporal phase.
Compare no phase, fixed multiscale phase and learned phase under the same shared
token coordinates; give the Transformer an appropriate RoPE configuration.

Keep token index or original-byte position separate from learned scheduling
delay. Otherwise learning a route can unintentionally change linguistic
distance. Long-distance phase aliasing, decay, numerical precision and reset
behavior require tests. A shifted zero-initialized stream has relative-position
equivariance only when all other features and scheduling rules respect that
shift; it is not automatic for the full model.

Affine transitions compose as \((A_2A_1,A_2u_1+u_2)\). Input-known diagonal or
2-by-2 rotation blocks permit efficient scans, parallel gradients and recurrent
decode. Dense matrices retain associativity but can make composition expensive.
State-dependent coefficients are a different recurrence. Generated tokens are
unknown until emitted, so parallel fitting remains compatible with sequential
generation; neither tokenization nor position changes that fact.

## 285. Online learning is another priced control, not an exclusive privilege

Let \(f_t(w)\) be the causal next-token loss of a fixed-feature adaptive readout.
For convex losses, gradients bounded by \(G\), projected SGD in a convex set of
diameter \(R\), and comparator sequence \(u_t\) with path variation
\(V_T=\sum_{t<T}\|u_{t+1}-u_t\|\), the standard distance-potential argument gives

\[
 \sum_t[f_t(w_t)-f_t(u_t)]
 \le \frac{R^2+2RV_T}{2\eta}+\frac{\eta G^2T}{2}. \tag{285.1}
\]

The projection inequality telescopes; changing comparators adds at most
\(2R\|u_{t+1}-u_t\|\) per step. This motivates bounded local adaptation under
domain drift and a cost-controlled update rate. It does not prove that
adaptation beats a well-trained frozen predictor on stationary text. Deep,
nonconvex changing-feature learning falls outside these assumptions. The current
unprojected count-mixture pilot is not an implementation of this guarantee.

Predict and record loss before revealing the target; update only subsequent
predictions. Compare equal observations and persistent cache behavior. Begin
with local memory/head adaptation so old cached features have explicit meaning.
Backbone changes require declared recomputation, reset or retained-old-state
semantics, all charged. Transformers and attention/recurrent hybrids can also
adapt between tokens/chunks: [TTT layers](https://arxiv.org/abs/2407.04620) and
[Titans](https://arxiv.org/abs/2501.00663) are relevant controls.

## 286. A widening lead requires a measured scaling condition

For illustrative fitted curves \(L_E=L_{E,\infty}+A_EC^{-\alpha_E}\) and
\(L_R=L_{R,\infty}+A_RC^{-\alpha_R}\), a win at \(C_0\) does not determine
floors or exponents. With a common floor, the excess-loss ratio is

\[
 \frac{L_E-L_\infty}{L_R-L_\infty}
   =\frac{A_E}{A_R}C^{-(\alpha_E-\alpha_R)}. \tag{286.1}
\]

The ratio improves with budget only if \(\alpha_E>\alpha_R\) in the fitted
regime. Even then the *absolute* loss gap can shrink as both approach their floor.
Different floors, data saturation, communication and utilization can cause a
crossover. Fit data and capacity jointly: the
[compute-optimal training study](https://arxiv.org/abs/2203.15556) shows why
holding data fixed while growing parameters can misallocate a budget.

Measure the resource needed to reach several fixed quality targets, repeat the
promising budget points and predict a larger held-out budget before running it.
Compare dense, sparse MoE, selective SSM and attention/recurrent hybrids, rather
than only old character LSTM/Transformer screens. A lower cost intercept is
valuable even with equal exponents. A frontier claim needs actual modern
large-scale models, representative data/tasks and end-to-end resources.

## Development decisions from this note

1. Upgrade shared tokenization, positional controls and dense kernel/packing
   baselines before attributing a language advantage to sparse computation.
2. Preserve a parallelizable input-known affine core and persistent decode;
   add bounded retrieval and local adaptation as separate interventions.
3. Sweep stored capacity at fixed active work **and** measure useful exposure;
   independently sweep retrieval, memory, credit and depth at fixed resources.
4. Fit local marginal-value curves from paired development measurements, reserve
   confirmation data, and test the predicted allocation against uniform scaling.
5. Scale only measured quality/resource improvements. Publish failures and
   reversals. The [frontier protocol](../FRONTIER_COMPUTE_PROTOCOL.md) specifies
   gates; this note supplies neither frontier superiority nor a promise of it.
