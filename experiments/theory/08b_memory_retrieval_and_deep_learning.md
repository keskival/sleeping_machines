# Vector memory, retrieval, and deep local learning

[Theory index](../THEORY.md) · Previous: [08 vector memory and deep stacks](08_vector_memory_and_deep_stacks.md) · Global sections 107–111; section numbers remain stable. · Next: [08c sparse attention and depth bounds](08c_sparse_attention_and_depth_bounds.md)

## 107. Two memories: states that select when written, retrieval that selects when read

*Written 2026-09-28. The step from spoken digits to language. Prior art includes xLSTM's scalar and matrix memories
(Beck et al. 2024), its 7B language model (Beck et al. 2025), and a broad xLSTM scaling study (Beck et al. 2025); recurrent
recall–memory trade-offs (Arora et al. 2023, "Zoology"; "Based" 2024); recurrent/attention hybrids (Griffin, Jamba, Samba);
and heavy-hitter key retention (H2O). These are architectural and experimental precedents, not inherited guarantees for E77.*

**(a) A time-vector unit has a restricted exponential-gated accumulator.** For a fixed arrival schedule, one real mode
obeys the affine recurrence $z_k=a_kz_{k-1}+x_k$, where $a_k=e^{-\Delta t_k/\kappa}$ and each arrival contributes its
payload multiplied by its content-derived delay factor. Thus
\[
z(t)=\sum_s e^{-(t-t_s)/\kappa}e^{\tau r_s/\kappa}Bv_s,
\qquad
c(t)=\sum_s e^{-(t-t_s)/\kappa}e^{\tau r_s/\kappa}.
\]
This is algebraically the *same affine accumulator skeleton* as an exponential-gated normalized memory only under
restrictive choices: fixed event topology, identity candidate/payload map, scalar real mode, unit output gate, and gates
tied to elapsed time and content delay. The count channel is a normalizer analogous to a normalized recurrent memory.
Current E74/E77 layers do not implement the full xLSTM sLSTM cell (learned input/forget gates and memory mixing), nor the
mLSTM matrix state that stores key/value outer products and reads them with a query. E77's causal event-Hopfield and
token-level query/key/value retrieval are explicit associative operations, separate from its time-vector accumulator.

The distinction matters for transfer: xLSTM's scale results motivate deep stack design and careful gate/kernel engineering,
but do not establish E74/E77's trainability, scaling, or energy use. What the time-vector paradigm adds:
- messages below the cut are never sent (sparse writes);
- units emit only when they cross threshold (sparse reads, §105d);
- time is continuous, so the gates are functions of real elapsed time.
The sparse claims describe event structure; the current implementation still materializes dense time-by-batch-by-unit state.

**(b) Write-time selection cannot replace read-time retrieval (capacity bound).** A unit's gate uses the payload and the
unit's own query at the moment of writing. It cannot know which future question will be asked. To answer, for any of N
stored key–value pairs, a query that arrives later, any state must hold ≥ N log₂|V| bits: the answers for all N possible
questions are recoverable from it. This is the information bound behind Arora et al.'s recall–memory trade-off. A state of
M units × n modes × b effective bits therefore recalls at most M n b / log₂|V| pairs. In-context recall of arbitrarily many
facts needs **retrieval**: a query sent at reading time, to stored keys that reply.

**(c) Retrieval natively: a race opens a window, and arrivals within it are exponentially weighted.** Stored key units
reply to a query with delay κ(s_max − s), so higher scores arrive **first**; s_max is a bound on scores, known in advance.
The receiver:
1. opens a hold window of length κΔ at the first arrival (§61);
2. weights each arrival a in the window by e^{(t_close − a)/κ}, in a mode that grows over the bounded window;
3. accumulates a count channel alongside.
The ratio of the two channels is **exactly softmax attention over the keys within Δ of the best**. Keys outside the window
are ignored, and every key below the global cut never sends.
- *Error:* at most 2 max‖v‖ × the softmax mass more than Δ below the top.
- *Aggregation work:* the keys within Δ of the maximum, which is the work law of §106(a) with the cut relative to the maximum.
- *Search work:* this does **not** by itself make finding those keys sublinear. Unless the key store has an index or a
  routing structure, every stored key must still compare its score with the query (or an equivalent global search must
  be done). That costs O(Nd) score work for N keys of dimension d, even if only W ≪ N values reach the receiver. A
  massively parallel race can reduce decision latency while still spending linear energy. Sublinear total work requires
  an explicit assumption about key structure and an index that returns a candidate set of size C ≪ N; then score work is
  O(Cd) and payload aggregation is O(Wd). Exact arbitrary-key retrieval has no such guarantee for free.
- *Primitives:* this uses only the race (first arrival), the hold window, and the flow. The race picks where to look, and
  the window decides how much to average.
- *Memory:* key units are the events of the past. They can be retained by credit, keeping keys that were retrieved (the
  pruning of §93, and H2O's heavy hitters), so memory follows use.

**(d) The hybrid, and its cost law.** A time-vector language model has two parts:
- recurrent time-vector layers (a): sparse, constant work per character;
- a delay-coded retrieval layer (c): O(Nd) score work without indexing, followed by O(Wd) payload aggregation; with a
  suitable candidate index of size C, O(Cd + Wd).
Its unindexed work per character is e F g (n + 1) d + Nd + Wd + (spikes) · n d. A Transformer's is approximately
12 L d² + 2 L N d. Where retrieval is sharp (copying, names, induction), W may be much smaller than N, but that only
reduces aggregation unless candidate search is also sparse. E76 measures W on text; E77 must report both score candidates
and retrieved keys. A practical sublinear-work claim also needs an indexed retrieval experiment with recall and quality
measured against exact search.

**(e) Gates are trainable only near the cut.** A message below the cut is never sent, so its gate receives no gradient. The
off-state is absorbing unless shared parameters (the unit's query) lift the message above the cut. This is §57's routing
problem again, and the same remedy applies: near misses, meaning messages just below the cut, carry the counterfactual
signal, and cooled noise on the cut explores. Diagnostic: the fraction of synapses that send at least once, tracked over
training. If it collapses, the gates need a near-miss band.

**(f) What retained event streams guarantee about depth.** E77 preserves every previous event payload and appends the
new events emitted by the next time-vector layer. It also applies the gated Hopfield residual to that accumulated stream.
On a region with fixed event identities, order, and spike topology, write one transition as
\[
U_\ell=\operatorname{concat}(E_\ell,F_\ell(E_\ell,x)),\qquad
E_{\ell+1}=U_\ell+\alpha_\ell R_\ell(U_\ell),
\]
where $F_\ell$ creates the new event payloads and $R_\ell$ is the event-Hopfield correction. The Jacobian of the
inclusion map is $D U_\ell=[I;D F_\ell]$, so
\[
(D U_\ell)^T D U_\ell=I+(D F_\ell)^T D F_\ell\succeq I.
\]
If $\|D R_\ell\|_2\le K_\ell$ and $\alpha_\ell K_\ell<1$, then
\[
\sigma_{\min}(D E_{\ell+1}/D E_\ell)\ge1-\alpha_\ell K_\ell.
\]
This follows because left multiplication by $I+\alpha_\ell D R_\ell$ has minimum singular value at least
$1-\alpha_\ell K_\ell$, while concatenation cannot reduce the norm of any perturbation in the retained coordinates.
Across depth the retained-payload path is bounded below by $\prod_\ell(1-\alpha_\ell K_\ell)$; with
$\alpha_\ell=1/L$, uniformly bounded $K_\ell\le K$, and $K/L\le1/2$, it is at least $e^{-2K}$.

This gives E77 a real identity-inclusion path across event layers, rather than claiming that adding layers alone makes
them trainable. It remains deliberately narrow: it bounds propagation of perturbations in already-created payloads,
not the gradient to newly created payloads, whether useful routes are discovered, or whether hard event creation matches
the fixed-topology derivative. The complete layer condition is addressed in §107(g–h), and §113
composes sparse-versus-dense gradient perturbations across residual depth. Existing §97–§98 mistake
bounds apply to realizable ordered-pattern learners under their stated candidate-route assumptions; they do not
establish a bound for E77's language objective. E77 remains sparse in event-stream structure but its `TVLayer` uses
dense time-by-batch-by-unit state tensors and its retrieval scores all eligible query/key pairs. A genuinely sparse
execution kernel and indexed candidate search remain necessary for the manifesto's energy goal. Depth sweeps (2, 4, 8,
16) therefore track loss, layerwise gradients, active and retained events, route coverage, and search/simulator work.

**(g) Hopfield retrieval gives a depth-scalable local credit rule.** The modern Hopfield/softmax-attention equivalence
is a one-update statement: given stored key/value pairs $(k_j,v_j)$ and query $q$, set
$p_j=\exp(\beta q^\top k_j)/\sum_r\exp(\beta q^\top k_r)$ and $y=\sum_jp_jv_j$. For the
auto-associative case $v_j=k_j$, this is the modern Hopfield retrieval update; separate $k$ and $v$ give its useful
hetero-associative key/value generalization. This distinction matters: a Transformer attention head is a differentiable
associative lookup, while arbitrary learned values need not descend the classical Hopfield energy in query space.

In the auto-associative case, define $\Phi(q)=\beta^{-1}\log\sum_j\exp(\beta q^\top k_j)$. Then
$y(q)=\nabla\Phi(q)$ and $\nabla^2\Phi(q)=\beta\operatorname{Cov}_p(k)\succeq0$. The update is monotone
and its Lipschitz constant is at most $\beta D_K^2/4$. In particular, the repeated retrieval iteration
$q_{t+1}=y(q_t)$ is a contraction when $\beta D_K^2/4<1$, giving a unique fixed point and geometric convergence.
Above that sufficient threshold the proof stops: sharper retrieval and multiple attractors become possible, but
convergence and gradient stability require more structure. For arbitrary hetero-associative values $V\ne K$, the
query Jacobian is the cross-covariance below and need not be symmetric or derive from a scalar potential. E77 therefore
uses one learned retrieval update per event layer plus an identity path, not unanalysed repeated Hopfield settling.

For loss gradient $g=\partial L/\partial y$, direct differentiation gives
\[
\frac{\partial L}{\partial s_j}=p_j\,g^\top(v_j-y),\qquad
\nabla_qL=\beta\sum_jp_j[g^\top(v_j-y)]k_j,\qquad
\nabla_{v_j}L=p_jg.
\]
The score credit is centered: $\sum_j\partial L/\partial s_j=0$. A key is rewarded when its value improves on
the retrieved mean in the direction required by the loss; this trains *where to read* and *what to transmit* through
the same operation. For a fixed memory and key/value diameters $D_K=\max_{ij}\|k_i-k_j\|$ and
$D_V=\max_{ij}\|v_i-v_j\|$, the query Jacobian is a cross-covariance,
\[
D_qy=\beta\sum_jp_j(v_j-y)(k_j-\bar k)^\top,\qquad
\|D_qy\|_2\le\frac{\beta D_KD_V}{4}.
\]
The inequality follows from Cauchy–Schwarz for the covariance and the Hilbert-space variance bound
$\mathbb E\|X-\mathbb EX\|^2\le\operatorname{diam}(X)^2/4$. Crucially, this query sensitivity bound does not
grow with the number of stored events when the key/value diameters are bounded. It gives a concrete, memory-count
independent control variable for retrieval gain: inverse temperature and representation diameter.

**Sharp retrieval and learnable credit pull in opposite directions.** Let $j^*$ be the best key and assume a score
margin $m=\min_{j\ne j^*}q^\top(k_{j^*}-k_j)>0$. At inverse temperature $\beta$, the non-winner softmax mass
obeys
\[
1-p_{j^*}\le (N-1)e^{-\beta m}.
\]
Thus $\beta m\ge\log((N-1)/\epsilon)$ suffices for at most $\epsilon$ non-winner mass. With two competing keys,
the route probability is $p=\sigma(\beta\Delta)$ and $\partial p/\partial\Delta=\beta p(1-p)$. Credit is
largest at the ambiguous boundary $p=1/2$ and decays exponentially once one route dominates. A hard-excluded key
has exactly zero score credit. Sharper retrieval therefore improves selection error while concentrating trainable
credit into a narrow near-tie band; this derives the need for broad early retrieval, near-miss candidates, and gradual
sparsification. Counting only the keys aggregated after a route is chosen misses this learning cost.

Combining the error and Jacobian bounds makes the scaling tradeoff explicit: a worst-case query gain is
$K_q\le\beta D_KD_V/4$ before query/key/value projection and gate gains. If required $\beta$ grows with memory
count to keep fixed error under a fixed margin, a depth guarantee must track that increase. The model can reduce the
pressure by increasing useful score margins, controlling payload diameters, or shrinking the candidate universe with
a learned index that retains softmax mass. An index that secretly computes all-pairs scores does not change the work
law.

**(h) The sequence-level bound depends on key fan-out, not just one-query sensitivity.** The bound above differentiates
one output with respect to its query while holding its memory fixed. In a stack, however, an event is also a key/value source
for later queries. Let payloads be $x_i$, with $q_i=W_qx_i$, $k_j=W_kx_j$, $v_j=W_vx_j$, and let $p_{ij}$ be the causal
softmax weights under a fixed event order and fixed candidate mask. Define

\[
H=\max_j\sum_i p_{ij},\qquad Q=\frac{\beta D_KD_V\|W_q\|_2}{4},\qquad
B=\|W_v\|_2+\beta D_V\|W_k\|_2\max_i\|q_i\|_2.
\]

For $j\ne i$, direct differentiation gives
\[
D_{x_j}r_i=p_{ij}W_v+\beta p_{ij}(v_j-r_i)(W_k^\top q_i)^\top,
\]
so its operator norm is at most $p_{ij}B$. The query block on the diagonal is bounded by $Q$ from the cross-covariance
lemma. Therefore the maximum block-row sum of the full sequence Jacobian $D R$ is at most $Q+B$, and its maximum
block-column sum is at most $Q+BH$. The scalar block-norm majorizer obeys the Schur bound, yielding
\[
\|D R\|_2\le\sqrt{(Q+B)(Q+BH)}.
\]
This is a full sequence bound for the differentiable payload route at fixed topology, not a one-query bound. $H$ is the
maximum attention mass received by any one event across all later queries. It is near one for diffuse use and can grow
with sequence length when many queries reuse the same key; bounded key/value diameters alone do not control it.

For E77's gated residual correction $F_i(x)=g_iW_or_i$, where
$g_i=\sigma(w_x^\top x_i+w_r^\top r_i+b)$, $\|r_i\|\le V_{\max}$ and $\|\nabla\sigma\|\le1/4$ give
\[
\|DF\|_2\le K_F:=\|W_o\|_2\left(1+\frac{V_{\max}\|w_r\|_2}{4}\right)
\sqrt{(Q+B)(Q+BH)}
+\frac{\|W_o\|_2V_{\max}\|w_x\|_2}{4}.
\]
Thus the residual theorem in (g) applies with the measured $K_F$, provided the per-layer step satisfies
$\alpha_\ell K_{F,\ell}\le1/2$. This condition includes query, key, value, output, gate, and cross-query fan-out gains.
It is conservative, but it is computable without forming the full Jacobian. E77 now logs $H$, this sequence-level bound,
its depth-scaled value, and their sum across event layers. Hard changes in event creation/order or top-$k$ membership are
outside the certificate; the bound is local to a region with those choices fixed. It is a condition to measure, not a
claim that deeper E77 models already satisfy it.

Now compose event updates $h_{\ell+1}=h_\ell+\alpha_\ell F_\ell(h_\ell)$, where each $F_\ell$ is
one causal Hopfield message update plus a bounded readout/gate. If the *full fixed-topology Jacobian* obeys
$\|DF_\ell\|_2\le K$ and $\alpha_\ell K\le1/2$, then
\[
e^{-2K\sum_\ell\alpha_\ell}\le
\sigma_{\min}(D h_L/D h_0)\le\sigma_{\max}(D h_L/D h_0)
\le e^{K\sum_\ell\alpha_\ell}.
\]
This follows by bounding each factor $I+\alpha_\ell DF_\ell$ between singular values
$1-\alpha_\ell K$ and $1+\alpha_\ell K$, then multiplying. Choosing $\alpha_\ell=1/L$ yields
depth-independent credit bounds $[e^{-2K},e^K]$. E77's event-Hopfield residual is initialized with this $1/L$
scale; its learned maps and hard spike topology do **not yet enforce** the required uniform $K$. E77 logs learned
delay gain $\beta$, diameter upper bounds, and the sequence-level certificate from (h). The upcoming depth runs will show
whether their measured values satisfy the sufficient condition; the proof itself does not imply that they will.

There is a second structural limit. If candidate set $C(q)$ has dense softmax mass $1-\epsilon$, then
$\|y-y_C\|\le2V_{\max}\epsilon$. But hard top-$k$ gives excluded keys zero score gradient. Therefore dense
Hopfield retrieval is the trainability/recall reference; a sparse index is a separate learned component. The general
counterfactual route-learning problem is already worked out in §19 and §57: a noisy-choice boundary term credits untaken
routes, and cancellation preserves a near-miss charge and best partial window without running another full event path.
Section 112 specializes that existing rule to key/value mixtures and bounds the cheap local loss estimate. The remaining
engineering question is whether a candidate mechanism can expose useful near-misses and payloads without restoring
all-pairs scoring. Teacher-mass distillation can be a diagnostic arm. Top-$k$
aggregation after all-pairs scoring is not a compute saving. E77's new event-level Hopfield update attends only among
emitted events, keeps query/key/value maps separate, uses learned score-to-delay decay, and carries the retrieved
payload into the next event layer. It runs only when events exist, but current candidate scoring is still quadratic in
the number of emitted events; measured work and an index are still needed for a sparsity claim.

This is the mathematical reason to study associative event message passing for depth: (1) Hopfield's centered score
credit makes address and content train together; (2) its local sensitivity depends on score sharpness and payload
diameter rather than memory count; (3) residual step scaling can prevent exponential gradient collapse across depth
under an explicit bounded-gain condition; and (4) event sparsity can reduce *which payloads are updated*, while an
index is still needed to reduce *which keys are searched*. These are separate claims with separate measurements.

**(i) The fixed-schedule time-vector state admits an exact associative scan.** For each batch item, unit, and mode,
the current affine memory update is $z_k=A_kz_{k-1}+x_k$. Here $A_k$ is diagonal (in the current implementation it is
the per-unit complex decay $e^{\lambda}$), and $x_k$ is the sum of payload arrivals assigned to bin $k$. Represent one
step by the affine map $T_k=(A_k,x_k)$. Chronological composition is
\[
T_j\circ T_i=(A_jA_i,\;A_jx_i+x_j).
\]
Because this is composition of maps, it is associative. An exact parallel prefix scan over these pairs returns every
prefix state $z_k$ in $O(Gn)$ work and $O(\log G)$ parallel depth for $G$ bins and $n$ diagonal modes; the scalar count
channel is another scan of the same form. This replaces the serial dependency through the fixed affine state recurrence
without changing its mathematical result or adding all-pairs attention. Since both schedules compute the same
differentiable affine composition, their exact-arithmetic gradients are equal as well; floating-point association can
introduce reduction-order differences. It is a concrete route to parallelizing the memory-state portion of training.

The claim is deliberately scoped. It assumes the arrival schedule and topology are fixed. The scan still writes all
prefix states, so it does not remove the current $G\times B\times M\times n$ dense storage or prove sparse execution.
Spike threshold/reset-trace decisions, event creation, routing changes, and candidate search are separate operations;
the scan alone does not parallelize or sparsify them. An event-sparse version would need sorted arrivals, segment
composition over empty intervals, and explicit accounting for sort/dispatch/bytes. **Next derivation-to-measurement step:**
compare scan and sequential outputs plus gradients on identical fixed event schedules, then run one small scan pilot only
after the shared safe runner is free. Record wall time, peak memory, and bytes moved; correctness alone is not a speed claim.

**Test (E77).** A time-vector character language model on text8:
- two spiking time-vector layers over characters as events (half-integer times, so readouts are causal);
- a causal modern-Hopfield query/key/value update over emitted payload events after each event layer, with a 1/depth residual scale;
- non-spiking readout units giving a state per character;
- one delay-coded retrieval layer (exact softmax over keys within Δ of the best, keys = past states, values carry the next
  character);
- a readout.
It is trained with the same gradients as E74, at 1M and 10M characters, against E64b's converged LSTM and Transformer and
the stage-1 event model. Reported: bits per character, messages, spikes, keys per query, and score candidates per query.
The current dense score implementation evaluates every causal query–key pair; its retrieved-key count is an aggregation
count, not a measured sublinear search cost. The ablation without retrieval measures what (b) predicts it loses, while an
indexed-search variant is required to test the total-work claim.
## 108. The lower envelope: the full event language model is never worse than its best part

*Written 2026-09-28. Standard online-learning results (Cesa-Bianchi & Lugosi 2006; Herbster & Warmuth 1998; KT and
context-tree weighting, Willems et al. 1995), applied to the design of §95–§107. They turn the scaling figure's promise, "the
full design contains the counting stage, so it is never worse", into a theorem, and say where supremacy on language is
guaranteed and where it has to be earned.*

**Setting.** Experts e = 1..E each give a predictive distribution p_e(x_t | x_<t) over the next character. They can be the
native counting and copy experts of E62–E66, the time-vector model of E77, or anything else. The mixer is conserved
multiplicative credit (§83): weights w_e ∝ π_e Π_{s<t} p_e(x_s | x_<s)^η, and p_mix = Σ w_e p_e.

**Theorem (lower envelope).**
(i) With η = 1 (the Bayes mixture), for every sequence, −log₂ p_mix(x_1..T) ≤ min_e [−log₂ p_e(x_1..T)] + log₂(1/π_e).
    Per character, with a uniform prior, L_mix ≤ min_e L_e + log₂E / T. On a 1M-character test with E = 20 experts the
    overhead is 4·10⁻⁶ bits per character.
(ii) With fixed share α (weights mixed toward uniform at rate α after each step), for every sequence and every partition
    into m + 1 segments with one expert per segment, L_mix·T ≤ Σ_segments L_{e_i} + (m + 1) log₂E + (T − 1) h₂(α) +
    m log₂(1/α) + ..., where h₂ is binary entropy. With α ≈ m/T this is O(m log(ET/m)). The mixture tracks whichever
    expert is best *locally* (copying in a repeated passage, the time-vector model elsewhere), so it can beat every single
    expert, not only match the best.
*Proof.* Log loss is 1-mixable, so the Bayes mixture's regret is the log of the prior mass (Cesa-Bianchi & Lugosi, Thm
3.2). Fixed share is the Bayes mixture over switching sequences with a Markov prior (Herbster & Warmuth). ∎

**Consequences.**
1. **The design's scaling curve is the lower envelope of its parts':** L_design(D) ≤ min(L_counting(D), L_copy(D), L_TV(D)) + ε
   at every data size D. Adding a part never hurts, by more than log₂E/T.
2. **Where supremacy is guaranteed: small data.** A KT counting expert over contexts of order ≤ k has redundancy
   ≤ (|A| − 1)/2 · log₂ n_c + 1 bits per context seen n_c times. This is minimax-optimal for bounded-order Markov sources,
   and context-tree weighting extends it to the best tree of orders. Gradient-trained networks have no such small-data
   guarantee. Where the counting mixture already beats a converged LSTM or Transformer (E63/E66 against E64b at 1M and
   10M), the full design beats them too, by (i).
3. **Where it must be earned: large data.** There the time-vector expert (E77) has to reach the Transformer. By (ii), the
   design still gains wherever the parts are locally better, for example exact copies, rare words and names.
4. **Native and cheap.** The mixer is one multiplicative update per expert per character (conserved credit, §83), and sleeping
   experts (those without information) neither vote nor learn (§100 E66).

**What the windowed, selector-keyed Hedge of E63/E66 is.** It is a heuristic variant, tuned on validation, without the
guarantee. E78 reports all three: the Bayes mixture (i) and fixed share (ii), which carry the guarantee, and the tuned
variant.

**(e) The race mixer: products of experts, natively (geometric pooling).** A race over the next character whose clock
rates are exp(Σ_e w_{s,e} log p_e(c)) samples from the geometric mixture p ∝ Π_e p_e^{w_e}. That is logistic mixing, the
engine of the PAQ/cmix compressors. Its log loss is convex in w, so online gradient descent with the exact local gradient
∂/∂w_e = log p_e(y) − E_mix[log p_e] has regret O(√T) against the best *fixed weight vector*, per selector context. The
comparison class is products of experts, which contains every single expert (a one-hot w). A product can sharpen where
the experts agree and cancel where one is confidently wrong, so it can be far below every expert even without switching.
Each weight's update needs only its own expert's log-probability of the outcome and the mixture's expectation of it (a
small dense core over 27 characters).
*Measured before the full runs.* With 300k training characters and 30k test characters, geometric race mixing gives 1.95
bpc, against 2.28 for linear Hedge over the same experts and 2.61 for the best expert. With E78 at 1M, the Bayes mixture
equals the best expert (2.218, as (i) says), and fixed share gives 1.945, below E66's tuned Hedge (1.980).

**Fairness conditions for comparisons with Transformers.** Two properties give the native side an advantage, so both are
reported separately:
- *Copy memory:* it searches the whole test prefix; restricted to 256 characters it matches the E64 Transformer's context.
- *Online adaptation:* the mixer keeps learning while predicting, as compressors are scored; with weights frozen after the
  validation stream it is static, like the Transformer.

**Test (E78).** On the same 1M test characters: the E66 experts (counting orders, word-keyed counts, copy memories) and
E77's time-vector model, each alone and mixed three ways, at 1M training characters and later 10M. Prediction by (i)–(ii):
the mixture is at or below the best part everywhere, and strictly below where the parts' errors differ.
## 109. Race neurons are gated linear networks: provably sufficient local learning

*Written 2026-09-28. Prior art: Gated Linear Networks (Veness, Lattimore, Budden et al., AAAI 2021), backpropagation-free
networks in which every neuron predicts the target by geometric mixing of its inputs' predictions, with weights selected by
a data-dependent context and learned by online convex optimization. They are universal in the limit, competitive with
batch-trained MLPs after one online pass, and resistant to catastrophic forgetting. Their stated limit: neurons do not
learn feature representations.*

**Identity.** A race whose candidates' clock rates are exp(Σ_e w_e log p_e(c)), with the weight vector w chosen by a context
c(t) (which weight set is awake: a sleeping-expert selector, §100), samples from the geometric mixture that a GLN neuron
outputs. A network whose units are such races, each fed the output distributions of the layer below, *is* a GLN. The
race samples; the distribution is also available locally, as each candidate's integrated rate at the decision (§104f).

**Consequences.**
1. **Local learning is sufficient, not merely possible.** Each race neuron minimizes its own log loss of the target, which
   is convex in its active weights, by the exact local gradient log p_e(y) − E_neuron[log p_e]. No error is sent between
   neurons, so there is no credit-assignment problem to solve. By Veness et al., capacity grows with network size and
   context richness, reaching universality in the limit. This is the strongest available answer to "can event networks
   learn without backpropagation": for prediction built from expert opinions, yes, provably.
2. **Every layer's output is a calibrated predictor.** The final neuron's regret against its best input (§108) makes the
   network never worse than its best neuron, which is the lower envelope once more.
3. **Division of labour.** GLN race neurons do not build features. In the full design, features come from time-vector
   layers (§105–§107, trained by spike-time gradients) and from native detectors (order detectors, counting experts,
   copies). The race GLN combines them locally, with guarantees.
4. **Forgetting.** GLNs' credit assignment is known to resist catastrophic forgetting. This bears on our open continual-
   learning weakness (E23: the weight race forgot more than SGD). Untested here.

**Measured (small check: 300k training, 30k test characters).** A three-layer race GLN (6 context-gated neurons over the
native experts, 3 over them, 1 final) gives 2.042 bpc frozen and 1.977 online, against 2.049 and 2.006 for a single race
neuron. The gain is small at this size, because most of the ~1,600 context-specific weight vectors see little data. E81
measures it at 1M and 10M characters.
## 110. Deep event chains with local supervised credit

*Revised 2026-09-28, before the E83/E84 depth pilots.*

Let $H_\ell=F_\ell(H_{\ell-1};\theta_\ell)$ be the event stream emitted by
layer $\ell$, with raw events only at layer 1. In a strictly serial stack, the
final loss gradient for an early layer contains
$J_{F_L}\cdots J_{F_{\ell+1}}$. Exact event-time derivatives do not prevent
this product from contracting, amplifying, or losing rank.

Attach the same task readout $g_\phi$ to each depth during training and use
auxiliary losses, but use only the deepest readout at inference:

$$
\mathcal L=\ell(g_\phi(H_L),y)+\lambda\sum_{k=1}^{L-1}\ell(g_\phi(H_k),y).
$$

For an early layer $\ell<L$, its gradient contains the local term

$$
\lambda J_{\theta_\ell,H_\ell}^{\top}
J_{g,H_\ell}^{\top}\nabla_{g(H_\ell)}\ell,
$$

which has no downstream layer Jacobian product. The final-output term still
travels through every later layer, so inference remains a true depth-$L$
composition. Comparing $\lambda=0$ with $\lambda>0$ isolates whether local
supervision keeps early-layer gradients and activity useful as depth grows.

This is a gradient-path result, not a convergence or representation theorem.
The direct term can still be zero when no useful event fires, all candidate
messages are gated out, or the shared readout has no local sensitivity. The
auxiliary objectives can also interfere: an intermediate representation may
be pushed toward solving the final task before it has learned a useful
composition. Held-out deepest-layer quality, per-layer gradients, and firing
statistics are therefore all necessary. Section 113 addresses a separate question: when a sparse
fixed-support approximation stays close to the dense stack's gradients as depth grows.

The E83/E84 hidden topology is a strict adjacent-layer chain with fixed sparse
candidate masks; there are no raw-input skips above layer 1 and no all-past
attention. The readout uses a fixed sparse source mask shared across taps. We
count both candidate score pairs and accepted messages, since accepted-message
count alone hides the cost of rejected score tests. The time-vector reference
implementation still scans grid/event steps over its state cells; therefore
these pilots do not claim fully asynchronous execution or an energy advantage.

**Test.** E83 applies the paired $\lambda=0$ and $0.2$ conditions to
speaker-held-out SHD; E84 uses the same pair on the frozen day-5 market
validation protocol. At depths 2, 4, 8, and 16, width, data exposure, optimizer
updates, and seed are held fixed. Record deepest-output quality, intermediate
tap quality, layer gradient norms, spikes, candidate score pairs, accepted
messages, state-vector scan updates, RSS, and wall time. A depth claim needs
the deepest output to improve or remain competitive while gradients and event
activity persist. If auxiliary loss helps, repeat across seeds and then tune
its depth-dependent weight; one pilot seed is not evidence of scaling.
## 111. When auxiliary credit helps the deepest objective

The local term in §110 removes downstream Jacobians from an intermediate
layer's auxiliary gradient, but nonzero credit alone is not sufficient. Fix a
layer's parameters $\theta_\ell$ and write

$$
g_\ell=\nabla_{\theta_\ell}\mathcal L_{\mathrm{deep}},\qquad
a_\ell=\nabla_{\theta_\ell}\mathcal L_{\mathrm{aux}}.
$$

For a small SGD step on
$\mathcal L_{\mathrm{deep}}+\lambda\mathcal L_{\mathrm{aux}}$,
$\theta_\ell' = \theta_\ell-\eta(g_\ell+\lambda a_\ell)$, Taylor expansion
gives

$$
\mathcal L_{\mathrm{deep}}(\theta_\ell')-\mathcal L_{\mathrm{deep}}(\theta_\ell)
=-\eta\left(\lVert g_\ell\rVert^2+
\lambda\langle g_\ell,a_\ell\rangle\right)+O(\eta^2).
$$

Let $r_\ell=\lVert a_\ell\rVert/\lVert g_\ell\rVert$ and let
$c_\ell$ be the cosine between these gradients. The first-order deep loss
decreases exactly when $1+\lambda r_\ell c_\ell>0$. Aligned auxiliary credit
($c_\ell\ge0$) preserves descent. If the gradients conflict, it is safe only
when $\lambda r_\ell|c_\ell|<1$. As depth suppresses the deep gradient,
$r_\ell$ can grow; a fixed auxiliary weight that helps at depth 2 can then
overwhelm the objective at depth 8. If $g_\ell=0$, an auxiliary update can
still train the intermediate prediction, but this first-order argument gives
no guarantee that it improves the deepest prediction.

This gives a direct diagnostic instead of treating larger gradient norms as
proof of trainability: measure per-layer deep-loss norm, auxiliary-loss norm,
and their cosine on the same batch. E83/E84 now record this probe on the first
training batch of each epoch, alongside final-output quality and activity. The
probe is for plain gradient directions; AdamW's adaptive preconditioning and
weight decay can change the actual update direction, so the paired held-out
result remains decisive. If auxiliary gradients are large and anti-aligned,
reduce or schedule $\lambda$, or improve the intermediate targets; do not add
depth supervision blindly.

M3 is the most informative experiment in this list. It says which term carries the
learning signal, whether our estimator of it is good, and what fired-only is missing,
on a network small enough that every term can be computed exactly. It should run
before E11. E11 then becomes "replace each estimated term with the better local one

that M3 identifies".
