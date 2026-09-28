# Vector dynamics and scaling

[Theory index](../THEORY.md) · Global sections 104–106; section numbers remain stable. · Next: [08b memory retrieval and deep learning](08b_memory_retrieval_and_deep_learning.md)

## 104. Event networks as controlled differential equations: signatures, selective jumps, races in time

*Written 2026-09-27. Sources: neural CDEs (Kidger et al. 2020), neural rough DEs and Log-NCDEs (Morrill et al. 2021; Walker
et al. 2024), the linear-CDE theory of selective state-space models (Cirone, Orvieto, Walker, Salvi, Lyons, NeurIPS 2024),
neural jump SDEs (Jia & Benson 2019), EventProp (Wunderlich & Pehle 2021), event-by-event SSMs (Event-SSM: Schöne et al.
2024; S7: Soydan, Zubić et al. 2024), mixture of softmaxes (Yang et al. 2018). Numerical checks: `theory_104_checks.py` →
`results/theory/s104_checks.json`.*

**Question.** Neural ODEs are continuous-depth networks, and neural CDEs are continuous-time RNNs driven by a path. Is there a
connection with event networks beyond the analogy, and does it tell us how event times and payload content should interact?
The answer is yes, and the connection is exact. Four consequences are new here: (b), (d)–(g).

**(a) An event network is a CDE driven by a counting path.** Write an event stream on channels 1..J as the time-augmented
counting path X_t = (t, N¹_t, …, N^J_t), with N^j_t the number of channel-j events up to t, plus payloads ξ (the running sum of
payload vectors). A linear CDE driven by it,
  dZ_t = A₀ Z_t dt + Σ_j A_j Z_{t−} dN^j_t + B dξ_t,
is exactly an event unit. Between events it flows in closed form, Z ← e^{A₀Δt} Z. At an event on channel j it jumps: Z ← D_j Z
+ B u, with D_j = exp(A_j) (Marcus) or I + A_j (Itô). Leaky counters are real diagonal A₀; rhythms are imaginary eigenvalues;
delay lines are the LMU/HiPPO choice. Hold, trigger and reset nodes are D_j = 0 on a component. Every mechanism of §83–§103
whose state is linear between events is an instance. The nonlinearity lives in what generates events (races, thresholds), not
in the flow.

**(b) The signature of the counting path is the set of order-detector counts (proposition).** For a word I = (i₁…i_n) of
channels, the strictly ordered iterated integral
  S^I_{s,t} = ∫_{s<u₁<…<u_n≤t} dN^{i₁}_{u₁} ⋯ dN^{i_n}_{u_n}
counts the ordered n-tuples of distinct events labelled i₁, then i₂, …, then i_n inside (s, t]. That is precisely the
(counting version of the) order detector of §53–§54 and §97. Words that contain the time letter weight the tuples by their
gaps: the word (i, t, j) is Σ over ordered (i, j) pairs of (t_j − t_i).
*Proof.* Expand the integral over the atoms of the jump measures.
*Algebra.* Because the tuples are strictly ordered, the counts multiply by the **quasi-shuffle** (stuffle) product of Hoffman:
S^u·S^v = Σ over quasi-shuffles w of u and v of S^w. The quasi-shuffles are the interleavings plus the merges in which both
tuples use the same event, which requires the same channel. The shuffle product that holds for continuous paths is not the
right one here. Hoffman's exp/log isomorphism maps the quasi-shuffle algebra onto the shuffle algebra of the geometric
(Marcus) signature, so the order-detector counts generate the same function algebra as the signature.
*Consequences.*
(i) Universality: the time-augmented path is tree-reduced, so by Stone–Weierstrass (signature universality: Hambly–Lyons
uniqueness, and Fermanian 2020 as used by Cirone et al.), linear combinations of time-weighted order-detector counts
approximate any continuous function of the event stream uniformly on compact sets.
(ii) An AND of two detectors is a sum of longer detectors (their quasi-shuffles). This is the algebraic reason summed ANDs
worked (§83).
(iii) There are J^n words at level n. This is the P^depth growth of §97, and lazy growth on credit (§85, §93) is sparse
signature selection.
Booleans (min(count, 1)) are nonlinear functions of the counts and inherit the same universality through a readout.

**(c) What one layer can compute depends on the gate (Cirone et al. Thms 4.1 and 4.3, Prop. 4.5, applied to events).** For a
linear CDE with gate path ω and input path ξ:
- ω = t alone (S4/S5/LRU, and Event-SSM on events): the closure is linear filters, ψ(t) + ∫ φ(t − s) dξ_s, so one layer weights
  each past event by a fixed kernel of its age. All selection happens in the nonlinearity between layers.
- Dense A_i with ω containing the input: context-dependent filtering. The closure is Ψ(ω_[0,t]) + ∫ Φ(ω_[s,t]) · dξ_s, with Φ
  any continuous function of the path segment since s.
- Diagonal A_i: the closure is ψ(ω_t) + ∫ φ(ω_t − ω_s) dξ_s, a comparison of two points of the gate path. With the event gate
  ω = (t, N¹…N^J) this reads: **each past event is weighted by any continuous function of its age and of how many events of each
  channel have occurred since.** "Time since the last j" and "count of j since s" are the special cases our hold and counter
  nodes implement.
- Chaining diagonal CDEs recovers the dense closure (Prop. 4.5). In event networks the chain runs through events: a layer's
  output events are the next layer's counting path.
Event selectivity is therefore free. The identity of each event is a natural selective gate, which Mamba-style models have to
synthesize from Δ_t = softplus(W x_t).

**Proposition (exact sleeping execution).** Let A₀ and every D_j be block-diagonal by unit, with D_j = I on units not
subscribed to channel j. Then the lazy simulation reproduces the dense per-event simulation exactly: each unit stores its state
and its last update time τ_u, and is touched only by subscribed events (Z_u ← D_{u,j} e^{A₀,u (t − τ_u)} Z_u + B_{u,j} u).
The work is Σ_events |subscribers(j)| · d² for a dense d-dimensional core per unit (d for a diagonal core), instead of
events · H² for the whole state.
*Proof.* Flows of one block commute with themselves (e^{AΔ₁}e^{AΔ₂} = e^{A(Δ₁+Δ₂)}), and D_j = I off the subscribers. ∎
This formalizes the design constraint of §102, a dense small core inside each unit with sparse events between units, as a
block-diagonal linear CDE with event coupling. Event-SSM and S7 update every state component on every event; the subscription
structure is what the paradigm adds.

**(d) A race unit is an integrate-and-fire neuron with an exponential threshold; its gradient is EventProp's jump term.**
Let a unit's intensity be λ(t) = exp(w · Z(t)) for a CDE state Z. By the time-change theorem it fires first at T with
Λ(T) = ∫₀^T λ = E, E ~ Exp(1): it integrates its rate and fires at a random threshold. By the implicit-function theorem,
  ∂T/∂θ = −∂_θ Λ(T) / λ(T),
the jump condition that EventProp derives with the adjoint method for deterministic thresholds. §103's pathwise gradient is its
constant-rate case. Check (d): IFT −0.6542919449 against finite differences −0.6542919450.
*What the random threshold buys.* EventProp's gradients are exact only at a fixed spike count and are blind to spike creation
and deletion. With a threshold that has a density, the expected loss is smooth across creation and deletion. The pathwise
estimator is unbiased whenever the per-sample loss is Lipschitz in the times. Hence the design principle: **read out functions
of times, not of identities.** Times are continuous in the parameters and identities are not. This is why the time-normalized
readout λ_i T works (§102–§103).

**(e) Proportional hazards separate "which" from "when" (theorem).** Let λ_i(t) = e^{s_i} g(t), where the gain g ≥ 0 is common
to the race and may depend on time and on any state. Then the winner is independent of T, P(winner = i) = softmax(s)_i for
every g, and G(T) = ∫₀^T g ~ Exp(Σ_j e^{s_j}). Consequently e^{s_i}·G(T) is an unbiased local estimate of softmax_i.
*Proof.* In internal time G, the race is a constant-rate race (§101). ∎
*Converse.* If the ratios λ_i/λ_j vary during the race, the identity depends on T.
*Design rule.* Content (query–key match) sets the relative hazards. Context, time, attention gain and urgency set the common
gain. The gain then controls speed without distorting the decision: a race can be run at any tempo.
Check (b): the winner frequencies equal softmax in both the fast half and the slow half of T, with an oscillating g.

**(f) The compensator estimator: local probabilities for any race (theorem).** For arbitrary predictable intensities λ_i(t),
which may be time-varying, interacting or non-proportional, let T be the first event of the superposition. Then
  E[Λ_i(T)] = P(winner = i),   Λ_i(T) = ∫₀^T λ_i,   and Σ_i Λ_i(T) ~ Exp(1).
*Proof.* N_i(t ∧ T) − Λ_i(t ∧ T) is a martingale (Doob–Meyer compensator), and N_i(T) = 1[i wins]; apply optional stopping
(E Λ(T) = 1 < ∞). The sum is the superposition time-changed to unit rate. ∎
Each unit's own integrated hazard at the decision time (its "membrane potential" in (d)) is thus an unbiased estimate of its
winning probability. No normalizer is needed, and averaging R races reduces the variance to 1/R. This generalizes §101
(Λ_i(T) = λ_i T at constant rates).
Check (a), with 5 units and non-proportional exponential-in-time rates: E Λ_i(T) = (0.198, 0.194, 0.382, 0.155, 0.074)
against exact (0.197, 0.194, 0.381, 0.154, 0.073). Its variance is 1.4–23× smaller than the winner indicator's (0.036 vs 0.159, 0.170 vs 0.236,
…, 0.003 vs 0.068). Σ_i Λ_i(T) has mean 1.002 and variance 0.998.

**(g) A race whose logits move during the race is a continuous mixture of softmaxes (theorem).** For any intensities,
  P(winner = i) = ∫ (λ_i(t)/λ(t)) · λ(t) e^{−Λ(t)} dt = E_T[ softmax(log λ(T))_i ]:
the instantaneous softmax, averaged over the race's own decision-time density. With proportional hazards this collapses to one
softmax (e). With logits that drift during the race, for example a unit state flowing by its CDE or other events arriving
while the race runs, it is a mixture of softmaxes indexed by decision time. A single softmax over logits W h has a log-
probability matrix of rank at most d + 1 (the softmax bottleneck, Yang et al. 2018). A race is not bound by this.
*How much depends on deliberation.* Check (c) uses d = 4, 60 outputs, 300 contexts and linearly drifting logits, slowing the
race by a common factor e^{−m}. The share of the centred log-probability matrix beyond rank d + 1 is 0.0001 (m = 0, mean
decision time 0.004), 0.02 (m = 3), 0.14 (m = 6) and 0.22 (m = 9). Fast races are single softmaxes; slow races integrate the
logit trajectory and gain rank. **Deliberation time buys expressiveness at no parameter cost**, a speed–expressiveness
trade-off unique to computing in time.

**(h) What this changes in practice.**
1. **Unit core.** A small, per-unit, selective linear CDE driven by the counting path of the events the unit subscribes to:
   closed-form flow between events, channel- or payload-selected jumps, and exact lazy execution. Its expressiveness is
   characterized by (c) and its capacity by (b).
   *Prior art:* the non-selective (Event-SSM, SHD 95.9%) and selective (S7, SHD 96.3%) versions reach state-of-the-art
   accuracy on SHD, SSC and DVS-Gesture while updating every state component on every event (both select checkpoints on the
   test set, as Schöne et al. note). Our additions are subscription sparsity with exact lazy execution, and races (d)–(g) as
   the between-unit event generators.
2. **Readouts.** Race readouts should use times: proportional hazards for exact softmax behaviour (e), compensators for local
   probabilities (f), and slow races where output rank matters (g), for instance language-model outputs.
3. **Training.** Time-change gradients (d) through every race and ordinary chain rules through the linear flows. The flows are
   linear in the state, so no adjoint jumps are needed inside a unit, as Schöne et al. observe; jumps occur only at races.

**Grading.** (a) and (c) are known mathematics, applied here. (b) is elementary; its value is the identification, the
quasi-shuffle correction and the universality consequence. (d) is EventProp's jump plus the time-change theorem; the "times, not
identities" principle is the useful new reading. (e) and (f) are proved and checked numerically. (g) is proved; its
magnitude is measured on one synthetic family only.

**Test (E71).** SHD with the speaker-held-out protocol (§92): event-CDE units over the spike stream, comparing the non-
selective gate (Event-SSM) against the channel/payload-selective gate (the diagonal closure of (c)), then tonotopic subscription
sparsity with the work counted. The aim is to close the 0.675 → ~0.96 gap with a native, sparse mechanism, and to report the
accuracy–work frontier.
## 105. Time and vectors compute together: content-dependent delays, transported payloads, snapshot emission

*Written 2026-09-28. The architecture the paradigm was aiming at: events carry small vectors, and neither the times nor the
vectors are a side channel of the other. Prior art uses one side only. Delay-learning spiking networks (Hammouamri et al.
2024, 95.1% on SHD; Mészáros et al. 2025, 93.2%) send scalar spikes over static learned delays. Event-SSMs and S7 send vectors
with no delays, updating every state on every event. Spiking Transformers compute attention on clocked binary spike maps.
Attention by content-dependent delays with an exact softmax identity was not found in a search (2026-09-28).*

**Setting.** An event is (t, v) with v ∈ R^d small. On a synapse from a sender to unit j, the message carries a content score
r = q_j·v + c_ij. It is **sent only if r > 0**, and it **arrives after the delay δ = τ_j·r**. Unit j holds z_j ∈ C^n, which flows
as z ← e^{Λ_j Δt} z between arrivals and jumps by B v at each arrival. It fires at the first T with Re⟨w_j, z_j(T)⟩ = θ, and it
emits (T, y = φ(Re C_j z_j(T))). Four couplings follow: content → time (delay and gate), time → content (the flow transports each
payload: it decays and rotates by its age after arrival), content → time (threshold crossing), and time → content (the
snapshot at the firing time).

**(a) What a unit holds.** z_j(t) = Σ_arrivals e^{Λ_j (t − t_s − τ_j r_s)} B v_s. With a real mode of rate 1/κ this is
e^{−(t − t_s)/κ} · e^{(τ_j/κ) r_s} · B v_s. **A delay is a multiplicative gain in the exponent**: arriving later by τr means
decaying less, by the factor e^{τr/κ}. With complex modes the delay is also a rotation, e^{iω τ r}. Content therefore
reweights and rephases payloads through time alone, with no multiplier.

**(b) Lemma (delay-coded attention; exact).** Keys send at t₀ with scores s_j, all at or above a cut s_c, delays κ(s_j − s_c),
and payloads v_j. Keys below the cut send nothing. A receiver with one real mode of rate 1/κ, read at any t ≥ t₀ + κ(s_max − s_c),
holds N(t) = e^{−(t − t₀)/κ} e^{−s_c} Σ_{s_j ≥ s_c} e^{s_j} v_j. Its count channel (the same mode fed with 1) holds the same
factor times Σ e^{s_j}. Their ratio is **softmax attention over the keys above the cut**, exactly and deterministically.
*Proof.* Substitute the arrival times into the flow. ∎
- The error against full softmax is at most 2 max‖v‖ times the softmax mass below the cut.
- Cost is proportional to the keys that send, and latency is κ(s_max − s_c): the dynamic range of the logits is bought with time.
- Checked numerically: equality to machine precision with all keys sending and with the top half or top quarter sending (the
  error then equals the missing mass: 0.15 and 0.35 in the check).
- **Races and delays are the two ends of one trade-off.** Races (§101–§103) compute the softmax by *sampling* (fast, noise 1/R).
  Delays compute it by *waiting* (exact, latency proportional to the logit range).

**(c) The semiring.** Leaky integration of delayed arrivals gives κ log N(t) + t = κ · log Σ_s exp((t_s + δ_s)/κ + log w_s).
This is a log-sum-exp over arrival times, where delays add and weights enter as log-offsets. Its zero-temperature limit is
max-plus (latest arrival), while first-arrival races give min-plus (§98). **A delay network computes in the log semiring
natively**, and attention is that semiring's product of scores with values. Composition across layers adds delays and
multiplies gains: products are computed by adding times.

**(d) Emission couples back.** The firing time T solves Re⟨w, z(T)⟩ = θ: the unit fires when its delay-weighted evidence
suffices, so strong matching content fires it early. The emitted vector is the state at that moment, y = φ(Re C z(T)). It
carries what the evidence was, while T carries how decisive it was. Gradients are local:
- ∂T/∂θ = −∂V(T)/∂θ / V̇(T)  (§104d);
- ∂y = φ′ · Re C (∂z(T) + ż(T) ∂T);
- ∂(arrival)/∂v = τ_j q_j  (content moves arrivals).

**(e) What one layer computes.** In §104c's terms, the weight on a past payload is a function of its age *minus a content-
dependent shift*, φ(t − t_s − δ(v_s, j)), evaluated at a time T chosen by the evidence itself. This is outside the closure of
any linear CDE: the shift is nonlinear in content, and the readout time is state-dependent. It contains:
- static delays (the scalar delay-network limit: v constant, δ fixed);
- Event-SSM-style filtering (δ ≡ 0, emission at every event);
- attention (lemma (b)).

**(f) Attention roles and a trainable sparse path.** The event-state layers need not do attention; query/key/value retrieval is a separate operation. For a query $q_i$ and stored key/value pairs $(k_j,v_j)$, use

\[
s_{ij}=q_i^\top k_j/\sqrt{d_k}+b_{ij},\qquad
\delta_{ij}=\kappa(s_{ij}-s_c),\qquad
y_i=\frac{\sum_{j\in C_i}e^{s_{ij}}v_j}{\sum_{j\in C_i}e^{s_{ij}}}.
\]

The query is the receiver's trainable content template; the key controls the arrival time; the value is the transported payload. With all keys in $C_i$, the delay-coded layer is exactly softmax attention after its value and count channels are divided. For a loss gradient $g_i=\partial L/\partial y_i$,

\[
\frac{\partial L}{\partial s_{ij}}=p_{ij}\,g_i^\top(v_j-y_i),
\]

so the same residual $v_j-y_i$ that teaches which value was useful trains the query and key through $s_{ij}$. This is the crucial trainability signal: a model that only transports whichever sparse route already exists may never learn the missing key match. The exact dense identity proves a Transformer attention layer is representable; it does not make its $O(N^2d)$ comparisons efficient.

To preserve this signal while reducing wasted pair work, let $C_i\subseteq\{1,\ldots,N\}$ be a learned candidate set produced by a shared code, route graph, or coarse key index. Train key/query scores and values on candidates, while separately training the candidate mechanism against dense teacher mass or the task gradient. Begin with broad candidate coverage; progressively lower the candidate budget only when recall and downstream loss remain stable. Keep near-miss credit for excluded keys close to the selection boundary so the route can recruit a missing useful key. At inference, the target work is $O(|C_i|d)$ value aggregation plus the cost of finding $C_i$; all-pairs scoring hidden inside candidate selection gives no asymptotic saving. A dense training phase is acceptable if its cost is amortized by a materially cheaper deployment model, but report training and inference energy separately.

**An adaptive retrieval interface can choose another computation.** E77 now places a learned causal state route beside key/value retrieval. Its state update is $m_i=\sigma(f_k(k_i))\odot m_{i-1}+\sigma(w_k(k_i))\odot v_i$, with the state at position $i$ using only earlier keys. A learned gate mixes this linear-in-context route with retrieval. Gate one, zero interaction coefficients, and an infinite delay window recover causal softmax attention; gate zero gives a content-gated recurrent memory. This is one attention-capable module inside a heterogeneous network, not a mandate to turn every event-state layer into attention.

Within retrieval, a low-rank query/key interaction lets compatibility and transported values co-adapt:

\[
u_{ij}=(A_h^\top q_i)\odot(B_h^\top k_j),\quad
s_{ij}=q_i^\top k_j/\sqrt{d_k}+a_h^\top u_{ij},\quad
\tilde v_{ij}=v_j+W_h u_{ij}.
\]

Rank $r=\dim u$ controls interaction width. E77 applies this only to retrieval; its event-state layers retain their own temporal computation. Training uses a smooth near-miss window and evaluation a hard score window. The implementation still scores every causal query/key pair, so it has not yet reduced total search work. This is a candidate mechanism, not a measured improvement. Compare ranks 0/1/4/16, state-only, full softmax, and windowed modes; measure quality, query/key/value gradients, active candidates, depth stability, bytes moved, and energy.

**Expressivity floor and depth.** Do not turn every event-state layer into attention. A Transformer-level comparison concerns the complete stack: residual paths, normalization, position information, nonlinear tokenwise computation, retrieval, and the optimizer. E77 now exposes arbitrary event depth and sparse raw-input skip paths; its default two-layer setting is a prototype, not the limit of the architecture. A Transformer-equivalent path through the whole stack plus alternate event-state paths is the goal. Measure layerwise gradient norms, event activity, route coverage, quality, and work as depth grows; the one-layer softmax identity alone does not establish deep trainability.

**Test (E74).** SHD, selected on held-out speakers. A time-vector network: 16-dimensional payloads; 64 + 64 units with 8
complex modes each; layer 1 on tonotopic windows, layer 2 on a random quarter of layer 1; content-gated, content-delayed
messages; snapshot emission; exact spike-time gradients.
- Measures: accuracy, messages and spikes per utterance.
- Ablations: content delays off (δ fixed), gate off (all messages sent), snapshot off (payload = unit identity only).
## 106. Scaling and representation laws for time-vector networks

*Written 2026-09-28. Checks: `theory_106_checks.py` → `results/theory/s106_checks.json`.*

**(a) Work law for delay-coded attention.** In §105(b) a query pays for the keys that send, i.e. the keys above the cut. For
softmax mass 1 − ε, it pays for the smallest set of keys holding that mass.
- *Model.* N keys with scores s ~ N(μ, σ²). Softmax exponentially tilts the score distribution, e^s·N(μ, σ²) ∝ N(μ + σ², σ²), so
  mass 1 − ε lies above s_c = μ + σ² − σ z_ε, with z_ε = Φ⁻¹(1 − ε).
- *High-temperature phase* (σ < σ_c = √(2 ln N)): the keys that must send number
    W(N, σ) ≈ N · Q(σ − z_ε) ≈ N^{1 − (σ − z_ε)² / (2 ln N)},
  and the latency is κ(max s − s_c). The work fraction falls like a Gaussian tail in the *sharpness* σ.
- *Frozen phase* (σ > σ_c): this is the random energy model's freezing transition. The mass condenses on a vanishing fraction
  of keys, following Poisson–Dirichlet statistics with m = σ_c/σ. The count is independent of N as N → ∞ at fixed m, but
  approaches that limit slowly.
- *Checked* at N = 10⁴, 10⁵, 10⁶ and ε = 0.01:
  - the tilted law is within 1–5% for σ ≤ 3 (e.g. 250,389 keys needed against 250,266 predicted at N = 10⁶, σ = 3);
  - it under-predicts near σ_c (×1.1–1.3 at σ = 4, ×1.3–3.5 at σ = 5);
  - at σ = 6 the count grows sublinearly: 63, 149, 801 keys (N^0.55 over this range).
- *Scaling law.* The work of attention is set by how sharp the scores are, not by how many keys exist. A context-length
  exponent below 1 follows whenever sharpness grows with √(ln N). Dense attention pays N regardless of sharpness.
- *Test:* E76 measures the keys needed per query and head in trained character-level Transformers, as a function of context
  position. This gives the empirical work exponent for language with no distributional assumption.

**(b) Time–precision invariant.** Arrival jitter σ_t (substrate noise, or the simulation grid) perturbs each delay-coded
weight by the factor e^{±σ_t/κ}, a relative error ε_w ≈ σ_t/κ. Spanning a logit range Δs costs latency T_lat = κΔs. Hence
  T_lat · ε_w ≈ σ_t · Δs:
precision and latency trade at a rate fixed by the substrate's jitter and the logit range. This is a design equation for
hardware (§9): a clockless core with 10 ps jitter and κ = 1 ns gives 1% weights over a logit range of 10 in 10 ns.

**(c) Symmetries of the substrate (theorem: dilation and shift covariance).** Let a time-vector layer (§105) have units on a
lattice of positions p and log-scales m. Unit (p, m) has:
- mode rates Λ₀ ρ^{−m}, content-delay scale τ₀ ρ^{m} and reset time τ_R ρ^{m};
- gate, write and readout parameters that depend only on the offset between sender and receiver (Δp, Δm), not on absolute
  (p, m).
Then:
1. shifting the input's channels by k lattice steps shifts every spike from (p, m) to (p + k, m) at the same time;
2. dilating the input's times by ρ^j moves every spike from (p, m) at time T to (p, m + j) at time ρ^j T;
3. with the same structure in every layer, the whole network is equivariant to the group of shifts × dilations.
*Proof.* z_λ(ρ t) under the dilated stream equals z_{ρλ}(t) under the original: substitute t_s → ρ t_s in Σ e^{λ(t − t_s − δ_s)}
B v_s, with δ_s scaling with the unit. Potentials, and hence threshold crossings, are unchanged in value and scaled in time.
Emitted snapshots are equal. The gate and write parameters see only offsets. Induct over layers. ∎
*Checked:* a unit with complex modes, content delays, gate and reset, under a dilation of 1.7, fires the same 13 spikes at
exactly 1.7× the original times (error 6·10⁻¹⁴).
*Prior art:* our own §49 (ramp neurons are dilation-equivariant if thresholds scale; here the scale lattice absorbs dilation with no parameter change), scale-invariant temporal histories (Shankar & Howard 2012, log-spaced Laplace memories), scale-equivariant CNNs
(Sosnovik et al. 2020). *New here:* exact covariance for event networks with content-dependent delays, threshold firing and
snapshot payloads. It holds because every mechanism is defined in time, so dilation only relabels scales.

**(d) Why it matters: generalization across speakers.** Speech from a new speaker differs largely by:
- a shift along the cochlear (log-frequency) axis, from vocal-tract length;
- a change of tempo.
Both are group actions of (c). For a model equivariant in its layers and invariant at its readout, a variation that is exactly
a group action costs nothing to generalize over, and the estimation term of the §99 scaling law sees effective data multiplied
by the size of the orbit the data covers. Prediction: the gap between training-speaker and held-out-speaker accuracy measures
the part of speaker variation that is *not* a shift or a tempo change. Equivariant networks should shrink that gap, and our
SHD failures were exactly that gap (§92: 0.73 held-in vs 0.36–0.46 held-out speakers). Weight sharing also divides the
parameter count by the number of lattice sites.

**(e) Work per token: time-vector network vs Transformer.**
- *Transformer:* per token and layer, ≈ 12 d² (projections and MLP) + 2 N d (attention).
- *Time-vector layer:* per token, e_tok events × (fan-out F × gate pass rate g) messages × (d + n d) for the score and the
  write, plus spikes × n d for snapshots.
- *Attention part:* by (a), W(N, σ) · d instead of N d.
- *Sparse routing:* dense projections are replaced by gated routing, with a per-token cost of e_tok F g (n + 1) d, independent
  of the model width.
So the ratio of work at equal function is W(N, σ)/N on attention and e_tok F g (n + 1)/(12 d) on the rest. The first is fixed
by the data's sharpness (E76), the second by the sparsity learned (E74 reports messages and spikes).

**Tests.**
- E75: SHD with the equivariant time-vector network of (c): shared weights across 35 band positions × 3 time scales,
  offset-dependent gates, delays and writes, invariant readout. Compare with E74 (unshared) on held-out speakers, and on the
  train − held-out gap.
- E76: attention work law in trained Transformers (a).
