# Native addressed event learning

## 330. Preserve the substrate, stop requiring an attention scaffold

The user explicitly redirects the primary construction toward its own strengths.
The existing parallel KV language fits learn, but their memory/credit additions
produce small pilot gains and have not justified confident scaling. This is a
concrete failure of that implementation/protocol, not a reason to suppress the
race-attention algebra or earlier strong results.

The new integrated branch processes an observed (address, time, content) through
independent receiver races in eight blocks. Content mixes with addressed deep
context and rotating/decaying receiver state; each winning receiver commits one
update; unrealized alternatives teach score selection. Keys and content/value
maps remain distinct. Per-head query and receiver value/gate maps are independent;
channels align by their actual arrival and retain a learned next-block mix.

The per-position KV bank and its historical attention query are removed from
this branch. The branch consequently does not reproduce attention averaging or
store all past observations. Old attention/receiver comparisons remain preserved.
The new question is whether useful persistent event representations and selective
state are a better use of this substrate than a conventional attention scaffold.
Known input projections, gating and addressed memory are supporting primitives;
the combined temporal/routing construction and tested consequences are the case.

For source a, let r_a be the preceding deep output's latest arrival. Admit its
next input at max(t_input,r_a); evolve retained channels to that read time.
Other sources have their own r_b. This is a local causal dependency, not a
periodic global clock or a global next-event completion barrier. Queries are
explicit observed deadlines. During silence no state update is scheduled by
the emulator; decay/rotation is evaluated analytically on the next read.
The internal head/block joins are retained and charged; no clockless chip is
claimed merely because the software schedule is event-driven.

## 331. Capacity, activity and complete learning boundaries

With S observed source populations, H heads, depth L, pool P and width d/head:

    available receivers = S L H P,
    selected commits / event = L H,
    scored keys / event = L H P,
    training proposals / event = L H P.

Incoming projection, channel mixing, addressed-context reads, local evolution,
readout and numerical clocks also cost work. State space grows with occupied
sources and selected receivers. This is a capacity/activity separation, not a
claim that total training cost is independent of S: embedding gradients and Adam
may touch more parameters as more addresses appear within an update window.
All fitting operators are audited in the native-event pilot, including backward,
normalization, clipping and optimizer work. Stored-state payload and host RSS
are separate, and traffic/RNG/physical energy are not inferred from FLOPs.

Each synthetic population writes three events per source then queries every
source. Full per-population producer graphs remain live through supervision.
An optimizer update occurs only after all pending population queries complete,
averaged over the actual query count. This avoids overwriting parameters while
an old population's graph awaits credit. It is still autograd-based CPU learning,
not an implemented local-trace or on-chip learning system. Full configuration
recovery, query/target separation and numerical timing contracts precede fitting.

## 332. Experiments that distinguish capability from a favorable implementation

Order labels depend on the last two content signs, not source ID. Time labels
use the sign of Σ v_i[exp(-age_i/.7)-.6 exp(-age_i/4)] so elapsed time can change
the answer. The model sees only observed signs, addresses, times and query flags.
A state-clearing intervention checks memory reliance. A refitted rank-time model
receives event order/content but no physical intervals. A refitted pathwise-credit
model preserves the same realized forward clocks/values but removes the losing-
route content teacher while retaining interior timing/value gradients.

Occupied-capacity 4/16/64 holds total query/event training budget and per-event
candidate count fixed, while every source actually stores information. Input
addressing must be given to strong controls too; a per-source recurrent control
can also exploit addressed state. Three seeds and independent confirmation
populations are necessary before a promoted mechanism claim. Synthetic success
unlocks real irregular-stream adapters and measured-resource comparisons, not
universal language or energy superiority. The executable priority, gates and
remaining real-data/hardware gaps are in RESEARCH_VALUE_PLAN.md.

## 333. Share learned content maps without sharing every persistent state

The old H2 character-specific model has 3.82M parameters but sees only 2K–8K
characters in its current pilots. Under a rough uniform alphabet and balanced
pool, a receiver gets T/(27P) realized writes per pass: about 38 at T=2048, P=2.
Actual character frequencies and routing are unequal. This is an exposure
calculation, not a statistical generalization bound; counterfactual score credit
and recurrent histories provide additional signal. It identifies why adding
cache machinery or larger local maps need not repair small-data learning.

The native language adapter instead treats tokens as content in one addressed
conversation. Receiver maps are reused across all token marks, while independent
head states, temporal evolution and routes remain. A receiver then has roughly
T/P realized writes per pass under balanced routes. The ≈27× exposure change
does not imply 27× better quality, nor does it preserve the old parameter capacity.
It tests useful weight sharing separately from memory addressing. There is no
per-position cache or assertion of full attention equivalence.

A scalable next construction can use shared content maps plus sparse specialist
corrections, W_a=W_shared+U_a V_aᵀ, with small local rank and addressed state.
Only admitted corrections execute, but scoring/discovery and losing-route
teaching must still be paid. That is a future native capacity hypothesis, not
an implemented replacement for the current tested model or a novelty claim for
factorized matrices. Establish the shared-map floor first, then test whether
extra specialist capacity improves quality per complete work budget.

## 334. Long silence is cheap to simulate but may be expensive in information

For a local temporal mode, m(t+Δ)=exp(-γΔ) R(ωΔ)m(t). Rotation preserves norm;
positive decay attenuates it. An analytic read avoids periodic updates, yet
||m(t+Δ)||=exp(-γΔ)||m(t)||. Sparse execution alone does not guarantee useful
information after a long gap. The native order probe at 8×/64× elapsed time tests
that distinction; its operation count cannot substitute for retained accuracy.

One principled next hypothesis is a direct sum of protected content and
computational time modes: A=0 on a small content subspace and A=-Γ+Ω on the
temporal subspace, with Γ positive and Ω skew-symmetric. The former preserves
content during silence; the latter supplies learned order/interval computation.
Input-conditioned writing, gating and cross-channel mixing can still change
protected content on events. This does not require synchronously refreshing
every state or importing an attention cache. It removes time evolution only
from the reserved subspace and must be tested against the current all-temporal
model at equal dimension, with write/read/teacher costs charged. It is not yet
implemented. Do not modify frozen running sources to install it mid-campaign;
use the long-gap result to decide whether the diagnosed information loss makes
this the next architectural experiment.

## 335. Sparse activity, sparse optimizer work and learning exposure are distinct

Let E be the number of fitted events, U events per optimizer window and π_s
the probability of source s. For independent source draws, the expected number
of distinct addresses in a window is exactly

    M(U) = Σ_s [1 - (1 - π_s)^U].

For uniform addresses, M(U) ≈ S(1-exp(-U/S)): approximately U when U is small
relative to S, and S when the window covers the population. This is a model of
independent arrivals, not the synthetic protocol's enforced complete populations.
In that protocol every window touches all sources, so M=S exactly.

With p_local receiver parameters per source, N_shared shared parameters and
embedding dimension D, an approximate arithmetic boundary is

    C_fit ≈ E C_event + (E/U) c_opt [N_shared + p_local M(U) + S D].

C_event includes forward, producer backward, losing proposals and clock work;
c_opt includes normalization, clipping and Adam arithmetic. The final S D term
is the current dense embedding-gradient implementation, even though its forward
lookup is addressed. It cannot be omitted as dormant work. Likewise, a receiver
with a gradient tensor is paid when Adam visits it. This formula explains both
why update batching can matter and why inference sparsity alone cannot establish
training scaling. The pilot audits actual operators instead of fitting this
approximation to make a claimed saving. E counts all input events; changing the
denominator to queries without charging their producer events is invalid.

The exposure tradeoff is equally concrete. At T total source events, uniform
sources and balanced pool routing, a local receiver receives about T/(S P)
realized writes. Increasing S at fixed T gives more state but fewer observations
per local parameter. Poor quality can therefore reflect insufficient learning
exposure, even when activity remains bounded. That is a hypothesis to distinguish
with shared maps or a matched-per-address-data control, not an excuse to discard
the negative result. Sharing weights and keeping states separate is the first
implemented intervention (§333). Sparse specialist corrections become useful
only if they improve quality enough to pay discovery, teaching and optimization.

The architectural target is thus a three-way frontier: information retained,
quality learned and complete active work. Cheap races offer additional choices;
they do not make untrained or information-erasing choices useful automatically.

## 336. Depth can accumulate ordered interactions without rescanning the past

An elementary gated temporal construction makes the native expressivity target
concrete. Let content maps a(v) and b(v) be scalars, and maintain two states
r and s. Between events they decay at rates γ₁ and γ₂. At event j, first read
the evolved left-limit states, then write

    s_j = s_j⁻ + b(v_j) r_j⁻,
    r_j = r_j⁻ + a(v_j).

Reading r before its current write excludes self-pairs. At a later query t,
expanding the recurrence gives exactly

    s(t) = Σ_{i<j} exp[-γ₂(t-t_j)] b(v_j)
                      exp[-γ₁(t_j-t_i)] a(v_i).

This is a weighted ordered-pair interaction. Choosing a to detect one content
pattern and b another distinguishes A-then-B from B-then-A. It needs two stored
scalars and constant arithmetic per event, although its expansion contains
quadratically many event pairs. Independent mode channels generalize this to
vectors; event-conditioned gates and further levels provide richer ordered
interactions. Such state is useful when the task admits a compact temporal
summary. Arbitrary independent historical values cannot all be recovered from
two scalars; this identity does not subsume unrestricted cached attention.

The time derivative also has a compact exact eligibility at fixed parameters.
Let e=∂r/∂γ₁ and f=∂s/∂γ₁, holding the content maps and γ₂ independent of γ₁.
For an event gap Δ, evolve using the pre-evolution r:

    r⁻ = exp(-γ₁Δ) r,
    e⁻ = exp(-γ₁Δ) (e - Δ r),
    s⁻ = exp(-γ₂Δ) s,
    f⁻ = exp(-γ₂Δ) f.

Then f_j=f_j⁻+b(v_j)e_j⁻, with e unchanged by the a(v_j) write. A final loss
on s uses f without retaining or replaying all old pairs. The extra traces and
their updates cost work. This is exact for this stated linear/gated recurrence
at fixed parameters; it is not a proof of cheap exact gradients for arbitrary
deep nonlinear hard races, changing weight versions or the current full model.

The two-state identity and eligibility are checked against an independent
explicit pair sum and finite differences in test_ordered_temporal_algebra.py.
This is a derived expressivity/learning target, not a newly fitted architecture
or a claim that pair recurrences themselves are novel. The native model has
evolving memory, content-conditioned gates and deep channels, but its exact
left-limit timing/read semantics differ. Demonstrating a learned ordered basis
in that model requires its held-out order task and controlled interventions.
The useful direction is economical causal summaries plus selectively recruited
temporal features, rather than copying a dense attention layer by default.
