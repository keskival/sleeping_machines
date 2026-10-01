# Parallel temporal heads and architectural work scaling

## 313. Independent heads without simultaneous arrival

The earlier integrated KV prototype has one historical winner per event block.
Eight sequential blocks do not provide eight simultaneous attention heads.
That restricts which different relationships a block can retrieve at once.
The new model gives each head independent query, key, value and gate matrices,
sparse receiver pools, an indexed episodic bank and a distinct output channel.
No head's weights are tied to another head. Learned square channel projections
permit cross-head interaction without reducing H*d_head dimensions to d_head.

An incoming message updates a persistent receiver; its stored state evolves
while the KV winner arrives. For each head, use the already established temporal
transition T_h(delta) = exp(-r_h*delta) R(omega_h*delta), with paired decay rates
and rotations. The KV arrival adds a gated selected value to the evolved
receiver message. At the next block's read time t=max(t_1,...,t_H), evolve each
retained head buffer by T_h(t-t_h) and concatenate the channels. This is a read
of stored signals, not a requirement that spikes coincide. Old episodic K/V
records remain immutable content; their admissibility and recency score are
separate from the evolution of currently propagating receiver messages.

Heads within a block start together and complete independent receiver/KV races.
Their maximum added delay is 0.022; the join uses max, not a sum across heads.
Thus L*0.022<0.5 protects source order independently of H. The CPU emulator
loops through heads serially; this is not measured parallel wall-clock speed.
Waiting-state semigroup, causality, large clock origin, teacher/inference and
chunk equality, independent Q/K/V storage, every head's projection gradients,
cross-head mixing, counterfactual counts and serialized next-update recovery
must pass before fitting. Boundaries retain the same local conserved route
surrogate; they do not become exact gradients of nonlinear route changes.

## 314. Attention-only comparison at matched interfaces

Match depth L, total width d, head count H and head width w=d/H. N is available
history per head; C is the admitted shortlist per head. Let B be the common
per-layer projection/content work and S the additional state-evolution work.
The leading matrix arithmetic for one new prediction is

    F_T  ~= L [B + 4 N d] + E_T,
    F_SM ~= L [B + 2 C d + S] + E_SM.

Dense attention scores QK (2Nd) and mixes all values (approximately 2Nd).
Race retrieval scores its C keys (2Cd), delivers one value per head, and avoids
the dense weighted-value contraction. Value delivery is memory traffic, not
zero resource use. Elementwise softmax O(HN), temporal/index work O(HC+d),
rate-setting hardware and clock energy require separate accounting.

If C=N, both have Theta(Nd) per-step attention work: the leading inference
advantage is a constant factor of about two when attention dominates. If useful
retrieval maintains bounded C as N grows, the admitted neural work is constant
in N while full-bank dense work grows linearly. This stronger claim depends on
discovery cost and quality/coverage; our fixed index has not demonstrated it at
frontier contexts. Both uncompressed banks still use Theta(LNd) storage.
Autoregressive full-sequence work sums these per-step costs: dense full-bank
attention is quadratic in sequence length, while bounded-candidate retrieval
can be linear, conditional on that discovery/quality assumption. Depth remains
approximately linear for both. At fixed H both have quadratic projection work
in d; no different width exponent is claimed.

## 315. Counterfactual teaching and the actual optimizer interval

Do not equate the credit length b with optimizer interval U. The present
reference uses b=U=16. A new accumulator still backpropagates and detaches each
b-character credit segment, but sums gradients over U targets, divides by the
actual target count, clips once and updates Adam once. A partial final window
uses its actual count. Prediction precedes its target-dependent update. This
changes the optimization trajectory and needs measured accuracy/LR comparisons;
it is not a free equivalence to the old optimizer.

Ignoring small elementwise terms, a dense attention forward/backward costs
approximately 12Nd per layer. A race scores Cd, differentiates scores and uses
the admitted values in its local teacher. Mean centering and counterfactual
direction contractions add approximately 4Cd beyond ordinary backward, giving
about 10Cd for the attention portion. All addressed receiver alternatives are
also evaluated during teaching. Current emulation backpropagates their graphs
even when a realized-value gradient is zero; the projection does not silently
remove that work.

For M target presentations, total fitting estimates therefore have the form

    G_T  ~= M {3 L B + 12 L N d + 3 E_T + 19 P_T/U_T},
    G_SM ~= M {3 L (B+S) + 10 L C d + R_receiver
               + 3 E_SM + 20 P_SM,updated/U_SM}.

Adam and clipping use approximate 14 and 5 operations per updated parameter;
explicit accumulated-gradient normalization adds one for SM. P_updated means
parameters with an allocated gradient, including addressed losing proposals,
not only winning units. Unaddressed units have no gradient/optimizer step.
The representative audits remain authoritative for recorded experiments.
The same data budget can have different M because passes differ. Increasing U
reduces optimizer cost per target but can require more learning or changed LR.
Training savings are consequently not inferred from winner-only inference.

## 316. A leading-order equation for the implemented parallel-head model

For our model, each layer has a d-by-d channel map and H independent d-to-w
receiver queries (4d^2 FLOPs together). A receiver pool has p scored units;
their key reads cost 2p*d^2/H. The selected receiver's input/output/gate maps
cost 6d^2/H. Each head's KV query, key, value and gate maps cost 8d^2/H.
Let A be the vocabulary/pool count (27 here) and r the Transformer's FFN
expansion (4). Then leading inference is

    F_T ~= L [(8+4r)d^2 + 4Nd + O(HN+d)] + 2Ad,
    F_SM ~= L {[4+(2p+14)/H]d^2 + 2(C+p)d + S_inf}
             + 2d^2 + 2Ad.

The extra 2d^2 is the source/context gate. For teaching, p receiver proposals
replace one, changing the quadratic coefficient to 4+(8p+8)/H:

    G_SM/M ~= 3L {[4+(8p+8)/H]d^2 + 2(C+p)d + S_train}
              + 4L(C+p)d + 3(2d^2+2Ad) + 20P_updated/U.

The backward multiplier two is approximate. Linear state, hashing, gates,
normalization, waiting dynamics and special functions are not zero; declare
their allowances in any plotted scenario. The current figures use 128d at
inference and 192d during teaching, plus 4HC scalar work, as illustrative
allowances. These are architectural scenarios, not measured end-to-end energy
or fitted accuracy scaling laws. They match interface dimensions, not measured
quality or identical function classes: stateful content updates replace the
Transformer's FFN structure and hard selection replaces deterministic averaging.

Leading parameter counts are

    P_T ~= L(4+2r)d^2 + embedding/output terms,
    P_SM ~= L[2+(4Ap+4)/H]d^2 + source/embedding/output terms.

The larger available receiver capacity need not execute per character. Approximate
updated receiver capacity uses A_U distinct addressed symbols within an optimizer
window instead of A. The plots use the illustrative uniform-symbol expectation
A_U=A[1-(1-1/A)^U]; actual text is nonuniform and saved traces replace this model.
Matched widths do not imply matched parameter counts. A claimed iso-FLOP or
matched-quality advantage must still be established empirically.

## 317. What approximately recovering Transformer attention requires

For rates lambda_j=exp(q*k_j/sqrt(w)), an exponential race chooses j with exactly
the softmax probabilities. If mu=sum_j p_j v_j and V is one winning value,
E[V]=mu. A sampled output is not a deterministic weighted average, and nonlinear
following layers do not generally commute with expectation.

Averaging m independent winners with the same rates gives

    E[mean(V_1,...,V_m)] = mu,
    E[||mean(V)-mu||^2] = sigma_v^2/m.

Scores can be shared across repeated races, giving approximately 2Cd scoring
plus O(md) accumulation arithmetic, m value deliveries and m physical races.
Emulation additionally samples/divides mHC numeric clocks. Counterfactual credit
and downstream variance also require their own work. The current prototype
delivers one winner per independent head; it does not implement m samples of
each Transformer's attention head.

If a shortlist omits probability mass delta and ||v_j||<=R, its mean differs
from full-bank attention by at most 2delta R. Thus an RMS error bound is
sigma_v/sqrt(m) + 2delta R. Approximate attention containment requires m and
coverage to meet the desired error; it cannot be asserted from the race identity
alone. Across depth, layer errors propagate with subsequent Jacobian norms, so
small per-layer errors need not remain small without stability. These equations
identify testable directions—multiple heads, accumulated samples, useful sparse
discovery and state evolution—rather than proving frontier-quality subsumption.

## 318. Spend cheap temporal operations around expensive matches

Keep independent Q/K/V matrices across heads. Within a head, retain G partial
matches phi_g(q,k)=sum_{i in group_g} q_i k_i rather than immediately collapsing
all coordinates. Computing these partial sums still costs approximately 2Nw.
A temporal policy r uses s_r(j)=a_r dot phi(q,k_j)+b_r(age_j,local_state).
R policies add approximately 2NRG scalar work per head, which can be modest
when RG is small relative to head width w. Different policies give genuinely
different distributions; repeated identical policies only reduce sampling
variance. The bilinear family is restricted by the shared feature basis and
must not be equated with R arbitrary independent dense projections.

Each winner can drive a distinct evolving-state response. An arrival at t_r
updates a persistent channel, which is read after decay/rotation at a later
time. This computes temporal functions beyond a static attention average,
without requiring simultaneous arrival or gathering every historical value.
Fixed total channel width can be partitioned among policies; this preserves
next-layer width, but reduces each policy's payload. Full-width payload per
policy increases value traffic, stored state and downstream work and must be
charged. The current implementation has independent spatial heads, not these
additional within-head policies. This is a proposed ablation, not a result.

For shared scores and R full-width outputs, dense weighted aggregation costs
approximately 2RNw, versus R winning-value deliveries and O(Rw) accumulation
arithmetic. For fixed total output width split among policies, dense value work
instead sums to approximately 2Nw; the R-fold advantage cannot be claimed.
Physical race circuits, rate setting, random sources and state capacity still
cost resources. Avoided digital normalization is not zero-energy hardware.

Counterfactual teaching still needs losing-value information. For local credit
g, (v_i-mean(v)) dot g can be evaluated through local scalar contractions,
then scalar credits communicated instead of whole losing vectors. This changes
communication placement, not the Cd local reads/dots or key/query gradients.
Our CPU teacher reads all admitted values, and no measured traffic saving for
this proposed local-credit implementation is claimed. Output cross entropy and
current recency transforms also retain logarithmic operations.

## 319. Value proposition and falsifiable milestones

The hypothesis is broader than attention: trainable temporal competition,
persistent evolving states and counterfactual credit can combine sparse event
execution with substantial selectively recruited capacity. Approximate attention
containment requires enough samples, coverage and stable error propagation;
one winner does not prove Transformer subsumption. With all keys scored, the
attention substitution has the same context-length order, but avoids dense
value aggregation and explicit probability normalization. Context scoring,
projection work, state updates and teaching remain charged.

Extra expressivity may permit equal quality at smaller width, fewer layers or
fewer training presentations. This compression claim requires matched-quality
curves, rather than following from temporal expressivity alone. Dormant capacity
separates available parameters from executed parameters only if useful module
discovery stays economical. Adam/credit costs can still grow with the number of
addressed alternatives. Event-driven input is a natural interface for irregular
sensor streams; event-camera superiority has not been benchmarked here.

Evidence already includes structured-task sample efficiency, learned temporal
algebra, deep sparse language training and numerical deep-credit contracts.
Deep learning is demonstrated in bounded event-block schedules; universal
asynchrony and frontier language quality are not. Milestones are: repeatable
matched-quality training/inference advantage; better quality at fixed work;
useful capacity growth without proportional activity; then measured hardware
energy/traffic and asynchronous sensor benchmarks. Failed language promotions
are evidence about this design, not a reason to erase established primitives.

## 320. A repeated race can be an event process rather than a global reset

For a fixed query, let stored key j emit a Poisson process with fixed positive
rate lambda_j=exp(s_j). Their superposition has total rate Lambda=sum_j lambda_j.
The next arrival's mark J has probability p_j=lambda_j/Lambda. Exponential
memorylessness means the losing emitters keep their residual waiting times;
only the winner needs its next local waiting time. Repeating this produces
independent marks J_1,...,J_m with distribution p, and independent inter-arrival
times of rate Lambda. The m-th arrival time has mean m/Lambda. No explicit
probability normalization or global reset is needed to generate these marks.
This is a mathematical construction, not a measured circuit or implemented
language module. Physical emitters, calibration and communication cost resources.

The query/key setup costs approximately 2Nd once. A subsequent sample needs one
winner's value and its local new event, plus delivery/accumulation, rather than
rescoring every key. With fixed m and static values, averaging the m arrivals
recovers the attention mean with mean-square error sigma_v^2/m. For m=64 the
RMS sampling term is sigma_v/8; this is an error relative to value variance,
not a claimed accuracy guarantee. Required m can be large, and the approximation
is not uniformly cheap. At N=4096,m=64, value-only reads are 64 times fewer than
reading all values, whereas total logical K/V reads improve only by
2N/(N+m), approximately 1.97. Projection and teaching work remain additional.

A common rescaling of all rates changes expected latency but not mark
probabilities. A real circuit must fit rate range, timing noise and deadlines;
refractory/reset delays can invalidate the ideal Poisson model. Stopping after
a fixed number of arrivals differs from stopping at a fixed deadline. If zero
arrivals occur by a deadline, a defined silence/no-event response and its
learning objective are required. Rate setting need not itself be free.

For distinct policies, project retained partial match features into independent
rate banks as in §318. Their additional score transforms, emitter/state capacity,
value widths and credit must be charged. Winner messages can drive evolving
states instead of being averaged, yielding temporal computations beyond a
static attention mean. That changes the function and requires its own learning
comparison. Marks are independent of arrival times under the fixed-rate model;
transporting values to different ages changes the readout unless its temporal
kernel/normalization explicitly preserves the attention mean. Do not silently
claim exact attention equivalence for an evolving-state readout.

A minimal future contract should check mark frequencies and lag correlations,
waiting-time statistics, equivalence of winner-local renewal versus fresh
independent races, and average-value convergence. Training needs a separate
contract: unbiased sampled values alone do not reproduce deterministic
attention gradients through nonlinear downstream layers. Validate integrated
learning and finite-rate/deadline behavior before a long run. This proposal does
not displace the already running independent-head/optimizer campaign.

## 321. Event-triggered credit and on-substrate learning

At fixed parameter versions and realized stochastic/temporal choices, the
finite unrolled credit graph consists of local operations y=f_theta(x). A node
can execute forward after its inputs arrive and retain the required activation
trace. Backward begins when the declared loss adjoint is available; each node
accumulates all its downstream adjoints, then emits J_x^T*g and J_theta^T*g.
Messages carry graph/target/version identity and dependency completion. This
readiness schedule does not require a chip-wide oscillator. It preserves the
chosen differentiable/surrogate graph if dependencies and versions are retained;
it does not make a hard race's true discrete derivative equal its teacher.

This construction is applicable to finite computation graphs generally. The
proposed advantage here is economical event-local traces and sparse addressed
credit, integrated with delays, evolving state, independent channels and
counterfactual alternatives. Losing proposals and admitted historical values
remain part of the teaching graph. Required completion accounting, communication
and local storage are not free. Keeping enough activation/error precision and
physical delay/rate fidelity is a separate hardware requirement.

Updates can be triggered by completed local credit windows. For exact recovery
of the reference learner, parameter versions, target-weighted accumulation,
global clipping and optimizer ordering must match its declared semantics.
A global norm reduction can use messages rather than a global clock, but still
has a logical completion dependency. Advancing a forward path before its
required weight update or using per-node clipping changes the learning rule.
Bounded-staleness/local optimizers require numerical contracts and measured
integrated convergence; no general convergence theorem is asserted here.

Persistent optimizer moments cost storage for updated parameter blocks, and
bounded credit traces scale with addressed alternatives and credit length.
Dormant capacity alone does not remove these costs. Full learning can be native
to a suitably equipped event ASIC in principle; a chip with only inference
state-update circuitry is insufficient. Current implementations use CPU autograd
and block-delayed Adam, not asynchronous learning hardware.

A useful milestone is online prediction and adaptation on the same substrate,
with delayed feedback and no external parameter trainer. Replay identical
causal streams, compare frozen/exact-version/asynchronous-update arms, and
measure accuracy/likelihood, adaptation latency, retention/forgetting, trace
bytes, control/gradient traffic, deadlines and total inference-plus-learning
joules. The existing full-backbone online result is supporting software evidence,
not hardware validation. Include learning-capable neuromorphic controls and
competent synchronous training hardware; on-chip learning is not unique to this
project. State what the complete combined construction adds.

## 322. Order and elapsed time as structural inductive biases

Let Phi_Delta be the stored-state flow and Psi_a the update induced by content
a. Two events a,b separated by Delta yield Psi_b(Phi_Delta(Psi_a(s))). Swapping
their content generally gives a different state, because event updates and the
waiting flow need not commute. In an affine local approximation
Psi_a(s)=A_a*s+b_a, their difference is

    (A_b*Phi_Delta*A_a - A_a*Phi_Delta*A_b)*s
      + A_b*Phi_Delta*b_a + b_b - A_a*Phi_Delta*b_b - b_a.

Thus order sensitivity follows from the computational state/update structure,
without treating order only as an extra input feature. Trainable delays also
alter relative arrival order and route selection. Counterfactual credit can
teach these hard decisions; this is not a claim that their realized discrete
boundary has an ordinary smooth derivative.

The learned rotation/decay flow has the semigroup identity
Phi_(u+v)=Phi_u composed with Phi_v. Reading state after a gap therefore need
not require repeatedly scanning every intermediate clock tick. If a construction
uses only time differences, common timestamp translation leaves its computation
unchanged. That invariance is conditional: absolute phases/time features and
explicit deadlines can deliberately change it. Delay bounds and event scheduling
must preserve observation causality; the present language candidate uses a
bounded sequential source schedule, not unrestricted overlapping event streams.

These biases favor order-, lag- and history-dependent signals. They do not prove
that temporal precedence identifies causal mechanisms or that every causal task
is easy. The completed multi-seed order-chain learning result supplies concrete
structured-task evidence, with its declared priors and controls. Promote the
broader claim through count/content-matched temporal tasks, held-out delay ranges,
timestamp jitter, relative-time shifts, event-order interventions and real-stream
quality/work curves. Compare controls with competent positional/time encodings.
Keep joint-model and hardware advantages prospective until measured.

## 323. A concrete tabular connection: feature thresholds through delays

Preserving feature identity gives a simple representational construction. For
feature x_i, threshold theta and positive d0>a, beta, define two branch delays

    d_plus  = d0 - a*tanh(beta*(x_i-theta)),
    d_minus = d0 + a*tanh(beta*(x_i-theta)).

Both delays remain positive. The plus branch arrives first exactly when
x_i>theta; the minus branch arrives first when x_i<theta. Declare the tie policy
at equality. In ideal arithmetic this realizes a decision stump through arrival
order. Composing such addressed branch modules realizes a tree's conditional
path without executing every stored node. Multiple independent paths/heads can
supply an ensemble. An oblique threshold uses z=u*x-theta instead, charging its
feature contractions. This is a representational construction, not an implemented
or trained tabular benchmark, a new decision-tree algorithm or hardware result.

This identifies a concrete hypothesis: tree-like feature-selective boundaries
can coexist with learned vector content, temporal state and counterfactual route
credit in the substrate. Generic smooth dense feature mixing is not required
before every conditional decision. Feature/store addressing and all input reads
still cost work. A row identifier can address locally stored features, so branch
messages need not retransmit the full row at each step; its storage/communication
costs must nevertheless be included. Finite delay resolution and circuit noise
can blur a threshold and require precision/robustness tests.

Learning good trees is not established by representing them. Local losing-branch
content credit is not automatically the exact loss of a complete unexecuted
subtree. Any approximation needs defined semantics, its own contracts and
resource charges. Branch discovery, feature selection, ensemble size and credit
can consume the saved execution. Compare with boosted trees and modern tabular
Transformers at fixed tuning/data/quality budgets.

Static independent rows should reset row-specific state and respect feature-ID
presentation invariance. A feature-addressed store can gather a row before
conditional queries, avoiding dependence on arbitrary input transport order;
this join is an explicit dependency and its latency is charged. Value-encoded
delay is a computation, not observed physical elapsed time. Order bias is useful
for temporal streams but need not be imposed on unordered tabular records.
See TABULAR_RESEARCH_PROTOCOL.md for the proposed tests and primary references.

## 324. Integrated repeated arrivals with one shared key match

Concrete limitation addressed: one historical value per spatial head supplies
only one sampled relationship, even when several candidate contents are useful.
Sections 316/319/320 motivate several cheap temporal marks from one query.
This trial preserves the complete independent-head architecture, addressed
persistent receiver updates, deep time evolution, separate Q/K/V, index and
counterfactual teaching. It changes historical retrieval alone. Receiver races
still select one receiver; no dense carrier or tied spatial heads are added.
The m=1 implementation nests the prior model exactly, including delays,
outputs, gradients and RNG. Stored history/candidate discovery are unchanged.

For a fixed admitted query, set lambda_i=exp(s_i) once. Start each next clock
at E_i/lambda_i. Select the earliest absolute clock tau_r, retain every losing
clock unchanged, and renew only the winner by adding a fresh Exp(1)/lambda_i
gap. Ideal independent Poisson processes have IID categorical marks with
p_i=lambda_i/sum(lambda), and independent gaps Exp(sum(lambda)). This supplies
m samples without m key/query contractions or m rate settings. Repeated marks
can select the same value. More marks are not additional independent learned
policies, extra spatial heads or better candidate discovery.

The CPU protocol encodes each absolute arrival as
u_r=.001+.010*tau_r/(1+tau_r). Its receiver message evolves to u_m; each selected
historical message evolves by T_h(u_m-u_r). Average these m evolved messages,
apply the existing gate and add to the receiver. Different spatial heads retain
their own projections/channels and join as before. Since u_m<.011, the original
depth/order bound is preserved. This bounded warp is a monotone numerical
encoding: the warped clocks are not homogeneous physical Poisson processes.
Rate programming, finite deadlines, resolution, renewal, buffering and a
hardware realization of this encoding remain implementation questions. There
is no measured clock-free hardware or zero-energy claim.

For learning, admit all C candidate values once. Let c_i=v_i-mean(v),
g_r be the upstream message gradient, Delta_r=tau_r-tau_(r-1), tau_0=0.
The declared local per-arrival teacher is

    a_(i,r) = lambda_i Delta_r (c_i dot g_r),
    credit_(i,r) = a_(i,r) - 1{i=w_r} sum_j a_(j,r).

Its conserved sum can be evaluated without m separate C-by-d contractions:

    G = sum_r Delta_r g_r,      D = sum_i lambda_i c_i,
    credit_i = lambda_i (c_i dot G)
               - sum_(r:w_r=i) Delta_r (D dot g_r).

Thus this local teacher takes O(Cd+md), not O(mCd). It equals the explicit
sum of the declared per-arrival rules, not the exact nonlinear loss change
when a mark changes subsequent state or arrival order. Realized timing credit
additionally uses d u_r/d s_(w_r)=-.010*tau_r/(1+tau_r)^2 inside fixed-mark
regions; value gradients accumulate at delivered marks. Transport's gradients
reach message and delay arguments. Route-boundary fidelity remains an open
hypothesis requiring task evidence, not a consequence of credit conservation.

Inference leading historical work remains 2Cd plus O(md) transport/aggregation
per head; it delivers m values and renews m-1 emitters. CPU minimum scans cost
m(C-1) comparisons and are reported separately from FLOPs. Training retains
C value reads, aggregated losing-route credit, all backward/optimizer work and
m message transports. Projected arithmetic removes only explicit numerical
clock division/renewal/bounded-time encoding and rate exponentials, recording
rate settings/arrivals/renewals separately. It does not erase matches, state
operations, memory, physical clock cost or counterfactual learning.

Prepared matched trials: H2, d32/head, eight blocks, pool2, C<=8+4,
credit16, optimizerU128/warmup512/lr.004/seed6; 2,048 fit characters/four passes,
8,191 cold development targets, m=2 and m=4. Reuse the completed exact m=1
reference (3.778729 bpc; 19.999171 unit-special whole-fit GFLOPs) only after
full-configuration nesting/optimizer contracts and guarded smoke fits pass.
Existing H2/H4/data-scaling work retains priority. Seven read-only numerical
checks pass for nesting, mark/gap statistics, teacher aggregation/conservation,
fixed-mark timing derivatives, integrated key-budget/teacher equivalence and
complete forward/backward operator/clock-projection accounting.
They are not training or benchmark evidence. The deferred serial campaign
must pass complete operator accounting and memory gates before fitting, and
promotes one 8K comparison only if completed pilot quality supports it.

## 325. Forward history, temporal credit and the failed head/data gate

The completed U128/lr.004 eight-block runs improve with 2K→8K data, but H2
reaches 3.485055 development bpc and H4 3.542576. Both miss the predeclared
3.357342+.10 indexed single-head quality gate. More heads also change total
width, source/channel dynamics and parameters. This is a limitation of the
present implementation/protocol, not evidence that independent heads are
unhelpful or a ceiling on the sparse temporal substrate. Keep the negative
results, earlier controls and strong structured-task evidence together.

A specific implemented bottleneck is temporal credit truncation. Suppose an
entry written at u stores k_u=W_K phi_u and v_u=W_V phi_u. For a later target t,
local score credit a_(t,u) and delivered value cotangent g_(t,u) would contribute

    dL_t/dW_K += a_(t,u) q_t phi_u^T / sqrt(d),
    dL_t/dW_V += g_(t,u) phi_u^T,

plus upstream representation credit, inside the declared route surrogate.
Sealing the cache with detach removes these producer paths for entries before
the credit boundary. Their values still affect prediction and query/choice
credit still learns; the old key/value-producing maps receive no contribution
from those later reads. Long forward history is not long learning history.
Likewise carried receiver/context state retains information but its earlier
producing path is cut. The first event after a boundary can retrieve old content
without any gradient to its old KV projection writes. A numerical contract
verifies this and exact forward equality before/after detachment.

The current b=16 segments therefore restrict learned long-lag representations,
even when all historical positions stay in memory. This is an intentional
truncated-gradient protocol, not a newly discovered incorrect derivative.
Cached states created under older weights also remain a declared approximation;
increasing b does not make all historical weight versions current or restore
unlimited credit. Simply adding capacity/arrivals does not repair these paths.

Next matched intervention: increase b from16 to64 with U=64/lr.002, H2,
d32/head, depth8 and the same source/index/races. U remains fixed, separating
credit support from optimizer interval. No architectural mechanism or inference
operation is removed/replaced. Standard truncated backpropagation is the
learning primitive being varied, not the novelty claim. Longer credit retains
more state/KV graphs; it may train producers used up to64 positions within a
window and change gradient norms/work. It cannot train arbitrarily old writes.
Average producer-lag coverage is boundary-dependent, not exactly64 everywhere.
All additional backward and clipping/Adam work must be captured; peak memory
must pass a guarded full-size64-target contract and 129-character smoke first.
Causality and the fixed-weight forward/RNG path are unchanged by graph lifetime.

Reuse the strongest completed 2K optimizer pilot, U64/lr.002/credit16 at
3.732586 bpc, instead of assuming the cheapest U128 schedule scales best.
The next finite serial campaign runs a frozen-checkpoint information-flow audit,
the64-credit contracts/smoke, one matched64-credit2K fit and the existing
16-credit/U64 8K configuration. A >=.02 bpc2K gain admits one64-credit8K fit.
Choose completed8K quality/work before32K promotion under the existing .10 bpc
control tolerance. Second-seed8K and conditional131K follow only after a
completed32K gain/memory/time gate. None of these pending cells is evidence.

The diagnostic separately measures short-window frozen interventions, channel
spectra/source gate behavior, and a local content-credit comparison. For one
realized race time, continuation seed and finite candidate set, replay each
value choice, giving losses ell_i. The conditional categorical content gradient
is p_i(ell_i-sum_j p_j ell_j). Compare its direction with the actual centered
local teacher at the realized value; report alignment and candidate loss gaps.
This conditions away the race-time derivative and samples a tiny local subset.
It is not the full expected gradient through changing persistent histories,
noise or candidate discovery, and removal interventions without refitting are
not benchmark controls. Use it to choose a concrete repair, not as a supremacy
claim or an automatic justification for replacing temporal computation.
