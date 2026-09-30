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
