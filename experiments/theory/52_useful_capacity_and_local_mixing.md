# Useful representational capacity and operations between mixing stages

Global §§345–348. Builds on §170's measured conditioning problem, §§171–172's
reachable-control geometry, §§333–335's exposure/optimizer economics, §336's
ordered-pair recurrence and §§337–344's temporal gates, windows and trains.
These are derivations and design criteria, not a fitted optimal architecture.

## 345. Capacity is useful when it creates learnable task-relevant directions

Parameter count bounds available knobs but does not measure useful functions.
A parameter can duplicate another, affect no observed input, act only in a
saturated gate, or change a direction the readout ignores. Nonlinearity alone
also does not ensure that the nonlinear response carries task information.
An effective construction needs all of:

- Information survives the message/state path and is accessible at readout.
- Independent learned directions change relevant predictions or reachable
  routes, not just nuisance variation.
- Learning has credit, sufficient exposure and usable conditioning for those
  directions; differentiability is necessary for ordinary gradients but not
  sufficient for an effective optimizer or reliable discrete-route teacher.
- The function shares useful structure across examples: causal order, elapsed
  time, address-specific state, compositional content and optional activity.
- Improvement pays its inference, learning, optimizer, traffic and latency cost.

For a fixed route/time-history cell, stack logit Jacobians J on a declared
example set. With output curvature F, control metric M and regularization
lambda, §171's exactly attainable local quadratic loss reduction is

    R(g)=.5 g^T J (J^T F J + lambda M^-1)^-1 J^T g.

Two equally large parameter sets can have very different R because one lacks
rank, is ill-conditioned or points away from g. Shared controls require the
cross terms in §172; independently summing apparent event opportunities would
invent capacity. An independently estimated future loss gradient is needed to
assess transferable reserve rather than fitting the diagnostic examples.

A complementary local dimension measure is

    D_lambda=trace(C(C+lambda I)^-1), C=M^(1/2) J^T F J M^(1/2).

This counts well-excited, cost-scaled tangent directions rather than named
parameters. It is bounded by rank(J), which on N examples with C_classes
softmax outputs is at most min(parameters,N(C_classes-1)) for the conditional
logit tangent. This is a local diagnostic, not a theorem bounding all nonlinear
or discrete representational capacity. The expected hard-route model needs
boundary credit as well; its surrogate teacher is not automatically J.

More rare specialists can make sense, but about N*pi_i observations reach a
unit used with probability pi_i. Counterfactual scoring teaches route choice;
it does not automatically give each losing value map the same training as a
winner. Exposure, useful activation and credit are measured separately. Cheap
state capacity and expensive scarce examples are different constraints.

## 346. Why alternate local nonlinear computation and learned mixing?

Several affine maps without intermediate nonlinear/state/time operations fuse:

    W_k(...W_2(W_1 x+b_1)+b_2...)+b_k = W_eff x+b_eff.

They can change parameterization or optimization, but do not create a larger
class of affine input/output functions. Likewise independent coordinate maps
f_j(x_j) cannot generate nonzero mixed derivatives between different input
coordinates. At least one learned cross-coordinate mixing makes relevant
parts interact; a nonlinearity after that mixing creates conditional behavior.
For a scalar readout of mixed nonlinear features,

    y(x)=sum_r c_r sigma(w_r^T x),
    Hessian_x y=sum_r c_r sigma''(w_r^T x) w_r w_r^T.

Each feature contributes at most one local curvature direction. Increasing
feature count can increase mixed-interaction rank if those directions are
independent and the operating nonlinearity has nonzero curvature. Duplicate
features and saturation add little. Hessian rank is a local witness, not a
universal expressivity comparison between all architectures.

The useful local sequence is: put content/state into learned bases; select
relevant components using content, memory and time; form nonlinear interactions;
project the result back into persistent/message channels; then let a learned
cross-head map redistribute those channels before the next local computation.
Temporal evolution between arrivals supplies another nonlinear/ordered operator;
§336 already shows constant-work states expanding into many ordered pairs.
Repeating a large matrix at every step is not the only source of complexity.

### Economical cross-vector interactions

For incoming x and memory m, both width d, a rank-r bilinear residual is

    z=(A x) elementwise_times (B m),
    y=x+C z.

Coordinate j contains x^T A^T diag(C_j) B m, a cross-vector interaction matrix
of rank at most r. Its mixed derivative d²y_j/(dx dm) equals that matrix,
which is generally nonzero. A plain additive linear map Ux+Vm has zero such
mixed derivative. A, B and C have 3dr parameters and about 6dr projection FLOPs
plus O(r+d) scalar work. A fully general separate d-by-d bilinear matrix per
output has d^3 parameters; this low-rank family earns economical interaction
by restricting that family. Rank is a tradeoff, not lossless compression.

The new temporal reception gate is the same useful principle with m replaced
by a rotating/decaying clock pair: a projected content/state representation
is dotted with phi(time), then gates its contribution. Phase changes which
content directions are receptive. Independent heads retain independent learned
bases rather than repeatedly applying the same common projection.

## 347. Attending within a message and mixing it with another representation

A vector's raw coordinate order has no universal semantic meaning. Fixed groups
may be appropriate for known sensor channels, but learned messages need learned
bases. An economical content selector can use a shared receiver context q and
projected components k_j of an incoming message:

    score_j=<Q(context),K_j(message)>,
    relevant_j=sigmoid(score_j+time_gate_j),
    result=sum_j relevant_j V_j(message).

Queries decide relevance, keys expose matchable features, values preserve the
content being selected; independent head maps offer different relationships.
For a fixed small local group count this uses bounded work and no full history
attention matrix. Hard addressed choices instead of sigmoid mixtures need the
already declared counterfactual credit. Sigmoid gates are known primitives,
not claimed as an invention. Selection, value diversity and information
transport must be checked individually: identical values make a content
selection uninformative even when scores learn.

Cross-vector alignment matters too. Project incoming content and stored memory
into compatible learned spaces before dotting or multiplying them. Raw x dot m
is basis-dependent and can compare unrelated coordinates. Residual content
paths prevent the selector from irrevocably destroying a needed direction;
stable writes and read-before-write semantics preserve temporal causality.
Normalization can improve conditioning, but erases scale unless a separate
scale channel retains information the task needs. Gates near 0 or 1 have weak
ordinary derivatives; activity sparsity and trainability need simultaneous
measurement. A gate mask is not free traffic unless the implementation avoids
fetching its masked values.

The native parent already has learned incoming/write/output maps, content-
conditioned forget/write controls, nonlinear value gates and cross-head maps.
Its clock-feature extension adds content-dependent temporal reception while
retaining those paths. Low-rank bilinear replacements and extra within-message
selectors above are proposals, not installed in frozen parent runs. A concrete
conditioning/information failure, numerical contract and matched integrated
refit must precede any such substitution.

## 348. There is no universal optimal number of learnable operations per unit

Let q_i describe local widths, interaction rank, clock modes or train length.
Optimize expected task loss under measured inference, fitting, traffic and
latency budgets. One explicit Lagrangian is

    J(q)=R_task(q)+lambda_inf C_inf(q)+lambda_fit C_fit(q)
                    +lambda_bytes B(q)+lambda_latency T(q).

For an interior continuous allocation, the optimal marginal condition is

    -partial R_task/partial q_i = lambda_inf partial C_inf/partial q_i
                  +lambda_fit partial C_fit/partial q_i
                  +lambda_bytes partial B/partial q_i
                  +lambda_latency partial T/partial q_i.

For discrete operations, compare finite held-out improvements and actual cost
increments. This is an allocation criterion, not an assumed separable scaling
law: interactions between units, shared optimizer work and discovery change
all terms. Unit activity pi_i weights ordinary execution; scored candidates
and counterfactual values can cost even when a unit is not selected. Complete
Adam cost depends on real updated parameter blocks/windows, not pi_i alone.

Practical intuition: keep one expressive content/memory update between learned
mixing stages, then reuse its expensive projections for several cheap temporal
modes or bounded emissions. Add another dense projection when it enables a
needed independent interaction, rather than stacking affine maps that fuse.
Increase temporal modes until their task-aligned response rank or held-out gain
saturates, their gradient conditioning deteriorates, exposure is insufficient,
or timing precision cannot resolve them. For a mode with normalized frequency
omega and decay gamma, a timing error du changes its vector by at most roughly
sqrt(omega²+gamma²)|du| times amplitude locally. Extra high-frequency modes are
not usable capacity if clock jitter overwhelms them.

The first integrated clock ladder uses R=2 and R=4, identical depth/head/payload
and data, plus a same-clock waiting control. These are initial operating points,
not a claimed optimum. The next rank/mode/train/mixing ladder should measure
quality/work, feature covariance participation, loss-aligned timing gradients,
head response overlap, credit fidelity and actual value-map exposure. Rank and
nonlinearity diagnostics must use fitting-only data; frozen development decides
predeclared stage gates. The objective is a quality/resource frontier supported
by completed results, not maximal operation count or parameter count.

## 350. Rotating reception banks match content and elapsed time economically

The user's proposed neuron holds R learned planes (a_r,b_r), rotates each at a
learned speed omega_r, and optionally resets phase at its own emission. With
elapsed local age Delta since the declared reset:

    q_r(Delta)=exp(-gamma_r Delta)[a_r cos(omega_r Delta)+b_r sin(omega_r Delta)],
    score_r(x,Delta)=x^T q_r(Delta).

Its mixed content/time derivative is generally nonzero:

    d² score_r/(dx dDelta)=exp(-gamma_r Delta)
       [(-gamma_r a_r+omega_r b_r)cos(omega_r Delta)
        +(-gamma_r b_r-omega_r a_r)sin(omega_r Delta)].

Thus the bank creates content-time interactions that additive separate f(x)
plus g(Delta) cannot express. Frequency credit is
Delta exp(-gamma_r Delta) x^T[-a_r sin(omega_r Delta)+b_r cos(omega_r Delta)].
Ordinary learning reaches vectors, frequencies, decay and reception maps within
the fixed-history conditions. Learnable reset times additionally need the
hybrid sensitivity/boundary treatment in §§340–341.

With a time-independent component and enough Fourier modes, vector Fourier
approximation on a compact interval can approximate any continuous
*time-varying linear receptive direction* q(Delta), after a suitable continuous
periodic extension. This is a useful specified function class, not a claim to
approximate arbitrary nonlinear histories or to invent Fourier features.
Nonlinear gates and output mixing then compose different selected directions.

Do not rotate d-dimensional vectors on every tick. For each received message,
compute u_r=a_r^T x and v_r=b_r^T x once (about 4dR FLOPs). Each clock read then
combines those two scalars with its phase, costing O(R), plus the output map.
Multiple emissions can reuse the projected coefficients while their source
message/weight version stays fixed. Changing incoming content requires new
projections. A CPU reference evaluates phase analytically only at events;
physical oscillators, delay lines or local timekeepers have nonzero stability,
retention, reset and energy costs even without a global execution clock.

At fixed Delta, R scalar projections have rank at most min(R,d); they cannot
create information missing from x. Across time they expose different directions
and relationships with elapsed age. A small message can retain substantial
predictive information, but redundant maps on a low-dimensional nuisance
signal waste work. Timing-jitter attenuation, covariance rank, loss-aligned
phase gradients and held-out ablation distinguish these possibilities.

This matches signals with content-dependent temporal relationships and causal
resets. Stable static channels should coexist so nonperiodic information is
not discarded. Broad slow/fast scales and learned weights can accommodate an
unknown generating process; random high-frequency phase is a poor default for
irregular noisy clocks. The implemented new candidate resets its clock-feature
states per input event/head; it is not yet this persistent per-neuron reset
bank. Installing that bank requires a frozen integrated comparison and accurate
cost/exposure accounting, rather than silently relabeling the event-local code.

## 351. How to establish whether extra freedom earns its cost

A constant/zero learned mode is evidence about that optimizer/data result, not
by itself proof that the function family is useless. Distinguish intentional
zero output initialization (§343), absent credit, poor time-scale calibration,
insufficient specialist exposure, saturation, timing noise and redundancy.
Check whether a nonzero decoder actually opens gradients to the mode before
interpreting its disappearance. A mode useful to rare inputs can look inactive
on an average-only summary.

Use fixed protocols for frozen deletion and refitted controls. Frozen deletion
measures reliance of a particular model; refitting tests whether remaining
modules replace the function. Shared clocks/matches, exposure, data, selection,
optimization and full work must be documented. The initial delay campaign
uses identical clock/waiting behavior with and without learned temporal reception.
This tests the combined added branch, not separately its gate, rotation, decay
or count. Follow-up controls can set frequency zero, remove content-conditioned
gating, randomize phase with matched noise, or limit trains to one causal
emission. Avoid selecting many controls on development then claiming their
best result as confirmatory evidence. Independent repeats/real-data checks
follow a completed worthwhile pilot; retain both helpful and harmful outcomes.

## 352. Intuition narrows the next test; phase freedom is not automatically useful

Two naive variants are poor bets. First, rotating a single vector by a scalar
or using collinear plane vectors changes magnitude but not receptive direction.
Second, many fast phases driven by noisy clocks can become random gates rather
than useful content matching. With independent Gaussian timing jitter eta of
variance sigma_t²,

    E[cos(omega(Delta+eta))]=cos(omega Delta) exp(-omega² sigma_t²/2).

A learned frequency much faster than resolvable signal timing loses coherent
phase information. For a raw exponential clock T~Exp(Lambda), a complex decay
mode has mean E[exp((-gamma+i omega)T)]=Lambda/(Lambda+gamma-i omega).
High omega likewise attenuates its mean. This second identity is for raw clock
time, not a closed form for the implemented bounded U=T/(1+T) features.

Conversely, a learned two-vector plane can rotate relevance; distinct smooth
modes can represent several content/time relationships; bounded sigmoid gates
and a retained static path protect existing information. Independent per-head
projection planes and different scalar policies reuse expensive matches without
forcing an identical relationship in every channel. Event-local reset anchors
intrinsic delay features; persistent spike-reset clocks would instead describe
elapsed history and must earn that different inductive bias in a separate fit.

The most sensible first candidate therefore uses two/four smooth temporal modes,
content-dependent reception, a retained primary content/memory path and causal
joins. Its normalized initial frequencies range from pi/4 to pi over the clock
span, rather than 2pi,...,2piR. This repairs weak native phase sensitivity without
starting with many noise-driven oscillations. Frequencies stay learnable, so
this initialization is not a claimed ceiling on expressive capacity. Initial
decay and separate clock temperatures are trainable too. Extra dimensions are
accepted only through the completed quality/work/readout control gate.

Prediction: if smooth time-conditioned gating provides a useful nonlinear
interaction, the full variant should improve frozen language quality relative
to both the native parent and same-clock waiting control. If it merely supplies
random redundant projections, that improvement should fail or require more
work than it earns. Either outcome is informative and remains reported.

## 353. Concentrate extra interaction capacity after aggregation: a matched test

A task-optimal representation discards nuisance information while keeping a
sufficient predictive summary. Data processing cannot create missing input
information, but deeper state can aggregate a larger causal receptive field.
That can justify richer nonlinear processing later. It is a hypothesis about
information access and function complexity, not a general rule that raw inputs
have low entropy or early layers need little work. Some low-level features are
expensive, and a bottleneck can irreversibly lose their evidence.

In this native model the first block already receives prior deep context. All
depths maintain persistent state; they do not have a proven strict progression
of receptive-field sizes. Therefore a late-heavy allocation must be tested,
not inferred from depth alone. Learned feedback also means early preprocessing
is not acting solely on the current 27-symbol input token.

The new eight-block comparison assigns R_l=2 at every layer, or R_l=0 for the
first four and R_l=4 for the last four. Sum_l R_l=16 in both. Available receiver
capacity, head queries/key matches, selected commits, counterfactual proposals,
extra race/rate settings and total additional parameter/projection budgets
match. Policy composition and actual representations differ; measured complete
work can still differ because graph/transport costs and occupancy differ.
Each layer stores only its actual clock parameters: inactive padded parameters
are not silently updated by Adam. The uniform R_l=4 variant is a separate
larger-budget operating point, not part of the matched-allocation claim.

Full gradient/optimizer/recovery contracts and a small accounting smoke precede
the late-allocation pilot. Compare quality/work and content/phase gradients by
depth. Retain informative messages at early layers, using the parent maps and
static residual path. Promote based on completed data/control evidence; this
allocation does not replace persistent content or counterfactual learning.
