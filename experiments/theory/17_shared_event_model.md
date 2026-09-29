# Shared event models with separate task training

## 176. A common architecture does not require common weights

The experimental question is whether the same computational primitives learn
useful solutions across task families. Each task has its own parameter vector,
input alphabet, declared time units, memory contents and outcome space. There
is no joint multitask objective and no transfer of labels between datasets.

E120 combines three implemented components:

1. Addressed exponential numerator/mass memory and the depth-scaled, hard
   winner-only carriers of §§166–175. Three delayed alternatives are scored;
   exactly one vector correction and delay are transmitted. Local detached
   losing alternatives provide training-only score credit.
2. Conditional count/exposure evidence, indexed by observed context. This
   captures the local sufficient-statistic mechanism used in the language and
   event world models. Only visited contexts allocate state.
3. Relative pointer routes: an observed query chooses a source-symbol/offset
   route and reads its destination symbol. Mistakes promote consistent losing
   routes and demote the mistaken winner. The extracted primitive reproduces
   E61's updates exactly in the contract experiment.

The deep core is actually executed on every E120 neural benchmark. Evidence
enters a common context-dependent score readout; it does not select an entirely
different legacy executable. Still, this is a **partial synthesis**. Native
hold/veto conjunctions, periodic phase computation, learned emission/cancellation,
and a persistent streaming scheduler are not yet implemented in this core.
The presence of a generic deep network does not prove efficient realization of
those operations. In particular, an eight-layer carrier is not automatically
equivalent in sample efficiency to the successful discrete temporal detectors.

### Natural-score interface

Let an observed prefix produce deep features $h\in\mathbb R^d$. Local memories
return evidence vectors $e_j\in\mathbb R^C$. The common readout is

$$\eta=Wh+b+\sum_{j=1}^J(a_j+g_j^Th)e_j.\tag{176.1}$$

This is a product of categorical experts when $e_j=\log p_j$ and the outcome
objective is categorical. Setting the feature gates to zero gives ordinary
geometric mixing; setting $J=0$ recovers E119's readout. E120 implements a
subset of E79's evidence bank (KT orders 0–3), not the full word/copy mixture.
The expert weights may be signed, as in the unconstrained E79 mixer.

For a point process, $e_j$ and $\eta$ are **log intensities per physical second**,
not normalized mark probabilities. The likelihood, rather than a hidden-layer
architecture change, gives these scores their interpretation. E120's smoothed
joint count/exposure estimator is $(N_{bk}+\alpha)/(S_b+\beta)$; this shares
the mechanism of the earlier factored hazard/mark estimator without claiming
identical finite-prior predictions.

## 177. Queries are information boundaries, not just readout positions

Let $\mathcal F_u=\sigma\{(c_i,t_i,v_i):t_i\le u\}$ contain all observations
available at a query cutoff $u$. A valid predictor must be measurable with
respect to $\mathcal F_u$ and its previously fitted parameters/state.

**Prefix closure proposition.** If the encoder, evidence lookup and every
hidden transition depend only on an explicitly supplied $\mathcal F_u$ prefix,
and fitted state contains no outcomes after that cutoff in a prequential
protocol, then the final predictor is $\mathcal F_u$-measurable.

**Proof.** Encoded events and lookup keys are measurable functions of the
prefix. A finite composition of measurable functions remains measurable;
sorting uses deterministic tie-breaking, and hard route selection is also
measurable. Output normalization does not add information. The claim follows
by induction over transitions and layers. This says nothing about whether
the prediction finishes by $u$: computation can take additional time.

This boundary is essential when delays reorder events. Two observed tokens
at times $t_1<t_2$ can have $t_2+\delta_2<t_1+\delta_1$. If a whole training
sequence is processed and the older token's delayed output is used to predict
the second token, its receiver may already contain that token. An ordinary
array-position next-token loss is then invalid. E120 instead processes only
the observed context and scores one terminal query. The next type and gap in
the market adapter enter only the objective, never an event mark or lookup key.

The `ObservedPrefix` contract rejects events past the cutoff and groups each
query independently. This checks supplied event times, not the semantic origin
of arbitrary features: adapters must still establish that all marks, time
normalization and memory fitting use permitted observations. Memory fitting is
disjoint from neural fitting in the language, market and retrieval screens;
development outcomes update neither memory nor neural parameters.

### Computation completion and streaming cost

The current pooled readout waits for all supplied carriers; the last-token
readout takes the carrier of the last supplied token. Neither implements a
calibrated confidence stopping rule. Each layer adds a delay in $(1,11)$ ms,
so every supplied carrier finishes within $11L$ ms of its input timestamp in
the simulated clock. This is not a bound on CPU latency, and not all earlier
carriers necessarily reach a last-token readout before it fires.

Adjacent language/market queries currently **replay their context**. For $Q$
queries of length $C$, the core processes $QC$ input carriers, not $Q$. A
persistent scheduler must preserve pending delayed messages and define when a
query closes before this cost can become incremental. E120 records replay work
explicitly. No streaming-throughput advantage is claimed from these screens.

## 178. The same score credit, with a different statistical geometry

For a categorical label $y$,

$$\ell_{cat}=\log\sum_k e^{\eta_k}-\eta_y,\qquad
 r_k=\partial_{\eta_k}\ell=p_k-\mathbf1_{k=y}.\tag{178.1}$$

For a marked event arriving in gap bucket $b_*$ with mark $y$, let $s_b$ be
the time exposed to bucket $b$. A piecewise-constant hazard gives

$$\ell_{event}=\sum_{b,k}s_b e^{\eta_{bk}}-\eta_{b_*y},\qquad
 r_{bk}=s_be^{\eta_{bk}}-\mathbf1_{b=b_*,k=y}.\tag{178.2}$$

Both are **predicted sufficient statistic minus observed sufficient statistic**.
The event objective integrates silence analytically; it does not need a dense
time grid or a synthetic label at every instant. E120 checks its exact gradient
numerically against (178.2). A positive hazard in the unbounded last bucket
makes the next-arrival distribution proper.

For frozen evidence, (176.1) gives an effective readout
$R(E)=W+\sum_j e_jg_j^T$. Thus

$$\nabla_h\ell=R(E)^Tr,\quad
 \partial_{a_j}\ell=e_j^Tr,\quad
 \nabla_{g_j}\ell=(e_j^Tr)h.\tag{178.3}$$

The evidence therefore changes the **direction of deep credit**, not only the
prediction. A useful memory can reduce residual error; it can also dominate
the readout and leave little learning signal for the deep core. This motivates
reporting evidence-only performance alongside the combined model. Frozen
deletions of the head/evidence are diagnostics, not separately trained controls.

### Curvature and the clock direction

Categorical score curvature is $H_{cat}=\operatorname{diag}(p)-pp^T$ and
annihilates a uniform score shift. Hazard curvature is diagonal,
$H_{event}=\operatorname{diag}(s_be^{\eta_{bk}})$; a uniform score shift changes
the total event rate and **must not be removed as a categorical gauge**.
The Gauss–Newton credit geometry for parameter Jacobian $J_\eta$ is
$J_\eta^THJ_\eta$. A well-conditioned hidden transport map alone does not bound
this matrix: evidence magnitude, output saturation, exposure duration, and
directions invisible to the objective also matter.

This identifies two distinct requirements for a shared model:

- preserve payload/credit transport through depth;
- match readout conditioning and natural-score units to the task likelihood.

Categorical experts may be centered across outcomes without changing their
probabilities after mixing. Centering a hazard expert in the same way would
discard its clock information. E120 retains the absolute log intensities.
The exponential-family identities here are standard mathematics; the project
contribution is their explicit causal implementation and the cross-task audit.

### What is and is not inherited

The old conditional, fixed-route depth bound applies to the extracted carrier
core. It does not certify optimization across race switches, the new output
geometry, or arbitrary optionality policies. Inference still has small dense
vector maps and an outcome-dense query readout. Race credit evaluates three
payload alternatives during training and one selected payload at inference.
Each observed packet continues through all layers: input activity is sparse,
but adaptive deletion of unnecessary hidden carriers remains future work.

With $E$ carriers, fixed width $d$, depth $L$ and three alternatives, the scan
has $O(LEd)$ arithmetic and local maps $O(LEd^2)$; sorting costs
$O(LE\log E)$ in this implementation. Pointer search is $O(ER)$ for $R$ relative
offsets, and conditional evidence lookup returns $C$ scores per memory. These
costs must all be counted before claiming an advantage over dense baselines.

## 179. A successful memory can be destroyed by an unidentified readout direction

The first E120 recall model is perfect on its development contexts of length
17 (eight key/value pairs and a query), but only 23/256 correct at length 65.
Its extracted pointer memory alone remains 256/256 correct at both lengths.
This isolates integration, rather than retrieval, as the failure.

Write the static count coordinate as $c=\log(1+E)/10$. Every fitting context
has the same $E=17$, so $\operatorname{Var}_{fit}(c)=0$. The original
standardizer nevertheless uses

$$z_c=(c-\bar c)/\max(\sigma_c,10^{-4}).\tag{179.1}$$

On the fitting set $z_c=0$: a readout coefficient multiplying it is
**unidentifiable from those observations**. Its gradient is zero and its
random initialization can persist. At length 65, however,

$$z_c=\frac{\log(66)-\log(18)}{10\cdot10^{-4}}\approx1299.28.\tag{179.2}$$

The following whitening factor further amplifies the unobserved direction.
The audit measures an added head score as large as 78.97. This swamps the
correct retrieved evidence despite successful pointer generalization.

**Supported-metadata rule.** For static metadata with zero fitting variation,
project its standardized input contribution out of the readout. This imposes
zero dependence in an unidentifiable direction. Equivalently it chooses the
minimum-norm coefficient for that direction instead of an arbitrary initial
coefficient. It is an inductive assumption, not a proof that length can never
matter; variable-length fitting can identify a genuine count dependence.

E120 implements this by zeroing the input row of the whitener corresponding
to constant log count. The rule depends only on fitting data. It does not
project learned hidden features merely because their initial variance is
small: those features can acquire variation as the core learns. Variable-count
calibration is bitwise unchanged in the contract check.

**Frozen intervention:** changing only that one row, without retraining any
weight, restores 256/256 longer-context predictions; standard-context fit and
development accuracy stay perfect. This is a causal intervention on the
implemented failure, not an inference from correlated learning curves. A
fresh run with the supported-count rule is recorded separately. The original
failed result remains available.

The general architectural lesson is to preserve a specialist's invariances
when coupling it to new branches. An unconstrained additive branch can undo
a correct specialized answer even when its training loss looks excellent.
Test the composite on the specialist's extrapolation protocol, and inspect
the feature support and score contributions before changing hidden routing.

## 180. Why generic temporal features need not inherit modular generalization

The eight-epoch modular screen fits 176/1473 tuples but recognizes only 6/256
unseen tuples. This is a failed short generalization screen, not a long-run
grokking experiment. The earlier successful rhythm mechanism has not been
ported into the shared core.

A useful analytic diagnostic is available without another seed sweep. For
uniform independent $A,B,C\in\mathbb Z_p$, let $Y=A+B+C\pmod p$. Conditional
on any two operands, the remaining operand is uniform, hence

$$P(Y=y\mid A,B)=1/p.\tag{180.1}$$

The same holds for the other operand pairs. Consequently every representation
$f$ depending on at most two operands satisfies

$$\mathbb E[f(\mathbf1_{Y=y}-1/p)]=0.\tag{180.2}$$

Such features provide no population label correlation to a uniform categorical
readout. Finite training-set correlations can instead support memorization.
Simply passing information through more stable layers does not remove this
statistical obstacle for a low-order representation.

The group character $\chi_k(a)=e^{2\pi i ka/p}$ supplies a constructive contrast:

$$\chi_k(A)\chi_k(B)\chi_k(C)=\chi_k(Y).\tag{180.3}$$

Multiplication of unit complex payloads, or addition of their phases, composes
all three inputs with the correct algebra. The native rhythm prototype uses
related phase composition. This motivates a reusable trainable periodic-state
primitive with its own tests; inserting the known arithmetic answer into an
encoder would not demonstrate learning. The implemented deep tanh carriers can
in principle form higher-order interactions, so (180.2) is **not** a proof of
their incapacity or a diagnosis that every learned feature is low-order. It
identifies a representation/credit measurement to make and an existing
successful mechanism to preserve in the next synthesis step.
