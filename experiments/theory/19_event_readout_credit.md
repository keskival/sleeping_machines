# Learnable event readout and local supervision

## 185. The terminal statistic can conceal informative events

The current speech carrier pools final-layer payloads $x_i\in\mathbb R^d$
with packet count weights $c_i$. It observes the complete supplied utterance;
only one utterance-level label is supervised. This is a valid sequence-to-class
objective, but the statistic

$$m_0=\frac{\sum_i c_i x_i}{\sum_i c_i}\tag{185.1}$$

cannot learn which events should contribute most strongly to the decision.
High count need not mean high class information. After E122's successful
nuisance-augmentation screen, the 4,096-example continuation reaches 370/512
(72.3%) clean held-out answers. This motivates a direct readout hypothesis,
separate from adding depth or changing routing seeds.

### A causal scalar key and numerator/mass state

Let $q\in\mathbb R^d$ be a learned readout key, and define

$$s_i=q^Tx_i,\quad b_i=2\tanh(s_i/2),\quad
 a_i=c_i e^{b_i},\quad m=\frac{\sum_i a_i x_i}{\sum_i a_i}.\tag{185.2}$$

The forward state is an accumulated numerator $N$ and mass $A$. For each
observed winning payload, update $N\leftarrow N+a_ix_i$ and
$A\leftarrow A+a_i$. This is associative event accumulation with $O(Ed)$
work, not $O(E^2)$ event-pair attention. Every gain depends only on that
event's causally computed payload and fitted parameters. The current terminal
query implementation uses grouped reductions; a persistent online scheduler
remains separate work.

The bounded score implies $e^{-2}c_i\le a_i\le e^2c_i$, so a nonempty observed
prefix has positive mass. At $q=0$, $a_i=c_i$ and (185.2) is exactly (185.1).
This nests the old readout and enables an exact common starting checkpoint.
It adds 32 parameters at width 32 and one $d$-coordinate scalar map per event.
Only already selected hidden payloads enter this readout. Losing hidden routes
remain training-only alternatives under the existing carrier credit rule.

This is a content-weighted mean, not a new hidden hard race. Its weights vary
within a fixed positive range; it does not yet cancel uninformative packets or
choose an early emission time. It has one fixed learned readout key, not an
independent attention query at each event. Those distinctions determine both
expressivity and cost.

### The label's local event credit

Let $g=\partial_m\ell$ include the fitted head, conditioning and categorical
residual, and write $\pi_i=a_i/A$, $v_i=\operatorname{sech}^2(s_i/2)$. Then

$$\frac{\partial\ell}{\partial s_i}
 =\pi_i v_i\,g^T(x_i-m),\qquad
 \nabla_q\ell=\sum_i\pi_i v_i\,g^T(x_i-m)x_i.\tag{185.3}$$

The scalar teacher says whether increasing this event's contribution moves
the aggregate toward a better class decision. Payload credit is

$$\nabla_{x_i}\ell=\pi_i g+
 \pi_i v_i\,g^T(x_i-m)q.\tag{185.4}$$

The first term teaches the value; the second teaches its usefulness as a key.
This is the same distinction that makes attention values and selection scores
trainable, specialized to a single causal event accumulator. A teacher arriving
after the utterance can use stored local payload/eligibility state plus $g,m,A$;
the formula does not require the global counterfactual tree. The current
implementation uses autograd, rather than claiming a deployed local-learning
runtime.

At zero initialization,

$$\partial_q m=\operatorname{Cov}_{c}(x),\qquad
 \nabla_q\ell=\operatorname{Cov}_{c}(x)g.\tag{185.5}$$

Thus zero gain is a valid mean initialization without necessarily being a
dead gate. Its gain gradient is nonzero when the event covariance contains the
teacher direction. Identical payloads or a teacher orthogonal to their variation
give zero gain credit. The Fisher/Gauss–Newton geometry is
$\operatorname{Cov}_{c}(x)R^THR\operatorname{Cov}_{c}(x)$ for effective readout
map $R$ and categorical score Hessian $H$. This identifies the quantities to
inspect before adding more gain keys: covariance support and teacher alignment.
It does not prove optimization or new-speaker generalization.

The guarded E125 contract checks exact mean/checkpoint equivalence, query
separability and (185.3). On its speech batch the gain gradient norm is 2.1385
and its relative error against the explicit local formula is $9.83\times10^{-7}$.
The matched one-epoch continuation arms use the same 4,096 examples, initial
checkpoint, Adam state for existing parameters, sample order, augmentation
RNG and learning rate. Only the new gain parameters receive a fresh Adam state.
The old fitting-only readout calibration remains fixed in both arms.

The completed one-epoch result is 353/512 (68.9%) for mean pooling and
356/512 (69.5%) for learned pooling, versus 370/512 (72.3%) at their common
starting checkpoint. Both fitting accuracies are 79.7%; the learned key norm
is 0.0700. Its online fitting NLL is 0.9041 versus 0.9072. Thus the key receives
and uses credit, but this screen does not improve held-out recognition.
The three-answer net gain over the continuing mean is insufficient evidence
that terminal count dilution is the principal remaining bottleneck.

### Supported credit and transferable credit are different conditions

At mean initialization the utterance's key gradient is $C_u g_u$, where $C_u$
is its count-weighted event covariance. For a fitting distribution $P$ and a
held-out speaker distribution $Q$, let $G_P=\mathbb E_P[C_u g_u]$ and
$G_Q=\mathbb E_Q[C_u g_u]$. A small plain gradient step in the key alone gives

$$R_Q(q-\eta G_P)=R_Q(q)-\eta\langle G_Q,G_P\rangle
 +O(\eta^2\|G_P\|^2).\tag{185.6}$$

With a positive optimizer metric $M$, the first-order term instead is
$-\eta G_Q^T M G_P$. Nonzero per-utterance support and a well-conditioned local
Fisher therefore do not suffice: averaging may cancel those directions, or
the fitting and new-speaker teachers may disagree. This identity holds with
the carrier frozen and differentiable readout; the actual continuation also
updates the carrier and head, whose route boundaries require separate treatment.
It specifies a discriminating next audit—gradient alignment and cancellation
across fitting speaker groups, plus class-conditional retained information—
rather than assuming more keys, depth or epochs must improve generalization.
Held-out outcomes may diagnose this screen, but must not be used as training
updates or a substitute for a frozen official-test protocol.

## 186. A zero-update pass certifies a local teacher's fixed point

The phase teacher of §181 changes state only for a mistaken deterministic
class-clock race. If an entire pass over the fitting set performs zero updates,
all those predictions were made from one unchanged state and were correct.
Every later ordering of the same fitting examples therefore also makes zero
updates. Induction proves that the fitted phase state stays fixed forever.

E124 stops at this condition, using fitting outcomes only. It verifies exact
parameter equality with the phase state in E121's 200-epoch composite. This
reduces needless fitting presentations while preserving the learned rule and
its certificate. It is valid for this mistake-only teacher. A categorical SGD
optimizer can still change confidence after every class is correct, so the
same stopping proof does not apply to the deep neural branches or the learned
event gain. A fixed point also need not generalize; §183 provides the additional
task-specific certificate for the fitted phase rule.

## 187. Class-wise credit exposes progress concealed by an aggregate

The saved-prediction audit supplies a narrower question than overall SHD
accuracy. At the 72.3% checkpoint, class 3 has 135/206 fitting answers correct
but 4/26 held-out answers; class 19 has 126/204 versus 3/26. Their fitting counts
are close to the 204.8-example average. After the additional mean-pooling pass,
fitting correct counts fall to 82 and 86, while aggregate fitting accuracy
barely changes. Thus both within-fit class discrimination and new-speaker
transfer need examination. Scarce labels or universally absent deep credit
cannot explain this particular pattern by themselves. This is descriptive
evidence, not a proof of its causal mechanism.

Let $R_c(\theta)$ be the fitting loss conditional on class $c$, with exact
gradient $g_c$. The aggregate gradient is $\sum_c\pi_c g_c$. Even balanced
class frequencies do not guarantee that its descent improves every class:
for an update $\delta$, class $c$ improves to first order precisely when
$g_c^T\delta<0$. Class-wise gradient alignment, not frequency alone, decides.

### A diagnostic certificate for simultaneous local improvement

Fix a set $H$ of fitting-defined difficult classes, a positive-definite
parameter metric $M$, and a local radius $r$. The largest common first-order
improvement is the convex program

$$t^*=\max_{\delta^TM\delta\le r^2}\min_{c\in H}(-g_c^T\delta)
 =r\min_{\lambda\in\Delta_H}
   \left\|\sum_{c\in H}\lambda_cg_c\right\|_{M^{-1}}.\tag{187.1}$$

The equality follows by replacing the minimum over classes with a minimum over
their probability simplex, interchanging the compact convex linear game, and
maximizing the resulting linear functional over the metric ball. Thus a strict
common descent exists exactly when zero is outside the convex hull of these
gradients. If a nonzero positive combination cancels to zero, no infinitesimal
parameter update can strictly decrease all selected losses at once. This is
the finite-dimensional theorem of alternatives specialized to class credit.
Easy classes may instead enter as tolerated constraints
$g_c^T\delta\le\epsilon_c$, avoiding an unnecessary strict-improvement demand
for classes already fitted well.

The measured Gram matrix $g_c^TM^{-1}g_{c'}$ distinguishes aligned support,
conflicting credit and nearly absent directions. A useful new counterfactual
route should add an attainable correction whose class effects improve this
feasible region; merely duplicating correlated options cannot do so. Its
effects must be checked by realized paired route interventions, with their
downstream work charged, rather than inferred from route count or entropy.

This is a diagnostic program, not a convergence guarantee or an implemented
replacement optimizer. It applies inside a differentiable fixed-route region.
Route-boundary moves, finite updates and generalization require separate
analysis. The carrier's training-only counterfactual surrogate is not generally
the exact gradient of its hard forward function; a surrogate Gram matrix is
estimator telemetry, not an exact certificate for that function. Use ordinary
pathwise gradients for the local certificate and paired shadows for route
changes. Define interventions from fitting data; held-out classes diagnose
transfer and remain unavailable to the training update.
