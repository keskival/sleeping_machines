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
