# Joint race likelihood, silence credit and generic learning

[Theory index](../THEORY.md) · Global sections 197–201.

This develops an exact stochastic alternative to the current deterministic
race surrogate. It builds on noisy races (§5), competing-risk supervision
(§§115–118), realized credit geometry (§§190–194), and joint key/value/clock
alternatives (§196). Likelihood-ratio credit and conditional expectation are
established mathematical tools. The advance here is an explicit joint law,
its censored information geometry, and its application to the observed winner
coupling. This is a derived and numerically audited candidate, not an implemented
SHD or language training improvement or a claim of a new general estimator.

## 197. Factor route identity from arrival intensity

At an active node with causal history H and a finite candidate set, define

\[
 p_k=\operatorname{softmax}(s(H))_k,\quad
 \Lambda=\exp(\eta(H)),\quad \lambda_k=\Lambda p_k.
\]

Independent clocks with constant hazards lambda_k give the winning mark K and
waiting time T the joint density

\[
 f(k,t\mid H)=\lambda_k e^{-\Lambda t}
             =p_k\Lambda e^{-\Lambda t},\qquad t\ge0. \tag{197.1}
\]

Equivalently sample K from p and T from Exp(Lambda), independently conditional
on H. Emit only candidate K's paired key/value at that actual time. This is a
hard sampled race, not a weighted forward average. Independent exponential
clocks or the equivalent two-variable sampler have the same winning law.
For observed (k,t), the local scores are

\[
 \nabla_s\log f=e_k-p,\qquad
 \partial_\eta\log f=1-\Lambda t. \tag{197.2}
\]

A winner's log rate includes survival of **all** losing clocks. Thus the clock
score uses an already local total hazard, and the mark score includes every
present candidate. Route confidence and arrival intensity are separately
trainable. The conditional Fisher matrix is block diagonal:

\[
 I=\begin{pmatrix}\operatorname{diag}(p)-pp^T&0\\0&1\end{pmatrix}. \tag{197.3}
\]

The cross block vanishes because E[e_K-p | T,H]=0; E[(1-Lambda T)^2]=1.
Adding a fixed minimum delay preserves this law after translating t. A
parameter-dependent minimum delay changes support and needs boundary terms.
Incoming events may interrupt the clocks: use the survival-integrated,
piecewise history-conditioned intensity, not (197.1) across a changing history.

## 198. Exact deep credit includes jumps that an interior derivative misses

Let omega be a finite sampled causal execution, including marks, elapsed times
and censored windows. Assume parameter-independent random-clock support,
finite expectations, differentiability of local laws and enough domination to
interchange expectation and derivative. A learned value program may be smooth
conditional on omega, as in the separate-key phase. Its expected objective is
J(theta)=E[L(theta,omega)]. Then

\[
 \nabla J=E\left[\nabla_\theta L\big|_\omega+
  \sum_{i\in\mathrm{visited}(\omega)}(G_i-b_i)
                  \nabla_\theta\log f_i(\omega_i\mid H_i)\right]. \tag{198.1}
\]

G_i is downstream loss or a declared downstream loss/work sum. Costs already
fixed before decision i can be removed. Baseline b_i must be measurable before
that decision; its current fitted value is detached in the score update. The
score has conditional mean zero. Parameter sharing requires summing all
visited decisions, including the derivatives of their causal key states.
Direct smooth effects not mediated by a sampled decision belong to the first
term. Time reordering and different descendant winners are part of L's realized
execution, so this score expression does not require a smooth suffix.

Example: L(T)=1{T>c}. Its expected loss is exp(-Lambda c), with

\[
 \partial_\eta J=-\Lambda c e^{-\Lambda c}.
\]

The sample's ordinary time derivative is zero almost everywhere. The exact
score expectation E[L(T)(1-Lambda T)] equals the nonzero derivative. This is
the clock counterpart of E130's winner-boundary obstruction. An exact expected
gradient does **not** guarantee low variance or improvement of every finite
sample's hard loss.

This is a stochastic training/deployment model. Replacing its races with
argmax marks or mean delays changes the objective and requires a separately
measured policy-transfer comparison. No equivalence to the existing deterministic
checkpoint is asserted.

## 199. Counterfactual suffixes reduce variance at a real cost

Condition on H_i and the sampled waiting time. A mark counterfactual must replace
the emitted paired key/value and rerun its actual causal descendants. Let U_k
be their loss under candidate k. Future random variables can be coupled across
alternatives to reduce variance, provided each alternative retains its correct
conditional law. Use stable packet/decision identities for random-number
coupling when alternative histories have different lengths.

The exact conditional mark credit is

\[
 g_s=p\odot(U-\langle p,U\rangle\mathbf1). \tag{199.1}
\]

Averaging over all candidates and valid future randomness gives the true mark
score expectation. Conditional averaging over the sampled mark reduces its
covariance by the law of total variance. It does not necessarily reduce total
wall time, and U_k cannot generally be read from local projected crossing times.

With a single additional candidate k drawn from a full-support proposal q,

\[
 \hat g_s=\frac{p_k}{q_k}(U_k-B)(e_k-p) \tag{199.2}
\]

is conditionally unbiased for any candidate-independent B. Its second moment
is sum_k p_k^2 (U_k-B)^2 ||e_k-p||^2 / q_k. The optimum proposal, given those
unknown utilities, is proportional to
p_k |U_k-B| ||e_k-p||. Thus route probability alone is not the optimal use of a
counterfactual work budget. Proposal floors protect support; clipping weights
introduces bias. Reusing learned utility estimates for proposals is allowed,
but must not substitute them for actual sampled utility without reporting bias.

For the **joint** clock gradient, a sampled winning time is itself informative.
Holding T fixed is sufficient for mark credit, not for optimizing the arrival
rate. Use the time score, censored survival, or a correctly integrated joint
counterfactual. Fixed-history local tangent scores in the current core remain
approximations to U_k; richer payload alternatives alone do not fix that gap.

Optionality can enter U_k through a declared finite future adaptation/evaluation
budget (§§159–163,196). That targets a meta-objective rather than current NLL.
Independent adaptation and evaluation avoid same-sample optimism. A single
utility scalar can carry that evaluated reserve, but cannot represent arbitrary
shared downstream control conflicts. Count every adaptation, suffix replay and
extra stored state.

## 200. Silence provides intensity credit and reduces available information

Let a fixed deadline h censor a node: either a winner arrives at t<h, or no
message is emitted. The latter has probability S=exp(-Lambda h), log likelihood
-Lambda h and score

\[
 \nabla_s\log S=0,\qquad\partial_\eta\log S=-\Lambda h. \tag{200.1}
\]

With candidate-conditional costs C_k and nonresponse cost C_NR, locally held
independent of s and eta, define Cbar=sum p_k C_k. Then

\[
 J=(1-S)\bar C+SC_{NR},
\]
\[
 \nabla_sJ=(1-S)p\odot(C-\bar C\mathbf1),\qquad
 \partial_\eta J=\Lambda h S(\bar C-C_{NR}). \tag{200.2}
\]

When answering is better than nonresponse, the intensity update increases the
chance of an arrival. If premature answers are worse, it decreases that chance.
A firing penalty without a proper class/decision cost does not teach the correct
answer. Observed silence gives no mark identity; mark teaching requires observed
answers or valid counterfactual answer evaluations.

Integrating the fired and censored scores gives

\[
 I_h=(1-S)\begin{pmatrix}\operatorname{diag}(p)-pp^T&0\\0&1\end{pmatrix}. \tag{200.3}
\]

Both channels lose information in proportion to the probability of firing.
For example, the eta block equals integral_0^(Lambda h)
exp(-x)(1-x)^2 dx + S(Lambda h)^2 = 1-S. The mark/time cross block is zero.
The intensity natural gradient divides (200.2) by 1-S; as Lambda h tends to
zero its deterministic limit is Cbar-C_NR. The sampled variance and lack of
candidate observations do not disappear. Damping, information-aware budgets
and nondegenerate initial arrival probabilities are needed; unlimited inverse
Fisher scaling is not justified.

A label window can therefore teach through both an arriving mark and survival
up to the deadline. It need not prescribe a physically unjustified single target
arrival time. For real streams whose intensities change after new input, replace
Lambda h with integrated causal exposure and sum the appropriate local scores.

## 201. Representation quality and physical scaling are separate claims

The conditional smooth value phase supplies supported interior transport. The
joint race score supplies expected boundary credit. Neither proves useful
learned features, optimizer convergence, cross-speaker transfer or competitive
language scaling. E131's higher fitting accuracy and lower held-out score
illustrate that distinction.

A generic language experiment must remove explicit target-count and pointer
experts from its primary prediction path, train all intended representation
layers, and expose the contribution of its learned event states. The natural
objective is causal next-token NLL; unlike completed-utterance classification,
it provides a teacher after each next token arrives. This changes the frequency
and localization of credit, not the need for attainable alternate routes.

Expected policy work remains O(number of visited decisions × local candidates)
for local scores, plus actual value computation and teacher/eligibility storage.
Exact counterfactual suffix work can be much larger, and even one alternative
per layer may induce depth-dependent recomputation. An unbiased rule does not
remove that systems cost. Candidate discovery, context replay, memory traffic,
communication, optimizer state and supervision all belong in the ledger.

The decisive test is a held-out quality/resource frontier as data and capacity
grow, including total training and inference energy where measured. Lower energy
at useful quality is valuable even before better raw loss. See
[the language scaling protocol](../LANGUAGE_SCALING_PROTOCOL.md) for the concrete
model exclusions, matching rules, data units and measurement boundaries.
