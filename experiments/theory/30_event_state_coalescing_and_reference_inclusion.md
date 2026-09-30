# Learned event states: reference inclusion, exact pooling and teacher support

Derived 30 September 2026. The new six-block encoder is an architectural
experiment. It does not yet establish SHD parity or replace the strongest
common model. E142 establishes numerical identities; E143 tests a larger,
function-preserving residual learner. These are distinct evidence categories.

## 226. A precise forward inclusion, and a different optimization geometry

Write each real state pair as a complex number. A stable modal event update is

\[
 z_{i,j}=e^{(-a_j+i\omega_j)\Delta_i}z_{i-1,j}+b_j u_i,
 \quad a_j>0. \tag{226.1}
\]

The EventSSM linear operator has poles lambda_j, positive learned scales
delta_j and an event-independent input normalization

\[
 M_j=\lambda_j^{-1}(e^{\lambda_j\delta_j}-1).
\]

Its transition is obtained by setting -a_j+i omega_j=lambda_j delta_j and
b_j=M_j B_j. Since Re(lambda_j)<0 and delta_j>0, neither lambda_j nor
exp(lambda_j delta_j)-1 vanishes. Thus M is invertible: every unrestricted b
has a corresponding B=M^{-1}b. Real coordinate pairs and arbitrary real output
maps represent the real part of the complex output. This proves inclusion of
that **linear operator**, not of its complete nonlinear classifier, pooling
schedule, initialization or optimizer. [EventSSM equations 15–17](https://arxiv.org/html/2404.18508v2).
The implementation's positive rate floor restricts its exact admissible poles
to a>1e-6; the abstract operator permits every a>0.

The forward change of coordinates does not preserve Euclidean/Adam updates.
For b=M(p)B, with p the poles/scales, its local teachers obey

\[
 g_B=M^*g_b,\qquad
 g_p^{\rm ref}=g_p^{\rm direct}
       +\operatorname{Re}\langle g_b,(\partial_p M)B\rangle. \tag{226.2}
\]

Absorbing M changes which functional changes a parameter step makes. Function
inclusion therefore cannot substitute for a conditioning/teacher experiment.
Time-dependent ZOH normalization cannot generally be absorbed in a constant B.
Likewise, moving normalization across a nonlinear gate generally changes the
function. The implemented block supplies output gates, normalization, residual
payloads and winning clocks, but is not an exact published-block reproduction.

## 227. Exact affine packet endpoints and all their smooth teachers

For constant A and a causal packet P closing at T, define

\[
 b_P=\sum_{i\in P}e^{A(T-t_i)}B u_i.
 \tag{227.1}
\]

If P covers the raw events since the preceding closure T0, then
s(T)=exp(A(T-T0))s(T0)+b_P exactly. A learned source lookup can precede this
aggregation; the source addresses need not be collapsed into frequency counts.
Computing B times the source table once, then looking up transformed marks,
avoids a full matrix multiplication at every raw spike.

For endpoint teacher v, the local raw drive teacher is exp(A(T-ti))^T v.
For pair A=-a I+omega J, each contribution h_i satisfies

\[
 \partial_a h_i=-(T-t_i)h_i,\qquad
 \partial_\omega h_i=(T-t_i)J h_i. \tag{227.2}
\]

The endpoint identity holds for every smooth parameter value in the specified
family. Differentiating it proves equality of its source, B, a and omega
teachers; this is stronger than equal inference states at one parameter value.
E142 measures maximum endpoint error 1.11e-16 and teacher error 7.77e-16.

The first affine state is exact even if its output is nonlinear, because that
output does not feed back into the first state. However, omitting intermediate
nonlinear outputs changes the next layer's inputs. At a downstream closure its
state difference contains

\[
 \Delta s_2(T)=\sum_{i\in P}e^{A_2(T-t_i)}B_2 h_i
                    -e^{A_2(T-T_P)}B_2\widetilde h_P.
 \tag{227.3}
\]

A sum of original embeddings does not in general determine the nonlinear
h_i. Exact first-layer coalescing consequently does not certify the whole
raw-event deep network. The new model explicitly evaluates nonlinear outputs
only at nonempty closures. A 10 ms schedule bounds added first-packet latency,
but its learned decision accuracy still needs measurement. No empty ticks or
event-pair attention matrices are evaluated. Local vector projections remain
dense, and all raw transport and table-projection work must be charged.

## 228. Nest the learned solution while opening a larger representation

Let ell0(x) be the frozen strongest event classifier and h_theta(x) a new
generic nonlinear event encoder. Use

\[
 \ell(x)=\ell_0(x)+W h_\theta(x)+b,
 \quad W=0,\ b=0\text{ initially}. \tag{228.1}
\]

With nonzero randomly initialized hidden representations, initial logits and
decisions are exactly preserved. Completed-query cross entropy gives

\[
 g_W=(p-e_y)h_\theta^T,\quad g_b=p-e_y,\quad
 g_\theta=J_h^T W^T(p-e_y). \tag{228.2}
\]

Only the head learns on the first update. After W changes, hidden teachers can
become nonzero. This differs from initializing both hidden representation and
head to zero, which can leave the head teacher zero too. It does not guarantee
that subsequent hidden teachers are useful or that held-speaker risk improves.

E143 retains 53,296 immutable parent parameters and adds 395,814 parameters in
a parallel six-block, width-128, 64-pair event-state encoder. This is a larger
residual architecture with inherited training; it is neither a fourteen-layer
sequential model nor a from-scratch result for the original core. Charge both
branches at inference and the frozen parent's forward work at training.

## 229. Temporal supervision has an explicit nullspace

The completed-query mean/head depends on final vectors, not their emission
times. Holding those vectors fixed gives dL/dt_final=0. The final clock has no
label teacher. Earlier clocks can receive teachers because they change the
lags/order seen by downstream states. E142 verifies nonzero clock-weight
teachers in the first five blocks and no final-clock teacher.

A TV-quiz early decision requires a separate time-observing likelihood or
decision objective, including non-emission/survival and premature-error cost
(§§135,197–200). Attaching a uniform early class label is not a derivation of
that objective. This classifier uses completed-utterance supervision; it does
not implement a calibrated early-confidence policy.

Stable modal transitions have singular values exp(-a_j Delta), but the entire
gated residual network is not thereby uniformly conditioned. LayerNorm's
teacher scale depends on feature variance/epsilon; learned gates and output
maps are unconstrained, and pooling can cancel sample teachers. A conditional
bound on ||D f_l||<=K with residual gain beta/L yields products bounded by
(1±beta K/L)^L when beta K/L<1. E143 has no enforced global K. Its finite
six-layer teacher measurements are evidence of support, not a general deep
optimization theorem or a nonsingular class-learning Gramian.

## 230. Selectivity does not require giving up sparse affine composition

There is a further constructive extension: per-event A_i and b_i can be
computed from the incoming message, without depending on the current recurrent
state. The maps F_i(s)=A_i s+b_i compose as

\[
 (A_2,b_2)\circ(A_1,b_1)=(A_2 A_1,A_2 b_1+b_2). \tag{230.1}
\]

Associativity holds even when matrices do not commute. For diagonal complex
modes (real 2x2 decay/rotation pairs), composition costs O(m) per event. Exact
packet summaries retain the product A_P and accumulated b_P; a single constant
decay based on packet duration generally does not. Their reverse teachers are
ordinary local affine adjoints, including derivatives of message-conditioned
A_i and b_i. No complete counterfactual tree or dense token attention is needed
for these smooth interior teachers. State-dependent coefficients would break
this precomputable affine scan and require a different analysis.

This is the mathematical route to selective memory, relevant to
[S7's input-dependent state dynamics](https://arxiv.org/html/2410.03464v1).
It is not yet implemented in E143. Winning-clock boundary credit remains
separate: sorting uses the realized order, and a local loser-delay surrogate
does not compute the exact loss of a changed-order suffix. Exact coalescing,
stronger smooth representation learning and counterfactual policy quality are
three separate properties to measure.
