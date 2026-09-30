# Information, source messages and a useful sparse learning boundary

Analytic derivations, 30 September 2026. This note separates a provable
representation obstruction from the unproved claim that it explains our SHD
score. E139 implements the nested source-message intervention below. §§185,
194–196 and 210 already distinguish transferable credit, hard boundaries and
supervised observability; these results address information *before* that credit.

## 215. An input quotient places a ceiling on every downstream depth

Let X be the complete raw marked event stream, C=C(X) its coarse packet
representation, and Y its utterance label. For categorical logarithmic loss,
the best measurable predictor based on a representation has risk equal to its
conditional entropy. Therefore

\[
 R_C^*-R_X^*=H(Y\mid C)-H(Y\mid X)=I(Y;X\mid C). \tag{215.1}
\]

This follows by decomposing conditional cross entropy into conditional entropy
plus the nonnegative KL divergence from the true posterior to the predictor.
Any network using only C, regardless of depth or route optionality, has risk
at least R_C^*. A deterministic downstream feature cannot increase the
information available about Y. For optimal top-one accuracy,

\[
 A_C^*=\mathbb E\max_y\mathbb E[q_X(y)\mid C]
 \leq\mathbb E\max_y q_X(y)=A_X^*. \tag{215.2}
\]

The inequality is conditional Jensen for the convex maximum. Equality in
accuracy can hold even when the log-loss gap is positive: posterior confidence
may differ inside a coarse fiber while its best label stays the same. Conversely,
if two equally likely, differently labelled streams have identical C, no
coarse predictor can distinguish them, whereas an informative raw mark can.

These are Bayes-risk statements, not a numerical upper bound on 40-band SHD.
We have not estimated I(Y;X|C). The 700-to-40 channel reduction, closure-time
replacement and count-only packet mark are explicit quotients worth testing;
their existence alone does not prove that they removed useful label evidence.
Our 72.3% private-development score also cannot be directly subtracted from
published official-test scores to estimate the lost information.

## 216. Coalescing events is exact when their update algebra closes

For an affine event-state update s_i=A_i s_{i-1}+b_i, a consecutive block has
the exact descriptor

\[
 A_P=A_n\cdots A_1,\qquad
 b_P=\sum_{i=1}^n A_n\cdots A_{i+1}b_i. \tag{216.1}
\]

The composition (A_2,b_2) after (A_1,b_1) is
(A_2 A_1,A_2 b_1+b_2), an associative operation. For diagonal dynamics this
descriptor costs O(d) per source event and O(d) storage, without a dense time
grid. For a continuous linear state with impulsive events and generator A,
the exact source payload delivered at the causal packet closure T is

\[
 b_P(T)=\sum_{i\in P} e^{A(T-t_i)} B u_i. \tag{216.2}
\]

Here source timing performs computation. Replacing this payload by a count
times one channel embedding is exact only under extra assumptions. With scalar
decay time tau and offsets 0<=T-t_i<=w, the true mass lies between
n exp(-w/tau) and n. Closure impulses can thus overstate the integrated mass
by a factor as large as exp(w/tau). At w=10 ms,tau=20 ms this factor is about
1.649. This is an operator approximation bound, not a measured classification
error and not a statement that the existing packet model must implement raw
impulse integration.

A Taylor descriptor can retain moments
m_r=sum_i (T-t_i)^r B u_i/r!, r=0..R. The resulting payload is
sum_r A^r m_r; its norm error is at most
sum_i ||B u_i|| exp(||A||w)(||A||w)^(R+1)/(R+1)!.
This follows from the exponential-series remainder and triangle inequality.
Learned source embeddings with explicit timestamp features provide another
finite-dimensional marked descriptor. Neither descriptor is universally
lossless: input-conditioned nonlinear updates and intervening hard races do
not generally close in this affine algebra. If an early event changes a later
winner inside the block, its realized order/suffix must be retained or replayed.
More accurate payloads cannot justify deleting arbitrary computational races.

## 217. A nested source-message adapter with immediate supervised credit

Keep each old nonempty packet P, its count c_P and closing time T_P. Give
each raw cochlear event an injective address j_i=(band_i,fine_slot_i).
For 700 channels, 40 bands and 18 local slots suffice: the integer map used in
E139 is injective, including the bands containing 17 versus 18 original units.
Let E_j be an eight-component learned source embedding and
phi(t)=(1,exp(-t/.05),exp(-t/.2),exp(-t/.8)). Set

\[
 x_P=x_P^{old}+{1\over c_P}\sum_{i\in P}
                  \widetilde E_{j_i}\otimes\phi(t_i),\qquad
 \widetilde E_{bq}=E_{bq}-{1\over18}\sum_{r=0}^{17}E_{br}.
 \tag{217.1}
\]

Centering isolates fine contrasts instead of giving the coarse embedding a
second interchangeable parameterization. It is a coupling over 18 addresses
within one band. It has 5,760 stored parameters and at most 5,440 effective
contrast coordinates; some slots are unused by clean original channels but
may be reached by the existing relative-band augmentation. The representation
retains source-frequency/time interactions in 32 payload components; it does
not retain every within-packet ordering or an invertible copy of all events.

At E=0, packet values, clocks, counts, logits and all hard races equal the
parent exactly. The model class contains the old predictor, so its *optimal*
risk cannot be worse. Finite training is not guaranteed to find that optimum.
In particular, learning marks may change subsequent hard races. Zero nesting
is not a route-invariance certificate after updates; §§194–196 still apply.

For a packet teacher g_{P,ar}=partial L/partial x_{P,ar}, define the uncentered
source teacher

\[
 h_{j,a}=\sum_{P}\sum_{i\in P:j_i=j}{1\over c_P}
                     \sum_r g_{P,ar}\phi_r(t_i). \tag{217.2}
\]

The exact gradient of E_{bq,a} is
h_{bq,a}-(1/18)sum_r h_{br,a}. The completed-utterance label thus returns to
the originating events through stored local source addresses/times. It does
not need a label before the digit becomes identifiable. A zero source value
does not block this derivative: this is an additive residual with a fixed
identity injection, not a product of two initially zero trainable gates.
This source adjoint is exact for the supplied packet teacher. In E139 that
teacher includes the common core's declared surrogate route credit. Exact
source chain rule does not turn that surrogate into the exact derivative of
the discrete winning program; a fixed-route interior teacher and a policy
boundary teacher retain their different meanings.
Only actual winning deep continuations emit; the existing losing alternatives
retain their declared surrogate score credit. This adapter is not a new claim
of exact counterfactual credit across hard changes.

The additional source work is O(Nd), plus the 40x18 contrast table, for N raw
events and d=32. Deep-layer event work remains O(P L) with the same P packets
and depth L. Training stores a raw-source graph in this prototype; a local
adjoint/eligibility implementation can accumulate (217.2) instead. Both source
lookup/aggregation and backward work must be counted. This is not free
information and is not yet a measured energy advantage.

## 218. When do the new marks add trainable options rather than duplicates?

Let J_o,J_f be old and fine parameter-to-class-contrast Jacobians at a fixed
batch, with positive block metrics D_o,D_f. Put A_o=J_o D_o^(-1/2) and
A_f=J_f D_f^(-1/2). With a block-diagonal update metric, the local class-visible
control kernel grows from K_o=A_o A_o^T to

\[
 K_{o+f}=K_o+A_f A_f^T. \tag{218.1}
\]

This increase is positive semidefinite, but useful fitting correction still
requires a teacher component in its support. The genuinely new local output
subspace is measured by (I-P_o)A_f, where P_o projects onto range(A_o).
If it is zero, the new block offers no new output tangent direction at that
batch, although it can change update cost/conditioning. A finite fitting
Jacobian of a large old model can already span every batch output; tangent
novelty on that batch then is not the same question as better new-speaker
function behavior. Replacing the function domain by one batch can conceal
the representational improvement in §215.

For a fit gradient h_f^P and an independent target-distribution gradient
h_f^Q, an update -eta D_f^-1 h_f^P changes target loss to first order by
-eta (h_f^Q)^T D_f^-1 h_f^P. Extra positive fitting kernel capacity supplies
no sign guarantee for this cross-distribution term. Fine source contrasts can
recover class evidence *and* encode speaker nuisances. Retaining the proven
frequency/time augmentation, comparing clean held speakers, and checking
actual best-score improvement are therefore necessary. A larger lookup table
or a nonzero new-source teacher is not the success criterion.

## Concrete empirical implication

E138 increased the fitting budget at the strongest checkpoint's terminal
learning rate. It completed at 368/512 (71.875%), below the existing 370/512.
That result gives no accuracy-record advance. E139 changes the specific source
information boundary instead: identical packet extraction, augmentation and
old optimizer; a zero-initialized fine-source contrast branch with measured
source work and a nonzero local label teacher. The numerical contract precedes
training. Quality conclusions require its completed fitting/development result.
Published parity still requires the official SHD test protocol.

## 219. A constructive state-space subset inside winning message computation

The present normalized three-timescale receiver is not a general learned SSM.
To contain a known temporal operator, specify its state transition explicitly.
Use a winning-message layer with state s, elapsed arrival time Delta and route r:

\[
 s^+=e^{A_r\Delta}s+B_r u,\qquad
 v=C_r s^++D_r u,\qquad t^+=t+d_r. \tag{219.1}
\]

Only the chosen update changes the retained state. For a stable diagonalizable
real generator, a complex diagonal representation can be implemented by real
two-coordinate modes

\[
 e^{A_r\Delta}|_j=e^{-a_{rj}\Delta}R(\omega_{rj}\Delta),\qquad
 a_{rj}>0. \tag{219.2}
\]

This includes damped oscillatory memory, rather than three shared positive
averages. Coordinate transformations are absorbed in B_r,C_r. Defective
generators are not included exactly by this diagonal-mode argument; they may
be approximated, and no universal approximation claim is needed for this subset.
Event-input B_r u is a lookup when u denotes a source channel. Decay/rotation
costs O(d); unrestricted C_r costs O(d^2), or a declared sparse/factored output
map changes that cost and representable family. Payload-local dense maps do
not create quadratic attention over events, but their energy still counts.

Choose one route, or make one constant-delay route always win, and set all
layer clock origins to include that constant offset. Then elapsed event times
and (219.1) equal the chosen asynchronous linear SSM by induction on events.
Add the specified output gate, residual and normalization to contain the
corresponding complete block; merely implementing (219.1) is not equivalent
to an entire published six-layer classifier. The constant delay subset is an
expressivity witness. Competing learned delays extend its computation rather
than being required to express its linear memory.

With state teacher a^+, emitted-vector teacher g and a fixed winning route,
the local state adjoint is

\[
 q=a^++C_r^Tg,\quad
 a^-=e^{A_r^T\Delta}q,\quad
 g_u=B_r^Tq+D_r^Tg. \tag{219.3}
\]

For mode j, put z_j=e^{-a_{rj}\Delta}R(omega_{rj}Delta)s_j.
Its exact decay and frequency teachers are -Delta q_j^T z_j and
Delta q_j^T J z_j respectively, with J the 90-degree rotation generator.
Input/elapsed-time derivatives must also include any content dependence of
A,B,C,D. Under a fixed route, local state credit obeys
||a^-||<=exp(-a_min Delta)||q||. This upper stability bound does not prove
nonvanishing long-horizon or deep-classification credit.

For an alternate route, the counterfactual is the *paired new state, output
and clock*. Its state changes later event responses, so comparing emitted
vectors alone omits its continuation utility. Exact replay or the declared
joint stochastic policy credit in §§197–200 is required for that boundary.
Likewise, affine source coalescing in §216 does not exactly emulate arbitrary
intermediate nonlinear gates/races. Keep their relevant events, or explicitly
measure the approximation. This supplies a precise inclusion target and local
teachers while retaining sparse event execution; it does not assert that our
current SHD core has already implemented or learned it.
