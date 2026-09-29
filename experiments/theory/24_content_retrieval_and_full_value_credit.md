# Full value credit and content-dependent temporal retrieval

[Theory index](../THEORY.md) · Global sections 202–205.

## 202. Teaching all values is a different experiment from adding context maps

E131 changes only the new cross-receiver value maps. It does not test whether
all eight value layers, embeddings, memory time constants and the readout can
learn under computed immutable keys. E134 tests that larger value phase with
58,048 trainable parameters in both arms, fresh Adam, identical initial
predictions and the same fitting examples, augmentation and update budget.
The coupled arm freezes policy tensors but its keys still depend on changing
values. The separate arm freezes the functional key program itself.

For a realized fixed schedule, write a residual layer as
$x_{l+1}=x_l+\alpha_l F_l(x_l;\theta_l)$ and let
$\|D_x F_l\|_\infty\le a_l$. If $\alpha_l a_l<1$, the state Jacobian and
its dual adjoint have lower and upper transport bounds given by products of
$1-\alpha_l a_l$ and $1+\alpha_l a_l$. The exact parameter teacher is

\[
 \nabla_{\theta_j} L=B_j^T\Psi_{L:j+1}^T g,
 \qquad B_j=\partial_{\theta_j} x_{j+1}. \tag{202.1}
\]

The E134 contract finds nonzero gradients in every layer and predicts a
$-0.00022804$ fitting loss change; a radius $10^{-5}$ step realizes
$-0.00022674$ while actual winners and clocks remain identical. Large value
perturbations also preserve these keys. This verifies a supported conditional
teacher, not optimizer convergence, discriminative capacity or generalization.
The clean eight-layer state transport bounds are 0.25245 and 3.16764 for the
plain-memory configuration, including its two context channels.

## 203. Pooling changes the norm and geometry of supervised credit

For $u=\sum_i a_i x_i$, $a_i=c_i/C$, and a terminal teacher $h$, the stacked
event teacher is $g_i=a_i h$. Therefore

\[
 \|g\|_1=\|h\|_1,\qquad
 \|g\|_2=\|h\|_2/\sqrt{N_{\rm eff}},\qquad
 N_{\rm eff}=C^2/\sum_i c_i^2. \tag{203.1}
\]

Max-norm state transport gives a bound for the dual **1-norm** teacher. It
does not give a dimension-free 2-norm bound for tied parameter gradients.
Those gradients sum teachers times event features and can cancel even when
every local teacher survives. E128's class/speaker geometry measures one such
cancellation separately. Increasing depth or demonstrating nonzero gradients
does not by itself remove pooled-label dilution or create content selectivity.

## 204. Completed-prefix replay is not yet a persistent stream compiler

The present query computation consumes the supplied observed prefix and then
allows its internal delayed packets to finish. An input that becomes available
after the query cutoff can arrive before an older packet's scheduled internal
arrival. It can change the memory that the older packet sees. Consequently,
appending new input to cached **completed** prefix states need not reproduce
fresh prefix replay. This is a scheduling issue caused by computational delays.

A persistent implementation must declare the query's physical completion time,
input availability, pending-event policy and readout snapshot/version semantics.
An alternative is input backpressure, which changes the workload's time model.
Neither is a free removal of replay cost. Likewise, input features anchored to
the start of a short prefix, $e^{-t/\tau}$, decay away on a long absolute-time
stream. Relative ages or a time-gauge transformation must preserve the declared
representation when compiling it to persistent execution. E133's measured
prefix replay cost remains the reported cost until this equivalence is proved.

## 205. Learn content retrieval inside the temporal state

The old memory averages earlier payloads within each addressed time bank. Its
time selection is trainable, but it lacks a query-specific content kernel.
For positive features $\phi_k,\phi_q\in\mathbb R_+^r$, maintain

\[
 S_\tau(t)=\sum_{i\preceq t} c_i e^{-(t-t_i)/\tau}\phi_k(x_i)x_i^T,
 \quad z_\tau(t)=\sum_{i\preceq t} c_i e^{-(t-t_i)/\tau}\phi_k(x_i),
 \quad y_\tau(x)=\frac{\phi_q(x)^T S_\tau(t)}{\phi_q(x)^T z_\tau(t)+\epsilon}.
 \tag{205.1}
\]

The relation $i\preceq t$ uses the same causal receiver and deterministic tie
order as the existing memory. These states obey an affine decay/update scan;
there is no event-pair matrix. State size is $O(r d)$ per receiver/time bank,
and payload work is $O(E r d)$ for fixed feature rank and bank count, plus
sorting and the existing value projections. The hard race still emits exactly
one winning correction and delay. Losing payloads do not become forward signals.
This uses positive-feature linear retrieval, an established construction; the
candidate contribution is its controlled nesting in the current delayed races.

With weights $w_i=c_i e^{-(t-t_i)/\tau}\phi_q^T\phi_{k,i}$, denominator
$Z=\sum_i w_i+\epsilon$, $a_i=w_i/Z$, and teacher $g$, the independent
value, feature and arrival derivatives are

\[
 \partial_{v_i}L=a_i g,\quad
 \partial_{\phi_{k,i}}L=\frac{c_i e^{-(t-t_i)/\tau}}{Z}
   \phi_q\,[g^T(v_i-y)],\quad
 \partial_{\phi_q}L=\frac{(S-z y^T)g}{Z},\quad
 \partial_{t_i}L=\frac{a_i}{\tau}g^T(v_i-y). \tag{205.2}
\]

The time derivative is for an earlier independent arrival with the query time
and order held fixed. Keys/values/query derived from the same payload add their
ordinary chain-rule terms. Arrival-order boundaries still require distinct
credit; a smooth content kernel does not remove hard route boundaries.

### Balanced features preserve the old mean without a symmetry trap

Constant identical features nest the old temporal mean but provide no useful
query discrimination. Instead use paired bounded features, for $P=r/2$:

\[
 \phi_W(x)=\frac{[1+\eta\tanh(W_jx),\;1-\eta\tanh(W_jx)]_{j=1}^P}{\sqrt{2P}},
 \quad 0<\eta<1. \tag{205.3}
\]

Initialize the query matrix $Q=0$ and diverse bounded key rows $W$. Then
$\phi_Q^T\phi_W=1$ for every payload: the entire old regularized temporal
mean is recovered, while heterogeneous feature states provide a query
gradient. Key gradients start at zero under this balanced query and become
available after a query update. This is a specific, testable learning order.
The kernel remains between $1-\eta^2$ and $1+\eta^2$; the initial candidate
is deliberately a bounded content reweighting, not an arbitrary sharp softmax.

The paired kernel expands exactly to
$1+(\eta^2/P)\sum_j\tanh(Q_jx)\tanh(W_jx_i)$. Its affine state therefore
needs only the plain numerator/mass and $P$ signed key moments, with
$(P+1)(d+1)$ scalars per time bank. Positive *kernel weights* do not require
keeping redundant positive feature coordinates in storage. At $P=4,d=32$,
this is 165 rather than 265 scalars for the uncompressed feature implementation
with its extra scalar mass. The compressed denominator is
$1+(\eta^2/P)\sum_j\tanh(Q_jx)\bar z_j$, where both numerator and moments
are divided by $C+\epsilon$. At $Q=0$ it is exactly one, retaining the old
mean's regularization without a compensating division. The direct pairwise
formula is retained only as a small audit oracle.

If each row has 1-norm at most $b$ and $\|x\|_\infty\le R$, each key/query
relative feature perturbation is at most $\eta b/(1-\eta)$. The paired
kernel itself gives the sharper relative bound
$\gamma=\eta^2 b/(1-\eta^2)$ for either input, since its denominator is at
least $1-\eta^2$ and the derivative of tanh is at most one. Differentiating the normalized weighted mean
(including a fixed zero-valued prior of mass $\epsilon$) gives

\[
 \|D_x y\|_\infty\le 1+2R(\gamma_k+\gamma_q).
 \tag{205.4}
\]

The old unit-gain mean bound must therefore be replaced. A row-normalized
local correction has bound at most $a_l=\max(1,L_y)$; with a separate global
correction a conservative bound is $a_l\le2\max(1,L_y)$. Require
$\alpha_l a_l<1$ before reusing the depth certificate. Large feature strength,
unbounded projections or sharp retrieval can invalidate that certificate.
The implementation records expanded scan width, projections, retrieval and
materialized state. It must not hide its extra work inside an unchanged count
of scalar scan *compositions*. E135 audits these identities and learning
directions before interpreting a speech continuation.
