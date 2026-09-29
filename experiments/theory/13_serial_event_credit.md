# Serial event transport and a sequence-level credit certificate

29 September 2026. Continues §§164–165. E117 is a capacity diagnostic for
§157, with fixed sparse addresses and terminal supervision. The bounds below
are analytical; measured recognition belongs in the results section.

## 166. Stable memory is a sub-Markov operator on an event sequence

Let event i have a fixed address a_i, time t_i, positive count c_i, and
payload h_i. A receiver maintains, for each positive time constant tau_k,

\[
 z_{ik}=e^{-(t_i-t_{prev})/\tau_k}z_{prev,k}+c_i h_i,\qquad
 m_{ik}=e^{-(t_i-t_{prev})/\tau_k}m_{prev,k}+c_i.
\]

Only that receiver is updated. The elapsed time is since its last addressed
event, not since a global tick. Initially z=m=0. Define
u_ik=z_ik/(epsilon+m_ik), epsilon>0. For fixed addresses/times/counts,

\[
 u_{ik}=\sum_{j\preceq i,\ a_j=a_i}
 P_{ij}^{(k)}h_j,\quad
 P_{ij}^{(k)}=
 \frac{c_j e^{-(t_i-t_j)/\tau_k}}
 {\epsilon+\sum_{r\preceq i,a_r=a_i}c_r e^{-(t_i-t_r)/\tau_k}}.
\]

Here j≼i includes the declared causal tie order for simultaneous arrivals.
Every P coefficient is nonnegative and each row sum is at most one.
Consequently **the full history operator is nonexpansive in the maximum
event-coordinate norm**: ||P H||_inf <= ||H||_inf. This includes all reuse
of old events by future queries; it is not a bound on just one read.
Stacking a fixed number of banks as separate coordinates retains norm one.
Mass features m/(1+m) retain bounded rate information without changing this
payload derivative, because mass is independent of H.

This choice of norm is substantive. Reusing a past event many times can
increase Euclidean sequence norm. A uniform infinity-norm certificate does
not imply a length-independent Euclidean certificate; converting norms can
cost sqrt(E d). It does imply a useful **dual credit** bound below.

## 167. A whole-sequence residual map that cannot extinguish payload credit

Concatenate the current payload, the normalized memory banks, and the mass
features. Apply a linear map whose absolute row sums are at most one,
followed by tanh, and use the result F_l(H) as a residual correction:

\[
 H_{l+1}=H_l+\alpha_l F_l(H_l).
\]

E117 enforces the row bound by dividing each weight row by
max(1, sum_j |W_ij|). At fixed event addresses, times, counts and model
parameters, the concatenation has payload Lipschitz constant one in the
maximum norm, the linear map has constant at most one, and tanh has constant
at most one. Therefore ||F_l(H)-F_l(H')||_inf <= ||H-H'||_inf. The triangle
and reverse triangle inequalities give

\[
 (1-\alpha_l)\|H-H'\|_\infty
 \le \|H_{l+1}-H'_{l+1}\|_\infty
 \le (1+\alpha_l)\|H-H'\|_\infty,
 \qquad 0<\alpha_l<1.
\]

For L stages with alpha=beta/L<1, multiply the bounds:

\[
 a_L=(1-\beta/L)^L,\qquad b_L=(1+\beta/L)^L.
\]

With beta=1 and L=8, a_L=0.3436 and b_L=2.5658. In contrast to a product
of firing probabilities, these bounds stay separated from zero and infinity
as depth grows. Each residual is globally invertible on its finite payload
space: H=Y-alpha F(H) is a contraction. This is the standard invertible
residual argument, here applied to a sparse **history** operator. See
[Behrmann et al., 2019](https://proceedings.mlr.press/v97/behrmann19a.html).

At differentiable points, ||J||_inf <= b_L and ||J^{-1}||_inf <= 1/a_L.
Using induced-norm duality gives an explicit backward certificate:

\[
 a_L\|g_{out}\|_1\le\|J^Tg_{out}\|_1\le b_L\|g_{out}\|_1.
\]

**Thus total absolute payload credit cannot vanish across the stack**, even
when persistent memory fans one event into many later reads. This closes
the fixed-history sequence gap in §157 for this particular construction.
The bound is for payload credit, not every parameter gradient. A class head
may ignore informative coordinates; a pooled readout is not invertible;
parameter Jacobians can be rank deficient; useful features still have to
learn. It is not a theorem of convergence, generalization, Euclidean
dynamical isometry, or superiority to attention.

The constants do not cover learned event identities, counts, cancellations,
or event order changes. Learned positive decay times preserve the payload
bound at every parameter value, but their parameter derivatives require
separate analysis. Optional branches need the control and budget conditions
of §§157/163. E117's addresses are fixed; it does not yet exercise route
optionality or learned output delays.

### Topology and work

At each depth, one input packet addresses one local frequency group. The
next stage shifts the group boundaries by half a group. This permits
overlapping local histories to interact through successive nonlinear
transformations, while every packet retains exactly one continuation.
There are L E receiver updates, no silent-unit ticks and no E² pair matrix.
This is local state compression, not exact attention over every past value.

E117's CPU training reference evaluates an associative segmented scan, with
O(L E log E) scan compositions and O(L E k d log E) autodiff storage in the
straightforward implementation. Padding is avoided by grouping actual
arrivals. The serial recurrence uses O(L E k d) state arithmetic plus local
payload projections. Dense matrices still mix the coordinates inside each
active payload; sparsity is in addressed units and event times. No measured
energy advantage follows just from these operation counts. Diagonal
recurrence and forward normalization also have established precedent in
[Orvieto et al., 2023](https://arxiv.org/abs/2303.06349); E117's diagnostic is
the specific causal local routing, normalized history, and bounded serial
transport, not a claim to invent residual recurrence.

## 168. A causal count mark has a release time

The old `e71_event_cde.events` groups same-band spikes separated by short
gaps, then attaches the **final group count to the first spike**. A future
spike can change that earlier payload. This is offline compression, not a
causal packet stream, even if the following network is causal. The issue
does not by itself explain terminal chance accuracy, and correcting it
does not retroactively validate early-answer metrics.

E117 uses fixed half-open windows [j Delta,(j+1)Delta) relative to recorded
utterance start, and emits a count only at (j+1)Delta. Empty windows do not
emit. The packet count is measurable from the history at release; future
events cannot change it. At a boundary, process the closing packet before
new events from the next window. Packet latency is at most Delta, which
must be included in an eventual early-decision latency comparison. Times
are never divided by the future utterance duration.

Terminal pooling is count weighted, sum_i c_i H_i / sum_i c_i. Its direct
payload derivative is c_i/sum_j c_j times the head gradient. Although
individual event credit is small in long sequences, gradients into shared
parameters sum across those events. Suppression of information at every
depth is therefore not needed merely to control count-induced logit gain.
An explicit time×band input embedding preserves coarse temporal-frequency
statistics even in the shallow control; it is a declared representational
change from E83, not a matched one-variable repair of that model.

### Decisive measurements

1. Independent sequential versus scan outputs and gradients; causal prefix
   agreement; finite-difference payload derivatives; packet release times.
2. Deepest-only terminal train and held-out-speaker loss/accuracy. No
   auxiliary classifier is allowed to stand in for the deepest layer.
3. Matched shallow learning and a trained-stack bypass diagnostic. Bypass
   uses a head trained on different features, so it alone does not prove a
   benefit of depth. Frozen layerwise probes isolate accessible information.
4. Only after useful deep representations learn: adaptive timing/routing,
   causal prefix calibration, and measured work/energy at matched quality.

These experiments distinguish stable transport from useful computation.
