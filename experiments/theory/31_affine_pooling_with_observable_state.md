# Preserve the pooled state as well as the final state

Derived 30 September 2026. This extends §227's packet endpoints to an exact
weighted query over the intermediate **linear** states. It addresses a specific
reference-architecture difference: EventSSM pools state representations before
its dense output transformation. [EventSSM pooling, §3.2](https://arxiv.org/html/2404.18508v2).

## 231. An augmented local program, including the query

Suppose incoming messages determine affine updates
s_i=A_i s_{i-1}+b_i. Coefficients may depend on incoming message/time, but not
on the current recurrent state. To query a weighted mean of raw states, carry
q_i=e_i q_{i-1}+w_i s_i and n_i=e_i n_{i-1}+w_i, where e_i is zero at the
start of a new packet and one otherwise. This is a triangular affine map:

\[
 \begin{bmatrix}s_i\\q_i\end{bmatrix}=
 \begin{bmatrix}A_i&0\\w_i A_i&e_i I\end{bmatrix}
 \begin{bmatrix}s_{i-1}\\q_{i-1}\end{bmatrix}
 +\begin{bmatrix}b_i\\w_i b_i\end{bmatrix}. \tag{231.1}
\]

A descriptor (A,b,R,d,e,n) acts as s'=As+b, q'=Rs+e q+d, n'=e n_old+n.
For first descriptor 1 followed by descriptor 2, its closed composition is

\[
 \begin{aligned}
 A_{21}&=A_2 A_1,& b_{21}&=A_2 b_1+b_2,\\
 R_{21}&=R_2 A_1+e_2 R_1,& d_{21}&=R_2 b_1+e_2 d_1+d_2,\\
 e_{21}&=e_2 e_1,& n_{21}&=e_2 n_1+n_2.
 \end{aligned} \tag{231.2}
\]

The map's composition is associative, including resets. For diagonal complex
modes, all products above are pointwise O(m) operations. The weighted query
therefore needs a constant factor of additional modal work, without an
all-event dense output map or an event-pair matrix. An inclusive pair-reduction
prefix requires fewer than 2N descriptor combines, each charging its expanded
arithmetic. It is not the same cost as a basic two-field affine combine.

A receiver boundary sets A_i=0 and e_i=0, preventing both state and query
contamination across utterances. A packet boundary only sets e_i=0, retaining
the receiver's past state. The reference streaming program is exactly these
two local updates. At a packet's last raw event, q_i/n_i equals the weighted
mean of its raw linear states, while s_i remains the final state for the next
packet. Waiting to a later causal closure additionally transports s_i and leaves
the historical query unchanged.

## 232. Query-visible teachers survive exact aggregation

For a closure query h=g(C(q/n)+D u_bar), the local state-query teacher is

\[
 v_q=C^T Dg^T v_h/n,\qquad
 v_n=-\langle v_q,q\rangle/n. \tag{232.1}
\]

At a raw update, incoming adjoints v_s,v_q return through the augmented program:

\[
 v_s^{\rm old}=A_i^T(v_s+w_i v_q),\quad
 v_q^{\rm old}=e_i v_q,\quad
 v_{b_i}=v_s+w_i v_q. \tag{232.2}
\]

Coefficient A_i has teacher (v_s+w_i v_q)s_old^T. Weight w_i also receives
<v_q,s_i> plus its mass teacher, and any u_bar path. Thus a pooled query
directly teaches intermediate states even when only one vector is emitted.
Its exact identity for every parameter value also proves equality of the
coalesced and raw-program smooth adjoints. State/query correlations are kept
inside the descriptor instead of being discarded by an endpoint-only readout.

For complex coordinates, transpose above means the corresponding real-pair
adjoint (or conjugate transpose with real-part inner products). Numerical
contracts must include source/drive, decay, frequency and pooling-weight
teachers. A small state error alone is not sufficient.

## 233. What this closes, and what it leaves measurable

The implemented `affine_packets.py` carries exactly this augmented program;
its E144 experiment compares parallel descriptors with a serial weighted-query
reference and their teachers, including message-dependent modal transitions.
The completed numerical experiment gives maximum state/query error 2.81e-15
and teacher error 1.60e-14, with 45 expanded descriptor combines for 27 events.
It is not used by E143, which tests the previously declared endpoint design.

The descriptor contains the state average that a subsequent dense C map and
nonlinear gate can consume. It cannot reconstruct arbitrary separate nonlinear
raw-event emissions: those are different queries. It also does not guarantee
class observability, transfer, useful race-boundary credit or calibrated early
decisions. Its practical opportunity is narrower and concrete: richer reference
pooling can be represented exactly while evaluating expensive nonlinear payload
maps only for emitted packets. Streaming state/query memory is O(m) per active
receiver; the current autograd prefix materializes O(Nm) intermediates. These
memory boundaries must not be confused when discussing training efficiency.
