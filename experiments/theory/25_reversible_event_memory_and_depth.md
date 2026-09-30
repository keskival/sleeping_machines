# Reversible event memory, deep credit and observable state

[Theory index](../THEORY.md) · Global sections 206–209.

Orthogonal recurrent transforms, scattering matrices and reversible networks
are established tools. This note applies their norm-preserving structure to
addressed packets, winning races, local memory and computational delays. It
supplies a conditional transport theorem and an event-scan prototype. It does
not establish classification quality or a new general reversible-network idea.

## 206. Exchange a packet with one addressed memory state

Let a hard winning key select one receiver state $s\in\mathbb R^d$ and an
angle $\theta$. For arriving payload $v$, perform

\[
 \begin{bmatrix}s'\\v'\end{bmatrix}=
 \begin{bmatrix}\cos\theta I&\sin\theta I\\-\sin\theta I&\cos\theta I\end{bmatrix}
 \begin{bmatrix}s\\v\end{bmatrix}. \tag{206.1}
\]

Store $s'$, emit only $v'$ with the winning delay, and leave other receiver
states unchanged. Losers are counterfactual exchanges, not emitted vectors.
The block is orthogonal. On fixed keys, angles, arrival orders and clocks,

\[
 \|s'\|_2^2+\|v'\|_2^2=\|s\|_2^2+\|v\|_2^2. \tag{206.2}
\]

Every event acts as an orthogonal map on the **augmented** packet/state vector.
Composing events and layers therefore gives an orthogonal map regardless of
depth, including all initial/final receiver states. This avoids multiplying
a worst-case residual gain bound by depth. It supplies a richer memory
exchange at fixed norm than merely making each residual correction smaller
as depth increases. Its packet interactions still require useful key choices.

The condition that keys and angles are independent of the value input is
essential. A computed separate key program can supply them. If a value-dependent
angle is differentiated, the Jacobian also contains
$[v';-s']\nabla_v\theta$ and the isometry theorem no longer applies to that
full Jacobian. Policy learning and value transport must retain this distinction.

For a receiver's events, $s_i=a_i s_{i-1}+b_i v_i$ with
$a_i=\cos\theta_i$, $b_i=\sin\theta_i$. An affine prefix scan computes
inclusive states in fewer than $2E$ combines. An exclusive shift then gives
$v_i'=-b_i s_{i-1}+a_i v_i$. Segment boundaries insert the declared initial
state. No division by $\cos\theta$ is needed, including at a complete swap.
The sequential execution updates one vector state and emits one packet per
event, using $O(d)$ payload work and $O(d)$ state per receiver. Parallel training
materializes event states; sorting, policy evaluation and reverse credit remain
additional work. The E136 primitive exposes these boundaries explicitly.

## 207. The backward signal uses local post-exchange values

For incoming teachers $g_{s'},g_{v'}$, the exact reverse event is

\[
 g_s=\cos\theta\,g_{s'}-\sin\theta\,g_{v'},\qquad
 g_v=\sin\theta\,g_{s'}+\cos\theta\,g_{v'},\qquad
 \partial_\theta L=g_{s'}^T v'-g_{v'}^T s'. \tag{207.1}
\]

The augmented adjoint norm is preserved. The angle teacher needs the local
post-event state/payload and their returning teachers, rather than a dense
comparison to all earlier packets. Tied angle parameters still sum event
teachers and can cancel. A route alternative changes both emitted content and
the selected bank's future state; credit based only on its immediate emitted
vector misses the latter. Joint policy credit from §§196–200 applies to its
actual suffix or declared stochastic expectation, with that cost counted.

Timing can also compute without forcing value contraction. For skew-symmetric
$A$, an inter-arrival state evolution $s(t+\Delta)=\exp(\Delta A)s(t)$ is
orthogonal, and $\partial_\Delta s=A s(t+\Delta)$. Independent two-coordinate
rotations give an $O(d)$ implementation. Learned frequencies and winning
delays then control phase transformations of the memory. Positive leakage
$\exp(-\rho\Delta)$ can be added to implement deliberate forgetting, with
its contraction recorded separately. This connects the verified periodic
state computation to a general event-memory transport design. The timing
extension is derived here; E136 currently implements the packet/state exchange.

## 208. Boundary observability determines what the theorem can teach

The state is part of the computation, so it must be part of the theorem and
the query semantics. At $\theta=\pi/2$ with initial $s=0$, a one-event
exchange emits $v'=0$ and stores $s'=v$. A classifier reading only emitted
packets has lost all input information in this example despite a perfectly
orthogonal augmented program. Reading or explicitly draining final state
exposes that information. This does not require an invented extra physical
spike; a completed query must specify which receiver states it reads.

With initial states fixed, the input-packet embedding into all emitted packets
and final states has an isometric forward Jacobian $J$, $J^TJ=I$. For an
arbitrary terminal teacher, $J^Tg$ projects it onto reachable variations;
its norm can be zero when $g$ lies in the orthogonal complement. Thus the
theorem proves well-conditioned **reachable value transport**, not a lower
bound for every possible readout teacher or parameter gradient. Learned
readout alignment, class/speaker transfer, key-policy reachability, memory
capacity and optimization remain substantive requirements.

The prototype contract compares linear scans to sequential event execution,
checks norm and adjoint conservation, verifies local angle credit, and computes
the complete augmented Jacobian for twelve layers. It includes the hidden-state
counterexample so that a future sequence classifier cannot claim deep
trainability while omitting its memory from the supervised boundary. This is
a candidate composable memory primitive; the contract is not a classification
training result.

The subsequent SHD adapter makes twelve value exchanges conditioned on the
inherited eight-layer key program, with the final query reading all retained
layer states. A four-utterance contract gives nonzero angle teachers at every
layer, preserves actual winners/clocks under learned-value perturbations,
and keeps the key program immutable. The adapter has a larger 6,945-feature
terminal readout and 140,428 trainable parameters, mostly in that readout.
Fitting progress alone can therefore come from the decoder. Completed training
fits all 1,024 fitting examples, while resetting its learned angles retains every
fitting decision and lowers held-speaker accuracy from 57.8% to 54.3%. The
compact matched-angle intervention in §§210–213 determines what exchange
adaptation contributes under a smaller query; the isometry theorem alone
does not answer that question.

## 209. Angular reachability and a precise role for optionality

For an elementary exchange, the generator acting on its post-event primal
state is $a_k=[v';-s']$. Under the conditional orthogonal suffix $U_k$, its
terminal control tangent is $J_k=U_k a_k$, hence

\[
 \|J_k\|_2^2=\|s'\|_2^2+\|v'\|_2^2,\qquad
 \partial_{\theta_k}L=g^T J_k. \tag{209.1}
\]

Available angular variation does not decay merely because later layers are
added. Its **alignment** can still vanish: a radial loss teacher is orthogonal
to rotation tangents. A classifier needs an orientation-sensitive readout.
Tied angles aggregate tangents across events and can also cancel. Count the
rank and geometry of the supported tangents, rather than treating the number
of candidate routes times depth as independent useful choices.

For a declared second moment of future teachers $M_f=\mathbb E[g_f g_f^T]$
and control costs encoded by $\sigma_k$, define
$G=\sum_k\sigma_k^2 J_kJ_k^T$. Then

\[
 \mathbb E_{g_f}[g_f^T G g_f]=\operatorname{tr}(G M_f),
 \quad \operatorname{tr}(U G U^T)=\operatorname{tr}G. \tag{209.2}
\]

This gives an explicit limited case for scalar option reserve: under an
isotropic future teacher, its expected quadratic correction capacity is
proportional to trace G and survives an orthogonal suffix for a zero-mean
isotropic teacher. With a nonzero mean, its squared mean term must also be
included (§161, §212). Under anisotropic
future teachers, direction matters; trace alone is insufficient. This is a
limitation of that particular scalar statistic, not of a state-conditioned
scalar future-value critic that learns the relevant alignment from its state.
The Gramian is a
reachable local correction statistic, not proof that a future optimizer will
find useful routes. Actual optionality must retain its declared future task,
adaptation budget and proposal support, as in §§159–163.

Noncommuting controls explain another kind of future reserve. Elementary
skew generators on a connected component obey
$[A_{ab},A_{bc}]=A_{ac}$. When those edges can be revisited with independent
controls, their Lie closure generates rotations between indirectly connected
coordinates. Commuting or disconnected controls have smaller reachable
families. A sparse communication graph can thus offer rich transformations
without a dense simultaneous map. Causal event order and available repeated
visits must permit the control sequence; static graph connectivity alone does
not prove reachable transformations in an asynchronous schedule.

The current scalar-angle primitive applies the same exchange to every payload
coordinate. It generates an event/state-index rotation tensored with the
payload identity, rather than all coordinate rotations. Additional sparse
channel rotations can enlarge that family. Their angle teacher has the same
local antisymmetric pairing, but their actual work and control support must
be accounted for. This identifies concrete expressivity and initialization
questions beyond the norm-preserving depth theorem.
