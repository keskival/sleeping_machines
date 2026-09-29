# Active communication topology and causal context

## 188. Static group overlap does not guarantee an active path

The carrier alternates width-8 frequency partitions and partitions shifted by
four channels. Overlapping groups permit communication through existing
packets. They do not emit relay packets where no input occurred. Consequently
the active causal communication graph depends on the example, not just on the
nominal group layout or number of layers.

At layer $l$, define an event dependency edge $j\to i$ when the events have
the same receiver key and $j$ precedes $i$ in actual arrival order, including
the declared tie rule. The next payload depends on its own old payload and
this causal receiver history. Receptive sets therefore compose these
example-specific relations. Increasing depth creates useful paths only when
the required intermediate packets actually exist.

### An expressivity obstruction that survives arbitrary depth

Consider two unit-count events at fixed times 0.1 and 0.2. The first carries
bit $a$ through channel 0 or 1; the second carries bit $b$ through channel 38
or 39. They never share a receiver in either partition. At every depth their
payloads and delays depend separately on $a$ and $b$. The utterance count is
fixed at two, and the mean-plus-affine head has class-score difference

$$D(a,b)=f(a)+g(b)+k.\tag{188.1}$$

Whitening and scaling are fixed affine maps and preserve this form. In
particular, $D(0,0)+D(1,1)=D(0,1)+D(1,0)$. Strict XOR classification would
make the left side negative and the right side positive, a contradiction.
This rules out perfect classification of the four balanced patterns for
this configuration, regardless of hidden width, depth or optimization.

The scalar weighted pool does not remove this obstruction. Write its positive
mass as $A(a)+B(b)$. Multiplying the affine score difference by that mass
again gives a sum of a function of $a$ and a function of $b$, since the static
count feature is constant. The same four-pattern contradiction applies.
More generally, disconnected active components followed only by this readout
cannot express every joint decision over those components.

E126 checks the additive identity numerically (error $1.86\times10^{-9}$) and
the weighted numerator identity (zero measured error). This is a genuine
architectural limitation in the tested configuration. It does not prove that
SHD's observed class confusions are caused by disconnection: real utterances
contain many intermediate packets. Measure their active paths and retained
joint information separately. Evidence/periodic primitives and other model
configurations are outside the assumptions of this obstruction.

## 189. A causal context channel supplies missing joint information

At selected layers maintain an additional exponential numerator/mass state
per utterance and time constant, driven only by observed winning input
payloads. Each current event reads that state's causal normalized payload
$h_i^G$ and bounded mass feature. Local state remains available.

For choice $k$, augment the value and route score by

$$u_{ik}=\tanh(W_k f^L_{ik}+U_k f^G_{ik}+b_k),\qquad
s_{ik}=r_k^T f^L_{ik}+v_k^T f^G_{ik}+c_k.\tag{189.1}$$

The hard delay race still selects one $u_{ik}$ and one delay. Losing values
remain training-only alternatives. Global context is an additional local
state receiver for the active stream, not dense attention over all event
pairs. Three exponential states cost $O(EKd)$ updates at a selected layer;
their state size is $O(Kd)$ per utterance. Existing event sorting and maps
must also be charged. No new relay packet or hidden time tick is introduced.

Set $U_k=v_k=0$ initially. The old scores, values, winners and delays are then
identical. Normalize $W_k$ and $U_k$ separately to row $\ell_1$ bounds of one;
the new denominator is clamped at one and is constant near $U=0$. This avoids
an absolute-value normalization cusp at the new zero columns. It permits
ordinary first-order credit to the new map without destroying the old state.

The context allows the second event in §188 to observe the first. Two value
coordinates can form a symmetric bump
$\tanh(S+b)-\tanh(S-b)$ of their causal averaged signed content $S$.
Equal bits give larger $|S|$ than opposite bits. An affine head separates
their bump values. E126 constructs this computation and classifies all four
patterns with positive margin 0.00338. This is a representability witness,
not an empirical claim that gradient descent learned XOR.

### Conditional stability through depth

Fix receiver choices, winning choices, counts and arrival times. Normalized
memory is a causal positive averaging operator with row sums at most one.
The separately bounded local and context maps therefore make the residual
correction $F_l$ at most 2-Lipschitz in the maximum norm across packet payloads;
without context its bound is one. For $x\mapsto x+\alpha F_l(x)$,

$$(1-a_l\alpha)\|x-y\|_\infty\le
 \|x+\alpha F_l(x)-y-\alpha F_l(y)\|_\infty\le
 (1+a_l\alpha)\|x-y\|_\infty,\tag{189.2}$$

where $a_l=2$ for context layers and one otherwise. If $a_l\alpha<1$,
the inverse is Lipschitz; the adjoint Jacobian has corresponding lower/upper
bounds in the dual $\ell_1$ norm wherever differentiable. With $\alpha=\beta/L$
and $m$ context layers, the composed bounds are

$$\prod_l(1-a_l\beta/L)
 =(1-\beta/L)^{L-m}(1-2\beta/L)^m,
\quad \prod_l(1+a_l\beta/L).\tag{189.3}$$

These stay finite and nonzero as depth grows at fixed $\beta$, including the
all-context limit $e^{-2\beta}$ to $e^{2\beta}$. The eight-layer/two-context
configuration has lower bound approximately 0.252 and upper bound 3.168.
This protects payload/adjoint transport under the stated frozen topology and
clock conditions. It does not prove route discovery, individual parameter
gradient support, Euclidean conditioning independent of packet dimension,
generalization or optimization across changing arrival orders.

### Teaching a zero-initialized context map

For a realized choice, value-map credit at zero is the outer product of its
local preactivation teacher with $f^G_{ik}$, scaled by $\alpha$ and the tanh
derivative. The independent normalization is constant around the new zero
matrix. Existing parameters therefore retain their old exact gradients while
the added columns can receive a nonzero first-order signal. The new route
columns likewise receive score/timing and the existing detached loser credit
multiplied by global features. That surrogate remains distinct from an exact
gradient at a discrete route boundary.

The real-checkpoint E126 contract gives exact initial logits, summaries,
winners and existing parameter gradients. The four new map gradient norms
are 5.963, 0.236, 5.719 and 0.704. Query isolation is preserved within
$1.43\times10^{-6}$; changing a future input leaves the first context payload
exactly unchanged. Two width-32 context layers add 6,534 parameters. A matched
one-epoch SHD continuation starts from the 72.3% checkpoint with the old Adam
state, identical fitting/augmentation RNG and learning rate. Only new columns
receive fresh Adam state. The contract proves a reachable teaching direction
and removes the stated expressivity obstruction; held-out improvement remains
an empirical question.

This also constrains optionality. If every alternative at every layer stays
inside the same disconnected active components, adding more race alternatives
or shadow samples cannot represent the excluded joint decision. Their attainable
correction space inherits the same information restriction. The context columns
add a different source of conditional information while leaving the initial
forward choice unchanged. Count useful counterfactuals by independent attainable
corrections and communication support, not by nominal alternatives times depth.

The full-update continuation finishes at 351/512 (68.6%) versus the unchanged
starting checkpoint's 370/512. The added maps learn nonzero norms, but this
screen does not improve SHD. A second arm imposes the optimization constraint
$\Delta\theta_{\rm old}=0$ and teaches only the added context columns. All old
state tensors remain bit-for-bit fixed. It finishes at 3,349/4,096 fitting
utterances (81.8%) and 364/512 held-out utterances (71.1%). Both arms share
initial predictions, examples, augmentation RNG and update schedule. New-only
teaching performs better than the full-update arm and the matched old-model
continuation (353/512), but remains below the starting checkpoint. Thus the
new subspace is empirically teachable, without a new generalization best.
Holding old parameters fixed does not preserve their predictions once new
context changes the winning computation.

The class audit sharpens the distinction. New-only teaching recognizes class 3
on 121/206 fitting utterances and 4/26 held-out utterances, versus 82/206 and
1/26 for the matched mean continuation. Class 19 is 99/204 fitting and 1/26
held out, versus 86/204 and 2/26. Extra conditional information and supported
credit are therefore separate from cross-speaker transfer. The next diagnostic
must measure class/speaker teacher alignment and temporal information retained
in the added causal state, rather than infer transfer from parameter gradient
norm or aggregate fitting accuracy. Results and paired class counts are in
`results/e126/summary_20260929.json`.
