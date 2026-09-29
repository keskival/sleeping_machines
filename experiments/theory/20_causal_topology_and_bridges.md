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
