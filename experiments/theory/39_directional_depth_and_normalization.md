# Live teachers need usable finite output directions

Derived 30 September 2026 after E159/E154. Corrected six/twelve-block runs have
the same 400/512 decisions, 5,574/6,144 fitting answers and 513/657 reused-audit
answers. Removing the six appended trained blocks changes zero audit decisions.
Their output-map norms reach about 0.031, while norm-gain vector norms stay
about 4.4e-5. Live teachers do not establish useful additional representation.

## 261. A normalized correction can erase its radial teacher

For centered z, r²=||z||²/d, equal gain gamma and epsilon>0,

\[
 N(z)=\gamma z/\sqrt{r^2+\epsilon}. \tag{261.1}
\]

Its centered tangential eigenvalue is gamma/sqrt(r²+epsilon); its radial
eigenvalue is gamma epsilon/(r²+epsilon)^(3/2). At zero both equal
gamma/sqrt(epsilon). Once r² exceeds epsilon, radial credit has an extra
epsilon/(r²+epsilon) suppression. A tiny global gain bounds every correction,
including hidden directions invisible to the current head. It can make added
depth safe yet irrelevant. E154 proves its fitted decision irrelevance at
this budget; radial saturation alone is not established as the sole cause.

## 262. Normalize sufficient features before the zero output map

An alternative appended block is

\[
 \phi=\operatorname{LN}(s)\odot\sigma(G\operatorname{GELU}(x)+b),\quad
 v=x+\alpha R A\phi,\quad A_0=0. \tag{262.1}
\]

The signed stable modal state s uses actual event intervals; the block emits
one winning delayed vector. At A=0 it preserves values and old teachers,
with the same six-ms initial delay. New output credit is

\[
 \nabla_A\ell=\alpha R^T\sum_i g_i\phi_i^T. \tag{262.2}
\]

Nonzero normalized states give immediate support. For fixed features,
map-to-output dependence is linear: post-map normalization cannot erase its
radial teacher. State/gate teachers become observable when A changes. This
is a different residual operator, not exact inclusion of every old normalized
block or a global convergence guarantee.

## 263. Preserve invisible directions while conditioning the observer

For fixed initial H=P W, define

\[
 R=(I+H^T H)^{-1/2}. \tag{263.1}
\]

A sensitivity sigma receives output unit 1/sqrt(1+sigma²), giving observed
sensitivity sigma/sqrt(1+sigma²)<1. On ker(H), R=I. R is invertible; C=R A
spans the complete original output-map family. These are optimizer units,
not a reduction in expressive rank. Unlike a worst-direction scalar bound,
hidden directions that may support later computation retain unit scale.

With initialized norm gain one/bias zero, ||phi_i||<=sqrt(m). For fixed
features and the completed count-mean query at an identity suffix,

\[
 \|P\Delta z\|\leq\alpha\|\Delta A\|_2\sqrt m,\qquad
 KL(p\Vert p_{new})\leq\alpha^2m\|\Delta A\|_2^2/4. \tag{263.2}
\]

These are fixed-observer bounds. Simultaneous blocks have additive initial
linearized effects; later state, normalization, head and clock changes require
actual finite replay. Invisible excursions and cross-speaker transfer are not
bounded by this class observer. The live teacher can still cancel across data.

## 264. Fold units into the map and charge their training work

For H=U diag(sigma) V^T, use
A−V diag(1−1/sqrt(1+sigma²)) V^T A. Its factor rank is at most the number of
classes. It acts on a parameter matrix once per block forward, with 2*d*r*m
MACs, not on event pairs. At deployment materialize R A once into the ordinary
map and remove its parametrization; packet projection work is unchanged.
Local maps are dense, while event work remains linear in supplied packets.
Added layers still cost real work and latency.

`observer_conditioned_depth.py` implements this proposal. Check identity/old
teachers, the explicit new teacher, spectral bounds, finite KL and deployment
folding before actual fitting replay and a declared continuation. Initialization
contracts alone are not evidence of improved speech recognition.

**Completed E161 numerical contract:** initial values and old teachers agree
exactly in float64. All six new output teachers are nonzero (0.0398–0.0484);
the explicit formula differs by at most 1.11e-16. Deployment folding gives
identical logits, inverse-coordinate error is 4.44e-16 and the radial
normalization teacher differs by 4.06e-19. The fixed-observer example's
KL is 3.47e-6 under its 8.86e-4 bound. These are local mathematical checks.

**Completed actual E162 fitting replay:** old LR stays 0.0000203125. For six
new width-128 blocks with gains 1/6, a fresh coordinate step has
||delta A||₂<=eta sqrt(d*m); equation (263.2) motivates proposal rate
sqrt(4*kappa)/(sum alpha sqrt(m)*sqrt(d*m))=0.0001953125 for kappa=0.02.
This is a conservative fixed-feature proposal unit, not a joint-network bound.
Actual replay accepts it: batch CE falls from 0.622628 to 0.325345 and fitting
anchor mean KL is 0.004574. The initial head sensitivity is 912.9976; largest
conditioned sensitivity is 0.9999994, with minimum output unit 0.00109529.
The same declared data/teacher/moments are used for all six new-rate proposals.
Full-training utility must be read from completed E163/E164 evidence.
