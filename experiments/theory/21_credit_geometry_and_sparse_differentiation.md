# Credit geometry, realized utility and sparse differentiation

## 190. Counterfactual coverage does not require every value's backward graph

Write one layer's value alternatives as $v_k=V_k(f_k;\theta)$, its winning
choice as $r$, its relaxed routing probabilities as $p_k$, and its residual
increment as $\alpha$. The implemented zero-forward estimator is

$$y=x+\alpha v_r+
 \alpha\sum_k[p_k-\operatorname{sg}(p_k)]
                 \operatorname{sg}(v_k-v_r).\tag{190.1}$$

Here $\operatorname{sg}$ stops differentiation. The corresponding clock
estimator substitutes each delay difference for the vector difference. For
downstream vector and clock teachers $g$ and $q$, its first-order credit is

$$\nabla_\theta\ell=\nabla_\theta\ell_{\rm path}+
 \sum_k\nabla_\theta p_k\left[
       \alpha g^T(v_k-v_r)+q(d_k-d_r)\right].\tag{190.2}$$

Only the winning value needs ordinary differentiation. Every losing value
still needs evaluation to supply its router comparison. Consequently the
complete estimator can be implemented with all detached alternative values
and a differentiable winning value. This also applies to feature, memory and
earlier-layer credit: the direct value VJP reaches the winning input features,
while the score VJP reaches all score features. Detaching the entire memory
would remove that second path and would not be equivalent.

E127 implements an optional `value_backward="winner"` backend. It recomputes
the winner and adds its zero-valued differentiable difference to the original
detached winning value. Forward results remain exact; contraction rounding
can change the recomputed derivative slightly. Tracing retains the full graph
so a caller can still differentiate a returned alternative explicitly.

For $E$ events, $K$ choices, value width $d$ and feature width $f$, the original
dense value contractions cost approximately $6EKdf$ FLOPs: one forward and
two VJPs, each charged two per MAC. The implemented replacement costs
$2E(K+3)df$: $K$ detached forward maps, one winning recomputation and two
winning VJPs. At $K=3$ this is one third less value-map contraction work.
Normalization, memory, routing, nonlinearities, sorting and optimizer work
are additional costs. Both cost formulas assume both contraction VJPs are
needed; freezing inputs or values changes that boundary.

On four double-precision contracts, including context maps, empty losing
options and seven choices, measured first-order differences are below
$10^{-11}$. At two actual D8 checkpoints, logits and summaries are exact and
relative parameter-gradient errors are $7.23\times10^{-8}$ and
$8.87\times10^{-8}$. However, alternating one-thread CPU forward/backward
timings make this implementation roughly 12% slower. Selecting rows and
dispatching smaller maps outweighs its contraction reduction here. The
default remains `full`; this is an equivalent credit factorization, not a
measured efficiency gain. An implementation/hardware claim needs complete
timing or energy over the declared workload, not just the contraction bound.

## 191. A surrogate teacher must be checked against realized hard computation

Inside a fixed differentiable region of the hard model, let $G$ be its exact
loss gradient and $C$ the added counterfactual estimator term. The update uses
$S=G+C$. Under positive-definite optimizer map $M$, the actual first-order
change from $\delta=-\eta MS$ is

$$R(\theta+\delta)-R(\theta)
 =-\eta G^TMS+O(\eta^2\|MS\|^2).\tag{191.1}$$

Thus $S\ne0$, a large norm or apparent surrogate descent does not establish
descent of the hard loss. The relevant sign is $G^TMS$. A sufficient condition
is $\|M^{1/2}C\|<\|M^{1/2}G\|$, by Cauchy–Schwarz; this condition is not
necessary. At a route boundary, a finite realized change and its downstream
loss require a separate paired comparison. The surrogate estimates access to
that other computation rather than an interior derivative.

For example, a last-layer route score may influence only its outgoing delay
within one fixed winner region. A terminal mean readout consumes the vector,
not that delay. Its interior clock teacher is then zero, though the
counterfactual vector comparison can give a nonzero route teacher. This is
useful boundary credit, but cannot reduce terminal loss until it changes the
realized choice. Counting that gradient as an ordinary infinitesimal descent
direction would conflate two different mechanisms.

For class $c$ and fitting-speaker group $j$, use exact conditional gradients
$G_{c,j}$ and record the surrogate separately as $S_{c,j}$. A direction learned
on group A has first-order transfer to group B governed by
$G_{c,B}^TM S_{c,A}$. Same-group and cross-group Gram matrices therefore answer
different questions. Class-balanced fitting samples isolate direction conflict
from unequal class frequencies; disjoint fitting speakers test internal
transfer without allowing held-out labels to choose updates.

E128 measures these matrices at the frozen 72.3% parent with two zero context
maps. Old parameters are fixed. It verifies pathwise and surrogate forward
losses agree, then applies declared small parameter radii and records actual
class losses, winner changes and per-query arrival-order changes. Finite
changes are diagnostic fitting perturbations, not new benchmark accuracy.
The original validation and official test remain unavailable to the update.

E128's 240 fitting examples give average cross-speaker-group cosine -0.1183
for exact credit and -0.1228 for the surrogate. The aggregate gradient norm is
about 10% of the average per-example gradient norm. Ordinary balanced descent
would increase eight of the 20 sampled class losses to first order. However,
the exact common-descent value across those classes lies between 0.08692 and
0.08994 per unit new-parameter radius. This is a positive attainable-direction
certificate at this checkpoint, despite the interference in ordinary averaging.
Exact/surrogate class-gradient cosines range from 0.9928 to 0.9992. Thus the
added context's counterfactual estimator is largely aligned with its pathwise
teacher here; its nonzero credit is not the same as group-balanced transfer.
These are small fitting-sample statistics, not population or unseen-speaker
guarantees.

Finite balanced perturbations of radius 0.001 and 0.01 lower aggregate fitting
loss. At radius 0.001 they nevertheless change 36 pathwise or 48 surrogate
winners and 114 or 132 per-query/layer arrival orders. A fixed-topology gradient
certificate alone is therefore insufficient for an actual finite step. The
next teacher checks the resulting hard computation after every proposed update.

## 192. A feasible useful direction is more informative than route count

For exact class gradients $G_c$, the common-descent value at unit Euclidean
radius is $t^*=\min_{\lambda\in\Delta}\|\sum_c\lambda_cG_c\|$ (§187).
For any feasible convex mixture $v=\sum_c\lambda_cG_c$, finite-iterate bounds
are

$$\max\!\left(0,\min_c\frac{G_c^Tv}{\|v\|}\right)
 \le t^*\le\|v\|.\tag{192.1}$$

The lower bound is the measured common improvement of feasible direction
$-v/\|v\|$; the upper bound is that mixture's feasible value in the dual
minimum. Zero direction supplies the lower bound zero. For $v=0$, the upper
bound is zero and no strict common descent exists. A positive lower bound
certifies a common local direction without assuming the numerical solver
reached its optimum. A positive upper bound alone does not certify one.

Optionality mechanisms can now be judged by whether they add conditional
information and attainable directions that enlarge a useful feasible region.
They may permit a later representation or boundary move even when immediate
loss does not decrease. Their reserve remains a horizon-conditioned quantity;
it is not ordinary gradient norm, entropy or the immediate improvement
certificate above. Use these local certificates to diagnose available direct
credit, then use paired continuation adaptation to measure the additional
future reserve. Repeating correlated counterfactuals cannot resolve a missing
communication path or a conflicting teacher by itself.

## 193. Preserve useful counterfactual credit inside a common descent cone

Let $G_j$ denote exact conditional gradients for fitting-defined class/speaker
conditions and $S_j$ their surrogate counterparts. Normalize by
$n_j=\|G_j\|$ when nonzero, and let $\lambda$ be a feasible simplex mixture
whose direction $d_p=-\sum_j\lambda_j G_j/n_j$ has $a_j=G_j^Td_p<0$
for every condition. Define the separate added-counterfactual direction
$d_c=-\sum_j\lambda_j(S_j-G_j)/n_j$, with $b_j=G_j^Td_c$. Then

$$\rho=\min\!\left(1,\min_{j:b_j>0}\frac{-a_j}{2b_j}\right),
 \qquad d=d_p+\rho d_c\tag{193.1}$$

retains $G_j^Td\le a_j/2<0$ for every condition. For $b_j\le0$ any positive
strength helps or leaves the constraint unchanged. With no positive $b_j$,
use $\rho=1$. Rescaling $d$ by a common positive factor preserves all signs.
This retains the complete counterfactual comparisons while bounding their
interference with available exact descent. It changes credit allocation,
not inference: the layer still emits one winning vector and one delay.

The guarantee is first order and conditional on a differentiable hard region.
E129 therefore proposes a finite parameter-radius step, recomputes the hard
fitting losses, and halves the radius until the declared conditional constraints
hold or its finite search budget is exhausted. Old parameters remain fixed,
but predictions are checked because that alone does not preserve function.
Its fitting-only two-speaker-group/class conditions define the teacher; held-out
loss does not select a radius or stopping point. Conditions with zero gradient
require explicit tolerances rather than an invented strict-descent certificate.

This tests a useful local subspace when available. It is not a global claim
that every class must improve at every step. A nonconvex representation change
may require temporary loss increase. Horizon-conditioned optionality should
then be judged by paired downstream adaptation and later attainable utility,
with its shadow work charged. Forcing that reserve into immediate-descent
constraints would remove the future opportunity it is meant to measure.
