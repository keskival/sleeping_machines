# Supervised observability, compact queries and directional optionality

[Theory index](../THEORY.md) · Global sections 210–213.

The E136 transport theorem removes one source of depth-dependent numerical
damage. Its completed classifier also exposes a different bottleneck: a large
terminal decoder can fit without requiring substantial hidden adaptation.
This note separates transport, supervised observability, parameter control
and future optionality. These are local conditional results, not convergence
or statistical generalization proofs.

## 210. The class-visible control kernel

Let the completed event program have augmented boundary state $h$, class
logits $z=f(h)$, readout Jacobian $P=\partial z/\partial h$, and supported
parameter/control Jacobian $J=\partial h/\partial\theta$. All derivatives
condition on the actual hard race program unless its boundary contribution
is explicitly included. Let $D\succ0$ declare the control metric or cost.
For class loss teacher $g=\partial L/\partial z$, a small preconditioned step
has

\[
 \delta\theta=-\eta D^{-1}J^TP^Tg,\qquad
 \delta L=-\eta g^TKg+O(\eta^2),\quad
 K=PJD^{-1}J^TP^T. \tag{210.1}
\]

This is the **class-visible control kernel**. The raw reachable Gramian is
$G=JD^{-1}J^T$, but the readout sees only $PGP^T$. An orthogonal suffix
preserves the norms of raw control tangents while changing their alignment
with $P$. Norm-preserving transport therefore supplies no positive lower
bound for $g^TKg$ on its own. Principal angles between reachable directions
and returning teacher directions determine whether that transport is useful.
Tied parameters sum event tangents before forming $G$: treating each event
as an independent tunable control would overestimate reachable corrections.

Softmax teachers satisfy $\mathbf1^Tg=0$. Put
$H=I-\mathbf1\mathbf1^T/C$ and analyze $HKH$ on the class-contrast space,
rather than counting the irrelevant uniform-logit direction. A lower bound
requires a positive minimum eigenvalue on the actual teacher support, plus
a finite-step region in which the linearization remains valid. Clipping,+optimizer state and hard race changes modify that region; gradient presence
alone is insufficient.

For a fitted linear head with feature rows $\Phi_i$, the feature kernel
$\Phi\Phi^T$ can have full fitting-sample rank even when all hidden parameters
are fixed. Fitting progress then does not establish representation learning.
In E136, resetting the learned angles preserves 1,024/1,024 fitting decisions
with the trained decoder. Held decisions change from 296/512 to 278/512.
This establishes some angle contribution/coadaptation at that checkpoint,
while identifying the decoder's ability to retain fitting decisions. It is
not a retrained fixed-angle comparison or proof of which training mechanism
produced the original fit.

## 211. Factor a memory query without hiding its teacher

Index retained states by bank $b$ (including depth/receiver/option) and payload
coordinate $d$. Let $\bar s_{bd}$ denote the fixed fit-only calibrated state.
Use a rank-$r$ bank/channel/class query:

\[
 u_a=\sum_{b,d} A_{ab}V_{ad}\bar s_{bd},\qquad
 z_c=z_c^{\rm packet}+\sum_a U_{ca}u_a. \tag{211.1}
\]

Its represented head tensor is
$W_{cbd}=\sum_a U_{ca}A_{ab}V_{ad}$. It uses $r(C+B+d)$ state-head
parameters instead of $CBd$. Its class-visible state teacher is exactly

\[
 \frac{\partial L}{\partial s_{bd}}=
 \frac{1}{\sigma_{bd}}\sum_a (U^Tg)_a A_{ab}V_{ad}. \tag{211.2}
\]

The scale $\sigma$ belongs to the supervised boundary and must appear in
gain/gradient accounting. The query is low rank in its calibrated coordinates;
it does not preserve every distinction held by the full memory. Rank $r$ is
an explicit inductive constraint, not free compression of an arbitrary head.
The terminal state-logit contrast rank is at most $\min(r,C-1)$. The separate
packet head can add class-visible directions. Small rank does not by itself
preclude separating many classes, but useful spatial/channel separability
must be learned and measured.

Contract the payload axis first. The forward state query requires
$Bdr+Br+Cr$ MACs, plus its packet query, and reads $Bd$ state scalars.
At $B=216,d=32,C=20,r=16$, state-head parameters fall from 138,240 to 4,288.
Including the packet head gives 4,968 decoder parameters and 115,028 query
MACs; the old full decoder uses 138,900 MACs. The twelve-layer model has
6,476 trainable parameters when angles learn and 5,288 when they are frozen.
This saves parameter capacity substantially but changes total event-memory
work very little: the inherited key program and exchanges still run.
Physical traffic, backward work and joules require separate measurement.

There is no event-pair attention matrix. Each winning packet still addresses
one state bank and emits one packet. The query is performed at a declared
completed utterance, without assigning an early class label to incomplete
evidence. Early confident decisions need their own timing/censoring objective.
Reading all layer states creates direct supervision paths; this experiment
does not establish that a deepest-only serial composition would learn equally
well. The E137 matched intervention updates versus freezes exchange angles
with identical query capacity, initialization, sample order and fitting budget.

## 212. Separate current error, uncertainty and optionality

The expectation in §209 requires a **second moment**, not just a centered
covariance. For future class teacher $g_f$ with mean $\mu$ and covariance $C_f$,

\[
 \mathbb E[g_f^TKg_f]=\mu^TK\mu+\operatorname{tr}(K C_f). \tag{212.1}
\]

The first term measures correction aligned with the mean teacher. The second
measures capacity across uncertain future teacher directions. Conflating them
turns optionality into ordinary current-error credit. Even the covariance
term is useful optionality only for a declared future adaptation problem:
uncertainty in irreducible label noise is not an exploitable route reserve.

For $g=p-e_y$ and a conditional label distribution $q$,

\[
 \mu=p-q,\qquad C_f=\operatorname{diag}(q)-qq^T. \tag{212.2}
\]

A wrong confident prediction can have a large mean teacher and zero label
covariance for a deterministic label. Predictive entropy, inverse observed
likelihood and epistemic uncertainty are therefore different quantities. With
uncertain $q$, the law of total covariance separates expected conditional
label covariance from covariance of conditional predictions. Their sum alone
does not identify epistemic uncertainty. A future-value critic, an explicitly
modeled posterior, or measured adaptation outcomes must define what can be
improved. Error can trigger exploration without serving as its optionality
score.

Exclusive categorical route alternatives also are not independent controls.
For terminal alternative states $h_a$, probabilities $p_a$ and mean
$\bar h=\sum_a p_a h_a$, their local variation covariance is

\[
 G_{\rm route}=\sum_a p_a(h_a-\bar h)(h_a-\bar h)^T. \tag{212.3}
\]

Duplicating an identical alternative and splitting its probability leaves
this covariance unchanged. Merely counting choices can consequently inflate
optionality without adding any reachable direction. This covariance describes
a declared stochastic alternative model; actual hard races still emit one
winner, and an alternative's $h_a$ must include its state/timing suffix.

For a new independent small control tangent $j$ with weight $w$, a diversity
statistic has the exact incremental form

\[
 \log\det(\lambda I+G+wjj^T)-\log\det(\lambda I+G)
 =\log(1+w j^T(\lambda I+G)^{-1}j). \tag{212.4}
\]

It discounts redundant directions. It is a reachable-volume statistic, not
classification utility, and should be projected into a relevant teacher
metric if used to propose alternatives. No global route tree is needed to
maintain a local low-rank Gramian sketch, but the sketch, propagation and
proposal/suffix work must be counted. For a nonlinear suffix, transporting
the sketch is a local derivative approximation, not an exact finite-route
counterfactual.

## 213. Why separate scalar reserves lose correlations

For packet and state perturbations with cross-covariance $C_{sv}$, a local
exchange $s'=cs+bv$, $v'=-bs+cv$ transforms state covariance as

\[
 C_{s'}=c^2C_s+b^2C_v+cb(C_{sv}+C_{sv}^T). \tag{213.1}
\]

The packet covariance has the opposite cross term. Total augmented variance
is preserved, but its location and class-visible usefulness depend on those
correlations. In one dimension, let $\operatorname{var}(s)=
\operatorname{var}(v)=1$. At a 45-degree exchange, perfectly correlated inputs
have output variances $(2,0)$; perfectly anticorrelated inputs have $(0,2)$.
Both begin with identical individual scalar reserves $(1,1)$ and total 2.
Independent scalar propagation cannot distinguish the two, although a query
reading only the packet can.

This rules out a universal exact propagation rule based solely on independent
packet/state variance scalars. It does not rule out a state-conditioned scalar
critic: such a critic can encode future value from its context and be trained
on counterfactual adaptation outcomes. Directional sketches are one explicit
alternative. A sketch $Q$ with $C\approx QQ^T$ propagates as $Q'=UQ$ through
the conditional exchange, preserving its shared columns and correlations.
Independent sketch columns at each node would lose that information too.
The next policy-learning mechanism must declare what reserve it predicts,+which correlations/context it retains, and how realized future gains teach
that prediction. E137 tests the compact supervised query; it does not yet
train an optionality critic.
