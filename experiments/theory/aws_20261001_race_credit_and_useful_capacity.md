# A verified route-credit limit and two resource-conscious ways forward

Read alongside §§19, 313–325 and 354–357. This is an AWS mathematical audit,
not a change to any frozen model, a fitted comparison or a supremacy claim.

## What failure is addressed

The current `TemporalRoute.backward` preserves hard forward selection, pays for
losing value proposals and conserves its local score teacher. None of those
properties proves that the teacher follows the expected nonlinear loss.
The existing fixed-time H2 audit reported mean local cosine .717 and one opposed
direction; it did not cover a full sequence/time expectation. The new audit
constructs an explicit strictly convex counterexample. Conversely, it verifies
that the existing combined value/time teacher is exact in a useful linear case.
The implication is to measure nonlinear credit fidelity, not remove races,
persistent state, independent heads or useful content paths.

## Exact joint winner/time identity

Let λ_i=exp(s_i), Λ=Σ_i λ_i and p_i=λ_i/Λ. Independent exponential clocks
have joint winner/minimum density

    P(W=i,T∈dt)=λ_i exp(-Λt)dt.

Therefore W~Categorical(p), T~Exp(Λ), and W and T are independent. An equivalent
parameterization is τ~Exp(1), T=τ/Λ. This is a distributional identity; it
does not preserve a particular realization of the old per-emitter noise.

Consider branch costs F_i(t) for a single race followed by a smooth suffix, or
the differentiable conditional expected suffix cost with an exact derivative.
Assume integrable costs/derivatives and differentiation under the expectation.
Other direct parameter dependence is held fixed in this local score derivative.
Then

    J(s)=E_τ[Σ_i p_i F_i(τ/Λ)],
    ∂J/∂s_i=E_τ[p_i(F_i-Σ_j p_j F_j)-p_i T Σ_j p_j ∂_T F_j].             (1)

The first term allocates route probability. Its sum is zero. The second changes
the aggregate clock rate. Its sum is -T times the expected time derivative;
common-mode urgency is real learning, not a conservation violation.
For the implemented bounded delay, F'_i(T) includes
D'(T)=.010/(1+T)^2, where D=.001+.010T/(1+T).

The equivalent likelihood-ratio form is

    ∂J/∂s_i=E[(1{W=i}-λ_i T) F_W(T)].                                  (2)

It follows directly by differentiating the joint density. Integration by parts
connects (1) and (2), when the boundary terms vanish. Equation (1) integrates
the winner out; (2) does not require a time derivative but may have large loss
variance. Baselines must respect conditioning and any required correction.

These identities do NOT license ignoring later route boundaries. If a sampled
suffix changes hard routes as t changes, pathwise interior derivatives alone
need not equal the derivative of its expected cost. One must marginalize that
suffix, add its stochastic/boundary credit, or label the resulting reference
as local/conditional. The implemented audit has one race and smooth costs.

Known primitives are likelihood-ratio/pathwise estimation and conditional
Monte Carlo, not a new estimator family. See
[Schulman et al., stochastic computation graphs](https://arxiv.org/abs/1506.05254)
and [Liu et al., Rao–Blackwellized discrete gradients](https://proceedings.mlr.press/v97/liu19c.html).
The contribution here is the joint exponential-race specialization, comparison
with this repository's teacher, numerical contracts and proposed resource tests.

## A strictly convex opposed-direction witness

Take two equal-rate routes with scalar values v_0=0, v_1=2. The loss is the
Poisson negative log likelihood for observed count4,

    F(v)=exp(v)-4v+log(4!).

It is strictly convex. With no time-dependent cost, the true score gradient is

    p_0 p_1 [F(v_0)-F(v_1)] (1,-1)
      =(.4027359753,-.4027359753).

Enumerate both possible winners and integrate the current teacher's λ_i T
weight, using E[T]=1/Λ. For any scalar two-route smooth loss, its expected
score0 component simplifies to

    p_0 p_1 (v_0-v_1) [F'(v_0)+F'(v_1)]/2.                            (3)

Here it is -.0972640247, giving the opposite vector. The dot product with the
true gradient is -.0783434437. A sufficiently small update in the NEGATIVE
expected teacher direction increases the exact objective locally.
This does not prove a stochastic optimizer always worsens this loss, nor that
an active full model exhibits the witness. It disproves a universal nonlinear
gradient-fidelity guarantee for the current local operator.

Equation (3) is an endpoint trapezoidal approximation to the secant loss.
For scalar F∈C³ on the segment, the absolute score error is bounded by

    p_0 p_1 |v_0-v_1|³ sup|F'''| / 12.                               (4)

It is exact for quadratic scalar losses, but convexity alone does not fix its
sign. Curvature variation and a small true route gap can overturn descent.
In vector/state suffixes, this scalar witness does not provide a complete bound;
route replacement also changes persistent writes, addressed memory and later
events, which delivered-value interpolation alone does not capture.

## Why the clock interior is not by itself a bug

Existing timing credit assigns -T F'_W(T) to the sampled winner. That expression
alone differs from the aggregate-rate term in (1) when branch time sensitivities
differ. However, the existing λ_i T value teacher is not the categorical term
in (1) either. Their biases can cancel.

For F_i(T)=v_i D(T), the combined existing teacher is EXACT in expectation.
The relation

    Λ E[T D(T)] = E[D(T)] + E[T D'(T)]

follows by integrating the derivative of T D(T) against exp(-ΛT). Substitution
gives (1). The audit verifies equality at unequal rates to2.7e-16. Do not rewrite
the clock gradient on the basis of inspecting one component in isolation, or
combine the new rate term with the old boundary term without deriving the joint
estimator. Preserve original statements that the old winner derivative is exact
inside a realized smooth region; expected nonlinear route credit is separate.

## Conditional enumeration and bounded residual replay

At fixed T, the hybrid reference estimator

    G(W)=(e_W-p)F_W(T)-p T F'_W(T)

has conditional mean (1). Replacing it with that mean removes the positive
semidefinite conditional covariance Cov(G|T). The law of total covariance gives
the usual variance reduction against THIS estimator. It does not show lower
variance than every alternative, or superior wall time per useful update.

Enumerating all candidate losses requires actual branch-specific persistent
writes and downstream outcomes, not just reading all losing values. Dense
suffix branching over many races is prohibitive. Two candidates at one chosen
node need two local conditional continuations, not2^N full histories if the
rest of the suffix remains sampled; then its marginal/boundary scope matters.

A potential bounded-budget construction uses cheap branch surrogates A_i(T),
residuals R_i=F_i-A_i, and one alternative I~q with full support. Add

    (p_I/q_I)[(e_I-p)R_I-p T R'_I]

to the fully enumerated cheap-surrogate version of (1). Its conditional mean
is the exact correction when residual costs AND derivatives are exact. Stop
gradients through proposal/importance bookkeeping in the local credit update;
otherwise an unrelated derivative of q is introduced. Candidate selection,
suffix replay, state copying and derivative work are charged.

For trace variance, the optimal proposal under known residual vectors is
q_i∝p_i|| (e_i-p)R_i-p T R'_i ||. Those vectors are unknown before evaluation;
an approximate proposal with q_i≥ε p_i bounds p_i/q_i≤1/ε and preserves support.
It trades variance and replay budget rather than proving useful convergence.
If only approximate suffix derivatives are available, only the corresponding
conditional boundary correction is exact. Do not claim full-sequence unbiased
learning. Clockless native execution remains a later implementation question.

## Occupied state and statistical exposure

The source64 ladder increases private maps while dividing128 queries among64
addresses. Each private transition sees only2 queries/pass; at S4 it sees32.
Four passes produce8 versus128 query exposures per address. This is capacity
with reduced supervision, not a pure inference-activity scaling experiment.
The shared-rule/private-state battery directly tests this confound while paying
all map/optimizer work. It also removes private source embeddings, so map-only
attribution still requires the declared embedding-only control.

There is a legitimate state-capacity/resource target. Suppose S independent
streams each contain a uniformly random M-bit record X_s that must be recalled
after a gap. Let B be the information in the persistent state, with labels not
leaked through parameters, addresses, queues or clocks. For per-address error
probability e_s and a decoder seeing that state plus the queried address, Fano
and independence imply

    B ≥ Σ_s [M-h₂(e_s)-e_s log₂(2^M-1)].                              (5)

For exact recall B≥SM. The proof uses Σ_s I(X_s;state)≤I(X_1..X_S;state)≤B.
All retained information-bearing channels must be counted. Shared rules can
keep transition/optimizer parameter work independent of S while addressed state
scales with S; selected per-event work can stay fixed under supplied addressing.
This is also possible in conventional addressed recurrent systems. It motivates
a fair necessary-memory test, not unique architectural supremacy.

The current order label holds only two bits/source; a large float state can far
exceed this lower bound. Occupancy counters alone cannot establish efficient
useful capacity. A future independent-content recall task should vary required
information and query delay, charge discovery if addresses are learned, and
include an addressed shared-weight recurrent control. Keep it separate from the
currently prioritized protected/shared and paired-timing battery.

## Completed audit and next decision

`experiments/race_gradient_reference.py` and its saved diagnostic JSON reproduce
the polynomial finite-difference agreement (~1.8e-12), convex opposed-gradient
witness, linear joint-clock identity and conditional covariance check. Three
read-only tests pass. No model training or report-quality claim follows.

Next after the frozen battery: replay a bounded set of saved-model route choices
with actual state replacement and fixed downstream randomness. Measure true
conditional suffix loss differences versus current credit, direction/sign error,
time dependence and total replay cost. Verify that the selected-node diagnostic
does not alter checkpoint weights or RNG. Predeclare node sampling and include
early/deep nodes, useful/unused alternatives and long-gap cases.

Only if that audit identifies actionable error should a separately named
integrated learning intervention change the teacher. It must retain temporal
computation, sparse commits, independent heads, persistent addressed state and
counterfactual alternatives; pass gradient/recovery/accounting contracts and a
small matched fit before promotion. Compare quality per total fitting work,
not route movement or an uncharged teacher. Strong addressed-recurrent/time-aware
controls and independent paired seeds remain necessary for supremacy claims.


## Completed frozen checkpoint replay — 1 October, 23:35 UTC

`aws_frozen_native_route_audit_20261001T233000Z.json` replays the selected
completed native order checkpoint. Twelve predeclared head0 choices cover
address0's four events and blocks0/4/7 in one development population, using
continuation seed619. Alternatives replace the candidate's persistent write
as well as its value. Same sampled delay, downstream RNG consumption, total
race/commit counts and checkpoint parameter hash are verified.

Two of12 local score teachers oppose the scalar full-state boundary reference.
At event0/block4, alternative losses are3.572759628 and3.588039398; teacher
score0 is+5.2289e-6 while reference score0 is-7.1532e-5. At event7/block7,
losses are3.588039398 and3.620435715; teacher score0 is+1.7415e-4 versus
reference-4.6471e-3. Twelve nonzero cosines average2/3; with two routes these
are sign comparisons, not rich high-dimensional angle evidence.

Boundary reference uses λ_i T(F_i-mean_j F_j), followed by the same sampled-
winner conservation correction. Current teacher uses endpoint value derivatives
instead of actual branch-state losses. Winner clock interiors are excluded
from BOTH comparisons. This is a fixed-noise/time, one-population diagnosis,
not the full expected gradient or an estimated population error rate. It does
not distinguish value curvature from missing alternative-memory effects yet.

Audit cost:36 full forward replays,12 backwards,9,216 races;6.319s wall and
442,400KiB peakRSS. Arithmetic is uninstrumented, not free or an energy claim.
The checkpoint and outerTorch RNG were preserved; no weights were fitted.

This supplies an actionable conditional fidelity error in a trained integrated
model. Next isolate persistent-write versus delivered-value effects, then test
a separately named bounded credit correction. All replay/state/optimizer work
and any cache/RNG handling must be charged. The prioritized protected/shared
battery continues; neither these probes nor its pending scores imply supremacy.


## Addressed writes create an independent credit channel

There is a failure even without nonlinear delivered-value curvature. Take
two equal-rate routes with identical delivered values v0=v1=0 and identical
proposed memory content m0'=m1'=1. Old addressed state is(S0,S1)=(0,0).
Route0 writes only address0; route1 writes only address1. Let later loss be
F=S1, independent of arrival time. Branch losses are0 and1, so exact local
score credit is(-.25,+.25). The value-only teacher is zero.

The new deterministic addressed-write audit checks finite differences to
6.7e-12; two tests pass. Concatenating the proposed memory CONTENT with the
message does not repair this witness: both concatenated vectors still coincide.
The location of the persistent update changes future information paths. This
is a local structural blind spot, not proof of zero learning in a whole model:
other events and shared parameters can receive useful gradients.

For a linear future conditional loss

    F(v,S)=a·v+Σ_b c_b·S_b,

routei's utility, up to a common constant, is

    U_i=a·v_i+c_address(i)·(m_i'-m_old,address(i)).

The SECOND term uses the adjoint at the actual candidate write address and
its delta relative to old memory. It is exact in this linear setting. The
existing teacher sees the first term; ordinary producer backprop can train
the selected memory map, but integer selection of WHICH persistent slot
to commit does not expose the losing-address delta to its local value teacher.
In consume_event the selected memory proposal is assigned outside TemporalRoute,
so that local autograd operator cannot directly receive its later memory adjoint.

For a smooth conditional suffix with state/value perturbationδ_i and Hessian
norm bounded by M, the joint Taylor error is at most M||δ_i||²/2. This holds
only within a differentiable continuation, not across ignored downstream
hard-route changes. Actual scalar suffix replay is needed to audit that scope.
A state-aware local surrogate can therefore reduce a missing information path
without claiming exact arbitrary expected gradients.

If future addressed adjoints are available, evaluating the K state-dot-deltas
costs O(Kd) scalar/vector work plus candidate proposals already charged. That
is a conditional complexity claim: generating, storing, transporting and
versioning those adjoints is NOT free, and losing addresses may not be touched
by the realized suffix at all. Dense reverse-state scans or full suffix replay
would undermine the resource argument. Learned local route-value critics are
an alternative hypothesis whose approximation, exposure and fitting work must
be measured. The frozen checkpoint audit currently conflates this missing
state channel with value curvature; its next control must separate them.

A justified intervention would expose candidate addressed-write effects to
the teacher while retaining temporal races, independent heads, persistent
state, sparse commits and losing alternatives. It must keep winner-only
forward execution and pay the extra learning information paths. First compare
value-only, value-plus-addressed-state and bounded scalar-replay credit on a
small fully audited integrated task. Preserve the existing teacher as a matched
parent and prove zero-added-credit nesting before fitting. Do not modify the
currently running protected/shared battery.
