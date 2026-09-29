# Optionality: attainable futures, delayed choice, and transferable learning

29 September 2026. Extends §§149–153. These are analytic derivations and a
proposed measurement contract; no SHD gain is claimed from the new metric.

## 159. Preserve the set of futures before choosing its scalar value

A route changes more than the present prediction. It changes the states,
observations, parameter updates, and later decisions that can still be
reached. Define the budgeted attainable set

\[
 \mathcal A_b(s)=\{(\text{terminal prediction},\text{updated learner state},
                  \text{work},\text{latency}):
                  \text{causal feasible continuation from }s\text{ within }b\}.
\]

The state includes payloads, clocks, receiver states, pending/cancelled events,
candidate topology, and the learner's parameters and optimizer state when
virtual learning is allowed. The budget specifies event work, search/replay,
updates, and horizon. Future labels may enter supervised training credit;
they cannot enter an inference route before they are observed.

**Optionality is a property of this attainable set and the information
available when its choices are made.** Immediate loss evaluates the present
member of the set. Route entropy describes a distribution over route names.
Neither measures how many distinct, useful futures remain attainable.

For a fixed future utility u, a scalar continuation value is
\(V_b(s;u)=\sup_{a\in\mathcal A_b(s)}u(a)\), with expected utility replacing
u for stochastic, causal policies. This scalar is sufficient to compare
states under that u and b. It is not sufficient for every possible future
utility: two sets can have crossing preferences. For example, attainable
logit displacements {±e₁} and {±e₂} favor different future correction
directions. No utility-independent scalar preserves both orderings.

For local logit displacements, the support function
\(h_s(d)=\sup_{\delta\in\mathcal A_b^{\rm local}(s)}d^\top\delta\)
retains this directional information. A small set of projections h_s(d_j)
can approximate it without retaining the whole tree. Include zero displacement
as a no-op. Duplicating an identical route leaves this function unchanged.
Averaging h over a declared distribution of future directions is a valid
scalar summary; counting routes is not a substitute. Finite projections
can miss other directions and do not certify global reachability.

**Monotonicity has conditions.** Enlarging an attainable set cannot lower
its optimal value only when old actions, observations, and their costs remain
available. Extra parallel active routes consume work and may inhibit or
perturb old routes, so they do not automatically satisfy set inclusion.

## 160. The value of choosing after information arrives

Let ξ be future evidence, with conditional law μ(ξ|s), and let Q(a,ξ) be
the net utility of a later route a under that evidence. Assume the same
action set is available for the following two decisions. If ξ is observed
before choosing a,

\[
 V_{\rm adaptive}=\mathbb E_\xi\max_a Q(a,\xi),\qquad
 V_{\rm commit}=\max_a\mathbb E_\xi Q(a,\xi),\qquad
 \Omega_{\rm information}=V_{\rm adaptive}-V_{\rm commit}\ge0.
\]

Proof: for each fixed a, max Q≥Q(a,ξ); take expectations and then the maximum
over a. This is the decision-theoretic value of retaining a choice until
information arrives. A branch can have no current loss advantage and still
be worth taking because it preserves this flexibility. It can also be worth
an immediate cost c when its total continuation advantage exceeds c.

If the network observes only o(ξ), use
\(\mathbb E_o\max_a\mathbb E[Q(a,\xi)\mid o]\).
Maximizing separately for hidden ξ is an oracle error: it credits information
the network never receives. With no informative observation, the adaptive
value equals the commitment value. A high-variance outcome alone creates no
option value. Two routes with utilities (1,−1) and (−1,1) across two equally
likely, distinguishable future observations have value 1 versus commitment
value 0. One route with the same variance has option value zero. Any number
of duplicate routes also has zero additional value.

### What can propagate as one number?

For a fixed utility, causal policy class, belief state, and remaining budget,
ordinary dynamic programming returns a scalar adaptive value from each child:

\[
 V_b(s)=\max\{U_{\rm stop}(s),\sup_{a:c(a)\le b}
 [-c_u(a)+\mathbb E V_{b-c(a)}(F(s,a,\xi))]\}.
\]

Here c_u is utility-denominated cost, c is resource consumption, and terminal
classification reward is counted once. Zero-cost cycles must be excluded or
handled by a finite horizon. The state/belief must contain the information
needed for this Markov recursion. Observations become available in F; the
current action cannot condition on them beforehand.

An isolated option premium generally cannot obey the same scalar backup.
At a root choosing A or B, suppose child (adaptive, restricted) values are
A=(10,10), B=(9,0). Their premiums are 0 and 9. The root's premium is
max(10,9)−max(10,0)=0, not max(0,9)=9. The subtraction baseline matters.
Pass adaptive value alone if that is the objective; retain an appropriate
restricted value or policy state as well when reporting a premium. Over many
stages, “commitment” must mean a specified observation-blind policy class or
noncontingent plan. Precommitting to an arbitrary observation-dependent policy
is already adaptive and gives no such gap. A general open-loop comparator
may require extra belief/plan state; two local scalars need not suffice.

This refines §149's one-number claim: a **fixed continuation objective** can
be propagated as one value. It does not justify propagating an arbitrary
optionality bonus independently of its objective, baseline, and information.
The potential-difference telescoping warning in §152 still applies.

## 161. Same-sample virtual progress rewards gradient noise

The earlier one-step learning reserve adapted and evaluated on the same
example. That answers whether this example can be fitted after a virtual
update. It can overvalue routes that expose large, inconsistent gradients.
This is a separate defect in the measurement objective from proposal collapse.

Let a be a reusable route policy/intervention, not just an event ID that has
no counterpart in other examples. Let A and B be independent training samples
or small batches from the intended conditional task distribution. Apply the
same policy definition to both; update only its declared reachable suffix.
Write g_A=∇L_A(θ;a), g_B=∇L_B(θ;a), and hold a positive-semidefinite update
metric M fixed for this comparison. The first-order virtual gains are

\[
 P_{AA}=\eta g_A^\top M g_A+O(\eta^2),\qquad
 P_{AB}=L_B(\theta;a)-L_B(\theta-\eta Mg_A;a)
       =\eta g_B^\top M g_A+O(\eta^2).
\]

For identically distributed, independent A and B, let m=E g and Σ=Cov(g).
Then the first-order terms satisfy exactly

\[
 \mathbb E[g_A^\top M g_A]=m^\top Mm+\operatorname{tr}(M\Sigma),\qquad
 \mathbb E[g_B^\top M g_A]=m^\top Mm.
\]

**Consequence:** a route can increase the measured same-sample learning
reserve entirely by increasing gradient noise, with no increase in expected
transfer. This gives a specific possible explanation for activity/gain bonuses
that move dynamics while failing to learn. It is not yet a measured explanation
of the E83 optionality pilots. With different task distributions, the transfer
term is m_BᵀMm_A and may be negative: this also measures interference.

If L_B is H-smooth along the virtual update,

\[
 P_{AB}\ge\eta g_B^\top M g_A-
             \tfrac12H\eta^2\|Mg_A\|^2.
\]

Thus transferable first-order signal and quadratic update cost can be measured
separately. Large raw gradients can lose after the curvature term. For Adam,
clipping, or state-dependent M, replay the actual optimizer state and measure
the finite transfer loss; the factorization above requires fixed M and
conditional independence. Positive same-sample gain alone is insufficient.

The new diagnostic contract is therefore **adapt on A, evaluate on B**,
comparing factual and counterfactual policies on matched draws and with the
same step/work budgets. B comes from an internal training split; repeated
development choices must not consume the final speaker test. If an action is
truly sample-specific, the measured reserve remains within-example adaptation;
call it that and do not interpret it as representation generalization.

Optionality is still the reserve over later feasible choices. This transfer
measurement improves its terminal learning utility; it does not replace the
set of choices with an ordinary immediate-gradient score. A counterfactual
that initially harms loss can remain valuable if it enables later, feasible
updates that transfer. Compare complete continuations at equal budget.

## 162. Useful breadth, discovery cost, and the next experiment

For a fixed state and proposal q, let X≥0 be the gain of a feasible continuation
relative to a retained no-op, after any specified virtual update and independent
evaluation. If K gains are iid from that same proposal and 0≤X≤U,

\[
 \mathbb E\max(X_1,\ldots,X_K)
 =\int_0^U[1-F_X(t)^K]\,dt.
\]

This follows by integrating the tail probability of the maximum. It extends
§153's hit probability into an entire attainable-gain curve versus search
budget. The marginal value of one more draw is
\(\int_0^UF_X(t)^K[1-F_X(t)]dt\), which decreases with K. Stop searching when
the estimated marginal value is below its replay cost. Fixed proposals,
independence, and bounded gains are assumptions; adaptive proposals and
correlated descendants require a different conditional calculation.

Route count can grow while this curve stays flat. Duplicate actions,
near-identical descendant states, inaccessible useful alternatives, and
unobservable future evidence all cause this. The curve depends on proposal
mass, not just the number of named alternatives. Changing q is a policy change;
record propensities and budget. Best-of-K evaluated with noisy loss estimates
also has selection bias: validate a selected continuation on fresh internal
training data or account for that noise before crediting its reserve.

**A discriminating experiment after the event-kernel correction:**

1. Freeze a working representation and use a small, explicit family of reusable
   route interventions with matched state, search, and update budgets.
2. Measure same-example progress, cross-example transfer, and the finite-K
   attainable-gain curve separately. Include duplicates and uninformative
   future observations as negative controls.
3. Compare a present-loss policy, the existing same-sample reserve, and the
   transfer-based continuation reserve. Keep inference topology and total
   replay/update work fixed; compare both fit-set learning and speaker holdout.
4. Advance the mechanism only when the reserve predicts actual subsequent
   improvement on fresh examples and the budgeted accuracy/work frontier moves.

The key new falsifiable prediction is that useful route optionality tracks
**distinct attainable future corrections and transferable updates**, while
the gap between same-sample and transfer progress tracks gradient variance.
This retains the user's intended “choices that become useful later” meaning
and supplies a way to distinguish it from noise, entropy, or immediate credit.

`e116_optionality_contract.py` checks the finite examples exactly: complementary
observable futures have option premium 1; one volatile route and duplicated
routes have premium 0. Gradients +2/−2 have same-sample score 4 and expected
independent transfer 0; consistent gradients +1/+1 score 1 under both.
These illustrate the derivations, not learning performance or a novelty claim.

## 163. A useful shadow must be realizable by the router's actual controls

There is another distinction between an intervened event tree and a trainable
option. A replay can force an event while bypassing the equations that would
produce its timing, payload, reset, and descendants. The replay may improve
loss even though no allowed routing decision or small parameter update can
produce that continuation. Such a shadow measures an intervention, not yet
an actionable option for this network.

**Implementation connection.** Legacy TVLayer inserted forced spikes using
the post-arrival grid state while reconstructing ordinary spikes from the
previous state (§155). A forced birth could therefore emit a payload that
lowering the ordinary threshold would not reproduce. This is an additional
reason to establish a consistent transition kernel before trusting learned
option values. The new grid reference makes those payload conventions agree;
parameter realizability and changed event times still need separate checking.

Let ξ be the actual local controls: gate logits, thresholds, delay parameters,
or supported stochastic choices. For a differentiable deterministic control
vector, approximate a missed-event margin by
\(m(\xi+\delta)=m+a^\top\delta\), with m<0 and a=∇m. Under the
trust-region budget \(\delta^\top M^{-1}\delta\le\rho^2\), M positive
definite, a crossing is locally reachable only if

\[
 -m\le\rho\sqrt{a^\top Ma}.
\]

For an affine margin this condition is exact. Cauchy–Schwarz proves necessity;
the minimum-cost crossing achieves it with

\[
 \delta^*=\frac{-m}{a^\top Ma}Ma,\qquad
 \min\tfrac12\delta^\top M^{-1}\delta
       =\frac{m^2}{2a^\top Ma}.
\]

If a=0 and m<0, no first-order control can create the event. Nonlinear
history-dependent margins need a remainder bound or direct replay of δ*;
the linear estimate alone is not a guarantee across support changes. This
is a **controllability-normalized margin**, distinguishing a nearby event
that the trainable controls cannot move from a farther event that they can.

For several desired changes, use the joint constraints Aδ≥b, also including
the events that must remain unchanged. Independent event overrides do not
imply joint feasibility under shared weights. For example, margins ξ−1 and
−ξ−1 can each be opened by changing ξ, but can never both be positive in the
same deterministic model. A double-open shadow invents a route bundle that
does not exist. Adding independent stochastic controls could change that
answer; it changes the model and must be specified and costed.

Refine §159's attainable set to trajectories produced by **the actual
transition kernel with allowed controls and bounded work**, rather than all
forced-event combinations. Three checks now precede optionality credit:

1. Is the counterfactual's transition/payload/reset convention the same as
   the deployed mechanism's?
2. Can an allowed control produce it, alone and jointly with the other
   choices needed by its descendant continuation?
3. Does the resulting feasible continuation preserve useful future choices
   and produce transferable learning at the declared budget?

These conditions explain why a richer tree can still offer little trainable
optionality. They also suggest a better proposal: prioritize distinct,
control-reachable alternatives, replay the corresponding control changes,
then allocate deeper continuation search using its measured value and cost.
The trust-region calculation uses only each local control neighborhood;
it does not require dense all-network inference. Global compatibility may
still limit composition, and must be measured rather than assumed.
