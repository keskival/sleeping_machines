# Winner-dependent downstream teachers and counterfactual time laws

This records two scope corrections to note59 §400 and
sleeping_machines/exact_pi_race.py without deleting their results, changing
completed models or overwriting the separately owned source. Exact-probability
linearized credit remains a legitimate empirical candidate; it is not generally
an unchanged-expectation, zero-variance version of the old teacher.

## A downstream derivative is not independent of the winner

At a fixed entering prefix, let rates lambda_i, total Lambda, probabilities
pi_i=lambda_i/Lambda, W~pi and T~Exp(Lambda). W and T are independent, but the
downstream payload error g_W generally depends on the delivered value and
subsequent memory writes/routes. For this section let g_W be independent of T.
Define d_i^W=g_W dot(v_i-arithmetic_mean(v)). Native score credit is

    C_i = lambda_i*T*d_i^W - 1[W=i]*T*sum_j lambda_j*d_j^W.

Then

    E[C_i] = pi_i*E_W[d_i^W] - pi_i*sum_j pi_j*d_j^{W=i}.

The replacement using exact probabilities gives

    E[C'_i] = pi_i*E_W[d_i^W-sum_j pi_j*d_j^W].

They agree for a fixed downstream error, as the existing fixed-linear-error
contracts demonstrate. They do not generally agree when it depends on W.
Dependence on T introduces additional covariance with the T factor. The
replacement still depends on sampled downstream error/prefix/time, so its
complete gradient is not zero variance. Calling pi known does not make g_W
known for every alternative. Correct Rao–Blackwellization averages the entire
random teacher conditional on the retained information, not just its rate
coefficient. Neither local rule is a general exact nonlinear choice derivative.

Concrete convex witness: values0/1, rates1/3, target.6, loss.5*(v-target)^2.
Legal losses are.18/.08; pi=.25/.75. Exact categorical gradient is
[+.01875,-.01875]. The original teacher's expected gradient happens to equal
that exact vector for this two-candidate quadratic. The exact-pi replacement
has expected gradient[-.028125,+.028125], the opposite direction. This is an
analytic categorical enumeration using E[T]=.25, checked against the actual
backward implementations by race_teacher_expectation_contracts.py. It is not
a general defense of the original surrogate, which has other proved failures.

For a general vector-valued quadratic with fixed candidate values, this has a
particularly useful interpretation. Let mu=sum pi_i*v_i. The exact-pi teacher
uses the sampled error v_W-y; averaging W gives

    E[C'_i] = pi_i*(v_i-mu) dot(mu-y)
            = d/ds_i [.5*||mu-y||^2].

But the hard-delivery objective is

    E[.5*||v_W-y||^2] = .5*||mu-y||^2 + .5*E[||v_W-mu||^2].

The missing variance derivative is exactly what reverses the direction in the
witness. A teacher that suits an averaged dense value need not suit one hard
delivered value, even with identical available candidates and exact pi. The
identity is checked independently by ordinary autograd; it is scoped to a
quadratic/fixed candidates, not an identity for arbitrary cross entropy or
proof that soft attention is universally better. It explains why the training
objective must match actual sparse delivery semantics.

## Counterfactual identity and counterfactual first time are different

The entering-prefix exponential race has joint density

    p(W=i,T=t)=lambda_i*exp(-Lambda*t).

Consequently T|W=i is Exp(Lambda), the SAME for every i. To enumerate a pure
categorical alternative, preserve the current first time T while changing
delivered value AND actual memory commit. This is what notes84/86 implement.

In dvs_credit_fidelity_audit.py, the forced branch instead uses
time=noise[i]/lambda_i, the unconditioned individual candidate time. That time
has law Exp(lambda_i), not Exp(Lambda). At rates1/3 the corresponding means
are1/.333333 versus conditional first-time mean.25. Its existing forced-loss
table therefore measures a combined identity/time intervention. Multiplying
those losses by pi and differentiating only pi does NOT make it the exact
fixed-time categorical gradient or complete joint winner/time gradient.

Preserve curie_dvs_credit_audit_20261002T192000Z.json and its reported sign/
magnitude numbers as the original combined-intervention diagnostic. Do not
use them to conclude that pure route credit is near chance at all depths or
that coefficient noise is the main cause. A revised audit needs unchanged T,
unchanged future draws, legal writes, explicit time-derivative ownership and
independent contracts. Our earlier fixed-time actual-write audit172000Z keeps
its existing distinct scope (four previously used development prefixes).

## Decision implication

The newly completed fitting-only branch audit and functional-autograd contracts
separate sampled branch derivatives, both-branch averaging, native route/clock
paths and one fixed-time actual-choice residual. Independent whole-history
draws are finite diagnostics, not conditional-node variance measurements.
First two fitting prefixes are not representative of all gestures. Initial
smoke shows small differences between sampled and averaged branch derivatives
and mildly opposed but much smaller route/clock contributions; no strong
alternation or broad representation diagnosis follows. Complete the declared
second-model/independent-draw audit, publish both numerical results and these
scope corrections, then select a repair. Do not duplicate separately owned
exact-pi/tied-map campaigns or infer supremacy from a primitive witness.
