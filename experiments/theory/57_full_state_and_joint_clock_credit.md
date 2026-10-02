# Full-state credit and downstream timing discontinuities

Global §§373–375. Extends notes53–56 and the AWS trained-route audit without
changing a frozen model. Observed teacher errors motivate targeted diagnosis,
not replacing temporal races with a dense attention scaffold.

## 373. A route changes stored state as well as its message

Let F_ab be suffix loss when a selects the delivered value and b selects the
committed candidate memory/address. Zero is the sampled winner,one the
alternative. Fix the selected-node delay and continuation random numbers:

    F11-F00 = (F10-F00) + (F01-F00) + (F11-F10-F01+F00).

Subtracting g_v dot(v1-v0) splits the teacher error into delivered-value
residual,persistent-write effect and their interaction. The residual includes
nonlinear response AND downstream route switches. Hybrids are diagnostic
interventions,not new legal routes. `route_write_decomposition.py` tests this
on the same frozen checkpoint and predeclared probes.

Completed AWS factorial replay (2 October) finds mean absolute persistent-write
effect .012694 versus delivered-value linearization residual .000288 across
the12 probes; these absolute summaries are not additive attribution percentages.
The two earlier opposed signs follow actual writes: event0/block4's value-only
effect is +.001504 while write-only is −.016749; event7/block7's are −.001280
and +.033918. This isolates an actionable addressed-state credit path. It is
one population/address/noise realization, not expected-gradient accuracy.
The guarded audit paid48 forwards/12 backwards/12,288 races,7.107s wall and
399,708KiB RSS. Weights/RNG are preserved; arithmetic remains uninstrumented.

Under fixed smooth slot semantics,old states m_i and candidate writes u_i give
the full-state linearization from winner j to alternative i:

    g_v dot(v_i-v_j) + g_mi dot(u_i-m_i) - g_mj dot(u_j-m_j).

Sparse addressed adjoints can calculate these terms locally. Arrival metadata,
slot validity and other state channels must also be represented and charged.
New slots and discrete routing do not inherit a Taylor bound by naming a
state vector. If the explicit continuous suffix has beta-Lipschitz gradient
along the whole segment,the linearization remainder is at most
beta/2 times ||delta||². Actual conditional replay tests where that assumption
fails. No state-aware teacher is installed here.

## 374. Joint likelihood credit handles suffix jumps

At one isolated node,lambda_i=exp(s_i),Lambda=sum lambda_i,p_i=lambda_i/Lambda.
The joint winner/minimum density is lambda_W exp(-Lambda T). Hold prefix,
candidate construction and independent continuation noise xi fixed with
respect to the LOCAL scores. Differentiating the density under appropriate
integrability gives

    grad_s E[F_W(T,xi)] = E[(e_W-lambda*T) F_W(T,xi)].

This uses established score-function/likelihood-ratio principles; see
[Williams (1992)](https://link.springer.com/article/10.1007/BF00992696).
Parameters also changing candidates/prefixes require their direct derivatives.
A local correction does not certify unbiased learning for the whole model.
Conditional winner enumeration gives the integrand

    p elementwise_times F - p*(Lambda*T)*sum_i p_i F_i.

It requires actual branch-specific state/suffix losses,but no pathwise time
derivative of a discontinuous suffix. Converting time-score credit to a
pathwise derivative by integration by parts requires jump contributions;
ordinary fixed-route autograd supplies only smooth interiors.

Witness: F0=0,F1=1[T>t0]. Expected cost is p1 exp(-Lambda*t0),with gradient

    p1 exp(-Lambda*t0) * (e1-p-lambda*t0).

Smooth-interior timing derivatives are zero almost everywhere and omit the
last term. At lambda=(1,1),t0=1,this reverses score1's direction and misses
the nonzero common-rate derivative. This analytical witness is not a claim
that every trained timing gradient fails. Preserve the positive existing
combined-clock teacher identity for linear payload/bounded delay; neither
component inspection nor this witness licenses an untested partial rewrite.

## 375. Analytic surrogate plus bounded residual replay

A prefix-only baseline b has E[(e_W-lambda*T)b]=0. A time-dependent baseline
does not generally vanish; dropping its rate term loses credit.
For A_i(T)=a_i+b_i*T with coefficients fixed conditional on prefix/candidates
or independent calibration,exponential moments give

    G_A,i = p_i * [a_i-sum p*a + (b_i-2*sum p*b)/Lambda].

Set R_i=F_i-A_i. At sampled T,draw I from full-support q and add
(p_I/q_I)*(e_I-lambda*T)*R_I to G_A. Its expectation gives the exact LOCAL
joint-score correction when R is the actual conditional state/suffix loss.
Surrogate coefficients and proposal bookkeeping are not differentiated in
this policy term; direct prediction derivatives remain a separate path.
No suffix time derivative is required.

The coefficients must not depend on the current sampled winner/time through
its suffix adjoint. Merely detaching such an adjoint removes autograd links,
not statistical dependence; it does not establish the stated unbiased identity.
If q=p and I is the actual winner,the already observed suffix loss supplies
R_I without an additional replay. That preserves the identity under the same
independence conditions,but may have high variance. Extra alternative replay
is a variance/information tradeoff,not a mathematical necessity in this case.

One sampled residual avoids exhaustive suffix enumeration,but replay is not
free. A floor q_i>=epsilon*p_i bounds importance ratios; variance,prefix
storage,state copying,suffix length and optimizer work still decide utility.
The full-state correction contains clock-rate credit,not only conserved
winner credit. An approximate residual introduces bias.

Admission: factorial audit -> specific missing path -> joint state/time
derivation -> integrated numerical/recovery/accounting contracts -> matched
small fit against the unchanged teacher. Charge every replay and preserve
temporal races,private state,deep content and counterfactual teaching.
