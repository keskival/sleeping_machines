# The local information metric of a winner and its computational time

## What the completed intervention changes

Theory101's two frozen producers gain only .011155/.002257 mean per-history
NLL from temperature2, below both .02 nomination thresholds. Greater softness
alone has not earned an integrated fit. The tested construction preserves the
current-state first-time law, performs real sparse writes and retains temporal
content computation. It does not identify a global route-learning impossibility.
Results/sources remain immutable; no changed gate or unchanged scale-up.

Other-host exact replay's first seed similarly fails quality, while its strong
fit/dev gap motivates the independently owned regularization campaign. The
local next theoretical issue is the metric used to balance credit, including
the clock coordinate. This note derives a local identity, not a new architecture
or optimizer adoption. Native gradients, state and inference are unchanged.

## Exact diagonal information, once the first time is retained

At one common-start race, lambda_i=exp(s_i), Lambda=sum lambda and
pi_i=lambda_i/Lambda. Observation (W,T) has density

    p(W=i,T=t)=lambda_i exp(-Lambda t).
    h_i = partial_s_i log p = 1[W=i] - lambda_i T.

W~Categorical(pi) and Z=Lambda*T~Exp(1) are independent. Decompose the
local likelihood score into q=onehot(W)-pi and c=pi*(1-Z). Both have mean0,
their cross covariance is0, and

    E[qq^T] = diag(pi)-pi*pi^T,
    E[cc^T] = pi*pi^T,
    F_(W,T) = E[hh^T] = diag(pi).

Keeping computational time fills the categorical common-shift null direction.
Categorical-only Fisher is singular; the joint winner/time Fisher is positive
definite for finite scores. This is information in observed winner+first time,
not the metric of all independent losing clocks: observing EVERY raw clock
instead gives identity information in log rates, and requires extra observation.
Counterfactual learning can use residual clocks, but it must charge that work.

In explicit coordinates b=log Lambda and categorical logits a,
pi=softmax(a), s_i=b+log(pi_i), the local metric is block diagonal:

    F_bb=1; F_ba=0; F_aa=diag(pi)-pi*pi^T.

This is statistical orthogonality at a fixed entering state. If b and a depend
on shared message/memory parameters, or route changes affect later histories,
those actual parameter/history couplings remain. It does not justify severing
the content/time graph, dropping pathwise derivatives or adding clock scores
on top of an already correct reparameterized clock derivative.

## A calibrated direction is not free precision

For conditional expected return ell_i=E_T[L(i,T)] without explicit score
dependence inside L, the categorical gradient is

    g_choice_i=pi_i*(ell_i-mean_pi ell).

A correct common-clock contribution is g_clock_i=pi_i*kappa. Under the
joint local inverse metric, F^(-1)g_choice=ell-mean_pi ell, whose pi-weighted
mean is0, and F^(-1)g_clock=kappa*ones. Thus choice and speed have distinct
local directions. A finite score step along the centered utility preserves
log Lambda only to first order; explicit b,a coordinates preserve b exactly
at fixed entering state. Subsequent changed writes legitimately change clocks.

Near pi_i=0, the inverse metric amplifies uncertainty too. Even for the raw
zero-mean likelihood score, E||F^(-1)h||^2=sum_i 1/pi_i, whereas E||h||^2=1.
A useful preconditioner therefore needs measured counterfactual-return noise,
conditioning/damping and a bounded candidate budget. Large alternative credit
and simple marginal normalization do not automatically provide reliable updates.
Legal frozen baselines/control variates or paired returns may change that noise;
their quality and expensive utility discovery remain empirical questions.

For shared parameters theta with score Jacobian J, F_theta=J^T diag(pi) J.
The parameter metric can be singular even though local F is nonsingular.
Per-score division is generally NOT the same as a parameter natural gradient;
actual shared-parameter covariance and content derivatives must be retained.
This explains why an isolated score-variance gain can fail parameter checks,
without declaring the observed AWS failure a universal theorem.

## Bounded next decisions

Check this information identity and conditional-return gradient by exact
finite winner sums/exponential moments plus independent quadrature. Check the
shared-parameter pullback and the inverse-metric noise amplification. No fit
is admitted from these identities. An optimizer proposal would require native
parameter/input gradients, correct conditional route/common-clock credit,
restored optimizer/RNG identity and complete resource work before a tiny fit.

Prioritized integrated evidence remains the independently owned batched
regularization and remaining replay confirmations on saved DVS protocols;
AWS owns parameter-targeted replay allocation and deferred dense comparisons.
These contracts support credit calibration reasoning while the failed frozen
temperature screen prevents another low-value fit. Learned reception and
popcorn remain incomplete boundary/scheduler/learning mechanisms, not deployed
substitutions. No proof of advantage, regression or impossibility is claimed.
