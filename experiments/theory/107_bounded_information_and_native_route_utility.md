# Bounded score information and actual route utility

Let raw emitter gradients be g0,g1 and bounded score slopes d0,d1. At a fixed
history the winner/first-time Fisher pullback is G^T diag(pi0*d0^2,pi1*d1^2) G,
where G has rows g0,g1. Its two nonzero eigenvalues have product
pi0*pi1*d0^2*d1^2*(||g0||^2||g1||^2-(g0 dot g1)^2).
Thus hard saturation of one candidate forces local rank at most one. A
positive slope restores rank exactly when the raw emitter gradients were
independent; it supplies access to existing freedom, not new parameter freedom.
This is a conditional local statement. Other histories can increase the total
shared-parameter rank even if an individual history loses rank.

For |raw|<=R, the fixed alpha bridge slopes are at least
alpha*(1+(R/C)^2)^(-3/2). With bounded scores ±C, each choice probability is at
least 1/(1+exp(2C)). On the raw two-dimensional span, the information minimum
is bounded below by pi_min*d_min^2 times the smallest raw Gram eigenvalue.
This is a strictly positive but potentially tiny bound: C=12 permits choice
probability near3.8e-11 and large raw scores shrink sensitivity as R^-3.
There is no uniform remote conditioning guarantee or finite-precision theorem.
Kinks at ±C remain for alpha<1; finite-difference checks avoid the kinks.

Useful choice credit also needs utility differences. For fixed actual first
time, history and suffix law, conditional risk pi0*L0+pi1*L1 has derivative
with respect to raw0 equal pi0*pi1*(L0-L1)*d0 (opposite sign for raw1).
A positive information direction can have zero utility. Full-write suffix
replays are required to measure L0,L1; a frozen entering-state score gap alone
cannot justify an update or a benchmark nomination. Holding suffix utilities
fixed is a local-expectation derivative, not all pathwise message/time credit.

Next bounded diagnostic: the five strictly saturated seed7 histories from
note104, using the same two previously FIT-unused inputs258/981 and fixed
pass4 producer. Reveal these FIT labels for utility diagnosis (they are no
longer label-unused). Force each actual alternative at the factual FIRST
arrival, preserve observed prefix history/RNG and then allow actual suffix
memory and clocks to evolve with the original hard map. Verify original
winner recovers all original outputs/state/end RNG. Report every utility gap
and hard/bridge scalar derivative; no selected-sign gate, optimizer, DEV/test,
or whole bridge risk/quality claim. Charge the extra raw score taps and replays.

Also audit the completed restricted note106 smokes: exact budgets/source
lineage, full/partial work coverage, cap exposure at initial/DEV-selected
parameters on their32FIT inputs, native state bytes and a common-unit ledger.
Preserve the bridge's negative smoke NLL difference and both learning curves.
No larger fit is automatically admitted by a nonzero local utility derivative.
