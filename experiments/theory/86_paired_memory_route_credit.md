# Bounded paired actual-memory choice credit

The actual-write choice pilot improves seed6 by.111931NLL/4.6875pp, but seed7
regresses by.029693NLL at unchanged55.21% accuracy. The predeclared confirmation
gate fails. Do not run the unchanged full-data/longer campaign or promote the
best seed. Preserve both scopes and numerical successes. This note addresses
the earlier question about larger associative counterfactual pools with bounded
replay work, not a demonstrated cause or solution of that replication failure.

## Finite estimator and calibration

At one isolated entering prefix and fixed first-arrival time T, let pi be the
race winner probabilities and F_i(T) the complete legal delivered-value/write/
suffix loss with fixed independent future draws. Observe W~pi and F_W. Draw
I!=W from q(I|W), with support on every eligible alternative. Replay just that
actual alternative, preserving T and future draws. Set

    g_hat = [pi_I/q(I|W)]*(F_I-F_W)*(e_I-pi).

Taking expectation over W and I, the term I=W contributes zero, so

    E[g_hat] = sum_W pi_W sum_I pi_I*(F_I-F_W)*(e_I-pi)
             = sum_I pi_I*F_I*(e_I-pi)
             = pi elementwise_times (F-sum pi*F).

This is the exact conditional categorical-score component, not the complete
winner/time density gradient. It uses the *loss difference* between two legal
full-state outcomes; shared additive loss offsets cancel sample by sample.
No inaccurate intermediate-head baseline is necessary. For two candidates,
the sole alternative has q=1 and every realized estimate equals the full
enumerated categorical derivative. For larger pools the estimator is unbiased
under the stated fixed-prefix/time sampling assumptions, with nonzero variance.
Sample reuse/conditioned weights retain the limitations in notes82/85.

Choose the alternative proposal

    q(I|W)=(1-epsilon)*pi_I/(1-pi_W)+epsilon/(M-1), I!=W,
    q(W|W)=0, 0<=epsilon<1.

Then pi_I/q(I|W)<=(1-pi_W)/(1-epsilon)<=1/(1-epsilon), and the estimator norm
is at most sqrt(2)*|F_I-F_W|/(1-epsilon). This bounds the importance multiplier
independently of pool size; it does not bound loss differences themselves.
The uniform part gives each nonwinner probability at least epsilon/(M-1),
retaining exploratory support without making uncertainty itself a reward.
An uncertainty/optionality-weighted proposal is a separate intervention and
must retain propensities/support. No advantage for entropy preference is shown.

The implementation samples q with a fresh auxiliary exponential race, consuming
M random exponentials/clip/window. q and importance weights are proposal
bookkeeping, not new inference routes. Their score/proposal arithmetic is paid;
RNG counts are separate from the established unit-special arithmetic convention.
Current winner/future native draws keep the matched per-pass convention; auxiliary
draws use the serialized Torch RNG for exact recovery, without label/identity seeds.

## Integrated boundary and prior work

Retain native time decay/rotation, physical delays, race choices, separate
keys/values, sparse persistent state commits, shared content arguments and the
existing factual timing/content derivatives. One fixed event9 head/window
receives paired actual-state choice credit. Inference adds no proposal draws,
parameters, winners or deliveries. Fitting pays one full alternative forward
regardless of M, but scoring/candidate formation still scales with eligible M;
losing candidate payload maps do not automatically get direct gradients from
no-grad shadows. Shared-map exposure and all-history/jumping-time credit remain
separate open mechanisms. The method does not create free memory discovery.

This applies established score-function and sampling/control-variate principles,
not a claim to invent them. See [Williams1992](https://doi.org/10.1007/BF00992696)
for score-function learning and [Kool et al.2020](https://arxiv.org/abs/2002.06043)
for unbiased sampling-without-replacement gradient estimation/control variates.
Our finite identity and bounded proposal above are independently derived;
neither citation establishes this integrated native model's quality advantage.

Before any fit: enumerate W/I expectations on2/3/8/64-candidate banks, compare
independent autograd expected risk, test loss-offset invariance, constant-loss
zero credit, propensity bounds and exact two-route nesting. Integrate with the
unchanged native driver, compare all two-route state/gradients to theory84,
verify eight-route legal commits and partial-window recovery/accounting. These
small contracts are not a completed larger-pool benchmark. Other-host tied-map
capacity fits remain owned there; do not duplicate or overwrite their results.
