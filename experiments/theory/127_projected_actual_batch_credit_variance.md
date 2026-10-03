# Actual-batch conditional credit variance with parameter projections

124 computes exact parameter covariance only at B2; fitting uses U16.
125's class0/1 update-versus-anchor tradeoff reinforces the need not to
generalize that tiny batch directly. Repeating every U16 race VJP would
spend thousands of full graph traversals and hundreds of megabytes.

For pool2, normalized conditional-choice contribution is exactly

    v_jr = a_jr J_jr,
    a_jr = pi0*pi1*(Q0-Q1)/B,
    J_jr = grad_theta(s0-s1).

Q and a are detached. The Jacobian includes all factual-history paths,
with the native factorized-clock custom backward. Define d=s0-s1, a
dummy cotangent w and h(w)=grad_theta(sum w*d)=J^T w. Then

    grad_w [sum_i h_i(w) z_i] = J z.

This reverse-over-reverse construction differentiates the ARTIFICIAL
cotangent, not theta twice. The native custom backward is linear in
upstream value/time errors, so it reproduces its declared surrogate
Jacobian even though forward rates/first-time are saved detached. Do not
call this a physical fixed-noise Hessian or whole-risk derivative.

For each independent Rademacher parameter direction z (coordinates +/-1,
NO1/sqrt(P)), projected v_jr*z gives a trace probe

    X(z) = sum_j R_j(R_j-k)/(k(R_j-1))
                 * sum_r (v_jr*z - mean_r(v_jr*z))^2.

For PSD conditional covariance A, X=z^T A z and E[X]=tr(A).
Var[X]=2*sum_(i!=j) A_ij^2 <=2*tr(A)^2. The mean of32 independent
sign directions has worst-case relative RMS at most25%. Empirical sample
standard error is descriptive, not guaranteed95% coverage. Record ALL
probe values, empirical SE/relative SE and this loose bound. Divide by
the EXACT combined-gradient squared norm. k8/k32 share directions for
paired comparison; projection noise is distinct from route-sampling noise.

## Prespecified contract and actual-batch audit

First recreate124's D2/D4/D6 B2 models, weights, seed171323 and factual
score histories. Use saved Q/banks; no additional contract shadow bank.
For THREE fixed sign directions per depth, every projected per-race
entry must match its exact saved parameter vector dot z. Summed projected
route credit must also match the exact full-route vector dot z. Preserve
caller RNG, all weights and original source/data/parent/artifact bytes.

Then run p16/D4 andD6/H2/pool2 initialized seed7, original fine FIT0..15,
full causal episodes, same noise171323. Evaluate all alternative returns
ONCE, factual and full-choice parameter gradients, and32 projected
directions per depth. Mask legal races per episode. No DEV/test arrays,
new optimizer step, quality fit, priority policy or architectural change.
Models/depth factories differ as already stated in124, not an isolated
depth perturbation. Q remains detached, so losing contents are not
directly taught. Ordinary native races/private state/keys/values remain.

Charge all64 main projected pullbacks,9 exact-bank contract pullbacks,
complete full shadow forward banks, factual/full gradient backprop and
all vector operations. Count lanes/events/targets/races; total diagnostic
FLOPs/traffic/energy unknown, not zero. Do not infer Adam-update variance
from raw trace alone, or nominate a sampling fit from these estimates.

One unique run_safe queue, one thread, virtual3GB/RSS1.25GB and at least
8GiB available. Timeout300s: prior exact B2 audit used292.6s mostly on
1008 individual VJPs; this reduces pullbacks to73 mixed products while
raising return-bank batch size to the actual16. Stop on any contract,
source, finite-value or resource failure; preserve failed logs/results.
