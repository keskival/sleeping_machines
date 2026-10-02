# Bounded terminal pair credit after the integrated diagnostic

This specializes the existing full-support counterfactual estimator in
Theory23 §199 to Note68's protected two-read terminal risk. It is a derivation
for a future comparison, not an implemented or completed resource advantage.
For the present C near11, enumerating about112 pair losses adds only2.56%
estimated fitting work in the seed6 full comparison. Sampling is not yet
justified by that small total overhead or by a presumption of better quality.

For fixed observed bank and native prefix history, let

    q_ij = pi1_i pi2_j,       L = sum_ij q_ij ell_ij.

Draw K alternative pairs from a full-support proposal r. The proposal must
not inspect the unknown target to choose candidates; if r uses learned
parameters, stop its derivative when forming importance weights. A single
pair gives the conditional gradient estimator

    g = (q_ij/r_ij) * [ (ell_ij-B) grad log q_ij + grad ell_ij ].

Its expectation is grad L because sum q grad log q =0. B may depend on fixed
context and the supervised label but must be independent of the sampled
pair; freeze it before the draw or fit it from independent samples. Reusing
the same loss sample to choose B generally biases the score term. No baseline
is subtracted from the direct decoder/value loss gradient. Differentiate the
candidate scores and actual loss, not the sampled indices or the proposal.

For first-head scores, grad log q=e_i-pi1, and the second head has the
analogous vector. Thus this estimator credits actual nonlinear pair losses,
rather than a value derivative linearized at a sampled winner. It remains
exact only in conditional expectation for terminal content risk. It neither
fixes the earlier native hard-route teacher nor provides write/clock credit.

The proposal mixture r=(1-epsilon)q+epsilon*u with uniform u over occupied
pairs has support epsilon/C² and q/r at most1/(1-epsilon). It controls
importance-weight magnitude, not all loss/gradient variance, and does not
guarantee rapid discovery or useful key contrast. Clipping weights introduces
bias. Note70's uniform-read alias and zero-residual key cold start still apply;
an unbiased estimator cannot create population signal at a stationary point.

Two C-key scans remain paid. The sampled terminal decoder/loss work becomes
K pairs and at most2K distinct interpreted values rather than C² pairs and
C values, although duplicate-pair reuse and full probability/key gradients
have their own costs. All-candidate optimizer and key work, proposal/RNG,
discovery, indices, traffic and measured host time must be recorded. This is
bounded pair-learning work, not zero-cost dormant capacity or constant-time
discovery. Retain actual two-race inference and all native temporal/state
mechanisms in a future implementation.

Required next comparison: exact small-C risk remains the reference, with
actual finite-difference/gradient-mean contracts, identical inference and
data, multiple fixed seeds, full fitting/inference accounting and no hidden
pair selection from evaluation labels. Only promote sampling after completed
quality/work evidence supports it. General learned contextual writes and
producer credit remain separate prerequisites for the main language thesis.
