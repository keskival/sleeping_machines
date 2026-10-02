# Bounded full-state joint-clock correction

Concrete failure: a current native DVS alternative has beneficial delivered
content but a much worse persistent write and complete suffix. Its value-only
teacher favors the worse route. Exact terminal content credit fails its
unchanged256-fit gate; that terminal race has no supervised future write use.
Theory57/80 therefore prioritize earlier actual write/suffix outcomes.

Retain the original sparse temporal architecture, parameters, physical delay,
key/value separation, candidate pool, selected updates and inference code.
Change fitting score credit at one middle-prefix race head/window, across all
independent clips. Rotate layer/head over four fixed passes. Ordinary sampled
content/state derivatives and all uncorrected route estimators remain.

At the corrected head, raw minimum time T and scores s define lambda=exp(s),
pi=lambda/sum(lambda). The factual loss F_W is already observed. Replay one
full independent-clip window with each clip's other candidate forced, committing
its actual memory and continuing with unchanged future random draws. Preserve
the current raw first time; earlier prefix and other head decisions match.
The original two-candidate state already computes all proposals. This replay
produces F_0,F_1, including future clock changes and discontinuous histories.

Replace, rather than add to, the selected score tensor's value/time derivative:

    g_i = pi_i (F_i-b) - lambda_i T sum_j pi_j(F_j-b).

This is the conditional-winner mean of the joint winner/time score-function
derivative from theory57. In expectation over T and independent future draws
it is the exact isolated-node likelihood component under its fixed-prefix
assumptions. The direct sampled candidate/content derivative is retained.
Replacing the entire local score derivative avoids double-counting its original
interior timing derivative. Upstream shared arguments receive the replaced
score derivative as well as their existing content/state paths.

Use b from the existing class head applied to the mean candidate messages
constructed before the corrected head's draw, with no sampled delay or suffix
adjoint. Conditional on the entering prefix/candidates it is independent of
the current winner and T, so its expected joint-score contribution is zero.
Detach it in this estimator and pay its head/loss arithmetic. This baseline
may still be a poor variance reducer; detachment alone is not the independence
argument. No time-dependent or winner-dependent fitted suffix baseline is used.

Only one head/event receives full likelihood credit in a window. Earlier
prefix estimators and uncorrected downstream derivative paths remain local;
no exact whole-model gradient, optimal variance, or zero discovery cost follows.
Factual labels enter supervised losses and the baseline only, never clocks,
candidate construction, fixed site schedule or replay noise. Inference adds
no fitted parameters, experts, delivered values or latency.

The full prefix replay currently recomputes unchanged earlier events and other
independent clips. Charge it honestly; cached-prefix/fused suffix execution is
a later optimization with its own contracts. The small memory budget has
priority over an unbounded shadow tree. Numerical contracts precede integrated
full/partial-window smokes and the gated pilot in COUNTERFACTUAL_CREDIT_PLAN.md.
Sampled score-function gradients can have high variance even with perfect
local expectations. A poor fit will test this correction/coverage/baseline,
not prove that useful temporal computation or full-state credit is impossible.
