# Race sampling, Adam rectification and actual functional effect

Theory124 completed five contract groups in292.571s/866372KiB. Its saved
initialization-only p16/D2/D4/D6 vectors give exact k8 combined-gradient
MSE ratios .017934/.084633/.305063. Mean raw-gradient cosine is
.991217/.961076/.905864, but fresh-Adam displacement cosine is
.702075/.561206/.567056. k32 reduces variance, without bringing Adam
displacements close. These are conditional parameter-space observations,
not trained failure causality or heldout improvement.

## Correction beside the original immutable evidence

124's cases.shadow_lanes/events count the FIRST complete alternative bank.
The original-driver equivalence contract evaluates that bank AGAIN.
Across the three main depths this is4032 shadow lanes/84672 shadow events,
plus16 lanes/16 events in the tiny contract. Original wording 'paid once'
does not describe total diagnostic execution. All numerical data, source,
original wording and prior evidence are retained; total FLOPs remain unknown,
not zero. The subset backward contract differentiates the cached-Q objective,
not an independent BL.route_term(k=8) execution. The loader reads FIT0..15;
only0/1 enter124. Report these distinctions explicitly.

## Why a norm or cosine alone is insufficient

Fresh Adam after scalar clipping gives, coordinatewise,

    delta_i = -eta * g_i / (|g_i| + epsilon/k_clip).

Away from epsilon this behaves like a sign transform. A coordinate with
small full mean can acquire almost a full-size sign step under unbiased
route noise, even if the factual gradient dominates the global L2 norm.
For an approximately Gaussian coordinate, sign-flip probability is
Phi(-|mean|/sigma), not a function of global cosine. No Gaussian
approximation is assumed in the completed measurements. Scalar clipping
does not restore that lost coordinatewise signal-to-noise; its scale
largely cancels in the fresh-step transform. Trained moments differ.

Conversely, large parameter disagreement can lie in directions with small
functional effect. Locally, delta logits approximately J_logits delta_theta;
discrete route changes and clipping boundaries require actual finite
forwards rather than claiming this linearization always applies. Thus do
not launch a larger sampling fit merely because124 found large step noise.

## Prespecified actual finite-update audit

Reuse124's EXACT represented double weights, complete normalized factual
and route-vector banks, source/data hashes and stored sample indices.
Reconstruct all D2/D4/D6 factories and assert their parameter vector equals
the saved vector bitwise. Rebuild full/sample combined gradients from
saved vectors and verify all64 saved sample metrics numerically, without
another expensive full counterfactual or gradient sweep.

For EACH depth, execute one fresh actual clip1 Adam.003 step in54 total
discarded independent forks: factual-only and full route credit, plus
the FIRST EIGHT stored draws for k8 and k32. No selection by loss/variance.
Check every actual stored displacement against the validated formula;
no Adam moments from a trained checkpoint are implied.

Evaluate each fork and the unchanged initial model on SAME FIT0/1 and
separate fixed FIT2..15 anchors, at common noise seeds171323/171324.
Labels never enter pre-update predictions. Record every logit/probability,
cross-entropy change, prediction KL relative to full-credit and original
models, parameter-update norm, and all declared sample metrics. These
are controlled finite FIRST-step FIT diagnostics; no DEV/test array,
heldout quality, complete expected risk, convergence or advantage claim.
Anchor examples use the same FIT-only normalization, not a new data split.

Counterfactual return banks are reused, not re-executed. Cached gradients
still represent the conditional-choice surrogate with detached losing
contents. Charge54 executed optimizer forks and1824 prediction-target
evaluations (96 baseline plus1728 fork evaluations). Numerical contracts
add their declared work. Total FLOPs/traffic/energy remain unknown.
Sparse addressed memory, key/value separation, computational delays/races
and full factual episode paths remain unchanged; no architectural
substitution, inference claim or active AWS edit is proposed.

## Decision and resource gate

A benign functional result weakens parameter-angle alarm as a sufficient
diagnosis. Consistently worse same-FIT/anchor steps make initialization
sampling interference more concrete, without proving trained causality.
Never promote k32 solely from this audit; other-host live-gate comparisons
and the AWS10M quality matrix remain the integrated quality priorities.
No new local dense fit. One unique run_safe queue, one thread, virtual3GB,
RSS1.25GB, at least8GiB available, timeout180s. Its bounded forward-only
forks reuse the292.6-second saved gradient work rather than repeat it.
