# Chronological learning under distribution shift

This is a protocol, not a completed comparison or a runnable stream adapter.
Separate three mechanisms: persistent-state adaptation with frozen parameters;
online parameter learning from newly revealed targets; self-supervised test-time
training (TTT). Ordinary state updates alone are not TTT.

First require useful stationary performance from the selected integrated
H2/eight-block construction and its exact parent. Screen 128 independent
four-source populations (512 scored queries), seed6, in three regimes:
covariate drift in gaps/mark scales with unchanged order labels; concept drift
in the two-mode timing rule; recurring regimes to measure retention. Use
separate calibration streams to fix update rate, replay and drift settings.
Compare frozen and adaptive weights on the same stream for each regime.
Only promising contrasts expand to seeds6/7/8 and gradual drift.

For every query, predict with the current weight version, record its probability,
and score that prediction when the target arrives. Only then may that target
teach the model. Never rescore old predictions with adapted weights. Finish
old-version graphs or replay them correctly before modifying weights. Keep
label delays, adaptation examples, update/replay budgets and memory boundaries
equal across families. Report chronological pre-update loss, regime recovery,
forgetting, full adaptation work, state/replay bytes and learning latency.
Bootstrap independent whole streams; fix architecture and hyperparameters before
confirmation. Self-supervised next-mark learning needs a causal objective and
separate adapter; it is not silently substituted for supervised online learning.

Under a *local* strongly convex smooth risk with curvature mu/L, a gradient
step of size eta<=1/L has contraction q<=1-eta*mu. With moving minimizer
drift delta_t, gradient noise xi_t and teacher bias b_t, a tracking bound is

    error_(t+1) <= q*error_t + delta_t + eta*(||xi_t|| + ||b_t||).

This qualified model identifies conditioning, exposure, credit bias and update
latency as possible advantages. It is not a global theorem for the routed
nonconvex learner. Demonstrated superiority requires the causal experiment.
Real pushing/event adapters and the complete adaptation ledger are pending;
existing completed online-language evidence is preserved separately.
