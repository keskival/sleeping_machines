# Quantify score-bound exposure and test an explicit gradient trap

## What is still unknown

Theory103/104 identifies hard-clamp rank loss and restores native local emitter
sensitivity in five selected cases. It has not measured how broadly saturation
appears or whether saturation blocks useful counterfactual returns. Count the
already frozen99/100 arrays; no new native forward, loss, optimizer or DEV/test.

The43,008-race profile99 stores postclamp scores. Equality to a bound is a
CAP-OCCUPANCY PROXY, not proof that every raw score is strictly outside. The
5,376-race-pair100 decomposition provides static+dynamic reconstructed raw
scores. Use a conservative2e-5 margin, exceeding observed reconstruction error,
to mark definite |raw|>12+2e-5; mark boundary/ambiguous cases separately.
Preserve both cohorts and noise scopes rather than pooling unequal observations.

Report candidate cap fraction, any/both capped race fractions, observed capped
winner fraction, expected capped-winner probability and event/layer/head groups.
When raw saturation is confirmed, diag(raw-to-score derivatives) gates the
winner/time Fisher: F_raw=diag(pi_i*kappa_i^2). The hard clip can remove one
or both local coordinates; no marginal clamp count is a utility-weighted
parameter credit or proof of the benchmark bottleneck.

## An exact expected-risk optimization witness

For two addressed choices with fixed conditional utilities(1,0), initial raw
emitters(13,13), hard cap12 produces equal scores and zero gradient to BOTH
raw emitters. This is an open flat cell, not a symmetry with a nonzero tangent.
Lowering the costly emitter to11 in a finite perturbation improves expected
utility; the representational family admits progress, but exact local gradient
descent cannot leave the cell alone. Other model parameters, momentum, decay
or noise could leave it; no universal training impossibility is claimed.

Retain computational time in the witness: common first T=Z/Lambda with
Z~Exp(1), physical delay .001+.010*T/(1+T), add .1*E[delay/.001] to the
expected conditional utility.48-node exponential quadrature evaluates it.
Two fixed gradient-descent traces, alpha0 and bridge alpha.1 from104,400steps,
step size1, identical initial raw emitters and exact expected gradients.
These are synthetic scalar-risk steps, NOT fitted DVS encoders or an accuracy
benchmark. Save every iterate; a positive bridge should escape the flat cell
and reduce utility+latency risk, while hard stays bitwise at its initialization.
Check the hard zero gradient, finite-hop improvement and positive native-map
gradient before descent. No hyperparameter search or training nomination.

Exact return enumeration has no estimator noise in this witness. A stochastic
native learner still needs useful returns, variance calibration and shared
parameter/history derivatives. Restored derivative magnitude alone is not
sufficient. Every clock/race/map/quadrature/gradient operation is paid, with
audit FLOPs/traffic/energy unknown, not zero. One safe unique1thread/120s queue.
Integrated priority stays with existing deeper replay/growth owner campaigns.
