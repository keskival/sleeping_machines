# Critic calibration and actual parameter variance

Concrete failure: norm-only and signed/label-aware frozen critics worsen
conditional score-coordinate variance. Critic cancellation remains correct;
utility scale and feature quality have not been isolated. This experiment
reuses the complete cached signed critic, with no new critic/producer update.
Fit a scalar alpha=clip(dot(g,q)/dot(q,q),0,1) on the original first32 FIT
prefixes, in route-score coordinates. Freeze alpha before examining next32.
Compare raw, calibrated and zero critic. Same k1/k2 versus plaink4 gate
(MSE<=baseline in both seeds7/8). This is a distinct calibration hypothesis,
not more epochs or a changed interpretation of earlier failure.

For shared parameter site vectors d_r, uniform k-of-R correction has exact
variance trace R(R-k)/(k(R-1))*sum_r||d_r-mean(d)||². Derive via inclusion
covariance; exhaustive R4/k1/2/4 numerical contract verifies it. Unlike the
disjoint score-space formula this includes cross-site cancellation/coherence.
Audit all15523 producer parameter coordinates on first2 critic-heldout FIT
prefixes, index32/33, both seeds. Actual vector-Jacobian products of recorded
scores, same factual race/noise/state and cached conditional utility. Compare
raw/calibrated/zero separately; alpha is not fitted on these parameter results.

Sparse temporal inference, actual alternative writes, separate keys/values,
factorized clock and corrected first-time replay unchanged. Only detached
learning critic scale changes. No inference substitution or quality evaluation.
Conditional route-estimator variance is not full risk-gradient unbiasedness or
supremacy. Existing producer/prior diagnostic costs retained; new VJP work
additional and FLOPs unknown. Unique one-job guarded queue,2GiB RSS/6GiB
virtual,8GiB available floor,600s timeout,16CPU/31GiB CPU-only host,one thread.
