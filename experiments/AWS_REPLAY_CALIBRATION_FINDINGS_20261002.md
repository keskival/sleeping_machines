# Replay critic calibration fails score and actual parameter variance screens

Training-only scalar calibration on the original32 FIT examples gives alpha
.691188569(seed7) and1(seed8); heldout next32 have k2/plaink4 score-MSE
2.149378/2.493344. Both fixed nominations fail. Raw critic failures retained.
No extra critic/model updates, new dev evaluation or long fit.

Actual full shared-parameter route gradients on heldout indices32/33 verify
all recorded score probabilities bitwise against the cached factual episode.
Finite-population variance includes cross-site covariance, rather than treating
parameter coordinates as separate race scores. Exhaustive R4 contracts pass.

| Seed | FIT index | Calibrated k2 / plain k4 parameter variance | Cross-site coherence |
| --- | --- | --- | --- |
| 7 | 32 | 2.383887 | 1.815013 |
| 7 | 33 | 2.189517 | 1.550583 |
| 8 | 32 | 37.462736 | 1.281145 |
| 8 | 33 | 1.674462 | 1.009641 |

Every bounded parameter example also fails. The extreme seed8/index32 ratio
shows score-space aggregates can hide parameter-specific failures. Scope:
conditional route-subsampling variance at fixed factual episodes, not complete
risk-gradient exactness, optimizer behavior or quality. Producer previously saw
all FIT labels; critic holdout is not producer cross-fitting. k1 and raw/zero
results, baseline magnitudes and calibrated scales retained in JSON.

1.955s/408688KiB,80 site-score VJPs per seed (two utilities times20sites times
2prefixes);160total VJPs. Original producer fits, cached replay/critic work
remain paid; new VJP FLOPs unknown,not zero. No new replay or optimizer work.
Unique guarded CPU queue,2GiB RSS/6GiB virtual,8GiB host available floor.
No unchanged reduced-replay fit admitted. Prioritized integrated corrected
replay quality remains independently owned. AWS resumes separately preregistered
compact current-context readout work, retaining its dense primitive/control scope.
