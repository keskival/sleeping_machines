# Parameter-aware replay magnitude priorities

Failure: score-based weightedk3 improves fresh FIT64..95 aggregate score variance
(.768144/.693763), but shared-parameter cases fail (.600428/4.011873,
.639390/1.009399). Preserve that positive score evidence and failed gate.
Concrete reason for departure: model gradients are J_r^T g_r, not g_r alone;
conditional shared-parameter variance depends on Jacobian geometry/cross-site
covariance. Train the SAME fixed64tree/depth6/minleaf8 predictor on log norm
of actual full parameter route vectors. This changes training target only;
features, race/time/law, clock, writes, sparse inference and proposal floor
remain. No new producer update or extra epochs on the failed model.

First32 FIT examples only, saved utility cache and recreated factual scores,
1280new training VJPs total across seeds7/8; verify cached probabilities bitwise.
TRAIN median parameter norm*.001 floor>=1e-12. Freeze predictor before NEW
critic-heldout FIT96..127. Producers have seen all FIT labels; no quality or
producer-cross-fitting claim. k3 sequential weightedwithoutreplacement,
exact marginal/pair probabilities and Horvitz-Thompson credit;90% predicted
norm/10%uniform. Every conditional scoreaggregate AND every parametercase
(indices96/97) must<=uniformk4 in BOTH seeds. Same original stringent gate,
no post-result relaxation. Uniformk3 control mandatory. Six-lane proposed
correction versus eight-lane baseline is not diagnostic actual work: diagnostic
still enumerates40lanes/prefix. Include predictor/VJP/inclusion costs unknown,
original producer/cache costs retained. No DEV/test, long model fit or naive
importance weighting. Unique guarded2GiB RSS/6GiB virtual,8GiB floor,600s CPU.
