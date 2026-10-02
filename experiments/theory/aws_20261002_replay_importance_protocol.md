# Importance sampling instead of a weak replay critic

Concrete failure: fitted critics and training-only calibration fail conditional
variance gates. Separate hypothesis: utility is concentrated by race position,
so allocate correction sampling with training-only priorities. Core time/races,
actual writes, key/value separation and counterfactual returns unchanged; no
critic or model retraining. This changes replay allocation, not inference.

Saved trained coarse seeds7/8, cached first32 FIT gradients determine p_r:
90% normalized sqrt(mean||g_r||²) plus10% uniform. Freeze this ordinal-site
proposal before next32 FIT rows. k1/k2 draws WITH replacement; correct each
selected route by1/(k p_r). Duplicate draws may reuse a return but weights
retain multiplicity. Positive floor prevents missing-site bias. Compare exact
conditional score-space variance to existing k4 UNIFORM WITHOUT replacement.
Do not hide this sampling distinction. Gate ratio<=1 in both seeds per k.

For shared-coordinate vectors, variance is
(sum_r||g_r||²/p_r-||sum_r g_r||²)/k. Exhaustive three-site draw enumeration
checks mean/variance for k1/k2. Score-coordinate blocks are disjoint here, so
the second norm is sum_r||g_r||². Shared-parameter covariance still must be
measured before a learning nomination. No dev/test access, long fit or claim
of quality. Original producer/cache/critic work retained, extra arithmetic not
free. Unique guarded CPU queue,2GiB RSS/6GiB virtual,8GiB available floor,600s.
