# Conditional magnitude priorities and their oracle ceiling

Training-only ordinal priorities fail k2/plaink4 gates. A read-only cached
calculation shows the exact score-space oracle may permit reduction: derive
p_r proportional||g_r|| using constrained minimization of sum||g_r||²/p_r.
Its minimized k2 variance is ((sum||g_r||)²-sum||g_r||²)/2. This oracle needs
every expensive return; no operational advantage is claimed from that number.

Distinct feature-conditioned hypothesis: predict log route magnitude from the
existing signed/label-aware per-race features (both candidates concatenated).
Train first32 FIT rows on saved corrected utilities; fixed ExtraTrees64trees,
maxdepth6,minleaf8,seed681. No hyperparameter selection. Target floor fixed from
TRAIN median magnitude*.001,at least1e-12. Predict exp minusfloor,positiveclamp;
90% normalizedprediction plus10%uniform. Freeze before next32 FIT rows in both
seeds7/8. k2 WITH replacement, correct multiplicity by1/(2p). Gate conditional
score-MSE<=uniformWITHOUTreplacementk4 in both seeds. Report uniformreplacement
control and diagnostic oracle with all costs/scope. Parameter covariance and
quality still require integrated evidence; no long fitting admission here.

No producer/core change. Temporal races, sparse actual writes, key/value
separation and first-time law preserved. Proposal labels are learning-only;
critic/subset dependency is absent because predictor is frozen. Existing cached
producer/replay/critic costs retained; new tree fit/evaluation additionally paid,
FLOPs unknown. Unique guardedCPUqueue,2GiB RSS/6GiB virtual,8GiBfloor,600s.
