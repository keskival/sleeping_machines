# AWS independent replay-critic variance screen

User requests unqueued corrected replay variants without known law errors.
Other host owns corrected k4/depth fits and all-race shadow lanes. This screen
does not duplicate them or bypass their quality gate. Read theory92 and the
current replay plan: preserve factual first time, actual alternative writes,
factorized common-clock credit, and a critic frozen before correction sampling.
Sixteen guarded factorized/replay/critic/lane/core tests pass before admission.

Fixed initial fine-packet native p16/depth2/heads2/pool2 encoders, seeds7/8.
First four FIT examples supply all legal replay targets, next four FIT examples
are held out from critic training. No producer update or development quality
selection. Train width32 GELU critic for100 Adam steps at.003. Freeze before
evaluating k1/k2 correction against k4 zero-critic baseline. No post-hoc tuning.

For site-score gradient g_r and critic prediction q_r, the concatenated
score-coordinate estimator is q_r+(R/k)I[r selected](g_r-q_r). Exact conditional
squared-error expectation is (R/k-1) sum_r ||g_r-q_r||². Exhaustive R4 subset
contracts verify mean and variance for k1/2/4. Gates: heldout MSE no greater
than baseline k4 in BOTH seeds, separately for k1 and k2. This is nomination
for further variance work, never a prediction-quality or supremacy gate.
Shared parameter gradients require cross-site covariance and are not inferred
from this disjoint score-coordinate result. Critic R² and every replay are
retained; diagnostic total FLOPs remain unknown, not zero. Hardware16CPU/31GiB,
one guarded job,2GiB RSS cap,6GiB address space,8GiB available floor,600s timeout.

Second separately named scope: apply the identical fixed critic screen to saved
full984/eight-pass coarse4/.25-clock encoders seeds7/8. These encoders have seen
all FIT labels, so only critic supervision is held out; this is conditional
variance evidence, not producer-cross-fitted generalization. Keep initial-screen
failure intact and do not retune critic width, updates, learning rate or gate.
