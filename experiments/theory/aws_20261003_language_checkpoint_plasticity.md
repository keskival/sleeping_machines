# Deep language plasticity: immutable 250k checkpoint analysis

The first publication milestones for both original depth-eight teacher controls
are now committed and pushed: private56e6939 and sharedbe9d535. Both snapshots
contain253952targets,992Adamupdates, full private state, RNG, model weights,
optimizer moments and frozen source/data protocol. These are partial learning
artifacts, not completed10M heldout results.

Other-host theory123 measures extreme growth initialization at gate bias-20:
most nonlinear candidate contributions round away and clipped gate/output
updates are epsilon-suppressed. This motivates checking the ACTUAL streaming
language snapshots rather than assuming a similar bottleneck across models.

Read-only script `experiments/analysis/aws_language_checkpoint_plasticity.py`
verifies checkpoint SHA256 against committed metadata, exact saved cursor,
optimizer update count and moment shapes. Frozen model state dictionaries have
parameters only, no registered buffers. Registration-order traversal with storage
alias deduplication reconstructs the unique Adam parameter list;384private and
174shared match. Shared map aliases are explicitly retained in the result.
No forward/backward, optimizer step, FIT/DEV token loading or new training occurs.

| Family | Unique gate/output tensors | Coordinates | sqrt(bias-corrected v) <= Adam epsilon | Bias-only sigmoid range |
|---|---:|---:|---:|---:|
| Private teacher | 96 | 16896 | 0 | .39544–.62027 |
| Depth-shared teacher | 6 | 1056 | 0 | .41115–.54896 |

Adam epsilon is1e-8. We report the saved denominator retention
sqrt(v_hat)/(sqrt(v_hat)+epsilon), bias-corrected moment direction
m_hat/(sqrt(v_hat)+epsilon), and per-parameter distributions in
`experiments/results/diagnostics/aws_language_checkpoint_plasticity_20261003T032000Z.json`.
No moments are absent in these selected gate/output tensors. The stored
moment direction is a description of saved optimizer state, NOT a simulated
next step, actual functional movement or a fresh gradient. Bias-only sigmoid
is NOT the actual input-dependent gate sigmoid(W*features+b).

Scope: at this milestone these teachers do not exhibit the particular extreme
bias closure or second-moment epsilon floor observed in the growth probe.
Historical nonzero moments do not prove every current route is learning, that
all depth is useful, or that the model is adequately fitting/generalizing.
The replay model has not yet reached its first250k milestone; do not compare
its unequal-exposure online score with these checkpoints. Await its immutable
snapshot for the same read-only diagnostic. No active protocol change follows.

The next numerical contract remains detached-return-centering, queued behind
the occupied host lock. New upstream causal-prefix cache contracts preserve
arithmetic savings but measure WORSE CPU wall; do not replace the current
winner-reuse implementation with that cache merely because FLOPs decrease.
Segment-batched initialization/growth experiments remain complementary and
have different state-reset/credit protocols, not matched streaming controls.
