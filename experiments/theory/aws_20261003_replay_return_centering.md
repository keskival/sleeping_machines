# Return centering: derivation and precision localization

The concrete failure is the preserved production float32 coordinate gate in
notes 116–117, not a demonstrated learning failure caused by rounding. The
integrated depth-eight private/shared models retain temporal races, addressed
persistent state, separate keys/values and full conditional losing-route credit.
The existing AWS 10M-character runs remain unchanged.

For detached route returns Q and pi = softmax(s), score credit is
  d(sum_i pi_i Q_i)/ds_j = pi_j (Q_j - sum_i pi_i Q_i).
Subtracting any common detached baseline B gives exactly the same gradient:
  sum_i pi_i (Q_i - B) = sum_i pi_i Q_i - B,
because sum_i pi_i = 1. A convenient available baseline is the factual winner's
suffix return. It requires no new replay, sampling or candidate discovery.
Baseline and all counterfactual returns MUST be detached: otherwise the
additional path changes factual/value credit. The scalar surrogate value shifts;
reported factual cross entropy must remain separate and unchanged.

An illustrative IEEE float32 witness uses already represented returns
Q=[10000,10000.0009765625], pi=[.5,.5]. Separately rounded products and sum can
round the common mean to 10000, yielding score gradient [0,.00048828125].
Centering first yields [-.000244140625,.000244140625], the exact gradient for
these represented returns. This is an arithmetic illustration, not a measured
PyTorch kernel result: contraction/fusion order can differ. Centering cannot
recover information already lost when forming Q, or eliminate rounding in the
model's contractions. It is not evidence that precision explains underfitting.

## Completed secondary analysis

`experiments/analysis/aws_replay_precision_localization.py` reads the saved
117 gradient vectors only: no model execution, backward or optimizer. Result:
`experiments/results/diagnostics/aws_replay_precision_localization_20261003T022200Z.json`.
The unchanged per-coordinate gate is |error| <= 3e-6 + 3e-4 |reference|,
comparing each float32 program against its own same-represented double program.

| Family | Estimator | Failed coordinates | Channel-mix share of squared gradient error |
|---|---|---:|---:|
| Private | Original full replay | 32 | 67.36% |
| Private | Winner reuse | 33 | 67.73% |
| Depth-shared | Original full replay | 75 | 79.46% |
| Depth-shared | Winner reuse | 101 | 79.63% |

These are not old/new mismatch counts. Large error share in channel-mix
parameters does not isolate a cause: route credit propagates into them, and
ordinary matrix arithmetic also affects them. Full per-parameter attribution
is saved; the analysis neither establishes a precision repair nor a quality
benefit. The independent completed ~49.66% full-fitting-work reduction remains
valid with its numerical qualifications and no language-quality supremacy claim.

## Required comparison before adoption

Use a separate helper and immutable uniquely tagged guarded diagnostic queue;
do not modify currently frozen training sources. Compare raw and centered
surrogates for original/full and winner-reuse replay in float32 and double,
using precisely the same represented weights, nonempty state, inputs and RNG.
Check EVERY parameter gradient against the corresponding double reference,
retain the original thresholds, and record failures rather than relaxing them.
Assert identical factual logits/state/end RNG and all factual/shadow route
histories. Preserve all alternative returns, lanes and events. Measure the extra
subtraction/optimizer work rather than calling it free. If the gradient contracts
pass, test actual traced/untraced accumulation and bitwise interruption recovery
before considering a separately named quality arm of at least 10M characters.
No centered quality run is admitted by this note.

All three AWS guarded CPU slots are occupied by the prioritized 10M matrix.
The proposed diagnostic waits for a safe slot; this analysis does not bypass
host reservations or interrupt healthy fits to obtain a small contract result.
