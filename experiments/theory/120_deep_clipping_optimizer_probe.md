# Deep fitting: clipping mechanism rather than a raw-norm inference

The user requests a systematic, divided diagnosis of deeper models. Independent
audits find no double normalization or hidden scheduler in the DVS path:
episode losses are summed, divided once by actual window size, clipped once,
then plain Adam at the configured fixed learning rate is applied.

The completed fine-packet D4 clip4 controls contradict prediction P408 on its
available FIT32 diagnostic: seed7 .718192 -> .945086, seed8 .587894 -> .711417.
DEV results are mixed. These are first32 FIT examples at DEV-selected weights,
not the complete training objective or a common final epoch. Preserve the old
claim and numbers, but amend its interpretation in a separate synthesis.

For g_t and k_t=min(1,c/(||g_t||+1e-6)), Adam uses
m_t=beta1*m_(t-1)+(1-beta1)*k_t*g_t and
v_t=beta2*v_(t-1)+(1-beta2)*k_t^2*g_t^2.
A positive constant scaling of the ENTIRE history cancels in m/sqrt(v) up to
epsilon. Fresh Adam's coordinate displacement is -eta*g/(|g|+epsilon/k).
Thus a threefold raw norm does not establish a threefold smaller update.
Changing only the present gradient against stored moments does not cancel;
variable k_t changes time/sample weighting and can change direction. This
does not eliminate clipping as a contributor; it specifies what must be tested.

New bounded probe uses actual paired online_model+optimizer snapshots, never
best_state paired with final moments. Available checkpoints are D2 fine20,
seed6/eight-pass original native teacher, and D4 coarse4 seed7/four-pass
factorized pilot. Their data/exposure/credit differ: comparisons BETWEEN them
cannot identify depth effects. Within each checkpoint, three prespecified
FIT windows 0..15,16..31,32..47 and separate FIT anchor96..111 are fixed.
No DEV/test is consulted or step selected. The saved completed epoch fixes
the original training-noise seed; the anchor uses the same seed.

Replay each actual driver gradient once while intercepting its single clip
and optimizer step. Retain normalized raw gradients. Fork identical weights,
raw gradient and actual Adam moments under cap1/cap4/no cap and learning-rate
multiplier1/.25. Measure pre/postclip norm, k, per-layer/block gradient and
actual parameter displacement, gradient/step inner product, finite SAME-FIT
and independent-anchor loss/KL. Sparse hard route changes make finite loss
changes a diagnostic, not a guarantee that a biased local teacher is exact.
All forks are discarded. Save vectors, original checkpoint hashes and cursor.

Also use each protocol's freshly initialized model on fixed FIT0..15 to test
the fresh-step formula with the actual deep gradient. These two fresh models
are not a matched-depth benchmark. A small deterministic double Adam contract
tests constant HISTORY scaling. Exact model and moment restoration, gradient
normalization/clip interception, finite outputs and caller-RNG preservation
are required. Do not infer trained clipping frequency from three frozen
windows. Total probe FLOPs/traffic/energy unknown, not zero.

The mathematical question is whether epsilon/history-weighting changes actual
updates; the empirical question remains fitting plus heldout performance.
Completed clip controls already answer the simple 'loosen cap' proposal.
Do not duplicate the separately owned pending lr.006/D2 clip4 runs. No
architectural substitution: time computation, races, keys/values, persistent
private updates and each saved model's original teacher are retained.
CPU-only unique one-job run_safe queue, one thread, 3GB virtual/1.25GB RSS,
at least8GiB available and180s timeout, before any longer experiment.

## Pre-admission corrections from failed attempts

The first fresh-step check used a universal2e-7 subtraction tolerance and failed:
packet-scaled frequency weights can exceed10, so their float32 storage ULP is
larger. The formula check now uses a coordinate storage/arithmetic bound of
4*machine_epsilon*max(1,|weight|). Failed queue/log remain; no successful gate
was revised. The second attempt finds an old batched-kernel hash; exactbytes
from155fbca^ are archived under8f93f5... and explicitly loaded for the D4
checkpoint. The third/fourth attempts discover a coarse preprocessing hash
gap: originala0496fe9... versus recomputed3fefa180... . All other data/source
metadata and original count-artifact hash match, but this does NOT prove
identical transformed input bytes. The full-gradient probe therefore records
the old/new metadata and labels D4 as a CONTROLLED CURRENT-FIT-input fork
at genuine saved weights/moments, not exact historical input replay. Only
that explicit known hash pair is admitted; other changes still refuse.
This is a separately stated numerical mechanism test, not a replacement
quality protocol or evidence that the transform mismatch caused old failures.
Exact historical preprocessing remains an artifact gap. The D2 transform
metadata reproduces exactly. No additional DEV or test selection introduced.
