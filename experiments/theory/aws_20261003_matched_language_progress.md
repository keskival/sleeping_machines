# Matched depth-eight language progress: online evidence only

All three integrated streaming fits have immutable250k/500k checkpoints.
Read-only analysis verifies checkpoint hashes, IDENTICAL10M fitting-data hash,
matching prefix lengths and matching Adam update counts. It reads saved cumulative
factual losses and optimizer tensors only; no model execution, new FIT/DEV/test
reads or optimizer steps. Script: analysis/aws_language_matched_progress.py.
Result: results/diagnostics/aws_language_matched_progress_20261003T083000Z.json.

| Arm | Prefix targets | Adam updates | Cumulative online BPC | Online BPC on latest interval |
|---|---:|---:|---:|---:|
| Private corrected full replay | 253952 | 992 | 3.309384 | 3.309384 |
| Private teacher | 253952 | 992 | 3.320448 | 3.320448 |
| Depth-shared teacher | 253952 | 992 | 3.331464 | 3.331464 |
| Private corrected full replay | 503808 | 1968 | 3.131119 | 2.949932 |
| Private teacher | 503808 | 1968 | 3.154944 | 2.986728 |
| Depth-shared teacher | 503808 | 1968 | 3.171832 | 3.009584 |

The second interval is targets[253952:503808],249856additional targets and976
additional Adam updates. Compute interval BPC from the DIFFERENCE of cumulative
factual-loss sums divided by interval targets and log(2); do not subtract BPCs.
Replay's second-interval loss is.036796bpc below the private teacher and.059652
below the shared teacher. Preserve this supported positive indication rather
than dismissing it because it is partial. Its scope is precise: predictions
made DURING fitting, with parameters changing throughout the interval, one seed.
It is not a completed quality score, heldout gain, useful-depth lesion result,
iso-FLOP comparison or proof of generalization. RNG pairing and identical
initialization across the original/fast implementations are not asserted.
No pending10M result cell is filled from this table.

All six snapshots have ZERO coordinates with bias-corrected sqrt(v)<=Adam
1e-8 epsilon among the saved unique gate/output parameters. This excludes that
particular stored-second-moment floor here; it does not measure fresh gradients,
input-dependent gate activation, route coverage or causal contribution of depth.

Full replay remains much more expensive than the teacher despite winner-reuse
savings. This online lead is a learning diagnostic, not resource supremacy.
Completed final1MDEV with common quality/work accounting remains the deciding
comparison. Active fits are unchanged. Future90M configurations await the
owner's413v4 DEV evidence and revised assignment; those segment-reset models
and their credit repairs have a distinct protocol from this full replay fit.
