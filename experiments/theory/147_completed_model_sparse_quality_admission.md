# From completed model quality to verified sparse quality

3 October 2026. This is a validation of the existing integrated model, with no
architecture substitution, learning change or new training. Read THEORY143–146,
the datacenter milestones and the current owner HANDOFF. The complete construction
retains computational delays, learned hard temporal races, deep private persistent
state, separate keys/values and alternative-value route credit. Silence-aware
supervision remains part of the broader substrate direction; this language scorer
does not add a silence experiment.

## What the newly completed evidence changes

The saved p64/D4/pool2 four-pass model scores **1.954701 T256 test BPC**, versus
2.183315 for its one-pass reference. That is 0.228614 BPC improvement and 14.65%
lower character perplexity for 4.00164 times estimated total fitting work.
Clipping remains at 1.0. The longer cosine schedule and more presentations both
change; this is evidence that training budget matters at depth4, not a causal
identification of clipping or deeper feature extraction.

Against the saved four-pass four-layer Transformer, the native model is
**0.046448 BPC worse**, with **8.2913 times less estimated whole-fit arithmetic**
and 7.6654 times fewer parameters. The perplexity ratio is 1.03272. Preserve this
promising raw work gap and the remaining quality gap together. Neither a
post-hoc quality tolerance nor a single seed establishes iso-quality supremacy.
The operation conventions also differ; no energy or service-cost ratio follows.

The first completed90M p32/D4/pool4 run scores **1.998416 T256 test BPC** at
107.606868 estimated TF, versus 1.954701 at 107.193761 TF for the four-pass10M
p64 model. The work differs only0.3854%, but width, pool, corpus variety,
presentations, update count and learning schedule differ. This does not isolate
an optimal capacity/data allocation. The saved larger90M controls remain better
at much greater fitting work. Current AWS pool2/depth8/width64 and curie
tied-pool/seed/horizon comparisons address these uncertainties; preserve them.

These comparisons are source-bound in
`results/diagnostics/native_completed_budget_audit_20261003T194000Z.json`.
The completed source files and their historical report evidence stay intact.
Test scores are reporting-only; further selection uses DEV and an independently
reserved confirmation protocol.

The initial193500Z metadata artifact is preserved with unchanged numbers. Its
parent-hash list omitted controls loaded directly by the report helper;194000Z
adds those bindings. The original analysis source is archived beside its record
under `archive/provenance/native_completed_budget_audit_20261003T193500Z.py`.

## Concrete implementation and deferred ladder

The final90M checkpoint is now present, unlike the absent curie four-pass weights
in this execution context. Its parent result SHA is
`67b047b3a12775d8597e2c072ba3bcd603bd1199fa3313d0f6faee8712effdfb`;
its actual final-weight SHA is
`e8b386dee21470434c2cc5179c6d95d06b5138073b2dd7ee04ce70a2e5da7901`.
The original pilot queue remains preserved and unrun. The new prepared ladder
uses this completed quality checkpoint:

`queue/trained90m_sparse_validation_20261003T192500Z/manifest.json`.

1. Run the unchanged prepacked contract driver on the completed90M weights:
   twelve synthetic plus four actual-trained FIT cases, FP32 and represented
   weights promoted to FP64, every observed winner, final state/cache, RNG,
   private ownership and mutation rejection.
2. Two bounded DEV prefixes, length8192 at T128 and T256, compare all eager and
   prepacked sparse logits before permitting full scoring.
3. Full original DEV and TEST at both saved lengths, in four separate one-job
   queues. Each requires matching passed contracts and its passed DEV prefix.

`trained_sparse_rescore.py` implements the paired scorer.
`sparse_rescore_protocol.py` provides standard-library source/provenance guards
and the exact evaluation ledger. `prepare_trained_sparse_validation.py` freezes
the queue and manifests, without importing a tensor runtime or scheduling jobs.

Admission rejects pilots, wrong/missing/duplicated cases, source/weight/argument
changes, winner disagreements, failed cache/state checks and incomplete lifecycle
coverage. All prerequisites are checked **before Torch/NumPy imports**. Existing
output paths cannot be overwritten. Evidence/source hashes are rechecked after
execution. The original model receives no update and caller RNG must be restored.

The original producer did not save a dataset-byte hash. The rescore records the
loaded split's symbol hash; matching saved quality is checked independently.
This adds reproducible rescore data identity, without retroactively asserting a
cryptographically verified identity for the historical scorer's dataset file.

Full scoring retains the original seed314159, the saved FP32 values, original
lane grouping, reset-per-window state, and the producer's exact exclusive-stop
window range. First window scores all positions; following windows score their
second half. T128 and T256 each score999936 targets from a1M-character split,
excluding target index0 and preserving the original tail omissions.

All full-split paired logits must be finite and within the existing FP32
atol/rtol1e-4 contract. Both paired BPC values must agree within1e-5; on a full
split each must also reproduce the saved compiled-producer BPC within1e-5.
Failure is retained and does not admit sparse quality. A prefix passes only its
bounded parity test, without being called a completed full-split quality result.
The full-split scorer compares every logit; it does not expose every intermediate
winner or state. Those contracts apply to the prerequisite FIT trajectories.

For exact real arithmetic, cross-entropy
`ell_y(z)=log(sum_i exp(z_i))-z_y` satisfies
`|ell_y(z+delta)-ell_y(z)| <= 2*||delta||_infinity`:
log-sum-exp shifts by at most the largest absolute perturbation, as does the
target logit. Thus an absolute logit perturbation bounded by epsilon bounds the
mean BPC change by `2*epsilon/log(2)`. This does not certify route equality or
floating-point reductions. The implementation therefore checks actual logit
agreement, score agreement and saved quality separately.

## Work boundaries and host admission

Each full T128 split evaluates1999744 positions **per backend**; T256 evaluates
1999616. The two-backend check pays3999488 or3999232 positions respectively,
rather than calling that999936 inference positions. Four full stages together
pay15997440 positions; the two prefix stages add64000. The original sixteen-case
contract adds3540 padded positions. Planned total16064980 positions,2048 forward
calls, **zero backwards/optimizer updates**. Actual completed calls/positions
are recorded separately if a stage fails early.

The worker charges one private model snapshot and one retained stack, plus
per-call key scoring, selected-map gathers, state/cache, input/readout and
metadata checks. Tensor payload ledgers are not measured DRAM traffic. Paired
diagnostic wall time includes both backends and verification; it is not an
isolated serving speed comparison. No new FLOP, energy or hardware result is
filled from this prepared ladder.

All seven queues are **PREPARED UNRUN**. No Torch/NumPy/model job is executed in
this Docker context, whose lock/process view does not cover the physical curie
trainer. No coordinator, waiter, background task or AWS allocation change was
started. After a genuine physical reservation, each numerical stage must use
`queue/run_safe.sh`, one CPU thread, enabled RSS watchdog and8192MiB available
memory floor. Prepared caps: VMS3000000KiB, groupRSS1250000KiB;420s contracts,
600s prefixes, conservative21600s full-stage ceiling. Recheck actual occupancy,
memory and prefix workload before full admission. Changed caps/settings/sources
require a new immutable tag rather than editing a successful job.

The stdlib check passes100 window-boundary enumerations,18 invalid-contract
rejections, and refusal of the actual ladder's missing native admission before
runtime imports. Native parity, full sparse quality, runtime speed and energy
all remain pending. The next research priority remains the successful integrated
route-credit model on the existing owner queues; this validation is deferred
until physical admission and does not displace those chains.
