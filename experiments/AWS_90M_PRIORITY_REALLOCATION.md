# Proposed checkpoint-preserving 90M priority allocation

The existing immutable matrix holds a global host reservation across six 10M
fits. Its three current fits alone have about 61/61/93 training hours remaining
at checkpoint-measured throughput (observations on 3 October 2026), excluding
final development evaluation. Additional 10M fits extend that delay. A 90M
queue waiting for this reservation cannot use a free slot until the original
coordinator finishes. The current 90M waiter is safe, but admission is days away.

The read-only forecast is saved in
`results/diagnostics/aws_language_queue_forecast_20261003T133000Z.json`.
These are operational projections, not training outcomes or pending quality.

## Proposed order

1. Preserve all completed results, milestone archives and live source-exact
   model, Adam, cursor, chronological RNG and private-state checkpoints. Record
   hashes, targets and guard commands before any interruption.
2. At completed optimizer-window snapshots, close the original admission
   coordinator and guards cleanly. Preserve their lifecycle and logs. Account
   for any discarded uncheckpointed work as unknown and nonzero, bounded by the
   snapshot cadence: up to 4,095 targets per current streaming fit.
3. Acquire the normal host reservation through a new immutable scheduler with
   at most three CPU slots and an 8 GiB available-memory floor. Never admit a
   new job while an old guard owns its reservation or slot. Each contract,
   throughput pilot and fit uses a unique one-job queue through run_safe.sh.
4. Prioritize the assigned payload-32/depth-4/pool-4 model with linear route
   credit at 90M. Retain private corrected replay and its private teacher
   control at 10M through exact-source recovery. During compilation, admit only
   workloads satisfying measured capacity and combined RSS reservations.
5. As the 90M slot frees, admit remaining assigned 90M arms, then recover the
   shared teacher and pending 10M controls. Preserve all earlier evidence. Do
   not change model settings, data protocol or estimator within a resumed tag.
6. Publish each completed fit and pilot with its JSON, final weights, recovery
   checkpoints, hardware, elapsed time and logs. Retain consistent work units,
   unequal-pass qualifications and the assigned development-selection rule.

This is an allocation proposal, not an executed interruption or a completed
replacement scheduler. The three original fits remain healthy and running.
The user's scheduling preference is pending; independent analysis continues.
The proposed choice changes allocation and order, not scientific protocols.
