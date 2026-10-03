# Current AWS work

**Queued by the user's request (3 October 06:30 UTC): native language at 90M characters.** Protocol and admission:
[AWS_NATIVE_LANGUAGE_90M.md](AWS_NATIVE_LANGUAGE_90M.md). Contracts and pilots
`queue/aws_language_90M_contracts_20261003T063000Z.txt`, then three one-job arms (p16/d8, p32/d8, p32/d8/pool4; all
compiled, near-identity init, one pass, checkpointed). Admit them into free guarded slots as the current fits finish.

The prioritized active integrated experiment is
`gym/plans/aws_capacity_exposure_20261002T072141Z/manifest.json`:
common-source-seed native H2/d8/depth8, occupied sources64, private/shared
rules, matched8 fit queries/source/pass, four passes and seeds6/7/8.
See [the capacity protocol](AWS_CAPACITY_EXPOSURE.md). Three guarded CPU slots
on this AWS host only;4GiB RSS/6GiB VMS/job and8GiB available-memory floor.
Do not start a second worker or modify its frozen sources.

As of this review,15/18 stages are complete and published; three pilot fits are
active. Completed results are95.02%/98.34% private-rule seeds6/7 and99.90%
shared-rule seed6 on development. Keep remaining cells pending. Total data is
larger than the16-source references; this is not an iso-data supremacy result.
The new capacity-summary supervisor waits for full completion, validates every
result, and publishes a common-unit quality/work inventory automatically.

Completed predecessors remain preserved:

- Native fresh-data timing and shared+common-source confirmation both passed
  their prespecified gates. The independent chain and replication summary are
  complete; do not restart them.
- Crossed rule/source-seed pilots isolate the combined effect. Common seed
  explains most quality gain; shared rules alone did not improve mean accuracy.
  See [the full factorial findings](AWS_RULE_SEED_FINDINGS_20261002.md).
- The causal two-trace timing reference completed1024/1024; it uses known
  generator constants and is a diagnostic, not a learned competitive control.
- Banknote remains11/12 final comparisons. CatBoost seed8 was stopped after
  an anomalous3128s with no completed candidates; its failure is preserved.
  Logistic has lower reserved-test NLL than our variant. Do not rerun a completed
  tag or report the full banknote gate as complete.

Other hosts own current count/statistic-retrieval, late-projection and value-
credit language comparisons. Do not duplicate their queued variants. Review
LOCAL_HANDOFF and THEORY before assigning a new architectural comparison.
Large Transformer controls remain AWS work under their original protocols and
resource requirements; completed90M controls are historical evidence, not a
request to blindly rerun old queues. No additional newly assigned AWS battery
was found in this pull. Preserve invalid-protocol quarantine and pending cells.
