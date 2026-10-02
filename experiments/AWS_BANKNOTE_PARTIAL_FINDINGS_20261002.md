# Banknote reserved-test evidence: the development lead did not replicate

The historical seed6 development screen (ours95.31% versus trees92.97%) remains
preserved. Reserved-test results from all three native fits now exist. No
settings were changed after those scores. CatBoost seed8 is incomplete after
an anomalous runtime; its failed attempt and partial state remain preserved.

| Family | Seed6 test NLL | Seed7 test NLL | Seed8 test NLL |
|---|---:|---:|---:|
| Native integrated | .142593 | .249356 | .313822 |
| Original trees | .206543 | .206543 | .206543 |
| Logistic | .090604 | .090604 | .090604 |
| CatBoost | .120693 | .107803 | pending |

All cells score the same281 reserved rows/270 feature groups with frozen
selected models. The logistic rule is deterministic across these seeds.
Repeated rows and repeated seeds are not treated as independent examples.

The originally specified crossed seed/group bootstrap, retaining98.33%
intervals for the planned three control comparisons, gives control-minus-ours
NLL differences:

- Logistic: mean−.144653, interval[−.266014,−.033775]. This completed paired
  comparison favors logistic under the protocol; the native development lead
  does not establish reserved-test quality supremacy.
- Trees: mean−.028715, interval[−.173752,.127717]. Native seed6 is better,
  seeds7/8 worse; the interval does not establish either family's NLL advantage.

Only two comparisons are analyzed here. The full three-control primary gate is
incomplete, and cannot be filled with a prediction for CatBoost seed8. Saved
partial analysis `results/diagnostics/aws_banknote_partial_analysis_20261002T014700Z.json`
contains result hashes and exact paired summaries. This is approximate
uncertainty over three training seeds and one fixed split, not broad-domain
performance. No test-directed retuning of this confirmation set is admitted.

Research implication: retain the useful sparse/temporal mechanism findings,
but do not generalize this small-data tabular variant's development score into
a quality win. Prioritize the already frozen event replications/fresh-data
confirmation and statistic-assisted integrated language direction. A new
banknote architecture would need a separately specified data protocol.
