# Gesture temporal-resolution diagnostic

Hyperparameters selected by fitting-user GroupKFold NLL; no development selection or official-test access.

|Bins|Dimensions|Dev accuracy|Dev NLL|Stored bytes|Sequential ms/query|Coarse quality gate|
|---|---:|---:|---:|---:|---:|---|
|1|32|68.750%|0.806788|352440|0.116785|FAIL|
|4|128|77.604%|0.686661|993432|0.147519|PASS|
|20|640|74.479%|0.707992|4880984|0.388310|reference|

All984 fitting gestures/192 held-out-user development gestures;27 fold fits+3 full refits paid, identical learner family. Every candidate/fold and final probability saved. CPU one-thread3 deterministic inference repeats, fit-only log-count scaling and coalescing included. Whole-fitting/inference solver FLOPs unknown for every row, not zero. Storage is serialized model size, not live RSS or traffic. Development comparison is exploratory, not new confirmation or evidence that the native core computes this representation.

Whole research job wall 6.705s; peak RSS 222000KiB. Four bins suggest useful coarse information with lower representation dimension; one bin loses useful temporal information in this fitted family. Finer input representation contains every coarsening, so this is an estimation/optimization observation rather than an information-theoretic reversal.
