# Fixed native encoder affine/quadratic optimization study

Both frozen nomination gates FAIL; numerical folding/native-state checks remain valid. All12 encoder/readout/seed rows are retained.

|Encoder|Seed|Degree|Accuracy|NLL|Native inference MFLOPs/query|Encoder-fit GFLOPs|Combined fit FLOPs|
|---|---:|---:|---:|---:|---:|---:|---|
|initial|6|1|57.812%|1.162621|not measured|0.000000|unknown|
|initial|6|2|64.062%|1.065509|not measured|0.000000|unknown|
|fitted|6|1|68.750%|0.835175|0.138119|4.218015|unknown|
|fitted|6|2|68.750%|0.870350|0.201534|4.218015|unknown|
|initial|7|1|56.771%|1.154207|not measured|0.000000|unknown|
|initial|7|2|59.896%|0.994128|not measured|0.000000|unknown|
|fitted|7|1|66.667%|0.939479|0.138007|4.218015|unknown|
|fitted|7|2|69.792%|0.899185|0.201422|4.218015|unknown|
|initial|8|1|60.938%|1.097227|not measured|0.000000|unknown|
|initial|8|2|64.583%|0.976572|not measured|0.000000|unknown|
|fitted|8|1|68.750%|0.936226|0.138035|4.218015|unknown|
|fitted|8|2|71.354%|0.885890|0.20145|4.218015|unknown|

Fitted affine means68.056%/.903627; fitted quadratic69.965%/.885142. Quadratic improves over refit affine by only.018485NLL/+1.910pp mean and worsens seed6 NLL by.035176, so it misses.05/+3pp/everyseed gate. Quadratic-native NLL gains .020633/.157749/.034854 miss everyseed>=.05 gate despite retained learned-encoder mean benefit. Initial affine58.507%/1.138019 and quadratic62.847%/1.012070 controls remain visible.

Same984 fit/192 development, C(.01,.1,1) selected on fitting-user GroupKFold NLL,108CV/12fullrefits,6feature replays. Encoder labels already learned on all fit targets; these are conditional decoder folds, not cross-fitted encoder evaluation. Every predictor uses the unchanged coarse native prefix and a folded local affine/quadratic head, no resident memory reader or dense prefix carrier. All5 sequential head calls and conversions are charged. Native inference ledgers/3repeat sequential walls saved. Combined fitting FLOPs UNKNOWN because solver FLOPs are unknown; parent4.218015GFLOPs alone is not combined fitting work. Tensor bytes exclude shared fitted packet-normalization arrays and serialization metadata; portable heads reference the committed data artifact/config.

Standalone original encoder/head states and frozen fit/development features are saved with checksum lineage, allowing later diagnostics without unpaid repeated replay. Neither failed gate admits unchanged scaling or architecture promotion. RBF context information remains stronger than this fitted polynomial class; conventional controls remain77.604%/.686661 (coarse SVC) and66.667%/.902951 (97476-byte historical compact prototype).

Whole study wall 126.813s; peakRSS 608044KiB. Solver warnings: 1. Model selection, preprocessing, cloning, evaluation and all exports included in job wall; traffic/energy unmeasured.
