# Completed AWS joint-event controls

Current v2 table accuracy 58.398438%; fixed calibration threshold 78.398438%.

|Control|Seed|Accuracy|NLL|Whole-fit GFLOPs|Fit MFLOPs/query|Inference MFLOPs/query|Calibration|
|---|---:|---:|---:|---:|---:|---:|---|
|gru|6|49.707031%|0.693136|12.404265|3.028385|1.020051|FAIL|
|gru|7|50.781250%|0.692962|12.401601|3.027735|1.020051|FAIL|
|gru|8|50.878906%|0.693016|12.401505|3.027711|1.020051|FAIL|
|transformer|6|50.097656%|0.693282|48.362315|11.807206|4.219196|FAIL|
|transformer|7|50.000000%|0.693550|48.497936|11.840316|4.219196|FAIL|
|transformer|8|50.000000%|0.695585|48.496172|11.839886|4.219196|FAIL|

Fixed base calibration. Unit-special convention; sampled-window fitting estimates, not energy. No history ladder admitted automatically.
Controls remain diagnostic; no native or overall supremacy claim. All three seeds reported, no best-seed selection. Failed prerequisites remain archived in preceding plans. Selected predictors saved; no confirmation tuning.
