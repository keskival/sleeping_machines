# Completed exact prefix-replay saving

Fitting work ratio 0.892624; 10.738% lower fitting work with identical entire curves, predictions, model weights, Adam and recovery state.

|Implementation|Accuracy|NLL|Whole fit GFLOPs|Fit MFLOPs/target|Inference MFLOPs/target|Wall seconds|Peak RSS KiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|full|25.000%|2.311338|0.143986|2.999703|0.591695|13.609|459616|
|reuse|25.000%|2.311338|0.128525|2.677607|0.591695|12.653|459672|

Same24 real fitting gestures/two passes (48 targets),8 development, p16/L2/H2/pool2. Selected updates and scored keys below include alternate simulation; hard inference remains unchanged. Source/data hashes and executed stage ledgers accompany raw results. Wall time is one concurrent-host observation; the saving claim is charged arithmetic equivalence, not a general latency/energy claim. Snapshot/copy memory traffic is not measured. Initial small contract fits and failed/research work are distinct from these per-fit totals.

Scope: improving the integrated actual-write learning implementation over its full-replay reference, with exact empirical quality/update retention. No claim over strongest gesture controls. Other hosts own substantive credit replication and capacity fitting; this optimization can support them without altering the estimator.
