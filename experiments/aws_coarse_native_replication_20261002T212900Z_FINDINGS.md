# Completed native temporal-resolution screen

|Variant|Seed|Accuracy|NLL|Whole-fit GFLOPs|Fit MFLOPs/target|Inference MFLOPs/target|Gate|
|---|---:|---:|---:|---:|---:|---:|---|
|fine|7|49.479%|1.434727|2.285696|2.232125|0.591695|reference|
|coarse_fastclock|7|53.125%|1.323325|0.548517|0.535661|0.137993|PASS|
|coarse_matchedclock|7|55.208%|1.220806|0.548517|0.535661|0.137965|PASS|

Same256 distinct fitting gestures,4passes/1024 fitting targets,192 development,p16/L2/H2/pool2,8 available receivers,4 selected updates/event. Coarse arms receive four250ms releases plus1s query; fine receives twenty50ms releases plus1s query. Actual scored keys and updates remain fully charged. Fine within-quarter timing is discarded. Old984-fit native/control references remain independent and stronger controls retain77.604%/.686661 at984fit. No fair quality claim from unequal data.

Gate: >=.02 NLL improvement, <=1pp accuracy decline, <=.50 fitting-work ratio. All arms reported with whole-fit/per-target common units; NumPy preprocessing FLOPs/traffic and energy unknown. Full wall includes input coalescing/normalization and evaluations. Development selection over fixed4passes is exploratory. Passing arms nominate unchanged seed7 matched replication; failed arms do not escalate automatically.
