# Completed full-data native coarse comparison

Same984 fitting gestures,8passes/7872 fitting targets,192 development,U16,Adam.003,clip1,p16/L2/H2/pool2. All seed rows/fixed gates below; no best-seed or ensemble selection.

|Variant|Seed|Accuracy|NLL|Whole-fit GFLOPs|Fit MFLOPs/target|Inference MFLOPs/target|Gate|
|---|---:|---:|---:|---:|---:|---:|---|
|fine|6|57.812%|1.179750|17.573520|2.232409|0.591779|reference|
|coarse_fastclock|6|61.458%|1.071827|4.218015|0.535825|0.138035|PASS|
|coarse_matchedclock|6|66.667%|0.890983|4.218015|0.535825|0.138119|PASS|
|fine|7|55.729%|1.172640|17.573520|2.232409|0.591737|reference|
|coarse_fastclock|7|70.312%|0.933333|4.218015|0.535825|0.137923|PASS|
|coarse_matchedclock|7|61.979%|1.056934|4.218015|0.535825|0.138007|PASS|
|fine|8|68.750%|0.898591|17.573520|2.232409|0.591751|reference|
|coarse_fastclock|8|60.417%|1.028562|4.218015|0.535825|0.138007|FAIL|
|coarse_matchedclock|8|69.271%|0.920744|4.218015|0.535825|0.138035|FAIL|

Strong fitting-user-selected RBF controls on the SAME984fit/192dev data:
- 1bins: 68.750%,NLL 0.806788; serialized 352440bytes, sequential 0.116785ms/query. Solver fitting/inference FLOPs unknown.
- 4bins: 77.604%,NLL 0.686661; serialized 993432bytes, sequential 0.147519ms/query. Solver fitting/inference FLOPs unknown.
- 20bins: 74.479%,NLL 0.707992; serialized 4880984bytes, sequential 0.388310ms/query. Solver fitting/inference FLOPs unknown.

Full-data developmental architecture comparison. All eight passes/7872 fit targets and per-seed curves retained. Native arithmetic+unit-special estimates, preprocessing wall included but NumPy FLOPs/traffic/energy unmeasured. No official-test/fresh-confirmation or broad supremacy claim.

Packets:4x250ms+1squery or20x50ms+1squery, all original raw counts retained, fine within-quarter timing discarded.8 available receivers and4 selected writes per observed event. No extra dense carrier/decoder or teacher substitution. Fixed per-seed advancement gate: >=.02NLL improvement, <=1pp accuracy decline, <=.50 native-fit work. Diagnostic intervals and all actual activity/wall/RSS remain in raw analysis. Successful screens do not erase a failed full-data seed. No automatic extension after a failure.
