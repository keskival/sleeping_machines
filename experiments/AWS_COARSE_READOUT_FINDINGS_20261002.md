# Frozen coarse native readout probes

|Encoder|Seed|Readout evidence|Accuracy|NLL|Probe bytes|
|---|---:|---|---:|---:|---:|
|initial|6|context|61.979%|1.069564|348060|
|initial|6|resident|68.750%|0.882296|1336941|
|fitted|6|context|67.708%|0.834528|279662|
|fitted|6|resident|69.792%|0.809460|1195535|
|initial|7|context|63.542%|0.979591|357724|
|initial|7|resident|69.271%|0.880663|1352669|
|fitted|7|context|67.708%|0.893706|286814|
|fitted|7|resident|72.396%|0.887607|1178239|
|initial|8|context|59.375%|1.058901|365708|
|initial|8|resident|68.229%|0.896475|1379389|
|fitted|8|context|71.875%|0.841134|260350|
|fitted|8|resident|76.042%|0.790254|1151519|

Context nomination gate passes: meanNLL improvement .099764 and meanaccuracy +3.125pp over native, everyseedNLL positive. Resident-over-context gate fails: meanNLL gain .027349<.05 despite meanaccuracy +3.646pp. Initial reservoir controls retained: fitted context mean69.097%/.856456 vsinitial61.632%/1.036019; fitted resident72.743%/.829107 vsinitial68.75%/.886478.

These are frozen RBF probes, not installed native heads or sparse learned retrieval. Full resident reads128memoryvalues+8ages+8seenflags in addition to32contextvalues. Fit-selected SVC controls see984 fit/192 reused development; solver FLOPs unknown, all72 internal fold fits+12final fits and6wholeprefix replays retained. Fitted encoders pay their earlier4.218015GFLOPs each; reuse across probe arms is explicit rather than free or counted twice. Matching NumPy/Torch state/winners and saved-development probabilities pass; no official-test access. Internal hyperparameter folds condition on a fitted encoder that used all fit labels; not cross-fitted encoder estimates.

Whole probe job 53.605s/516444KiB. Useful information is readable from trained context; this is not proof that a cheap polynomial head will learn it end-to-end. A zero-nested degree2 query-head comparison is the next integrated hypothesis, after numerical/learning/accounting prerequisites.
