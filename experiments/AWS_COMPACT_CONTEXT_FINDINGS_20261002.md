# Compact current-context head: all contracts pass, practical gates fail

Fixed33-anchor local RBF readout on six frozen native contexts (three fitted/
initial seed pairs), with mandatory raw4/raw20 prototype controls.33 distances
and all basis outputs are used: this is dense local readout, not sparse race
attention. Native clocks/keys/value proposals/selected memory writes unchanged.

| Predictor | Accuracy % | Dev NLL | Whole fit GF | Per-target fit MF | Standalone bytes | Seq ms/prefix |
| --- | --- | --- | --- | --- | --- | --- |
| Initial native6 | 58.854 | 1.268551 | Unknown | Unknown | 114506 | 1.826 |
| Fitted native6 | 67.708 | .901193 | Unknown | Unknown | 115106 | 1.843 |
| Initial native7 | 62.500 | 1.145877 | Unknown | Unknown | 114554 | 1.832 |
| Fitted native7 | 64.583 | .938003 | Unknown | Unknown | 115078 | 1.831 |
| Initial native8 | 55.729 | 1.170969 | Unknown | Unknown | 114554 | 1.839 |
| Fitted native8 | 68.750 | .915083 | Unknown | Unknown | 115054 | 1.853 |
| Raw4 prototype | 65.625 | .888167 | Unknown | Unknown | 40966 | .01545 |
| Raw20 prototype | 66.146 | .902387 | Unknown | Unknown | 184334 | .02820 |

All984 FIT/192 DEV, no test. Native producer has8passes/7872 presentations,
head sees984 examples plus fitting-user3-fold C(.1,1,10) selection. Native
head folds share label-trained encoder (not producer cross-fitting). Raw heads
use984 and same folds/basis/decoder family. Whole/per-target COMBINED fit work
unknown in EVERY row because KMeans/logistic solver work unmeasured. Fitted
native producer4.218015GF and.535825MF/target remain charged; initial/raw producer
fit0,not total fit0. Cached feature replay and original extraction retained.
Total inference FLOPs unknown; decoder-only operation estimate explicitly scoped
in ledger and cannot be juxtaposed with full native costs.33 anchor/all-value
lookup is charged, not sparse or dormant capacity proof.

Staged nomination FAIL: per-seed native NLL gains−.010209/.118931/.005661 miss
everyseed>=.05. Learned-versus-initial mean gain.277040 passes its isolated
condition; retain this positive representation evidence. Practical storage
nomination FAIL: native/raw4 serialized ratio~2.81, versus raw20~.624. Native
matches raw20 quality margins in seeds6/8 but not7. Isolated fine-control byte
savings are not practical advantage over the coarse control. Strong raw4 SVC
77.604%/.686661 and earlier native leading results remain, not overwritten.

Before optimizer fitting:48 state/query/target/repeated-execution comparisons
and six original Torch/core reference contracts pass. Demand-only decoder removes
interim logits that have no state feedback; final aligned query vector, every
winner/memory/arrival/context/noise/wait counter identical. All192 portable
predictions perarm match cached basis inference; export roundtrips and3 sequential
repeats are deterministic. Normalization, original core maps, cached rates/noise,
anchors, decoder and metadata included in uncompressed standalone exports.
Array-only bytes75312native/38832raw4/182192raw20 also reported separately;
serialized bytes include NPZ metadata/headers. Neither is RSS or physical traffic.

72 fold decoder fits,8full fits,352class-specific KMeans fits paid.23.296s,
690896KiB,unique guarded CPU queue2GiB RSS/6GiB virtual,8GiB available floor.
No unchanged basis expansion/extra-epoch promotion. Positive RBF information
survives, but a33-anchor approximation does not preserve the strongest result.
