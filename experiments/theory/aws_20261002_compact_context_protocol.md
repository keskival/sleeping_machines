# Compact local nonlinear native readout with demanding raw controls

RBF context probes improve all three fitted coarse encoders' NLL, whereas
end-to-end degree2 and frozen polynomial nominations fail. Test a bounded
RBF/prototype LOCAL decoder of the32-dimensional current native query vector.
No prefix carrier, all-resident read or new attention mechanism enters.
Keep encoder states/clocks/hard races/keys/selected writes unchanged. Encoder
training remains the paid earlier counterfactual native stage; this readout
phase trains no new producer credit. Do not promote it as an end-to-end fit.

Freeze one basis: three prototypes per each of11 fitting classes (33anchors),
KMeans n_init5/max_iter100/seed681, fit-only feature standardization (std floor
1e-6 for latent context, .5 for raw logcounts), gamma1/(D*fitvariance),C(.1,1,10).
Three fitting-user GroupKFold folds re-estimate normalizer and class prototype
basis within each fold; choose pooled validation NLL then refit all984. Frozen
encoder folds condition on an encoder that used all fit labels. No dev choice;
no official test. Six native fitted/initial encoders (all3seeds), AND raw20-bin
and4-bin33-prototype controls. The4-bin raw control is mandatory: omitting it
would give a misleading memory comparison against a larger fine-input control.
All72 internal decoder fits plus8full refits and shared basis fits recorded.

Demand-only numeric native inference is a separate engineering implementation:
cache the last aligned native vector, use an empty interim readout, and classify
only at the explicit observed finalquery. Original logits have no feedback to
state/races. Verify every route/memory/arrival/context/delay/noise counter against
the frozen original numeric port and original Torch core, plus roundtrip model
exports/repeated causal predictions and target mutation. No shared port edit.
Charge all candidate keys and selected values plus all33 query anchor distances,
RBF exponentials and11-class decoder contraction. Prototype readout scores all
anchors and uses all basis outputs; it is not sparse race attention.

Comparisons use same standalone uncompressed array export for native/raw
predictors, including packet normalization, cached rates/noise, all core maps,
anchors, decoder/scalers and metadata. Count tensor/array bytes separately from
serialized bytes; neither is live process RSS or physical traffic. Measure
same host1thread3repeat sequential prediction calls, whole prefix preprocessing
included. Fitted native parent4.218015GF/seed remains paid, but combined fitting
FLOPs unknown due KMeans/decoder solver. Reuse saved frozen feature cache with
source/input/checkpoint hashes; its earlier replay work stays linked, not free.

Fixed stage nomination: fitted native prototype beats original native by>=.05
NLL at EACHseed without>1pp accuracy decline, and trained native versus matching
initial prototype has>=.05 meanNLL gain. Independent practical storage gate
against raw4 AND raw20 prototypes: eachseed native NLL<=control+.02, accuracy
>=control−1pp and exported bytes<=.80controlbytes. Report latency too; passing
storage gate alone would be an isolated storage/quality advantage, never total
supremacy. All seed/arms and historical strongest4-bin SVC77.604%/.686661 and
compact20-bin66.667%/.902951 stay visible. Failed gates stop unchanged scaling.
