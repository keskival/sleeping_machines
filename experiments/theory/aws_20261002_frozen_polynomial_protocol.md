# Frozen native polynomial optimization diagnostic

The zero-nested end-to-end degree2 first pilot fails NLL by.166618 despite
correct derivatives and1.0836 fitting/1.4588 inference work ratios. Frozen RBF
context information passes its nomination gate. These do not distinguish
polynomial inadequacy, decoder optimization or coupled encoder drift. Do not
retrain the failed variant unchanged or claim the context lacks information.

Fix six encoders: all three full984-fit/eight-pass matched-clock coarse seeds,
and their exact matching initial reservoirs. At each encoder compare affine
and the SAME standard degree2 polynomial local query-vector class. Train
regularized multinomial logistic readouts by L-BFGS on fixed fitting features;
C(.01,.1,1), three fitting-user GroupKFold splits, fold-local centering/std
floor1e-6, pooled fitting-validation NLL selection then984-fit refit. All108
fold/12final fits retained. This is conditional frozen-encoder head selection,
not cross-fitting an encoder that already used fitting labels. Development
192 chooses nothing; official test unopened. Save features for reproducible
reuse with checkpoint/input hashes, save every selected head.

Fold feature normalization algebraically into native readout coefficients:
W_eff=W/scale,b_eff=b-W_eff*center. Split linear and quadratic columns to emit
the same native prediction class without a runtime scaler. Readout runs in
float64 with float32 native state, to avoid cancellation from small feature
scales; count the conversions/head storage and all5 sequential head calls.
The head has no feedback into context/state/clocks/keys, so inference native
state and routes remain those of the frozen encoder. Validate the folded
native end-to-end logits/probabilities against explicit transformed features
on the whole192 development prefixes, including causal target mutation and
NumPy/Torch original-core state/winner checks.

No resident/dormant-value read, prefix buffer/dense carrier or RBF decoder
enters this polynomial predictor. Prior encoder training retains original
counterfactual/full-prefix temporal learning; this final decoder phase sends
no new credit to encoders. Distinguish staged learned predictors from an
end-to-end polynomial training result. This comparison is diagnostic first,
not automatic promotion into the research architecture.

Frozen nomination: fitted quadratic beats fitted affine by meanNLL>=.05 and
meanaccuracy>=3pp, eachseedNLL gain nonnegative. To nominate a staged learned
predictor, also demand each quadratic fitted head improve over its original
native NLL by>=.05 without>1pp accuracy decline, and mean fitted-vs-initial
quadratic NLL advantage>=.05. Every arm/seed remains visible; do not revise
thresholds or addC/decodefamily after seeing development. A failed decoder
does not prove zero conditional information; it stops this unchanged line.

Report actual folded native prediction quality, stage work and3repeat sequential
wall/storage for selected fitted heads. Parent encoder fits4.218015GF/seed
remain charged, feature replay and108CV+12refits paid. Solver FLOPs unknown,
so COMBINED fitting FLOPs stay unknown, not4.218GF alone. Native inference
ATen estimates charge all prefixes/head calls; no energy/traffic claim. Compare
strongest coarse RBF77.604%/.686661 and historical native controls explicitly.
No practical supremacy claim from retained-state/reservoir or diagnostic quality.
