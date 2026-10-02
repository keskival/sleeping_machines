# Producer-held decoder selection and sample-size-consistent regularization

2 October2026. Consequence of notes94–95: most trained full-state probe gain
can be explained by a stronger query-head regularizer. Query C.1 reaches
.889842 versus query C1 .999857; full state C.1 .859084 and C1 1.278955.
These controls disfavor a unique memory-access diagnosis. Layer0/1 still give
finite probe improvements, but no causal depth or sparse superiority claim.
Existing labels-trained encoder features make conditional decoder CV a different
quantity from whole-pipeline generalization. Investigate its selection behavior
before inventing another route/content operator.

## A formal selection counterexample

A binary producer stores every observed fitting label as Q_j=Y_j in {-1,+1}.
For unseen inputs Q is an independent equiprobable sign, unrelated to Y. A
linear decoder with positive slope w has conditional seen-feature CV risk
log(1+exp(-w)), strictly decreasing in w. On a new input its expected loss is
one half[log(1+exp(w))+log(1+exp(-w))], strictly increasing for w>0, with optimum
w=0. Enumerating three slopes .5/2/5 verifies opposite confidence preferences.
Thus correct fitting-fold scaler/decoder fitting does not make conditional
CV unbiased when the producer saw those held-fold labels. This is a constructed
selection failure, not proof that native features are memorized labels or that
all frozen-encoder CV is wrong. Producer-trained validation must preserve that
conditioning distinction.

## Existing-checkpoint experiment: no new producer training

Use both native local pilots256-fit/192-dev/four-pass/U16, seeds6/7. Choose their
ONLINE model after exactly four passes, not the minimum-development checkpoint;
reproduce the saved pass4 predictions. These producers learned only the first
256 fitting labels. The remaining728 FIT examples were never optimized by those
producers and become producer-held decoder selection data. Original input
scaling used all984 unlabeled fitting counts; that is recorded and unchanged.
No official test or extra producer pass. The fixed model choice avoids a new
validation-based epoch selection, but this entire research dev set has already
been inspected and stays exploratory.

For each encoder compare two prespecified rules over THREE affine L2 heads:

- Conditional decoder CV: three stratified folds inside the first256 examples,
  using representations whose producer had already seen their labels.
- Producer-held: fit each candidate on all256 examples and select on the
  other728 fitting examples, whose labels did not train the producer.

Each rule selects by fitting-data NLL. After selection refit its head once on
all984 fitting features and evaluate the192 development examples. This changes
readout data exposure relative to the saved256-only pilots, so do not call the
quality difference matched-data or iso-FLOP advantage. Both rules see the same
fixed core and final readout data; only their prespecified selection differs.
The728 targets are deliberately selection data, not a final generalization
claim. Save every cell and both outcomes whether selection agrees or differs.

## Match the regularizer by mean objective, not by C alone

The locally installed scikit-learn logistic solver implements mean pointwise
loss plus lambda/2||W||^2 with lambda=1/(n*C) for unweighted samples; intercept
is unpenalized. Verified in its _logistic.py l2_reg_strength and _linear_loss.py
l2_penalty before admission. Using unchanged C across folds and refit changes
lambda as sample count changes. That is legal as a declared algorithm, but
confounds a comparison about regularization strength.

This experiment fixes lambda across each candidate's folds,256 fitting and
984 final refit. Candidate lambdas are1/(984*C_ref), C_ref=.1/1/10, using the
existing three nominal final-readout settings. At each actual n use
C_actual=1/(n*lambda). Fitting-fold feature centering/std floor.5 remains; the
penalty is in standardized feature coordinates, not asserted equal to a native
raw-head weight penalty. This avoids another unwarranted transplant of a good
frozen-readout setting into a joint core update. A native raw-coordinate penalty
would need its own derivation and integrated fit.

## Fold the final affine head into the unchanged native model

If standardized feature x'=(x-center)/scale has coefficients A and intercept b,
set native W=A/scale and native bias=b-sum(A*center/scale). No additional inference
operator or parameter is needed. Require all192 calibrated batched probabilities
to match the solver, eight independent winner-only serial predictions to agree,
and every nonhead parameter to remain bitwise unchanged. Original pass4
probabilities must also reproduce before any calibration. The native event
core still performs physical-time evolution, races, actual sparse state updates
and persistent representation; its inference operator/activity boundary is
unchanged. Generic head fitting does not establish the sparse architectural
thesis by itself, but this controlled native calibration can guide later joint
learning. Save decoder and ported native head tensors for review/reproduction.

Work includes each saved producer's2.285696GFLOPs fit, all984/192 feature
replays, each candidate solver/final refit and additional head-port validation
replays. Known core fitting/replay/verification work is recorded separately;
solver/transform/grid/traffic/energy are unknown rather than zero. Complete
captured32/24 core shapes are multiplied by actual counts, with seed/node/time
and version provenance. No cheap-total-fit claim while solver work is unknown.
One serial safe CPU job, one thread/nice19,3M KiB VMS/1.25M KiB RSS watchdog,
8GiB memory floor. No dense Transformer/LSTM or duplicate other-host campaign.

Next decision depends on completed BOTH selection rules/seeds, not anticipated
scores. If producer-held selection consistently avoids an overconfident head,
test a separately declared joint-native conditioning/regularization repair with
correct raw-coordinate metric, numerical contracts and a tiny integrated fit.
If it does not, retain that failure and pursue the smaller state-access signal
with a properly contracted sparse temporal receiver. No automatic architecture
substitution, whole-gradient exactness or superiority is admitted here.
