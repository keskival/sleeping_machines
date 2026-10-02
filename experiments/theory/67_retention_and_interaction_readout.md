# Retained bits and an interaction readout

The balanced joint episode has one bit of target information in the complete
prefix and zero in its restricted query suffix/count inputs. A failed linear
readout need not mean the prefix has been forgotten. Consider a frozen query
representation h = c + a S1 + b S2 for independent balanced signs. Individual
bit information can be present even though E[S1 S2 h] = 0. A linear parity
readout has no mean target direction at its balanced zero-logit initialization.
But

    E[S1 S2 h h^T] = a b^T + b a^T.

Products of retained features expose an interaction. This is a statement about
a restricted representation model, not an assertion that the actual core has
this form. Within each four-example group, compute the exact finite Walsh
coefficients of h for the constant, first bit, second bit and parity. Their
magnitudes and consistency across held-out noise suffixes distinguish retention
and interaction in the actual checkpoint. Nonzero coefficients alone do not
establish useful prediction.

## Concrete diagnostic and architectural scope

When the current native/tapped/full/shallow cycle completes, freeze each
selected encoder and compare two trainable residual readouts on the same
fitting query features:

1. Affine residual on the standardized query vector.
2. The same affine residual plus all upper-triangular products of that vector.

Both residuals start at zero, exactly nesting the selected parent's two-symbol
logits. Standardization uses fitting features only. This is a standard degree-2
polynomial feature map, not an XOR extraction or a new attention primitive.
No target, bit identity or suffix-group identity is supplied at prediction.
Each predictor processes its actual observed complete prefix through the
unchanged integrated temporal core, with clocks, sparse receivers, persistent
state, keys/values and the admitted counterfactual-trained weights preserved.

The architectural intervention is at the query readout, after aggregation.
It adds local vector products and a small learned linear map, not a replacement
of temporal recurrence with a dense model. It introduces no learned KV retrieval
or historical producer credit. The frozen encoder is a diagnostic control;
success does not demonstrate end-to-end feature learning or nominate this as
the main architecture without a later integrated refit/comparison.

Include the native full encoder's **initial** checkpoint as a frozen control.
If its readout learns equally well, the result supports a useful temporal
reservoir plus learned decoding, not improved encoder representations. Report
its zero encoder fitting work separately; trained encoders must pay their
entire earlier fit, not only readout fitting. One-bit recall probes are
diagnostic losses trained on additional bit targets, never inputs to the parity
predictor, and must be labelled as extra supervision/work.

## Prespecified bounded followthrough

Use the parent's eight fitting noise groups, four bit pairs each, under four
coupled race-noise views (128 feature observations). The bit targets only
support separate retention probes. Original development groups remain reused
exploration. Before fitting any readout, reserve new seed-73001 noise suffixes,
32 complete groups/128 labels, for a single fixed-readout evaluation. None of
their losses selects regularization, feature degree, optimizer budget or
readout weights. Multiple declared arms on this set are still one synthetic
distribution/one encoder seed, not a general confirmation campaign.

Fit zero-initialized residual coefficients with regularized logistic loss,
lambda = 1e-5, using full-batch L-BFGS at most 100 iterations and tolerances
1e-8 gradient/1e-10 change. Bias is unpenalized; all other residual coefficients
have the same penalty. This is convex in the fixed feature coefficients. Charge
every closure/optimizer operation, feature-statistic/design construction and
all prefix feature replays. Report actual closure count and fitting derivative;
do not infer convergence merely from termination. No new core optimizer step.

Numerical contracts must verify polynomial adjoints, exact zero residual
nesting, actual causal integrated prediction, restored fitted-readout output,
and frozen encoder/feature fingerprints. Operator coverage remains mandatory.
Run all jobs serially through the host guard; no unbounded history graph and
no duplicate of other-host natural-language studies.

The relevant positive result would be a fixed predictor's new-suffix loss below
the one-bit restricted-count bound, with at least 75% accuracy and .8 bits
loss, and the nonlinear readout improving the corresponding affine residual
by .05 bits. This demonstrates useful joint prediction under this information
restriction and declared supervision. It does not establish natural-language
semantic abstraction, an advantage over timestamp-aware dense controls or
iso-quality resource supremacy. Preserve every unsuccessful arm and any
initial-reservoir success beside it. A later encoder refit, learned retrieval
or joint route teacher is a separate hypothesis requiring new prerequisites.
