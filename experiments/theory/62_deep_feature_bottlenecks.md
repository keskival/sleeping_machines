# Deep features: information, state use and credit are separate contracts

2 October2026. Read the preserved negative findings and corrected interpretation
in [the language diagnosis](../LANGUAGE_LEARNING_DIAGNOSIS_20261002.md).
This note supplements §§389–391; it does not replace temporal computation,
hard-route counterfactual credit or available capacity beyond active work.

## What the completed evidence actually separates

All eight native layers receive nonzero gradients. The earlier native-state
baseline also reproduces the saved AWS training trajectory. These are evidence
against a general gradient/optimizer regression; they do not establish useful
deep representation learning. Saved language predictions depend on history
beyond a bigram, but dependence is weaker evidence than a held-out benefit.

The matched2K full/minimal16/64-credit fits show that changing graph reach alone
does not earn a useful quality gain here. A material frozen-gradient difference
cannot be substituted for measured fitting improvement. Do not infer that every
long-range information path is adequate from that result either: longer credit
cannot reconstruct information a forward state has already lost.

The new frozen attribution keeps count evidence and its cursor fixed. Erasing
the full model's source context and receiver contents once worsens one32-target
slice by.048141bpc; receiver erasure alone does not hurt. Thus learned recurrent
information is present. Its apparent concentration in the source-context path
is a concrete hypothesis about allocation of useful memory, not a proof that
the eight receiver layers are unnecessary. They help produce that context, and
their contents can rebuild after the intervention. Whole-development evidence
still favors the much cheaper minimal core.

## Correct mixture gradients can still allocate little useful deep credit

With evidence C and fixed gate values, the cascade is affine in its base q:

\[
p_y=a_y(C)+e(C)q_y(X),\qquad
\partial_{z_j}[-\log p_y]=r_y(q_j-1[j=y]),\qquad
r_y=\frac{e q_y}{p_y}.
\]

This identity is verified on the actual trained gated models with D/theta held
at their identical-forward values, not assumed from an initialized carrier.
For64 fitting targets, full-core mean/median r are.071734/.004530; minimal
.057129/.003138. The dynamic gate adds a separate derivative through its
q-dependent features. It must not be silently omitted from an actual gradient
or represented by r alone. All layers receive that actual dynamic-gate credit.

Small r is proper mixture credit, not an autograd defect. Multiplying every
gradient by a common constant is not generally a proportional reduction in an
Adam update. The measured norm ratio alone therefore cannot explain failure.
Heterogeneous r can instead concentrate learning on few unpredictable examples,
while deep memory competes with cheap evidence-dependent smoothing for those
examples. Test residual exposure and predictive information, not just magnitude.

If counts C already predict most targets, a new feature Z can improve an
otherwise optimal log-loss predictor only through I(Y;Z|C). This is a statement
about useful conditional information, not a ceiling on architecture or a claim
that our current predictors are optimal. A base optimized as a residual can
perform badly by itself while improving the composition. A count-statistic
readout can improve composition without demonstrating deeper recurrent features.

## Retained mechanisms and the next local comparison

Concrete failure addressed: full native language quality does not justify its
cost against a minimal count-composition control, despite credit reaching its
layers. Every token uses one source address and a small shared receiver pool;
there is no native per-position key/value retrieval bank. Temporal decay,
repeated nonlinear updates and interference are plausible information losses,
not a measured complete causal explanation or finite-precision impossibility.

Prioritized local repair diagnostic is ContextAddressedNativeModel at the
existing H2/d16/depth8 setting. It retains the native temporal races, evolving
messages, persistent addressed receivers, separate receiver keys/values and
unrealized-route surrogate credit. It adds one fixed context-hash slot read and
one preceding-context outcome write per event. Slot count/storage, hashing,
linear maps, losing-route learning and optimizer work must be charged. This
tests capacity and evidence paths, **not learned context pooling or race
attention**; collisions mix contexts blindly. Its zero read map nests native
and stages write-map learning after read-map learning begins.

TappedNative is a secondary path-length diagnostic, retaining the same core
and adding bounded layer-input buffers/interpolated reads and one extra square
matrix multiply per layer/event. A detached buffered value can still teach a
current read map; lack of credit to its old producer does not imply no benefit
at16-event truncation. Learned delays at a clamp or beyond the stream may have
zero gradient. Compare the actual retained information and cost, rather than
assuming exponential receptive-field reach makes that information useful.

Full-shape contracts pass for both repairs: exact zero forward/parent gradients,
trained causality/chunk invariance, real target-weighted normalization, and
serialized next-Adam parameters/predictions/moments. Numerical contracts are
not useful-feature evidence. The long-range driver now separates U64 updates
from credit length; whole-operation short accounting smokes precede further
fits. It still needs whole-driver boundary checkpoint/recovery admission.

Required comparisons: native versus repaired core with matching construction
of core weights, race RNG, data/passes, optimizer interval and selection rule;
full/minimal repair controls where evidence-derived smoothing could explain a
gain. Distinguish cold development state from frozen fit-prefilled replay and
charge warm-up work before comparing to saved count references. Measure each
causal local order independently on synthetic targets: inserted cues invalidate
the previous all-orders independence proof. Reserve abstraction claims for
tasks with demonstrated local-control insufficiency and generalization to
unseen contexts/rules, rather than memorized fixed hash contexts.

The unchanged AWS integrated common-seed capacity/exposure campaign remains
the priority integrated research model overall. Local fixed-hash repairs are
diagnostic stages with learned address pooling, KV race attention and complete
producer credit still open; no successful cheap carrier is promoted to the
main architecture. No broad mathematical impossibility or supremacy follows
from the current results.
