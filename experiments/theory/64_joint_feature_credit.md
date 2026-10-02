# Joint features and why gradient reach is not sufficient

Let S1,S2,D be independent balanced signs, and Y=S1S2. Then
I(Y;S1)=I(Y;S2)=0, while I(Y;S1,S2)=1bit. This is an exact finite
distribution, not an assertion about the statistics of text8.

For logistic sign loss ℓ=log(1+exp(−Y w F1F2)), at w=0,

\[
\partial_w E[\ell]= -\tfrac12 E[Y F_1F_2].
\]

If F1=S1 and F2=D, that derivative is zero. If both relevant features are
delivered, F1=S1,F2=S2, it is −1/2. Likewise a linear logit w1S1+w2S2
has zero balanced gradients at w1=w2=0 even with both bits available. The
information needs an interaction; mere layerwise derivative reach does not
establish that a model is learning it. Arbitrary nonzero initialization and
other objectives change these gradients, so this is not a universal no-go.

If two independent hard selections deliver their relevant features with
probabilities r1,r2 and independent distractors otherwise, the expected
interaction gradient at zero is −r1r2/2. Improving one choice while the other
is purely uninformative gives no immediate mean gain. Uniform selection from C
candidates yields a C⁻² joint exposure factor. For m-way parity the analogous
factor is the product of relevant-delivery probabilities. This concerns a
specific unstructured routing policy; it is not a capacity ceiling, a universal
sample-complexity law or evidence that every sparse route must suffer it.

The implication for the substrate is concrete: single-choice local teachers
can miss useful combinations of unrealized alternatives when other required
features are absent. Inspect feature retention, joint exposure and conditional
credit, not just nonzero per-layer norms. Joint counterfactual supervision,
structured candidate discovery or a curriculum exposing constituent recalls
are distinct mitigation hypotheses. Charge their extra candidates/value work;
do not label a gradient through the realized nonlinear route an exact estimator
of all route changes. The current temporal hard-route surrogate remains scoped
as before, and this note does not modify frozen teachers.

The paired-parity generator supplies identical query suffixes/count vectors and
all bit combinations. Frozen text-model probes show dependence only. A future
integrated fit must demonstrate retained input information and target credit
under matched full/shallow controls before invoking this explanation for a
measured failure. If fixed-context hashes never read the relevant old address,
no projection-credit repair can recover its missing evidence; learned keys or
a demonstrated other information path must address that failure.

`tests/test_joint_feature_credit.py` verifies exact balanced derivatives and
the r1r2 attenuation. These are theoretical numerical contracts, not fitted
language or semantic-feature results. There is no mathematical prohibition on
progress here: the full input contains the required bit, and a correct retained
representation plus interaction can predict it. Learning/resource efficiency
of a particular architecture still needs reproducible evidence.
