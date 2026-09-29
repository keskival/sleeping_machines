# Winning races, losing alternatives, and conditioning the credit

29 September 2026. E118 preserves a genuine discrete forward race. Its
purpose is to combine §167's stable carrier idea with winning/losing route
distinctions, not to replace those races with a dense average.

## 169. One winning continuation, several counterfactual computations

For an arrival (t,h,c), a local receiver has K normalized temporal memories
u_r. Each candidate proposes a bounded correction and a positive delay:

\[
 f_r=\tanh(W_r[h,u_r,m_r/(1+m_r)]+b_r),\quad
 d_r=d_{min}+d_{span}\,\sigma(-s_r),\quad
 r_* = \arg\min_r d_r.
\]

E118 uses K=3 and different learned decay constants. The score s_r is a
learned affine function of the candidate's payload and memory features.
Only the winner emits (t+d_{r_*}, h+alpha f_{r_*}); losing continuations
emit nothing. The winner changes the value transformation and the arrival
time of a real downstream event. The next receiver processes events in
their actual arrival order, including overtaking caused by learned delays.
Each arrival produces exactly one winner, so serial support is protected
without making losing signals identical to winning ones. Creating, holding,
or cancelling additional carriers remains a separate extension.

All K state banks are updated locally and all K scores are evaluated.
Inference evaluates only the winning value projection. Counterfactual
training evaluates losing values as extra work, which is explicitly counted.
There is neither all-pairs attention nor a soft average in the forward
payload. Local coordinate matrices are dense; the event topology is sparse.

### Score credit from a losing timed message

With downstream cotangents g_h and g_t, the first-order cost contrast is

\[
 A_r=\alpha g_h^T(f_r-f_{r_*})+g_t(d_r-d_{r_*}).
\]

Let p=softmax(-d/T). A zero-forward surrogate adds
sum_r (p_r-stopgrad(p_r)) stopgrad(alpha(f_r-f_*)) to the payload and the
analogous term for delay. Its score credit is

\[
 \frac{\partial\widetilde L}{\partial d_j}
 =-\frac{p_j}{T}\left(A_j-\sum_rp_r A_r\right).
\]

This compares alternative computations and times with the actual winner.
It does not give losing values ordinary pathwise training gradients. A
losing expert's value parameters learn when that expert wins; its current
counterfactual value can guide the router toward or away from it.

This is a biased local first-order estimator of a discrete decision, not
an exact derivative of the hard argmin. Its accuracy requires small enough
changes and a suffix smooth along the intervention. Winner changes,
reordering downstream events, and shared parameter effects can break that
approximation. A candidate vector is a plausible alternative computation,
but independently changing every event's score is not generally realizable
by shared router weights (§163). Exact shared-parameter replays and
cross-example transfer remain necessary before interpreting surrogate
credit as useful optionality. No entropy bonus is called optionality here.

The implementation witnesses check that changing only losing value maps
does not alter the forward output; losing maps get zero ordinary value
gradient in a one-event race; counterfactual router credit is nonzero; and
the training forward equals the winner-only inference forward.

### What remains of the depth certificate

For frozen choices **and frozen event times/order**, choosing one of K
norm-bounded residual maps preserves §167's payload bound. The full learned
race also changes delays and can switch branches. Therefore its complete
Jacobian is not certified by that bound. In particular, time sensitivity
cannot be silently included in a payload-only theorem. The constructive
advance is a certified conditional carrier map plus explicit discrete and
temporal mechanisms to audit, not a proof of globally trainable routing.

## 170. Surviving class information can still be badly conditioned

E117's eight-layer, 128-fit/128-held-out-speaker pilot ended with deepest
training accuracy 23/128, held-out accuracy 9/128, and fit NLL 2.7697.
All eight layers retained gradients. A frozen-feature probe then fitted a
ridge classifier using **only the fitting speakers**: deepest features
gave 84/128 fit and 36/128 held-out correct. The input representation gave
83/128 and 39/128. All depths are reported, not selected on development
accuracy; ridge regularization was fixed at 0.01. The transform and head
differ from the SGD head, so this is evidence of accessible information and
an optimization gap, not a one-factor causal ablation or a gain from depth.

At the final depth, the standardized feature covariance has participation
ratio (tr C)^2/tr(C²) about 2.12 despite 33 coordinates. Large common
directions and much smaller discriminating directions coexist. Near uniform
class probability p, the last linear layer's Hessian has Kronecker form

\[
 \nabla_W^2 L \simeq
 (\operatorname{diag}(p)-pp^T)\otimes E[xx^T].
\]

The expression is exact when p is the same for all examples. After centering
x and accounting separately for the intercept, feature covariance controls
the non-class part of this curvature. For balanced uniform p, nonconstant
class directions have eigenvalue 1/C_classes; poorly conditioned feature
directions therefore create poorly conditioned head learning, even though
credit reaches every event layer. Adam's coordinate scaling is not full
covariance whitening.

E118 measures initial pooled features on fitting examples only, centers and
standardizes them, then freezes the regularized inverse-square-root map
(Cov+0.1 I)^(-1/2). It does not use held-out statistics or labels to fit this
map. Its whitened covariance eigenvalues are lambda/(lambda+0.1), bounded
by one. Initial near-null directions remain weak; the transform does not
create information. Features drift as the model learns, so initial
conditioning is not a uniform optimization guarantee. The head starts
with small weights to avoid initially confident wrong predictions.

The terminal feature statistics are legitimate for this diagnostic. An
early-decision implementation would need a causal running feature estimate
and prefix calibration; a full-utterance normalization must not be supplied
to an earlier prefix. Terminal readout conditioning is not yet an anytime
classifier or a measured training-energy advantage.

### Matched next comparison

Both E118 arms start with the same model, fit-only transform, data order,
eight layers and optimizer budget. One arm gets ordinary winner pathwise
credit; the other additionally gets the explicit losing-route contrast.
Report both endpoints, their losses and routing utilization. A gain in this
pair would isolate the added surrogate within this new model, while a
comparison with E83 or E117 changes several structural choices at once.
