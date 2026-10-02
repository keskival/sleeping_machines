# Credit calibration, signal and common-clock coordinates

User asks whether credit can be normalized, balanced or calibrated. Current
fitting already sums then averages gradients over the target window, clips
the global norm to1 and uses Adam. Frozen audit191600Z shows how joint-clock
noise can contaminate that global balance: on full native weights, norm3.3901
becomes6.4002, cosine to local.3940, clipping.2950 becomes.1562. The unchanged
decoder's gradient is consequently attenuated. Choice correction stays3.3915/
.99998/.2949. This is one draw, not expected covariance or an Adam trajectory.

## Calibration that preserves the specified derivative

At fixed prefix/time, exact legal choice credit is
g_i=pi_i*(F_i-R), R=sum pi_i F_i. It is invariant to a shared loss offset:
replacing every F_i by F_i+c leaves g unchanged. This is principled centering,
not an arbitrary boost for small gradient components. Actual counterfactual
losses calibrate magnitude and sign to downstream legal writes/suffix outcomes.
Joint winner/time credit has a separate baseline condition: b independent of
the current winner and time removes zero-mean noise, whereas an arbitrary
time-dependent baseline generally removes useful clock credit too (note57).

A positive definite preconditioner P independent of the current noise gives
g dot E[P*g_hat]=g dot P*g>0 for an unbiased g_hat at a fixed prefix. A
scalar objective normalization changes its units; unequal component weights
are a preconditioner or a changed objective and must be specified accordingly.
Gradient RMS is not a measure of useful signal. Normalizing each noisy sample
can introduce bias: scalar g_hat=+2 with probability1/3 and-1 with probability
2/3 has mean zero, but unit-normalized mean is-1/3. Fixed positive scaling
cannot manufacture signal from this zero expectation. Clipping/Adam are
practical transforms with their own noise/dynamics, not an unbiasedness proof.

Positive scaling in *score* space is not automatically a positive parameter
preconditioner. With score Jacobian J=[[1,0],[2,1]], score gradient g=[1,-1]
and positive output scaling D=diag(10,1), the true parameter gradient J^T*g
is[-1,-1], whereas J^T*D*g is[8,-1]. Their dot product is-7: updating parameters
opposite the rescaled credit increases the local loss at first order. This
zero-sum g is admissible categorical choice credit. A proper parameter-space
metric or a tested score-target/KL projection needs the actual Jacobian; simply
boosting low-probability score gradients through a coupled emitter is not a
universal descent guarantee. This does not invalidate the exact chain-rule
credit of the current integrated driver or authorize decoupling content/time.

## Factor total firing rate from relative choice

Let common log-rate c and relative logits u define

    s_i = c+u_i-logsumexp(u),
    lambda_i = exp(c)*softmax(u)_i, Lambda=exp(c).

At fixed candidates/prefix/T, conditional winner enumeration of joint density
credit yields h_s,i=pi_i*(F_i-b)-lambda_i*T*(R-b). The coordinate pullback is

    h_c = sum_i h_s,i = (1-Lambda*T)*(R-b),
    h_u,i = h_s,i-pi_i*sum_j h_s,j = pi_i*(F_i-R).

Thus the relative-choice coordinates are exactly independent of the shared
clock-baseline error; all of that error is confined to the common-clock
coordinate. The clock component still matters for actual useful latency and
downstream timing jumps. Independent clock baseline/stepsize calibration can
retain it. Keys/values remain separate; shared incoming content/memory can
feed both emitters and still receive both paths.

There is no free fix from a coordinate identity. Taking u=old scores and
c=logsumexp(old scores) reproduces the original rates AND the original gradient:
the c pullback restores the same shared-clock term to the old score parameters.
Detaching c would be a deliberate gradient approximation, not that identity.
To benefit structurally, separate learned output maps would need an integrated
comparison, with common content gradients still potentially noisy. No such
architectural substitution is implemented or fitted here. All eligible-key
normalization/discovery work must be charged, especially for large pools; an
explicit logsumexp is not free asynchronous hardware computation.

Required numerical contracts independently differentiate the joint winner/time
log density, test loss-offset invariance and the complete clock/choice pullback,
recompose old scores to show forward/gradient identity, and demonstrate the
sample-normalization bias witness. These are mathematical/numerical contracts,
not a new fit or a clock-normalized architecture result.

The completed first fixed256-fit seed6 actual-write choice pilot gives
1.305937->1.194006NLL and54.17->58.85% accuracy at1.33867 fitting work ratio.
That supports confirmation, not normalization causality or superiority over
full-data RBF/compact controls. Seed7 controls/treatment remain the next
gate. No global no-go theorem, guaranteed cross-benchmark gain, or successful
larger-capacity claim follows from these calibration identities.
