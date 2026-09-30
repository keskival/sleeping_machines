# Local teacher statistics when old and new parameter blocks share a clip

Derived 30 September 2026. E138–E140 do not improve the strongest SHD score.
This note identifies a specific optimizer confound to isolate, not a proved
explanation of the whole quality gap. Rich nonlinear temporal representation
and capacity remain independent requirements.

## 224. Clipping a joint gradient changes a new learner's sample weighting

Let Theta denote the old trained parameters, phi new source/temporal controls,
and g_Theta,g_phi their minibatch teachers. Joint norm clipping at radius c gives

\[
 \widehat g_\phi=\kappa g_\phi,\qquad
 \kappa=\min\{1,c/(\|g_\Theta\|^2+\|g_\phi\|^2)^{1/2}\}. \tag{224.1}
\]

Each minibatch keeps its new-block direction, but their average need not:

\[
 \mathbb E\widehat g_\phi=(\mathbb E\kappa)\mathbb E g_\phi
                             +\operatorname{Cov}(\kappa,g_\phi). \tag{224.2}
\]

The covariance is vector-valued. It can reverse the average teacher. For two
equally probable batches, take scalar new teachers 1 and -0.9, and old norms
100 and 0. With c=1, the original new mean is +0.05, but the jointly clipped
mean is approximately (0.01-0.9)/2=-0.445. A step descending the clipped mean
ascends the original new-block expected loss locally. These teachers can be
realized by local affine sample losses; no stochastic routing is needed for
the counterexample. Clipping may still be a useful stability policy, but its
cross-sample direction is not an unbiased teacher for the unweighted risk.

Changing only the old parameter's units changes its raw gradient norm and
therefore kappa, even when the computed function and the new parameter's
teacher are unchanged. Raw joint clipping is consequently not invariant to
an unrelated old-block reparameterization. Block metrics/trust regions, frozen
old blocks or a justified importance-weight correction must explicitly choose
the intended teacher statistics. Dividing by kappa can restore its mean but
also restores the large/noisy updates clipping was intended to control.

Adam does not remove this distinction: its first moment estimates the clipped
teacher, and its second moment estimates a differently weighted square. There
is no general cancellation of a sample-dependent kappa between these moments.
In a scalar stationary regime its positive denominator cannot correct the sign
reversal above. This is not a claim that every clipping/Adam combination harms
learning or that clipping should be removed globally.

## 225. A constrained local-learning intervention with a fixed parent

E140 records 1,535 clipped steps out of 1,536. Its mean joint preclip norm is
about 11.97, while its new-source norm averages about 0.157; the phase teachers
are also much smaller. These aggregate numbers establish disparate block
scales and widespread clipping. They do not determine the covariance in
(224.2) or prove that it caused the held-speaker plateau.

E141 freezes the complete parent state and learns only phi: 5,760 source
parameters and 384 phases. It uses the same source model, parent, fitting and
held examples, augmentation, sample order, rates and one-pass budget as E140.
Its clip sees only the active new teachers. When ||g_phi||<=c, they are not
weighted by the old block's norm. The parent optimizer/moments are retained
unused, and an exact state comparison verifies that no old parameter or
calibration buffer changed. Thus the intervention separates old-function
adaptation and joint clip weighting from availability of the new teachers.
It does not isolate those two consequences from each other.

Freezing old parameters does not freeze the model's decisions: learned input
marks and memory phases can change winners, clocks and outputs. The initial
phi=0 program remains exactly the strongest checkpoint. Subsequent policy
credit still uses the declared surrogate; sparse actual forward winners are
unchanged as a mechanism. The final quality criterion is improvement over
370/512, with official-test parity remaining separate. New-block clipping and
the completed outcome must be measured before this is recommended as a general
training rule.

**Completed intervention:** E141 changes all eight phase blocks, learns source
marks and leaves every parent parameter/buffer exactly unchanged. None of its
1,536 updates clips the new teacher. Nevertheless it finishes at **369/512
(72.0703%)**, below the parent 370/512; fitting accuracy is 4,935/6,144 (80.3223%)
and pooled held-speaker NLL is 0.989443. Removing this cross-block weighting is
therefore insufficient at the recorded one-pass budget. The clipping identity
is valid; these results do not identify clipping as the cause of the accuracy
gap or establish a superior training rule. The next structural investigation
must address nonlinear temporal representation and source/state capacity.

For an eventual function-preserving widening (§223), keep the old and new
teacher metrics explicit. Adding observable duplicate coordinates should not
silently change the old optimizer's first step solely through a larger raw
joint norm. A blockwise preconditioned trust budget can control each functional
update while preserving old moments. Hard-route changes still need their own
finite-utility check; an interior metric is not a boundary-credit theorem.
