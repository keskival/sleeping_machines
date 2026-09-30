# A gradient cap is not a probability-space step cap

Derived 30 September 2026. E150's clean-head initialization has augmented
fitting NLL 0.6358 under a matched reconstruction, but its first encoder pass
averages NLL 5.0118. E152's paired head starts at augmented NLL 0.3087 and
406/512 development accuracy; its ensuing updates also initially raise loss.
The finite replay below isolates the actual first update before attributing
this disruption to representation limits, missing routes or augmentation alone.

## 249. Fresh Adam can largely cancel the size of a clipping factor

For a coordinate with fresh moments, bias-corrected Adam's first update is

\[
 \Delta\theta_j=-\eta\frac{g_j}{|g_j|+\epsilon}. \tag{249.1}
\]

Clipping g to c g, 0<c<=1, changes this to
-eta g_j/(|g_j|+eps/c). When |g_j| greatly exceeds eps/c, the coordinate
step remains approximately eta times its sign. With N active coordinates,
its parameter-space norm can approach eta sqrt(N). Therefore a cap on the
teacher's norm does not impose that cap on the Adam step or on changed logits.
This is a direct consequence of the optimizer formula, not a novel Adam result.
With existing moments the exact step differs and must be replayed from those
moments. The joint clipping covariance issue of §§224–225 is separate.

An affine readout rewrite can preserve predictions while changing the hidden
teacher by V^T r (§240). A well-conditioned fixed-feature head also need not
have a well-conditioned full encoder Jacobian. Thus fresh Adam after absorption
needs a function-space step check; resetting its moments is not innocuous.

## 250. Observe the step through the categorical event query

For proposed parameter displacement d and query logit Jacobian J_i,

\[
 D_{KL}(p_i(\theta)\Vert p_i(\theta+d))
 =\tfrac12d^T J_i^T H_i J_i d+o(\|d\|^2),\quad
 H_i=\operatorname{diag}(p_i)-p_i p_i^T. \tag{250.1}
\]

The relevant curvature is the complete encoder/query Fisher. It contains
source, state, gate and timing effects, not just a head-feature covariance.
No event-pair attention matrix is needed: each supplied utterance produces
one C-class probability vector. Finite replay measures its actual distortion,
including changed winners/order and nonlinearities; it avoids assuming the
linearization remains accurate. KL by itself would preserve wrong predictions,
so the calibration also requires real supervised loss descent on the update
batch. It is a local step criterion, not an optionality objective.

## 251. Calibrate shared depth comparisons on fitting data only

E156 restores the same selected E152 checkpoint and the same actual first
four augmented fitting utterances for every proposal. The following 32 fitting
utterances supply anchors; none comes from development or the reused audit.
It replays Adam with factors 1,1/4,1/16,1/64,1/256,1/1024, identical teacher,
optimizer state and clipping. A proposal is admissible if update-batch CE does
not increase and anchor mean KL is at most 0.02. Choose the largest declared
factor admissible for both the unchanged D6 and identity-grown D12.

The replay records actual CE changes, parameter-step norm, KL and both the
clipped surrogate and exact interior pathwise directional predictions. The
pathwise gradient excludes the training-only clock-choice surrogate. This
distinguishes a large nonlinear step from a surrogate direction that fails
even locally; crossing order/race boundaries still requires the finite result.
Only completed replay results can establish which happens in this example.

E159 retains old Adam state and parameter order, scales both arms by
this common fitting calibration, and keeps old/new clipping ownership separate.
The D12 new group shares the scale; it adds real parameters and event work.
Existing old teachers match at identity initialization (§242), so its first
old update has the same ownership as the D6 control. Later representations
intentionally diverge. The calibration cost must be charged to both protocols.

## 252. The guarantee is local; the empirical question is deeper learning

A finite accepted first step has its observed descent and probability bound
on the declared fitting/anchor examples. A fixed calibrated learning rate
does not enforce a trust region on every subsequent update, bound all new
speakers, or prove convergence. Rechecking selected steps would be an explicit
additional training mechanism and budget. The current experiment makes the
one calibration choice, retains the 8-GiB host reserve, and reports all outcomes.

This explains a concrete distinction: class information in a trained event
query, stable gradient transport at added depth, and a safe realized optimizer
step are three different conditions. E152's fixed-feature gain demonstrates
the first; E151 supplies a constructive second; E156/E159 tests the third.
Each condition is stated as a measurable mechanism rather than a route-count
or raw-gradient heuristic.

**Completed D6 E156 replay:** the initial four-utterance CE is 0.622628 and
the independent fitting-anchor CE is 0.238607. Original LR 0.000325 raises
them to 1.867725/2.098338, with anchor mean KL 1.845976. Its exact interior
directional loss prediction is -5.848710 while the actual batch change is
+1.245098: the finite step defeats the locally useful teacher. At 1/16 scale,
LR 0.0000203125 yields CE 0.333181/0.243864, anchor KL 0.004412 and a batch
change of -0.289447. The physical directional prediction is -0.365528.
Gradient norm before clipping is 69.9252; the original/1/16 parameter-step
norms are 0.202736/0.012671. The first-step moment state is empty. This is
causal evidence of excessive step size in this absorbed encoder continuation,
not a retrospective explanation of every historical SHD failure.

**Restored scheduler audit:** E155's intended calibrated `lr` was overwritten
by the optimizer's saved `initial_lr` when `LambdaLR` initialized. Actual epoch
rates stayed 0.000325. Those completed files are historical uncalibrated runs.
E159 resets each group's scheduler base to its calibrated rate and asserts the
actual group rates before training. Its D12 online NLL is 0.396784 at verified
LR 0.0000203125, versus 3.507343 for E155 at the overwritten rate. Development
is 400/512, below the 406/512 starting model: corrected rate stability and
generalization improvement remain different empirical questions.
