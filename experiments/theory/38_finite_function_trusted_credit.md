# Use realized event continuations to accept learning steps

Derived 30 September 2026 during the E155 update investigation. Subsequent
inspection finds that E155's scheduler silently overwrote its calibrated rate;
E159 corrects that implementation and has priority. The mechanism here is a
prepared bounded training algorithm, not evidence that per-update replay is
necessary after the corrected static rate or improves recognition.

## 257. Parameter proposals have real event counterfactuals

At theta, form one Adam proposal d from the current fitting utterances A,
with the retained moment state and declared old/new clipping groups. Draw an
independent fitting group B, excluding A. Run the actual event encoder on A+B
to record class probabilities p and label CE before moving parameters.
For alpha in {1,1/4,1/16,1/64,1/256,1/1024}, recompute the complete encoder at
theta+alpha d. Its actual winning clocks, arrival order, nonlinear values and
completed queries determine the counterfactual predictions.

This is a one-dimensional parameter ray with six finite proposals, not the
whole routing tree or an imagined differentiable frozen suffix. Its teacher
can contain the declared local loser-clock surrogate; realized acceptance
checks its actual outcome. Losing proposed updates do not alter inference.

## 258. An observed local learning contract

Accept the largest declared alpha such that

\[
 L_A(\theta+\alpha d)\leq L_A(\theta),\quad
 L_{A+B}(\theta+\alpha d)\leq L_{A+B}(\theta),\quad
 \overline{D_{KL}(p\Vert p_{\alpha})}_{A+B}\leq\kappa,
 \quad\kappa=0.02. \tag{258.1}
\]

Current-batch descent checks actual useful credit. Paired descent tests
interference on additional fitting evidence; KL checks the magnitude of the
function change. These three are different criteria. If no declared proposal
passes, parameters stay at theta. Adam's teacher moments advance once, since
one fitting teacher was observed; no additional moment update is taken per
replay. This is an explicitly different optimizer from unrestricted Adam.

Every accepted update has its observed finite descent and probability bound
on these supplied fitting examples, including changed hard decisions. This
does not prove monotone whole-epoch risk, unseen-speaker improvement or global
convergence. Probability preservation can constrain correction of confident
errors; kappa is a declared experimental budget. The label criteria retain
the actual supervised objective. This is not a future-optionality score.

## 259. Correct numerical and data boundaries

Before/after replay uses identical packed A+B examples so batched floating
scan differences do not become an acceptance signal. The original gradient
is computed on A. B is sampled by a separate fitting-only RNG; development
and the reused audit never enter gradients, proposal selection or KL checks.
Channel/time transformations use the existing label-preservation assumptions.
The extra B labels used to accept updates are part of the training protocol
and cost, not unseen evaluation evidence.

Proposal displacements are captured once and each alpha is relative to the
original theta. Interpolating an already interpolated proposal would be an
implementation error. Parameters, teacher moments, replay counts, accepted
alpha/KL and actual CE changes are recorded. Rejected proposals are retained
as diagnostic counters; their values are not emitted by the trained network.

## 260. Charge bounded replay; judge it by quality and physical work

E158 keeps one actual vector/delay emission per supplied packet at inference.
Training uses one backward pass per fitting batch, one joint reference forward
and at most six joint counterfactual forwards. Work is bounded by a declared
constant times event work, with no event-pair attention matrix or global route
tree. Its extra source, projection, gate, state-composition and clock counts
are logged separately and must be added to gradient-forward work.

This spends more training work per batch to prevent the measured destructive
updates. The practical criterion is transferable quality and total work to
reach it, compared with the unrestricted trajectories and static calibration.
Only a completed E158 result can establish improvement. One resource pilot
does not establish recognition quality, energy savings or state-of-the-art parity.
