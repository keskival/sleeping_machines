# Distinguish closed-gate gradients from usable new-depth plasticity

The direct legacy D2-to-D4 growth initializes appended unit gate biases at
-20. For v=x+alpha*y*sigmoid(G*gelu(y)+b), both the nonlinear correction and
its gate derivative can be near 2e-9 times their ungated values. A strictly
positive gradient is therefore not a quantitative plasticity contract. A
small correction can also disappear when added to a larger FP32 input even
while autograd supplies its mathematical derivative. Added channel maps,
private writes and computational clocks retain separate active paths; a
closed value gate is not equivalent to absence of all extra computation.

Fresh Adam updates a coordinate by -eta*g/(abs(g)+epsilon) after clipping.
For abs(g) much greater than epsilon, gradient scaling nearly cancels. For
abs(g) much smaller than epsilon, the displacement is instead approximately
-eta*g/epsilon. Report the actual active-coordinate attenuation fraction
abs(g)/(abs(g)+epsilon), realized FP32 displacement and storage rounding,
rather than assuming either complete cancellation or completely frozen gates.

The fixed source-valid fine-packet D2 producer is
`local_dvs_clock_full_20261002T153000Z`. Use its selected best_state with
deliberately FRESH Adam, not its terminal moments. Construct legacy direct
growth to p16/D4/H2/pool2 with the original seed and old gains. Four fixed
initializations share every weight except appended gate bias: -20 (legacy),
-8, -4 and 0. The latter three are diagnostic interventions that change the initial
function and do not establish a near-identity repair. No outcome selects an
arm or nominates a long fit.
The -4 arm was added before numerical execution as a prespecified gate-only
counterpart to the other host's new section410 skip-init proposal. It keeps
the trained shallow parent and the legacy transport construction, so it does
not reproduce that full scratch-trained skip-init protocol. It is not a
competing initialization fit.

Use only original fine-packet FIT indices0..15 and the parent's exact
fit-only transform. Do not read DEV/test arrays. For each arm use factorized
episode races and fixed shared noise171323, the same16-example mean loss,
one clip1 operation and one actual fresh Adam update at .003. Measure before
and after gate preactivations/activations/derivatives, actual candidate and
selected proposals minus incoming FP32 messages, gate-bias changes, per-layer
and parameter-block normalized raw/clipped gradients, epsilon attenuation and
actual displacements. Keep inactive padded lanes out of activation statistics.

Instrumentation intercepts torch.sigmoid only for the native four-dimensional
gate tensor, reads channel-mix outputs through removable hooks and forwards
every LaneRace call to its unchanged implementation. Check instrumented
logits and EVERY normalized gradient bitwise against a plain backward in each
arm. Check the fresh-step formula with a coordinate FP32 storage bound,
caller RNG restoration, kernel-function restoration and original parent/base
weight/artifact/source immutability. Save full gradient/displacement vectors.

Finite SAME-FIT loss/KL is a diagnostic of these four isolated steps, not a
learning curve or generalization result. Changed winners can affect finite
loss. Each arm starts from its own fixed intervention; cross-arm initial
quality differs. Total64 fitting targets/four optimizer updates does not
include additional plain correctness backwards and telemetry forwards as if
they were free. Campaign wall/RSS includes them; audit FLOPs/traffic/energy
remain unknown, not zero. The original selected parent quality/work is retained.

Run only under one unique bounded safe queue, one CPU thread,3GB virtual/
1.25GB RSS caps,8GiB available reserve and180-second timeout. Expected workload
is a few tiny episode forward/backward passes, not a model-fitting campaign.
Only a completed result can establish whether closed-gate branch contributions
vanish or whether epsilon materially attenuates the new parameter steps.
Progressive-gain continuity is addressed separately by frozen note121.
