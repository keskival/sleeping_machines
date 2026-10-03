# Preserve actual residual gains through progressive DVS growth

The frozen `dvs_grow_depth_benchmark.py` copies parent tensors but `gain` is a
Python float absent from `state_dict`. It restores every old layer to
`0.5/sqrt(parent_depth)`. Direct D2 to D4 is consistent: the old two layers use
0.35355339059327373 and the new two use 0.25. Growing this model to D6 resets
the oldest two layers to 0.25, a 29.2893% decrease in residual amplitude. This
changes the parent's computation before learning and confounds a progressive
growth interpretation. The completed D4 results remain valid for their
actual construction. Historical progressive results must retain their actual
reset, rather than being retrospectively assigned the intended gains.

The new sibling reconstructs each completed parent from recognized frozen
source bindings and its selected checkpoint. Ordinary clock-calibrated
parents have uniform gain. Legacy grown parents have exactly the gain vector
their legacy constructor assigned, including a reset when applicable. New
parents preserve the reconstructed actual parent vector and append only the
new depth's standard gains. Both result and recovery checkpoint record this
metadata and every ancestor result/checkpoint hash. Missing lineage, cycles,
unknown constructors, incompatible dimensions/clocks, changed model source,
mismatched selected checkpoint or contradictory metadata refuse admission.
The wrapper annotates each original-driver snapshot at its exact per-run
checkpoint destination, so a process interruption before the driver returns
still leaves the gain metadata. Other serialization destinations are unchanged.

For the corrected D2 to D4 to D6 construction the gain vector is

    (0.35355339059327373, 0.35355339059327373,
     0.25, 0.25, 0.20412414523193154, 0.20412414523193154).

No temporal or sparse mechanism is removed. The sibling retains closed new
gates, identity channel initialization, sparse private writes, computational
delay, inherited top transport and optional first-time-preserving replay
credit. Inference and learning work have the same operators as the legacy
growth arm. Artifact validation costs time and memory and is not reported as
zero work. Fresh Adam, shifted race draws, additional passes and extra depth
remain real changes; this repair alone is not exact parent-function or old
optimizer-step preservation and does not demonstrate benchmark advantage.

The existing growth contract merely requires one positive gate-bias gradient.
At bias -20, sigmoid is about 2.06e-9. This can strongly attenuate branch
credit, particularly relative to Adam's epsilon; nonzero gradient alone does
not demonstrate practical plasticity. The added channel and transport maps
still have independent live credit. Consequently growth accuracy does not
by itself establish that the new nonlinear message branches learned. Inspect
saved gate activations, branch contributions and realized parameter steps
before replacing closed-gate growth with a live-output identity construction.
Theory34 provides a previously tested live-output construction for a different
modal event encoder; transferring it to this receiver must retain temporal
and sparse contracts in a separately admitted sibling.

The standalone contract driver is to run only under one guarded, uniquely
named queue. Its tiny numerical checks cover direct D2-to-D4 bitwise legacy
nesting of logits, every gradient, state and caller RNG; heterogeneous old
gain/tensor preservation at D6; actual source-bound factory and checkpoint
lineage reconstruction; future D6-to-D8 reload; historical legacy reset
interpretation; invalid lineage refusal; and parent/source immutability.
It performs no optimizer updates or DVS fitting. Publication must use a
completed contract artifact, not this proposed protocol as evidence.

The required later quality comparison is corrected progressive growth versus
the frozen reset arm and a shallow continuation with equal total fitting
passes and explicit optimizer ownership, with gate contribution and full
fit diagnostics. Retain all old results and record this protocol correction
beside any legacy progressive-growth claim. This narrow repair does not
resolve generalization, route-credit variance, clamp saturation or temporal
Jacobian conditioning; those are independent diagnoses.
