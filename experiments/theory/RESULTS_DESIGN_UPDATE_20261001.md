# What the completed results change, 1 October 2026

This empirical update applies notes 43, 48, 51 and 52. Outcomes are measured;
causal interpretations remain hypotheses. These single-seed results do not
establish frontier supremacy.

## Useful depth; additional paths must earn their cost

The older width-200 race network on the depth-three Random Hierarchy Model
reaches 70.12%, 83.72%, 85.06% and 84.12% at depths 1–4, with 64K fitting
examples and ten passes. Residual-2 variants at depths 2–4 reach 72.72%,
74.54% and 72.56%. Completed provenance is under
`experiments/results/aws_20260929/`; the report visualizes these together.

This supports compositional depth on that task, but not automatically adding
residual interactions in this implementation. Residual-depth4 is still
improving: slower convergence, interference and conditioning are competing
explanations, not established causes. The task itself has depth three; this
is not evidence that language models should have only three blocks.

Retain the integrated model's eight blocks. Before another path substitution,
measure content survival, gradient reach and conditioning, then compare
matched-budget convergence. More paths are useful when they produce learnable
task-relevant directions (§345), not merely more parameters.

## A promising native quality/work tradeoff

The native d16/H2/eight-block language2K pilot reaches 3.764712 bpc with
3.778244 whole-fit GFLOPs. The saved d32/H2/eight-block KV2K construction
reaches 3.732586 bpc with 22.753030 GFLOPs under the same four-pass fitting
and frozen development protocol: 6.02x less work at 0.032126 bpc worse.
Sources are the completed `local_native_language_D2048_H2_d16_depth8_s6_20261001T174000Z`
and `local_episodic_pair_heads_H2_D2048_d32_depth8_s6_b64lr002_20260930T231500Z`
JSONs under the native_language and episodic_language result directories.

Width, capacity, sharing and history organization differ: this supports the
whole native construction, not an isolated attention-removal explanation.
Neither score is near frontier quality. AWS sparse32K's 3.106041 bpc uses an
older six-block construction, so it is not another point on this native curve.

Prioritize shared content/state learning and measure the quality/work frontier
with more data. Preserve historical KV attention as a comparison rather than
assuming it is the best use of temporal computation.

## Non-redundant temporal functions; fitted benefit pending

Notes 51–52 derive content-dependent rotating reception, differentiable finite
integration windows, and repeated spikes with information absent from an
identical first spike/payload. Numerical contracts passed. This establishes
useful function classes and local trainability under the stated smooth-region
assumptions; better fitted predictions remain untested.

R2/R4 reception pilots compare the native base and a refitted waiting-only
control. Uniform versus late-block allocation holds the total clock budget
fixed. Additional temporal machinery must pass the declared quality/work gate
before larger promotion. Window and spike-train primitives are not integrated
into these language fits. Birth/deletion and hard-route boundaries still need
the specified credit; interior differentiability alone is insufficient.

## Dormant inference state is not free training memory

AWS's unchanged source64/payload8 contract peaked at 2,585,864 KiB, exceeding
the original guard; a unique measured probe passed with a larger cap. This is
a workload-envelope finding, not a quality or architecture failure.

Track available state, scored candidates, selected commits, counterfactual
graphs and optimizer exposure separately. Do not provision a wider source64
fit locally from sparse inference activity alone. Bounded eligibility and
recomputation are future hypotheses to compare against exact small gradients,
not reasons to modify frozen running code.

## New completed pilots, 22:18 UTC

Seven AWS recovery pilots and the local R2 language2K fit are now complete.
All AWS screen points are seed6, four passes, payload8/eight blocks/two heads.
Event fits have 128 queries per pass and 64 development queries; language
has 512 fitting characters and 1,023 development targets. These are much
smaller protocols than the earlier successful native order pilot. Result
tags begin `aws_fast_matrix_recovery_20261001T213409Z_`.

**Extra clock reception has not earned promotion yet.** Local R2 reaches
3.795698 bpc versus native 3.764712, with 4.032017 versus 3.778244 whole-fit
GFLOPs: .030986 bpc worse and 6.72% more work. Inference is .105568 versus
.097376 MFLOPs/character (8.41% more). Both curves still improve at pass four.
This constrains uniform R2 at this budget, not all temporal features. R4,
late allocation and the selected waiting-only control remain pending. Do not
extrapolate late improvement to eventual superiority or alter frozen fits.

**State contributes; elapsed-time benefit is not isolated.** Timing accuracy
is 78.125%, falling to 54.6875% when state is cleared. However the independently
refitted rank-time control reaches 79.6875%, slightly above observed timing.
The observed-time model falls to 70.3125% under a frozen rank-time intervention;
that is sensitivity, not proof that elapsed-time information is necessary.
The narrow synthetic gap distribution and content may allow approximate
prediction without exact ages. A future timing protocol should vary ages more
widely and include paired identical marks/order with distinct timing labels,
then confirm with refitted controls and independent seeds.

**Order memory is sensitive to irrelevant gaps.** Order accuracy is 51.5625%
versus 28.125% after state clearing. Stretching gaps by 8 or 64 leaves the
order labels unchanged but reduces accuracy to 29.6875% or 26.5625%. A useful
design hypothesis is separate stable/order-sensitive memory and elapsed-time
channels with learned coupling, rather than forcing every content direction
to decay/rotate. Input clock precision, conditioning and fitted time ranges
must also be diagnosed. This is not a reason to discard temporal computation.

**Counterfactual credit is not yet buying quality in this small screen.**
Refitted order pathwise credit reaches the same 51.5625% accuracy; NLL is
1.028594 versus counterfactual 1.020832. Counterfactual work is .265264 versus
.262997 GFLOPs. The local teacher's small NLL advantage does not establish
improved accuracy or general uselessness. More learning and route-utility
diagnostics are needed before claiming a benefit or removing this mechanism.

**Capacity/activity separation survives; learning exposure becomes limiting.**
At 4/16/64 occupied sources, available receivers are 128/512/2,048 while
commits stay 16 and scored keys 32 per event. Whole-fit work rises only
.265/.280/.341 GFLOPs, but accuracy declines 51.56/48.44/32.81%. The fixed
128-query pass exposes each source only 32/8/2 times, and separate source
parameters grow 42,932/157,940/617,972. Shared learned maps with private state
are therefore a worthwhile future matched comparison; the current points do
not isolate whether exposure, capacity or optimization causes the decline.
At sources64, development has only one independent population: its bootstrap
interval collapses and provides no meaningful population uncertainty. Increase
independent populations before making a capacity-generalization claim.
