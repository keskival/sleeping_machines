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
