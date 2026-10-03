# Actual-batch projection results: sampling noise persists, no fit promotion

127 completed four contract groups in56.096s/629624KiB. At each initialized
depth2/4/6, THREE fixed parameter-sign directions reproduce EVERY race's
exact124 vector dot product. Maximum discrepancy is1.38e-14. Thus the
artificial-cotangent mixed derivative implements the actual declared native
score Jacobian, including temporal/factual-history paths; it is not a
local-score substitute or a claim about physical stochastic Hessians.

| Depth/batch | k | Combined-gradient MSE ratio | Empirical SE | Exact versus estimated |
| --- | --- | --- | --- | --- |
| D4/B2 | 8 | .084633 | not applicable | exact124 |
| D4/B16 | 8 | .080239 | .003471 |32 parameter-sign probes |
| D4/B16 |32 | .017051 | .000738 |32 parameter-sign probes |
| D6/B2 | 8 | .305063 | not applicable | exact124 |
| D6/B16 | 8 | .147045 | .006484 |32 parameter-sign probes |
| D6/B16 |32 | .033145 | .001462 |32 parameter-sign probes |

All rows are same initialized seed7/p16/H2/pool2/fine21-event representation,
with fixed factual history noise171323. B16 uses FIT0..15 with original
FIT-only normalization; B2 uses0/1. These are different deterministic
batches, not an IID resampling proof. No new optimizer step or quality fit.
Empirical SE is descriptive, not a guaranteed95% interval. Worst-case
relative RMS bound for32 signs is25%; all probe values are saved.

Raw k8 relative noise RMS from the two B16 estimates is approximately
.283/.383. Sampling variance is therefore not automatically negligible
at the ACTUAL batch size in this initialization history. D4's denominator
and trace both shrink relative to B2, so the relative ratio scarcely
changes; D6's ratio about halves. Neither result proves why trained models
fail, estimates Adam variance, measures useful feature depth, or validates
a sampling-budget fix. Larger k32 lowers raw variance, while125 already
shows larger budgets do not consistently improve finite anchor loss.

This stage executes13440 full alternative shadow lanes/282240 shadow
events ONCE,64 main projected pullbacks plus9 exact-bank contract
pullbacks, factual/full gradient backprops and mixed-graph construction.
No comparison with124's292.6s as an equivalent-program speedup: the
workloads/batches/output guarantees differ. The new method makes an
otherwise expensive actual-batch diagnostic affordable. Total diagnostic
FLOPs, traffic and energy remain unknown, not zero; no inference saving.

The randomized-trace primitive is established, not our invention:
[Avron and Toledo, Randomized algorithms for estimating the trace of an
implicit symmetric positive semi-definite matrix](https://www.cs.tau.ac.il/~stoledo/Bib/Pubs/randomized-trace.pdf)
analyzes such estimators and their convergence. Our construction applies
it to finite-population conditional route covariance, with score-Jacobian
mixed products and exact native per-race contracts. The formulas and this
application are derived in127; no novelty priority claim.

Decision: preserve sampled-credit risk as a concrete, INITIALIZATION-only
diagnostic, but do not displace owned live-gate/gain-preserving integrated
comparisons or streaming AWS10M with an ungated sampling sweep. Next
learning decision requires completed matched quality and full-FIT/common-
epoch, lesion/equal-pass continuation where available. Unresolved losing-
content exposure, route changes and long-horizon credit remain separate
mechanisms. The newly corrected DEV-only90M selection rule protects the
eventual comparison; no pending quality cell is filled with these probes.
