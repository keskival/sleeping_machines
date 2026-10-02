# Bounded order task: learned-control result and exact scope

All twelve final scores and the prespecified analysis are complete on fresh4096-query seed4201. Both history widths reach100% accuracy in all declared training seeds. The initial six-control publication retained pending native cells; the completed numbers below now replace those pending labels.

| Model | Fresh mean accuracy | Fresh mean NLL | Fit GFLOPs | Fit MFLOPs/query | Inference MFLOPs/query |
|---|---:|---:|---:|---:|---:|---:|
| native_private | 0.976074 | 0.097618 | 1.324095 | 0.646531 | 0.106736 |
| native_shared | 0.991374 | 0.062436 | 1.034232 | 0.504996 | 0.106743 |
| history32 | 1.000000 | 0.831460 | 0.006263 | 0.003058 | 0.001034 |
| history128 | 1.000000 | 0.181817 | 0.024799 | 0.012109 | 0.004106 |

Whole fits include all32 Adam steps, backward, clipping, feature arithmetic and loss. Numerical-recovery steps are separate prerequisite work and included in measured whole-job wall. Special functions remain separately saved. Both decoders use the declared three-mark bound; no race/counterfactual mechanism is present. Native historical fits remain~1.034GFLOPs(shared)/~1.324GFLOPs(private) and are not reclassified as zero-cost because weights are reused.

History controls achieve perfect classification at much lower work, but their NLL is worse: native predictions are more confident under this distribution. There is no quality dominance across both metrics. This prevents a broad resource-advantage claim on this bounded task. Preserve the strong native capacity result and completed timing/shared comparisons as mechanism evidence with this limitation beside them. No retuning from fresh scores or replacing old evidence.

The native intra-family gate also fails: shared-minus-private mean accuracy is+1.52995pp, adjusted98.33% interval[−1.36719,4.61426]pp, exceeding the permitted−1pp lower-bound deficit. Its fitting arithmetic ratio is.7810–.78115 (about21.9% less work); the mean resource/quality improvement is preserved, but the predeclared uncertainty gate is not met. Three seeds remain limited evidence. No reinterpretation of the margin after results.

## Why this task permits such a cheap control

The generator labels the signs of the second and third marks: y=2*1[m2>0]+1[m3>0]. Let classes0..3 correspond to sign pairs(−,−),(−,+),(+,−),(+,+). Class scores s_y=a_y*m2+b_y*m3 select the correct class whenever both marks are nonzero, because each coefficient independently maximizes its contribution. Thus four linear scores over two retained values can decode the exact task; depth8 and route discovery are not necessary for this labeling function. This is a derivation about this generator, not a theorem against the research architecture.

For binary marks, two sign bits per observed source are sufficient after the latest mark, with a causal two-slot shift on each mark. Uniform independent class labels also carry two bits/source: exact recovery of all S labels requires distinguishing4^S assignments, or at least2S bits of label-bearing state under finite-bit lossless semantics. Slot addressing/control metadata and physical execution add costs; this information bound is not a free implementation. The observed source ID already supplies lookup addressing.

The next advantage comparison should therefore stress evidence access, learned retrieval, variable context or useful joint-feature credit beyond this two-bit sufficient statistic. Other hosts own current joint/retrieval learning experiments; do not duplicate their variants. Keep temporal races, sparse addressed updates and counterfactual learning explicit, and use controls with matching information and charged discovery. A handcrafted difficult task is still a mechanism test until real-data/strong-control evidence supports a broader claim.
