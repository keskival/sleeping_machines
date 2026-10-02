# Bounded order task: learned-control result and exact scope

All six learned history-control final scores completed on fresh4096-query seed4201. Both widths reach100% in all declared training seeds. Native fresh scores and its intra-family gate are still pending at this publication; do not fill them from development.

| Control | Fresh accuracy, each seed | Fit GFLOPs | Fit MFLOPs/query | Inference MFLOPs/query |
|---|---:|---:|---:|---:|
| History32 | 100% | 0.006263 | 0.003058 | 0.001034 |
| History128 | 100% | 0.024799 | 0.012109 | 0.004106 |

Whole fits include all32 Adam steps, backward, clipping, feature arithmetic and loss. Numerical-recovery steps are separate prerequisite work and included in measured whole-job wall. Special functions remain separately saved. Both decoders use the declared three-mark bound; no race/counterfactual mechanism is present. Native historical fits remain~1.034GFLOPs(shared)/~1.324GFLOPs(private) and are not reclassified as zero-cost because weights are reused.

This prevents a broad resource-advantage claim on this bounded task. Preserve the strong native capacity result and completed timing/shared comparisons as mechanism evidence with this limitation beside them. No retuning from fresh scores or replacing old evidence.

## Why this task permits such a cheap control

The generator labels the signs of the second and third marks: y=2*1[m2>0]+1[m3>0]. Let classes0..3 correspond to sign pairs(−,−),(−,+),(+,−),(+,+). Class scores s_y=a_y*m2+b_y*m3 select the correct class whenever both marks are nonzero, because each coefficient independently maximizes its contribution. Thus four linear scores over two retained values can decode the exact task; depth8 and route discovery are not necessary for this labeling function. This is a derivation about this generator, not a theorem against the research architecture.

For binary marks, two sign bits per observed source are sufficient after the latest mark, with a causal two-slot shift on each mark. Uniform independent class labels also carry two bits/source: exact recovery of all S labels requires distinguishing4^S assignments, or at least2S bits of label-bearing state under finite-bit lossless semantics. Slot addressing/control metadata and physical execution add costs; this information bound is not a free implementation. The observed source ID already supplies lookup addressing.

The next advantage comparison should therefore stress evidence access, learned retrieval, variable context or useful joint-feature credit beyond this two-bit sufficient statistic. Other hosts own current joint/retrieval learning experiments; do not duplicate their variants. Keep temporal races, sparse addressed updates and counterfactual learning explicit, and use controls with matching information and charged discovery. A handcrafted difficult task is still a mechanism test until real-data/strong-control evidence supports a broader claim.
