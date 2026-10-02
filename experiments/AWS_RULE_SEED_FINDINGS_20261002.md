# Completed rule/source-seed isolation

| Shared rules | Common seed | Seed6 accuracy | Seed7 | Seed8 | Mean | Fit GFLOPs mean | Fit MFLOPs/query mean | Inference MFLOPs/query mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| False | False | 0.4414 | 0.2578 | 0.4297 | 0.3763 | 0.2744 | 0.5359 | 0.1067 |
| False | True | 0.7539 | 0.5117 | 0.6914 | 0.6523 | 0.2743 | 0.5358 | 0.1067 |
| True | False | 0.4023 | 0.3203 | 0.2500 | 0.3242 | 0.2570 | 0.5020 | 0.1068 |
| True | True | 0.7539 | 0.4453 | 0.7031 | 0.6341 | 0.2570 | 0.5019 | 0.1067 |

Common seed gains private/shared: 27.604166666666668/30.989583333333336pp. Shared rules gain with private/common seed: -5.208333333333331/-1.822916666666663pp.

Interpretation: removing private source embeddings accounts for most of the combined quality improvement in this small-fit regime. Shared maps with private embeddings do not retain the original combined gain. Common-seed/private rules is strongest in mean accuracy here, while shared/common still reduces parameter/optimizer work. Preserve every cell; no best-seed selection or automatic supremacy attribution.

Same128 fitting queries ×4 passes,256dev, H2/d8/L8/P0/S16. Arithmetic2 FLOPs/MAC; specials separately charged in raw rows. Historical whole fits are included with512-query fitting denominator. Every event selects16 updates and scores32 candidates with512 available addressed receivers. Initializers differ across rule-sharing settings; these are exploratory architecture comparisons. The already observed confirmation set is not fresh evidence for these new cells.

Next capacity probe holds8 fitting queries/source/pass,4 passes and selected work/event fixed while increasing occupied sources16→64. Fit512 queries, dev256, sameU64/lr/d8/H2/L8; report larger populations and optimizer-window composition as protocol differences. Shared/common and private/common are both retained. No task quality is assumed invariant.
