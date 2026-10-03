# Random-segment exposure and a discriminating future comparison

This is a source-bound combinatorial audit, not a new training result or an
architectural substitution. The active AWS 90M and depth-eight jobs are unchanged.
`analysis/aws_segment_exposure.py` reads no data or model; its exact two-draw
exhaustive fixture passes. Results are in
`results/diagnostics/aws_segment_exposure_20261003.json`.

The compiled language driver independently draws B segment starts per window
uniformly from M=N-S-1 positions, with replacement. Each draw trains on S next
characters. There are M-S+1 interior target positions covered by S starts each;
the two boundary bands have multiplicities 1,...,S-1. One possible final next
character target is unreachable under the present exclusive upper bound.
For K=B*W draws, linearity of expectation gives exact expected unique coverage:

    (M-S+1) * [1-(1-S/M)^K]
    + 2 * sum(c=1..S-1) [1-(1-c/M)^K].

This calculation does not assume independent coverage of neighboring positions.
It does assume the driver's independent uniform draws. Interior exposure is
Binomial(K,S/M), approximately Poisson at this scale, with mean KS/M.

| FIT range | Budget | Presentations | Expected distinct target positions | Expected fraction |
|---|---:|---:|---:|---:|
| 10M | 1 pass equivalent | 9,994,240 | 6,319,089 | 63.191% |
| 90M | 1 pass equivalent | 89,997,312 | 56,889,864 | 63.211% |
| 90M | 2 pass equivalents | 179,994,624 | 77,819,072 | 86.466% |
| 90M | 4 pass equivalents | 359,997,440 | 88,351,501 | 98.168% |
| 90M | 6 pass equivalents | 539,992,064 | 89,776,853 | 99.752% |

These are expectations, not realized coverage or unique-information counts.
Text repetitions, endogenous state/routing, optimizer order and reset context
all matter. No comparison with saved dense controls follows without auditing
those controls' sampling and schedules. In particular, this does not explain
away their quality lead or establish a loss improvement from broader coverage.

A future bounded integrated comparison could change only data ordering: random
starts versus a shuffled nonoverlapping tiling, at the same presentation and
optimizer budgets. It would retain races, computational time, separate keys and
values, sparse persistent writes, depth and existing linear route credit, with
unchanged segment reset. A full tiling reaches essentially all target positions
once, but its fixed segment boundaries change context-position exposure; a
prespecified offset or multiple offset seeds are necessary to study that
confound. It must not be silently substituted within an existing successful tag.

Before admission: inspect existing queues to avoid duplication; contract no
FIT/DEV/test leakage, exact target budgets, deterministic seed/cursor recovery,
all-target weighting and unchanged work accounting. Select by DEV only and keep
the reporting test untouched until the specified final evaluation. This is a
proposed comparison, not an admitted queue. Finish assigned AWS 90M arms first.

The practical hypothesis is that more useful exposure per paid presentation
may improve learning at fixed work. Repeated presentations may also benefit
optimization more than new positions; only the controlled experiment decides.
Neither outcome changes the architecture's expressivity or proves supremacy.
