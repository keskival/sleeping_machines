# Coverage budget: measured tradeoff and a fixed-state allocation derivation

## Actual admitted work, not a quality win

136 local_trained_route_coverage_work_20261003T054700Z completes SIX groups,
60.076030s/362876KiB. FIVE actual EXISTING BL fitting windows on SAME
132 source-bound trained p4D4 weights/12-step Adam moments, original
FIT0..3/B4, first134noise fixed, original sampler RNG. Factorized/no-choice,
sampled k1/8/32 and all168. EVERY traced/untraced next parameter AND Adam
state agrees; serialized full recovery agrees. Factual CE identical across
all arms and exactly132's corresponding float32 CE. All11updates discarded.

| Credit/site count | Actual step GF | Fit MF/presentation | Native inference MF/target | Traced shadow lanes |
|---|---:|---:|---:|---:|
| Factorized, no choice | .002028565 | .50714125 | .162245333 | 0 |
| k1 | .003327802 | .83195050 | .162245333 | 8 |
| k8 | .012406626 | 3.10165650 | .162245333 | 64 |
| k32 | .043533818 | 10.88345450 | .162245333 | 256 |
| k168 | .219918762 | 54.97969050 | .162245333 | 1344 |

Each row4presentations/one actual clip1 historicalAdam.003 update; SAME
12-target native inference boundary, one-special-function-one convention.
2379parameters,16available private receivers,168selected/336keyscores per
presentation unchanged. Full losing returns still evaluate keys/values and
backward/normalization/clipping/optimizer work is charged. No-choice changes
the mean teacher and is NOT k0 of the same route-credit variance curve.

| k | 134 estimated raw variance ratio | Actual step work ratio | Product heuristic |
|---|---:|---:|---:|
| 1 | 1 | 1 | 1 |
| 8 | .160450686 | 3.728174332 | .598188128 |
| 32 | .070498973 | 13.081853428 | .922257237 |
| 168 | .046226289 | 66.085290531 | 3.054877751 |

k8 is the best MEASURED variance-work heuristic point here, about40.2%
better than k1. It is not a prediction/Adam/convergence or benchmark win:
134uses four-history DOUBLE projected covariance;136uses one-history
actual FLOAT32 work and just one fork per configuration. Local FIT/anchor
NLL changes no-choice/k1/k8/k32/full are respectively -.040751/-.013688,
-.038340/-.011477, -.039690/-.006799, -.040479/-.015607,
-.040739/-.023932. These adjacent FIT anchors are not IID/DEV/test and
are not used for selection. The lowest one-fork anchor loss is full, while
the heuristic favors k8: preserve both, do not convert the heuristic into
a quality claim or select by these five local scores.

Traced5updates, exact untraced5, serialized full1:11discardedupdates.
Traced1672fullshadowlanes/35112events, verification3016lanes/63336events,
extra inference/admission/evaluation72predictiontargets. Total campaign
FLOPs/traffic/energy unknown, not zero. Work table is actual traced STEP
work, not full diagnostic campaign. All numerical/protocol/result sources
freeze at completion; no active AWS driver/queue replaced.

## What the variance-times-work objective actually guarantees

At FIXED theta, let G_k be unbiased for the SAME DECLARED preclip native
teacher mean, with independent draws and constant known cost C(k).
Averaging n such vectors gives covariance trace V(k)/n. For compute budget
M and continuous approximation n=M/C(k), trace=V(k)*C(k)/M. Thus minimizing
V*C improves fixed-state Monte Carlo estimation under those assumptions.
This is not a theorem about moving theta, hard-boundary whole expected
risk, nonlinear clipping/Adam, random cost stopping, quality or convergence.
Our finite empirical estimates also carry uncertainty. A cost-correlated
stopping rule is not automatically an unbiased fixed-n average.

For all rows sharing R and site variance V1, exact finite-population scaling
and law of total covariance give

    V(k) = V_history + V1*(R-k)/(k*(R-1)) = a + b/k,
    a = V_history - V1/(R-1), b = V1*R/(R-1).

Suppose actual cost is affine C(k)=C_fixed+c*k. Then

    V(k)*C(k) = a*C_fixed + b*c + a*c*k + b*C_fixed/k,
    d/dk = a*c - b*C_fixed/k^2.

For a>0,b>0,c>0,C_fixed>0, continuous interior optimum

    k_star = sqrt(C_fixed*b/(c*a)), constrained to [1,R].

Check nearby integers and active bounds. If a<=0 the objective decreases
on the legal interval, so the heuristic favors full support. Negative a
is allowed by finite-population algebra; it is not negative actual V(k).
More support is not always optimal when the history floor is positive.

134 estimates R168,V1=7.046586,V_history=.341525 give
 a=.299329828,b=7.088781128. The k1/k8 measured-cost line gives
 C_fixed=2.030827143MF,c=1.296974857MF per extra site/episode.
It predicts k32/full work within .00047%/.00175% of measurements, but is
an approximation, not a byte-exact accounting formula. Interior k_star
6.089498; k6 was NOT run or validated. k8 is the best tested heuristic
point. Do not start a quality sweep or claim a learned scheduler from this.

## Heterogeneous memory/route coverage and adaptation

For rows with legal R_j>1 and independently sampled k_j, let
 S_j=sum_r||v_jr-mean_r(v_jr)||^2 and b_j=R_j^2*S_j/(R_j-1).
Then V=a+sum_j b_j/k_j, a=V_history-sum_j b_j/R_j.
R_j=1 has zero site variance and mandatory single-site cost; fold that
cost into C_fixed. With C=C_fixed+sum_j c_j*k_j, a>0 and interior choices, write
 k_j=t*sqrt(b_j/c_j), D=sum_j sqrt(b_j*c_j). The objective becomes
 (C_fixed+t*D)*(a+D/t), minimized at t=sqrt(C_fixed/a), hence

    k_j_star = sqrt(C_fixed*b_j/(a*c_j)).

This is direct variance/allocation calculus, no novelty or convergence
claim. A fixed-cost constrained objective gives the same square-root
relative allocation via Lagrange multipliers. Respect bounds [1,R_j] and
recompute effective constants when coordinates saturate. More variable
and cheaper credit receives more coverage; storing more memory alone does
not say which addresses are useful. Candidate discovery/cost remains paid.

A larger initial pool then shrinkage is justified ONLY if the measured
site-noise/cost versus history-floor ratio evolves that way. Depth, learning
stage, route support and memory uncertainty can move the optimum in either
direction. Choosing k before drawing a uniform subset preserves its full-
site conditional mean. Adaptive stopping based on that same subset's
observed returns generally does NOT preserve R/k unbiasedness. Use an
independent pilot not reused in the main estimate, or properly derived
inclusion probabilities/credit, charging pilot work. No policy is implemented
or admitted by the calculus alone.

## Research decision and report provenance

Do not promote expensive conditional-content averaging:133 only0.36%
less actualAdam variance for75.5% more work. Instead, the next specific
optimization diagnostic is an ACTUAL warm-Adam coverage ensemble at the
same trained state, preserving native physical clocks/races/private values
and measuring fixed FIT effects+cost. Distinguish raw-gradient estimation
advantage from a practical learned-model advantage. Existing completed
full-data deep controls and pending source-owned quality runs take precedence
over another duplicate p4 fit or unlimited k sweep.

Preserve AWS corrected private/shared native full replay10M and original
teachers/factorized controls. Other-host v2c compiled layer steps and
live-gate/capacity experiments are separate protocols/sources; do not alter
active frozen modules. Compilation improves implementation throughput;
selected-write counts alone do not prove zero candidate/optimizer cost.

The new report preserves all earlier pages and raw-work/quality references.
First055000Zpublication failed an orphan-footer guard (page204); previous
report restored. Exact failed producer/publisher/source and queues/logs
retained under archive/failed_runs/trained_route_report_20261003T055000Z;
failed PDF stays at .git/report-validation. Shorten text only, new unique
055400Zretry, numerical results unchanged. Successful publication details
are recorded in LOCAL_HANDOFF; producer/publisher freeze only on success.
