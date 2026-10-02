# Training-only ordinal replay priorities fail heldout variance

First32 FIT prefixes determine90% sqrt(mean site-score energy) plus10%uniform.
Freeze before next32 FIT rows, saved trained coarse producers seeds7/8.
k1/k2 importance draws WITH replacement; correction1/(kp) preserves multiplicity.
Positive floor and exhaustive draw mean/variance contract pass. Baseline is k4
uniform WITHOUT replacement, sampling distinction retained.

| Seed | k1 / plain k4 conditional score MSE | k2 / plain k4 conditional score MSE | Expected unique sites at k2 |
| --- | --- | --- | --- |
| 7 | 4.149312 | 2.074656 | 1.936188 |
| 8 | 6.106031 | 3.053016 | 1.929643 |

Both gates fail. Expected unique sites alone is not work/quality advantage.
No producer update, new replay, DEV or test access. Original producer/cache/
critic costs paid; additional arithmetic FLOPs unknown. Ordinal-site structure
alone does not reliably predict heldout route utility. Do not scale unchanged.
Feature-conditioned magnitude priority is a separate frozen hypothesis; oracle
magnitude priorities require all expensive returns and are diagnostic only.

## Feature-conditioned magnitude prediction: useful signal, gate still fails

Fixed64-tree depth6/minleaf8 predictor trains on first32 FIT log route norms,
using both candidates' signed/value/label-aware features. Next32 heldout,
90%normalizedprediction/10%uniform, k2 WITH replacement, exact correction.
Seed7 MSE1.206815× plaink4,seed8 1.519582×: BOTHFAIL. This is better than
uniform WITH replacement2.375× and ordinal priorities2.074656/3.053016×,
but does not yet justify the proposed half-replay budget.

Diagnostic oracle p proportional exact route norm yields.631115/.519480×.
It needs EVERY expensive replay utility, so this is a mathematical lower bound
showing remaining allocation headroom, not a deployable advantage. Do not report
its numbers as reduced-work training. Saved models/proposals/cases retained.
Conditional score-space scope only, no parameter covariance/quality claim.

## Three distinct weighted sites: fresh FIT score gains, parameter gate fails

Prespecified fresh critic-heldout FIT64..95, frozen predictor trained0..31.
Sequential weighted WITHOUTreplacement, exact enumerated marginal/pair
probabilities, correction1/inclusion. Same positive90/10proposal. Exhaustive
mean/shared-vector variance/uniform nesting contract passes. Six proposed lanes
versus eight baseline; diagnostic actually enumerates40lanes/prefix.

| Seed | k3 weighted / uniform k4 score variance | Uniform k3 / k4 | Parameter index64 ratio | Parameter index65 ratio |
| --- | --- | --- | --- | --- |
| 7 | .768144 | 1.416667 | .600428 | 4.011873 |
| 8 | .693763 | 1.416667 | .639390 | 1.009399 |

Positive23.19%/30.62% score variance reductions on fresh prefixes are supported;
retain them. Strict combined nomination FAILS due parameter cases. This is
conditional score/route variance, not quality or supremacy. Labels have been
seen by producers; only priority supervision is held out. Exact inclusion
computation averages .742/.747ms/prefix; tree/discovery/VJP costs additional.
3.578s/518380KiB,2560actual diagnostic shadow lanes and80confirmationVJPs.
No unchanged integratedfit. Next distinct hypothesis directly predicts
parameter route norm on TRAIN instead of score norm; freezes before another
fresh critic-heldout block, with the SAME stringent gate.
