# Corrected replay critics: two bounded variance screens fail

Sixteen guarded law/clock/critic/lane/core tests pass (34.73s). AWS independently
checks a phase2 prerequisite; other host owns phase0 and all-race shadow fits.

| Encoder | Seed | Critic k1 / plain k4 MSE | Critic k2 / plain k4 MSE | Mean heldout critic R² |
| --- | --- | --- | --- | --- |
| Initial fine | 7 | 4.218306 | 2.083742 | -.021707 |
| Initial fine | 8 | 4.213718 | 2.081475 | -.035445 |
| Trained coarse | 7 | 5.129628 | 2.429824 | -.174411 |
| Trained coarse | 8 | 4.745302 | 2.247774 | -1.209840 |

Every nomination fails. Fixed width32/GELU critic,100 Adam updates at.003,
first4 FIT examples training,next4 heldout,seeds7/8. Corrected first-time actual
write replay enumerates all alternatives. Initial fine84 races/prefix;
trained coarse20. Critics frozen before subset evaluation. Current driver
features: scores, probabilities, value norms, centered-value norms, position,
depth. Trained producers already saw all FIT labels; only critic training
is held out. These are separate scopes, not matched producer learning fits.

Exact conditional concatenated score-coordinate MSE:
`(R/k-1) sum_r ||g_r-q_r||²`. Exhaustive R4 contracts prove mean and variance
for k1/2/4. Shared parameter-gradient variance also needs cross-site covariance;
this screen does not measure it or predict quality. Unbiased correction alone
does not establish useful variance reduction or disprove replay credit.

Initial1.903s/443592KiB,trained1.033s/433184KiB. Paid replay lanes2688/640,
200 critic updates per screen. Full diagnostic FLOPs unknown,not zero.
Saved critics/features/utilities and both JSON ledgers retained. Unique guarded
queues,2GiB RSS/6GiB virtual,8GiB available floor,600s timeout,single CPU thread.
No development quality evaluation or official test. Fixed protocols retained.

Next gap: norm-only features omit signed messages and supervised query context.
Utility may differ when these features agree, creating irreducible conditional
variance. This is a candidate explanation, not measured causality. Richer
critics need separately fixed features/conditioning and heldout variance
contracts before reduced-replay long fits. No retuning these failed screens.

## Distinct signed-message/label-aware critic also fails

A separately frozen feature hypothesis adds candidate values, centered values
and supervised labels (learning only). First32 FIT rows train, next32 holdout;
same width32/100 updates/.003, saved trained coarse producers seeds7/8.
Critic k1/plaink4 MSE4.524748/5.263726; k2/plaink4 2.143301/2.493344.
Both nominations fail. Mean per-prefix R²−15.173313/−209.190677; tiny-return
prefix denominators make that mean sensitive, so aggregate route-score MSE
is the primary gate. Feature addition alone does not solve conditional utility
prediction in this bounded fit. This does not prove feature insufficiency is
or is not the root cause. No learning-quality claim or reduced-replay fit.
3.415s/435624KiB,5120 replay lanes/200 critic updates, total FLOPs unknown.
Artifacts and immutable feature protocol retained alongside previous failures.
Next useful diagnostic is utility scale and heldout calibration against zero,
with training-only shrinkage and shared-parameter covariance measured before
further critic training. Other-host corrected replay quality gate still governs
any long campaign.
