# Language representation and credit: evidence before repair

The user's concern is whether learned language representations have lost useful
depth and reduced to surface counting. The current integrated model, the older
temporal carrier and the carrier/count composition are different constructions.
Their results should not be treated as a single model's version history.

## Completed evidence

* The older learned selective carrier at131K scores2.586650bpc. Resetting
  history at every character worsens it to4.519205; removing all memory
  corrections gives3.834631. Every layer's content/memory maps changed during
  fitting. This supports learned history dependence, not semantic abstraction.
  Source: `results/parallel_language/local_language_representation_20260930T162337Z.json`.
* The newer native model scores3.764712 at2K and3.557380 at8K, without count
  tables. Its completed contracts show all head-query gradients, exact optimizer
  recovery, causal predictions and correct accumulated-gradient normalization.
  Different data/capacity/history paths prevent interpreting the older131K
  score as a direct regression of this8K construction.
* Count-carrying native2K gives2.733599, but its untrained base/escape
  initialization already gives2.740653. The0.007054 improvement measures fitting
  the whole composed model, including escape parameters; it does **not** isolate
  the native representation's contribution.
* Carrier/count131K w32/w128 scores2.313495/2.313448. The tiny width gain is
  real evidence that added width has not earned its fitting cost in this
  composition. Deep content maps nevertheless move at every layer. Parameter
  movement alone does not establish useful depth. The count-composed checkpoints
  are absent in this checkout; claimed base-only8.17/11.34bpc in Theory§389
  have not been independently reproduced here.

## Interpretation corrections

Equal mean cross entropy does not identify a model's context dependencies or
representation. Theory§388's equivalent count order is a **quality calibration**,
not evidence that the native model computes only bigrams or uses nothing longer.
Models with different conditional predictions can have identical average loss.
Test dependence by keeping recent suffixes/noise identical and intervening on
earlier state; test useful dependence by scoring the intervention's loss change.
Neither test alone establishes semantic or hierarchical abstraction.

The composed base is trained on `p(y)=a_y+e*q_theta(y)`, not standalone
cross entropy of q. Its current logit gradient is

    responsibility_y * (q - one_hot(y)),
    responsibility_y = e*q_y/p_y.

After several escape levels, e is their product. A poor standalone q is
compatible with a useful conditional escape predictor; it is not independently
a disconnected-gradient diagnosis. Holding escape parameters fixed and replacing
the trained q with initialization/uniform is needed to isolate q's contribution.
Holding q fixed and varying escape parameters isolates their contribution.
Theory§389's missing count-conditioning path is an actionable hypothesis, not
yet an identified unique cause. Conditioning on count state may help but a
stream model can also retain/reconstruct relevant evidence.

If count state is hidden from the base, the optimum minimizes conditional
expected mixture loss. Its active-coordinate stationarity condition is

    E[P_y * e / (a_y + e*q_y) | base_features] = multiplier.

In general this optimum is not the arithmetic conditional expectation of each
count-conditioned residual optimum. Averaging and minimizing nonlinear loss
do not commute. Preserve the measured width failure while testing the proposed
information-path repair against alternatives.

## Bounded diagnostic and mitigation

`experiments/language_learning_audit.py` tests the actual current composition's
logit-gradient identity and all six layers' gradient reach, without an optimizer.
It also restores the saved old selective carrier and native8K checkpoints,
measures layer gradients on a warmed16-character window and frozen dependence
on1/2/4/8/16/64-character histories at32 fixed development positions.
Coupled per-position native race noise preserves identical suffix draws.
This diagnostic is intentionally small, with every replay/target/backward,
wall time and RSS charged. It is not the full saved development score.

The guarded one-job queue is inserted only after the active state-credit
baseline finishes. Only its coordinator is reserved; trainer and watchdog
continue. Finally resumes the coordinator, including audit failure. Audit
timeout300s, groupRSS1,250,000KiB, VMS3,000,000KiB and8GiB available-memory floor.
No model or active training source changes. Larger language fitting is not a
repair for an unmeasured credit fault.

The other thread's prepared count-message/escape-gate modules remain intact.
They add count-conditioned logits and per-position escape parameters while
retaining the native temporal/sparse/counterfactual core. This local readout
repair should not be described as demonstrated deeper recurrent representation
learning. Its existing combined queue must be split into one-job queues with
frozen contracts/smokes before admission. Do not duplicate the same fits or
launch it beside the active state-credit comparison.

Next comparisons: fixed-escape base replacement; count-conditioned versus
unconditioned base; warmed layer/history dependence; and matched tasks with
identical local n-grams but different long-range targets. Credit reach and
superiority remain separate experimental questions.

## Completed guarded audit, 02:38 UTC

`results/diagnostics/local_language_learning_audit_20261002T023510Z.json`
completed in33.212s /333,336KiB. All model weights are preserved; zero optimizer
steps. The actual composition's logit gradient agrees with the responsibility
identity within1.86e-9; all six content layers have nonzero gradients. Mean
responsibility on the64-target initialization probe is.034259, not a fitted
whole-corpus statistic. There is no general gradient disconnection in this test.

Both saved models receive gradients at every depth on a warmed16-character
window. Carrier depth norms range.419–.692, native1.027–3.487; these aggregate
different parameter groups and are not cross-model conditioning comparisons.
The first five carrier clock maps receive gradients; the final arrival is
unread by the fixed deep-message readout and its clock has no gradient.

Keeping the last16 input characters and their noise fixed while resetting
earlier native state changes the predictions: meanKL.022519nats from64-character
history over32 positions. Resetting to8 changes meanKL to.080978 and worsens
this slice's loss from3.218881 to3.358732bpc. This directly contradicts a strict
bigram-only dependency reading. At16, loss is3.203912, slightly better than64;
the audit does not establish a useful benefit beyond16, semantic features or
full-development quality. Carrier slice1/4/8/16/64 losses are4.463787/2.076002/
1.715860/1.732411/1.761144. Longer memory is not uniformly beneficial here either.

Correction to the initial clock interpretation (02:59 UTC): the earlier text
incorrectly said every fitted composed clock had zero displacement. The saved
JSONs' actual `diagnostics[].parameter_change_norms["clock.weight"]` show:

| Composed width | Layers1–5 clock-weight displacement norms | Layer6 |
| --- | --- | --- |
|32|1.1950,1.1041,.9710,.9678,1.2109|0|
|128|1.8651,1.8830,1.7446,1.7413,1.8982|0|
|256|2.0674,2.1272,2.2207,2.1328,2.1369|0|

Only the final clock is unchanged, consistent with its unused arrival time.
This removes the purported clock-learning discrepancy. The missing fitted
checkpoints still prevent new base-replacement interventions here, but the
JSONs themselves already establish clock-parameter movement. As with other
parameter movement, this does not prove useful representation depth.

Contemporary strength0 integrated order finishes at54.296875%/.978725879,
matching the earlier AWS parent54.296875%/.978725864. Its four development
epochs match the saved trajectory within floating-point tolerance. That is
positive regression evidence for this unchanged event-model/optimizer path,
not evidence of learned language abstractions. The guarded strength1 fit
started02:38:06 after the language audit; it remains pending.

## Credit-horizon theory and the next discriminating test

Truncation cuts parameter credit through old state even when that state is
retained for prediction. It introduces gradient bias; adaptive horizons and
compensated stochastic truncation are established ways to study that tradeoff
([Aicher, Foti and Fox,2020](https://proceedings.mlr.press/v115/aicher20a.html),
[Tallec and Ollivier,2017](https://arxiv.org/abs/1705.08209)). Neither result
proves our current short horizon is the measured bottleneck. Learnable local
updates can still learn occupancy schedules or reuse carried information.

For this substrate, compare warmed persistent-state gradients at16/32/64
credit targets with identical target suffixes and race noise. Separate extra
context from extra gradient reach: replay the same prefix without gradients
for all arms, enable the graph only for the chosen credit horizon, and measure
the gradient norm/direction and truncated-vs-longer differences. Bound cost
before any longer fit; do not silently convert this into dense all-history
training. Then use equal-local-n-gram long-range tasks to measure predictive
benefit, retaining addressed writes, races and unrealized-route credit. These
are proposed controlled diagnostics, not completed evidence or new queued fits.

## Completed credit-horizon audit and mitigation contracts, 03:05 UTC

`results/diagnostics/local_language_credit_horizon_20261002T030100Z.json`
completes in11.075s /530424KiB. The native8K checkpoint's predictions match
exactly across16/32/64 graph horizons for the same128-token context and16-target
loss suffix.16 versus64 gradient difference norm is30.887% of the64-token
gradient (cosine.956998);32 versus64 is15.145% (cosine.988989). All layers
receive gradients. Norms7.083/6.379/6.664 show why truncation should not be
interpreted simply as reducing gradient magnitude: omitted terms may cancel
others. This diagnoses a real credit-horizon effect on one frozen slice, not
proof of improved fitting quality, semantics or unbiased route credit. No
optimizer/weight change,384 forward tokens,112 graph tokens,3 backwards.

Full native H2/d16/depth8 count-repair contracts pass in13.678s /421616KiB for
CountMessage+EscapeGate, message alone and gate alone: exact zero nesting,
matching parent gradients, all head-query and enabled repair gradients, and
bitwise recovery of the next actual Adam update and persistent count position.
Source: `results/diagnostics/local_credit_followthrough_20261002T025250Z_contracts.json`.
This removes the missing recovery prerequisite; fitting still needs its own
accounting smoke and frozen one-job queue. The other thread owns those fits.

The integrated addressed-write comparison also completes:54.296875%/.978725879
versus55.078125%/.977812493, at.265264 versus.277294 whole-fitGFLOPs. The
.78125-point accuracy gain/.000913NLL gain with1.04535× work fails the
predeclared5-point/.02NLL/≤2× gate. Do not scale this variant automatically.
Historical stronger timing/shared-map results and the core thesis remain;
this small write-teacher variant has not established worthwhile improvement.

## Completed longer-credit fitting test

The matched four-arm2K comparison completes. Full16/64:2.695122/2.694298bpc;
minimal16/64:2.693828/2.693725. Full-core credit gain.000824bpc, with1.00490×
fitting work, and full64 is.000573bpc worse than minimal64. The nomination gate
fails; longer credit alone did not resolve the quality failure in this setting.
Source: `results/diagnostics/local_count_credit64_analysis_20261002T060000Z.json`.
All contracts/smokes/fit/recovery/provenance/ledger stages and guarded report
publication complete, preserving the original16-credit controls. No scale-up.

This revises the frozen-gradient interpretation: material omitted gradient
terms do not imply a material fitting benefit. Information sufficient for
long-range prediction, credit fidelity, exposure and useful learned computation
still need discriminating integrated tests; no global impossibility theorem
or single identified cause follows from this restricted negative result.

## Frozen trained-state attribution, 08:04 UTC

`results/diagnostics/local_deep_core_attribution_20261002T075000Z.json`
completes12.850s/522736KiB with no optimizer or parameter changes. Exact saved
full/minimal64-credit checkpoints and fitting/development hashes verified.
On32 development targets128–159, retain count tables, absolute cursor,
arrival timestamps and per-position race noise. Erase content once at128,
then permit ordinary state rebuilding. Dynamic gate may respond to changed q.

| Intervention | Full composed bpc | Minimal composed bpc |
|---|---:|---:|
|Intact|3.454094|3.490371|
|Erase carried source context|3.491624|3.488527|
|Erase receiver contents|3.449488|3.490132|
|Erase both|3.502234|3.488218|

Full recurrent content helps by.048141bpc here, principally through the
top-level context path. Receiver contents alone do not improve this slice.
Per-layer receiver interventions are saved; their small/nonmonotonic effects
do not show individual layers are useless, since source context survives and
memory rebuilds. A uniform base with the intact gate frozen scores3.505888
(full) /3.491742(minimal). This is evidence against an entirely inert full
recurrent predictor, not proof of semantic abstractions or a general full-core
advantage: the completed whole-development comparison still favors minimal.

Actual trained fitting targets64–127 have direct base responsibility mean
.071734(full)/.057129(minimal), median.004530/.003138. With D/theta frozen at
the actual gate's values, the output derivative matches r(q−onehot)/64 to
9.31e−10/4.66e−9. Dynamic gate output-gradient norm ratios to standalone q
are.206793/.195473; gate path difference norms.004102/.004259. Actual
dynamic-gate loss sends nonzero gradients to all8(full)/1(minimal) layers.
This establishes scarce direct residual credit for most targets, not broken
autograd. Uniform gradient rescaling can be partly compensated by Adam;
heterogeneous responsibilities, missing predictive information and sample
allocation remain distinct possible bottlenecks. The base is a residual, so
its5.14/5.01bpc standalone fitting scores do not imply broken learning.

## Proposed long-range protocol repaired before fitting

`results/diagnostics/local_long_range_protocol_20261002T075400Z.json`
completes.982s/271400KiB, three stream pairs each for lag48/induction128.
Lag dev targets include18–25 earlier cue symbols per~284targets; the asserted
uniform24 target law is false. Induction also contains inserted queries and
copied targets, with queries selected from observed pairs; iid fillers do not
prove all-orders suffix independence. Individual causal add-one count orders
1–8 score4.654–4.869bpc(lag) /4.598–4.821(induction); observed near-chance
controls are evidence about these samples, not a proof for every order.
Former per-target minimum over order losses used labels to select a predictor;
tests now evaluate each causal order separately. No such bound is evidence
for a deployable model. Only6.57–7.08% of positions are task targets; this is
objective exposure, not measured target gradient mass.

Old per-chunk Adam would give2048 updates(c16) versus512(c64) over8K×4.
Driver now uses target-weighted accumulation with independent U64, yielding
512 updates in either case and correct partial-window normalization. The old
multi-job queue is retired before any launch; its commands are preserved in
`archive/protocol_audits/curie_long_range_core_20261002T072000Z.pre_audit.txt`.
Use new tags/one-job queues for changed settings. Cold addressed-memory
evaluation is causal but lacks count references' preloaded fit memory; report
that difference explicitly. Driver still lacks full fitting FLOP/recovery
admission, so long fits await those prerequisites.

Full H2/d16/depth8 memory-repair contracts complete31.250s/450832KiB:
`results/diagnostics/local_deep_memory_contracts_20261002T080200Z.json`.
Both addressed and tapped models exactly nest native forward/parent gradients
at zero repair, preserve learned causal/chunk-invariant prediction, correctly
normalize actual gradients, and recover the next Adam update, predictions and
moments bitwise with new persistent state/RNG. The addressed read and write
maps both receive gradients after the initial read-map update; all tap maps
and some in-range unclamped delays receive gradients. Zero-read initialization
stages write-map learning, rather than immediately failing it. This is a
numerical prerequisite, not useful-feature evidence. State accounting now
includes context slots and tap buffers (integer/Python metadata separate),
and detach diagnostics include their differentiable entries.

Paired192-character/one-pass integrated accounting smokes complete with every
fitting floating operation traced; completed ledger
`results/diagnostics/local_deep_feature_preflight_20261002T081500Z.json`.
Native/addressed whole-fit .088376/.089829GFLOPs, per-target
.462702/.470307MFLOPs over191targets/3U64 updates, inferred work ratio1.01644.
CPU arithmetic includes unit-weight specials, loss/backward/normalization/
clipping/Adam; excludes dev, RNG, hash/integer metadata and traffic. Native/
addressed listed state2448/21648B, with150occupied extra slots of4096available.
Both initial scores match bitwise under identical core initialization; training
RNG is reset after constructors. Dev4.543183/4.538435bpc over128targets is
only an accounting smoke, not a quality pilot or useful-feature result.
Inference .097376/.101568MFLOPs/target is one traced target after127warm
tokens, not throughput. Original text-only empty filler-group NaN is undefined;
preserve those JSONs and this annotation. Driver now writes null for absent
groups and rejects other nonfinite JSON. Final25 tests pass11.79s, including
the real driver budget/partial-window and null serialization tests.

## Completed value-credit experiment and revised diagnosis, 10:16 UTC

Theory Note 63 and `results/diagnostics/local_value_credit_analysis_20261002T090800Z.json`
record five matched exploratory seed-6 fits: 1,024 fitting characters, four
passes, 4,092 targets, 64 updates and 2,047 development targets. The standalone
late-projection full core scores 3.942092 bpc against its same-width shallow
control's 4.086241, a .144148 improvement. Its minimal control gives 4.539614.
Thus added core computation earns quality in this small uncomposed setting;
the earlier count-mixture near-equivalence does not generalize to all models.
Different-depth initialization and unequal work prevent interpreting this as
isolated semantic depth or iso-FLOP superiority.

Original addressed memory scores 3.902970 against native full's 3.968133,
a .065164 bpc improvement for about 1.65% additional estimated fitting work.
Late projection restores a verified fixed-feature derivative through detached
memory but loses .039123 bpc to original. Its predeclared gate fails. Frozen
fitting loss is also worse (3.727163 versus 3.659509), so an explanation based
solely on excess development overfitting is insufficient. Extra fit replay
history hurts late full by .038790 bpc. These observations preserve useful
memory evidence while rejecting the projection repair as a demonstrated
quality fix under this protocol. The historical nonlinear feature producers
remain truncated; no generic disconnected-gradient regression was identified.

The frozen followthrough audit changes no weights and takes zero optimizer
steps. Full native/original/late models preserve earlier-prefix sensitivity
at gaps 8, 32 and 64 under identical query suffixes and actual order-1–8 count
vectors. Full paired output KL spans .000214–.001943 nats; shallow is much
smaller, and minimal has zero measured feature/output difference at gaps 32
and 64. This demonstrates a retention difference, not learned parity: none
of these text checkpoints was fitted to the probe. Balanced opposite-label
pairs establish a 1-bit loss lower bound only for predictors restricted to
those query suffix/count inputs. The full prefix determines the label.

Same-window frozen order-4 Kneser–Ney gives 3.751541 bpc and still beats all
neural arms. All 20 fixed-order/method/history controls remain separately
reported; one count pass with fit-prefilled memory differs from four gradient
passes with cold neural state, and development adaptation differs again.
Matching average losses cannot establish that a neural model computes counts.
Frozen head gradients are nonzero, and shallow covariance participation rank
is higher despite worse loss. Neither gradient reach nor rank certifies useful
deep features. Fitting is also not demonstrably at a stationary optimum.

The concrete theoretical next step is in Note 65: mean plus an occupied bit
can hide sample precision, a value-map repair cannot supply evidence from an
unread address, and joint route utility may remain zero until an interaction
decoder learns. These are distinct conditional limitations, not a mathematical
barrier to improvement. Keep core temporal/race/state/counterfactual mechanisms;
test retained evidence and joint useful credit under bounded integrated
controls before further scaling. The 100-page report contains completed
quality/resource tables and the frozen inference-fusion audit, preserving
historical comparisons and the failed gate. No additional local fit is queued.
