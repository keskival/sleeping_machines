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
