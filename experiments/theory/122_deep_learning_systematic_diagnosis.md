# Systematic deep-learning diagnosis: distinguish credit, conditioning and use

This synthesis answers the 3 October request for exhaustive, systematic,
divided analysis of deeper-model learning. Three independent audits covered
completed experiment/protocol evidence, optimizer/clipping implementation,
and architecture/transport. Notes120/121 specify new bounded interventions;
their completed results are separate immutable artifacts. Nothing below
promotes a dense carrier or removes computational time, temporal races,
sparse private state, separate keys/values or deep counterfactual credit.

## What the completed comparisons actually establish

The fine-packet DVS matrix has984 FIT/192 subject-disjoint DEV gestures,
eight passes/7872 presentations/496 Adam updates, p16/H2/pool2/U16/lr.003.
Default clip1. FIT32 is the first32 fitting gestures evaluated at the
DEV-selected weights; it is neither whole-FIT loss nor common-final-epoch
optimization loss. All rows below are completed, not online scores.

| DVS variant | Seed | Selected FIT32 NLL | DEV accuracy | DEV NLL | Whole-fit GF | Fit MF/presentation |
| --- | --- | --- | --- | --- | --- | --- |
| D2 factorized | 7 | .576049 | 56.771% | 1.116189 | 18.2437 | 2.3175 |
| D4 factorized | 7 | .718192 | 55.208% | 1.130272 | 34.3476 | 4.3633 |
| D4 clip4 | 7 | .945086 | 57.292% | 1.172124 | 34.3476 | 4.3633 |
| D4 sampled8 | 7 | .715109 | 55.729% | 1.330390 | 214.3941 | 27.2350 |
| D4 all-race | 7 | .763093 | 55.208% | 1.152624 | 3815.2066 | 484.6553 |
| D4 factorized | 8 | .587894 | 59.896% | 1.093247 | 34.3476 | 4.3633 |
| D4 clip4 | 8 | .711417 | 58.854% | 1.035432 | 34.3476 | 4.3633 |
| D4 sampled8 | 8 | .630686 | 57.292% | 1.101878 | 214.3941 | 27.2350 |
| D4 all-race | 8 | .553204 | 64.583% | 1.020627 | 3815.2066 | 484.6553 |
| D6 factorized | 7 | .658653 | 59.375% | 1.085895 | 50.7214 | 6.4433 |
| D6 sampled8 | 7 | .952533 | 47.917% | 1.336483 | 313.4628 | 39.8200 |

Exact work and native inference numbers are in the source JSONs and generated
report common-unit appendix; this rounded table uses the same presentation
denominator for every row. Unit-special FLOPs are estimates from complete
first/last sampled-window operator audits, not measured energy or wall cost.
Files: `experiments/results/dvs_native/curie_dvs_batched_p16d{2,4,6}pool2_...`
with factorized/le8/leall tags20261002T224500Z/230500Z; clip4 tags
20261003T004500Z. Source/result files remain unchanged.

**Revision beside §408, not silent replacement.** P408's prediction that cap4
repairs depth4 fitting fails on the available FIT32 diagnostic in BOTH seeds:
NLL worsens .226894/.123523. Heldout behavior is mixed. Its stronger assertion
that route credit cannot matter is also unsupported: full all-race seed8
improves FIT32 by.034690 and DEV NLL by.072620/accuracy4.6875points; seed7 fails.
The positive result must remain visible, with its replication failure and
roughly111-fold counted fitting cost. Neither is a universal cure.

Full corrected AWS coarse depth4 replay separately lowers FIT-subset loss
relative to teacher/factorized (.807254 versus .856196/.890972) but fails
heldout gates. AWS depth8 replay's seed7 private/shared positives fail the
unchanged seed8 confirmation. These results show usable deep credit and fitting
signals, plus substantial generalization/seed sensitivity. They do not justify
declaring a complete gradient-learning regression or mathematical impossibility.
The active ten-million-character AWS language matrix remains the quality
priority; tiny1025-character learning smokes only admit implementations.

New completed owner controls arrived during the audit: D4 seed7 lr.006 has
FIT32 approximately.960, worse than.718; D2 seed7 cap4 approximately.683,
worse than.576. They strengthen the failure of the TESTED larger-step cures.
They do not exclude all optimizer/moment-conditioning mechanisms or prove
hard-route destabilization without actual functional/route telemetry. Owner
§410 now queues scratch near-identity gate bias-4 D4/D6 and separate segment-
reset10M language arms; reuse those comparisons rather than duplicate an
initialization sweep. They retain clocks/races but have different language
state/data/evaluation protocol from streaming AWS10M, so tables must label it.

## 1. Clipping: correct equations, actual code, discriminating quantities

The audited DVS driver sums losses, divides every gradient once by ACTUAL
episode count, clips once, and steps plain Adam. No warmup/scheduler, stale
stacked-parameter cache or double normalization was found. Clip4 wrapper
changes the actual cap and restores its monkeypatch in finally. This excludes
those particular implementation bugs, not every learning defect.

Let h_t=k_t g_t, k_t=min(1,c/(||g_t||+epsilon_clip)). Adam uses

    m_t = beta1*m_(t-1) + (1-beta1)*h_t
    v_t = beta2*v_(t-1) + (1-beta2)*h_t^2
    Delta theta_t = -eta*mhat_t/(sqrt(vhat_t)+epsilon_adam).

If the ENTIRE gradient history is scaled by positive constant a, moments
scale as m->a m, v->a^2 v. The update cancels that scaling except epsilon.
Fresh Adam gives -eta*g/(|g|+epsilon_adam/k). Thus threefold raw norm under
fixed clipping does NOT imply a threefold smaller Adam displacement. This is
consistent with the original [Adam paper](https://arxiv.org/abs/1412.6980);
the application to these logs is our explicit derivation, not its benchmark.

Variable clipping does matter. It changes moment-history/sample weighting,
directions, effective signal-to-noise and relative age of sparse coordinates:

    E[k g] = E[k] E[g] + Cov(k,g).

An unrelated large block can change every other block's sample weighting.
Note29 gives a constructed mean reversal and a failed cross-block-freeze
intervention. Note36 records a real large Adam step that raises SHD fitting CE
.622628->1.867725, while1/16 learning rate lowers it to.333181 and drastically
reduces anchor KL. That is evidence about a DIFFERENT saved model, illustrating
overshoot rather than proving it causes these DVS results.

At mature moments, a previously dormant coordinate's first nonzero gradient
can yield magnitude eta*(1-beta1)/sqrt(1-beta2), about3.16eta for default
betas, before accounting for prior moments/bias correction/epsilon. A tiny
raw gradient therefore need not mean a tiny step; a tiny branch-output
derivative can still suppress its FUNCTIONAL effect. Separate parameter
displacement from the computation it changes.

Required telemetry is actual pre/postclip norm, k_t, per-block displacement,
update/weight ratio, moment age and epsilon-dominated coordinates, finite
same-FIT CE, independent FIT-anchor CE/KL and route changes. Saved benchmark
JSONs omit k_t and displacements; raw-norm probes cannot reconstruct their
history. Note120 forks genuine paired online weights/Adam moments. Never
pair final moments with DEV-selected best_state: selection restores best
weights AFTER the final paired online snapshot.

The saved fine D2 and coarse D4 probe protocols differ in encoding, seed,
passes and gradient teacher. Cross-model differences do not identify depth;
only within-checkpoint forks isolate immediate cap/LR effects. Missing matched
curie D2/D4 checkpoints remain an artifact gap. No frozen three-window probe
establishes historical clipping frequency or a quality improvement after refit.

## 2. Gradient transport: amplitude is not conditioning or useful direction

For fixed routing, total temporal/state computation has a full unrolled
Jacobian J = J_event,T ... J_event,1, including recurrent context, private
memory writes, normalization, key reads and arrival derivatives. Elementary
transport D(age)R(age) has singular values equal to its coordinate-pair
decays. Small decay across ONE head-alignment interval does not bound the
singular spectrum of J or useful task directions after depth and recurrence.
Large parameter norms do not exclude cancellation, gradient shattering,
high curvature, projection through normalization or wrong surrogate direction.

For residual x'=x+alpha F(x), J=I+alpha J_F. If sum||alpha J_F|| is controlled,
products admit useful finite upper/lower conditioning bounds away from
singularities; merely scaling alpha by1/sqrt(depth) does not prove such a bound
for coherent residual directions or long recurrent event sequences. LayerNorm
removes its shift direction and approximately its scale direction: inspecting
transport rotation alone misses this geometry. We should probe VJP/JVP norms
and alignment along the FULL event graph and across independent FIT examples,
not declare transport solved from3% local decay/.11rad rotation.

[Identity Mappings](https://arxiv.org/abs/1603.05027) analyzes direct residual
paths; [Shattered Gradients](https://arxiv.org/abs/1702.08591) studies loss of
gradient correlation with depth. These are diagnostic analogies. Our native
core already has content residuals, but channel mixing, source recurrence,
time-dependent alignment and hard routes complicate the complete product.
No substitute of a conventional synchronous network is proposed.

## 3. Added depth must be live, reconstructible and demonstrably used

The legacy growth construction sets added sigmoid gate biases to-20. At that
bias sigmoid and its derivative are about2.06e-9. Gate input can change this;
Adam can partly cancel small gradients above epsilon. Therefore this is a
plasticity RISK, not proof of frozen gates. Message branch contribution,
opening and functional sensitivity need measurement. Channel mixes and
clock/transport parameters have other live learning paths.

Grown D4 seed7 factorized improves DEV60.938%/1.109089 and FIT32.600796 over
scratch D4. It also gets eight EXTRA passes, fresh Adam, changed parent race
noise and heterogeneous gains. Charge parent20.0739GF plus new34.3476GF =
54.4215GF. Further fitting old layers can explain the gain without learned
added message branches. Grown all-race FIT32.534146 is better while DEV
57.812%/1.150003 is worse. Neither is an isolated gain from depth.

**Concrete construction bug:** unit.gain is a Python float absent from
state_dict. D2->D4 sets old gains.353553 and new gains.25; legacy D4->D6
resets ALL old gains to.25, changing oldest nonlinear amplitude by29.3%.
Reconstructing a grown D4 checkpoint with the ordinary factory similarly
misreads it as uniform.25. Note121's new sibling records source-checked
lineage and per-layer gains, reproduces legacy parents as ACTUALLY constructed,
then preserves their vector in new children. It retains closed gates, clocks,
races, sparse private memory and original optimizer/noise protocol; it does
not claim to solve branch plasticity or make growth fully function preserving.

Identity message proposals alone do not preserve a temporal model's complete
function: added positive race delays alter future ages/readiness, and extra
noise draws change races. Do not copy Net2Net's identity claim without an
event/time/noise contract. [Net2Net](https://arxiv.org/abs/1511.05641) supplies
the function-preserving motivation; applicability here requires this extra work.

If gate saturation is CONFIRMED as the bottleneck, a later separately admitted
construction can use a small signed linear branch coefficient with a live
coefficient derivative, or zero output projection onto a live hidden branch.
At y=x+alpha F_theta, alpha=0 initially gives dL/dalpha=<dL/dy,F> while dL/dtheta
is initially zero; the coefficient can open before branch weights learn.
[ReZero](https://arxiv.org/abs/2003.04887) is a precedent, not an original
claim for this scalar primitive. The first live update, clocks/state and
extra inference/learning work must be contracted in our integrated model.
This remains a proposed intervention until measured need/admission, not an
unannounced architecture replacement.

## 4. Route credit: fidelity, support, variance and delay are different axes

For a categorical race with probabilities pi and alternative suffix returns
Q_i, score credit is c_i=pi_i*(Q_i-sum_j pi_j Q_j). Full alternative returns
capture changed private writes, future messages and future route choices;
payload-only local teachers can miss these. Useful full-route effects are
demonstrated in completed deep/horizon diagnostics. Exact conditional
enumeration is still not a blanket exact whole-stream gradient theorem.

Sampled k races out of R scales each sampled contribution byR/k. For uniform
sampling without replacement, total-score estimator covariance is proportional
to R^2*(1-k/R)*S/k (S the finite-population covariance under its conventional
normalization). Thus fixed k=8 grows worse as R grows if heterogeneity persists.
At21events, D4/H2 hasR168 and D6/H2 R252. Adam then receives a different
noise distribution and clipping factors, so 'unbiased route sum' does not
imply equal optimizer trajectories or equal fitting. Need fixed-state
parameter-space variance/step comparisons before more sampled-depth fits.

Also distinguish dT/ds_i=-T*pi_i, the factorized common-clock teacher in
batched_episodes, from the literal fixed-noise branch-interior derivative
-T*1[i=winner] of min(e_i/exp s_i). Notes80–89 discuss joint choice/time
laws, winner-dependent teachers and their limits. Gradient/Adam dot products
in the new probe are SURROGATE predictions, not certified fixed-noise descent.
Finite hard-model loss and heldout-anchor KL are observations even if the
surrogate predicts incorrectly. Separate content, relative choice and common
clock components; measure interference after the shared parameter Jacobian
pullback, not only in score coordinates. Alternation and phase offsets have
prior failed/mixed controls; do not launch them all at once.

## 5. Exposure, memory and representation versus readout

Available receivers = depth*heads*pool. Key scoring, selected writes, losing
value proposals, replay lanes and optimizer visits have separate costs.
Depth increases capacity, gradient sources, paths and race count together;
it does not by itself increase examples received by each private parameter.
Sharing maps across depth preserves private memories but changes parameter
exposure; AWS shared-depth controls correctly test this separately.

Full replay's Q_i are computed under no_grad and detached. This expands route
utility coverage, NOT direct losing-content gradients. Factorized content
backprop still trains factual winners (and shared ancestors). Correct policy
credit can move a unit into future use, but does not guarantee immediate
learning of its value function. Sparse content-gradient exposure, score-credit
support and available capacity must have separate measurements.

The DVS race seed is common to EVERY example/window of a given pass:
100000+seed+10000*epoch. With fixed21-event length, the same race-position
uniforms repeat across984 examples; only eight per-pass noise environments
are used. Winners still depend on content/state and change with weights, so
this is not proof of identical routing. Shared-noise correlations may limit
effective exploration and amplify gradient covariance; they are a testable
contributor distinct from clipping. Changing to independent per-window/lane
noise requires causal identity-free seeding, sequential/vectorized/recovery
contracts and a separately fixed comparison. Do not assume it helps or mix it
with a credit/regularization/growth repair. If Rao-Blackwellizing content
gradients is proposed, derive its full conditional estimator and charge losing
branch BACKWARD work; merely removing stopgrad at every race can double count
representation derivatives and is not an admissible shortcut.

To establish deeper features, compare saved INITIAL and trained encoders with
the same prespecified small decoder, probe usable predecessor information,
and lesion added computation at fixed weights. Decoder fit gains alone can
reflect readout adaptation; dormant storage alone does not prove useful
capacity. Prior retention/interaction notes67/68 and real-feature note75
give both positive learned-information findings and failed readout hypotheses.
No conclusion that the entire architecture merely counts n-grams follows
from one weak shallow/private variant or a failed deep benchmark.

## 6. Input/time scale and generalization remain material

Completed native coarse4 controls reduce21events to5, discard within250ms
timing and often improve DEV versus fine20 while cutting fitting work.
Seed6 matchedclock DEV NLL.890983 versus fine1.179750; seed7 1.056934 versus
1.172640; seed8 .920744 versus .898591 is worse. Faster coarse clocks reorder
which seeds improve. Encoding, time conditioning and subject invariance matter;
these comparisons do not prove temporal detail intrinsically useless.

A model can underfit whole-FIT, overfit subjects and have a decoder bottleneck
simultaneously. Compute complete FIT at common epochs before claiming global
underfitting; use FIT-only step calibration and fixed independent FIT anchors.
Preserve DEV selection rules and avoid retrospectively tuning on failed DEV
seeds. Existing regularization/coarse/sharing results should select a narrow
follow-up only after the mechanism diagnosis, not a combinatorial sweep.

## Decision order and resource boundary

1. Preserve completed clipping and full-credit positives/failures; revise §408
   beside its history. Reuse pending lr/D2-cap controls rather than duplicate.
2. Complete the saved actual-moment/FIT-only step probe and numerical lineage
   repair. Apply no benchmark cap/LR recommendation from raw norm alone.
3. Retrieve matched fine D2/D4 online checkpoints when available; measure full
   FIT/common-epoch losses, live added branch movement and lesion effects.
4. Use frozen gradient decomposition/variance and full-event Jacobian probes
   to choose ONE repair: functional step calibration, non-saturated depth
   entry, better conditional credit allocation, or parameter exposure.
5. Contract every retained mechanism and recovery/work, small integrated fit,
   then unchanged second-seed confirmation. Only then long AWS comparison.

Exact winner reuse already cuts corrected replay fitting about49.66% with
unchanged tiny-smoke learning. Notes118/119 causal-prefix caching prove another
arithmetic reduction but the grouped executor is over2x SLOWER than winner
reuse in T16 wall; no promotion. Compact replay execution is an efficiency
task independent of proving deep feature quality. Charge all learning work,
not only selected inference, and never convert partial10M scores into
completed benchmark claims. Guarded serial jobs preserve8GiB available.

There is no impossibility theorem established here. Finite memory and finite
precision impose real information bounds; hard route support, conditioning
and statistical generalization impose learnability tradeoffs. The observed
positive deep fits show progress is possible in some settings. What remains
open is a replicated, useful quality/resource advantage of the full sparse
temporal construction, not whether every failure is a single tuning bug.

## Completed optimizer and lineage diagnoses

`diagnostics/local_deep_clipping_optimizer_probe_current_inputs_20261003T025400Z.json`
completes5 contracts,100.552s/382576KiB. Eight actual driver gradients and
48 discarded Adam forks retain full gradient/update vectors. Both original
checkpoint hashes, online cursor and genuine moments are bound. No refit or
DEV/test selection. Four prior failed queues/logs are retained: an overly
small storage-rounding bound, absent historical kernel, and two preprocessing
preflight failures. Exact old D4 kernel bytes8f93f5... are archived and used.

The D4 coarse statistic-byte transform hash fails exact original reproduction:
a0496fe9... versus this host's3fefa180... . All other metadata/count-artifact
hashes match. The admitted D4 probe explicitly uses CURRENT recomputed FIT
inputs at genuine saved weights/moments; no bitwise original-input claim or
cause-of-old-quality inference. D2 transform metadata matches exactly. This
gap reinforces saving preprocessing arrays and environment alongside models.

Fresh cap1 versus cap4/no-cap changes the actual full step norm only.019%
(D2) and.190%(D4), despite gradient norms shrinking by factors.374716 and
.295487. Full parameter-step norms are.373657/.373729 (D2) and
.501875/.502832(D4). Same-FIT/anchor losses are nearly identical across caps.
Quarter LR produces one-quarter displacement and much smaller prediction KL.
These two protocols differ; the cross-depth step ratio is not depth causality.

At trained D4 moments, changing ONLY the present cap1->4 multiplies raw
clipped gradients4/4/3.602 across three windows, but actual step norms rise
only1.3656/1.4655/1.3613. Surrogate descent cosine rises.0905->.3155,
.1510->.3455,.0545->.2973. Same-FIT losses decrease more with cap4 in all
three; independent FIT-anchor loss changes cap1=.00487/-.08828/.02082,
cap4=-.07074/-.30569/-.01406. These single-step improvements coexist with
the completed eight-pass cap4 FIT32 regressions. The moments were trained
under cap1; increasing CURRENT gradient weight against that history is not
training the ENTIRE history under cap4. This reconciles the results without
declaring either false, and prioritizes sample/moment weighting over a
constant gradient-scale explanation. It does not establish an adaptive-cap
schedule or select a new cap from heldout scores.

`diagnostics/local_depth_growth_lineage_contracts_20261003T025000Z.json`
completes12 contracts,1.560s/261216KiB. Heterogeneous gain preservation,
direct old/new bitwise nesting, D8 reload, actual snapshot metadata and
invalid-protocol refusal pass. The external pending legacy progressive D6
queue is affected; preserve it as historical/proposed evidence and use a
new uniquely tagged corrected driver after its owner verifies parent files.
Neither numerical fix proves useful added depth or benchmark superiority.

DVS batched_episodes retains each entire episode graph; there is no internal
detach. The language16-token horizon cannot explain that DVS failure. A
separate actual gate/branch plasticity probe is now specified in note123,
including the possibility that tiny added value contributions disappear
when added to nonzero float32 incoming values. Nonzero autograd gradients
alone do not demonstrate meaningful finite forward changes.

## Completed added-branch plasticity probe: a concrete limited mechanism

`diagnostics/local_depth_growth_plasticity_probe_20261003T030400Z.json`
completes6.142s/370288KiB, four prespecified -20/-8/-4/0 gate-only arms at
identical p16 grown D4 parameters from selected saved D2 parent. Exact fine
FIT-only transform, fixed16 FIT examples/one normalized clip1 freshAdam.003
step per arm. No DEV/test arrays read or step selected. Every instrumented
logit/gradient is bitwise identical to plain computation, caller RNG/kernel
functions/hooks/parent/source immutable; vectors are saved. Four updates/
64 target exposures, plus plain backward and before/after verification work.
Total FLOPs/traffic/energy unknown, not zero.

At -20 added layer3/4 mean gates2.2814e-9/2.2206e-9. Only.2651%/.2511%
of candidate proposal coordinates differ from incoming float32 values:
approximately99.7% of nonlinear contributions ROUND AWAY. Selected branch
delta norms4.24e-9/3.41e-9. After one step gates remain at2.26e-9/2.21e-9.
All active gate and output gradients are below Adam epsilon after clipping.
Mean fresh sign-step fractions for gates are7.03e-5/7.19e-5 (about.007%).
Aggregate stored gate displacement1.40e-5/1.50e-5 and output1.96e-5/2.74e-5.
This is measured strong attenuation, not merely a nonzero-gradient argument.

But the input maps are NOT frozen: raw gradient norms.04635/.13888 and
displacement.09587/.09595, mostly full fresh sign steps. Private memory
keys/time and other paths carry live credit even when message outputs are
nearly silent. This corrects any inference that all added state/content
learning disappears. It does not identify which live paths caused previous
growth accuracy gains; inference lesions and equal-pass continuation remain
necessary. Default scratch D4 gates differ from -20, so the result explains
the limited growth construction, not every scratch-depth failure.

At -4 mean gates.01982/.01930, ALL measured proposal contributions visible,
gate displacement.09752/.09710, output.09551/.09530; only.276%/.368% of
active gate coordinates belowepsilon. Initial SAME-FIT NLL .868078 versus
.866363 at -20, a small but real change of function. Poststep .434085
versus.435219; all four arms improve this tiny FIT batch. These are not
generalization/quality evidence or a bias selection sweep. -4 is the
prespecified gate-only counterpart of owner §410, not its full scratch
initialization/transport protocol. Use the existing owned quality comparisons.

Priority after this diagnosis: source-complete growth reconstruction, useful
live nonlinear message entry, measured FULL-FIT/common-epoch behavior and
functional update calibration. Keep counterfactual quality/variance and
shared-noise exposure separate. The data favor a multi-part optimization and
generalization diagnosis over a universal clipping fix.
