# Findings log

## E171–E177: causal evidence, complete learning work and persistent language

### Highlighted-result review and aligned language scoring

The numerical E171 review reproduces the seven consolidated task screens from
saved weights and the selected speech model's 408/512 development answers.
Fitting/development identities are disjoint; target mutations leave constructed
inputs unchanged. Causal prefix probes pass for the reviewed shared/event-state,
LSTM, Transformer and Hawkes input paths. Source review also identifies
historical E79 partial-word target leakage and older native market threshold
fitting on held days. Detailed decisions and limits are kept in
`EXPERIMENTAL_REVIEW.md`; those historical claims are not used in the report.

E173 rebuilds train-only, strictly causal context evidence on 10M characters.
The count/copy mixture reaches **1.726986 test bpc without word context**;
adding causal word context gives **1.719360**. Both fit mixture weights on an
additional 1M validation labels and freeze parameters during testing. E174
rescoring of unchanged saved neural weights gives **1.799344 LSTM** and
**1.908275 Transformer**, on the same **999,999** test positions and checksum.
The score gains are 0.072358/0.181289 bpc for the no-word mixture; optional word
context adds 0.007626. Model capacities and fitting/validation budgets differ.
This is a specialized prediction result, not generic representation supremacy
or measured energy. All compared streams start cold and score causal context.

### Full optimizer-step arithmetic

E172 v2 traces a representative four-query fitting batch for each common and
Transformer screen. The total includes **forward/loss, backward, clipping and
Adam**. Every observed floating operator has a declared arithmetic formula or
explicit special-function/comparison/data-movement classification. Common/TF
arithmetic ratios are **1.913 text, 1.914 market, 1.970 composition, 1.437 MNIST,
0.453 gestures**. These are complete-step estimates, not historical whole-run
budgets. Preprocessing, evidence fitting, calibration, inherited fitting and
search remain additional costs. No physical memory traffic or joules are measured.

### Eight-layer persistent language learning

E175 implements retained modal states and an actual delayed-message queue.
Fifteen token arrivals cause exactly 120 layer deliveries. Chunk splits and
future-suffix mutation change earlier predictions by zero. All eight value
teachers are nonzero; seven hidden-clock teachers are nonzero. The final clock
is unobserved by a completed, untimed query within its deadline.

E176 fits 8,192 character targets for four passes, with 1,024 validation targets,
31 warm characters and 64-character truncated credit. It uses 28,403 parameters,
width 32 and sixteen modal pairs per block. Validation bpc is **5.328981** before
fitting, then **3.831130, 3.527552, 3.420379, 3.351248**. Frozen fitting bpc ends
at **3.142180**. All eight layer teachers are nonzero in each pass. Each pass
consumes 8,223 source characters and makes 65,784 block deliveries, without
prefix replay. Total CPU wall time is 364.93 s; peak RSS 373,580 KiB. Target
ranges match E133, but topology, capacity, history and update counts differ.
This is a one-seed development result, not a matched depth/control scaling study.

### A causal tokenization control with exact character likelihood

Theory §§269–274 constructs a complete prefix dictionary. Leaves are uniquely
parsed tokens released at their last observed character. Subtree probability
ratios define normalized next-character probabilities; phrase scores telescope
to the learned leaf likelihood. Partial final phrases are marginalized, so
character and compressed arms can score exactly the same raw targets. The local
teacher is the difference of compatible-leaf posteriors, not a confidence proxy.

E177 verifies normalization to 2.22e-16, token/character score telescoping exactly,
character-stream equivalence to 4.44e-16, and zero chunk/future-suffix errors.
All eight layer teachers are nonzero. In its synthetic contract 48 characters
release 21 tokens, with one unfinished character and 168 block deliveries.
This establishes scoring and causal execution, not a quality or energy result.
The vocabulary is fitted on training only; frequency counts never supply model
output probabilities. Vocabulary capacity and tokenizer work must be charged.

## E161–E166: directional depth and a stronger single deployment encoder

**Deployment improvement, not a new overall accuracy record:** the selected
single six-block encoder matches E143's **408/512 (79.6875%)** with development
NLL **0.625161** versus **0.684276**, and **395,814** versus **449,110** parameters.
On the reused 657-utterance audit it gets **518/657 (78.84%)** versus 510/657:
60 selected-only correct and 52 combined-only correct, net +8. This is the
private training-speaker protocol; no official SHD test or published parity.

### The intervention follows the measured depth bottleneck

E154's bounded D12 output maps learn, but their six added blocks change no
audit answers. §§261–264 derives the centered normalization's tangential and
radial teachers. Post-map normalization can attenuate its map's radial update;
a tiny scalar output gain also shrinks hidden directions invisible to the
class head. The fitted deletion establishes decision irrelevance, not unique
causal attribution to radial saturation.

`observer_conditioned_depth.py` instead normalizes modal-state features before
the zero output map and uses R=(I+H^T H)^(-1/2), H the fitting-derived class
contrast head. R is invertible and retains unit scale on ker(H). The output
map's expressive family is unchanged; its visible update units are conditioned.
There is no arbitrary small gain on all hidden coordinates. The fixed transform
folds into ordinary output weights at deployment, with no extra packet map.
This changes normalization, optimizer coordinates and new-group update units
together, not a matched single-factor intervention or complete reference inclusion.

E161 preserves initialization/old teachers exactly, gives nonzero teachers in
all six new maps, checks the explicit formula to 1.11e-16, full-rank coordinate
reconstruction to 4.44e-16 and deployment folding exactly. E162 uses the same
actual first fitting examples as E156: fixed old rate 0.0000203125 and derived
new rate 0.0001953125 reduce batch CE 0.622628 -> 0.325345; anchor mean KL is
0.004574 under the 0.02 budget. No development/audit labels select this rate.

### Useful extra depth is still a generalization gap

The declared E163 continuation keeps old optimizer state/order, sample order,
augmentations and one pass over 6,144 examples, matching E159's D6 control.
It adds 300,306 trainable parameters; full D12 has 696,120. Online NLL is
0.397281, fitting **5,577/6,144 (90.77%)**, development **401/512 (78.3203%)**
and NLL **0.643979**. The extra maps reach parameter norms 0.632–0.660 and
norm-gain vector norms around 11.04; their effect is no longer negligible.

But its own trained prefix/head gets **408/512**, NLL **0.625161**, when only
the six appended blocks are removed. Full versus prefix changes ten held
predictions: one full-only correct, eight prefix-only correct, net -7.
On the reused audit, full D12 gets **509/657**, NLL **0.688439**; its prefix
gets **518/657**, NLL **0.668562**. Deletion changes eleven predictions:
zero full-only correct and nine prefix-only correct. The correction now makes
real decisions and has negative net held-speaker value at this budget.

The matched separately trained D6 control gets 400/512 and 513/657. The
directional model's pruned prefix thus improves development by eight answers
and audit by five over that control, after training in the deeper graph.
This suggests useful training coupling and harmful fitted appended corrections;
it does not isolate a depth benefit from the changed coordinates/regularization,
nor establish that longer deep training will fix transfer. Representation
transport, output conditioning and cross-speaker utility remain distinct.

E165 selects prefix versus full by the two already recorded development
scores, never by the audit. This is explicit post-hoc architecture selection.
It exports only the six old blocks and trained head, retaining their Adam
state and order/augmentation RNG. E166 independently restores that ordinary
encoder and optimizer and reproduces the entire 512-example development score
exactly. Checkpoints and executed sources are hash-verified; no completed
historical result or successful queue is overwritten.

**Cost:** selected prefix CPU audit forward time is **11.904 s** versus
**23.705 s** for the combined model; full directional D12 takes **14.123 s**.
These are single warmed observations, not joules. The deployment parameter
reduction is about 11.9%. E163's full recorded wall time is 591.47 s and training
390.64 s; peak RSS 1,875,648 KiB, minimum sampled host available 10,378 MiB.
It records **6,039,797,760 additional training-forward coordinate-fold MACs**,
which must be added to its other partial forward ledger. Backward/optimizer,
sorting, memory traffic and energy remain incompletely measured. Pruning
does not erase the full D12 fitting cost, fitting-only calibrations, paired-head
work or inherited E143 training. All jobs run one at a time through the safe
queue with the RSS watchdog and explicit 8-GiB reserve.

**Next analytical/empirical question:** what makes the appended updates useful
on fitting speakers and harmful on held speakers? Measure fitting-only view
covariance and independent-example transfer of the full new block, not merely
the first map. Preserve signed temporal amplitude and exact weighted-state
queries when comparing pooling/normalization. Retain the best deployment
checkpoint while testing a predetermined deeper-training/representation change;
do not claim SOTA parity from this private consolidation result.

## E149–E159: one encoder, live depth and realized optimizer steps

**The best speech-family score remains E143's 408/512 (79.6875%).** A single
six-block temporal encoder now retains **406/512 (79.296875%)**, with 395,814
deployed parameters instead of the two-branch model's 449,110. Its development
NLL is 0.631004 versus 0.684276 for the combined model. These weights inherit
the E143 encoder and use its combined predictions as a fitting-only teacher;
this is deployment consolidation, not a from-scratch or equal-training-cost
comparison. Neither model has reached official SHD reference parity.

### Why the readout needed the augmentation distribution

E149 derives finite categorical-gauge absorption certificates and fits an
affine replacement head on fixed encoder features. Offline feature whitening
is folded into the ordinary head, adding no inference transformation. The
contract preserves 240/240 synthetic decisions, checks the finite KL bound,
and reproduces folded logits to 2.14e-14. An arbitrary parent is not guaranteed
to lie in the new encoder's affine feature span.

E150's clean-feature fit reaches 6,011/6,144 fitting answers but only 374/512
held answers. E152 uses the same frozen features and one clean plus one
augmented view per fitting utterance, with an explicit within-view covariance
penalty. It reaches 406/512 before any further encoder update: **32 additional
held answers with unchanged representations**. The number of differing
clean/augmented fitting predictions falls from 1,169 to 480. Exact finite-view
Jensen risk falls from 0.153232 to 0.041113, about 73.2%.

| Fixed encoder; fitted head | Clean fit NLL | Augmented fit NLL | Held correct / 512 | Held NLL |
| --- | --- | --- | --- | --- |
| E150 clean fitting, matched reconstruction | 0.153118 | 0.635833 | 374 (73.05%) | 0.817668 |
| E152 paired views and nuisance covariance | 0.217752 | 0.308660 | 406 (79.30%) | 0.631004 |

The changed views and regularization are a joint intervention. No single-factor
attribution is claimed. All head fitting and whitening use only the 6,144
fitting IDs. The encoder inherits three E143 passes; E152 then tried two
additional passes, neither improving its epoch-zero score. Private development
selects epoch zero. Raw data are streamed through the augmentation cache;
the experiment does not duplicate the entire raw-event dataset in memory.

Theory §§245–248 proves exactly, for a fixed label and finite view distribution,
that average view CE minus CE at the mean logits is average
KL(p(mean logits) || p(view logits)). Its upper bound depends on class-visible
within-view covariance. E153 checks the identity to 2.11e-15. Nuisance variation
and useful future route optionality are separate quantities; rewarding all
variance would oppose this measured improvement.

### Depth can preserve the old function and still have new teachers

E151 grows six blocks to twelve with identity residual values, nonzero hidden
state, zero output maps and a LayerNorm gain chosen to keep the first output
map teacher live. Initial class outputs and all old parameter teachers agree
exactly; all six new output maps have nonzero label teachers, with the explicit
formula checked to 1.11e-16. New internal input/pole/clock teachers initially
wait for the output maps to become nonzero. This avoids a double-zero gate.
The extra six winning emissions add 36 ms initially, so the preservation is
for completed untimed queries, not deadline-equivalent behavior.

An observer-conditioned initialization (E157, §§253–256) then bounds new
class-visible residuals using the actual inherited head norm. Initial norm
gain is 4.84056e-6 and its optimizer-group rate scales by 0.00153072. Maps
retain a live, smaller first teacher. This is a concrete layer/optimizer
construction, not a global convergence theorem or established depth benefit.

### The realized Adam step, and a scheduler restoration bug

E156 restores the first four actual augmented fitting examples and replays
the same Adam teacher/moments at six declared step scales. At LR 0.000325,
batch CE increases from 0.622628 to 1.867725, although the physical interior
directional prediction is negative. Independent fitting-anchor KL is 1.845976.
At 1/16 rate (0.0000203125), batch CE instead falls to 0.333181 and anchor KL
is 0.004412. This isolates an excessive realized first step in this particular
continuation. Adam normalization can largely cancel the magnitude of gradient
clipping; a teacher-norm cap is not a class-probability step cap (§§249–252).

The original identity-grown D12 fails the same anchor budget even at 1/1024
scale. E157's observer-conditioned D12 accepts the same 1/16 factor as D6,
with batch CE 0.332073 and anchor KL 0.004448. All calibration labels are from
fitting, not development or the reused audit.

**Implementation finding:** E155 loaded an old optimizer `initial_lr` of
0.000325. `LambdaLR` restored that value over the manually calibrated `lr`.
The actual recorded epoch rate therefore stayed 0.000325. Its D6 pass exactly
reproduces E152's unstable first-pass online NLL, 3.548693291276829. E155's
result files and executed source are preserved. They do not test calibrated
training and must not be cited as failure of that policy or its depth theorem.
E159 explicitly resets every group's `initial_lr`, verifies scheduler rates
against the intended rates before any update, and records them in its result.

The corrected D12 continuation completes one pass at the verified old-group
rate 0.0000203125. Online NLL is **0.396784**, fitting is **5,574/6,144 (90.72%)**,
and development is **400/512 (78.125%)**, NLL **0.639008**. Its 696,888 deployed
parameters include 301,074 new parameters. It retains useful training at
twelve blocks, but loses six held answers relative to its 406-answer starting
head. The matched corrected six-block result also reaches **400/512**,
**5,574/6,144** fit and online NLL **0.396765**. Its held NLL is 0.639012;
all 512 decisions match D12. D6 training takes 304.93 s versus D12's 373.26 s.
Additional depth buys no accuracy gain in this comparison.

**Completed E154 reused-audit/deletion check:**

| Deployed checkpoint | Correct / 657 | Audit NLL | Warm CPU forward seconds | Parameters |
| --- | --- | --- | --- | --- |
| E143 combined | 510 (77.63%) | 0.686446 | 23.705 | 449,110 |
| E150 clean head, selected continuation | 482 (73.36%) | 2.213027 | 11.800 | 395,814 |
| E152 paired head, selected pass zero | 507 (77.17%) | 0.684873 | 11.794 | 395,814 |
| E159 corrected D6 | 513 (78.08%) | 0.684352 | 11.797 | 395,814 |
| E159 corrected D12 | 513 (78.08%) | 0.684344 | 13.835 | 696,888 |
| Same trained D12, appended blocks deleted | 513 (78.08%) | 0.684316 | 11.752 | 395,814 |

The 657 utterances are disjoint from current fitting/development, but already
audited in E147. The paired head loses three net correct answers to the combined
model while roughly halving this CPU observation. Corrected D6/D12 gain three
audit answers over the combined model, despite losing six development answers
relative to their own starting head. These are not official-test or new private
development records. Timings include packing/query, exclude loading and are
one warmed sequential observation each, not joules or a timing distribution.

Deleting the appended blocks changes **zero audit predictions**. Their output
map norms are 0.0307–0.0332, but norm-gain vector norms stay around 4.4e-5.
Live teachers and changed weights therefore do not establish useful added
representation. Deletion retains the fitted prefix/head; it is not a retrained
architecture control. E160 verifies source/result/checkpoint hashes, identical
depth data/rates and safe-runner success/resource logs.

E158 implements an optional bounded, finite function-space replay policy:
one teacher/Adam proposal, at most six actual parameter-ray replays, fitting
CE checks and a fitting-anchor KL cap. It charges extra forwards and defines
moment ownership even when a parameter proposal is rejected. It is not yet a
completed experiment or a necessary fix; E159's scheduler correction takes
priority. Do not report the prepared policy as an empirical finding.

**Resource boundary:** E159 D12's full recorded wall time is 563.51 s, including
loading and evaluation; training alone is 373.26 s. Peak RSS is 1,894,904 KiB
and sampled host MemAvailable stays above 10 GiB. Runs remain sequential under
the safe-runner lock, RSS watchdog and 8,192-MiB reserve. Source projection,
modal scans and local vector maps are real work; recorded forward ledgers
exclude parts of backward, optimizer, sorting and physical memory traffic.
These are not measured joules or a training-energy supremacy claim.

## E143–E148: a new SHD record, learned timing and disjoint transfer

**New private-development record: 408/512 (79.6875%), up from 370/512
(72.265625%) by 38 correct answers / 7.421875 percentage points.** The selected
checkpoint also improves all 657 other eligible utterances of the same held
speakers from 463/657 (70.4718%) to 510/657 (77.6256%). No fitting/development
utterance belongs to that additional audit. This is one exploratory inherited
model and two held training speakers, not an official SHD test or SOTA claim.

| Passes over 6,144 fitting utterances | Fit correct | Fit NLL | Held correct / 512 | Held NLL |
| --- | --- | --- | --- | --- |
| 0: unchanged parent | — | — | 370 (72.27%) | 0.982730 |
| 1 | 5,203 (84.68%) | 0.497160 | 389 (75.98%) | 0.795071 |
| 2 | 5,346 (87.01%) | 0.422942 | 402 (78.52%) | 0.703540 |
| 3 | 5,394 (87.79%) | 0.380833 | 408 (79.69%) | 0.684276 |

The original D8 width-32 carrier stays exactly immutable. A **parallel** D6
width-128 signed event-state encoder adds 395,814 parameters to its 53,296,
for 449,110 total. This is not fourteen sequential layers or a from-scratch
replacement result for the original core. The generic new branch contains
64 real coordinate-pair modes, source lookup before causal coalescing, learned
input/output maps, nonlinear gates, LayerNorm, residual values and winning
clocks. Only supplied nonempty packets are evaluated; no empty ticks or dense
event-pair attention are introduced. Local vector maps remain dense.
Its initially zero correction head exactly preserves the original logits;
nonzero hidden initialization allows representation teachers after the first
head update. Completed-utterance CE leaves the last emission clock unobserved.

The three passes, data, augmentation, seed 6, width/state sizes and optimizer
were declared before outcomes. Adam's epoch rates are 0.001, 0.000775 and
0.000325. The best private-development epoch is the final one. Additional
capacity, a fresh learner and a larger update budget change together; the
record is an engineering improvement, not a single-factor attribution.
On the 512 development utterances, 49 previously wrong answers become correct
and 11 previously correct become wrong. On the 657 disjoint audit utterances,
the corresponding counts are 62 and 15. Its NLL falls from 0.997235 to 0.686446.
The audit uses the selected checkpoint unchanged, with no optimizer updates
or reselection. It is a same-speaker transfer check; historical project-wide
non-exposure of those utterances is not asserted.

**Which learned components the fitted solution uses (E145):**

| Restored initialization; trained correction head retained | Correct / 512 | NLL |
| --- | --- | --- |
| None: learned model | 408 | 0.684276 |
| Hidden clocks only | 395 | 0.724534 |
| Source embedding only | 373 | 0.958682 |
| State stack only | 358 | 0.977332 |
| Decay/frequency parameters only | 407 | 0.683846 |
| All hidden parameters | 374 | 0.968766 |

Clock reset changes 20 predictions: 14 trained-only correct and one reset-only
correct. Only the new hidden emission times change in this intervention, so
the fitted model's learned delays contribute to classification. State-stack
reset changes 79 predictions and loses 50 net correct answers; source reset
loses 35. These are uncoordinated resets, not matched retrained controls;
their effects cannot be added or taken as architectural necessity. Pole reset
changes one decision and slightly improves NLL: modal-parameter learning has
little measured incremental influence at this budget, although the temporal
states remain part of the computation. The learned state/value maps, source
representations and clocks are the stronger fitted dependencies.

**Analytical advance and numerical checks:** §§226–230 constructs the exact
linear EventSSM operator mapping while retaining its optimizer-coordinate
difference, proves first-affine packet endpoint/teacher preservation and
specifies selective affine composition. E142 checks endpoint/teacher errors
1.11e-16/7.77e-16 and all six representation teachers. §§231–233 extends the
packet algebra to preserve its weighted raw-state query as well as final
state, with exact local adjoints. E144's maximum state/query error is 2.81e-15
and teacher error 1.60e-14; expensive output maps can be applied at 9 closures
instead of 27 raw events in that numerical example. E143 does not use this
query-pooling extension; it retains the original declared endpoint design.

§§234–236 connects delay counterfactuals to a Krylov control orbit. At a fixed
later observation, moving one impulse by delta changes its contribution by
(exp(-A delta)-I)h. Adjacent-interval credit cancels the duplicated older-history
term. Multiple small delay alternatives are first-order collinear even when
their finite orbit has high algebraic rank. E148 verifies the exact state
counterfactuals and local teacher, with maximum error 2.06e-13. Its four singular
scales approach epsilon powers 1,2,3,4; at epsilon=0.001 the chord condition
number is about 1.33e9. This identifies a usable-reserve limitation that route
counts or entropy miss. It motivates distinct payload/temporal programs and
measured task-visible conditioning, not blind delay noise or a seed sweep.

**Physical boundary and safety:** the complete E143 timer is 1,982.24 seconds,
including preparation, training and initial/epoch evaluation. Training alone
takes 417.40, 414.61 and 414.70 seconds per pass. Peak RSS is 1,881,256 KiB
(1.79 GiB); sampled host MemAvailable stays at least 10,361 MiB. Every job runs
under the host lock, watchdog and explicit 8,192-MiB reserve. Across training,
148,378,296 raw sources become 1,334,237 new deep-input packets; the frozen
parent separately processes 18,373,004 coarse packets. Smaller new packet
counts do not imply a complete energy saving: wider maps, source transport,
both branches, backward and optimizer work must be charged. The partial
forward ledger and full provenance are in `results/e146/event_state_summary_20260930.json`;
physical traffic and joules are unmeasured. The exact completed calibration
source is archived, and all recorded final sources are SHA256 verified.

**Next:** preserve this transferable improvement while testing a single generic
encoder, exact pooled-state queries and useful route initialization/credit.
Compare frozen-feature training at the same capacity/budget before attributing
the gain quantitatively to deep adaptation. Freeze an official training and
selection protocol, reach the named approximately 96% official-test reference
range, and measure physical work at a fixed quality target. The present result
solves a real development plateau; it does not establish frontier parity.

## E138–E141: source information and a trainable temporal-mode intervention

**Historical comparison:** the four completed continuations below do not beat
the then-strongest 370/512 (72.265625%) common-model development score. None delivers
a new best. This private speaker-held-out protocol is not the official SHD test.

| One-pass continuation from the strongest parent | Fit correct / 6,144 | Held correct / 512 | Held NLL | Wall seconds | Peak RSS KiB |
| --- | --- | --- | --- | --- | --- |
| E138: existing source representation, terminal warm LR | 4,962 (80.76%) | 368 (71.875%) | 0.973563 | 625.69 | 665,076 |
| E139: learned fine channel/time source messages | 4,968 (80.86%) | 369 (72.070%) | 0.976990 | 658.80 | 1,718,160 |
| E140: fine sources plus learned temporal phase | 4,967 (80.84%) | 368 (71.875%) | 0.976120 | 996.79 | 1,835,124 |
| E141: same fine/phase learner, original parent frozen | 4,935 (80.32%) | 369 (72.070%) | 0.989443 | 961.25 | 1,788,360 |

E138–E140 use the same 6,144 fitting examples, disjoint 512 held-speaker examples,
warm old Adam state, core LR 0.00007, sample order and band/time augmentation.
E139 introduces 5,760 zero-initialized source-contrast parameters with LR 0.0003.
Its initial logits, hard winners, clocks and sparse deep packet extraction match
the parent. All 700 original cochlear channels have distinct source addresses;
raw event times enter the message before packet coalescing. Fine weights change
to norm 0.9712 and have nonzero gradient, but the single extra held decision is
insufficient to exceed the parent, and held NLL is slightly worse than the plain
continuation. Restoring these marks alone is not demonstrated as the solution.

E139 processes 49,437,469 augmented raw source events and 6,123,083 deep input
packets. Source projection has 1,581,999,008 multiplies and the same number of
aggregation additions, 195,938,656 packet divisions and 148,312,407 time-feature
exponentials, plus source contrast-table work. The old three-option/eight-layer
training evaluates 146,953,992 value alternatives. These are forward work counts,
not complete training FLOPs, physical memory traffic or joules. Peak RSS includes
the cached raw fitting data; a streaming loader would change this boundary.
All runs remain sequential under the host lock and explicit 8,192-MiB reserve.
The recorded continuation wall times start after model/data preparation and
include initial/final evaluation; complete guarded-job wall times also include
startup and loading. Neither timing boundary is a joule measurement.

E141 uses the same examples, augmentation, ordering, source/phase rates and
initial function as E140, but freezes every original parameter and calibration
buffer. No new-teacher update clips (0/1,536), versus 1,535/1,536 in E140.
All phase blocks change and the fine embedding norm reaches 1.03577. The parent
state remains exactly unchanged, yet accuracy still fails to beat 370/512.
The clipping covariance identity in theory §§224–225 therefore identifies a
possible optimizer confound, not the demonstrated cause or solution of the
quality gap. Sampled host MemAvailable remains at least 10,510 MiB.

**Next structural experiment (E142–E143):** `event_state.py` implements signed
unnormalized modal states, nonlinear vector gates, residual depth and winning
clocks. It learns source vectors before causal coalescing. E142 verifies exact
first-affine endpoint/teacher preservation (1.11e-16/7.77e-16), agreement with
complex modal recurrence, nonzero six-layer representation teachers, and one
emitted vector per supplied packet. Completed-query supervision has no final
clock teacher, as its output time is not observed by the loss. This does not
establish SHD accuracy, nonlinear raw-event equivalence or global trainability.

E143's measured 16-example resource calibration uses 550,060 KiB peak RSS and
1.141 seconds for four optimizer steps. Its eight-example development sample
is a resource check, not accuracy evidence. The exact calibration source is
archived under `reference/e143_event_state_calibration_20260930.py`; its SHA256
matches the completed calibration record. The larger uniquely named queue
trains 395,814 new parameters in a parallel six-block width-128 encoder while
retaining the immutable 53,296-parameter parent. Its initially zero correction
head preserves 370/512 exactly. Three passes over 6,144 examples, Adam 0.001
with cosine decay to 0.0001, and 10 ms nonempty global closures are declared
before outcomes. Its completed improvement is documented above; this is not a fourteen-layer
sequential classifier or a from-scratch benchmark. Theory §§226–230 distinguishes
linear reference inclusion, optimizer geometry and exact packet credit from
the remaining nonlinear pooling, selective-state and route-boundary questions.

Theory §§215–219 now separates the input quotient's irrecoverable Bayes-risk
gap from optimization failure, derives exact affine block coalescing and the
local source-label teacher, and specifies a signed learned temporal-mode
operator with paired winning state/output/clock semantics. The standalone
operator reproduces a diagonal complex SSM to 2.09e-17 state error; its decay/
frequency teachers and semigroup agree at float64 precision. It is not a complete
nonlinear reference classifier or a trained policy.

E140 changes the common receiver to damped coordinate-pair rotations, with
dimensionless phase per memory timescale. It adds 384 scalars and no emitted
events. Zero phase exactly nests the parent; all eight phase blocks receive
credit. The primitive analytic teacher matches to 1.09e-14, and the linear-work
scan agrees with the sequential recurrence to 4.44e-16. The whole-model restored
1e-5-L2 phase probe lowers fitting NLL by 7.15e-7, versus surrogate prediction
1.40e-6. This discrepancy preserves the distinction between an exact local
memory adjoint and the core's approximate hard-route teacher. Theory §§220–223
derives this operator, its nonzero phase teacher at initialization, conditional
depth bounds and a subsequent optimizer-preserving width construction.
The same-budget E140 run completes at 368/512 (71.875%), with fitting NLL
0.59909 versus 0.60256 for E139, and held NLL 0.97612 versus 0.97699. Every
phase block learns: final block norms range from 0.6462 to 1.3364. Thus this is
not a missing-gradient or frozen-parameter result, but it gives no accuracy
record advance. Recorded continuation wall time rises 51.3% relative to E139;
peak RSS is 1,835,124 KiB. Sampled available memory stays at least 10,444 MiB,
above the 8,192-MiB guard. The complete source/phase model has 59,440 stored
parameters. No extra events are emitted, but rotations, trigonometric work and
their backward graphs add cost. Conditional stability and available local
credit have not supplied competitive content-dependent representation learning.

Completed source/continuation records:
`results/e122/d8_n6144_best_warm_s6_e1_20260930.json`,
`results/e139/d8_fine_source_n6144_warm_s6_e1_20260930.json`,
`results/e139/fine_source_contract_v2_20260930.json`,
`results/e139/temporal_modes_contract_20260930.json`,
`results/e140/phase_contract_20260930.json`.
The completed phase record is
`results/e140/d8_phase_source_n6144_warm_s6_e1_20260930.json`.

## E137: compact twelve-layer learning isolates useful angular adaptation (30 September 2026)

A rank-16 bank/channel/class memory query has **4,968 decoder parameters**.
The learned-angle model trains **6,476 parameters**, versus 140,428 for the
full-state E136 prototype (21.7× fewer trainable parameters). All arms also
retain the **53,296 frozen pretrained key parameters** and their computation;
this is not a 21.7× total-model or energy reduction.
All 1,024 query-fitting utterances were already in the key parent's 4,096-fit
set. All 512 held utterances were in its development split, with disjoint
fitting/held speakers. This is a supervised-key transfer/intervention screen;
`results/e137/pretraining_overlap_20260930.json` preserves the overlap audit.

Matched three-pass outcomes on the same 1,024 fitting/512 held utterances:

| Angular adaptation | Fitting correct | Held correct | Fitting NLL | Held NLL |
| --- | --- | --- | --- | --- |
| Learned | 690/1,024 (67.38%) | 243/512 (47.46%) | 1.3654 | 1.7220 |
| Frozen | 591/1,024 (57.71%) | 181/512 (35.35%) | 1.7322 | 2.0507 |

Both embeddings and compact heads learn; only angle adaptation is disabled in
the control. Initial state, logits, calibration, immutable key checkpoint,
examples, sample order, learning rate, update budget and current source hashes
match exactly. The held gain is **62/512, or 12.11 percentage points**; 102
utterances are correct only with learned angles, versus 40 only with frozen
angles. The paired descriptive SE is 2.27 points, excluding seed and speaker-
population uncertainty. Every learned angle layer receives nonzero credit.
This is evidence of useful angular adaptation under matched decoder capacity,
not just gradient support or memorization by a huge head.

The compact model remains below E136 full-state final accuracy 57.8% and the
separately trained common model's 72.3%. Its fitting loss continues to fall,
so capacity, optimization budget and speaker transfer remain distinct gaps.
All 768 optimizer steps clip in both compact arms. The rank constraint is
explicit; there is no theorem that a rank-16 query preserves arbitrary memory
classification. Direct all-layer state queries do not establish deepest-only
serial composition, and inherited supervised keys prevent a from-scratch claim.

A restored final-checkpoint audit on 64 fit and 64 held utterances finds angular
mean-gradient cosine **−0.0399**, versus **+0.0347** for the query and **+0.1324**
for the embedding. Angular squared-mean/mean-squared batch gradient ratio is
0.0587 on fitting data; the distinct-batch inner-product estimate is −0.2007.
These are noisy descriptive statistics from 16 minibatches per split, not a
population diagnosis. A normalized angle step of L2 size 0.001 changes fit/held
NLL by **−0.001682/+0.000190**; first-order predictions are
−0.001712/+0.000156. An independently restored query step of the same size
changes losses by **−0.001891/−0.000108**, close to predicted
−0.001893/−0.000109. Actual finite signs agree with the observed credit geometry.
No development-label updates are retained. This identifies fitting-speaker
invariance and reusable gradient alignment as targets beyond mere transport.

The learned run resumes after a daemon restart from its completed first-pass
checkpoint, preserving optimizer and next-pass RNG under a new result/queue
name. The old partial result remains intact. Accounted completed-boundary wall
time is 447.8 seconds, excluding lost uncheckpointed work and including repeated
resume setup, versus 399.0 seconds for the control. Peak RSS is 517,072/443,428
KiB. The RSS watchdog stays enabled; sampled host MemAvailable never drops
below **11,765 MiB**, above the 8,192 MiB floor. Only one queue job runs at once.
Terminal-query forward contractions are 115,028 MACs versus 138,900 for the full
head; complete training work, physical traffic and joules remain unmeasured.

Completed records: `results/e137/compact_comparison_20260930.json`,
`credit_statistics_20260930.json`, `compact_contract_20260930.json`,
`geometry_contract_20260930.json`, `resource_guards_20260930.json`.

## E136: completed twelve-layer memory-query learning and attribution (30 September 2026)

The observable-state classifier completes three passes on 1,024 unaugmented
fitting utterances and 512 held training-file speakers. It inherits immutable
keys from the E122 eight-layer/4,096-fit checkpoint; those key layers are reused
cyclically through twelve trainable exchanges. Class loss is assigned only at
the completed utterance. No official test data are read.

The full-state query reaches fitting accuracy **98.1%, 100%, 100%** across its
three passes; held-speaker accuracy is **59.0%, 56.3%, 57.8%**. Every angle layer
receives nonzero credit and changes. With the emitted-packet-only query, fitting
accuracy is **31.2%, 38.5%, 46.0%** and held accuracy **18.4%, 25.0%, 29.7%**.
Both retain the same actual winning key program and calibration, but this is
**not capacity matched**: reading all states adds 138,240 active decoder
weights. Full-state training has 140,428 active parameters versus 2,188 for the
packet query (the masked nominal head still stores unused weights).

Resetting all exchange angles in the full-state trained checkpoint, while
retaining the learned embedding/decoder/calibration/keys, preserves 1,024/1,024
fitting decisions. Fitting NLL rises from 0.001576 to 0.005248. Held decisions
fall from **296/512 to 278/512 (57.81% to 54.30%)**, and held NLL rises from
1.5946 to 1.7816. This demonstrates angle contribution/coadaptation while
identifying the decoder's ability to retain fit. It is a frozen intervention,
not a retrained fixed-angle control. Neither prototype beats the separately
trained common model's 370/512 (72.3%) development result.

Both complete screens take about 424 seconds on the recorded one-thread CPU,
with peak RSS about 523,000 KiB and the host memory floor preserved. These are
not energy or matched-quality work claims. The all-layer memory query creates
direct supervision paths; a deepest-only serial-composition claim still needs
its own intervention. Contracts and completed outputs:
`results/e136/scattering_query_comparison_20260929.json`,
`scattering_angle_ablation_20260929.json`, `reachability_contract_20260930.json`.

The next intervention, E137, uses a bank/channel/class factorized query with
4,968 head parameters. Learned/frozen angle arms start with exactly equal
logits and share data, sample order, keys, calibration, readout capacity and
optimizer budget. The compact query and its exact state teacher agree with a
dense oracle to at most 4.2e-17; all twelve angle layers receive credit on the
four-utterance contract. Completed empirical comparison is required before
attributing transfer to angular adaptation. Theory §§210–213 derives the
class-visible control kernel, costed query, categorical route covariance and
correlation-preserving optionality propagation. It also reconciles §209's
second-moment identity with the existing mean/variance distinction in §161.

## E134: full value learning separates gradient support from transfer (29 September 2026)

Eight layers, 4,096 fitting utterances, 512 held-out training-file speakers,
one augmented pass, the same parent checkpoint, fresh Adam at 0.0003, batch
four, seed six. Both arms train the same 58,048 value/embedding/memory-time/head
parameters, with no local loser surrogate. Policy tensors are frozen in both;
only one computes functional keys from an immutable separate program. Initial
predictions, sample order, augmentation and update budget match exactly.

Both arms finish at **349/512 (68.16%)**, below the parent's **370/512 (72.27%)**.
Fixed/coupled held NLL is 1.1071/1.1006 versus 0.9827 at the parent. Clean fitting
NLL improves to 0.6076/0.6108 from 0.6204; fitting accuracy is 79.91%/79.71%.
Every value map changes and receives nonzero gradients. Fixed key winner counts
and clocks are preserved exactly; coupled functional schedules change. The
finite credit contract agrees with predicted loss reduction at a small step.

This rules out a gain from functional key freezing **in this full-value phase**.
It does not erase E130's isolated harmful boundary crossing, and it does not
attribute all remaining errors to routing. Interior credit support, optimizer
behavior, representation selectivity and transfer are distinct gaps. The
parent remains the strongest common SHD result. Fresh-optimizer continuation
differs from the earlier parent training; this is not a matched initialization
or from-scratch architecture comparison.

1023/1024 fitting steps clip the total gradient in each arm. Fixed/coupled
wall times are 511.3/370.2 seconds with the same cached initial-evaluation
boundary, and peak RSS is 582,340/588,060 KiB respectively (read result files
for recorded hardware). Extra immutable key computation is charged. No total
training FLOPs, memory-traffic or joule measurement is available. Completed
audit: `results/e134/full_value_comparison_20260929.json`.

## E136: reversible value transport and the memory boundary (29 September 2026)

Theory §§206–208 replaces accumulated residual gain bounds with an orthogonal
packet/state exchange conditioned on independent keys/angles. One addressed
state is updated and one packet emitted per event; losing alternatives remain
counterfactuals. The affine event scan uses fewer than 2E combines and handles
complete swaps without division by a small cosine. The local angle teacher is
`g_state·packet_out - g_packet·state_out`, including the state suffix's utility.

The numerical contract computes the full **180×180 augmented Jacobian through
12 layers**. Singular values are [0.9999999999999989, 1.0000000000000007], and
the maximum orthogonality error is 3.3e-16. Scan/sequential outputs are exactly
equal in the fixture; payload-plus-state and adjoint squared-norm discrepancies
are 7.1e-15 and 1.8e-15. Angle finite-difference error is 3.3e-10; the local
angle teacher matches exactly.

The contract also verifies the essential counterexample: a complete exchange
can store all input information in memory while emitting zero. Consequently,
a sequence classifier must expose or drain retained memory at its supervised
query. The proof includes initial/final states and conditions on keys and
angles. It ensures conditioned reachable value transport, not arbitrary
value-dependent policy gradients, useful readout alignment, parameter-gradient
noncancellation or classification convergence. The primitive contract alone is an implemented mechanism/theory advance,
not an accuracy result; the subsequently completed classifier is recorded above. `results/e136/scattering_contract_20260929.json`.

## E135: completed content-key continuation and attribution (29 September 2026)

The full-value content continuation finishes at **349/512 (68.16%)**, equal to
the plain E134 control. Held NLL is 1.107008 versus 1.107095, and clean fitting
NLL 0.607567 versus 0.607569. All eight query and key matrices change and receive
nonzero gradients. Removing content retrieval from the trained checkpoint while
preserving its values, head and actual key program leaves accuracy at 349/512
and changes held NLL to 1.107046. On 64 clean fitting utterances, content
changes memory payloads by at most 0.01156 and logits by at most 0.002122
(mean absolute logit change 0.0002973). This identifies **weakly exercised
retrieval**, rather than an absent query/key gradient, in this continuation.

The first-order kernel's content weight ratio is analytically capped at 5/3.
Its exact balanced query teacher is a key/value covariance aligned with the
returning task teacher; the direct formula matches autograd to 4.2e-17. Its
actual normalized matrix partitions tighten the trained conditional transport
interval from [0.00336, 21.25] to [0.02420, 12.11]. Neither interval is an
optimizer/generalization proof. A small fixed-schedule fixture verifies its
partitioned Jacobian and inverse bounds numerically.

At this budget the extra retrieval costs 1580.9 versus 511.3 seconds on the same
one-thread CPU boundary, and peak RSS 1,270,016 versus 582,340 KiB. Its value
memory scan has exactly 5× the plain state width; recorded content scan,
projection and retrieval contractions total 133.83G partial forward FLOPs,
of which 80.61G are expanded scan contractions. These are not total training
FLOPs, memory traffic or joules. The new primitive has earned a verified
teacher and implementation, but no quality/resource advantage here. Next
changes must increase useful function movement/selectivity or change the
supervised memory interface; another seed does not address this diagnosis.

Completed records: `results/e135/content_comparison_20260929.json`,
`content_ablation_20260929.json`, `covariance_contract_20260929.json`.

## E135: content-key temporal memory contracts (29 September 2026)

A bounded positive retrieval kernel is accumulated in affine temporal state
inside the existing winning-value stream. A paired-feature initialization
recovers the old temporal mean **exactly**, while diverse keys provide a
nonzero query teacher. Query teaching then unlocks key teaching. Algebraically
compressing paired features to a mean plus four signed content moments reduces
the candidate's state from 265 to **165 scalars per time bank** at width 32;
the old plain memory uses 33. Kernel weights remain positive, and the event
scan has linear payload work with no production event-pair matrix.

The contract nests the parent's logits exactly, preserves actual hard winners
and clocks, and matches a direct reference to 6.7e-16. Payload/time/count/tau/
query/key gradient discrepancies are at most 2.5e-14. Initial query/key gradient
norms are 0.4863/0; after a query step the key norm becomes 0.07086. These are
small deterministic mechanism checks, not speech accuracy. The conditional
depth bound must include content-dependent memory feedback: the conservative
initial transport interval is [0.00345, 21.12], substantially looser than for
plain means. This does not establish well-conditioned optimization.

Theory §205 derives the covariance that actually teaches query selection and
the extra state/projection/retrieval work. The speech continuation uses the
same parent, augmented fitting pass and full-value optimizer budget as E134,
plus 2,048 content query/key parameters; completed outcomes are recorded
separately. Core hard races and sparse event packet semantics are unchanged.
Contract: `results/e135/content_contract_20260929.json`.

## E133: learned deep language prediction without explicit experts (29 September 2026)

The same common model, evidence_count=0, no pointer/copy/phase predictor: 8,192
training next-character targets, 32-character causal contexts, 1,024 reserved
validation-region targets, four passes, width 32, seed 6. The eight-layer run
improves from 4.7523 to **3.3951 dev bpc**, versus **3.4638** with one layer.
Fitting scores are 3.0643/3.2343. Every layer's value, route and log-time-constant
parameters change; all eight value-gradient norms remain nonzero. This provides
a generic neural trainability/depth foothold without an explicit statistical
expert. It does not reproduce the strongest specialized language result.

Count/time/last-character-preserving input probes increase the eight-layer loss
to 3.8046 (shuffle preceding order) and 3.7771 (unrelated preceding context).
One-layer scores are 3.7750/3.7251. These are frozen perturbations rather than
retrained controls. The last-character-only deletion gives severe out-of-support
errors (36.91/18.28 bpc) and cannot establish a useful context advantage because
it changes length and time-feature support. The matching count-preserving probes
are the interpretable sensitivity measurements.

Depth brings additional cost: 53,430 versus 7,671 parameters; 170.52 versus
27.44 s total CPU wall time. Recorded training-forward map/scan contractions
are 111.38G versus 14.04G FLOPs, excluding backward/optimizer and other work.
One instrumented 16-query training batch has forward/backward contraction
estimates 54.01M/108.13M for depth eight and 6.82M/13.61M for depth one. These
operator counts have incomplete coverage and are not hardware measurements.
Physical memory traffic and energy are unavailable. Inference map/scan/head
contractions are 1.269M/0.162M per replayed query. Both depth choices therefore
remain quality/work tradeoffs; this is not a measured efficiency improvement.

The screen is one seed, equal width/data/presentations, not matched parameter
count or a tuned Transformer/RNN/state-space comparison. Official test data are
untouched. Exact commands, layer diagnostics, source/data hashes and profiler
coverage: `results/e133/generic_language_d{1,8}_s6_20260929.json` and
`generic_language_audit_20260929.json`. The current deterministic surrogate is
used; E132's stochastic joint score law is not deployed in this run.

## E132: exact joint race credit, deadline teaching and information geometry (29 September 2026)

The proposed exponential race separates mark probabilities from total arrival
intensity. The joint likelihood includes winning identity, waiting time and
survival when no message arrives. Its censored Fisher matrix is block diagonal
and both mark/clock blocks scale with firing probability. This derives how a
label window can teach arrival intensity despite nonresponse, and why silent
initialization starves both channels of information.

Numerical contracts give maximum joint-gradient finite-difference error
8.83e-12 and Fisher quadrature error 4.86e-17. A depth-three, three-choice
conditional tree enumerates all 27 actual leaves and matches the score gradient
to finite differences within 1.53e-11. A discontinuous deadline loss has zero
ordinary sampled time derivative but nonzero exact clock credit; Monte Carlo
agrees with that expectation. Conditional candidate averaging reduces the
measured estimator variance on the synthetic example. Actual suffix replay
cost must still be charged. These are analytic contracts, not a trained speech
or language result; the current deterministic model is unchanged. Theory
§§197–201 and `results/e132/joint_race_contract_20260929.json` preserve the proof
scope and audited quantities.

## E128–E131: useful deep credit exists; winner coupling can defeat a finite representation update (29 September 2026)

**Fitting-only geometry:** 240 utterances, six per class in each of two disjoint
fitting-speaker groups, at the 72.3% checkpoint with zero context columns.
Exact and surrogate aggregate group cosines are -0.1183/-0.1228; their mean
gradient norms are about 10% of the average individual norm. Ordinary balanced
descent would worsen eight sampled class losses. Nevertheless, an exact
common-direction certificate across all 20 classes has lower/upper improvement
bounds 0.08692/0.08994 per unit parameter radius. Exact/surrogate class-gradient
cosines are 0.9928–0.9992. New context credit has useful support; interference
and realized boundary changes remain separate issues. No held-out labels were
used. Full matrices: `results/e128/class_speaker_geometry_20260929.json`.

**A finite counterfactual teacher:** E129 combines exact common descent with
the surrogate increment while preserving half of each predicted conditional
improvement. All 40 class/group predicted derivatives are negative, but twelve
dyadic finite radii fail the declared hard-loss constraints. Every trial lowers
average fitting loss. No update is accepted; the restored checkpoint retains
370/512 held-out answers. This is a failed stringent local constraint, not
evidence that ordinary aggregate-loss training is impossible.

**Causal isolation of the discrepancy:** E130 applies the same fitting-only
direction with realized choices, frozen winners, frozen arrival order and both
frozen. At radius 2.44e-5, one realized winner changes and the worst conditional
loss rises by 3.01e-5. Frozen winners make every condition improve; freezing
only order leaves that increase. At radius 0.001, fixing winners cuts prediction
error from 0.00251 to 0.0000451. This establishes the winner-change obstruction
for the tested update, not the cause of every recognition error. Frozen
histories are diagnostics rather than deployment models.

**Architecture guided by that isolation:** E131 introduces a separate local
key stream. It computes actual hard choices/clocks from each observed query;
the value stream learns new joint content under those computed choices.
Initial predictions match all 4,096 fitting and 512 held-out parent answers.
Small/large value perturbations leave audited key winners and clocks exactly
unchanged. A value step predicts loss change -0.0008240 and realizes -0.0008225.
Query isolation and checkpoint roundtrip pass. Two added value maps contain
6,336 trainable scalars; the key stream adds 52,616 frozen parameters and real
execution cost. Matched value-only continuations use the same source, examples,
augmentation, calibration, Adam state and update budget. The separate-key arm
finishes at 364/512 (71.1%), compared with 365/512 (71.3%) for the shared-stream
arm and 370/512 (72.3%) at the parent. Fitting accuracy increases to 81.84%/81.64%
from 80.08%; held-out NLL worsens to 1.0176/1.0173 from 0.9827. Every old value
parameter and cloned key parameter remains unchanged. Separate-key winners and
clocks stay fixed on all three clean evaluation splits, while shared-stream
choices change. This validates structural decoupling, not a held-out gain.
Forward value-map evaluations are 65.30M versus 97.95M, with 32.65M of the former
belonging to keys; the separate key scan adds 65.14M compositions. Wall times
have different initial-evaluation reuse, so they cannot establish a speedup.
No energy is measured. `results/e131/key_value_comparison_20260929.json` records
the matched contracts and work. Theory §§190–196 separates supported credit,
finite utility, key/value learning and future optionality.

**Equivalent credit, different runtime (E127):** winner-only value
differentiation retains every loser score comparison and exact forward values;
real-checkpoint gradient errors are below 9e-8 relative L2. It reduces estimated
value-map forward/backward contractions by one third, including recomputation,
but is about 12% slower on this one-thread CPU. The default stays `full`.
Neither contraction work nor CPU time is a joule measurement.

## E122–E125: deep speech improves; consolidated advantages receive explicit work ledgers (29 September 2026)

**Preserved arithmetic:** the common class's phase-only configuration reaches
3,440/3,440 unseen mod-17 triples. It stops after 47 complete fitting passes
(69,231 presentations), at a zero-update fixed point of the local teacher.
All 69 phase scalars exactly match the guarded E121 composite after its
200-epoch budget; the update count remains 29,003. Training/evaluation takes
3.29 s with peak RSS 248,440 KiB. The certificate proves the rule on all 4,913
possible triples. No unused embedding, generic carrier or dense head executes.

**New same-example dense controls and work audit:** one width-32/two-layer
setting, seed 6; the arithmetic controls use E121's fit split and 200-epoch
schedule. Both fit every training triple but fail to learn the unseen rule.
Recall controls receive all 4,000 pointer-fitting and 512 neural-fitting
examples, eight times. These are small configuration controls, not tuned best
LSTM/Transformer models. The stronger historical retrieval controls remain
separately documented.

| Task / configuration | Held-out accuracy | Estimated logical operations / query |
|---|---:|---:|
| Arithmetic: common periodic path | 100% | 188 |
| Arithmetic: common two-layer carrier + phase | 100% | 71,260 |
| Arithmetic: LSTM | 67/3,440 (1.95%) | 104,518 |
| Arithmetic: Transformer | 124/3,440 (3.60%) | 155,592 |
| Recall at 4× context: common two-layer + pointer | 256/256 (100%) | 728,602 |
| Recall at 4× context: LSTM | 19/256 (7.42%) | 2,242,695 |
| Recall at 4× context: Transformer | 19/256 (7.42%) | 4,460,917 |

The arithmetic primitive uses about 556×/828× fewer estimated operations
than these LSTM/Transformer controls. The configured two-layer recall model
uses about 3.08×/6.12× fewer. The ledger charges actual configured maps,
router choices, scan combines, normalization, pointer search and clock
diagnostics. Its units are 2 per MAC and 1 per other scalar operation,
nonlinear function or estimated comparison; logical reads are separate.
These are inference work estimates, not measured joules or training counts.
The latest default-core extraction contracts still pass exact legacy logits,
gradients and winners. Source, IDs and full ledger: `results/e123`–`e125`.

**Speech:** matched 2,048-example continuations improve pooled held-out
accuracy from 350/512 (68.4%, ordinary inputs) to 369/512 (72.1%, time/channel
augmentation), from the same checkpoint, sample order and training budget.
The paired gain is 36 corrected and 17 lost answers, concentrated on speaker 3.
Adding fitting data to 4,096 and two further passes reaches 370/512 (72.3%).
These are development utterances from held-out training speakers, not the
official test set; the second 256 examples share those speakers.

**A targeted readout experiment:** zero-initialized bounded event weighting
has exact mean equivalence and its local covariance/teacher gradient passes
the numerical contract. One matched extra epoch reaches 353/512 (68.9%) with
the old mean and 356/512 (69.5%) with the learned key, both below their 72.3%
starting point. Weighted versus mean changes five errors to correct and two
correct answers to wrong (descriptive discordance p=0.453). Nonzero key credit
is established; improved recognition is not.

**A more specific remaining gap:** the saved-prediction class audit finds
that the 72.3% checkpoint recognizes class 3 on 135/206 fitting utterances
but only 4/26 held-out utterances; class 19 is 126/204 versus 3/26. Their most
common held-out confusions are 3→8 (15 cases) and 19→9 (20). Both classes have
ordinary fitting support, near 205 examples. The extra mean/weighted passes
reduce fitting recognition of both classes as well, despite nearly unchanged
aggregate fit accuracy. Weighted pooling mainly helps class 7 and leaves these
confusions. This separates weak class fitting plus speaker transfer from a
uniform inability to learn. It does not yet identify the representation or
credit mechanism causing those confusions. Next inspect class/speaker gradient
alignment and retained temporal information, using fitting-only interventions.
Details: `e125/class_summary_20260929.json`; theory §§185–187.

**Presentation:** the regenerated report is a project entry point: plain-language
identity, strongest evidence, consolidated accuracy/work plots, principles,
applications and frontier potential. Ongoing speech is an appendix; detailed
diagnoses and operational material remain in research documentation. Both PDF
editions, Markdown, figures and completed JSON summaries are in the host commit
helper. In-progress JSON is skipped.

## E121: arithmetic capability restored in the shallow shared model (29 September 2026)

The two-layer shared model now retains the missing periodic computation.
All runs use the same 1,473 mod-17 triples (30%), seed 6, width 32, example
order and 200-epoch neural optimizer schedule. Every one of the **3,440 unseen
triples** is evaluated; the old 256-example slice remains separately recorded.

| Configuration | Fit correct / 1,473 | Unseen correct / 3,440 | Training/evaluation wall time |
|---|---:|---:|---:|
| Plain two-layer core | 354 | 70 (2.0%) | 63.7 s |
| Phase memory + fixed bounded correction | 1,471 | 3,391 (98.6%) | 75.7 s |
| Phase memory + margin guard | 1,473 | 3,440 (100%) | 77.3 s |

The local phase state has 69 learned scalar parameters, initialized randomly.
It receives position-tagged operand symbols and a supplied period of 17. Its
rotation/reflection algebra realizes the earlier E41 chain; no arithmetic
answer is encoded in its inputs or update. A hard class-clock race answers.
The phase-only module is 100% correct on every unseen triple in both coupled
runs. Both phase runs make exactly 29,003 local mistaken-example updates.
The shared neural core has 14,060 Adam parameters and executes two layers.

**An identified integration error:** the fixed correction changes 49 correct
phase answers at small margins, while preserving all 3,224 queries covered by
the original margin certificate. An adaptive bound `min(0.25, clock_lead/4)`
protects every clock winner without consulting its label. The separately
trained guarded run has 3,440 certified queries and zero winner changes. The
neural branch can train confidence but cannot correct a wrong phase class in
this mode. This establishes retained shared-model arithmetic, not independent
arithmetic discovery by the generic carrier. The interventions also change
the teaching rule and readout; this is not a phase-state-only ablation.

The theory is in §§181–182. Serial/reduced state agreement, signed occurrence
derivatives, repeated-symbol credit, the margin bound and serialization are
checked in the guarded contracts. The old fixed-bound result is retained.
The earlier isolated phase probe reaches perfect accuracy on the 256-example
slice after 50 epochs in 5.8 seconds total; it omits the generic carrier.
Peak process RSS for these shared runs is about 329 MiB. No joules were measured.

This answers the representation question in the tested setting. It does not
establish a scaling law, a global convergence theorem or a multi-seed benchmark
lead. E41's older 99.4–99.9% figures use different seeds and a memorizer/sleep
branch. The E120 eight-layer/eight-epoch failure remains an initial screen,
not an equal-budget depth comparison. Task-specific depth is deliberate:
shallow synthetic configurations and deep speech share implementation and
primitives, with separate fitted weights.

## E120: one shared core, separate task fits (29 September 2026)

The shared package now contains the E119 deep event memory/carrier, conditional
count/exposure evidence, E61 relative pointer routing, causal observed-prefix
queries and categorical/hazard objectives. **There is no joint training.**
Each task uses separate weights, memory contents and appropriate targets.
Hidden races remain winner-only; losing payloads supply training-only score
credit. The common eight-layer core runs on every new neural task.

**Completed screens:** text8 3.022 → 2.915 development bpc (2,048 neural fit /
256 dev, with a separate 32,768-character evidence bank); temporal composition
249/256 (97.3%); pooled MNIST 194/256 (75.8%); first-second DVS Gesture 26/44
(59.1%, users disjoint); corrected recall 256/256 at both standard and four-times
context. The SHD extraction preserves 151/256 from the frozen E119 checkpoint,
with exactly matching logits, gradients and winners on the audited batch.
These are small development screens, not new full-scale superiority claims.

**A mechanism isolated and repaired:** the first combined recall model fell to
23/256 at four-times context although its pointer alone stayed perfect. Every
fitting example had the same event count, so the count-feature coefficient was
unidentified. A 1e-4 standard-deviation floor turned the longer input into a
1299.28-unit feature, adding an untrained score as large as 78.97. Projecting
only that count contribution restored 256/256 with every learned weight frozen.
A new run with fitting-only count support also retained perfect extrapolation.
Variable-count calibration is unchanged. Both failed and corrected runs remain.

**Two useful failures:** market training loss fell to 1.546 nats/event while
development loss worsened to 3.823, versus fixed evidence 3.670. A frozen deletion
of the additive neural head improves that to 3.549; text likewise improves from
2.020 to 1.942 nats. The deep feature gate remains in that deletion: these are
not retrained memory-only controls. Modular addition fits 176/1473 but recognizes
6/256 unseen tuples after eight epochs. The successful older periodic primitive
has not been ported; this is not a long-run grokking experiment.

**Theory §§176–180:** derive the common natural-score interface and its credit;
prove explicit prefix closure; distinguish categorical shift invariance from
hazard clock information; identify the readout-support failure; show why any
two uniform modular operands have zero population label correlation for a
three-operand target. This yields concrete coupling and representation tests.

**Cost/protocol:** one guarded CPU job at a time, one thread, ≥8 GiB host reserve;
new training peak RSS below 0.5 GiB. Prefixes are replayed, local maps/readouts
remain dense, and no joules are measured. No official real-data test is read.
Memory evidence is fitted on disjoint examples before neural fitting. Native
hold/veto, periodic state, persistent online scheduling and continual-learning
coverage remain to integrate. See [SHARED_MODEL.md](SHARED_MODEL.md).

The report is reorganized into a concise accomplishments-first narrative;
the previous long account is retained under report/archive/. Ongoing speech
work remains in an appendix. The default report now builds Markdown and PDF
from the same editorial source, with a named copy to survive cross-host status
PDF refreshes. The host commit helper includes all new package/report sources,
queues and result JSON, while excluding checkpoints and logs.

---


## 2026-09-29 — Linear-work memory funds stronger eight-layer SHD recognition (E119)

**Completed final endpoint: 151/256 (58.98%) development accuracy**, compared with the earlier E118
104/256 (40.625%) on identical held-out-speaker examples. Sixty-four earlier errors are corrected and
seventeen earlier correct answers are lost. Fit accuracy is **827/1024 (80.76%)**. Best development
checkpoint is epoch 7 at **163/256 (63.67%)**, distinct from the final epoch 8. Final fit/dev NLL is
0.5570/1.6600; the earlier dev NLL was 1.7496. Online training NLL falls from 2.5123 to 0.5786.
The endpoint is selected by the predeclared eight-epoch budget, not by the peak development score.

The architecture is unchanged: width 32, eight bounded carrier layers, three local competing options,
only the winning value/delay emitted, with loser score credit during training. All eight routers
receive nonzero gradient and all three options win some events in each layer. There is no new
optionality reward. The progression increases fitting data from 512 to 1,024 and epochs from four
to eight, and uses a fixed cosine learning-rate schedule .003 → .0003. Conditioning remains fit-only.
This is a larger-budget learning result, not a single-factor attribution or a comparison to a
published official-test score. The official SHD test file remains unopened.

**An exact execution improvement:** pair reduction/prefix reconstruction replaces the O(E log E)
doubling memory scan with O(E) memory combines and an O(E) autograd graph. Sorting remains
O(E log E), and local vector maps remain dense. On the same earlier checkpoint and 256 examples,
all predictions agree, as do winners on the audited batch; relative parameter-gradient L2 error is
2.34e-7. Memory combines fall **21,553,320 → 3,925,048 (5.49x)**. Eight warm timing repetitions
on one Intel i5-4690 CPU thread give median inference **97.1 → 63.8 ms/batch of four (1.52x)**,
and forward/backward **347.2 → 206.8 ms (1.68x)**, excluding optimizer updates.
The singleton diagnostic initially rejected disconnected time/tau autograd inputs; these are exactly
zero derivatives for one event. Handling that mathematical zero allowed the full five-size audit to
complete. The failed first queue log is retained; it was not a failed model-equivalence result.

**Frozen final model, causal coalescing:** 10/20/40/80 ms windows give 151/142/130/114 correct out of
256, at 100/64.49/42.49/27.85% of the original input packets. Median model evaluation times are
5.109/2.791/2.030/1.563 seconds over 256 examples (three warm repetitions, loading/coalescing excluded).
Extra input delay is at most 0/10/30/70 ms. Counts are preserved and released only at closure.
The 20 ms setting saves 35.5% of packets and about 45% of CPU evaluation time while losing 3.52
accuracy points. This is an internal accuracy/work/latency tradeoff, not a measured energy win.

**Theory §§173–175:** derive linear-work memory and its exact reverse recurrence; factor local
log-time-constant eligibility into gap/time-scale, evidence mixture, and payload contrast; bound
root temporal-feature perturbation under causal coalescing. The latter is not a whole-model bound
because event histories and winners can change. Prefix scanning is established prior machinery;
its application removes redundant work from this event model without softening forward races.

**External references refreshed:** EventSSM 95.9%, S7 96.3%, and the dataset-maintainer leaderboard's
96.26 ± 0.08% provide accuracy targets. Chen et al.'s FPGA gives 93.4% with reported powers whose
power/throughput ratios are approximately 2.71 mJ/utterance (processor) and 16.44 mJ (whole SoC).
The original dataset paper reports an 85.7% LSTM, correcting the report's earlier approximately 70%
summary. Primary sources and comparison boundaries are in `SHD_FRONTIER_PROTOCOL.md`.

The guarded training run took **739 seconds**, peak process RSS **599,480 KiB (585 MiB)**, with one
thread and over 11 GiB available host memory in the watchdog samples. Per-epoch checkpoints now
include Adam and RNG state for true continuation. RAPL energy counters are unreadable here; no
joules claim is made. `results/e119/summary_s6.json` verifies data/prediction correspondence and
hashes all completed input artifacts. The report/PDF place this development evidence in Appendix A.

## 2026-09-29 — Eight-layer race learning, useful depth, and a conditioning diagnosis (§§166–172)

**First positive depth comparison in the new E118 carrier architecture:** 512 fitting / 256 held-out-speaker
development utterances, four epochs, seed 6, width 32 and counterfactual loser score credit. Depth 8 reaches
**104/256 (40.625%)**, versus **65/256 (25.391%)** for depth 1. Fitting accuracy is 258/512 versus 229/512;
held-out NLL is **1.7496 versus 2.2544**. Of the same held-out examples, D8 corrects 50 and loses 11 relative
to D1. Data/order/update count and conditioning procedure match. D8 has 53,296 parameters versus 7,537 and
does more event work. This is a development result with extra capacity, not matched-compute supremacy.

The smaller 128-fit/128-held-out, eight-epoch comparison does **not** show a consistent depth gain:

| Depth | Credit | Fit correct | Held-out correct | Held-out NLL |
|---|---|---:|---:|---:|
| 1 | Winner pathwise | 70/128 | 33/128 | 2.3988 |
| 8 | Winner pathwise | 64/128 | 34/128 | 2.4595 |
| 1 | Plus loser score | 70/128 | 41/128 | 2.3500 |
| 8 | Plus loser score | 76/128 | 30/128 | 2.4387 |

For D8, adding loser credit improves fitting and slightly improves held-out log loss but loses four correct
classifications. The initial predictions, conditioning statistics and data order match exactly between these
two arms. Last-layer router gradient is zero for ordinary terminal pathwise credit (winning delay cannot
change the terminal pooled class score), and nonzero for loser credit; this expected distinction is explicit.
All alternatives are locally scored, only the winner emits, and losing values get no ordinary value-path
gradient. Counterfactual training evaluates three value alternatives instead of one. A contract checks that
changing only losing value maps leaves the actual forward answer unchanged.

**Concrete conditioning intervention:** E117's fixed-route D8 carrier pilot reached 23/128 fitting and 9/128
held-out. A frozen deepest-feature ridge probe recovered 84/128 and 36/128, so information survived despite
the poor trained head. Input-feature probes were slightly better held-out (39/128), not evidence for useful
depth. A matched head-only experiment then held the deepest features, zero initialization, Adam and eight
epochs fixed. Fit-only covariance conditioning changed held-out accuracy **7/128 → 39/128** and NLL
**2.9728 → 2.2408**, isolating head optimization geometry as a real bottleneck in this representation.

**Analytical advances:** §§166–167 construct a normalized causal history operator with a whole-sequence
maximum-norm bound and dual L1 payload-credit bound, including history reuse. The D8 fixed-history gain
interval is [0.3436, 2.5658]. Real race switches/delays are outside that conditional certificate. §§169–170
derive timed losing-route score credit and the class-head covariance curvature. §§171–172 model optionality
as budgeted reachable correction directions. A local Gramian recursion is exact for independent controls;
shared router parameters add cross terms that can cancel, invalidating a naive sum over event alternatives.
Finite linear contracts check the support function, quadratic reserve and shared-control cancellation.

**Counterfactual transfer is still open:** actual shared-router replays on four fitting/four held-out examples
show loser-credit steps helping the former and hurting the latter at all three tested step norms. At norm
0.01, loss changes are −0.02898 / +0.05080. Ordinary pathwise credit has the opposite signs, +0.01464 / −0.00608.
These tiny diagnostics cannot estimate population utility. They motivate measuring transferable correction
directions, rather than rewarding same-sample gradient energy or merely more near-tied routes.

All scores are terminal, on the SHD training split with speakers 3/6 held out; no official test access.
Input packets release counts at the end of causal 10 ms windows. There is no silent-unit time grid;
CPU training uses an associative scan over actual arrivals and local dense vector projections. D8 runs took
178 s each for 128 examples and 335 s for 512; the runner kept process RSS below 0.9 GB and more than 11 GB
host memory available. Results and exact pair checks: `results/e118/matched_summary_20260929.json`.

Next: distinguish depth from parameter capacity, determine which deeper transformations add class-relevant
information, expand speaker coverage, and calibrate early decisions. Stable transport, useful depth and
transferable optionality are distinct questions; this run advances the first two without declaring them solved.

## 2026-09-29 — Count placement, evidence scale, and deep support (§§164–165)

A matched depth-4 screen used 512 fitting and 256 held-out-speaker development
utterances, seed 6, two epochs, corrected grid emission, deepest primary loss,
auxiliary weight 0.2, and no counterfactual route updates. Absent count marks
and shared-vector additive marks both ended at **14/256 terminal and anytime
accuracy**. Their final L4 support was **12.11% versus zero**; late-prefix NLL
was 2.9927 versus 2.9984. The count projection learned a nonzero norm, but
preserving multiplicity this way did not yield recognition.

A subsequent, matched `address_neutral` mode routes the first input layer from
the original embedding while adding the count only to its integrated payload.
It ended at **13/256 terminal, 14/256 anytime**, with **91.80% L4 support** and
late-prefix NLL **297.1397**. Training loss rose **232.64 → 497.94**. Thus high
activity and large gradients coexist with failed learning. The shared-vector
arm became silent; input address separation retained strongly miscalibrated
evidence. These are one-seed development results, not an SHD accuracy gain or
proof of a unique causal failure. The third arm was chosen after the pair.

The score/payload split is opt-in. Existing behavior remains the default;
checkpoint audit loaders preserve the new mode, and boundary-credit traces
use the actual route vector. The executable routing contract and exact result
paths are in `e83_countmark_summary.py` and
`results/e83/countmark_matched_20260929.json`. The initial deepest-loss gradient
norms matched across all three arms. Zero auxiliary gradient at L4 is expected
because auxiliary heads exclude that layer; it is not a new defect.

§165 derives the competing loss incentive: near uniform prediction,
expected CE = log(C) − alpha A + alpha² V/2 + higher-order terms. Uninformative
evidence has no alignment term A but pays a variance penalty V. Suppressing
such evidence can lower loss before representations learn. This supports the
§157 sparse serial residual continuation design, with bounded payload/state
gain and explicit event budgets, followed by terminal learning, prefix
calibration, and transferable route optionality. The loss calculation is
analytical; it has not uniquely established the cause of the observed collapse.

All three training jobs completed serially under `run_safe.sh`, in 311/297/322
model-reported seconds. Observed runner heartbeats stayed below 620 MB RSS
with over 11 GB available host memory. The current grid reference still
reports about 293k hidden state-vector updates per utterance; these results
establish no asynchronous execution or energy advantage.

## 2026-09-29 — Hybrid emission contract and optionality synthesis (§§155–163)

- **Confirmed implementation defect:** legacy `TVLayer` detects firing after current-bin arrivals but builds the
  outgoing payload from the previous state. The one-arrival witness returns payload 0 and derivative 0 instead of
  the consistent grid values 1.9544986 and 1.0852314. Arrival crossings require jump semantics, not an autonomous
  implicit-crossing derivative. Added opt-in `--spike_reconstruction grid`, with matching detection/reset/payload
  timestamps. Legacy remains the reproducibility default; this is a clocked reference, not an async solver.
- **Frozen prevalence audit:** 128 held-out-speaker development utterances; L1–L4 emitted events 892/171/36/32;
  current-bin state jumps coincide with 853/81/10/8; zero prior state occurs in 0/26/2/4. Coincidence does not
  establish that the jump alone triggered a spike. Result: `results/e83/emission_contract_s6_n128.json`.
- **Matched correction training completed:** seed 6, depth 4, 120 fit / 128 held-out, four epochs, same optimizer
  and route-credit configuration as `bundle_control`. L4 coverage ends at 96.88% versus 1.56%; terminal accuracy
  7/128 versus 8/128; race-plus-fallback 11/128 versus 8/128. No reliable recognition improvement is established.
  Grid train loss 191.46 → 60.51 → 25.06 → 17.64; held-out prefix NLL 22.30/128.11 versus 3.02/3.25 control.
  Post-arrival propagation survives, but evidence scale remains poor. The intervention also changes timing/reset
  discretization. Runner completed safely; 582 s model wall time, observed heartbeat RSS about 0.6 GB and host
  available memory above 11 GB. Result tag `emission_grid_recongrid`; no trained checkpoint was requested.
- **Causally eligible L4 boundary replay:** 1,024 development utterances, 256 batches. 76 batches have eligible
  nonrefractory receivers after a real selected-message arrival; 30 have a nearest candidate in the ±0.5 band.
  Of 21 in-band natural-off toggles, 14 help and 7 harm; mean batch-mean deepest loss delta −0.000708, SE 0.00527.
  The 46 out-of-band fallbacks have mean +0.005173, SE 0.002432. Do not pool these into a claim about near-boundary
  births or interpret the selected replay effects as accuracy improvement.
- **Analytical advance in optionality:** §§159–163 define budgeted attainable futures and distinguish information
  arriving before a choice from an oracle choosing on hidden future information. A standalone premium cannot in
  general use the same scalar backup as continuation value. Same-sample virtual progress has first-order expectation
  η[mᵀMm + tr(MΣ)], while independent adapt/evaluate progress has ηmᵀMm under fixed metric and iid assumptions.
  Thus the old metric can reward gradient noise. This is a derived mechanism to investigate, not a measured causal
  explanation of prior SHD failures. Exact E116 examples illustrate duplicate-route invariance and the noise term.
- **Actionable optionality (§163):** forced births in the legacy kernel used different payload semantics from natural
  births. A shadow must correspond to allowed controls, not just a forced event. Derived the minimum local
  control cost m²/(2aᵀMa) to cross an affine margin; shared-control constraints can make individually reachable
  events jointly impossible. This is a new proposal-design criterion, not a trained result.
- **Bird's-eye synthesis:** current task advantages belong to different model families. Priorities are correct
  hybrid credit, retained class information, useful serial depth, transferable optionality, calibrated stopping,
  affordable candidate search, and measured energy. Local work remains SHD; the AWS sibling owns non-SHD runs.
- **Report:** plain-language introduction and measured advantages now lead; SHD development moved to Appendix A.
  Expanded potential covers training economics, dormant capacity, persistent mobile learning, science/industry,
  hardware and infrastructure, and learned computational optionality. New results are visualized in the appendix.


One entry per result: what we ran, what came out, what it teaches us, what changes.
Newest first. Numbers are single seeds unless stated.

**E83 conditioned margin-support audit (§154).** A frozen seed-6 depth-4
control was replayed over 128 held-out-speaker utterances. We counted
nonfiring, nonrefractory receiver-time cells only after at least one actually
selected upstream message had arrived, and restricted candidate times to the
causal delay horizon. In the current $[-0.5,0)$ spike-proposal band, counts
were 22,553 / 4,313 / 195 / 50 across L1--L4, while the number of utterances
with at least one candidate was 128 / 114 / 24 / 4. In $[-0.25,0)$ the counts
were 4,205 / 1,074 / 57 / 7 across 128 / 81 / 13 / 2 utterances. Adjacent
time cells are correlated, so they are not independent route alternatives.
Many wider-band margins cluster at the $-1$ zero-voltage baseline; blindly
expanding the proposal could force ungrounded events. This measures a sharp
loss of near-threshold candidate support with depth, not candidate utility or
an accuracy gain. The next causal test must replay sampled births through the
actual suffix and measure deepest-head loss and descendant work. The read-only
audit ran through `run_safe.sh` in 35 seconds with the RSS cap at 1.8 GB and
host available memory above 12 GB. See §154 and
`experiments/queue/e83_conditioned_margin_audit.txt`.

**E83 six-epoch continuation check (§153).** We extended the matched seed-6
depth-4, 120-train/128-held-out-speaker immediate-credit and two-rollout
continuation-value runs from two to six epochs. Both stayed at 6/128 held-out
terminal accuracy in every epoch except the option arm's first epoch (7/128);
race coverage was zero throughout. Final training loss was 2.9950 and 2.9957,
near uniform 20-class loss. The immediate arm's L1--L4 support moved from
98.4% / 40.6% / 2.3% / 0% to 100% / 1.6% / 0% / 0%. The option arm began at
98.4% / 56.3% / 11.7% / 2.3% and ended at 100% / 8.6% / 0% / 0%; its L4
activity appeared again at epoch 3 but was absent at the end. Thus the update
briefly expands deep support and leaves more L2 activity, but this does not
carry class signal or create a persistent deep path. L4 pathwise gradient
norm was zero at the end in both arms. The added epochs reject simple
undertraining as the sole explanation for this configuration. The result
points toward censored proposal support and route survival; §153 derives why
increasing rollouts cannot recover a useful route omitted by the proposal.

**State-conditioned optionality pilot (§152).** We formalized optionality as a
state-, label-, horizon-, proposal-, and rollout-budget-conditioned reserve:
the expected best terminal loss among the factual continuation and $K$ sparse
counterfactual futures, compared with factual loss. A separate beneficial-mass
statistic records how much proposal probability reaches futures improving by a
declared margin. This separates uncertainty over class labels, surprise on the
observed label, entropy over route proposals, and future route value; none is
the inverse of another. We also derived that signed optionality differences
telescope along a full trajectory, so they must enter a continuation backup
or action-selection target rather than be blindly summed as an extra reward.

A matched depth-4, seed-6 experiment compared immediate-only threshold utility
with a two-rollout, continuation-aware utility ($\lambda=1$), both with causal
per-layer suffix-gradient substitutions, 120 training examples, 128
held-out speakers, and two epochs. Both arms ended at 6/128 terminal accuracy
(4.69%). The rollout arm's mean reserve change was $6.28\times10^{-5}$ then
$9.73\times10^{-4}$ loss units/action. At the 0.05-loss cutoff, both parent
and child had zero improving sampled futures: 0/48 per side in epoch 1 and
0/54 per side in epoch 2. Epoch-2 layer support was
83.6% / 4.7% / 1.6% / 0.8% (L1--L4), compared with 100% / 14.8% / 0% / 0% for
immediate-only. Nearly all measured suffix-correction norm still landed on
the output head. The treatment took 224 s versus 177 s control, at about 603
MB peak RSS and over 11.7 GB host memory available. This validates execution
of the estimator and matched update path, but provides no accuracy evidence
that optionality helps. Later route alternatives are still too rarely
reachable, and the estimator is specific to its sparse proposal. §152 records
the state-value formulation, uncertainty distinctions, breadth diagnostic,
potential-shaping caveat, and this negative training result.

**E83 scalar option-value training pilot (§151).** We turned the frozen
counterfactual scalar into a local threshold update and ran three serialized,
resource-capped depth-4 arms on seed 6: pathwise control, immediate-only
counterfactual updates ($\lambda=0$), and scalar optionality ($\lambda=10$).
All used the same 120-example training split, 128 held-out speakers, deepest
readout, and two epochs. Every arm ended at 6/128 held-out accuracy (4.69%),
the chance-level outcome; no recognition gain appeared. Race coverage was zero
in all arms, so every answer came from the terminal fallback rather than an
early confidence-triggered emission.

The scalar arm did move intermediate support: by epoch 2, held-out L2 support
was 27.3%, versus 8.6% in the pathwise control and 13.3% in the immediate-only
arm. L4 support was 0.78% (1/128) in all three. The sampled intervention
cascades added hidden events during training, but those events did not become
a sustained path to the deepest classifier. The scalar arm's mean suffix-step
advantage was positive but small in epoch 1 (+0.000143), then slightly
negative in epoch 2 (−0.0000031); its scalar utility also changed from
+0.00297 to −0.000252. This is an informative failure of the first update
rule, not a rejection of all scalar option-value methods: it localizes the
remaining problem to candidate availability, depth survival, and persistence
of task utility.

Epoch-2 pathwise gradient norms directly track this loss of credit to depth:
L3/L4 norms were 6e−5/0 for pathwise control, 0/0 for immediate-only credit,
and 3e−5/0 for scalar optionality. The scalar arm's mean L2 event count was
about 0.47 per held-out utterance (27.3% support and 1.71 spikes per active
example); L4 stayed at 0.78% support and about 0.02 events per example. The
integer-rounded `spikes_per_utt` field reports zero for sub-half counts, so
that display must not be read as zero events. Intermediate support therefore
did not imply a sustained event path or useful gradient at the deepest layer.
The next analysis should trace forced event births through their exact
descendants and measure both event survival and deepest-loss gradients,
rather than counting added events alone.

The implementation samples a layer and then a near-threshold event with
margin-weighted probability, but does not apply inverse-propensity weights.
Its expected update is therefore proposal-weighted. The scalar rule is exact
for its chosen local relaxation, while unbiasedness for a uniform candidate
objective is not established. The branch utility also uses a one-step clipped
suffix-SGD proxy; its scale is optimizer-dependent and $\lambda$ remains
uncalibrated. Next compare logged-propensity updates against the current
proposal-weighted rule while separately measuring forced-event survival to
L4 and deepest-head loss. All jobs ran under the serialized safe runner at
about 600 MB RSS and over 11 GB available host memory.

**E83 scalar route optionality through descendants (§§149–150).** We ran six
guarded frozen-checkpoint audits on the same stratified 32-example
held-out-speaker subset, selecting eight factual errors. The alternatives
included route swaps at two margin widths, closed-route birth, spike birth,
and their combined proposal pool. Route-only, route-birth, and spike-only
audits produced no immediate-loss or finite suffix-step advantage among their
sampled leaves. In the spike-only arm, verified interventions added events in
L1/L2/L3 on 13/11/3 leaves, but none reached L4. This shows that making more
local events is insufficient unless their descendants reach the classifier.

The combined action pool produced 31 leaves; all 31 forced interventions were
verified. Five leaves added L4 events (eight added L4 events total across
these separate replays). Four of 31 lowered deepest-head loss, four improved
the matched one-step clipped suffix-SGD progress, and three improved both.
One branch worsened current loss while increasing finite suffix progress.
The recursively backed-up scalar was positive on one of eight error trees at
each tested learning-option weight λ=0, 1, and 10, and on two trees at λ=100;
the second tree flips only because that uncalibrated high weight values its
larger suffix step over its worse current loss. The result is a small
checkpoint-conditioned route to deeper label credit, not a training gain or
SHD accuracy improvement. Action proposals were not paired with common random
numbers across arms, and no route-birth leaf was sampled in the combined arm.
The audit stores branch traces for inspection, while the proposed recursion
itself propagates one scalar and does not require a global tree.

**Theory update.** §149 now defines the terminal value as immediate
label-loss advantage plus measured finite suffix-learning progress, and backs
that scalar through counterfactual descendants. §150 distinguishes event
propagation from reachable class utility and reports the six audits. λ needs
calibration against real held-out loss after a sparse scalar update; this
virtual step remains a development diagnostic. The next decisive experiment
is a bounded update on training data with recorded proposal propensities and
held-out-speaker evaluation after training.

The jobs ran serially through `experiments/queue/run_safe.sh` with 3.6 GB
virtual-memory cap, 2.6 GB process-group RSS cap, 8 GB host-availability
floor, and 20-minute watchdog. Peak observed RSS was 407 MB and available host
memory stayed above 11.8 GB; all completed jobs exited 0. The first spike-mask
attempt exited before useful work due to a tensor-broadcast shape error; the
mask was corrected and the guarded rerun succeeded.

**E83 1,024-example spike-pair audit: event propagation is not pair synergy (§146).**
The frozen held-out-speaker audit tested every binary outcome for one selected
near-boundary pair per batch. L2 yielded 141 control and 69 late-only pairs;
L3 yielded 31 and 11. Among natural-off pairs, opening both raised downstream
spike counts in 38/64 and 19/34 L2 cases, and 6/10 and 3/5 L3 cases. The same
double opening lowered deepest-only loss in 8/64 and 4/34 L2 cases, and 5/10
and 4/5 L3 cases. At L3 the candidate pool is tiny: the late-only median
double-open loss change was −0.032, but a single +1.40 harmful outlier made
the mean harmful. These are checkpoint-conditioned interventions, not an
accuracy gain.

The four-corner difference-in-differences $\Gamma=L_{11}-L_{10}-L_{01}+L_{00}$
isolates credit specific to the pair. Across 113 natural-off pairs, no
double-open intervention improved deepest-only loss when both singleton
openings were individually non-helpful. With $|\Gamma|>0.01$ as a descriptive
threshold, only 2/42 L3 pairs and 0/210 L2 pairs crossed it; median $|\Gamma|$
was zero. L2 event openings frequently created deeper activity without
changing class loss, so spike counts alone do not measure useful propagation.
This spike-event audit selects any two distinct units within 50 ms; it does
not enforce a shared downstream receiver and therefore does not test the
topology-conditioned same-receiver route-pair mechanism.

**Theory update.** §146 derives the exact mixed derivative of the independent
logistic event relaxation: $\partial^2\tilde L/(\partial m_i\partial m_j)=
p_i(1-p_i)p_j(1-p_j)\Gamma/\tau^2$. Pair replay work is warranted where a
shared receiver and overlapping arrivals make an interaction plausible; the
four-corner estimate determines whether such pair-specific credit exists.
The next mechanism test should compare topology-matched shared-receiver pairs
against time-matched nonshared controls and log message acceptance, event
time/payload changes, deepest loss, and replay work. No trainable pair update
or SHD accuracy improvement is established.

The four jobs ran serially through `experiments/queue/run_safe.sh` with 3.6 GB
virtual-memory cap, 2.6 GB process-group RSS cap, and 8 GB minimum available
host memory. Peak observed RSS was about 334 MB; `MemAvailable` remained above
11.1 GB. No guard tripped.

**E83 layer-balanced route pairs: deep support without class learning (§143).**
This matched D4 seed-6 follow-up changed only pair selection relative to the
global-pair arm: same initialization, 120 examples, four epochs/120 updates,
one pair shadow per minibatch, and 128 held-out examples. It selected
43/34/15/28 route pairs from layers 1–4 over the run; the global sampler had
sent all 120 pair shadows to layer 1. Layer 4 then retained 100% of held-out
example support in every epoch. Yet final anytime-with-fallback accuracy was
6/128 (4.69%), compared with 8/128 (6.25%) for the no-pair control and 14/128
(10.94%) for the global-pair arm. Paired exact McNemar tests were
$p=0.791$ (balanced vs control) and $p=0.180$ (global vs control). These
single-seed differences are inconclusive.

Support came with an unstable activity regime: the balanced run's mean
per-utterance layer event counts at epoch 4 were [122, 189, 363, 1,647],
and its four-epoch averages were [142, 192, 334, 1,421]. Adjacent activity
amplification ratios from those averages were 1.35, 1.74, and 4.25. Its
training losses stayed between 6,574 and 12,932; final late-prefix NLL was
14,699, and the classifier predicted only two of 20 classes. Therefore
restoring deep support did not produce a useful representation; it exposed a
second failure mode, overactive deep event generation with collapsed class
evidence. A matched intervention associates this with layer-balanced route
updates, but does not isolate the pair sampler from the threshold/reset
dynamics or noisy route utility.

The paired replays give additional separation between action utility and
learning: joint opening lowered its matched loss in 38/120 selected pairs
(31.7%), while $\Gamma<0$ in 30/120 (25.0%). Per-epoch raw counterfactual to
pathwise gradient norm ratios ranged 0.0011–0.0146, and their cosine ranged
from -0.202 to +0.251. Useful counterfactuals exist, but they were neither
common nor a reliably aligned update. The balanced policy samples layers
uniformly by design; absent inverse-propensity weighting it estimates a
layer-equalized objective rather than the full candidate-pool gradient.

**Theory update.** §143 now formalizes the joint operating-point requirement:
track active-example support $c_\ell$, conditional event multiplicity
$\mu_\ell$, and total activity $n_\ell=c_\ell\mu_\ell$ separately. It
derives a cost-constrained route utility that includes the counterfactual
change in per-layer event/message work, and identifies the next experiment:
collect matched class-loss/work deltas before setting any activity budget.
E83 also scans hidden states on a 1 ms grid, so its current SHD training
result is not evidence of sparse asynchronous training cost.

**Frozen validation route/work audit (§143).** With no parameter updates, the
balanced checkpoint was evaluated on the same 128 held-out-speaker examples.
One pair was sampled per eligible layer and minibatch: 115 pairs total,
32/32/19/32 from layers 1–4; eligible pool sizes were 214,920/228/38/273.
Joint opening improved the matched prefix loss in 5/32 (15.6%), 10/32
(31.3%), 11/19 (57.9%), and 16/32 (50.0%) pairs by layer. Median
$L_{11}-L_{00}$ was 0.000/0.236/-0.268/-0.001, while the means were
19.93/74.07/1.36/0.46. Early-layer means are outlier-sensitive; the audit
does not estimate general utility from one collapsed checkpoint.

The measured downstream work deltas give a sharper route hypothesis. A
layer-2 pair added on average 4.24 L4 spikes and 10.76 L4 readout updates per
example; layer 3 added 1.79 and 4.04; layer 4 added 0.23 and 0.45. Layer-1
pairs added 0.78 and 1.83. Layer-2 averages conceal mostly zero late-layer
work changes plus a small number of large cascades. Next, test a matched
late-layer-only pair sampler (last half of the stack) to see whether it
retains useful deep alternatives while avoiding the early-layer cascade.
This uses validation utility to form a training hypothesis; the same
validation results are development evidence, not final test evidence.

Reconstructing the clipped four-outcome derivative from those same paired
losses and route scores ($\sigma=0.25$) gives mean $\partial L/\partial s$
of +0.291, +0.517, -0.013, -0.114 per route in layers 1–4. Gradient descent
would close routes for positive derivatives and open routes for negative
ones; negative-derivative fractions were 14.1%, 35.9%, 60.5%, 50.0%.
Thus equal shadow allocation is not equal useful credit: early proposals
mostly say “close,” while the late-layer signal is more favorable but weak
and noisy. This is one checkpoint's validation-conditioned calculation,
not a general layer ranking.

**Matched late-layer pair run and second-order proposal starvation (§144).**
The D4 seed-6 late-balanced arm changed only pair selection, restricting one
pair shadow per minibatch to L3/L4. It produced 9 deep pair shadows in epoch
1, then none in epochs 2–4. The per-epoch candidate-pair counts by layer were
`[193140, 3, 6, 18]`, `[188476, 0, 0, 0]`, `[180439, 0, 0, 0]`, and
`[180459, 0, 0, 0]`. L4 validation support was not literally zero in every
later epoch, so the sharper explanation is pair co-occupancy failure: no two
closed, near-time routes shared a deep receiver, despite occasional deep
events. Final anytime accuracy was 6/128 (4.69%), versus 8/128 (6.25%) for
the matched control. This is not evidence that late alternatives are harmful;
the treatment had no deep pair proposals for three of four epochs.
Training loss decreased 59.78→3.07, but held-out prefix NLL was 2.985/3.167
in the two time strata and accuracy stayed near the 5% chance rate.

For each minibatch/receiver group `g`, let `n_g` be the number of eligible
closed routes before the code filters adjacent arrivals by time gap and
distinct source event. The exact pair count is bounded by
`sum_g (n_g - 1)_+`. Under a sparse Poisson occupancy model, this bound's
expected value and the probability of at least two alternatives both scale
as `lambda^2/2`, while single-route availability scales as `lambda`. Thus
higher-order counterfactuals disappear faster than first-order credit as
event flux thins. Importance weighting corrects
sampling among existing pairs; it cannot repair a zero candidate set. The
frozen time-window sweep tested 25, 50, 100, 250, 500, and 1,000 ms on the
same 120-example fit subset. Pair counts at 1,000 ms were `[189987, 6, 0, 1]`
for L1–L4, compared with `[181577, 1, 0, 0]` at 25 ms. Even across a full
second there were no L3 pairs, only one L4 pair, and L2 pairs in just 6/30
batches. Near-closed route records themselves were `[191787, 111, 3, 13]` by
layer. This rules out the 25 ms cutoff as the main cause; source-route
co-occupancy is the bottleneck. Widening the window measures candidate count,
not counterfactual usefulness, since widely separated arrivals may not
interact under the receiver kernel. The next design must preserve first-order
shadows, instrument co-occupancy by layer, and use sparse prefix-expansion
replays to create downstream counterfactual events when justified. That
estimator is not implemented yet.

**Refractory-aware spike-boundary audit.** A matched no-pair control and the
late-only checkpoint were evaluated on the same 128 held-out-speaker examples
(32 batches). Each batch/layer supplied its closest spike toggle; local
boundary candidates were then restricted to margin $\le0.25$ and
nonrefractory state. Valid candidate counts by layer were `[22, 21, 8, 0]`
for control and `[21, 19, 2, 1]` for late-only. Separating the fused main loss
from the weighted auxiliary losses, the L1 main-loss toggle helped 12/22
control candidates (mean $+0.0266$) and 16/21 late-only candidates (mean
$-0.0094$, median $-0.0020$); the auxiliary term has the same direction.
Only 13 batches had a valid L1 candidate in both arms; the mean difference
between the two arm-specific selected spike-on utilities was $-0.037$ (SE
$0.035$), and the selected unit/time can differ by checkpoint. This is
suggestive rather than a reliable treatment effect. L2 is not a robust opening signal: 12/19
late-only candidates helped, yet mean main-loss change was $+0.0040$ (median
$-0.00063$), with large variance. L3/L4 have only 2/1 valid late-only
candidates. The earlier all-candidate averages were misleading because they
included forced spikes outside the margin band. A matched deepest-only replay
of these same toggles found exactly zero L1/L2 main-loss changes, although
late-only L1's all-depth mean was $-0.00936$. The L1 intervention added no
downstream hidden spikes and changed its own readout edges by 1.238/example.
Thus the apparent L1 utility is a shallow-head shortcut, not a deep credit
signal. The next test should use a deepest-only primary objective and/or a
work-capped sparse suffix-expansion replay that demonstrates downstream event
changes; neither remedy has yet been trained.

**E83 receiver-bundle route-credit screen (§142).** This matched D4 seed-6
comparison used the same initialization, 120 training examples, four epochs
(120 optimizer updates), split RNG streams, and 128 held-out examples; both
arms retained the one-route-per-layer counterfactual update. The treatment
added one receiver-bundle pair shadow per minibatch, costing three replays.
Terminal accuracy ended at 8/128 (6.25%) for control and 14/128 (10.94%) for
the pair arm. For the actual anytime prediction with terminal fallback, the
paired counts were 8 versus 14 correct, with 4 control-only and 10 pair-only
correct examples (exact McNemar $p=0.180$). This is a small exploratory
direction, not evidence of a reliable accuracy gain.

The main objective issue remains visible: control training loss fell from
1752.8 to 3.01, nearly the uninformative 20-class value $\ln20=2.996$; held-
out late-prefix NLL finished at 3.246. The pair arm ended at train loss 3.23,
late-prefix NLL 3.916, and 3.9% layer-4 support versus 1.6% for control. Both
deep supports collapsed over training. The pair arm temporarily retained
51.6% layer-4 support at epoch 2 versus 11.7% in control, but this did not
persist or improve that epoch's accuracy (3.9% versus 9.4%). A falling scalar
loss therefore did not mean that the classifier learned the task.

The global proposal policy selected 30 pairs per epoch from 197k–206k
eligible adjacent candidates, and all 120 selected pairs came from layer 1.
Only 7/120 joint openings reduced the matched loss and 8/120 had negative
interaction $\Gamma$; the first epoch accounted for most of the positive mean
interaction. The policy paired closed scores in $[-0.5,0)$ by batch/receiver,
kept adjacent source-time pairs within 25 ms, then sampled globally. This was
not a threshold-margin or vector-compatibility selector. The result shows
that generic temporal proximity produces little useful pair credit and that
global candidate counts starve deeper layers. It does not reject cooperative
routes; it identifies the proposal policy as the next variable to isolate.

The follow-up keeps one pair per minibatch and changes only selection:
choose uniformly among eligible layers, then uniformly among that layer's
pairs, while logging layerwise candidate counts and inclusion probabilities.
The route formalism also now distinguishes the hard deterministic edge gate
from its logistic training relaxation and records two additional architectural
limitations: the current receiver query is input-independent, and the same
score controls both gate admission and positive delay. The four-outcome
gradient is exact for its stated independent-gate surrogate before clipping;
the proposal sampler, clipped contrasts, and separate local SGD step are
explicitly identified as bounded learning heuristics. Alternate delay/value
policies and receiver-state-conditioned queries remain untested.

**E83 route-credit theory and compute-matched screen (§§138–141).** The
analysis quantifies the pathwise-gradient penalty from sparse deep support:
if a sample contributes gradient $X$ only when layer $\ell$ is active, then
$G=A X$ has mean $c_\ell\mu_\ell$ and covariance
$c_\ell\Sigma_\ell+c_\ell(1-c_\ell)\mu_\ell\mu_\ell^\top$. For IID minibatches,
the probability of no active example is $(1-c_\ell)^B$; with batch size four
and measured layer-4 coverage 1.56–7.03%, this is 75–94%. Under
conditional-gradient-noise dominance, task-direction SNR scales as
$\sqrt{B c_\ell}$. This is exact under the stated sampling assumptions, but
does not establish that active events carry useful class information.

The compute-matched D4 comparison held architecture, initialization, and 120
optimizer updates fixed, with nested 120-example/four-epoch versus
480-example/one-epoch subsets and the same 128 evaluation examples within
each seed. Seed 6 favored the larger subset (20/128 vs 8/128; paired exact
McNemar $p=0.0227$); seed 7 reversed direction (8/128 vs 20/128;
$p=0.0357$). Layer-4 support remained 1.56–7.03%, and late-prefix NLL was
3.02–6.58, above uniform $\log20=2.996$. The data-diversity effect is not
reproducible in these two runs and did not repair deep support or posterior
quality. The optimizer-induced cascade hypothesis also remains unverified:
global gradient clipping does not bound AdamW's gate-margin movement, and
per-update margin traces have not been collected.

Sections 139–140 separate four objects that had been conflated: the fixed
candidate mask, the event-conditioned graph, the realized route-and-fire
graph, and the graph receiving loss credit. Exact reconstruction of the
seed-6/7 D4 masks gives 90.2%/98.0% first-to-fourth unit-pair reachability;
all 140 input bands reach layer 4. Yet only 1.56–7.03% of examples actually
reach it in the matched arms. Sparse layer-1 skips raise support to 99–100%
without paired recognition improvement. Static disconnection is therefore
not the broad failure; useful events and label credit still fail downstream.

The route calculation makes the sparse-MoE analogy precise: a competing
route's relaxed gradient depends on the loss difference between paired route
outcomes. Current E83 shadows test one near-boundary route at a time, so they
miss a possible cooperative crossing where two individually subthreshold
messages jointly create a useful spike. A two-gate derivation identifies the
needed four replays (00/10/01/11) and the interaction loss. This is a
testable mechanism, not yet an established cause. Section 141 generalizes the
idea to dormant proposals: record why a candidate did not fire, replay a
sparse set through the full downstream state, and credit its local margin
only when the paired loss says the alternative helped. Route closure,
threshold failure, race loss, and refractory blocking need distinct margins;
absence by itself is not a negative label.

Finally, §140 shows why depth can add useful choices but cannot be increased
blindly: candidate paths grow combinatorially while strict-chain sample
support multiplies by conditional survival at every transition. Preserving
half of full input support at depth 8 or 16 requires average per-transition
survival of at least 0.906 or 0.955. These results motivate a fixed-checkpoint,
margin-stratified dormant-event audit before another depth sweep; they are
not evidence of supremacy or of route-pair synergy in the current model.

**E83 all-depth D4 control and strict-chain support invariant (§137).** The
matched four-epoch depth-4 `all_depths` run used 128 train / 32 held-out
speakers, seed 6, and the same event-prefix objective and route shadows as its
queued deepest-only control. Held-out spikes per utterance rounded by layer
were `[14, 6, 5, 28]`, `[16, 2, 1, 0]`, `[9, 1, 0, 0]`, and `[10, 1, 0, 0]`.
Terminal accuracy was 6.25%, 3.125%, 6.25%, and 6.25%; epoch-4 prefix NLL was
`[3.1373, 3.4115]`, above the uniform 20-class NLL of 2.996. On the first
training minibatches of epochs 3 and 4, main-loss gradient norms in layers 3
and 4 were exactly zero. These are first-minibatch gradient measurements,
not claims that every update in those epochs was zero.

The matched deepest-only arm ended at 5/32 (15.6%) versus 2/32 (6.25%) for
`all_depths`. This is a first above-chance-sized endpoint in the matched D4
screen, but it is not strong evidence yet: on the shared examples there were
five cases correct only for deepest-only and two correct only for fusion
(exact two-sided McNemar p=0.453). The deepest-only prefix NLLs were
`[4.0081, 4.7412]`, worse than uniform NLL 2.996, and accuracy moved
9.4% → 0% → 3.1% → 15.6% across its four epochs. The nominal one-sided
binomial tail for 5/32 at 5% chance is 0.020, before accounting for the
multiple arms/epochs inspected. Larger held-out evaluation is needed.

That larger readout comparison is now complete for seed 6, with 128 examples
in both the train and held-out-speaker subsets. The `all_depths` and
`deepest` arms used the same seed, selected examples, and training random
stream; only the readout fusion setting differed. At the fixed fourth-epoch
endpoint, terminal accuracy was 17/128 (13.3%) for `all_depths` and 7/128
(5.5%) for `deepest`. Layer-4 active-example coverage was 61.7% versus 4.7%,
respectively; layer-2 coverage was 100% versus 46.1%. For the actual
thresholded output policy (first crossing, otherwise terminal fallback), the
paired predictions were correct on 20/128 versus 7/128 examples, with 19
all-depth-only and 6 deepest-only correct cases (exact McNemar p=0.0146).
This is a nominal single-seed, two-speaker result, not a speaker-general
claim. The fixed-threshold race emitted on 42/128 all-depth examples, 10
correct (23.8% accuracy among emissions); their mean reported confidence was
63.8%. Deepest-only emitted on 2/128 and neither was correct. All-depth
late-prefix NLL was 17.57, while deepest-only
was 4.66, both above uniform 2.996. The objective improves class decisions
and preserves deep support in this pair, but its accumulated evidence is
poorly calibrated. Removing the layer-4 branch left all-depth terminal
accuracy unchanged at 13.3%, so this result does not establish that the
deepest branch contributes the accuracy gain.
At this endpoint, standalone head accuracies were `[7.0, 10.2, 8.6, 10.2]%`;
removing each head in turn from the fused terminal logits left
`[10.2, 7.0, 10.2, 13.3]%`. The second-layer head has the largest measured
leave-one-out effect, while layer 4 has none. These are readout-head
ablations on one trained checkpoint, not retrained depth ablations.

The seed-7 matched replication narrows the claim. At the fixed endpoint,
`all_depths` reached 9/128 terminal accuracy versus 7/128 for `deepest`; its
race-plus-fallback outputs were correct on 9 versus 6 examples, with 9
all-depth-only and 6 deepest-only cases (exact McNemar p=0.607). Layer-4
coverage was 22.7% versus 16.4%, a smaller difference than seed 6. The
all-depth race emitted 8 answers with mean confidence 63.1%, none correct;
late-prefix NLL was 5.74 versus 4.09 for deepest-only. The seed-6 paired
accuracy advantage therefore did not replicate, while the calibration failure
did. Seed-7 leave-one-head-out fused accuracies were
`[2.3, 7.8, 10.9, 7.8]%` (omitting layers 1–4) versus 7.0% with all heads;
removing layer 4 slightly improved the endpoint. The layer contribution
pattern varies by seed, with no stable deepest-layer gain.

The seed-7 `deepest` replication has also finished: terminal accuracy was
7/128 (5.5%), layer-4 coverage 16.4%, late-prefix NLL 4.09, and the fixed
0.6 race emitted four answers with none correct. Thus deepest-only remains
near chance on a second training/evaluation subset, with the same two held-out
speakers. Its matched seed-7 `all_depths` arm completed; it did not reproduce
the seed-6 paired advantage.

The seed-6 deepest-only skip ablation has finished. A sparse layer-1 event
skip raised layer-4 coverage from 4.7% to 99.2% and broke strict support
nesting as intended (29 measured nesting violations); mean deep candidate
scores rose only 1.8% (46,246 to 47,096). Terminal accuracy moved from 7/128
to 9/128, while late-prefix NLL improved from 4.66 to 3.71, still above
uniform 2.996. At the fixed 0.6 threshold the skip emitted five times and
none were correct. The paired race-plus-fallback comparison had 6 strict-only
and 8 skip-only correct cases (exact McNemar p=0.791). The skip fixes the
intermediate support bottleneck at small sparse fan-out cost, but it does not
yet produce useful class evidence or a reliable stopping signal. The 1.8%
figure counts candidate event–receiver pairs; the simulator still performs
288,008 vector-state updates per utterance on a 1 ms grid, so this is not an
end-to-end energy measurement.

The seed-7 skip replication reached 100% layer-4 coverage but only 5/128
terminal accuracy, late-prefix NLL 48.80, and zero correct answers among
seven fixed-threshold emissions. Total candidate-score work was 22.3% above
its strict-chain control; paired race-plus-fallback accuracy was 5 versus 6
correct (6 strict-only, 5 skip-only; exact McNemar p=1.0). The skip therefore
restores support in both seeds without a paired classification gain, while
its event activity and evidence scale vary sharply by seed. Support survival,
event-rate control, and class credit are separate requirements.

The analysis yields an exact structural invariant for this model: if a strict
chain layer receives no events, its zero state and positive threshold produce
no output events. Per-utterance active-example support is therefore nested
with depth; `all_depths` changes the loss paths but cannot make a silent hidden
layer receive input. Mean event count and active-example support are distinct:
event multiplicity can cascade on a shrinking active subset. This clarifies
how the D8 count cascade can coexist with deep support loss. The count-mark
arm kept layer-4 coverage at 56–94% across its four epochs and produced zero
support-nesting violations, yet accuracy fell to 0/32 after epoch 1 and
epoch-4 prefix NLL was `[8.4786, 27.28]`. Thus count input changed deep
dynamics but did not create class evidence in this seed. Added held-out
coverage and spikes-per-active-utterance fields, plus a support-nesting
violation counter. The seed-7 deepest-only and both sparse layer-1-skip
controls are complete. The skip restored layer-4 support in each seed but had
no paired accuracy gain; its seed-7 arm also had unstable late-prefix loss.
The skip masks use a separate topology RNG
so the matched adjacent-layer masks remain unchanged.
E83 now defaults to `--rng_protocol split`, which separates evaluation
selection, training subset/order, augmentation, prefix sampling, and
route-shadow sampling. Existing results used the prior shared-stream behavior;
pass `--rng_protocol legacy_shared` to reproduce it. The split implementation
is not yet validated with a paired run across evaluation limits. Use split
streams before comparing different evaluation sizes. Do not add a hidden-spike
boundary update until its paired loss deltas show useful class credit.

## 2026-09-28

**E77 1M depth-8 benchmark resource screen.** The parameter-matched 8-layer
Transformer control completed 4,882 updates on 1M text8 training characters
(4,999,168 token positions), with 1.251M parameters. Its best validation BPC
was 2.322 at update 4,881; the frozen 1M-character test BPC was 2.352. It ran
for 1,617 seconds at about 3,507 token positions/second and stayed near 1.2 GB
RSS. The matching E77 depth-8, width-128 job at 1M characters, five passes,
batch 4 and 256-character windows was stopped by `run_safe.sh` before training
when its process group reached 3.834 GB, above the 3.5 GB cap. `MemAvailable`
remained above 10 GB; the host was not at risk. **Learning:** the guard caught
a batch/activation configuration that was too large before it could become a
host-wide failure. The 1M E77 job will not be retried. A matched 100k-character
screen now uses batch 2 and 128-character windows for both E77 and Transformer;
depth 16 is deferred until that runtime is measured.

**E77 exact post-reset activity matching and depth-8 gradient reach (§§133–134).**
The raw voltage-quantile estimate failed at the default E77 width: on a matched
depth-4, 10k-character, seed-77 run it yielded selected-checkpoint test rates
$[0,0.007,0.009,0.004]$ spikes/character. Only $[1,9,12,6]$ of 13 validation
checkpoints had nonzero gradients by layer. We changed calibration to save
pre-reset voltage traces and replay the exact refractory/reset recurrence,
then search threshold against the realized event count without rerunning the
whole network. At initialization, rates matched the 0.1 target to
$[0.096,0.082,0.094,0.100]$; all four layers had gradients at all 13
checkpoints. Test rates rose to $[0.204,0.190,0.456,0.802]$ as the
representation changed. Quantile-only and exact-rate runs scored 3.765 and
3.824 BPC, respectively, on only 1k test characters; this is not a quality
comparison. It is evidence that the initialization statistic, not the
language-model loss, caused the earlier silent stack.

The exact-rate procedure also carried a width-8 model to depth 8. With 4,096
training characters, 16 updates, and one seed, all eight layers had nonzero
gradients at each of 16 validation points. Initial rates were
$[0.102,0.086,0.109,0.102,0.086,0.102,0.098,0.102]$ and selected-checkpoint
test rates were $[0.115,0.125,0.217,0.075,0.081,0.138,0.124,0.121]$.
The 4-layer control under the same tiny setup had nonzero gradients on
$[16,16,16,12]$ validation points. Test BPC was 4.319 at depth 8 and 4.254 at
depth 4, each on 512 characters; neither supports a quality or depth
comparison. **New learning:** the mechanism that reopened the depth-4 path
also reaches depth 8 while keeping event activity sparse. This is a concrete
trainability result, not scaling or supremacy evidence.

A fresh depth-4 route-shadow pass sampled 64 near-boundary route openings per
layer. The minibatch-trajectory means and cluster SEs for
$L_{open}-L_{closed}$ were $[+0.000360\pm0.000218,-0.001115\pm0.000325,
+0.000220\pm0.000278,-0.000435\pm0.000290]$. Layer 2's openings tended to
help, but adjacent updates share a moving model, so these are not independent
replicates. Across the run, the counterfactual/pathwise norm ratio averaged
0.77% and cosine 0.0040; the optimizer correction remains disabled. This
supports continued layer/condition-stratified measurement, not a larger gain.

The new local Bayesian derivation (§134) formalizes early evidence trust: for
a locally linear conditional, posterior mean gain scales with posterior
covariance, input-state novelty, and observation noise. Repeated observations
shrink gain only in the state directions actually observed. Activity alone
calibrates event scale, not task relevance; output credit is still required.
Representation drift requires discounted precision or reset. No
posterior-driven synaptic gain or online threshold homeostasis is implemented
yet. All jobs ran through the memory-guarded runner; one first attempt was
stopped at its RSS cap before the calibration search was optimized.

**E77 depth-4 event bootstrap and route-credit diagnostic (§133).** A matched
16-update, seed-77 text8 micro-pilot exposed a trainability blocker: with the
fixed threshold $\theta=1$, test activity was [0.018, 0, 0, 0] events/character
across four layers, and only layer 1 had any nonzero gradient (5/16 updates).
The deeper stack was effectively absent from learning. We added a
label-free, depth-ordered threshold calibration from empirical pre-reset
voltage quantiles, targeting 0.1 spikes/character/layer. The matched calibrated
run set thresholds [0.664, 0.229, 0.429, 0.283], produced [0.065, 0.052,
0.044, 0.035] test events/character, and had nonzero per-layer gradients on
15/16, 16/16, 14/16, and 16/16 updates. This is direct evidence that the
previous initialization silenced the deep event stack and that a measured
activity scale can restore gradient paths. The tiny run (4,096 training
characters, 512 validation/test characters, width 8, 16 updates, one seed)
does not establish language-model quality or depth scaling: test BPC was
4.341 fixed versus 4.284 calibrated, too little data for a performance claim.

The calibrated model also received 64 near-boundary route shadows per layer
(256 total). Minibatch-clustered layer means of $L_{open}-L_{closed}$ and
standard errors were: layer 1 −0.000034 ± 0.000237; layer 2 −0.000440 ±
0.000236; layer 3 −0.000045 ± 0.000103; layer 4 −0.000092 ± 0.000477. Every
approximate 95% interval includes zero. Only 3.1–15.6% of sampled openings
helped by layer; the global counterfactual/pathwise norm ratio averaged
0.61%, with cosine −0.0019. A weak prior should therefore remain broad, and
the optimizer still applies no counterfactual update. This is a posterior
uncertainty result, not a reason to turn up the gain. E77's token-level query/
key module still materializes a dense causal $L\times L$ score matrix; this
experiment measures deep hidden-route trainability, not sparse retrieval cost.
All runs used the memory-guarded queue.

**E83 causal prefix supervision and route-gradient statistics (§§130–132).** The per-utterance normalized-time sampler was future-dependent: it used each utterance's last event to choose a prefix. E83 now samples a fixed, shared 0–1000 ms physical-time window, stratified into exogenous query strata. Under log loss, the population optimum at each query is the causal posterior $P(Y\mid\mathcal F_t)$; choosing query time using future duration would instead reweight labels by duration. The code also fixed a PyTorch autograd failure in grouping equal event times (`unique_consecutive` had no derivative), replacing it with detached group IDs and differentiable group-time means. This makes the prefix-time path differentiable; it does not remove E83's 1 ms hidden-state scan.

The exact route boundary term from §§19/57 is now measured by toggling one near-boundary route, rerunning the whole downstream stack, and comparing factual and shadow prefix loss at identical sampled query times. A first Horvitz–Thompson implementation multiplied each sampled route by candidate-count/sample-count. At depth 4, 128 train / 32 held-out, two epochs and seed 6, this drew 128 shadows per epoch from about 480k eligible routes. Train loss rose 1,974 → 17,044, prefix NLL and layer-4 activity exploded, so that estimator is too noisy at this budget. The fixed-population formula in §132 explains the scale: total-estimator covariance has the factor $N^2(1-m/N)/m$.

A bounded normalized local rule now averages sampled boundary terms within each layer, clips $L_{on}-L_{off}$ to ±5, clips the global correction-gradient norm to 1, and applies a separate SGD step of 0.001. The matched seed-6 pathwise-only run stayed near chance: terminal accuracy 6.25% (2/32), with held-out prefix NLL by stratum [2.995, 3.202] in epoch 2. The local-counterfactual run also ended at 6.25%, but its epoch-2 NLL was [2.996, 19.735], so this pilot did not improve the posterior and showed severe late-window overconfidence. Race coverage was 3.12% with no correct emitted answer. It did not reproduce the HT loss explosion, but neither did it establish stable learning.

The route statistics explain why a global gain is unjustified. In epoch 1, 121 shadows had mean signed open-minus-closed loss +0.172, SD 1.176, and only 14.9% showed that opening helped. Per-layer mean deltas were [+0.051, +0.344, +0.262, −0.010]; layer 4 was less unfavorable (36% helpful), but its mean remained small relative to its SD. Epoch 2's overall mean was +0.0063 (SD 0.0644; helpful fraction 11.6%). The counterfactual/pathwise gradient cosine was 0.017 then 0.0038. Raw gradient norm ratios were 4.47% then 1.64%, but the applied clipped update norms were only 4.5e−4 and 1.5e−4 per batch. These results call for uncertainty-aware, layer-specific analysis: a weak prior at initialization should not suppress route evidence, but the sampled evidence itself remains noisy and does not support a larger update. §132 derives a hierarchical posterior/shrinkage diagnostic and separates it from trust-region step-size control.

A separate 80/40 depth-2 smoke reinforces the scale issue: mean absolute shadow delta 2.6e−4, SD 1.28e−3, 10% helpful openings, gradient norm ratio 1.17e−5, and cosine −0.002. The bounded update was nearly negligible there. Both pilots remain at chance-scale accuracy; use them to diagnose credit magnitude, not to claim trainability. The guarded depth-4 run peaked below 0.6 GB RSS with host available memory above 11.7 GB. Inspection also found that payload-agreement accounting included fallback cases in an emitted-only denominator; the E83 counter is fixed, and the saved result summary was corrected from its per-sample output vectors.

**Deep sparse-stack stability and full-sequence attention bound (§§113–114; derivation plus E114 diagnostic).** Section 113 composes local approximation and Jacobian error through residual depth; under residual scale $1/L$ and bounded per-layer constants, these bounds do not grow exponentially, but they do not show that the task gradient is useful. Section 114 lifts fixed-support softmax truncation to the full sequence Jacobian, with operator norm at most $\sqrt{RC}$ and a shared-key column term containing $F_j$. E114 checked 144 fixed-support synthetic cases using central finite differences at lengths 3, 5, and 8, with diffuse/reused keys, several retained supports, and three score temperatures. No absolute violation exceeded $10^{-8}$. Among the 90 cases with bounds at least $10^{-8}$, the largest Jacobian and forward-error ratios were 0.897 and 0.687; the largest positive absolute excesses were $1.32\times10^{-10}$ and $7.4\times10^{-16}$. Ratios for the all-keys-retained cases are omitted because the exact bound is zero and finite-difference residue makes the ratio meaningless. This supports the formulas on small synthetic inputs; it is not an architecture-scale result. Uniform state-region bounds, sparse Jacobian Lipschitz constants, parameter-VJP error, learned support changes, and task-gradient alignment remain open.

**xLSTM transfer and affine-scan lemma (analysis, not an experiment).** Corrected the earlier overstatement: a real-mode,
fixed-schedule time-vector state is a restricted normalized exponential accumulator, not a full xLSTM layer. It lacks
sLSTM's learned gate/memory mixing and mLSTM's query-addressed matrix memory; E77's Hopfield key/value updates are an
explicit separate operation. xLSTM's 7B model and its 672-run, 80M–7B parameter scaling study are precedents for deep
non-Transformer scale methodology, not transferred evidence for E77 ([7B model](https://arxiv.org/abs/2503.13427),
[scaling study](https://arxiv.org/abs/2510.02228)). New derivation: for fixed event bins, each E74 state step is the
diagonal affine map $(A_k,x_k):z\mapsto A_kz+x_k$. Composition
$(A_j,x_j)\circ(A_i,x_i)=(A_jA_i,A_jx_i+x_j)$ is associative, so a prefix scan can compute all state bins with linear
work and logarithmic parallel depth; the count state follows the same construction. This preserves the recurrence in
exact arithmetic but keeps dense prefix-state storage and does not parallelize topology changes, thresholds, or resets.
**Next:** compare sequential and scan values/gradients on fixed schedules, then time a small serialized pilot. No speedup
or energy saving has been measured.

**Hopfield event-message architecture and depth bound (not yet measured on language).** E77 now applies a causal key/value
associative update over emitted event payloads at each event layer, then forwards the updated payloads. Queries, keys,
and values use separate trainable maps; a gated identity path preserves the original event, with update scale 1/depth.
Scores become arrival advantages through $\\delta=\\tau(s_{max}-s)$, whose exponential decay reproduces the softmax
weights. This connects the Hopfield address and payload operations to the project's delay/vector message primitive.
For a fixed memory, the query Jacobian is a key/value cross-covariance and is bounded by
\\beta D_KD_V/4, independent of the number of memories. If the complete fixed-topology correction Jacobian is bounded
by $K$, 1/depth residual scaling gives singular values of the depth Jacobian in $[e^{-2K},e^K]$. E77 records key/value
diameter upper bounds and the implied query-Jacobian bound as direct diagnostics. Sequence-level sensitivity also
depends on maximum key fan-out $H=\max_j\sum_i p_{ij}$, since one stored payload may influence many later queries;
§107(h) derives a conservative operator bound including $H$ and the query/key/value, gate, and output gains. This is a
conditional fixed-topology state-path guarantee, not proof that event routes are discovered or that hard event births are differentiable. Exact
all-past-event retrieval is the trainability reference; optional top-k limits aggregation but still scores every pair.
E77 logs event pair count, retrieval entropy, score spread, update size, key fan-out, the sequence Jacobian bound and
residual-scaled layer sum, and query/key/value/output/gate gradient norms. The revised E77 architecture is queued, not
yet run. A depth 2/4/8/16 Transformer control now matches E77's 256-character windows, sampled training batches, 5M token
exposure, batch size and optimizer-update count. Widths 104/104/112/136 match E77's parameter count within 4% at all
four depths.
Separate ablations now turn off token retrieval and event-Hopfield updates one at a time.

**Hopfield stability/precision tradeoff (derived).** With values equal to keys, softmax retrieval is $\nabla\Phi$ for
the convex log-partition $\Phi(q)=\beta^{-1}\log\sum_j e^{\beta q^\top k_j}$, and its Hessian is the PSD key
covariance. The update is contractive if $\beta D_K^2/4<1$. For two choices, score credit is $\beta p(1-p)$:
it peaks at a tie and vanishes as retrieval becomes certain. More generally, a score margin $m$ gives non-winner mass
at most $(N-1)e^{-\beta m}$, so raising precision narrows the region that receives learning credit. Separate key and
value codes enable arbitrary hetero-association but remove the general energy-descent guarantee. These derivations
motivate a measured temperature/near-miss sweep before claiming that sharper, sparser retrieval will train as well.

**Adaptive retrieval and depth, theory-to-code update.** E77 has event-state layers, a recurrent state route, event-level Hopfield payload updates, and a token query/key/value retrieval interface. Event-state layers remain non-attention layers; the associative update is a separate operation at emitted events. E77 accepts configurable depth and adds sparse raw-event skips to deeper blocks. This prototype is not yet trained or validated. Sparse describes its synaptic fan-out and event-stream wiring, not a measured sparse execution kernel: `TVLayer` allocates dense time-by-batch-by-unit states, while event and token retrieval score all eligible pairs. The attention window or event top-k limits value aggregation, not candidate search. **Design correction:** sparse aggregation does not imply sparse search or low energy; indexed candidate search and an event-driven simulator remain open engineering problems.

**E79 data/capacity scaling and copy-window ablation.** The race mixture's 256-character-copy setting scores 1.808 / 1.613 / 1.504 frozen bpc (1.782 / 1.593 / 1.483 with online weight adaptation) at 1M / 10M / 90M training characters. Expert count rises with data (K=5/6/7), so this is evidence that the native expert mixture improves as both data and capacity grow, not an isolated data-scaling law. Removing the copy-window cap changes frozen test bpc from 1.808 to 1.779 at 1M, 1.613 to 1.612 at 10M, and 1.504 to 1.512 at 90M. Thus unbounded copy helps modestly at 1M but has no consistent advantage at larger scales; the 0.008 difference at 90M is one seed and has no uncertainty estimate. **Learned:** keep a 256-character copy window in scale comparisons, and treat long-range associative retrieval as its own capacity path. E79 is ahead of the completed 1M one-layer LSTM (2.179 bpc), but adds count/copy memory and has no matched compute/energy accounting. E61 learned query/key race retrieval solved a synthetic recall task in 1–4k examples and extrapolated to 4× context. E77 has no completed LM result yet; these findings motivate its deeper adaptive model but do not establish its superiority.

**Attention sparsification and counterfactual credit (§112; mathematical result, not yet an experiment).** Conditioning dense softmax on a fixed candidate set gives an exact query-gradient residual: omitted-group covariance plus a between-group covariance between key direction and value advantage. Its norm is at most β ε (3/2 − ε) D_K D_A. Mass recall therefore controls absolute gradient error but not relative error: in the two-key example, 99.9% retained mass still leaves the sparse query gradient zero while the omitted key carries the dense gradient. A new H-smooth extension closes the nonlinear-loss gap: the full-loss query-VJP error is bounded by β ε D_K[(3/2 − ε)D_A^C + H D_V²/4], with corresponding key- and value-VJP bounds. The H term is the upstream-gradient change induced by the retrieval error, not just the local support mismatch. This remains an absolute, fixed-support bound and cannot promise alignment when the true gradient is small. Adding a missing key changes the output to (1 − r)y + rv; the finite loss change differs from r·gᵀ(v − y) by at most (H/2)r²‖v − y‖². This connects the already-derived attention score credit (§105/§107) to the existing sparse-expert counterfactual route rule (§19/§57). Substituting this estimate into the existing route-boundary gradient bounds its error by (H D_V² / 2) Σ ρσ(m) r² ‖∇θ(b_u − b_c)‖, giving a principled way to allocate exact shadow runs to near-miss keys. Next compare exact dense and truncated autodiff VJPs against the new bounds while varying support mass and downstream curvature, then compare bound-based shadow allocation with fixed top-m at equal shadow work. No such attention-index experiment has run yet.

**E64b gradient language-model baselines.** The validation-selected 1M, 20-pass LSTM completed at 2.1385 valid / 2.1794 test bpc (338,395 parameters); the 2-layer width-256 Transformer completed at 2.3403 / 2.3667 (1,658,907 parameters). Both best checkpoints are at their final validation point. At 10M, the 2-layer width-512 LSTM completed six passes at 1.7448 validation / 1.7993 test bpc (1,199,323 parameters). The 4-layer width-256 Transformer then completed all 4,882 updates in four passes (batch 32; 3,238,427 parameters): best validation BPC 1.864707 at step 4,880 and held-out test BPC 1.908253. It is 0.1199 higher on validation and 0.1090 higher on test than the LSTM. This is a same-data comparison, not a capacity- or training-budget-matched comparison; each run uses one seed, and the LSTM's best validation point is its last checkpoint.

**90M E79 queue state.** The native race-mixture 90M result already exists at `experiments/results/e79/race_mixer_D90000000_K7_e77none.json`; the queue runner lost its successful-completion marker. The duplicate E79 90M queue line is now commented out so a resumed chain will not repeat it. This was not a 90M Transformer run. The Transformer currently advancing is the 10M, four-layer E64b baseline.

**E79 at 10M versus the completed LSTM.** On the same 10M-character training prefix and 1M-character test segment, the frozen E79 race mixture scores 1.6130 test bpc versus 1.7993 for the LSTM: a 0.1863 bpc lead. E79 uses six native experts plus its copy memory; the LSTM has 1,199,323 parameters. This is a single-seed, same-data result, not a compute-, parameter-, or training-budget-matched architecture comparison. The completed 10M four-layer Transformer scores 1.9083 test bpc on that segment; E79 is 0.2953 lower, but uses six experts plus copy memory and is not matched for parameters, training, or compute.

**E64b LSTM versus the E79 race mixture at 1M and 10M text8 characters.** The validation-selected 256-hidden-unit LSTM used 20 passes and early
stopping on 200k validation characters: 2.138 validation / 2.179 test bits per character, 338,395 parameters. E79's frozen race
mixture on the same 1M-character training and test segments scores 1.808 with its copy expert limited to the Transformer's 256-character
context (1.782 when weights also adapt online). The frozen mixture is 0.371 bpc below the LSTM and 0.559 below the two-layer Transformer;
the online-adapted score is 0.397 below the LSTM. This is the clearest current real-language performance lead for the native experts.
It is not yet a compute/energy-matched architecture comparison: its count tables and copy index are extra state. It also does not test
the deep E77 time-vector language model. At 10M, the frozen race mixture scores 1.613 test bpc against 1.799 for the completed
two-layer width-512 LSTM on the same test segment. That 0.186 bpc lead is also single-seed and lacks matched compute or parameter
accounting; the four-layer Transformer trace has no endpoint or test result (last logged at 90%; the serialized lock remains held, though its process is not visible in the current process listing). **Next:** prioritize the deep
market and speech event models, then return to additional language baselines only if they answer a specific unresolved comparison.
Report memory, scoring work, and adaptation policy alongside loss before making a scaling claim.

**E83 — sequence objective audit and depth-4 screening controls.** SHD supplies one class label per spoken utterance. The original E83 implementation trained one loss per utterance, but averaged softmax(V) across the entire shared simulation grid. Empty prefixes and the artificial delay tail contributed uniform class guesses; a short utterance's output could therefore change when batched with a longer one. The completed depth-2 jobs are retained as debugging data but excluded from trainability claims.

E83 now compares a causal confidence race against two established utterance readouts. The race stays silent until the first temperature-scaled class probability crosses a threshold; only that first class is emitted. Its train loss scores the probability that the correct class wins the event race, includes survival (no output yet) before a winner, and discounts late correct outputs. It does not train against the final class at every prefix. Per-utterance endpoint masks prevent batch padding from changing the objective. Cramer et al. used max-over-time readout potentials for their SHD SNN experiments and reported better results than last-step loss; Spyx uses cross-entropy on integrated readout potentials ([Cramer et al.](https://kip.uni-heidelberg.de/Veroeffentlichungen/download.php/6616/temp/4143-3.pdf), [Spyx SHD tutorial](https://spyx.readthedocs.io/en/latest/examples/surrogate_gradient/SurrogateGradientTutorial/)).

The first race prototype failed: at 80 train / 40 held-out examples for one epoch, it emitted for every item and reached 2.5% emitted accuracy. A revised shared confidence gate and temperature-scaled class race reached 92.5% coverage but only 10.8% accuracy among emitted answers, with 2.5% max-over-time accuracy. These tiny one-epoch runs are smoke diagnostics, not model comparisons. A guarded, matched depth-4 screening run compared two-epoch objectives on 512 training and 128 held-out-speaker examples. Integral pooling ended at 4.69% max-over-time accuracy; terminal fallback reached 5.47%, race coverage was 17.97%, and emitted accuracy 4.35%. Max pooling ended at 4.69% max-over-time accuracy; at the default race threshold its fallback accuracy was 3.91%. The `anytime` race-plus-fallback objective reached 6.25% max-over-time accuracy (8/128) and 5.47% emitted accuracy at 100% coverage; all thresholds from 0.3 through 0.9 still emitted every item, with peak confidence 1.0. Layer 4 firing increased from 932 to 1,024 spikes/utterance. Under a 5% chance model, 8/128 has a 0.31 upper-tail probability and a 3.2–11.8% Wilson interval, so this is not reliable evidence above chance; the decisive signal is false confidence, not early useful recognition. The matched stable-cause race-only control also finished near chance: 3.12% max accuracy, 97.66% coverage, 4.8% emitted accuracy, with mean peak confidence 0.981. A readout-only shadow probe at seed-2 initialization (not trained weights) forced 32 near-gate routes on four held-out-speaker utterances. 31/32 changed max-pooled CE by exactly zero; one reduced it by 0.045. The estimated counterfactual-gradient norm was 2.2% of pathwise norm, cosine 0.012. This small mechanism probe is consistent with a max-pooling winner-gap dead zone (§129), but says nothing conclusive about hidden route births or trained checkpoints. All jobs stayed under the one-process memory-guarded runner. E83 stores per-sample vector outputs and final-layer firing rasters for separate visualization.

**Anytime stream-to-class theory (§§122–§128).** With a true posterior at every prefix, emitting at the first crossing of $1-\epsilon$ bounds the error among emitted answers by $\epsilon$; ordinary full-recording calibration does not establish this, so first-crossing calibration must be measured. For a marked point process, posterior evidence consists of both event log-likelihood jumps and survival evidence from intervals with no event. If each message changes only $r$ class logits, indexed max and log-sum-exp trees can maintain the exact confidence threshold in $O(r\log C)$ updates, with $O(C)$ paid only when materializing a full class-probability payload. A portable reference head is added at `experiments/sparse_anytime_readout.py`; it is not yet connected to E83 or benchmarked. Section 128 separates posterior estimation from the stopping policy: sampled-prefix log loss is proper for $P(Y\mid E_{\le t})$ even with only one utterance label, while the race objective alone does not identify calibrated prefix probabilities. This applies to SHD and event-camera clips, which share one-label-per-stream supervision but differ in event marks, rates, and nuisance variation.

**E83 gradient-graph and readout audit (§§127, 129).** The fixed-support controls all share a hard routing implementation: `TVLayer` removes nonpositive message scores using `keep = r.detach() > 0`, and computes spike identities in `no_grad`. Thus closed routes receive exactly zero task gradient; open routes can receive pathwise credit through their delay and payload; firing-time refinement differentiates a spike only after it has fired. Intermediate readouts improve credit for realized events but do not create the missing route-birth/firing boundary term. The route masks are also fixed random/tonotopic buffers, so topology itself cannot be recruited. A subsequent readout-only shadow probe forced 32 near-gate final routes at seed-2 initialization, on four held-out-speaker utterances. 31 interventions changed terminal max-pooled CE by exactly zero; the one nonzero intervention reduced it by 0.045. This is consistent with the max-pooling winner-gap dead zone derived in §129: a route can alter its local trace while remaining below the current per-class temporal maximum, leaving terminal scores and loss unchanged. The measured boundary-gradient norm was 2.2% of pathwise norm, cosine 0.012. Because this probe used untrained weights and no hidden spike insertions, it is only a mechanistic clue. **Next discriminating work:** save a trained checkpoint, shadow near-threshold hidden spikes through the remaining layers, and compare max, integral, and smooth-max posterior heads under the same event support and sampled-prefix proper log loss. Report exact loss differences, winner gaps, and gradient alignment before adding counterfactual credit. Use a fixed noise band and shadow budget. No more objective sweep answers this mechanism question.

**E83 depth-four failure mechanism, spike-boundary audit, and depth-eight rate profile (§137).** The seed-6 128/32 pathwise event-prefix run reports held-out spikes per utterance rounded to `[4, 2, 0, 0]`; layers 3–4 average below 0.5 events per utterance. On epoch 2's first minibatch, the main-loss gradient norms were exactly zero for every hidden layer, while the auxiliary gradient norms for layers 3–4 were also zero. This confirms that the hard support cuts label credit on silent deep-path batches. The normalized route-shadow run had more held-out activity `[21, 15, 4, 14]` but still 6.25% terminal accuracy and late-prefix NLL 19.735: routing credit alone did not solve recognition. A paired hidden-spike audit on 128 held-out utterances found near-threshold margin candidates (within ±0.25) averaging 349/batch, 67/batch, 7.9/batch, and 6.6/batch across the four layers; layer 4 had none in 24/32 batches. Spike-on improved the loss in only 16/32, 17/32, 14/32, and 14/32 interventions, with near-zero mean effects. This confirms scarce deep support but does not justify a single-spike update. A separate input audit found that E83 drops the merged event-count mark: it is returned by preprocessing, but the network uses only band identity. Multiple raw spikes occur in 55.6% of fitting groups and 43.5% of held-out-speaker groups, so this is a concrete information bottleneck whose class value remains to be tested. The depth-eight all-depths pilot oscillated between early extinction and an activity cascade: layer counts moved from `[16, 9, 3, 6, 27, 51, 141, 250]` at epoch 1 to `[24, 3, 1, 1, 4, 6, 32, 65]` at epoch 2, surged to `[59, 35, 97, 286, 1137, 1929, 4125, 4888]` at epoch 3, and fell to `[27, 3, 2, 2, 3, 11, 47, 58]` at epoch 4. Late-prefix NLL swung 39,814.7 → 1,692.1 → 19,787,863.2 → 22.9; terminal accuracy remained 3.1–12.5%. We stopped after epoch 4 for instability. The next controlled representation test preserves the count as a sparse vector mark before adding spike credit or threshold calibration.

**E83 all-depth readout audit (§145): the L1 signal is a shallow bypass.** On the same seed-6 checkpoints and held-out examples, we replayed the valid nonrefractory, in-band spike toggles against both fused all-depth and deepest-only main loss. The late-only L1 all-depth delta averaged −0.00936 (16/21 helpful), while all 21 matched deepest-only L1 deltas were exactly zero; every valid L1/L2 deepest-only delta was zero in both arms. Work deltas show why: L1 spike-on added 0.1429 L1 spikes/example but no downstream hidden spikes; it changed 1.2381 L1 sparse-readout edges/example. Thus the fused loss directly rewards the L1 classifier head even when no event cascade reaches deeper layers. The observed local utility is not evidence of deep compositional credit. Deepest-only L3/L4 counts were just 2/1 and cannot support estimates. The next depth test must train the primary deepest-only objective or explicitly create a plausible sparse prefix event, replay its suffix, and measure downstream changes under a declared work cap. This was a frozen validation replay, not an accuracy gain.

**E74/E75 — time-vector speech pilots do not yet learn well, and the memory ceiling was too tight for their default batches.**
E74 on 2k training / 500 held-out utterances at batch 32 completed 6 epochs in 31 minutes, but held-out speaker accuracy peaked at
0.146 and ended at 0.120. E82's partial readout sweep peaked at 0.184 after 240 updates with a non-spiking state readout and layer
normalization; the queued normalized spike readout stayed near chance. E75 at batch 16 reached only 0.044 after 2 epochs before a
38 MB CPU allocation failed; a batch-32 restart failed on a 65 MB allocation. Host available memory remained above 10 GB, so these
failures are consistent with the 3.5 GB per-process virtual-address limit, not host exhaustion. **Learned:** the current speech
representation/readout and optimization are not close to a useful baseline; exact equivariance alone is not enough. Do not spend on
full SHD runs until smaller pilots show a learning signal. **Change:** future E74/E75 pilots and ablations use batch 4, results now
include batch size in their filenames, and the safe runner also monitors whole-job RSS. These lower-batch pilots are queued; they have
not yet established whether the memory issue or the learning issue is resolved.

## 2026-09-26

**E24 — grokking on (a + b) mod 31, half the pairs for training (first runs).** Dense MLP, full-batch AdamW
(wd 1.0): train 1.0 by step 1k, test 0.00 until ~3k, then 0.87 at 20k while the weight norm falls: the regime
exists on this CPU (92 s). Race without sleep: train 0.99–1.0 within ~50 epochs, test 0.002 (below chance 0.032)
through 1,500 epochs. *Learned:* the race memorizes as a partial-key lookup: an unseen pair activates the codes
of training pairs sharing an operand, whose sums are all different, so it is reliably wrong. Contrary to §52.2,
plasticity never goes quiet (57M events by epoch 1,300, still rising) and the weight norm grows; random-feedback
hidden credit keeps the codes churning. Sleep sweep running.

**E23 — class-incremental split-MNIST (5 blocks), the readout was the problem.** Single-head forgetting is
saturated for both learners (race 0.968, MLP 0.979 at 1k frames per task; MLP 0.984 at 4k): each block drives
the old tasks to exactly 0.00, so §50's predictions cannot be tested on it. Added a task-aware readout (only the
task's classes may win; the race is re-run with the other outputs' thresholds out of reach) and plasticity per
block. Task-aware forgetting: race 0.167, MLP 0.030 (1k per task); MLP 0.029 at 4k. *Learned:* §50's prediction
is reversed so far. The race output has no bias (the bias column is zeroed, thresholds fixed), so the prior shift
of each block is carried by old classes' feature weights (THEORY §51). *Changes:* `--price` (learned output
thresholds) added; ablations queued.

**Engineering.** The container was restarted without a GPU (host driver is nouveau; not needed: the code is
numpy-only). The venv lives in the session scratchpad and had to be rebuilt; the interrupted E23 job reran.

**E17 — a continually learning race on the BTCUSDT trade stream (preregistered; 21 confirmatory days,
prequential, day-block 95% intervals).** Direction accuracy on moves ≥ 1 bp: race 0.593 (88% of episodes
decided, 6.7 s mean decision), frozen race 0.595, online logistic regression 0.583 (decides at 10 s), momentum
0.591. Preregistered rules: competitive (met, 33% earlier) and nominally better than logistic regression (+1.0
[0.6, 1.5]). A fairness check made after seeing the results overturns "better": logistic regression at the
race's coverage is as accurate (+0.1 [−0.3, +0.5]); momentum ties the race. Continual − frozen −0.2 [−0.6,
+0.1]: no benefit from test-time learning. The hold race (learning when to trade) traded 1.4% of episodes at
chance accuracy, tying logistic regression's most confident trades. Every learner loses money after a 2 bp
cost. *Learned:* the race can match baselines on a real asynchronous stream while deciding a third earlier;
nothing here is an edge. Plain 10 s momentum is ~59% right on ≥ 1 bp moves, more than the preregistered null
expected (look-ahead checked).

**E20 — exact spike-time gradients on the race architecture (debug: 10k images, 2 epochs).** Gradients verified
by finite differences (error 1e-4 to 1e-3 at step 0.01). Adam needs steps relative to each layer's weight scale
(absolute steps diverged). Depth 1: 0.922 (local rule at this size about 0.85). Depth 3 collapsed to chance (0.10)
exactly as THEORY §27/§30/§36 predicted, and the theory's remedies rescue it: fast homeostasis 0.64, Ward centering
of timing credit 0.40, **both 0.785** (slow homeostasis 0.01 fails, as §36's timescale argument says). *Learned:*
the first theory-derived fix that changes a result materially; deep exact training of race networks is possible
with activity owned by the thresholds.

**E22 — Spiking Heidelberg Digits, first pass (validation, seed 0, 10 epochs): poor.** Race depth 1: counterfactual
0.353, fired-only 0.314, **frozen hidden 0.385**; depth 2 0.356; dense MLP 0.559 / 0.589 (depth 1 / 2). Synaptic events
per utterance 87–139k (MLP ~288k multiply-accumulates). *Learned:* far behind the MLP, and hidden learning hurts.
Suspected cause: hasty decisions. The first output crossing commits on the first few hundred ms of a one-second
utterance, where early spikes are not stronger evidence, and the rule learns from those premature decisions.
Pilots with later output decisions (theta_out 3, 10) and speed–accuracy curves queued.

**E22 decision-time diagnosis (5 epochs).** Output threshold 1 / 3 / 10: 0.301 / 0.361 / 0.273; frozen hidden (threshold
3) 0.380. Speed–accuracy curves rise with later decisions but saturate near 0.36; the relative (MSPRT) rule does
not beat the absolute race. *Learned:* hasty decisions are a small part of the SHD gap; the local learning rule is
the main problem (a frozen random hidden layer still beats trained ones).

**E20/E21 diagnostics (MNIST, depth 2).** Exact gradients with no cancellation at the MLP's 5-epoch budget:
0.9675 vs MLP 0.976, so the race architecture itself is close to dense. Fermi–Dirac (soft k-winner) training,
evaluated hard: 0.942 vs 0.930 trained hard (k = 3, 2 epochs), about 60% of the cancellation cost recovered.
Both are training-time diagnostics under the 2026-09-26 direction decision (ROADMAP).

**E20 at full length (depth 2, 2 epochs, decay + clipping; the constant-rate runs diverged).** Winners per group
3 / 5 / 10: 0.930 / 0.926 / 0.951; local rule (E14, 3 epochs) 0.952; backprop MLP (5 epochs) 0.976. *Learned:*
race cancellation costs about 2 points under exact gradients, and a further ~2.5-point gap remains without it, at
an unequal budget (2 vs 5 epochs). Matched-budget runs (5 epochs, depth 2 and 4) queued.

**E17 follow-up, continual learning as tracking (THEORY §43; exploratory).** Step size η 0.001 / 0.003 /
0.01 / 0.03: 0.583 / 0.593 / 0.593 / 0.578; change-gated 0.578 (mean gate 0.15); frozen 0.595. No setting
beats freezing. *Learned:* the test was confounded, because η and the gate also applied while learning from
scratch in the pilot week; a corrected run (pilot at full rate, tracking variants afterwards) is queued. What
stands: three weeks of BTCUSDT show no drift the race can exploit, and a large step (0.03) chases noise.

**Credit conservation at full length (2 seeds, width 400, 3 epochs).** 0.960 / 0.952 / 0.941 at depths 1 / 2 /
3 vs 0.949 / 0.942 / 0.924 without: +1.0 to +1.7 at every depth, both seeds. The theory's prediction holds; the
debug size (+8–10) overstated it. Non-negative weights at depth 3: 0.932 vs 0.937 (1 seed).

**Theory §27–41 (THEORY.md) and first measurements.** Exact: a Ward identity (timing credit sums to the
deadline's credit), topical-map non-expansiveness with a certified jitter radius, and a commitment-cost
identity (near-miss credit is the gradient of σ × surprisal). Scaling predictions queued as M31–M43. Measured
so far: the input's entropy exponent α ≈ 0.95 (M39), refuting a predicted input-redundancy pyramid; smoke tests
show a common-mode share around 0.03, against a predicted value near 1 (untrained; decisive run queued).

**Engineering.** The host hung a third time: the queue ran one job while an ad-hoc debug run ran beside it.
Now every computation goes through the queue, the watchdog is at 6 GB and checks every second, and `dev.sh`
gives new containers hard memory, CPU and GPU settings.

## 2026-09-25

**Pivotal credit (THEORY §26; debug, 10k images, 2 epochs, output conservation on).** Computing
the first-order jump through the real output weights, gated by arrival before the decision, for
the top hidden layer: depth 1 0.921 (0.925 with group conservation) vs 0.895 random feedback;
with random feedback deeper ("pivot at the top"): depth 2 0.896 vs 0.880, depth 3 0.876 vs 0.870.
Failures on the way, each informative: (a) arrival gating created dead-late units (fixed by a
time-residue weight); (b) pivotal credit through real hidden-to-hidden weights collapses at
depth 3 (0.10); (c) zero-sum credit within hidden groups breaks even the working rule (0.87 →
0.10), refuting my explanation of (b); (d) the exact race Jacobian (evidence shares w/A, a
conservative backward flow) does not fix (b) either (depth 3 0.096; depth 2 0.865, worse than
pivot at the top). The deep path's instability is open. Leads; full-length runs queued.

**M23 at full length (2 seeds, width 400, 3 epochs) — the debug lead did not hold.** Depth 3:
shadow neuron window 0.4: 0.916 / 0.916; window 0.15: 0.920 / 0.924; sampled binary 0.925 /
0.921; **hard binary window 0.928 / 0.931**; residue weighting (E14) 0.924. Depth 1 shadow 0.951
(vs 0.949), depth 2 shadow 0.930 (vs 0.942). *Learned:* the shadow neuron's 5-point debug lead
reversed with full training (and it costs twice the forward compute); weighting-free, sort-free
selection by a hard window is as good or slightly better than continuous weighting at depth 3.
Debug leads from 1-epoch runs are unreliable for rankings; confirm at full length before
presenting them. (Bug found here: result filenames omitted the window, so the w0.4 depth-3 files
were overwritten; values recovered from logs; filenames now include every varied setting.)

**E14 depths 4–5 (seed 0).** Counterfactual vs fired-only: depth 4 0.910 vs 0.891 (+1.9),
depth 5 0.897 vs 0.872 (+2.5). With depths 1–3 (+0.2, +1.2, +1.5, 2 seeds) the gap grows
monotonically with depth.

**M30 — learnt capacities (debug, depth 3).** Target firing rates proportional to each node's
information about the label: 0.733 (linear) and 0.736 (log-ratio) vs 0.746 with equal targets.
*Learned:* no gain from this simple version; the capacity structure holds but this choice of
capacities is not the missing piece.

**M29 — homeostasis as Sinkhorn (debug, depth 3).** Log-ratio (Sinkhorn dual) threshold updates
balance hidden usage better than the linear rule (0.95–0.98 vs 0.89–0.93) but accuracy drops as
balance is forced (0.741 / 0.612 / 0.562 vs 0.746). *Learned:* thresholds really act as prices on
a capacity constraint (THEORY §25), but equal capacities are the wrong target; capacities should
be learnt.

**E14 complete (2 seeds, width 400, 3 epochs, validation).** Counterfactual vs fired-only vs
frozen: depth 1 0.949 / 0.947 / 0.873 (gap +0.2; seeds −0.1, +0.4); depth 2 0.942 / 0.930 / 0.708
(gap **+1.2**, both seeds +1.2); depth 3 0.924 / 0.909 / 0.565 (gap **+1.5**; seeds +1.2, +1.7).
*Learned:* confirmed at 2 seeds: counterfactual credit's advantage appears from depth 2 and holds
at depth 3, modest (~1.2–1.5 points) but consistent; a frozen stack collapses with depth, so deep
race networks depend on hidden credit. Depths 4–5 and credit conservation on top are queued.

**M28 — simplex (multiplicative) learning: negative.** Non-negative, credit-conserving nets,
debug size: exponentiated-gradient updates of each node's evidence mix (urgency additive) reach
0.06–0.31 vs 0.85 additive (depth 1) and 0.08–0.15 vs 0.75 (depth 3). *Learned:* the simplex
view is an exact description (neurons are linear in their evidence mix) but the wrong learning
geometry: multiplicative steps cannot recruit absent evidence (zero or tiny weights), which is
most of what learning has to do. THEORY §24.

**E14 full length, seed 0 (width 400, 3 epochs, validation).** Counterfactual vs fired-only:
depth 1 0.945 vs 0.946 (−0.1), depth 2 0.940 vs 0.933 (+0.6), depth 3 0.926 vs 0.909 (+1.7);
frozen hidden 0.875 (d1), 0.712 (d2). *Learned:* the gap grows steadily with depth as predicted
(§16), but at full training it is ~2 points at depth 3, not the ~19 of the 1-epoch debug run:
longer training lets fired-only credit partly catch up. Seed 1 and depths 4–5 pending; credit
conservation (+8–10 in debug) is queued on top of this.

**Formal consequences, checked (THEORY §22).** (1) Time-shift equivariance is exact: uniform
input delays of 0.05 and 0.2 leave all 300 decisions and hidden firing sets unchanged, times
shifted to within 2·10⁻⁷. (2) Weight norm is urgency: ×1.5 fires 0.084 earlier. (3) **Credit
conservation**, predicted by the dequantized-min derivative: normalising competitor credit
raises depth-3 accuracy 0.649 → 0.746 (residue weighting) and 0.698 → 0.782 (shadow neuron)
(debug size, one seed): the largest single gain at depth. (4) Monotone (non-negative-weight)
networks lose almost nothing (0.848 vs 0.852 at depth 1; 0.745 vs 0.746 at depth 3). Full runs queued.

**M23 (debug size) — sort-free near-miss selection and the two-channel neuron (user's ideas).**
Depth 3, width 200, 5k images, 1 epoch, one seed. Continuous residue weighting 0.649; hard Δ
window (binary) 0.584; sampled binary eligibility (Bernoulli(exp(−Δ/σ))) 0.548; fired-only 0.525;
**two-channel shadow neuron, window 0.4: 0.698** (window 0.05/0.15: 0.62). *Learned (tentatively):*
letting losers keep integrating and emit shadow spikes into a separate learning channel, with
closeness selected by time rather than by sorting or weighting, is at least as good as the
residue machinery at depth. It is the straight-through pattern with a real counterfactual backward
path, and the substrate-natural form of the idea (THEORY §20). Full runs queued.

**M20 — beams as asynchronous shadow events.** One discrete-event pass computes the factual
history plus one branch per hidden-group collapse ("the group's last winner does not fire"), with
shadow continuation of the group's members and per-branch output deltas. Over 50 samples (small
network, 6 branches per sample): factual winner agreement 100%, **branch winner agreement 100%**
against the batch computation of the same swaps. Cost per sample: ~506 output shadow
re-predictions and ~81 hidden shadow updates (vs 611 scheduled events in total), mostly from every
branch re-predicting every output on each output input. *Learned:* branches can run exactly and
asynchronously in the same event queue (§14.6 holds); the overhead is in re-prediction and has
obvious savings (only outputs whose ranking can change). Now a regression test.

**E16 — counterfactual routing gradient in an ordinary MoE (numpy; 8 linear experts; top-1).**
A first version showed a dramatic gain for the boundary term. That was an artifact: my gate
baseline did not implement the exact Switch-style gradient. With exact gradients, on MNIST every
router collapses to one expert (a single linear classifier already reaches ~0.89, so routing is
irrelevant there). On a synthetic task that *needs* routing (8 clusters, each with its own linear
labelling), 3 seeds, 20 epochs: gate + load balancing 0.843 ± 0.015; gate 0.816 ± 0.010;
boundary with 1 shadow expert 0.757 ± 0.029 (σ 0.3: 0.769; σ 3: 0.639); dense top-2 0.716;
boundary with 2 shadows 0.661. The boundary term helped early (5 epochs: 0.695 vs 0.652) and hurt
later. *Learned:* the counterfactual routing gradient is exact for the *current* experts but
**myopic**: an expert that is not routed an input is not learning from it, so its current loss
understates what it could become. Mostly the alternative is worse, so the term reinforces the
existing routing and reduces exploration (balance 0.88 vs 0.97). Routing is a bandit whose arms
improve when pulled (THEORY §15, §19); a useful routing term must value an alternative's
*learning potential*, not just its present loss. *Next:* an optimistic or lookahead variant
(the alternative's loss after one hypothetical update), before any MoE claim.
**Follow-up (lookahead, 10 seeds):** valuing the alternative after one hypothetical step on the
input removes the harm: 0.864 ± 0.018 (95% CI) vs gate + load balancing 0.866 ± 0.012 (paired
difference −0.002 ± 0.022, 6/10 wins), gate 0.855 ± 0.017, dense top-2 0.738 ± 0.021. At 3 seeds
it had looked like a win (0.865 vs 0.843); 10 seeds show a tie, at twice the expert compute.
*Learned:* the myopia diagnosis is right (lookahead fixes it), but on this task the counterfactual
routing gradient does not beat the standard remedy. No MoE claim.


**E15 (debug size) — credit percolation with local layer-by-layer feedback.** Depth 3, width
200, 5k images, 1 epoch. Reach per layer (input side → output side): fan-in 16 counterfactual
0.68/0.51/0.51, fired-only 0.25/0.24/0.21 (above threshold, no decay, as predicted); fan-in 2
fired-only 0.11/0.13/0.18 (decays toward the input, as predicted), counterfactual
0.20/0.16/0.23 (noisy). No clean transition yet at this size; the full sweep is queued.
**Surprise:** sparse hidden-to-hidden connectivity raises deep accuracy (≈0.79 at fan-in 2
vs 0.68 at 16), matching the E9 fan-in result: sparsity seems to help race networks in its
own right, plausibly because dense layers are decided by a few very early spikes.
**Link (user):** the same branching condition governs router training in sparse MoE (top-1
routers get no gradient toward unselected experts); residue-based near-miss credit could train
routers without running unselected experts (THEORY §17.5).

**E9 sparse fan-in (debug size).** 3k images, 1 epoch, validation: fan-in 32 gives 8.1k
synaptic events per image vs 113.9k dense (14× fewer), with *higher* accuracy (0.817 vs
0.792). For scale: the MLP that matched round-3 accuracy (32 hidden units) needs ~25k
multiply-accumulates. If fan-in 32 holds ~0.96 at ~8k events, the inference-energy verdict
flips in the race's favour. *Next:* E9 pilots (3 epochs; fan-in 16/32/64, with and without
growth) are queued; if they hold, a full 10-epoch test-set run for the energy table.

**DRTP control (debug size).** Hidden credit from a random projection of the label alone
(direct random target projection), the pure form of the template mechanism: one hidden layer
0.54 vs 0.59 for counterfactual credit (close, consistent with templates); **depth 3: 0.20 vs
0.63**. *Learned (tentatively):* in shallow nets our hidden credit behaves much like label
templates, but at depth the error-gated, near-miss structure is what keeps race networks
trainable. That is the distinctive part of the idea. Full runs (M22, E14 with crl_drtp) queued.

**M3 against a smooth race-margin objective, and along training.** Output rule +0.82 against
the margin objective. Hidden credit: random feedback −0.10, fired-only −0.17, true-weight
feedback +0.28, sign feedback +0.31 (seed 0; reliability 0.79). Along training (error
objective): random feedback ≈ 0 at 0 and 1 pretraining epochs; sign feedback rises to
+0.45–0.50 after one epoch. *Learned:* the "smoother objective" explanation of the puzzle
fails: random-feedback hidden credit follows neither objective's gradient, yet adds 16 points.
*New hypothesis (template mechanism):* with random feedback each hidden node is pushed to fire
for the classes its feedback row favours, so the hidden layer forms label-conditioned templates
that the output learns to read; label-driven feature formation, not gradient descent. It would
also explain why gradient-aligned (true-weight) feedback did worse in rounds 1–2: its targets
move as the output weights change. First diagnostic (debug size): output–feedback alignment
0.19 vs 0.04 frozen; hidden class selectivity 0.29 vs 0.24 (chance 0.11). M22 (3 seeds,
including true-weight and sign feedback under round-3 settings) queued.

**E14 (debug size) — counterfactual credit with depth.** User's hypothesis: the
counterfactual part matters only beyond one hidden layer. 5k images, 1 epoch, width 200,
one seed: gap (counterfactual − fired-only) +2.2 at depth 1, +1.9 at depth 2, **+19.3 at
depth 3** (0.630 vs 0.437). Share of hidden nodes receiving credit: counterfactual 29% / 44–62%
/ 54–78% by depth, fired-only 17% / 24% / 27–29%. *Learned (tentatively):* fired-only credit
collapses with depth because it cannot reach nodes whose influence runs through events that did
not happen; counterfactual credit keeps deep race networks trainable. Accuracy still falls with
depth at this training length. Full runs (2 seeds, 3 epochs, width 400, plus frozen) queued next.

**E13a (debug size) — reward only.** On 2k images, 2 epochs: near-miss guess 0.30,
reward-modulated winner-only 0.19, pool policy gradient 0.15, supervised reference 0.59.
*Learned (tentatively):* spreading credit over the close calls when wrong roughly doubles
what the standard spiking reinforcement rule reaches. Pool policy gradient may need another
temperature (σ variants queued). 3 seeds at full size queued.

**Queued to resolve open questions:** M3 against a smooth race-margin objective (does the
hidden credit follow a smoother objective than 0/1 error?); M3 along training; E9 sparse
fan-in (16/32/64, with and without growth) as the energy viability gate.

**M18 / M18b / M21 — 3 seeds (small network, 10k images, 3 epochs).**

| rule | accuracy | weights changed |
|---|---|---|
| counterfactual credit | 0.821 ± 0.006 | 12.1M |
| fired-only credit | 0.818 ± 0.005 | 6.8M |
| repair + homeostasis + thin margins | 0.798 ± 0.014 | 0.40M |
| repair | 0.767 ± 0.021 | 0.27M |
| repair + homeostasis | 0.756 ± 0.020 | 0.28M |
| output-only repair | 0.661 ± 0.005 | 0.40M |
| frozen hidden | 0.659 ± 0.010 | 1.4M |
| label-free competitive hidden (M21) | 0.467 ± 0.022 | 13.5M |

*Learned:* history repair comes within ~2.3 points of the gradient-like rules while changing
31× fewer weights; widening thin margins gave ~3 points, homeostasis nothing. Label-free
competitive learning makes the hidden layer much worse than leaving it random: labels are
essential for the hidden layer.

**Puzzle.** Label-driven hidden credit is worth +16 points (fired-only 0.818 vs frozen 0.659),
yet its alignment with the gradient of expected 0/1 error is ≈ 0 even at initialisation (M3 at
0 pretraining epochs: +0.03 random feedback, +0.00 fired-only; estimate reliability 0.77). So
its usefulness is not captured by alignment with that gradient. Candidate explanation: the
0/1-error gradient is dominated by the few samples on a decision boundary, while the rules
improve a smoother objective that pays off later. M19 (smooth loss) and alignment against a
smooth objective should tell.

**M21 (debug size) — can the hidden layer learn without labels?** A label-free competitive
rule (winners move toward their input, rows normalised, homeostasis) with the same output
learning reached 0.29 (η 0.02) and 0.20 (η 0.1) on 2k images, *below* a frozen random hidden
layer (0.33); fired-only credit, which uses labels, reached 0.56. *Learned:* this **contradicts**
the M3 reading below. Label information in the hidden update matters a lot, even though its
alignment with the true gradient, measured at one late snapshot, is weak. Likely reconciliation:
weak local alignment that is consistent over training (M3 measures one point on a pretrained
network). *Next:* M21 at full size (queued); measure alignment along training, not only after it.

**M18 — history repair, full size (small network, 10k images, 3 epochs, seed 0).**
Repair 0.749 vs counterfactual credit 0.818 and fired-only 0.823, while changing 46× fewer
weights (0.27M vs 12.6M). Hidden repairs add ~9 points over output-only repair (0.655).
*Learned:* single-event repair is a real hidden-layer learning signal and extremely sparse, but
not yet competitive on accuracy. The baselines also learn on thin margins and run homeostasis;
repair did neither. *Next:* M18b adds both (queued), then multi-step delegation.

**M13 — learning by half-space projection (single layer, validation).** 7 settings end at
0.899–0.905 after 5 epochs vs 0.909 for the near-miss rule. Needs a minimum correction
deadline (c ≥ 0.2–0.5), because decisions made at the first instant leave no charge to act
on, and a capped step (PA-I); without them it diverges (0.25). *Learned:* a valid
reformulation with no learning rate, not an improvement. The half-space property is also known
(RELATED_WORK.md). *Next:* keep it as the geometric core of repair, not as a rule of its own.

**M19 (debug size only).** After fixing a capped-time bug, the exact gradient of a
cross-entropy over output firing times barely aligns with the gradient of expected *error*
(−0.03 vs +0.61 for the E6 rule), at a size where the estimate is unreliable. *Hypothesis:* the
cross-entropy pushes confident samples too, while error only moves near boundaries. Full size
queued.

**M3 — does each rule point downhill? (3 seeds, 200 coordinates, estimate reliability
0.74–0.80 hidden, 0.95–0.99 output).** Output rule: +0.36 to +0.73. Hidden rules: random
feedback ≈ +0.05, fired-only ≈ −0.01, true-weight feedback ≈ +0.21, sign feedback ≈ +0.17.
Only 13–25% of hidden weights have any nonzero true gradient, rising with timing noise.
*Learned:* at this snapshot, hidden credit carries little gradient information, which fits
counterfactual and fired-only credit tying. (An earlier reading, that the hidden layer's gain
comes from its own competition and homeostasis, is contradicted by M21 above.)
The zero-gradient majority is the blind spot that motivates the tree-of-histories view (§14).

**E9 P1 — cascade.** Only 0.07% of work happens after the output decides. *Learned:* the cost
sits before the decision; only sparse fan-in and routing can reduce it.

**E6 round-3 controls.** Single layer 0.921, frozen hidden 0.895, counterfactual 0.959,
fired-only 0.961, hidden 2000 0.960. *Learned:* depth adds ~4 points under round-3 settings;
the new synapse model gave the single layer ~2.5; width and credit type add nothing.

**Energy re-check.** Round-3 hidden networks: ~108k synaptic events per image, ~3.6× the
inference energy of an equally accurate 32-unit MLP; training 1.2–1.5× cheaper than unbatched
dense training. Only the single racing layer wins at inference (~2.4×).

**Engineering.** Parallel runs hung the host twice; logs could not say why. Everything now runs
one job at a time with a watchdog. Hidden batch assumptions surfaced when moving to one frame
per update (homeostasis inside `teach`; a frame with no hidden spike). A waiter loop that
matched its own command line never started the queue; found by checking, not assuming.
