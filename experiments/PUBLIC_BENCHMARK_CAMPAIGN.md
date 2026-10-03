# Public benchmark campaign: fixed suite, completed evidence and full resources

Requested by the user on 3 October 2026. Goal: reproducible wins on external
benchmarks, with asynchronous and synchronous inputs and sparse native state
updates. No new benchmark win has yet been established by this campaign.

## Coordination

The main agent's [SOTA_TARGETS.md](SOTA_TARGETS.md) leads with NeuroBench
Mackey–Glass and primate reaching, official loaders and its curie DEV queue.
This AWS suite complements that work; do not duplicate its queue or tune on
its reporting test. The NeuroBench leaderboard lists Mackey–Glass LSTM
sMAPE13.37 and ESN14.79.
[Official leaderboard](https://github.com/NeuroBench/neurobench/blob/main/leaderboard.rst).

## Priorities and concrete public targets

| Priority | Benchmark | Plausible edge to test | Public reference / initial gate |
|---|---|---|---|
| 1 | ECG200, official UCR split (100 TRAIN / 100 TEST) | Small-data temporal computation and compact recurrent memory; cheap full fits permit systematic diagnosis | Author implementation: FCN88.9%, ResNet87.4%, MLP91.6%, encoder92.3% (10-run means). First aim93/100 after DEV-only selection and full TRAIN refit; historical-baseline win, not current SOTA claim. |
| 1 | JapaneseVowels, UEA (270 / 370, 12 channels) | Short variable-length synchronous series without forced padding; shared vector operations with private temporal state | Author MTS FCN99.3%, ResNet99.2%, MLP97.6%. First reproduce variant/splits, then match99% at smaller full-resource boundary; tiny accuracy headroom means resource frontier is the sensible target. |
| 1 | PenDigits, UEA (7494 / 3498, two channels, eight steps) | Compact addressed-state classifier with enough examples for replication and fast native fits | Strong matched control required; no verified numerical published target assigned yet. Data are spatially resampled pen trajectories, not raw event timestamps. |
| 2 | SHD, official20-class test | Raw asynchronous timing, silence, persistent memory; avoid dense time-grid work | EventSSM95.9%; dataset-maintainer references include stronger models. Existing speaker-held-out DEV scores are not official-test results. |
| 2 | DVS128 Gesture, official subject split | Asynchronous event-only inference and sparse state updates at competitive gesture quality | EventSSM official code v0.2 reports99.2%, with a tokenization correction. Original96.5% is historical, not the current strongest verified target. Existing coarse-packet DEV is a different protocol. |
| Deferred | PhysioNet2012 irregular records; LRA | Missingness/physical-time support; longer sparse-capacity tests | Promising mechanism matches, but new preprocessing/selection protocols and expensive long fits. LRA already has strong state-space baselines; beating its original Transformer alone is not frontier success. |

Sources checked3October2026:
[archive ECG200](https://www.timeseriesclassification.com/description.php?Dataset=ECG200),
[JapaneseVowels](https://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels),
[PenDigits](https://www.timeseriesclassification.com/description.php?Dataset=PenDigits),
[author neural controls](https://github.com/hfawaz/dl-4-tsc),
[SHD maintainers](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/),
[EventSSM implementation and corrected results](https://github.com/Efficient-Scalable-Machine-Learning/event-ssm),
[PhysioNet protocol](https://physionet.org/content/challenge-2012/1.0.0/),
[LRA official suite](https://github.com/google-research/long-range-arena).

Architecture suitability is a hypothesis. Existing gesture/generalization gaps
make event-frontier accuracy a demanding target; no feasibility guarantee or
promise that these tasks will be beaten. The suite is fixed before any new
TEST score; report unsuccessful datasets as well as successful ones.

## Implemented data and native harness

`public_benchmarks/downloads.json` records official archive URLs/ZIP SHA256.
The three archives are downloaded. `data_manifest.json` records expected
published dimensions/counts, immutable TRAIN/TEST byte hashes and deterministic
stratified TRAIN-only80/20 FIT/DEV indices (splitseed20261004). Data parsing and
split contracts passed with standard-library code. TEST files were byte-hashed;
no TEST parsing/model score has occurred.

`public_benchmarks/run.py` uses the existing AddressedEventHeads native core,
compiled episode kernels, private persistent state, separate keys/values,
temporal races and linear local-expectation categorical message credit. It
retains all episode BPTT and an explicit terminal query. This is not full causal
counterfactual return replay. Dense input channels arrive synchronously at each
ordinal step; sparse native writes happen within the model. Actual physical
asynchronous datasets are the second track, not a reinterpretation of these
ordinal series as physical events.

FIT-only per-channel normalization, ordinal timestamps normalized by FIT maximum
length and a terminal-query flag are explicit. JapaneseVowels variable-length
TS is used without padding; the source author MTS variant must be checked before
comparing published scores. No silent filtering of examples or test-tail changes.

Screening never opens TEST. Checkpoints preserve model, actual Adam, epoch,
NumPy shuffle RNG, historical best DEV weights and counted windows. Fixed seed6
screens three configurations(p16/D2/pool2, p32/D4/pool2, p32/D4/pool4), lr.003,
40epochs, batch32. All candidates see identical fixed FIT/DEV. Numerical port
contracts and bounded throughput pilots precede these fits. Select minimum DEV
NLL, tie-break on counted fitting work then parameters, and refit full official
TRAIN for the selected epoch count with seeds6/7/8. Preserve all prior negative
screens. Access TEST only at final fixed fits, save every prediction/identity,
report all seeds, and never use test outcomes to choose another configuration.
A future revision after final test needs a new independent confirmation policy.

`baseline.py` adds a matched1NN resampled Euclidean diagnostic. It is not a
strong modern frontier or evidence for our architecture. Before an investor-facing
win, reproduce a named strong relevant baseline or verify the exact published
split/variant/scorer. Tiny TEST sets require intervals and paired predictions,
not choosing the luckiest seed. Beating a historical named baseline is labelled
as such; no broad SOTA claim follows.

## Resource and admission rules

Native fitting work traces a full eager optimizer window each epoch including
loss/backward/clip/Adam and extrapolates per presented target. Variable lengths
and padding make this an estimate; record the sampled work distribution. Include
screening, refits and failure/verification work in campaign totals, with DEV/test,
packing and compilation separately. Same-column whole-fit GFLOPs/per-target
MFLOPs for every comparable model. Trained inference tracing is required before
an inference advantage; storage is not liveRSS/DRAM and FLOPs are not joules.

All contracts, pilots, controls and fits receive unique one-job queues through
run_safe.sh. AWS maxthree one-thread jobs, inherited host reservation/per-slot
locks, RSS watchdog, at least8GiB available. Current90M depth8/width64 and private
streaming jobs keep exact settings and recovery. Campaign source freezing and
guarded scheduling must precede numerical execution; no fourth job or free-lock
container shortcut. Prepared queues do not count as executed results.

Source-exact language recovery loads archived original driver and compiled
module bytes through frozen_language.py after upstream extensions changed
shared files. Source aliases in results record the actual executed bytes;
upstream changes are retained. Prepared queue parsing was corrected to exclude
comments before any execution.

## Claim cards and phase decisions

Each final benchmark card must identify the dataset variant, immutable data
hashes, official split/metric, fixed configuration/epoch selection, every seed,
per-example predictions, paired comparator and uncertainty. Report seed spread
on the same examples separately from a test-sampling interval; repeated seeds
do not multiply TEST sample size. ECG200 has only100 TEST cases: a one-case
lead is descriptive and does not establish a statistically reliable difference.
JapaneseVowels' small remaining accuracy margin means a measured comparable-
quality deployment-resource win is more meaningful than an isolated accuracy
rounding difference. PenDigits' matched DEV1NN already reaches99.47%, so it
is a demanding control, not a deliberately weak target.

A public point-score comparison requires a completed official test under the
verified reference protocol. An efficiency comparison additionally requires
trained-state forward equivalence, identical scoring/quality boundary, all
candidate discovery and preprocessing, and compatible MAC/FLOP conventions.
Historical reference means and our three seeds are not a paired statistical
comparison. A leaderboard submission is prepared only from completed certified
evidence; this campaign does not post submissions or contact maintainers.

Initial40epoch screens remain fixed. Weak outcomes are retained and diagnosed
from TRAIN/DEV. They do not justify retuning on the same official TEST after
final refits. Future architecture hypotheses require a revised confirmation
policy or independent target; no automatic never-ending sweep on reporting
test labels. All incomplete/failed admissions have their queues/logs preserved.

Reproduction: python3 experiments/public_benchmarks/fetch.py reconstructs the
exact hashed archive bytes; committed data_manifest.json fixes splits. Numerical
queues live under queue/aws_public_campaign_20261004T000200Z/manifest.json, run
only through the guarded scheduler. status.py inventories completed evidence
without model execution. The current coordinator publishes each completed
result/checkpoint/guard log directly to main, independently of partial language
checkpoint publishers.
