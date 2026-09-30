# SHD: accuracy and energy reference points

Updated 2026-09-30. These are verified primary-source reference points, not an
exhaustive claim about the highest result anywhere in the literature.

## What must be compared

Standard SHD has 20 classes, 8,156 training and 2,264 test utterances, and 700
input channels. Two speakers occur only in the test set. Our current deep
carrier development runs use speakers 3/6 of the training file as held-out
speakers. Their scores must remain separate from official test accuracy.
[Dataset authors and leaderboard](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).

| Reference | Reported SHD accuracy | Resource evidence and scope |
|---|---:|---|
| Zhang, Wang and Shen 2026, multiscale residual encoding | 96.44% | Publisher abstract; full architecture/training/selection protocol not yet inspected |
| Cramer et al., original dataset paper | 85.7% LSTM | Corrects the earlier report summary of approximately 70% |
| EventSSM | 95.9% | Official implementation; asynchronous event input |
| S7 | 96.3% | Paper reports 0.5M parameters; event input |
| Sun et al. 2025, attention for delay SNNs | 96.26 ± 0.08% | Dataset-maintainer leaderboard; no joules comparison here |
| Chen et al. 2025, delay SNN on FPGA | 93.4% deployed | 8-bit weights; processor 282 mW, complete SoC 1.71 W, approximately 104 samples/s |

Sources: [Cramer et al., Section III](https://www.kip.uni-heidelberg.de/Veroeffentlichungen/download.php/6616/temp/4143-3.pdf),
[Zhang et al., publisher abstract](https://www.sciencedirect.com/science/article/abs/pii/S0893608026003345),
[EventSSM code and results](https://github.com/Efficient-Scalable-Machine-Learning/event-ssm),
[S7 Table 1](https://arxiv.org/html/2410.03464v1#S4.T1),
[dataset leaderboard](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/),
[Chen et al., Section IV](https://arxiv.org/html/2511.01158v1#S4).

For the FPGA reference, power divided by throughput gives approximately
**2.71 mJ per utterance for the processor**, or **16.44 mJ for the entire SoC**.
These are derived values from the authors' reported powers/throughput. They are
not interchangeable measurement boundaries. Their 93.4% accuracy also differs
from the strongest accuracy references. This is a concrete engineering target,
not evidence that our CPU model currently beats it.

## What the completed local experiment establishes

**Current accuracy gate:** E143's inherited D8 parent plus parallel D6 temporal
encoder reaches 408/512 (79.6875%) on our fixed private speaker-held-out sample.
E152 retains 406/512 (79.296875%) after deployment consolidation to one six-block
encoder. E159's corrected twelve-block continuation reaches 400/512 (78.125%)
after one matched fitting pass. The strongest historical eight-layer parent
alone is 370/512 (72.265625%); architectural controls do not replace the new gate.
Published parity requires a comparable result on the 2,264 official test
utterances, not a numerical comparison between different held-out partitions.

All head projection, paired-view whitening and finite optimizer calibration
use fitting IDs only. The 657 additional same-speaker utterances audited in
E147/E154 are disjoint from current fitting and development, but reused across
investigations. They are not an untouched test. Report model inheritance,
extra head-fitting views, exact actual optimizer rates and completed source
hashes; E155's scheduler-overwritten rates must not be presented as calibrated.

### What the reference learning structure includes

EventSSM uses learned raw-event embeddings, six state-space blocks, nonlinear
output gates and residual/normalization paths. Its temporal transition depends
on actual event intervals. The paper reports 64/128 state sizes and several
event augmentations. [Primary paper, methods and experimental setup](https://arxiv.org/html/2404.18508v2).

S7 adds input-dependent transitions and reports 96.3% with 0.5M parameters.
It includes event pooling and asynchronous integration; processing raw event
identities need not require every deep layer to process every source event.
[Primary paper, sections 3.3–4.2](https://arxiv.org/html/2410.03464v1).

Our existing core has 32-component carriers, three normalized exponential
memory scales per receiver and fixed alternating frequency partitions. Its
hard winner/surrogate credit is not an exact inclusion of the above smooth
temporal learners. Theory §§215–219 defines both an information-preserving
source intervention and an explicit learned temporal-mode inclusion target.
E139's new branch learns fine source identities/times before the original
packet coalescing. It preserves the best checkpoint at zero, but is not claimed
to preserve every raw-event ordering or implement the complete reference model.
Its completed one-pass score is 369/512 (72.07%), below the 370/512 best.
E140 adds trainable signed rotations over elapsed time, nesting the old memory
at zero phase. Its local teacher, initialization and serial/parallel contracts
pass; the completed one-pass result is 368/512 (71.875%), also below the best.
**Updated development result:** E143's inherited D8 parent plus parallel D6
width-128 signed-state encoder reaches **408/512 (79.69%)** after three passes,
with 449,110 total parameters. The original parent stays unchanged. On all 657
disjoint remaining utterances of the same held speakers, it improves from
463/657 to 510/657. Architecture/checkpoint selection uses the original private
sample; the added audit has no updates or reselection. This is not an official
test result or a from-scratch comparison. Clock-only reset loses 13 correct
answers and state-stack reset loses 50, holding the trained head fixed.

Earlier native E51 records include official-test evaluations. The current
common-model continuations do not access that partition; do not describe the
entire project's test history as untouched.

E141's frozen-parent source/phase continuation also finishes below the best,
at 369/512 (72.0703%). It leaves the entire inherited parent state unchanged
and clips none of the 1,536 active-new-block updates. This does not identify
joint clipping as the cause of the accuracy plateau. The next E143 architecture
adds a six-block signed temporal-state encoder as an initially zero logit
residual; it is a larger inherited parallel model, not a from-scratch result
for the original core. Published parity still requires the official protocol.

### Evaluation protocol and acceptance gates

1. Use the existing disjoint training-speaker development sample during
   architectural work. Record accuracy, NLL, absolute utterance IDs, model and
   optimizer lineage, preprocessing and work. Select on fitting/development
   data; never retain an update computed from development labels.
2. Before the official evaluation, freeze architecture, preprocessing,
   augmentation, training schedule and model-selection rule. A fitting-only
   validation partition or a predetermined epoch budget supplies selection.
   Refit on the designated official training data; keep the official test out
   of routine tuning. Record all official-test accesses.
3. Mark reference protocol differences. EventSSM's paper explicitly selected
   its best epoch on the test set; the delay-learning paper also used the SHD
   test set for validation. S7 describes validation-epoch selection while
   following EventSSM's setup. This is a comparability detail, not an excuse
   for our present quality gap. [EventSSM §4](https://arxiv.org/html/2404.18508v2),
   [learned delays §4.2](https://arxiv.org/html/2306.17670v3).
4. E143 now exceeds 370/512 on the existing development sample, reaching
   408/512. Retain this gain in a single generic learner and reach the
   leading verified official-test range (about 96%); specify the named
   reference, selection procedure and uncertainty when claiming parity.
   Recheck the primary literature at that point: the table is not an exhaustive
   or permanent global state-of-the-art claim.
5. Report forward/backward source projection, packet aggregation, all race
   alternatives, state updates, optimizer work, wall time and memory. Measured
   joules to a fixed quality target is a separate acceptance gate. Low work
   does not substitute for the requested accuracy parity.

`results/e119/scan_audit_s6.json` compares the same trained eight-layer E118
checkpoint under two event-memory implementations. All 256 development
predictions agree (104 correct), and the tested race winners agree. The
linear-work scan uses 3,925,048 vector combines rather than 21,553,320.
Warm median times per batch of four fall from 97.1 to 63.8 ms for inference
and from 347.2 to 206.8 ms for forward/backward, without optimizer updates.
RAPL `energy_uj` counters are unreadable in this container. These results
establish an internal work and latency improvement, not an energy frontier win.

The larger E119 run has now completed: final accuracy **151/256 (58.98%)**, best development
checkpoint **163/256 (63.67%)**, and fitting **827/1024 (80.76%)**. A frozen-model audit of
20 ms input windows retains **142/256 (55.47%)** while using **35.5% fewer packets** and
about **45% less model evaluation time**. This trades accuracy and input latency for CPU work;
it does not establish a published-benchmark or measured-joules win.

## Accuracy route

1. Keep winners as the only forward continuations and retain causal packet
   closure times. Fit-only readout conditioning addresses the demonstrated
   optimization bottleneck; linear-work memory makes more updates affordable.
2. Increase data and updates under a schedule fixed before looking at its
   results. The E119 run uses 1,024 fitting examples, the same 256 development
   examples, eight epochs, and learning rate 0.003 decreasing to 0.0003.
   It saves optimizer and random-generator state at every epoch.
3. Read fitting and speaker-held-out curves together. Strong fit with poor
   transfer points toward speaker invariance/augmentation; weak fit points
   toward representation or optimization. Analyze the temporal eligibility
   factors in THEORY §174 before changing time constants or race margins.
4. Once a configuration is selected, train with the full designated training
   budget and evaluate the official test partition under a recorded protocol.
   Reproduce a useful depth comparison at matched capacity or training cost.
   The current larger run changes examples, epochs and schedule together: it
   is an engineering progression, not a single-factor causal ablation.

## Energy route

- Preserve both accuracy and outputs when optimizing kernels first. The E119
  change removes redundant work without removing the races or counterfactual
  score credit. Sorting and local dense projections remain optimization targets.
- Measure warm batch-one streaming inference, batching separately, preprocessing,
  state maintenance, routing, output decisions, and idle baseline. Report the
  complete device boundary as well as an accelerator-only boundary if available.
- On a quiet host with readable counters, record package/DRAM energy separately;
  do not sum a package counter with an overlapping core subcounter. Record
  counter ranges/wraps, sample counts, wall time and accuracy together. CPU
  package energy includes unrelated work and is not a process meter.
- Compare training joules to reach a fixed held-out target separately from
  inference joules per utterance at a specified accuracy. Counterfactual work,
  all epochs, and optimizer updates belong in the training budget.
- Quantization, event reduction and early answers require new accuracy/latency
  measurements. Lower operation counts alone do not establish lower joules.

Reproduction commands live in unique one-job queue files. Use `run_safe.sh` with
at least 8 GiB host available-memory reserve. Do not run another benchmark while
its host-local lock is held. The AWS sibling retains ownership of non-SHD work.
