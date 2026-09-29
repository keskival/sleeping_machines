# SHD: accuracy and energy reference points

Updated 2026-09-29. These are verified primary-source reference points, not an
exhaustive claim about the highest result anywhere in the literature.

## What must be compared

Standard SHD has 20 classes, 8,156 training and 2,264 test utterances, and 700
input channels. Two speakers occur only in the test set. Our current deep
carrier development runs use speakers 3/6 of the training file as held-out
speakers. Their scores must remain separate from official test accuracy.
[Dataset authors and leaderboard](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).

| Reference | Reported SHD accuracy | Resource evidence and scope |
|---|---:|---|
| Cramer et al., original dataset paper | 85.7% LSTM | Corrects the earlier report summary of approximately 70% |
| EventSSM | 95.9% | Official implementation; asynchronous event input |
| S7 | 96.3% | Paper reports 0.5M parameters; event input |
| Sun et al. 2025, attention for delay SNNs | 96.26 ± 0.08% | Dataset-maintainer leaderboard; no joules comparison here |
| Chen et al. 2025, delay SNN on FPGA | 93.4% deployed | 8-bit weights; processor 282 mW, complete SoC 1.71 W, approximately 104 samples/s |

Sources: [Cramer et al., Section III](https://www.kip.uni-heidelberg.de/Veroeffentlichungen/download.php/6616/temp/4143-3.pdf),
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
