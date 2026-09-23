# Sleeping Machines: first experiments

*Experiment log, 23 September 2026. Interactive version of this report: the page built from `report/index.html`.*

Sleeping Machines proposes that computation can happen in **time rather than memory**: candidate events race, the first to fire cancels the rest, and the cancelled ones keep a **trace of how close they came** so that a later teaching signal can use it. This report covers the first experiments designed to test that proposal. Predictions were written down before each evaluation was run, and the report includes the parts that did not work.

![One race: B fires, A and C are cancelled but keep their distance to threshold](report/figures/race.png)

**Contents:** [Summary](#summary) · [Method](#method) · [Scorecard](#scorecard) · [E4 Counterfactual credit](#e4--a-cancelled-node-can-be-taught) · [E2 Adaptive decisions](#e2--races-decide-as-fast-as-the-evidence-allows) · [E6 Hidden layers](#e6--hidden-layers) · [Energy](#energy) · [Limitations](#limitations) · [Next](#next-steps) · [Reproducing](#reproducing)

## Summary

- **A cancelled node can be taught (E4).** In a race among K output nodes, the right answer usually loses and never fires, so rules that credit only nodes that fired (the STDP / reward-modulated family) can only punish the wrong winner. Keeping each cancelled node's distance to threshold Δ lets a delayed teaching event promote the right answer and push down only the competitors that came close. At K = 128 this reaches **0.79** accuracy while the reward-modulated rule is at chance, using **6%** of the plasticity updates of uniform credit. It still trails a global-gradient reference (0.93).
- **Races decide as fast as the evidence allows (E2).** An accumulator race dominates a fixed-time decoder at every decision time and matches the MSPRT, the statistically optimal stopping reference, with nothing but additions and a threshold. Decision time adapts to difficulty (0.29 s on easy trials, 1.69 s on hard ones) and 74% of the input events are never processed.
- **Credit reaches hidden layers (E6, interim).** On latency-coded MNIST a hidden layer trained by counterfactual race learning beats both a single racing layer and a frozen random hidden layer, and counterfactual hidden credit beats fired-only hidden credit. It is **not competitive with dense networks yet**: interim 0.88 vs 0.978 for a dense MLP and 0.92 for a plain linear classifier.
- **Energy gains are single-digit factors so far.** Priced with published per-operation energies, event-driven learning is about 6.5× cheaper than unbatched dense training at K = 128, roughly at parity with batched dense training, and more expensive on today's neuromorphic silicon. Orders-of-magnitude savings, if they exist, need the dormant-capacity scaling experiment that has not been run yet.

## Method

**Event engine.** `sleeping_machines/sim.py` is a discrete-event simulator with no global clock. It keeps a pool of pending future events, jumps to the next one, allows pending events to be cancelled, and counts every operation (events scheduled, fired and cancelled; synaptic operations; spikes; plasticity updates). All work and energy figures in this report come from these counters.

**Nodes.** Non-leaky integrate-to-threshold units: each input spike adds its weight to the potential; the first node to reach threshold fires. In a race, the winner's fire event inhibits the others and cancels their pending fire events. At that moment every cancelled node freezes Δ = (θ − v)/θ, its normalised distance to threshold.

**Teaching.** A teaching event naming the correct answer arrives after the race, possibly after a delay. Each synapse holds a presynaptic trace x = exp(−age/τ). All learning rules are local and triggered only by the teaching event.

**Discipline.**
- Every experiment has a preregistration file (`experiments/E*_PREREGISTRATION.md`) written before its evaluation seeds were run.
- Hyperparameters are tuned per rule and per problem size on separate tuning seeds or a validation split.
- Results are reported on held-out seeds or the test set, with 95% confidence intervals where there are several seeds.
- Amendments made after tuning are dated and explained in the preregistration files.

## Scorecard

| Test | Prediction (written before the evaluation) | Verdict | Evidence |
|---|---|---|---|
| E4 · P1 | At K = 3, rules that credit only the fired node learn almost as well as counterfactual ones. | 🟡 Partly | fired rules learn (0.75–0.79) but trail cf_margin (0.99) by 20 points |
| E4 · P2 | As K grows, fired-only rules collapse while counterfactual rules keep learning. | 🟢 Supported | K = 128: fired_reward 0.014 vs cf_winner 0.414 |
| E4 · P3 | Near-miss weighting matches uniform credit with under 25% of the plasticity updates. | 🟢 Supported | 0.79 vs 0.40 accuracy with 6% of the updates (after two fixes to the uniform baseline) |
| E4 · P4 | Credit survives delays up to about τ; an intervening race erases it. | 🟢 Supported | 0.91 at D = τ in quiet conditions; 0.12 at D = 2 when races intervene |
| E2 · P1 | The race's speed–accuracy frontier lies above fixed-time decoding. | 🟢 Supported | higher accuracy at every matched decision time |
| E2 · P2 | Decision time falls monotonically with evidence strength. | 🟢 Supported | 1.69 s → 0.29 s across coherence 0.05 → 0.8 |
| E2 · P3 | The race needs under 20% more time than the MSPRT at equal accuracy. | 🟢 Supported | no extra time needed in this setting |
| E6 · 6a | A hidden layer trained by CRL solves XOR-in-time; a single layer cannot. | 🟢 Supported | single layer 0.59, frozen hidden 0.63, CRL 0.92 |
| E6 · 6b | Latency-coded MNIST at 97% or better with local, event-driven learning. | 🔴 Not yet | interim best 0.88 |

## E4 · A cancelled node can be taught

*Single racing layer, sparse spike patterns over 256 channels, 6,000 teaching events, 10 held-out seeds.*

| rule | target (never fired) | competitors |
|---|---|---|
| `fired_only` | not updated | winner depressed on error |
| `fired_reward` | not updated | winner potentiated if right, depressed if wrong (reward-modulated STDP) |
| `cf_winner` | potentiated on error | winner depressed |
| `cf_uniform` | potentiated on error | every non-target depressed by 1/(K−1) |
| `cf_margin` | potentiated on error | non-target k depressed by exp(−Δₖ/σ), skipped when negligible |
| softmax (reference) | global gradient, not event-driven | |

![E4 accuracy vs number of classes](report/figures/e4_accuracy.png)

| rule | K = 3 | K = 8 | K = 32 | K = 128 |
|---|---|---|---|---|
| softmax (reference) | 0.997 | 0.989 | 0.972 | 0.926 |
| fired_only | 0.753 | 0.270 | 0.042 | 0.009 |
| fired_reward | 0.790 | 0.847 | 0.390 | 0.014 |
| cf_winner | 0.980 | 0.908 | 0.700 | 0.414 |
| cf_uniform | 0.977 | 0.901 | 0.666 | 0.400 |
| **cf_margin** | **0.991** | **0.969** | **0.921** | **0.792** |

<p>
<img src="report/figures/e4_work.png" width="49%" alt="E4 accuracy vs plasticity work">
<img src="report/figures/e4_delay.png" width="49%" alt="E4 delayed teaching">
</p>

**Reading.**
- Counterfactual credit is what keeps learning working as the number of alternatives grows. `cf_margin` is also the cheapest learner at K = 128: 2.4M synaptic updates against 39M for `cf_uniform` and 22M for softmax.
- Credit survives a delayed teacher while nothing else happens. Once the trace decays past τ, or the next race overwrites the frozen record before the teacher arrives, it is lost.
- The gap to the global-gradient reference grows with K. Local counterfactual credit is better than fired-only credit, not a replacement for a gradient.

## E2 · Races decide as fast as the evidence allows

*Four classes; evidence arrives as Poisson spikes whose coherence (how strongly they favour the right class) varies unpredictably between trials; 5 seeds × 2,000 trials.*

The race node for class k adds 1 for each spike in its own group and subtracts 1/(K−1) for every other spike; the first node to reach θ fires, and all pending input events are cancelled. It is compared with a fixed-time decoder (count spikes until T, pick the largest) and with the MSPRT on the exact likelihoods.

<p>
<img src="report/figures/e2_frontier.png" width="49%" alt="E2 speed-accuracy frontier">
<img src="report/figures/e2_coherence.png" width="49%" alt="E2 decision time by coherence">
</p>

| decoder | accuracy | mean decision time | input events processed |
|---|---|---|---|
| race θ = 15 | 0.794 | 1.03 s | 208 |
| MSPRT α = 0.2 | 0.785 | 1.12 s | 223 |
| fixed-time T = 1 | 0.735 | 1.00 s | 200 |
| fixed-time T = 2 | 0.797 | 2.00 s | 400 |

**Reading.** The race matches the optimal-stopping reference using only additions and a threshold. It reaches the fixed-time decoder's T = 2 accuracy in half the time and with half the events.

## E6 · Hidden layers

*Latency-coded MNIST: a brighter pixel spikes earlier; one spike per pixel; 1,000 hidden nodes in groups of 10 with one winner per group.*

**Counterfactual race learning (CRL).**
- The output error (+1 for the target; −exp(−Δ/σ) for near-miss competitors) is sent back as events through feedback synapses: symmetric, sign, or fixed random (feedback alignment).
- Hidden nodes that fired, *and hidden nodes that were cancelled close to threshold*, update their input synapses from their stored traces.
- For these non-leaky single-spike neurons, the firing time is piecewise constant in the weights, so the exact gradient is zero almost everywhere. The frozen Δ plays the role that a surrogate gradient plays in conventional spiking-network training.
- Training uses a batched closed-form solver. It agrees with the discrete-event engine on 100% of decisions once simultaneous events are given the same semantics in both: events at one instant are integrated together, and simultaneous crossings are resolved by overshoot.

**Interim results (60,000 training images, test set; runs in progress, 10 epochs each):**

| model | epoch 1 | epoch 2 | notes |
|---|---|---|---|
| dense MLP, 1000 hidden, backprop (reference) | 0.939 | 0.956 | 0.978 after 10 epochs |
| linear classifier (reference) | | | 0.922 after 10 epochs |
| single racing layer | **0.864** | 0.862 | declined to 0.824 by epoch 10 without learning-rate decay |
| frozen random hidden layer | 0.760 | 0.773 | |
| CRL · fired-only hidden credit | 0.852 | 0.868 | |
| CRL · feedback alignment | **0.869** | **0.880** | |
| CRL · symmetric feedback | 0.863 | | |

**What was learned on the way:**
1. **Hasty decisions.** In latency-coded MNIST all saturated pixels spike at t = 0, so a race can be decided on that first instant. Median decision time was 0.4–1.2% of the input window, and a race cannot use evidence that arrives after it has decided.
2. **Lateral inhibition helps.** Letting every spike also inhibit all outputs by the mean weight makes the output race run on *relative* evidence, as the E2 race does. With a high enough threshold and learning-rate decay, the single racing layer improves from 0.86 to **0.90** on the validation split. That is still below the linear classifier.
3. **Symmetric feedback diverged** until its class-common component was removed. The output signal rarely sums to zero, so a feedback matrix with positive mean pushed every hidden node the same way.

The full-data runs with lateral inhibition, learning-rate decay and several winners per hidden group come next.

## Energy

The simulators count synaptic events, spikes, plasticity updates and multiply-accumulates. `sleeping_machines/energy.py` prices those counts on hardware profiles built from published per-operation energies:

| profile | per operation |
|---|---|
| dense int8 accelerator, weights in local SRAM | 1.55 pJ per multiply-accumulate (0.31 pJ when a batch of 256 shares each weight fetch) |
| dense fp16 training | 4.0 pJ per multiply-accumulate, 5.4 pJ per weight update (both less when batched) |
| **idealized event-driven, near-memory** | 1.3 pJ per synaptic event (8-bit weight fetch + 16-bit add), 2 pJ per spike, 2.6 pJ per weight update |
| event-driven, weights in a large shared SRAM | 12.6 pJ per synaptic event |
| Loihi, measured (14 nm) | 23.6 pJ per synaptic event, 120 pJ per weight update |

Arithmetic and SRAM figures are 45 nm values from Horowitz (ISSCC 2014); Loihi figures are from Davies et al. (IEEE Micro 2018). The "suitable architecture" assumption is that a binary spike turns a synaptic operation into a weight fetch plus an addition, with no multiplication and no clock. These are order-of-magnitude estimates, not chip measurements.

![Training energy per teaching event, E4 at K = 128](report/figures/energy_e4.png)

**Reading.**
- At K = 128, `cf_margin` costs about **5.3 nJ per teaching event** on the idealized event profile. That is 6.5× less than unbatched dense softmax training (34 nJ) and about equal to batched dense training (5.6 nJ). The dense model is also more accurate.
- On Loihi's measured costs the event rule would cost 124 nJ.
- On MNIST the comparison is always against the cheapest dense model that is at least as accurate. A linear classifier with 7,840 multiply-accumulates reaches 0.92, which a racing network has not matched yet.
- The single racing layer uses only ~460–830 synaptic events per image, so at *lower* accuracy it is much cheaper. The matched-accuracy table is produced by `experiments/energy_report.py` once the E6 runs finish.

## Limitations

- Everything runs in simulation on CPU. Energies are estimates from operation counts, not measurements.
- E4 and E2 use synthetic tasks designed to isolate one mechanism each; they show that a mechanism works, not that it is useful on real data.
- The single-spike, non-leaky neuron is deliberately simple. Leaky neurons, multiple spikes per node and learned delays are not tested yet.
- MNIST latency coding favours hasty decisions and is a poor showcase for timing. A benchmark where timing carries the information (Spiking Heidelberg Digits) is the fair test.
- One training seed per E6 configuration so far.

## Next steps

1. **E6, second round.** Full-data runs with lateral inhibition, learning-rate decay, several winners per hidden group, and a learned decision threshold; then learned delays.
2. **E5, dormant capacity.** Measure whether inference and learning work stay flat as network capacity grows while most of it stays silent. Compare against sparse dense baselines (mixture-of-experts style), not only dense ones.
3. **Spiking Heidelberg Digits** on a GPU, against published learned-delay results.

## Reproducing

```bash
python experiments/e4_counterfactual.py tune && python experiments/e4_counterfactual.py main && python experiments/e4_counterfactual.py delay
python experiments/e2_race_decisions.py
python experiments/e6_hidden.py mnist --variant crl_fa --hidden 1000 --epochs 10
python experiments/dense_frontier.py
python experiments/energy_report.py && python experiments/build_report.py && python experiments/make_figures.py
```

Dependencies: `numpy`, `matplotlib`. Related prior art is listed in the [README](README.md#related-prior-art).
