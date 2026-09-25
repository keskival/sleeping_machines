# E8 — What the race does that classical spiking networks do not

Written 2026-09-25, before any E8 code. Literature numbers are verified before
they are quoted; every baseline here is run by us, with operations counted the same way.

## Question

Classical spiking networks share our vocabulary (spikes, thresholds, events), so
"it spikes" is not an advantage. We claim three real differences:

1. **No clock.** Our simulation and our hardware model jump from event to event,
   exactly; clocked SNNs update every neuron every step.
2. **Cancellation.** Our inhibition removes pending future work; in a clocked SNN an
   inhibited neuron keeps being updated.
3. **Near-miss credit with a tiny learning state.** A cancelled node keeps one value
   (Δ) that a later teacher can use. BPTT keeps the whole state history; e-prop
   updates a trace per synapse per step; STDP cannot credit neurons that never fire.

Each is turned into an experiment that can fail.

## Models (all ours, same data, same counting)

- **race**: our network (single layer, and the E6/E9 hidden network).
- **clocked IF**: the same non-leaky neurons and weights, simulated with a clock at
  step dt. Isolates the cost and error of the clock itself.
- **LIF + surrogate-gradient BPTT**: leaky neurons, fixed window T, fast-sigmoid
  surrogate, the standard modern SNN recipe. A small numpy reference first
  (784-100-10); PyTorch only if needed.
- **e-prop**: the same LIF network with online eligibility traces and random feedback.
- **STDP winner-take-all** (Diehl & Cook style, unsupervised + label assignment),
  **reward-modulated STDP**, **Tempotron** (for E8d).
- **TTFS exact gradient**: backprop through exact spike times of our own neurons
  (from E11): the "same neuron, non-local learning" ceiling.

## Experiments

**E8a — the cost of a clock.** Take trained race weights, run clocked IF at
dt ∈ {10⁻¹, 10⁻², 10⁻³, 10⁻⁴} of the input window; train LIF-BPTT at each dt.
Measure accuracy, decision agreement with the exact race, and operations (neuron
updates + synaptic events).
*Prediction:* clocked cost rises ~10× per decade of resolution while accuracy
saturates by 10⁻²; the race's cost is flat and exact. *Fails if* a coarse clock
(10⁻¹) already matches race accuracy at lower cost — then exact timing is not
worth its price on this task, and we say so.

**E8b — learning state and learning work.** For each learner, the peak bytes that
must be kept to make one update, and operations per training sample, as the input
window T and the width grow.
*Prediction:* BPTT memory ∝ T × neurons; e-prop work ∝ T × synapses; ours ∝ nodes
(one Δ each) plus the input spike times, independent of T. Expected ≥ 100× less state
than BPTT at T = 100 steps. *Fails if* e-prop's state and work are within 3× of ours
at equal accuracy — then our learning-cost advantage is only over BPTT.

**E8c — dead neurons.** Fraction of hidden units that receive no credit on a sample,
and that never change over training, as width grows 100 → 4000, for surrogate BPTT,
e-prop and the race (Δ credit and fired-only credit).
*Prediction:* surrogate methods credit every unit a little (dense, costly) or, with a
narrow surrogate, leave many dead; the race credits a sparse, targeted set, and
fired-only leaves most units unused. This is where Δ credit should show value even
though its accuracy benefit has not appeared yet (E6 r3).

**E8d — supervised local learning at scale.** E4 extended with STDP-WTA,
reward-modulated STDP and Tempotron, K = 3 … 128 (and 1024 if cheap), 10 seeds.
*Prediction:* only counterfactual credit stays well above chance at K = 128.
The existing E4 result predicts this; the named baselines make it a comparison
reviewers recognise.

**E8e — accuracy per total work.** The scoreboard that matters: accuracy against
total operations (training + inference), and against learning state, for every
model. Run after E9 and E11 so the race is at its best.

## Honest expectations

Surrogate-gradient LIF networks will likely be more accurate, and leaky, recurrent
neurons have temporal memory ours lack. The claim we test is not "more accurate
than SNNs" but "comparable accuracy for far less clocked work, learning state and
learning work", plus capabilities they lack (credit for silent units, adaptive time).

## Protocol

Same tuning budget for every model on the validation split (a fixed number of
pilot configurations each). E8a–c: 3 seeds; E8d: 10 seeds; small networks,
one job at a time through `run_queue.sh`.
