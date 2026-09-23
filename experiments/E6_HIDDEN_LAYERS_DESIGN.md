# E6 — Counterfactual race learning in hidden layers

Written 2026-09-23. A design, not yet a result.

## Why this is the experiment that matters

A single racing layer is a linear classifier with an unusual readout. What made
deep learning work is credit assignment through hidden layers. Spiking and
event-driven systems have lagged there: either they train with backprop through
time on a simulated clock (accurate, but neither local nor event-driven), or they
use local rules such as STDP, which trail badly in accuracy.

So the question for Sleeping Machines is: **can a teaching event deliver useful
credit to hidden nodes, including hidden nodes that never fired, using only
information stored locally at events?**

## Key observation: the counterfactual trace acts as a surrogate gradient

Surrogate-gradient SNN training replaces the zero-or-undefined derivative of
"spike vs. membrane potential" with a smooth bump that peaks at threshold, e.g.
σ'(θ − v). The counterfactual eligibility in E4 is

    e_k = exp(−Δ_k / σ),   Δ_k = (θ_k − v_k) / θ_k   frozen at cancellation,

which has the same shape: it is largest for nodes that nearly fired and vanishes
for hopeless ones. The difference is *where and when* it is computed:

| | surrogate-gradient BPTT | counterfactual race learning |
|---|---|---|
| when | every simulated timestep, stored for the whole sequence | once per node per race, at fire/cancel |
| memory | O(T · N) activations | O(N): one snapshot per node and one trace per synapse |
| credit signal | exact backprop through time | a delayed teaching event plus a feedback projection |
| hidden nodes that never fired | through the surrogate | through the snapshot, which is the same idea stored as an event record |

If this analogy holds, counterfactual race learning is surrogate-gradient
learning *collapsed to events*. That is a clear, defensible, testable claim.

## The learning rule (CRL: counterfactual race learning)

Architecture: input spikes → hidden layer → output race.

- The hidden layer consists of G competition groups (cortical-column-like), each
  a k-winner race: the first m nodes of the group to reach threshold fire; the
  rest are cancelled and keep a Δ snapshot.
- Output: a K-way race, as in E4.
- Hidden spike times feed the output layer. Earlier spikes contribute earlier,
  so the output race sees time-coded evidence.

When the teaching event arrives:

1. Output layer, exactly as in E4 `cf_margin`: the target is potentiated,
   near-miss competitors are depressed, each scaled by its presynaptic trace.
2. The output error signal is s_o = +1 for the target, −e_o for competitor o.
3. It is sent back as events through feedback synapses B (o→h): δ_h = Σ_o B_oh · s_o.
   - `crl_sym`: B = Wᵀ, i.e. weight transport, the upper bound.
   - `crl_fa`: B fixed and random (feedback alignment), fully local.
   - `crl_sign`: B = sign(Wᵀ), a cheaper middle ground.
4. Hidden update: ΔW_hi = η · δ_h · e_h · x_i, where e_h = 1 for fired
   hidden nodes and exp(−Δ_h/σ) for cancelled ones.
   - `crl_fired_only` ablation: e_h = 0 for cancelled hidden nodes. This is the
     key test of whether counterfactual traces matter in depth.

Phase 2 adds learnable **delays** per synapse, using the timing counterfactual:
an input that arrived *after* a node was cancelled, but would have pushed it
over threshold, is advanced if that node should have won. This makes time,
not only weights, the learned substrate, which is the heart of the manifesto.

## Implementation note

For non-leaky integrate-to-threshold nodes, the event simulation has a closed
form. Sort the input spikes by time, take the cumulative weighted sum, and find
the first crossing. That gives exact first-spike times in batched numpy.
Training uses the batched solver. The event engine is kept for (a) an
equivalence test on a subset (identical decisions and spike times) and (b)
counting work. Leaky nodes and multi-spike hidden units would need the full
engine or an exact-time solver, and are deferred.

## Evaluation ladder

| stage | task | why | what "success" means |
|---|---|---|---|
| 6a | XOR-in-time: classes defined by conjunctions of input-group timing | not linearly separable, so one layer must fail | CRL ≫ single layer ≈ chance; `crl_fired_only` is clearly worse than CRL |
| 6b | MNIST, latency-coded (brighter pixel → earlier spike, one spike per pixel) | the standard time-to-first-spike (TTFS) benchmark, with published local and backprop baselines | ≥ 97% puts it in range of TTFS-backprop results; ≥ 95% beats unsupervised STDP-WTA |
| 6c | SHD (Spiking Heidelberg Digits: 700-channel cochlea spikes, 20 spoken digits) | a practical benchmark where timing is essential and learned delays are state of the art | any local, event-driven rule in the 80s would be notable; BPTT with learned delays reaches the mid-90s |
| all | work per inference (synops, events) and per learning step (plasticity updates) | the efficiency claim | report accuracy vs work frontier against a dense MLP (MACs) and a surrogate-gradient SNN of equal size |

Literature numbers to verify before quoting: TTFS-backprop MNIST (Mostafa 2017;
Comsa et al. 2020; Göltz et al. 2021, roughly 97–98%); STDP-WTA MNIST
(Diehl & Cook 2015, about 95%); SHD (Cramer et al. 2020 baseline; Hammouamri
et al. 2023, learned delays, about 95%).

## Baselines built here, not just cited

1. The same architecture trained by **backprop through the exact spike times**,
   via the analytic dt/dw of the closed-form solver. This is the ceiling for this network.
2. A dense MLP with the same hidden size. This is the ceiling for accuracy and the reference for work.
3. `crl_fired_only`, and output-only learning with a frozen random hidden layer
   (reservoir/extreme-learning-machine style). If CRL cannot beat a frozen random
   hidden layer, the hidden credit is not doing anything.

## Honest expectations

- 6a should work. If it doesn't, the rule is broken.
- 6b at 97% or more is plausible but not assured. Feedback-alignment-style rules
  typically lose one to two points against backprop on MNIST.
- 6c is where a real claim could be made, and also where things are most likely to fail:
  SHD needs memory across hundreds of milliseconds, which this non-leaky,
  single-spike design only gets through delays (phase 2). I would plan on 6c
  taking several iterations.
- Being competitive with the state of the art means being competitive *within event-driven,
  local learning*, while reporting honestly against BPTT and ANNs.
