# Findings log

One entry per result: what we ran, what came out, what it teaches us, what changes.
Newest first. Numbers are single seeds unless stated.

## 2026-09-25

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
