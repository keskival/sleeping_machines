# Findings log

One entry per result: what we ran, what came out, what it teaches us, what changes.
Newest first. Numbers are single seeds unless stated.

## 2026-09-25

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
