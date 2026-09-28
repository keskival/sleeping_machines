# Partial evidence, slow state, speech, and sparse inference

[Theory index](../THEORY.md) · Previous: [06 world models and sparse topology](06_world_models_and_sparse_topology.md) · Global sections 89–93; section numbers remain stable.

## 89. The race's shortcut: partial-evidence routes, and crediting the latest instant

*Written 2026-09-27, after the E56 error analysis, before the 5-seed runs.*

**Observation.** On E34's task with learned windows, every error on seed 4 is a lost race: another class fires first.
The winners use a route that holds motif A's part and is triggered by the *cross* pair (A's end → B's start), so they fire
when B begins, before B is complete. On other classes' examples a noise spike on B's first channel inside the gap window
completes that route: it is right about 99% of the time. Its rare false fires demote it, but (a) the earned margin
protects it (precision ≈ 0.99 clears a 0.9 gate) and (b) on the next misses cooled exploration re-credits the prefix
instant, whose drive is high, about 80% of the time; the complete route's trigger (B's part) starts near zero and
never catches up. This is the speed–accuracy trade-off of any first-to-fire readout: commit early on partial evidence
and pay its error rate; a model that decides after the whole episode (a Transformer) does not face it.

**Proposition (latest-instant credit).** For a class whose pattern ends with a fixed last event, every earlier instant
at which a route can fire is a prefix of the pattern; a prefix route is valid only if no other class or decoy shares the
prefix, whereas a route at the pattern's last event with the rest of the pattern held is valid whenever the pattern
determines the class. Crediting, on a miss, the latest candidate instant of the example therefore always credits a
superset of the complete evidence. Instants after the pattern (noise) vary between examples, so by §83(i) their units
lose weight relative to the pattern's (they are promoted in a fraction f < 1 of the misses). Unlike exploration, this
choice is deterministic and needs no temperature: it removes both the prefix trap of §84 and the shortcut above.
*Observed (one seed each):* E53 (depth 3) 0.998 after 1,000 episodes with 435 updates (exploration: ≈ 5k episodes,
≈ 1,100 updates).

**Predictions (5 seeds, 40k episodes).** P1: E53 with latest-instant credit ≥ 0.99 at the final checkpoint on 5/5 seeds,
with fewer updates than exploration (≈ 1,100). P2: E34's task with learned windows and latest-instant credit ≥ 0.995 on
≥ 3/5 seeds (the shortcut removed). P3: with the earned margin added, no seed below its no-margin value.

**Results (5 seeds).** P1 holds with the earned margin (E53: 0.997–0.999 on 5/5 at every checkpoint, 476–560 updates,
half of exploration's); without it the same values from the first checkpoint (5k episodes; 550–690 updates) with one final
dip. P2 holds: E34's task with learned windows and latest-instant credit 0.9987 / 0.9967 / 0.9993 / 0.990 / 0.998
(≥ 0.995 on 4/5; mean 0.9965) from 40k examples seen once, against the Transformer's 0.9955–0.998 after 2M examples and
0.9935 / 0.9965 given the same 40k examples 50 times without weight decay (0.982 / 0.985 with): parity at equal data,
from one pass instead of fifty. Both pieces are needed: with fixed windows latest-instant credit keeps
the old plateaus (0.988–0.995), because the complete route (A held with the gap part, B's part as trigger) is only
expressible once the gap part's window is learned. P3 fails: with the margin gated at precision 0.9, two seeds of E34's
task fall to 0.976–0.981, the margin again protecting routes that are right ≈ 99% of the time. The gate must exceed
the precision the task requires (a route wrong 1% of the time must stay correctable when the target is 99.8%): gate
0.99 queued.
**Follow-ups.** Gate 0.99: depth 3 stable (0.997–0.999, 476–565 updates) but E34's task with learned windows unchanged
(seeds 3–4 at 0.971 / 0.982): the level of the gate is not the cause. Near-miss credit at the latest instant instead of the
firing instant makes E34's task worse (0.92–0.98, unstable) while depth 3 is unaffected. The reason: the latest candidate
instant of an example is often a noise event after the pattern. On a miss (rare) that costs little, since inconsistent
noise fades (§83(i)); near-miss credit acts on every correct answer, so it keeps promoting noise instants. Latest-instant
credit is right for misses; the margin needs another anchor (open). Best configuration on E34's task so far: learned
windows and latest-instant credit without a margin (0.990–0.9993; occasional dips).
## 90. The world model needs slow state, and slow state must be counted, not tracked

*Written 2026-09-27, after E52 (validation), E57 and E58.*

**The gap.** A Transformer Hawkes process with the event network's own piecewise-constant hazard family, trained on days
1–4 and frozen, scores −2.43 nats per event on day 5; the semi-Markov event network (last two event types, gap window)
scores −3.12. Only the context differs: the Transformer attends over the last 128 events and their times, and so can
read the recent regime (how fast events are arriving, which side is pressing).

**Slow state, native.** Leaky counters updated only at events, c ← c·e^(−Δ/τ) + 1, are event nodes whose level is the
recent rate at scale τ; quantized, they join the event network's state, and rare (type, regime, window) cells back off
to the plain semi-Markov estimate. With rate counters at 5 s and 60 s (and an order-flow counter) day 5 improves from
−3.12 to −2.76 (half the gap), and the held-out test days from −2.38 / −2.10 to −2.18 / −2.00 (E57), at a few extra
counter updates per event. Backoff strength barely matters (m = 5 or 20: within 0.01 nats).

**Counting versus tracking (a principle for frozen transfer).** A factorized version (per-feature multiplicative factors
on the hazard, learned by exponentiated-gradient Poisson regression, E58) looks better on day 5 (−2.63) but is worse on
the test days (−2.37 / −2.23, below even the plain model on day 7), and worse the larger its step (η = 0.03: −2.65 /
−2.42). With a constant step, multiplicative updates weight recent evidence exponentially: the frozen factors describe
the end of the last training day, not the regime structure, and its quantile edges (from the first events of day 1) do
not travel either. Counts average all the evidence and fixed rate edges mean the same thing on every day, so they
transfer. For a model that is frozen and then deployed, estimate by counting; constant-step multiplicative updates
belong to tracking (online use), where recency is the point.

**Against the Transformer on the test days (E52 test, selected on day 5 per hazard family).** Coarse windows: THP −1.97 /
−1.82 nats per event (≈ 108k multiply-adds) vs the event network with regime counters −2.18 / −2.00, and with a third
backoff level of per-type counters (τ = 2 s) −2.16 / −1.97; fine windows: THP −1.83 / −1.65 vs −1.91 / −1.73 (variants chosen on day 5;
an earlier −2.15 / −1.96 was a variant not preferred by validation). The Transformer is the better world model by 0.08–0.19 nats per event; the counted event network costs ≈ 30–40 operations
per event (≈ 1/3000–1/4000).
## 91. Latest-instant credit under trailing noise: a drift bound, and where the margin should stand

*Written 2026-09-27, before E91.*

**Setting.** A class node with summed potentials and conserved multiplicative credit; on a miss it promotes, by (1 + α),
the deficient roles at the latest candidate instant of the example (§89); on a false fire it demotes, by (1 − β), the
contributors at the firing instant. The target route fires its trigger unit l* at the pattern's last event and is valid
(never complete in a negative). Let q be the probability that an example's latest candidate instant lies after the
pattern (trailing noise), f_j the probability that a non-target unit j is in the credited trigger set on a miss, and ρ the
rate, per miss, of false fires to which l* contributes.

**Proposition (drift).** For every distractor j, Λ_j = log(g_{l*} / g_j) (renormalization cancels, §83(i)) changes on a miss
by +log(1 + α) when the credited instant is the pattern's end and j is not credited there (probability ≥ 1 − q − f_j),
by −log(1 + α) when j is credited and l* is not (probability ≤ f_j), and by at most −log(1/(1 − β)) per false fire in which
l* is demoted and j is not. Hence

  E[ΔΛ_j per miss] ≥ (1 − q − 2 f_j) · log(1 + α) − ρ · log(1/(1 − β)).

When the right side is positive, every distractor's share decays geometrically and the trigger role concentrates on l*
after M = O((log Q + log((1 − θ)/θ)) / drift) misses: logarithmic in the candidate basis Q, and slower as trailing noise
grows, as 1/(1 − q − 2f). As q → 1 − 2f (the pattern's end is rarely the latest instant) learning stalls. The same bound
holds for the hold role with the window before the credited instant.

**Consequence for the margin (§86, §89).** Near-miss credit acts on correct answers, which are frequent, so whatever
instant it anchors to is promoted far more often than misses are credited. At the firing instant it protects shortcuts;
at the latest candidate instant it multiplies the trailing-noise term q by the rate of correct answers. The anchor
consistent with the bound is the latest instant at which the node's *own* drive already exceeds θ: trailing noise units
carry little weight, so this is the pattern's end once the complete route has weight, and the margin then protects what
the node already recognizes rather than whatever came last.

**Predictions.** P1 (E91 noise sweep, E53 with latest-instant credit, 5 seeds, noise probability 0.1 / 0.25 / 0.4 / 0.55):
the measured q rises with the noise, and the updates needed grow at least as fast as 1/(1 − q); accuracy stays ≥ 0.99
while 1 − q is well above 2f, and degrades at the highest noise. P2 (E91 anchor, E34's task with learned windows): the
margin anchored at the latest supra-threshold instant keeps every seed at or above its no-margin value (0.990–0.999)
without the dips.

**Result, P1 (E91 noise sweep, 5 seeds, 20k episodes): holds, quantitatively.** Measured q (the fraction of positive
examples whose latest candidate instant is after the pattern) 0.105 / 0.284 / 0.459 / 0.62 at noise 0.1 / 0.25 / 0.4 /
0.55; updates to reach 0.99 (mean over seeds) 359 / 453 / 603 / ≥ 934 (two seeds never reach it at the highest noise);
ratios to the lowest noise 1 / 1.26 / 1.68 / ≥ 2.6 against 1/(1 − q): 1 / 1.25 / 1.65 / 2.35. With the one remaining
parameter of the bound fitted at the highest noise (f ≈ 0.03), 1/(1 − q − 2f) predicts 1.27 and 1.73 for the middle
levels (measured 1.26, 1.68). Accuracy 0.999–1.000, 0.997–0.999, 0.995–0.999, then 0.925–0.995 at the highest noise;
activity grows with the noise (26 → 90 events per example).

**Result, P2 (E91 anchor): fails, identically.** The supra-threshold anchor reproduces the firing-instant margin seed for
seed (0.992 / 0.991 / 0.999 / 0.976 / 0.981). A shortcut that is right ≈ 99% of the time is exactly what such a node
recognizes: its latest supra-threshold instant is its firing instant, and the complete route never crosses θ. Any margin
anchored in what the node already recognizes protects the shortcut; on this task the near-valid route is the problem, so
a margin cannot tell it from the valid one by the node's own drive or precision. Open: a margin that requires the
route to be complete (for example, protection only for routes whose trigger is the latest part of every positive they
fired on).
## 92. Spoken digits across speakers: validate on unseen voices, code bands relative to the voice

*Written 2026-09-27, before E59.*

**Protocol flaw found.** E51 chose its configuration on a random 10% of the training utterances, spoken by the training
speakers, and scored 0.73 there but 0.647 on the test set, whose utterances come mostly from two speakers never heard in
training (1,840 of 2,264). Test accuracy barely moved across very different configurations (0.647 at B = 140, O = 40;
0.649 at B = 35, O = 5): selection optimized speaker-specific detail. Validation must hold out whole speakers (here
speakers 3 and 6, 1,169 training utterances).

**Relative coding.** A voice shifts formants along the cochlear (band) axis; a class-conditional event model on absolute
bands learns the training voices' positions. Native invariance: each utterance keeps a running sum and count of the
bands of its spikes (two counters, updated per spike), and both the conditioning state (the last spike's band) and the
predicted next band are coded relative to the running centroid. The transform is shared by all class models, so the
likelihood comparison stays exact. This is the reference-frame idea of §56 (a relative time needs a reference event)
applied to frequency.

**Predictions (E59).** P1: on held-out speakers, relative coding beats absolute coding at the best configuration of each.
P2: the configuration chosen on held-out speakers transfers to the test set, with a gap much smaller than E51's
0.73 → 0.647. P3: relative coding raises test accuracy by ≥ 0.02 over E51's 0.647.

**Results (E59).** Held-out speakers 3 and 6 are far harder than the test's unseen speakers: absolute coding 0.359–0.379
across the grid. P1 holds at every configuration: relative coding 0.441–0.464 (≈ +0.09). Selected per coding on the
held-out speakers only and scored once: absolute (B = 70, O = 40) test 0.657, relative (B = 140, O = 40) test 0.675. P3
holds (+0.028 over E51's 0.647; +0.018 over absolute coding under the same protocol). P2 as stated is moot: the
held-out-speaker validation now errs on the pessimistic side (0.46 vs 0.675 on test), the safe direction. The gain on
the test set is smaller than on the held-out speakers, whose voices lie further from the training voices. A published
LSTM reaches ≈ 0.70; the counted event models, with no gradients, are now within ≈ 0.025 of it.
## 93. Inference activity follows the learned structure: pruning the candidate generator by credit

*Written 2026-09-27, before E93's full runs.*

**Problem (§85).** A chain unit at level k fires for every part within W₂ after a firing level-(k − 1) unit, so a network
that searched a basis of 10⁷–10¹⁰ candidates keeps paying for all their extensions: 44 / 155 / 454 events per example at
depth 3 / 4 / 5, although only a few routes carry weight.

**Rule.** After a warm-up, a unit is extended to the next level only if one of its children carries weight on some class
node (> 0.05), re-evaluated periodically (a sleep phase, the reuse filter of §72 applied to the candidate generator).
Usefulness flows down from the children: extending only units that are themselves credited would fail at bootstrap,
since a deep pattern's lower composites often lie outside the credited window.

**Why accuracy survives.** A valid route's units carry weight θ or more (they fire the class), so their parents stay
extended; extensions whose children carry no weight contribute nothing to any potential above threshold, so removing
them changes no decision except by removing noise units that could otherwise be credited. *Observed (one seed, depth 3,
4k examples):* 44 → 26 events per example, accuracy 0.965 → 0.989.

**Prediction (E93).** At depth 3 / 4 / 5 the pruned networks match the unpruned accuracy (E54: 0.98–1.00 / 0.999–1.000 /
0.94 at 4k) while events per example fall by ≥ 40%, and by more at greater depth, since the pruned fraction of the
extension tree grows with depth.

**Results (E93, 5 seeds, 40k examples; pruning after 3k).** Depth 3: accuracy identical seed for seed to the unpruned
network (0.999 / 1.000 / 1.000 / 0.981 / 0.999), events per example 44 → 26 (−42%). Depth 4: 1.000 / 0.995 / 1.000 /
0.999 / 1.000 (unpruned 0.999–1.000), events 147–168 → 38–39 (−75%), at the price of slower convergence (1.0 reached by
25–35k examples instead of 10–15k; 2.1k–2.5k updates instead of 1.9k–2.0k). The saving grows with depth, as predicted.
Depth 5: first run killed by a watchdog that counted page cache as memory (0.6 GB in use); rerun queued.
