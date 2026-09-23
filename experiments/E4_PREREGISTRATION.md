# E4 — Learning from counterfactual traces of cancelled events

Written 2026-09-23, before any E4 code was run.

## Claim

In an event-driven race, the output that *should* have won has usually been
cancelled and never fired. It is gone by the time a delayed teaching event arrives. A local learning rule that
only credits nodes that fired (STDP / reward-modulated STDP style) cannot
directly promote that node. If a cancelled node keeps a local record of how
close it came (its distance to threshold when it was inhibited), then the
teaching event can promote it, and the record also says which competitors are
threats worth suppressing.

## Setup

- K output nodes, non-leaky integrate-to-threshold, fed by M input channels.
- Each sample activates a noisy subset of its class prototype channels plus
  random distractor channels, each spiking once at a random time in the window.
- The first node to reach threshold fires after a short delay. Its inhibition
  cancels the others' pending fire events and freezes their distance-to-threshold
  snapshot Δ_k.
- A teaching event naming the target arrives after the window, with delay D.
- Presynaptic eligibility: x_i = exp(-(t_teach - t_spike_i) / τ).

Rules, all local and all event-triggered (applied at the teaching event only):

| rule | target (never fired) | competitors |
|---|---|---|
| `fired_only` | none | winner depressed on error |
| `fired_reward` | none | winner +/− by correctness (R-STDP-like) |
| `cf_winner` | potentiated on error | winner depressed |
| `cf_uniform` | potentiated on error | all non-targets depressed by 1/(K−1) (see amendment) |
| `cf_margin` | potentiated on error | non-target k depressed by exp(−Δ_k/σ), skipped below ε |

Reference (not local, not event-driven): online softmax regression on the bag of
active channels, trained by SGD on the same sample stream.

The learning rate (and σ for `cf_margin`) is tuned per rule on separate tuning
seeds at K=32. Evaluation uses different seeds.

## Predictions (falsifiable)

P1. At K=3, fired-only rules learn: suppressing wrong winners is almost as good
    as promoting the target.
P2. As K grows (3, 8, 32, 128), fired-only rules degrade sharply in test accuracy
    at a fixed sample budget. The cf rules stay close to the reference.
    *Falsified if* `fired_reward` stays within 5 points of `cf_winner` at K=128.
P3. `cf_margin` reaches accuracy within 2 points of `cf_uniform` or better, with
    far fewer synaptic plasticity updates at large K (fewer than 25% at K=128).
    `cf_uniform` may itself be destabilised at large K by depressing every
    hopeless alternative.
P4. (E4b) With no intervening activity, credit survives teaching delays up to
    about τ and then fades. When a new race happens before the teaching event
    arrives, the snapshot is overwritten and learning degrades.

## What would count against the thesis

If `fired_reward` matches the cf rules at large K, cancelled-event traces add
nothing over ordinary reward-modulated plasticity for this problem class.

## Amendment 1 (2026-09-23, after tuning, before the evaluation seeds were run)

The original `cf_uniform` depressed every non-target by the full learning rate.
On the tuning seeds it scored 0.00 at every setting. The cause is structural:
each error potentiates one node and depresses K−1, so all weights drift down
until the layer is silent. Keeping that version would make it a strawman. It now
depresses each non-target by 1/(K−1), balancing potentiation, and was retuned.
The original grid is kept in `results/e4/tuning_grid.json`.
