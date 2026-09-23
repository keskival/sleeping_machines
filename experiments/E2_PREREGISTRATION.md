# E2 — Integrate-and-race decisions compute adaptively

Written 2026-09-23, before any E2 code was run.

## Claim

A race of evidence accumulators, where the first to reach threshold fires and
cancels the rest, is a complete decision procedure that decides easy inputs
early and hard inputs late. Nothing schedules its stopping time: stopping is the
first fire event, and cancelling the pending input events is where the work is
saved.

## Setup

- K classes, G input channels per class, each channel an independent Poisson
  spike source. The target class's channels fire at r0·(1+c). All other
  channels fire at r0·(1 − c/(K−1)), so the total input rate carries no
  information.
- Coherence c is drawn per trial from {0.05, 0.1, 0.2, 0.4, 0.8}, and is unknown to
  every decoder.
- Every input channel generates its next spike lazily as a pending event. Once the
  decision is made, those pending events are cancelled and never processed.

Decoders:

| decoder | rule |
|---|---|
| `race` | node k: +1 per spike in group k, −1/(K−1) per spike elsewhere; first to θ fires |
| `fixed_time` | count spikes per group up to a fixed T, then argmax (a clocked, non-adaptive decoder) |
| `msprt` | multi-hypothesis sequential probability ratio test on the exact Poisson likelihoods, marginalised over c; stops when the posterior max exceeds 1−α |

Sweep θ, T and α to trace accuracy against mean decision time and against
input events processed.

## Predictions

P1. The race's accuracy/time frontier lies above `fixed_time` at every mean
    decision time. *Falsified if* any `fixed_time` point dominates the race curve.
P2. The race's mean decision time falls monotonically as coherence rises, so
    work adapts to difficulty. `fixed_time` spends the same time on every input by construction.
P3. The race comes within a modest distance of `msprt`, the Bayes-optimal stopping
    reference: less than 20% more time at equal accuracy for accuracy ≥ 0.8.
