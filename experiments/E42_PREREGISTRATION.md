# E42 preregistration: learning when to transact, on the BTCUSDT trade stream

*Written 2026-09-27, before any E42 learner was run on any data. Design and tuning use only the 7 pilot days
(2026-08-25 … 08-31); the 21 confirmatory days (2026-09-01 … 09-21) are evaluated once, after this file and the code are
committed. Disclosure: the confirmatory days were used before, by E17, for a different question (10-second direction
prediction); E17's result (direction ≈ 59% predictable on moves ≥ 1 bp, every learner losing money after costs) is known
and motivates the cautious hypotheses below.*

## Why E17 was the wrong question

E17 asked "will the price be higher in 10 s?" every 10 s and charged costs afterwards. A trader holds a position and
decides when changing it pays for its cost. E42 poses that problem.

## Task

- **Position** x ∈ {−1, 0, +1} (short, flat, long), one unit, held between decisions.
- **Decision points**: asynchronous, at each trade whose price differs by at least 0.5 bp from the price at the previous
  decision point (price events), capped at one per second. No decisions on a clock.
- **Profit**: between decision points k and k+1, x_k · 10⁴ · ln(P_{k+1}/P_k) bp, minus c · |x_k − x_{k−1}| bp at each change.
  Cost levels: c = 2 bp (maker-level) and c = 10 bp (taker-level), analysed separately; the learner is trained per cost.
- **Teacher (costs built in)**: the profit-maximizing position sequence in hindsight over a lookahead of L = 10 minutes,
  computed by dynamic programming over {−1, 0, +1} with the cost c. The label at decision k is that sequence's position at
  k. **Causality**: the label for k is released for learning only after t_k + L; the learner acts at k using only data up
  to t_k.
- **Inputs to the event learner**: the trade stream in the 10 s before the decision as spikes on 24 channels (aggressor side
  × size quartile × tick direction; first spike per channel, E17's encoding) plus the current position as three channels.

## Learners

- **event** (E35-type detectors, nothing given: learned hold, trigger and veto channels, hold durations; errors-only):
  three detectors race for target position −1 / 0 / +1; if none fires, the position is kept.
- **B-flat**: always flat (profit 0).
- **B-hold**: long throughout (buy and hold).
- **B-mom**: momentum with hysteresis: go long (short) when the 60 s return exceeds +h (below −h), flat when |return| < h/2;
  h tuned on the pilot days per cost level.
- **B-logit**: online logistic regression on E17's window features plus position, trained on the same teacher labels,
  acting through the same rule (argmax target position).

## Metrics (confirmatory days, prequential, one pass)

- Net profit per day in bp after costs, with a 95% day-block bootstrap interval; turnover (position changes per day);
  fraction of time in the market.

## Hypotheses and decision rules (per cost level)

- **H1 (profitable)**: event learner's net profit > 0 with the interval excluding 0. *Expected: not met at c = 10 bp;
  uncertain at c = 2 bp.*
- **H2 (knows when not to trade)**: at c = 10 bp, event learner's net loss is no worse than 0.5 bp/day below B-flat
  (it learns to stay out), and its turnover is below B-mom's.
- **H3 (better than the baselines)**: event learner − best of {B-mom, B-logit} > 0, interval excluding 0.
- **H4 (cost of decisions)**: event learner's operations per decision (synaptic events) are reported against B-logit's
  multiply-adds; not a pass/fail criterion.

A negative H1 with a positive H2 is a meaningful result: a learner that correctly decides that transacting does not pay.
