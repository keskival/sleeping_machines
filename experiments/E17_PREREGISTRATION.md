# E17: a continually learning race network on a live market event stream

Written 2026-09-26, before any market data was downloaded or inspected. Settings marked
**pilot** are chosen on the pilot days only and dated before any confirmatory day is scored.
Everything else is fixed now.

## Question

Can an asynchronous, continually learning race network predict the short-term direction of a
market price from the raw trade stream, and does it do so

1. at least as accurately as simple, strong baselines that decide at a fixed clock;
2. **earlier** (its own weaving decides when to commit, THEORY §21.9, §38);
3. **better when it keeps learning** than when frozen, because market dynamics drift?

This is a study of the model class on real, non-stationary event data. It is **not** a trading
system. No live trading and no real money at any stage. A profit proxy is reported as a
diagnostic only.

## Data

- **Binance spot BTCUSDT aggregated trades** (public daily files, data.binance.vision), with
  millisecond (or microsecond) timestamps, price, quantity and aggressor side.
- **This is a crypto market, not a stock market.** Freely available stock tick data with raw
  timestamps was not found. The FI-2010 stock benchmark contains pre-processed snapshots without
  timestamps, so it cannot test asynchrony. Crypto trades run around the clock and are a genuine
  asynchronous event stream with the same microstructure questions.
- **Days:** 2026-08-25 to 2026-09-21 (28 days). **Pilot:** 2026-08-25 to 2026-08-31 (7 days).
  **Confirmatory:** 2026-09-01 to 2026-09-21 (21 days). The learner runs through all days in
  order as one stream, and only confirmatory days are scored.

## Task and causality

- **Decision episodes** start every W = 10 s on a fixed grid, t₀. At t₀ the race restarts (an
  episode reset, as in E7), with reference price p₀ = the last trade price before t₀.
- **Input:** each trade in [t₀, t₀ + W) is a spike at time (t − t₀)/W · H on a channel given by
  aggressor side (2) × size quartile (4; quartile edges from pilot days) × tick direction versus the
  previous trade (up, same, down: 3) × occurrence rank within the episode (the first 8
  occurrences of each type). That gives 192 input channels, one spike each at most.
- **The race decides when it commits.** The first output crossing (up or down) at time t_c ≤ t₀ + W
  is the decision. If no output crosses by the horizon, the episode **abstains** (no position).
- **Label:** the sign of p(t_c + τ) − p(t_c) with τ = 10 s. Moves smaller than 1 basis point are
  **flat**: excluded from direction accuracy and not taught.
- **When to trade (added 2026-09-26, still before any data was inspected).** A second race variant
  has three outputs: up, down and **hold**. Hold is taught when |p(t_c + τ) − p(t_c)| is below the
  trading cost c = 2 bp (a move too small to pay for), and up or down otherwise. The network thus
  learns *whether* to trade, not only which way, and *when* (its crossing time). This is THEORY
  §38's rule, "commit when the commitment cost is below a price", with the price set by the
  transaction cost. A hold decision is scored as no position.
- **No look-ahead, by construction:** predictions for a chunk of 32 episodes use weights updated
  only with labels revealed before that chunk started. A label is revealed at t_c + τ ≤ t₀ + 20 s,
  so chunk j is taught after chunk j + 1 has been predicted.

## Learners (same stream, same labels, same causality)

- **Race** (continual): the E14 network (1 hidden layer of 200 nodes in groups of 10, k = 3,
  crl_fa, output conservation), two output classes, learning online throughout.
- **Race, frozen:** the same network, learning stopped at the end of the pilot days.
- **Race-hold** (continual): the three-output variant above.
- **B1, online logistic regression** on window features (buy minus sell volume, trade count,
  signed trade count, return over the window, last aggressor side), SGD, deciding at the end of the
  window (t₀ + W). This is a synchronous, full-information decider.
- **B0, momentum and reversal:** predict the sign of the window's return (or its opposite); which
  one is fixed on the pilot days.

Pilot settings (race depth 1 or 2, learning rate, output threshold) are chosen on pilot days only.

## Measures (confirmatory days, prequential)

- **Direction accuracy** on non-flat labels among decided episodes, with a 95% interval from a
  day-block bootstrap.
- **Coverage:** the fraction of episodes decided (not abstained).
- **Decision time:** mean t_c − t₀ (the baselines use W by construction).
- **Profit proxy per decision:** sign × return over τ, minus a cost of 2 bp (optimistic, maker-like)
  and of 10 bp (taker fee). Reported, never interpreted as tradable.
- **Adaptation:** continual minus frozen accuracy, per day.
- **Trade selection (race-hold):** the profit proxy per episode (a hold earns 0), the fraction of
  episodes traded, and the accuracy on traded episodes, compared with the two-output race and with
  B1 restricted to its most confident episodes at the same trade fraction.

## Decision rules (fixed now)

- **The race is competitive** if its accuracy is within 1 point of B1 (or better) while deciding at
  least 20% earlier on average.
- **The race is better** if its accuracy exceeds B1's by at least 1 point, with the day-block
  interval excluding 0.
- **Continual learning helps** if continual minus frozen is at least 1 point, with the interval
  excluding 0.
- **Learned trade selection helps** if race-hold's mean profit proxy per episode at 2 bp exceeds
  the two-output race's, and B1's at the same trade fraction, with the day-block interval
  excluding 0.
- Otherwise the result is reported as negative. Chance is 50%. A result near 50% for every learner
  is the expected null for 10 s crypto direction, and is reported as such.

## Known limits

- Aggregated trades carry no order-book quotes, so the spread and queue position are unknown, and
  the cost model is crude.
- One asset, 28 days, one direction task. Anything positive would need replication across assets
  and periods before being believed.
- Resource rule for this host: every run goes through the one-job queue with the memory watchdog.
  Days are processed one at a time.
