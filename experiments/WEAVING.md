# How weaving progresses through the network and time

An intuitive description of the weaving operator (THEORY §21.9–21.10) as the code runs it
(`e14_depth.py`, `e6_hidden.py`: hidden groups of 10 ramp neurons, k = 3 winners per group,
an output race among 10 class neurons, a horizon H).

**In one sentence:** each group of neurons holds an open set of possible futures, and every
spike irreversibly settles part of it. The settled region spreads through the network as a
front, from the input towards the output, until the output race settles the answer.

## 1. Before anything happens: an open pool

At t = 0 nothing is decided. Every hidden neuron has a *projected* crossing time: when it would
reach threshold if no further input arrived. With no input yet, all of these are "never". The
*pool* is the set of every way the upcoming races could turn out.

## 2. Input arrives, and forecasts only move earlier

The image arrives as a latency code: bright pixels spike early, dim ones late. Each arriving
spike starts a ramp in every neuron it connects to, so each neuron's projected crossing time
jumps earlier. With excitatory weights a forecast can only move earlier, never later (THEORY
§31.1, §34). You can picture each group of 10 as 10 runners whose estimated finish times keep
improving as evidence comes in.

## 3. The first weave: a crossing becomes history

When a neuron actually crosses threshold, that event is **woven**. Its spike time is now fixed
history, and nothing arriving later can change it. The spike goes out immediately to the next
layer. Only inputs that arrived before the crossing belong to its history (the causal rule,
§21.9). A ramp's input starts from zero, so an input arriving just before or just after the
crossing makes no jump: this boundary is continuous.

## 4. The group closes: winners fixed, losers frozen as near misses

After the k-th (third) crossing in a group, the race is over:

- **Winners** keep their spikes, each at its own crossing time.
- **The other seven are cancelled** at that instant (`freeze`) and will not spike on this input.
- **Each loser's residue** at cancellation, (θ − v)/θ (how far it was from threshold), is frozen.

The frozen residues are the near misses: the only trace of the futures that weaving cut off, and
what learning reads. The gap between the k-th winner and the closest loser (D_k) measures how
contested the race was. Decided by noise alone, it averages σ/k (§28). It is also exactly the
robustness margin of that race to timing jitter (§34.4).

If fewer than k neurons cross before the horizon, the group closes at H.

## 5. The front moves up a layer

Layer 2 faces the same situation one step removed. Its "future inputs" are not pixels but
layer 1's spikes, many not yet woven. So layer 2 forecasts on top of a partly settled layer 1:
a pool conditioned on another pool (§21.10). Each group closes on its own third crossing, and
the settled region grows as a **diagonal front** through layers and time:

```
 time →      0 ........................................ H
 input   ████████████████████ (pixels keep arriving)
 layer 1   ░░▓▓▓▓███████████          groups close one by one
 layer 2       ░░░▓▓▓▓█████████
 layer 3            ░░░▓▓▓▓███████
 output                  ░░░░▓▓█  ← first output crossing = decision
         ░ open futures   ▓ partly woven (some groups closed)   █ all woven
```

At any moment each layer has three parts: woven history (closed groups), live races (groups
still running, forecasts improving), and untouched futures. The front is not synchronised:
nothing waits for a whole layer, and each group closes when its own race ends.

## 6. The decision closes everything

The output layer runs one more race between the class neurons. **The first output crossing is the
decision.** That single weave closes the pool of the whole network for this input, and whatever is
still open is dropped.

If no output neuron crosses by the horizon H, the network is forced to decide: it picks the output
with the highest potential (an "urgent" decision). The fixed horizon is the only point where an
external clock enters, and the only source of non-conserved credit (§30.1).

## 7. What learning sees afterwards

The backward pass walks the woven record from the output down:

- **Along actual spikes:** credit follows the woven path, what happened.
- **At each closed group:** credit also flows to the frozen near misses, weighted by e^{−Δ/σ},
  what nearly happened. This weight is the derivative of the soft minimum (§21.6). At depth
  this channel is the only one that does not fade (§31.3).
- **The shadow neuron (§20):** its losers keep integrating after cancellation in a second
  compartment, recording what *would* have happened had weaving not cut them off, including
  evidence that arrived after the collapse.

## The one-line intuition

A building of rooms, each with runners racing on stopwatches. Each room stops its race when three
runners finish and writes down how close the fourth came. The finished results fire the starting
guns of the rooms upstairs. The building's answer is whichever top-floor runner finishes first.
Learning rereads the finish sheets, including the notes saying "fourth place was only this far
behind".

## Properties established so far

| Property | Where |
|---|---|
| Weaving is a stopping time: a collapse uses only inputs up to that time | THEORY §21.9 |
| The settled region spreads as a causal front; each layer's pool is conditioned on the one below | §21.10 |
| With excitatory weights, crossing times are topical (monotone, shift-equivariant) functions of input times; the front moves by at most the input jitter, except at race boundaries | §34.3 |
| A decision cannot change under jitter smaller than ε* = min(½ output margin, ½ race gaps, horizon distances) | §34.4 |
| The horizon is the only clock; timing credit sums exactly to the horizon's credit | §30.1 |
| The near-miss weight is the derivative of the dequantized (soft) minimum | §21.6 |
| Committing to a branch costs σ × its surprisal (σ × entropy on average): weaving pays temperature × information | §35.1 |
| Near-miss credit is the gradient of that commitment cost | §35.2 |
| The soft value is a submartingale: new evidence is its martingale part, commitment its increasing compensator | §35.3 |
| With excitatory weights the future is bracketed, so a race (and the decision) can be woven early with zero rollback | §35.4 |

These are properties of the framing assembled from known mathematics (THEORY §21.8, §32);
results specific to the weaving operator itself start in THEORY §35.
