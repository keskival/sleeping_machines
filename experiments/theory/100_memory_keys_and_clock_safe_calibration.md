# Memory-conditioned route confidence and clock-safe calibration

The all-site99 profile finds trained top-route>=.99 in35–44% of current races,
with almost uniform first-event routing and much sharper later routing. This
suggests a state-dependent confidence source, but does not establish poor route
utility, an optimizer regression or a need for uniform load balancing.

The exact native entering score is

    r_i = q^T key_i/sqrt(P) + clock_bias_i
          + q^T key_read_i(memory_i)/sqrt(P),
    s_i = clamp(r_i,-12,12).

The first two terms are static at fixed entering q; the last is a persistent-
memory read. A losing receiver may have little memory while a repeatedly
selected receiver has strong memory, so memory-conditioned routing can
reinforce a selected address. It can also be correct evidence-based retrieval.
High confidence does not distinguish those possibilities. Separately measure
static/dynamic gaps, memory norms, unclamped scores and hypothetical component-
only concentration before changing normalization. Component-only scores with
q held fixed are a local algebraic counterfactual, not an actual prefix rerun.

The bounded collector adds read-only hooks on existing query/key-read operators;
it does not recompute or change the actual forward. Use the same two fixed-
pass4 producers and initial reservoirs,16 prespecified producer-unseen FIT
prefixes, one fixed noise history314159. Reconstruct every clamped score within
float32 tolerance, and preserve exact final logits, every persistent tensor,
every weight and end-of-prefix RNG state. Save all5,376 race decompositions
and layer/head/first-event/later-event summaries. No target loss, head fitting,
DEV/test evaluation or training admission. This narrower one-history audit is
not an independent statistical confirmation of99.

An additional explicit RNG contract compares99's original arrival collector
against the native baseline, including end-of-prefix generator state, logits
and persistent state. That supplements its earlier output/state/source-noise
checks without silently revising the completed result. All hooks and profile
reductions are paid; full FLOPs/traffic/energy remain unknown, not zero.

## Analogous router stabilization is informative, not a drop-in objective

[ST-MoE (Zoph et al.,2022), sections3.1/3.4](https://arxiv.org/pdf/2202.08906)
uses router z-loss, the mean squared logsumexp of router logits, to improve
training stability and address roundoff around exponentiation. Its numerical
findings motivate examining router magnitude and clipping. They do not prove
that its auxiliary loss improves this temporal substrate.

Here logsumexp(s)=log Lambda is a COMPUTATIONAL CLOCK COORDINATE. Subtracting
it from all scores preserves categorical routing but forces total raw rate1:
it changes the first-time law from Exp(Lambda) to Exp(1). Likewise penalizing
its magnitude is a real clock-speed regularizer, not merely removing a softmax
gauge. It can destroy useful delay/evidence computation if copied blindly.

One lawful analytical decomposition is s_i=b+u_i with sum exp(u_i)=1;
b=log Lambda and u_i=log pi_i. Relative race-window support in99 depends on
pi and preserves common-clock influence on physical timing. Any future key
calibration must state how b, relative routing and evolving values are retained,
which Jacobians it changes and its additional fitting/inference cost.
More degrees of freedom can improve local conditioning, as90/91 show, but the
failed unchanged second-seed offset result remains beside that theory.

If dynamic keys dominate confidence, the next decision still requires actual
full-write counterfactual utility or a target-independent, preregistered
integrated intervention with matched content/waiting/work. If static keys
dominate, memory normalization is a poorly motivated change. No choice here
admits another guessed regularization, entropy, key normalization or long fit;
the other host's corrected replay/shadow quality comparison remains prioritized.
