# E7 — A race network that lives in a causal stream

Written 2026-09-25, before any E7 pilot was run. The numeric thresholds in
"Decision rules" are filled in from the pilots (validation split only) and dated
**before** any confirmatory run. Everything else below is fixed now.

## Why

E6 trains the way clocked hardware likes: minibatches of 32, ten IID epochs,
a global learning-rate schedule, homeostasis from batch means, a separate test
phase, a label on every sample. None of that is needed by an event-driven learner;
it is inherited. E7 drops it and asks what the race can do when it lives in time:
one stream, no epochs, every frame predicted before it can be learned from, labels
scarce and sometimes late, and slow state carried from frame to frame.

We are not trying to be biologically faithful. The rule is narrower: **where we
deviate from biology only because of synchronous-hardware habits, we do not carry
the deviation over.**

## Audit: deliberate choices and inherited habits

| choice | origin | in E7 |
|---|---|---|
| minibatch updates | GPU throughput | gone: one update per frame (Stage 0 checks it changes nothing) |
| epochs, IID reshuffling | batched SGD | gone: one pass over a stream |
| global LR decay | SGD convergence | gone; per-synapse consolidation instead |
| homeostasis from batch means | vectorised statistic | per node, every frame, with or without a label |
| dense feedback `s @ B` | cheap matmul | counted as events to eligible hidden nodes only |
| train phase, then test phase | benchmark protocol | primary measure is prequential; held-out accuracy kept for comparability |
| a label for every sample | benchmark convention | labels are scarce (10%), late (2 frames), or asked for by the learner |
| frames with a global t = 0 | datasets are frames | kept as **episode resets** (a legitimate boundary, like knowing one moved one's eyes); slow state crosses them |
| dense image → latency code | MNIST is a frame | kept as a **benchmark**, not a sensor model; the encoder's cost is not counted and the report says so |
| one spike per input, non-leaky neurons | exact closed-form solver | deliberate; kept |

## The learner

The E6 network (input → 100 groups of 10 hidden nodes, 3 winners each → 10-way
output race, ramp synapses, collapsing bound), with the E6 round-3 settings. The fast
state (the race) restarts every frame, so the closed-form solver and its
equivalence test still hold. With every mechanism below switched off, the stream
learner is bit-identical to E6 at one frame per update (`tests/test_e7_stream.py`).

Slow state carried across frames:

- **prior**: the last winner's threshold is lowered by a fraction β, decaying
  with τ = 3 frames; cleared at an episode reset.
- **tags**: a label that arrives d frames late teaches the synapses of *its own*
  frame (their stored charge), scaled by exp(−d/5). The control, "no tags",
  applies the late label to whatever is active now.
- **carry**: a label seen in an episode teaches the later views of that episode.
- **continuity**: with no label, hidden nodes that won on the previous view and
  nearly won now are pulled toward winning (no label involved).
- **consolidation**: effective rate η / (1 + c·Σ|Δw|) per synapse.
- **asked labels**: the learner requests a label when its own race is uncertain
  (deadline-forced decision, or a rival within the margin), under a token bucket
  with the same average budget as random labelling.

## The streams (latency-coded MNIST, shifted views)

- `episodes`: 5 views of one image (random shifts up to 2 px), then another image.
- `shuffled`: the same frames and labels in random order, resets at the same
  places. **This is the control for every claim about time.**
- `iid`: one view per image.
- `blocked`: classes in pairs {0,1}, {2,3}, … one pair after another.

Baselines on identical streams: an MLP of the same hidden size trained by SGD one
frame at a time (with the same carry and asked-label rules where they apply), and
the same MLP with a 1000-example reservoir replay buffer.

## Hypotheses

- **H0 (Stage 0).** One frame per update learns as well as minibatches of 32:
  held-out accuracy after one pass, 3 seeds each, differs by less than the
  seed spread. If not, batching was doing real work and we say so.
- **H1 (carried prior saves work).** On `episodes`, the prior cuts input events
  consumed before a decision on later views, without costing accuracy on first views.
- **H2 (time is used, not just more updates).** At 10% labels, carry and/or
  continuity raise prequential accuracy on `episodes`, and the gain is **absent or
  much smaller on `shuffled`**.
- **H3 (asking beats random).** At the same label budget, asked labels give
  higher prequential accuracy than random ones.
- **H4 (tags make late labels usable).** With labels 2 frames late, tags beat no-tags.
- **H5 (forgetting).** On `blocked`, the race forgets less than the MLP trained
  by SGD; consolidation reduces forgetting further. Replay is reported alongside,
  and we expect it may beat us.

## Measures

Primary: prequential accuracy (each frame predicted before learning), its curve
over the stream, and input events consumed before the decision (first view vs
later views). Secondary: held-out accuracy at the end; forgetting (per-task peak
minus final, tasks 1–4); labels used; training work per frame including feedback
and homeostasis events; hidden-code overlap within vs across episodes (a collapse
detector for continuity).

## Protocol

1. Pilots: validation split (last 10k training images held out), 10k frames,
   seed 0. Used to choose β, continuity rate and consolidation strength, and
   to set the thresholds below.
2. Confirmatory: test set, 30k frames, **5 seeds**, only the settings chosen in (1),
   plus every control. Reported as mean ± 95% CI over seeds; a difference counts
   only if its CI excludes zero.
3. Everything that fails is reported, with what we learned from it.

## Decision rules (filled in from pilots, before confirmatory runs)

_To be written after the pilots. Each hypothesis gets a threshold and a
direction; nothing is changed after the confirmatory runs start._
