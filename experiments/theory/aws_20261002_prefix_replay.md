# Exact prefix reuse for actual-write credit

Failure addressed: the promising actual-write pilot pays an entire second
forward to change one event9 route. The first nine events are identical under
both interventions. Snapshot addressed state, context and RNG immediately
before event9. Replay only event9 through the query from detached cloned state.
Keep the factual graph intact: no producer derivatives are truncated there.
The alternate loss is already evaluated under no_grad in the original driver.
Therefore identical conditional losses give identical corrected score credit,
factual content/time gradients and every parameter update, with lower replay
work. This is implementation reuse, not a changed estimator or architecture.

For 21-event clips, replayed events fall from42 to33 per clip/window:21.43%
less simulated forward event activity. Whole-fit savings are smaller because
factual backward/Adam and replay setup remain. Charge stacking, tensor copies,
state/RNG snapshots, losses and all suffix candidate computation. Snapshot
storage grows with live addressed state; measure RSS rather than assume free.
No inference, quality, traffic or energy advantage follows from this identity.

Contracts: every factual/alternate logit, loss, route, memory/arrival/context,
all parameter gradients and next Adam updates versus full replay, fixed RNG,
multiple rotating sites and float32/64. Actual interruption recovery and full
operation coverage precede integrated fitting. Initial tests are engineering
contracts; local host owns quality replication. AWS does not duplicate it.
