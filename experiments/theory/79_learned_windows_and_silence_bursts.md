# Learned integration windows and silence-terminated bursts

The user's question identifies a mechanism gap: the current native layer
selects one value per head/event. Receiving several useful arrivals can expose
relations before a deeper state update. Theory51's compact smooth window is
implemented and numerically tested as a primitive, not fitted as a full native
layer. The completed fixed-count repeated-arrival language fit is related but
different: its best four-arrival fit improves .007978bpc for4.57% extra estimated
fitting work, below the .02 promotion gate. E14's `crl_window` instead bounds
near-miss credit; it is not learned message-window integration. These results
must not be presented as tests of the proposed silence rule.

## Two distinct temporal operators

A causal support window integrates arrivals with age in(0,H). The polynomial
kernel in theory51 vanishes with its first derivative at both boundaries,
allowing exact ordinary membership gradients for a fixed finite spike set.
Window duration is bounded even in a continuously active stream. Its five
moments need an expiry queue retaining live payloads; charge both that storage
and discovery/projection/scheduling work.

The popcorn rule instead resets a deadline t_last+H after every incoming
message. Close a burst only when no new message arrives before that deadline.
Inputs exactly at the deadline win the scheduling tie and extend the burst.
Accumulate a persistent vector by event maps and analytic silent evolution;
emit only at the completed deadline. An unfinished burst is not flushed at
EOF or at an earlier query. Silence is observed until a real clock deadline;
the later input must not retroactively cause an earlier answer.

`silence_burst.py` implements the additive/decaying reference. It updates at
arrivals and deadlines, with one state vector/active flag/deadline per addressed
receiver; no idle ticks or all-prefix buffer. A nonlinear local message/state
map can enrich it, with its own derivative and inference-work contracts. A full
asynchronous layered implementation needs a generation-aware priority queue:
new inputs cancel stale timers, and emitted timestamps feed downstream layers.
All value updates, actual emissions, scheduler and losing-option work are paid.
This primitive does not yet implement that whole sparse addressed architecture.

## Exact history-cell gradients do not teach every merge/split

For a fixed burst partition and a fixed finite observation horizon, decay,
message values, input times and emission times are differentiable. Deadline
emission has dt_emit/dH=1. Composition through two layers admits ordinary
gradients as long as no ordering or burst membership changes. At a consecutive
input gap g=H the histories change: one merged emission versus two split
emissions, generally with different content, times and downstream state.

That is a nonzero loss jump. Deterministic history-cell autograd omits the
distributional boundary term. Increasing H through the gap changes expected
loss by a merge-versus-split utility, weighted by gap density (and dH/dtheta).
Exactly at a horizon, an emission appearing/disappearing is another boundary.
Differentiating only t_emit or treating hard membership as a detached mask does
not solve either credit problem. Smooth zero-boundary windows eliminate their
fixed-spike membership jump; upstream spike birth is still separate.

An implementable exact finite alternative is a fixed bank of positive timeout
options H_k and a learned categorical policy pi_k=softmax(s)_k. Conditional
on the entering state, evaluate each complete causal timed outcome L_k:

    R=sum_k pi_k L_k,
    dR/ds_k=pi_k(L_k-R).

This supplies unrealized merge/split/horizon credit without a derivative
through a discrete boundary. Inference chooses one hard timeout; training
enumeration is paid. For two global layer options, evaluate every pair; for
per-burst policies the number of histories can grow exponentially. Candidate
sampling must retain support and account for propensities; no cheap exact
whole-sequence claim follows. Callback outcomes include their actual emission
times, not a soft mean time or a label-dependent temporal mask.

The numerical contracts test scheduled causality/ties, no premature EOF flush,
fixed-partition finite differences through two layers, and exact two-layer
finite-option risk gradients. No reference contract or synthetic capability
fit establishes natural-stream advantage.

## Silence grouping has a sharp resource and starvation law

For stationary Poisson arrivals at rate lambda, a gap exceeds H with
probability q=exp(-lambda H). A burst has geometric message count with mean
exp(lambda H). From its first message through its last-message timeout,

    E[burst duration]=(exp(lambda H)-1)/lambda,
    long-run emission rate=lambda exp(-lambda H).

To derive duration, E[N-1]=exp(lambda H)-1 and a short exponential gap has
mean1/lambda-H/(exp(lambda H)-1); add the final H timeout. The expectation
needs stationary Poisson input; it is not a model of the measured camera data.
An uninterrupted finite-gap stream can postpone emission indefinitely. A
learned positive H therefore needs a measured latency/activity objective or
a declared maximum-burst deadline. Such a cap changes the pure popcorn rule
and must remain explicit. Constant accumulator storage does not remove the
cost of consuming all messages, nor guarantee prediction sufficiency.

## Real-stream integration admission

The existing DVS adapter emits twenty equally spaced50ms packets per gesture.
A timeout acting only on that raw schedule mostly chooses one burst versus
one burst per packet. It cannot establish learning useful natural silence.
Native internal race arrivals can vary, but waiting beyond the1s query would
change the permitted latency boundary and must be charged to all controls.

Before fitting a popcorn-native extension: define its receiver/message path,
whether it accumulates realized races or several candidate arrivals, scheduler,
credit to optional arrivals and timeout alternatives, inference/query deadlines,
and a causal irregular packet adapter retaining useful real timing. Compare
against the original winner core, the smooth-window core and strong conventional
controls with identical observed information. Do not displace integrated core
experiments with a carrier-only or isolated timeout fit. The user's mechanism
is worth a controlled test; it is not ruled out by the current restricted fits.
