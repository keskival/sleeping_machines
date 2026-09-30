# Integrated sparse temporal language: computation, credit and scale

Derived 30 September 2026. This is the prioritized combined-mechanism candidate,
not a completed frontier benchmark. The preceding dense carrier remains a
distinct control. Preserve its learned-content evidence and the older structured
temporal, retrieval and phase advantages.

## 299. A content-bearing event chooses persistent receivers through time

An observed character supplies a learned embedding. It is mixed with the last
deep message through an input-dependent gate, so sequential context is available
even before a receiver has previously seen that character. At depth l a query
of this content meets keys of a small addressed pool. Each key combines a
learned prototype and a learned read of that receiver's persistent state.

Rates are exp(clamp(q dot k / sqrt(d) + clock_bias, -12,12)). Independent
exponential clocks choose one receiver with exactly the corresponding softmax
probabilities. The winner emits its input-dependent value, not a constant unit
vector. A monotone time map f(T)=0.001+0.010 T/(1+T) bounds latency without
changing which clock wins. Rates, candidate keys and emitted values remain
separate quantities. No explicit normalization sum/division computes choice
probabilities in this winner path; the emulator still computes rates and noise.

The current index is observed character identity and the graph has fixed depth.
This is a declared sparse addressing prior, not learned topology discovery or
arbitrary context attention. Race choices within pools are learned. State-key
reads cost work even when a candidate does not emit a value.

## 300. State only changes along the realized event path

For a selected receiver, its old state h and last arrival t_old are transported
to incoming time t before adding the new content drive:

    h_new = exp(-r (t-t_old) forget(x)) R(omega (t-t_old)) h + write(x) W_in x.

A normalized read W_out h_new+x is gated and added to incoming x. The selected
state and arrival are committed. Its message reaches the next depth at t+f(T).
Other receiver states are unchanged. Training computes losing proposed values
as counterfactuals, without committing their unrealized state histories.

With D*.011<.5<1, every selected chain finishes before its query and next input.
This bounded event schedule is causal and requires no empty ticks. It does not
prove arbitrary overlapping-queue scheduling. Float64 arrivals and integer
source positions retain sub-token timing at large indices; float32 payloads
remain separate. The six-depth candidate has 324 receivers for two options in
each of 27 character pools. Exactly six states update per character.

## 301. Route credit, arrival-time credit and value credit have distinct scope

For candidate values v_i and a noise-independent baseline b, the conserved
teacher of §296 supplies score-output credit

    J_i=lambda_i T (v_i-b) - 1[W=i] T sum_j lambda_j (v_j-b).

The implementation uses the pool value mean as b. Credit sums to zero for each
realization and has the soft-attention expected output Jacobian for fixed
candidates and upstream error. For a nonlinear loss and changing persistent
route histories it is an explicit local surrogate, not the exact gradient of
the whole expected network objective. Counterfactual reads are charged.

Values receive the realized winner's cotangent. Units need exploration to
learn their payloads; a winner-only value teacher is not a theorem of optimal
optionality. Additional counterfactual value teaching and learned topology
remain hypotheses to test if alternatives undertrain. Arrival-time credit has
an exact interior term, holding winner fixed:

    d f(T)/d score_W = -0.010 T/(1+T)^2.

This term changes downstream elapsed-time computation. Its final-arrival term
is absent under the bounded completed-message query, as in §291. Unlike the
earlier shared-payload clock, however, the final race selects different receivers
and values: its clock scores still receive counterfactual choice credit. A zero
interior arrival-time derivative therefore does not remove its semantic route.
Numerical
contracts check live query/key/value/retention gradients, causality, partition
invariance, training-forward/inference equality and selected-state counts.

## 302. Capacity, activity and learning work are separate scaling variables

There are 27 D C receiver units, D selected value/state updates and D C addressed
key scores per character. Training evaluates D C proposed values for route
credit. Doubling C doubles available units but not selected state updates;
it does increase scoring and counterfactual arithmetic. Dormant parameters
outside addressed pools have no new local update, although truncated history
can still deliver credit to previously active units. Adam and clipping are
charged to their actual gradient-bearing tensors in the representative trace.

Increasing payload width is a different scaling intervention: local maps cost
quadratically in d. The first ladder holds d=16 and D=6, varies data at C=2 and
tests C=4 at fixed small data. All runs begin from scratch with seed 6. The
character alphabet, cold development interval and frozen selection are fixed;
credit truncation is 16, differing from the earlier carrier's 64. Those models
therefore do not form a single-mechanism ablation or a matched scaling law.

Whole-stream selected updates, candidate visits and RNG draws are recorded.
Representative saved-parameter arithmetic includes forward, counterfactual
learning, backward, clipping and Adam; sparse address/optimizer activity varies
through the stream. This extrapolation is not a whole-run instruction or joule
certificate. The architecture's strongest prospective advantage is useful
quality from sparse activity, not a preset speedup factor. Data/quality gates
prevent blind promotion of a failing design. The full architecture has priority;
carrier-only and hybrid retrieval experiments are deferred controls.
