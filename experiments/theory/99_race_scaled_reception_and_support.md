# Race-scaled reception separates support from a common speed shift

Read97/98 and the current handoff before using this construction. The completed
one-site1ms test failed without receiving any losing message in either trained
producer. Its fixed3ms mean gains were only.0011/.0023 NLL. We must distinguish
categorical concentration, absolute clock scale and useful alternatives before
another integrated fit. The other host owns corrected replay/shadow-lane quality
and AWS owns replay-critic calibration; this is an independent reception audit.

## What a fixed millisecond width couples accidentally

Adding b to all entering scores multiplies every rate by exp(b). With fixed
Exp(1) draws it scales every raw arrival A_i by exp(-b) without changing winners.
The bounded physical map g(t)=.001+.010t/(1+t) is nonlinear, so it can change
whether a loser lies within a fixed physical H of the first arrival. Thus a
fixed-width failure cannot by itself diagnose categorical certainty or absent
optional information. Time continues to compute; this identifies a reception
parameterization question, not an argument for removing physical timing.
The law concerns a common shift of ENTERING scores; shifting emitter biases
through a score clamp need not produce that common shift. Clamp exposure is
therefore reported separately rather than ignored.

Instead close the raw-clock window at (1+c)T, T=min A, c>=0. The physical
deadline is g((1+c)T), and a candidate is heard iff A_j<=(1+c)T. Under a common
speed shift both sides scale equally, so the heard set is exactly invariant
for each fixed draw. Winner and actual physical timing/content evolution still
change with scores. This adds a reception degree of freedom without cancelling
the computational clock or decoupling representation from time.

No sum of all rates is needed to implement the deadline: it uses the realized
first raw clock and a scalar multiplier. Access to raw clock or inversion of
the physical map is an implementation assumption; hardware cost is not proved
free. It is a fixed-first deadline, not a silence-reset popcorn rule. All key
discovery, heard values/writes, timers, transport and losing-credit work are paid.

## Exact capacity/activity and latency laws

For fixed entering scores, W and T are independent, pi_i=lambda_i/Lambda,
T~Exp(Lambda), and losing residual X_j|W,T~Exp(lambda_j). Here R=cT. Therefore

    P(j heard | W=w) = E_T[1-exp(-lambda_j cT)]
                    = c*pi_j/(1+c*pi_j), j!=w,
    E[receivers] = 1 + sum_w pi_w sum_{j!=w} c*pi_j/(1+c*pi_j).

This closed form integrates current race noise, not downstream utility or the
history-dependent distribution of future scores. It is exactly invariant to
common score shifts. At c=0 it nests the winner-only law; c->infinity receives
every candidate for positive finite rates. If categorical probabilities are
already concentrated, moderate c can still deliver almost only the winner.
For pool2, pi=(1-epsilon,epsilon), expected extra activity at fixed c is
epsilon[c+c/(1+c)]+O(epsilon^2). A scale-invariant window therefore does not
solve lack of support caused by categorical concentration.

The added physical waiting time is

    H_c(T)=.010*c*T / [(1+T)(1+(1+c)T)].

Its exact maximum is .010*(sqrt(1+c)-1)/(sqrt(1+c)+1), reached at
T=1/sqrt(1+c) for c>0. The final local delay g((1+c)T) remains below.011s,
the original native race's bound. Actual waits increase and need measurement;
bounded worst-case local latency does not make waiting or extra values free.
Source admission, joins and query readiness continue to use actual times.

## Credit remains coupled, including membership changes

The boundary identity in97 applies with R=cT: ordinary unconditional residual
history-cell derivatives plus dq_j times the actual heard/unheard downstream
loss difference, plus winner-choice credit. Here

    dq_j = exp(-lambda_j cT)*lambda_j*(T dc + c dT + cT ds_j),
    dT = -T sum_i pi_i ds_i.

For a common score shift, ds_j=db and dT=-T db, so dq_j=0 exactly while physical
deadline and evolved values can still change. This makes the relative-support
versus common-clock ownership explicit. Discarding the boundary term still
omits real changes in information and writes. The zero-width forward can nest
the original model; a trainable positive softplus cannot reach exact zero with
a finite parameter, and a hard categorical timeout bank has a different
training/inference policy. No complete native learner is installed here.

## Bounded frozen support audit, declared before observing results

Use the existing native256-label/four-pass seed6/7 fixed final producers and
their exact initial reservoirs. All84 race sites per21-event prefix,16
prespecified evenly spaced producer-unseen FIT indices linspace(256,983,16),
eight independent whole-history noise seeds314159+1009k. Common draw seed
across clips in a history matches established conditioning. There are no
optimizer steps, target-dependent decisions, DEV/test evaluations or new heads.

Record exact entering scores, raw arrival draws, winner, probabilities, entropy,
score clamp exposure and physical gaps. For fixed H1/3ms record actual and
conditional expected heard count. For relative c1/4 record actual and exact
entering-score expected count plus physical wait. Aggregate every depth/head,
each site and query/nonquery separately. Never infer utility from support.
Preserve full profiles for later prespecified information/credit diagnostics.

Contracts check rate-shift invariant membership/count, exact closed count law,
conditional-time quadrature, independent finite differences of count derivatives,
zero-width nesting and tight physical-wait bound. The collector must reproduce
the original logits, persistent state and RNG consumption and leave weights
bitwise equal. Measure wall/RSS and exact observed event/key/write counts. Audit
FLOPs/traffic/energy remain unmeasured rather than zero; prior core fits remain
charged. Profiles do not execute any hypothetical extra receive/write outcome.

There is no automatic training nomination gate: support alone cannot establish
useful message information. If a particular width/site has support, separately
preregister a small actual-write/transport/amplitude-controlled FIT intervention
before changing a learning queue. If support is concentrated away, keep the
failure beside98 and investigate emitter uncertainty/actual utility with the
already prioritized corrected replay evidence rather than retuning a width on
DEV. Neither branch proves or rules out the complete architecture's advantage.
