# Multi-arrival race reception: the missing boundary is computable

Read notes51,55,79 and92 first. This is an independent mechanism-gap investigation,
not a replacement for the other host's corrected first-time replay experiment.
The concrete limitation is one delivered value and one committed receiver per
native head/event. Several complementary arrivals might improve an interaction
before a deeper update. Current frozen probes and failed fits do not prove that
this is the observed failure, and ordinary window autograd is not a remedy by
itself. No new integrated fit is admitted by this note.

## A complete arrival law, not just the winner and first time

For independent raw clocks A_i~Exp(lambda_i), lambda_i=exp(s_i), common admission
time and a common increasing physical delay map g, define W=argmin A and T=min A.
Writing Lambda=sum lambda and pi_i=lambda_i/Lambda gives the joint construction

    W~Categorical(pi), E0~Exp(1), T=E0/Lambda,
    X_j~Exp(lambda_j) independently for j!=W,
    A_W=T, A_j=T+X_j.

Indeed its density is lambda_W exp(-Lambda*T) times the product of loser
lambda_j exp(-lambda_j X_j). This equals the original joint clock density after
the unit-Jacobian change of variables. Residual arrivals cannot be omitted when
several messages are received. Independence requires the stated exponential
common-start/common-map conditions; heterogeneous starts or delay maps need a
different conditional law. The established first-time replay correction stays
valid for winner-only reception and is not automatically a window gradient.

Consider a FIXED deadline g(T)+H after the first arrival. This is not popcorn:
it does not reset on subsequent messages. The winner is heard, and each loser
is heard iff X_j<=R(T,H). For native g(t)=.001+.010t/(1+t),

    R=H(1+T)^2 / [.010-H(1+T)]

when H<.010/(1+T); above that span all candidates are heard. Conditional on W,T,
loser membership is independent Bernoulli with q_j=1-exp(-lambda_j R). A heard
residual can be reparameterized exactly as -log(1-U*q_j)/lambda_j. Actual timed
values and actual receiver writes must enter the complete downstream loss;
averaging clocks or replacing a write by a delivery changes the objective.

For finite R, expected selected receivers are 1+sum_{j!=W}q_j. This exposes the
capacity/activity tradeoff directly. H=0 recovers exactly the winner-only law;
covering the remaining span activates every candidate. A wide window is not
free dormant capacity. Scores/discovery/scheduling must still be charged.

## Boundary flux gives faster/slower and width credit together

Let L be a differentiable complete timed outcome inside each history cell,
including real persistent writes and future queries. Fix W,E0 and reparameterize
losers by X_j=E_j/lambda_j. Away from a membership change ordinary derivatives
include clock, evolution, content and all downstream maps. They omit the jump
when X_j crosses R. Under continuous independent clocks, dominated derivatives
and integrable outcomes, the expected derivative adds for each loser

    lambda_j exp(-lambda_j R)
      * [dR + R ds_j]
      * E_other[L(heard j at the deadline)-L(unheard j)].

Here dR includes the common-first-time derivative dT=-T sum_i pi_i ds_i and
the width derivative; the ordinary part uses dX_j=-X_j ds_j. Equivalently the
coefficient is dq_j. The two legal approaches are:

1. Differentiate the exact membership-weighted, conditional-residual expectation:
   ordinary conditional inverse-CDF derivatives PLUS Bernoulli membership credit.
2. Differentiate unconditional residuals inside history cells PLUS the paired
   boundary term above. Winner choice adds sum_i dpi_i E[L|W=i] in both cases.

Do not combine conditional inverse-CDF derivatives with the full boundary term:
that double counts redistribution inside the truncated residual distribution.
Do not use Bernoulli membership credit alone with an unconditional residual
sampler and claim exactness. A baseline for sampled choice/membership credit
must be fixed before the corrected sample or independent of its action, as in92.

The boundary pair is evaluated at one common deadline; it changes actual j
delivery AND j's persistent write, with all other sampled arrivals kept fixed.
Its loss difference is stopped before multiplying dq_j. A frozen deterministic
message perturbation therefore changes timing, membership, writes and contents
through their respective chain-rule terms. Coupling is not an obstacle to this
decomposition. Quantized clocks/atoms, coincident boundaries, history-dependent
rates or nontransverse events require additional analysis, not this formula.

When the span is covered, membership derivatives are zero; conditional residual
content/time derivatives remain. At finite positive rates exp(-lambda R) kills
the diverging cutoff as the span is approached. Zero width has a one-sided
boundary derivative even though it emits only the winner; a softplus width
cannot nest exact zero with a finite unconstrained parameter. A finite bank
including zero can nest the forward model, but its categorical expected-risk
training and hard-argmax inference are different policies and must be disclosed.

## Exact numeric reference, resource limits and next admission

The new contracts use three independent candidates, a common clock, decaying
received messages, actual separate addressed memory writes and a later fixed
address read. Deterministic quadrature compares full conditional expectation
gradients with ordinary history-cell plus boundary plus winner credit for every
score, width, message coordinate and decay parameter. Finite differences provide
an independent check. Density, physical/raw cutoff, zero-width nesting, cap,
normalization and expected receiver activity are checked separately. This is a
small mechanistic reference, not the full native model or benchmark advantage.

Enumerating every membership history is exponential in pool size. The boundary
identity does not require that enumeration: sample factual other arrivals and
evaluate a heard/unheard pair per candidate, or sample one candidate with known
full-support propensity and inverse-propensity weight. Either method pays every
complete replay/write/suffix, has sampling variance, and is not an exact cheap
per-example loss. Joint interacting reception histories remain important even
when a first derivative has an additive flux representation. At many layers,
each conditional credit must be inserted into the actual history with correct
future randomness; a local illustrative contract is not full-sequence credit.

The earliest integrated test should be a declared one-site reception window,
with actual sparse receiver commits, physical transport to the real deadline,
unchanged keys/values and earlier mechanisms, and a winner-only nested control.
Keep arrival/source readiness, no future query access, pending-deadline handling
and RNG/recovery contracts. Include a matched amplitude/normalization control:
multi-message sums can change confidence without adding useful information.
Charge inference key discovery, selected value proposals/writes/transports,
timers, training boundary candidates/replays and Adam. Admit a small integrated
learning/accounting smoke only after these contracts pass, before any quality
pilot. The fixed raw DVS packet cadence is not natural-silence evidence; a
silence-reset popcorn extension still requires its own scheduler and law in79.
