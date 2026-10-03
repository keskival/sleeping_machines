# Trained write fidelity and calibration against the unexplained residual

3 October 2026, 12:55 UTC. Read140/142/143 and §§413–414 first.
Only standard-library scalar/source checks executed in this continuation.
Actual trained native forward/backward contracts remain pending.

## The decision this should resolve

Preserve the positive credited-language results: p32/U2 improves2.507→2.370,
p32/U4 improves to2.343 at unchanged eight selected writes, and p64/U2 reaches
2.184(T128)/2.183(T256), close to the saved one-pass LSTM2.171. Their completed
quality/work and single-seed scope remain in the appendix. More stored capacity
raises fitting work; winner-only inference has separate pending trained-weight
admission. None is evidence that temporal/deep learning is mathematically
impossible. None identifies a single cause of every historical depth failure.

`linear_rw` diverged; written-only `linear_rwn` was stable but worse at U2.
Do not guess another large write-credit fit or declare clipping the cause
without measuring the missing utility. This continuation prepares a small
read-only factorial audit on the **actual completed p64/D4/H2/U2 checkpoint**.
The same hard temporal races, persistent addressed memory, key/value separation,
content maps and losing-value learning remain intact. There is no architecture,
loss, optimizer or active source change and no new fit proposed here.

## Legal route alternatives versus diagnostic hybrid branches

At a declared node, the entering prefix fixes candidate messages, proposals,
scores and the race first time. Compare four forward suffixes:

- factual delivery and factual commit: Q00;
- alternative delivery, factual commit: Q10;
- factual delivery, alternative commit: Q01;
- alternative delivery and alternative commit: Q11.

Q00/Q11 are the two legal categorical choices at fixed first time. Q10/Q01
are diagnostic hybrids, not new legal model routes. Native commit includes
the selected memory, timestamp and readiness bit. Future state, clocks,
queries and hard routes evolve normally in each replay. All race noise
vectors, including those after the intervention, must match exactly.

The exact finite differences decompose as

`Q11-Q00 = (Q10-Q00) + (Q01-Q00) + (Q11-Q10-Q01+Q00)`.

The last term is interaction: message and write effects need not add.
With a factual value coefficient `aV_i=gV·v_i`, message-teacher residual is

`Q11-Q00-(aV_alt-aV_factual)`

`= value-linearization residual + commit effect + interaction`.

The first residual includes later route switches and surrogate adjoints,
not just smooth loss curvature. Commit effects include real timestamp/seen
changes and stored-key utility; note142 explains why neither all homogeneous
changes nor all timestamps can simply be dismissed as coordinate noise.

W and T are independent in the entering exponential race. Conditional on
that first T and future independent draws, legal losses Q_i give local
choice credit `pi_i (Q_i-E_pi Q)`. Forcing an alternative at its own losing
candidate time would instead mix identity and clock effects (89/92).
This job explicitly keeps the sampled first T. It does not compute the
separate current-time derivative or the whole expected sequence gradient.

## Measure all cheap coefficients with the same factual cotangents

To avoid conflating a coefficient change with changed downstream credit,
measure gV and post-write G_i under the successful **message-only** rule.
The diagnostic runs the eager reference with linear value credit. An
independent zero leaf at the selected memory update collects each slot's
cotangent without adding score/content credit. Every original parameter
gradient, factual logit and loss must nest exactly against an unobserved
message-only pass. The helper removes every write-score auxiliary throughout
that probe, so it does not covertly measure an already modified learner.

At the same factual history, compare coefficients

`aV_i`, `aV_i + G_i·(m_new_i-m_i)`, `aV_i + G_i·written_i`.

Record the homogeneous contribution separately. These are candidate local
additions using fixed factual cotangents, not the full gradients produced
by enabling linear_rw/rwn throughout a new backward pass. G may be zero
for an unused final write; no arbitrary nonzero cotangent is fabricated.
A final-event contract verifies that commit-only has no future loss effect.

## Calibrate an addition against residual utility, keeping useful credit

With coefficients a, exact local choice utility Q and positive pi, the
pi-whitened score-gradient error is

`sum_i (cQ_i-ca_i)^2/pi_i = Var_pi(Q-a)`.

A positive scalar multiplying an entire opposed teacher cannot reverse
its direction. But optional write credit is an **addition** to the useful
value teacher b. For candidate addition d, keep b fixed and minimize

`E(alpha) = Var_pi(Q-b-alpha*d)`.

Let e=Q-b. Then

`E(alpha) = Var_pi(e) - 2 alpha Cov_pi(e,d) + alpha² Var_pi(d)`.

For nonnegative alpha and nonconstant d, the oracle local minimizer is

`alpha* = max(0, Cov_pi(e,d)/Var_pi(d))`.

The possible local error reduction is

`max(0,Cov_pi(e,d))² / Var_pi(d)`.

If d is constant, it supplies no categorical credit and the reduction is
zero. Crucially, alignment with **Q alone** is not the criterion. Example:
Q=(1,0), b=(2,0), d=(-1,0). The extra term opposes Q but perfectly corrects
the overestimated base at alpha1. The Q-aligned d=(1,0) instead worsens
that base for every positive alpha. Normalize the missing utility, rather
than assuming all useful-looking components should be amplified together.

`route_fidelity_geometry.py` reports both full-teacher diagnostics and
these residual-addition diagnostics. Scalar checks verify baseline
invariance, the pi metric identity, zero constant credit, opposed whole
teachers, the two addition witnesses and factorial interaction. Baseline
subtraction precedes centering so a constant represented coefficient does
not acquire artificial variance from weighted-mean roundoff.

These are oracle measurements on fixed contexts, **not chosen training
scales**. The emitter Jacobian can couple keys, clocks and contents; its
pullback changes the relevant metric. Clipping and warm Adam are nonlinear
(139), and current final checkpoints contain no historical optimizer
moments. No fabricated warm-Adam alignment is reported. Independent FIT
calibration and actual-update comparison would be required before a fit.
Two routes provide a particularly limited local directional space; eight
sites at one span do not establish population bias or training reliability.
More precisely, the centered two-route space is one-dimensional: any
nonconstant addition with the correct residual sign admits a perfect
per-case oracle scalar fit. Zero per-case error is therefore not compelling
evidence for a usable training scale. The audit also reports one shared
oracle alpha across all eight equally weighted local metrics:
`max(0, sum Cov_pi(e,d) / sum Var_pi(d))`. Its remaining error and the
per-case alphas expose heterogeneity; they are still reused-case diagnostics,
not cross-validated calibration or a shared-emitter parameter metric. The
scalar witness has cases needing alpha1 and2: both fit individually, while
the best shared alpha1.5 retains.125 summed local error. A later U4 audit
would test additional directional freedom after actual admission, not infer
it from the present U2 projection.

## Bounded implementation and admission

`language_route_fidelity.py` requires a completed nongrown U2/linear result,
actual final weights, matching batched/core/factory hashes and fresh output
before importing Torch/NumPy. It preserves model tensors, Python gains and
caller RNG, and restores inherited/owned `LaneRace.apply` descriptors and
the write helper even on exceptions. It executes no optimizer update.

The native admission first uses a tiny double integrated model to check
every-gradient nesting, fixed-time factorial semantics, causal prefixes
and the last-event write boundary. Then original float32 trained weights,
FIT[0:33],32 prediction positions, events7/23, first/last depth, head0,
seeds314159/322078 yield eight paired sites. Scores, selected times,
messages and all future noise vectors must match before the intervention;
commit-only leaves outputs through the current event unchanged. All checks
must pass before a completed diagnostic JSON is written. Unexpected contract
failures remain unsuccessful guarded jobs with their preserved runner log.

Per-target CE uses model precision and sequence sums accumulate in float64
to preserve small suffix differences. Coefficient dots use float64 on the
captured represented operands. Units are **summed nats over32 positions**,
not the original training mean/clipped update; all three coefficients and
their legal Q use the same denominator. No full expected-gradient identity
or literal bitwise reconstruction of an installed score VJP is claimed.

Actual trained work: **34 full forward replays,10 backward replays**,1,088
forward input positions and8,704 forward races for D4/H2. Native admission
adds nine five-position forward replays and three backwards. All are paid
diagnostic work; FLOPs, traffic and energy are not measured or zero. Wall
time/RSS include admission and observer overhead, not an isolated speed test.

Unique pending one-job queue:
`queue/local_language_route_fidelity_20261003T125500Z.txt`.
Use only `experiments/queue/run_safe.sh` on the actual producer after physical
host reservation: one CPU thread, VMS3,000,000KiB, RSS watchdog1,250,000KiB,
minimum8,192MiB available,420-second timeout. The real p64 checkpoint and
host-global reservation remain unavailable in this container. This job
must not displace the active curie integrated chain or AWS fits/90M admission.
No native execution took place here; help, syntax, source/missing-checkpoint
guards, scalar math and fake-runtime callback restoration are the only
executed checks. No pending cell enters a quality table.

On completion, retain raw legal/hybrid losses and every failure. First ask
whether useful write utility is large, whether the cheap term explains the
**remaining** value residual, and whether interactions dominate. Only then
choose between better state/time utility, independent scalar calibration
or leaving value-only credit unchanged. A cheap terminal local-loss repair
would not substitute for memory-sensitive suffix utility if that is the gap.

## 13:30 UTC update: pool4 failure and the shared memory score path

The shared pool4 linear_rwn fit also diverged (2e62072/b99f2c7); the earlier
stable-but-worse U2 result is retained. Write-address credit is withdrawn
from the active queues; value credit and independent-seed confirmations
remain priority. This strengthens the need for actual utility/stability
measurements; it does not prove a particular feedback cause.

Both value and write choice coefficients pull back through the same score
read of memory: holding incoming q fixed, `d s_i/d m_i = K_i^T q/sqrt(P)`
inside the clamp. Thus both transmit route-score gradients to memories.
The write addition is distinct because it also changes the effective
**persistent-update** Jacobian. Let S be the local score Jacobian, B map
one probability coordinate into its slot's detached delta, and Jpi be
the softmax Jacobian. The zero-valued write auxiliary adds `B Jpi S` to
the factual memory-update Jacobian. Value auxiliary instead changes the
emitted-message Jacobian. These are different feedback locations, not
presence versus absence of the shared score-memory path. Neither identity
attributes the measured divergence; both mechanisms require trained-state
measurements, and the actual forward state remains unchanged by auxiliaries.
