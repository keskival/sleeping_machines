# Coupled physical-time evolution offsets and alternating updates

User proposes a learnable evolution offset, trained with messages and frozen
with race parameters, while explicitly preserving time–representation coupling.
Concrete failure addressed: message evolution currently has to use the phase
implied by the physical elapsed time and frequency. A desired content phase
change may therefore compete with route/clock changes or alter frequency for
every age. The fitting-only audit89 does not establish this as the dominant
quality bottleneck; this is a separately user-requested bounded intervention.

Retain physical elapsed age a, damping rate rho and frequency omega, and use

    read_beta(m,a) = exp(-rho*a) * R(omega*a + beta) * m,
    beta = pi*tanh(raw_beta).

Beta is per layer/head/carrier pair, initialized exactly zero:32 parameters
at p16/L2/H2. Damping continues to use nonnegative physical age; actual emission,
race minimum, receiver readiness and stored physical timestamps are untouched.
Beta is a bounded phase calibration, not an independent latent clock. Its
derivative with respect to age still contains omega and damping rho; freezing
beta does not remove timing credit through age or replace time computation.
There is no future input access, timestamp advance or negative-age amplification.

The physical flow Phi_a(m)=exp(-rho*a)R(omega*a)m remains a semigroup. The read
map Phi_a C_beta is a calibration at reception, not itself a claimed semigroup:
applying its beta twice changes phase twice. Store raw emitted payload with its
physical time and re-read it once per reception, as the addressed model does;
do not repeatedly transport an already calibrated stored value. Constant phase
can be absorbed into an unconstrained adjacent linear map in restricted cases.
Thus this is an optimization/conditioning hypothesis and explicit parameter
choice, not a novel primitive or proven increase in representational capacity.
Nonlinear persistent state, routed content and subsequent races stay integrated.

Offset is applied AFTER the current race forms scores and chooses its payload.
It cannot directly change that already-computed choice, but can change later
representations, memory updates and routes. Global decoupling is neither claimed
nor desired. Keep native losing-route score teachers and current timing credit;
do not mix this first experiment with a new credit estimator or shared-map repair.

## Parameter ownership and comparisons

Route block: queries, unit keys/key-read maps and clock biases. Message block:
payload/state maps, their decay/frequency controls, transport decay/frequency,
decoder and raw_beta. Shared embedding/content/source-gate/channel-mixing maps
receive the ordinary complete derivative each update. This explicitly retains
both derivative paths; an update to shared maps may change both routes/messages.

Joint mode updates every block. Alternating mode uses message then route on
successive windows: raw_beta is active only in message windows. Set inactive
parameter grad=None before clipping/Adam; retain inactive parameters AND their
momentum/step state. The shared content weight's saved Adam step determines the
phase, so interrupted recovery reproduces the next phase exactly. Compute the
full derivative first, then apply block ownership: no implicit content detach.
Active-gradient clipping is part of the declared algorithm; it changes global
balance, and alternate blocks see fewer updates at a fixed presentation count.
Report these differences instead of calling the comparison a free second step.

Before fits: zero-offset native logits/state/routes and every original parameter
gradient identity; nonzero serial/batched full state/gradient agreement; fixed
clock input age and offset finite-difference contracts; beta derivative nonzero
and clock derivative retained; actual alternating inactive Adam-state identity;
continuous/interrupted next-phase model/optimizer/RNG recovery; full/partial
window operation accounting. A copied batched kernel receives only the offset
addition and refuses terminal-pair mode; base kernel remains source-frozen.

Then joint/alternating24fit/eightdev/two-pass readiness fits, same seed/U16+U8.
Only measured complete contracts/readiness admit fixed256/192/four-pass seed6
joint-offset versus saved native local control; same initialization/data/noise,
separate charged work. Gate >=.02NLL gain, <=1pp accuracy decline, <=1.50 fitting
work ratio, RSS<900000KiB. Test alternating against that joint-offset control
only after its own readiness. Independent seed7 confirmation is necessary
before full-data scale-up. Preserve all three possible outcomes: offset helps,
alternation helps, or neither helps. No after-result tuning/extra epochs.

Other-host exact-pi/tied/depth experiments stay separately owned. Note89
records teacher/time-law limitations of the old fidelity audit beside the
original interpretation. This change neither depends on that audit nor implies
that the main learning problem has been solved.

## Additional freedom as local controllability, not automatic gradient surgery

Let s denote the current choice scores and z a received representation. In a
linearized comparison, original parameter increments yield
(Delta_s,Delta_z)=(J_s*Delta_theta,J_z*Delta_theta). A locally post-race offset
adds columns (0,J_beta) when the entering prefix is held fixed. The attainable
set contains the original one, since Delta_beta=0 reproduces it. Consequently
the best local squared target residual cannot increase. This is a statement
about attainable increments, not an SGD convergence or validation guarantee.
If J_beta lies in the existing fixed-score content tangent image
J_z(kernel(J_s)), the new coordinates add no first-order controllability.
Otherwise they relax a genuine representational tradeoff. Shared global offsets
can still alter later/previous-prefix computations; zero local score derivative
is not a zero full-sequence routing Jacobian.

For physical age a at fixed parameters, the two independent directions are
d/da[exp(-rho*a)R(omega*a+beta)m] (damping plus rotation) and
d/dbeta[...] (rotation). With nonzero damping and a nonzero message they are
generically linearly independent per carrier pair: the offset can rotate without
requiring the damping change induced by waiting. With rho=0 and a single band,
offset/time directions are collinear, so this particular freedom is locally
redundant. If omega=0, offset can still rotate a message although age contributes
only damping. These rank cases are required numerical contracts, not assumed
quality effects. The physical-age derivative remains present in all cases.

Relevant established intuition: [PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
projects conflicting task gradients, and [CAGrad](https://arxiv.org/abs/2110.14048)
balances task improvement while retaining an average-loss objective. These
address distinct task losses. Our routing/content terms belong to one coupled
prediction objective and use scoped surrogates: a negative inner product alone
does not justify projecting a true chain-rule term away. Private offset parameters
and a tested update schedule are the first intervention; no gradient surgery is
introduced by this analogy. The controllability derivation above is independent
of those papers and claims no new general optimization theorem.
