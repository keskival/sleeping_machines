# Analogous credit systems and alternating learning, 2 October

Primary papers consulted in response to the user's request. Established ideas
are precedents, not evidence that they fix this substrate. The implementation
remains unchanged; numerical/gradient/work contracts precede any new fit.

| Analogue | What it addresses | Transfer to this substrate |
|---|---|---|
| [Stochastic computation graphs](https://arxiv.org/abs/1506.05254) | Combines score-function and differentiable paths; legal baselines | Audit branch, categorical and clock terms separately, with correct stochastic ancestors |
| [RELAX](https://arxiv.org/abs/1711.00123) | Unbiased learned control variates, trained for gradient variance | Clock baseline/critic must cancel its own expectation derivative correctly; an arbitrary time-dependent subtraction is biased |
| [Rao–Blackwellized straight-through Gumbel](https://arxiv.org/abs/2010.04838) | Reduces estimator variance through conditional averaging | Average nuisance randomness where inexpensive; variance reduction does not remove straight-through bias |
| [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416) | Alternating phases mitigate policy/value interference; cloning constrains policy drift | Test short message/route phases, explicit shared-map ownership and a route-drift budget |
| [ST-MoE](https://arxiv.org/html/2202.08906v2) | Router numerical stability, load balancing, quality versus clipping tradeoffs | Measure exposure and numeric scale; do not equate balance with correct alternative utility |
| [Expert Choice](https://arxiv.org/abs/2202.09368) | Controls expert training allocation through expert-selected batches | Exposure is a real resource; batch-global selection is not automatically causal sparse event routing |
| [EventProp](https://arxiv.org/abs/2009.08378) | Event adjoints and derivative jumps for spiking dynamics | Retain reset/time sensitivities; fixed event-topology derivatives do not alone recruit absent routes |
| [DiCE](https://arxiv.org/abs/1802.05098) | Correct higher-order stochastic derivatives under automatic differentiation | Required before differentiating through sampled learning to reward optionality or future adaptation |

PPG's value prediction is an auxiliary objective, whereas our message maps
directly affect the main prediction and clocks. Its result motivates an
experiment, not an identification of our message layer with a critic. RELAX
uses explicit unconditional/conditional relaxed samples and correction terms;
subtracting a learned surrogate alone is not the RELAX estimator. EventProp's
specified spiking dynamics differ from addressed memory and route replacement.

## Alternating updates: preserve computation, change the schedule

Let phi denote disjoint route-output parameters, psi disjoint message/value
parameters, and omega shared incoming-content/state processing. Forward values
continue to affect routes and times, and routes continue to affect subsequent
values. Alternation can schedule updates without removing these dependencies.
But detaching content when computing route credit removes an actual chain-rule
path; that is a different gradient approximation, not merely a schedule.

For smooth J(phi,psi), short sequential block steps differ from simultaneous
steps by a term of order eta_phi*eta_psi times a cross-Hessian action. A useful
staleness bound is

    ||g_phi(psi+Delta)-g_phi(psi)|| <= L_phi,psi*||Delta||,

when the cross derivative is bounded on the segment. This explains why a
router trained against rapidly changing messages can chase outdated utility.
It gives no universal superiority for alternation: long freezes can lock in
bad routes, starve message exposure or repeatedly chase stale state. Adam,
clipping and hard topology add further dynamics beyond this smooth argument.

Initial comparison, only if the frozen note87 audit supports interference:
joint updates versus one message update then one route update. Match total
presentations, optimizer work and measured fitting arithmetic; a two-phase
cycle is not a free second update. Start with disjoint output blocks and leave
omega under the ordinary summed derivative, or explicitly preregister its
ownership on each phase. Partition clock parameters separately from relative
choice and record every parameter's actual dependencies. Momentum/Adam state
of inactive blocks must stay fixed, not drift via a nominal zero-gradient step.
Recompute utilities after each parameter update; historical memory features
need source weight/version identity. Keep full-support proposals so message
phases can teach alternatives rather than only reinforcing existing winners.

Measure utility rank drift on fixed independent-noise fitting probes, route
distribution KL, parameter-space gradient cosine/covariance, alternative-map
exposure and held-out quality. A small KL or shorter phase is a stability
device, not a quality proof. Alternation and branch-content replay should be
separate comparisons; do not combine two repairs and lose attribution. PPG
suggests a later trust penalty if measured route drift is harmful, not an
immediate new loss or an unbounded schedule sweep.

## Three formal details worth carrying into the next audit

**A variance-optimal baseline is not generally mean loss.** For a prefix-fixed
baseline b, parameter-space score vector h with E[h]=0, and estimator h*(F-b),
minimizing trace covariance gives

    b* = E[F*||h||^2] / E[||h||^2].

If the estimator also includes pathwise term v, its optimal scalar baseline
is E[(h*F+v) dot h]/E[||h||^2]. This accounts for shared parameter Jacobians
and covariance between message and route credit. Fit the baseline from prior
or independent draws and freeze it for the current sample; optimize the
measured parameter-space variance, charge the estimator cost, and retain the
baseline legality proof. Current per-pass common draws require independently
sampled diagnostic noise before estimating this covariance. A zero denominator
means no score component to calibrate, not a division by zero.

**MoE common-logit regularization is not a harmless gauge here.** ST-MoE uses
(logsumexp(s))^2 as a router z-loss. Softmax choices ignore s->s+a. An
exponential race instead has total rate Lambda=sum exp(s), so this shift also
changes expected first time1/Lambda. In note85's coordinates, the same penalty
is simply c^2, explicitly regularizing the computational clock toward a unit
rate. Units and desired time computation matter. Do not import it without
measuring numeric failure and declaring that clock objective change.

**Almost-everywhere pathwise derivatives can miss expected boundary flux.**
For one noisy threshold a(theta) separating two legal outcomes,

    J = integral[-infinity,a] L1(theta,z)p(z) dz
        + integral[a,infinity] L2(theta,z)p(z) dz,

and the derivative includes

    [L1(theta,a)-L2(theta,a)]*p(a)*a'(theta)

in addition to the branch-interior derivatives (assuming a smooth fixed noise
density and the required integrability). The boundary has zero probability
at a fixed parameter but nonzero flux as it moves. Exact smooth event/reset
adjoints and counterfactual choice/support credit solve different pieces.
Our current one-node categorical contracts do not prove all future timing
jump/route-creation terms. Higher-order optionality credit likewise cannot be
obtained by repeatedly differentiating detached first-order teachers; DiCE
demonstrates the missing-distribution-dependency problem. Keep that work
deferred until first-order utility and useful representation learning improve.

Decision: frozen joint-route/content/time audit first; select ONE repair from
exposure, interference or clock variance; contracts, tiny fit, matched work and
independent confirmation next. Strong controls remain the practical target.
