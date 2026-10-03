# Credit geometry after clipping and actual warm Adam; safe admission

## Concrete question and retained construction

Note 137's raw-gradient variance/work heuristic favors k8 at one restricted
trained p4D4 state. That does not imply that k8 minimizes ACTUAL Adam
update noise or prediction error. Note 138 prespecifies the required 48 real
warm-update forks, including same-noise all-site references. No temporal/
race/private-memory/key-value mechanism is removed or replaced.

The operator taking raw normalized gradient g to a parameter update is
nonlinear and anisotropic. It mixes coordinates through global clipping,
then rescales them using historical moments. A large raw covariance trace
in one direction can become a small update trace, while a small raw term
on another coordinate can matter more. This can explain a failed raw
variance criterion without a capacity obstruction or a missing derivative.

## Exact real-arithmetic clipping Jacobian, including the implementation floor

Let r=||g||, clip threshold tau>0, e=1e-6 as in the actual clipping rule.
On the unclipped branch r+e<tau, h(g)=g and J_clip=I. On the clipped branch,
with r>0 and away from its boundary,

    h(g)=tau*g/(r+e),
    J_clip = tau/(r+e) * [I - g*g^T/(r*(r+e))].

For u=g/r, tangential eigenvalues are tau/(r+e); radial eigenvalue is
 tau*e/(r+e)^2. With e=0 the radial derivative vanishes entirely. The
actual floor leaves a small radial response. Thus replacing covariance
by a scalar norm reduction misses which directions survive. The boundary
is piecewise, not differentiable in the same way as an interior point.
None gradients are excluded from the active norm; keep the active mask
fixed locally rather than treating disconnected and zero gradients alike.

## Actual historical Adam diagonal, not a fresh sign-step approximation

For a fixed active parameter coordinate, historical moments m,v, and NEXT
coordinate step t (which can differ when parameters were previously unused),
write beta1,beta2, eta and eps from the actual saved optimizer group. With
weight_decay=0, maximize=False and amsgrad=False, as in the admitted teacher,

    a=(1-beta1)/(1-beta1^t), b=(1-beta2)/(1-beta2^t),
    n=[beta1*m+(1-beta1)*h]/(1-beta1^t),
    s=[beta2*v+(1-beta2)*h^2]/(1-beta2^t), d=sqrt(s),
    Delta(h)=-eta*n/(d+eps).

For d>0, the derivative is

    D_i=-eta*[a/(d+eps) - n*b*h/(d*(d+eps)^2)].

The full raw-gradient-to-update Jacobian is J_update=diag(D_i)*J_clip,
with zero outputs for inactive parameters. Its diagonal need not have a
uniform magnitude or even a uniform sign: increasing a coordinate's
current gradient can also increase its denominator and alter historical
moment alignment. This is why current-gradient rescaling is not equivalent
to rescaling the whole Adam history. If d=0 and n=h=0, the composite update
has derivative -eta*a/eps, despite naive sqrt autograd producing a 0/0
chain. If d=0 with nonzero historical numerator, one-sided behavior requires
separate treatment; do not hide it with an arbitrary denominator floor.

These formulas describe the ideal real-arithmetic optimizer map. Actual
FLOAT32 stored weights/updates include rounding and can change active hard
routes. Do not claim the derivative of the quantized stored program equals
this smooth map. The finite actual-update vectors in138 are the primary
comparison; this Jacobian is a diagnostic approximation to explain them.

## Conditional finite-population variance in the correct update geometry

At fixed entering/noise history E and same-noise FULL-site raw gradient
G_full(E), locally linearize the ideal update map. Site perturbations
eta_k=G_k-G_full have conditional mean zero under the declared independent
uniform subset law. Then

    Delta(G_k)-Delta(G_full) ~= J_update(E)*eta_k,
    trace Cov[Delta(G_k)|E] ~= trace[J_update(E)*Sigma_sites,k(E)*J_update(E)^T].

The approximation requires small perturbations, fixed masks and no clipping
boundary crossing; k1's large rare-route perturbations may violate it.
The exact MSE includes nonlinear mean shifts and higher-order terms.
Note 138 records raw, clipped and actual-update differences rather than assuming
this approximation holds. Expected clipped/update means are not preserved
merely because preclip route estimators are unbiased.

For homogeneous legal R and FIXED J_update evaluated at the full gradient,
finite-population covariance scales by gamma_k=(R-k)/(k*(R-1)). Therefore
its transformed trace has the same algebraic scaling, but a DIFFERENT
coefficient. Between-history variability of full actual updates provides
another floor. A new variance/work square-root optimum could be derived
from those coefficients under the same fixed-state/constant-cost assumptions
as note 137. The raw-gradient optimum 6.09 need not be the update optimum.

Output/loss sensitivity supplies yet another metric. Only a declared
fixed-topology linearization can replace J_update by a prediction Jacobian
composition; native hard route changes and actual finite FIT loss changes
must still be measured. No promise of heldout features or supremacy follows
from a covariance identity. No new optimizer or architecture is implemented
from this derivation alone.

## A convex counterexample: unbiased route sampling can become ascent

This is an exact constructed example, not a measured native failure. Let
the mean loss be L(theta)=theta^2/2, evaluated at theta=0.1. Ten equally
likely component losses are ell_i(theta)=theta^2/2+a_i*theta, with
a_1=9.9 and a_2..a_10=-1.1. Their mean is exactly L; every component is
convex. Uniform one-component gradients are 10 with probability .1 and
-1 with probability .9, hence E[g]=.1, the correct full gradient.

Actual threshold-one clipping with floor e gives

    E[h(g)] = .1*10/(10+e) - .9/(1+e) < 0,
    h(E[g]) = .1 > 0,                  e=1e-6.

The expected clipped SGD step is positive, against the desired negative
step. Indeed E[L(theta+Delta)-L(theta)] equals
theta*E[Delta]+E[Delta^2]/2 and is strictly positive for every positive
learning rate in this example. There is no representational obstruction:
the exact full-gradient step has the correct direction.

Fresh Adam does not remove the example. Its first update is
Delta(g)=-eta*h(g)/(|h(g)|+eps_Adam), with bias correction included.
For the actual eps_Adam=1e-8, the frequent negative component again
dominates the expected step, which is positive. Removing clipping alone
also leaves the fresh-Adam near-sign imbalance. This complements note 120's
fresh scaling cancellation; warm moments require the actual138 forks.

Generally, write c(g)=min(1,tau/(||g||+e)). At a fixed history,

    E[h(g)] = E[c]*G_full + Cov(c,g),
    E[h(g)]-G_full = (E[c]-1)*G_full + Cov(c,g).

Thus the sign/content correlation with clipping strength matters, beyond
raw trace or the mean clip factor. Wider unbiased site support can reduce
both variance and this transformation bias. It does not merely rescale
an already correct update. A larger k does not guarantee a better result
for every distribution, especially at equal total work.

The native sampled teacher scales selected contributions by R/k; a rare
large sample can dominate the global norm and change how unrelated content
coordinates are clipped. This is a concrete hypothesis connecting sparse
credit to apparent depth plasticity. The observed restricted site-noise
dominance supports testing it; it does not demonstrate that its mean update
has the constructed example's reversal. Note 138's paired full-reference errors
and retained finite FIT predictions distinguish this question from raw
variance alone. Alternating route/content updates or adding phase freedom
does not itself correct this estimator-transformation bias.

## Local bias bounds and allocation after optimizer conditioning

If an ideal update map U is twice differentiable on the convex region
containing G_full and its site perturbations, with second-derivative
bilinear operator norm at most M and fixed active masks, conditional
E[eta_k]=0 gives

    ||E[U(G_full+eta_k)]-U(G_full)|| <= M/2 * E[||eta_k||^2].

This follows by Taylor's integral remainder and cancellation of the linear
term. It applies neither across a clipping kink nor to rounded float32
updates by assertion. The nonsmooth boundaries must be checked or bounded
separately; exact finite forks remain preferable here.

For heterogeneous legal site counts R_j, conditional independence of
episode-subset draws, fixed history and a fixed full-gradient Jacobian J,
write v_jr for each route's already batch-normalized vector and vbar_j
for its site mean. The linearized update site trace is

    sum_j b_j^U * (1/k_j - 1/R_j),
    b_j^U = R_j^2/(R_j-1) * sum_r ||J*(v_jr-vbar_j)||^2,

for R_j>1; R_j=1 has no sampling variance. With per-site work c_j and a
fixed site-work budget, interior allocation is k_j proportional to
sqrt(b_j^U/c_j), bounded to 1..R_j. Raw-gradient allocation replaces J
with I and can give a different answer. Costs must include discovery,
returns, probes and optimizer work; this is not a free uncertainty bonus.
Any fitted allocation needs an independent pilot or valid inclusion law,
as note 137 explains, and must be compared under sustained integrated learning.

Reproducible static verification: `python3 experiments/check_credit_geometry.py`.
Standard-library checks pass clipped/unclipped Jacobians, the composed warm
Adam Jacobian including heterogeneous coordinate steps and a positive
sensitivity, the convex ascent example in exact rational arithmetic, and
all subsets of a transformed finite population. Maximum finite-difference
absolute error is 6.97e-12. These are mathematical/source checks, not native
gradient contracts, fitting runs or benchmark evidence.

## Other-host progress and the remaining learning-mechanism gap

Completed compiled reset-language p16/D8 at 10M has test score 2.718723394 bpc at
T128 and 2.719413474 at T256, DEV 2.676420348, a nominal one-pass budget
of 9,994,240 presented characters sampled in random reset segments,
54,907 parameters. Preserve this progress beside the prior
2.899 result; multiple recipe changes prevent isolated attribution to
clipping, depth or compilation. Its saved step-work extrapolation is
405,023.02246 unit-special FLOPs/character and 4,047.897292 whole-fit GFLOPs;
evaluation is separate. These are completed result fields, not a new fit.
Source: [completed 10M result](../results/language_batched/curie_language_batched_10M_p16d8_skip2_l64_lr004_cmp_s6_20261003T054000Z.json).

Source inspection matters for transfer: language_batched_benchmark.step
backpropagates factual cross-entropy. LaneRace.backward and its compiled
equivalent give selected-value credit and common first-time clock credit;
this driver does not add the losing-write categorical replay term. Thus
this temporal/private-state forward model's positive fit is evidence for
its declared learning recipe, not a completed corrected-replay comparison.
The separately queued original-teacher/corrected private/shared replay
matrix remains necessary. Do not treat a restricted teacher's weakness as
a test of absent mechanisms, or edit active drivers to combine protocols.

The user's newly queued AWS 90M compiled language arms retain their own
contracts/pilots, DEV-only selection and guarded admission. Current local
compiled language/DVS work and AWS 90M must not be displaced by note 138's job. The
restricted p4/D4/B4 diagnostic cannot identify deep nonlinear language
features, generalization or advantage by itself. Subsequent transfer needs
source-bound actual trained language states and an explicitly matched
counterfactual teacher under an affordable numerical protocol.

## Historical driver provenance

07eaaae introduced compiled support in the shared BL driver. Preserve that
active source. Exact historical SHA-256 4b163a25008ea8261acbee465770bad84dff3dbc1aa3c0ff74f39571fd83e8f8
bytes are archived from `07eaaae^` under `archive/frozen_sources/<digest>`.
`legacy_batched_driver_binding.py` explicitly loads them for note 138 and maps
ONLY this known historical source path to its verified archive in a copied
publication view. Saved JSON, numerical fields and original source hashes
are untouched. Root readable_report applies this view; earlier frozen
producer modules remain unchanged. Changed or missing archive bytes fail.
This makes historical evidence publishable after a supporting driver evolves,
without attributing old numbers to the new compiled implementation.

## Physical-host admission: the container-local lock is insufficient evidence

The new curie handoff reports tmux curie_chain10 compiled 10M training on
the physical host. This /workspace environment is an overlay container;
its process list, tmux socket and /tmp runner lock do not show that run.
A free container-local lock is NOT proof that the physical curie host is
idle. Shared available-memory readings alone cannot establish exclusivity.
The user has been asked for missing physical-host state. Until confirmed
idle or separate, note 138 remains queued; no new fitting/profiling job is launched.
The AGENTS one-training-job rule and the 8 GiB floor take precedence.

Potential prior operational overlap is now explicit:134/136 ran around
05:37/05:42, while the later handoff reports compiled curie work active
around05:45. Without physical-host process/lock visibility, host-wide
exclusive timing is not established. Keep their numerical contracts,
operation counts, data and completed results; do not use those wall times
as isolated speed benchmarks. Numerical/FLOP variance-work heuristics do
not charge or claim isolated wall-time or measured energy advantages.
This caution is beside the retained evidence, not a deletion or rewrite.

Next safe step requires a host-global lock/status visible to both execution
contexts, or explicit confirmation of physical idleness/separate host.
Do not let a waiting job auto-start merely because this container's /tmp
lock becomes free. No elapsed timeout is a host-idle confirmation. AWS has
its explicitly authorized three-slot scheduler exception; curie does not.
Preserve AWS integrated full replay10M/teachers/controls and the other host's
compiled language/gate/capacity work while the bounded diagnostic waits.
