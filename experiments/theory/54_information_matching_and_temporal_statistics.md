# Information matching: a precise version of the antenna intuition

Global §§358–362. Connects §§171–172/345's reachable learning geometry to
conditional sufficient messages and computational time. The metaphor suggests
a design criterion; it does not assert literal electromagnetic resonance or a
universal optimal neuron. Results supporting a fitted advantage remain separate.

## 358. Match learnable directions to predictive distinctions

Let C be causal context and M=(M1,...,Mk) the received messages. For a
deterministic representation T(M,C), data processing gives

    I(Y;T | C) <= I(Y;M | C).

It loses no label information exactly when Y is conditionally independent of M
given (T,C). With unrestricted optimal log-loss predictors, the compression
penalty is

    H(Y|T,C)-H(Y|M,C) = I(Y;M|T,C).

Fitted probe loss gaps are diagnostics, not exact mutual-information estimates:
finite data, optimizer failures and restricted probes can create a gap. Deeper
state adds earlier causal observations; a nonlinear transformation of a single
message cannot manufacture information absent from its history.

Formalize impedance matching locally. Let A(X) map small changes in a
generative factor alpha to the desired predictive change, and J(X) map model
parameter changes to its output tangent. Let F be predictive curvature and M0
a parameter step metric. The attainable regularized correction to a target
tangent a is

    delta_theta = (E[J^T F J] + lambda M0^-1)^-1 E[J^T F a].

If E[J^T F a]=0, that local model tangent cannot correct this distinction even
with nonzero derivatives and many parameters. This does not prove a globally
trained nonlinear model cannot learn a new tangent. Poor conditioning can
likewise make an available distinction expensive to learn.

With Gtheta=E[J^T F J], Galpha=E[A^T F A], whitened coupling is

    B = Gtheta^(-1/2) E[J^T F A] Galpha^(-1/2).

On nonsingular supported subspaces its singular values are canonical
correlations in the predictive metric, between zero and one. Pseudoinverses or
declared regularization handle nullspaces. Alignment, observable rank and
conditioning—not raw parameter count—make the antenna analogy precise.
Estimate these on fitting/calibration data, then measure prediction benefit
on independent data; fitting an alignment diagnostic to development labels
would spend selection budget. No generic claim that natural generators always
have low rank is needed.

## 359. Pool evidence, including its reliability, rather than raw values

For conditionally independent observations yi=Hi z+epsilon_i with Gaussian
noise covariance Ri and Gaussian prior, each observation sends information
parameters

    Lambda_i = Hi^T Ri^-1 Hi,   eta_i = Hi^T Ri^-1 yi.

The posterior has Lambda=Lambda0+sum Lambda_i,
eta=eta0+sum eta_i, and mean Lambda^-1 eta. This is an exact derivation for
that linear-Gaussian assumption, not the semantics of current learned vectors.
Raw averaging cannot distinguish equally averaged observations of unequal
reliability. Repeated evidence and missing evidence are also different: a
message should preserve a precision/count statistic when the task needs it.

Conditional independence justifies additive log likelihoods, not arbitrary
pooling. Shared latent causes or correlated noise require appropriate joint
terms. In mixtures, evidence can change which cause explains another signal;
this creates conditional interactions. A context-dependent message map or
low-rank bilinear term (A x) elementwise_times (B context), followed by learned
mixing, can represent these interactions at O(d*r) projection work (§346).
It is useful only if the conditioned directions generalize and pay their cost.

Order-sensitive roles also matter. Anonymous sums identify (x1,x2) with
(x2,x1), so they cannot preserve a target that distinguishes those orders.
Identified channels, temporal state or role-dependent projections can retain
the distinction. For real scalar pairs, sum alone also loses dispersion:
(1,-1) and (2,-2) have the same sum but different products. Adding sum of
squares recovers the product by x1*x2=((x1+x2)^2-(x1^2+x2^2))/2. This exhibits
a useful added statistic rather than adding redundant affine maps.

Cheap learned statistics, several independent addressed/head channels and
event-time integration are therefore sensible tests. Hard winner selection
can be excellent for a dominant cause, but a target depending on multiple
independent pieces may need several arrivals or integrated evidence. The
existing numerical windows/trains make this possible in principle; they are
not yet fitted replacements for the complete native model.

## 360. Match temporal modes to task observability and precision

A decaying/rotating mode can cheaply compute a temporal feature between
arrivals. Its frequency is useful when phase changes the needed prediction.
For y=a^T exp(-gamma t)R(omega t)m, the frequency derivative has magnitude
bounded by t exp(-gamma t)||a||||m||. It vanishes when no elapsed time is
observed, the readout ignores that mode, or relevant state has decayed away.
Protected coordinates retain order/content while temporal coordinates model
duration; event gates and learned mixing couple them.

For independent Gaussian timestamp jitter epsilon with variance sigma_t^2,
E[exp(i omega epsilon)]=exp(-omega^2 sigma_t^2/2). High-frequency phase can
therefore be attenuated by measurement uncertainty. On a uniform sampling grid
with step Delta, omega and omega+2*pi*k/Delta alias. Irregular times can break
that exact equivalence, but do not eliminate finite precision or conditioning.
Learned time directions need observable information, not simply fast rotation.

In a stochastic dynamical generator, protecting a mean forever is not the
same as preserving confidence forever. Even when the latent transition is
identity, process noise accumulates uncertainty during silence. The current
protected variant has no explicit Bayesian confidence channel; confidence-aware
silence processing is a future comparison, not a property acquired by naming it.

## 361. A cheap race can communicate aggregate evidence as well as a winner

For independent exponential clocks with rates lambda_i=exp(score_i), let
Z=sum lambda_i. Then

    P(W=i)=lambda_i/Z;   P(Tmin>t)=exp(-Z*t).

Conditional on rates, W and Tmin are independent. Winner identity encodes
relative preference; arrival-time distribution encodes total rate. If
eta=log Z, the timing log-likelihood derivative is 1-Z*Tmin, whose mean is
zero and variance one. Thus one race has unit Fisher information about eta;
m independent races have m units. A single stochastic arrival is not an exact
deterministic log-sum-exp, nor is physical rate setting free.

This supplies a concrete additional meaning for received times: strength of
aggregate evidence or an uncertainty statistic, not just which value won.
Reused scalar matches can drive different calibrated policies, exposing
different aggregates without repeating every key/query projection. Their
O(R*K) policy settings, precision, clocks and content readouts still cost work;
the existing R2 language fit has not established a benefit. The next test must
hold content, matches and budget fixed and make aggregate evidence predictive,
with frozen/refitted timing-removal controls. Optional sparse value deliveries
need separate accounting; losing-value training may still read many values.

## 362. There is no universal optimum number of freedoms per unit

Add a degree of freedom when it creates an observable, conditioned direction
that predicts independent data better than its added fitting/inference cost.
Use task-relevant rank, per-route exposure, gradient alignment and ablations.
Extra clocks, moment statistics and bilinear interactions are different ways
to achieve this; none should be installed everywhere by default.

The first implemented tests are protected/time state and shared/private maps
under §§354–357. Next conditional tests: precision/count messages for unequal
reliability, role-aware pooling and cheap joint statistics, aggregate-rate
reception, then low-rank interactions after aggregation. Preserve incoming
content and core races/counterfactual credit. Isolate each addition before a
combined fit; compare full work and independent held-out quality before scaling.
