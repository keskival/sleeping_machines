# Generators, predictive spectra and sampling of evolving state

Global §§363–368. Continues the antenna analogy through precise special cases
and counterexamples. The following are analytical criteria, not evidence of
benchmark supremacy or literal resonance between all generators and models.

## 363. Matching a generator is realizability, not automatic learnability

If the true conditional law is p*(y|history), expected log loss decomposes as

    E[-log q_theta(Y|history)] = H(Y|history)
                               + E[KL(p*(.|history) || q_theta(.|history))].

If a receiver family contains the generator's observable conditional law, its
best achievable approximation KL is zero. This helps, but says nothing about
whether finite data and the chosen optimizer find that law. A smaller or
differently structured receiver may represent the same observable distribution
more economically; generator parameters need not be identifiable. Hidden-state
symmetries, redundant modules and stochastic history complicate parameter
recovery. Predicting the distribution is the objective, not copying the teacher.

The appropriate experiment crosses generators AND receivers. Use frozen native
temporal, attention and dense-recurrent teachers, then fit the receiver families
on independent generated streams. Store teacher parameters, seeds and input
histories; reveal no hidden teacher state to a receiver. On independent prefixes,
score the teacher conditional probabilities against receiver probabilities to
measure conditional KL directly, alongside sample log loss and complete work.
Equal source data, horizon, tuning budget and capacity/resource curves prevent
declaring one's own teacher distribution a general benchmark win. This battery
cell requires causal generator adapters; it is planned, not currently runnable.

## 364. Predictive modes are Fourier-like only under additional structure

Consider past X and future block Y. The conditional expectation operator

    (T f)(x) = E[f(Y)|X=x]

maps L2(P_Y) to L2(P_X) with norm at most one by conditional Jensen. In a finite
distribution, or when this operator is compact, it has a singular system:

    T f = sum_k sigma_k u_k <v_k,f>.

These modes identify predictable future functions and their corresponding past
features. If T is Hilbert–Schmidt, the optimal rank-r Hilbert–Schmidt
approximation error is sum_{k>r} sigma_k^2. Fast singular-value decay would
support economical predictive compression. This bound concerns the operator
and its L2 metric; it is not automatically a cross-entropy or compute bound.

For general distributions T need not be compact. For continuous X=Y it can be
the identity on an infinite-dimensional L2 space, with no decay of predictive
directions. There is therefore no universal small spectral receptor. Fourier
modes arise from translation symmetry; arbitrary nonlinear hierarchical
generators need not diagonalize in that basis. Hierarchical conditional
factorization and spectral factorization are related design ideas, not identical.
Ordinary covariance modes can also track high-power nuisance instead of the
target. Use past–future/predictive modes and task curvature, not variance alone.

In a fixed-Jacobian, full-batch squared-loss linearization, prediction error
obeys de/dt=-K e. Along a kernel eigenmode u_k it decays as
e_k(t)=e_k(0)exp(-lambda_k*t). Alignment of needed corrections with learnable
high-eigenvalue modes gives a precise optimization meaning to matching. Feature
learning changes K, and stochastic/cross-entropy dynamics are different;
this linearized conclusion is not a universal neural learning law.

Related primary source: Jacot, Gabriel and Hongler (2018),
[Neural Tangent Kernel](https://arxiv.org/abs/1806.07572). Our proposed diagnostics
use the task/resource metric and event observations of §§358–360, not a claim
that the NTK result itself is novel.

## 365. Weights store shared rules; activations need not be unexplained residue

Weights choose the transition/readout rules learned across examples.
Activations instantiate those rules for the current content, context, time and
state. A useful activation can be an explanatory feature, posterior statistic,
memory, routing key, uncertainty or explicit prediction error. It is not
generally leftover surprise. Scalar surprisal -log p(Y|X), a signed innovation
and a learned feature are different objects.

At a correctly specified binary predictor with success probability p,
E[p-Y|X]=0 while conditional entropy can be positive. Thus unpredictable
surprise remains even when the expected parameter gradient is zero. Learning
cannot absorb irreducible randomness except by memorizing sample noise.

If the architecture explicitly represents innovation
r=observation-E[observation|past], then E[r|past]=0. For a FIXED target Y and
growing causal information F_t, m_t=E[Y|F_t] is a martingale, and changes
m_t-m_(t-1) are orthogonal to past measurable functions in L2. This is a rigorous
predictable-part/innovation decomposition. It does not apply automatically to
successive neural layers: a deterministic compression can discard information,
and changing next-token targets are not the same fixed Y.

Design implication: test predicted content PLUS calibrated evidence/innovation
channels, rather than subtracting a guessed prediction everywhere. In the
Gaussian case of §359, precision and weighted evidence are exact sufficient
messages. More general conditional dependencies require learned interactions.

## 366. Sampling belongs to reception and observability, not depth by itself

Discrete network layers are compositions, not automatically time samples.
A residual update z_(l+1)=z_l+Delta*f(z_l,t_l) approximates an ODE only under
an explicit vector field, step sizes and regularity assumptions. Arbitrary
layers need not describe such a flow. A uniquely solvable smooth deterministic
ODE flow preserves distinctions in its state; readouts, event resets and
dimension changes can still compress them. Continuous depth is not unlimited
free expressive capacity, and numerical ODE evaluations cost work.

For a between-event linear flow E(t), scalar/vector reception C and independent
noise covariances Ri, observations y_i=C E(t_i)z0+noise have information Gramian

    G = sum_i E(t_i)^T C^T Ri^-1 C E(t_i).

The initial state is locally identifiable from these observations exactly when
G has full rank. A static scalar projection may have rank one; rotation can
expose another direction at a later reception. Sampling once per full rotation
aliases those directions; an appropriate non-aliased reception can recover them.
This is a precise useful role for rotating projections and repeated arrivals.

Classical bandlimited sampling applies only if its signal assumptions hold.
An event process with state-dependent arrivals instead has information in
the times, marks AND silence. For conditional event intensity lambda(t), the
negative log likelihood includes -sum_i log lambda(t_i)+integral lambda(t)dt,
plus the mark likelihood. Ignoring the integral drops silence evidence. Current
analytic flows/races avoid some numerical integration, but do not certify
cheap inference or learning for an arbitrary intensity/function family.

Linear-Gaussian state/uncertainty propagation is established prior art; see
[Kalman (1960)](https://doi.org/10.1115/1.3662552).
The proposed substrate question is whether sparse event writes, analytically
evolving/observable state and counterfactual learning can combine these useful
structures at better measured quality/work. Test reception Gramian conditioning,
jitter, aliasing, pooling sufficiency and finite route credit before long fits.

## 367. Hierarchical resolution of explaining factors, rather than input sampling

The user's intended scale is explanatory resolution: coarse shared causes,
conditional substructure and finer variations. An exact multiresolution analogy
exists when explaining-factor information forms nested sigma-algebras
F0 subset F1 subset ... subset FL. For an L2 target function f, define

    g_l = E[f | F_l],   d_l = g_l - g_(l-1).

Distinct detail terms are orthogonal, and

    ||f||_2^2 = ||g_0||_2^2 + sum_l ||d_l||_2^2 + ||f-g_L||_2^2.

This is a conditional-expectation multiresolution decomposition. Unlike Fourier
frequency, its scales are information partitions and conditional explanatory
detail. It does not establish that ordinary neural layers produce nested
factor information, or that layer count fixes a factor resolution. Unknown
latent factors are not automatically available as observed inputs. A receiver
must infer them economically from its causal observations.

For a Markov generative hierarchy Z_(l+1) -> Z_l -> observations X, propagating
only E[Z_l|X] is generally insufficient for the next nonlinear factor. If
E[Z_(l+1)|Z_l]=h(Z_l), then

    E[Z_(l+1)|X] = E[h(Z_l)|X], not generally h(E[Z_l|X]).

For h(z)=z^2, the difference is Var(Z_l|X). Thus a message's uncertainty can
be exactly the information a higher explanatory factor needs. Several modes,
moments or conditionally selected messages can preserve it; raw mean pooling
can erase it. Full posterior propagation is sufficient under the assumed
hierarchy but may be expensive, so structured approximations need task tests.

Learning speed is a separate issue: shared factors receive many examples and
well-conditioned predictive directions learn faster (§364); private rare
details get less exposure. There is no general law that later layers learn
at a lower physical or hierarchical sampling rate. Sparse specialist capacity
should be allocated to conditional details whose held-out predictive gain pays
their discovery, delivery and learning work.

Pursue this with known-factor generators: vary coarse/fine dependence,
heterogeneity, uncertainty and irrelevant gaps independently; measure which
factors are accessible at each block using calibration-only probes. Compare
content-only means, identified channels, confidence/moment messages and
low-rank interactions, then confirm on real streams. Current native feedback
means even its early blocks read prior deep context; a clean factor-to-layer
ordering is a hypothesis, not an established architecture property.

## 368. Existing generator–recognizer formalisms and our implementation question

The most literal precedent is the [Helmholtz machine / wake–sleep](https://www.cs.toronto.edu/~hinton/absps/ws.htm):
a hierarchical stochastic generator and a learned recognition pathway infer
latent causes. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
formalizes a learned recognition distribution q_phi(z|x) for a generative
p_theta(x,z). Its standard identity is

    log p_theta(x) = ELBO(theta,phi;x)
                    + KL(q_phi(z|x) || p_theta(z|x)),
    ELBO = E_q[log p_theta(x,z)-log q_phi(z|x)].

Thus a dual receptor approximates a POSTERIOR, not necessarily the generator's
network or its exact inverse. The likelihood/reconstruction, latent prior and
posterior uncertainty determine what should be communicated. Hierarchical
models can carry distributions or sufficient statistics rather than only
point estimates. Current Sleeping Machine fits are supervised predictors, not
implemented variational recognizers merely because this interpretation is useful.

[Factor graphs and sum–product](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf)
give the explicit local message algebra for a factored joint law: messages
multiply compatible evidence and sum/integrate over hidden variables. Tree
inference is exact; loops and restricted messages need approximation. This
helps specify when pooling is valid and which conditional interactions a small
event message must retain. A temporal race does not automatically implement
every required sum–product or marginalization operation.

For sequence learning, [spectral HMM identification](https://arxiv.org/abs/0811.4413)
provides a precise connection between an observable predictor and a latent
generator under rank/separation assumptions. Its guarantees depend on
conditioning; they do not promise efficiently learnable receptors for all
hierarchical distributions. Predictive state and observability are natural
bridges to persistent event representations without recovering every latent
parameter of a generator.

Our candidate contribution is the execution/learning construction: useful
recognition/prediction messages processed through delays, analytically evolving
state, independent sparse routes and counterfactual credit, with dormant
capacity and complete resource accounting. These existing theories help choose
what information to retain. They are not architectural substitutions or evidence
of a benchmark/energy advantage before completed matched experiments.
