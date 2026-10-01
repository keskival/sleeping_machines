# From a generator to necessary receiver operations

Global §§369–372. A scoped attempt at the missing bridge: generator assumptions
-> necessary predictive information -> representation/operators -> learning
and resource tests. These conditions constrain implementations; they do not
provide a universal recipe for an optimal efficient differentiable learner.

## 369. Minimal predictive state comes before a module catalogue

For histories h,h', define h~h' if their conditional future laws agree for
every permitted future input/action sequence. A sufficient predictive state
must distinguish inequivalent histories. In a supervised task one can instead
use the relevant future target laws; that weaker equivalence can compress more.
Full hidden-generator recovery is unnecessary if two causes predict identically.

Minimal state realization and statistical sufficiency concern this equivalence,
not a particular neuron type. Finite precision, allowed operators, training
data, stability and resource budgets determine how costly its representation
is. A single arbitrary real number could encode unbounded discrete history;
that is not a finite-precision, stable, trainable memory construction.

For a continuous family of observed histories controlled by alpha, let b(alpha)
be a vector of future conditional probabilities or expected test functions.
If b=g(s(alpha)) with differentiable n-dimensional state s, then

    rank(Db/Dalpha) <= n.

This yields a LOCAL state-dimension requirement under smoothness. It does not
lower-bound every discontinuous/infinite-precision code. Linear observable
predictor rank also has realization results; it cannot be imposed unchanged
as the minimal dimension of every nonlinear predictor.

## 370. Three operator requirements with falsifiable witnesses

**Order roles.** Any symmetric pooling T(x1,x2)=T(x2,x1) identifies swapped
observations. If the target distinguishes them, no decoder of T can be correct
for both. Some role/order-sensitive operation is necessary: identified channels,
causal state, role-dependent projections or another distinction-preserving
mechanism. This does not uniquely require our races or recurrent construction.

**Conditional uncertainty.** Under a nonlinear hierarchy h(z)=z^2,
E[h(Z)|X]=E[Z|X]^2+Var(Z|X). If two contexts have the same conditional mean
but different variance, a mean-only packet is insufficient. A second moment,
distribution, mixture or another equivalent statistic is necessary for that
prediction. In linear-Gaussian independent evidence, precision plus weighted
evidence is sufficient (§359); arbitrary correlated/non-Gaussian laws need more.

**Robust order retention across unbounded silence.** Suppose all retained
directions contract with rate at least gamma>0, initial state separation is
at most2B, and the decoder has a uniform Lipschitz constant L. The separation
of output predictions after a gap delta is at most

    2 L B exp(-gamma*delta).

A fixed positive output margin cannot survive arbitrarily large gaps under
these assumptions. A noncontracting memory direction, renewed information,
bounded admissible gaps or a decoder whose sensitivity grows with the gap is
needed. Unbounded amplification changes stability/noise/precision costs.
Protected modes supply one solution; local identity preservation does not
alone prove whole-model gap robustness, because routes and event writes change.

These witnesses make design changes testable: the missing statistic/symmetry/
memory direction is specified before adding machinery. Multiple implementations
can satisfy the same requirement; benchmark costs decide between them.

## 371. Distinguish genuine redundancy from unexposed useful freedom

Conditional Fisher information is

    I(theta)=E[grad_theta log p_theta(Y|H) grad_theta log p_theta(Y|H)^T].

A null direction is first-order unobservable under this distribution. It is
not automatically an exactly redundant parameter or safely removable module.
Four cases must remain separate:

- Structural equivalence/gauge: an entire parameter path leaves all relevant
  predictions unchanged. Reparameterization may remove it without loss.
- Distribution/task equivalence: predictions differ only on unobserved or
  irrelevant inputs. A changed deployment law can make them useful again.
- Local flatness: for f=(a*b)x at a=b=0, both first derivatives vanish while
  the mixed derivative is nonzero. A joint movement creates a useful function;
  this initialization also gives ordinary gradients no way to start learning.
- Poor exposure/conditioning: a rare route or saturated gate can hide useful
  directions. More data, appropriate nonzero feature initialization or route
  credit can expose them; none is guaranteed to improve held-out quality.

Likewise two independent fair bits can each be individually uninformative
about their XOR while jointly determining it. Pruning from single-feature
correlation alone can remove complementary information. Use grouped ablations,
cross derivatives and independent task-loss tests, alongside Fisher spectra.
Inspect which producer/event graphs the teacher actually credits. A positive
local route teacher is not a guarantee of favorable discrete suffix loss.

## 372. A resource-aware receiver specification

For each known generator, specify a target distortion (e.g. conditional KL),
finite precision/noise, causal access, operator grammar and learning budget.
Then compare sufficient-state candidates under whole inference/learning work,
traffic, memory and empirical learnability. Predictive rate–distortion supplies
a representation objective; identifiability/observability checks what can be
learned; operator and routing contracts check what can be executed.

The proposed evidence chain is:

1. Derive an information witness or necessary symmetry/rank/memory condition.
2. Build the smallest integrated variants satisfying or violating it, retaining
   other mechanisms and counting every added operation/parameter/teacher.
3. Verify information/gradient/checkpoint contracts before fitting.
4. Measure cross-generator generalization, independent prediction and resources.
5. Test the useful variant on recognized real distributions before scaling.

Our first implementation uses the protected-memory requirement and shared
statistical exposure. Next candidate packets preserve uncertainty/conditional
interaction statistics. The resulting minimum depends on the declared
generator/task/operator family; a generally optimal module catalogue is still
an open problem, not something this note has proved.
