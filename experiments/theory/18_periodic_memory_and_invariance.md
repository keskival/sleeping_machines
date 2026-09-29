# Periodic memory, protected composition and nuisance variation

## 181. A periodic payload with exact local credit transport

The first shared-model arithmetic screen omitted the rhythm computation used
by E41. Increasing generic depth cannot certify that its optimizer reconstructs
that missing primitive. We instead extract the primitive, while making depth
an independent task configuration. E121 uses two carrier layers for arithmetic;
speech retains eight. The fitting weights are separate in every task.

### Affine actions on a circle

Let a node carry a phase $\phi\in\mathbb R/P\mathbb Z$. An observed symbol $x$
applies

$$T_x(\phi)=s\phi+a_x\pmod P,\qquad s\in\{-1,+1\}.\tag{181.1}$$

The offsets $a_x$ are learned. In the vector representation
$z=\exp(2\pi i\phi/P)$, this is a rotation of $z$ when $s=1$ and a rotation
of its conjugate when $s=-1$. The two real coordinates remain on the unit
circle. Each event transmits one updated state; there is no all-pairs attention
or hidden time grid.

More generally write a transition as $(s,a)$. Composition is

$$(s_2,a_2)\circ(s_1,a_1)=(s_2s_1,s_2a_1+a_2\pmod P).\tag{181.2}$$

This associative law permits a sequential event implementation or an exact
reduction over observed events. It is a subgroup of circle isometries. E121
uses a fixed reflection $s=-1$, so for a sequence of length $m$ and initial
phase zero,

$$\phi_m=d+\sum_{j=1}^m s^{m-j}a_{x_j}\pmod P.\tag{181.3}$$

The shared input adapter supplies position-tagged symbols for the three
operands, giving 51 independently initialized offsets, 17 class phases and
one offset $d$: 69 learned scalars. It supplies the period $P=17$. It does
not initialize phases from operand values, supply the sum as an input, or
use the arithmetic formula in teaching. Position tags are already present
in the plain E120/E121 controls.

### A realizable computation and its credit

For three symbols, (181.3) is $d+a_A-a_B+a_C$. A modular sum is realizable
by choosing $a_A=A$, $a_B=-B$, $a_C=C$ and suitable class clock phases.
This is an expressivity witness, **not the initialization or update rule**.
The offsets start independently at random and only labeled fitting examples
adjust them. For E41 the correspondence is
$a_A=d_1[A], a_B=e_1[B], a_C=e_2[C], d=-d_2$.

For class clock $k$ at $\theta_k$, its waiting time and winning answer are

$$w_k=(\theta_k-\phi_m)\bmod P,\qquad \hat y=\arg\min_k w_k.\tag{181.4}$$

The earliest clock determines the answer. Distinct losing clocks are not
averaged into the winning phase. The present query implementation calculates
all $C$ clock scores; a dedicated clock scheduler is not implemented.

On any smooth phase chart,

$$\left|\frac{\partial\phi_m}{\partial\phi_j}\right|=1,\qquad
 \frac{\partial\phi_m}{\partial a_{x_j}}=s^{m-j}.\tag{181.5}$$

Thus state credit neither vanishes nor explodes with sequence length. Circle
distance handles chart wraps; hard class selection still has switching
boundaries. If a symbol occurs repeatedly, its parameter derivative is the
**sum** of its signed occurrences, which can cancel or grow. The isometry
bound is for state transport and each occurrence, not a bound of one on every
tied-parameter gradient. It does not prove global convergence.

E121 uses the error-only local timing surrogate inherited from E41. For target
$y$, margin $h$ and a noisy teaching phase $\tilde\phi$, define

$$g=\operatorname{wrap}_{(-P/2,P/2]}(\theta_y-\tilde\phi-h),\quad
 u=\eta\operatorname{sign}(g)/(m+3).\tag{181.6}$$

Each occurrence gets $\Delta a_{x_j}=s^{m-j}u$; also $\Delta d=u$ and
$\Delta\theta_y=-u$. Away from wraps and the cusp, these directions descend
$|g|$. For shared symbol parameters put
$q_x=\sum_{j:x_j=x}s^{m-j}$: the first-order change is
$\Delta g=-u(2+\sum_xq_x^2)$. This explains why credit must accumulate repeated
occurrences rather than overwrite them. The fixed step can cross a boundary;
it is not an exact line search. A correct deterministic phase race skips
teaching. Internal teaching noise has standard deviation
$2.21\exp(-U/20000)$ after $U$ mistaken updates and is absent at inference.
Neither this noise schedule nor the denominator is a convergence theorem.

The phase update is not categorical SGD. The shared neural branch uses
categorical cross-entropy and Adam; the phase memory uses local timing credit.
Both see only fitting labels. This explicitly combines two learning rules,
as the shared pointer and conditional memories already do.

### Preserve a winning computation when adding a neural branch

Let phase scores be $q_k=-w_k$. The coupled readout is

$$z_k=q_k+B\tanh(h_k/B),\qquad B\ge0,\tag{181.7}$$

with the correction defined as zero for $B=0$. Here $h_k$ is the ordinary
shared-core head. E121 fixes $B=0.25$ before fitting. It does not choose this
bound on unseen outcomes.

**Margin certificate.** If the phase winner $k_*$ has lead
$\Delta=q_{k_*}-\max_{k\ne k_*}q_k>2B$, then $k_*$ also wins (181.7).

**Proof.** Any competitor gains at most $2B$ relative to the winner, so its
corrected deficit is at least $\Delta-2B>0$.

This is a certificate of preservation, not correctness. It can preserve a
confident wrong phase answer; local phase teaching remains necessary. Small
clock margins allow the neural branch to change decisions. Its own derivative
is $\operatorname{sech}^2(h_k/B)$, so large head scores saturate neural credit.
This limitation is preferable to silently claiming an unrestricted trainable
mixture. E121 reports phase-only accuracy, coupled accuracy, margin coverage
and how often the neural branch changes the winner.

All configured carrier layers execute and receive their own loss credit.
The phase memory is an additional observed-event state branch. Perfect coupled
arithmetic would establish preserved capability in the shared implementation;
it would not establish that the generic deep carrier independently discovers
arithmetic, or that the phase state has already become an internal hidden route.

### Closing the uncovered small-margin cases

The initial completed E121 run exposes the certificate's boundary: its phase
memory is perfect on all 3,440 unseen tuples, but the fixed $B=0.25$ correction
changes some answers whose lead is below $2B$. No violation of the certificate
occurs. This is a failure of incomplete protection, not failed phase learning.

An optional guarded readout sets

$$B(x)=\min\{0.25,\Delta(x)/4\}.\tag{181.8}$$

For every positive phase lead, the corrected lead remains at least
$\Delta-2B(x)\ge\Delta/2>0$. At a tie, $B(x)=0$ and the existing deterministic
tie-break is retained. This rule uses the model's clock lead, never a label.
It therefore preserves **every** phase prediction by construction. The neural
branch can adjust confidence within that class decision; it cannot repair a
wrong phase winner. Such errors must be corrected by phase teaching or by a
separate learned choice of computational primitive. E121 retains the fixed-bound
run and separately fits this guarded variant with the same schedule. This is
deliberate computational ownership, not evidence of a neural arithmetic gain.

## 182. Speech augmentation as a specific nuisance hypothesis

The eight-layer E119 model already learns speech: final development accuracy
is 151/256, versus 5% uniform chance. Its fit/development gap motivates a
generalization intervention. E122 continues the same final checkpoint and
Adam state with 2,048 fitting utterances. Two arms have identical fitting IDs,
order RNG and four-epoch schedule; only one perturbs fitting observations.

### Time scaling and the objective it actually changes

For input event times $t_i$, let $T_u$ send $t_i\mapsto e^u t_i$ with
$u\sim\operatorname{Uniform}[-a,a]$, $a=0.15$. This is a hypothesis that modest
tempo changes preserve the utterance label. It uses no future duration to
normalize the prefix. Conditional on the drawn $u$, order is preserved.

On a smooth region with unchanged hard routes, define the scaling generator
$D=\sum_i t_i\partial_{t_i}$. A symmetric expansion gives

$$\mathbb E_u\ell(z(T_ux),y)=\ell(z(x),y)
 +\frac{a^2}{6}D^2\ell(z(x),y)+O(a^4).\tag{182.1}$$

For categorical residual $r=p-e_y$ and score Hessian
$H=\operatorname{diag}(p)-pp^T$,

$$D^2\ell=(Dz)^TH(Dz)+r^TD^2z.\tag{182.2}$$

The first term penalizes sensitivity in predictive directions; the second
need not be positive. Therefore augmentation is not universally equivalent
to a positive Jacobian regularizer. At race switches this smooth expansion
does not apply; the sampled objective remains defined and explores changed
routes. Those are precisely reasons to test the intervention against a
matched continuation instead of asserting improvement from the formula.

The exponential memory $\exp(-\Delta t/\tau)$ would be equivariant if both
time and its time constants were rescaled. The present model fixes the input
time constants and constrains layer delays, so input scaling alone is **not
an exact symmetry**. Augmentation asks the learned decision to tolerate the
nuisance despite this sensitivity.

The other intervention translates the 40 channel indices by a sampled integer
in $\{-2,-1,0,1,2\}$, dropping events outside the range. It approximates modest
frequency variation. Cropping breaks invertibility; fixed receiver groups and
symbol embeddings break exact translation equivariance. Thus the joint arm
tests a combined nuisance hypothesis and cannot identify which transform helps.
Count marks are retained on surviving packets. Empty crops fall back to the
original channel sequence.

### How the result must be read

Evaluate clean fitting utterances and two disjoint groups of 256 held-out
utterances from training speakers 3/6. The first group is the existing E119
development slice; the additional group checks sensitivity to that repeatedly
inspected slice. These groups contain the same speakers and are not new-speaker
replications. Neither is the official SHD test set. Record final and intermediate
curves, retain both arms, and compare paired predictions. More fitting data and
the continuation schedule are shared changes; their effect relative to E119 is
not separately identified. A single matched seed is a mechanism screen, not an
energy or state-of-the-art claim.

## 183. Certifying a learned modular rule with min-plus composition

Accuracy measures the answer on evaluated tuples. The learned phase parameters
also permit a compact certificate over **all** tuples of the declared task.
This audit uses the known target algebra after training; that algebra is not
part of the model's teaching rule or input encoding.

For three position-specific codebooks, write fitted phases as

$$a_{1,A}=\kappa A+c_1+\epsilon_1(A),\quad
 a_{2,B}=-\kappa B+c_2+\epsilon_2(B),\quad
 a_{3,C}=\kappa C+c_3+\epsilon_3(C)\pmod p,\tag{183.1}$$

and class phases as $\theta_y=\kappa y+c_\theta+\epsilon_\theta(y)\pmod p$.
Here $\kappa$ is an integer winding. For prime $p$, any nonzero winding
permutes the ideal equally spaced class clocks. We fit intercepts by circular
means and choose the winding with the smallest sum of worst residual magnitudes.
All quantities are read from the trained parameters, without outcome-based
selection on development examples.

For a composite period the exact condition is $\gcd(\kappa,p)=1$: multiplication
by $\kappa$ is an automorphism of $\mathbb Z_p$ precisely for those windings.
Otherwise distinct sums differing by $p/\gcd(\kappa,p)$ have the same ideal
phase. No readout of that ideal phase alone can distinguish them. Residual
features could add information, but would no longer inherit the equally spaced
clock guarantee. This gives an explicit representation constraint when extending
the primitive to other periods, independently of carrier depth or optimizer.

Let $c=c_1-c_2+c_3+d$, $h=(c_\theta-c)\bmod p$ and
$E(A,B,C)=\epsilon_1(A)-\epsilon_2(B)+\epsilon_3(C)$. For
$y=A+B+C\pmod p$, the unwrapped target wait is

$$w_y^*=h+\epsilon_\theta(y)-E(A,B,C).\tag{183.2}$$

Its remainder modulo $p$ is exactly the actual target waiting time. A simple
sufficient certificate is
$\sum_{j=1}^3\|\epsilon_j\|_\infty+\|\epsilon_\theta\|_\infty<\min(h,1-h)$
when $0<h<1$. This locates the phase strictly between the preceding ideal
clock, including its perturbation, and the target clock. It may be too loose
because it combines errors that cannot occur together at a given sum.

### A tighter certificate preserves the target constraint

For arrays on $\mathbb Z_p$, define cyclic min-plus and max-plus convolution:

$$(f\star_{\min}g)(y)=\min_{a+b=y\bmod p}\{f(a)+g(b)\},\qquad
 (f\star_{\max}g)(y)=\max_{a+b=y\bmod p}\{f(a)+g(b)\}.\tag{183.3}$$

Both are associative. The exact extrema conditional on target $y$ are

$$L_y=(\epsilon_1\star_{\min}[-\epsilon_2]\star_{\min}\epsilon_3)(y),\quad
 U_y=(\epsilon_1\star_{\max}[-\epsilon_2]\star_{\max}\epsilon_3)(y).\tag{183.4}$$

Hence all target waits before wrapping belong to

$$[\ell_y,u_y]=[h+\epsilon_\theta(y)-U_y,\;
                 h+\epsilon_\theta(y)-L_y].\tag{183.5}$$

Sort the **actual** class clocks, not their ideal order, and let $g_y$ be the
forward circular spacing from the preceding actual clock to $\theta_y$.

**Clock-cell certificate.** If for every class $y$,

$$\ell_y>0\quad\text{and}\quad u_y<g_y,\tag{183.6}$$

then every tuple has the correct winning class. The smallest
$\min_y\{\ell_y,g_y-u_y\}$ is a phase slack for the certificate.

**Proof.** Equation (183.5) bounds the target wait for every tuple with that
sum. Equation (183.6) places its phase after the preceding actual clock and
before the target clock. The target is therefore the unique first clock. The
strict inequalities exclude all clock boundaries. Apply this for every $y$.
The guarded neural readout preserves that winner by §181.

For $m$ signed symbol codebooks this dynamic program costs $O(mp^2)$ and
$O(p)$ working state, compared with $p^m$ input combinations. The implementation
currently materializes an $O(p^2)$ temporary for vectorized pair composition;
row-wise reduction would attain the stated working-state bound. It records
both this certificate and an exhaustive $p=17,m=3$ verification of the extrema.
The audit is not a training algorithm. A certificate failure need not imply
wrong answers: wrapping or loose decompositions may prevent this sufficient
condition from succeeding. A successful certificate proves the learned finite
task computation, not discovery from every initialization or generalization
to untrained moduli. Its useful design principle is to propagate attainable
error ranges under a compositional constraint, rather than add unrelated
worst-case errors across the whole tree.

The E121 algebra audit succeeds for the trained seed-6 representation: winding
$\kappa=4$, target offset $h=0.35708$, minimum clock-cell slack $0.01197$ phase
units. The crude sum of maximum residuals is $0.58978$ and fails its sufficient
bound, while the target-constrained min/max composition proves every class.
Exhaustive checking finds zero errors among all 4,913 triples and agrees with
both computed extrema. This is a concrete case where preserving reachable
combinations in the formalism gives information lost by a single global bound.

## 184. Computational ownership preserves both answers and work

The guarded composite proves which primitive supplies the modular answer.
It also reveals an unnecessary cost: the generic carrier executes and adjusts
confidence even though it cannot change the winning class. Leaving that branch
active cannot inherit the old rhythm computation's sparse work advantage.

The common model now permits an explicit phase-only configuration: one local
periodic state path, a class-clock race, and no generic carrier layers. It uses
the same `SharedEventModel` query interface and the same `PhaseMemory` local
teacher. The decision to configure this primitive is made per task before
training, not selected per example from its label. The phase-only path returns
a unit-circle vector payload and the class-clock scores.

Because the phase teaching rule has no dependence on the neural branch, with
identical example-order and exploration RNG states its parameter trajectory
is identical by induction over fitting examples. E124 checks equality of the
whole fitted phase state against E121's guarded composite. Thus removing the
unused carrier preserves its phase computation and all class decisions.
The resulting configuration has 69 learned phase scalars and no Adam parameters.
It does not provide the speech model's representation capacity: the eight-layer
speech configuration still needs the generic carrier. Configurable computational
primitives and depth belong to the same model family; they need not all execute
on every task.

### Work accounting must follow the configured computation

For a carrier width $d$, $K=3$ alternatives and $E$ packets, the inference value
maps cost $Ed(2d+1)$ MACs per layer; routing costs $EK(2d+1)$ MACs. Training also
evaluates the losing value maps. Each counted affine scan composition touches
$K(d+1)$ scalar numerator/mass coordinates. Readout whitening costs $(d+1)^2$
MACs per query, followed by $(d+1)C$ for the head. Weight normalization, memory
division, exponential decays, sorting and lookup are additional work. E124's
ledger includes them as declared estimates instead of calling one entire
width-32 vector message a single scalar operation.

The phase-only path has $O(E+C)$ arithmetic plus the present $O(C\log C)$
clock-margin diagnostic. It returns all class scores; a physical event scheduler
is not assumed. Pointer search still has $O(ER)$ candidates for $R=4$ offsets,
even when only one winning destination is read. Compare full configured work,
not the historical native primitive's cost attached to a different executable.

The work plots use a declared logical-operation estimate: two units per MAC,
one per other scalar arithmetic/nonlinear operation or estimated sort comparison.
They report logical memory reads separately. These units do not give equal
energy to a remainder, exponential, SRAM read and GPU multiply. The ledger does
not estimate backward or optimizer work from inference counts. Hardware joules
and latency measurements must follow the actual workload and memory boundary.
