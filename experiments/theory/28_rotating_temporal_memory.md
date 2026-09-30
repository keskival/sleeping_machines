# Trainable temporal phase without adding event traffic

Derived and implemented 30 September 2026. E140 extends the existing common
receiver memory; it does not replace its actual winning vectors and delays.
The normalized spectral operator here is narrower than the general signed
affine state update in §219, but nests the strongest trained model exactly.

## 220. A phase-bearing message memory with a linear-work scan

For one addressed receiver, input vectors x_i, positive counts c_i and one
time scale tau, define a pairwise rotation block R(theta). Let

\[
 m(t)= {\sum_{i:t_i\leq t} c_i e^{-(t-t_i)/\tau}
            R(\theta(t-t_i)/\tau)x_i
        \over \sum_{i:t_i\leq t}c_i e^{-(t-t_i)/\tau}+\epsilon}.
 \tag{220.1}
\]

Each coordinate pair has its own dimensionless phase theta; each receiver
time bank has its own phase vector. At theta=0 this is the old normalized
exponential memory, including its epsilon. Theta=1 means one radian of
rotation over one memory time scale; the physical angular frequency is
theta/tau. This uses a comparable parameter scale in fast and slow memories.

Conjugate input events by their absolute arrival times:

\[
 \widetilde x_i=R(-\theta t_i/\tau)x_i,\qquad
 n_i=e^{-\Delta_i/\tau}n_{i-1}+c_i\widetilde x_i,\qquad
 M_i=e^{-\Delta_i/\tau}M_{i-1}+c_i,
 \quad m_i=R(\theta t_i/\tau)n_i/(M_i+\epsilon). \tag{220.2}
\]

The equality follows from the rotation group law. Receiver boundaries reset
n,M; a stable key sort retains actual chronological/tie order within each
receiver. The existing affine-prefix algorithm evaluates the conjugated
numerator and ordinary mass in fewer than 2P vector combines. Extra rotations
are O(P K d) for P packets,K time banks,d components. There is no event-pair
matrix or work on empty time ticks. An online receiver can instead apply the
relative rotation R(theta Delta/tau) directly to its previous numerator;
that form also avoids large absolute phase arguments.

E140 uses the scan for training and preserves the existing query semantics.
The absolute-phase implementation can lose floating precision at very large
timestamps/frequencies; a relative online kernel or receiver-local time origin
is the appropriate systems implementation. Learned delays alter subsequent
arrival intervals, and therefore change both the rotations and future races.
Time is an argument of the computation, not just a scheduling annotation.

## 221. The phase teacher exists at the real-memory initialization

Let J be the 90-degree rotation generator on one coordinate pair. At zero
phase, the exact local memory derivative is

\[
 {\partial m(t)\over\partial\theta}\bigg|_0
 =J\,{\sum_i c_i e^{-\ell_i/\tau}(\ell_i/\tau)x_i
             \over M(t)+\epsilon},\qquad \ell_i=t-t_i. \tag{221.1}
\]

For packet teacher g, the local phase teacher is the inner product of g with
this rotated lag-weighted trace. Thus a zero phase can be recruited from the
ordinary real-vector memory. It need not wait for another initially zero gate.
The teacher can still vanish when no past vector contributes, or when the
lag trace and class teacher have the relevant orthogonality; nonzero is not
universal. The contract measures it in the actual fitted SHD model.

By comparison, retaining only a cosine-weighted scalar feature gives
partial_theta cos(theta ell/tau)|_0=0. Discarding its sine/paired coordinate
creates a first-order dead zone at exactly the natural zero-frequency
initialization. Keeping both coordinates avoids that particular obstacle.
This is a concrete representation/initialization distinction, not an assertion
that every oscillatory architecture must initialize at zero.

The source gradient and the memory phase teacher are exact local chain rules.
The common model still supplies its existing surrogate for changing hard
winners. The tiny whole-network finite probe improves fitting loss but differs
from the surrogate prediction; it is not promoted to exact boundary credit.

## 222. Conditional stability and measured initialization

Because R is orthogonal and the normalized positive weights sum to at most
one, ||m(t)||_2 <= max_i ||x_i||_2. In infinity norm a coordinate-pair rotation
has row sum at most sqrt(2), so the memory operator norm is at most sqrt(2).
For a fixed event count, receiver, clock and winner schedule, the existing
value row normalization and tanh Lipschitz bound therefore give a correction
Jacobian norm at most sqrt(2) in the complete packet infinity norm.

With residual factor alpha=beta/L, alpha sqrt(2)<1 yields

\[
 (1-\alpha\sqrt2)\|\delta x\|_\infty
 \leq\|D f_l\,\delta x\|_\infty
 \leq(1+\alpha\sqrt2)\|\delta x\|_\infty. \tag{222.1}
\]

For fixed beta, the complete conditional stack bounds remain finite and
positive as depth grows, tending to exp(+-beta sqrt(2)). These constants
exclude key/clock/phase parameter variation, normalized mass dependence on
changed counts and schedule boundaries. They do not imply supervised
observability, favorable stochastic optimization or new-speaker transfer.
The full teacher and realized finite changes still need measurement.
In particular, these bounds apply to the realized conditional value program;
they do not bound every extra zero-forward surrogate routing derivative.
The dual statement is a packet-teacher bound in the 1-norm: the transpose of
the conditional stack transports a nonzero output teacher between the same
product bounds. This follows from ||J^T||_1=||J||_infinity and the corresponding
inverse bound. It is not a dimension-free 2-norm singular-value claim, nor a
bound on sums of shared-parameter gradients across packets. Query projection
and parameter sharing can still hide or cancel those transported teachers.

## 223. Preserve a trained optimizer when expanding payload capacity

Temporal operator richness and payload width are separate capacity choices.
A principled widening can duplicate a payload: T x=(x,x). Split each old input
coefficient between the duplicate columns using nonnegative fractions rho and
1-rho, and duplicate output rows. Then W_wide T=T W for hidden maps, while
H_wide T=H for the head. Absolute row sums are preserved by this sign-preserving
split, so the old normalized-value constraint remains valid. Biases, source
embeddings, time banks and phase coordinates duplicate consistently. Normalizers
must be mapped too; copying only the weights does not preserve a calibrated model.

Independent split fractions can expose both copies to nonzero class teachers,
while keeping the initial function identical. Merely adding zero hidden rows
and zero head columns creates a product-of-zero learning dead zone. Duplication
with an observable split avoids that particular obstruction; it does not
guarantee independent features will subsequently emerge.

Changing parameter shapes and starting a fresh Adam would confound capacity
with the optimizer history, which already matters in our continuations. Instead
parameterize widened maps as W_wide=E_rho(W_old)+Delta W, with zero Delta W.
Keep the old parameter objects and moments, and optimize the new residuals as
a separate block. If F(Theta,0)=F_parent(Theta) in the relevant smooth region,
then partial_Theta L(Theta,0)=partial_Theta L_parent(Theta) by identity of the
functions. The original optimizer receives its original gradient/history;
the new block can receive credit through both exposed copies. At hard routing
boundaries the policy credit still needs its declared estimator, and all wider
maps/state/work must be charged. This gives an explicit capacity intervention
without discarding the strongest checkpoint or assuming width alone will solve
SHD. E140 changes temporal phase at width 32; this widening is an analytically
specified subsequent intervention, not an implemented benchmark result.

E140's completed contract records: zero initial logit error; identical actual
winners/clocks; serial/parallel memory discrepancy 4.44e-16; local phase-teacher
discrepancy 1.09e-14 in float64. All eight trained-checkpoint layers have
nonzero phase credit. There are 384 new phase scalars. The 1e-5-L2 restored
whole-network phase probe changes fitting NLL by -7.15e-7, versus the surrogate
prediction -1.40e-6. No probe update or development-label update is retained.
The corresponding same-budget fine-source continuation is the control for
quality; the highest common-model score remains the independent acceptance gate.
