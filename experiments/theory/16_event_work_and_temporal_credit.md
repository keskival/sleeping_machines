# Event work and temporal credit

## 173. Linear-work event memory and its exact reverse credit

E118's memory is already an event recurrence. Its first parallel implementation
performed more work than that recurrence requires. For one receiver and memory
bank, write the numerator and mass together as a vector:

\[
 z_i=a_i z_{i-1}+v_i,\quad
 a_i=\exp(-\Delta t_i/\tau),\quad v_i=c_i[h_i,1].
\]

A receiver boundary sets its first coefficient to zero. Composition of two
successive affine maps is associative:

\[
 (a_2,v_2)\circ(a_1,v_1)=(a_2a_1,v_2+a_2v_1).
\]

**Construction.** Pair adjacent maps, recursively scan the pairs, and reconstruct
the missing even prefixes from the preceding odd prefix. In zero-based indexing,

\[
 A_j=a_{2j+1}a_{2j},\quad V_j=v_{2j+1}+a_{2j+1}v_{2j},
\]
\[
 z_{2j+1}=\operatorname{scan}(A,V)_j,\qquad
 z_{2j}=v_{2j}+a_{2j}z_{2j-1}\quad(j>0),\quad z_0=v_0.
\]

The number of vector combines satisfies

\[
 C(E)=C(\lfloor E/2\rfloor)+\lfloor E/2\rfloor+
       \lfloor(E-1)/2\rfloor<2E.
\]

The previous doubling implementation uses
\(\sum_{2^j<E}(E-2^j)=\Theta(E\log E)\) combines. The replacement has
O(EKd) arithmetic for K memory banks and d payload coordinates, and logarithmic
parallel span. Its autograd graph and saved tensors also have linear size.
This is an application of established work-efficient prefix scanning, not a
claim to have invented parallel scans. The contribution here is executing the
same segmented, delayed carrier memory and its credit with that work bound.

**The adjoint is itself an event recurrence.** With local state derivative
\(r_i=\partial L/\partial z_i\) before recurrent dependencies,

\[
 \lambda_i=r_i+a_{i+1}\lambda_{i+1},\qquad
 \frac{\partial L}{\partial v_i}=\lambda_i,\qquad
 \frac{\partial L}{\partial a_i}=\langle\lambda_i,z_{i-1}\rangle.
\]

The coefficient at the next receiver boundary is zero, so credit cannot cross
between receivers. For a nonboundary arrival with positive gap,

\[
 \frac{\partial a_i}{\partial\log\tau}=a_i\Delta t_i/\tau,
 \qquad \frac{\partial a_i}{\partial\Delta t_i}=-a_i/\tau.
\]

Thus training needs neither silent-time updates nor an event-by-event Jacobian
matrix. E119 uses ordinary autograd through the linear-size scan graph; the
formula also specifies a future fused backward kernel. It preserves derivatives
through payloads, counts, arrival times and time constants. Receiver sorting
and the discontinuous change of event order are separate: the derivative is
conditional on the realized order, with the same convention as E118.

**Scope of the work claim.** The present CPU code still sorts arrivals and
receiver addresses, costing O(E log E). Local vector projections are dense and
training with loser credit computes three candidate values per event. Inference
computes one winning value. The *memory scan* is linear work; the whole program
is not claimed to be linear time or globally sparse in every coordinate.
An online receiver implementation has linear memory-update work, but the
scheduler's cost and actual hardware energy still have to be measured.

### Completed equivalence and cost experiment

`e119_scan_audit.py`, seed-6 E118 checkpoint, one CPU thread:

- Five sizes (1, 2, 3, 17, 128) compare the two memories and gradients in float64.
  Singleton streams have mathematically zero time/tau credit; disconnected
  autograd inputs are correctly interpreted as zero in this diagnostic.
- On a four-utterance trained-model batch, maximum logit error is 1.43e-6,
  relative parameter-gradient L2 error 2.34e-7, with no race-winner changes.
- All 256 held-out predictions agree: both give 104/256 correct.
- Forward memory combines fall from 21,553,320 to 3,925,048 (5.49x fewer).
- Eight warm, alternating-order timing repetitions give median forward/backward
  times 347.2 ms versus 206.8 ms per four utterances (1.68x faster); inference
  times 97.1 versus 63.8 ms (1.52x). No optimizer update is included in this timing.
- A single whole-development pass takes 6.61 versus 4.51 seconds. These are local
  CPU timings, not a hardware-independent speed ratio or measured joules.

Floating-point reassociation can change extremely close winners on other
inputs; the equivalence is algebraic, not a promise of bit-identical training.
The checked checkpoint/batches do not exhibit that problem.

## 174. When can a delay receive useful credit?

Nonzero payload credit alone does not ensure that a learned delay is useful.
The normalized memory makes the temporal sensitivity explicit. Hold previous
numerator/mass and the current event fixed. Let

\[
 P=a M_{i-1},\quad B=c_i+\epsilon,\quad
 u=P/(P+B),\quad v=c_i h_i/B,\quad
 \bar h_{i-1}=N_{i-1}/M_{i-1}.
\]

For nonzero previous mass, the actual regularized memory is exactly

\[
 m_i=u\bar h_{i-1}+(1-u)v.
\]

Hence the local eligibility for the log time constant is

\[
 \frac{\partial m_i}{\partial\log\tau}\Big|_{N_{i-1},M_{i-1},h_i,c_i}
 =\frac{\Delta t_i}{\tau}u(1-u)(\bar h_{i-1}-v).
\]

For the interarrival gap it is
\(-u(1-u)(\bar h_{i-1}-v)/\tau\). At a reset there is no previous mass,
so these local terms vanish. The mass feature \(M_i/(1+M_i)\) has an additional
derivative and must be included when applying this analysis to E118.

This identifies three multiplicative conditions for timing to learn:

1. The memory must contain both prior and new evidence: u near zero or one
   suppresses the mixture's local sensitivity.
2. Their payloads must differ in a direction the downstream loss uses.
3. The time scale and gaps must make the eligibility appreciable.

At fixed gap and tau, the mixture factor is largest at u=1/2. This is a local
sensitivity statement, not an instruction to force every memory into the same
balance: long-memory banks legitimately use u close to one. Different banks can
cover different evidence ages. Larger gradients by themselves need not improve
classification. The reverse recurrence in §173 combines these local terms with
future evidence and the terminal class derivative.

**An important task distinction.** For terminal count-weighted pooling, changing
only the final layer's emitted time cannot change the logits. Its ordinary
pathwise time derivative is exactly zero. Winner changes can still alter its
payload; the loser-value contrast supplies score credit for that choice. An
anytime classifier needs an additional causal stopping/readout objective before
last-layer emission times carry the intended answer-latency semantics. E119
therefore measures terminal recognition, not confidence-triggered answering.

**Next discriminating measurement.** Inspect u(1-u), payload contrast and their
loss-aligned product by layer/bank, alongside held-out finite route replays.
This separates absent temporal eligibility from poor alignment and poor
cross-speaker transfer. Uniform route entropy or more candidates does not make
these three conditions hold. No measured optimal time-constant initialization
is claimed by this derivation.

Reference for the scan primitive: Guy E. Blelloch, [Prefix Sums and Their
Applications, CMU-CS-90-190 (1990)](https://www.cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html).

## 175. Causal coalescing is a separate accuracy/work tradeoff

The exact scan change preserves the mathematical model. Reducing the number of
input packets is a different intervention. E119's frozen-model screen merges
adjacent 10 ms packets for the same band into windows of 20, 40 or 80 ms and
releases each sum at its new window closure. It preserves the total count and
label, adds at most 10, 30 or 70 ms of input delay, and never attaches future
counts to an earlier input timestamp. Inference still chooses and emits exactly
one winning continuation per remaining arrival.

The source of approximation can be bounded at the root representation. For a
coordinate \(e_b\exp(-t/\tau)\), moving a packet to a closure \(t+\delta\),
\(0\leq\delta\leq D\), changes its count-normalized contribution by at most

\[
 |e_b|\exp(-t/\tau)(1-\exp(-\delta/\tau))
 \leq \max_b|e_b|(1-\exp(-D/\tau)).
\]

The same bound holds after count-weighted averaging because its weights sum to
one; the constant temporal coordinate and total-count feature are preserved.
This quantifies why fast temporal features are more sensitive than slow ones.
It is **not** a bound on the final classifier: coalescing also changes the event
history, memory sampling, and later race winners. The fixed-history payload
certificate cannot be applied to that different event set without further work.

The proper experiment therefore freezes weights and measures classification,
packet work and CPU time together across those causal windows. Any chosen
window is a development choice. A lower work count with worse accuracy is a
tradeoff, and a shorter CPU time remains distinct from lower measured energy.

### Completed coalescing screen

The final 1,024-fit model gives 151/142/130/114 correct out of 256 with
10/20/40/80 ms packets, respectively. Packet counts become
245,928/158,591/104,493/68,496. Thus 20 ms removes 35.5% of packets while
losing 3.52 percentage points; increasing the window further loses more
accuracy. This establishes a tradeoff and demonstrates that discarded temporal
resolution matters for this trained model. See `results/e119/packet_pareto_d8_n1024_s6.json`.
