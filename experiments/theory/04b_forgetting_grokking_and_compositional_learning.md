# Forgetting, grokking, and compositional learning

[Theory index](../THEORY.md) · Previous: [04 residual depth and grokking](04_residual_depth_and_grokking.md) · Global sections 50–55; section numbers remain stable.

## 50. Why error-gated race learning forgets boundedly, and softmax SGD does not

The first theory aimed at the project's own niche (ROADMAP, 2026-09-26): continual learning.

### 50.1 The near-miss rule is ultraconservative

On a labelled sample, the output rule with credit conservation (§22.3) does nothing if the target wins with
margin; otherwise it promotes the target (+1) and demotes only competitors that came within the eligibility
window, with weights normalised to sum to −1. For the step-synapse race with simultaneous inputs (E4's
setting), a node's potential is v_c = ⟨w_c, x⟩ and the race picks the argmax. The update is then exactly an
**ultraconservative multiclass algorithm** (Crammer & Singer 2003): it touches only the target and the
competitors in the "error set", with competitor coefficients summing to −1. Their theorem gives a mistake
bound: on data separable with margin γ inside radius R, the number of updates is at most about 2(R/γ)²,
**independent of how long training continues**. For ramp synapses the same structure holds within a causal
set (§44: the update is urgency-preconditioned gradient), so the bound transfers to the linearised piece.

### 50.2 Bounded interference in class-incremental learning

Train on task A, then task B (new classes). An old class c's row changes during B only on updates where c is in
the error set (a near miss), so

    ‖Δw_c‖ during B  ≤  η·R·N_c(B),     N_c(B) ≤ M_B ≤ 2(R/γ_B)²

and the new classes' rows also change only on B's mistakes. **Forgetting caused by B is bounded by B's mistake
count, and stops growing once B is learned**, however long B continues.

### 50.3 Softmax cross-entropy with SGD keeps interfering

Cross-entropy SGD updates every competitor on every sample by η·p_c(x) > 0, and the target row on every sample.
On separable data the loss and p_c decay only like 1/t while weight norms grow like log t (the implicit-bias
dynamics of Soudry et al. 2018). The accumulated push on old classes is Σ_t η·p_c,t ~ η log T, and the new
classes' rows keep growing. **Forgetting keeps growing, logarithmically, with time on the new task.** Hidden
layers add representation drift, which is again error-gated in the race and not in SGD.

### 50.4 A sharp side prediction: homeostasis breaks the guarantee

Homeostasis updates thresholds on *every* frame, label or not, so it is not conservative. Its drift grows with
time on task, not with mistakes. **In race networks, homeostasis should be a leading source of forgetting.**
Turning it off (or gating it by error) should flatten forgetting further.

### 50.5 Predictions (M51, E23: class-incremental split-MNIST, 1k / 4k / 16k frames per task)

(i) Race forgetting is roughly flat in frames per task; MLP–SGD forgetting grows with it (roughly log).
(ii) The race without homeostasis forgets less than with it, with the gap growing with time on task.
(iii) The race's number of weight updates per task saturates; the MLP's grows linearly.

*Borrowed:* ultraconservative online algorithms and their mistake bounds (Crammer & Singer 2003); the
implicit-bias dynamics of cross-entropy (Soudry et al. 2018). *New here, as far as checked:* the
identification of the conserved near-miss race rule as ultraconservative, the resulting O(M_B) vs O(log T)
forgetting contrast, and homeostasis as the non-conservative leak.
## 51. The class prior belongs in the prices: why the race forgets old discriminations

*Written 2026-09-26 after E23's first readout, before the ablations below were run.*

### 51.1 What E23 showed first

With a single head, both the race and the MLP forget everything (forgetting 0.97 / 0.98 at 1k frames per task):
the newest classes win on every input. That is the known task-recency bias of class-incremental learning, and it
saturates the metric, so §50's predictions cannot be tested on it. The informative readout is **task-aware**
accuracy (only the task's own classes may win; for the race, the other outputs' thresholds are put out of reach
and the race is re-run). There the prediction reversed: race 0.17 forgetting, MLP 0.03.

### 51.2 The gap in §50.2

The mistake bound counts every update of B, including those where an *old* class wins or nearly wins on a B
input. In class-incremental learning that is most of B's early updates, and each one lowers an old class's
weights on B's features. §50.2 bounds ‖Δw_c‖, but not *where* the change goes: it goes onto shared features,
which are what separate c from its task-mate c′. So a small norm bound does not protect discrimination.

### 51.3 A prior channel absorbs the shift

When the label prior shifts (a new block), the loss-optimal response is mostly a per-class offset. SGD has one: the
bias gradient is p − y, whose mean over the block is exactly the prior mismatch, and a uniform offset across a
task's classes leaves their within-task ranking unchanged. The race's output thresholds are fixed (the bias column
is zeroed), so the whole prior shift is written into feature weights. **In time, a class bias is a price:** a
threshold, the chemical potential of §48. Let the teaching signal move it, θ_c ← θ_c·exp(−η_θ s_c). A few price
updates then give the old classes the margin that ends B's near misses (ultraconservative updates stop at margin),
so fewer updates reach old-class weights.

### 51.4 Predictions (M52, E23 at 1k frames per task)

(i) Race with learned prices (η_θ ∈ {0.003, 0.01, 0.03}): task-aware forgetting falls toward the MLP's; single-head
forgetting stays near 1 (a prior channel cannot fix recency; that needs a balanced prior or replay).
(ii) Output-only learning (frozen hidden) forgets less than full learning, and removing homeostasis helps only a
little: the leak of §51.2 is at the output, not in homeostasis.
(iii) Old-class weight updates during later blocks drop with prices.

*Borrowed:* task-recency bias and bias correction in class-incremental learning (Wu et al. 2019, BiC; Masana et
al. 2022 survey). *New here, as far as checked:* the reading of the class prior as a race price, and the gap
between a norm bound and discrimination in §50.

### 51.5 First results (E23, 1 seed, validation)

Task-aware forgetting, race vs MLP: 0.167 vs 0.030 at 1k frames per task, 0.158 vs 0.021 at 16k. Output-only
race (frozen hidden): 0.192; the same without homeostasis: 0.183.

- §50 (i), race forgetting flat in time on task: **holds** (0.167 → 0.158). MLP forgetting growing like log T:
  **refuted**; it falls (0.030 → 0.021). The implicit-bias argument of §50.3 is about norms, and, like §50.2, says
  nothing about where the change goes.
- §50 (ii), homeostasis as the leak: **not supported** with a frozen hidden layer (0.183 vs 0.192).
- §51 (ii), the leak sits at the output: **confirmed**. A purely ultraconservative linear race on fixed features
  forgets more than the full network. The race is flat in time, as the mistake bound says, but at a level 5–8×
  the MLP's, because the bounded number of updates lands on the wrong coordinates.
## 52. Grokking in the race: sleep turns an absorbing memorization into a phase transition

*Written 2026-09-26, before E24's race runs were read. Setting: E24, (a + b) mod p from two one-hot input
spikes, a fraction of the p² pairs for training. A dense MLP with AdamW groks there on this CPU (p = 31, half
the pairs: train 1.0 by step 1k, test 0.00 until ~3k, 0.87 at 20k, weight norm falling).*

### 52.1 Two circuits, two costs

A network that fits n training pairs can do it in two ways.

- **Memorization is memory indexing.** Each training pair gets its own hidden winner pattern, and the output
  row of its label is tuned to that pattern: a lookup table keyed by the pair. The table's norm grows with
  the number of entries: to give n patterns margin γ with near-orthogonal k-winner codes needs roughly
  ‖W‖²_mem ≈ c_mem · n/γ².
- **Generalization is a relation.** Modular addition is addition of phases (a ↦ e^{2πia/p}); the known
  generalizing circuit uses a few Fourier frequencies (Nanda et al. 2023), with a norm C_gen/γ² that does
  **not** grow with n.

The minimum-norm solution with margin γ is therefore the lookup table below a critical data size
n* ≈ C_gen/c_mem and the relation above it (the "circuit efficiency" account of Varma et al. 2023, in race
units). This is the manifesto's claim in miniature: a lookup is the spatial solution, a relation the
compressed one, and only a pressure towards small norm makes the system prefer the relation.

### 52.2 Without sleep the race cannot grok (a consequence of §50)

The output rule is ultraconservative (§50.1): it updates only on mistakes and near misses (margin window
`margin`). Once every training sample wins with margin, **the rule is inert**: the memorizing solution is an
absorbing state. Nothing moves the weights towards smaller norm, so the relation never takes over, whatever
the training time.

Softmax SGD differs: every sample always updates by p_c > 0, and the implicit bias of cross-entropy
(Soudry et al. 2018) drifts the weights towards the max-margin direction at rate ~1/log t. That is why dense
networks can grok slowly even without weight decay, and why weight decay speeds it up. **The same property
that bounds the race's forgetting (§50) forbids its grokking.**

### 52.3 Sleep is the leak that restores the drive

Sleep downscaling (the synaptic homeostasis hypothesis, Tononi & Cirelli 2003/2014): after each waking epoch,
every weight shrinks, w ← (1 − λ)w. Waking and sleeping together minimise

    λ/2 · ‖W‖²  +  Σ_samples hinge(margin − Δ(x, y))

by stochastic subgradient steps: the waking rule is the hinge subgradient (it fires only inside the margin
window, and §44 makes it exact up to a per-node positive preconditioner at the output), sleep is the L2 step.
That is **Pegasos** (Shalev-Shwartz et al. 2007), which converges to the regularized max-margin solution. By
§52.1 that solution is the relation when n > n*. So the race groks when it sleeps, and only then.

Sleep is non-conservative, like homeostasis (§50.4): it erodes margins everywhere, not only where there
were errors. **Sleep trades the forgetting guarantee for generalization.** The two E23/E24 axes are one
dial.

### 52.4 Timescales and the phase diagram

Control parameters: the data fraction n/p², the sleep strength λ, and the waking rate η.

- **Grokking time.** The memorizing part of W is defended only while it carries margin that the relation does
  not already provide. Once the relational component can carry the margin, the table decays geometrically
  during sleep: t_grok − t_fit ≈ (1/λ) · log(‖W_mem‖/‖W_gen‖). **Delay ∝ 1/λ**, and larger initial norms
  (higher `init_frac`) lengthen it logarithmically (Omnigrok's "LU mechanism": grokking needs the initial norm
  to be above the generalizing one).
- **Too much sleep underfits.** The waking rule restores at most η·(rate of margin violations)·R per epoch,
  while sleep removes λ‖W‖². When λ exceeds roughly η·R/‖W_gen‖ even the relation cannot hold its margin: the
  network neither memorizes nor generalizes.
- **Too little data never groks.** Below n* the minimum-norm solution *is* the table; sleep only makes it more
  efficient. Reducing the data after grokking should undo it ("ungrokking", Varma et al.).

So there are three phases: memorization (λ → 0, or n < n*), grokking (intermediate λ, n > n*), and
confusion (λ large).

### 52.5 Order parameters the race makes visible

The event substrate exposes quantities that dense networks hide:

- **Code sharing.** Memorization gives nearly one hidden winner pattern per pair; the relation gives patterns
  shared across the pairs with equal a + b (their fibre). The number of distinct hidden codes among training
  pairs, or the mutual information I(code; pair) − I(code; label), should drop sharply at grokking (a
  neural-collapse-like order parameter, measured in spikes).
- **Latent heat in plasticity.** The rule is error-gated, so plasticity events per epoch are a direct readout
  of how much margin is being re-carved. Prediction: after the fit, plasticity falls, then **peaks at the
  transition** as the table is dismantled and the relation built, then falls to a low floor. A peak in
  susceptibility at a phase transition, measurable for free.
- **Certified radius.** Max-margin solutions have larger margins in time units, so the median certified
  jitter radius ε* (§34.4) should jump at grokking, tying §40's generalisation-through-timing-robustness to
  this transition.

### 52.6 The hidden layer is the risk

The argument is exact for the output layer (§44). Hidden credit comes through random feedback, which by §46
fits label templates layer by layer rather than composing. If the hidden layer cannot form phase-like
features, the sleeping race is a max-margin readout on a slowly changing, nearly random code: a kernel
machine. Random k-winner codes of the pair then need a width that grows with p² to generalize, and the race
would generalize only through width, not grok. Frozen hidden vs random-feedback hidden separates the two.

### 52.7 Time coding could lower the critical data size

The manifesto's thesis is that relations are cheap in time. Modular addition is addition of phases, and a race
neuron computes weighted means of input times (P3). If operands arrive as delays on a cyclic clock (a at
phase a/p, b at phase b/p), the relation is a coincidence detector over summed delays, with a norm of a few
synapses per output, so C_gen, and with it n*, should fall well below the one-hot case. **Prediction:
phase-coded operands grok from smaller training fractions than one-hot operands.** The encoding needs a cyclic
(wrapping) readout, which the race does not yet have; to design.

### 52.8 Predictions (M53, E24)

(i) Race without sleep: fits the training set, test stays near chance for any training length; plasticity
decays towards zero after the fit.
(ii) Race with sleep, intermediate λ: delayed generalization, with a delay that scales roughly as 1/λ; large
λ underfits.
(iii) A plasticity peak and a drop in the number of distinct hidden codes at the transition.
(iv) Frozen hidden layer: generalization, if any, grows smoothly with width and not with training time.
(v) Below a critical training fraction no λ groks; the critical fraction is lower for phase-coded operands
(when built).
(vi) Sleep increases forgetting in E23 (§52.3's dial).

*Borrowed:* grokking (Power et al. 2022); circuit efficiency and ungrokking (Varma et al. 2023); the norm
account and initial-norm dependence (Liu et al. 2022, Omnigrok); Fourier circuits for modular addition (Nanda
et al. 2023); Pegasos (Shalev-Shwartz et al. 2007); the implicit bias of cross-entropy (Soudry et al. 2018);
sleep as synaptic downscaling (Tononi & Cirelli). Biologically motivated mechanisms that help grokking in MLPs,
including homeostasis and lateral inhibition, are studied by Leon (2026). *New here, as far as checked (web
search, 2026-09-26, no demonstration of grokking in spiking or event networks found):* that an
ultraconservative race rule makes memorization absorbing; that sleep downscaling turns it into a
Pegasos-driven phase transition; that one dial trades forgetting against generalization; and the event-level
order parameters (code sharing, plasticity peak, certified radius).

### 52.9 First test, and a correction: in a thresholded race, downscaling is price inflation

**Result (E24, p = 31, half the pairs, 5,000 epochs, no deadline, no homeostasis).** No λ grokked. λ ≤ 3e-4:
memorization, test 0.000–0.004 throughout. λ = 1e-3: slow collapse (train 0.97 → 0.51). λ ≥ 3e-3: the network
**dies** (weight norm → 0, train 0.00, plasticity 0). The grokking phase is missing, and the "confusion" phase
is not underfitting but silence. Without sleep, plasticity did not stop either (§52.2 assumed it would): random-
feedback hidden credit keeps the codes churning while the test error stays at the lookup's floor.

**What §52.3 got wrong.** A node fires when its potential reaches θ, so (W, θ) → (cW, cθ) is a gauge symmetry
of the race (the dilation of §49, applied to potentials). Shrinking W with θ fixed is not a norm penalty; it is
**raising every price**. Below threshold nothing fires, no error is registered (a silent output is not a
competitor), and the error-gated rule cannot recover: silence is a second absorbing state, next to
memorization. Pegasos needs two things the plain race lacks:

1. **The network must keep deciding while margins shrink.** A deadline (the leader fires at the horizon)
   makes the output an argmax again, so sleep shrinks margins measured in units of θ, and the waking rule
   defends only the margins that the data needs. That is Pegasos's absolute margin, in θ units.
2. **Hidden prices must follow the drive.** Homeostasis lowers hidden thresholds as downscaling lowers the
   drive, so hidden codes survive. The gauge-invariant quantity that sleep then shrinks is ‖w_n‖/θ_n only
   to the extent that homeostasis lags, so hidden-layer sleep is weak by design; the output layer carries
   the max-margin pressure.

So **sleep needs prices** (P5): downscaling regularizes only when thresholds are dual variables that re-balance
activity. This is the grokking counterpart of §51, where the missing prior channel was also a price.

**Revised prediction (M53, before the rerun):** with deadline and hidden homeostasis, the silent phase
disappears; an intermediate λ shows delayed generalization; large λ underfits instead of dying. If the
intermediate phase still does not appear, §52.6's risk (random-feedback hidden credit cannot build relational
features) is the leading explanation, and a frozen or wider hidden layer should behave the same.
## 53. Delays instead of lookup: modular arithmetic on a ring, learned by replay

*Written 2026-09-26 after E24's race runs (all test ≤ 0.008 against chance 0.032, feedback-alignment MLP 0.000:
only backprop groks there) and after E25 pilots. The E24 encoding put both operands at t = 0, so time did no work.
This section re-poses the task the way the README's Sleep Sort note suggests.*

### 53.1 The compiled ring

A ring of p relay nodes, each firing the next after one delay unit, is a cyclic clock: a spike injected at node a
sits at node (a + t) mod p at time t. Inject operand a as a position, let operand b set the read time (a delay of
b units on its line), and a coincidence detector per node reports (a + b) mod p. Addition is waiting; the modulus
is the cycle. 4p synapses, 2b + p + 1 synaptic events per query, all p² pairs correct, nothing learned
(`e25_delay_ring.py compiled`). The lookup table needs p² conjunction nodes and cannot answer an unseen pair.

### 53.2 What the substrate assumes: characters

Give every operand line and every class detector a learnable delay on a ring of any period; write it as a phasor,
z = e^{2πi·delay/period}. The prediction is the class whose detector phase is nearest to the sum of the two
operand delays:

  ŷ(a, b) = argmax_c Re( z̄_c · z_a · z_b ).

This fits a labelling exactly iff y = h(f(a) + g(b)) for some maps into a cyclic group (h injective on the used
classes): the labels factor through **one character** of an abelian group. It contains a + b, a − b, relabelled
sums, a² + b², and a·b on the nonzero residues (a cyclic group of order p − 1, so the delays must learn the
discrete logarithm). It excludes a² + ab + b² and random tables. 3p parameters. The substrate knows it is
composing phases; it is not told which operation, which frequency, or which encoding of the operands.

### 53.3 Learning is synchronization; replay is its power method

With the classes as labelled constraints z_a z_b z̄_y ≈ 1, learning is angular synchronization on the
tripartite hypergraph of training triples (a, b, y). The replay rule

  z_a ← unit( Σ_{samples with a} z_y z̄_b ),  and likewise for z_b and z_y,

is local (a delay moves to the circular mean of what the samples it took part in say it should be: in time,
the teacher's arrival minus the partner operand's arrival) and is the generalized power method for
synchronization. It needs all stored samples at once, so it is a sleep-phase computation. The online,
error-gated version of the same geometry (§35's weaving, in continuous delays) fails even on the training set
where replay succeeds. (Corrected in §54: the cause is the push on the wrong winner, not the loop.)

Replay is not required, but forgetting is. Keep one phasor trace per delay, add each sample's vote as it
arrives (S_a += z_y z̄_b, with the delays read from the current traces), and downscale all traces by (1 − λ)
in a sleep phase between epochs. Without downscaling the first, random-phase votes are never outweighed and
the traces freeze into an inconsistent state (pilot, p = 97, 20% of pairs: test 0.01). With λ = 0.5 the
same rule reaches test 1.000 from 20% of pairs. This is §52's sleep, now doing what §52 predicted it would do:
the online rule is a power method whose stale early iterates must be forgotten. In E24's race the same
downscaling only destroyed training accuracy; the difference is the hypothesis class, not the sleep.

### 53.4 Two thresholds: statistical and computational

- **Identifiability.** 3p phases with a gauge (a global phase per group and the frequency choice m ∈ Z_p^*)
  against n constraints of log p bits: a consistent fit is forced to be the relation once n exceeds about
  3p, i.e. frac ≳ 3/p. Below it, fits that memorize exist even in this tiny class (E25 pilots: p = 31,
  5% of pairs, train 1.0, test at chance). This is §52.1's lookup/relation dichotomy with parameter
  counting instead of norms.
- **Search.** The power method from random starts succeeds only well above that: pilots at p = 97 succeed from
  10% of pairs (8 restarts chosen on training error, test 1.000) and fail at 3–5% (train 0.1–0.4). As in
  sparse synchronization and planted problems, a gap between what the data determines and what local
  iteration finds is expected; its width is the measurement.

### 53.5 Relation to grokking

A grokking MLP ends in the circuit Σ_ω cos(ω(a + b − c)) (Nanda et al. 2023): it builds a phase representation
of the operands out of weights, slowly, under weight decay. The ring has that representation as physics:
delays add and cycles wrap. The prediction is that the delay substrate reaches the relation from a much smaller
fraction and with no slow memorize-then-generalize phase, because its hypothesis class holds nothing but
characters. The price is the class: one ring is one character, so a² + ab + b², which the MLP can grok, is out of
reach. A bank of K rings summing votes is a K-term character expansion, the MLP's grokked form, but it does not
rescue poly: e^{2πi m(a² + ab + b²)/p} = e^{2πi m a²/p} · e^{2πi m ab/p} · e^{2πi m b²/p}, and the middle
factor, as a p × p matrix in (a, b), is a DFT matrix, full rank. Each frequency needs K ≈ p separable rings, so
the bank grows to ~p² parameters, a table. Compression by delays exists exactly for separable compositions.

### 53.6 Predictions (M54, E25)

(i) Compiled ring: accuracy 1.0 at every p, events 2b + p + 1.
(ii) Replay generalizes (test ≥ 0.99) above a critical fraction f_c(p) that falls with p, while memorizing
    (train 1, test at chance) is possible below it; f_c between 3/p and ~10/p.
(iii) The online error-gated learner and the discrete near-miss learner fail where replay succeeds.
(iv) The class boundary: add, sub, perm, sq, mul learned; poly and rand not, at any fraction.
(v) The backprop MLP needs a larger fraction than replay at the same p, and thousands of steps.
(vi) A bank of rings learns poly only with K ≈ p rings per frequency (no compression): the delay substrate's
    advantage is confined to separable compositions, and the sample complexity for poly should look like a table's.
## 54. In a race, winning is positional: teach by pulling, never by pushing

*Written 2026-09-26 from E26 pilots (p = 31, 30% of pairs, 1–2 seeds). Full sweeps queued (`queue/e26.txt`,
`queue/e26b.txt`).*

§53's replay and trace learners work, but they are dense: every sample updates, in epochs, with global
normalisation. E26 asks the question in native terms. Passive delay ring; a query is two operand spikes; class
detectors race, and the first to coincide fires and cancels the rest; learning happens only on errors and touches
only the three delays involved plus, optionally, the wrong winner.

**Observation.** With the usual two-sided rule (pull the teacher earlier and push the wrong winner later), all p
detectors collapse onto a single phase (pilot: 30 of 31 inter-detector gaps < 0.1) and accuracy stays at
chance, even on the training set, with or without timing noise σ ∈ {1, 2, 3, 8}. Repelling crowded runners-up
does not fix it. With the push removed, the same sparse rule learns everything jointly from random delays:
test 0.86–0.91 on unseen pairs from 30% of pairs, about 37k updates in 100k samples, σ = 0.

**Why.** In a race, a class wins by being *earliest*, not by others being late: cancellation already implements
the competition. The teacher pull has a fixed point per class (its detector listens just after the phase its
samples produce), so pull-only learning is a set of independent contractions. The push has no fixed point of its
own: the pushed detector's position is set by *other* classes' errors, and each push hands the lead to the next
detector just behind it. The ring of detectors behaves like a queue, and pushes feed it back toward the read point
until it is one clump. A dense softmax needs the push because scores are not exclusive; a race does not, and
the push is actively harmful. (§53.3's "frustration" of online learning was this push, not the loop.)

**Consequence for the main architecture.** The race rule used everywhere since E6 has exactly this term: every
competitor gets −elig (a near-miss-weighted push later). If the argument holds beyond the ring, it is also a cause
of the race's weak results where competitors crowd (SHD, E23 forgetting, E24). Test: `--compete 0` on E22 (SHD)
and E24 (grokking), queued.

**Predictions (M55).** (Status: (i) confirmed at full length, 3 seeds; (iii) refuted for the weight-based race, see §60.) (i) E26 push = 0 generalizes above a critical fraction, push = 1 never does; (ii) annealed
timing noise changes sample efficiency but not the push result; (iii) `--compete 0` does not lower SHD accuracy,
and raises it if competitor crowding is a cause of the gap.
## 55. Where supremacy can and cannot be claimed

*Written 2026-09-26.*

The one formal separation found in the literature goes against spiking networks: indexing needs Ω(n/log²n)
spiking gates vs O(√n) sigmoid gates (Lynch, Musco & Parter 2017, Neuro-RAM). Indexing is the dense world's
native operation. The question is where the reverse holds.

**Not on operation counts for static functions.** The compiled ring (§53.1) adds mod p with O(1) events given
a shared pacemaker, but a dense network fed the operands as scalars computes cos(2π(a + b)/p) in O(1)
operations too. The apparent separation against a one-hot MLP is an encoding effect, not an effect of asynchrony.
Any static function has a clocked implementation whose op count matches the event count up to the encoding.

**Where the clockless system is different in kind.** A clocked system pays per tick × unit whether or not
anything happened; an event system pays per event. So a defensible separation needs:

1. **Streams whose information rate is far below any usable clock rate:** cost ∝ informative events vs ∝ T/dt.
   The clock can't be slowed without missing timing that matters (SHD-style precise timing inside long silence).
2. **Decisions whose latency is set by the evidence (E2):** the event system answers at the first sufficient
   event; a clocked pipeline answers after its fixed depth × tick.
3. **Learning whose cost ∝ errors (E26):** no epochs, no backward pass over time.

All three must hold on one task at matched accuracy, with the dense side allowed the same priors and input
encoding.

**A quantitative criterion for the input side.** A clocked system that must resolve timing δ pays at least
(channels × duration / δ) input samples; an event system pays one per spike. The separation factor is therefore
1 / ρ_δ, with ρ_δ = spikes per channel per δ-bin. Measured on SHD (test set, 227 utterances): 8,414 spikes per
0.71 s utterance on 700 channels, so ρ = 0.017 at δ = 1 ms (59×), 0.068 at 4 ms (15×), 0.17 at 10 ms (6×).
Dense SHD models do well with 10 ms bins, so SHD offers only about 6× on the input side: it is dense in time, a
poor supremacy benchmark. The benchmark must have ρ_δ ≪ 0.01 at the precision the task truly needs, e.g. rare
informative events in long silence, where the factor grows with the silence. That task family is the benchmark target: sparse event streams with rare, precisely timed informative
events (mostly-silent keyword spotting, event-camera onsets, anomaly onset), measured in events, latency, and
updates.
