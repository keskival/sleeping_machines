# Sequential classification and output races

Written 2026-09-28. This note specifies the SHD objective for a classifier
whose output is an event: once one class is sufficiently supported, it emits
that class and stops. It separates this online decision task from ordinary
utterance-to-class scoring at the end of a recording.

## 115. A supervised first-output race

Let $V_c(t)$ be the readout potential for class $c\in\{1,\ldots,C\}$,
computed only from input events observed by time $t$. Define its instantaneous
class probability, using an output temperature $T_o$ selected on validation data:

\[
p_c(t)=\operatorname{softmax}(V(t)/T_o)_c.
\]

The deployed decision is causal. Given confidence threshold $\theta$, it
emits at the stopping time

\[
\tau=\inf\{t:\max_c p_c(t)\ge\theta\},\qquad
\hat y=\arg\max_c p_c(\tau).
\]

Before $\tau$, the output is silence (no class has won); after $\tau$, the
decision is final. Since $p(t)$ is a function of the event history through
time $t$, future events cannot revise an already-emitted answer. The
confidence threshold is a decision rule, not a claim that softmax is
calibrated; held-out accuracy, coverage, and latency must be reported together.

### Differentiable likelihood for training

Hard first-crossing decisions do not provide a useful gradient when no class
crosses the threshold. Relax them to competing class hazards, constant within
each simulation bin:

\[
q_{\max}(t)=\max_c p_c(t),\qquad
\Gamma(t)=\rho\,\eta\,
\operatorname{softplus}\!\left(\frac{q_{\max}(t)-\theta}{\eta}\right),
\qquad \lambda_c(t)=\Gamma(t)p_c(t),\quad \Lambda(t)=\Gamma(t).
\]

Here \(\eta\) is the softness of the confidence boundary and \(\rho\) sets
the output-event rate. In a bin of width \(\Delta\), the probability that no
class emits before bin \(t\) is

\[
S_t=\exp\!\left(-\Delta\sum_{s<t}\Lambda(s)\right).
\]

The probability that class \(c\) is the first output in bin \(t\) is

\[
P(c,t)=S_t\left(1-e^{-\Delta\Lambda(t)}\right)
             \frac{\lambda_c(t)}{\Lambda(t)}.
\]

For the shared gate above, $\lambda_c=\Gamma p_c$ and
$\Lambda=\sum_c\lambda_c=\Gamma$, so the cause fraction simplifies
exactly:

\[
\frac{\lambda_y}{\Lambda}=p_y,
\qquad \log\frac{\lambda_y}{\Lambda}
=\operatorname{logsoftmax}(V/T_o)_y.
\]

This identity matters computationally as well as algebraically. Computing the
ratio from float32 hazards and then taking $\log(\max(p_y,\epsilon))$ can
zero the class gradient when a wrong class saturates softmax and $p_y$ falls
below the clamp. The implementation now obtains the cause log-probability
directly with stable `log_softmax`; it introduces no extra class operation
because the readout already compares all 20 classes.

For utterance label $y$, the undiscounted objective is the negative log
probability that the correct class wins before the per-utterance deadline:

\[
\mathcal L_{race}=-\log\sum_{t\le H}P(y,t).
\]

This likelihood penalizes an early wrong output twice: it assigns that bin's
cause probability to the wrong class, and its event hazard reduces survival
for later correct output. A no-output trace has near-zero correct-win
probability and therefore high loss. The experiment also uses an explicit
latency preference $e^{-\gamma t}$:

\[
\mathcal L_{race,\gamma}=-\log\sum_{t\le H}e^{-\gamma t}P(y,t).
\]

This is a time-discounted success objective, not a normalized event
likelihood. $\gamma=0$ recovers the proper race likelihood; $\gamma>0$
trades some accuracy for earlier correct answers. It must be swept on the
speaker-held-out validation set, since the dataset has no label for the
earliest time at which a listener should decide.

This is a sequence-level likelihood, not a cross-entropy target copied onto
every prefix. Earlier bins contribute survival: no class should have won yet.
When the readout is uninformative and class probabilities are equal, the cause
term is class-neutral; as evidence separates the classes, the likelihood
rewards the correct class for winning the race. As $\eta\to0$, the shared
emission hazard vanishes below $\theta$ and grows above it; with a high event rate, the soft race approaches
the deterministic threshold-crossing rule (ties remain a discrete boundary).
At finite $\eta$ and $\rho$, this is a stochastic stopping-time surrogate,
not the exact likelihood of the deployed deterministic threshold rule: it can
assign event mass before a hard crossing and spread event mass over several
bins after it. A nonzero $\gamma$ further optimizes discounted utility, not a
normalized probability. Validation must therefore measure the hard rule's
accuracy, coverage, and latency frontier; low training loss alone does not
establish that the deployed race is well trained.

### Coverage-complete race and vector-valued answer

At a hard threshold crossing, E83 emits the full class-probability vector
$u(\tau)=p(\tau)\in\mathbb R^{20}$ as the signal payload. The argmax decodes
the class; the remaining coordinates retain uncertainty for downstream
message passing. The `anytime` scheme handles a missing crossing by emitting
a terminal max-potential classification at the end of the utterance. If
$S_H$ is the probability of no race event by deadline $H$, and
$q_y^{end}=\operatorname{softmax}(V^{max}/T)_y$, its correct-output
probability is

\[
P(\mathrm{correct})=P(\mathrm{correct\ early\ race})+S_Hq_y^{end}.
\]

The terms are disjoint: either the correct class wins early, or no race event
occurs and the terminal classifier supplies the answer. At $\gamma=0$ this
is a normalized probability of a correct answer. A silent trace with a
classifiable terminal state gets a gradient through the fallback term; an
incorrect early winner spends the race probability mass and cannot be
repaired by the fallback. A fallback guarantees an answer at the recording
boundary, but it is late and does not count as an early decision. Measure
fallback rate separately from race coverage and emitted-answer accuracy.

For E83, $T_o$ is configurable, $\Delta=1$ ms, and each sample's deadline is its own
last input time plus the maximum allowed path delay and a 60 ms settling
window. The deadline limits the likelihood only; the inference-time crossing
rule itself does not need to know the future utterance duration. If no class
crosses, the model emits no answer. A forced end-of-recording prediction is
reported separately for conventional full-coverage benchmarks.

### Why the former E83 objective was confounded

The earlier readout used

\[
\bar p=\frac1G\sum_{t=1}^{G}\operatorname{softmax}(V(t))
\]

and applied one cross-entropy to \(\bar p\). It was technically one loss per
utterance, but it did not implement the intended first-output task. During
empty time, the zero readout has \(\operatorname{softmax}(0)=\mathbf1/C\).
If only \(G_v\) of \(G\) bins contain useful evidence, then

\[
\bar p=\frac{G_v}{G}\bar p_v+
      \left(1-\frac{G_v}{G}\right)\frac{\mathbf1}{C}.
\]

Thus silent prefixes and the artificial delay tail pull each answer toward
chance. Worse, batches share one simulation length set by their longest item:
the same short utterance can receive a different loss and prediction depending
on which longer utterance it is batched with. The old depth-2 runs are therefore
debug observations only; they are not trainability or depth evidence.

## 116. What established SHD supervision does and does not answer

The original Heidelberg benchmark study used last-time-step cross-entropy for
LSTMs and also evaluated max-over-time membrane-potential loss; the max-over-
time objective improved its SNN and LSTM results over last-step scoring. That
is a sequence label applied to a temporal trace, not a per-token language-model
loss. A current Spyx SHD tutorial instead applies cross-entropy to the sum of
readout voltages over time. These are suitable sequence-classification
controls, but neither directly measures first-confident-output behavior.

E83 therefore reports two axes:

1. **Full-coverage benchmark control:** cross-entropy/accuracy from the
   per-class maximum potential over the valid sequence window, matching the
   original SNN readout family; integral-potential pooling is a second control.
2. **Online race:** fraction of utterances that cross the confidence threshold,
   accuracy among emitted answers, and latency from the first input event.
   A no-output case is not silently converted into a correct-at-end answer.

The published methods do not establish a best objective for a deep,
sparse, delayed message-passing classifier. That is an empirical question.
The first comparison should keep topology, data order, seed, and optimizer
budget fixed across race, integral, and max-over-time objectives. Only after
the loss and output behavior are verified should depth 2/4/8/16 be compared.

## 117. Verification obligations

- A controlled prefix task with a known evidence-arrival time must show that
  the model stays silent before the evidence and emits the right class after
  it; include distractor evidence and a no-evidence/no-output case.
- On SHD, use the speaker-held-out validation partition for model and
  confidence-threshold selection; reserve the official test speakers for a
  single final evaluation.
- Report coverage, emitted-answer accuracy, latency, and full-coverage SHD
  accuracy. Accuracy alone can hide a race that never emits.
- The current E83 evaluator computes the trace through the deadline, then
  records the earliest causal threshold crossing. It does not stop the full
  layer simulator at that point; wall-clock or energy savings from early halt
  are not yet measured.
- Compare scores for each utterance independently of batch composition. The
  pooling deadline must use that utterance's end time and its own maximum path
  delay, not the longest sequence in the minibatch.
- Sweep the output temperature and confidence threshold on validation data and
  plot the accuracy-coverage-latency frontier. Do not call raw softmax
  probability “certainty” until calibration has been checked.

## 118. Failure modes are part of the result

The race is a hypothesis, not a guarantee of useful early classification.
There are three distinct failure modes:

1. **Premature wrong winner.** Once emitted, a wrong class cannot be repaired.
   The race likelihood penalizes wrong cause mass and survival loss, but it
   cannot invent class information absent from an early prefix. Measure the
   first-emission error and latency jointly.
2. **No winner.** In race-only mode, if no confidence reaches $\theta$, the
   classifier remains silent. Its smooth hazard loss penalizes low
   correct-win probability, but this surrogate does not guarantee a crossing
   under the hard inference rule. The `anytime` objective adds a terminal
   max-potential fallback and optimizes early correct-race probability plus
   no-race survival times terminal classification probability. It guarantees
   an answer at the supplied end-of-utterance boundary, not an early answer.
   Report fallback rate so full coverage cannot hide failed early decisions.
3. **False confidence.** Softmax is a normalized score, not a calibrated
probability. An apparently high confidence can be wrong, especially under
held-out speakers or distribution shift. Calibrate $T_o$ and $\theta$ on
validation data only, then freeze both for test.

## 119. General event-stream-to-class recognition

SHD is one member of a broader task class. Let an input be a marked point
process $E_{\le t}=\{(t_i,x_i,m_i):t_i\le t\}$ and let $Y$ be one label for
the whole recording. For SHD, $x_i$ is a frequency band; for event-camera
recognition it can be pixel coordinates and polarity; for gesture or
industrial monitoring it can be a sensor identity and value. In all these
cases there may be no frame, phoneme, or per-event target. The causal object
to estimate is the posterior

\[
\pi_t(c)=P(Y=c\mid E_{\le t}).
\]

A deep representation is useful when it lets a simple readout recover this
posterior while discarding nuisance variation such as speaker, tempo, camera
motion, or illumination. It cannot create class information absent from the
observed prefix: by data processing,
$I(Y;Z_t)\le I(Y;E_{\le t})$ for a representation $Z_t=f(E_{\le t})$.
Depth can reorganize accessible information, but a layer that discards
class-relevant distinctions makes them unrecoverable downstream.

With zero-one class loss, the immediate Bayes error at time $t$ is
$R_{stop}(\pi_t)=1-\max_c\pi_t(c)$. A rational stopping rule compares this
with the expected future value of more evidence:

\[
V(\pi,t)=\min\left\{
  1-\max_c\pi(c)+C_{stop}(t),\quad
  C_{wait}(\delta)+\mathbb E[V(\pi_{t+\delta},t+\delta)\mid\pi_t=\pi]
\right\}.
\]

Thus a fixed confidence threshold is an approximation to an application
specific optimal-stopping boundary, not a universal property of the task.
The martingale argument for waiting and the exact selective-risk condition for
a first-crossing emission are derived in §§122–123. SHD's utterance label
alone does not reveal the earliest acceptable answer time; neither does a
clip label for event-camera recognition.

## 120. Vector sufficiency and event-rate impedance

The class output can itself be a message value. E83 currently emits
$u_t=\pi_t\in\mathbb R^{20}$ at a race crossing; the argmax identifies the
class and the rest of the vector carries uncertainty. A smaller payload can
use class code vectors $v_c\in\mathbb R^d$ and emit
$u_t=\sum_c\pi_t(c)v_c$. This compression is safe only if the code vectors
remain decodable over the confidence range used for stopping. A codebook
should be orthogonal or margin-separated, and its vector decoder should be
validated independently; a hard argmax followed by a lookup is only a
transport encoding, not an extra learning signal.

For a linear time-vector state driven by marked events,

\[
z(t)=\sum_{t_i<t}e^{A(t-t_i)}Bv_i.
\]

If arrivals are approximately Poisson with rate $\nu$ and centered payloads
of covariance $\Sigma$, the stationary covariance is

\[
P=\nu\int_0^\infty e^{As}B\Sigma B^*e^{A^*s}\,ds,
\qquad AP+PA^*+\nu B\Sigma B^*=0.
\]

For scalar decay $A=-1/\tau+i\omega$, this scales as
$P\propto\nu\tau B\Sigma B^*/2$. A receiver's firing probability therefore
depends jointly on input event rate, payload covariance, decay constants,
readout weights, threshold, and reset. Reusing the same weight scale and
threshold at every depth does not preserve activity when these quantities
change.

Define $N_\ell$ as the number of emitted events per fixed set of utterances
from layer $\ell$, and $g_\ell=N_{\ell+1}/N_\ell$ as a coarse event-throughput
ratio. A first-order chain approximation gives
$N_L\approx N_1\prod_{\ell<L}g_\ell$: repeated $g_\ell\ll1$ suggests a
shrinking communication budget, while repeated $g_\ell\gg1$ warns of an
activity cascade. This is a diagnostic, not a sufficient condition for
accuracy: a few high-information vector messages can outperform many noisy
ones. E83's current pilots show both regimes depending on the objective. The
integral epoch-1 validation counts were $49,9,4,5$ spikes per utterance across
four layers; the race run's Layer 4 count rose from $53$ at epoch 1 to over
$2{,}200$ by epoch 2, while accuracy stayed near chance. Those runs used the
same topology but different losses, so they point to objective-sensitive
impedance and stability, not an architecture-wide activity law.

## 121. Sparsity has three separate meanings here

E83 has sparse connectivity masks and sparse emitted messages, but its current
simulator still advances every layer on a 1 ms grid. The non-spiking readout
materializes a $G\times B\times20$ score trace and compares all 20 classes at
each bin. This is a small output head rather than quadratic attention, but it
is time-synchronous work; the current result does not demonstrate fully
asynchronous inference or energy savings. Connectivity sparsity, activity
sparsity, and sparse execution must be measured separately.

A direct asynchronous readout can update class evidence only when a hidden
message arrives. For an event from sender $j$ with payload $v_k$,

\[
s_c(t_k^+)=s_c(t_k^-)+a_{j,c}^{\mathsf T}v_k
\quad\text{for sparse edges }(j,c),\qquad
\pi_k=\operatorname{softmax}(s(t_k^+)/T_o).
\]

It emits the vector $\pi_k$ only when the stopping rule is met; otherwise an
end-of-utterance event invokes the terminal fallback. With no between-event
leak, class state changes only at arrivals, so this removes the $G\times20$
readout sweep. With leaky state, crossings can occur between arrivals and
must be scheduled or bounded analytically. That event-time head is a distinct
experiment; it should follow a verified sequence classifier so that changes
in loss and execution are not conflated.

The immediate theoretical and experimental unknowns are therefore: whether
the stable race cause gradient learns at all; whether max/integral terminal
loss gives a class-sufficient depth-4 representation; what per-layer payload
rate/covariance keeps the event chain informative without runaway firing; and
how much of the remaining work is the 1 ms simulation grid rather than the
sparse topology. The current matched controls address the first two; the
rate-matching and asynchronous-readout claims remain open.

An initial 80-example smoke of the first race prototype emitted on every
utterance but achieved only 2.5% accuracy on 40 held-out-speaker examples
(20-way chance is 5%). That was a failed prototype, and too small to judge the
race hypothesis. It built classwise hazards from already-peaked probabilities.
The revised form uses one confidence gate, a temperature-scaled softmax cause
distribution, and an explicit latency discount. Its 80/40, one-epoch smoke
reached 92.5% coverage but only 10.8% accuracy among emitted answers; its
full-sequence max-over-time accuracy was 2.5%. This is still a failed learning
pilot with very few examples and one epoch. The high confidence/low accuracy
is a direct warning that the softmax score is not calibrated. The integral and
max controls from the same tiny smoke were also near chance; the integral
scores there used an uncalibrated raw millisecond sum and are superseded by
the dimensioned 1 ms integration factor now in the code. In the current
depth-4, 512/128 held-out-speaker pilot, integral pooling after two epochs
reached 4.69% max-over-time accuracy and 5.47% with terminal fallback (20-way
chance is 5%); early-race coverage was 17.97%, and its emitted accuracy was
4.35%. The matched max control also ended at 4.69%. Stable-cause anytime
reached 6.25% max-potential accuracy (8/128; chance-tail probability 0.31),
with 100% coverage and saturated confidence. Stable-cause race-only reached
3.12% max-potential accuracy, 97.66% coverage, and 4.8% emitted accuracy at
epoch 2. These are not reliable recognition results. The controls are
diagnostic, not a competitive SHD benchmark.

**References.** Cramer et al., *The Heidelberg Spiking Data Sets for the
Systematic Evaluation of Spiking Neural Networks* (2022),
[paper](https://kip.uni-heidelberg.de/Veroeffentlichungen/download.php/6616/temp/4143-3.pdf);
Spyx, [SHD training tutorial](https://spyx.readthedocs.io/en/latest/examples/surrogate_gradient/SurrogateGradientTutorial/)
and [integral cross-entropy definition](https://spyx.readthedocs.io/en/latest/reference/fn/).

**Evidence status.** The race likelihood and causal decision rule are
analytical specifications. The old E83 softmax-mean failure follows directly
from its pooling equation. The shared-gate form has only a tiny failed smoke;
the guarded depth-4, 512/128, two-epoch controls have finished. Integral and
max pooling ended at 4.69% max-over-time accuracy. The stable-cause anytime
objective reached 6.25% max-over-time accuracy (8/128; chance-tail probability
0.31), 5.47% emitted accuracy, and 100% coverage at threshold 0.6; all tested
thresholds emitted every item and peak confidence saturated at 1.0. Layer 4
activity rose from 932 to 1,024 spikes per utterance. This diagnoses
overconfidence and activity growth, not a reliable above-chance result. The
matched stable-cause race-only control ended at 3.12% max accuracy, 97.66%
coverage, 4.8% emitted accuracy, and 0.981 mean peak confidence. A subsequent
guarded readout shadow probe used a fresh seed-2 initialization, not trained
weights: 31 of 32 near-gate final routes changed max-pooled CE by exactly zero;
one reduced it by 0.045. Its counterfactual-gradient norm was 2.2% of the
pathwise norm, with cosine 0.012. This small, untrained probe supports a
max-pooling winner-gap dead zone (§129), not a conclusion about trained hidden
routes. Separately, structured E59 event models reached 67.5% test accuracy
on SHD, so the task itself is learnable even though E83's learned deep stack
has not reproduced that result.

## 122. A calibrated posterior gives a selective-risk guarantee

Let $\mathcal F_t$ be the information in the event prefix through time $t$,
and suppose $\pi_t(c)=P(Y=c\mid\mathcal F_t)$ is the true posterior. For
threshold $\theta=1-\epsilon$, define the first-passage time

\[
\tau_\theta=\inf\{t:\max_c\pi_t(c)\ge 1-\epsilon\}.
\]

At any bounded stopping time $\tau_\theta$, if the system emits
$\hat Y=\arg\max_c\pi_{\tau_\theta}(c)$, then conditioning on the observed
prefix gives

\[
P(\hat Y\ne Y\mid\mathcal F_{\tau_\theta})
=1-\max_c\pi_{\tau_\theta}(c)\le\epsilon
\quad\text{on }\{\tau_\theta<\infty\}.
\]

Taking expectations yields the selective-risk bound
$P(\hat Y\ne Y,\,\tau_\theta<\infty)\le\epsilon P(\tau_\theta<\infty)$.
This is a precise foundation for “emit once sufficiently confident”; it does
not require a per-event label. It applies to speech, event-camera clips, and
other stream-to-one-label tasks. It guarantees neither that a crossing occurs
nor that the threshold is latency-optimal. A deadline and terminal fallback
separately provide full coverage.

The assumption is stronger than ordinary classifier calibration. Reliability
must hold for the posterior at the *selected first-crossing prefixes*, over
the deployed stopping policy, not merely for shuffled complete recordings or
for average confidence over all time bins. Global expected calibration error
does not imply this guarantee. If one had a uniform pathwise bound
$\lVert q_t-\pi_t\rVert_\infty\le\delta$ between emitted score $q_t$ and the
true posterior, thresholding $\max_c q_t(c)\ge1-\epsilon$ would instead give
error at most $\epsilon+\delta$ at emission. In practice that uniform bound
is not known; prefix-conditioned reliability plots, selective risk and
coverage on speaker/device-held-out streams are necessary empirical checks.

## 123. Why waiting, and when an early answer has value

Under the true posterior, $\pi_t(c)=E[\pi_{t+\Delta}(c)\mid\mathcal F_t]$;
the posterior is a martingale. Since $\max_c$ is convex,
$E[\max_c\pi_{t+\Delta}(c)\mid\mathcal F_t]\ge\max_c\pi_t(c)$. Thus more
evidence cannot increase optimal expected zero-one classification error.
This does *not* mean every sample path's confidence rises monotonically: new
events can reverse the leading class. It does mean a system that must always
answer and has no latency, energy, or deadline cost has no Bayes decision
reason to stop early. The useful module is a sequential decision system whose
application supplies a cost of waiting or a deadline.

For a posterior state $\pi$ at elapsed time $t$, the stopping Bellman equation
with zero-one loss and delay cost $c(t,\Delta)$ is

\[
V(\pi,t)=\min\left\{1-\max_c\pi(c),\;
c(t,\Delta)+E[V(\pi_{t+\Delta},t+\Delta)\mid\pi_t=\pi]\right\}.
\]

The optimal boundary generally depends on the application, remaining deadline,
and predictive dynamics; a fixed threshold is a convenient policy family,
not a universal optimum. A joint accuracy/coverage/latency/energy curve is
therefore more informative than a single threshold score. For asynchronous
systems, report latency both in physical time and in processed event count:
two streams with equal wall-time duration can have very different event
rates, and the cost of waiting may be tied to either clock.

## 124. Requirements for a reusable sparse stream classifier

The event-camera analogy exposes what a general module must preserve. Model
inputs as marked events $(t_i,x_i,m_i)$; for an event camera, $x_i$ includes
pixel location and $m_i$ polarity, while speech uses frequency channel and
amplitude. Each causal layer should update only states reached by the new
event or its sparse messages, and carry elapsed time explicitly. Simultaneous
events should be aggregated or processed with permutation-invariant updates;
an arbitrary tie order must not make the prediction depend on serialization.
The state should support both fast evidence and longer-lived context (for
example, multiple decay scales), while monitoring per-layer event rates and
payload covariance so deep stacks do not silently extinguish or amplify the
signal.

The readout should maintain a compact class-evidence state and update it only
on incoming messages. Its emitted payload can be the calibrated class
posterior vector, preserving uncertainty for downstream consumers. A
confidence crossing produces an early message; an explicit end-of-stream
event invokes a terminal answer or a configured abstention. The contract must
state how resets, missing events, bursts, clock drift, and distribution shift
affect state and calibration. Sparse edges alone are not evidence of sparse
runtime: operation counts and measured energy must include event generation,
routing, state updates, readout maintenance, and any timer work needed for
between-event threshold crossings.

The reusable abstraction is therefore a causal event-state encoder plus an
anytime calibrated decision rule, with vector-valued answer, deadline
semantics, and measured sparse execution. SHD is a useful first test because
its supervision is only one label per utterance; event-camera recognition is
a natural next domain with the same task structure but different marks, rates,
and nuisance variation.

## 125. Exact sparse confidence checks need not rescan every class

The current implementation forms all $C$ class scores at each simulation
time and applies softmax, so its readout work is $O(C)$ per time bin. That
operation is not an all-pairs attention layer, but it is still a dense scan
over classes. An event-driven readout can avoid that scan when each incoming
message changes only a sparse subset of class logits.

Let $s_c$ be the maintained class logit and suppose event $k$ changes only
classes in $A_k$, with $|A_k|=r_k$. The exact maximum posterior is determined
by

\[
\log\max_c p(c)=\frac{\max_c s_c}{T_o}
 -\log\sum_{j=1}^{C}\exp(s_j/T_o).
\]

Maintain two indexed trees over classes: a max tree whose root stores
$\max_c s_c$, and a log-sum-exp segment tree whose leaves store $s_c/T_o$
and whose internal nodes combine children by $\operatorname{logaddexp}$. A
point update to one class changes $O(\log C)$ nodes in each tree. Therefore
an event that changes $r_k$ classes updates the exact threshold statistic in
$O(r_k\log C)$ time, and tests $p_{max}\ge\theta$ at the root without
recomputing a $C$-way softmax. Simultaneous events should first be accumulated
into one per-class delta so ties remain order-invariant.

On a threshold crossing, emitting the full posterior vector still costs
$O(C)$ to materialize and transmit: that cost is required by the requested
payload dimension. If the downstream contract permits a class identifier,
confidence, or top-$K$ posterior with an explicit residual-mass bound, the
output cost can be reduced. Thus sparse decision maintenance and full-vector
payload size are separate design choices. The tree method is exact for sparse
point updates and fixed $T_o$. A class-dependent time decay changes many
leaves at once and removes the update advantage; shared additive shifts are
free because softmax is invariant to them, while shared multiplicative
changes to logits generally are not. Calibration is still required: an exact
softmax computation of uncalibrated scores does not make the threshold a
reliable error bound. A portable reference head implementing these semantics
is in `experiments/sparse_anytime_readout.py`; it has not yet been integrated
with or benchmarked on E83.

## 126. A point-process classifier must learn from events and silence

For a marked point process with class-conditional intensity
$\lambda_c(t,m\mid\mathcal F_{t^-})$, let
$\Lambda_c(t)=\int\lambda_c(t,m\mid\mathcal F_{t^-})\,dm$ be the total event
rate under class $c$. The exact class log posterior, up to a shared
normalizer, is

\[
s_c(t)=\log\pi_0(c)
 +\sum_{t_i\le t}\log\lambda_c(t_i,m_i\mid\mathcal F_{t_i^-})
 -\int_0^t\Lambda_c(u)\,du,
\qquad \pi_t=\operatorname{softmax}(s(t)).
\]

Each event contributes a mark-and-time likelihood jump; every interval with
no event contributes survival evidence. Thus a sequence-to-class learner
does not need labels on prefixes: a single utterance or clip label supervises
the final class likelihood. A proper terminal cross-entropy trains that
likelihood, while an early-output objective can reward correct first passage
and account for deadline or delay cost. The prefix posterior still needs
calibration under the stopping policy before its threshold has a risk
interpretation.

Silence has an execution consequence. If $\Lambda_c$ is the same for every
class, the no-event term is a shared logit shift and cancels from $\pi_t$; the
posterior only changes at input events. If class-specific rates differ,
silence itself changes the posterior between arrivals. A correct immediate
threshold policy must then schedule a timer for a possible between-event
crossing, or prove that no such crossing can occur before the next event. If
the survival update changes only a small class subset, the sparse trees in
§125 can process it as another sparse logit update. If it changes every class,
that timer has an $O(C)$ evidence update unless a special factorization is
available. This is why an event-stream module must specify whether it models
silence, not merely how it handles observed spikes.

## 127. E83 currently trains the realized event support, not support changes

The E74/E83 time-vector layer has two hard support decisions. For an input
message $v_i$ and candidate receiver $j$, it forms
$r_{ij}=q_j^\top v_i+c_{ij}$ and sends only if $r_{ij}>0$; a sent message
arrives at $a_{ij}=t_i+\tau_j\max(r_{ij},0)$. Separately, a receiver emits
only at discrete times when its threshold test fires. The current code makes
the first decision with `keep = r.detach() > 0`, and makes the firing mask
inside `no_grad`. Arrival-time refinement supplies an implicit gradient for
the time of a spike that already occurred, but neither support decision has a
gradient for a route or spike that did not occur.

This can be stated exactly on a fixed input batch. For a closed route
$r_{ij}\le0$, the message is removed before its arrival and its clamped delay
has zero derivative, so the task loss has
$\partial L/\partial r_{ij}=0$. For an open route the loss can change $r_{ij}$
through its delay and payload flow; that is pathwise credit conditional on
the route already existing. For a unit that never fires, there is no emitted
payload in the downstream graph, so its firing parameters receive no
end-to-end output credit on that example. Intermediate readout losses shorten
the path for realized spikes, but do not by themselves teach a closed gate or
a silent unit to create an event. These claims concern this implementation's
gradient graph, not the expressivity of the architecture.

There are two further restrictions. First, $q_j$ and $c_{ij}$ affect the
continuous loss only through the delay of an already-open route; once its
delay is clamped at $d_{max}$, even that derivative is zero. They do not get a
smooth signal for the usefulness of the route's payload. Second, E83's
tonotopic first-layer mask and random later-layer/readout masks are fixed
buffers sampled at initialization. A boundary signal could recruit a closed
edge inside that candidate mask, but no optimizer can recruit an edge omitted
from the mask without a separate topology candidate mechanism. E83 is
therefore a fixed-random-sparsity depth screen, not a learned-topology test.

The project already has the missing mathematical object in §§19 and 57: a
counterfactual boundary term for route birth/death. For a binary route with
score $r$, add logistic decision noise $\xi$ of scale $\sigma$ and execute
$h=\mathbf1[r+\xi>0]$. Then $P(h=1)=p=\operatorname{sigmoid}(r/\sigma)$,
and for route-specific losses $L_1,L_0$ the exact boundary derivative is

\[
\frac{\partial\mathbb E[L]}{\partial r}
=\frac{p(1-p)}{\sigma}(L_1-L_0).
\]

When $L_1<L_0$, gradient descent raises the score of the useful route. The
factor $p(1-p)/\sigma$ restricts work to the near-boundary band. This is the
missing boundary component; it supplements the pathwise derivative of delay
and payload within an open route. For E83,
$L_1$ must include the downstream consequences of the hypothetical vector
message or spike. A tagged shadow continuation of a bounded number of
near-misses gives the exact difference; a local linear estimate
$\nabla_VL^\top\Delta V$ is cheaper but must be audited for curvature and for
downstream support changes. The same calculation applies to a threshold
margin $g=V-\theta$ for an otherwise silent unit.

This yields a principled next step, rather than another loss sweep: measure
the distribution of closed-route scores and non-firing threshold margins;
on a small diagnostic batch shadow the top near-boundary candidates through
the remaining depth; compare their exact $L_1-L_0$ with local estimates; and
compute the missing boundary-gradient norm and direction relative to the
current pathwise gradient. Only if near-miss alternatives carry useful loss
differences should E83 add the §19/§57 credit, with a stated noise band and
shadow-work budget. The current depth-4 objective controls do not test that
mechanism because they all use the same fired-only support gradient.

## 128. Estimate the prefix posterior first; optimize stopping separately

There are two different learning targets in an anytime classifier:
$\pi_t(c)=P(Y=c\mid E_{\le t})$, the class posterior at each prefix, and the
stopping policy that decides when its risk is low enough to emit. A
confidence-race objective trains the probability of a correct integrated
winner (possibly discounted by latency). It does not uniquely identify the
instantaneous prefix posterior: many different score trajectories and hazard
gates can produce the same integrated class-win mass. Consequently a small
race loss or high coverage does not establish calibrated confidence at the
first-crossing time.

For any fixed prefix distribution, ordinary log loss has the exact
decomposition

\[
\mathbb E[-\log q_t(Y)\mid E_{\le t}]
=H(\pi_t)+D_{KL}(\pi_t\Vert q_t),
\]

so it is uniquely minimized by $q_t=\pi_t$. This remains true when the
training label is only attached to the complete utterance: each sampled
training prefix carries that utterance's eventual class, and across examples
the proper score estimates the conditional class distribution. It does not
assert that an individual prefix already determines the answer. At an empty
prefix, a balanced task's optimum is the class prior (uniform here), which
correctly stays below any useful confidence threshold.

A principled sequence-classification objective can therefore sample a modest
set of prefix times $t\sim\nu$ and minimize
$\mathbb E_{t\sim\nu}[-\log q_t(Y)]$, including event arrivals and the
end-of-stream prefix. The sampling measure $\nu$ states which latencies matter;
it need not evaluate every millisecond. Then calibrate $q_t$ and choose the
stopping boundary using the application's delay/error costs and
held-out-prefix risk curve. The race likelihood remains useful as a
decision-focused fine-tuning term, but should not be the sole evidence that
$q_t$ is a posterior. This separates three questions that E83 currently
conflates: whether the representation contains class information, whether
training produces calibrated prefix probabilities, and whether the chosen
stopping policy reaches a useful accuracy/latency frontier.

## 129. Max pooling creates a counterfactual credit dead zone

For one class, let the terminal score be the temporal maximum
$m_c=\max_t V_{t,c}$. Suppose a hypothetical route changes readout potentials
only on a set of times $A_c$, by increments $\delta_{t,c}$. The counterfactual
score is

\[
m'_c=\max\left(m_c,\max_{t\in A_c}(V_{t,c}+\delta_{t,c})\right).
\]

Define the winner gap $g_{t,c}=m_c-V_{t,c}\geq0$. If every increment is at
most its gap, $\delta_{t,c}\leq g_{t,c}$, then $m'_c=m_c$ exactly. If this
holds for every class, the whole pooled logit vector and any terminal loss
are unchanged: $L_1-L_0=0$. Thus adding logistic route noise does not by
itself create useful credit at a max-pooled head. It only estimates the loss
difference of the two executions; the readout can erase an otherwise real
change in the event trace. A max head's route-credit signal is concentrated
on alternatives that change a temporal winner.

By contrast, an integrated score changes by $\sum_t\delta_{t,c}$ (with the
declared time-step factor), while a smooth maximum
$m_{c,\tau}=\tau\log\sum_t\exp(V_{t,c}/\tau)$ gives a contribution weighted
by $\exp(-g_{t,c}/\tau)$. The smooth maximum removes the exact dead zone but
can still suppress alternatives far below the current winner. These readouts
are compatible with event-driven implementation: an event updates only its
affected class accumulator, and a log-sum-exp tree can maintain the temporal
normalizer; materializing the full class posterior is a separate $O(C)$
operation at a requested output.

The guarded E83 readout shadow audit is consistent with this prediction but
is only a mechanism probe: at seed-2 initialization (no trained checkpoint),
four held-out-speaker utterances and 32 near-gate final-readout insertions,
31 changed terminal max-pooling CE by exactly zero and one reduced it by
0.045. The estimated boundary-gradient norm was 2.2% of the pathwise norm
with cosine 0.012. The 4/4 baseline predictions were wrong and per-example
losses were 549–1,995, so this is not evidence about a trained model or
benchmark performance. It shows that final-readout near misses under max
pooling are mostly invisible in this deliberately small, untrained probe.

This narrows the next discriminating work. First repeat the intervention on a
saved trained checkpoint. For any route with negligible terminal $L_1-L_0$,
inspect the maximum winner gap and trace change; then compare max, integrated,
and smooth-max posterior heads at the same event support and proper sampled-
prefix log loss. Shadow hidden-unit threshold crossings through the remaining
layers separately, because those can change downstream event support and are
not covered by a final-head intervention. Keep the route-noise scale and
shadow budget fixed, and report the exact loss-difference distribution and
gradient alignment before deciding which boundary estimator merits training.

## 130. The label is a backward teaching pulse over a causal time window

Let $\mathcal F_t$ contain exactly the marked events observed by physical
time $t$, and let $\pi_t(c)=P(Y=c\mid\mathcal F_t)$. For a declared measure
$\nu$ over query times, the sequence label defines the prefix-prediction risk

\[
R_\nu(q)=\mathbb E\!\left[\int -\log q_t(Y)\,\nu(dt)\right].
\]

If the query clock is fixed independently of the example's future, conditioning
on $\mathcal F_t$ gives

\[
\mathbb E[-\log q_t(Y)\mid\mathcal F_t]
=H(\pi_t)+D_{KL}(\pi_t\Vert q_t).
\]

Thus hard utterance labels at randomly sampled *causal prefixes* teach the
posterior; they do not claim that every prefix already identifies its class.
The output error at one sampled prefix is
$\partial L/\partial z_c=(q_t(c)-\mathbf1[c=Y])/T$. In expectation it is
zero at the correct posterior. A sample time may be scored with a one-hot
utterance label, while the population objective still represents uncertainty
through its posterior optimum.

The time-sampling rule is part of the theorem. If a query time is chosen as a
fraction of that same utterance's final event time, the sampling weight depends
on future data. Under a future-dependent weight $w$, the pointwise optimum is

\[
q^*_t(c\mid\mathcal F_t)=
\frac{\mathbb E[w\mathbf1[Y=c]\mid\mathcal F_t]}
     {\mathbb E[w\mid\mathcal F_t]},
\]

which need not equal $\pi_t$. E83's first window sampler used each example's
last event to set its upper bound, so it did not have the claimed causal
posterior interpretation when duration and class covary. The corrected design
draws stratified query times from a fixed physical-time horizon shared across
examples, with the horizon stated in milliseconds; the terminal EOS score is
an additional, separate target. The horizon is an experimental choice that
sets which response times receive training weight.

Here is the asynchronous credit path. Between arrivals the state follows a
flow $h_i^- = \Phi_{\Delta_i}(h_{i-1}^+)$; at event
$(t_i,x_i)$ it jumps by $h_i^+=\Psi(h_i^-,x_i;\theta)$. At a sampled query
time $\tau$, inject the readout error
$\delta_\tau=\nabla_{z_\tau}[-\log q_\tau(Y)]$. Set the state adjoint
$a_\tau=J_G(h_\tau)^\top\delta_\tau$. Moving backward across a jump and
the preceding silent interval gives

\[
a_i^- = J_{h}\Psi_i^\top a_i^+,
\qquad
a_{i-1}^+ = J_h\Phi_{\Delta_i}^\top a_i^- ,
\]

and the parameter contribution at the event is
$J_\theta\Psi_i^\top a_i^+$ (plus the flow-parameter term, if the flow is
learned). Only events at or before $\tau$ are traversed. With a sparse event
graph, this reverse sweep follows stored event edges and their actual delays;
it need not visit empty millisecond bins. The equivalent forward eligibility
recurrence is $E_i^-=J_h\Phi_{\Delta_i}E_{i-1}^+$,
$E_i^+=J_h\Psi_iE_i^-+J_\theta\Psi_i$, followed by the three-factor product
$\delta_\tau J_GE_\tau$. Full eligibility is generally expensive; sparse
event adjoints or local eligibility traces are the implementation choices.
This is the precise meaning of a label signal traveling backward and using
the local state: the label supplies the modulatory error, while stored local
Jacobians or eligibility carry the credit to the events that formed that
state.

For $K$ sampled prefixes, the training gradient is their weighted sum of
backward pulses. Stratified Monte Carlo gives an unbiased estimate of the
window integral without a dense loss at every time bin. A sparse readout may
materialize the $C$-class log normalizer at those few query times; inference
can maintain its exact confidence tree only at class-logit arrivals and
materialize the vector at emission. This saves the time-by-class sweep in the
head. It does not make the current E83 hidden simulator asynchronous: its
time-vector layers still scan a 1 ms grid, and autograd retains their pathwise
graph. The current event readout is an additive logit accumulator driven by
hidden payload, gap, elapsed time, and EOS features. Its posterior only changes
at message or EOS events. It cannot emit a new confidence crossing during a
silent interval unless a timer/survival update is added; §126 gives the exact
class-conditional survival term.

## 131. A losing route gets the boundary signal from its shadow

The prefix pulse above differentiates the realized event path. It cannot
credit a route that was rejected by E74's hard content gate. For candidate
message $k$, with score $r_k$ and hard activity $h_k=\mathbf1[r_k>0]$, add
logistic boundary noise $\xi_k$ of scale $\sigma$ during the gradient
derivation. Then $p_k=P(h_k=1)=\operatorname{sigmoid}(r_k/\sigma)$ and the
exact derivative of the noise-averaged loss is

\[
\nabla_\theta\mathbb E_\xi[L]
=\text{pathwise terms}
+\sum_k \frac{p_k(1-p_k)}{\sigma}
   (L_{k,1}-L_{k,0})\nabla_\theta r_k .
\]

$L_{k,1}$ and $L_{k,0}$ are two executions with this one route on and off,
holding the other gates fixed. For a closed route, force it on and subtract the
ordinary loss; for an open route, drop it and subtract the shadow loss from
the ordinary loss. The difference must be measured after the entire remaining
network, since inserting one vector can change later spikes and which later
routes win. This is the same boundary term derived in §§19 and 57, now paired
with the causal prefix-window loss. The prefix labels say which complete
outcome was correct; the route's shadow says whether this lost message would
have improved those time-indexed predictions.

Exact shadowing every candidate is wasteful. If $S_\ell$ is the set of
near-boundary candidates in layer $\ell$, then uniform sampling of $m_\ell$
routes with the Horvitz–Thompson factor $|S_\ell|/m_\ell$ is unbiased for the
sum of boundary terms restricted to $S_\ell$. That fact does not control its
variance: a large candidate set can multiply one noisy shadow difference by a
large $|S_\ell|/m_\ell$.

E83 tested this estimator at depth four. With 128 training examples and one
shadow per layer and minibatch, it sampled 128 shadows per epoch from about
480,000 eligible routes. The loss then rose from 1974 to 17044, the final
layer's activity grew from 1034 to 1691 spikes per utterance, and prefix NLL
exploded. This is direct evidence that the unbiased sum estimator is unusably
high-variance in this configuration; it is not evidence that the exact
boundary derivative is wrong.

The next implementation uses a bounded, normalized local rule. For each
layer, average only the sampled route terms, clip the loss difference, and
clip the resulting gradient norm:

\[
\tilde g_\ell=\operatorname{clip}_{G}\!\left[
\frac{1}{m_\ell}\sum_{k\in sample_\ell}
\frac{p_k(1-p_k)}{\sigma}
\operatorname{clip}_{d}(L_{k,1}-L_{k,0})\nabla_\theta r_k
\right].
\]

This estimates a mean near-route signal per layer, not the full sum over
candidates. Loss clipping also makes it biased for the original logistic-noise
objective. The separate update $\theta\leftarrow\theta-\eta_{cf}\tilde g$
keeps its scale explicit and prevents a rare large shadow from overwhelming
the pathwise optimizer. Since $r=q_j^\top v+c_{ij}$, its local Jacobian
updates the gate, bias, and the source payload computation; the shadow
continuation itself is detached. This is a controlled heuristic motivated by
the exact derivative, not a new unbiased estimator or a trainability theorem.
Candidates outside the near-boundary band and edges outside the fixed
connectivity mask remain unreachable by this signal. Hard silent-neuron firing
still needs a separate boundary term.

The pathwise-only depth-four control stayed stable but near chance on its
32-example held-out subset. Thus the current evidence identifies a concrete
variance failure in one counterfactual estimator; it does not yet show that
the normalized update improves recognition. Report eligible routes, shadow
count, signed and absolute $L_{on}-L_{off}$, clipping rate, update norm,
activity, and test accuracy separately.

## 132. What gradient statistics say about the learning signal

Let a sampled candidate's vector contribution be
$u_k=b_k\nabla_\theta r_k$, where
$b_k=p_k(1-p_k)(L_{k,1}-L_{k,0})/\sigma$. In one layer's eligible band
$S$ of size $N$, the exact restricted gradient is $G=\sum_{k\in S}u_k$.
Uniform sampling of $m$ candidates without replacement gives the unbiased
total estimator

\[
\widehat G_{HT}=\frac Nm\sum_{k\in sample}u_k,
\qquad
\operatorname{Cov}(\widehat G_{HT})=
\frac{N^2}{m}\left(1-\frac mN\right)S_u,
\]

where $S_u$ is the finite-population covariance of the candidate vectors.
The standard deviation therefore carries an $N/\sqrt m$ scale. A large
candidate count does not by itself make the estimate informative: if most
candidate effects cancel, its signal-to-noise ratio can get worse as the
band grows. E83's roughly 480,000 near routes and 128 shadows per epoch imply
about 3,750 candidates per sampled route when totals are compared per layer;
the exact per-minibatch ratio varies with the event counts. This makes the
observed instability of the scaled total estimator analytically plausible.

The normalized sample mean
$\widehat{\bar G}=m^{-1}\sum u_k$ instead estimates $\bar G=G/N$ and has
covariance $(1-m/N)S_u/m$. It removes the $N$ multiplier but changes the
quantity being optimized: it does not estimate the full candidate sum. The
loss-difference clip makes it biased even for this mean. A separate update
with global gradient cap $G_{max}$ has Euclidean step norm at most
$\eta_{cf}G_{max}$ per minibatch. The cap controls update size, not usefulness;
clipping can also hide a badly scaled signal, so both raw and applied norms
must be logged.

The route coefficient has a useful local bound. Since
$p(1-p)\le1/4$, its magnitude before clipping is at most
$|L_{on}-L_{off}|/(4\sigma)$. With E83's $\sigma=0.25$ and loss-difference
clip 5, each scalar coefficient is at most 5 before the global gradient
clip. Its parameter gradient splits into direct router terms and upstream
payload eligibility:

\[
\nabla_q r=v,\qquad \nabla_{c_{ij}}r=1,\qquad
\nabla_{\theta_{payload}}r=q^\top
\frac{\partial v}{\partial\theta_{payload}}.
\]

Thus a counterfactual can teach the query and route bias even when the
message lost, while teaching the source representation depends on the
payload Jacobian through earlier layers. Per-layer norm and cosine are needed
to tell these channels apart.

For pathwise gradient $g_p$ and counterfactual gradient $g_c$, the local
first-order change from adding $-\eta g_c$ to ordinary SGD is
$-\eta\langle g_p,g_c\rangle$ in the pathwise objective. Cosine sign gives
the direction of interference; the norm ratio gives its scale. A positive
cosine supports local cooperation, a negative cosine predicts a first-order
increase, and a near-zero cosine means the extra update is nearly orthogonal.
This interpretation is Euclidean: E83 uses AdamW for the pathwise step and a
separate clipped SGD correction, so optimizer preconditioning and curvature
can change the actual interaction. Layerwise cosines, raw norm ratios,
clipped update norms, shadow-delta variance, and helpful-route fraction are
therefore complementary diagnostics rather than a single “gradient quality”
score.

A Bayesian summary can separate uncertain layer means from a deliberately
weak initialization prior. For scalar shadow utility, use a robust sampling
model such as
$\Delta_{\ell k}\sim t_\nu(\mu_\ell,s_\ell)$,
$\mu_\ell\sim\mathcal N(0,\tau_0^2)$, with a broad $\tau_0$ before learning;
report $P(\mu_\ell<0\mid data)$, since negative $L_{on}-L_{off}$ favors
opening. A Beta-binomial model for the helpful-route fraction is a useful
secondary check, but discards effect magnitude. The posterior should be
updated from fresh shadows as the network changes, or discounted by recency;
an old route-value posterior is stale after the representation moves.

This posterior is a diagnostic, not the step-size rule. The action gradient
depends on the vector $\nabla_\theta r_k$, and the scalar mean delta alone
does not identify its direction, covariance, or optimizer-metric length.
Estimate uncertainty of the sampled gradient or of its projection onto the
proposed update, then use a separate trust radius in the optimizer's
parameter metric and verify the paired loss change on held-out prefixes.
With only about 30 shadows per layer in E83, the observed per-layer means
are all small relative to their across-shadow standard deviations; a weak
prior cannot turn that into confident evidence. More shadows or a better
stratification by route score and layer are needed before tuning gain.

E83 evaluates each shadow at the same sampled prefix times as its factual
execution. This common-random-number pairing cancels part of the query-time
Monte Carlo noise in $L_{on}-L_{off}$. Prefix-time stratification separately
controls the variance of the window-integrated proper log score. Keeping data,
prefix, and shadow random streams separate makes matched comparisons
reproducible and prevents route sampling from changing the training examples.

## 133. Deep language modeling already has causal labels; hidden route birth is the gap

For a character stream $x_0,\ldots,x_{L-1}$, E77 predicts each next character
from its prefix, with objective

$$
\mathcal L(\theta)=\frac{1}{BL}\sum_{b=1}^B\sum_{k=0}^{L-1}
 -\log p_\theta(x_{b,k+1}\mid x_{b,\le k}).
$$

This is ordinary causal supervision at every prediction position. The SHD
problem of choosing when to attach one utterance label does not transfer: the
language model has no missing target time or unobserved class posterior. Its
output softmax is over the small character alphabet, while hidden event
messages and their candidate graph can remain sparse. This does not make all
of E77 sparse: its token-retrieval module currently materializes dense causal
$[B,H,L,L]$ query-key scores, so its retrieval search still has quadratic
training cost. The route-credit probe isolates a separate hidden-topology
question and does not resolve that cost.

E77's content route is $g=\mathbf 1[r>0]$, where
$r=q_j^\top v_e+c_{ej}$. A route that is open receives ordinary derivatives
through delay and payload; a closed route has no path through those operations.
Under logistic score perturbation of scale $\sigma$,
$p_\sigma(r)=\operatorname{sigmoid}(r/\sigma)$, the local expected objective
has boundary derivative

$$
\nabla_r\mathbb E[\mathcal L]
=\frac{p_\sigma(r)(1-p_\sigma(r))}{\sigma}
  (\mathcal L_{\rm open}-\mathcal L_{\rm closed}).
$$

The loss difference must be measured by toggling the route and replaying its
downstream consequences, including later event generation and the causal
token losses. For a route arriving at $t_e$, only prediction states at or
after that arrival can change. Using the identical token batch in both runs
cancels unrelated sample variation. The E77 probe samples near-zero scores
separately in each layer and averages within layer; it does not multiply by
the inverse sampling probability over the huge candidate population. It logs
the paired loss deltas and counterfactual/pathwise gradient alignment before
allowing a separate, clipped counterfactual update. This is deliberately a
measurement path first: E83's near-zero cosine and unstable total estimator
are reasons to measure transfer, not to assume it.

There is an earlier prerequisite: a layer that never emits has no realized
pathwise credit, and its route shadows can have exactly zero effect if toggles
still do not create an event. Let $V_{btk}$ be a layer's pre-reset voltage
under a short sample of actual training inputs, with $M$ receiver units. For
an initial aggregate event budget $\rho$ per character, use the empirical
voltage quantile as an initial threshold estimate,

$$
\theta=\widehat F_V^{-1}(1-\rho/M),
$$

where $\widehat F_V$ is the empirical voltage CDF. Without resets this gives
approximately $M\Pr[V>\theta]=\rho$ threshold attempts per time bin. It is
only a scale guess: reset, temporal dependence, and finite samples change the
realized count. E77 now replays the exact threshold/reset recurrence on saved
voltage traces, then searches the threshold against the realized per-character
spike count. This avoids rerunning the whole network for each candidate.
Calibration is depth ordered, so each deeper layer's traces come from the
already calibrated event stream below it.

The distinction mattered at default width. On the same one-seed, depth-4
10k-character smoke, the quantile-only estimate produced near-silent initial
rates and test rates $[0,0.007,0.009,0.004]$; the revised procedure matched
the initial rates to $[0.096,0.082,0.094,0.100]$ around a target of 0.1. All
four hidden layers had nonzero gradients at all 13 validation checkpoints,
versus $[1,9,12,6]$ checkpoints under quantile-only calibration. By the
selected checkpoint, event rates had moved to $[0.204,0.190,0.456,0.802]$.
Thus activity calibration reopened the deep learning path in this smoke, but
the learned representation changed the rate substantially. This is an
initialization diagnostic, not evidence for language-model quality, stable
homeostasis, or scaling. Online threshold adaptation remains unimplemented.

For uncertainty, route deltas should be grouped by layer and relevant
conditions (score band, active/closed status, event age, and token region).
With a broad zero-centered prior on a layer mean $\mu_\ell$ and a
Student-$t$ likelihood for heavy-tailed shadow deltas, initially weak prior
precision lets the first observations move the estimate; posterior uncertainty
falls only when repeated shadows agree. If the representation shifts, discount
or reset stale observations. The posterior over route utility answers whether
opening routes is promising; it does not set the parameter step size. Bound the
actual update in the optimizer metric and report both its norm and its cosine
with the causal pathwise gradient. This distinction preserves early
responsiveness without turning a noisy first shadow into an unbounded update.

**Prediction.** If route birth is a material trainability bottleneck in E77,
the shadow term should become measurable at depth and should align with
pathwise changes in future-token loss for some layers/conditions. If it is
negligible, poorly aligned, or unstable across repeated batches, optimize the
existing smooth delay/payload gradients and investigate event firing,
conditioning, or the data/compute setup instead. A successful smoke or one
seed is not a depth-scaling result.

## 134. Local Bayesian trust should follow conditional evidence, not a global clock

The prior/evidence proposal has a precise local interpretation. Let a unit's
conditional response be linearized around its current parameters, with local
eligibility or state feature $z$ and scalar supervised credit $y$:

$$
y\mid z,w\sim\mathcal N(z^\top w,\sigma^2),\qquad
w\sim\mathcal N(m_0,\Lambda_0^{-1}).
$$

After observed state/credit pairs $(z_i,y_i)$, the Gaussian posterior has

$$
\Lambda_n=\Lambda_0+\sigma^{-2}\sum_{i=1}^n z_i z_i^\top,\qquad
m_n=\Lambda_n^{-1}\left(\Lambda_0m_0+
\sigma^{-2}\sum_{i=1}^n z_i y_i\right).
$$

For one new observation, writing $\Sigma_n=\Lambda_n^{-1}$, the exact update
is

$$
m_{n+1}-m_n=
\frac{\Sigma_n z}{\sigma^2+z^\top\Sigma_n z}
\left(y-z^\top m_n\right).
$$

This gives the proposed schedule without a global epoch clock: a broad prior
(small $\Lambda_0$) gives a novel, informative conditional observation more
influence; repeated consistent observations of the same state direction add
precision and shrink its later influence. The gain also depends on observation
noise and feature novelty. High-noise credit should not be trusted merely
because it arrived early, and evidence in one state direction does not make an
unvisited conditional well known. A scalar count or global learning-rate
decay cannot express that geometry.

Three qualifications matter for Sleeping Machines. First, unlabeled activity
is evidence about voltage scale and event frequency, but not evidence about
which class or token is correct. Task credit must still reach the conditional
state, including counterfactual routes that did not fire. Second, the Gaussian
formula is exact only for a fixed linear-Gaussian conditional; nonlinear
neurons require a local Laplace/online natural-gradient approximation, and
heavy-tailed route deltas call for robust likelihoods. Third, representation
drift makes old precision stale. For a changing conditional, use discounted
sufficient statistics (or a drift-triggered reset) and measure uncertainty
per layer and state stratum. Otherwise accumulated confidence can freeze a
unit after its input semantics have changed.

This suggests a discriminating experiment, not a new optimizer setting yet:
log local state occupancy, effective sample size, predictive residual scale,
and posterior variance for each unit/route stratum; compare constant-gain,
global decay, and evidence-conditioned gain at matched data and compute. Use
held-out causal loss and gradient alignment to judge the update, and keep a
trust radius in the optimizer metric. The scalar route-utility posterior in
§133 and this parameter posterior are different objects: the former estimates
whether opening a route helps; the latter estimates how much a local
conditional parameter remains uncertain. E77 currently implements neither
posterior-driven synaptic gains nor online homeostasis.

## 135. The TV-quiz task is posterior filtering followed by a stopping rule

Represent an utterance as a marked event history $\mathcal F_t$ and let $Y$
be its one final class. The online object is the causal prefix posterior
$\pi_t(c)=P(Y=c\mid\mathcal F_t)$. For any fixed, exogenous query-time law
$\nu$, the proper training risk is

$$
\mathcal R_\nu(\theta)=
\mathbb E_{(X,Y)}\mathbb E_{t\sim\nu}
[-\log\pi_\theta(Y\mid\mathcal F_t)].
$$

Its population minimizer is the true conditional posterior at each sampled
prefix. Reusing the utterance label at several prefixes does not assert that
each prefix already determines the class: at an empty prefix the correct
target is the class prior. The query-time law must be chosen without looking
at the future endpoint. Only after posterior estimation should we calibrate
the first-output time
$\tau=\inf\{t:\max_c\pi_t(c)\ge1-\epsilon\}$ and emit the full vector
$u(\tau)=\pi_\tau$. If this is a true posterior at a stopping time, conditional
error among emitted answers is at most $\epsilon$. A no-crossing fallback at
an observed EOS is a separate, late answer; raw softmax confidence needs
selected-prefix calibration before the guarantee applies.

For class-conditional point-process intensities, the exact log posterior is

$$
s_c(t)=\log\pi_0(c)+
\sum_{t_i\le t}\log\lambda_c(m_i,t_i\mid\mathcal F_{t_i^-})
-\int_0^t\Lambda_c(u\mid\mathcal F_{u^-})\,du,
\qquad \pi_t=\operatorname{softmax}(s(t)).
$$

An event contributes a mark/time likelihood jump. An interval of silence
contributes the negative integrated class event rate. For constant rates
between hidden events, a silent gap of duration $\Delta$ changes class log
odds by $-(\Lambda_c-\Lambda_d)\Delta$. The label can therefore teach the
network from both a positive event and a missing event. At a sampled prefix,
the posterior cross-entropy derivative is
$\partial L/\partial s_c=p_c-\mathbf1[Y=c]$; the survival term carries this
credit through the local state that predicted the gap. This answers how to
train when no output has fired: score posterior error at causal prefixes and
include no-event likelihood, instead of waiting for an output event to exist
before assigning credit.

There is a code/theory mismatch in E83's sparse event readout. Class logits
accumulate hidden-event updates and one EOS update. Between hidden events, a
prefix query sees no clock/survival update, so its posterior is piecewise
constant. The gap feature is attached only when the next event arrives and
cannot support an earlier crossing. This is correct only under
$Y\perp\text{survival to }t+\Delta\mid\mathcal F_t$. The direct test is the
held-out predictive value of the gap:

$$
I(Y;\text{no event in }(t,t+\Delta]\mid\mathcal F_t),
$$

measured by paired prefix log loss with and without a class-conditional
silence update, sampling queries inside silent intervals. If this gain is
zero, event-triggered updates suffice. If it is positive, attach a learned
class-rate state to the current causal hidden state. Each new hidden event
updates its mark score and rate; between events the rate integrates
analytically, and inference schedules a timer only for a possible confidence
crossing. Under constant rates, each logit is linear in gap duration, so a
candidate crossing solves $\pi_c(t)=\theta$ without millisecond polling.

The systematic SHD sequence is therefore: verify a small-set terminal fit and
the speaker split; fit causal prefix posteriors with fixed-time proper log
loss; measure event-mark and silent-survival information separately; calibrate
first-crossing coverage, emitted accuracy, latency, and EOS fallback on
validation speakers; then increase depth and measure pathwise and lost-route
credit by layer. This isolates identifiability, representation, posterior
estimation, topology credit, and stopping policy instead of asking a single
race loss to solve all five.

## 136. Sparse multi-depth evidence reduces serial credit bottlenecks

E83's strict stack gives the final readout only the last layer's event stream.
An early event can affect the objective only if it survives and is transformed
by every later hard route. With per-layer event transmission probabilities
$q_1,\ldots,q_D$, a simplified independent-route calculation gives the
survival factor $\prod_{k=1}^{D-1}q_k$; actual route events are dependent, but
the product exposes a depth-sensitive failure mode. Local auxiliary heads
improve layerwise optimization, yet do not make early evidence part of the
final prediction.

The new `all_depths` readout gives each layer $k$ its own sparse event logit
stream

$$
z_c^{(k)}(t)=b_c^{(k)}+
  \sum_{e\in E_k:\,t_e\le t}\phi_{k,c}(v_e,\Delta_e,t_e),
\qquad z_c(t)=\sum_{k=1}^{D}z_c^{(k)}(t).
$$

Cross-entropy is applied to $z(t)$ at fixed, causal physical-time prefixes.
The terminal EOS terms from all heads are aligned at the common stack
deadline. These are learned additive discriminative potentials; summing them
does not assert statistical independence between levels. The existing local
prefix losses remain as deep supervision. In the counterfactual estimator, a
route toggle is scored by the change in this same fused objective, so an
opened route receives credit for both its direct readout contribution and any
downstream events it causes.

This relaxes a serial credit bottleneck; it does not prove that the
compositional stack is trainable at arbitrary depth. A model might solve the
task mostly through its shallow branches. Therefore compare `deepest` and
`all_depths` on matched data, seeds, widths and update budgets, report each
layer's event contribution, and ablate branches at inference. The added work
is sparse: it is proportional to the number of emitted events times each
head's sparse class fan-out, not to every simulation tick. Readout work grows
roughly with the number of active layers. This also does not fix §135's
separate gap: class logits still wait for an event or final EOS instead of
changing continuously with evidence from silence. Results from this design
are pending; the implementation alone is not evidence of accuracy or
supremacy.

## 137. Deep event trainability requires support survival and boundary credit

Section 127 predicted two distinct hard-support gaps in E83: a closed message
route and a hidden unit's absent spike. The seed-6 depth-four measurements now
separate them. Write a hidden firing margin as

$$
g_{bjt}=V_{bjt}-\theta(1+R_{bjt}),\qquad h_{bjt}=\mathbf 1[g_{bjt}\ge0],
$$

where $R$ is the refractory trace used by `TVLayer`. The ordinary autograd
path differentiates payloads and the refined time of a spike conditional on
$h=1$; it does not differentiate the Boolean fire mask, which is constructed
under `no_grad`. If a unit has no emitted event, its payload is absent from the
downstream event list, so that example supplies no task gradient through that
unit's firing decision. An auxiliary classifier at that depth still sees its
EOS/bias features, but it cannot recover the missing input-dependent event
path.

There is a stronger support invariant for the current strict chain. With no
incoming events, the layer's state is identically zero, its membrane potential
is zero, and its positive threshold prevents a spontaneous spike. Therefore,
for every utterance $x$,

$$
E_\ell(x)=\varnothing\;\Longrightarrow\;E_{\ell+1}(x)=\varnothing,
\qquad
\mathcal A_{\ell+1}\subseteq\mathcal A_\ell,
$$

where $E_\ell(x)$ is the emitted event set and
$\mathcal A_\ell=\{x:E_\ell(x)\ne\varnothing\}$ is the active-example
support. Hence event coverage $c_\ell=P(x\in\mathcal A_\ell)$ cannot
increase with depth in this architecture. This is exact, not a mean-field
approximation. Mean event count has a different factorization,
$n_\ell=c_\ell\,\mathbb E[|E_\ell|\mid x\in\mathcal A_\ell]$: coverage can
shrink while event multiplicity on the surviving examples explodes. Thus
depth-eight spike-count cascades do not contradict support extinction. Sparse
`all_depths` readout supplies direct losses at existing layers but does not
alter this hidden support invariant. A sparse skip from the raw event stream,
a skip from an earlier active representation, a nonzero baseline drive, or
another explicit support-recruitment mechanism would change the invariant;
each has different work and stability costs and must be compared as an
architectural ablation. E83 now has an optional sparse layer-1-to-deeper-layer
event skip for that controlled comparison. It still cannot recover an example
on which layer 1 itself emits nothing, so the experiment isolates intermediate
chain extinction rather than solving every possible silence state.

This is the exact boundary term for a smoothed fire decision. Add independent
logistic perturbation $\epsilon$ of scale $\sigma$ to $g$, so
$p=\Pr[g+\epsilon\ge0]=\operatorname{sigmoid}(g/\sigma)$. Let $L_1$ and
$L_0$ be the same causal prefix objective after forcing this spike on or off,
replaying its refractory effect, and running the remaining network. Then

$$
\frac{\partial\mathbb E[L]}{\partial g}
=\frac{p(1-p)}{\sigma}(L_1-L_0).
$$

The existing E83 counterfactual code estimates this form for content routes,
not for hidden spike birth/death. Its route shadow cannot substitute for a
spike shadow: the two binary variables have different margins and different
state consequences. A spike shadow must include the changed refractory trace
in its layer and every downstream event it causes. If only $m$ of $N$ eligible
margins are sampled uniformly, the Horvitz–Thompson estimate of the total
boundary sum is $(N/m)\sum_{i\in S}p_i(1-p_i)(L_{i,1}-L_{i,0})\nabla g_i/\sigma$.
That total can have high variance; a clipped layer-mean update is more stable
but is a different, biased objective, as §132 already establishes for routes.

The other condition is event-rate impedance. Define
$n_\ell=\mathbb E[N_\ell]$, the mean number of layer-$\ell$ events per
utterance, and $\rho_\ell=n_{\ell+1}/n_\ell$ when $n_\ell>0$. The product of
these ratios is only a first-moment branching approximation because events
are correlated and one unit may fire repeatedly. Still, a near-zero $n_\ell$
removes most downstream examples from the pathwise credit support, while
repeated $\rho_\ell\gg1$ grows event work and accumulated class evidence.
Section 120 explains why a fixed threshold and weight scale do not preserve
this operating point: the input rate, payload covariance, decay, and reset
change with depth.

The E83 evidence matches both predicted regimes:

| Run and evaluation event counts per utterance | Held-out result | Reading |
|---|---:|---|
| Depth-4 pathwise, `[4, 2, 0, 0]` after rounding to whole events | 6.25% (2/32) | Layers 3–4 averaged below 0.5 events per utterance. In epoch 2's first training minibatch, the main-loss gradient norms for all four hidden layers were exactly zero; the auxiliary norms were 5.72, 1.48, 0, 0. |
| Depth-4 route-counterfactual, `[21, 15, 4, 14]` | 6.25% (2/32) | Route openings restored some deep activity, but the model remained at chance; the route update still did not measure spike birth/death. Late-prefix NLL reached 19.735. |
| Depth-8 `all_depths`, epoch 1 `[16, 9, 3, 6, 27, 51, 141, 250]` | 12.5% (4/32), preliminary | Early event attenuation is followed by late amplification; training loss was $2.78\times10^6$ and late-prefix NLL 39,814.7. |
| Matched D4 seed-6, eval128: `all_depths` `[39, 16, 7, 22]`; `deepest` `[12, 1, 1, 1]` | 17/128 vs 7/128 terminal; race-plus-fallback 20 vs 7 | Multi-depth loss preserves support and improves paired decisions; layer-4 ablation has no terminal-accuracy effect and late-prefix NLL is 17.57. |
| D4 layer-1 skip, seeds 6 and 7 | 9/128 and 5/128 terminal | Layer-4 support rises to 99.2% and 100%, but paired accuracy does not improve; seed-7 late-prefix NLL reaches 48.80. |

In the original 32-example screen, the matched depth-four `all_depths` run
shows that adding direct sparse
readouts does not itself reopen hidden support. Across its four epochs,
held-out event means rounded from `[14, 6, 5, 28]` to `[16, 2, 1, 0]`,
`[9, 1, 0, 0]`, and `[10, 1, 0, 0]` per utterance. Terminal accuracy was
6.25%, 3.125%, 6.25%, and 6.25%. On the first training minibatches of epochs
3 and 4, the main-loss gradient norms in layers 3 and 4 were exactly zero.
Readout fusion alone therefore did not solve the event-support problem in
that trajectory; the
matched deepest-only control below shows a different, still inconclusive
accuracy signal.

The matched deepest-only control has now finished. At its fixed fourth-epoch
endpoint it classified 5/32 held-out utterances correctly (15.6%), versus
2/32 (6.25%) for `all_depths`; the first three deepest-only epochs were
9.4%, 0%, and 3.1%. Conditional on a fixed 5% chance classifier, 5/32 has an
uncorrected one-sided binomial tail of 0.020, but this was one of several
arms/epochs inspected. More directly, on the same 32 examples the arms had 5
deepest-only-only correct cases and 2 `all_depths`-only correct cases, with an
exact two-sided McNemar $p=0.453$. The final deepest-only prefix NLLs were
`[4.008, 4.741]`, worse than the uniform 20-class NLL $\log 20\approx2.996$.
This is a first above-chance-sized accuracy signal in the matched D4 readout
screen, not a validated performance result: it is small-sample, statistically
inconclusive as a paired head comparison, and poorly calibrated. The larger
held-out evaluation must use a fixed endpoint and report both accuracy and
proper loss.

A fresh 128-example held-out run of the `all_depths` arm has now reached
17/128 terminal accuracy (13.3%) at its fixed fourth-epoch endpoint. Under
the simple independent 5% chance model, the one-sided binomial tail is
$2.31\times10^{-4}$; the observed rates were 11/75 (14.7%) and 6/53 (11.3%)
on the two held-out speakers, rather than coming from only one voice. At the
fixed 0.6 stopping threshold, the output race emitted on 42/128 examples,
with 10/42 correct (23.8%). The emitted class payloads had mean maximum
confidence 63.8%, so this is a direct sequential-calibration failure, not
merely a low-coverage policy. Mean emission latency was 377 ms, and anytime
accuracy was 15.6% with terminal fallback. The final late-prefix NLL remained 17.57, and the
epoch-4 layer-4 standalone head was 10.2%; removing it from the fused logits
left fused accuracy at 13.3%. Standalone head accuracies were
`[7.0, 10.2, 8.6, 10.2]%`; leave-one-head-out fused accuracies were
`[10.2, 7.0, 10.2, 13.3]%`. The layer-2 head has the largest measured
leave-one-out contribution, while layer 4 has none. These are readout
ablations on one checkpoint, not retrained depth ablations. This is a
discrimination lead but not calibrated posterior learning or evidence of a
deepest-branch gain. The
nominal chance tail is exploratory: only two held-out speakers, one seed,
multiple prior arms, and repeated epoch inspection limit its interpretation.

The seed-6 128-example control is now paired against `deepest` at the same
training and evaluation limits. These two commands have identical selected
examples and training RNG streams; `readout_fusion` is the changed factor.
At epoch 4, terminal accuracy was 17/128 for `all_depths` and 7/128 for
`deepest`; layer-4 active-example coverage was 61.7% and 4.7%, respectively.
For the deployed output rule (first threshold crossing, otherwise terminal
fallback), saved paired predictions were correct on 20/128 versus 7/128
examples: 19 discordant cases favored `all_depths` and 6 favored `deepest`
(exact McNemar $p=0.0146$). This is nominal evidence for the fused
objective/readout package on this fixed evaluation subset; the examples come
from only two held-out speakers, so utterance-level pairing does not establish
speaker-level replication. At threshold 0.6, `all_depths` emitted 42 answers,
10 correct, versus 2 answers and no correct emission for `deepest`. Despite
the accuracy difference, late-prefix NLL favored `deepest` (4.66 versus
17.57), with both above uniform 2.996. Thus additive evidence from multiple
depths preserves support and changes decisions, but also yields severe
overconfidence. Removing the deepest head from fused terminal logits left
`all_depths` accuracy at 13.3%; deeper-branch utility has not been shown.

The same-seed-7 all-depths comparison did not reproduce that paired result.
At the fixed endpoint it reached 9/128 terminal accuracy versus 7/128 for
`deepest`; race-plus-fallback predictions were 9 versus 6 correct (exact
McNemar $p=0.607$). Layer-4 coverage was 22.7% versus 16.4%, and late-prefix
NLL was 5.74 versus 4.09. The all-depth race emitted 8 answers at mean
confidence 63.1%, with none correct. Thus the seed-6 classification advantage
is unconfirmed across seeds, while race overconfidence appears in both.
Seed-7 leave-one-head-out terminal accuracies were `[2.3, 7.8, 10.9, 7.8]%`
versus 7.0% fused; removing layer 1 reduced accuracy, while removing layers
2–4 slightly or substantially improved it. The contribution pattern differs
from seed 6, where removing layer 2 had the largest negative effect and layer
4 was neutral. There is no stable evidence yet that the deepest layer
contributes useful class evidence.

The supervision scheme remains appropriate for sequence-to-class recognition.
At an exogenous causal prefix $T$, let $p_T=P(Y\mid\mathcal F_T)$ be the
true posterior and $q_T$ the model output. The population risk decomposes as
$\mathbb E[-\log q_T(Y)]=\mathbb E[H(p_T)+D_{KL}(p_T\Vert q_T)]$,
so its optimum is the conditional class posterior; it does not require an
output at utterance onset. E83 samples two fixed-time prefixes plus EOS, then
applies the 0.6 race as a separate stopping policy. The loss is proper at its
sampled times, but two samples do not establish posterior quality at
event-triggered stopping times. The observed confidence/accuracy gaps
identify posterior estimation and calibration as current failure modes, not
a need to force a class event while the network is silent.

This support difference predicts how often a small minibatch cannot carry
deep evidence at all. If utterances independently activate the deepest layer
with probability $c$, a batch of size $B$ has
$N_{\rm active}\sim\mathrm{Binomial}(B,c)$ and
$P(N_{\rm active}=0)=(1-c)^B$. With $B=4$, the observed seed-6 deepest-only
coverage $6/128$ implies $P(N_{\rm active}=0)=0.825$; seed-7 coverage $21/128$
implies $0.488$; all-depth seed-6 coverage $79/128$ implies $0.0215$. This is
a support-only approximation, assuming training examples follow the held-out
coverage: an active event is necessary for hidden task credit, but it does
not guarantee a nonzero or useful gradient. It quantifies why batch size four
can make support collapse dominate stochastic updates.

The seed-7 deepest-only control reached 7/128 terminal accuracy (5.5%),
16.4% layer-4 coverage, and late-prefix NLL 4.09. At the fixed 0.6 threshold
it emitted four times with no correct answer. This second seed supports the
deepest-only failure diagnosis. Together with its matched all-depth arm, it
shows that the apparent fusion benefit varies across seeds.

The seed-6 deepest-only test of an earlier-layer event skip now falsifies the
claim that support survival alone will solve recognition. Sending layer-1
events directly to layers 3–4 raised final layer-4 coverage from 4.7% to
99.2%; the measured nesting-violation counter rose to 29, as expected when
the strict-chain condition is relaxed. Deep candidate scores increased only
1.8% (46,246 to 47,096 per utterance), while terminal accuracy moved from
7/128 to 9/128 and late-prefix NLL improved from 4.66 to 3.71, still worse
than uniform. The fixed-threshold race emitted five times and got none right.
On paired race-plus-fallback predictions, 6 examples favored strict-chain
and 8 favored skip (exact McNemar $p=0.791$). This isolates a real support
mechanism with a small sparse work increment, while showing that the newly
recruited paths have not yet acquired useful class evidence. Seed 7 repeated
the support effect without accuracy gain. The 1.8% work change is in candidate event–receiver scores, not
total inference energy: the current simulator still performs 288,008 vector
state updates per utterance on its 1 ms grid.

The seed-7 skip result reproduced the coverage change, reaching 100% layer-4
support, but terminal accuracy was 5/128, late-prefix NLL was 48.80, and
none of seven fixed-threshold emissions were correct. Candidate-score work
was 22.3% above its strict-chain seed-7 control. Paired race-plus-fallback
predictions were correct on 5 versus 6 examples (6 strict-only, 5 skip-only;
exact McNemar $p=1.0$). The two seeds support a causal effect on event
support, not class accuracy; seed-dependent event rates and poor evidence
calibration remain. Therefore support survival, activity impedance, and
label-useful credit must be diagnosed as distinct mechanisms.

The larger-evaluation command is a fresh training run, not a re-evaluation of
the original 32-example checkpoint. All historical E83 artifacts used one
NumPy generator for held-out subset selection and subsequent training
permutations and augmentations. Changing `eval_limit` therefore changed the
training random stream after initialization. The matched 128-example arms
share the same `eval_limit` and are valid within-protocol comparisons, but
their contrast with the earlier 32-example screen cannot be attributed to
evaluation size alone. The implementation now defaults to
`--rng_protocol split`, assigning evaluation selection, training subset/order,
augmentation, prefix sampling, and route-shadow sampling to separate streams;
`legacy_shared` remains available for historical reproduction. The split
protocol has not yet been validated by a paired run across evaluation limits.

The first row establishes a concrete cause of the pathwise depth-four failure:
on many examples the network has no realized deep event path for the label to
train. The second row prevents overclaiming: event absence is not the only
cause of chance accuracy. The pathwise prefix cross-entropy is proper at its
fixed causal query times, but properness does not supply derivatives for
missing support. The third row shows why merely lowering thresholds is not a
complete solution: the same small stack can move from early extinction to a
late activity cascade, and additive event logits then become badly scaled.

The depth-eight fused run was stopped after four of its eight requested
epochs. Its event counts swung from `[16, 9, 3, 6, 27, 51, 141, 250]` at epoch
1, to `[24, 3, 1, 1, 4, 6, 32, 65]` at epoch 2, to
`[59, 35, 97, 286, 1137, 1929, 4125, 4888]` at epoch 3, then back to
`[27, 3, 2, 2, 3, 11, 47, 58]` at epoch 4. Terminal accuracy was
12.5%, 3.1%, 9.4%, and 6.25%; late-prefix NLL was 39,814.7, 1,692.1,
19,787,863.2, and 22.9. This alternating extinction/cascade is the measured
instability; further epochs of the same setting were not useful evidence.

A paired spike audit used 128 held-out-speaker utterances (32 batches) from
the trained depth-four route-counterfactual checkpoint. It shadowed the
closest firing margin once per layer and batch, toggling the spike and
replaying the full stack. Candidate margins within $\pm0.25$ averaged 349 per
batch in layer 1, 67 in layer 2, 7.9 in layer 3, and 6.6 in layer 4. Median
counts were 347.5, 64.5, 1, and 0; no layer-3 margin was in-band in 8/32
batches, and no layer-4 margin was in-band in 24/32. Spike-on improved the
batch loss in 16/32, 17/32, 14/32, and 14/32 interventions. Mean batch
objective $L_1-L_0$ was $+0.00047$, $-0.00055$, $+0.00202$, and $+0.00260$,
with standard deviations $0.00842$, $0.00274$, $0.02249$, and $0.01027$.
These selected local interventions do not estimate the total boundary
gradient, but they reject the simple claim that a useful single spike
insertion consistently improves the current model's class loss. Deep support
is scarce, and where an event can be toggled its present payload is not
reliably class-useful.

There is also a concrete input-information bottleneck. `events()` groups
same-band spikes within 2 ms and returns $c_i=\log(1+n_i)$ as an event mark.
E83's `to_events` unpacks this count but forms the input only as the band
embedding $e_{b_i}$; $c_i$ is discarded before the first vector state.
Across the full fitting split, 55.6% of merged groups contained multiple raw
spikes (2.25-fold count compression); on held-out speakers, 43.5% did (1.79-
fold compression). The per-class held-out share ranged from 0.343 to 0.528.
The coarsening $(b,t,c)\mapsto(b,t)$ obeys
$I(Y;B,T)\le I(Y;B,T,C)$, with equality only if
$I(Y;C\mid B,T)=0$; that sufficiency condition has not been tested. This is
not proof that the count mark explains the accuracy gap, but it is input
information the current model cannot reconstruct after preprocessing.

The one-factor additive-count ablation has completed at the same D4,
128-train/32-held-out, seed-6 budget. It confirms a dynamical effect: layer-4
held-out support coverage was 0.625, 0.9375, 0.6562, and 0.5625 over the four
epochs, with zero support-nesting violations; conditional layer-4 events per
active utterance were 29.3, 80.5, 25.3, and 22.1. Yet terminal accuracy was
9.4%, then 0% for the remaining three epochs, and epoch-4 prefix NLL was
$[8.4786, 27.28]$. The count mark can keep a deeper path active, but this
single-seed run did not turn that activity into class evidence. This rules out
the dropped count as a sufficient explanation and argues for measuring
coverage and class utility separately.

The next comparisons separate the remaining causes. The matched deepest-only
versus `all_depths` readout controls and 128-example paired evaluations are
complete. Continue measuring active-example coverage
$c_\ell$ and conditional event multiplicity alongside mean spikes per
utterance; the exact nesting invariant predicts zero support violations for
the strict chain. If larger held-out checks do not validate the current
deepest-only signal, compare a sparse raw-event skip to the strict chain at
the same depth, seed, loss, and optimizer budget. Such a skip can break
absorbing silence without making the layer dense, but may let deep layers
shortcut the learned hierarchy, so include per-depth branch ablations and
event-work counts. Keep the 128-example spike audit as a measurement baseline;
do not add its noisy single-spike update until its utility is replicated.
Threshold calibration and boundary credit come after representation and
support, with a declared throughput target and fresh paired deltas. Count
restoration has shown that support can change without accuracy changing; rate
calibration must likewise be evaluated for class utility, not event count. If
neither input marks nor support paths help, move to topology or
speaker-invariant representation analysis instead of another
classification-loss sweep.

## 138. Deep event learning has a support-weighted gradient and a gate-space stability problem

Section 137 proves that strict-chain support is nested. The consequence for
learning is stronger than a count of zero-gradient batches: support changes
the mean and variance of the pathwise gradient itself. Fix a layer's
input-dependent pathwise parameter gradient on one example, and write it as

$$
G=A X,\qquad A=\mathbf 1[x\in\mathcal A_\ell],\qquad
c_\ell=\Pr(A=1),
$$

where $X$ is the gradient conditional on that layer having an event path.
Let $\mu_\ell=\mathbb E[X\mid A=1]$ and
$\Sigma_\ell=\operatorname{Cov}(X\mid A=1)$. Then, exactly,

$$
\mathbb E[G]=c_\ell\mu_\ell,\qquad
\operatorname{Cov}(G)=c_\ell\Sigma_\ell+
c_\ell(1-c_\ell)\mu_\ell\mu_\ell^\top.
$$

For a minibatch of $B$ independent examples,
$\Pr(M_\ell=0)=(1-c_\ell)^B$. In any task-relevant direction $u$,
the minibatch signal-to-noise ratio is

$$
\operatorname{SNR}_u=
\frac{\sqrt{B}\,c_\ell|u^\top\mu_\ell|}
{\sqrt{c_\ell u^\top\Sigma_\ell u+
c_\ell(1-c_\ell)(u^\top\mu_\ell)^2}}.
$$

When conditional gradient noise dominates, this scales approximately as
$\sqrt{B c_\ell}$; meanwhile the expected gradient magnitude scales as
$c_\ell$. This is a support penalty even when each active example has a
perfectly ordinary gradient. It does not assert that the conditional signal
$\mu_\ell$ points in a useful direction. If $u^\top\mu_\ell$ is near zero,
raising event coverage alone cannot fix label learning.

For the four equal-update depth-four arms, measured layer-4 coverage ranged
from 1.56% to 7.03%. If those rates represent training minibatches of size
four, the binomial approximation gives a 75–94% chance that a batch contains
no layer-4 example. This estimate assumes independent, representative
examples; the exact per-utterance support nesting does not require that
assumption. It explains why a small batch can repeatedly omit the deepest
pathwise signal, but not why accuracy is poor when a deep event does occur.

The omitted signal at a hard spike boundary can be written separately. For a
margin $g(\theta)$ and a logistic threshold perturbation of scale $\sigma$,
let $p=\operatorname{sigmoid}(g/\sigma)$ and let $L_1,L_0$ be downstream
losses after replaying the network with the event forced on or off,
including its refractory effect. The smoothed expected loss is
$\bar L=pL_1+(1-p)L_0$, so its derivative decomposes as

$$
\nabla_\theta\bar L=
p\nabla_\theta L_1+(1-p)\nabla_\theta L_0+
\frac{p(1-p)}{\sigma}(L_1-L_0)\nabla_\theta g.
$$

The first two terms are pathwise credit inside the two fixed event outcomes;
the last is the spike birth/death boundary credit. E83's ordinary gradient
does not contain that last term because the hard event identity is detached.
A replay shadow can estimate its loss difference, but inserting this term
would train the explicitly randomized/smoothed gate objective. It is not the
ordinary derivative of the deterministic hard-gate loss, which is zero almost
everywhere and discontinuous at the boundary. Thus the theory identifies the
missing credit and the objective it would optimize; it does not yet justify
the estimator's variance or its gain on SHD.

A further mechanism can make support change abruptly even when global gradient
clipping is enabled. AdamW takes a coordinatewise preconditioned step

$$
\Delta\theta_t=-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta\lambda\theta_t.
$$

Clipping the raw gradient norm does not generally bound
$\|\Delta\theta_t\|$ or a gate's margin change in this optimizer geometry.
On an isolated first step, multiplying every gradient by a clipping factor
$\alpha$ also multiplies $\hat m$ by $\alpha$ and $\hat v$ by $\alpha^2$;
when $\epsilon$ is small, the Adam ratio is nearly unchanged. This is a
direct algebraic reason that a raw-gradient clip is not a trust region for
hard events.

Within a fixed event pattern, take a firing margin $g_i(\theta)$ with
Hessian spectral norm bounded locally by $H_i$. A step cannot change that
gate's sign if

$$
|g_i(\theta)|>
|\nabla g_i(\theta)^\top\Delta\theta|
+\tfrac12 H_i\|\Delta\theta\|_2^2.
$$

The ratio of the right side to $|g_i|$ is a gate-space step-size diagnostic:
values below one certify local sign stability under the stated curvature
bound; values above one identify margins that may flip. E83's alternating
extinction and activity cascades make optimizer-induced gate changes a
plausible explanation, but epoch-level event counts do not establish that
cause. The required evidence is a per-update record of pre/post margin
histograms, actual AdamW displacement, predicted margin displacement
$\nabla g_i^\top\Delta\theta$, and observed gate flips. No such trace has yet
been collected.

This yields four separate necessary checks for deeper supervised event
models: (1) examples survive to the layer that should learn; (2) active
messages carry label-useful information; (3) missing useful spikes receive a
measured boundary signal; and (4) optimizer steps preserve useful gates long
enough for that signal to accumulate. They are not interchangeable. The
layer-1 skip raises layer-4 support without accuracy, the spike audit finds
mixed utility among selected candidates, and data-diversity effects reverse
direction across the two equal-update seeds. These results locate distinct
open conditions instead of selecting one universal cause.

### Compute-matched data diversity screen

The two-arm comparison at each seed held optimizer updates to 120, architecture
and initialization fixed, and used a nested 120-example versus 480-example
training subset; both arms evaluated on the same 128 examples within each
seed. In seed 6, terminal accuracy was 8/128 for 120 examples over four epochs
and 20/128 for 480 examples over one epoch (paired exact McNemar $p=0.0227$).
In seed 7, the direction reversed: 20/128 versus 8/128
($p=0.0357$). Each arm pair therefore changes example diversity while keeping
updates fixed, but the sign reversal means these two small runs do not
establish a repeatable data-diversity benefit. Across all four arms,
layer-4 support was only 1.56–7.03% and late-prefix NLL was 3.02–6.58, above
the 20-class uniform value $\log 20\approx2.996$. The screen points to strong
initialization/optimization sensitivity and leaves the gate-space mechanism
unresolved; it is not a reason to launch more seed-only repetitions.
## 139. Route reachability is not route credit: the missing comparison may be a pair

For sparse routing, distinguish four graphs: (1) the **candidate graph** in
the fixed connectivity mask; (2) the **event-conditioned graph** whose edges
are scored for a realized source event and payload; (3) the **realized graph**
whose content routes open and whose receiver units fire; and (4) the **credit
graph** whose alternatives are actually evaluated by the supervised loss.
Only the first is captured by a static adjacency mask. A path in that graph
is necessary for learning through that path, but does not imply an event,
message, spike, or gradient occurred there.

For E83's D4 settings, each of the three later 16-by-16 masks is sampled at
fan probability 0.25 and repaired if any row or column is empty. Recreating
the exact masks from the two recorded seeds gives edge counts \([54,61,74]\)
for seed 6 and \([64,70,73]\) for seed 7. A static path exists between 90.2%
of first-layer/fourth-layer unit pairs in seed 6 and 98.0% in seed 7. After
including the actual local first-layer frequency mask, all 140 input bands
have a candidate path to layer 4 in both seeds; the median band reaches 16 of
16 layer-4 units. These are exact reachability counts for these two
initializations, not proof that any path is semantically useful.

The matched D4 runs nevertheless have just 1.56–7.03% realized layer-4
support. The bottleneck therefore occurs after static path construction:
source events disappear, content gates close, or arriving vector messages
fail to create a threshold crossing. This rules out a broad static
disconnection in these two masks as the explanation for near-silent depth.
It does not distinguish dynamic route selection from insufficient membrane
drive or unhelpful payloads.

The MoE analogy identifies an additional learning condition. Let two competing
routes have scores \(s_a,s_b\) and relaxed choice probability
\(q=\operatorname{sigmoid}((s_a-s_b)/\tau)\). If their full downstream losses
under matched input/context are \(L_a,L_b\), then

$$
\frac{\partial\mathbb E[L]}{\partial(s_a-s_b)}
=\frac{q(1-q)}{\tau}(L_a-L_b).
$$

When \(L_a<L_b\), gradient descent raises the relative score for route \(a\).
This gradient needs both route outcomes: if the losing route is never executed
or shadowed, \(L_b\) is unknown and its comparative utility cannot be learned.
Training does not require both routes on every example; it does require a
sparse sample of paired route evaluations with the same example, prefix
times, and downstream randomness. This is the same requirement that appears
in sparse mixture-of-experts routing.

E83's current shadow toggles one masked message route open/closed and replays
the downstream network, but only when the event-conditioned score satisfies
\(|r|\leq\texttt{cf\_band}\). Far-closed route instances have zero direct
shadow probability under that estimator; a candidate edge outside the fixed
mask has no proposal at all. Shared receiver vectors can move some scores
indirectly, but do not guarantee that every suppressed route is discovered.
Hence static path existence alone is not enough: every potentially useful
route, or a structured proposal for it, must have nonzero counterfactual
sampling probability.

The single-edge comparison also misses synergistic routes. E83's receiver
adds arriving vector contributions before applying a hard firing threshold.
Under a fixed event ordering, let two incoming candidate routes contribute
margin increments \(h_a,h_b\) to baseline margin \(g_0\). It can happen that

$$
g_0<0,\quad g_0+h_a<0,\quad g_0+h_b<0,\quad
g_0+h_a+h_b\geq0.
$$

Each route alone leaves the receiver silent; together they create a spike.
The one-route loss deltas are then zero even though the pair changes the
downstream class evidence.

This complementarity is exact in a two-gate relaxation. Let independent
candidate gates open with probabilities \(q_a=\operatorname{sigmoid}(s_a/\tau)\)
and \(q_b=\operatorname{sigmoid}(s_b/\tau)\); let \(L_{ij}\) be the complete
replay loss when gate states are \(i,j\in\{0,1\}\). Then

$$
\frac{\partial\mathbb E[L]}{\partial s_a}=
\frac{q_a(1-q_a)}{\tau}
\left[(1-q_b)(L_{10}-L_{00})+q_b(L_{11}-L_{01})\right].
$$

If neither singleton helps but the pair does, so \(L_{10}=L_{01}=L_{00}\) and
\(L_{11}<L_{00}\), gate \(a\) receives credit proportional to \(q_b\). If
route \(b\) is never present in a rollout or forced shadow, \(q_b=0\) and the
gradient to \(a\) is exactly zero; the same holds symmetrically. The missing
object is the four-way comparison \(L_{00},L_{10},L_{01},L_{11}\), not
another scalar update to a route that has never jointly participated.

The principled sparse option is to preserve proposal support over plausible
masked routes and add a small number of cooperative shadows targeted at a
common receiver: sample pairs whose arrivals overlap in its temporal window
and whose combined potential approaches its firing margin. Replay the four
gate states 00, 10, 01, and 11 on the same utterance/prefix, estimate each
single-route utility and the interaction
\(\Gamma_{ab}=L_{11}-L_{10}-L_{01}+L_{00}\), then shrink uncertain estimates
and limit the change in gate and firing margins. A negative \(\Gamma_{ab}\)
means the pair reduces loss beyond its isolated effects. Pair selection must
stay sparse; enumerating every edge pair would replace the architecture's
advantage with dense work.

The next informative audit is a fixed-checkpoint route-reachability and
cooperation test, not another seed sweep: measure all four graph levels, and
test a predeclared sparse set of singleton/pair shadows near receiver
thresholds. Record pair-created spikes, class-loss deltas, route proposal
probabilities, and shadow cost. This would tell us whether the unresolved
failure is absent source events, far-closed routes, cooperative threshold
crossing, or payload/class utility. The present evidence establishes broad
candidate-graph reachability and poor realized support; it does not yet
establish that route-pair synergy is the cause.

## 140. More depth creates combinatorial choices, but path survival multiplies

Depth is useful because it composes choices: a width-\(M\) hidden stack with
per-edge candidate density \(p\) has, under independent random masks, an
expected

$$
\mathbb E[P_{u\to v}^{(D)}]=M^{D-2}p^{D-1}
$$

static routes between a fixed first-layer unit \(u\) and final-layer unit \(v\)
over \(D-1\) transitions. For the current \(D=4\), \(M=16\), \(p=0.25\)
setting, this expectation is 4; adding layers makes the number of candidate
compositions grow rapidly. Each extra layer can therefore create more
possible expert compositions for the same input.

But more **candidate** compositions are not more realized or trainable
choices. Let \(s_\ell=\Pr(x\in\mathcal A_{\ell+1}\mid x\in\mathcal A_\ell)\)
be the conditional example-survival probability. Under the strict chain,

$$
c_D=c_1\prod_{\ell=1}^{D-1}s_\ell.
$$

If survival were constant at \(s\), then \(c_D=c_1s^{D-1}\). To preserve at
least half the examples at layer 8 from full layer-1 support requires
\(s\geq0.5^{1/7}\approx0.906\) at every transition; at depth 16 the
requirement is \(s\geq0.5^{1/15}\approx0.955\). This is a simple survival
identity, not an assertion that examples are independent across layers.
The gradient to a deepest-only loss is subject to the same support product.
All-depth supervision shortens that path for each head, but does not restore
the final hidden layer's support.

Thus depth should follow, not precede, route acquisition and survival. Simply
stacking more layers raises the number of possible expert sequences while
making any specific sequence less likely to be active and credited. More
width adds local alternatives but also raises candidate fan-out and possible
threshold bursts. Sparse skips can protect support but earlier E83 tests
showed that restoring support alone did not teach the class. A trainable deep
design needs (a) broad but sparse candidate reachability, (b) paired or
cooperative counterfactual evaluation so losing choices receive label
utility, and (c) a calibrated survival/threshold mechanism. Then increase
depth while monitoring per-layer conditional survival and route-credit
coverage. This predicts a path to deep models; it is not yet a proof that the
current E83 cell scales.

## 141. A non-event should carry its failure margin and a counterfactual loss

A useful training signal can originate at a candidate event that did not
occur. But “did not fire” is not one state with one correction: an existing
source may have had its edge closed, the receiver may have stayed below its
threshold, the message may have lost a timing race, or the receiver may have
been refractory. These decisions have different local controls. A zero or
negative label sent backward without identifying the cause could push the
wrong control and make event rates unstable.

Represent a candidate event $e$ by its source, target, proposed arrival time
$t_e$, payload $v_e$, and local decision margin $m_e$. Choose the sign so the
candidate is admitted when $m_e>0$. Examples are

$$
m_e=r_e \quad\text{(content route)},\qquad
m_e=V_e-\theta_j \quad\text{(receiver threshold)},
$$

$$
m_e=t_{\mathrm{competitor}}-t_e \quad\text{(earliest-arrival race)},\qquad
m_e=t_e-(t_{\mathrm{last},j}+R_j) \quad\text{(refractory availability)}.
$$

For a fixed factual trace, let $L_{e,0}$ be the causal prefix loss when the
candidate stays absent, and $L_{e,1}$ the loss when it is admitted and the
network is replayed through all downstream state changes: later events,
refractory/reset state, route winners, and readout. Define its contextual
utility as

$$
U_e=L_{e,0}-L_{e,1}.
$$

$U_e>0$ means this specific event would have helped on this example and
prefix. It does not mean every event of its class is useful. If a smooth
Bernoulli relaxation $z_e\sim\mathrm{Bernoulli}(p_e)$ is used to derive an
update, with $p_e=\operatorname{sigmoid}(m_e/\tau)$, then

$$
\nabla_\theta \mathbb E[L_e]
=-\frac{p_e(1-p_e)}{\tau}U_e\nabla_\theta m_e.
$$

Loss descent therefore moves the margin toward admitting a helpful
candidate and away from admitting a harmful one. The deployed computation
can remain a hard event decision; the shadow estimates a boundary update for
the smoothed decision rule, not the almost-everywhere derivative of the
deterministic threshold. For an actual race, the alternate replay must change
the winning event consistently, including cancellation of the former
winner, rather than append a second event to a trace that allows only one
winner.

This gives a precise interpretation of propagating “did not fire” events:
carry a sparse record of the dormant candidate and its cause/margin, obtain a
paired downstream loss by forcing the relevant alternative, then send the
signed utility through the local eligibility $\nabla m_e$. The
counterfactual event need not be inserted into the factual state. If it
would create a downstream event that also did not occur, replay or shadow
that changed continuation as part of the same intervention; otherwise
$U_e$ omits the event's actual consequence. Cooperative proposals still
need joint shadows as in §139.

This rule has a support condition. A margin update can teach only controls
with nonzero sensitivity to $m_e$, and the counterfactual can evaluate only
events generated from an available source and a legal candidate edge. A
candidate edge outside the fixed mask needs a separate sparse structural
proposal. If deterministic routing never samples or shadows a dormant
candidate, its utility is unobserved; a small exploration probability over
eligible candidates prevents its proposal probability becoming exactly
zero. Sampling probabilities must be recorded. Reweighting every rare
shadow to estimate the full candidate sum is unbiased in principle but can
have the high variance observed in §131; bounded, normalized, stratified
utility updates are a safer first test, with their bias reported.

The sparsity constraint is substantive: enumerate proposals only from
realized source events and sparse candidate neighbors, then spend a bounded
shadow budget on near-margin cases plus a small exploration sample. Do not
form all absent source-target-time combinations. For each sampled shadow,
record cause, margin, sampling probability, paired loss delta, whether a new
spike/race winner appeared, downstream event count, and replay work. Keep
separate statistics for route closure, threshold failure, race loss, and
refractory blocking so a useful threshold correction is not mistaken for a
useful routing correction.

This makes the benefit of a richer pool conditional, not automatic. Suppose
there are $N$ eligible dormant proposals on a fixed example and $K$ have
positive, task-useful utility, so $\rho=K/N$. If $m$ proposals are sampled
uniformly without replacement, the probability of observing at least one
useful alternative is exactly

$$
1-\frac{\binom{N-K}{m}}{\binom{N}{m}}
\;\approx\;1-(1-\rho)^m.
$$

At fixed shadow budget $m$, adding mostly unhelpful candidates can lower
$\rho$ and reduce discovery probability. If instead samples are drawn from a
useful margin/cause stratum, repeated paired utility estimates with
conditional variance $\sigma_U^2$ have standard error $\sigma_U/\sqrt m$
under independent sampling. Thus the pool helps through *coverage* and
utility SNR, not raw cardinality. Stratified sampling improves coverage but
changes the target distribution; record propensities and either report
stratum-conditional utility or use bounded importance weights. Candidate
pair discovery is stricter: a singleton pool does not reveal a synergistic
pair, so overlapping proposals near one receiver must be explicitly paired
and sampled.

**What this advances and what remains open.** Sections 19, 57, 131, 138,
and 139 establish why hard decisions need paired counterfactual outcomes,
why support and event-boundary credit are distinct, and why joint
alternatives can matter. The new unification is to index each dormant
proposal by its causal failure margin and differentiate that margin using
end-to-end paired utility. No E83 run has yet measured this four-cause
candidate record or shown that its updates improve class accuracy. The next
informative test is a fixed-checkpoint audit: sample non-events by cause and
margin, replay singletons and a small set of threshold-overlap pairs, and
report utility, variance, event-rate effects, and sparse replay cost before
enabling updates.

## 142. Counterfactual-rich topology needs route alternatives, not dense firing

This is the same core problem as routing in sparse differentiable computation
graphs, including sparse expert models: the forward pass selects only a small
set of branches, while the learner needs comparative utility for plausible
alternatives. The event-network-specific questions are how message time,
vector payload, receiver integration, threshold firing, race arbitration, and
refractory state change that utility. Merely naming a route counterfactual is
not a new routing principle.

The previous section treats one dormant proposal. In a deep stack, a useful
event may require several gates in series, or a small bundle of messages in
parallel. A topology that stores only the realized event trace loses both
kinds of alternative. The design target is therefore a sparse *proposal
topology*: it keeps a bounded set of legal route alternatives and enough
local state to explain why each was rejected, while the factual forward pass
still emits only events that won the actual decisions.

For a source event $i$ and candidate receiver $j$, store a sparse proposal

$$
e=(i,j,t_e,v_e, m_e^{route},m_e^{fire},m_e^{race},m_e^{ref},\pi_e),
$$

where $t_e$ and $v_e$ are its proposed arrival time and payload; the four
margins are conditional on the preceding decisions being opened; and
$\pi_e$ is its counterfactual sampling probability. The proposal index is
the sparse candidate graph, not a dense source-by-target matrix. The factual
event is produced only if the route is legal/admitted, the receiver can fire,
the refractory state permits it, and the relevant race is won. These causes
are not mutually exclusive: opening a route may still leave its receiver
subthreshold, so causal utility must be measured by interventions, not
assigned from a single categorical “failure label.”

This topology admits four counterfactual sizes, in increasing cost:

1. **Single proposal:** open/close one existing edge, or add/drop one
   contribution at a receiver. This estimates the conditional utility of a
   route in the current context.
2. **Receiver bundle:** replay a small pair or tuple of temporally
  overlapping messages together. This measures threshold cooperation such
  as §139's $L_{00},L_{10},L_{01},L_{11}$ interaction.
3. **Short path:** start from a dormant near-threshold source or route, open
   a bounded sequence of gates through depth, then replay its entire
   downstream continuation. This tests a useful path whose individual
   gates cannot get a signal because their partners are absent.
4. **Structural proposal:** test a sparse edge outside the current mask.
   Promote it to the trainable candidate graph only after repeated paired
  evidence supports it; otherwise the current mask makes its proposal
  probability exactly zero.

For two currently closed gates $a,b$, let $q_a,q_b$ be their independent
smoothed opening probabilities and let $L_{00},L_{10},L_{01},L_{11}$ be the
four losses after replaying the same input with neither, either singleton,
or both routes open. Their expected loss is

$$
\bar L=(1-q_a)(1-q_b)L_{00}+q_a(1-q_b)L_{10}
 +(1-q_a)q_bL_{01}+q_aq_bL_{11}.
$$

If $q_i=\operatorname{sigmoid}(s_i/\tau)$, the exact relaxed gradient is

$$
\frac{\partial\bar L}{\partial s_a}=\frac{q_a(1-q_a)}{\tau}
 [(1-q_b)(L_{10}-L_{00})+q_b(L_{11}-L_{01})],
$$

with the symmetric expression for $b$. The interaction
$\Gamma=L_{11}-L_{10}-L_{01}+L_{00}$ is negative when joint opening gives a
super-additive reduction in loss. Four matched outcomes distinguish that
cooperation from two helpful singleton routes. They also show a limit: the
exact independent-gate gradient still weights a route's conditional paired
utility by the partner's opening probability. Pair shadows reveal the
conditional utility; escaping a very small partner probability requires a
separately justified exploration distribution or a different joint gate,
not a claim that replay alone removes the attenuation.

The E83 pilot implements one bounded instance: it pairs two near-boundary
closed routes that target the same receiver and have source times within a
fixed window, then replays the three non-factual outcomes. At most one pair is
sampled per minibatch; the pair sample uses an independent random stream, so
the matched pathwise control has the same data order and initialization.
This specifically tests receiver-level cooperation. It does not test
cross-layer short paths, out-of-mask topology growth, or threshold/race/
refractory alternatives, and three replays per pair add training work while
leaving inference routing unchanged.

The proposal rule in this pilot is a screening heuristic, and is now stated
precisely so it is not confused with the derivation: retain closed route
scores $s\in[-0.5,0)$; group by batch and receiver; sort each group by source
time; keep only adjacent pairs with a gap at most 25 ms; then sample at most
one candidate pair uniformly per minibatch. If a minibatch has $N$ such
candidates and samples $m=\min(1,N)$, each candidate's inclusion probability
is $m/N$. Uniform sampling makes the selected mean an estimator for this
filtered stratum, but the positive near-boundary/receiver/time filters give
zero support to other alternatives. The current JSON records total eligible
and selected counts but not each minibatch's inclusion propensity, so it
cannot recover an importance-corrected utility over a broader route
population. The rule is not chosen from the receiver's threshold margin or
the vector messages' projected effects. The gate-gradient contrasts are also
clipped to $[-5,5]$, averaged over selected pairs, globally norm-clipped to
1, and applied as a separate SGD step after AdamW. Therefore the four-loss
derivative is exact before those stated sampling/clipping/update choices;
the complete optimizer is a deliberately bounded local-credit heuristic.

The follow-up layer-balanced sampler keeps the one-pair budget but first
chooses uniformly among layers with at least one eligible pair, then chooses
uniformly among that layer's eligible adjacent pairs. A selected pair in a
layer with $N_\ell$ proposals and $K$ eligible layers has inclusion
probability $1/(K N_\ell)$; the implementation logs the eligible counts by
layer and these propensities. This deliberately changes the target from the
global pair-population mean to equal opportunity for each eligible depth.
The comparison with the global sampler therefore tests a credit-allocation
policy, not an unbiased estimate of one shared population gradient.

A mechanism-based proposal can use the receiver's actual threshold geometry.
Before an intervening spike/reset, let its factual state be $z_j^0(T)$ with
refractory trace $R_j(T)$, and define the gap to firing as

$$
g_j(T)=\theta_j(1+R_j(T))-\operatorname{Re}\langle w_j,z_j^0(T)\rangle.
$$

For a closed route from source vector $v_i$ with proposed arrival $a_i$,
its linearized drive at $T$ is

$$
\phi_i(T)=\mathbf1[T\ge a_i]\operatorname{Re}\langle
w_j,e^{\lambda_j(T-a_i)}B_jv_i\rangle.
$$

A candidate pair is a plausible threshold-cooperation proposal when each
singleton remains below the positive gap but
$\phi_i(T)+\phi_k(T)\ge g_j(T)$ for some nearby $T$. This score incorporates
relative delays, decay/rotation, and the signed vector projections that
actually drive the receiver. It is only a proposal approximation: another
event or a counterfactual reset can change $z_j,R_j$, so exact joint replay
still determines utility. A principled next sampler should reserve a
nonzero uniform exploration share, allocate the rest to these predicted
threshold crossings, log the resulting inclusion probabilities, and
stratify by layer/cause. That tests whether the route is locally able to
fire before asking whether its emitted payload improves the class loss.

The general within-layer object is a **conditional marked-message policy**.
Let $x_i=(t_i,v_i)$ be a source event, $h_j(t^-)$ the receiver's local state
just before a candidate arrival (decaying vector state, refractory state, and
pending-event summary), and let $r_i$ denote an action. An action may suppress
the message, choose a receiver, or choose a delay/value mode. The policy is
$\pi_\theta(r_i\mid x_i,h_j)$ and emits a marked event

$$
m_i(r_i;h_j)=\big(a_i,u_i\big),\qquad
a_i=t_i+\delta_\theta(x_i,h_j,r_i),\quad
u_i=V_\theta(v_i,h_j,r_i).
$$

For E74/E83's present edge, $r_i$ is simply admit/suppress,
$u_i=B_jv_i$, and $\delta_\theta=\tau_j[s_{ij}]_+$. The inference policy is
currently the hard threshold; the logistic $\pi$ is a local training
relaxation, not sampled stochastic inference. Since a source may fan out,
the default choice is a vector of per-edge gates, allowing several messages
to be active in parallel. A categorical target choice is appropriate only
when the hardware or model explicitly imposes one-route capacity.

The present E74/E83 gate is an especially restricted case: its score is
$s_{ij}=q_j^\top v_i+c_{x_i,j}$, where $q_j$ is a learned but
input-independent receiver vector. Thus it is a content-matched sparse edge
filter, not yet a state-conditioned attention query; the score does not read
$h_j(t^-)$ when deciding whether or when to admit the message. The conditional
policy above is the broader formalism. A state-conditioned version could use
$q_j(t^-)=Q_\theta(h_j(t^-))$ and score
$s_{ij}=q_j(t^-)^\top K_\theta(v_i)$ only over the receiver's sparse legal
candidate list. That retains event-driven candidate work while allowing
context-dependent retrieval, but requires route decisions to be processed in
time order against the receiver state they actually change.

For two proposed messages arriving at $a_1,a_2$, their contribution to a
linear pre-fire state at time $T$ is

$$
\Delta z_j(T)=
\mathbf1[T\ge a_1]e^{\lambda_j(T-a_1)}B_jv_1+
\mathbf1[T\ge a_2]e^{\lambda_j(T-a_2)}B_jv_2,
\quad
\Delta V_j(T)=\operatorname{Re}\langle w_j,\Delta z_j(T)\rangle.
$$

Thus the counterfactual is not “the same signal with an arbitrary second
delay”: each candidate has its own conditional $(a_i,u_i)$, and the receiver
integrates both at their own times. The linear state contribution is
additive, but threshold crossing time, reset/refractory state, emitted
payload, and all downstream routes are nonlinear functions of the pair.
When action 2 is chosen after action 1 changes the receiver state, its policy
must condition on that replayed state; an independent-gate product is then
only a declared approximation. A faithful shadow replays the ordered pair
through the state transition and measures the resulting full loss.

An expanded experiment can factor each action into edge admission, a small
delay alternative $\delta$, and a payload mode $\eta$, with
$m_i=(t_i+\delta_i,V_{\eta_i}(v_i,h_j))$. It should first stratify
counterfactual pairs by relative arrival $a_2-a_1$ and vector alignment after
the receiver's decay/rotation, then evaluate matched singleton and joint
replays. The current SHD pilot does neither delay-mode nor payload-mode
interventions: for its near-closed routes, $[s]_+=0$, so forced-open messages
arrive at their source times with the existing payload transform. This
distinction prevents the present gate-pair result from being overread as
evidence about learned timing or vector alternatives.

There is also a concrete parameter coupling to audit in the current layer:
the same score $s_{ij}$ decides whether the route exists and sets its delay
$\delta_{ij}=\tau_j[s_{ij}]_+$. On the open side, before delay saturation,
$\partial a_{ij}/\partial s_{ij}=\tau_j$; on the closed side the message is
absent and this delay derivative is zero. A route-credit update that raises
$s_{ij}$ to make a useful edge more likely therefore also retimes its message
later. The current shadow measures the finite presence contrast at fixed
parameters; its local logistic derivative does not measure a separate
counterfactual delay policy. This can make the learned gate gradient fight
the temporal objective, especially when the receiver is sensitive to a narrow
arrival window.

A clean architecture ablation would factor admission and timing into separate
scores, $s^g_{ij}$ and $d_{ij}$:

$$
z_{ij}=\mathbf1[s^g_{ij}>0],\qquad
a_{ij}=t_i+\operatorname{softplus}(d_{ij}),\qquad
u_{ij}=B_{ij}v_i.
$$

Only admitted messages instantiate or execute their payload path, so this
does not require dense firing. Gate shadows can then estimate route utility,
while ordinary active-path gradients train delay and value maps; paired
delay/value alternatives can separately test timing or representational
interactions. Whether this factorization improves SHD is an open experiment,
not a presumed fix.

The path case exposes an additional credit bottleneck. Suppose a candidate
path needs $K$ independent gates $z_k\sim\mathrm{Bernoulli}(p_k)$, and for
this local derivation its complete activation changes the loss from $L_0$
to $L_1$. With $p_k=\operatorname{sigmoid}(m_k/\tau_k)$,

$$
\mathbb E[L]=L_0+\left(\prod_{k=1}^{K}p_k\right)(L_1-L_0),
$$

and therefore

$$
\frac{\partial\mathbb E[L]}{\partial m_k}
=\frac{p_k(1-p_k)}{\tau_k}
  \left(\prod_{q\ne k}p_q\right)(L_1-L_0).
$$

If the whole path is useful ($L_1<L_0$) but its other gates are almost
never open, the isolated gradient to gate $k$ is almost zero. For equal
opening probability $p$, the path utility is attenuated by $p^{K-1}$ in each
gate's gradient. This is a direct mathematical reason that merely adding
depth or more independent route choices need not make a useful path
discoverable. A sampled **joint path shadow** can reveal $L_1-L_0$ even when
ordinary executions never realize the path; its local gate updates then
need a credit-allocation rule. The simplest bounded option is to use each
gate's conditional marginal while the other path gates are forced open. If
strong interactions make that order-dependent, average marginal utilities
over a small sample of opening orders (a sampled Shapley allocation) or
retain only an explicit small bundle gate. This is a proposal for testing,
not a proven optimizer.

The topology should make these audits naturally sparse:

- Each active source event visits its bounded outgoing candidate list and
  records the best few rejected proposals plus a small exploration sample;
  it does not score every possible target.
- A receiver keeps the local pre-threshold potential, threshold margin,
  earliest arrival and runner-up gap, refractory-availability margin, and
  compact IDs for the contributions that formed those values. These are
  sufficient to propose targeted singleton and overlap shadows.
- A bounded shadow queue stores candidate payload/time and causal ancestry,
  then replays only selected branches through downstream state. Counterfactual
  state stays separate from the factual state; at inference it is not
  propagated or paid for.
- A small reservoir of absent structural edges supports topology growth.
  It is sampled by sparse co-activity/arrival compatibility, then admitted
  only if repeated held-out paired utility justifies the added degree and
  work. Fixed-mask edges with no exploration can never be discovered by
  gradient descent alone.
- Credit is local to the margin that controls the intervention, but utility
  is measured end-to-end. Keep cause-specific utility statistics and cap
  changes in the optimizer metric and in predicted firing/race margins.

This creates a hierarchy of actual and counterfactual routes without making
the inference graph dense: realized event paths remain sparse; dormant
alternatives are metadata plus a bounded training-time shadow budget. The
cost of training is $O(E_{active}+S\,C_{shadow})$ for $S$ selected shadows,
not all possible edges or edge pairs. A short path shadow can still be
expensive because it replays downstream state, so report both $S$ and
measured replay work.

**Systematic validation order.** First, at a fixed checkpoint, inventory
proposal counts and failure margins by layer/cause, including proposals
created only by opening a prior dormant path. Second, compare factual loss
with paired singleton, receiver-bundle, and short-path replays on the same
utterance and prefix, using common random numbers for any stochastic parts.
Third, estimate utility sign, variance, coverage, pair/path interaction,
event-rate changes, and shadow work under a fixed budget. Fourth, enable only
the best-supported local margin updates with a trust region; compare against
pathwise-only learning on the same initialization and examples. Finally,
grow the candidate topology only where held-out counterfactual utility
repeats. This is how richer counterfactuals can improve route learning
without replacing sparse event computation with dense synchronous activity.

**Known versus open.** The product attenuation above is exact for the stated
independent-gate, single-path relaxation; it proves why a useful deep route
can be invisible to ordinary gate gradients. Existing experiments establish
that static E83 paths are plentiful while realized deep support is low, and
that single-edge shadows do not test joint threshold crossings. They do not
yet show that paired or path shadows have useful SHD utility, that local
credit allocation trains those routes, or that the additional training-time
work is offset by inference savings. Those are the next falsifiable claims.

### A route's decision-boundary counterpart

For a hard router choosing $a=\arg\max_k s_k(\theta)$, compare the winner
with each alternative $b$, not just with one generic “off” state. The pairwise
margin is $m_{ab}=s_a-s_b>0$. A counterfactual flip is a perturbation
$\delta\theta$ such that $m_{ab}(\theta+\delta\theta)\leq0$. Linearizing
with $g_{ab}=\nabla_\theta m_{ab}$, the smallest flip under a
positive-definite optimizer metric $H$ is

$$
\delta\theta^*=-\frac{m_{ab}}{g_{ab}^\top H^{-1}g_{ab}}H^{-1}g_{ab},
\qquad
d_{ab}^2=\frac{m_{ab}^2}{g_{ab}^\top H^{-1}g_{ab}}.
$$

$d_{ab}$ measures local accessibility of the alternative; it does not say
whether that alternative is useful. For that, replay both choices on the
same input and compare $U_{b|a}=L(a)-L(b)$. Route exploration should measure
boundary distance, utility, and utility uncertainty separately. A close
alternative may be useless or harmful, and a far alternative may be useful
but invisible to a policy that samples only near boundaries. For event
networks, the same contrast can be a firing-threshold margin, race winner
gap, refractory-availability gap, or joint path boundary.

“Dual” is appropriate as a conceptual name for this choice/alternative
pair, but it is not automatically a formal convex dual. In the constrained
minimum-change problem above, a Lagrange multiplier is the strict
optimization-theory dual variable. If multiple gates, delays, and receiver
conditions must change together, the counterfactual is an intervention set
or alternate path, not one scalar opposite route.

### A route has several independent axes of alternatives

A richer route family should factorize what can change:

| axis | actual choice | counterfactual alternative |
|---|---|---|
| topology | which legal sender–receiver edge exists | a sparse dormant edge is opened or grown |
| gate | whether a message is admitted | toggle its content score across the gate margin |
| payload | which vector transform/value is sent | alternate value map, sign, gain, or subspace |
| time | message delay/arrival | an alternate delay or timing window |
| integration | which contributions jointly charge a receiver | add/drop a temporally overlapping bundle |
| firing | whether integrated state crosses threshold | counterfactually create/delete the receiver event |
| arbitration | which event wins a race | replace the winner with a specific competitor |
| state | whether refractory/reset permits the event | replay with the matched availability intervention |
| continuation | which downstream path follows | shadow a short multi-layer path or path bundle |
| readout | whether/when a class emits | compare class alternatives and stopping times |

These are not mutually exclusive choices. A single candidate can lose its
edge, be too late, arrive during refractory state, and remain insufficient
to cross threshold. So store a *margin vector* and define alternatives by
interventions on one coordinate or a small set of coordinates; let the
downstream replay reveal whether changing that coordinate was sufficient.
When the mechanism is nonseparable, test interactions explicitly using
paired bundles or short paths. The number of possible configurations is
combinatorial, but the proposal graph and shadow sample can remain sparse.

For a multiway router with chosen route $a$ and alternatives $b_1,\dots,b_K$,
the correct counterfactual object is the set of matched route utilities
$U_{b_j|a}$, not a binary “wrong” flag. When routes are sets (top-$k$ keys,
multiple arriving messages), compare selected set replacements and additions;
when payloads and delays are continuous, probe local perturbations as well as
discrete alternatives. Attention gives a useful instance: competing keys
change retrieval content, while key/value representation, delay, and
whether two retrieved values combine are separate axes. The mechanism must
preserve matched compute when comparing alternatives, or extra active work
will be confounded with better routing.

### Literature boundary and novelty

Counterfactual route comparison itself is not new. A 2026 MoE study samples
equal-compute expert alternatives, measures token-level loss utility, and
reports that a router-only update can help on difficult tokens
([Yoon et al., arXiv:2605.07260](https://arxiv.org/abs/2605.07260)). A separate
2026 preprint uses counterfactual expert impact to alter inference-time
routing ([Hu et al., arXiv:2604.14246](https://arxiv.org/abs/2604.14246)).
HNCA derives lower-variance credit for discrete stochastic units
([Young, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20874));
EventProp derives exact event-based gradients for particular spiking
dynamics ([Wunderlich & Pehle, 2021](https://arxiv.org/abs/2009.08378)).
These works mean we should not claim generic novelty for “try an alternate
route and measure the loss.” The potentially distinct contribution is a
cause-factorized, temporally structured counterfactual topology for sparse
vector messages—covering edge admission, payload, delays, threshold
cooperation, race winners, refractory state, and path alternatives under a
bounded shadow budget. That combination is a research hypothesis, not an
established novelty claim or a demonstrated result; a fuller literature
review and direct comparisons are required. The paired E83 pilot only tests
the receiver-bundle slice of that hypothesis.

## 143. Deep event learning needs both support and an activity operating point

The strict-chain support invariant in §137 tracks whether an utterance has
any event at depth $\ell$, but that binary statistic is only half of the
trainability problem. Let $N_\ell(x)$ be the number of emitted events,
$c_\ell=\Pr[N_\ell>0]$ the example support, and
$\mu_\ell=\mathbb E[N_\ell\mid N_\ell>0]$ the conditional event
multiplicity. Then

$$
n_\ell=\mathbb E[N_\ell]=c_\ell\mu_\ell.
$$

In a strict chain, $c_{\ell+1}\le c_\ell$ even when $n_{\ell+1}$ grows:
the surviving examples can emit many more events. Define the measured
activity amplification $\rho_\ell=n_{\ell+1}/n_\ell$ when $n_\ell>0$.
The product $\prod_\ell\rho_\ell=n_D/n_1$ is an exact identity for these
empirical means; interpreting each factor as a branching reproduction number
is only an approximation because messages combine, neurons can fire
repeatedly, and refractory state couples events. It is nevertheless a useful
warning signal: a layer may be support-starved while the work and downstream
evidence on the surviving examples are exploding.

There is a matching readout issue. If the class state is additive,

$$
\ell(T)=b+\sum_{e:t_e\le T} m_e,
$$

then $\|\ell(T)-b\|\le\sum_e\|m_e\|$. Thus raw per-event class
increments can grow linearly with event count when their signed means align;
zero-mean independent increments instead have a square-root scale. Which
regime holds is a representation question, not guaranteed by sparse fanout.
Consequently a usable deep operating point needs three diagnostics together:
nonzero example support, bounded event/work growth, and class-aligned
per-prefix evidence. Raising support alone is not a success criterion.

### Matched E83 intervention: support recovery versus useful computation

The layer-balanced follow-up to §142 changed only the pair proposal policy
in a matched D4 seed-6 run: same initialization, 120 training examples, 120
updates, one pair replay per minibatch, and 128 held-out examples. It selected
43, 34, 15, and 28 pairs from layers 1–4, respectively, whereas the global
sampler's 120/120 choices came from layer 1. The balanced policy therefore
did what it was designed to do—expose deeper route gates to pair credit—but
that intervention alone did not learn the classifier.

Layer-4 example support stayed at 100% in every balanced epoch, but final
held-out accuracy was 6/128 (4.69%), versus 8/128 (6.25%) for the matched
no-pair control; paired exact McNemar $p=0.791$. The global-pair arm scored
14/128 (10.94%), but its paired comparison with control was inconclusive
($p=0.180$). The balanced arm ended with only two predicted classes, a
late-prefix NLL of 14,699, and mean per-utterance layer event counts
$(122,189,363,1647)$. Averaged over its four epochs, the counts were
$(142,192,334,1421)$, so the measured adjacent-layer amplification ratios
were approximately $(1.35,1.74,4.25)$. This is a high-activity, collapsed
classifier, not a useful deep sparse representation. The one-seed matched
comparison implicates layer-balanced route updates in this activity regime,
but it does not isolate whether the cause is pair selection, noisy utility,
or the pre-existing threshold/reset dynamics.

The pair replays themselves were not consistently helpful: joint opening
reduced the matched class loss in 38/120 selected pairs (31.7%), and the
interaction $\Gamma<0$ in 30/120 (25.0%). Across epochs the raw
counterfactual-gradient/pathwise-gradient norm ratios were 0.0011–0.0146 and
cosines ranged from $-0.202$ to $+0.251$. These measurements separate three
claims that should not be conflated: deep alternatives were proposed; some
replays had positive local utility; the resulting update did not produce
reliable task accuracy. The layer-balanced sampler estimates a deliberately
layer-equalized proposal objective; without propensity weighting it is not
an unbiased estimate of the global candidate-pool gradient.

### Frozen validation audit: utility and descendant work depend on depth

The balanced checkpoint was next evaluated without updates on the same 128
speaker-held-out validation examples. One pair was sampled uniformly within
each layer/minibatch when eligible, giving 115 matched pairs: 32/32/19/32
from layers 1–4. The eligible pair pool sizes were
214,920/228/38/273. Layer 2 and 3 therefore offered far fewer alternatives
than layer 1, and layer 3 was absent in some minibatches.

The fraction whose joint opening lowered the matched prefix loss increased
with depth: 5/32 (15.6%) in layer 1, 10/32 (31.3%) in layer 2, 11/19 (57.9%)
in layer 3, and 16/32 (50.0%) in layer 4. Median $L_{11}-L_{00}$ was
0.000/0.236/-0.268/-0.001 respectively; the corresponding means were
19.93/74.07/1.36/0.46. The large early-layer means are outlier-sensitive,
so medians and helpful fractions are reported alongside them. These are
properties of one collapsed checkpoint, not estimates of general route
utility.

The same replay measures downstream work. A layer-2 pair added on average
4.24 layer-4 spikes and 10.76 layer-4 readout updates per example; a layer-3
pair added 1.79 and 4.04; a layer-4 pair added 0.23 and 0.45. Layer-1 pairs
added 0.78 layer-4 spikes and 1.83 layer-4 readout updates on average. The
layer-2 means hide a highly sparse distribution (median work deltas are zero
for its later-layer spikes/readout): a small number of shadows create large
cascades. This supports a specific next intervention—sample pair credit
from the last half of the stack while measuring its accuracy and event
growth—rather than uniformly pushing all layers. It is still an exploratory
selection hypothesis; the held-out outcomes have informed the proposal and
must not be presented as an untouched final test.

The four-outcome gate derivative also makes the layer difference explicit.
Reconstructing it from the same shadows using the logged route scores,
$\sigma=0.25$, and the training rule's $\pm5$ clipping gives mean
$\partial L/\partial s$ per route of $+0.291,+0.517,-0.013,-0.114$ for
layers 1–4. The fraction of sampled route margins with negative derivative
(so gradient descent would open the route) was 14.1%, 35.9%, 60.5%, and
50.0%. Early-layer pair credit points toward closing most sampled routes;
late-layer credit is more often favorable but close to zero in mean and
high-variance. Uniformly allocating the same shadow count to each layer
therefore does not imply a uniform useful update. This is a descriptive
calculation on held-out labels and one final checkpoint; it does not establish
that layer-3/4 routing will generalize or that this local gradient can win
against the pathwise update.

### Cost-constrained route utility

The §142 utility compares class loss alone. It assigns no penalty to a route
that lowers one batch's loss while multiplying downstream event count and
readout work. A direct extension is a constrained objective with per-layer
work $C_\ell$:

$$
\min_\theta\;\mathbb E[L_{\rm class}]
\quad\text{subject to}\quad
\mathbb E[C_\ell]\le B_\ell,
$$

whose Lagrangian is

$$
\mathcal J=\mathbb E[L_{\rm class}]
+\sum_\ell\lambda_\ell(\mathbb E[C_\ell]-B_\ell),\qquad
\lambda_\ell\ge0.
$$

For a matched route or pair shadow, use the same replay to measure both
$\Delta L_{\rm class}$ and $\Delta C_\ell$; the local utility then becomes
$U=-(\Delta L_{\rm class}+\sum_\ell\lambda_\ell\Delta C_\ell)$.
This gives the route gate a principled reason to recruit an event only when
its class benefit justifies its marginal downstream cost. Dual ascent on
$\lambda_\ell$ can enforce declared work budgets, but its stability depends
on the measured response of work to the chosen gate/threshold controls; that
monotonicity has not been established for the current reset dynamics.
Coverage should be monitored separately so the upper work constraint does
not reward the all-silent solution. A first experiment should collect
matched per-layer event, message, and replay-work deltas before selecting
budgets or multipliers. The frozen validation audit above now provides those
primitive deltas for receiver-bundle pairs, but not hardware-calibrated cost
weights or an optimal budget. Thus it narrows the next sampler and supplies
the data for a later constrained update; it does not yet validate that
training method.

## 144. Deep counterfactual support is itself sparse

The late-layer-only run (§143) isolates a limitation that is distinct from
ordinary layer support. In the implementation, a pair proposal requires two
distinct source events whose closed routes both target the same receiver and
whose arrival times are within a window $W$. The implementation groups
eligible routes by example and receiver, sorts them by source-event time,
and retains only adjacent entries within $W$ that come from distinct events.
Let $n_g$ be the eligible-route count in one example/receiver group. Its
exact pair count is

$$
P_\ell=\sum_g\sum_{j=1}^{n_g-1}
\mathbf 1\{t_{g,j+1}-t_{g,j}\le W\}
\mathbf 1\{e_{g,j+1}\ne e_{g,j}}.
$$

Therefore $P_\ell\le\sum_g(n_g-1)_+$. A single-route shadow needs only
$n_g\ge1$; a pair shadow needs $n_g\ge2$ plus valid time and event
identities. Under the sparse-occupancy approximation
$n_g\sim\operatorname{Poisson}(\lambda_g)$, the upper envelope is

$$
\mathbb E[(n_g-1)_+]=\lambda_g-1+e^{-\lambda_g}
=\tfrac12\lambda_g^2+O(\lambda_g^3),
\qquad
\Pr[n_g\ge2]=1-e^{-\lambda_g}(1+\lambda_g)
=\tfrac12\lambda_g^2+O(\lambda_g^3).
$$

Thus, at low event flux, the pair pool has a quadratic sparse-occupancy upper
envelope while the first-order pool shrinks linearly; the actual time and
event-identity filters can only reduce pair availability further. In a strict
chain, if source-event flux into layer $\ell$ is proportional to the previous
layer's realized support, the pair proposal bound compounds this support
loss. This is a conditional occupancy result, not a claim that every deep
model follows the Poisson approximation; the exact diagnostic is to count
$n_g$ by layer and receiver, then retain the observed gap and source identity
filters.

The matched late-only run demonstrates the distinction. It found 27 candidate
pairs across L3/L4 in epoch 1 and shadowed 9 of them. In epochs 2–4 it found
zero L3/L4 pairs, even though held-out L4 support remained nonzero in all
three epochs (3.1%, 0.8%, 0.8%). Candidate counts were overwhelmingly at L1: the per-epoch
layer counts were $[193140,3,6,18]$ in epoch 1, then $[188476,0,0,0]$,
$[180439,0,0,0]$, and $[180459,0,0,0]$. Final anytime accuracy was 6/128
(4.69%), versus 8/128 (6.25%) for the matched control. The run therefore
shows pair-proposal starvation, not that late route alternatives have
negative utility. The frozen validation audit found late pairs often helpful,
but that audit's one-checkpoint outcomes cannot create training support.

A no-update audit of the final late-only checkpoint tested windows from 25 to
1,000 ms on the same 120-example fit subset. Pair counts were
$[181577,1,0,0]$ at 25 ms and $[189987,6,0,1]$ at 1,000 ms across L1–L4.
At the widest window, L2 pairs occurred in only 6/30 batches, L3 in 0/30,
and L4 in 1/30. The near-closed route record counts were
$[191787,111,3,13]$. Widening the window therefore recovers almost no deep
co-occupancy; the 25 ms rule is not the dominant bottleneck. A full-second
window is only a candidate-count diagnostic, since late arrivals may no
longer interact under the receiver's integration kernel. Future ablations
must report both pair propensity and the state-dependent pair utility.

This makes the support condition in counterfactual gradient estimators
explicit. If $\mathcal P_\ell(x,\theta)$ is the available pair set and
$\pi(i\mid\mathcal P_\ell)$ is a sampling probability, inverse-probability
weighting can correct which available pair was selected. It cannot recover
the contribution of a missing pair set:

$$
\widehat g_\ell=
\mathbf 1\{\mathcal P_\ell\ne\varnothing\}
\sum_{i\in S_\ell}
\frac{\Delta_i\nabla p_i}{\pi(i\mid\mathcal P_\ell)},
\qquad
\mathbb E[\widehat g_\ell\mid x]
=\Pr(\mathcal P_\ell\ne\varnothing\mid x)\,
\mathbb E[\widehat g_\ell\mid\mathcal P_\ell\ne\varnothing,x].
$$

Increasing the pair budget or changing layer weights has no effect when the
indicator is zero. Conversely, broadening proposals without controlling the
resulting event cascade can recreate the layer-balanced arm's activity
explosion. A principled next design therefore keeps first-order route shadows
available when pairs are absent, measures pair occupancy separately from
event support, and adds higher-order shadows only where the observed
co-occupancy supports them. To give a deep route credit before factual
upstream activity exists, a shadow must create a plausible prefix event and
replay its sparse descendants; its proposal probability and event-work cost
must be logged. Such a prefix-expansion estimator is a design requirement,
not yet an implemented or validated method. An arbitrary larger time window
is not a principled substitute: choose $W$ from the receiver's integration
kernel and test its effect on both proposal availability and class-loss
utility.

### A first-failure spike can seed a sparse suffix replay

A refractory-aware paired audit compared the late-only checkpoint with its
matched no-pair control on the same 128 held-out speakers. For each layer and
batch, it toggled the closest threshold candidate, but local-boundary
analysis retains only candidates within $\pm0.25$ and outside refractory
state. The valid counts in L1–L4 were $[22,21,8,0]$ for control and
$[21,19,2,1]$ for late-only. Separating fused main-posterior loss from
weighted auxiliary losses, L1 main-loss spike-on helped 12/22 control
candidates (mean $+0.0266$) and 16/21 late-only candidates (mean $-0.0094$,
median $-0.0020$); the auxiliary loss moved in the same direction. Only 13
batches had valid candidates in both arms; the mean difference between the
two arm-specific selected spike-on utilities was $-0.037$ (SE $0.035$), and
the selected unit/time can differ by checkpoint. This is a
validation-conditioned lead, not a reliable treatment effect. For L2, 12/19
late-only main-loss toggles
helped individually, but their mean was $+0.0040$ (median $-0.00063$); the
control mean was $+0.0179$. L3/L4 have only 2/1 valid late-only candidates.
Raw helpful fractions across all selected spikes were misleading because
most deep toggles were outside the margin band. At this point, L1 appeared to
be a candidate prefix-credit hypothesis; §145 tests whether that utility
actually traverses the deep stack.

For a candidate spike with margin $m$, define the local randomized event risk
on the fused main-posterior loss, keeping deep-supervision loss separate:

$$
\widetilde L_{main}(m)=p_\tau(m)L_{main,on}
 +[1-p_\tau(m)]L_{main,off},
\qquad p_\tau(m)=\sigma(m/\tau),
\qquad
\frac{\partial\widetilde L_{main}}{\partial m}
=\frac{p_\tau(1-p_\tau)}{\tau}
(L_{main,on}-L_{main,off}).
$$

The training objective may also add $\alpha L_{aux}$, but its shadow delta
must be logged separately so an auxiliary win cannot masquerade as improved
sequence classification. A negative main-loss on-minus-off utility raises
the margin under gradient descent. The current evidence
supports a controlled L1-focused pilot: sample genuinely in-band,
nonrefractory first-layer failures, replay their ordinary sparse suffix, and
record per-layer event-work changes and the local-update/pathwise-gradient
alignment. Do not force out-of-band deep spikes. The per-layer work constraint
and prefix-expansion update remain unimplemented; this audit is not a
training result.

## Scope of the evidence

The D4 E83 implementation hard-gates message candidates but advances hidden
states on a 1 ms simulation grid; its reported state scans scale with grid
length times hidden width. The successful small E77 depth-8 check establishes
gradient reach for a different language-model setup, not SHD recognition,
and neither result establishes low-cost event-driven training. An
asynchronous sparse implementation, a stable per-layer activity regime, and
repeatable class learning remain separate requirements. The matched deepest-
only replay in §145 rules out treating the all-depth L1 utility as evidence
for a deep boundary update: it changes the shallow head but has no measured
deep loss effect. The next falsifiable depth experiment should either train a
deepest-only primary objective and diagnose the resulting support/gradient
failure, or use a sparse prefix-expansion counterfactual that explicitly
creates an in-band event and replays its suffix through ordinary dynamics.
It must compare deep and fused losses, log downstream spike/message/readout
work, boundary/pathwise gradient norms and cosine, layer support and event
multiplicity, prefix NLL, and anytime accuracy against a matched control. Any
shadow update needs a declared replay-work cap; the audit does not yet supply
or validate such a training mechanism.

## 145. All-depth readout can create a shallow counterfactual shortcut

The preceding audit measured spike-on utility against an all-depth fused
classifier. That objective directly exposes every hidden layer to its own
sparse class readout. Write the causal prefix logits as

$$
z_{all}(t)=\sum_{\ell=1}^{D}z_\ell(t),\qquad
L_{all}(t)=\operatorname{CE}(y,z_{all}(t)),
$$

whereas the matched deepest-only objective is

$$
z_{deep}(t)=z_D(t),\qquad
L_{deep}(t)=\operatorname{CE}(y,z_D(t)).
$$

For a layer-1 spike intervention, decompose the induced logit change as
$\delta z_{all}=\delta z_1+\sum_{\ell=2}^{D}\delta z_\ell$. If the event
changes the layer-1 head but no downstream hidden event or readout evidence,
then $\delta z_\ell=0$ for $\ell\ge2$. The all-depth loss can still change:

$$
\Delta L_{all}=
\operatorname{CE}(y,z_{all}+\delta z_1)-\operatorname{CE}(y,z_{all}),
$$

while $\Delta L_{deep}=0$ exactly whenever $\delta z_D=0$. Thus a nonzero
all-depth shadow delta does not by itself demonstrate credit through a deep
composition. It may be credit for a shallow classifier branch. This is a
causal distinction between direct readout utility and serial route utility,
not a general criticism of deep supervision; auxiliary heads can aid
optimization, but their contribution must be distinguished from the primary
deep output.

The matched frozen audit used the same seed-6 control and late-only checkpoints,
128 held-out-speaker examples, margin band $|m|\le0.25$, nonrefractory
candidates, and selected interventions under both fusion rules. For late-only
L1, 16/21 all-depth main-loss toggles helped and their mean was $-0.00936$;
under deepest-only loss, all 21 deltas were exactly zero. All valid L1/L2
deepest-only deltas were zero in both arms. The late-only L3/L4 deepest-only
counts were only 2/1, so they are not estimable. The L1 interventions changed
hidden-spike counts per example by $[+0.1429,0,0,0]$, accepted hidden messages
by $[0,+0.2857,0,0]$, and sparse readout-edge updates by
$[+1.2381,0,0,0]$. In particular, the L1 event did not create a downstream
hidden-spike cascade; it activated its own readout path. These exact zeros are
specific to the tested counterfactuals/checkpoints and do not prove that a
different architecture cannot propagate a useful L1 change.

This resolves the prior ambiguity: the all-depth L1 result is a shallow
readout shortcut, not evidence that the learned early event is composable
through layers 2–4. It also sharpens the experiment needed to test depth.
Train a matched primary deepest-only objective, or define a sparse
prefix-expansion counterfactual that explicitly inserts a plausible event and
replays its suffix through ordinary receiver, threshold, and refractory
dynamics. For every intervention, log both $\Delta L_{deep}$ and
$\Delta L_{all}$ together with per-layer event/message/readout-work deltas.
Only downstream changes that improve the primary deep loss count as serial
credit. Any prefix expansion must include its proposal probability and a
declared replay-work cap. This pilot is not implemented; the audit establishes
the shortcut mechanism in the tested model, not a successful remedy.
