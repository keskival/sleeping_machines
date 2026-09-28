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
an initial aggregate event budget $\rho$ per character, set

$$
\theta=\widehat F_V^{-1}(1-\rho/M),
$$

where $\widehat F_V$ is the empirical voltage CDF. Without resets this gives
approximately $M\Pr[V>\theta]=\rho$ threshold attempts per time bin; reset,
temporal dependence, and finite calibration samples alter the realized count,
so log the actual firing rate. Calibrating in depth order lets each layer see
the already-calibrated event stream below it. This is a data-derived
initialization, not a learned posterior or proof of stable depth. It uses the
observed conditional activity instead of a guessed threshold prior while the
network has little task evidence. Later threshold changes should be slower or
confidence-weighted, because noisy minibatch quantiles otherwise make event
rates oscillate; that online homeostasis is not yet implemented.

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
