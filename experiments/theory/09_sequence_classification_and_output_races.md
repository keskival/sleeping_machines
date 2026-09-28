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
2. **No winner.** If no confidence reaches $\theta$, the deployed classifier
   remains silent. The training loss penalizes this through low correct-win
   probability, but a threshold above the model's achievable confidence can
   still make coverage collapse. Keep no-answer as an explicit outcome and
   report coverage; for benchmark scoring, report any forced end-of-utterance
   answer separately.
3. **False confidence.** Softmax is a normalized score, not a calibrated
   probability. An apparently high confidence can be wrong, especially under
   held-out speakers or distribution shift. Calibrate $T_o$ and $\theta$ on
   validation data only, then freeze both for test.

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
the dimensioned 1 ms integration factor now in the code. A matched depth-4,
512-example, five-epoch objective comparison is running. Even if its losses
learn, no claim of early reliable output follows without a useful
accuracy-coverage-latency frontier.

**References.** Cramer et al., *The Heidelberg Spiking Data Sets for the
Systematic Evaluation of Spiking Neural Networks* (2022),
[paper](https://kip.uni-heidelberg.de/Veroeffentlichungen/download.php/6616/temp/4143-3.pdf);
Spyx, [SHD training tutorial](https://spyx.readthedocs.io/en/latest/examples/surrogate_gradient/SurrogateGradientTutorial/)
and [integral cross-entropy definition](https://spyx.readthedocs.io/en/latest/reference/fn/).

**Evidence status.** The race likelihood and causal decision rule are
analytical specifications. The old E83 softmax-mean failure follows directly
from its pooling equation. The shared-gate form has only a tiny failed smoke;
the larger objective comparison is guarded and underway.
