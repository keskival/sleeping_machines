# Sleeping Machines v2: Formalism, Analysis, and Experimental Program

## Abstract

Sleeping Machines is a computational paradigm in which the **future field of causal events is part of machine state**. Instead of representing computation primarily as repeated synchronous transformations of a dense state tensor, a machine maintains local state together with pending events carrying relative temporal delays.

The central primitive is:

\[
\boxed{\text{race}\rightarrow\text{fire winner}+\text{cancel losers}+\text{remember counterfactuals}}
\]

A losing event need not be discarded. Its predicted firing time, cancellation status, and race margin can remain in local history and participate in later learning. This makes a cancelled event a genuine counterfactual learning object.

This document develops a mathematical formalism and an experimental program for testing whether these ideas form a viable basis for trainable asynchronous sparse architectures. The experiments are feasibility tests, not a proof of universal superiority.

---

## 1. Machine state

At time \(t\), define

\[
M_t=(S_t,F_t,H_t,\Theta).
\]

- \(S_t\): active/local state.
- \(F_t\): pending future events.
- \(H_t\): relevant history, including cancelled events.
- \(\Theta\): trainable parameters.

The crucial distinction is

\[
e\notin F_t \quad\not\Rightarrow\quad e\notin H_t.
\]

An event can leave the pending set because it fired or was cancelled while remaining represented in history.

### Event

An event is

\[
e=(\tau,m,x,\kappa),
\]

with residual latency \(\tau\ge0\), mark/type \(m\), payload \(x\), and metadata \(\kappa\). Its absolute arrival time is

\[
T(e)=t+\tau.
\]

Its lifecycle is

\[
\text{pending}\rightarrow\{\text{fired},\text{cancelled}\}.
\]

---

## 2. Counterfactual memory

Suppose

\[
T_A<T_B.
\]

Then A fires and B is cancelled. A minimal implementation might erase B. Sleeping Machines instead retains, locally,

\[
B,\quad T_B,\quad \Delta_B=T_B-T_A,\quad \text{status}(B)=\text{cancelled}.
\]

The quantity

\[
\Delta_B>0
\]

is the race margin. A close loser is a particularly plausible counterfactual alternative.

Define counterfactual eligibility

\[
E_B^c=g(\Delta_B),
\]

for example

\[
E_B^c=e^{-\beta\Delta_B}.
\]

Thus near-misses receive more learning credit than alternatives that were far too late.

This is one of the most distinctive parts of the formalism.

---

## 3. Event weaving

The next causal event is

\[
e^*=\arg\min_{e\in F_t}T(e).
\]

The machine advances directly to its event time rather than stepping through a global clock.

Define

\[
W:(S,F,H,\Theta,e^*)\mapsto(S',F',H',\Theta').
\]

Conceptually,

\[
W=W_H\circ W_C\circ W_F\circ W_S\circ W_Q,
\]

where scheduling, state transition, firing, cancellation, and history update are applied in the appropriate causal order.

A local race therefore has four consequences:

1. one candidate fires;
2. competing candidates are cancelled;
3. cancelled alternatives may be retained;
4. their local traces may affect later learning.

---

## 4. Causality

Let \(\mathcal F_t\) denote information available by time \(t\). A valid event process may depend only on \(\mathcal F_t\).

Deterministic routes satisfy

\[
T_e=t+d_e,\qquad d_e\ge0.
\]

Stochastic routes can be represented by conditional intensities

\[
\lambda_e(t\mid\mathcal F_t)\ge0.
\]

For all candidates,

\[
\Lambda(t)=\sum_e\lambda_e(t)
\]

is the total hazard. This connects the formalism to standard point-process mathematics.

---

## 5. Temporal races

### Hard race

Given arrival times \(T_1,\ldots,T_n\),

\[
T^*=\min_iT_i,
\qquad
i^*=\arg\min_iT_i.
\]

The winner propagates; the losers are cancelled.

### Soft race

For training, use

\[
T_\beta=-\frac1\beta\log\sum_i e^{-\beta T_i}.
\]

Its derivative is

\[
\frac{\partial T_\beta}{\partial T_i}
=
\frac{e^{-\beta T_i}}{\sum_j e^{-\beta T_j}}.
\]

As \(\beta\to\infty\), this converges to the hard minimum.

This gives a practical route:

\[
\text{soft differentiable race}
\rightarrow
\text{annealing}
\rightarrow
\text{hard event-driven execution}.
\]

---

## 6. Temporal softmax

If

\[
T_i\sim\operatorname{Exp}(\lambda_i),
\]

then

\[
P(i\text{ wins})=\frac{\lambda_i}{\sum_j\lambda_j}.
\]

With

\[
\lambda_i=e^{z_i},
\]

this becomes exactly

\[
P(i\text{ wins})=\frac{e^{z_i}}{\sum_je^{z_j}}.
\]

Therefore softmax competition has a temporal race interpretation.

---

## 7. Trainable temporal geometry

A connection has parameters

\[
\theta_{ij}=(w_{ij},d_{ij},\rho_{ij},\ldots).
\]

A presynaptic event

\[
(t,x)
\]

is transformed into

\[
(t+d_{ij},w_{ij}x).
\]

The important point is that \(d_{ij}\) is a computational parameter, not merely a hardware delay.

The network therefore learns a **causal temporal geometry**.

---

## 8. Three-factor temporal learning

Let \(E_e(t)\) be local eligibility and \(M(t)\) a delayed teaching/modulatory signal:

\[
\Delta\theta_e=\eta M(t)E_e(t).
\]

For weights and delays,

\[
\Delta w_e=\eta_wME_e^w,
\qquad
\Delta d_e=\eta_dME_e^d.
\]

Eligibility may contain ordinary activity and counterfactual terms:

\[
E_e=E_e^{active}
+\alpha E_e^c\mathbf1[e\text{ cancelled}].
\]

The distinctive rule is that a cancelled route can remain locally eligible without having fired.

---

## 9. Signed counterfactual delay learning

For a delayed teacher specifying the desired winner,

\[
M_e=
\begin{cases}
+1,&e\text{ should win but lost},\\
-1,&e\text{ should lose but won},\\
0,&\text{otherwise}.
\end{cases}
\]

For a cancelled candidate,

\[
\Delta d_e=-\eta_dM_eE_e^c.
\]

Example:

\[
T_A=10.0\text{ ms},\qquad T_B=10.8\text{ ms}.
\]

A wins, but the delayed target says B should have won. B's local history contains its 0.8 ms miss, so its delay can be shortened. Repetition can move B through

\[
10.8\rightarrow10.4\rightarrow10.1\rightarrow9.9\text{ ms},
\]

without requiring B to have fired.

The experimental program explicitly tests this mechanism against controls where cancelled events are discarded.

---

## 10. Predictive event learning

For stochastic future events, a point-process negative log likelihood is

\[
\mathcal L
=
-\log\lambda_{e^*}(t^*)
+
\int_0^{t^*}\Lambda(t)\,dt.
\]

This provides a standard mathematical route for learning distributions over future events rather than only deterministic delays.

---

## 11. Temporal attention

Query-key similarity \(s(q,k)\) can modify delay,

\[
d(q,k)=d_0-\alpha s(q,k),
\]

or hazard,

\[
\lambda(q,k)=\lambda_0e^{\beta s(q,k)}.
\]

Relevant interactions then tend to arrive sooner or win more often.

However, this does not by itself solve the quadratic candidate-generation problem. A scalable architecture still needs sparse candidate generation, hierarchical routing, locality, hashing, or similar mechanisms.

---

## 12. Sparse computation

Let \(N_{active}\) be processed events and \(N_{capacity}\) possible nodes.

The desired regime is

\[
N_{active}\ll N_{capacity}.
\]

Approximate event work is

\[
C_{event}\approx
\sum_{e\in E_{active}}\operatorname{fanout}(e)
+C_{race}+C_{queue}.
\]

Dense timestep simulation instead pays roughly

\[
C_{dense}\approx N_{steps}N_{nodes}.
\]

The engineering hypothesis is therefore that useful computation can be concentrated in sparse event cascades.

This is an execution-complexity hypothesis; it is not automatically a physical energy claim.

---

## 13. Structural plasticity

The learned graph can change through:

- creation;
- deletion;
- strengthening;
- weakening;
- advancing routes;
- retarding routes;
- changing event reliability/hazard.

Thus the object being learned is potentially

\[
G=(V,E,\Theta_E)
\]

with temporal parameters on its edges.

The long-term objective is to learn the causal temporal graph itself.

---

## 14. Biological predictions

The formalism predicts measurable signatures:

1. timing manipulations should matter even when spike counts are held fixed;
2. relative event order should matter;
3. synaptic delays should become task-specific;
4. close losing alternatives should leave measurable traces;
5. those traces should affect subsequent plasticity;
6. delayed modulatory signals should alter the timing of locally eligible alternatives;
7. useful computation should concentrate into sparse event cascades.

These are hypotheses to test, not assumptions.

---

# 15. Experimental program

## Experiment A — timing necessity

Generate samples containing one A, one B, and distractors.

Class 0:

\[
T_A<T_B.
\]

Class 1:

\[
T_B<T_A.
\]

Event counts are identical.

Compare:

- count-only MLP;
- GRU using event order and time;
- temporal race.

Then scramble type/time associations while preserving event counts and the set of event times.

A successful result should show a substantial timing-dependent accuracy drop.

---

## Experiment B — learned temporal geometry

Compare temporal races with:

- trainable delays;
- fixed delays.

On a task requiring temporal reordering, learned delays should improve accuracy and move toward the task's useful ordering.

The delay values themselves should be inspected, not just the final score.

---

## Experiment C — soft-to-hard compilation

Train the temporal race while annealing \(\beta\) from a soft regime to a sharp regime.

Evaluate both the differentiable model and the genuine hard minimum.

A small soft-to-hard accuracy gap supports the claim that gradient-friendly temporal competition can compile into event-driven execution.

---

## Experiment D — counterfactual learning

Compare:

1. **none** — cancelled events discarded;
2. **binary** — cancelled event remembered with constant eligibility;
3. **margin** — cancelled event remembered with \(e^{-\beta\Delta}\).

The desired winner alternates over training so both advancing and retarding delays are required.

A strong result is:

\[
\text{margin} > \text{binary} > \text{none}
\]

in learning speed or final target-winner accuracy.

---

## Experiment E — event economy

Use a toy sparse network to compare:

- dense node-by-timestep work;
- scheduled events;
- fired events;
- cancelled events;
- queue/race work.

This establishes whether the proposed execution model actually exploits event sparsity.

It does not by itself establish silicon energy efficiency.

---

# 16. Experimental controls

The experiments should be accompanied by ablations:

- remove timing information;
- scramble timing;
- freeze delays;
- remove counterfactual history;
- retain cancellation but remove race margin;
- use binary rather than graded eligibility;
- disable structural plasticity;
- compare soft and hard races;
- increase distractor density;
- increase race width;
- compare event-driven work with dense timestep work.

A particularly important control is a conventional SNN/LSM implementation with comparable parameter count. If it reproduces the results without counterfactual temporal state, the novelty claim becomes weaker.

---

# 17. Proposed architecture: Sparse Temporal Causal Network

Each node stores

\[
S_i,F_i,H_i
\]

and each connection stores

\[
(w_{ij},d_{ij},\rho_{ij}).
\]

Runtime:

```text
while pending events exist:
    take earliest event
    advance local time to its arrival
    resolve the local race
    fire the winner
    cancel competing events
    retain counterfactual traces
    update local state
    schedule downstream events
    accumulate eligibility
```

The critical operation is:

```text
remember(cancelled_event)
```

rather than

```text
discard(cancelled_event)
```

---

# 18. Differentiable reference model

A minimal candidate arrival model is

\[
T_i=x_i+d_i-\frac{1}{\beta}\log(w_i+\epsilon).
\]

The class output can be generated with the soft race

\[
T_\beta=-\frac1\beta\log\sum_i e^{-\beta T_i}.
\]

This deliberately small model tests the primitive rather than claiming to be a complete neural architecture.

---

# 19. Hard event-driven implementation

Use a priority queue whose entries contain

\[
(t,\text{node},\text{event type},x,\text{lineage}).
\]

Extract the minimum arrival time, resolve local competition, propagate the winner, and retain losing alternatives in local history.

No global timestep is required by the computational model.

---

# 20. Benchmark suite

Initial baselines:

- count-only MLP;
- MLP with engineered timing features;
- GRU/LSTM;
- standard SNN;
- liquid state machine;
- temporal race;
- temporal race with learned delays;
- temporal race with counterfactual eligibility.

Report:

- accuracy;
- convergence speed;
- latency;
- processed events;
- scheduled events;
- cancelled events;
- active parameters;
- memory;
- approximate operations.

For eventual hardware studies also measure joules/sample, power, latency, and throughput.

---

# 21. Falsification criteria

The paradigm is weakened if:

1. count-only models solve timing-dependent tasks equally well;
2. learned delays provide no benefit where timing should matter;
3. soft races cannot be hardened without substantial accuracy loss;
4. counterfactual memory provides no learning benefit;
5. local learning requires essentially the same global backpropagation cost;
6. sparsity disappears under realistic workloads;
7. candidate generation secretly becomes dense;
8. cancellation and queue overhead dominates;
9. ordinary SNNs reproduce the same effect without counterfactual temporal state.

The final point is crucial: the proposal should not be presented as merely a rebranding of SNNs.

---

# 22. Relationship to existing fields

Sleeping Machines overlaps with SNNs, liquid-state machines, temporal point processes, neural ODE/CDE models, three-factor learning, and neuromorphic hardware.

Its proposed distinguishing feature is the explicit computational status of future events and the retention of cancelled alternatives as local counterfactual state.

Thus the strongest novelty claim is not:

> spike timing matters.

It is:

> future-event state, temporal races, cancellation, and counterfactual eligibility can form first-class primitives for trainable sparse computation.

---

# 23. Research roadmap

### Stage 1 — primitive validation

Run the included suite.

### Stage 2 — multi-layer causal network

Add sparse recurrent connectivity, multiple races, explicit cancellation, counterfactual memory, and learned delays.

### Stage 3 — local learning

Replace end-to-end gradients with

\[
\Delta\theta_e=\eta M E_e.
\]

Measure the cost in accuracy and convergence.

### Stage 4 — representation learning

Introduce event embeddings, hierarchical races, temporal attention, and sparse routing.

### Stage 5 — realistic workloads

Test event-camera streams, asynchronous sensors, symbolic temporal reasoning, long-context prediction, and sparse control.

### Stage 6 — hardware

Map the event model to asynchronous processing elements, local state, event queues, programmable delays, local plasticity, and sparse interconnect.

---

# 24. What would constitute convincing evidence?

A strong sequence would be:

1. timing-dependent tasks defeat count-only models;
2. temporal races solve them;
3. learned delays outperform fixed delays;
4. soft races can be hardened;
5. cancelled events improve local temporal learning;
6. removing counterfactual memory removes that advantage;
7. the effect survives in multi-layer sparse networks;
8. event counts remain far below dense timestep-equivalent work;
9. local eligibility plus delayed teaching remains effective;
10. conventional SNN baselines cannot reproduce the result without introducing essentially the same mechanism.

That would constitute a credible empirical basis for Sleeping Machines as a computational formalism.

---

# 25. Core thesis

The strongest formulation is:

> A useful class of neural computation may be represented more naturally as an evolving causal field of future events than as repeated synchronous transformations of a dense state tensor.

The distinctive learning hypothesis is:

> A losing event is not necessarily computationally irrelevant. Its predicted time, cancellation status, and race margin can form a local counterfactual memory that enables temporal credit assignment.

The core primitives are therefore:

\[
\boxed{
\text{race}\rightarrow\text{fire}+\text{cancel}+\text{remember}
}
\]

and

\[
\boxed{
\text{counterfactual eligibility}
\times
\text{delayed teaching signal}
\rightarrow
\text{change temporal competition}.
}
\]

If these mechanisms scale while preserving sparse event-driven execution, they provide a plausible basis for a trainable asynchronous computational substrate.
