# Sleeping Machines: A Formalism for Asynchronous Temporal Computation and Learning

## Abstract

This document develops the `Sleeping Machines` idea into a candidate mathematical formalism for computation in which **time, rather than addressable memory, is a primary computational resource**.

The central object is not a clocked vector of node states, but a causal process consisting of:

1. a current state,
2. a sparse set or distribution of pending events,
3. causal transformations that create, cancel, delay, accelerate, or inhibit future events,
4. an event-selection mechanism that advances computation from one causal event to the next.

This formulation is deliberately close to the intuition in the original manifesto: nodes may "sleep", wake when a causal event reaches them, and emit new events after state-dependent delays. The resulting machine is asynchronous, sparse, and naturally parallel.

The most important extension proposed here is to make **latency a learnable computational parameter**, alongside synaptic strength. A connection is therefore not merely

\[
w_{ij},
\]

but

\[
\theta_{ij}=(w_{ij},d_{ij},\sigma_{ij},\ldots),
\]

where \(w\) controls the influence of an event and \(d\) controls when that influence becomes available. This makes the temporal geometry of the network part of the learned representation.

A practical learning system can then be constructed around **eligibility traces over causal events** and a sparse global learning signal:

\[
\Delta\theta_e
=
\eta\,M(t)\,E_e(t),
\]

where \(E_e\) is a locally maintained eligibility trace and \(M\) is a delayed error, reward, novelty, or prediction-error signal. This is a temporal analogue of three-factor plasticity and is closely related to e-prop and event-driven learning in spiking neural networks, but the proposed formalism treats *events and their latency distributions themselves* as the primitive computational state.

The goal is not to claim that this formalism is already superior to transformers or conventional neural networks. The goal is to define a sufficiently precise experimental substrate that the hypothesis can be falsified.

---

## 1. Motivation

The original Sleeping Machines hypothesis starts from a simple observation: a computation can be represented by **when something happens**, not only by **where something is stored**.

Sleep-sort is a toy example. Given numbers \(x_i\), one can represent each number as a delay and emit an event after that delay. The order in which events emerge is then the sorted order.

The important abstraction is not sorting itself. It is:

> **A system can represent information by temporal relationships and perform computation by causal interaction between delayed events.**

The conventional von Neumann abstraction makes memory addresses and synchronous state transitions primary. A Sleeping Machine instead makes:

- causal events,
- delays,
- temporal coincidence,
- cancellation,
- competition,
- propagation,
- and sparse activation

primary.

This is potentially interesting for biological neural computation and neuromorphic hardware because biological systems have:

- sparse activity,
- asynchronous events,
- relatively fixed local connectivity,
- substantial computation in membrane and synaptic dynamics,
- and no global high-frequency clock coordinating every operation.

Existing spiking neural networks already exploit many of these properties. The purpose of this formalism is therefore **not** to rediscover SNNs under another name. It is to make a stronger statement:

> The computational primitive should be a *causal temporal process*, with event latency and future-event distributions treated as first-class state variables.

The distinction matters because a conventional SNN is often still described as a discretized dynamical system:

\[
x_{t+1}=F(x_t,u_t).
\]

A Sleeping Machine should instead admit a representation such as

\[
(S_t,\mathcal F_t)
\longrightarrow
(S_{t+\Delta},\mathcal F_{t+\Delta}),
\]

where \(\Delta\) is itself an event-dependent quantity and \(\mathcal F_t\) is a sparse population of pending future possibilities.

---

# 2. Formal computational state

Let a Sleeping Machine at physical time \(t\) be

\[
\mathcal M_t=(S_t,F_t).
\]

Here:

- \(S_t\) is the **present state**,
- \(F_t\) is the **future-event field**.

The crucial difference from an ordinary recurrent network is that future activity is explicitly represented.

## 2.1 Present state

Let

\[
S_t \in \mathcal S.
\]

A node can have arbitrary internal state. For a minimal machine, this could be a scalar membrane-like potential:

\[
v_i(t)\in\mathbb R.
\]

A richer node might have

\[
s_i(t)=
(v_i(t),a_i(t),r_i(t),q_i(t),\ldots),
\]

where \(a_i\) might represent adaptation, \(r_i\) refractory state, and \(q_i\) some learned contextual variable.

The formalism does not require the state to be neuronal. It can be any computational state.

---

# 3. Future-event field

The future is represented by events of the form

\[
e=(\tau,\phi,\xi,\kappa),
\]

where:

- \(\tau>0\) is the remaining latency,
- \(\phi\) identifies the operation or destination,
- \(\xi\) contains payload,
- \(\kappa\) identifies the causal lineage or event type.

Thus

\[
F_t=\{e_1,e_2,\ldots\}.
\]

A deterministic event has a definite latency:

\[
\tau_e=d_e.
\]

A stochastic event has a latency distribution:

\[
\tau_e\sim P_e(\tau\mid S_t).
\]

This immediately generalizes the original README's idea of a "pool of future potential events".

## 3.1 Survival-function representation

For probabilistic events it is often more convenient to represent the future by a survival function

\[
G_e(\tau)
=
P(T_e>\tau).
\]

Its hazard rate is

\[
\lambda_e(\tau)
=
-\frac{d}{d\tau}\log G_e(\tau).
\]

The hazard representation is particularly useful because the probability that event \(e\) occurs in a short interval \(dt\), conditional on not having occurred yet, is

\[
P(e\text{ fires in }[t,t+dt)\mid e\text{ alive})
\approx
\lambda_e(t)\,dt.
\]

Therefore the Sleeping Machine can be formulated as a **competing-risks process**: many possible events sleep concurrently, and whichever causal event becomes active first advances the computation.

---

# 4. The event transition operator

Let

\[
\mathcal M_t=(S_t,F_t).
\]

Let

\[
\Delta t=\min_{e\in F_t}\tau_e.
\]

The next causal event is

\[
e^\*=\operatorname*{arg\,min}_{e\in F_t}\tau_e.
\]

The machine advances directly to the next event:

\[
t' = t+\Delta t.
\]

No global clock tick is required.

The transition is

\[
(S_t,F_t)
\xrightarrow{e^\*}
(S_{t'},F_{t'}).
\]

Define

\[
\Phi_{e^\*}(S_t,F_t)
=
(S_{t'},F_{t'}).
\]

This is the fundamental transition operator.

A conventional discrete-time neural network therefore appears as a special case in which all events are constrained to occur at integer multiples of a global \(\Delta t\).

---

# 5. Causal semantics

The central invariant is:

\[
e\text{ at }t
\quad\text{may depend only on information available at }t^-.
\]

No transition may depend on a future event.

For an event \(e\) generated at time \(t\),

\[
P(e\mid\mathcal H_t)
\]

must be measurable with respect to the causal history \(\mathcal H_t\).

This gives the formalism a natural filtration:

\[
\mathcal H_t
=
\sigma\{e:\operatorname{time}(e)\le t\}.
\]

Every future-event distribution must satisfy

\[
P(F_{>t}\mid\mathcal H_\infty)
=
P(F_{>t}\mid\mathcal H_t).
\]

This is essentially the mathematical statement that the machine is causal.

---

# 6. Primitive temporal operations

A useful computational formalism needs a small set of primitives from which larger computations can be constructed.

## 6.1 Emit

Create an event after delay \(d\):

\[
\operatorname{emit}(\phi,x,d).
\]

Operationally:

\[
F\leftarrow F\cup
\{(d,\phi,x)\}.
\]

## 6.2 Cancel

Remove an event:

\[
\operatorname{cancel}(e).
\]

Cancellation is important because computation is not only about producing signals. It is also about eliminating counterfactual futures.

## 6.3 Delay

Transform

\[
d\mapsto d+\delta.
\]

The delay may depend on current state:

\[
d' = d+\delta(S).
\]

## 6.4 Advance

Reduce all outstanding delays by the elapsed causal interval:

\[
d_i\mapsto d_i-\Delta t.
\]

## 6.5 Accumulate

Update a state variable when an event arrives:

\[
v_j\leftarrow v_j+w_{ij}x_i.
\]

## 6.6 Threshold / race

Emit an event when a condition becomes true:

\[
g(S_t)>0
\Rightarrow
\operatorname{emit}(\phi,x,0).
\]

A particularly important special case is the **race primitive**:

\[
e^\*=\arg\min_i\tau_i.
\]

The first branch to arrive wins.

This gives a temporal analogue of an argmax.

---

# 7. Temporal logic

The original causal rule

> Emit A after delay \(x\) if no B arrives before that

can be formalized as

\[
A
\leftarrow
\operatorname{timeout}(x)
\land
\neg\operatorname{before}(B,x).
\]

More explicitly, define

\[
T_A=t+x.
\]

Then

\[
A
\iff
B\notin\mathcal H_{[t,T_A)}.
\]

This is a genuine causal operator.

It has an important property: the system does not need to inspect the entire future. It only needs to wait until either:

1. \(B\) arrives, or
2. the timeout occurs.

Thus temporal logic can be implemented through local races.

---

# 8. Temporal coincidence

Two events

\[
e_i=(t_i,x_i),
\qquad
e_j=(t_j,x_j)
\]

may interact according to their relative timing

\[
\Delta t=t_i-t_j.
\]

Define a coincidence kernel

\[
K(\Delta t).
\]

For example,

\[
K(\Delta t)
=
A_+e^{-\Delta t/\tau_+}\mathbf 1_{\Delta t>0}
-
A_-e^{\Delta t/\tau_-}\mathbf 1_{\Delta t<0}.
\]

This is structurally similar to STDP kernels, but in the Sleeping Machine it is not merely a learning rule. It can be a **computational primitive**.

The same temporal relation can simultaneously affect:

- state,
- event latency,
- event probability,
- event amplitude,
- and learning eligibility.

---

# 9. Signals are event streams

Instead of representing an input as a vector

\[
x\in\mathbb R^n,
\]

represent it as a marked point process

\[
X=\{(t_k,m_k)\}_{k=1}^{N}.
\]

The mark \(m_k\) can contain:

- channel identity,
- amplitude,
- polarity,
- learned embedding,
- precision,
- or a symbolic token.

A conventional vector can be encoded as a burst of simultaneous events:

\[
x_i
\mapsto
(t_0,i,x_i).
\]

But this is only one encoding.

The richer representation is

\[
x
\mapsto
\{(t_k,i_k,a_k)\}.
\]

The network can then compute directly on temporal structure.

---

# 10. Synapses as temporal operators

A synapse should be represented by

\[
\theta_{ij}
=
(w_{ij},d_{ij},\rho_{ij},\ldots).
\]

The simplest transformation is

\[
(t,x)
\mapsto
(t+d_{ij},w_{ij}x).
\]

Thus a synapse does two independent things:

1. changes **what** arrives;
2. changes **when** it arrives.

This suggests a key architectural hypothesis:

> Learned temporal delays may carry representational information just as learned weights do.

The network therefore learns a temporal geometry.

---

# 11. Node dynamics

A minimal node can integrate arriving events:

\[
v_i
\leftarrow
\alpha_i(\Delta t)v_i
+
\sum_{e\to i}w_e x_e,
\]

where

\[
\alpha_i(\Delta t)=e^{-\Delta t/\tau_i}.
\]

When

\[
v_i>\theta_i,
\]

the node emits:

\[
e_i=(t,\phi_i,q_i).
\]

After firing,

\[
v_i\leftarrow v_i-v_{\text{reset}}
\]

or

\[
v_i\leftarrow v_{\text{reset}}.
\]

This recovers the leaky integrate-and-fire neuron as one implementation of the formalism.

But it is not mandatory. A node could instead implement:

- a nonlinear oscillator,
- a stochastic accumulator,
- a finite-state machine,
- a symbolic matcher,
- a differentiable state-space model,
- or a learned local dynamical system.

---

# 12. Temporal computation as path competition

Consider a network containing many possible causal paths from input \(X\) to output \(Y\).

Each path \(p\) has latency

\[
T_p
=
\sum_{e\in p}d_e
+
\sum_{v\in p}\delta_v.
\]

If the output selects the earliest arrival,

\[
Y=\arg\min_p T_p,
\]

then the machine computes a shortest-path-like operation.

If the output selects the highest amplitude among arrivals,

\[
Y=\arg\max_p A_p,
\]

it computes a conventional winner-take-all operation.

More interestingly, if amplitude and latency interact,

\[
Y
=
\operatorname{softargmin}_p
\left(
T_p-\beta A_p
\right),
\]

the system obtains a continuous competition between evidence and temporal priority.

This gives a plausible mathematical interpretation of the "neurons integrate confidence and fire when confidence becomes sufficiently large" idea in the README.

---

# 13. Temporal softmax

A particularly useful primitive is a temporal analogue of softmax.

Given alternatives with scores \(z_i\), define their firing latency

\[
d_i=\tau_0e^{-\beta z_i}.
\]

Then the largest score tends to produce the earliest event.

Alternatively, define stochastic hazards

\[
\lambda_i
=
\lambda_0e^{\beta z_i}.
\]

For competing exponential races,

\[
P(i\text{ wins})
=
\frac{e^{\beta z_i}}
{\sum_j e^{\beta z_j}}.
\]

This is exactly the categorical softmax distribution.

This is an important bridge between conventional neural computation and temporal computation:

> **Softmax can be implemented as a race between stochastic event generators.**

The normalization is performed by the competition itself rather than by explicitly computing a sum followed by division.

---

# 14. Temporal attention

Suppose queries \(q_i\), keys \(k_j\), and values \(v_j\) are represented as events.

A conventional attention score is

\[
a_{ij}
=
\frac{q_i^\top k_j}{\sqrt d}.
\]

In a Sleeping Machine, convert this score into a delay or hazard:

\[
d_{ij}
=
\tau_0e^{-\beta a_{ij}},
\]

or

\[
\lambda_{ij}
=
\lambda_0e^{\beta a_{ij}}.
\]

Then relevant keys arrive sooner or generate more likely events.

A temporal attention mechanism could therefore be implemented as:

1. generate candidate interactions;
2. encode similarity as latency/hazard;
3. allow events to race;
4. integrate the earliest or strongest arrivals;
5. suppress later redundant events.

This does **not** mean that a temporal architecture automatically beats transformer attention. The important point is that it provides a concrete experiment:

> Compare a dense dot-product attention layer with an event-race implementation of the same mathematical selection operation.

---

# 15. Sparse connectivity

Let a network contain \(N\) nodes and \(E\) connections.

A conventional dense layer has

\[
E=O(N^2).
\]

A sparse Sleeping Machine instead aims for

\[
E=O(kN),
\qquad k\ll N.
\]

But sparse connectivity alone is not sufficient.

The key requirement is:

\[
\text{active events per unit time}
\ll
E.
\]

An ideal execution engine should therefore process only events that actually occur.

The computational cost becomes approximately

\[
O(N_{\text{events}}\log E)
\]

for a priority-queue implementation, or potentially closer to

\[
O(N_{\text{events}})
\]

with bucketed timing or specialized hardware.

This is where asynchronous computation could provide a genuine advantage over dense synchronous matrix multiplication.

---

# 16. Learning: the central proposal

The main proposed learning mechanism is **event-local eligibility propagation with delayed global credit assignment**.

The fundamental parameter of an edge is

\[
\theta_e=(w_e,d_e).
\]

For every edge, maintain an eligibility state

\[
E_e(t).
\]

When an event traverses the edge, update eligibility locally.

Then, when a learning signal \(M(t)\) becomes available,

\[
\Delta\theta_e
=
\eta M(t)E_e(t).
\]

This is a three-factor rule:

1. local presynaptic activity,
2. local postsynaptic/temporal state,
3. global or regional modulatory/error signal.

Three-factor learning rules are already a major framework in biologically plausible learning and neuromorphic computation. e-prop similarly uses local eligibility traces combined with a learning signal to approximate temporal credit assignment. Event-driven SNN research has also specifically investigated ways to preserve sparse event execution during learning.

The novelty here is to make **temporal latency itself a first-class learnable parameter**.

---

# 17. Eligibility for weight learning

Suppose an event \(x(t)\) crosses edge \(e=(i,j)\).

The local contribution to the postsynaptic state is

\[
\delta v_j(t)=w_e x(t).
\]

Define a local eligibility trace

\[
\dot E^w_e
=
-\frac{E^w_e}{\tau_E}
+
x_i(t)\,
\psi_j(t),
\]

where \(\psi_j\) is a postsynaptic sensitivity term.

For a threshold neuron, \(\psi_j\) can be a smooth surrogate derivative around threshold.

Then

\[
\Delta w_e
=
-\eta_w\int M(t)E^w_e(t)\,dt.
\]

In event-driven form, there is no need to integrate continuously. The trace is updated only when relevant events occur.

---

# 18. Eligibility for delay learning

This is the more interesting part.

Suppose

\[
t_j=t_i+d_e.
\]

A small delay change modifies downstream timing:

\[
\frac{\partial t_j}{\partial d_e}=1.
\]

For a temporal loss

\[
L=L(\{t_k\}),
\]

we therefore have

\[
\frac{\partial L}{\partial d_e}
=
\sum_{k\in\operatorname{desc}(e)}
\frac{\partial L}{\partial t_k}
\frac{\partial t_k}{\partial d_e}.
\]

Exact calculation can require propagating temporal credit through the causal graph.

The proposed approximation is to maintain a **temporal eligibility trace**

\[
E^d_e(t).
\]

For example,

\[
\dot E^d_e
=
-\frac{E^d_e}{\tau_E}
+
x_i(t)
\frac{\partial y_j(t)}
{\partial t_j}.
\]

Then

\[
\Delta d_e
=
-\eta_d
\int M(t)E^d_e(t)\,dt.
\]

The important conceptual change is:

> The network learns not only "how much signal should pass?" but "how early or late should the signal arrive?"

---

# 19. A simpler practical delay-learning rule

For an initial implementation, exact derivatives are unnecessary.

Suppose the target output event occurs at

\[
t^\*,
\]

and the network predicts

\[
\hat t.
\]

Use temporal error

\[
\epsilon=t^\*-\hat t.
\]

For an edge whose activity contributed to the selected output, update

\[
d_e
\leftarrow
d_e-\eta_d\,\epsilon\,E_e^d.
\]

Interpretation:

- if the output arrived too late, useful causal paths become faster;
- if it arrived too early, useful paths become slower.

Clamp delays to

\[
d_{\min}\le d_e\le d_{\max}.
\]

This is extremely easy to test experimentally.

---

# 20. Learning event selection

The network should not learn every possible connection equally.

Introduce an edge activity variable

\[
a_e\in[0,1].
\]

Define an \(L_1\)-like sparsity objective

\[
L_{\text{sparse}}
=
\lambda\sum_e a_e.
\]

The total objective becomes

\[
L
=
L_{\text{task}}
+
\lambda_s L_{\text{sparse}}
+
\lambda_d L_{\text{delay}}
+
\lambda_w L_{\text{weight}}.
\]

An edge can be pruned when

\[
a_e<\epsilon
\]

for sufficiently long.

This creates a learnable sparse causal graph rather than a fixed sparse graph.

---

# 21. Structural plasticity

Weights should not be the only learned variables.

A practical system should allow:

### Synapse creation

Create an edge

\[
i\rightarrow j
\]

when repeated causal correlation suggests that it is useful.

### Synapse strengthening

Increase \(w_{ij}\) when the edge consistently contributes to successful outputs.

### Synapse weakening

Decrease \(w_{ij}\) when it is irrelevant or harmful.

### Synapse deletion

Remove edges whose long-term utility is below a threshold.

### Delay adaptation

Adjust \(d_{ij}\).

Thus learning operates on both:

\[
\text{parameters}
\]

and

\[
\text{topology}.
\]

This is important because the desired architecture is sparse and asynchronous. Dense networks with most connections permanently present defeat much of the hardware motivation.

---

# 22. Credit assignment

The major difficulty is temporal credit assignment.

A final error at \(t=T\) may depend on events occurring much earlier:

\[
e_1\rightarrow e_2\rightarrow\cdots\rightarrow e_n\rightarrow y.
\]

Backpropagation through time stores or reconstructs the entire trajectory.

The proposed alternative is to let each edge maintain a decaying local memory:

\[
E_e(t)
=
\int_{-\infty}^{t}
K(t-s)\,
C_e(s)\,ds,
\]

where \(C_e(s)\) is a local causal coincidence signal.

Then the delayed teaching signal simply gates this memory:

\[
\Delta\theta_e
=
\eta M(t)E_e(t).
\]

This is the central mechanism by which an asynchronous machine can learn without globally replaying every event.

---

# 23. Eligibility as a causal derivative

A useful interpretation is:

\[
E_e(t)
\approx
\frac{\partial S(t)}{\partial\theta_e}.
\]

The trace therefore approximates the sensitivity of the current state to the local parameter.

The global signal

\[
M(t)
\]

then converts local sensitivity into a useful update.

This gives a bridge between biologically local plasticity and gradient-based optimization.

The exact equivalence need not hold. A valuable research question is:

> How accurately can local event-driven eligibility approximate the true gradient while retaining sparse asynchronous execution?

That can be measured directly.

---

# 24. Learning through prediction rather than labels

Supervised labels are not required.

A particularly natural objective for Sleeping Machines is **next-event prediction**.

Given history

\[
\mathcal H_t,
\]

the network predicts

\[
P(e,\Delta t\mid\mathcal H_t).
\]

Train using temporal negative log likelihood:

\[
L
=
-\log P(e^\*\mid\mathcal H_t)
-
\log p(\Delta t^\*\mid e^\*,\mathcal H_t).
\]

This turns the machine into a temporal generative model.

The network learns:

- what happens next,
- when it happens,
- and which causal path is responsible.

This is potentially much closer to the natural objective of a temporal causal machine than reconstructing arbitrary dense tensors.

---

# 25. Hazard-based predictive learning

Let each candidate event \(i\) have hazard

\[
\lambda_i(t).
\]

The total hazard is

\[
\Lambda(t)=\sum_i\lambda_i(t).
\]

The probability that no event occurs before \(t\) is

\[
S(t)
=
\exp\left(
-\int_0^t\Lambda(u)\,du
\right).
\]

If event \(i\) occurs at time \(t_i\), the point-process likelihood is

\[
\log p(i,t_i)
=
\log\lambda_i(t_i)
-
\int_0^{t_i}\Lambda(u)\,du.
\]

Therefore

\[
L
=
-\log\lambda_i(t_i)
+
\int_0^{t_i}\Lambda(u)\,du.
\]

This is an especially attractive learning objective because it directly trains the quantities that the machine already uses operationally.

There is no conceptual mismatch between the forward computation and the loss.

---

# 26. A concrete first learning algorithm

The recommended first implementation is deliberately simple.

## Network

Each neuron has:

\[
(v_i,\theta_i)
\]

and each synapse has

\[
(w_{ij},d_{ij},E^w_{ij},E^d_{ij}).
\]

## Forward event

When neuron \(i\) fires at time \(t\):

\[
v_j
\leftarrow
v_j+w_{ij}x_i
\]

and schedule

\[
e_{ij}
=
(t+d_{ij},j,w_{ij}x_i).
\]

## Local eligibility

At transmission:

\[
E^w_{ij}
\leftarrow
\rho E^w_{ij}
+
x_i\psi_j,
\]

\[
E^d_{ij}
\leftarrow
\rho E^d_{ij}
+
x_i\dot\psi_j.
\]

The exact local terms can initially be simplified.

## Output error

For a target event time \(t^\*\) and prediction \(\hat t\),

\[
M=t^\*-\hat t.
\]

## Update

\[
w_{ij}
\leftarrow
w_{ij}
+
\eta_w M E^w_{ij},
\]

\[
d_{ij}
\leftarrow
d_{ij}
-
\eta_d M E^d_{ij}.
\]

## Sparse update

Only synapses with

\[
|E_{ij}|>\epsilon
\]

are updated.

Only active neurons and pending events are processed.

This is a complete trainable asynchronous model.

---

# 27. Better learning: temporal backpropagation through races

Once the basic system works, replace the heuristic temporal error with differentiable race dynamics.

Suppose candidate paths have arrival times

\[
T_1,\ldots,T_n.
\]

Hard winner:

\[
T_{\min}=\min_iT_i.
\]

Use a soft minimum during training:

\[
T_\beta
=
-\frac1\beta
\log
\sum_i e^{-\beta T_i}.
\]

As

\[
\beta\rightarrow\infty,
\]

\[
T_\beta\rightarrow\min_iT_i.
\]

This gives a differentiable approximation to temporal competition.

The derivative is

\[
\frac{\partial T_\beta}{\partial T_i}
=
\frac{e^{-\beta T_i}}
{\sum_j e^{-\beta T_j}}.
\]

Thus the temporal race itself produces a soft credit-assignment distribution.

This is extremely promising because the machine's computational mechanism supplies its own analogue of attention weights.

---

# 28. Temporal backpropagation without dense matrices

For a path

\[
p=(e_1,e_2,\ldots,e_n),
\]

the arrival time is

\[
T_p=\sum_kd_{e_k}.
\]

Therefore

\[
\frac{\partial T_p}{\partial d_e}=1
\]

for edges on the path.

For the soft minimum,

\[
\frac{\partial T_\beta}{\partial d_e}
=
\sum_{p\ni e}
P_\beta(p),
\]

where

\[
P_\beta(p)
=
\frac{e^{-\beta T_p}}
{\sum_qe^{-\beta T_q}}.
\]

This is a sparse path-credit mechanism.

Instead of multiplying dense Jacobian matrices, the learning signal can propagate only along active causal paths.

---

# 29. Combining local learning and global optimization

A practical architecture should have two learning regimes.

### Fast local plasticity

Run continuously:

\[
\Delta\theta_e
=
\eta M E_e.
\]

This supports online adaptation.

### Slow global optimization

Occasionally sample trajectories and optimize a more accurate objective using:

- surrogate gradients,
- truncated temporal backpropagation,
- evolutionary search,
- or differentiable approximations of the event simulator.

This is analogous to biological systems having fast synaptic dynamics and slower structural/homeostatic adaptation.

It also avoids making the entire system differentiable merely to obtain trainability.

---

# 30. Homeostasis

Pure Hebbian temporal learning can collapse into pathological activity.

Introduce homeostatic terms.

For neuron \(i\), define target firing rate \(r_i^\*\).

Penalize

\[
L_{\text{homeo}}
=
\sum_i(r_i-r_i^\*)^2.
\]

For temporal activity, also regulate event density:

\[
L_{\text{event}}
=
(\bar r-r^\*)^2.
\]

This encourages a useful sparse operating regime.

A particularly interesting possibility is to regulate **temporal occupancy** rather than firing rate:

\[
\rho_i
=
\frac{\text{time neuron }i\text{ is causally relevant}}
{\text{total elapsed time}}.
\]

This may be more natural for Sleeping Machines than conventional firing-rate regularization.

---

# 31. Information-theoretic interpretation

The network can be viewed as compressing computation into sparse temporal events.

Let \(X\) be the input and \(Y\) the desired output.

The network constructs a latent temporal process

\[
Z=\{(t_i,m_i)\}.
\]

Training attempts to maximize useful information

\[
I(Z;Y)
\]

while minimizing computational cost

\[
C(Z).
\]

A conceptual objective is

\[
L
=
L_{\text{task}}
+
\lambda C(Z).
\]

Possible computational costs include:

\[
C(Z)
=
\alpha N_{\text{events}}
+
\beta N_{\text{active synapses}}
+
\gamma T_{\text{latency}}.
\]

This makes energy/sparsity an explicit part of the optimization rather than an afterthought.

---

# 32. Why this could be different from an SNN

The proposal should be considered a generalization of event-driven neural computation rather than simply another name for SNNs.

The distinctive primitives are:

1. **future-event state is explicit**;
2. **latency is a learned parameter**;
3. **event cancellation is computationally fundamental**;
4. **races are first-class operations**;
5. **probability can be represented as event hazard**;
6. **the machine advances from causal event to causal event rather than from clock tick to clock tick**;
7. **learning acts on temporal geometry and topology as well as weights**.

If these properties do not produce empirical advantages, the formalism has failed its purpose.

---

# 33. Biological prediction

The biological hypothesis is therefore more precise than "the brain uses spikes".

The hypothesis is:

> Biological neural circuits may exploit temporal races, delays, coincidence, cancellation, and event sparsity as computational primitives, with synaptic plasticity learning both signal influence and temporal routing.

This predicts that useful computational information should often be found in:

\[
\Delta t
\]

between causally related events, rather than only in firing rates.

It also predicts that disrupting temporal precision can impair computation even when mean firing rates remain approximately preserved.

Experiments should therefore compare models using:

- spike rates,
- spike timing,
- causal event order,
- latency distributions,
- and temporal motifs.

---

# 34. How to recognize temporal computation experimentally

A biological system should be considered evidence for this formalism if the following observations occur.

## 34.1 Temporal scrambling hurts performance

Preserve the spike counts but randomly perturb spike times.

If performance collapses, timing is computationally important.

## 34.2 Causal order matters

Swap two events while preserving their marginal statistics.

If behavior changes according to which arrived first, the system implements temporal logic.

## 34.3 Latency carries learned information

After training, measure whether synaptic delays become task-specific.

## 34.4 Sparse causal paths explain output

Record only events causally connected to an output and ask whether a small fraction of the network explains most of the decision.

## 34.5 Counterfactual events disappear

Test whether early events suppress or cancel alternative future activity.

This would be especially strong evidence for the "collapse of counterfactual futures" hypothesis.

---

# 35. Hardware architecture

A hardware Sleeping Machine should not resemble a GPU.

A plausible implementation consists of:

- asynchronous processing elements,
- local state memory,
- sparse synaptic routing,
- delay lines or timestamp queues,
- local event buffers,
- priority/race resolution,
- and local learning state.

Each processing element sleeps until:

1. an input arrives,
2. its internal state crosses a threshold,
3. a timer expires,
4. or a modulatory event arrives.

The machine consumes energy primarily when events occur.

This is the computational architecture required to make the sparsity claim meaningful.

---

# 36. Software reference implementation

The first software simulator should be an event-driven discrete-event simulation.

Represent an event as:

```python
Event(
    time: float,
    target: int,
    value: float,
    source: int,
    kind: int,
)
```

Maintain:

```python
priority_queue[time] -> events
```

The main loop is conceptually:

```python
while queue:
    t, events = pop_earliest()

    state = decay_state(state, t - last_time)

    for event in events:
        state[event.target] = apply_event(
            state[event.target],
            event
        )

    fired = detect_events(state)

    for neuron in fired:
        schedule_outgoing_events(neuron, t)
```

Learning adds:

```python
update_eligibilities(events, state)
apply_learning_signal(modulatory_signal)
```

This simulator is sufficient to test the mathematics before specialized hardware exists.

---

# 37. Minimal benchmark suite

The research program should begin with problems where temporal computation is genuinely relevant.

### Level 1: temporal primitives

- sorting,
- temporal XOR,
- first-arrival classification,
- sequence matching,
- timeout detection.

### Level 2: sequence learning

- delayed XOR,
- copy/reversal,
- sequence prediction,
- temporal grammar recognition.

### Level 3: event streams

- event-camera classification,
- speech/event audio,
- asynchronous sensor fusion.

### Level 4: difficult long-memory tasks

- associative recall,
- temporal parity,
- long-range dependencies,
- hierarchical sequence prediction.

### Level 5: language

Only after the architecture demonstrates strong temporal computation should it be tested against token prediction.

---

# 38. Required baselines

Every experiment should compare against:

1. dense MLP,
2. RNN/LSTM,
3. GRU,
4. conventional SNN,
5. reservoir computing / LSM,
6. sparse SNN,
7. transformer or efficient transformer where appropriate.

The important metrics are not only accuracy.

Measure:

\[
\begin{aligned}
&\text{accuracy}\\
&\text{latency}\\
&\text{number of events}\\
&\text{active synapses}\\
&\text{memory}\\
&\text{training FLOPs}\\
&\text{inference FLOPs}\\
&\text{energy}\\
&\text{parameter count}.
\end{aligned}
\]

A Sleeping Machine should not be declared successful merely because it obtains comparable accuracy.

It should ideally obtain a better **accuracy / active-computation** ratio.

---

# 39. The strongest possible test

The most interesting experiment is not "can it classify MNIST?"

It is:

> Can a Sleeping Machine learn a task where the optimal computation is naturally sparse, asynchronous, and temporally structured, while using substantially less active computation than a dense model?

Construct a synthetic task where:

- input events arrive asynchronously,
- only a small subset is relevant,
- causal relations occur over widely separated delays,
- irrelevant events are numerous,
- and the output depends on the order and timing of a few events.

Then compare:

\[
\frac{\text{task performance}}
{\text{active events}}
\]

across architectures.

This directly tests the proposed computational advantage.

---

# 40. Possible relationship to transformers

A transformer performs

\[
QK^\top
\]

to determine relationships between all tokens and then performs a weighted aggregation.

A Sleeping Machine could instead construct a sparse temporal interaction graph.

Each candidate relationship produces a potential event:

\[
(i,j)
\mapsto
e_{ij}
\]

with latency

\[
d_{ij}=f(q_i,k_j).
\]

Only events that win temporal races need to become computationally significant.

This suggests a possible route toward **event-sparse attention**:

\[
O(N^2)
\quad\longrightarrow\quad
O(Nk)
\]

where \(k\) is the number of temporally selected interactions.

The key challenge is avoiding the hidden \(O(N^2)\) cost of generating all candidate events. A genuinely useful architecture therefore needs hierarchical or learned routing so that candidate generation is itself sparse.

---

# 41. Temporal routing as learned indexing

This leads to an important architectural primitive.

Instead of asking:

> Which memory address should I read?

the machine asks:

> Which event should arrive next?

A learned temporal router maps state \(s\) to a distribution over delays:

\[
P(d\mid s).
\]

The network therefore performs **computation by routing time**.

This may be analogous to an address calculation in conventional computers, except the "address" is temporal.

---

# 42. A temporal programming language

A practical high-level language could eventually contain primitives such as:

```text
emit X after 10 ms
wait for A
race A against B
cancel C if D arrives
accumulate evidence
fire when confidence > θ
predict next event
delay X according to score
broadcast winner
```

The semantics of each primitive can be defined by the event calculus above.

This would make Sleeping Machines not merely a neural architecture, but a **programming model for asynchronous causal computation**.

---

# 43. Universality

A serious formalism should eventually prove computational universality.

A straightforward route is to show that the machine can implement:

- a finite-state automaton,
- Boolean gates,
- a register,
- a delay element,
- and conditional branching.

For example, Boolean values can be encoded by events:

\[
0=\text{absence},
\qquad
1=\text{event}.
\]

NOT can be implemented by timeout:

\[
\operatorname{NOT}(A)
=
\operatorname{timeout}(T)
\land
\neg A_{[0,T)}.
\]

AND can be represented by coincidence:

\[
A\land B
\iff
|t_A-t_B|<\epsilon.
\]

OR can be represented by a race:

\[
A\lor B
=
\operatorname{first}(A,B).
\]

Once reliable storage and fanout are established, one can construct ordinary Boolean computation.

This would establish that temporal computation is not merely useful for special-purpose problems.

---

# 44. Complexity theory

The more interesting question is whether some problems become substantially cheaper in this model.

For a temporal race over \(N\) candidates, the *physical* system can allow all candidates to evolve in parallel while observing only the first event.

The idealized latency is therefore

\[
T=O(1)
\]

in the number of candidates, assuming:

- \(N\) physical parallel units,
- bounded propagation delay,
- and a physical race mechanism.

The resource cost remains

\[
O(N)
\]

in physical resources.

Thus the relevant complexity measure becomes multidimensional:

\[
(\text{time},\text{space},\text{events},\text{energy}).
\]

This is analogous to parallel computation: the machine trades physical parallelism for latency.

A useful theory must therefore avoid claiming "constant complexity" without specifying the resource model.

---

# 45. The real research question

The central question is not:

> Is time another way to encode information?

That is already obvious.

The important question is:

> **Can temporal encoding make the *algorithmic control flow itself* sparse and asynchronous in a way that cannot be efficiently reproduced by a dense synchronous architecture without paying substantially more computation?**

If yes, Sleeping Machines could represent a genuinely useful computational paradigm.

If no, it may simply be an alternative mathematical description of SNNs and event-driven dynamical systems.

That distinction should guide the research.

---

# 46. Proposed research roadmap

## Phase 1 — Formal simulator

Implement:

- causal events,
- delays,
- cancellation,
- races,
- node state,
- stochastic latency,
- event logging.

## Phase 2 — Temporal primitives

Demonstrate:

- sorting,
- AND/OR/NOT,
- temporal XOR,
- sequence matching,
- prediction.

## Phase 3 — Learning

Implement:

- STDP-like eligibility,
- three-factor learning,
- delay learning,
- structural plasticity,
- sparse pruning.

## Phase 4 — Differentiable temporal races

Implement:

\[
\operatorname{softmin}
\]

and differentiable latency learning.

## Phase 5 — Event-driven sequence models

Compare against:

- LSTM,
- GRU,
- SNN,
- LSM,
- transformer.

## Phase 6 — Event-stream workloads

Use:

- event cameras,
- audio,
- asynchronous sensors,
- real-time control.

## Phase 7 — Hardware

Implement the event engine on:

- FPGA,
- neuromorphic hardware,
- asynchronous digital logic,
- or mixed-signal hardware.

---

# 47. Recommended initial implementation

The most useful first prototype should **not** attempt a full biological simulation.

Build a small event-driven Python/PyTorch-compatible simulator with:

```text
Neuron:
    state
    threshold
    refractory state

Synapse:
    source
    target
    weight
    delay
    eligibility_weight
    eligibility_delay

Event:
    timestamp
    source
    target
    value
```

Use a binary heap for pending events.

Then implement exactly three learning rules:

### Rule A — local STDP

\[
\Delta w
=
\eta K(t_{\text{post}}-t_{\text{pre}}).
\]

### Rule B — three-factor learning

\[
\Delta w
=
\eta M E.
\]

### Rule C — delay learning

\[
\Delta d
=
-\eta_d M E_d.
\]

This will establish whether learning temporal geometry actually works.

---

# 48. A particularly promising architecture

The architecture I would test first is a **Sparse Temporal Reservoir with Learned Delays (STRLD)**.

It contains:

1. a sparse recurrent reservoir;
2. fixed or slowly learned local nonlinear dynamics;
3. learned synaptic weights;
4. learned synaptic delays;
5. event-triggered execution;
6. temporal race / winner mechanisms;
7. local eligibility traces;
8. a small global learning signal.

The initial training strategy should be:

\[
\boxed{
\text{reservoir}
+
\text{delay plasticity}
+
\text{three-factor learning}
+
\text{sparse structural plasticity}
}
\]

rather than full backpropagation.

This gives the hypothesis the strongest possible test because it preserves the properties the formalism is supposed to exploit.

---

# 49. What success would look like

A convincing result would look something like:

\[
\text{accuracy}_{SM}
\approx
\text{accuracy}_{Transformer}
\]

while

\[
N_{\text{events}}
\ll
N_{\text{dense operations}},
\]

and

\[
E_{\text{inference}}
\ll
E_{\text{transformer}}.
\]

Even more interesting would be:

\[
E_{\text{training}}
\ll
E_{\text{backpropagation}}
\]

while retaining competitive sample efficiency.

That would establish a genuinely different computation/learning regime.

---

# 50. What would falsify the hypothesis

The theory should be considered unsuccessful if experiments show that:

1. temporal delays do not improve representation beyond equivalent weights;
2. sparse event execution provides no meaningful computational savings;
3. delay learning is unstable or useless;
4. local eligibility learning cannot scale beyond toy problems;
5. temporal races are easily simulated by dense layers with negligible overhead;
6. event-driven hardware saves little energy after routing and memory costs are included.

These are important outcomes, not failures of the research process.

---

# 51. Summary of the proposed formalism

The entire machine can be summarized as

\[
\boxed{
\mathcal M_t=(S_t,F_t,\Theta)
}
\]

where

\[
F_t=\text{pending causal events},
\]

and

\[
\Theta=
\{(w_e,d_e,\ldots)\}.
\]

The next event is

\[
e^\*
=
\arg\min_{e\in F_t}\tau_e.
\]

The state transition is

\[
(S_t,F_t)
\xrightarrow{e^\*}
(S_{t+\tau^\*},F_{t+\tau^\*}).
\]

Learning maintains

\[
E_e(t)
\approx
\frac{\partial S(t)}{\partial\theta_e},
\]

and updates

\[
\boxed{
\Delta\theta_e
=
\eta M(t)E_e(t).
}
\]

For temporal prediction, use a point-process objective:

\[
\boxed{
L
=
-\log\lambda_{e^\*}(t^\*)
+
\int_0^{t^\*}\Lambda(t)\,dt.
}
\]

For temporal competition, use

\[
\boxed{
T_\beta
=
-\frac1\beta
\log\sum_i e^{-\beta T_i}.
}
\]

For sparsity:

\[
\boxed{
L_{\text{total}}
=
L_{\text{task}}
+
\lambda_sN_{\text{events}}
+
\lambda_gN_{\text{active edges}}.
}
\]

This gives a complete mathematical direction from the original intuition to a trainable asynchronous computational architecture.

---

# 52. Relationship to existing work

This proposal is adjacent to several established areas rather than isolated from them.

Spiking neural networks already exploit event-based computation and temporal dynamics. Surrogate-gradient methods provide one route to training them, while recent work has specifically examined sparse surrogate gradients because ordinary surrogate-gradient training can compromise the sparsity that motivates event-driven systems.

Liquid State Machines and reservoir computing provide another close connection: recurrent temporal dynamics can perform useful computation while substantially reducing the amount of trainable structure. STDP has also been used to adapt reservoirs.

Three-factor learning rules provide the clearest existing conceptual foundation for the proposed eligibility mechanism:

\[
\dot w=F(M,\text{pre},\text{post}).
\]

The proposed extension is to treat delay and causal timing as explicit learned parameters and to make the event field itself part of the machine state.

Recent event-driven learning work also demonstrates that learning can be organized around actual events rather than global time steps, which is an important practical precedent for this direction.

The research question is therefore not whether event-driven neural learning is possible—it is—but whether the more general **temporal-event calculus** proposed here gives useful computational primitives and scaling properties beyond established SNN/LSM formulations.

---

# 53. References and useful starting points

- Tero Keski-Valkama, *Sleeping Machines* (2021), DOI: `10.5281/zenodo.13207423`.
- Wulfram Gerstner et al., work on spiking neural dynamics and temporal coding.
- Emre O. Neftci, Hesham Mostafa, Friedemann Zenke, *Surrogate Gradient Learning in Spiking Neural Networks* (2019).
- Nicolas Frémaux and Wulfram Gerstner, *Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules* (2016).
- Guillaume Bellec et al., work on e-prop and eligibility propagation for recurrent spiking networks.
- Work on Liquid State Machines and STDP-based reservoir learning.
- Recent work on event-driven learning and sparse surrogate-gradient training for SNNs.

---

# 54. Final perspective

The strongest version of the Sleeping Machines hypothesis is therefore not:

> "Neurons compute using time."

It is:

> **Time can serve as an address space, a control-flow mechanism, a representation, a source of competition, and a medium for credit assignment.**

If this is true computationally, then the architecture of a learning machine can be radically different from a sequence of dense matrix multiplications.

Instead of

\[
x_{t+1}=f(Wx_t),
\]

we can think in terms of

\[
\boxed{
\text{events}
\rightarrow
\text{delays}
\rightarrow
\text{causal races}
\rightarrow
\text{state changes}
\rightarrow
\text{new events}.
}
\]

And instead of learning only

\[
W,
\]

we learn

\[
\boxed{
(W,D,G)
}
\]

where:

- \(W\) determines influence,
- \(D\) determines temporal geometry,
- \(G\) determines sparse causal topology.

That is the core research program: **learn the causal temporal graph itself**.

The decisive experiment is whether such a graph can learn useful computation while remaining sparse enough that the physical machine performs only a tiny fraction of the work that a synchronous dense implementation performs.
