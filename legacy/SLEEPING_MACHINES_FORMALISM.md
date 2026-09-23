# Sleeping Machines v3 — Event-Oriented Computation and Learning

## Thesis

Sleeping Machines is a computational paradigm in which the **future event field is part of machine state**. Computation proceeds by causal events, temporal races, firing and cancellation. Crucially, cancelled events need not be erased: their predicted times and race margins can remain as local counterfactual state and later receive credit from a delayed teaching event.

The main empirical hypothesis is stronger than event-driven inference:

> **Both computation and learning can be implemented as sparse asynchronous causal event processes, so dynamic work depends primarily on causal activity rather than on the capacity of the dormant computational fabric.**

Training-step count is therefore not the primary metric. The relevant quantities are node activations, active synaptic operations, event routing, race resolution, eligibility updates, and plasticity updates.

## 1. State

At causal time \(t\):

\[
M_t=(S_t,F_t,H_t,\Theta_t).
\]

- \(S_t\): local instantaneous state.
- \(F_t\): pending future events.
- \(H_t\): relevant history, including fired and cancelled events and local eligibility traces.
- \(\Theta_t\): trainable parameters.

The important distinction is

\[
e\notin F_t \;\not\Rightarrow\; e\notin H_t.
\]

A cancelled candidate may leave a compact record

\[
C_e=(\hat t_e,\Delta_e,\xi_e,E_e),
\qquad
\Delta_e=\hat t_e-\hat t_w>0.
\]

## 2. Event semantics

Represent an event as

\[
e=(t_e,m_e,x_e,s_e,d_e,\kappa_e),
\]

with scheduled causal time, mark, payload, source, destination, and causal context.

The lifecycle is

\[
\text{pending}\rightarrow\{\text{fired},\text{cancelled}\}.
\]

A machine advances directly to the next relevant event:

\[
t_{n+1}=\min_{e\in F_{t_n}}t_e.
\]

There is no semantic global timestep. A priority queue is merely one software implementation.

## 3. Temporal races

For \(K\) candidate alternatives with times \(T_1,\ldots,T_K\),

\[
w=\arg\min_kT_k.
\]

A differentiable soft race is

\[
T_\beta=-\frac1\beta\log\sum_k e^{-\beta T_k},
\qquad
p_k=\frac{e^{-\beta T_k}}{\sum_j e^{-\beta T_j}}.
\]

An exponential race gives

\[
T_k\sim\operatorname{Exp}(\lambda_k),
\qquad
P(w=k)=\frac{\lambda_k}{\sum_j\lambda_j}.
\]

These are useful mathematical/reference forms; physical execution can be a hard event race.

## 4. Why the learning experiment must be multiway

With two alternatives,

\[
P(B)=1-P(A),
\]

so reducing A is equivalent, after normalization, to promoting B. That makes a two-way counterfactual experiment unable to distinguish the mechanism we actually care about.

With three or more alternatives,

\[
P(A)+P(B)+P(C)=1.
\]

Reducing A redistributes probability among **both** B and C. It does not specifically promote B. Thus a proper test uses \(K\ge3\) and asks whether a cancelled target route can receive target-specific credit.

## 5. Counterfactual eligibility

When route \(w\) wins, a cancelled route \(k\) retains its margin

\[
\Delta_k=T_k-T_w.
\]

A natural local eligibility is

\[
E_k^c=f(\Delta_k),
\]

for example

\[
E_k^c=e^{-\beta\Delta_k}.
\]

Near misses receive more eligibility than hopeless alternatives.

The important causal sequence is

```text
candidate events
      |
      v
     race
   /     \\
fire    cancel
  |         |
output   counterfactual trace
             |
          ...sleep...
             |
       teaching event
             |
          Δw / Δd
```

## 6. Event-oriented plasticity

A generic three-factor local update is

\[
\Delta\theta_e=\eta M(t)E_e(t).
\]

For delay parameters,

\[
\Delta d_e=\eta_d\,s_e\,M(t)E_e^c(t),
\]

where \(s_e\) advances or retards the route according to the teaching event.

No dense timestep loop is semantically required. A synapse can receive an event, store an eligibility trace, sleep, and wake only when a later relevant event arrives.

## 7. Event-oriented energy/work

A normal CPU is a functional simulator, not a neuromorphic energy measurement. We therefore count dynamic work:

\[
E=c_wN_w+c_sN_s+c_rN_r+c_qN_q+c_mN_m+c_pN_p.
\]

Here the terms count node transitions, active synaptic operations, routing, race/scheduling, state access, and plasticity.

A dense clocked reference repeatedly visits dormant structure, approximately scaling like

\[
E_{dense}\sim T N_{nodes}\bar k
\]

per episode (with the appropriate forward/backward multiplier during training).

The event-oriented hypothesis is instead that

\[
E_{event}\sim N_{causal\ events}+N_{eligible}+N_{plasticity},
\]

which can remain largely independent of dormant capacity.

## 8. Benchmark

The supplied Python program deliberately avoids a sorted A/B sequence. It uses asynchronous streams of marked events with real-valued payloads, distractors, several latent output alternatives, and temporal jitter. Train and test share the same latent route templates.

It measures:

1. **Functional viability** — a temporal race solves the task.
2. **Timing dependence** — timestamp/content permutation ablation.
3. **Clean delay learning** — explicit delays are separated from route influence.
4. **Multiway counterfactual credit** — winner-only, active-only, binary counterfactual, margin counterfactual, and oracle reference.
5. **Delayed teaching** — the local trace survives until a later teaching event.
6. **Dynamic-work scaling** — event-oriented training versus a dense clocked reference as dormant capacity grows.

## 9. What would constitute a meaningful result?

A strong initial result would show:

- the temporal computation works;
- timing contains usable information beyond event counts;
- explicit delay parameters can be learned without a weight/delay confound;
- with \(K\ge3\), cancelled-event eligibility gives target-specific credit unavailable to winner-only learning;
- the credit survives delayed teaching;
- training itself touches only causally active state;
- event-oriented dynamic work scales with activity rather than dormant capacity;
- the result survives reasonable energy-cost sweeps.

This would not prove superiority over transformers or existing SNNs. It would establish the viability of the proposed computational regime and its key scaling hypothesis.

## 10. Core proposition

\[
\boxed{
\text{future alternatives}
\rightarrow\text{temporal competition}
\rightarrow\text{fire/cancel}
\rightarrow\text{counterfactual trace}
\rightarrow\text{delayed local plasticity}
}
\]

The distinctive claim is not simply that spikes can encode time. It is that **causal future alternatives, their competition, their cancellation, and the resulting learning traces can all be first-class asynchronous events**.
