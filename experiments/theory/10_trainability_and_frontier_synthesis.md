# Trainability and the frontier: synthesis and executable contracts

29 September 2026. Read with the [manifesto](../../README.md),
[mathematical program](../MATHEMATICAL_PROGRAM.md), and existing derivations
in §§19/57 (route credit), §§107–114 (payload depth), and §§130–154 (SHD).

## 155. A hybrid event must carry the state that caused it

### Two different kinds of threshold crossing

Write a receiver's continuous state as z and its refractory state as R.
Between messages, dz/dt = Λz. At an arrival a carrying v,

\[
 z(a^+) = z(a^-) + Bv,\qquad
 g(z,R)=\operatorname{Re}(w^\top z)-\theta(1+R).
\]

There are two cases, even with a fixed winner and route:

1. **Flow crossing:** g reaches zero during smooth autonomous evolution.
   If the crossing is transverse, its implicit derivative is
   \(D_\xi T=-(g_zD_\xi z+g_RD_\xi R+g_\xi)/(g_z\dot z+g_R\dot R+g_t)\),
   where numerator derivatives hold time fixed. This requires a nonzero
   denominator and no jump at T.
2. **Arrival crossing:** g(a−)<0≤g(a+) because a message arrives.
   Here T=a on the region where this same arrival triggers the event.
   The payload must be evaluated from the right-limit state z(a+).
   Its derivative contains the direct term B Dv, in addition to
   dependencies through the previous state and arrival time. Dividing by
   the autonomous voltage slope is not the derivative of this crossing.

The event that wins can change under parameter perturbations. That remains
the boundary-credit problem of §§19/57, separate from computing either
fixed-history derivative correctly. Coincident messages also need a declared
ordering or simultaneous-batch jump convention; the two can differ.

### A concrete defect in the current TVLayer

The legacy grid implementation detects firing from
\(Z_k=e^\Lambda Z_{k-1}+X_k\), which includes current arrivals. It reconstructs
the firing state as \(e^{\Lambda s}Z_{k-1}\), excluding X_k, then performs a
clipped implicit-time step. A current arrival can therefore cause a spike
whose vector omits that arrival. Clipping the denominator cannot repair the
missing jump or make a smooth-crossing formula valid at a discontinuity.

**Exact witness.** A single real state starts at zero; B=w=C=1, embedding=0,
threshold=1, decay rate=10⁻⁶/ms. One payload v=2 arrives at 0.5 ms. The legacy
code detects the spike at grid edge 1, emits it at 1.5000002 ms, and returns
y=0 with dy/dv=0. The exact jump semantics emit at 0.5 ms from state 2.
A consistent grid reference emits at 1 ms from
\(2e^{-10^{-6}/2}\), giving GELU payload 1.9544986 and derivative 1.0852314.
The executable witness reproduces both grid results. It proves a missing
payload-credit channel, not a general claim that every legacy spike is wrong
or that this is the only cause of poor recognition.

The opt-in `spike_reconstruction=grid` reference uses the same detection edge
for reset, payload, and timestamp. It preserves payload/delay derivatives
through the delivered impulse X; the discrete firing time has no pathwise
derivative in this reference. It is a coherent clocked diagnostic, not the
final asynchronous event solver. The legacy mode remains available for
matched comparisons and old checkpoints. Changing the default silently
would invalidate those comparisons.

### What the frozen checkpoint says

On 128 held-out-speaker development utterances, the legacy depth-4 control
emits 892/171/36/32 events in L1–L4. Current-bin input jumps accompany
853/81/10/8 of them. Zero prior states occur in 0/26/2/4 emitted events:
those events have no state-dependent legacy payload at all. Refined times
fall outside the detection interval in 793/86/12/13 cases. Accompanying a
jump does not prove the jump alone caused the crossing. These counts identify
where the reconstruction contract needs checking; they do not measure
accuracy gains. Source: `e83_emission_contract_audit.py`,
`results/e83/emission_contract_s6_n128.json`.

**Matched correction pilot.** The four-epoch seed-6 grid reference uses the
same 120 fitting/128 held-out utterances, architecture, optimizer schedule,
and route-credit settings as `bundle_control`. Its L4 support ends at
124/128 (96.9%) versus 2/128 (1.6%); terminal accuracy is 7/128 versus 8/128.
Grid training loss falls 191.46→60.51→25.06→17.64, but held-out prefix NLL
ends at 22.30/128.11, versus 3.02/3.25 in control (uniform is 2.996).
The corrected model keeps propagating events, but produces badly scaled
class evidence. This establishes neither a recognition gain nor a complete
cause of the old failure. Reconstruction also changes timing/reset semantics,
so the experiment does not isolate payload inclusion from those changes.
The run completed in 582 s of model-reported wall time, with observed runner
heartbeats around 0.6 GB RSS and more than 11 GB host memory available.

**Next analytic and implementation obligation.** Implement exact state jumps
at ordered arrivals, distinguish arrival-triggered emissions from transverse
flow crossings, and include the appropriate time/reset derivatives. Compare
an asynchronous solver against a converging grid reference on streams with
both crossing types. Event scheduling must include autonomous crossings and
silence-dependent evidence; processing arrivals alone is insufficient for
oscillatory states. Charge root isolation, scheduled events, cancellation,
and credit storage to the work budget.

## 156. Event support, representation, and useful descent are different gates

The old support analyses remain valid: a strict layer chain cannot transmit
through an empty event set; near-miss weighting cannot recover candidates
outside proposal support. But restoring firing does not establish trainability.
The experiment with 100% deep support and collapsed class predictions already
falsifies support as a sufficient criterion.

For a differentiable fixed-history region, let z_i be class logits,
e_i=softmax(z_i)−onehot(y_i), and J_i=∂z_i/∂θ. Stack examples with loss scaled
consistently into J and e. The parameter gradient is g=Jᵀe and

\[
 \|g\|^2=e^\top K e,\qquad K=JJ^\top.
\]

K is an empirical controllability matrix for the current logit residual:
nonzero events or large singular values in unrelated directions do not imply
that e has a component in its range. In particular, rank loss, identical
payloads, or absent sample paths can leave parts of the classification error
uncontrollable locally. Auxiliary-head gradients can be large while the
deepest-head K is zero. Measure each objective separately.

Within a region where loss is H-smooth, an update −η ĝ obeys

\[
 L(\theta-\eta\hat g)\le L(\theta)
 -\eta\langle g,\hat g\rangle
 +\tfrac12H\eta^2\|\hat g\|^2.
\]

This is a local descent certificate: the signal is alignment with the task
gradient, and the cost is update variance/curvature. A nonzero gradient norm,
large route entropy, or more counterfactual samples is not the certificate.
At event boundaries apply the analogous analysis to the explicitly smoothed
objective or measure paired finite loss changes; do not extend fixed-history
smoothness across an unmodeled jump.

### Why a representation probe is discriminating

Freeze an event representation φ(x), center it, and initialize a linear
softmax probe at uniform logits. Its weight gradient is

\[
 \nabla_W L=\frac1N\sum_i (\mathbf1/C-y_i)\phi_i^\top.
\]

Thus first-step class learning is controlled by label–feature covariance,
not the number of features alone. A convex readout fitted to raw event
counts/timing, then to each hidden layer, separates input information,
representation loss, and readout optimization. Failure of a linear probe
does not prove absence of all class information: it bounds what that readout
can extract. Fit-set recovery and held-out-speaker generalization are
different measurements. Compare both at fixed features and declared regularization.

For present SHD work, prioritize these gates:

1. Correct jump, reset, payload, time, and backward contracts (§155).
2. Recover a small fit set using a deepest-only terminal objective, with
   fixed-path gradient correctness and raw-input/frozen-layer probes.
3. Establish useful depth: compare matched 2/4/8-layer models and remove or
   bypass deep transformations. A shallow head solving the task is insufficient.
4. Establish speaker generalization and causal posterior calibration.
5. Add adaptive stopping, then compare accuracy/coverage/latency/work jointly.

Prefix cross-entropy at exogenous causal query times is already proper
(§130). Uninformative early prefixes legitimately learn the prior. A
terminal-only diagnostic isolates representation capacity; it does not
"repair" a supposedly invalid prefix target. No label-dependent stopping
time or future utterance duration may choose an online query.

## 157. A depth architecture consistent with the manifesto

The architectural target is a **sparse event continuation with learned
transformations**, plus competing optional branches. An incoming event has
one primary continuation that transports a payload through each stage;
stateful transformations update only addressed units. Routes may replace
that continuation or spend a bounded budget on alternatives. Learned holds,
delays, and cancellation remain first-class operations. Retrieval is used
where memory access helps; every layer need not perform attention.

For fixed identities and times, a residual payload map
\(h_{l+1}=h_l+\alpha_l F_l(h_l)\) with
\(\|DF_l\|\le c_l\) and \(\alpha_lc_l<1\) satisfies

\[
 \prod_l(1-\alpha_lc_l)\le\sigma_{\min}(J)
 \le\sigma_{\max}(J)\le\prod_l(1+\alpha_lc_l).
\]

This follows from the singular-value bounds for I+αDF and multiplication.
With bounded \(\sum_l\alpha_lc_l\) and each term bounded away from one,
the lower bound stays positive and the upper bound finite. The full sequence
operator must include persistent-state influence and shared-key fan-out
(§114); bounding only each event's d×d map is insufficient. This restates and
applies the existing depth certificate, not a new global convergence theorem.

The design implication is specific: carry information through a learnable
layer without requiring a fresh threshold crossing to exist first. The
earlier first-layer skip only bypassed layers; it did not enforce serial
payload preservation through each transformation. Residual continuation
must operate on arriving events, never on all silent units every tick.

There is an unavoidable cost: one continuation per event across D stages
costs O(DE). Free branching can instead create (1+b)^D events. Require a
per-layer event budget E_l≤cE_input, with replacement/merging consuming that
budget, and separately count candidates and shadow replay. Feature width,
state precision, and discarded information remain explicit design choices.
Near-identity payload transport addresses extinction and Jacobian collapse;
it does not ensure discovery of task features or good route proposals.

## 158. What a frontier claim still needs

| Gap | Existing foothold | Required bridge |
|---|---|---|
| Shared representation learning | E79 predictive mixture, E61 retrieval, synthetic compositional depth | One learned stack that combines them on real data |
| Hybrid credit correctness | Fixed-path and route-boundary calculus | Correct derivatives at state jumps, resets, scheduling, and cancellations |
| Useful deep optimization | Local fixed-support Jacobian bounds; depth-8 E77 gradient reach | Label-aligned descent under changing support, controlled estimator variance |
| Route discovery | Paired lost-race credit, scalar continuation values | Proposal coverage of useful alternatives at bounded total search/replay cost |
| Anytime decisions | Proper prefix targets and posterior stopping bounds | Calibration on selected stopping prefixes, silence handling, explicit no-answer outcomes |
| Sparse scaling | Cheap native motif/memory primitives | Candidate indexing, learned representation scaling, full execution/communication costs |
| Hardware advantage | Strong counted-work leads | Measured end-to-end training and inference joules at matched quality |

No known impossibility result forbids a better frontier. Expressive inclusion
does not order learnability or efficiency. Conversely, current E83 failure
does not establish an architectural limit: its implemented hybrid semantics
already violate the intended payload contract. The constructive route is to
close these gaps one by one, preserving the manifesto's event-based execution
and exposing the cost of learning the sparse structure.
