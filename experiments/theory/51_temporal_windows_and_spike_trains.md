# Useful delay computation, temporal integration and repeated firing

This extends §§173–174, 329 and 330–336; it does not replace the native
architecture or its recorded evidence. Read the retained-mechanism comparison
and the numerical contracts before a new full fit. Global sections 337–341.

## 337. A concrete timing-scale failure and a retained-mechanism repair

For a transported channel y(delta)=exp(A delta)y0, the bounded race delay
is delta=.001+.010 T/(1+T). A common score shift b rescales T by exp(-b).
Consequently |d delta/db|=.010 T/(1+T)^2 <= .0025 and

    ||d y/db|| / ||y|| <= .0025 ||A||

for a normal rotation/decay block. The initialized native frequencies have
magnitude at most pi/2: the full .010 delay span changes phase by at most
.0157 radians. This does not prove the model cannot learn through timing;
selection and external elapsed-time evolution still compute. It does identify
poor calibration for exploiting the endogenous delay itself. Raising all
persistent-state frequencies would also change response to unit-scale observed
gaps, so the proposed repair separates fast event-local clocks from long-lived
content/time state.

`ClockFeatureEventHeads` retains eight native blocks, independent head queries
and keys, source-local admission, incoming-content/memory mixing, hard primary
payload routes, winner-only inference proposals, and the existing local
counterfactual primary-score teacher. No KV bank or dense carrier is added.
It reuses each existing candidate match s_i for R additional clock policies
s_i^(r)=softplus(a_r)s_i+b_r. They sample independent clocks but deliver no
additional candidate values. A local rotating/decaying basis is reset per event
and latched at each arrival:

    u_r = (delta_r-.001)/.010
    phi_r = exp(-softplus(gamma_r)u_r) [cos(omega_r u_r),sin(omega_r u_r)].

A separate per-head learned projection mixes these features into the primary
content. Initial omega_r spans pi/4 to pi across the bounded delay span; the projection
starts at zero. All clocks are awaited. Primary content evolves from its arrival
to max_r delta_r before mixing; heads then join by their existing causal rule.
This is a candidate implementation, not a completed quality or energy result.
Physical oscillator precision, reset, latch, rate-setting and traffic costs
remain to be measured; CPU exp/sin/cos and projections remain charged.

For E events, L blocks, H heads, P candidates and d coordinates/head:

| Quantity | Native parent | R extra clock features |
|---|---:|---:|
| Key/query vector matches | ELHP | ELHP |
| Selected receiver commits | ELH | ELH |
| Training candidate-value proposals | ELHP | ELHP |
| Clock races | ELH | ELH(1+R) |
| Candidate clock-rate settings | ELHP | ELHP(1+R) |

Extra scalar policy transforms cost O(ELHPR). The 2R-to-d readout costs
approximately 4ELHdR FLOPs; feature formation and causal alignment also cost
work. Backward, clipping and Adam are counted. Rate settings are not free simply
because numerical softmax normalization is absent.

The zero-extra-clock model exactly nests parent initialization, realized
forward, random stream and gradients. The same-clock waiting control retains
all extra clocks and joins but omits feature readout. These models need their
own refits; the waiting control agrees with the enabled model initially because
its decoder is zero, not after training.

## 338. Clock time is a useful continuous nonlinear statistic

Independent exponential clocks with rates lambda_i=exp(s_i) have first time
T~Exp(Lambda), Lambda=sum_i lambda_i, and winner probabilities lambda_i/Lambda.
The winner and T are independent for fixed rates. Thus even a common score
shift, which leaves categorical choice unchanged, changes a continuous output.
For U=T/(1+T), 0<u<1:

    P(U<=u) = 1-exp(-Lambda u/(1-u)).

Clock time exposes aggregate evidence strength without numerically evaluating
logsumexp in the physical winner path. Shared matches with different learned
policies generate different temporal statistics, not repeated identical
attention probabilities. Whether these statistics improve prediction must be
measured. For fixed clock noise and an isolated minimum,

    d delta/d s_w = -.010 T/(1+T)^2,
    d delta/d s_loser = 0.

At a clock winner exchange, the minimum time is continuous. Provided integrable
dominating derivatives, a continuous downstream function of time admits the
pathwise expected gradient without a losing-value jump term. If a different
winner selects a different nonzero payload, that continuity argument fails.
`ContentAndClockRaces` therefore keeps the parent local counterfactual teacher
for its primary content race. It does not claim an unbiased arbitrary
full-sequence loss gradient for that teacher.

## 339. Learnable windows can have exact membership gradients

A layer need not cancel all but one event. For a finite set of addressed
arrivals (t_i,v_i), define age a_i=t-t_i and a trainable positive width
H=H_min+softplus(h). One constructive causal kernel is

    K(a;H,beta) = (a/H)^2 (1-a/H)^2 exp(-beta a), 0<a<H,
                 0, otherwise; beta>=0.
    y(t) = sum_i v_i K(t-t_i;H,beta).

Both K and its first derivative vanish at age zero and H. The continuation by
zero is C1 in age, H and beta for H>0. Therefore y is C1 in values, input times,
read times, width and decay for a fixed finite spike set, including window
entry/expiry. The representation is sparse in event support while admitting
ordinary exact gradients; no soft dense attention matrix is needed. A sharp
rectangular window instead jumps when it includes a nonzero value and does
not enjoy this proof. Immediate nonzero input effects can coexist in another
channel, with their own boundary analysis.

Event-efficient implementation: retain five exponential moments

    M_k(t)=sum_active v_i (t-t_i)^k exp(-beta(t-t_i)), k=0,...,4.
    M_k(t+Delta)=exp(-beta Delta) sum_{j=0}^k C(k,j) Delta^(k-j) M_j(t).
    y(t)=M_2/H^2-2M_3/H^3+M_4/H^4.

On arrival add v to M_0. Remove an expired event's moment contribution at the
next addressed arrival or read. Each event enters and leaves once; no updates
occur during silence. With fixed degree four, work is O((E+Q)d), not O(EQd)
for E inputs and Q reads. This bound assumes chronological inputs, an addressed
receiver and constant width during the evaluation segment. Receiver discovery,
sorting, scheduling and projections are separate costs. The reference
`temporal_window.py` uses a deque and autograd, with no event-by-query tensor.
It stores payloads needed for expiry: O(E_H d) live-window storage plus five
moment vectors. It is not a constant-storage full-history memory.

Training cannot change H midstream and reuse moments/expiry decisions as though
nothing changed. Replay the retained window at that weight version or restrict
updates to state-reset boundaries. Generalize different per-head widths with
separate local queues; backward follows event maps or checkpointed recomputation.
A fused fixed-degree kernel would reduce Python/graph overhead. Polynomial
moment cancellation at large ages can need compensated arithmetic, rebasing or
periodic event-triggered reconstruction; tests are float64 and do not establish
arbitrary-horizon float32 stability.

## 340. Repeated spikes: an exact hybrid-system proof, with explicit limits

Let continuous state obey dx/dt=f(x,theta), emit when g(x,theta,t)=0, then reset
x^+=R(x^-,theta,t). Assume C1 flow/guard/reset, isolated transverse crossings
(g_t+g_x f != 0), a finite event horizon, and no simultaneous ordering changes
or spike creation/deletion in a neighborhood. The implicit-function theorem
makes each crossing time differentiable:

    d tau/d theta = -(g_theta+g_x X^-)/(g_t+g_x f^-).

X^- is the state sensitivity evaluated at fixed time immediately before the
nominal crossing. At the event, the fixed-time sensitivity becomes

    X^+ = R_x X^- + R_theta
          + (R_x f^- + R_t - f^+) d tau/d theta.

The timing correction is the saltation term: differentiating only the reset
misses it. Induction over the finite event sequence proves differentiability
of every repeated spike, representation and parameter on that history cell.
Vector-Jacobian products avoid materializing dense saltation matrices. This
is established hybrid/event-gradient mathematics, not an invention of this
project: [EventProp](https://www.nature.com/articles/s41598-021-91786-z) treats
recurrent hard-threshold SNNs; see also the
[saltation review](https://arxiv.org/abs/2306.06862).

A concrete implemented example is dv/dt=I-beta v, threshold theta>0 and reset
zero. On a constant-current interval with I>beta theta:

    first_delay = log((I-beta v0)/(I-beta theta))/beta,
    repeat_period = -log(1-beta theta/I)/beta.

All actual spikes are t_first+j repeat_period up to the next input boundary.
`threshold_spike_train.py` solves this in event time and emits multiple spikes;
it does not iterate over empty time ticks. Nonnegative finite currents, positive
threshold/leak and finite horizon bound event count; an explicit count cap
rejects oversized trains. Gradients are exact within fixed-count regions,
including current, threshold, leak, initial state and segment boundaries.
Its composition with the compact window passes finite-difference contracts.
It is a primitive/reference, not yet a refitted full native language model.

## 341. Hard routing and spike birth still need credit beyond the history cell

Probability-zero ties do not justify differentiating an expected discontinuous
payload loss by pathwise derivatives alone. If a candidate event switches from
absent to present at T(theta,xi)=H(theta), its boundary contribution has form

    E[(L_present-L_absent) delta_Dirac(H-T) (dH/dtheta-dT/dtheta)].

Winner changes use the analogous difference between downstream histories. A
smooth zero-at-boundary temporal kernel eliminates this particular jump for
fixed candidate events; creating a new nonzero upstream spike or changing a
noncommuting downstream reset can still jump. Counterfactual replay estimates
the jump, and a local first-order value teacher approximates it. Keep this
approximation explicit. Replay only nearby alternatives when possible, but
charge candidate discovery and any replay. Near-boundary truncation is biased
unless an estimator accounts for excluded mass.

An alternative exact expected-gradient target is a stochastic marked point
process with smooth positive intensities lambda_j(t;theta). Conditional on a
history held fixed, its log density is

    log p = sum_events log lambda_j(t_event) - sum_j integral_0^H lambda_j(t) dt.
    d E[L]/dtheta = E[d_theta L | fixed history + (L-b) d_theta log p].

An appropriate history-independent baseline b can reduce variance. Under
regularity/integrability this reaches silent and unrealized alternatives through
hazard terms, but estimator variance and the survival-integral cost are real.
This is the fixed-history likelihood-ratio identity; do not add a reparameterized
event-time derivative to it without deriving the joint estimator, which could
double-count credit. A native local eligibility implementation remains a
separate milestone, not implemented by ordinary truncated autograd.

### Required experiment sequence

1. Numerical contracts: exact clock adjoints, conserved primary teacher,
   zero-clock parent nesting, causal joins and actual activity counts; compact
   moment outputs/all gradients versus explicit sums; threshold trains against
   closed forms, partition invariance and finite differences; train/window
   composition. Read-only proofs do not establish fitted quality.
2. Guarded full-eight-block optimizer/accumulation/recovery contracts, then
   small accounting smokes before any delay-feature fit. Frozen parent sources
   and current valuable fit remain unchanged.
3. Same-data native language2K baseline versus two/four-clock full variants,
   then a refitted same-clock waiting control. Charge full optimizer and all
   added clock policies/features. Promote on completed quality/work evidence.
4. Integrate window/train primitives into the native model only after specifying
   state-version replay, optional emission/silence credit and event budgets.
   Compare refitted winner-only, window and bounded-train variants on the same
   data; test clock precision and dormant-capacity behavior before long runs.

The value proposition is richer trainable functions per addressed active event,
not simply more spikes. Unbounded rate coding can spend away sparsity. Neither
these proofs nor pending definitions certify frontier quality or physical energy.

## 342. An information-bearing train, not copies of an early answer

Repeated emission is useful only if it makes a new causal computation visible.
A train that just repeats a payload at fixed offsets from its first spike does
not establish this. Here is a constructive separation for the implemented
threshold train and temporal window.

With reset zero and constant current I>beta theta, the inter-spike interval is

    p(I)=-log(1-beta theta/I)/beta,
    p'(I)=-theta/[I(I-beta theta)],
    p''(I)=theta(2I-beta theta)/[I^2(I-beta theta)^2] > 0.

So an interval is an exactly trainable, strictly curved nonlinear function of
local input strength; it is not an affine delay or duplicate marker. For a
subthreshold initial voltage v, the next-crossing delay has derivative

    d tau/d I = -(theta-v)/[(I-beta v)(I-beta theta)] < 0.

Apply the same prefix current I0 to two streams until time a, after exactly one
spike tau0<a. All observations, first-spike time and any causally formed first
payload are identical. After a, apply different currents I+>I->beta theta.
Their common voltage at a is v_a<theta, but the next spike times satisfy

    tau_plus=a+log((I+-beta v_a)/(I+-beta theta))/beta,
    tau_minus=a+log((I--beta v_a)/(I--beta theta))/beta,
    tau_plus < tau_minus.

Choose a read t with tau_plus<t<tau_minus, before the third plus spike, and a
window width with t-tau_plus<H<t-tau0. The smooth-window train readouts are

    y_plus=K(t-tau_plus;H,beta_window)>0,
    y_minus=0.

No downstream function of the already emitted common first time/payload alone
can distinguish these two streams. The second spike communicates new evidence
that did not exist when the first was sent. Away from the kernel's stationary
points and event-order/count boundaries,

    d y_plus/d I+ = -K_age(t-tau_plus) d tau_plus/d I+ != 0.

Current projection, threshold, leak, width and temporal readout parameters can
all receive ordinary chain-rule credit within this regime. Different kernels,
heads and event-local payload projections can read separate aspects of the
train; counting is only one possible representation.

The numerical contract uses beta=.7, theta=.4, I0=2.3, a=.25, I+=2, I-=.8,
t=.45 and H=.18. First times are exactly equal; only the plus stream emits a
second spike before the read. Its train readout is positive, its minus counterpart
zero, and the late-current gradient passes finite differences. The common
first time's derivative to late current is exactly zero; the second time's is
negative. No fitting is used to manufacture this separation.

This proves a useful information path under the stated causal early-first
interface. It is not an unrestricted expressivity theorem against a one-spike
model allowed to defer its sole message until all future evidence is seen or
encode the entire history into an arbitrary vector. Nor does it establish that
more spikes always help: each realized emission and its learning/scheduling
work must earn its cost. An integrated model should learn content-bearing
bounded trains, with optional continuation and silence supervision, rather
than spend a fixed high firing rate copying biological neurons.

## 343. Reception can match content against a rotating time representation

In the frozen native parent, candidate score is a query dot a key decoded from
raw stored receiver memory. Delivered memory evolves, but that raw key is not
aged before scoring. It would be inaccurate to describe that route as the
full time-evolved key/query operation requested here.

The new fast-clock candidate adds a per-head reception map. Its selected,
content/memory-mixed message v has evolved to the causal clock join. A learned
representation h_r(v) is compared with each locally latched temporal vector
phi_r(delta_r):

    z_r(v,delta_r)=<h_r(v),phi_r(delta_r)>/sqrt(2),
    g_r=sigmoid(z_r),
    v_next=v + W_out concat_r[g_r phi_r]/sqrt(R).

Each two-coordinate temporal pair has independent learned projection rows.
These are content-dependent time gates, beyond a fixed learned projection of
clock time. They retain the primary message, rather than discarding incoming
information. Clock minimum times depend on reused head key/query matches;
reception depends on both those times and the selected content/persistent
memory. The temporal vectors are latched at their own arrivals, then available
at the join; no clock-derived information is used before its clock fires.

For one pair phi(t)=exp(-gamma t)[cos(omega t),sin(omega t)] and fixed h,

    z(t)=exp(-gamma t)(h_cos cos(omega t)+h_sin sin(omega t)),
    d z/d t=exp(-gamma t)[(-gamma h_cos+omega h_sin)cos(omega t)
                           +(-gamma h_sin-omega h_cos)sin(omega t)].

Thus receptive strength changes analytically with arrival phase and envelope;
learning h changes the preferred reception phases for different contents.
All terms are ordinarily differentiable away from the already stated hard
route/count boundaries. This resembles a query/key compatibility operation,
but matches a local temporal vector rather than a bank of past token keys.
It adds approximately another 4ELHdR projection FLOPs plus O(ELHR) scalar
dots, sigmoids and products. These costs remain in both measured CPU and
projected arithmetic ledgers. No free attention or quality claim follows.

Both reception and output projections initially vanish; the output map learns
first, after which the reception map and temporal basis receive credit. Tests
explicitly cover that staging. The waiting control preserves the same clock
hardware and joins while omitting the full temporal reception/readout branch.
The new frozen campaign must compare completed refits before scaling it.

## 344. Arrival coordinates are not a globally ticking execution clock

The CPU simulator stores numerical arrival coordinates on a common axis and
admits globally nondecreasing observed events. This is an implementation and
input-protocol convention, not proof of an asynchronous chip. Native event
tasks supply measured event times; the language adapter assigns successive
tokens successive unit coordinates. These facts must remain explicit.

The native computation uses differences (elapsed ages), causal precedence,
source-local readiness and head joins. With a common offset c,

    input_time -> input_time+c,
    internal_arrival -> internal_arrival+c,
    age, bounded race delay, features, scores and payload predictions unchanged.

Induct over received events and depth: the max operation commutes with adding
c, all stored-channel transports use time differences, and clock features use
event-local delays. Therefore the function has no preferred absolute time
origin. Tests shift the native and new gated-clock models by 100 units, retain
the random stream and verify logits plus shifted arrivals within float64/float32
rounding tolerance. The proof is algebraic; floating subtraction eventually
loses precision for extreme offsets.

Globally clockless hardware can realize this with local delay elements,
receiver-local elapsed-time state and asynchronous arrival/handshake logic.
It still needs stable local time scales, reset/latch circuits and local causal
coordination. Comparisons between independent streams would require a declared
common reference or a synchronization protocol if an operator needs their
relative timing; the current native model keeps their memories independent.
Sensor timestamp synchronization and a global clock driving every neural
operation are different costs. Some older phase-arithmetic constructions use
an explicit periodic reference; do not generalize the native offset symmetry
to abolish that declared requirement. No fabricated-ASIC clockless energy
measurement is implied by using relative times in the simulator.

## 349. Time rotates the direction a message is projected onto

A dot product x^T q is a compatibility score along q. Calling that score an
attention-like primitive is reasonable; a complete attention operation also
needs a relevance-to-delivery/mixture rule. It is not automatically full
historical attention or learned content retrieval.

For one temporal reception pair, h(x)=A x and phi(t) a rotating vector:

    score(x,t)=<A x,phi(t)> = x^T q(t), q(t)=A^T phi(t).

The effective direction in original message space changes with time, in the
learned plane spanned by A's two rows. If they are independent it traces a
rotating ellipse, with decay modulating strength. If rows are collinear or
frequency is zero, that directional benefit collapses; mode count alone does
not certify useful capacity. Multiple independent planes increase the possible
receptive directions without a new full-width projection for every instant.

The implemented content-dependent readout is

    y=x + sum_r C_r phi_r(t_r) sigmoid(x^T A_r^T phi_r(t_r)/sqrt(2))/sqrt(R).

C_r projects each gated temporal pair back into the message. The result composes
several time-selected learned components, rather than merely returning one
scalar projection. A_r/C_r and temporal parameters are learnable; ordinary
chain-rule gradients reach them within the declared history cell. Clock times
also depend on incoming key/query matches, adding content-conditioned time
selection. Separate heads keep their own maps and channels.

The numerical witness maps one plane onto input coordinates 0 and 1. A quarter
rotation changes which coordinate controls reception, changes the composed
message, and passes finite differences. This demonstrates the requested
mechanism, not trained feature relevance or full-cache attention equivalence.
