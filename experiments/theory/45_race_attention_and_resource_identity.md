# Race attention, resource identity and the language design gap

Derived 30 September 2026. Historical results remain intact. This note corrects
specific mathematical scope and defines a mechanism test, not a supremacy claim.

## 294. What time removes, and what it does not

For rates lambda_i=exp(s_i), independent exponential clocks give
P(W=i)=lambda_i/Lambda and T~Exp(Lambda), with W independent of T. This is a
softmax sampler without an explicit summation/division circuit in the winner
path. If rates already exist physically, normalization emerges from competition.
Computing content scores, arming candidates, generating randomness, routing
values and delivering learning credit still cost resources. A CPU emulator
executes those operations explicitly; the identity is not an energy measurement.

The six-layer learned-language carrier is not this attention model. Time
transports persistent state by damped rotations; content gates select retention
and writing. All layers and dense maps execute per character. Its bounded-delay
races select a clock for a shared payload, not a key's value. Improved carrier
accuracy establishes trainable temporal content representations, not dormant
capacity or winner-only attention savings. Selective state-space precedent is
acknowledged: https://arxiv.org/abs/2312.00752. The research target remains
time-based computation plus sparse learned paths and counterfactual credit.

## 295. Correct covariance of the historical time-normalized value sum

Sections 102–103 use o=T sum_i lambda_i v_i. Put mu=sum_i p_i v_i and
Z=Lambda T. Then o=Z mu, Z~Exp(1), so exactly

    E[o]=mu; Cov(o)=mu mu^T; Cov(mean_R o)=mu mu^T/R.

The older §102 bound involving sum_i p_i^2 is invalid for this shared clock:
if every v_i=1, variance remains 1 for any number of keys. The expectation and
§103's fixed-input expected Jacobian remain valid. They do not imply an unbiased
gradient of a nonlinear loss on sampled outputs. Pathwise gradients of the
continuous time-sum train its own expected random-network objective. Closeness
to a deterministic Transformer requires concentration and suitable smoothness
throughout the network. Hard winner delivery needs a separate selection teacher.

A clock cutoff does not certify small skipped probability: a high-rate key can
happen to fire late. The retained-mass bounds of §112 apply when omitted mass
is actually bounded. Index quality must be established independently of timing.

## 296. Centering reduces noise without erasing the temporal mechanism

For a noise-independent baseline b, use

    o_b=b+T sum_i lambda_i (v_i-b).

Then E[o_b]=mu and Cov(o_b)=(mu-b)(mu-b)^T. An accurate contextual baseline
can reduce variance without more races. This still sums admitted values and
does not provide winner-only forward communication.

The indexed winner probe centers its training teacher. For each race r let
c_i=v_i-b, a=sum_i lambda_i c_i. The score-output Jacobian teacher is

    J_i^r=lambda_i T_r c_i - 1[W_r=i] T_r a.

Credit sums to zero for every realization and E[J_i]=p_i(v_i-mu). Choosing b
as the candidate mean reduces noise for similar values. Multiplying by a fixed
upstream error preserves this expectation. If the error depends on the sampled
winner, this is a surrogate, not an unbiased expected-loss gradient. Actual value
gradients follow delivered winners. Losing-value reads forming a are charged.

## 297. Test the missing mechanism against identical retrieval

The new probe retains the carrier and adds separate query/key/value maps with
16-dimensional value messages. The last observed character selects an inverted
index bucket with at most eight recent prefix representations and observed
successor values. A successor is inserted only when it becomes an input; unknown
prediction targets are never stored. Integer relative ages give learned recency
without absolute sub-token clock cancellation. Four races each deliver a value.

The control has the identical carrier, index, payload, initialization, training
characters and passes, but uses softmax over the shortlist. The completed
carrier-only screen remains a third comparison. This isolates race versus dense
shortlist aggregation and retrieval versus none. It does not establish learned
routing, full asynchronous scheduling or dormant-layer growth. All carrier
layers execute. Four deliveries are not cheaper than every possible shortlist;
actual candidate/delivery counts must accompany the quality result.

Candidate finding uses the supplied character index, rather than scanning all
keys and charging only winners. Training exposes counterfactual reads. CPU
stacking/gathers and RNG remain additional emulator costs. Representative
full-step arithmetic and a separately labelled synthetic saturated-index
scenario accompany whole-stream candidate/delivery counts. No physical joules
or global supremacy follows from an isolated mechanism test.

## 298. Compare resource boundaries before interpreting ratios

Historical §63 compares synaptic deliveries with binned-convolution MACs.
Sparse-stream extensions and cross-chip per-operation energy numbers are
scenarios, not measured whole-model energy ratios. The note itself explains
that event-driven digital convolution narrows the gap dramatically. Silence
avoidance and errors-only updates are real mechanisms; their strongest factors
depend on workload and implementation, not on every language model inheriting them.

The recent breadth table compares eight width-32 event layers with two width-32
Transformer layers, including losing-candidate learning and calibration. It
does not say the larger learned-language model uses more arithmetic than the
saved width-256, four-layer language Transformer, whose 10M/four-pass estimate
is 888.78T. Runtime comparisons need matched data, optimization, evaluation,
implementation and hardware. Old language records lack sufficient hardware
metadata for a certified cross-run speedup. FLOPs do not determine seconds or
joules; IO-aware exact attention is an established example of the distinction:
https://arxiv.org/abs/2205.14135.
