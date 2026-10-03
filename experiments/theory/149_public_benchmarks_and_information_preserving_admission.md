# Public benchmark selection and native speech admission

3 October 2026. This is an experimental decision, not a completed benchmark win.
Read together with 139 (physical-host admission), 140–145 (route/write credit),
147–148 (trained sparse inference) and PUBLIC_BENCHMARK_CAMPAIGN.md.

## Failure, intervention and retained mechanisms

The language-linear value-credit arm has completed positive learning and
capacity results. The real-event native gesture arm still trails its own strong
controls, while historical SHD development scores use a different carrier
family. Four 250-ms, spatially compressed gesture packets can discard information
before the learned event core receives it. Neither a poor adapter nor a poor
restricted credit estimator identifies an expressivity impossibility.

The new SHD admission uses the existing AddressedEventHeads/fast native model
and batched_episodes with linear alternative-value credit, unchanged. It adds
a **task adapter**, not a replacement temporal encoder: each nonempty causal
packet carries all 700 channel counts and channel-specific first time moments.
A fixed deadline query supplies completed-utterance supervision. Deep temporal
state evolution, learned bounded delays/races, private sparse commits, separate
keys/values and alternative value credit remain. All keys/proposals and the
1401-component input projection remain paid. Source address is one shared
utterance stream, not 700 disconnected source-specific memories.

Counts/moments are input features, not a fitted n-gram/count predictor. They are
also not a lossless raw-spike representation. The adapter preserves channel
identity and aggregate timing; it loses higher within-packet order/moments.
No claim about raw-event expressivity or an energy advantage follows from it.
The next comparison changes packet resolution before width/depth, and compares
competent models on identical packets as well as published raw-event references.

## A useful decomposition of the quality gap

Let E be the full observed event stream, A(E) its deterministic causal adapter,
Y its target, and q the learned predictor. Under log loss:

    E[-log q(Y|A(E))] - H(Y|E)
      = I(Y;E | A(E)) + E[KL(p(Y|A(E)) || q(Y|A(E)))].

The first term is lost task information; the second includes capacity and
optimization error. More depth or route credit cannot recover information
discarded by A. Conversely, improving A alone does not remove the second term.
Measure both with resolution controls and integrated learning; do not attribute
the whole gap to clipping or the route estimator from a single weak fit.

For a packet [a,b), channel j, store n_j and
m_j=sum((t-a)/(b-a))/n_j, with m_j=0 for empty channels. Emit at b;
the message uses log(1+n_j) and m_j-.5 for occupied channels, zero otherwise.
Every spike belongs to exactly one half-open packet, and all source channels
remain distinct. No future packet, target, speaker, utterance ID or whole-stream
normalization enters the content. A fixed 2-s deadline admits spikes in [0,2);
out-of-range spikes cause an error rather than silent cropping. Raw seconds are
floored to microseconds (less than 1us quantization). Time is expressed
in packet units so the existing bounded race delays are shorter than one packet
at the admitted D2/D4 depths. Coincident final packet/query times use declared
input order; the core retains its normal causal queueing semantics.

Resolution is frozen per arm. Raw observations and logical event count must both
be reported; skipping empty packets avoids periodic deep work, but parsing all
spikes, dense feature construction and all-candidate score/value work remain.
Sparse winner-only inference is a separate contracted backend milestone.

## Choice of benchmarks and work allocation

Keep current 90M language/seed/tied-map owner chains and trained inference queues.
Do not duplicate or interrupt them. Native SHD numerical contracts, one-update
smoke and a four-pass 64-utterance fit are the next *new* event admission stages.
This staged pilot only checks learning and complete work; its development NLL
is not an official score. Compare a selected larger fit with EventSSM and S7,
not only dense controls. DVS refinement follows its existing credit chain and
must first beat the current 77.6% application control on the same development
protocol. SSC is contingent on SHD confirmation, not a parallel broad sweep.

Target a Pareto improvement, not merely an accuracy number: predeclare quality
noninferiority, resource boundary and minimum resource reduction. Published
accuracy numbers alone contain no common FLOP/latency denominator. A reproduced
reference on common hardware is needed for a serving win; different chips and
logical event arithmetic do not establish joule superiority.

For native bank U, depth D, heads H, message width P and packet count M, the
current learning path includes roughly M*D*H*U*P^2 proposal work and all-key
scoring. Fixed selected commits do not make learning independent of U.
Useful tied-map capacity is a hypothesis to test after learning works. Preserve
the private state/maps distinction and the failed write-credit arms.

## Exact admission and selection boundaries

The new stages read only shd_train.h5. Speakers 3/6 remain development and are
already reused in this project; do not label them untouched confirmation.
Pilot examples are deterministically shuffled independently of the optimizer
seed, without label-dependent selection. A small pilot may miss classes and
cannot support benchmark claims. Larger search uses a bounded, declared DEV
tuning budget and official-test access history, then freezes all settings.
Historical E51 has read official test, so the whole project cannot claim never
to have accessed it. New stages cannot open the official test file.

Numerical admission must verify every parameter gradient for eager/compiled,
forward invariance of linear credit, causal-prefix inputs and an actual next
Adam update after save/recovery. The smoke measures wall/RSS and full-step work.
Pilot admission requires these records and a conservative measured workload
bound; full runs need separate unique queues and measured limits. No job from
this container may auto-start using its unshared /tmp lock.
