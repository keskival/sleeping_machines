# Routing, clocks and memory: measurement admission

3 October 2026, 09:20 UTC. This is a bounded forward diagnostic protocol,
not an architectural replacement or a completed numerical experiment.

## Evidence and the decision

The completed compiled one-pass native language arms retain useful progress:
payload32/head, H2/D8/pool2 achieves test **2.456 bpc**, against the saved
one-pass Transformer **2.427** and LSTM **2.171**. Pool4 at the same width and
depth gives **2.498**, despite more available capacity and unchanged selected
writes. The new D4/pool1 control gives **2.439**, versus D4/pool2 **2.507**;
pool1 removes selection and dormant slot capacity, so it stays a diagnostic
control. These are single-seed recipe comparisons, not isolated proofs of
the cause or comparable-quality superiority. Preserve the integrated v4
message/memory-credit arms and their existing queue.

The completed 50K-DEV diagnostics in §413 already refute the description of
trained routing as near-uniform: mean max pi is .817–.955 at pool2 and
.687–.840 at pool4. Their remaining measurement gaps are concrete:

1. `mean_pi_per_unit` is expected probability mass, not observed winners;
   averaging heads also conceals opposing head preferences.
2. Their argmax policy replaces clock noise with1, changing identity and
   computational delay together.
3. The four-seed mixture is compared with the first seed, without saving the
   other component losses or the actual Jensen gap.
4. All score trajectories/probabilities are retained. A streaming collector
   can bound memory by the lane chunk instead.

Use the versioned sibling `language_routing_measurements.py`, leaving the
completed driver/results and queued v4 producer unchanged. Bind the exact
original83265f batched source to the actual saved pool2/D8 weights, verify
the producer's model-factory hashes, and preserve all source/weight/result
hashes, RNG, Python residual gains and parameters. The sibling rejects
newer batched producers rather than silently interpreting their weights
under this original forward protocol.
The saved parent does not separately hash its precise-rotation dependency.
Verify7237331f temporal transport against both archived producer anchors
07eaaae^/e096ca6^ and the current file; load_batched refuses any change.
This adds explicit dependency provenance rather than silently assuming a
relative import still resolves to the original temporal implementation.

## Observed decisions versus expected routing mass

For race r and candidate i, save both W_ri=1[winner_r=i] and pi_ri. Their
denominators are the same number of actual races; neither is substituted
for the other. Independent Exp(1) candidate noise yields the categorical
law pi_i=lambda_i/sum(lambda), conditional on the entering history. Noise
is shared across lanes and histories depend on earlier draws, so aggregate
winner counts are not IID multinomial samples. Sharp probabilities alone
prove neither predictive slot specialization nor semantic memory use.

The collector immediately merges counts, probability sums, max pi, entropy,
sharp-race counts and clock statistics into separate (depth, head) buckets.
It includes both warming and scored input positions, explicitly labelled;
the first factual seed supplies these statistics. No score trajectory is
retained. All evaluated windows have the same length, so no inactive lane
is counted as an actual race.

## Clock conditioning before optimizer clipping

The original native delay is d(T)=.001+.010*T/(1+T). A common shift c of
the *transmitted post-clamp* scores rescales T(c)=exp(-c)*T and leaves pi
unchanged at a fixed history/noise. Consequently

    abs(dd/dc) = .010*T/(1+T)^2 <= .0025,

with maximum at T=1 and vanishing sensitivity at both fast and slow limits.
Report its mean and fraction below .000025 (one percent of the maximum).
This measures a local scalar clock derivative, not a parameter gradient or
loss sensitivity. The upstream hard clamp and downstream cotangents still
matter. Also count transmitted scores with abs(score)>=12, explicitly as
boundary hits; these do not distinguish equality from raw scores outside
the clamp. This is a measurement repair, not a new smooth-score bridge.

`argmax_preserve_first` replaces only the chosen identity with argmax(score)
while retaining the actual sampled min(noise/rate) in each current history.
`argmax_unit_clocks` instead uses noise=1, so the first time is1/max(rate),
matching the older intervention without a global Tensor monkeypatch.
Changed writes/messages legitimately change future state and rates. The
identity-only policy preserves the first time at its *own* entering history;
it does not preserve the entire factual clock trajectory across policies.

## Probability mixtures and paid work

At each scored target, for seedwise probabilities p_k of the actual symbol,

    -log2(mean_k p_k) <= mean_k[-log2(p_k)].

Accumulate each component's bits and the mixture's bits in double log space.
Report mean-seed bpc minus mixture bpc as the Jensen gap; separately report
mixture minus first-seed bpc, which can have either sign. A scalar fixture
p=(.9,.1) has mixture bpc1, mean-seed bpc1.736965594 and first-seed bpc.152003093:
the mixture loses to the first while satisfying Jensen. This is exactly the
distinction missing from the previous diagnostic, whose original scores
and signed first-seed improvements remain preserved.

The pilot uses DEV text8[90M:90M+2048], T128, stride64, first window scored
whole and subsequent windows on their second half. This gives30 windows,
1984 scored targets per policy,3840 forward input events per policy. Four
sampled passes plus two argmax policies cost **six** production passes:
23040 input events,368640 selected writes and737280 scored keys/computed
candidate values for H2/D8/U2. Count batched callback invocations separately
from per-lane races. Contract, statistic and scoring work is additional;
FLOPs/traffic/energy are unknown, not zero. Weight count is not execution
work. This small span is not a replacement for the completed50K diagnostic
or a new test benchmark.

## Why base memory half-lives do not bound receptive fields

§413's saved-state numbers ln(2)/rate, with median5.8–7.5 and maximum16–27
characters, are valid **at unit forget**. Actual candidate-state evolution
uses

    decay_j = exp(-rate_j * forget(x) * age),
    forget(x) = softplus(control(x))/ln(2).

The effective local half-life is ln(2)/(rate_j*forget(x)), not ln(2)/rate_j.
The positive forget operator has no strictly positive theoretical lower
bound, so base-rate maxima do not impose a finite maximum on the actual
local retention horizon. For repeated selected writes, a fixed pair's
homogeneous old-value contribution has magnitude attenuation
exp(-rate_j*sum_k forget(x_k)*age_k); paired rotations preserve its norm.
New additive writes, gates, mixing and predictive readout further separate
this attenuation from task-relevant context dependence.

There is another information path: the original body computes keys from
the stored m **before** candidate decay, `key + key_read @ m`. Nonselected
slots retain m until a later selected update. Their stored information can
therefore influence key scores and first-time computation even when the
candidate value would be strongly attenuated. This does not make discovery
free: all candidate keys and proposals are still evaluated here.

Thus the observed base-rate shortening is evidence of parameter adaptation,
and interference-driven short retention is a plausible hypothesis; neither
is a measured maximum context bound. T128/T256 similarity also cannot
identify causal history use. The next horizon test needs identical targets
and recent suffixes with controlled earlier histories, actual dynamic
forget/age statistics, and prediction/state changes. It belongs after the
current credit comparison, without replacing its priority or inferring a
mathematical barrier to the complete substrate.

## Status and guarded admission

`check_routing_measurement_summary.py` passes pure standard-library fixtures
for denominators, head separation, missing measurements, Jensen and clock
identities. Import/help inspection does not load NumPy/Torch. Native tensor
contracts are **pending**: collector/logit nesting and descriptor recovery,
constructed race identity/clock interventions, tiny double lane grouping,
then actual saved-state/weights/RNG/source preservation and race counts.
No model-runtime execution or completed result is claimed in this session.

One-job queue `local_language_routing_measurements_20261003T092000Z.txt` is
prepared, not started. The physical curie queue is reserved; the container
process table/lock cannot establish its idleness. Run only after a physical
host reservation, through run_safe, one CPU thread, VMS3000000KiB,
RSS1250000KiB, MemAvailable floor8192MiB and420s timeout. These are bounded
pilot caps informed by the roughly31GiB host and prior small native runs;
this streaming workload's actual peak and duration remain unmeasured.
Retain the existing v4 queue first and138's pending actual-warm-Adam test.
The parent names an untracked final checkpoint on the producer host; that
file is unavailable in this execution context. The diagnostic rejects its
absence before loading the model runtime. Run on the producer with that
actual checkpoint (or transfer the verified file), never reconstruct it
from reported scores. This is an additional reason the local forward
measurement remains pending, independent of physical-host occupancy.
