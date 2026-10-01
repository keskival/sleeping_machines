# Integrated sparse temporal language models

The research target is useful learned capacity with selective activity: time
performs computation, hard routes learn through counterfactual credit, and
content-bearing events mix with persistent memory. The current language models
combine these mechanisms without a dense language carrier or statistical expert.
We have explored a small part of this design space; larger-scale quality and
resource advantages remain empirical questions.

## What executes

An observed character's embedding mixes with the previous deep message. At each
event block, a learned query scores keys formed from receiver prototypes and
retained state. Independent exponential clocks select a receiver with exactly
the softmax probabilities of those scores. The winner mixes incoming content
with rotating/decaying memory, updates its state and sends a gated vector and
arrival time. Incoming information is retained, not replaced by a constant unit
vector. No explicit normalizing reduction/division evaluates the choice
probabilities. One sampled value differs from a deterministic weighted sum.

The prior single-head **eight-block** receiver configuration has 432 available
units, 16 addressed key scores and eight receiver-state updates per character.
Its episodic variant adds one historical race per block. These controls remain
preserved. The current full-architecture candidate uses H independent parallel
heads in each of eight blocks: separate receiver pools and Q/K/V/gate matrices,
H historical races, and H timestamped output channels. With payload 32/head,
H=2 gives total width 64; H=4 gives 128. Increasing heads here also increases
width and capacity, so this is not an isolated head-count ablation.

Each channel evolves through learned rotation/decay until the next read time.
The next block reads aligned channels at the latest arrival and applies a
learned cross-channel matrix. Exact simultaneous arrival and averaging heads
are unnecessary. CPU loops emulate these parallel events serially. Dormant
receivers retain their capacity without being updated on every character;
addressed losing proposals still execute during teaching and must be charged.
Keys, values, gates, retention and clocks receive learning signals. Credit is
truncated at 16 characters; forward state survives across chunks. The new
optimizer experiment accumulates detached-segment gradients over 64/128 targets,
then normalizes, clips and updates Adam once. Learning-rate and quality effects
are measured, not assumed equivalent to the old per-16-target optimizer.

## Completed shared-match repeated-arrival pilots

The [repeated-arrival model](../sleeping_machines/repeated_arrival_race_language.py)
keeps independent spatial heads and all sparse receiver mechanisms. Historical
retrieval sets each admitted key's rate once and reads m marks from the merged
Poisson process; only the winning emitter renews, while losing absolute clocks
remain unchanged. These marks share a policy, rather than replacing independent
Q/K/V heads. Earlier message buffers evolve to the last local arrival, then the
messages are averaged/gated into the evolved receiver. Keys are scored once;
inference delivers m values. Training reads all C admitted values once and sums
a conserved local per-arrival teacher in O(Cd+md). This remains a surrogate for
nonlinear route changes. See theory §324 for its derivation and cost boundaries.

m=1 nests the previous model exactly. Seven read-only numerical/operator checks
passed; full-configuration optimizer contracts and smoke fits have also passed in
unique guarded queues. Matched m2/m4 2K pilots use H2/d32/head/depth8, U128,
lr.004, four passes and seed6. The completed m1/m2/m4 results are 3.779/3.820/3.771
development bpc; m4 gives four times the value deliveries for 4.57% extra counted
whole-fitting work. Its .008 bpc gain misses the .02 promotion gate, so no larger
fit ran. The [completed manifest](queue/local_repeated_arrivals_after_heads_20261001T015000Z.json)
preserves the original finite schedule and definitions. Bounded numerical time
encoding is not a demonstrated homogeneous physical Poisson clock or measured
clockless energy advantage. Messages, renewals, teacher reads and emulator
minimum comparisons remain visible in the record.

## Receiver state and historical memory

| Ours | Representation | Candidate discovery |
| --- | --- | --- |
| [Sparse receivers](../sleeping_machines/sparse_race_language.py) | Compressed persistent per-unit state | Observed-character pool, two learned state keys per block |
| [Episodic KV pilot](../sleeping_machines/episodic_race_language.py) | Separate key/value at every observed position and block | Recent matching-character entries plus recent global positions |
| [Content-indexed episodic KV](../sleeping_machines/indexed_episodic_race_language.py) | Separate per-position keys/values, all retained until reset | Three-bit random-hyperplane index of learned queries/keys; own/neighbor buckets, recent and full-history samples |

The first KV index leaves old entries outside its tails inaccessible. The content
index admits samples throughout eligible historical buckets, with at most eight
indexed candidates plus four recent positions. It is an approximate discovery
scheme, not guaranteed full-bank attention. Hashing is an established primitive;
the experiment tests its integration with temporal races and sparse receivers.
Inference reads only winning values. Training reads all admitted counterfactual
values. Cached activations are detached at credit boundaries and retained under
the weights that created them.

## Completed evidence

| Ours: experiment | Completed development result | Scope |
| --- | --- | --- |
| Six blocks / payload 16 / 32K fit | 3.121 bpc; 31.053 whole-fit GFLOPs | Four passes, 8,191 cold development targets, one seed |
| Six → eight blocks / payload 32 / 2K fit | 3.633 → 3.542 bpc | Matched data/passes/seed; capacity and work also increase |
| Character-indexed KV, six / eight blocks | 3.620 / 3.539 bpc | Same 2K protocol; gains over receiver controls are 0.013 / 0.003 bpc |
| Eight-block content-index KV | 3.554 bpc versus receiver 3.542 | Same 2K protocol; no quality gain in this pilot. Full historical buckets are eligible; average 10.07 candidates / one value delivered |
| Eight-block / 8K fit matched pair | Receiver 3.311; content-index KV 3.357 bpc | Same 8K development stream; KV is worse by 0.047 bpc; 40.243 / 50.006 whole-fit GFLOPs |
| Two / four heads, eight blocks, 8K fit | 3.485 / 3.543 bpc; 79.953 / 193.751 whole-fit GFLOPs | U128/lr.004, four passes, same8K dev; both miss the prior indexed control+.10 quality gate |
| Two independent heads / eight blocks / 2K fit | 3.786 bpc; 27.731 whole-fit GFLOPs | 8K development stream, four passes, selected epoch 2; later passes overfit. New state/channel design also differs from older single-head models |
| Two-head accumulated updates / same 2K fit | U64/lr.002: 3.733 bpc / 22.753 whole-fit GFLOPs; U128/lr.004: 3.779 / 19.999 | U128 costs 27.9% less than U16, within .05 bpc of best; LR/warmup vary, one seed |
| Separate online full-backbone adaptation | 3.191 frozen → 3.096 online bpc | Inherited checkpoint, new 8K development stream, predict before block-delayed updates |

Records live in [parallel_language](results/parallel_language/),
[episodic_language](results/episodic_language/) and
[online_language](results/online_language/). The report visualizes completed
quality/work pairs, including saved LSTM/Transformer controls. These small-data
scores do not establish comparable-quality language superiority.

## Forward history and learning credit

Forward state, available retrieval history and gradient support are different.
All historical entries remain stored, but the16-character credit boundary
seals their producer graphs. Later query/route parameters can learn from their
contents while older key/value-producing maps receive no corresponding later
credit. Two read-only contracts verify this distinction and exact fixed-weight
forward equality across detach partitions. This is truncated backpropagation,
not unlimited long-context training or a derivative error.

The [credit campaign](queue/local_language_credit_campaign_20261001T074000Z.json)
follows the repeated-arrival trial. It first diagnoses saved H2/H4 checkpoints,
then guards full 64-credit gradient/update contracts and a129-character smoke.
The matched 2K comparison holds U64/lr.002 and the full architecture fixed;
only credit span changes16→64. The strongest16-credit schedule also gets its
prepared8K comparison. Longer8K,32K,secondseed8K and 131K fitting have explicit
quality/resource gates. See theory §325 for producer gradients and scope.

## Architecture work versus emulation

The KV records carry two ledgers: `work.projected_event_architecture` and
`work.cpu_emulator`. The projection replaces numerical clock exponentiation,
noise/rate division and bounded-delay simulation with physical competition.
Query/key/value and hash projections, admitted key scores, gated messages,
backward, counterfactual credit, clipping and actual Adam remain charged.
Special functions are separate, with a unit-weight total for historical
comparisons. Clock circuitry, rate setting, index/address operations, random
sampling and memory traffic have additional costs; FLOPs do not measure joules.

The character-indexed pilot averages 11.36 candidate keys but reads one value
per inference retrieval query. The explicit normalizing reduction and dense
value aggregation are avoided. This is a same-shortlist module comparison;
quality, index coverage and total resource use still determine its usefulness.

## Run state and next evidence

Read [HANDOFF.md](HANDOFF.md) and inspect live processes before launching work.
The parallel-head, repeated-arrival and longer-credit campaigns finished at
small-data quality gates. Their completed records and unused larger definitions
remain preserved; do not restart them as though they were active.
The current [historical write-credit campaign](queue/local_historical_write_campaign_20261001T152100Z.json)
runs H2/d32/head/depth8, with full and quarter-strength compact producer credit.
It preserves the forward sparse temporal architecture and adds one detached
normalized feature per stored K/V write. See [theory §§326–328](theory/49_historical_write_eligibility.md)
for the factorized key teacher, stale-write scope and 50% K/V tensor-storage cost.
It teaches sealed write maps without reopening their old representation graphs.

Six read-only numerical tests, full eight-block optimizer/recovery contracts and
an alpha1 smoke passed. Exact alpha0 forward/RNG/gradient/Adam nesting allows
reuse of the strongest completed matched 2K parent. The campaign completes an
alpha.25 smoke and both 2K pilots before any 8K promotion: >=.02 bpc pilot gain.
The saved indexed-control+.10 gate then decides 32K; measured data benefit,
memory and time decide 131K. See the same-stem status JSON for actual progress.
Every job uses a unique one-job queue and [run_safe.sh](queue/run_safe.sh), with
one trainer per host, an 8 GiB available-memory floor and RSS watchdog. The
coordinator publishes and commits completed evidence stage by stage. A prepared
larger definition is not evidence that it has run or succeeded.

Append-only tensor slabs and integer-array indices reduce historical cache
metadata and preserve all entries. Independent heads multiply storage. No
memory compression, eviction or changed candidate prior is hidden in packing.
The [scaling theory](theory/48_parallel_heads_and_work_scaling.md) retains
all-key matching in its main comparison and distinguishes logical accesses from
energy. Shared-match multi-policy races and local scalar counterfactual credit
are proposed extensions, not implemented or benchmarked results.

The [AWS 10M definition](AWS_INTEGRATED_10M.md) remains a separate committed
six-block receiver comparison, not an episodic-KV result. Longer KV runs require measured memory/runtime and quality before promotion. The older
six-block local checkpoint and recovery queue are preserved, without an implicit
restart that displaces the requested eight-block architecture.

Fixed receiver pools, bounded causal delays and approximate candidate discovery
are current constraints. This is sparse event execution with computation through
time; unrestricted overlapping schedules and learned topology remain open.
See [theory §§303–311](theory/47_language_capacity_and_online_learning.md) and
the [integrated construction](theory/46_integrated_sparse_temporal_language.md).

## Beyond character streams

The parallel-head candidate currently uses 27 character-specific receiver pools;
it is not an implemented joint sensor/language model. The broader event interface
and temporal/state/credit primitives motivate shared persistent representations
with modality-specific projections. The [event-stream and integration protocol](EVENT_STREAM_ADVANTAGE_PROTOCOL.md)
sets tests where instructions change event routing, events ground language and
both inform actions. Language index and physical elapsed time require distinct
encodings. Held-out cross-modal combinations and single-modality interventions
must establish any integration or compression advantage.
