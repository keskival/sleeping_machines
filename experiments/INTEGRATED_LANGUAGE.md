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

The primary **eight-block** receiver configuration has 432 available units,
16 addressed key scores and eight receiver-state updates per character. The
episodic KV variant adds one historical race per block: up to 16 sequential
selections, with one historical value delivered at each nonempty cache. Training
reads admitted losing values for a conserved local surrogate and charges them.
Keys, values, gates, retention and clocks receive learning signals. Credit is
truncated at 16 characters; forward state survives across chunks.

## Two distinct memory organizations

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
| Separate online full-backbone adaptation | 3.191 frozen → 3.096 online bpc | Inherited checkpoint, new 8K development stream, predict before block-delayed updates |

Records live in [parallel_language](results/parallel_language/),
[episodic_language](results/episodic_language/) and
[online_language](results/online_language/). The report visualizes completed
quality/work pairs, including saved LSTM/Transformer controls. These small-data
scores do not establish comparable-quality language superiority.

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
The [eight-block content-index campaign](queue/local_indexed_episodic_depth8_20260930T211500Z.json)
runs a matched 8K receiver/KV pair after the completed 2K pilot passes its
predeclared bounded-data gate (useful receiver depth gain and limited KV regression). All jobs
use unique one-job queues and [run_safe.sh](queue/run_safe.sh). Sources remain
fixed during runs; completed evidence publishes and commits automatically.

The [AWS 10M definition](AWS_INTEGRATED_10M.md) remains a separate committed
six-block receiver comparison, not an episodic-KV result. Longer KV runs need
packed cache storage and measured memory/runtime before promotion. The older
six-block local checkpoint and recovery queue are preserved, without an implicit
restart that displaces the requested eight-block architecture.

Fixed receiver pools, bounded causal delays and approximate candidate discovery
are current constraints. This is sparse event execution with computation through
time; unrestricted overlapping schedules and learned topology remain open.
See [theory §§303–311](theory/47_language_capacity_and_online_learning.md) and
the [integrated construction](theory/46_integrated_sparse_temporal_language.md).
