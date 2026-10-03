# Compact shared maps with private temporal state

3 October2026. Read THEORY398 and its scope correction,145–147 and the
datacenter milestones. This is an inference implementation for already queued
tied-map integrated models, not a core architecture or learning substitution.

## Concrete failure and retained mechanisms

The existing `_stacked` producer copies input/output/gate/control/key-read
weights for every receiver. When those modules are tied within a head, it
materializes U copies of the same maps. The winner evaluator then gathers
per-lane matrix copies even though all selected receivers use the same head
maps. Prepacking146 avoids repeated construction but still retains those
duplicates. Shared trainable parameter count alone does not repair this layout.

`sleeping_machines/compact_tied_inference.py` prepares one map bank per head.
It retains private keys, clock biases, damping rates, frequencies and every
receiver's memory/timestamp/readiness/key-read cache. Every candidate key is
scored; only the selected receiver is written. Incoming messages mix content
with persistent state and evolve through physical waiting times and transport
rotations. Clock law, noise order, head/depth order and source context remain
the same. Alternative-route credit and its teacher remain in the unchanged
training producer. Silence-aware learning remains a broader research mechanism;
this CPU inference path does not introduce a silence experiment.

No frozen original producer, training model, existing sparse evaluator or147
rescore source is edited. The new worker accepts only actual shared module
objects, not separately trained maps that happen to contain equal values.
Keys/clocks/timescales retain distinct parameter objects. Different gains
within a layer reject because the original stacked producer uses its first
unit's gain. One-source CPU FP32/64, quiescent snapshot, immutable weights and
one worker per process are the current scope. Online updates, concurrent
serving, GPU execution and persistence across requests need separate contracts.

## Ideal-program equivalence and finite precision

At a fixed layer/head, tied maps satisfy `W_(h,u)=W_h` and shared biases
similarly, for every receiver u. Let the stored memory be m_u and its cached
key read `r_u=k_u+K_h m_u`. Then the original race score is

`s_u=clamp(q dot r_u/sqrt(P)+b_u,-12,12)`.

The compact program stores the same k_u,b_u,m_u,r_u, rates and frequencies,
and uses the same q. Thus the same noise yields the same first time and winner
in exact arithmetic. For that winner, replacing a duplicated/gathered W_h by
a broadcast view of W_h changes no mathematical contraction. Its write,
read-cache refresh, emission time, transport and resulting context agree.
Starting from the same zero state, induction over heads/layers/events proves
equality of the ideal numerical program. The construction preserves the
existing tied model's representational capacity; it does not prove that tying
matches every untied model or that every larger state pool learns useful features.

Expanded stride-zero map views preserve the producer's batched contraction
notation while avoiding explicit `[lanes,P,P]` matrix gathers. Torch kernels
can still choose different arithmetic order or internal temporaries. Therefore
this argument does not establish FP32 route identities, runtime traffic or
speed. Native winner/state/cache/logit/RNG admission is required before any
heldout score is assigned to this worker. Failing a race contract is retained
as a numerical limitation, not silently accepted because mean loss is similar.

## Exact tensor layout ledger

Let D,H,U,P be depth, heads, receivers/head and even per-head width, and b be
the FP32/64 scalar byte count. Each shared bank contains

`A = 4P^2 + 3P + 2` scalar entries:

four square maps, the gate bias, two-row control map and its two biases.
Private packed storage per receiver is `P+P/2+1` b-byte entries plus P/2
double frequencies. Queries contain `D H^2 P^2` entries. Therefore

`compact packed bytes = b[D H A + D H U(P+P/2+1) + D H^2 P^2] + 8 D H U(P/2)`.

The ordinary expanded stack exceeds this by exactly

`b D H (U-1) A` bytes.

Shared dense-map storage is now independent of U. Private parameter storage,
state/cache and all-key discovery still grow with U. Actual source snapshot,
allocator/RSS, metadata traversal, input/readout, control operations, rate
transforms, temporaries and physical traffic remain additional. This repair
does not reduce the declared winner-map arithmetic or erase training's losing
proposals, discovery and optimizer work.

For D4/H2/U4/P64 FP32, ordinary packed payload2404736bytes becomes813248bytes:
1591488 fewer duplicate bytes. At64 lanes the private unit memory/cache/
timestamp/readiness payload is1067008bytes in either implementation. There
are still32 available/scored receivers and8 selected writes per position.
These are checked tensor shapes, not measured bytes transferred, latency,
joules, dollars or a competing-engine advantage.

## Contracts and deferred numerical admission

`check_compact_tied_inference.py` executes the actual packer against shape-only
fake tensors for50 combinations of width/pool/precision. It independently
enumerates packed payloads, checks private state/key counts and exercises
sharing rejection, private alias/gain rejection, deepcopy isolation, immutable
snapshots, retained-stack reuse and changed-source detection. Five malformed
frozen manifests reject before Torch/NumPy imports. These are stdlib contracts;
they do not execute a model or establish native numerical parity.

Prepared one-job queue:
`queue/compact_tied_sparse_contracts_20261003T211000Z.txt`, with frozen sibling
`.sources.json`. `compact_tied_inference_contracts.py` compares the original
batched producer with both the expanded prepacked worker and compact worker:
pools1/3/4/8, FP32/64, two seeds, variable lengths13/5/9. The32 pair cases
observe every active winner and final memory/timestamps/readiness/context/cache,
check caller RNG/weights and exact observer nesting, then verify ownership and
reject a changed shared map. Layout accounting is captured after the paired
comparisons; the execution ledger also pays subsequent lifecycle calls.

Planned synthetic work:144 forward calls,5616 padded lane positions,
3888 active positions, no backward or optimizer updates. Byte/layout and
verification work is paid; no total FLOP or physical-resource result is inferred.
The optional actual tied-parent mode requires a new unique source/parent/weight
manifest and actual saved weights; it then adds eight original/promoted FIT pair
cases and two lifecycles. Untied90M weights are not converted and presented as
trained tied quality. There is no completed tied language checkpoint here yet.

Everything remains **PREPARED UNRUN**. No tensor runtime, model forward,
training, profiler, waiter or scheduler was launched from this Docker context.
Use `queue/run_safe.sh` only after actual physical-host reservation, one CPU
thread, VMS3000000KiB/groupRSS1250000KiB/available8192MiB, timeout420s. These
are conservative small-contract ceilings; recheck actual occupancy and memory.
The Docker-local lock does not establish physical curie idleness. Changed source,
arguments, queue, checkpoint or limits require a new tag. Existing147 trained
sparse quality validation and active owner fits retain their own provenance.

## Stronger completed capacity evidence and priorities

The completed untied p32/D4 90M pool2/pool4 comparison uses the same declared
training/evaluation settings except pool size and tag, the same producer hashes,
89997312 presentations and8 selected writes. At T256, test improves
**2.045356 to1.998416**; DEV improves1.962040 to1.915118. Estimated whole fitting
work increases65.219569 to107.606868TF (1.649917x), with16 to32 scored keys
and512 to1024 available value scalars. Test character perplexity falls3.2013%.
All single seed; pool size also changes initialization/race streams. This is
positive useful-capacity evidence, not an iso-FLOP test or a credit-by-pool
factorial. It is separate from the untested compact tied worker.

Source-bound metadata:
`results/diagnostics/native_90m_capacity_comparison_20261003T211500Z.json`;
updated broader inventory `native_completed_budget_audit_20261003T211500Z.json`.
The owner has already integrated the completed pool2 row in the shared report
and rebuilt its PDF; no competing renderer was started here.

Keep the owner90M depth8/width64 chain and the new pool8 capacity and six-pass
tied/baseline queues prioritized through their existing admission protocols.
Curie p96 six-pass, tied pools, seeds and horizon controls remain owner work.
After native parity and actual tied quality, measure complete serving resources
at identical precision/inputs/seeds/quality, including snapshot/layout costs,
metadata and all-key scoring. More useful capacity and lower total resource
use together remain the target; no pending quality or supremacy cell is filled.
