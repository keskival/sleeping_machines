# Prepacked serving and datacenter admission

3 October2026. Read143–145, HARDWARE_VALUE_PROPOSITION.md and
DATACENTER_VALUE_MILESTONES.md. Native execution remains pending. Only
source/byte/standard-library lifecycle and payload checks ran here.

## Concrete failure and retained computation

The existing sparse evaluator computes only the winner's proposal and caches
each stored-memory key read. But `model._stacked(0)` copies all unit maps on
every invocation, even when weights are unchanged across requests. The p32/
D4/H2/U4 FP32 final stack alone occupies612736bytes. Selected writes stay
eight per evaluated position, and all32 keys are still scored.

A datacenter worker should pay weight assembly at a declared weight-version
boundary, rather than at every request. This is a runtime lifetime repair,
not an architectural substitution or a new estimator. It preserves temporal
races/delays, sparse persistent addressed writes, message content/evolution,
separate keys/values and the trained hard-route construction. No active model,
training core, current sparse evaluator or producer source is changed.

`sleeping_machines/prepacked_sparse_inference.py` introduces an independent
prepared worker. It deep-copies a quiescent model, freezes its parameters and
retains exactly one `_stacked(0)` result. A small model view supplies those
retained tensors to the unchanged `sparse_logits`. Content/source/channel/
transport/readout maps come from the same private model snapshot. Each call
still allocates fresh episode state/key caches and resets them exactly as the
existing evaluator does. This is not a persistent cross-request conversation
API, a GPU kernel, concurrent RNG scheduler or online-learning worker.

Every new weight version builds a new worker. Original model updates after
preparation cannot stale the private snapshot. Source versions are checked
across construction, so ordinary concurrent source mutation rejects preparation.
Use a quiescent boundary: this guard is not a general distributed atomic
snapshot protocol. Before each evaluation, parameter/buffer identities,
versions, shapes, dtype/device, module modes, gains, dimensions and packed
tensor versions are checked. Metadata traversal costs O(parameter objects),
including dormant units, and belongs in a runtime measurement. Deliberate
`.data` writes bypass version semantics and are unsupported; no writable
public model/update API is provided.

Construction locally disables inference_mode so retained tensors have version
counters, while numerical execution retains no-grad semantics. Serving uses
one worker per process: the original evaluator's process-global RNG fork is
preserved, without claiming safety for concurrent threads. Long-lived state,
parameter-version rollout and batch scheduling need separate contracts.

## Exact identities and numerical admission are different

At fixed weights `_stacked` contains the same stored maps, transformed positive
rates, double frequencies and per-layer gain on every call. Reusing that stack
preserves the ideal program and omits its repeated assembly. Actual FP32 route
ties, tensor strides, versions, ownership and instrumentation still require
runtime validation. Synthetic lifecycle checks alone cannot establish them.

`prepacked_sparse_contracts.py` reuses143's actual-min and return-frame observer.
It compares the original batched producer against the prepared sparse worker:
every active winner, logits, final memory/timestamps/readiness/context, cache
invariant, caller RNG, original weights/gains and exact observer nesting.
Synthetic pools1/3/4, both32/64precisions and two seeds form12 gated cases;
actual trained weights then form four cases on fixed FIT spans. Source-helper
wrapping points directly at the underlying no-grad producer body, so state
capture does not accidentally inspect the decorator's frame.

Each worker is constructed inside an outer inference_mode to exercise versioned
preparation. Additional numerical contracts change the original model after
construction and require unchanged worker output, restore original values,
then mutate the private snapshot and require rejection before forward.
No model fit, backward, optimizer, label loss or DEV/test rescore occurs.

The newly available actual checkpoint is the AWS491520-presentation p32/D4/
H2/U4 throughput admission pilot. It has no completed heldout quality and is
not the completed10M checkpoint or an interim90M quality result. The driver
records this distinction explicitly. Passing it would admit those finite
pilot histories, not attach the saved10M2.345 score to this backend.
Actual completed-fit parity and subsequent full rescore remain required.

Full passing admission pays80 forward invocations:64 observed/nested producer
calls plus16 lifecycle calls over eight workers. This is3540 padded/evaluated
lane positions and2600 active positions. Rejected private mutations execute
no forward. Observer/setup costs are paid; FLOPs and actual traffic/energy
are not measured, and whole diagnostic wall is not an isolated speed result.

Unique prepared one-job queue:
`queue/local_prepacked_sparse_contracts_20261003T150000Z.txt`, with frozen
source/parent/checkpoint/command manifest in its `.sources.json` sibling.
Any changed source bundle, arguments, checkpoint, parent or queue rejects
before Torch import. Runtime end checks revalidate source/evidence bytes.
Use onlyrun_safe, oneCPUthread, VMS3000000KiB/RSS1250000KiB watchdog,
minimum8192MiB available,420seconds, after actual physical reservation.
No new job was launched here; a free container-local lock is insufficient.
Do not displace current curie v8/v6/DVS or AWS90M/private recovery fits.

## Charge isolation and amortize only actual reuse

Let R be declared source payload read for one stack, W its final stack writes,
C the complete copied model tensor payload, and n the number of calls served
by this weight version. A restricted logical payload ledger gives

`ordinary repeated stack payload = n (R+W)`,
`prepared snapshot and once-stack payload = 2 C + R + W`.

It charges one source-model read and one private-model write. The positive
payload break-even is `n > 1 + 2 C/(R+W)`. It must be reapplied for each
weight version. These are declared tensor payload boundaries, not physical
DRAM bytes: rate/frequency temporaries, allocations, model/framework objects,
guard traversal, state/cache initialization, selected-map gathers, device
transfers, cache residency and all other computation remain extra.

For the p32/U4 all-FP32/no-buffer shape premise, R610688, W612736 and
C=177019*4=708076bytes. At ten calls the restricted ledger is12,234,240bytes
ordinary versus2,639,576bytes prepared, a9,594,664byte difference. The first
positive count is three calls. This is a shape scenario, not observed traffic,
speed, joules or dollars. Actual worker accounting computes C/W from native
tensors when admitted, rather than assuming every future buffer has FP32 size.

Retained private snapshot plus stack is1,320,812bytes under this premise;
original model and per-request state are additional while retained. A caller
may release its original model after preparation; construction still has a
peak involving both copies. Shape-derived retained payload is not peak RSS.
Shared-map models may still expand repeated matrices into the packed stack;
shared parameter count does not automatically make packing or storage free.

`prepacked_resource_geometry.py` gives this ledger with checked integer inputs
and a strict positive break-even. `check_prepacked_sparse_worker.py` exercises
snapshot isolation/reuse/private mutation/unsupported shapes/changed-source
rejection with a stdlib fake runtime. `check_prepacked_resource_geometry.py`
checks setup, equality, per-version break-even and the declared shape scenario.
All pass here with no NumPy/Torch imports. Native tensor behavior is unrun.

## What would strengthen a datacenter investment case

This repair closes an implementation gap in our own evaluator; competent dense
engines can also prepare weights once. It does not itself establish a competitor
advantage. After native admission, measure ordinary/prepared sparse execution
on identical frozen traces, precision, physical reservation and output mode,
including construction and guard overhead. Separately compare an optimized
dense/control service at declared quality, latency, throughput and residency.

For a system-level lower bound, if a workload needs F floating operations and
B actual transferred bytes on a device with applicable sustained compute R_F
and bandwidth R_B, ideal execution time is at least `max(F/R_F, B/R_B)`.
These must refer to the same workload and resource boundary. Launch, routing,
control, dependencies and queueing add work/time; latency percentiles do not
follow from that lower bound. Never replace measured B with the restricted
assembly-payload ledger or use vendor peak R as an observed throughput result.

The active90M and multi-pass models close quality/scale doubts; shared-map and
seed comparisons address capacity/exposure and reliability. Sparse serving
admission plus measured total resources close deployment doubts. Modern token/
dataset/control evidence under FRONTIER_COMPUTE_PROTOCOL remains necessary
before a general datacenter language claim. Joint event/language and physical
substrate milestones retain the broader vision beyond a narrow text8 result.
See DATACENTER_VALUE_MILESTONES.md for the ordered technical/customer proof
points; neither a speculative valuation nor more untested variants is evidence.
