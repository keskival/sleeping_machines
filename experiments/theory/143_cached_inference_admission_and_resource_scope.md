# Cached inference: route admission and the complete resource boundary

3 October 2026, 12:30 UTC. Source audit and standard-library contracts only.
No native numerical job, forward pass, fit or profiler launched here.

## Retain the positive integrated evidence

The priority remains the compiled, segment-batched native model with
`--route-credit linear`: computational delays, temporal races, sparse
addressed persistent writes, separate keys/values, deep state and losing-value
credit. This note changes neither the architecture nor its learning rule.
The current curie integrated chain and AWS revision-2 90M comparisons retain
priority; these inference contracts are pending after physical-host admission.

Completed 10M, one-pass, seed-6 results, with the same T256 test boundary:

| Native model | Test bpc T256 | Fitting TFLOPs, whole fit | Fitting MFLOPs/input position | Writes/input position |
|---|---:|---:|---:|---:|
| p32/D4/H2/U2, linear credit | 2.371491 | 7.242661 | .724684 | 8 |
| p32/D4/H2/U4, linear credit | 2.345157 | 11.949789 | 1.195668 | 8 |
| p64/D4/H2/U2, linear credit | 2.183315 | 26.787462 | 2.680290 | 8 |

These are traced/extrapolated unit/special fitting estimates. Each fit used
1,220 updates and 9,994,240 sampled training positions, state reset per segment.
Doubling p32 capacity improves .026334 bpc at unchanged selected writes,
with 1.649917 times total fitting work: useful capacity beyond activity is
supported, and training is not free. The saved one-pass Transformer's T256
test score is 2.426909; the LSTM's is about 2.171. Width plus credit has nearly
closed that latter gap. Single seeds and different dense/native work-estimate
conventions remain explicit; no comparable-quality supremacy claim follows.

Saved references: `results/language_batched/curie_language_batched_10M_`
`{p32d4_pool4,p64d4_pool2}_linear_l64_lr004_cmp_s6_20261003T101000Z.json`
and `curie_language_batched_10M_p32d4_pool2_linear_l64_lr004_cmp_s6_20261003T070000Z.json`.
Write-only `linear_rwn` at U2 was stable but worse than value-only credit
(2.384 versus 2.370 at T128). Preserve that negative result; do not promote it.

## Exact real-arithmetic cache invariant

The entering race reads `r_i = k_i + K_i m_i`, where `m_i` is the stored
memory. Age-dependent transport is applied at the selected write, not at
every losing key read. With fixed parameters and unchanged `m_i`, `r_i`
is unchanged. Initialize `r_i=k_i` at zero state, then refresh only the
selected slot after its actual memory write. Induction gives the same
entering scores, shared-noise winner, winning proposal, timestamps and
next state as computing every proposal, in exact real arithmetic.

This is the useful algebra of `sleeping_machines/sparse_inference.py`.
It retains real timing computation and addressed state; it is not a dense
carrier substitution. Inference no longer needs losing proposal maps for
credit. Training still needs the candidates used by the local-expectation
teacher and charges that computation.

Cache validity has a parameter-version condition. Updating `k_i` or `K_i`
invalidates that slot's cached read even if memory did not change. A query
map or clock bias update changes scores, but does not itself change `r_i`.
Other value/control maps affect the next write. The present evaluator creates
its cache privately from fresh parameters and zero memory on each call, so
it does not reuse stale reads across optimizer updates. It is not yet an
API for persistent caches across calls or online learning.

## Floating-point routes require their own witness

Real-arithmetic equivalence does not imply every floating-point winner
is identical. Gathered versus batched contractions and different addition
association can perturb content or clocks. A near-tie can change identity
and therefore future private memories. Similar output logits cannot prove
identical routes: two distinct candidates can deliver identical values.
The prototype float64 tests check output tolerance on small random fixtures;
their code does not directly count winners or compare the complete state.
Retain those results with that scope, rather than treating a comment about
identical routes as an observed route contract.

For one paired race, let `ell_i=log(T_i)` and
`epsilon_i=abs(ell'_i-ell_i)`. If the reference winner is `w` and every
loser satisfies

`ell_j - ell_w > epsilon_j + epsilon_w`,

then `ell'_j - ell'_w > 0`: the candidate winner must also be `w`.
This is a sufficient, local perturbation certificate, not a global
rounding bound. An uncertified race can still have the same winner.
The numerical helper evaluates measured log margins; it does not provide
interval arithmetic for the logarithms. Exact ties follow the producer's
first-index rule and do not receive a strict-margin certificate.

`cached_inference_contracts.py` observes the actual production `min(-1)`
decisions and producer-return local state without editing either evaluator.
It first checks 12 small synthetic cases: U1/3/4, both float32/float64,
two seeds and unequal episode lengths. If those pass, it checks four cases
on the actual saved trained checkpoint: its original float32 weights and
the same represented weights promoted to float64, two seeds, fixed FIT
spans only. No targets, loss, parameter updates or heldout selection.

Every active race must have matching winner indices; final memory, arrival
stamps, readiness, source context and logits must agree within the declared
precision tolerances. Every final cached read is checked against `k+K m`.
Padding is excluded from active-winner totals. The job also checks caller
RNG, every stored parameter, Python gains, absent gradients and exact output
nesting with/without the observer. Every route disagreement is preserved
with its times and indices; a failed contract writes a new failure result
and exits unsuccessfully, so it cannot admit this backend. Intermediate
payloads are not individually captured. No passing case is a full DEV/test
rescore or a runtime speed measurement.

The source/factory hashes must match the checkpoint producer. Missing actual
weights, changed producers and existing output paths are rejected before
Torch/NumPy import. The benchmark driver has gained new options since these
fits; its current hash is recorded for the small data adapter, not asserted
to be the original training driver. Both actual-weight inference paths here
are eager; original reported language scores used compiled layer steps.

## Arithmetic advantage and costs omitted by FLOPs

Saved same-shape 1,024-character accounting files, 12:00 UTC, p32/D4/H2:

| Evaluator/pool | MFLOPs/evaluated input position | MFLOPs/scored target |
|---|---:|---:|
| Winner-only U2 | .163227 | .304691 |
| Winner-only U4 | .164301 | .306695 |
| Batched emulator U4 | .393573 | .734669 |

The U2-to-U4 winner-only arithmetic increase is about .66%. This supports
the concrete inference mechanism gain. These traces are shape measurements
on freshly initialized models, not rescoring the saved trained checkpoints.
There were 1,792 evaluated input positions and 960 scored targets; overlap
must not be silently dropped when comparing work per scored target. The
appendix uses per-input-position inference for every row, including controls.

The current implementation calls `_stacked(0)` on every invocation. This
stacks **every** unit's key, four P-by-P maps, gate/control biases, controls,
rates and frequencies, plus queries. Copying matrices is not multiply-add
work. Consequently, sparse proposal arithmetic does not make its current
parameter preparation independent of pool size.

For `S=D H U` slots, scalar byte width `b`, even per-head width `P`, the final
stacked tensor payload is

`S [ b(4P²+4P+P/2+3) + 8(P/2) ] + b D H P(H P)` bytes.

Frequencies are converted to float64; other fields use model precision.
Source parameter payload read for this stack is

`b [ S(4P²+5P+3) + D H P(H P) ]` bytes.

Their sum is a minimum tensor-payload read-plus-final-write accounting
boundary, not measured DRAM traffic: intermediate stacks/conversions can
add work and hardware caches affect physical traffic. Unit state and key
cache for `n` lanes additionally occupy

`n S (2 P b + 8 + 1)` bytes.

For p32/D4/H2/U4, float32, eight lanes: final stack **612,736 bytes**,
source parameter payload **610,688 bytes**, unit state/cache **67,840 bytes**.
For p64/D4/H2/U2: **1,333,440**, **1,331,392**, **66,688** bytes respectively.
Raw model tensors, context, input/output, allocator metadata and temporaries
are additional. At p32/U4096 the same formula gives a 560,398,336-byte
final stack before these additions; that is a static count, not an executed
large-pool experiment. No RSS, traffic, energy or speed advantage is proved
by this formula. It identifies what to measure and potentially amortize.

## Admission and next decisions

Pure standard-library checks pass:

```
python3 experiments/check_sparse_inference_accounting.py
python3 experiments/check_cached_inference_observer.py
```

These verify scalar margin/count fixtures and fake-runtime observer recovery
(including inherited method restoration and exceptions), not Torch contracts.
Native contracts are **pending** in two unique one-job manifests:

- `queue/local_cached_inference_p64_contracts_20261003T123000Z.txt`
- `queue/local_cached_inference_pool4_contracts_20261003T123000Z.txt`

Run only on the producer with its actual checkpoint, after reserving physical
curie, through `experiments/queue/run_safe.sh`: one thread, VMS cap
3,000,000 KiB, RSS watchdog 1,250,000 KiB, minimum 8,192 MiB available,
420-second timeout. They do not displace the existing integrated chain.
Neither checkpoint is available in this container. A container-local free
lock does not reserve the physical host; no runtime job was launched.

After successful admission, verify actual-weight full DEV/test parity under
the existing window/seed protocol before associating sparse inference with
the reported quality. Then measure preparation, state traffic and end-to-end
wall time alongside arithmetic before increasing pool size substantially.
Amortizing immutable parameter stacks is a possible implementation step,
requiring explicit parameter-version ownership; it is not installed here.
Broader or indexed discovery would require its own route/coverage contracts,
not an inference from these flat all-key races. The immediate priority stays
the completed language gains, credited width/capacity comparisons, horizon
arms and the provisioned AWS 90M protocol.
