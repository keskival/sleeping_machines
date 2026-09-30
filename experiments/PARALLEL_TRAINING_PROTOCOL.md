# Parallel sequence training with sparse event inference

The current affine event memory supports parallel sequence training. That does
not establish that this implementation contains arbitrary LSTMs or softmax
Transformers as exact special cases, or that its sparse execution is faster.
Those are separate representation and hardware questions.

## What already parallelizes

For each receiver and memory timescale, the incoming-state recurrence is
`s_i = a_i * s_(i-1) + z_i`. Once a layer's incoming payloads, times and receiver
keys are known, its coefficients are known before evaluating the recurrence.
Affine transitions compose associatively:

```text
(a2, z2) after (a1, z1) = (a2 * a1, z2 + a2 * z1)
```

[event_memory.py](../sleeping_machines/event_memory.py) implements a segmented
scan with linear combine work and logarithmic dependency depth. Receiver resets
use zero decay. The router and value maps then operate over event arrays, and
learned outgoing delays determine arrival order in the next layer. Layers still
depend on previous layers. Sorting, scan buffers, losing alternatives and
autograd all incur work; the present PyTorch reference is not a fused GPU kernel.

An ordinary LSTM's gates depend on its preceding hidden state, so its full
nonlinear recurrence does not become this simple scan merely by rewriting its
cell update. Likewise, an arbitrary query-specific softmax distribution over
all earlier keys is not exactly a fixed-width affine memory. A general event
framework could implement both computations, while retaining their costs.

Parallel training and persistent recurrent inference already coexist in
[linear attention](https://proceedings.mlr.press/v119/katharopoulos20a.html)
and selective state-space models such as
[Mamba](https://arxiv.org/abs/2312.00752). The original
[Transformer paper](https://arxiv.org/abs/1706.03762) also distinguishes sequence
parallelism from arithmetic complexity and dependency path length. These are
relevant controls, rather than evidence of an event-model advantage.

## Architecture comparison

| Arm | Training computation | Persistent inference |
| --- | --- | --- |
| Current event carrier | Segmented affine scans, races and candidate maps | Sparse event updates and learned delays |
| Input-gated event scan | Input-dependent diagonal decay/injection, parallel scan | Retained state per active receiver |
| Event scan plus bounded retrieval | Scan plus selected key/value retrieval | Persistent state and indexed event memory |
| Dense Transformer control | Optimized causal attention and feed-forward maps | KV cache |
| LSTM and selective-SSM controls | Recurrent LSTM and scan-based SSM | Retained recurrent state |

The input-gated arm must compute its transition coefficients from layer inputs,
not the unknown state being scanned. Nonlinear recurrent feedback is a distinct
sequential arm. Retrieval candidates must be discovered without hidden all-pairs
scoring; index construction and maintenance count toward training and inference.

## Gates before an architecture claim

1. Compare scan and sequential execution on identical inputs and weights:
   states, outputs and vector-Jacobian products for inputs, decay and parameters.
   Cover receiver boundaries, tied arrivals, extreme gaps and reordered events.
2. Verify future-token perturbations cannot change earlier predictions. Compare
   full-sequence training with chunked persistent execution and recurrent decode
   under the same tie policy. An autoregressive decoder still waits for each
   generated token even when fitting known sequences is parallel.
3. Benchmark forward **and backward**, with optimizer updates, across increasing
   event counts and context lengths. Count sorting, scan materialization,
   alternatives and retained state. Warm kernels; measure wall time and peak
   memory on the same hardware. FLOPs alone cannot predict throughput.
4. Train learned language arms on identical causal targets and data budgets.
   Freeze development-based choices before final evaluation. Compare both fixed
   parameter/data budgets and the work required to reach a fixed held-out loss;
   include any evidence fitting, inherited checkpoints and search cost.
5. Repeat promising arms over several seeds. Report total training arithmetic,
   time, memory and available energy measurements beside quality. A sparse
   forward path alone does not establish cheaper training or superiority.

Use the [language scaling protocol](LANGUAGE_SCALING_PROTOCOL.md), a provisioned
GPU host, and [AGENTS.md](../AGENTS.md). Prioritize the deferred large Transformer
comparisons and keep one guarded job active per host. This document proposes the
comparison; these new architecture arms have not been trained or benchmarked.
