# Online language adaptation and frontier comparisons

## Current evidence

The corrected [E173 language benchmark](e173_causal_language.py) learns expert
mixing weights on validation and freezes them on test. Its context/copy cache
changes with observed text, while learned counts and weights remain frozen.
The E176/E178 neural language screens also freeze weights during evaluation.
Online training loss during fitting is not evidence of test-time adaptation.

[online_language_ablation.py](online_language_ablation.py) supplies an initial
development-only comparison from the saved E173 pilot. It changes only mixing
weights, with identical frozen count experts and copy-cache behavior. Its rate
comes from the checkpoint's earlier validation split; new development positions
are disjoint. Every target is scored before its update. This is a local adaptive
readout experiment, not a learned deep TTT backbone or a frontier benchmark.

## Hypotheses to separate

| Axis | Experiment | Required accounting |
| --- | --- | --- |
| Sequence parallelism | Batched affine scans versus sequential execution | Forward/backward, sorting, buffers and optimizer |
| Persistent state | Streaming memory versus repeated prefix replay | State bytes, cache lifetime and per-character work |
| Sparse retrieval | Indexed bounded candidates versus dense attention | Candidate recall, index maintenance and search |
| Online learning | Frozen weights versus local head, memory and route updates | Score first; count all adaptation work |
| Selective activity | Winner-only value credit versus all-candidate credit | Include loser discovery, replay and gradients |
| Capacity beyond activity | Larger dormant banks at fixed active budget | Total parameters/state versus active work |

Cross these axes through staged ablations. Compare each intervention with its
own frozen/unchanged control before combining it with the others. Preserve
failures. Adding flexibility also adds discovery, maintenance and optimization
costs, which belong in the ledger.

## Causal protocol

At position `t`, predict the next character from the previously observed
prefix; record its log loss; then reveal the character and update the permitted
memory or parameters for subsequent predictions. Fix rates, clipping, forgetting
and adaptation budgets on development, not the scored evaluation stream.

Declare resets at document/stream boundaries, carried optimizer moments and
whether a chunk update learns only from earlier chunks or uses a causally masked
within-chunk rule. Do not score targets after fitting on them. Test future
perturbations, current-target independence and full-stream/chunk equivalence.
Distinguish observations used for state updates from those used for gradients.

Begin with frozen backbone plus adaptive head/memory so parameter updates do not
silently invalidate earlier cached features. Whole-backbone adaptation must
declare whether prior states/KV entries are recomputed, retained under old
weights or reset, and charge the resulting cost.

Report frozen and adaptive scores separately, adaptation curves under domain
change, forgetting on revisited domains, instability and total cost per original
character. A language stream can supply its own next-character learning targets
after observation, so no human labels are needed for these causal updates.

## Existing research and comparison targets

Sequence-parallel fitting does not prevent a Transformer from adapting between
tokens or chunks. Gradients within each fixed-parameter chunk can still be
parallel. Serial adaptation dependencies, optimizer/state traffic and KV cache
consistency affect the cost. They are not an exclusive architectural barrier.

[TTT layers](https://arxiv.org/abs/2407.04620) treat recurrent hidden state as a
learnable model and use self-supervised update rules. They are relevant to local
plastic event memory; the present fixed update ablation is not that meta-learning
implementation. [Titans](https://arxiv.org/abs/2501.00663) also combines attention
with learned test-time memory. Both warrant controls alongside ordinary head
adaptation rather than assuming attention cannot learn online.

As of 2026-09-30, include representative modern efficient architectures:

- An optimized causal Transformer with FlashAttention/SDPA and an adaptive-head
  version, plus the historical LSTM control.
- [Mamba-3](https://arxiv.org/abs/2603.15569) as a current selective-SSM reference.
- [Gated DeltaNet](https://arxiv.org/abs/2412.06464), including an attention hybrid;
  [Qwen3-Next](https://qwen.ai/blog?id=qwen3-next) demonstrates a deployed hybrid
  of gated recurrence and attention.
- TTT/learned-memory controls with the same allowed adaptation data and budget.

Use official implementations, record their revisions, and distinguish matched
small-scale architecture experiments from evaluation of full pretrained models.
Do not infer frontier parity from text8, a single development window or unmatched
published perplexities with different tokenizers/data. Start with the
[parallel training protocol](PARALLEL_TRAINING_PROTOCOL.md) and
[language scaling protocol](LANGUAGE_SCALING_PROTOCOL.md), then scale a repeated
quality/work advantage on a provisioned GPU host. Existing deferred large
Transformer runs retain priority; each host uses one guarded queue job at a time.
