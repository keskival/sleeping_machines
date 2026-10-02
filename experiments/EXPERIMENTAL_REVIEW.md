# Review of highlighted experimental evidence

30 September 2026. This review checks input/target causality, fitting partitions,
selection, baseline implementation, reproduced scores and work accounting.
It is a code and numerical review, not a certification of every archived run.
The guarded E171 audit reconstructs seven consolidated task screens and the
selected speech model from their saved weights. Original results are retained.

## Findings and claim decisions

| Evidence | Finding | Decision |
| --- | --- | --- |
| E79 native language mixture, including the old 10M/90M headlines | E63's partial-word key resets when the **target** is a space. Holding every observed character fixed while changing the target changes its predicted distribution. | Quarantined outside active results; numerical claims removed. Rebuild counts and rerun. E173 is the corrected replacement, with and without the word expert. |
| E44/E48/E49/E52/E57 historical market world-model comparison | Large-trade threshold is the 99th percentile pooled across all seven pilot days, including evaluation days. Both native and neural references are affected. | Quarantine the held-day comparison and its quality/work headline until rerun with fitting-only preprocessing. |
| E64 language baselines | Causal masks and recurrent states pass suffix-mutation probes. LSTM scores 999,999 targets; historical Transformer scores 999,936; mixture originally scores 1,000,000. | E174 rescores unchanged saved weights on identical targets 1..999,999, including the Transformer tail. |
| E120/E123 common text, market, temporal, MNIST and DVS screens | Reconstructed predictions agree exactly; NLL differences are at most float reduction rounding. Targets never enter packed input. Fitting and development identities are disjoint. | Retain as bounded development results, with different depth, batch sizes, parameters and evidence access stated. |
| E124/E123 consolidated modular arithmetic | All 3,440 unseen decisions and both dense controls reproduce. The learned phase primitive is supplied period 17; no target is supplied at prediction. | Retain the specialized rule-generalization result. It is not generic deep representation learning. |
| E120/E123 consolidated recall | Pointer and neural fitting samples are disjoint from development; saved pointer/carrier and dense-control predictions reproduce. Pointer search has a relative-offset inductive bias. | Retain the synthetic retrieval result; report its fitting budget and search cost. |
| E165 selected speech prefix | 408/512 answers reproduce, with zero fitting/development utterance overlap. Raw event marks precede fixed packet closure times. | Retain private development evidence. Repeated development selection and the reused 657-utterance audit do not establish untouched official-test accuracy. |
| E35/E34/E53/E54 native synthetic temporal models | Source review finds prediction functions use observed times and learned weights; labels are used in teaching/evaluation. Synthetic task generators and fixed evaluation seeds are reused across many settings. Archived learned weights are unavailable for some runs. | Retain exploratory accuracy observations, label reused evaluation populations. No claim of a fresh confirmatory test or exhaustive checkpoint reproduction. |
| Native activity vs Transformer work plots | Event deliveries, thresholded synaptic activity and dense MACs are different quantities. Native NumPy simulations also evaluate candidate arrays, scans and dense products which the event count omits. | Remove 10,000x/100,000x total-computation implications. Keep activity observations distinguished from executed arithmetic and physical energy. |
| E124 training work column | The historical numbers omit backward and optimizer, and estimate dense references without actual batch padding. | Replace the primary comparison with E172 full optimizer-step accounting. Preserve the old partial ledger only as historical scoped evidence. |

## What causality means here

For target x[t], a predictor must be measurable from observations x[:t], fitted
parameters and its prior state. Changing x[t:] cannot change that distribution.
Availability includes preprocessing: a threshold, vocabulary, normalizer or
feature selector fitted on a future evaluation day is outside that information
set even when every recurrent layer is causal.

E171 finds zero prediction changes in its LSTM, causal language Transformer,
Transformer point-process, four-layer race-payload and six-layer modal-state
prefix probes. It finds a decisive change only in the native partial-word
expert. These are finite probes plus a source review, not universal proofs of
all possible parameter states.

The classification Transformer references are bidirectional **inside an already
observed prefix**. That is valid for a completed speech/gesture/image query and
for predicting one next event from an observed context. It does not support an
earlier token-by-token output from the same full prefix. Likewise, pooled speech
classification is a completed-utterance task, not yet a calibrated quiz-race
policy. Its fixed packet closures delay release; no count is backdated to the
first spike. MNIST latency encoding starts with an available static image.
DVS uses annotated gesture starts/ends; this is segmented recognition, not
autonomous discovery of gesture boundaries.

### Market details

The consolidated market adapter fits its size threshold and conditional hazard
memories on 25 August, fits the neural models on 26 August and evaluates on
29 August. Past price changes and trade types form the observed context.
Next-event type, gap bucket and integrated silence exposure enter the loss,
not the input. E171 reconstructs 3.823226 vs 4.207972 nats/event on the 256
development queries. Its fixed evidence alone scores 3.670: this is not a
generic learned-backbone superiority result or a trading-profit result.

Historical threshold pooling is a real protocol error, not proof that the
entire observed performance gap was caused by leakage. Correct reruns are
required to determine its size. Historical market baselines also differ in
day resets, warm-up exclusions and initial-event handling: E49 starts each
day's GRU at zero without ingesting the first event; native semi-Markov/regime
models carry their state across days; E52 excludes the first half-context and
uses a different hazard parameterization. These require alignment before a
strong comparison. Equal-timestamp events also need a declared ordering and
zero-gap policy; several historical implementations clamp gaps to 1 microsecond.

The older E17 trading quantiles use all seven **pilot** days. That is permissible
as preprocessing for subsequent confirmatory days but is not strictly causal
pilot prequential evaluation. Its first-day input-drive calibration similarly
uses the full first pilot day. E42's hindsight teacher is a target, released
after a declared horizon; its median-gap-to-event-count conversion needs a
timestamp-based availability audit before using pilot scores as online evidence.
Trade-tape aggressor prices in E55 are proxies for executable quotes, not order
book execution with latency, queue priority and fill accounting. No trading
profit claim is supported by the present report.

## Corrected language comparison and scope

E173 rebuilds the word counts with an exclusive previous-space boundary and
tests all expert distributions against current/future target mutations. It
fits counts on the first 10M text8 characters; mixing weights use a separate
1M validation stream; both expert arms are frozen on the 1M test stream.
The copy cache stores at most the last 256 characters. It does not fit counts
on test labels. Both scored streams start cold, and target index zero is
excluded so E174 can use exactly the same evaluation positions.

Completed corrected scores: **1.726986 bpc without the word expert**;
**1.719360 bpc with the causal word expert**. The word expert changes this run
by 0.007626 bpc, not the much larger old leaked improvement. The unchanged
LSTM rescore is **1.799344 bpc** and Transformer **1.908275 bpc**, on exactly the
same 999,999 targets and data hash. E173 uses 10M fitting characters plus 1M
validation labels to fit mixing weights; the LSTM/Transformer use different
capacity, training passes and validation-selection budgets. This is a native
predictor comparison, not matched generic architecture supremacy or an energy
advantage. The old 90M result remains withdrawn.

## Tokenization and persistent streaming

Character IDs already are tokens. Text8 has a fixed 27-symbol alphabet; no
held-data vocabulary fitting is needed. Current common-language queries replay
32 characters and assign artificial 10 ms spacing. That is a causal offline
prefix encoding, not an efficient persistent stream or physical speech timing.
E175 checks a new generic persistent event-state path: one consumption per
token, retained modal memories, actual delayed-message queue, no empty ticks,
chunk-preserving state and explicit credit truncation. Its correctness contract
is separate from a trained language benchmark.

Word/subword tokenization may improve finite-model quality and work per byte.
Vocabulary/merge rules must be fitted on training only; tokens are released
after their text is observed. Compare on the same raw-text splits, count all
bytes represented, and include tokenizer work and buffering latency. Bpc,
bits/token, context lengths, optimizer steps and byte presentations must be
reported together. A tokenization change does not by itself repair credit flow
or establish a depth/scaling advantage.

## Reproduction and work boundaries

- Numerical review: `e171/highlight_review_v2_20260930.json`.
- Corrected native language: `e173/causal_language_10m_20260930.json`.
- Aligned saved controls: `e174/aligned_{lstm,tf}_10m_20260930.json`.
- Complete-step arithmetic audit: E172; forward/loss, backward, clipping and Adam.
- Stream contract: E175; no language quality claim until trained.

E172 v2 covers every observed floating operator in all ten measured steps.
Common/Transformer total arithmetic ratios are 1.913 text, 1.914 market,
1.970 composition, 1.437 MNIST and 0.453 gestures. These are four-query batches
with actual padding, not historical whole-run totals. All figures include
backward, clipping and Adam, not a fixed multiplier on forward work.

E175 gives exactly zero chunk-split and future-suffix prediction error in its
contract; all eight value teachers are nonzero. Seven hidden clock teachers are
nonzero; the final clock is unobserved by an untimed, within-deadline readout.
Fifteen token arrivals deliver 120 block events. E176's 128-target pilot safely
reduces development bpc from 5.170 to 5.091 in one pass, with all eight layers
receiving credit. This tiny pilot establishes execution and loss reduction,
not useful language quality or a scaling advantage.

Fused-kernel operation formulas are estimates of mathematical arithmetic.
Unsupported floating operators must remain listed; a stage cannot silently
contribute zero because a profiler lacks a formula. Comparisons, transcendental
functions and memory/control operations are separate. A representative complete
step is not the total historical training budget: preprocessing, expert fitting,
calibration, inherited weights, failed trials and evaluation still cost work.
No report result currently measures whole-system joules or physical memory traffic.
