# Completed native split-event screen: 2 October 2026

All11 pilots and their contracts/accounting smokes completed on AWS. Source
hashes, finite metrics and operator coverage validate against the frozen
[manifest](gym/plans/aws_split_event_20261001T230029Z/manifest.json).
[Machine-readable analysis](gym/plans/aws_split_event_20261001T230029Z/analysis_20261002.json)
preserves paired contrasts. No holdout was read by this screen.

Eight blocks, H2, d8, pool2, seed6;128 fitting queries/pass, four passes,
256 development queries; Adam.003,64-query windows. Whole fitting work
includes producer graphs, losing proposals, backward, normalization, clipping
and optimizer updates. Multiply-add counts as two operations; special
functions count separately and at unit weight in the total below.

| Ours: task/construction | Dev accuracy % | Dev NLL | Parameters | Whole fit GFLOPs | Fit MFLOPs/query |
|---|---:|---:|---:|---:|---:|
| Order S4 private/P0 |54.30|.978726|42,932|.265264|.518|
| Order S4 private/P2 |47.27|1.106010|42,356|.256057|.500|
| Order S4 shared/P0 |70.70|.855592|14,180|.261445|.511|
| Order S4 shared/P2 |56.64|1.036723|13,988|.252370|.493|
| Order S16 private/P0 |44.14|1.119434|157,940|.280463|.548|
| Order S16 private/P2 |44.92|1.214667|155,828|.271079|.529|
| Order S16 shared/P0 |75.39|.832724|14,180|.261816|.511|
| Order S16 shared/P2 |56.25|1.035879|13,988|.252737|.494|
| Paired timing S4 private/P0 observed |95.31|.185005|42,898|.264993|.518|
| Paired timing S4 private/P0 rank |50.00|.703494|42,898|.264996|.518|
| Paired timing S4 private/P2 observed |82.42|.513100|42,322|.255789|.500|

## What improved, and why it matters

**Identifiable elapsed-time learning:** short/long timing populations have
identical marks, addresses and rank order, with opposite labels. Evaluation
couples the race noise within each pair. Every rank-only prediction therefore
gets exactly one member right, giving an exact50% ceiling for any such
receiver. Ours observed-time learns95.3125%=244/256 and loses the advantage
when state is cleared. This closes the information-identifiability gap in the
previous78.125% versus79.6875% unpaired timing comparison. It establishes useful
timing-dependent persistent computation in the integrated model, not superiority
over a time-aware recurrent or Transformer control.

There are32 independent timing population pairs. The paired gain is45.31pp,
exploratory95% cluster-bootstrap interval[42.97,47.66]. Those intervals are
conditional on one selected seed6 checkpoint, development selection and this
task distribution; they do not cover seed uncertainty or screening multiplicity.

**More useful addressed capacity through shared learning:** with16 occupied
sources, shared/P0 learns75.390625%=193/256, versus44.140625%=113/256 private/P0.
NLL improves25.6%, whole-fit work falls6.65%, and parameters fall11.14-fold.
Both retain512 receiver states,16 commits and32 scored keys/event. Shared
weights receive all128 queries/pass; private source rules see8 each. AtS4,
the corresponding gain is16.41pp,70.70% versus54.30%. Data are identical within
each comparison; independent populations are64 atS4 and16 atS16. Paired
exploratory95% gain intervals are[8.20,24.61] and[25.39,37.11]pp respectively.

This is the desired distinction between learning reusable processing and
storing independent stream information. Shared rules ALSO replace private
source embeddings with one common seed. The result identifies a useful combined
construction, not map sharing alone. Source-count comparisons change per-source
exposure and cannot establish a universal capacity-scaling exponent.

## What the protected subspace did and did not solve

P2 remains writable at events and constant during silence; P0 evolves all
coordinates. AtS16/private,64× gap accuracy rises21.09→34.38%; atS16/shared,
20.31→47.27%. But S4/shared stays near chance under long gaps, and protection
reduces ordinary-gap sharing scores70.70→56.64% and75.39→56.25% atS4/S16.
Timing likewise drops95.31→82.42%. Do not promote half-protection as a general
improvement. The original100% native512-query/eight-pass order result remains
valid at a different budget/width.

The protected prefix also removes the faster initialized temporal modes from
the remaining suffix. Retention, dynamic dimension and initial spectrum all
change together. A matched-spectrum control must precede attributing this loss
to protection itself. Order labels are gap-invariant, so large silent-gap
failures remain a real information-retention problem to diagnose.

## Next admission

First replicate the two successful complete mechanisms at the SAME small
budget across seeds6/7/8 and an untouched1024-query synthetic holdout. Reuse
seed6 selected checkpoints without optimizer steps, validating source/data/dev
identity and charging historical fitting work. Fit only seeds7/8 with matched
controls; keep all seeds, not the best one. Use population-pair clustering and
crossed seed/population uncertainty, with correction across the two primary
contrasts. Do not read the new holdout while choosing the protocol.

Then isolate source-seed/map sharing and protected-spectrum effects, and admit
larger integrated fits only from completed replicated evidence. Time-aware
addressed recurrent controls, real-event data, learned discovery, optional
activity and physical energy remain separate requirements for domain supremacy.
