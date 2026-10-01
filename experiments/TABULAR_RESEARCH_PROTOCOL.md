# Tabular prediction: sparse conditional routing as a hypothesis

The first real-data adapter, guarded smokes and most full AWS pilots have
completed. Banknote native quality leads the small tree screen; wine native
currently loses. See [first-screen findings](FIRST_SCREEN_FINDINGS_20261001.md).
The [banknote confirmation protocol](AWS_BANKNOTE_CONFIRMATION.md) adds repeated
seeds, stronger controls and frozen reserved-test scoring after the prioritized
event battery. The integrated language campaign remains the local priority.

## Why the architecture might help

Hard routes and optional modules could represent irregular, conditional feature
interactions while executing a small part of a larger model. Counterfactual
credit can teach unselected routes, and separate feature keys/content can retain
attribute identity before learned mixing. Missing or expensive observations
could activate defaults or feature-acquisition routes. This is a possible useful
inductive bias, not a proof of equivalence or superiority to boosted trees.

Tree ensembles are important controls. [Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815)
identify irregular target functions, uninformative features and feature orientation
as issues for the evaluated deep models. Modern dense tabular methods are also
strong: [TabPFN, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6)
uses transformer pretraining to learn a prior over tabular prediction tasks.
Do not claim that Transformers intrinsically cannot model tabular data.

A concrete construction in [theory §323](theory/48_parallel_heads_and_work_scaling.md)
uses paired delays d0±a*tanh(beta*(x_i-theta)) to select the side of a feature
threshold by arrival order. Composing addressed modules can express tree paths,
with multiple paths supplying an ensemble. This establishes representational
compatibility, not successful learning or superiority. Local counterfactual
values need not equal a full unexecuted subtree's loss; that approximation and
all teaching/feature-access work require tests.

## An adapter that preserves the task

An independent row is a set of (feature identity, value, type, observed/missing)
records. Preserve meaningful feature axes, fit encoders/scalers/categories on
training rows only, and reset row-specific state between independent examples.
Never manufacture a sequential dependence from arbitrary dataset row order.
Reordering feature presentation should preserve predictions once feature IDs
and the declared information set are fixed, unless a justified acquisition
policy explicitly depends on observation availability. Cross-row statistics,
in-context labelled rows or online adaptation must be declared as extra inputs.

Static data has no native elapsed-time semantics. Value-to-delay encoding is a
learned representation, not an observed physical time. Candidate discovery,
encoding, feature reads and counterfactual alternatives must all be charged.
Silence must not silently conflate an observed zero with a missing field.
Use observed/missing indicators or a declared observation deadline. Sparse
execution can save subsequent module work even if every input field is read;
a feature-read saving requires an actual selective acquisition policy.

## A compact evidence ladder

1. Numerical input/target independence, row-state reset, feature-ID consistency,
   presentation-order invariance and missingness contracts.
2. Small synthetic controls isolating axis-aligned thresholds, rare relevant
   features, irrelevant-feature scaling and conditional interactions. Include
   a rotated-coordinate version: the point is to identify the bias, not choose
   only friendly tasks. Check learning before increasing dormant capacity.
3. A predeclared small real-data suite with classification and regression,
   numerical and categorical features, missingness and several dataset sizes.
   Preserve standard splits when appropriate; use grouped/temporal splits where
   independence assumptions fail. Tune on development data, then freeze before
   scoring the test. Avoid choosing the suite after seeing ours' scores.
4. Compare boosted trees, random forests, a tuned dense model/tabular Transformer,
   and an applicable pretrained tabular method. New local Transformer/LSTM
   training remains reserved for AWS; all host training, including tree fitting,
   follows unique guarded one-job queues. A protocol is not a queued run.

Use at least three seeds for promotion, equal development/tuning budgets, quality
versus complete fit/inference work, latency, model/storage memory and feature
access counts. For a pretrained baseline, distinguish marginal task adaptation/
inference from its upstream pretraining rather than inventing per-task training
costs. For tree controls report measured resources and their relevant operation
ledger; a comparison solely in neural FLOPs is incomplete.

Useful ablations: input-ID-preserving versus premature dense feature mixing;
learned versus fixed routing/delays; full versus missing counterfactual credit;
stateful versus per-row reset; larger dormant pools at fixed admitted activity.
A gain would support general sparse conditional computation, even if physical
asynchrony is not essential for static rows. It would not establish online,
clockless hardware or event-camera superiority without their own tests.

## First real-data screen prepared — 1 October 2026

`native_tabular_benchmark.py` implements feature-ID preserving independent rows,
train-only scaling and exact feature-duplicate group isolation for UCI banknote
classification and red-wine regression. Missing and observed zero have distinct
marks; canonicalizing feature identity makes presentation permutations identical.
The adapter uses the unchanged native/clock-feature core, all feature reads and
full eight-block counterfactual learning work. It does not selectively acquire
input fields or implement a tree-path topology, optional heads or physical time.

The AWS fast matrix includes native R0/R2 and HistGradientBoosting pairs, 128 fit /
128 dev rows, four neural checkpoints or four tree fitting candidates. Actual
neural forward/loss/backward/clipping/Adam work is traced; all tree-candidate
fitting wall time is counted, without inventing neural FLOPs for trees. Raw rows
are parsed but reserved test labels are never scored. Numerical read-only checks
and a guarded full-depth contract have passed; completed pilot evidence is
preserved beside its limitations in the findings/report.
Data manifests include hashes, original UCI URLs and CC BY 4.0 attribution.
