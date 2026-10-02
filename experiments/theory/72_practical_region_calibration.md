# Practical headroom before another advantage campaign

The user explicitly requires advantage in regions not already practically
optimal with existing solutions. Beating an information-deprived table is
insufficient. Keep the prior diagnostics, but calibrate against a conventional
predictor that consumes the relevant causal evidence.

## Completed stronger table for the proposed joint text/event task

`joint_event_stateful_table.py` retains the observed question text and latest
observed timestamp of each of four marks. For each question string it learns
which mark age to split, its threshold, both leaf probabilities and split
direction. Neither the generator's word/mark mapping nor DELTA is supplied.
Every question uses a generic search over four observed age features. Labels
are used only in fitting and scoring, never extraction or candidate prediction.
Target mutation, appended future observations and common clock-shift contracts
pass. Missing marks have a shift-invariant explicit sentinel.

Same512 distinct fitting episodes/seed1301, one pass; fixed learner, no dev
tuning. Development256/seed2301:99.609375%,NLL0.051631959.
Confirmation1024/seed3301:99.51171875%,NLL0.055257691. Fitting wall0.027026s,
confirmation feature extraction/prediction0.043617s, maxRSS67,940KiB.
There are20 question strings. Source/data hashes and logical sorting,
threshold, content, timestamp, count and probability work are preserved in
`results/diagnostics/local_joint_stateful_table_20261002T134700Z.json`.
These logical counts and Python wall are not relabelled as neural FLOPs.

At most0.48828125 percentage points of confirmation accuracy remain above
this completed control. The proposed20-point calibration gate is impossible
against that table on these episodes, even for a perfect learner. Thus the
joint recency task is a mechanism/capability diagnostic, not the required
advantage region. NLL may still improve, but neither accuracy saturation nor
this tiny fitting cost supports a broad practical advantage claim. Do not
promote a gain against the time-blind table while omitting this control.

## What the relation confirmations change

Joint seed7/8 reach99.21875/100% on new seed75001; matched local reaches100%
in both. Joint loss0.0496041/0.000147362 versus local0.003160244/0.003534957.
Both prespecified .05-bit joint-over-local gates fail. Preserve seed6's
0.647883-bit gain, but it is not replicated evidence of reliable superiority.

Delivered-value-zero accuracy is98.828125/100/75/99.21875% for s7joint,
s7local,s8joint,s8local. Thus three selected models learn useful prediction
in native recurrent context even without delivered content. The full prefix
and learned state can support the relation; the restricted query count inputs
cannot. This does not isolate a training benefit from the outcome bank:
width, update budget and learning rate also differ from the failed older core
stage. A matched bare-core refit would be needed for that attribution.

No generic disconnected-gradient regression or architectural impossibility
is identified. Nor is semantic language competence or valuable extra depth
proved. Note70's uniform alias is a conditional terminal obstruction, not
a claim that every completed model uses only pooled outcome values.

## Regional admission and next priority

Keep language evidence strata and pooled-memory campaigns owned by the other
host/thread; they currently fail to beat the strongest unseen-context count
reference. Do not duplicate them or declare low evidence alone proves
headroom. The completed AWS bounded-history controls likewise prevent a broad
advantage claim on their simple order task.

Inspect real DVS128 Gesture next, reusing recorded subject-disjoint adapters
and causal first-second observations. Current dense gesture controls are
majority-class failures on88/44 examples, not a credible practical ceiling.
Before a new integrated fit, establish strong class-count, time-aware linear
and kernel controls under a fixed common representation, fit-only scaling,
user-disjoint development and a reserved confirmation protocol. Retain
timestamps, sparse state/races and counterfactual learning in the integrated
candidate. Charge feature extraction, discovered candidates, loss/backward
and optimizer work. If practical controls already saturate the metric, do
not promote that protocol as the new advantage region. Official test remains
untouched during architecture/protocol selection. No DVS improvement is yet
claimed or predicted from these admission steps.
