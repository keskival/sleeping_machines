# Confirm the banknote quality signal against stronger controls

The seed6 first screen gives ours95.31% accuracy/.1553 NLL versus the small
boosted-tree screen92.97%/.2319. This is a promising development signal,
not a test win or demonstrated resource superiority. We selected banknote
after this pilot; this protocol cannot establish broad tabular supremacy.

The integrated model remains unchanged: native R0,d8,H2,depth8,pool2,
128 fitting rows,four passes,128 dev rows,Adam.003,16 rows/update. No new
reception or architectural substitution. All feature reads, losing-route
learning, backward, clipping and Adam remain counted. Freeze model choices,
controls and seed6/7/8 queues before any confirmation scores are observed.

Compare four families: ours, the original HistGradientBoosting control,
CatBoost, and logistic regression. Each family has four development selection
opportunities: ours epochs1–4; original trees16/32/64/128; CatBoost256 trees
with depth3/6 crossed with L2 strength1/5; logistic C=.01/.1/1/10. All
independently fitted control candidates are charged. These are equal counts of
selection opportunities, not equal tuning effort, parameter count or compute.
CatBoost uses one CPU thread and no development early stopping. See the
[official API](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier).

Seed6 native **reuses its completed fit**, provided the exact saved selected
checkpoint, settings, data hashes and development score match. Its original
whole fitting work/wall remain charged; new optimizer steps are zero. Preserve
that checkpoint on AWS. A missing/mismatching checkpoint stops for review;
never silently repeat or change it. Seeds7/8 are the two new native fits.

Only after development selects the checkpoint/candidate, score all281 reserved
rows (270 feature groups) with frozen weights, training-only scaling and cold
row state. Do not select the best seed or ensemble. Save row identities,
probabilities and hit/loss arrays for paired analysis. Duplicate groups remain
isolated from fitting/development. Raw labels are parsed during dataset checks,
but never scored or used to select models before this confirmation stage.

Paired primary differences are control-minus-ours NLL; positive favors ours.
Analysis crosses three seed draws with shared resampled feature groups, without
treating repeated test rows as independent examples. Three control comparisons
use Bonferroni98.33% NLL bootstrap intervals; accuracy95% intervals are
descriptive. Three seeds and one split limit uncertainty claims. Report every
seed, all controls and incomplete stages. A positive lower bound against all
controls supports this prespecified quality comparison, not generality, compute
or physical energy. Later architecture changes require a new confirmation set.

## Execution after the prioritized split-event battery

Prepared plan:
`gym/plans/aws_banknote_confirmation_20261001T234000Z/manifest.json`.
It contains four contracts,four full-accounting smokes and12 final comparisons;
all prerequisites precede any pilot. Do not start a second worker while the
split-event battery owns the host lock. Its promising event follow-ups remain
a higher research priority; inspect their result gates before spending on
unplanned scale-up. This banknote protocol is a bounded follow-up, not a
replacement for that architectural investigation.

```bash
.venv-docker/bin/python -m pip install -r experiments/tabular_confirmation_requirements.txt
tmux new-session -d -s aws-banknote-confirmation-20261001T234000Z '.venv-docker/bin/python -u scripts/run_aws_matrix_recovery.py --manifest experiments/gym/plans/aws_banknote_confirmation_20261001T234000Z/manifest.json --jobs 3 >> experiments/queue/aws_banknote_confirmation_20261001T234000Z.out 2>&1'
```

Inspect existing jobs/locks,host memory and GPU occupancy before admission.
Use the approved bounded three-slot CPU scheduler on ip-172-31-47-132 only;
caps2441MiB RSS/3907MiB VMS/job,8GiB memory floor,one thread/job,serial git
publication. Checks/smokes1800s,finals5400s based on the earlier native
~994s wall/RSS under the same128-row protocol. Smoke near-cap stops promotion.
Failures and checkpoints remain preserved. No local trainer is added.

After every completed final has been published:

```bash
.venv-docker/bin/python experiments/tabular_confirmation_analysis.py --manifest experiments/gym/plans/aws_banknote_confirmation_20261001T234000Z/manifest.json --output experiments/results/diagnostics/aws_banknote_confirmation_20261001T234000Z_analysis.json
```

That analysis performs no fitting. Commit its completed JSON on main; report
publication reads completed raw results only. Neural FLOPs, control measured
wall time, inference,preprocessing and test-evaluation cost remain distinct.
Control FLOPs are unavailable and cannot be filled with guessed numbers.
