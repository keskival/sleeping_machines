# Depth8 corrected replay: seed7 nomination

Allsix models:256 FIT/192 DEV, four passes (1024 presentations), payload16, depth8, twoheads, pool2, coarse4/.25clock. Everyparameter gradient and actual recovery/work contracts pass; allsix smokes learn.

| Family/credit | DEV accuracy | DEV NLL | FIT first32 diagnostic NLL | Whole-fit GFLOPs | Per-target MFLOPs |
|---|---:|---:|---:|---:|---:|
| private teacher | 48.438% | 1.369199 | 1.288000 | 2.022431 | 1.975030 |
| private factorized | 47.396% | 1.399592 | 1.242583 | 2.124006 | 2.074225 |
| private replay | 56.771% | 1.316790 | 0.861515 | 109.118229 | 106.560770 |
| depth teacher | 52.604% | 1.410495 | 1.069838 | 1.989397 | 1.942771 |
| depth factorized | 46.875% | 1.395239 | 1.097126 | 2.090973 | 2.041966 |
| depth replay | 52.083% | 1.348356 | 1.024945 | 109.085195 | 106.528511 |

Both private and shared corrected replay meet the predeclared .03 NLL gain against BOTH matched controls with at most1pp accuracy decline. Both therefore require unchanged seed8 confirmation. This is positive deep replay quality evidence, not supremacy: replay training costs roughly52–55 times the controls. Private replay remains the strongest DEV quality here; mapsharing does not improve its final NLL. Reused DEV epoch selection and single-seed evidence; officialtest untouched. Full replay uses163840 shadowlanes and819200 replayevents per fit, all charged.

Numeric gates:
```json
{
  "private": {
    "teacher": {
      "nll_gain": 0.05240932572633028,
      "accuracy_gain": 0.08333331346511841
    },
    "factorized": {
      "nll_gain": 0.0828021764755249,
      "accuracy_gain": 0.09374997019767761
    }
  },
  "depth": {
    "teacher": {
      "nll_gain": 0.062138427708608335,
      "accuracy_gain": -0.0052083532015482215
    },
    "factorized": {
      "nll_gain": 0.04688262939453125,
      "accuracy_gain": 0.05208331346511841
    }
  }
}
```

## Unchanged seed8 confirmation: both gates fail

Same256FIT/192DEV/fourpasses/1024targets and settings as seed7.

| Family/credit | DEV accuracy | DEV NLL | Whole-fit GFLOPs | Per-fit-target MFLOPs | Inference MFLOPs/target |
|---|---:|---:|---:|---:|---:|
| private/teacher | 55.729% | 1.331247 | 2.022431 | 1.975030 | 0.475909 |
| private/factorized | 48.438% | 1.270459 | 2.124006 | 2.074225 | 0.475993 |
| private/replay | 54.167% | 1.282510 | 109.118229 | 106.560770 | 0.475699 |
| depth/teacher | 48.438% | 1.283150 | 1.989397 | 1.942771 | 0.475825 |
| depth/factorized | 50.000% | 1.271398 | 2.090973 | 2.041966 | 0.475783 |
| depth/replay | 48.958% | 1.270943 | 109.085195 | 106.528511 | 0.475811 |

Private replay retains .048736NLL improvement versus teacher, but loses
1.5625pp accuracy and is .012051NLL worse than factorized control: confirmation
FAILS. Shared replay gains only .012207/.000455NLL (teacher/factorized) and
misses both .03NLL gates; factorized accuracy drops1.0417pp: FAILS. Preserve
the positive depth8 seed7 results, with limited robustness beside them.
No unchanged full984FIT gesture promotion and no supremacy claim.

Both independent native language causal/recovery contracts and1025-char
learning checks pass (private5.311638→4.832740BPC; shared5.273659→4.807838BPC).
These small fits are admission only, not language advantage evidence. Separate
private/shared original-teacher models proceed to10,000,000-character FIT and
1M disjointDEV; corrected full replay is not yet installed in language.
