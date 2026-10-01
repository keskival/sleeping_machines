# What the first AWS screen teaches us

Snapshot: 16 of17 pilot results published locally. Wine R2 remains pending.
Source fingerprints of all16 completed pilots match the saved implementations.
Source protocol: [AWS_RUN_FAST_MATRIX.md](AWS_RUN_FAST_MATRIX.md), recovery
manifest `gym/plans/aws_fast_matrix_recovery_20261001T213409Z/manifest.json`.
These are seed6 development screens, not independent benchmark confirmations.

| Contrast | Completed observation | Interpretation |
|---|---|---|
| Banknote,128 fitting/128 dev rows | Ours R0:95.3125%,NLL.155305; trees:92.96875%,NLL.231868 | Promising quality advantage: three additional correct examples and33.02% lower log loss. Four neural checkpoints/four tree candidates selected on dev; untouched test, paired seeds and stronger controls remain necessary. |
| Banknote reception | R2:89.0625%,NLL.269967; .379 versus R0 .335 whole-fit GFLOPs | Added reception harms this fit and costs12.91% more. Static feature coordinates are not evidence of physical async timing. |
| Wine regression | Ours R0 RMSE.823788 versus trees.648887 | Trees win this screen; ours27.0% higher RMSE. Diagnose learning/conditioning before increasing capacity. R2 pending. |
| Order depth | L8/L4 accuracy51.5625% each; NLL1.020832/1.186294; fit.265264/.133942GF | Deeper processing improves log loss13.95%, at1.98x fitting work. Same accuracy does not imply redundant depth. |
| State dependence | Order51.56% falls to28.13% with state cleared; gaps8x/64x yield29.69/26.56% | Learned state matters, but useful information fails to survive silence despite invariant order labels. Protected channels are a targeted response. |
| Capacity/exposure | S4/16/64 accuracy51.56/48.44/32.81%; .265/.280/.341GF | Updates16 and scores32 per event stay fixed while available state grows128/512/2048. Private rules get32/8/2 queries per pass. More dormant capacity has not become more useful capacity here. Shared rules/private state are the next contrast. S64 dev has only one independent population. |
| Timing/rank | Observed78.125% versus refitted rank79.6875% | Original distribution does not identify a timing benefit. Paired same-order/marks,opposite-label tasks provide a provable information gap for the next test. |
| Counterfactual/pathwise order | Same51.5625%; NLL1.020832/1.028594 | Small loss gain with .86% added work, insufficient evidence for a decisive credit advantage under this budget. |
| Tiny language reception/waiting | Native4.200314bpc; R2 reception4.194859; R2 waiting4.195463 | Reception improves only.000604bpc over waiting, below.01 gate; added temporal processing has not earned its cost. |

Banknote R0 whole fitting work is.335379GF, .655037MF per row presentation,
and .139238MF per inference row. Neural fitting includes actual traced
forward/loss/backward/clipping/Adam. Trees' FLOPs are unavailable. All candidate
tree fits take.0866s; audited neural fitting takes981.8s. The quality lead is
not a compute, wall-time or physical-energy lead. Audit overhead and CPU
simulation are included in neural wall time.

The stronger local native order512-query/eight-pass100% result remains valid;
the128-query/four-pass screen is a different budget. Local matched language2K
native3.764712bpc/3.778244GF is also separate: R2/R4/late variants currently
worsen quality. Relative to the saved KV2K construction3.732586/22.753030GF,
native uses6.02x less counted work at.032126bpc worse, with changed width,
capacity and memory construction. This is a supported near-quality tradeoff.

Next admitted AWS protocol:
[AWS_SPLIT_EVENT_BATTERY.md](AWS_SPLIT_EVENT_BATTERY.md),11 pilots/33 guarded
stages. It tests protected state and shared rules, and makes elapsed time
identifiable. Prefer this already-coordinated plan over the earlier unlaunched
local draft. No claim that these changes will necessarily win.
