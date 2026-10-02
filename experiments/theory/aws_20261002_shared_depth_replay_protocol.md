# Depth-shared receiver maps with corrected deep replay

Concrete failure: completed matched coarse DEPTH4 pilot,256FIT/fourpasses,
full replay lowers fitting-subsetNLL.856196->.807254 versus originalteacher,
but DEV1.342018->1.374384. Original gate FAIL, no unchanged full/seed extension.
This is evidence that deep credit moves fitting, not generalization advantage.

Distinct architectural hypothesis: tie input/output/gate/control/key-read maps
across depth and pool, separately perhead. Private keys, clockbiases, timescales,
arrival state and memories stay separate at EVERYdepth/receiver. Queries,
channelmix, transport and sourcecarry remain. Every event still traverses four
temporal computing layers and eight sparse writes; all16 keys/candidates and
all80legal shadow lanes remain charged. Small content-memory messages,
computational delays/races, separate keys/values, deep persistent state and
counterfactual learning retained. This is familiar parameter sharing supporting
the complete substrate, not a dense/synchronous substitution.

Reason: shared maps get selected-event exposure across layers/receivers and
lower unique fitting parameters; frozen returns may drift less across cold
branches. Theory398 risk law assumes an estimation model and is motivation,
not a general guarantee. Key-read maps can receive nonwinner score credit,
so do not falsely claim every shared map previously learned ONLY on winners.
Lower fit/dev gap and better heldout behavior must be demonstrated, not inferred.

First numerical contract compares an untied reference with IDENTICAL copied
weights to shared forward/state. Shared parameter gradients must equal sums
of all corresponding untied gradients; private gradients unchanged. Double
depth4 fullreplay sequential/forked versus batched everyparameter, alias
ownership, optimizer uniqueparameter enumeration, interrupted Adam/RNG/cursor
and full operator accounting before24FIT/8DEV/two-pass p16 smokes.

Matched teacher/factorized/fullreplay SHARED-model256FIT/192DEV/fourpass
seed7 pilots, sameU16/Adam.003/clip1/.25clock/coarse4. No new noise, optimizer,
normalizer or reception schedule. Fullreplay nomination: NLL>=.03better than
BOTHsharedcontrols and accuracydecline<=1pp, thenunchanged ALL3seed8 confirmation.
Also compare every shared arm to saved SAMEcredit untiedpilot for architectural
attribution. Record uniqueparameters,16capacity/8writes, whole/per-target fit
and infer work, separate allreplay discovery/losingvalue/optimizer cost. No
supremacy or fullfitprediction. Gatefailure stopsunchangedscale. DEVreused epoch
selection, no officialtest. Otherhost fineD4sampled8/regularization ownership
retained. Unique guardedCPUqueues,2GiB RSS/6GiBvirtual,8GiBfloor,1200s timeout.
