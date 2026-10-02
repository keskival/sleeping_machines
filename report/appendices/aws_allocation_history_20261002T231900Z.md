## AWS corrected replay critics: conditional variance gates fail

Sixteen guarded numerical tests pass before two FIT-only frozen screens.
Against plain k4, critic k2 score-coordinate MSE ratios are2.083742/2.081475
for initial fine seeds7/8 and2.429824/2.247774 for trained coarse seeds7/8.
All k1/k2 nominations fail; mean heldout critic R² negative. Critic fixed before
subset sampling and first-time-preserving actual-write replay remain correct.
Unbiasedness is not sufficient for variance reduction. This conditional
score-space evidence does not measure shared-parameter gradient covariance,
learning quality or supremacy. Both screens retain targets/critics/replays;
diagnostic FLOPs unknown,not zero. No unchanged reduced-replay long fit.
Full scope/costs: experiments/AWS_REPLAY_VARIANCE_FINDINGS_20261002.md.

A distinct signed-message/label-aware critic on64 FIT prefixes also fails:
k2/plaink4 conditional score-MSE2.143301/2.493344 in seeds7/8. Labels are
learning-only detached critic inputs; inference unchanged.32 critic train/
32 holdout,100 fixed updates. This is a negative bounded variance screen,
not a model-quality result. Replay lanes5120 and critic updates200 retained;
full diagnostic FLOPs unknown. Previous norm-only evidence preserved.

## AWS critic calibration: actual parameter variance also fails

Training-only shrinkage yields k2/plaink4 conditional score-MSE2.149378/2.493344
(seeds7/8). Actual shared-parameter variance, including cross-site covariance,
is2.383887/2.189517 on seed7 heldout FIT indices32/33 and37.462736/1.674462
on seed8. Every nomination fails. Exhaustive finite-population and cached
factual-probability contracts pass. No new model/critic optimization or quality
claim;160new VJPs paid, FLOPs unknown. Prior failures remain. Full record:
experiments/AWS_REPLAY_CALIBRATION_FINDINGS_20261002.md.

## AWS compact native-context prototype readout: practical gate fails

Same984 FIT/192 DEV,threefrozen native fitted/initialpairs,33anchors and
fit-user3foldCselection; all state/query/core/portable prediction contracts pass.
Fitted native67.708%/.901193,64.583%/.938003,68.750%/.915083; raw4 compact
65.625%/.888167 andraw20 66.146%/.902387. Native~115KB standalone export
vs40.966KBcoarse/184.334KBfine. BOTHfixed stage/storage nominations FAIL.
Learned-over-initial mean.277040NLL gain and fine-export savings retained;
strong coarsecontrol prevents broad advantage. Allanchors scored/delivered:
dense local RBF readout, not sparse attention. Combinedfit/per-targetfit and
wholeinference work unknown forallarms; parentnative4.218015GF/cache costs
retained. Full common-unit table/three-repeat latency/exports:
experiments/AWS_COMPACT_CONTEXT_FINDINGS_20261002.md.

Training-only ordinal replay priorities also fail: k2/plaink4 conditional
score-MSE2.074656/3.053016(seeds7/8). Positivefloor,correct1/(kp) importance
weights and exhaustive contracts pass. With-replacementproposal vswithout-
replacementbaseline explicit. No quality or reduced-workfit claim.

## Fresh replay allocation signal, with parameter failure retained

Three learned weighted distinct sites on fresh FIT64..95 reduce conditional
score variance23.2%/30.6% versus four uniform sites (seeds7/8); uniformthree
would increase variance41.7%. Exact WITHOUTreplacement inclusion weights and
numerical mean/variance contracts pass. Combined nomination still FAILS:
shared-parameter ratios.600428/4.011873 and.639390/1.009399 on fixed examples.
Six proposedshadowlanes vs eight baseline is unexecuted projected allocation,
NOT the actual all-target diagnostic cost. No quality/supremacy or unchanged
learningfit claim. Full cases/costs: AWS_REPLAY_IMPORTANCE_FINDINGS_20261002.md.
