## AWS frozen polynomial optimization study complete: gates fail

All three984-fit coarse encoders and matching initial reservoirs compare
fit-user-selected affine/degree2 logistic heads;108CV/12refits,6replays,
192 development, official test unopened. Folded coefficients exactly reproduce
explicit normalized-feature logits; full native query predictions agree within
frozen tolerance. All5 sequential head calls charged, native state unchanged.
Fitted affine68.056%/.903627mean; quadratic69.965%/.885142. Polynomial adds
only.018485NLL/+1.910pp and worsens seed6 relative to affine, failing .05/+3pp/
everyseed nomination. Parent-native polynomial NLL gains .020633/.157749/
.034854 fail everyseed>=.05. Initial affine58.507%/1.138019 and quadratic
62.847%/1.012070 retained. Numeric contracts do not guarantee predictive gates.

Native parent4.218015GFLOPs is charged for fitted encoders,0 for initial
reservoirs; COMBINED fit FLOPs remain UNKNOWN because decoder solver FLOPs are
unknown. Tensor-state/storage, inference MFLOPs,3repeat walls and every fold
are in AWS_FROZEN_POLYNOMIAL_FINDINGS_20261002.md/raw ledger. Packet-normalization
arrays and serialization metadata are outside its tensor-only storage figures.
All selected portable encoder/head states and reusable feature arrays preserved.
This restricts a polynomial hypothesis; it does not overturn the positive RBF
information result or imply a general architectural impossibility. No unchanged
polynomial escalation, test access or broad advantage claim.
