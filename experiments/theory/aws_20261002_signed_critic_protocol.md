# Signed-message and label-aware replay critic

Concrete failure: local norm-only critic fails heldout conditional variance in
both initial and trained scopes. Hypothesis: its features omit signed candidate
content and the supervised target that determines return. Add each candidate
value, centered value and eleven-class label to the existing features. Labels
are legitimate learning-only control-variate inputs; critic remains detached
and frozen before independent correction sampling. No inference or temporal
core substitution; actual writes, separate keys/values, factorized clock and
legal conditional replay remain. Critic work and label conditioning explicit.

Fixed saved trained coarse seeds7/8, first32 FIT rows critic training,next32
critic-heldout, same100 Adam updates,width32,lr.003. No development quality.
Compare exact score-coordinate k1/k2 MSE against plaink4; each gate ratio<=1
in both seeds. All earlier failed results preserved. New feature hypothesis,
not an unchanged long-fit extension. No practical quality/parameter-gradient
variance claim. Producer has seen FIT labels; no producer cross-fitting.
Unique guarded queue,2GiB RSS/6GiB virtual,8GiB available floor,600s timeout.
