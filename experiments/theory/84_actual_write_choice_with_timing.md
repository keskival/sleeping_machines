# Actual-write categorical credit with native timing retained

Failure being addressed: completed joint winner/time likelihood correction
regresses matched256-fit NLL by.144750. The frozen four-prefix audit identifies
a common-clock term900–1300 times the conditional choice magnitude under an
inaccurate intermediate-head baseline. This motivates isolation of write-choice
utility; it does not establish variance as the cause of the quality regression.

At the fixed entering prefix and sampled first time T, replay the actual other
winner's delivered value and persistent commit, and continue the complete suffix
with the same future draws. With pi=softmax(s), replace the selected head's local
value score teacher with g_i=pi_i*(F_i-sum_j pi_j F_j). Independently preserve
the reference native timing derivative, assigned to the actual winning score:

    -error_delay * .010*T/(1+T)^2.

The categorical component is the exact derivative of conditional expected legal
outcome loss at fixed T/prefix. The native timing component is unchanged; it is
not an exact derivative through downstream hard decision jumps. Ordinary factual
candidate/content/state gradients and every uncorrected local route teacher
remain. Scores share content/memory/key/clock arguments, so new route utility
propagates to earlier message and memory producers. Sum choice-score credit is
zero; this does not imply that the sum of exp(scores) physical rates is fixed.

Inference architecture is unchanged: p16/L2/H2/pool2, time decay/rotation,
computational delays, temporal races, separately computed keys and values,
persistent sparse addressed commits and source mixing. One corrected event9
head per clip/window, rotating head then layer across the four fixed passes.
No extra parameters, deliveries or inference candidates. Fitting pays one
complete no-grad alternative forward plus the factual backward/optimizer.
The discarded intermediate baseline is not needed for categorical enumeration.
Label use remains supervised losses only, never features, clocks or site policy.

Numerical contracts: independent finite risk derivative plus a separately
autodifferentiated native delay term; preserved winning payload gradients and
uncorrected head credit; exact factual state/logits; actual alternate commits
and prefix times; integrated every-parameter gradient difference equals the
upstream VJP of the replaced score residual; actual interrupted Adam/cursor/RNG
recovery; complete accounting. Then24/8/two-pass U16+U8 learning smoke.

Matched256/192/four-pass seed6 pilot admitted only after these pass. Same
saved local reference, .02NLL improvement/<=1pp accuracy decline/<=1.50 paid
whole-fit work/RSS<900000KiB gate. No post-result schedule or pass extension.
Seed7 local+treatment only if the first gate passes. This is a scoped integrated
credit comparison, not exact whole-model learning or practical supremacy.
