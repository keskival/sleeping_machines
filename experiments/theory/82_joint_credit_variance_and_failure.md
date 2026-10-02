# Correct credit, baseline variance and the failed fit

Completed fixed256/192/four-pass seed6 full-state joint-clock pilot:
47.3958%/1.450687NLL versus saved local54.1667%/1.305937. Work ratio1.33904;
quality gate fails. Preserve this result: exact isolated-node likelihood credit
does not suffice for improved fitting in this implementation. No confirmation
or larger unchanged fit is admitted.

At a fixed entering prefix and independently fixed future noise, set
u=Lambda*T, R(T)=sum_i pi_i F_i(T). The conditional-winner joint score is

    h_b(T) = c(T) + pi*(1-u)*(R(T)-b),
    c_i(T) = pi_i*(F_i(T)-R(T)).

The first term exchanges probability between choices. The second changes the
common clock scale. Since u has an Exp(1) law, a baseline independent of the
current winner/time has zero expected contribution. Detachment alone does
not make an arbitrary suffix-dependent baseline independent.

If F_i(T) are time-independent, R is constant and choosing b=R removes all
time variance: h_R=c. An inaccurate b injects score-space variance
||pi||^2*(R-b)^2 while contributing zero useful mean common-clock signal.
Discarding the clock term for time-dependent outcomes is generally biased;
the timing-jump contracts demonstrate that its mean can reverse the choice
component's direction. The solution cannot simply declare clocks irrelevant.

For h_0 and k(T)=pi*(1-u), the minimum-variance scalar baseline at a fixed
prefix/future draw is b*=E[k dot h_0]/E[||k||^2]. This minimizes score-space
trace variance, not necessarily parameter-space variance after the producer
Jacobian, clipping or Adam. Computing b* from many actual time/route replays
is an oracle diagnostic whose entire replay work must be paid; it is not a
cheap baseline implementation or improved benchmark. A small time quadrature
is only an estimate with discontinuous suffix decisions; compare resolutions
and retain differences rather than claiming convergence or exact integrals.

The completed pilot corrects only one event/head per window. All earlier and
other downstream route teachers remain approximate. Its pre-draw baseline
applies the existing decoder to intermediate mean candidate values; this head
was trained at the final query, so prediction of suffix loss is unestablished.
The matched protocol shares current per-pass race draws across clips/windows.
Such correlations can prevent batch-size variance reduction; repeated reuse
also does not provide fresh noise conditional on weights learned with those
draws. Isolated frozen-node unbiasedness is not an assertion of unbiased
training-process gradients under this reuse schedule.

Next bounded audit: first four fitting prefixes from local/treatment selected
checkpoints, event9/both layers/heads, actual legal route and full-suffix
outcomes across8/16 exponential-time quadrature nodes. Compare baseline,
choice/common-clock magnitude and oracle removable variance, preserve all
resolution differences. No optimizer or development selection. This can
nominate an independent-noise/control-variate intervention with matched
controls and recovery/accounting contracts, not establish the cause of the
quality regression. Candidate support, information retention and decoder
learning remain competing explanations. Do not restart the unchanged fit.
