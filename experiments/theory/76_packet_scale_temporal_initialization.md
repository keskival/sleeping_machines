# Packet-scale initialization without replacing temporal computation

The fixed first real native fit learns above its initialization but trails the
strong nonlinear controls through pass6. Its remaining passes must finish and
stay recorded; no quality gain or clock diagnosis is yet established. A
concrete conditioning mismatch motivates the next matched test independently
of the later selected score.

## Physical dimensions and the candidate

TemporalUnit and transport initialize decay times1..100 and rotation
frequencies in[-pi/2,pi/2]. In the original position-scaled language setting
these are event-scale quantities. DVS timestamps use physical seconds with
50ms observed closures. The unmodified decay times therefore span20..2000
packet intervals, and the largest phase per50ms is only.07854 radians.

Set a target-independent initialization step h=.05 seconds from the existing
observed encoding, and multiply actual decay rates and rotation frequencies
by1/h=20. Initial decay times become.05..5 seconds (1..100 packet intervals),
and maximum phase per packet becomes pi/2. Adjust softplus raw rates using
the inverse map, including the existing1e-6 floor. Initialize only: all these
parameters remain trainable with the same optimizer, clipping and fit budget.
Leave content/maps/gates, addresses, race scores, race draws and the bounded
physical race delay.001..011 seconds unchanged. No new values/keys/parameters,
routes, dense carrier, output head or model capacity is introduced.

For a decay/rotation generator A, the calibrated evolution is

    exp((A/h)*a) = exp(A*(a/h)).

This identity explains the temporal basis change, not an exact equivalence of
the whole coupled core: its external times and race delay remain physical,
so memory, scores and winner trajectories can change. Two layers take at
most.022 seconds of native computation, below the regular50ms input spacing;
ordinary inter-packet queue overload is therefore not the present diagnosis.
The last packet/query share1s and their local admission still applies.

## Credit and resource implications

For an isolated exponential decay, sensitivity to log-rate is

    d exp(-lambda*a) / d log(lambda) = -lambda*a*exp(-lambda*a).

Its magnitude peaks at lambda*a=1. This motivates basis coverage of the
observed interval; it does not guarantee larger raw-parameter gradients or
better prediction. Rotation derivatives likewise do not become useful solely
because their magnitude changes. Note51 previously identified very small
delay-phase changes and introduced additional event-local clocks. Here the
observed physical interval differs, so a matched initialization suffices as
the first test; those additional clocks are not substituted or duplicated.

Changing initial timescales adds no recurrent arithmetic beyond the existing
decay/rotation. Initial parameter transformation and model construction are
included in workflow wall. Scoring, losing-value teachers, all prefix events,
backward and Adam remain paid. Persistent sparse state, separate keys/values,
computational delays and counterfactual credit remain the integrated core.

## Required comparison

Rerun the actual native reference/fast/all-parameter gradient, trained
teacher/inference, causality/metadata and interrupted model/Adam/cursor/RNG
contracts at the new initialization. A bounded full-shape real-data learning
and accounting smoke must pass before a full fit. One locked one-thread queue,
RSS watchdog and8GiB host-memory floor remain mandatory; choose timeout from
the original full fit and measured calibrated smoke.

Full candidate keeps984/192 data, p16/L2/H2/pool2,8 passes,U16,Adam.003,s6
and minimum fixed-pass devNLL selection. Compare both completed initializations
against each other and every preserved strong control; do not extend a weak
fit, tune h on development, infer useful depth or declare independent
practical advantage. Frozen feature/readout diagnosis remains complementary.
If no improvement occurs, preserve it as a failed initialization hypothesis,
not a general ceiling on temporal computation or reason to weaken controls.
