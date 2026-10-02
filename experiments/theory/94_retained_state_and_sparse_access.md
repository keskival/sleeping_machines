# Retained state versus sparse predictive access

2 October2026. Diagnostic after the completed noise-covariance gate fails; no
new main architecture is substituted. Frozen query probe75 shows useful learned
features: selected67.1875%/.999857 versus initial57.8125%/1.131057 under its
fitting-only decoder selection. The original native head scores65.1042%/.963161,
so higher probe accuracy alone did not improve NLL. Neither the available
persistent state nor its predictive information was measured by that32-feature
query probe. A failed finite decoder family cannot certify information absence.

## Information bottleneck versus access bottleneck

Let Q be the actual query feature and M the causal pre-query persistent state.
At a population optimum with flexible decoders,

    H(Y|Q)-H(Y|Q,M) = I(Y;M|Q) >= 0.

A positive conditional-information gap means additional prediction-relevant
evidence is retained in state but absent from Q. It does not identify whether
sparse keys, value delivery, state/clock processing or the final decoder lost
access. A finite probe estimates neither conditional entropy optimum exactly;
its improvement is evidence of accessible information under that chosen family
and protocol, not a lower-bound theorem without statistical assumptions. Lack
of improvement does not prove absent information. Ages/occupancy are part of
M, so a gain cannot be attributed only to message payloads or deeper features.

The population augmentation cannot worsen best attainable risk because a
flexible decoder can ignore M. Finite data, selected hyperparameters and
regularization can still worsen validation. This is distinct from note91's
local parameter-tangent result: retained predictive information is a statement
about state sufficiency, whereas attainable score-preserving movement is a
statement about optimization geometry. Either bottleneck can occur without
the other. No surface-n-gram-only or universal gradient-regression conclusion
follows from either finite diagnostic.

## Fixed causal feature protocol

Use the exact initial and dev-selected original native full984-fit encoders
and the existing probe75 reference/artifact. Preserve sources, parameters,
routes and the fixed314159 evaluation draw. Extract actual query head input
Q32 using the batched reference already contracted in note77. Reproduce original
saved probabilities and the old probe's saved probabilities via its exact
readout artifact; no refit/retuning of that reference is needed.

Replay each observed prefix BEFORE its final query and concatenate every
receiver's raw stored memory, age at source query admission and seen flag.
Admission=max(observed query timestamp, preceding context's readiness). Unknown
slots have zero memory/age and seen=0. Use stable layer/head/unit order and
include explicit ages because physical time and representation are coupled.
This causal state exists before the query computation; no target, identity,
future event or label-dependent address is introduced. Augmented Q+M has
32+128+8+8=176 coordinates for p16/L2/H2/pool2. It densely exposes all eight
available receivers for this diagnostic; selected activity of the original
native encoder remains four commits per event. Do not call this sparse readout.

Serial pre-query memory/ages/occupancy must match batched extraction on two
fitting and two development clips for both encoders, with float32 tolerances;
original and old-probe probabilities must reproduce. Every encoder parameter
must stay bitwise fixed, with zero encoder optimizer updates. Readout uses the
same nine generic linear/RBF cells, folds and fitting-only scaling as probe75:
three linear C values and six RBF C/gamma combinations. Select by3-fold FIT-only
NLL then fit once on all984 examples and evaluate all192 dev examples. Preserve
all cells, chosen decoder and feature artifacts. The trained encoder previously
saw all fitting labels and was dev-selected; conditional decoder CV is not
unbiased whole-pipeline validation. No official test is accessed.

A prespecified access signal needs augmented NLL improvement>=.05 and accuracy
no more than1pp worse than the corresponding original query probe. Report both
initial and selected encoders regardless of this signal. It gates a later
fitting-only partition diagnostic (payload versus ages/occupancy and early/late
state), not an automatic architecture change or long fit. Do not select a new
main model from its development score. Practical superiority still requires
an integrated sparse temporal fit and completed matched-quality/resource
comparisons with the preserved full RBF73.4375%/.706478 and compact prototype
66.6667%/.902951 controls. RBF readouts are diagnostic controls and charge their
support vectors/traffic/latency before any deployment claim.

## Work and host boundary

Charge the existing selected encoder's20.075193 whole fitting GFLOPs plus
complete pre-query AND full-query replays for all984/192 examples. Initial
reservoir has no encoder fitting, but still pays both replays and decoder
selection. Kernel computes all candidates for this diagnostic; it is not a
winner-only inference efficiency claim. Capture each actual32/24-target replay
shape, count batch multiplicities and expose raw random-generation work.
Feature ages/materialization, third-party decoder/grid/solver arithmetic,
traffic and energy remain unmeasured rather than zero. Report known core replay
work separately from incomplete total probe fit work; do not compare that
partial total against complete integrated fitting work as an advantage.

One uniquely named guarded CPU job, one thread/nice19,3,000,000KiB VMS,
1,250,000KiB RSS watchdog and8GiB available-memory floor. Native vectorization
keeps the diagnostic bounded; no new Transformer/LSTM or duplicate other-host
credit/depth training. The noise audit's small covariance result remains beside
its original hypothesis; freshness across adaptive updates is still untested.
