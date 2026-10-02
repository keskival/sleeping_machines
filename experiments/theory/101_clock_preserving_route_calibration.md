# Calibrate hard route uncertainty while retaining the common clock

Theory99/100 identifies memory-conditioned concentration, not proven harmful
routes. The completed other-host all-race replay fit shows no first-seed gain;
its generalization gap motivates diagnostics, not a mathematical claim that
credit or representations can no longer improve. This independent frozen test
asks whether less decisive routing helps producer-unseen inputs before any key
normalization or regularization fit. Time, actual sparse receiver writes and
the remaining native message/state/transport construction are retained.

## A conditional inverse transform with no additional random draw

At fixed entering scores s, original independent clocks A_i=E_i/exp(s_i) have
winner W~Categorical(pi), first T~Exp(Lambda), pi=softmax(s). Choose a fixed
loser index j(W), e.g.0 unless W=0 then1. Its conditional residual X_j=A_j-T
is independent Exp(lambda_j), independently of W,T. Thus

    V=1-exp(-lambda_j X_j) is Uniform(0,1), independent of W,T,
    U=sum_{i<W} pi_i + pi_W V is Uniform(0,1), independent of T.

For desired pi_tau=softmax(s/tau), choose W_tau by inverse CDF using U and keep
the actual T. This exactly has the factorized law W_tau~Categorical(pi_tau),
T~Exp(Lambda), independently. It is the law of competing rates
lambda'_i=Lambda*pi_tau_i, sampled through conditional coordinates. Every race
consumes the original number of Exp draws, and tau1 nests its exact original
winner, clock, RNG and state. No loser value needs evaluation at inference.

Reusing the original first time with a differently weighted winner directly
from the original clocks is generally WRONG: new identity can correlate with
that first time. The auxiliary residual uniform is the necessary conditioning
step. It must not be naively differentiated as a score-dependent random base.
Training needs correct factorized clock and actual conditional route credit;
the present positive-temperature class refuses training. Tau1 retains all old
gradients by calling the original reference. Finite-precision ties/CDF endpoint
rounding are explicit limitations; mathematical law assumes continuous clocks.

This changes uncertainty while retaining Lambda at each entering state. It does
not guarantee equal full-prefix clocks: altered writes/content legitimately
change later entering states and rates. Common computational speed remains
available rather than normalized to1. Sparse winner-only value delivery and
one actual receiver commit per head remain; new categorical transforms are
charged CPU work, not a free or hardware-certified calibration.

## Fixed bounded intervention, distinct unseen-FIT cohort

Reuse native256-fit/four-pass seed6/7 fixed final online producers, with exact
initial reservoirs. Select16 evenly spaced indices from256..983 EXCLUDING the
16 indices used in98/99/100; use four independent whole-history noise seeds
411173+1009k. No DEV/test, best-epoch choice, decoder/encoder fitting, target-
dependent forward, extra data/time, or hyperparameter selection.

Four declared settings: original tau1; tau2 all layers; tau4 all layers;
tau2 layer0 only. Record every model/configuration outcome and every prediction,
actual key scores/writes/readiness. Report mean per-history NLL (expected noisy
inference loss estimate) and accuracy, not NLL of an ensemble as though it were
one sparse inference. Original2.285696GF core fit each trained producer remains
charged; representative traced prefix work and whole-job wall/RSS are distinct.

Nomination to a later integrated learning/accounting SMOKE requires tau2/all
to improve mean per-history NLL by>=.02 with no accuracy decline in BOTH
trained seeds. Tau4/layer0/initial outcomes are diagnostic only and cannot
substitute for a failed prespecified gate. Passing still needs full native
parameter/input gradients, corrected choice/common-clock estimator and real
recovery/accounting contracts before training. Neither passed law contracts
nor a frozen FIT nomination establishes benchmark advantage.

Before this intervention, guarded tests must establish uniform auxiliary law,
independent first-time/categorical calibrated law, exact original RNG nesting,
common score-shift coupling, and actual native tau1 output/state/all-gradient
equality. Positive-temperature training remains refused. All sampled candidate
clock discovery, inverse-CDF work and old optimizer work count. Full audit
FLOPs/traffic/energy remain unknown, not zero. Failed calibration would reject
this unchanged proposal, not all clock/message/routing improvements.
