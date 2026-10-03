# Joint conditional branch-content and sampled route credit

## Failure, retained mechanisms and scope

The current batched learner replays alternative suffix losses without
gradients. These correct choice credit, but directly teach only factual
message contents on a realization. This is not proof of a missing expected
content gradient: factual selection can already be unbiased. The failure
to investigate is sparse content exposure/variance.124–128 demonstrate
choice sampling disturbance; they do not establish content variance or a
quality repair. Ordinary deep language gates are not bias-20 growth gates.

Retain the exact native forward: computational delays, hard temporal
races, addressed private state, separate keys/values, message-memory
mixing, full episode paths and factorized clock credit. Only a training
estimator changes; inference, candidate support and available capacity do
not. This is a sibling, never an active-driver replacement or dense carrier.
Full losing-branch forward AND backward work must be charged.
Clipping/Adam are nonlinear: preservation of the preclip estimator mean
does not preserve the expected clipped gradient or optimizer update.
Raw surrogate objective values are not prediction metrics; record factual
CE and conditional-site CE separately.

## Derivation without double-counting

For episode j, uniformly select one legal race r among R_j, independently
of weights/noise/labels. Retain its entering prefix, actual first time T
and future draws. Let pi=softmax(entering scores), and L_i be the FULL
live forced-delivery/write suffix loss for alternative i. Define

    objective_j = sum_i stopgrad(pi_i)*L_i
                + R_j*sum_i pi_i*stopgrad(L_i).

Average across actual B examples. The first component REPLACES factual
loss gradient with its conditional branch average; adding factual loss
would double-count content. The second is the existing uniform single-
site unbiased estimate of the sum of all conditional-choice gradients.
Route scaling R belongs ONLY to categorical credit, never branch content.
The separate detachments prevent extra probability derivatives in content
or extra losing-content derivatives in the route-only component.

For fixed theta/entering history/T/independent future draws, current
winner W~pi is independent of T. Native backward uses -T*pi, independent
of forced identity; forced winning branch exactly reproduces factual
primal and pathwise gradient. Thus the weighted branch average is the
conditional mean of the native pathwise gradient at this site. Its
conditional current-winner variance disappears for ONE episode, and its
mean is preserved. Conditional choice term is invariant to current W.
This is NOT conditioning on the full candidate-exponential vector, which
would reveal W. Shared candidate noise across episode lanes means a
per-episode Rao–Blackwell statement does not prove reduced BATCH gradient
variance; covariance or total learning variance can worsen. Uniform site
sampling and other noise remain. No exact whole-risk derivative theorem.

Future hard topology uses native gradients and branch-specific histories.
The pure-Torch reference must hold each recorded branch's own categorical
topology and factorized clock latent Z=Lambda*T fixed, then use T=Z/Lambda.
Holding candidate exponentials fixed gives winner-only time derivatives;
holding physical T fixed deletes useful timing credit. Neither is the
correct reference. This is an independent reference for the DECLARED
native pathwise clock/value algebra, not future discontinuity derivatives.

## Numerical admission before any fit

Tiny native p4/D2/H2/pool2 one/two-event cases, unequal episode lengths,
early/late selected races. Compare EVERY parameter gradient against:
individually replayed live branches and the explicit detached decomposition;
factual forced-winning branch; independent pure-Torch factorized-time
reference for recorded factual and forced histories; existing BL k1
choice-only driver for identical selections. Assert exact current first-
time preservation, causal factual predictions under relabeling, unchanged
original weights/RNG/kernel, finite gradients and selected-race validation.

At a first-event receiver, the factual losing output map should have zero
payload gradient while the joint estimator has the correctly pi-weighted
gradient. Enumerating current W proves the conditional mean and computes
its factual-winner variance; expose nonzero losing derivatives without
claiming all rare branches get large gradients or batch variance improves.

Actual normalized clip1 Adam, saved next-update recovery and complete
forward/backward/normalization/clip/optimizer/inference operator accounting
must pass before a FIT-only small integrated smoke. No new Transformer/
LSTM, no DEV/test quality nomination, no active AWS protocol changed.
One unique one-job run_safe queue, one thread,3GB virtual/1.25GB RSS,
8GiB available floor,180s timeout. Preserve failures and freeze successful
source/protocol/result bytes. Subsequent fits require separate unique queues.
