# Depth, finite-race sampling and actual parameter credit

Theory122/123 identify practical growth plasticity and reject the simple
gradient-norm/Adam explanation. Owned near-identity D4/D6 quality comparisons
are pending. Do not duplicate them. A separate unresolved hypothesis concerns
fixed-eight race sampling as depth increases: D6 sampled8 collapses while
factorized D6 learns; D4 full credit improves one seed while sampled8 fails.
This is evidence motivating a conditional mechanism audit, not proof of cause.

## Existing negative controls must constrain this proposal

Note93's trained D2 common/independent whole-history noise covariance ratios
are approximately1.00166/.99119 for route credit and.99787/1.00178 for combined
credit. Both independent-noise gates fail. Do not nominate a new noise policy
from general covariance intuition. Existing branch-exposure diagnostics block
score paths at selected shallow nodes. Shallow priority-allocation experiments
show score variance improvements can coexist with parameter variance increases
(4.012x for one distinct-priority case;1.137x in parameter-targeted confirmation).
The new question is uniform site sampling AFTER the actual shared parameter
Jacobian, at matched initialization protocols across depth2/4/6.

## Fixed protocol and numerical scope

Use two original fine-packet FIT examples0/1, exact FIT-only normalization,
p16/H2/pool2, initialization seed7, clock step.05, depths2/4/6. No DEV/test
arrays read. Factories differ in parameter count, gain and RNG consumption;
same seed does not imply identical shared-prefix weights. This is a matched
configuration ladder, not an isolated causal depth perturbation or a refit.
Promote SAME represented float32-initialized weights to double to isolate
sampling variance from return-centering/float32 issues already owned elsewhere.
Use fixed factual common noise171323 and every actual-write, first-time-
preserving counterfactual suffix return. Enumerate complete episodes exactly
as the frozen batched driver, including all alternative values and scoring.

For episode j and legal race r, define the normalized parameter contribution

    v_jr = (1/B) d/dtheta [sum_i pi_jr,i stopgrad(Q_jr,i)].

This includes score pullback through the complete factual computation. It is
the declared conditional-choice surrogate, not an exact whole-risk or literal
fixed-noise branch derivative. The separate factual gradient retains the
driver's factorized common-clock and factual winner-value credit. Detached
Q does not directly teach losing message parameters.

Cache all v_jr ONCE by actual autograd VJPs. Their sum must equal full route
objective backward. One fixed sampled8 sum must equal an independent actual
sampled objective backward. A tiny R4/k2 enumeration proves the sample mean
and exact covariance identity on actual parameter vectors, not just scalar
score utilities. Kernel/factual logits/RNG/weights must remain unchanged.

For uniform sampling k races without replacement from episode j's R_j,

    Ghat_j = (R_j/k) sum_(r in S_j) v_jr,
    E Ghat_j = sum_r v_jr,
    tr Cov(Ghat_j) = R_j(R_j-k)/(k(R_j-1))
                    * sum_r ||v_jr - mean_r(v_jr)||^2.

Episode subset samplers are independent; sum these traces. The B normalization
is already inside v. This computes exact conditional trace covariance without
constructing a full parameter covariance matrix or assuming Monte Carlo
convergence. Report k8, a fixed larger k32 and all-race where applicable;
fractional k/R and cancellation separately. All formulas use actual R per
episode. Never call the small-pool permutation enumerator: it explicitly
allows only R<=32/k<=3 and could hang at R84..252/k8.

Generate64 fixed subset draws from cached vectors for descriptive distributions.
Add the unchanged factual gradient. Compare raw norm/cosine, clipping factor,
direction against full combined credit, relative discrepancy, and fresh Adam
displacement direction/sign changes. Validate its first-step formula with
two actual discarded Adam forks per depth, pairing each model with FRESH
moments. No claims about trained moment history or convergence.

Save all per-race vectors, factual/full gradients, losses/returns, sample
indices and seed. Record shadow lanes/events, legal races, available capacity,
selected activity and parameter counts. Total diagnostic FLOPs/traffic/energy
remain unknown, not zero; cached simulation draws do not establish cheaper
execution of their gradients. Sparse native inference is unchanged.

## Resource and decision boundary

Based on Theory123's6.142s for four two-backward p16/D4/16-example probes,
the smaller B2 full-prefix VJP ladder is bounded by300s, virtual3GB/RSS1.25GB,
one thread, unique one-job run_safe queue and at least8GiB MemAvailable.
Check queue/results/report/memory/jobs first; no simultaneous training.
Publish any null result beside the hypothesis. High INITIAL route variance
cannot itself explain a trained failure. Low combined/update disturbance
weakens sampling as the early bottleneck even if relative route noise is high.
Do not change active language sources, launch a sampling sweep, or nominate a
priority/critic/representation substitution from this audit alone.
