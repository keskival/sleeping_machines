# Reusable rules and private predictive state

Global §§376–378. Completed screen evidence is in
[SPLIT_SCREEN_FINDINGS_20261002.md](../SPLIT_SCREEN_FINDINGS_20261002.md).
This note explains the observed mechanism contrasts and their limits. It does
not revise frozen forward mechanisms or assert asymptotic supremacy.

## 376. Timing information, not merely ordering

Consider each balanced short/long pair with the same marks M and observed
order O, but opposite deterministic labels. Let Q denote which age regime is
queried. Conditional on each pair, Q is uniform, and Y_short = 1-Y_long.
Thus P(Y=1 | M,O) = 1/2 and H(Y | M,O) = 1 bit, whereas
H(Y | M,O,Q) = 0. Hence

    I(Y; Q | M,O) = 1 bit.

An order-only model that receives identical inputs and coupled internal noise
predicts the same class for both members. It gets exactly one right. No
parameter count or training budget defeats this missing-information ceiling.
The completed refitted rank control is exactly 50%; native observed-time is
95.3125%. This shows reception of useful time information, not a lower bound
on the quality of time-aware conventional models. The bounded rejection
generator and complete-pair split are part of the result's scope.

The generator uses

    r(q) = sum_j m_j [exp(-(q-t_j)/.7) - .6 exp(-(q-t_j)/4)].

Its exact sufficient state is two scalars per stream:

    du/dt = -u/.7,  dv/dt = -v/4;
    at a mark m: (u,v) <- (u+m,v+m);
    predict 1[u-.6v > 0].

There is no periodic integration requirement: between event times, multiply
each state by the corresponding closed-form decay. This is an analytic oracle
for this synthetic generator, not a trained comparator or a novel universal
learner. It proves why time-evolving local state is a sensible receiver for
this task and why an explicit time-aware recurrent control is necessary.
Mark-sign symmetry makes labels balanced conditional on each query-age regime,
so query time alone cannot explain the learned accuracy either. Clearing
source state reduces the completed native model to 50%.

Next informative tasks should vary unknown spectra, interacting marked sources,
irregular gaps and hierarchical factors; then use real recorded events. Do
not equate fitting one two-exponential kernel with frontier expressivity.

## 377. Share a receiver law, not stream information

Assume independent streams have the same conditional transition/readout law
with unknown parameter theta, and private evolving state z_s. A statistically
matched receiver shares theta while retaining each z_s independently. Private
maps instead fit theta_s from n_s examples. Under a correctly specified,
identifiable regular finite-dimensional model with independent examples,
asymptotic estimator covariance is approximately

    Cov(theta_s) = (n_s I)^(-1),
    Cov(theta_shared) = (sum_s n_s I)^(-1).

For equal exposure n_s=N/S this is S I^(-1)/N versus I^(-1)/N. In a smooth
task metric with local sensitivity matrix G, the corresponding local quadratic
excess risks scale like tr(G I^(-1)) S/(2N) and tr(G I^(-1))/(2N).
This is a regular local statistical calculation, not a convergence theorem
for the nonconvex sparse model. Within-population correlations, optimizer
behavior and implicit regularization can change the approximation.

If streams differ, sharing adds bias. In a local quadratic approximation,
private rules become useful when their reduction in mismatch cost exceeds
their increased estimation variance. Stream embeddings or hierarchical
deviations theta_s=theta+B e_s can interpolate between complete sharing and
complete separation. The current sharing experiment removes private source
embeddings as well as sharing maps, so it cannot identify these terms
individually. Preserve that confound and test an embedding-only control.

The screen gives the predicted qualitative exposure effect: shared/P0 gains
16.41 percentage points at S4 and 31.25 at S16 with fixed total fitting data.
At S16, 14,180 shared parameters versus 157,940 private parameters learn
75.39% versus 44.14%, while both keep 512 receiver states. This is useful
capacity without multiplying the receiver law, not evidence that all stored
capacity is fully utilized or that every heterogeneous task should share.

For supplied addresses, available state can scale as O(S L H K d), while
selected transitions and key scores per observed event stay O(L H) and
O(L H K). Transition parameters can stay O(L H K d²), independent of S.
Full fitting includes those transitions, all retained producer graphs and
counterfactual proposals, plus n_updates times the shared optimizer work.
Stored graphs, versioned state adjoints, address discovery, queue metadata and
communication are additional costs. None disappears by declaring dormant
capacity. Shared-weight addressed recurrent systems can share these benefits;
our distinction must also earn its time/routing/learning advantages.

## 378. Separate invariant memory from task-sensitive time

For an order task invariant under strictly monotone time warps, the relevant
history is the equivalence class that preserves marked event order. A sufficient
receiver must preserve this information despite arbitrary silent duration.
If every retained content coordinate contracts toward one fixed state during
silence, there is no other information-bearing channel, and reception has
fixed finite resolution or nonzero observation noise, arbitrarily long gaps
can erase distinguishable histories. Exact unbounded-precision real states
need not lose information at any finite gap merely because they contract.
Finite-gap accuracy may
still be excellent, but it does not imply arbitrary-gap sufficiency.

Conversely, a timing task needs directions that distinguish elapsed intervals.
A useful state decomposition is an invariant content/history subspace and a
time-sensitive computational subspace, with writable mixing at events. This
is an information requirement, not a prescription that half the dimensions
must be protected. The best division depends on the predictive factor ranks,
rates, memory horizon, message bandwidth and available learning evidence.

P2 improves some S16 stretched-gap results but weakens ordinary timing/order.
It simultaneously changes silent retention, dynamic dimension and initialized
frequency/decay coverage. Therefore the next attribution control must preserve
the dynamic suffix's spectrum while changing retention, and separately vary
protected dimension. A failed fixed split does not refute invariant state;
neither does improved long-gap accuracy prove the current split is optimal.

Before broader architectural changes, finish the fixed-budget replication,
isolate those confounds and the addressed-write credit channel from note57,
and compare useful accuracy against full fitting/inference work. These results
favor reusable temporal processing and independent persistent state over
blindly adding source-specific parameters or protecting a fixed fraction.
