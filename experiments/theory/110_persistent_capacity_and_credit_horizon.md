# Persistent capacity and credit horizon are different resources

The installed note108/109 language numerical port fixes within-chunk actual
write utility, not learning beyond a detached16-target boundary. Persistent
state can retain a feature whose first useful label arrives after that boundary.
Neither normalized credit nor full within-chunk counterfactual support can
recover the omitted derivative by themselves. This is a training-horizon limit,
not a proof of restricted architecture capacity or a diagnosis of current text8.

Concrete witness: observe x at t0, write z0=theta*x, transport z_D=rho^D*z0,
and predict delayed label x with squared loss. The full encoder derivative is
rho^D*x*(rho^D*theta*x-x). If the entering future-chunk state is detached and
theta appears only at the original write, the derivative is exactly zero.
A finite change to theta improves the delayed loss, so the architecture admits
learning while the restricted estimator supplies no encoder credit. Shared
parameters used again later can still get other derivatives; that does not
restore this omitted original-write path automatically. Decoder adaptation
may partly compensate, but is not equivalent to useful new feature learning.

A race version writes value x or0 at the factual first time. The delayed
conditional choice derivative is pi0*pi1*(L0-L1), with L0=.5*(rho^D-1)^2 and
L1=.5. If all earlier losses are independent of that write and the delayed
label lies outside its credit chunk, its within-chunk route utilities are
identical and choice credit is0, despite both routes being explicitly replayed.
Full support and exact local-expectation arithmetic cannot fix a zero observed
utility difference created by truncation. This does not claim all native
routes or text8 labels have this structure.

For a fixed linear contraction mode with retention rho, the absolute geometric
credit tail after H steps is bounded by rho^H/(1-rho) times its bounded local
loss sensitivity. Its fraction of the infinite geometric sum is rho^H. A
100-event decay retains~85% of this weighting beyond H16. Native whole-state
Jacobian contraction is NOT established: physical rotation is orthogonal,
fixed-input unit damping contracts, but learned source carry/gates, clock
history and addressed interactions need separate bounds; forget gates can
approach0. This linear witness is a diagnostic contract, not a native bound.

Under a fixed replay-event budget, longer horizon plus fewer sampled races
is a distinct axis from wider same-horizon alternatives. Naive fullT16/L8/H2/
pool2 costs512lanes/8192shadow events,512shadow events per target. T64/k8
uniform distinct race sampling costs16lanes/1024shadow events,16per target.
The factual64events and their larger credit graph are still paid. These are
exact execution counts, not measured FLOPs/wall or proven efficiency: sampling
scales each chosen contribution by R/k=128 and can introduce large variance.
For a single informative route, variance is(R/k-1)*g*g^T=127*g*g^T. Perfect
sampling unbiasedness is compatible with poor learning progress. Changing H
also changes the detached-history objective; do not call it merely variance
reduction of the same objective.

Next empirical priority, once exact-source saved language checkpoints are
available: compare fixed prefix conditional returns within16 against a causal
longer suffix on unseen FIT inputs, measure shared-parameter horizon signal,
variance and every paid replay cost before choosing H or k. Existing10M teacher
runs remain unchanged; no long sampled-credit fit follows from this toy witness.
