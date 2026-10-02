# Integrated degree-2 readout after frozen context information diagnosis

Read theory67 and the preserved failure of degree2 on its restricted earlier
balanced task. New real coarse native context probes pass the independently
frozen nomination gate: mean NLL improvement .099764 and accuracy +3.125pp,
with NLL positive at all3 fitted seeds. Resident-state probe does not pass
its additional gain gate. No readout or feature probe proves trainability.

Failure addressed: information readable by a fitting-only nonlinear probe is
not fully used by the existing affine native decoder. Add a standard degree2
upper-triangular feature map of the CURRENT32-dimensional native vector,
with residual11-class linear weights starting at zero. Scale products by
1/sqrt(32) for fixed dimension normalization. This exactly nests initial logits
and all pre-existing gradients while exposing nonlinear joint features. It is
an established polynomial primitive, not a new race or retrieval mechanism.

Retain native p16/L2/H2/pool2, four causal250ms packets/1s query, .25clock
initialization, persistent state, continuous delay/races, key/value separation,
sparse selected writes and local unrealized-route teachers. No all-memory read,
extra prefix buffer/dense carrier or label-dependent input enters. Producer
credit remains the existing full-prefix graph. Score teachers remain scoped
local linearizations; this change does not correct whole-core hard-route bias.

The5808 new residual weights add local products, contraction, parameter/Adam
work. In batched fitting, the head is called at the observed query. Existing
sequential inference calls the head at EVERY event, discarding nonquery logits;
all5 head computations must be charged until a separately validated demand-only
implementation exists. Do not present one head call as current inference.
Old affine-only NumPy port must not silently omit the new residual; this model
has a different head interface and that port cannot pack it unchanged.

Contracts: initial RNG/forward/every original parameter-gradient nesting; explicit
quadratic output/gradients; nonzero residual-learning gradients; mutated targets
cannot change forward inputs/logits; native independent-vs-batched forward,
state and all gradients after a nonzero residual; interrupted Adam/cursor/RNG
recovery and complete operation coverage. Then24fit/8dev/two-pass U16+partialU8
learning/accounting smoke, with<900000KiB RSS and preserved8GiB host floor.

Pilot after prerequisites:256fit/192dev/fourpasses/U16/seed6/Adam.003/clip1,
exact matching initializer/data/order/query loss; reuse saved matched-clock
coarse seed6 reference from aws_coarse_native_20261002T212600Z. Frozen gate:
NLL gain >=.05, accuracy decline <=1pp, native whole-fit work <=1.25x and
inference work <=1.50x reference. All added feature/losing-value/optimizer work
paid; NumPy preprocessing FLOPs/traffic/energy unknown. Gate not widened later.
Seed7 unchanged comparison only after seed6 passes; full-data comparison only
after seed7 passes. Earlier failed full-coarse seed gate remains explicit; this
is a new measured hypothesis, not extra passes on an unchanged failed model.
No official test or broad superiority claim from small development evidence.
