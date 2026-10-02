# Which retained information matters: payload, clock and layer partitions

2 October2026. Admitted only after completed selected-encoder state-access
signal from note94: query-only67.1875%/.999857, augmented71.3542%/.859084.
Initial augmentation also helps57.8125%/1.131057 ->63.5417%/.886238. Frozen
encoders remain unchanged. This is positive information-access evidence under
finite decoder protocols, not a new sparse model or practical benchmark win.

## Four partitions before seeing any partition result

Reuse the exact cached176-coordinate features from note94; no extra core replay.
Every partition retains the SAME query feature32:

- payloads: query plus raw stored payloads of both layers,160 coordinates;
- clocks: query plus both layers' ages and seen flags,48 coordinates;
- layer0: query plus first-layer payload/age/seen,104 coordinates;
- layer1: query plus second-layer payload/age/seen,104 coordinates.

Layout: query0:32, layer0 memory32:96/age96:100/seen100:104, layer1 memory
104:168/age168:172/seen172:176. Independently check exact partition lengths,
union and intersections before fits. Both initial/selected encoders get all
four partitions, with unchanged nine-cell three-fold fitting-only selection,
fold scalers, random seeds and full984-fit/192-dev data. Preserve every CV cell,
chosen decoder, full probabilities and artifacts. Neither a positive nor a
negative partition result is hidden. Fitting is a generic diagnostic readout;
no encoder weight, memory update or race law is retrained here.

Payload improvement would support retained message-content access. Clock-only
improvement would instead motivate handling silence/ages/occupancy correctly.
A layer1 gain over query supports accessible second-layer state under this
probe, but does not prove two-layer core superiority over a matched shallow
fit. A layer0 gain can likewise be retained early information bypassing a
lossy read path. Partition feature counts and regularization differ, so a
ranking does not identify a causal layer failure or certify population
conditional information. Core depth, keys/value separation, sparse selection
and temporal computation remain the main mechanism commitments.

## Matched decoder-setting controls

The full-state selected probe chose linear C=.1 by fitting CV; the old query
probe chose C=1. That creates a finite-regularization explanation in addition
to retained-information access. Therefore two FIXED full-fit controls per
encoder are declared before any of these new outcomes:

- Query feature alone, using the already fitting-selected augmented decoder
  configuration (selected linear C=.1; initial RBF C10/gamma1).
- Full augmented state, using the already fitting-selected query configuration
  (selected linear C1; initial RBF C10/gamma1).

No grid or selection is added for these controls. This swaps existing
configurations using fitting decisions already recorded by the parents, not
development labels. All transformations and solver options stay the same.
For RBF the established dimension/variance normalization remains, so this is
matching named solver settings, not a claim of identical kernels in different
feature spaces. A gain at both fixed C values would make a simple C-change
explanation insufficient, without creating an information-theoretic theorem.

## Decision and resource boundary

Use the completed partitions to design a bounded integrated access repair.
A dense probe is not that repair. More message deliveries, a learned receiving
window or a small sparse race over stored state are candidates only when the
partition identifies useful payload/state information. Their actual time law,
legal writes, losing-message/route credit, producer gradients, persistent state,
optimizer recovery and complete work must be contracted before a tiny fit.
A new main model still needs unchanged second-seed confirmation and matched
full-data quality/resource comparison; stronger raw RBF remains73.4375%/.706478.
No current result proves useful depth or superiority over that control.

The selected encoder retains its20.075193GFLOPs original fit and1.735341GFLOPs
known diagnostic core replay from note94, plus both stages' UNKNOWN generic
solver/grid/materialization arithmetic. Initial pays its replay and decoder
costs without an encoder fit. Reusing cached features adds zero CORE replay,
not zero total work. All solvers, storage, traffic and inference work are
separate from the sparse inference boundary. One serial safe CPU queue,
one thread,3M KiB VMS/1.25M KiB RSS watchdog and8GiB available-memory floor;
no local Transformer/LSTM or duplicate other-host long campaign.
