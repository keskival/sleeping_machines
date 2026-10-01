# Event-triggered hardware: an energy hypothesis

The target is globally clockless temporal/event execution, with local state,
message-triggered modules and limited local coordination. This is an architecture
target, not a description of the CPU simulator or a completed chip. The current
parallel-head model has a local latest-arrival read/join and bounded event-block
scheduling; global clock removal does not remove causal ordering, communication,
arbitration, handshakes or time references. No FPGA/ASIC implementation or
Sleeping Machines hardware joule measurement is claimed.

## What clock removal alone can save

At equal completed task, quality, throughput and latency, let f be the fraction
of baseline total energy removable with its global clock, and h the added
asynchronous-control/timing energy as a fraction of that same total. If all
other work stays equal:

    E_async / E_baseline = 1 - f + h,
    energy reduction = f - h,
    efficiency ratio = 1 / (1 - f + h).

Illustrative f=0.10,0.30,0.50 and h=0 give10%,30%,50% savings, or1.11x,1.43x,2x
energy efficiency. These are scenarios, not measured fractions or predictions.
For f=.30 and h=.03, the saving is27% / efficiency1.37x. Measure f after the
baseline's competent clock gating, not against a deliberately ungated design.
Removing clock distribution alone does not establish orders-of-magnitude savings.

The larger hypothesis also changes active work and traffic. For duration T and
K observed events, a useful decomposition is

    E_event = P_idle*T + K*(e_match + e_active + e_message)
              + E_deadlines + E_learning + E_timing/control.

State can evolve analytically between reads in the algorithm, avoiding periodic
scans. Physical evolving state, timestamps and calibrated delays still require
an implementation and energy budget. Leakage/retention, fan-out, candidate
matching, routing and idle link power can dominate; they are not zero. A task
whose decisions or likelihood depends on silence must pay for the required
readout/deadline/survival computation. Dormant capacity costs storage and must
not require exhaustive global search on every input.

## Feasible validation path

An FPGA prototype can validate event queues, bank-local state, independent heads,
arrival-order semantics, fixed precision and sparse active updates. A conventional
FPGA implementation will use clocks and clock enables; measure its actual clock,
idle and active resources. An asynchronous ASIC is the more direct test of
physical delay/race computation and local event-triggered communication.

Replay identical saved event traces; test output/learning fidelity, jitter,
rate calibration, head joins, throughput, deadlines and backpressure. Measure
whole-task and idle power, joules/event, transferred bytes, bank accesses and
queue/control energy. Include rate setting, RNG, losing-value training credit,
gradient storage and optimizer updates. Compare a competent synchronous or
clock-gated version at matched quality and constraints. Separate inference and
training; inference circuitry alone does not establish cheap on-chip learning.

Primary precedents: [Intel Loihi 2 technology brief](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/neuromorphic-computing-loihi-2-brief.pdf)
describes fully asynchronous neuron cores with spike communication; asynchronous
circuits themselves are not our novelty. [AMD's clocking guide](https://www.amd.com/content/dam/xilinx/support/documents/user_guides/ug472_7Series_Clocking.pdf)
describes FPGA clock resources/enables, and [AMD power-analysis methodology](https://docs.amd.com/r/en-US/ug907-vivado-power-analysis-optimization/Vectorless-Power-Analysis)
requires activity assumptions for estimates. Their hardware evidence does not
supply the clock fraction or energy ratio for Sleeping Machines.

## The value of Transformer-equivalent clockless training and inference

If the architecture reproduces relevant Transformer quality, stable convergence,
throughput and deadlines at lower whole-system energy, a constant-factor gain is
already useful. Supporting both full learning and inference makes a stronger
substrate claim than inference-only acceleration. This supplies a known useful
workload before testing extra temporal expressivity, compressed models, dormant
capacity and cross-modal integration. These further gains remain hypotheses.
Compare a competent synchronous ASIC as well as GPU emulation to isolate the
clockless architecture from the benefit of custom silicon alone.

Use mutually exclusive baseline fractions f_compute, f_memory, f_clock and
f_fixed that sum to1. At equal task and completion constraints, ideal clock
removal with compute/memory component reduction factors r_C,r_M gives

    E_new/E_ref ~= f_compute/r_C + f_memory/r_M + f_fixed + h,

where h includes replacement timing, control and extra communication energy.
Do not multiply the individual efficiency ratios: these are different portions
of the energy budget. As a purely illustrative scenario, fractions40%,40%,10%,10%,
r_C=r_M=2 and h=5% give E_new/E_ref=.55:45% less energy,1.82x efficiency.
This is not a chip estimate; the real fractions, overheads and component savings
must be measured. Operation counts alone cannot supply these fractions.

Full training must charge gradients, losing-route credit, memory precision,
optimizer state/updates and inter-module gradient communication. Asynchronous
handshakes do not eliminate the dependencies of a chosen learning rule.
Demonstrate the intended update semantics or measure convergence under a stated
alternative. Delays as computational variables are a stronger architectural
hypothesis than merely implementing unchanged dense operations in self-timed
logic. Replicated quality with lower total energy would nevertheless be a solid
milestone, independent of whether the additional expressivity later succeeds.

## Native learning during use

A learning-capable event ASIC could update delays, gates, routes and local
content/state parameters where they reside, in response to observations and
later error or reward messages. This would support continual adaptation without
sending every learning example and weight update to an external trainer. The
potential benefits are local feedback latency, reduced external data movement,
offline autonomy and adaptation to a changing environment. These are hardware
hypotheses; their value depends on useful retained quality and total energy,
including learning state and communication.

Our completed CPU full-backbone online experiment improves 3.191 to 3.096 bpc on
a new 8K development stream, predicting before each 16-character block update.
It demonstrates adaptive learning under that protocol, not a clockless chip or
fully asynchronous optimizer. Current fits use autograd, truncated credit,
shared update windows and global gradient-norm clipping. Mapping these exactly
requires their dependencies; changing them to local/event-triggered updates
requires an explicit learning rule and integrated convergence comparison.

An asynchronous reverse-credit graph can be scheduled by data/error readiness,
with local traces, dependency completion and parameter-version tags, instead of
a chip-wide clock. That is a possible execution construction, not proof that
our current trainer has no synchronization or fits efficiently on silicon.
Losing-route proposals still require credit work. Error delivery, trace storage,
optimizer moments, atomic updates and in-flight state/weight versions must be
included in an on-chip prototype and resource ledger. Exact gradient semantics
and a more asynchronous/stale-update learning rule are distinct experiments.
See theory §321 for the conditional construction and validation milestone.

Many inference accelerators intentionally omit full learning circuitry. An
ordinary neural network is not intrinsically GPU-only: a learning-capable ASIC
can be designed for it. On-chip neuromorphic learning also has precedents:
[Intel's Loihi 2 brief](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/neuromorphic-computing-loihi-2-brief.pdf)
describes programmable learning with local third-factor traces and asynchronous
neuron cores. The stronger research claim is a scalable combination of deep
content-bearing temporal computation, counterfactual hard-route credit, sparse
activity and useful on-device learning, with measured quality/resource benefits.
No uniqueness claim for on-chip learning itself is made.
