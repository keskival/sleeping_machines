# Research farm: staged shopping list

Prepared 1 October 2026. These are procurement envelopes, not purchase orders.
No AWS capacity has been provisioned by this plan. Product/spec sources were
checked against primary vendor pages; request current regional quotes before
spending. USD budgets below exclude staff, tax, shipping and ongoing power.

## Buy throughput for uncertainty reduction first

The current native CPU emulator uses one PyTorch thread and many small event
operations. Its completed native2K fit/evaluations take about 17 minutes; the
order capability fit takes roughly 2.5 hours. A GPU does not automatically
speed up this serial Python/event graph. Several independent high-clock CPU
workers yield more variant/seed/data evidence before a GPU optimization effort.
Deferred large Transformer/LSTM controls should run on a provisioned AWS GPU
worker, preserving the user's local reservation for integrated models.

Run exactly one training job per worker host, including CPU jobs on a GPU host.
Do not defeat the policy with container-local locks. Parallelism is across
independent hosts; a multi-GPU host is reserved for one distributed training
job, not simultaneous competing experiments. Keep at least 8 GiB available
on every host and keep RSS/GPU watchdogs active.

## Starter procurement, approximately $15–25k

| Item | Quantity / specification | Planning allowance | What it unlocks |
|---|---|---:|---|
| Independent CPU workers | 4 boxes, 8–16 modern high-clock cores, 64 GiB RAM, 2 TB NVMe, wired networking | $6–10k total, get builder quotes | Four serial guarded benchmark streams, independent seeds, temporal/tabular/language screens |
| Coordinator/storage machine | Existing workstation if reliable; otherwise 64–128 GiB RAM and mirrored 4 TB SSDs | $1–3k | Immutable manifests, result validation, dashboards and report publishing; no training role while coordinating |
| Backup and network | Versioned offsite artifacts plus local mirror; 2.5/10 GbE, UPS | $1–2k | Checkpoint recovery, durable evidence, stable unattended workers |
| FPGA starter boards | 2 Arty A7-100T boards plus cables and breakout fixtures | Boards listed at $314 each; allow $0.8–1.2k with accessories | Receiver/state banks, race/timing quantization, event queues, joins and RTL contracts |
| Power/logic instruments | Precision rail power analyzer, bench supply, scope/logic analyzer, shunts/fixtures | $2–4k, quote instruments | Idle/active joules, energy per event, timing jitter and backpressure traces |
| Initial AWS reserve | CPU burst, one L40S GPU worker when needed, short FPGA sessions | $4–6k spend cap | Large dense controls and short larger-model/hardware experiments before buying expensive accelerators |

The Arty board has 101,440 logic cells, 240 DSP slices and 256 MB DDR3L,
and supports the free Vivado WebPACK tool flow. It is an economical contract
and small-block prototype, not a full frontier-model deployment board.
[Digilent specification and current listed price](https://digilent.com/shop/arty-a7-100t-artix-7-fpga-development-board/).

A Joulescope JS220 is a candidate for low-power rail profiling; choose an
instrument/shunt arrangement that fits actual board voltage/current and
bandwidth, and separately measure system input energy. A rail instrument is
not automatically a server AC power meter.
[Manufacturer product](https://www.joulescope.com/products/js220),
[manufacturer guide](https://download.joulescope.com/products/JS220/JS220-K000/users_guide/Joulescope%20JS220%20User%27s%20Guide.pdf).
The instrument allowance above is a budget, not a verified JS220 quote.

## AWS worker capacity

| Role | Initial host envelope | Upgrade trigger |
|---|---|---|
| Native event/language CPU emulator | c7i/c7a-class x86 host with 16–32 GiB RAM and good measured single-thread throughput; use 32 GiB for headroom | Actual smoke peak RSS, producer-graph length or per-source capacity needs more RAM; benchmark cost/target before committing long runs |
| Dense controls / batched GPU models | g6e.xlarge or g6e.2xlarge, one L40S, 32/64 GiB host RAM | Chosen model/batch/context does not fit measured usable GPU memory, or bigger batch gives a measured cost/target improvement |
| Large-memory CPU/compile | 64–128 GiB host for compilations, checkpoint analysis or large producer graphs | Real peak RSS/compile logs justify it; cores alone do not accelerate the current serial emulator |
| FPGA burst | f2.6xlarge: one VU47P FPGA, 256 GiB host RAM, 16 GiB HBM and 64 GiB FPGA DDR4 | Starter-board resource/traffic limits obstruct the already validated useful block |

AWS lists 48 GB GPU memory for G6e L40S cards; EC2's OS-facing instance
specification lists approximately 44 GiB usable. Use `nvidia-smi` and measured
allocations, not the marketing number, for admission. Set the PyTorch memory
fraction and keep one trainer on that instance.
[G6e product table](https://aws.amazon.com/ec2/instance-types/g6e/),
[EC2 OS-facing accelerated-instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html).

F2 offers one-, two- and eight-FPGA sizes, and its developer AMI includes
Vivado without an additional software charge. Start with one FPGA when the
prototype is ready. Cloud FPGA sessions serve throughput/large-design tests;
local boards remain more convenient for accessible rail/timing measurements.
[F2 product and developer environment](https://aws.amazon.com/ec2/instance-types/f2/).

Record region, instance SKU, actual quoted/on-demand/Spot rate, storage/network
charges and hardware inventory in each run. This plan intentionally does not
invent a universal hourly price. Quote via the
[AWS Pricing Calculator](https://calculator.aws/).
Use interruption-tolerant Spot workers only for drivers with verified checkpoint
recovery. Keep decisive long controls and hardware sessions on stable capacity
when interruption would invalidate the intended measurement.

## Small lab, approximately $50–100k

Expand to 8–12 independent CPU workers after the starter farm is continuously
occupied by worthwhile jobs. Add one 128–256 GiB build/analysis host, 10 GbE,
20–40 TB of redundant local artifacts and versioned offsite copies. Reserve
$15–30k of this envelope for cloud GPU/FPGA work and independent controls.
Purchase one 48–96 GB GPU workstation only after measured utilization and
training throughput justify ownership; obtain vendor quotes and fit a real
smoke in its usable VRAM first. Prefer a second independent worker to an unused
multi-GPU tower if experiments remain small and policy permits one job per host.

Add a larger FPGA/SoC platform after synthesis reports establish needed DSP,
BRAM, external memory and interface resources. Price that board and any required
Vivado license together. Bring in part-time RTL verification/hardware expertise;
equipment without an executable timing/learning contract will not remove the
research uncertainty.

## Larger funded program, $250k+ equipment/cloud envelope

Separate replicated model evidence, fused CPU/GPU emulator work, FPGA execution
and ASIC feasibility into workstreams with milestone gates. Budget engineers
separately from these equipment envelopes. Rent an 80-GB-class or larger GPU
node/cluster when a specified useful fit actually needs it; choose current SKUs
and regional quotes then. Do not buy a frontier GPU cluster just to accelerate
an unfused Python event loop.

Clockless ASIC validation needs asynchronous circuit design, timing/rate/RNG
circuits, precision/noise analysis, memory/learning architecture, EDA access,
PDK/library compatibility, verification and a foundry/test/package quote. Set
its budget from a concrete small-block specification. An FPGA validates the
scheduler/arithmetic/traffic construction but normally uses clocked fabric;
its power does not establish the ASIC's clock-removal saving. The
[hardware value proposition](HARDWARE_VALUE_PROPOSITION.md) defines that scope.

## Hardware validation sequence

1. CPU fixed-point reference and saved input/output/learning traces.
2. VHDL contract simulation with GHDL, randomized ordering/tie/reset/backpressure
   cases, and exact trace comparison. No VHDL implementation is currently claimed.
3. Small-board synthesis/resource/timing reports and recorded tool versions.
4. Bank-access/traffic/latency and whole-task idle/active energy on the board,
   against a competent clock-gated implementation at the same precision/task.
5. FPGA training contracts: losing-route credit, gradient/optimizer/version
   semantics; inference-only fidelity is not proof of cheap native learning.
6. A quoted asynchronous ASIC block once its algorithm, precision and hardware
   data make the expected upside concrete.

[GHDL documentation](https://ghdl.github.io/ghdl/) supports VHDL simulation;
use a pinned released toolchain in the farm rather than silently tracking its
development documentation. Store waveforms and synthesis/activity artifacts
alongside the source commit and benchmark record.

The [gym-farm specification](GYM_FARM.md) describes the model-in, benchmark-matrix-out
workflow and separates implemented jobs from future adapters.
