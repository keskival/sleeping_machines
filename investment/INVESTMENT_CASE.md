# Sleeping Machines: the investment case

**A learning architecture and computing substrate for intelligence everywhere.**

Prepared 3 October 2026. Investment thesis and proposed commercial strategy;
completed research evidence is identified separately from product ambitions.

## The proposition

Sleeping Machines aims to make capable intelligence economical wherever it is
needed: in datacenters, on personal devices and inside machines that interact
with the world. We are developing models and their execution substrate together.
Time performs computation; messages race to select useful work; persistent
addressed memories retain context; counterfactual credit teaches alternatives
that did not win. The long-term implementation target is a globally clockless,
event-driven substrate supporting both inference and learning.

The investment opportunity is ownership of a useful new way to build, train and
execute AI. A successful platform could earn revenue through model/runtime
licensing, deployment software, accelerator IP and eventually hardware. Each
route draws on a common architecture and toolchain. The initial product must
solve a specific customer problem; the broader platform is the expansion path.

The ambition is substantial: next-generation frontier models with useful memory
and adaptive behavior at a better quality/resource frontier, supported by
computing substrates suited to continuous interaction. Frontier capability,
commercial serving savings and fabricated clockless chips remain milestones.
The current case rests on a working research program with concrete positive
results and a staged way to turn them into customer value.

## Why this could become a foundational platform

AI needs capability under constraints. A datacenter buyer cares about useful
output per dollar, rack watt and unit of resident memory. A mobile buyer cares
about capability under battery, heat, storage and connectivity limits. A robot
needs timely decisions, persistent context and adaptation as its environment
changes. These are different commercial requirements, with a shared technical
question: how much useful intelligence can a system obtain from the work it
actually performs?

Our architecture explicitly separates available capacity, scored keys,
selected state updates, delivered values and counterfactual learning work.
This makes capacity beyond selected activity a design objective. Deep memories
and learned routes could allocate work to the current need while preserving a
larger store of skills and context. The present implementation still scores
all keys in its candidate pools; scalable discovery is a required advance.

If these mechanisms improve the quality/resource frontier, the benefit can be
spent on lower cost or greater capability at the same budget. For example,
a demonstrated 40% reduction in a relevant total cost would permit about
1.67 times the corresponding work at the same budget. That is arithmetic under
the stated boundary, not an observed gain or evidence of better scaling
exponents. A widening advantage with scale must be measured.

## The advantage portfolio

| Mechanism | Potential customer advantage | Present evidence and next boundary |
|---|---|---|
| Time as computation and race attention | Select and combine information through learned delays and arrival order; match irregular streams | Temporal/race implementations and integrated fits exist; precision, jitter and physical timing need hardware validation |
| Sparse addressed updates and separate keys/values | Deliver and update useful content without computing every candidate value during inference | Winner-only evaluator and arithmetic ledgers exist; actual-trained parity, rescore and total runtime remain pending |
| Persistent deep event representations | Maintain context across observations and compute from incoming messages plus stored state | Integrated language/event models learn; latest language scores use segment resets, not demonstrated indefinite memory |
| Capacity beyond selected activity | More useful state or skills at bounded selected work | Doubling p32 receiver slots improves quality at eight writes per input; key scoring, learning and storage grow |
| Counterfactual route credit | Train sparse hard choices and deepen useful computation without dense inference | Alternative-value credit improves completed depth-4 and depth-8 language fits; training alternatives still cost work |
| Event-driven, globally clockless hardware | Reduce unnecessary switching and clock distribution; exploit local computation and communication | Hardware target and energy accounting derived; no fabricated Sleeping Machines chip or measured joule advantage |
| Online learning | Adapt to users, environments and drift near the point of use | Causal full-backbone CPU pilot improves predictions; stable continual learning and on-chip optimizer execution remain open |
| Computation during silence | Preserve relevant temporal evolution without periodic scans; reason about absence when needed | Temporal algebra and silence-aware supervision are part of the research; deadlines, readouts and physical retention are paid |
| Models and substrate developed together | Align learning, memory, execution and physical implementation rather than optimize one cost in isolation | Shared theory, implementations, contracts and audited results exist; an integrated commercial stack is still to be delivered |

Clockless circuits, sparse models, recurrent state and on-chip learning have
precedents. Intel's Loihi 2 describes asynchronous cores and programmable
learning. Our proposed differentiation is the complete trainable construction:
deep content-bearing temporal computation, addressed persistent memory and
credit to unrealized hard routes, translated into a useful system. [Intel Loihi 2](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/neuromorphic-computing-loihi-2-brief.pdf)

## What already makes the case tangible

These completed language comparisons use text8, 10M fitting characters, one
pass and the saved controls' T256 evaluation window. Each native row is a single
seed. Fitting work includes backward, route-credit and optimizer estimates.
Inference work is per evaluated input position; native winner-only values are
shape traces pending actual-trained backend admission, not measured serving.

| Model | Test bpc, lower better | Whole fitting TFLOPs est. | Fit MFLOPs/input position est. | Inference MFLOPs/input position est. |
|---|---|---|---|---|
| Native p32/D4, timing credit | 2.5064 | 7.22 | 0.72 | 0.163 |
| Native p32/D4, alternative-value credit | 2.3715 | 7.24 | 0.72 | 0.163 |
| Native p32/D4/pool4, alternative-value credit | 2.3452 | 11.95 | 1.20 | 0.164 |
| Native p64/D4, alternative-value credit | 2.1833 | 26.79 | 2.68 | 0.605 |
| Native p96/D4, alternative-value credit | 2.1625 | 58.65 | 5.87 | 1.324 |
| Saved LSTM-256 | 2.1706 | 20.31 | 2.03 | 0.677 |
| Saved Transformer-256x2 | 2.4269 | 111.26 | 11.13 | 3.710 |

Three observations are particularly relevant to investment:

- **A learning repair unlocks meaningful capability.** Value-informed route
  credit improves p32/D4 by 0.135 bpc with about 0.3% more counted fitting work
  and unchanged hard forward behavior. The improvement also appears in a
  completed depth-8 model. This is evidence that learning the alternatives
  matters, rather than evidence that additional activity alone explains gains.
- **The sparse native construction has a credible quality/work foothold.**
  Credited p32/D4 beats the saved one-pass Transformer by 0.0554 bpc with about
  15.4 times less estimated fitting work and 15.2 times fewer parameters.
  This is a scoped exploratory comparison against that saved small control;
  it does not establish superiority to optimized contemporary frontier models.
- **Quality and useful capacity improve.** The p96 model slightly exceeds the
  saved LSTM's quality, while using about 2.9 times its fitting work. At p32,
  doubling slots from 16 to 32 improves 2.3715 to 2.3452 with eight selected
  writes unchanged, but 16 to 32 scored keys and about 1.65 times fitting work.
  These results establish progress and a capacity/activity distinction;
  total-resource leadership over the LSTM is still open.

A separate integrated full-backbone online CPU experiment scores 3.1909 bpc
with frozen weights versus 3.0957 with adaptation on 8,191 new development
targets. Predictions precede each 16-character block update; both arms have
persistent event state. This supports useful causal adaptation in one
checkpoint/window/rate. It does not establish lifelong learning, reduced total
retraining cost or a clockless hardware learner.

The program also spans event speech, event vision, tabular tasks and temporal
prediction. Those experiments support a broader research direction, with
different maturity levels and strong controls that sometimes lead. They do not
yet support one universal advantage claim. Causal count references remain
strong in their established region. Invalid target-dependent mixtures remain
quarantined and are excluded from this investment case.

Completed parents and exact values are indexed by the accompanying evidence
manifest. [Research status](../report/sleeping_machines_status.pdf)

## Markets: one foundation, distinct products

**Datacenters and frontier-model developers.** Start with a reproducible
quality/cost improvement on a defined workload. Buyers could license a model,
runtime or accelerator design that increases useful output within their power,
memory and latency budgets. Efficient inference is one entry; efficient
training and stronger scalable models are larger opportunities. A software
proof can precede the capital demands of custom silicon.

The IEA's updated outlook puts datacenter electricity at 485 TWh in 2025 and
projects 950 TWh in 2030. This demonstrates the scale of the constraint; all
datacenter electricity is not our addressable AI spend. Our inference is that
validated quality per watt and dollar could have substantial economic value.
[IEA outlook](https://www.iea.org/reports/key-questions-on-energy-and-ai/executive-summary)

**Mobile, wearables and personal devices.** The desired product is an SDK/model
and later licensed compute IP for continuously available contextual AI:
speech, sensing, assistance and personalization under a device power budget.
Persistent state and local learning could reduce cloud dependence and feedback
latency. Keeping processing local can reduce raw-data transfer, but the product
must implement its privacy and update policy. Online learning must be shown
to improve usefulness without unacceptable forgetting or additional energy.

GSMA's 2026 report describes 5.8 billion unique mobile subscribers and 8.8
billion wireless connections. Those are ecosystem counts, not compatible
device shipments or paying customers. The opportunity is an OEM deployment
that becomes a repeatable design win, then expands across device families.
[GSMA Mobile Economy 2026](https://www.gsma.com/solutions-and-impact/connectivity-for-good/mobile-economy/wp-content/uploads/2026/02/The-Mobile-Economy-2026.pdf)

**Robotics and autonomous machines.** Continuous sensor fusion, memory,
low-latency response and adaptation offer a natural application for an event
substrate. The proposed product is a model/runtime or embedded accelerator
that improves a measured perception or adaptive-control workload. Learning
from deployment could eventually address changing payloads, environments and
sensor characteristics. Closed-loop transfer, latency, robustness and retained
skills require robotic demonstrations; current text/event experiments do not
establish those properties.

IFR reports five million industrial robots operating in 2025 and more than
600,000 new installations that year. This is a concrete industrial ecosystem,
not a forecast of our sales; service robots and future autonomous devices are
additional possibilities. [IFR World Robotics 2026](https://ifr.org/ifr-press-releases/news/five-million-robots-now-operate-in-factories-globally)

**Everything between edge and cloud.** Industrial monitoring, network/telecom
streams, connected vehicles, private enterprise AI, environmental sensing and
adaptive forecasting share aspects of the same opportunity. Some need sparse
continuous sensing; others need capable private inference or adaptation. They
are expansion options after a repeatable first product, with their own data,
quality and deployment protocols. We should not sell one benchmark as proof
that all these markets have already been solved.

## Define the upside in economic terms

The following are transparent, independent scenarios, not forecasts, observed
prices or a computed market size. They show how a narrow deployment can become
a venture-scale business without requiring an immediate win in every domain.

| Route | Explicit hypothetical assumptions | Annual company revenue implied |
|---|---|---|
| Datacenter software/IP | Reach $1B/year of eligible customer execution spend; reduce total eligible cost 20%; capture 10-20% of the $200M value created | $20-40M |
| Mobile compute IP | Win 100M newly licensed devices/year at $0.25-$1.00 per device | $25-100M |
| Robotics/industrial runtime | Serve 1M paying active units at $20-$100 per unit/year | $20-100M |
| Scaled platform | Achieve $250M recurring annual revenue across a validated product portfolio | $250M; requires adoption and sustained delivery |

The datacenter scenario's $200M saving precedes our fee: a $20-40M fee leaves
$160-180M for customers before migration, support and update costs. Final
customer savings must include all those costs. Device royalties are paid on actual licensed units; existing installed
devices are not annual shipments. Robotics revenue assumes a recurring product,
not an upfront chip sale. The routes can overlap, so do not sum them into a TAM.
Chip revenue, margins and financing needs differ substantially from software/IP.

For scale intuition only, $250M recurring revenue at an assumed 8-15 times
revenue valuation would imply $2.0-3.75B enterprise value. Neither that revenue
nor those multiples is a prediction or a current comparable. A frontier-model
and substrate platform with durable billion-dollar annual revenue could
support much larger, potentially tens-of-billions outcomes under suitable
economics. The route to that upside is owning a widely adopted layer of AI
infrastructure, with defensible quality/cost benefits and meaningful value
capture. Exceptional architecture alone does not guarantee distribution.

This is why the upside can justify early investment: capital today buys an
opportunity to establish that layer before the complete platform is proven.
The opportunity has correlated technical risks across its products; several
applications do not constitute independent chances of success.

## Defensibility and capital efficiency

The potential moat combines a learning method, an execution contract,
model/runtime engineering, memory/routing design, hardware mappings and a
growing corpus of reproducible evidence. Successful deployments can add
integration know-how, task-specific learning policies and developer adoption.
The combination may be harder to reproduce than any isolated primitive.
Distribution, measured benefit and the ability to keep improving will matter
as much as the architecture's initial originality.

This is proposed defensibility, not an established patent portfolio. The
historical manifesto is public and cites a 2021 origin; corporate ownership,
contributor rights, dependencies, licenses and protectable implementations
still need a documented inventory. No exclusive rights or freedom-to-operate
opinion is asserted here. Public research can support credibility and adoption
while the company develops commercial execution capabilities.

There are financing precedents for both parts of the thesis: Liquid AI
announced a $250M Series A for efficient general-purpose models in December
2024; Innatera reported a EUR15M Series A for neuromorphic edge technology in
March 2024. They demonstrate investor interest in these categories. Their
teams, maturity and funding amounts do not establish our valuation.
[Liquid AI announcement](https://www.liquid.ai/blog/we-raised-250m-to-scale-capable-and-efficient-general-purpose-ai), [Innatera announcement](https://www.innatera.com/newsroom/innatera-raises-e15m-for-neuromorphic-edge-ai/)

The proposed sequence is capital-efficient: establish the algorithm and
software resource frontier; secure a workload/design partner; validate
fixed-precision hardware semantics on FPGA; fund ASIC development after
quality, economics and demand justify it. A clocked FPGA can validate event
semantics; it cannot demonstrate the energy of a fabricated clockless ASIC.

## A basis for today's valuation

A reasonable opening fundraising position remains **approximately US$10M
pre-money**, with **US$5-15M** as a discussion range, conditional on a credible
founding team, usable rights, a coherent milestone budget and investor demand.
This is a negotiated pre-seed position based on technical progress and platform
optionality, not an appraisal of the repository or a value inferred from the
full market. Unknown team, corporate, customer and ownership facts materially
affect whether that position is attainable.

For context, Carta's 2025 data reports median post-money SAFE caps around $10M
for $250K-$1M rounds and $15M for $1M-$2.5M rounds. Its Q2 2026 report describes
continued concentration in early AI financing. SAFE caps and priced pre-money
valuations are different quantities; neither dataset prices this company.
[Carta 2025 review](https://carta.com/sg/en/data/state-of-pre-seed-2025/), [Carta Q2 2026](https://carta.com/data/state-of-pre-seed-q2-2026/)

At $10M priced pre-money, a $1.5M investment implies $11.5M post-money and about
13.0% new-investor ownership; $2M implies $12M and 16.7%, before option-pool or
other financing effects. A $1.5-2M raise is an illustrative planning envelope,
not a costed financing requirement. Build an actual staffing, compute, partner
integration and runway budget before setting the amount.

A higher price should follow stronger evidence: independently replicated
modern quality/resource advantage, a paying deployment or credible design win,
useful adaptation under drift, and validated hardware economics. The broad
vision explains how large the company could become; these milestones explain
why investors should pay more as the company progresses.

## What the next investment should buy

1. **A reproducible model advantage.** Finish the assigned language fits and
   replications; establish a modern representative comparison with complete
   resource accounting. Keep strong counts, recurrent and attention controls.
2. **A deployable sparse backend.** Admit actual-trained winner/state/cache/RNG
   parity and heldout rescore, then measure setup, residency, latency,
   throughput and total cost. The prepared packed-weight worker is an initial
   runtime implementation; its lifecycle checks pass, native admission is open.
3. **One customer workload and one adoption route.** Define the buyer, useful
   quality level, deployment constraints and acceptable integration cost.
   Proposed first route: a software/model/runtime proof for datacenter AI,
   while validating event-stream strengths for later edge/robotics products.
4. **A credible adaptation result.** Demonstrate prequential gains under drift,
   retained stationary quality and bounded update resources against competent
   online baselines. Distinguish state updates from learned weight adaptation.
5. **A hardware execution contract.** Export validated traces; establish event
   ordering, precision, timing, local state and backpressure. Measure clock,
   communication, memory and learning overhead before an ASIC commitment.

The main failure conditions are explicit: gains disappear against competent
controls; discovery/learning/traffic consume the savings; depth or adaptation
fails to retain useful information; timing precision/control defeats hardware
economics; or customers cannot capture enough value after integration. These
determine which product path to pursue and when to change course.

The investable thesis is a staged path from an unusual trainable architecture
to a repeatable economic advantage, then to a platform spanning the cloud and
physical world. We can earn substantial value with a successful first product
while retaining the much larger ambition of frontier intelligence and a new
computing substrate.

## Evidence and commercial diligence

[Evidence manifest](evidence_20261003.json) records completed-parent hashes,
precise metrics, arithmetic conventions and primary market sources accessed
on 3 October 2026. [Technical milestones](../experiments/DATACENTER_VALUE_MILESTONES.md),
[hardware thesis](../experiments/HARDWARE_VALUE_PROPOSITION.md) and
[frontier protocol](../experiments/FRONTIER_COMPUTE_PROTOCOL.md) supply the next
validation boundaries. No customer commitments, revenue, patent grants,
fabricated silicon or current frontier-scale capability are represented as
existing assets in this case.
