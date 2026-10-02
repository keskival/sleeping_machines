# Sleeping Machines

Deep learning that computes with time

Tero Keski-Valkama and Karoliina Salminen · Research report · 2 October 2026

Messages carry content and an arrival time. Nodes mix incoming vectors with persistent memory, gate their updates and compete through learned delays. Arrival order and winning races determine the computation. The goal is useful intelligence with much less active work.

## The differentiators at a glance

- **Time performs computation.** Delays, races and phase transformations implement useful functions.
- **Hard routes can learn.** Winning messages execute; unrealized alternatives receive counterfactual credit.
- **Deep, persistent event representations.** Vector messages and local memory carry information and credit through layers; retrieval and temporal primitives share the model family.
- **Capacity beyond activity.** The scaling goal is more useful dormant capacity, selectively recruited and judged by prediction quality at a given total work budget.

## The strongest demonstrated results

- **Generalization.** Race retrieval reaches **100% at four times the training context** within 4,000 examples in all five runs. A learned phase rule solves **all 3,440 unseen modular triples**, using the supplied period 17.
- **Learning from fewer examples.** Depth-three event chains reach **99.73–99.93%** after 2,000 examples seen once; saved Transformer controls reach **33.25–40.80%** with the same number of distinct examples and repeated fitting. Depth-four chains reach 99.9–100%.
- **Learned representations.** Completed temporal-carrier development screens reach **2.572 bpc at 131K** and **2.210 at 1M fitting characters**, four passes. A learned speech encoder reaches **79.69%** on 512 private development utterances. Embeddings, temporal state and vector maps learn. Calibration: closed-form Kneser–Ney counts of the same fitting data score 2.349 / 2.007 bpc on the same targets and lead these small-data comparisons. They are strong references where local statistics are well supported; architecture advantage requires practical headroom (§§376,393–394).

![accomplishments](report/figures/accomplishments.png)

Left: means and recorded ranges, five event runs and two Transformer runs; 2,000 distinct examples, seen once / presented 400,000 times. Right: all five event runs reach 100% within 4,000 examples; the control is the best saved result across seven Transformer configurations and their learning curves. These synthetic tasks use different architectures and structural priors. Sources: E53/E36 and E61.

## New evidence: quality and complete work

**Tabular: competitive accuracy, no confirmed win.** Ours averages **91.8%** reserved-test accuracy versus **94.0%** for boosted trees across three seeds. Logistic regression reaches **94.7%** and lower log loss (**0.091** versus ours **0.235**). The native eight-block model mixes content and memory through parallel temporal receiver heads.

![banknote reserved test](report/figures/banknote_reserved_test.png)

Means and individual seeds6/7/8 on281 reserved rows (270 feature groups). 128 fitting/128 development rows; four fixed selection opportunities. Accuracy uncertainty includes zero difference, but does not establish statistical equivalence. The original development lead, 95.3% versus 93.0%, is retained in Appendix B. CatBoost seed8 was stopped without a score; full confirmation is incomplete. Tree FLOPs are unavailable and CPU fits are faster; no resource advantage over trees is established.

**Statistical memory where counting is strong (10M characters).** On the same999,999 test targets, ours count/copy race mixture scores **1.727bpc** versus **1.799** for LSTM and **1.908** for Transformer; closed-form counts alone (untuned mkn, order 7) score **1.788**. Here counting statistics are near-optimal and the dense controls sit at their level; learned models overtake them only with far more data and parameters (Theory §§381, 394). Our statistical memory therefore adds a useful information path on top of near-optimal counts. It is not the learned native model. Capacity and fitting budgets differ. Appendix B charges floating mixing work and reports integer table work separately.

**Work between two learned language models.** Ours native2K uses **3.78 whole-fit GFLOPs** versus **22.75 GFLOPs** for the saved KV2K construction: **6.02× less counted work**, at 3.765 versus 3.733 development bpc (0.032 worse). Both use four passes and 8,191 scored development targets; width, capacity and memory construction differ. Complete CPU fitting traces include counterfactual learning and Adam. This compares two learned models with each other. Near-optimal count references for this small-data regime are shown in Appendix B as calibration (Theory §§393–394).

**Native data scaling.** The same 54,907-parameter construction improves from **3.765 to 3.557 bpc** when fitting data grows from2K to8K characters, using **15.12 whole-fit GFLOPs**. Both use four passes and the same 8,191 development targets; this is one-seed completed data-scaling evidence.

This banknote comparison concerns one task. Strong synthetic order/retrieval evidence on the preceding page remains valid under its own protocols. Appendix B retains the full cross-domain comparisons and resource ledgers.

## Native strengths: useful time and private state

**Elapsed time carries useful information.** Ours reaches **95.31%** on paired short/long sequences with identical marks, addresses and event order but opposite labels. The refitted order-only control reaches exactly **50%**; paired noise makes that ceiling exact. Clearing persistent state also reduces ours to 50%.

![native mechanism evidence](report/figures/native_mechanism_evidence.png)

**Learning can be shared while memories stay private.** At 16 occupied sources, shared processing raises accuracy from **44.14% to 75.39%**, with **11.1× fewer parameters** and **6.6% less whole-fit work**. All 512 receiver slots remain available; each event commits 16 states and scores 32 keys in both constructions. Sharing also removes private source embeddings; these effects are tested together.

| Construction | Dev accuracy<br/>% | Parameters | Whole fit<br/>GFLOPs | Fit/query<br/>MFLOPs | Infer/query<br/>MFLOPs |
| --- | --- | --- | --- | --- | --- |
| Ours: elapsed time | 95.31 | 42,898 | 0.265 | 0.518 | 0.110 |
| Ours: order-only control | 50.00 | 42,898 | 0.265 | 0.518 | 0.110 |
| Ours: 16-source shared rules | 75.39 | 14,180 | 0.262 | 0.511 | 0.110 |
| Ours: 16-source private rules | 44.14 | 157,940 | 0.280 | 0.548 | 0.110 |

Eight blocks, two independent heads, d8, pool 2; 128 fitting queries/pass, four passes, 256 development queries, seed 6. Timing uses 32 independent population pairs; 16-source order uses 16 populations. Exploratory population-bootstrap gains are 45.31 pp [42.97, 47.66] for time and 31.25 pp [25.39, 37.11] for shared rules; 95% intervals condition on selected checkpoints, not independent seeds or confirmation. Complete counted fitting includes losing proposals, backward, clipping and Adam; special functions have unit weight. These are synthetic mechanism advantages, not superiority over time-aware dense models or measured energy.

Sources: [frozen11-pilot protocol](experiments/AWS_SPLIT_EVENT_BATTERY.md) and [validated findings](experiments/SPLIT_SCREEN_FINDINGS_20261002.md). Full variants and gap-retention diagnostics remain in Appendix B.

## A general architecture for content, time and selective activity

Language tokens, irregular observations and action requests can be expressed as content-bearing events with timestamps and source identities. Sleeping Machines aim to learn through this common interface: local state evolves between arrivals, delays perform computation, and only recruited modules act. This broader design is the central research target.

![general temporal interface](report/figures/general_temporal_interface.png)

## Why this could matter across domains

- **Ordering as computation.** Earlier events change the state/routes encountered by later ones; elapsed time changes that state. The structured order-learning results support this prior for sequence-sensitive signals.
- **Asynchronous sensing.** Updates can follow observations and required deadlines rather than a periodic sweep of all modules. Silence remains informative when the objective depends on waiting time.
- **Instruction-conditioned control.** Language can guide event routing and memory; observations can ground language and update a world state that informs timed actions.
- **Useful dormant capacity.** Stored modules need not all execute for each input. The gain depends on economical discovery and credit, and is judged at a fixed total work budget.
- **Distributed hardware.** Local event-triggered state and communication can reduce global coordination. Globally clockless ASICs are a target; conventional FPGA prototypes retain clocks. Hardware joule savings remain to be measured.

## What is established, and what is next

| Ours: family evidence | Completed result / scope | Next generality test |
| --- | --- | --- |
| Temporal reasoning | 99.73–99.93% event-order accuracy; five runs, declared structured task | Unseen delays, gaps and concurrent streams |
| Deep learned context | 3.121 development bpc; sparse six-block / 32K fit; eight-block models also train | Matched-quality work and capacity scaling |
| Auditory events | 79.69% on 512 private development utterances; selected temporal encoder | Aligned official-test real-stream comparison |

Joint multimodal learning and robot reliability remain research targets. Existing results use separately trained variants; native language uses 27 character pools. Sparse routing motivates tabular prediction: paired delays can represent feature thresholds (theory §323). Preserve feature IDs and avoid invented row order. Trees and tabular Transformers remain controls. The banknote screen leads the original trees; broader superiority requires the stronger-control confirmation.

## The research upside: five routes to useful advantage

The investment thesis is a trainable substrate with a known useful workload and broader capability beyond it. Reproducing relevant Transformer quality and convergence with lower whole-system energy would already be valuable. Temporal expressivity, selective capacity and cross-modal integration offer additional, independently testable upside.

| Strength | Potential benefit | Evidence needed |
| --- | --- | --- |
| Temporal races and local state | Less digital normalization and value aggregation; a path to globally clockless hardware | Matched-quality full learning/inference, precision, throughput and measured system joules |
| Evolving state and reused matches | More useful transformations per expensive match; potentially smaller models | Width/depth/data sweeps at fixed quality and complete work |
| Capacity beyond activity | More specialized stored representations without executing all modules per observation | Improve quality as capacity grows; keep discovery, teaching and execution budgets economical |
| Common content/time interface | Language-guided sensing and event-grounded reasoning/control in a shared model | Joint held-out modality combinations, task success and causal deadline tests |
| Native learning during use | Adapt delays/routes/content locally on an event ASIC; reduce external trainer traffic | CPU online: 3.191→3.096 bpc; asynchronous on-chip credit/updates remain untested |

## Quantitative scenarios, with conditions

- **Compression.** At equal quality, half the width would give one-quarter projection arithmetic and roughly half the matching/state work. This compression is unproven.
- **Energy scenario.** Assume baseline shares of 40% compute, 40% memory, 10% clock, 10% fixed. Halving compute/memory energy, removing the clock and adding 5% control gives 45% savings (1.82×). These are assumptions, not a chip forecast.
- **Dormant units.** H2 stores 864 receivers and selects 16 updates/character (54× capacity/activity). Teaching evaluates 32 receiver alternatives and admitted historical values; shared maps execute. This is not a 54× resource saving.

## Why this is a research program worth testing

Multi-run structured learning/generalization, temporal algebra and deep trainable event representations provide starting evidence. The current language fits have not established a matched-quality resource advantage. Optimizer, width/head/data scaling and repeatability tests address that gap. Theory §320 proposes repeated Poisson arrivals that reuse matched keys without resetting all losing clocks.

Current autograd, global clipping and block-window Adam do not demonstrate fully asynchronous learning. That needs dependency/version-aware credit and tested updates. On-chip learning has precedents ([Intel Loihi 2](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/neuromorphic-computing-loihi-2-brief.pdf)); the proposed contribution is the complete temporal/sparse-credit construction. Compare competent synchronous learning ASICs and charge gradient/optimizer traffic. See HARDWARE_VALUE_PROPOSITION.md and EVENT_STREAM_ADVANTAGE_PROTOCOL.md.

## The hypothesis: more capability per unit of active work

Sleeping Machines combine trainable delays, temporal races, evolving local state and counterfactual credit. The hypothesis is that these mechanisms can approximate useful attention with less selected arithmetic and value movement, then use richer temporal computation and dormant capacity to reach comparable quality with smaller models or less fitting. A common content-and-time event interface can support tokens and irregular sensor streams, with task-specific adapters and losses. Existing cross-task models train separately; the integration target is language-guided event routing and shared state: events ground language and both inform actions. Shared-weight multimodal learning remains a further milestone.

A softmax race samples exactly from its distribution. One winner does not equal its weighted average. Averaging m independent winners has mean-square error variance/m; approximate Transformer containment also requires historical coverage and stable propagation through depth. Temporal state permits additional computations beyond this attention analogue.

## Matched attention work: retain all query/key matches

Let d be total width, L depth, N historical keys, r the feed-forward expansion and B = (8 + 4r)d² the shared projection/content work per layer. S is extra evolving-state work per layer. Two FLOPs per multiply-add:

| Per token / target | Transformer | Ours: race substitution |
| --- | --- | --- |
| Inference | L(B + 4Nd) | L(B + 2Nd + S) |
| Training, approximate | 3LB + 12LNd + 19P/U | 3L(B + S) + 10LNd + 20P/U |
| Logical value reads (FP32) | 4LNd bytes | 4Ld bytes |
| Logical key + value reads | 8LNd bytes | 4L(N + 1)d bytes |

P is updated parameter count and U targets per Adam update. The race training term includes admitted losing-value credit; it is not winner-only training. Our normalization of accumulated gradients adds one operation per updated parameter. Shared embeddings/output and lower-order operations are added in the plotted scenario.

## What would make the case decisive?

- **Matched-quality efficiency.** Repeated completed comparisons of full fitting work and inference work.
- **Capacity beyond activity.** More useful stored modules with nearly fixed routing and execution budgets.
- **Temporal expressivity.** Reuse expensive matches for distinct cheap races and evolving-state responses; test whether this reduces required width or depth.
- **Common event interface.** Tokens, irregular sensors and instruction-conditioned control can use content-and-time events. Real-stream, joint-reasoning/control and hardware-energy advantages require their own benchmarks.

This is a research hypothesis and an architectural comparison, not a frontier-language or measured-energy claim. Current deep sparse learning and structured-task results establish meaningful mechanisms; compression and broad language advantage need further evidence.

## Expected architectural work and access scaling

![full bank temporal scaling](report/figures/full_bank_temporal_scaling.png)

Scenario: d = 256, four heads, r = 4, U = 128 and S = 128d per layer. Context plots use L = 8; the depth plot scores N = 4,096 keys. All keys are scored in both models. Both retain linear context and depth terms, and quadratic width terms. At fixed width, the attention-only arithmetic limit is about 2× at inference and 1.2× during counterfactual training; common projection work lowers these total-work ratios. If richer temporal computation reaches the same quality at width αd and depth βL, projection work scales by βα² and context work by βα. Those additional savings require matched-quality evidence.

![full bank temporal traffic](report/figures/full_bank_temporal_traffic.png)

Winner-only retrieval reduces logical value reads by N in this one-sample scenario. Including the key reads, total K/V access improves by at most about 2×. These counts are logical accesses, not measured off-chip transfers, cache behavior or joules. Multiple winners increase value reads. Explicit digital probability normalization is avoided in a physical race, but clock circuitry and rate setting still have costs.

Shared content/projection structure isolates the attention substitution; this is not a quality-matched fit of our current receiver model. Additional receiver-alternative teaching, indexing and scheduling must be charged when present. Bounded candidate search is a separate coverage hypothesis. See theory note 48, §§313–323; measured quality/work curves remain in the appendix.

## Why this research matters

Sparse neural computation promises to spend work only where information changes. The hard part is teaching useful deep representations when routes can be silent, discrete or absent. A cheap forward pass is insufficient if discovering those routes consumes the savings.

| Research lineage | What it established | The question we pursue |
| --- | --- | --- |
| Neuromorphic and spiking networks | Event-driven signals and trained spike timing; EventProp differentiates at events. | How do inactive alternatives receive useful credit without exhaustive replay? |
| Temporal logic and learned delays | Race/delay algebra computes with time; delay learning already improves SNN recognition. | Can temporal computation coexist with rich vector content and deep learned state? |
| Sparse conditional models / MoE | Selected experts allow capacity to grow faster than active work; Switch trains at scale. | Can message timing, communication and correction work also become selective? |
| Asynchronous state-space models | EventSSM learns asynchronous streams with parallel scans; this is a strong precedent. | Can hard races and counterfactual alternatives add quality per unit of total work? |

### What is distinctive here

We combine computation through trainable time, content-bearing messages, persistent local state and credit to unrealized alternatives. Optionality asks whether distinct, reachable future corrections remain available under a work budget. The theory connects temporal algebra, key/value separation, credit transport and supervision that includes silence. The contribution is this construction and its tested consequences; learned delays and sparse capacity are established ideas.

The integrated language candidates now exercise learned content, temporal races, sparse persistent receivers and counterfactual credit together. An eight-block variant also races historical key/value messages. These small-data experiments explore only a small part of the design space; larger-scale quality and complete resource advantages remain under test.

Our earlier deep sparse-routing pilots often lost activity and useful credit before the final layers. Counterfactual proposals alone did not reliably fix that. The subsequent vector-state and persistent-memory work addresses those observed obstacles. Completed structured-task gains motivate the larger learned-model tests; broad quality, training efficiency and energy must still be measured together.

Primary precedents: [EventProp](https://www.nature.com/articles/s41598-021-91786-z); [Space-Time Algebra](https://arxiv.org/abs/2001.04242); [Learning Delays in SNNs](https://arxiv.org/abs/2306.17670); [Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html); [EventSSM](https://arxiv.org/abs/2404.18508). Our routing failures and revised interpretations remain in the theory index and findings.

## One architecture, several learned computations

The common idea is local computation triggered by an arrival: retain memory, combine the incoming vector with that memory, and choose an outgoing time. A delay changes which messages meet and which race finishes first. This makes timing part of the learned function. The model family implements this idea at several levels of generality.

![shared architecture](report/figures/shared_architecture.png)

| Model | Mechanism | Evidence | What it establishes |
| --- | --- | --- | --- |
| Ours: integrated sparse temporal language | Learned content, state-dependent key races, selected receiver updates and episodic KV | Completed 32K receiver / 2K depth and KV screens | Combined mechanisms train; bounded candidate coverage and causal schedule |
| Ours: learned event-state encoders | Source embeddings, temporal modes, nonlinear vector maps and competing clocks | Language E176; speech E165 | Learned representations and persistent state |
| Ours: routed event query encoders | Candidate payloads, receiver memory and hard value/time races | Breadth E120; language E133 | Trainable event depth across tasks; text/market breadth variants add statistical evidence |
| Ours: structured event mechanisms | Temporal chains, relative pointers and phase composition | E34/E53/E54, E61, E124 | Sample efficiency and generalization with declared structural priors |
| Ours: statistical controls | Conditional counts, backoff and copy probabilities | E173; online pilot | Separate baselines; no event backbone |

**Incoming content is retained.** In the event-state encoder, an incoming vector is projected into rotating, decaying memory. A learned memory read is mixed with a direct input path, normalized and gated. The outgoing vector adds that correction to the incoming vector. The payload therefore depends on both current content and history; timing supplies an additional control.

Each task has separately fitted weights. Current learned encoders use fixed depth and locally dense vector maps; the language scheduler retains state and delayed messages across chunks. The routed query encoder instead rebuilds a supplied context per query. Clock learning uses declared surrogate credit through a hard schedule; it is not an exact derivative of every order change.

## How the model computes and learns

**Time performs computation.** An arrival time is a computational value. A delay adds to that value; a first-arrival race computes a minimum and selects a payload; coincidence detects the latest required arrival. Inhibition can veto a path. With a declared periodic reference, phase transformations compose reusable modular relations. Learned timing therefore changes the function and its causal paths, beyond deciding when a fixed dense calculation runs.

- **Local temporal memory** accumulates observed content and elapsed time without evaluating empty time ticks.
- **Computation through delays** uses waiting times, arrival order and clock races to transform information and select outcomes.
- **Hard races and counterfactual credit** choose the emitted vector and delay. Losing alternatives teach better routes while remaining distinct from the winning forward message.
- **Separate keys and values** let content-dependent keys set routes and clocks while value learning preserves the selected schedule.
- **Trainable depth** carries representations and learning credit through a hierarchy. The reversible construction preserves conditional norms under its stated assumptions; readout visibility and routing remain necessary.
- **Structured memories** include relative pointers, learned phase transformations and conditional evidence. Their task gains retain their declared priors; statistical experts are labeled separately.
- **Appropriate supervision** teaches a completed class decision or the next event's type and waiting time, including information carried by silence.

For a temporal pattern such as A followed by B, a learned delay can bring A's trace into coincidence with B. A competing path can veto the match when C intervenes. The timing-pattern and compositional experiments test these mechanisms; the language and speech encoders learn richer vector messages and temporal state.

The simulator stores arrival coordinates on a common axis; the native function uses local elapsed intervals, precedence and causal joins, not a globally ticking execution clock. A time-origin shift preserves predictions. CPU serialization and a fabricated clockless ASIC are separate implementation claims. The primitives and their symmetry limits are developed in [temporal computation theory, §56](experiments/theory/05_temporal_computation_and_scaling.md); [counterfactual learning](experiments/theory/01_foundations_and_counterfactual_credit.md), [key/value separation](experiments/theory/22_key_value_separation_and_race_boundaries.md) and [reversible depth](experiments/theory/25_reversible_event_memory_and_depth.md) give the learning contracts. Clockless delay/race networks compute relative timing relations; phase arithmetic requires its reference. Each implemented model uses a declared subset. Candidate discovery and training alternatives are charged to the work ledger.

## Useful functions from time, reception and repeated events

A dot product is a compatibility score along one direction. Projecting a message into a learned two-dimensional plane and rotating a local clock vector changes which content direction is receptive. Separate clock modes and independent heads can gate different components, then compose a new vector while retaining the incoming content. This creates content–time interactions, rather than just delaying a fixed computation.

![temporal reception windows trains](report/figures/temporal_reception_windows_trains.png)

| Ours: construction | Analytical function and learning | Implementation status |
| --- | --- | --- |
| Temporal reception | Content dot a rotating learned direction; exact ordinary phase/projection gradients within a route history | Full native candidate; two/four-mode fits and same-clock waiting control |
| Learnable integration window | Compact smooth kernel; exact membership gradients; five moment vectors, no silent-time ticks | Primitive contracts passed; full-model integration remains |
| Information-bearing train | Same first spike, different later evidence, distinguishable train readout; exact fixed-count gradients | Closed-form event solver and train/window contracts passed |

Parameters are useful when their interactions preserve relevant information, reach the readout and receive adequate credit and exposure. Consecutive affine maps can fuse into one; gated products and temporal state create new interactions. The appropriate mode count or local rank is determined by marginal held-out quality per complete work, not by maximizing parameter count.

The figure is an analytical construction, not measured model performance. Hard destination changes and spike creation/deletion still require boundary or counterfactual credit. More emissions are charged; no biological rate-code or free-energy claim. Theory §§337–352 gives the proof, costs, timing-noise limits and frozen/refitted ablation protocol.

## The ambition: useful capacity without proportional activity

The proposed shift is to compute through event timing and selectively active paths. A larger network should be able to retain more useful dormant structure while spending work on the paths a query needs. Counterfactual credit must teach those hard choices, including useful alternatives that did not win. The earlier temporal-chain, pointer and phase results test parts of this case and remain central evidence.

### What temporal softmax actually provides

If candidate clocks have rates exp(score), their first-arrival winner has exactly the softmax choice probabilities. Competition supplies normalization in time. It can avoid an explicit normalizing sum/division in the winner path when those rates are physically available. Score formation, candidate discovery, value delivery and learning still cost work. This identity is not a measured near-zero-energy attention system.

| Ours: mechanism | Completed evidence or status | What remains |
| --- | --- | --- |
| Temporal chains / hard pointers / phase rules | Strong structured-task accuracy, transfer and work comparisons | Transfer the useful priors to broad learned representations |
| Temporal content carrier | Learned embeddings, state, gates and delay-dependent transport | Every layer executes; does not demonstrate dormant-unit scaling |
| Integrated sparse temporal model | Hard races, content memory, sparse state updates and counterfactual teachers pass contracts | Scaling quality and complete learning work are under test |
| Capacity beyond activity | Gains depend on useful sparsity and learning | Measure marginal useful capacity with bounded active work |

### A direct mechanism experiment

The native candidate combines the mechanisms: an addressed content/time event enters eight blocks with two independent receiver heads, selecting one persistent content-bearing unit per head at each depth. Separate state-dependent keys set rates; the winner mixes incoming content and retained memory, then emits a vector and learned arrival time. Addressed losing values receive counterfactual score credit during training. The dense carrier and carrier-plus-retrieval variants remain diagnostic controls.

Observed-source pools and fixed depth are declared priors; learned topology growth and unrestricted asynchronous schedules remain open. RNG and physical traffic are additional. The old time-normalized value sum has a shared random amplitude; its covariance and cutoff claims are corrected beside the original theory, not silently deleted. A centered, conserved teacher is now tested. Its fixed-error expected Jacobian is not an unbiased sampled-loss gradient. Theory §§294–298: experiments/theory/45_race_attention_and_resource_identity.md.

## A native path to more capability per unit of work

The next integrated experiments test what this substrate does naturally: learned time computation, sparse addressed memory and hard choices trained through counterfactual credit. A derived gated-state kernel accumulates ordered interactions without enumerating past pairs; learning useful such representations is the target. Episodic race attention remains a useful preserved comparison; the native branch does not require a per-position attention bank.

![native addressed event path](report/figures/native_addressed_event_path.png)

| Ours: native construction | Exact activity boundary |
| --- | --- |
| S observed sources; L blocks; H heads; P candidates | Available receivers: S × L × H × P |
| One selected receiver per head/block | Selected state commits per event: L × H |
| All addressed candidates are scored | Key scores and training proposals per event: L × H × P |

Useful capacity can grow while selected activity stays fixed, but its value must be learned. At a fixed data budget more local maps receive fewer examples. Sharing learned maps while retaining separate state is one way to improve learning exposure; the native language adapter tests this directly. It changes capacity and is a whole-construction comparison.

Analytic state evolution avoids periodic simulation during silence. Decay can still erase information, so long-gap accuracy is measured separately from operation count. A protected-content subspace alongside evolving time modes is a derived next hypothesis, to test if the current construction loses useful memory.

The current eight-block/two-head/pool2 model selects 16 commits and scores 32 keys per event. Shared maps, content transforms, losing proposals, backward, Adam and source-local causal waits remain paid. The ordered kernel is a restricted algebraic identity, not an achieved capability of the fitted model. Full-depth contracts and accounting smokes pass; quality pilots, refitted controls and independent seeds determine further scaling. Theory §§330–336; completed results and the executable priority appear in Appendix B.

## Useful old evidence: a completed joint-learning intervention

Protected outcome state and two learned key/value races reach 96.09% on new suffixes, versus 78.12% with matched local-credit training. The query suffix and its actual count inputs are identical within each opposite-label group. Predicting the distant relation therefore requires additional observed evidence.

![local joint outcome 20261002T125700Z analysis learning](report/figures/local_joint_outcome_20261002T125700Z_analysis_learning.png)

Reserved loss:0.102 versus 0.750bits/query. All1,024 fitting presentations, complete prefix and optimizer work charged: 0.2379 versus 0.2319GFLOPs estimated. Joint training teaches the loss of candidate pairs; inference delivers only two values after scoring all occupied keys. Initial models and inference mechanisms match.

Shallow joint reaches95.31% at 0.1534GFLOPs. This task supports protected evidence and terminal joint learning; useful extra core depth is not established. Earlier native/tapped fits and frozen readout failures remain in the appendix.

Unchanged seed7/8 confirmation: 0 of2 declared joint-versus-local gates pass. All four arms and delivered-value interventions are reported separately in the appendix; the new suffix set does not select settings.

One fitted seed,128 reserved synthetic queries. Observed predecessor addresses are fixed; terminal content-risk derivatives are exact conditionally, earlier native route derivatives remain scoped. The bound concerns query-count inputs, not all counting. No natural-language or iso-quality resource superiority is inferred.

## Ours: the integrated sparse temporal language experiment

This candidate has no dense language carrier. Each event mixes its embedding with the previous deep message and traverses contextual key races. Only selected receivers update their persistent rotating/decaying state and emit values. Time is part of the computation; inactive receivers are not evaluated on empty ticks.

![full sparse language path](report/figures/full_sparse_language_path.png)

The first six-depth, 16-dimensional candidate provides 324 units but updates only six states per character. Each depth scores two addressed keys; training additionally evaluates both candidate values to teach hard choices. Unaddressed pools remain dormant. The capacity experiment doubles available units to 648 while retaining six selected state updates. Key scoring and counterfactual work still grow and are charged.

Checks establish causal predictions, identical chunked execution, precise clocks at 10M positions, equality of training forward values and winner-only inference, learned key/value/memory gradients and conserved route credit. The smoke fit learns, but is not a quality benchmark. Completed data, depth and memory interventions test progress before larger promotion.

A matched payload-32 / 2K depth screen improves 3.633 to 3.542 development bpc from six to eight receiver blocks, for 32.6% more fitting work. The current eight-block KV variant adds historical races, giving sixteen selection steps after warmup. Fixed data/passes/seed do not isolate depth from increased capacity; this is exploratory evidence, not a scaling law.

Fixed observed-character pools, bounded delays and six sequential event depths; no learned topology or complete frontier-language claim. Interior arrival-time derivatives and counterfactual score surrogates have distinct scope. FLOPs include teaching alternatives and optimizer work; representative sparse traces do not certify whole-run instruction or energy counts. Theory §§299–302; source: sleeping_machines/sparse_race_language.py.

## Ours: queries, memory and context

Race attention specifies how a candidate wins; memory organization specifies what the candidates represent. The integrated language model races between compressed persistent receivers. It does not replace a Transformer token KV cache entry for entry.

![language queries and memory](report/figures/language_queries_and_memory.png)

At each depth, a learned map turns the incoming message into a query. A candidate key combines its learned prototype and a read of its retained state. Query–key compatibility sets an exponential clock rate; the first arrival wins with the corresponding softmax probability. The observed character addresses two candidates per depth. Queries cannot search arbitrary past-token keys in this construction. Winner-only delivery also differs from a deterministic softmax-weighted sum, although its one-step expectation equals that sum.

| Model / storage formula | Raw state / one stream | Forward history |
| --- | --- | --- |
| Ours: width 16, 324 states + clocks + last message | 22.84 KiB | Carried until stream reset |
| Ours: width 32, 324 states + clocks + last message | 43.16 KiB | Carried until stream reset |
| Transformer: conceptual FP32 KV, 4 layers × 256 positions × width 256 | 2,048 KiB | At most 256 characters |

The raw width-32 state is about 47× smaller than this conceptual KV allocation and does not grow with history length. This is a storage-formula comparison, not matched recall capacity or measured total RAM: ours compresses history, whereas KV entries retain separate position-addressable representations. The saved Transformer actually recomputes windows without an implemented KV cache. Its 256-character window is a model setting, not an intrinsic dataset limit. Ours retains forward state beyond its 16-character training-credit horizon.

Ours raw bytes = 324 × (4d + 8) + 4d; conceptual Transformer KV bytes = 2 × 4 × 256 × 256 × 4. Excludes weights, gradients, optimizer, activations, object/index overhead and traffic. Long-range recall and comparable-quality memory advantages remain to be measured. Recurrent compression resembles the memory organization of selective state-space models (Mamba, Gu & Dao, arXiv:2312.00752); our hard temporal races and counterfactual route teacher are separate mechanisms. Compression is not required by races: a separate integrated per-position KV experiment retains historical entries and tests sparse value delivery. Theory §§308–310.

## Ours: completed integrated-language stages

| Ours: fit / payload / pool | Development bpc ↓ | Fitting GFLOPs ↓ | Capacity / selected states |
| --- | --- | --- | --- |
| 32,768 / 32 / 2 | 3.106 | 110.09 | 324 / 6 |
| 32,768 / 16 / 2 | 3.121 | 29.88 | 324 / 6 |
| 8,192 / 16 / 2 | 3.398 | 7.49 | 324 / 6 |
| 8,192 / 16 / 4 | 3.426 | 14.15 | 648 / 6 |

These are the integrated model stages, with identical cold development targets. Each character selects one unit at each depth; addressed alternatives teach the races. Capacity and selected activity are different counts. The fitting ledger includes counterfactual values, backward, clipping and Adam.

Quality, data efficiency and work must be judged together. Completed earlier carrier results remain preserved. Comparing these models also changes payload size, capacity and truncated credit, so a score difference does not isolate one mechanism. No official test or energy measurement is implied.

One seed; observed-character index; no statistical expert. Whole-stream candidate scores, selected updates and teaching visits are recorded. Arithmetic is a representative saved-parameter extrapolation; sparse optimizer activity depends on the actual input and credit history. RNG, indexing and memory traffic are separate. Source: experiments/results/parallel_language/local_full_sparse_language_*.json.

## A demonstrated advantage: generalization with less work

The retrieval and phase computations preserve useful rules when the evaluation extends beyond the fitting examples. The saved compact Transformer and LSTM controls use the same synthetic evaluation targets. **Higher and further left is better:** more accurate answers from less estimated inference work. All points use one logical operation ledger.

![consolidated work frontiers](report/figures/consolidated_work_frontiers.png)

| Ours: event computation | Held-out capability | Estimated work per query |
| --- | --- | --- |
| Periodic path; 69 learned scalars | 3,440/3,440 unseen triples | 188 logical operations |
| Two-layer carrier + hard pointer | 100% at four times context | 728,602 logical operations |

The phase rule costs 188 logical operations per triple; the saved LSTM and Transformer cost 104,518 and 155,592 and score 1.95% and 3.60%. At four times the recall context, the event encoder plus learned pointer scores 100% at 728,602 operations; both compact controls score 7.42% at 2.24M and 4.46M operations. The phase model receives a periodic representation with period 17, and retrieval has a pointer mechanism. These useful priors explain the task advantage and are part of what must transfer to harder tasks.

**Learning work is also selective.** The periodic teacher makes 29,003 mistaken-example updates and 145,015 learned-scalar update visits. The dense arithmetic controls make 4,800 Adam steps: 92.2M parameter visits for the LSTM and 132.5M for the Transformer. These count parameter updates, excluding optimizer state and backward arithmetic; they are not training FLOPs or joules.

Arithmetic: 1,473 fitting triples, a 200-epoch budget, all 3,440 unseen triples; supplied period 17. The phase-only path stops after 47 passes, when an entire fitting pass makes no updates. Recall: 4,000 pointer-fitting examples plus 512 neural-fitting examples; the dense controls receive all 4,512 examples for eight epochs. Width 32 and two generic layers where present, seed 6, one small dense setting. Work is an analytic logical-operation estimate, including configured vector maps, routers, scans, normalization, clock candidates and pointer search. These inference counts are not measured joules or backward/optimizer counts. Full work definitions appear in the evidence appendix.

## A demonstrated advantage: learning temporal structure

Event chains learn timing patterns and compose recognizable parts into ordered structures. The preserved five-run results show accurate recognition and strong sample efficiency. The right panel compares distinct examples rather than incompatible activity counters.

![supremacy map](report/figures/supremacy_map.png)

| Preserved comparison | Ours | Transformer | Fitting protocol |
| --- | --- | --- | --- |
| Timing patterns | 99.95–100%; five runs | 99.60–99.80%; two runs | 200k examples once / 1M with relative-time bias |
| Shared-motif composition | 99.00–99.93%; five runs | 99.60–99.85%; two runs | 40k examples once / 1M with relative-time bias |
| Depth-four order | 99.90–100%; five runs | 98.95–99.05%; two runs | Same 40k distinct examples; one pass / 50 passes |
| Depth-three order at 2k examples | 99.73–99.93%; five runs | 33.25–40.80%; two runs | One pass / repeated fitting on the same 2k examples |

The strongest depth-four comparison has approximately ten times fewer classification errors despite the event learner seeing each fitting example once. The sample-efficiency curve makes the next question concrete: can learned embeddings and broader event representations retain that advantage when the inputs no longer supply known temporal parts?

Completed E35, E34, E53/E54 and E36 files. Ranges describe the recorded runs, not confidence intervals. Architectures and optimization differ; synthetic evaluation sets were reused during research. Event deliveries remain activity measurements. Total arithmetic, backward and optimizer work require a declared ledger; activity divided by dense MACs is not a training-cost or power ratio.

## Why asynchronous, sparse computation matters

Persistent local state can retain experience without repeatedly reconstructing a whole history. An asynchronous node updates when useful information arrives. A hard race emits one chosen message. These mechanisms create a path to spending less computation and moving less data per useful answer. On an event-oriented processor, inactive nodes and communication links could remain idle.

### Spend the next unit of work where it helps

A larger budget can buy a longer memory, better retrieval, a deeper representation for difficult inputs or more local adaptation. The development rule is to measure the held-out improvement from each choice and allocate work to the most useful one. The phase and pointer results show why an appropriate computation can be much cheaper than a generic dense approximation. The scaling program tests how much of this flexibility survives when that computation must itself be learned.

| Architecture | Fitting known sequences | Generating or processing a stream |
| --- | --- | --- |
| LSTM | Gates depend on prior hidden state; recurrent work is sequential | Retained hidden state; dense gate updates per token |
| Transformer | Causal attention permits sequence-parallel fitting | Sequential token generation with a key/value cache; adaptation is possible |
| Ours: Sleeping Machines | Affine event memory supports parallel scans once incoming values/times are known | Retained local state and delayed-message queue; only due arrivals execute |

Parallel fitting and sequential generation are compatible. The precise-clock language scan gives a 12.30× complete-step CPU speedup against serial execution of the same width-256 model, including backward, clipping and Adam. Predictions, states, gradients, chunk boundaries and causality are checked. This follows a bounded-delay schedule; arbitrary reordering networks require their own contract.

Parallel recurrent computation also appears in [linear attention](https://proceedings.mlr.press/v119/katharopoulos20a.html) and [selective state-space models](https://arxiv.org/abs/2312.00752). EventSSM already processes asynchronous events with scans. These are important controls. The distinctive hypothesis here is the combination of learned timing, sparse communication, credit to alternatives and independent compute budgets; it must earn its advantage empirically.

Primary FLOPs describe the declared event algorithm, including required vector maps, candidate computation, scans, backward and optimizer updates. Simulator padding and dispatch are separate implementation overhead. Physical power also depends on memory, queues, communication and hardware utilization. Demonstrated work savings and projected power savings are distinguished; total device joules have not yet been measured.

## A mathematical foundation for trainable computation

| Principle | What it enables |
| --- | --- |
| Stable transport through depth | The reversible packet/memory construction preserves conditional value and credit norms under its stated operator and boundary assumptions. Readout visibility, routing and optimization remain separate requirements. |
| Active communication support | Inputs need causal paths through which to interact. A context channel supplies joint information when sparse packets leave local groups disconnected. |
| Credit to unrealized alternatives | A losing payload or timing choice can show how a different route would change the outcome, while forward computation remains a hard race. |
| Periodic state as an isometry | Learned rotations/reflections have unit-magnitude occurrence derivatives. Their composition supports reusable arithmetic instead of a table of observed tuples. |
| Certified composition | Target-constrained min/max composition of phase errors certifies the fitted modular rule across all 4,913 possible tuples; exhaustive checking confirms it. |
| Natural supervised credit | Categorical and event likelihoods both credit predicted sufficient statistics minus observations. Silence enters through integrated exposure. |
| Useful optionality | Reserve consists of distinct, attainable future corrections under a causal work budget. Reachability and transferable learning matter alongside immediate loss. |
| Statistically useful credit | Expected improvement must overcome the curvature cost of fitting noise. Cross-example teacher agreement separates reproducible correction from raw gradient magnitude. |

### From mathematics to an engineering discipline

The theory connects representation, topology, clocks and optimization. Expressivity describes what a network can compute; transport describes whether information and credit survive; the objective describes what the teacher asks it to learn. These pieces must agree. The periodic certificate is one concrete case where the formal model explains and verifies a learned computation.

A common implementation makes these principles reusable across tasks. Efficient primitives can own a computation when its structure is known; a deep carrier can learn representations when it is not. The research objective is to combine this flexibility with affordable route discovery and increasingly capable models.

Formal derivations and their assumptions are indexed in the project's theory notes. Conditional stability and a certificate for a fitted rule do not establish global optimizer convergence or a scaling law. The program builds on established deep spiking and sparse conditional computation; its focus is the combination of useful temporal operators, hard causal routing and counterfactual learning. Related survey: [Direct training of deep spiking networks](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full).

## Potential grounded in the completed evidence

The objective is capable models that spend computation where it improves an answer. The saved structured-task results and learned stream pilots provide specific starting points. Moving from those mechanisms to frontier prediction requires useful representations, longer context and measured quality at a fixed total resource budget.

| Opportunity | Present foundation | What would establish the larger case |
| --- | --- | --- |
| Compact prediction and memory | Persistent learned state; pointers generalize to longer contexts. | Competitive held-out language quality, retention and complete fitting/inference cost. |
| Continuous perception | Learned temporal speech representations and composition. | Full speech/vision tests and confidence-based early decisions at measured latency. |
| More useful training per budget | Structured-task sample efficiency; exact causal parallel scans. | Quality improvements at equal total fitting work, including route discovery. |
| Dormant skills and selective depth | Hard race routing and compute-allocation theory. | Learned marginal work allocation that outperforms fixed allocation on real tasks. |
| Mobile and industrial autonomy | Local state and event-triggered updates. | Device joules, memory traffic and task quality measured on deployable implementations. |

### Why the hardware and economic implications could be large

If comparable quality needs less total training and inference energy, a fixed power and capital budget can support more capable models, more research or continuous adaptation on robots, phones and instruments. Persistent local state and selective communication would favor hardware that handles message delivery, queues and memory efficiently. These are conditional consequences of measured savings, rather than savings inferred from event counts.

The near-term experiment asks where the next unit of computation helps: longer memory, richer content transformations, retrieval, depth or local learning. Held-out gains and complete work determine promotion. A successful scaling result must show that those gains continue across independently fitted sizes and budgets.

## Ours: language learning and staged scale-up

These learned models use character embeddings, gated residual content transformations and persistent rotating/decaying state. Temporal modes encode relative token distance. No statistical count, copy or word experts provide their predictions. The goal is competitive quality from a learned backbone before claiming a language compute advantage.

![language capacity scaling](report/figures/language_capacity_scaling.png)

| Ours: width | Parameters | Development bpc ↓ | Total fitting GFLOPs ↓ |
| --- | --- | --- | --- |
| 32 | 21,741 | 2.858 | 76.11 |
| 64 | 80,301 | 2.727 | 272.30 |
| 128 | 308,013 | 2.643 | 1026.18 |

All curves use six layers, seed 6, 131,072 fitting characters, four passes and the same 8,191 cold-context development targets. Credit is truncated every 64 characters; memory persists. FLOPs include forward/loss, backward, clipping and Adam; special functions are reported separately in each result. These runs vary capacity at equal data/passes, rather than equal compute, and establish neither a scaling law nor official-test superiority.

### From longer memory to selective content

| Ours: memory variant | Parameters | Development bpc ↓ | Fitting GFLOPs ↓ |
| --- | --- | --- | --- |
| Constant / inherited | 308,013 | 2.643 | 1,026.18 |
| Constant / long decay | 308,013 | 2.752 | 1,026.18 |
| Constant / long spectrum | 308,013 | 2.858 | 1,026.18 |
| Input gates / inherited | 309,561 | 2.587 | 1,040.61 |
| Input gates / long decay | 309,561 | 2.627 | 1,040.61 |

Identical width, seed, data, four passes and 64-character credit horizon. Constant-memory arms vary initial timescales/frequencies. Input-gated arms add content-dependent write/forget controls (0.50% more parameters, 1.41% more fitting arithmetic). Longer decay alone worsens this fit. These are single-seed development comparisons; complete numerical contracts precede training.

## Ours: content and memory in fitted language models

An event carries information about its input. The learned vector is a transformation of incoming content and persistent state, with a residual path and an output gate. The new candidate also gates memory writing and forgetting from incoming content. These checks measure whether the fitted predictions use those paths.

![language content memory audit](report/figures/language_content_memory_audit.png)

Resetting all history before each character preserves its current embedding and learned content transformations, but raises the gated model's development loss from 2.587 to 4.519 bpc. Zeroing incoming embeddings raises it to 7.569. Both present input and earlier messages contribute to prediction.

### Selective retention adds a useful control

The two input gates start at one, preserving the constant-memory model exactly at initialization. During fitting they learn different write strengths and forgetting factors for different incoming vectors. A factor below one slows decay; above one accelerates it. The controls are known from the causal previous layer, so serial execution and parallel affine scans retain their checked outputs and teachers.

In the matched small fit, gates improve 2.643 to 2.587 bpc for 0.50% more parameters and 1.41% more fitting arithmetic. This is a local quality/work improvement, with one seed. All six layers still execute for every character; dormant-unit scaling remains a separate target.

Frozen selected checkpoints, identical 8,191 cold development targets, no training or official-test reads. These interventions disrupt a trained model; they establish fitted dependence, not the quality of retrained ablated architectures or lossless storage. Source: parallel_language/local_language_representation_20260930T162337Z.json.

## Ours: larger language development stages

Each stage fits independently from initialization. Data, capacity and memory controls are declared below. Development selects weights within the fixed four-pass budget; ongoing training logs are never substituted for a completed result.

| Ours: memory / width | Fit characters | Parameters | Development bpc ↓ | Fitting TFLOPs ↓ |
| --- | --- | --- | --- | --- |
| Input gates / 256 | 131,072 | 1,208,889 | 2.572 | 4.009 |
| Input gates / 128 | 1,048,576 | 309,561 | 2.210 | 8.325 |

![language compute choices](report/figures/language_compute_choices.png)

On the identical 131K-character/four-pass screen, widening the gated model buys 0.014 bpc for 3.85× the total fitting arithmetic. Adding input gates at width 128 instead improves 0.057 bpc for 1.41% more arithmetic. This makes the allocation question quantitative: measure useful correction before spending broadly on width. These are finite, single-seed interventions, rather than a scaling law.

The fixed-capacity data comparison and fixed-data capacity comparison answer different questions. Equal passes and data do not imply equal compute. The pipeline checks finite learning, trained value blocks, complete work and source provenance before promotion. A development gain is not an official-test or frontier claim.

Count reference (Theory §§376–380): on the same 8,191 development targets, interpolated Kneser–Ney counts of the same fitting characters score 2.349 bpc at 131K and 2.007 at 1M, with one counting pass and no gradient work. Every completed earlier fit without count-carrying receivers from 2K to 1M characters is above this bar. The fixed-step estimator analyzed in Theory §377 has a variance floor; this is not an impossibility theorem for learned recurrent gates or short credit. It motivates testing count-carrying receivers with escape races and a learned base measure. Counts are reference predictors, not neural controls or a large-data comparison.

Precise clocks; float32 payloads; causal persistent state; 64-character credit horizon. Special functions, evaluation passes and physical traffic are separate from the arithmetic ledger. One seed, no statistical experts. Source: experiments/results/parallel_language.

## What establishes the larger advantage

The ambition is a common model family whose strongest mechanisms remain useful as tasks, data and capacity grow. Arithmetic and retrieval retain their demonstrated strengths in the consolidated implementation. The strongest language mixture is specialized; the generic backbone still needs to demonstrate competitive learned representations. Character and subword-token budgets must be distinguished.

| Objective | Decisive evidence |
| --- | --- |
| Generic language scaling | Train a learned event backbone that owns the prediction. Scale through declared data budgets with matched Transformer, recurrent and state-space references; record loss, capacity, complete training work, memory traffic, time and joules. |
| Preserve capabilities | Repeat established generalization and sample-efficiency results within the common model family, with task-appropriate depth and explicit resource accounting. |
| Strong real-event recognition | Accurate speech and event-camera decisions on complete held-out benchmarks; calibrated confidence and time-to-answer. |
| Learn routes and representations at scale | Reliable deep credit and useful counterfactual alternatives as width, depth, memory and data increase. |
| Efficient persistent operation | Maintain local state and pending messages across queries, preserving causal predictions while reducing repeated work. |
| Lower total energy at useful quality | Measure training and inference joules, memory traffic, latency and communication under declared hardware and quality targets. |

Success on these dimensions would turn the current task-level advantages into a broader foundation for frontier models. The project's distinctive resources—timing, local memory, hard selection and credit to alternatives—remain the guide for architecture and learning.

## Appendix A. Deep event recognition

The strongest completed speech result in the model family is **79.69%** on 512 private held-speaker utterances (selected single six-block prefix; trained with twelve blocks). Published official-test results below use a different partition; they are reference targets.

| Ours: private development configuration | Correct | Accuracy ↑ |
| --- | --- | --- |
| Original eight-layer parent | 370/512 | 72.27% |
| Parent + six-block residual; pass 3 | 408/512 | 79.69% |
| Single six-block temporal encoder; pass 0 | 406/512 | 79.30% |
| Calibrated six-block continuation | 400/512 | 78.12% |
| Bounded twelve-block encoder | 400/512 | 78.12% |
| Directional twelve-block encoder | 401/512 | 78.32% |
| Selected single six-block prefix; trained with twelve blocks | 408/512 | 79.69% |

![e143 temporal residual learning](report/figures/e143_temporal_residual_learning.png)

A six-block width-128 temporal encoder learns corrections while the inherited eight-layer parent stays frozen. It adds 395,814 parameters to the parent's 53,296. Signed modal states, nonlinear gates and residual vectors learn from all source identities and original event times before causal pooling. This is a larger parallel model, not fourteen sequential layers; completed-utterance supervision does not yet teach calibrated early answers. On 657 disjoint utterances from the same held speakers, accuracy improves from 70.47% to 77.63%.

| Published reference; official-test protocol | Reported accuracy ↑ |
| --- | --- |
| Ours: full official SHD comparison | Pending |
| EventSSM: asynchronous learned state-space layers | 95.9% |
| S7: input-dependent temporal state | 96.3% |

Our private sample uses training-file speakers 3/6; official test accuracy is unmeasured. The temporal residual uses three passes and a fresh optimizer; its larger capacity and budget are not a matched single-factor comparison. Its listed score selects the best private-development epoch; the curve shows all three. The original parent was fitted on 4,096 examples. Continuation timers exclude loading; the residual timer includes it. Both include evaluation and are not energy measurements. One seed. References: [EventSSM](https://arxiv.org/html/2404.18508v2), [S7](https://arxiv.org/html/2410.03464v1). These published scores were checked against the original papers; they are targets for a full official-test comparison, not scores on our private split.

## Appendix A (continued). Learned timing and transfer

**Learned delays contribute to the answer.** Restoring the hidden clocks to their initial values, with source vectors, state/value maps and the trained classifier retained, loses 13 correct answers. Timing changes the temporal interactions used by the representation; it performs computation.

![e145 learned timing and transfer](report/figures/e145_learned_timing_and_transfer.png)

| Ours: same trained readout; subsystem reset | Correct | Accuracy ↑ |
| --- | --- | --- |
| All learned parameters retained | 408/512 | 79.69% |
| Only hidden clocks reset | 395/512 | 77.15% |
| Only source vectors reset | 373/512 | 72.85% |
| Only state stack reset | 358/512 | 69.92% |
| Only decay/frequency parameters reset | 407/512 | 79.49% |

The temporal stack and source vectors also learn useful coordinated representations. Resetting the stack loses 50 correct answers; resetting sources loses 35. Decay/frequency resets change one decision. These changes depend on the fitted solution's coordination; their effects cannot be added or treated as a matched comparison of retrained architectures.

The 657-utterance audit is disjoint from fitting and the development sample and uses the selected checkpoint unchanged. It gains 62 correct answers and loses 15, for a net improvement of 47. Both samples use the same two held training speakers. Official-test and additional-speaker generalization are the next evaluation targets.

### A stronger mathematical account of routing

An affine packet summary can preserve both the final state and the average of raw temporal states, including their teachers. This permits richer pooling before expensive nonlinear maps. A second derivation shows why many almost-equal delays may offer little usable choice: their effects point in nearly the same direction. Diverse payloads and temporal modes, sufficient delay spread and downstream visibility determine useful route reserve.

One exploratory run: 449,110 total parameters, inherited parent plus a trained six-block encoder. Three passes use about 33 minutes including preparation/evaluation and 1.79 GiB peak RSS; the guarded host retains at least 10,361 MiB sampled available memory. Complete physical work and joules remain unmeasured. Formal statements and numerical contracts are in THEORY §§226–236.

## Appendix A (continued). One temporal encoder

A six-block temporal encoder retains nearly all the combined model's development accuracy through one ordinary query head. Deployment removes the frozen parent: 395,814 parameters replace 449,110. Modal states, gated vector messages and winning delays remain. Its weights inherit earlier encoder training; combined teacher predictions are used only to initialize the head.

| Ours: private development configuration | Correct | Accuracy ↑ | NLL ↓ |
| --- | --- | --- | --- |
| Combined model: parent + temporal correction | 408/512 | 79.69% | 0.684 |
| Single encoder: clean head, before continuation | 374/512 | 73.05% | 0.818 |
| Single encoder: paired head, selected pass zero | 406/512 | 79.30% | 0.631 |
| Single encoder: selected trained prefix | 408/512 | 79.69% | 0.625 |

![e152 single encoder learning](report/figures/e152_single_encoder_learning.png)

Fitting the head on clean and transformed speech improves held accuracy by 32 answers with the temporal features frozen. Its covariance penalty suppresses class-visible nuisance variation. The right panel compares the heads on the same features; the left shows all subsequent unrestricted encoder passes. The selected trained prefix reaches 518/657 (78.84%) versus 510/657 (77.63%) for the combined model on the reused disjoint audit. Audit labels do not choose the checkpoint.

Training the directional twelve-block extension, then selecting its six-block prefix on development, retains the combined model's 408 correct answers with lower NLL and 395,814 deployed parameters. The extra training blocks are removed after their learned contributions reduce held accuracy. This improves deployment quality/work; it does not establish a positive deep-block accuracy gain.

Seed 6, private train-file speakers 3/6; official-test parity remains unmeasured. Head fitting uses 6,144 unique fitting utterances: one clean view for the first arm, clean plus one transformed view for the paired arm. The arms also change regularization and use three/two encoder passes respectively, so total budgets are not matched. All continuation epochs are plotted; selection uses development accuracy, then NLL. Extra teacher/cache/head work and inherited fitting must be charged. Modal/vector maps are locally dense; no empty ticks or event-pair attention are added. The selected prefix additionally inherits the full twelve-block fitting pass; pruning does not erase that training cost. Prefix/full choice is post-hoc private-development selection. Formulae and numerical checks: THEORY §§237–264. One CPU forward evaluation takes 11.90 s for the selected prefix versus 23.71 s for the combined model. Packing/query included, loading excluded; one timing observation, not joules.

## Appendix A (continued). Making depth useful

Identity growth preserves the classifier and old teachers while added output maps receive label credit. They must also change useful features. A tightly bounded twelve-block extension learns weights but changes no audit decisions when its six appended blocks are removed.

| Ours: one matched fitting pass | Parameters | Fit accuracy ↑ | Private accuracy ↑ | NLL ↓ |
| --- | --- | --- | --- | --- |
| Six-block control | 395,814 | 90.72% | 78.12% | 0.639 |
| Twelve blocks: bounded outputs | 696,888 | 90.72% | 78.12% | 0.639 |
| Twelve blocks: directional units | 696,120 | 90.77% | 78.32% | 0.644 |

![e164 depth use and work](report/figures/e164_depth_use_and_work.png)

Directional conditioning normalizes temporal-state features before their output map. An invertible coordinate change rescales classifier-sensitive directions and preserves hidden null directions for later computation. The fixed transform folds into an ordinary map at deployment. Initial outputs and old teachers remain exact; fitting replays verify the actual proposed update.

The directional model reaches 509/657 on the reused audit. Removing its six appended blocks changes 11 predictions: 0 are correct only with the blocks and 9 only without them. This measures fitted contribution with the trained prefix/head retained; it is not a retrained architecture comparison.

All arms inherit the paired-head checkpoint and use 6,144 fitting utterances, the same order, channel/time transformations and one encoder pass. Old-group LR is 0.0000203125; directional new groups use 0.0001953125 from fitting-only replay. Changed normalization and update coordinates form one intervention. New blocks initially add 36 ms latency; labels supervise completed untimed utterances. Audit reuse is explicit; official-test parity remains unmeasured. CPU points are one warmed observation per model, packing/query included and loading excluded; energy is unmeasured. Weight-coordinate folding, all source work and added depth must be charged during training. Local maps remain dense, with no empty ticks or event-pair attention. Theory §§249–264.

## Appendix B. Breadth of the common implementation

These small development screens test one implementation across tasks. Text and market variants include separately fitted statistical evidence. Ours denotes Sleeping Machines; TF is the saved Transformer. Accuracy improves upward; prediction loss and FLOPs improve downward.

| Task / quality direction | Ours: quality | TF: quality | Ours: inference MFLOPs ↓ | TF: inference MFLOPs ↓ |
| --- | --- | --- | --- | --- |
| Text8<br/>bpc ↓ | 2.915 bpc | 3.729 bpc | 1.26 | 1.84 |
| Market event prediction<br/>nats/event ↓ | 3.823 nats/event | 4.208 nats/event | 1.26 | 1.84 |
| Temporal composition<br/>accuracy ↑ | 97.3% | 90.2% | 0.29 | 0.38 |
| MNIST<br/>accuracy ↑ | 75.8% | 69.9% | 1.66 | 2.57 |
| Event-camera gestures<br/>accuracy ↑ | 59.1% | 15.9% | 23.99 | 124.23 |

![breadth work ratios](report/figures/breadth_work_ratios.png)

Ours uses eight layers and TF two, both width 32, with the same neural-fitting examples, encoding, objective and eight epochs. Inference counts maps/scans (ours) and maps/attention (TF): two FLOPs per multiply-add, excluding padding, scalar nonlinearities and expert preparation. Training includes backward, clipping, Adam and our evidence/calibration; its ledger follows.

**Lower core inference work in every screen.** Gestures use **5.18× less** with 59.1% versus 15.9% accuracy. That 44-query screen has an underfitting TF control; a general vision claim requires complete benchmarks and stronger references.

**Why training can cost more:** the race core evaluates all three candidate vector payloads during training, versus only the winner during inference. Its eight layers also exceed the reference's two. With short contexts, that work outweighs the saved attention cost; these rows do not show a training efficiency advantage. On the longer event-camera prefixes, the common model's estimated total uses 49.0% of the reference training arithmetic, including calibration.

Seed 6; neural fit/development counts: text 2,048/256, market 512/256, temporal 1,024/256, MNIST 1,024/256, gestures 88/44. The common text model also has a separately fitted 32,768-character evidence bank; market evidence is fitted on a prior day. The references have no such bank. MNIST uses pooled training-set images; gestures use first-second prefixes and disjoint users. Batch 16, or four for gestures. No official real-data test. Market fixed evidence: 3.670 nats/event.

## Appendix B (continued). Total training cost

The ledger estimates the entire completed fitting budget for each reported model. It includes prediction and loss, backpropagation, gradient clipping and Adam across all eight epochs. The common model also pays for its evidence bank and initial readout calibration. These models were trained from initialization; there is no inherited neural fitting to omit.

![e172 complete training work](report/figures/e172_complete_training_work.png)

| Model/task | Forward + loss | Backward | Clip | Adam | Evidence + calibration | Total |
| --- | --- | --- | --- | --- | --- | --- |
| Language: Ours | 62.633 | 130.873 | 0.165 | 0.658 | 2.930 | 197.26 |
| Language: TF | 31.246 | 63.990 | 0.083 | 0.334 | 0 | 95.65 |
| Market: Ours | 15.657 | 32.716 | 0.041 | 0.164 | 0.733 | 49.31 |
| Market: TF | 7.811 | 15.996 | 0.020 | 0.081 | 0 | 23.91 |
| Temporal: Ours | 7.725 | 16.546 | 0.081 | 0.324 | 0.364 | 25.04 |
| Temporal: TF | 4.773 | 9.540 | 0.041 | 0.162 | 0 | 14.52 |
| Mnist: Ours | 40.394 | 84.376 | 0.083 | 0.333 | 1.886 | 127.07 |
| Mnist: TF | 33.026 | 68.489 | 0.049 | 0.197 | 0 | 101.76 |
| Dvs: Ours | 51.580 | 107.386 | 0.028 | 0.112 | 2.404 | 161.51 |
| Dvs: TF | 99.023 | 230.261 | 0.014 | 0.057 | 0 | 329.35 |

All table values are estimated GFLOPs for the whole fitting run, not per query. A multiply-add counts as two operations. Forward/loss and backward use the saved E172 four-query operator trace scaled by recorded map/scan work. Ours uses its logged candidate-map/scan counts. Reference padding is reconstructed from all fitting prefix lengths, the original shuffle seed and batch sizes, including fused attention products. Clipping and Adam are charged once per original step. Other arithmetic and the small calibration eigensolver are estimates.

Event-target arithmetic excludes padding and simulator dispatch/allocation; required candidate maps, losing-value teaching, scans and learning remain charged. References use the same FLOP convention. Evaluation, search, encoding, special functions, integer/index work, comparisons and memory traffic are outside these totals. These single-seed screens have different depths/quality and do not measure event hardware, matched-quality cost or energy. Ledger: [estimate_training_work.py](experiments/estimate_training_work.py).

## Appendix B (continued). Completed language references

The Transformer and LSTM benchmarks have already been run. Their completed result files remain in the repository and are reused as reference targets for the full learned-event benchmark. Lower bits per character (bpc) means better prediction.

| Model | Fitting characters / passes | Test bpc ↓ | Full training FLOPs ↓ |
| --- | --- | --- | --- |
| Ours: learned event-state model (planned) | 10M / four passes | Pending | Pending |
| LSTM; width 512, one recurrent layer | 10M / six passes | 1.799 | 432.59T |
| Transformer; width 256, four layers | 10M / four passes | 1.908 | 888.78T |
| LSTM; width 512, one recurrent layer | 90M / 6 passes | 1.661 | 3.89P |
| Transformer; width 256, four layers | 90M / 4 passes | 1.604 | 8.00P |

The 10M references score exactly the same 999,999 text8 targets in [95M,96M), with frozen validation-selected weights and cold initial context. The 90M LSTM uses the same test interval and its saved recurrent scoring protocol. All use the historical 27-character alphabet and 200,000-character validation selection; data budgets, capacities and fitting passes differ.

Training estimates include every fitting step, forward/loss, backpropagation, gradient clipping and Adam. Backward is approximated as twice forward; multiply-add counts as two FLOPs. G/T/P mean billion/trillion/quadrillion. Validation/test evaluation, memory traffic and runtime are outside these arithmetic totals.

### Ours: learned-language benchmark status

The planned comparison uses 10M fitting characters, four passes, 200,000 validation characters and the same 1M test interval. Priority is the integrated sparse/timed architecture documented in the next appendix. Completed dense-carrier fits remain diagnostic controls; their quality does not establish sparse-model performance. Numerical contracts and progressively larger integrated development fits precede promotion. The aligned full-test result and its fitting work remain pending. The preserved 28,403-parameter pilot's 3.351 development bpc uses a smaller fitting budget and different split; it is not a comparable test result.

Earlier 1M-character references also remain saved: LSTM **2.179** and Transformer **2.367 test bpc**, each with twenty fitting passes. The separate count/copy baseline and the cross-task Transformer/retrieval LSTM comparisons remain in their labeled sections and Appendix B.

Saved evidence: [10M LSTM aligned result](experiments/results/e174/aligned_lstm_10m_20260930.json); [10M Transformer aligned result](experiments/results/e174/aligned_tf_10m_20260930.json); [90M LSTM saved result](experiments/results/aws_20260929/aws_e64_lstm_D90M_baseline_20260929/lstm_D90000000_s512_p6_dr0.1_v.json); [90M TF saved result](experiments/results/aws_20260929/aws_e64_tf_D90M_baseline_rss6g_20260930/tf_D90000000_s256_L4_p4_dr0.1_v.json). An earlier 90M Transformer attempt was interrupted by its RSS watchdog before producing a completed test result; its provenance is preserved.

## Appendix B. New completed AWS language evidence

The six-block sparse receiver model reaches 3.106 development bpc on 32K fitting characters. It learns meaningful prediction, but this is the older single-head, observed-character-pool variant, not the newer eight-block native/content-gated reception model. The large dense controls are completed reference targets with much better held-out quality and much larger fitting budgets.

| Model | Fit chars / passes | Dev bpc ↓ | Test bpc ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ |
| --- | --- | --- | --- | --- | --- |
| Ours sparse d32/L6 | 32,768/4 | 3.106 | Not scored | 114.25 | 0.872 |
| LSTM512 | 90M/6 | 1.616 | 1.661 | 3,893,396.04 | 7.210 |
| Transformer256/L4 | 90M/4 | 1.580 | 1.604 | 8,000,253.35 | 22.223 |

| Model | Parameters | Guarded wall h | Peak RSS MiB | Infer MFLOPs / char ↓ |
| --- | --- | --- | --- | --- |
| Ours sparse | 1,388,871 | 0.39 | 558.4 | 0.0885 |
| LSTM512 | 1,199,323 | 10.64 | 2246.9 | 2.4024 |
| Transformer256/L4 | 3,238,427 | 31.18 | 3822.5 | 14.8104 |

All rows use the same units and whole-fit/per-target denominators within each column. Sparse work is a representative full-step arithmetic estimate with unit-weight special functions; dense work uses shapes and backward = twice forward, including clipping/Adam. Both exclude evaluation and traffic. Dense inference uses the saved shape convention, including Transformer window overlap; sparse inference is a winner-only trace. Guarded wall and RSS include different simulator/runtime overheads and are not energy measurements.

The sparse development set has 8,191 targets; dense selection uses 200,000 validation characters and the saved 1M test interval. Neither their development scores nor their fitting budgets are matched. Raw resource gaps cannot establish comparable-quality or iso-FLOP supremacy. The interrupted Transformer attempt remains preserved; only its completed retry appears as a quality point. All completed points, including the 90M Transformer, are retained in the common quality/work and inference figures.

## Appendix B. AWS hierarchy depth: a useful constraint

These are older race-network diagnostics on the fixed depth-three Random Hierarchy Model. They test composition and depth, not the newer integrated language/event construction. Every shown model uses width 200, 64K fitting examples, ten passes, architecture seed 0 and rule seed 0.

![aws hierarchy depth](report/figures/aws_hierarchy_depth.png)

| Model | Depth | Final held-out accuracy | Guarded wall s | Peak RSS MiB |
| --- | --- | --- | --- | --- |
| Ours plain race | 1 | 70.12% | 70.4 | 129.3 |
| Ours plain race | 2 | 83.72% | 227.4 | 202.4 |
| Ours plain race | 3 | 85.06% | 382.2 | 249.7 |
| Ours plain race | 4 | 84.12% | 536.5 | 252.1 |
| Ours residual-2 | 2 | 72.72% | 266.1 | 217.4 |
| Ours residual-2 | 3 | 74.54% | 610.4 | 369.3 |
| Ours residual-2 | 4 | 72.56% | 1094.7 | 574.0 |

Plain depth 1/2/3/4 gives 70.12/83.72/85.06/84.12%: depth helps to three blocks, then regresses slightly. The completed residual-2 depth 2/3/4 variants give 72.72/74.54/72.56%, below their plain counterparts. Depth-four residual accuracy rises throughout its ten passes; its longer-budget convergence is untested. This constrains the tested implementation and budget, not all residual paths.

Only completed provenance files produce rows. Plain and residual depth-four runs are distinct. Final epoch is reported; evaluation curves are visible throughout fitting, so this is exploratory held-out evidence, not independent confirmation. Single rule/model seed; no 64K matched dense-control or complete FLOP/energy supremacy is inferred.

## Appendix B. AWS early temporal mechanism screen

Small completed integrated pilots from the committed fast matrix. These answer mechanism questions before larger scaling; their development budgets differ from the saved main language and native512-target results. Contracts and accounting smokes are excluded.

![aws fast screen temporal](report/figures/aws_fast_screen_temporal.png)

| Model | Fit / passes / dev | Dev quality | Whole fit GFLOPs | Fit MFLOPs / target | Infer MFLOPs / target |
| --- | --- | --- | --- | --- | --- |
| Ours timing full | 128/4/64 | 78.12% | 0.265 | 0.518 | 0.1101 |
| Ours timing rank | 128/4/64 | 79.69% | 0.265 | 0.518 | 0.1100 |
| Ours order full | 128/4/64 | 51.56% | 0.265 | 0.518 | 0.1104 |
| Ours order pathwise | 128/4/64 | 51.56% | 0.263 | 0.514 | 0.1104 |

| Model | Parameters | Receivers/commits/matches per event | Wall s | Seed |
| --- | --- | --- | --- | --- |
| Ours timing full | 42,898 | 128/16/32 | 791.8 | 6 |
| Ours timing rank | 42,898 | 128/16/32 | 799.8 | 6 |
| Ours order full | 42,932 | 128/16/32 | 789.3 | 6 |
| Ours order pathwise | 42,932 | 128/16/32 | 768.4 | 6 |

FLOPs count MAC as two and special functions once; all losing-value credit and Adam are charged. Temporal inference per query includes intervening input events; language per target is per character. Capacity/activity counts are per event, not per query. CPU simulation/audit overhead, RNG, traffic and physical energy are separate; AWS wall observations may include authorized CPU concurrency. Temporal tasks and altered source counts are explicitly named; different tasks do not form a single accuracy scaling curve. Single-seed development evidence, not supremacy. Paired independent seeds and frozen held-out confirmation precede benchmark promotion.

Interpretation: observed-time accuracy 78.12% versus refitted rank-time 79.69% does not yet demonstrate an elapsed-time advantage. State-clearing and stretched-gap probes are diagnostic interventions, not refitted controls.

Next experimental questions: state clearing and stretched silent gaps damage order predictions; private source rules lose training exposure as capacity grows. The next integrated battery tests protected memory, shared rules with private state, and paired timing whose labels cannot be inferred from rank alone.

## Appendix B. AWS early temporal mechanism screen

Small completed integrated pilots from the committed fast matrix. These answer mechanism questions before larger scaling; their development budgets differ from the saved main language and native512-target results. Contracts and accounting smokes are excluded.

![aws fast screen temporal](report/figures/aws_fast_screen_temporal.png)

| Model | Fit / passes / dev | Dev quality | Whole fit GFLOPs | Fit MFLOPs / target | Infer MFLOPs / target |
| --- | --- | --- | --- | --- | --- |
| Ours order sources16 | 128/4/64 | 48.44% | 0.280 | 0.548 | 0.1103 |
| Ours order sources64 | 128/4/64 | 32.81% | 0.341 | 0.666 | 0.1104 |
| Ours order shallow control | 128/4/64 | 51.56% | 0.134 | 0.262 | 0.0569 |

| Model | Parameters | Receivers/commits/matches per event | Wall s | Seed |
| --- | --- | --- | --- | --- |
| Ours order sources16 | 157,940 | 512/16/32 | 829.1 | 6 |
| Ours order sources64 | 617,972 | 2048/16/32 | 980.5 | 6 |
| Ours order shallow control | 21,684 | 64/8/16 | 403.3 | 6 |

FLOPs count MAC as two and special functions once; all losing-value credit and Adam are charged. Temporal inference per query includes intervening input events; language per target is per character. Capacity/activity counts are per event, not per query. CPU simulation/audit overhead, RNG, traffic and physical energy are separate; AWS wall observations may include authorized CPU concurrency. Temporal tasks and altered source counts are explicitly named; different tasks do not form a single accuracy scaling curve. Single-seed development evidence, not supremacy. Paired independent seeds and frozen held-out confirmation precede benchmark promotion.

Capacity interpretation: commits and matches remain fixed while available state grows, but fixed queries reduce per-source training exposure. The 64-source development set has one population; its collapsed bootstrap interval is not useful uncertainty.

## Appendix B. AWS early language mechanism screen

Small completed integrated pilots from the committed fast matrix. These answer mechanism questions before larger scaling; their development budgets differ from the saved main language and native512-target results. Contracts and accounting smokes are excluded.

![aws fast screen language](report/figures/aws_fast_screen_language.png)

| Model | Fit / passes / dev | Dev quality | Whole fit GFLOPs | Fit MFLOPs / target | Infer MFLOPs / target |
| --- | --- | --- | --- | --- | --- |
| Ours native | 512/4/1023 | 4.2003 | 0.290 | 0.142 | 0.0301 |
| Ours reception | 512/4/1023 | 4.1949 | 0.325 | 0.159 | 0.0348 |
| Ours late | 512/4/1023 | 4.1962 | 0.319 | 0.156 | 0.0341 |
| Ours waiting | 512/4/1023 | 4.1955 | 0.304 | 0.149 | 0.0318 |

| Model | Parameters | Receivers/commits/matches per event | Wall s | Seed |
| --- | --- | --- | --- | --- |
| Ours native | 14,971 | 32/16/32 | 166.1 | 6 |
| Ours reception | 16,123 | 32/16/32 | 239.2 | 6 |
| Ours late | 16,123 | 32/16/32 | 205.1 | 6 |
| Ours waiting | 16,123 | 32/16/32 | 207.6 | 6 |

FLOPs count MAC as two and special functions once; all losing-value credit and Adam are charged. Temporal inference per query includes intervening input events; language per target is per character. Capacity/activity counts are per event, not per query. CPU simulation/audit overhead, RNG, traffic and physical energy are separate; AWS wall observations may include authorized CPU concurrency. Temporal tasks and altered source counts are explicitly named; different tasks do not form a single accuracy scaling curve. Single-seed development evidence, not supremacy. Paired independent seeds and frozen held-out confirmation precede benchmark promotion.

## Appendix B. Protected state/shared rules: order

Completed integrated pilots only. Protected modes retain information during silence; temporal modes still evolve. Shared learned rules retain private addressed state and remove private source embeddings. Paired timing keeps marks/order identical while labels differ; rank-only prediction has an exact 50% paired ceiling under coupled noise.

| Construction | Dev accuracy% | Dev NLL | Whole fit GFLOPs | Fit MFLOPs/query | Infer MFLOPs/query |
| --- | --- | --- | --- | --- | --- |
| Ours S64 private/P0/observed/s6 | 95.02 | 0.1889 | 1.364 | 0.666 | 0.110 |
| Ours S64 private/P0/observed/s7 | 98.34 | 0.0558 | 1.364 | 0.666 | 0.110 |
| Ours S64 private/P0/observed/s8 | 98.83 | 0.0430 | 1.364 | 0.666 | 0.110 |
| Ours S64 shared/P0/observed/s6 | 99.90 | 0.0279 | 1.054 | 0.515 | 0.110 |
| Ours S64 shared/P0/observed/s7 | 99.90 | 0.0869 | 1.054 | 0.515 | 0.110 |

| Construction | Fit/dev/passes | Parameters | State slots | Updates/scores per event |
| --- | --- | --- | --- | --- |
| Ours S64 private/P0/observed/s6 | 512/1024/4 | 616,964 | 2048 | 16/32 |
| Ours S64 private/P0/observed/s7 | 512/1024/4 | 616,964 | 2048 | 16/32 |
| Ours S64 private/P0/observed/s8 | 512/1024/4 | 616,964 | 2048 | 16/32 |
| Ours S64 shared/P0/observed/s6 | 512/1024/4 | 14,180 | 2048 | 16/32 |
| Ours S64 shared/P0/observed/s7 | 512/1024/4 | 14,180 | 2048 | 16/32 |

Exact full fitting includes producer graphs, losing proposals, backward, clipping and Adam; specials have unit weight. Independent population/pair uncertainty is distinct from seed uncertainty. Protected-prefix initialization also removes faster initial temporal modes; any timing change is not isolated spectral evidence. Scope remains synthetic pilot quality, not physical energy.

## Appendix B. Protected state/shared rules: order

Completed integrated pilots only. Protected modes retain information during silence; temporal modes still evolve. Shared learned rules retain private addressed state and remove private source embeddings. Paired timing keeps marks/order identical while labels differ; rank-only prediction has an exact 50% paired ceiling under coupled noise.

| Construction | Dev accuracy% | Dev NLL | Whole fit GFLOPs | Fit MFLOPs/query | Infer MFLOPs/query |
| --- | --- | --- | --- | --- | --- |
| Ours S64 shared/P0/observed/s8 | 97.95 | 0.0723 | 1.054 | 0.515 | 0.110 |
| Ours S16 shared/P0/observed/s7 | 44.53 | 1.3721 | 0.262 | 0.511 | 0.110 |
| Ours S16 shared/P0/observed/s8 | 70.31 | 0.8630 | 0.262 | 0.512 | 0.110 |
| Ours S16 shared/P2/observed/s7 | 26.17 | 1.3999 | 0.253 | 0.494 | 0.107 |
| Ours S16 shared/P2/observed/s8 | 51.56 | 1.0933 | 0.253 | 0.494 | 0.107 |

| Construction | Fit/dev/passes | Parameters | State slots | Updates/scores per event |
| --- | --- | --- | --- | --- |
| Ours S64 shared/P0/observed/s8 | 512/1024/4 | 14,180 | 2048 | 16/32 |
| Ours S16 shared/P0/observed/s7 | 128/256/4 | 14,180 | 512 | 16/32 |
| Ours S16 shared/P0/observed/s8 | 128/256/4 | 14,180 | 512 | 16/32 |
| Ours S16 shared/P2/observed/s7 | 128/256/4 | 13,988 | 512 | 16/32 |
| Ours S16 shared/P2/observed/s8 | 128/256/4 | 13,988 | 512 | 16/32 |

Exact full fitting includes producer graphs, losing proposals, backward, clipping and Adam; specials have unit weight. Independent population/pair uncertainty is distinct from seed uncertainty. Protected-prefix initialization also removes faster initial temporal modes; any timing change is not isolated spectral evidence. Scope remains synthetic pilot quality, not physical energy.

## Appendix B. Protected state/shared rules: order

Completed integrated pilots only. Protected modes retain information during silence; temporal modes still evolve. Shared learned rules retain private addressed state and remove private source embeddings. Paired timing keeps marks/order identical while labels differ; rank-only prediction has an exact 50% paired ceiling under coupled noise.

| Construction | Dev accuracy% | Dev NLL | Whole fit GFLOPs | Fit MFLOPs/query | Infer MFLOPs/query |
| --- | --- | --- | --- | --- | --- |
| Ours S16 private/P0/observed/s7 | 25.78 | 1.4432 | 0.280 | 0.548 | 0.110 |
| Ours S16 private/P0/observed/s8 | 42.97 | 1.3150 | 0.281 | 0.548 | 0.110 |
| Ours S16 private/P0/observed/s6 | 75.39 | 0.7592 | 0.280 | 0.548 | 0.110 |
| Ours S16 private/P0/observed/s7 | 51.17 | 1.0313 | 0.280 | 0.548 | 0.110 |
| Ours S16 private/P0/observed/s8 | 69.14 | 0.8689 | 0.280 | 0.548 | 0.110 |

| Construction | Fit/dev/passes | Parameters | State slots | Updates/scores per event |
| --- | --- | --- | --- | --- |
| Ours S16 private/P0/observed/s7 | 128/256/4 | 157,940 | 512 | 16/32 |
| Ours S16 private/P0/observed/s8 | 128/256/4 | 157,940 | 512 | 16/32 |
| Ours S16 private/P0/observed/s6 | 128/256/4 | 157,700 | 512 | 16/32 |
| Ours S16 private/P0/observed/s7 | 128/256/4 | 157,700 | 512 | 16/32 |
| Ours S16 private/P0/observed/s8 | 128/256/4 | 157,700 | 512 | 16/32 |

Exact full fitting includes producer graphs, losing proposals, backward, clipping and Adam; specials have unit weight. Independent population/pair uncertainty is distinct from seed uncertainty. Protected-prefix initialization also removes faster initial temporal modes; any timing change is not isolated spectral evidence. Scope remains synthetic pilot quality, not physical energy.

## Appendix B. Protected state/shared rules: order

Completed integrated pilots only. Protected modes retain information during silence; temporal modes still evolve. Shared learned rules retain private addressed state and remove private source embeddings. Paired timing keeps marks/order identical while labels differ; rank-only prediction has an exact 50% paired ceiling under coupled noise.

| Construction | Dev accuracy% | Dev NLL | Whole fit GFLOPs | Fit MFLOPs/query | Infer MFLOPs/query |
| --- | --- | --- | --- | --- | --- |
| Ours S16 shared/P0/observed/s6 | 40.23 | 1.2576 | 0.262 | 0.512 | 0.110 |
| Ours S16 shared/P0/observed/s7 | 32.03 | 1.4567 | 0.262 | 0.512 | 0.110 |
| Ours S16 shared/P0/observed/s8 | 25.00 | 1.5001 | 0.262 | 0.512 | 0.110 |
| Ours S16 private/P0/observed/s6 | 44.14 | 1.1194 | 0.280 | 0.548 | 0.110 |
| Ours S16 private/P2/observed/s6 | 44.92 | 1.2147 | 0.271 | 0.529 | 0.107 |

| Construction | Fit/dev/passes | Parameters | State slots | Updates/scores per event |
| --- | --- | --- | --- | --- |
| Ours S16 shared/P0/observed/s6 | 128/256/4 | 14,420 | 512 | 16/32 |
| Ours S16 shared/P0/observed/s7 | 128/256/4 | 14,420 | 512 | 16/32 |
| Ours S16 shared/P0/observed/s8 | 128/256/4 | 14,420 | 512 | 16/32 |
| Ours S16 private/P0/observed/s6 | 128/256/4 | 157,940 | 512 | 16/32 |
| Ours S16 private/P2/observed/s6 | 128/256/4 | 155,828 | 512 | 16/32 |

Exact full fitting includes producer graphs, losing proposals, backward, clipping and Adam; specials have unit weight. Independent population/pair uncertainty is distinct from seed uncertainty. Protected-prefix initialization also removes faster initial temporal modes; any timing change is not isolated spectral evidence. Scope remains synthetic pilot quality, not physical energy.

## Appendix B. Protected state/shared rules: order

Completed integrated pilots only. Protected modes retain information during silence; temporal modes still evolve. Shared learned rules retain private addressed state and remove private source embeddings. Paired timing keeps marks/order identical while labels differ; rank-only prediction has an exact 50% paired ceiling under coupled noise.

| Construction | Dev accuracy% | Dev NLL | Whole fit GFLOPs | Fit MFLOPs/query | Infer MFLOPs/query |
| --- | --- | --- | --- | --- | --- |
| Ours S16 shared/P0/observed/s6 | 75.39 | 0.8327 | 0.262 | 0.511 | 0.110 |
| Ours S16 shared/P2/observed/s6 | 56.25 | 1.0359 | 0.253 | 0.494 | 0.107 |
| Ours S4 private/P0/observed/s6 | 54.30 | 0.9787 | 0.265 | 0.518 | 0.110 |
| Ours S4 private/P2/observed/s6 | 47.27 | 1.1060 | 0.256 | 0.500 | 0.107 |
| Ours S4 shared/P0/observed/s6 | 70.70 | 0.8556 | 0.261 | 0.511 | 0.110 |

| Construction | Fit/dev/passes | Parameters | State slots | Updates/scores per event |
| --- | --- | --- | --- | --- |
| Ours S16 shared/P0/observed/s6 | 128/256/4 | 14,180 | 512 | 16/32 |
| Ours S16 shared/P2/observed/s6 | 128/256/4 | 13,988 | 512 | 16/32 |
| Ours S4 private/P0/observed/s6 | 128/256/4 | 42,932 | 128 | 16/32 |
| Ours S4 private/P2/observed/s6 | 128/256/4 | 42,356 | 128 | 16/32 |
| Ours S4 shared/P0/observed/s6 | 128/256/4 | 14,180 | 128 | 16/32 |

Exact full fitting includes producer graphs, losing proposals, backward, clipping and Adam; specials have unit weight. Independent population/pair uncertainty is distinct from seed uncertainty. Protected-prefix initialization also removes faster initial temporal modes; any timing change is not isolated spectral evidence. Scope remains synthetic pilot quality, not physical energy.

## Appendix B. Protected state/shared rules: order

Completed integrated pilots only. Protected modes retain information during silence; temporal modes still evolve. Shared learned rules retain private addressed state and remove private source embeddings. Paired timing keeps marks/order identical while labels differ; rank-only prediction has an exact 50% paired ceiling under coupled noise.

| Construction | Dev accuracy% | Dev NLL | Whole fit GFLOPs | Fit MFLOPs/query | Infer MFLOPs/query |
| --- | --- | --- | --- | --- | --- |
| Ours S4 shared/P2/observed/s6 | 56.64 | 1.0367 | 0.252 | 0.493 | 0.107 |

| Construction | Fit/dev/passes | Parameters | State slots | Updates/scores per event |
| --- | --- | --- | --- | --- |
| Ours S4 shared/P2/observed/s6 | 128/256/4 | 13,988 | 128 | 16/32 |

Exact full fitting includes producer graphs, losing proposals, backward, clipping and Adam; specials have unit weight. Independent population/pair uncertainty is distinct from seed uncertainty. Protected-prefix initialization also removes faster initial temporal modes; any timing change is not isolated spectral evidence. Scope remains synthetic pilot quality, not physical energy.

## Appendix B. Protected state/shared rules: paired_timing

Completed integrated pilots only. Protected modes retain information during silence; temporal modes still evolve. Shared learned rules retain private addressed state and remove private source embeddings. Paired timing keeps marks/order identical while labels differ; rank-only prediction has an exact 50% paired ceiling under coupled noise.

| Construction | Dev accuracy% | Dev NLL | Whole fit GFLOPs | Fit MFLOPs/query | Infer MFLOPs/query |
| --- | --- | --- | --- | --- | --- |
| Ours S4 private/P0/observed/s7 | 90.23 | 0.3483 | 0.265 | 0.518 | 0.110 |
| Ours S4 private/P0/observed/s8 | 91.80 | 0.1989 | 0.265 | 0.518 | 0.110 |
| Ours S4 private/P0/rank/s7 | 50.00 | 0.7153 | 0.265 | 0.518 | 0.110 |
| Ours S4 private/P0/rank/s8 | 50.00 | 0.7151 | 0.265 | 0.518 | 0.110 |
| Ours S4 private/P0/observed/s6 | 95.31 | 0.1850 | 0.265 | 0.518 | 0.110 |

| Construction | Fit/dev/passes | Parameters | State slots | Updates/scores per event |
| --- | --- | --- | --- | --- |
| Ours S4 private/P0/observed/s7 | 128/256/4 | 42,898 | 128 | 16/32 |
| Ours S4 private/P0/observed/s8 | 128/256/4 | 42,898 | 128 | 16/32 |
| Ours S4 private/P0/rank/s7 | 128/256/4 | 42,898 | 128 | 16/32 |
| Ours S4 private/P0/rank/s8 | 128/256/4 | 42,898 | 128 | 16/32 |
| Ours S4 private/P0/observed/s6 | 128/256/4 | 42,898 | 128 | 16/32 |

Exact full fitting includes producer graphs, losing proposals, backward, clipping and Adam; specials have unit weight. Independent population/pair uncertainty is distinct from seed uncertainty. Protected-prefix initialization also removes faster initial temporal modes; any timing change is not isolated spectral evidence. Scope remains synthetic pilot quality, not physical energy.

## Appendix B. Protected state/shared rules: paired_timing

Completed integrated pilots only. Protected modes retain information during silence; temporal modes still evolve. Shared learned rules retain private addressed state and remove private source embeddings. Paired timing keeps marks/order identical while labels differ; rank-only prediction has an exact 50% paired ceiling under coupled noise.

| Construction | Dev accuracy% | Dev NLL | Whole fit GFLOPs | Fit MFLOPs/query | Infer MFLOPs/query |
| --- | --- | --- | --- | --- | --- |
| Ours S4 private/P0/rank/s6 | 50.00 | 0.7035 | 0.265 | 0.518 | 0.110 |
| Ours S4 private/P2/observed/s6 | 82.42 | 0.5131 | 0.256 | 0.500 | 0.106 |

| Construction | Fit/dev/passes | Parameters | State slots | Updates/scores per event |
| --- | --- | --- | --- | --- |
| Ours S4 private/P0/rank/s6 | 128/256/4 | 42,898 | 128 | 16/32 |
| Ours S4 private/P2/observed/s6 | 128/256/4 | 42,322 | 128 | 16/32 |

Exact full fitting includes producer graphs, losing proposals, backward, clipping and Adam; specials have unit weight. Independent population/pair uncertainty is distinct from seed uncertainty. Protected-prefix initialization also removes faster initial temporal modes; any timing change is not isolated spectral evidence. Scope remains synthetic pilot quality, not physical energy.

## Appendix B. Native replication scope

Shared S16/P0 order replications: seed7: 44.53%, seed8: 70.31%. The original seed6 screen was75.39%; training-seed variation remains material. Private S16 replication controls and fresh-population confirmation are required for the paired sharing claim; a completed shared-only score cannot pass that gate.

Observed-time paired replications: seed7: 90.23%, seed8: 91.80%. Original seed6 was95.31%, with the exact rank-only paired ceiling50%. These reuse the development distribution and were chosen after seed6; they are training-seed evidence, not independent confirmation.

8 of8 reserved replication pilots are complete in this checkout. Every completed seed is listed in the preceding common-unit tables; pending cells carry no score. The frozen AWS replication/confirmation chain owns the remaining work; no local duplicates.

## Appendix B. Native frozen confirmation

Completed 1,024-query synthetic holdout evaluations only. All fitted seeds and matched controls remain visible; selected checkpoints use development NLL before confirmation. Existing seed6 and AWS-replication weights are reused with their original whole fitting work charged.

| Construction | Dev accuracy % | Holdout accuracy % | Holdout NLL | Whole fit GFLOPs | Fitting lineage |
| --- | --- | --- | --- | --- | --- |
| Ours: paired timing observed/seed 6 | 95.31 | 93.07 | 0.2230 | 0.265 | Reused; charged |
| Ours: paired timing observed/seed 7 | 90.23 | 86.52 | 0.3979 | 0.265 | Reused; charged |
| Ours: paired timing observed/seed 8 | 91.80 | 86.04 | 0.3296 | 0.265 | Reused; charged |
| Ours: paired timing rank/seed 6 | 50.00 | 50.00 | 0.7039 | 0.265 | Reused; charged |
| Ours: paired timing rank/seed 7 | 50.00 | 50.00 | 0.7162 | 0.265 | Reused; charged |
| Ours: paired timing rank/seed 8 | 50.00 | 50.00 | 0.7140 | 0.265 | Reused; charged |

Primary gains require the complete three-seed crossed population/pair analysis with correction across two contrasts. Partial scores cannot pass a gate. Synthetic mechanism confirmation is distinct from time-aware dense controls, real-data supremacy and physical energy.

## Appendix B. Native frozen confirmation

Completed 1,024-query synthetic holdout evaluations only. All fitted seeds and matched controls remain visible; selected checkpoints use development NLL before confirmation. Existing seed6 and AWS-replication weights are reused with their original whole fitting work charged.

| Construction | Dev accuracy % | Holdout accuracy % | Holdout NLL | Whole fit GFLOPs | Fitting lineage |
| --- | --- | --- | --- | --- | --- |
| Ours: order S16 shared/seed 6 | 75.39 | 79.59 | 0.8253 | 0.262 | Reused; charged |
| Ours: order S16 shared/seed 7 | 44.53 | 41.80 | 1.3791 | 0.262 | Reused; charged |
| Ours: order S16 shared/seed 8 | 70.31 | 70.80 | 0.8565 | 0.262 | Reused; charged |
| Ours: order S16 private/seed 6 | 44.14 | 47.75 | 1.1264 | 0.280 | Reused; charged |
| Ours: order S16 private/seed 7 | 25.78 | 31.15 | 1.4322 | 0.280 | New fit |
| Ours: order S16 private/seed 8 | 42.97 | 35.84 | 1.3393 | 0.281 | New fit |

Primary gains require the complete three-seed crossed population/pair analysis with correction across two contrasts. Partial scores cannot pass a gate. Synthetic mechanism confirmation is distinct from time-aware dense controls, real-data supremacy and physical energy.

## Appendix B. Persistent-write credit diagnosis

A frozen selected native checkpoint is replayed with alternative delivered content, alternative persistent write, and both. The exact four-corner decomposition separates those effects from their interaction and the current message-linearization residual. The forward architecture and fitted weights remain unchanged.

| Ours: frozen audit component | Mean absolute loss effect |
| --- | --- |
| Full alternative branch | 0.020980 |
| Persistent-write effect | 0.012694 |
| Delivered-value linearization residual | 0.000288 |
| Delivery/write interaction | 0.000066 |

The two opposed directions in the earlier audit are explained by persistent writes: at event0/block4 the value-only change is +0.001504 but the write-only change is −0.016749; at event7/block7 they are −0.001280 and +0.033918. The route chooses a memory address as well as a message. Training must teach that future state effect.

Twelve fixed probes, one population/address/noise seed, fixed selected-node time. Mean absolute write effect 0.012694 versus value residual 0.000288; these absolute summaries are not additive percentages. Hybrids are diagnostic interventions, not legal proposed routes. Value residual includes nonlinear response and downstream route switches. This is conditional fidelity evidence, not an expected-gradient failure rate.

Charged replay counts:48 forwards,12 backwards,12,288 races; 7.107s wall, 390.3MiB peak RSS. Arithmetic is uninstrumented. Checkpoint weights and outer RNG are preserved. Any new state-aware teacher needs integrated contracts, full accounting and a matched small fit.

Source: [completed factorial audit](experiments/results/diagnostics/aws_route_write_decomposition_20261001T235000Z.json); [joint state/time theory](experiments/theory/57_full_state_and_joint_clock_credit.md).

## Appendix B. Ours and boosted trees: banknote

Independent feature-ID rows, state reset between rows, train-only scaling and duplicate-feature group isolation. Ours uses eight native event blocks with parallel heads and content/state mixing; R2 adds temporal reception. Static processing coordinates are not physical asynchronous samples.

| Model | Fit/dev rows | Dev NLL / RMSE ↓ | Dev accuracy / MAE | Whole fit GFLOPs | Fit MFLOPs/row | Infer MFLOPs/row |
| --- | --- | --- | --- | --- | --- | --- |
| Ours R0 | 128/128 | 0.1553 | 0.9531 | 0.335 | 0.655 | 0.139 |
| Ours R2 | 128/128 | 0.2700 | 0.8906 | 0.379 | 0.740 | 0.163 |
| Boosted trees | 128/128 | 0.2319 | 0.9297 | Not counted | Not counted | Not counted |

| Model | All fit wall s | Peak RSS MiB | Tree nodes / bytes |
| --- | --- | --- | --- |
| Ours R0 | 981.79 | 445.8 | Not applicable |
| Ours R2 | 1452.02 | 453.7 | Not applicable |
| Boosted trees | 0.09 | 404.0 | 890/49,840 |

Four development checkpoints or four separately fitted tree candidates; all candidate tree fitting wall time is charged. Neural fit arithmetic is an actual forward/loss/backward/clipping/Adam trace, with specials counted once; preprocessing, evaluation and RNG are separate. Tree FLOPs are unavailable and are not manufactured. Neural wall time includes CPU simulation/audit instrumentation. These are small exploratory development results; reserved test labels are not scored. Strong tabular/frontier superiority requires larger frozen protocols and independent seeds.

Reception ablation: ours R2 reaches89.06% accuracy /0.270 NLL versus native R0 95.31% /0.155. Whole fitting work increases from0.335 to0.379 GFLOPs. The added reception capacity has not earned its cost in this single-seed static-data screen.

Confirmation protocol: native checkpoint reuse plus seeds7/8, original trees, CatBoost and logistic regression; four development selection opportunities per family, then frozen reserved-test scoring. See experiments/AWS_BANKNOTE_CONFIRMATION.md. No pending test score is reported.

## Appendix B. Ours and boosted trees: wine_red

Independent feature-ID rows, state reset between rows, train-only scaling and duplicate-feature group isolation. Ours uses eight native event blocks with parallel heads and content/state mixing; R2 adds temporal reception. Static processing coordinates are not physical asynchronous samples.

| Model | Fit/dev rows | Dev NLL / RMSE ↓ | Dev accuracy / MAE | Whole fit GFLOPs | Fit MFLOPs/row | Infer MFLOPs/row |
| --- | --- | --- | --- | --- | --- | --- |
| Ours R0 | 128/128 | 0.8238 | 0.6422 | 0.816 | 1.594 | 0.341 |
| Ours R2 | 128/128 | 0.7579 | 0.6205 | 0.919 | 1.795 | 0.398 |
| Boosted trees | 128/128 | 0.6489 | 0.4890 | Not counted | Not counted | Not counted |

| Model | All fit wall s | Peak RSS MiB | Tree nodes / bytes |
| --- | --- | --- | --- |
| Ours R0 | 2553.48 | 468.1 | Not applicable |
| Ours R2 | 3547.22 | 485.4 | Not applicable |
| Boosted trees | 0.11 | 404.3 | 464/25,984 |

Four development checkpoints or four separately fitted tree candidates; all candidate tree fitting wall time is charged. Neural fit arithmetic is an actual forward/loss/backward/clipping/Adam trace, with specials counted once; preprocessing, evaluation and RNG are separate. Tree FLOPs are unavailable and are not manufactured. Neural wall time includes CPU simulation/audit instrumentation. These are small exploratory development results; reserved test labels are not scored. Strong tabular/frontier superiority requires larger frozen protocols and independent seeds.

Reception helps this regression pilot: ours R2 RMSE0.758 versus R0 0.824 (8.0% lower), for0.919 versus0.816 whole-fit GFLOPs (12.6% more). Trees retain lower RMSE0.649. This positive within-model effect contrasts with banknote/language reception failures; it is not a cross-family win.

## Appendix B. Addressed-state write credit

Completed integrated pilots only: private S4/P0, eight blocks, two independent heads, d8/pool2, 128 fitting queries per pass/four passes,256 development queries, seed6. Both retain hard temporal races and winner-only inference. The zero-credit model exactly nests the parent; added memory/time credit is a local surrogate, not an arbitrary unbiased sequence-gradient estimator.

![state credit quality work](report/figures/state_credit_quality_work.png)

| Model | Accuracy % | NLL | Whole fit GFLOPs | Fit MFLOPs/query | Infer MFLOPs/query |
| --- | --- | --- | --- | --- | --- |
| Ours: write credit | 55.08 | 0.9778 | 0.277 | 0.542 | 0.110 |
| Ours: baseline | 54.30 | 0.9787 | 0.265 | 0.518 | 0.110 |

| Model | Fit/dev/passes | Parameters | State slots | Updates/scores per event |
| --- | --- | --- | --- | --- |
| Ours: write credit | 128/256/4 | 42,932 | 128 | 16/32 |
| Ours: baseline | 128/256/4 | 42,932 | 128 | 16/32 |

All fitting forward/loss/backward/normalization/clipping/Adam and losing proposals are charged. Special functions have unit weight beside arithmetic; integer discovery/traffic/energy remain separate. Training-only auxiliary state views and whole-process RSS are recorded in each result. Numerical/optimizer prerequisites and accounting smokes are excluded from benchmark plots. Matched gains: 0.78 percentage points / 0.00091 NLL, at 1.045× fitting work. The predeclared larger-fit gate (5 points/.02 NLL/at most2× work) fails. This reuses exploratory development populations; independent seeds and fresh confirmation remain required.

## Appendix B. Learned history and credit reach

Frozen checkpoint audit on32 fixed development positions. Both learned models depend on history; the native predictions change even when identical16-character suffixes receive the same race noise. The native model is therefore not strictly a bigram predictor. Equal average loss to a count model calibrates predictive quality; it does not identify learned features or context dependence.

![language learning context](report/figures/language_learning_context.png)

| Saved model | Layers with gradients | Layer norm range | Replay tokens | Backwards |
| --- | --- | --- | --- | --- |
| Ours: carrier,131K | 6 | 0.419–0.692 | 3120 | 1 |
| Ours: native,8K | 8 | 1.027–3.487 | 3120 | 1 |

The actual current count-composition logit gradient matches responsibility-weighted cross entropy to1.9e-09; all six carrier layers receive gradients. Mean responsibility is3.43% on this64-target initialization probe. This supports investigating attenuated task signal and conditioning, rather than assuming a general gradient disconnect.

No optimizer steps, weight changes or official-test access. History controls reset state and replay the retained suffix at its absolute positions; per-position native noise is coupled. This32-position slice is not the full saved development quality or a matched-data model comparison. Gradient norms aggregate different parameter groups; they show reach, not superior conditioning or unbiased hard-route credit. The64-character history is not uniformly better than16 on this slice; useful long-range/semantic features remain open. All replays/backwards are counted; arithmetic is uninstrumented. Wall33.21s, peakRSS325.5MiB. See experiments/LANGUAGE_LEARNING_DIAGNOSIS_20261002.md.

## Appendix B. Credit horizon with fixed context

Completed frozen native8K/H2/d16/depth8 diagnostic. All three arms retain the same128-token development context, score the same last16 targets and receive identical per-position race noise. Only graph reach changes:16,32 or64 tokens. Predictions match exactly; the gradient changes, separating retained information from the credit used to learn how to retain it.

| Graph horizon | Slice bpc | Gradient norm | Graph tokens | Replay tokens |
| --- | --- | --- | --- | --- |
| 16 | 4.016190 | 7.0827 | 16 | 128 |
| 32 | 4.016190 | 6.3785 | 32 | 128 |
| 64 | 4.016190 | 6.6635 | 64 | 128 |

| Short/long credit | Difference norm | Relative difference % | Gradient cosine |
| --- | --- | --- | --- |
| 16 versus 32 | 1.7577 | 27.56 | 0.9713 |
| 16 versus 64 | 2.0581 | 30.89 | 0.9570 |
| 32 versus 64 | 1.0092 | 15.15 | 0.9890 |

The16-versus64 gradient difference has norm30.89% of the64-token gradient, with cosine0.9570. The32-versus64 difference is15.15%. Every layer receives credit, and short-credit norms can be larger because omitted contributions can cancel retained ones. This is evidence of material truncation effects on this probe; it does not establish that increasing the horizon improves fitting quality.

Frozen weights; zero optimizer steps and no official-test access. One16-target slice,384 replay tokens,112 graph tokens,3 backwards. Wall11.08s, peakRSS518.0MiB. Arithmetic is uninstrumented. Native hard-route credit is a surrogate;64 tokens is a comparison, not an all-history unbiased reference. Relative difference divides the norm of the gradient difference by the longer-credit gradient norm, not a percentage of predictive quality or retained features.

## Appendix B. Banknote: no confirmed advantage

Competitive accuracy, no confirmed advantage: the three-seed reserved test does not sustain the development lead. Ours averages 91.8% versus 94.0% for the original trees. Logistic regression has lower test log loss. The full four-family confirmation remains incomplete. The reserved-test accuracy point estimates are close, but no equivalence margin was specified. Parity is therefore a descriptive reading, not a proven equivalence claim.

![banknote reserved test](report/figures/banknote_reserved_test.png)

| Model | Completed seeds | Mean test accuracy % | Mean test NLL |
| --- | --- | --- | --- |
| Ours native | 3 | 91.81 | 0.2353 |
| Boosted trees | 3 | 93.95 | 0.2065 |
| CatBoost | 2/3 | Pending | Pending |
| Logistic | 3 | 94.66 | 0.0906 |

For the completed ours/tree comparison, the paired accuracy difference is −2.14 percentage points (descriptive95% crossed seed/feature-group interval −6.90 to +2.43). The control-minus-ours NLL difference is −0.0287 (98.33% interval −0.1738 to +0.1277). Logistic regression improves NLL by0.1447 (98.33% interval0.0338 to0.2660). These three-seed intervals are approximate and share one test split.

Partial analysis: local_banknote_partial_confirmation_20261002T013000Z.json;11/12 final cells, 4000 bootstrap draws,270 feature groups, three seeds. NLL intervals allow for three control comparisons; accuracy intervals are descriptive. This does not replace the incomplete full four-family gate. The original95.3% versus93.0% development screen and all per-seed work remain below.

## Appendix B. Frozen banknote confirmation

Completed test scores only. Checkpoints/candidates were selected on128 development rows after fitting 128 rows. Reserved feature groups were scored after choices were frozen; all rows start with cold state. The per-seed ledger preserves completed scores; the summary identifies families with all three seeds.

| Model/seed | Dev NLL | Test NLL | Test accuracy% | Whole fit GFLOPs | Fit MFLOPs/row | Infer MFLOPs/row |
| --- | --- | --- | --- | --- | --- | --- |
| Ours native/s6 | 0.1553 | 0.1426 | 94.31 | 0.335 | 0.655 | 0.139 |
| Ours native/s7 | 0.2594 | 0.2494 | 90.75 | 0.335 | 0.655 | 0.139 |
| Ours native/s8 | 0.3270 | 0.3138 | 90.39 | 0.335 | 0.655 | 0.139 |
| Boosted trees/s6 | 0.2319 | 0.2065 | 93.95 | Not counted | Not counted | Not counted |
| Boosted trees/s7 | 0.2319 | 0.2065 | 93.95 | Not counted | Not counted | Not counted |
| Boosted trees/s8 | 0.2319 | 0.2065 | 93.95 | Not counted | Not counted | Not counted |

| Model/seed | Charged fit wall s | Test wall s | Peak RSS MiB | Fit provenance |
| --- | --- | --- | --- | --- |
| Ours native/s6 | 981.79 | 4.54 | 334.3 | Historical fit reused |
| Ours native/s7 | 976.56 | 4.65 | 447.0 | New fit |
| Ours native/s8 | 989.25 | 4.61 | 447.0 | New fit |
| Boosted trees/s6 | 0.09 | 0.00 | 434.3 | New fit |
| Boosted trees/s7 | 0.09 | 0.00 | 434.9 | New fit |
| Boosted trees/s8 | 0.09 | 0.00 | 434.9 | New fit |

Native forward/loss/backward/clipping/Adam are traced. Seed6 reuse retains its original full fitting charge; it adds no optimizer steps. Control fitting includes all four independent candidates. Their FLOPs are unavailable. Audit instrumentation, preprocessing and physical energy are separate. Repeated seeds share test rows and must not be pooled as independent observations. Paired seed/feature-group analysis and all three prespecified seeds are required for the confirmation claim.

## Appendix B. Frozen banknote confirmation

Completed test scores only. Checkpoints/candidates were selected on128 development rows after fitting 128 rows. Reserved feature groups were scored after choices were frozen; all rows start with cold state. The per-seed ledger preserves completed scores; the summary identifies families with all three seeds.

| Model/seed | Dev NLL | Test NLL | Test accuracy% | Whole fit GFLOPs | Fit MFLOPs/row | Infer MFLOPs/row |
| --- | --- | --- | --- | --- | --- | --- |
| CatBoost/s6 | 0.1023 | 0.1207 | 93.95 | Not counted | Not counted | Not counted |
| CatBoost/s7 | 0.1042 | 0.1078 | 95.37 | Not counted | Not counted | Not counted |
| Logistic/s6 | 0.0832 | 0.0906 | 94.66 | Not counted | Not counted | Not counted |
| Logistic/s7 | 0.0832 | 0.0906 | 94.66 | Not counted | Not counted | Not counted |
| Logistic/s8 | 0.0832 | 0.0906 | 94.66 | Not counted | Not counted | Not counted |

| Model/seed | Charged fit wall s | Test wall s | Peak RSS MiB | Fit provenance |
| --- | --- | --- | --- | --- |
| CatBoost/s6 | 0.35 | 0.00 | 461.6 | New fit |
| CatBoost/s7 | 0.35 | 0.00 | 460.8 | New fit |
| Logistic/s6 | 0.01 | 0.00 | 429.6 | New fit |
| Logistic/s7 | 0.01 | 0.00 | 429.7 | New fit |
| Logistic/s8 | 0.01 | 0.00 | 429.9 | New fit |

Native forward/loss/backward/clipping/Adam are traced. Seed6 reuse retains its original full fitting charge; it adds no optimizer steps. Control fitting includes all four independent candidates. Their FLOPs are unavailable. Audit instrumentation, preprocessing and physical energy are separate. Repeated seeds share test rows and must not be pooled as independent observations. Paired seed/feature-group analysis and all three prespecified seeds are required for the confirmation claim.

## Appendix B (continued). Ours: language work as scaling develops

This ledger updates from completed integrated-model stages. It shows the emerging work advantage alongside its quality and data budget. Per-target fitting work removes the difference in the number of presentations; it does not establish equal-quality superiority.

![integrated language work progress](report/figures/integrated_language_work_progress.png)

| Model | Fit / passes | bpc / split ↓ | Whole fit GFLOPs ↓ | Fitting MFLOPs / target ↓ | Forward MFLOPs / position ↓ |
| --- | --- | --- | --- | --- | --- |
| Ours / d32 / p2 | 32,768 / 4 | 3.106 / dev | 114.247 | 0.872 | 0.089 |
| Ours / d16 / p2 | 32,768 / 4 | 3.121 / dev | 31.053 | 0.237 | 0.026 |
| Ours / d16 / p2 | 8,192 / 4 | 3.398 / dev | 7.788 | 0.238 | 0.025 |
| Ours / d16 / p4 | 8,192 / 4 | 3.426 / dev | 14.721 | 0.449 | 0.032 |
| LSTM / width 512 | 10M / six | 1.799 / test | 432,592.997 | 7.210 | 2.402 |
| Transformer / width 256 | 10M / four | 1.908 / test | 888,775.443 | 22.223 | 7.405 |

Compare within a column: whole-fit totals use GFLOPs for every model; per-target and forward work use MFLOPs for every model. One GFLOP is 1,000 MFLOPs. Whole-fit totals also depend on the number of training presentations; the per-target column divides that out.

All table values, figures and ratios use arithmetic plus one operation per special function, matching the historical neural estimate convention. This is not a physical energy cost. Ours arithmetic-only whole-fit totals (GFLOPs): 32,768 / pool 2: 110.088; 32,768 / pool 2: 29.883; 8,192 / pool 2: 7.492; 8,192 / pool 4: 14.152. Separate special-function counts are preserved in each result.

**The raw work gap is substantial.** The completed 32,768-character integrated stage's representative forward estimate is **84× smaller** than the larger saved Transformer estimate; fitting work per target is **25× smaller**. These are configuration-level work ratios. Our development score and the reference official test score use different targets and data budgets. The gap is not a matched-quality supremacy claim.

Ours: d denotes payload width and p pool size; fixed character pools and event depths. Each result records its validation interval and credit horizon. References: width-512 LSTM or four width-256 Transformer layers, 256-position fitting chunks and 999,999 aligned official test targets. Ours uses representative operator traces including counterfactual credit, backward, clipping and Adam; neural references use shape formulas and backward ≈ twice forward. RNG, indexing, memory traffic and evaluation passes are additional. Same-quality and iso-FLOP conclusions await comparable completed runs.

## Appendix B (continued). Ours and neural controls: accuracy versus FLOPs

Each point is a completed model, not a projected scaling law. Left: ours on cold development characters, with integrated models and earlier carrier controls labelled separately. The new 2K screens score 2,047 development targets; the earlier ladders score 8,191. Right: saved neural test results. Lower bpc means better prediction; lower fitting work means fewer estimated operations. No curve is drawn between different model families or scoring splits.

![language quality vs work](report/figures/language_quality_vs_work.png)

| Model type | Fitting budget | bpc / split ↓ | Whole fit GFLOPs ↓ | Fitting MFLOPs / target ↓ |
| --- | --- | --- | --- | --- |
| Ours: integrated d32/p2 | 32,768 / 4 passes | 3.106 / dev | 114.247 | 0.872 |
| Ours: carrier w128g | 1,048,576 / 4 passes | 2.210 / dev | 8,373.302 | 1.996 |
| LSTM: 512 | 90,000,000 / 6 passes | 1.661 / test | 3,893,396.042 | 7.210 |
| Transformer: 256x4 | 90,000,000 / 4 passes | 1.604 / test | 8,000,253.349 | 22.223 |

The table selects the largest fitting budget currently completed for each family; the best score breaks ties. Point numbers refer to the following variant ledger, which lists all plotted variants. Variant labels: I = ours integrated payload/pool/data; IKV adds per-position race memory (S uses the content index); C = ours carrier width/data (g means content gates); L = LSTM width/data; T = Transformer width x layers/data; s denotes seed. K is 1,024 characters in ours labels; M is decimal million in neural labels.

Estimates include learning, clipping and Adam, with unit-weight special functions. Ours uses representative operator traces; neural controls use shape formulas and backward approximately twice forward. Scoring splits, data, passes, capacity and credit differ; these panels are evidence inventories, not an iso-FLOP or equal-quality benchmark.

## Appendix B (continued). Accuracy versus inference FLOPs

Inference predicts with frozen weights: no backward pass, clipping or optimizer update. These are the same completed checkpoints, quality scores and point IDs as the fitting graph. Ours uses saved forward operator traces; the integrated models read only winning values. LSTM and Transformer costs use shape estimates. Development and test evidence remain separate.

![language quality vs inference](report/figures/language_quality_vs_inference.png)

| Model type | bpc / split ↓ | Inference MFLOPs / character ↓ | Cost boundary |
| --- | --- | --- | --- |
| Ours: integrated d32/p2 | 3.106 / dev | 0.0885 | Winner-only inference trace |
| Ours: carrier w128g | 2.210 / dev | 0.6389 | Saved forward operator trace |
| LSTM: 512 | 1.661 / test | 2.4024 | Recurrent shape estimate |
| Transformer: 256x4 | 1.604 / test | 14.8104 | Overlapping-window shape estimate |

Solid Transformer points estimate its saved 256-position scorer: full windows advanced by 128 positions, approximately two forward positions per scored character (boundary/tail overhead omitted). Hollow points show a hypothetical one-step decode with cached keys/values and 256 available positions, using L(24d² + 4Td + 30d + 20T) + 54d + 135 unit-weight operations, T = 256. No cached decoder was run. Its plotted bpc belongs to the saved window scorer; learned positions reset between windows, so cache reuse has not been shown to preserve those scores.

Two FLOPs per multiply-add; special functions count as one operation. Ours traces include numerical clocks and loss scoring; neural elementwise overhead is approximate. Traces are representative warm-state costs, not full-stream measurements; growing KV candidate occupancy can change work. The separate KV pages also show projected event-architecture costs that remove numerical clock simulation. RNG, indexing, memory traffic and physical race energy are additional. These are work estimates, not latency or joules, and differing data, quality and evaluation protocols prevent a supremacy conclusion.

## Appendix B (continued). Completed language variants and work

| Variant | Params K | Fit / passes | bpc / split ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ | Inference MFLOPs / char ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Ours: AWS I32/p2/32K/s6 | 1,388.9 | 32,768 / 4 | 3.106 / dev | 114.247 | 0.872 | 0.0885 |
| 2. Ours: I16/p2/32K/s6 | 361.4 | 32,768 / 4 | 3.121 / dev | 31.053 | 0.237 | 0.0255 |
| 3. Ours: I16/p2/8K/s6 | 361.4 | 8,192 / 4 | 3.398 / dev | 7.788 | 0.238 | 0.0255 |
| 4. Ours: I16/p4/8K/s6 | 720.0 | 8,192 / 4 | 3.426 / dev | 14.721 | 0.449 | 0.0321 |
| 5. Ours: IHR2x32/b64/u64@0.002D8/2K/s6 | 3,819.5 | 2,048 / 4 | 3.740 / dev | 23.019 | 2.811 | 0.5060 |
| 6. Ours: IHR2x32D8/2K/s6 | 3,819.5 | 2,048 / 4 | 3.786 / dev | 27.731 | 3.387 | 0.5059 |
| 7. Ours: IHR2x32/u128@0.004D8/2K/s6 | 3,819.5 | 2,048 / 4 | 3.779 / dev | 19.999 | 2.442 | 0.5058 |
| 8. Ours: IHR2x32/u64@0.002D8/2K/s6 | 3,819.5 | 2,048 / 4 | 3.733 / dev | 22.753 | 2.779 | 0.5059 |
| 9. Ours: IHR2x32/u64@0.004D8/2K/s6 | 3,819.5 | 2,048 / 4 | 3.800 / dev | 22.750 | 2.779 | 0.5060 |
| 10. Ours: IHR2x32/u128@0.004D8/8K/s6 | 3,819.5 | 8,192 / 4 | 3.485 / dev | 79.953 | 2.440 | 0.5059 |
| 11. Ours: IHR2x32/u64@0.002D8/8K/s6 | 3,819.5 | 8,192 / 4 | 3.490 / dev | 89.999 | 2.747 | 0.5058 |
| 12. Ours: IHR4x32/u128@0.004D8/2K/s6 | 7,778.3 | 2,048 / 4 | 3.981 / dev | 48.431 | 5.915 | 1.2898 |

Each row retains its original architecture, fitting budget and score. The selected 10M LSTM/Transformer rows use the aligned 999,999-target scores; other neural rows retain their original E64 test scorers. The 90M LSTM uses its saved recurrent scoring protocol. Carrier and integrated development scores use frozen evaluation; integrated official scores appear only after their full test completes. Validation/test work, RNG and physical traffic are outside fitting totals. Sources: E64/E174, saved AWS E64 results and the completed parallel_language and episodic_language JSON records. The global ledger uses emulator floating arithmetic consistently; fitting work per target divides by actual training target presentations. The separate KV page reports architectural projections. No new dense model was trained.

## Appendix B (continued). Completed language variants and work

| Variant | Params K | Fit / passes | bpc / split ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ | Inference MFLOPs / char ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| 13. Ours: IHR4x32/u128@0.004D8/8K/s6 | 7,778.3 | 8,192 / 4 | 3.543 / dev | 193.751 | 5.914 | 1.2882 |
| 14. Ours: IKV32D6/2K/s6 | 1,413.6 | 2,048 / 4 | 3.620 / dev | 9.035 | 1.103 | 0.1466 |
| 15. Ours: IKV32D8/2K/s6 | 1,883.9 | 2,048 / 4 | 3.539 / dev | 11.992 | 1.465 | 0.1941 |
| 16. Ours: IKVS32D8/2K/s6 | 1,883.9 | 2,048 / 4 | 3.554 / dev | 11.999 | 1.465 | 0.1968 |
| 17. Ours: IKVS32D8/8K/s6 | 1,883.9 | 8,192 / 4 | 3.357 / dev | 50.006 | 1.526 | 0.1969 |
| 18. Ours: I32D6/2K/s6 | 1,388.9 | 2,048 / 4 | 3.633 / dev | 7.207 | 0.880 | 0.0890 |
| 19. Ours: I32D8/2K/s6 | 1,850.9 | 2,048 / 4 | 3.542 / dev | 9.557 | 1.167 | 0.1173 |
| 20. Ours: I32D8/8K/s6 | 1,850.9 | 8,192 / 4 | 3.311 / dev | 40.243 | 1.228 | 0.1173 |
| 21. Ours: IHR2x32/m2/u128@0.004D8/2K/s6 | 3,819.5 | 2,048 / 4 | 3.820 / dev | 20.495 | 2.503 | 0.5165 |
| 22. Ours: IHR2x32/m4/u128@0.004D8/2K/s6 | 3,819.5 | 2,048 / 4 | 3.771 / dev | 20.914 | 2.554 | 0.5268 |
| 23. Ours: IHR2x32/wc0.25/u64@0.002D8/2K/s6 | 3,819.5 | 2,048 / 4 | 3.722 / dev | 23.469 | 2.866 | 0.5059 |
| 24. Ours: IHR2x32/wc1/u64@0.002D8/2K/s6 | 3,819.5 | 2,048 / 4 | 3.724 / dev | 23.464 | 2.866 | 0.5058 |

Each row retains its original architecture, fitting budget and score. The selected 10M LSTM/Transformer rows use the aligned 999,999-target scores; other neural rows retain their original E64 test scorers. The 90M LSTM uses its saved recurrent scoring protocol. Carrier and integrated development scores use frozen evaluation; integrated official scores appear only after their full test completes. Validation/test work, RNG and physical traffic are outside fitting totals. Sources: E64/E174, saved AWS E64 results and the completed parallel_language and episodic_language JSON records. The global ledger uses emulator floating arithmetic consistently; fitting work per target divides by actual training target presentations. The separate KV page reports architectural projections. No new dense model was trained.

## Appendix B (continued). Completed language variants and work

| Variant | Params K | Fit / passes | bpc / split ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ | Inference MFLOPs / char ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| 25. Ours: Native H2d16/p2/2K/s6 | 54.9 | 2,048 / 4 | 3.765 / dev | 3.778 | 0.461 | 0.0974 |
| 26. Ours: Native H2d16/p2/8K/s6 | 54.9 | 8,192 / 4 | 3.557 / dev | 15.116 | 0.461 | 0.0974 |
| 27. Ours: R2/uniform/reception/2K/s6 | 57.1 | 2,048 / 4 | 3.796 / dev | 4.032 | 0.492 | 0.1056 |
| 28. Ours: R4/uniform/reception/2K/s6 | 59.3 | 2,048 / 4 | 3.795 / dev | 4.183 | 0.511 | 0.1109 |
| 29. Ours: R4/late/reception/2K/s6 | 57.1 | 2,048 / 4 | 3.795 / dev | 3.979 | 0.486 | 0.1041 |
| 30. Ours: R4/uniform/waiting/2K/s6 | 59.3 | 2,048 / 4 | 3.764 / dev | 3.897 | 0.476 | 0.1007 |
| 31. Ours: C128/128K | 308.0 | 131,072 / 4 | 2.643 / dev | 1,032.197 | 1.969 | 0.6296 |
| 32. Ours: C32/128K | 21.7 | 131,072 / 4 | 2.858 / dev | 77.197 | 0.147 | 0.0470 |
| 33. Ours: C64/128K | 80.3 | 131,072 / 4 | 2.727 / dev | 274.735 | 0.524 | 0.1675 |
| 34. Ours: C128g/128K | 309.6 | 131,072 / 4 | 2.587 / dev | 1,046.656 | 1.996 | 0.6389 |
| 35. Ours: C256g/128K | 1,208.9 | 131,072 / 4 | 2.572 / dev | 4,025.494 | 7.678 | 2.4571 |
| 36. Ours: C128g/1024K | 309.6 | 1,048,576 / 4 | 2.210 / dev | 8,373.302 | 1.996 | 0.6389 |

Each row retains its original architecture, fitting budget and score. The selected 10M LSTM/Transformer rows use the aligned 999,999-target scores; other neural rows retain their original E64 test scorers. The 90M LSTM uses its saved recurrent scoring protocol. Carrier and integrated development scores use frozen evaluation; integrated official scores appear only after their full test completes. Validation/test work, RNG and physical traffic are outside fitting totals. Sources: E64/E174, saved AWS E64 results and the completed parallel_language and episodic_language JSON records. The global ledger uses emulator floating arithmetic consistently; fitting work per target divides by actual training target presentations. The separate KV page reports architectural projections. No new dense model was trained.

## Appendix B (continued). Completed language variants and work

| Variant | Params K | Fit / passes | bpc / split ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ | Inference MFLOPs / char ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| 37. L256/1M | 338.4 | 1,000,000 / 20 | 2.179 / test | 40,628.875 | 2.032 | 0.6770 |
| 38. L256/10M | 338.4 | 10,000,000 / 1 | 2.171 / test | 20,306.115 | 2.032 | 0.6770 |
| 39. L512/10M | 1,199.3 | 10,000,000 / 6 | 1.799 / test | 432,592.997 | 7.210 | 2.4024 |
| 40. T112x8/1M | 1,250.6 | 1,000,000 / 5 | 2.352 / test | 51,107.144 | 10.223 | 6.7999 |
| 41. T256x2/1M | 1,658.9 | 1,000,000 / 20 | 2.367 / test | 222,614.402 | 11.133 | 7.4192 |
| 42. T256x2/10M | 1,658.9 | 10,000,000 / 1 | 2.427 / test | 111,261.602 | 11.133 | 7.4192 |
| 43. T256x4/10M | 3,238.4 | 10,000,000 / 4 | 1.908 / test | 888,775.443 | 22.223 | 14.8104 |
| 44. L512/90M | 1,199.3 | 90,000,000 / 6 | 1.661 / test | 3,893,396.042 | 7.210 | 2.4024 |
| 45. T256x4/90M | 3,238.4 | 90,000,000 / 4 | 1.604 / test | 8,000,253.349 | 22.223 | 14.8104 |

Each row retains its original architecture, fitting budget and score. The selected 10M LSTM/Transformer rows use the aligned 999,999-target scores; other neural rows retain their original E64 test scorers. The 90M LSTM uses its saved recurrent scoring protocol. Carrier and integrated development scores use frozen evaluation; integrated official scores appear only after their full test completes. Validation/test work, RNG and physical traffic are outside fitting totals. Sources: E64/E174, saved AWS E64 results and the completed parallel_language and episodic_language JSON records. The global ledger uses emulator floating arithmetic consistently; fitting work per target divides by actual training target presentations. The separate KV page reports architectural projections. No new dense model was trained.

## Appendix B (continued). Ours: per-position race KV memory

Both integrated models fit 2,048 characters for 4 passes, with payload 32, 6 sparse receiver depths, seed 6 and 2,047 identical cold development targets. The KV arm retains separate historical keys and values at every depth; learned queries select one value through time. Incoming content is retained and gated with the retrieved message. No dense carrier is added.

![episodic language comparison D2048 depth6 character](report/figures/episodic_language_comparison_D2048_depth6_character.png)

| Ours: memory | Dev bpc ↓ | Projected fit GFLOPs ↓ | CPU fit GFLOPs ↓ | Projected forward MFLOPs/char ↓ |
| --- | --- | --- | --- | --- |
| receiver | 3.633 | 7.206 | 7.207 | 0.0890 |
| kv | 3.620 | 9.034 | 9.035 | 0.1464 |

Completed KV improvement over receiver memory: +0.013 bpc (positive is better). The index admits up to 8 recent matching-character entries plus 4 recent positions. Older entries outside these tails cannot be addressed by this index. Duplicates are removed. Development averages 11.36 keys scored and one value delivered per retrieval query. All 12,282 entries remain stored (3.00 MiB raw keys/values); the oldest selected entry is 1,829 characters old. This bounds reads, not stored history.

Physical clock competition replaces explicit numerical rate exponentiation and noise/rate division plus the bounded-delay simulation in the projected ledger. Query/key/value maps, scored candidates, gated content, backward, counterfactual teaching, clipping and actual Adam remain charged. Counts are representative first/mature/partial traces; special functions have unit weight here and are separate in JSON. Physical rate setting, clock circuits, index/address operations, RNG and traffic need their own implementation costs; FLOPs do not certify energy.

Temporal races avoid the explicit normalizing reduction/division and deliver one value at inference; training reads all admitted values for route credit. The orange bar is an analytical same-shortlist aggregation comparison, not another trained model. Candidate coverage is approximate and does not guarantee full-bank attention equivalence. Random-hyperplane indexing is an established primitive (Charikar, STOC 2002); novelty is not claimed for this index. Historical activations are detached at the credit boundary and are not recomputed after parameter updates. One seed and a small data budget; no equal-quality Transformer or frontier claim.

## Appendix B (continued). Ours: per-position race KV memory

Both integrated models fit 2,048 characters for 4 passes, with payload 32, 8 sparse receiver depths, seed 6 and 2,047 identical cold development targets. The KV arm retains separate historical keys and values at every depth; learned queries select one value through time. Incoming content is retained and gated with the retrieved message. No dense carrier is added.

![episodic language comparison D2048 depth8 character](report/figures/episodic_language_comparison_D2048_depth8_character.png)

| Ours: memory | Dev bpc ↓ | Projected fit GFLOPs ↓ | CPU fit GFLOPs ↓ | Projected forward MFLOPs/char ↓ |
| --- | --- | --- | --- | --- |
| receiver | 3.542 | 9.556 | 9.557 | 0.1172 |
| kv | 3.539 | 11.990 | 11.992 | 0.1939 |

Completed KV improvement over receiver memory: +0.003 bpc (positive is better). The index admits up to 8 recent matching-character entries plus 4 recent positions. Older entries outside these tails cannot be addressed by this index. Duplicates are removed. Development averages 11.36 keys scored and one value delivered per retrieval query. All 16,376 entries remain stored (4.00 MiB raw keys/values); the oldest selected entry is 1,829 characters old. This bounds reads, not stored history.

Physical clock competition replaces explicit numerical rate exponentiation and noise/rate division plus the bounded-delay simulation in the projected ledger. Query/key/value maps, scored candidates, gated content, backward, counterfactual teaching, clipping and actual Adam remain charged. Counts are representative first/mature/partial traces; special functions have unit weight here and are separate in JSON. Physical rate setting, clock circuits, index/address operations, RNG and traffic need their own implementation costs; FLOPs do not certify energy.

Temporal races avoid the explicit normalizing reduction/division and deliver one value at inference; training reads all admitted values for route credit. The orange bar is an analytical same-shortlist aggregation comparison, not another trained model. Candidate coverage is approximate and does not guarantee full-bank attention equivalence. Random-hyperplane indexing is an established primitive (Charikar, STOC 2002); novelty is not claimed for this index. Historical activations are detached at the credit boundary and are not recomputed after parameter updates. One seed and a small data budget; no equal-quality Transformer or frontier claim.

## Appendix B (continued). Ours: per-position race KV memory

Both integrated models fit 2,048 characters for 4 passes, with payload 32, 8 sparse receiver depths, seed 6 and 2,047 identical cold development targets. The KV arm retains separate historical keys and values at every depth; learned queries select one value through time. Incoming content is retained and gated with the retrieved message. No dense carrier is added.

![episodic language comparison D2048 depth8 semantic](report/figures/episodic_language_comparison_D2048_depth8_semantic.png)

| Ours: memory | Dev bpc ↓ | Projected fit GFLOPs ↓ | CPU fit GFLOPs ↓ | Projected forward MFLOPs/char ↓ |
| --- | --- | --- | --- | --- |
| receiver | 3.542 | 9.556 | 9.557 | 0.1172 |
| kv | 3.554 | 11.997 | 11.999 | 0.1965 |

Completed KV improvement over receiver memory: -0.012 bpc (positive is better). The content index uses three random-hyperplane bits of learned keys/queries, with up to 8 recent/full-history samples in the query bucket and its one-bit neighbors, plus 4 recent positions. Duplicates are removed. Development averages 10.07 keys scored and one value delivered per retrieval query. All 16,376 entries remain stored (4.00 MiB raw keys/values); the oldest selected entry is 2,026 characters old. This bounds reads, not stored history.

Physical clock competition replaces explicit numerical rate exponentiation and noise/rate division plus the bounded-delay simulation in the projected ledger. Query/key/value maps, scored candidates, gated content, backward, counterfactual teaching, clipping and actual Adam remain charged. Counts are representative first/mature/partial traces; special functions have unit weight here and are separate in JSON. Physical rate setting, clock circuits, index/address operations, RNG and traffic need their own implementation costs; FLOPs do not certify energy.

Temporal races avoid the explicit normalizing reduction/division and deliver one value at inference; training reads all admitted values for route credit. The orange bar is an analytical same-shortlist aggregation comparison, not another trained model. Candidate coverage is approximate and does not guarantee full-bank attention equivalence. Random-hyperplane indexing is an established primitive (Charikar, STOC 2002); novelty is not claimed for this index. Historical activations are detached at the credit boundary and are not recomputed after parameter updates. One seed and a small data budget; no equal-quality Transformer or frontier claim.

## Appendix B (continued). Ours: per-position race KV memory

Both integrated models fit 8,192 characters for 4 passes, with payload 32, 8 sparse receiver depths, seed 6 and 8,191 identical cold development targets. The KV arm retains separate historical keys and values at every depth; learned queries select one value through time. Incoming content is retained and gated with the retrieved message. No dense carrier is added.

![episodic language comparison D8192 depth8 semantic](report/figures/episodic_language_comparison_D8192_depth8_semantic.png)

| Ours: memory | Dev bpc ↓ | Projected fit GFLOPs ↓ | CPU fit GFLOPs ↓ | Projected forward MFLOPs/char ↓ |
| --- | --- | --- | --- | --- |
| receiver | 3.311 | 40.241 | 40.243 | 0.1173 |
| kv | 3.357 | 49.998 | 50.006 | 0.1967 |

Completed KV improvement over receiver memory: -0.047 bpc (positive is better). The content index uses three random-hyperplane bits of learned keys/queries, with up to 8 recent/full-history samples in the query bucket and its one-bit neighbors, plus 4 recent positions. Duplicates are removed. Development averages 10.40 keys scored and one value delivered per retrieval query. All 65,528 entries remain stored (16.00 MiB raw keys/values); the oldest selected entry is 8,135 characters old. This bounds reads, not stored history.

Physical clock competition replaces explicit numerical rate exponentiation and noise/rate division plus the bounded-delay simulation in the projected ledger. Query/key/value maps, scored candidates, gated content, backward, counterfactual teaching, clipping and actual Adam remain charged. Counts are representative first/mature/partial traces; special functions have unit weight here and are separate in JSON. Physical rate setting, clock circuits, index/address operations, RNG and traffic need their own implementation costs; FLOPs do not certify energy.

Temporal races avoid the explicit normalizing reduction/division and deliver one value at inference; training reads all admitted values for route credit. The orange bar is an analytical same-shortlist aggregation comparison, not another trained model. Candidate coverage is approximate and does not guarantee full-bank attention equivalence. Random-hyperplane indexing is an established primitive (Charikar, STOC 2002); novelty is not claimed for this index. Historical activations are detached at the credit boundary and are not recomputed after parameter updates. One seed and a small data budget; no equal-quality Transformer or frontier claim.

## Appendix B (continued). Ours: cheaper learning updates

Credit still propagates over 16-character segments. Gradients are summed over U targets, normalized by their actual count, clipped once and used for one Adam update. All models here use two independent heads, payload 32/head, eight blocks, seed 6, four passes over 2,048 fitting characters and 8,191 frozen development targets.

![parallel optimizer work](report/figures/parallel_optimizer_work.png)

| Ours: U / learning rate | Dev bpc ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ | Adam + clip GFLOPs ↓ |
| --- | --- | --- | --- | --- |
| U16 / 0.001 | 3.786 | 27.731 | 3.387 | 10.521 |
| U64 / 0.002 | 3.733 | 22.753 | 2.779 | 5.225 |
| U64 / 0.004 | 3.800 | 22.750 | 2.779 | 5.225 |
| U128 / 0.004 | 3.779 | 19.999 | 2.442 | 2.645 |

The cheapest schedule within the declared 0.05 bpc tolerance of the best pilot uses U128, lr 0.004: 3.779 bpc and 27.9% less whole fitting work than reference. The best quality is 3.733 bpc. These completed results support optimizer amortization in this configuration, not language-model supremacy.

Learning rates and warmup differ across configurations, so this is not an isolated optimizer-interval ablation. Selected checkpoints minimize frozen development loss over the fixed four passes; all fitting work remains charged. Each result is one seed. Candidate scoring, losing-value credit, backward and gradient accumulation/normalization remain in the operator ledger. Larger-data and repeat-seed comparisons must establish transfer of the selected schedule.

## Appendix B (continued). Ours: shared-match temporal arrivals

2 independent spatial heads, payload 32/head, 8 blocks. All rows fit 2,048 characters for 4 passes and score 8,191 cold development targets. Each query forms its candidate matches once. Multiple temporal marks reuse those rates; only the winning emitter renews its clock. The receiver and each selected message evolve until the last local read, then the messages are averaged and gated.

![repeated arrival quality work 1df4adaf89](report/figures/repeated_arrival_quality_work_1df4adaf89.png)

Ours delivers 4 times as many historical winner messages for 4.57% additional whole fitting arithmetic in this completed screen. Quality changes from 3.779 to 3.771 bpc. This supports cheap arrival multiplicity under shared matches; it does not establish language-model superiority.

| Ours: arrivals / head | Dev bpc ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ | Inference MFLOPs / char ↓ |
| --- | --- | --- | --- | --- |
| m=1 | 3.779 | 19.999 | 2.442 | 0.5058 |
| m=2 | 3.820 | 20.495 | 2.503 | 0.5165 |
| m=4 | 3.771 | 20.914 | 2.554 | 0.5268 |

Candidate discovery and independent Q/K/V projections are retained. Multiple marks reuse one rate setting within each query; trained scores and candidate trajectories can differ across runs. More marks can retrieve the same value; they do not create extra learned spatial heads or discover absent candidates. Training reads all admitted values once and aggregates the conserved per-arrival teacher in O(Cd + md). All delivered messages, temporal transports, backward, clipping and Adam remain charged.

Completed single-seed development screens; m=1 reuses the saved reference under exact nesting contracts. Unit-weight special functions are included; CPU minimum comparisons/RNG and memory traffic are separate counters. The bounded numerical time encoding is not a demonstrated homogeneous physical Poisson clock. This local counterfactual teacher is a declared surrogate, not an exact gradient through nonlinear route changes. No matched-quality dense-model, physical-energy or frontier superiority is inferred.

## Appendix B (continued). Ours: independent temporal heads

Each head has its own receiver pool, historical bank and query/key/value/gate matrices. A winning content vector evolves through learned rotation and decay until its channel is read. The next block reads at the latest parallel arrival, preserves each channel and learns their mix. Heads need not arrive simultaneously. This is implemented in the CPU emulator; execution there is serial.

![parallel temporal heads](report/figures/parallel_temporal_heads.png)

![parallel temporal head pilots](report/figures/parallel_temporal_head_pilots.png)

| Ours: heads / update | Fit / passes | Dev bpc ↓ | Whole fit GFLOPs ↓ |
| --- | --- | --- | --- |
| H2 / U128 / lr 0.004 | 8,192 / 4 | 3.485 | 79.953 |
| H2 / U64 / lr 0.002 | 8,192 / 4 | 3.490 | 89.999 |
| H4 / U128 / lr 0.004 | 2,048 / 4 | 3.981 | 48.431 |
| H4 / U128 / lr 0.004 | 8,192 / 4 | 3.543 | 193.751 |

Payload 32 per head: H2 total width 64, H4 total width 128; eight blocks. More heads also increase capacity, and source/channel dynamics differ from the old single-head model. The baseline H2 pilot selects epoch 2 and overfits later; no head-count quality benefit is established. Adam interval U is separate from 16-character credit. Training reads admitted losing values and charges gradients, clipping and optimizer work. Contracts pass for causality, independent projections, evolving channels, all-head gradients and exact next-update recovery. All completed variants remain in the ledger; this table shows the latest four records.

## Appendix B (continued). Ours: completed head/data scaling

Fixed d32 per head, eight blocks, pool2, credit16, U128/lr.004, four fitting passes, seed6. Every point scores the same 8,191 frozen development targets. More data improves these configurations, while four heads increase both width/capacity and fitting cost.

![parallel head data work](report/figures/parallel_head_data_work.png)

| Ours: heads / fit | Dev bpc ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ |
| --- | --- | --- | --- |
| H2 / 2,048 | 3.779 | 19.999 | 2.442 |
| H2 / 8,192 | 3.485 | 79.953 | 2.440 |
| H4 / 2,048 | 3.981 | 48.431 | 5.915 |
| H4 / 8,192 | 3.543 | 193.751 | 5.914 |

The earlier 8K single-head controls reach receiver 3.311 and indexed KV 3.357 bpc. The multihead construction also changes source/channel dynamics and total width, so this is not a pure head-count ablation. The completed parallel-head 8K results missed the declared 0.10 bpc tolerance of the indexed control; the campaign stopped before 32K/131K promotion. Route-credit fidelity, recurrent/channel conditioning, candidate coverage and optimization are diagnosis targets. These results constrain this implementation rather than the whole substrate.

One seed and small fitting budgets. Whole fitting includes all four passes, backward, admitted losing-value credit and optimizer work. Development selection uses the lowest full development loss over those passes. Logical FLOPs and unit-weight special functions do not measure wall time, physical traffic or energy; no language supremacy follows from these points.

## Appendix B (continued). Ours: longer temporal credit

Independent H2 heads, d32/head, 8 event blocks, pool2. 2,048 fitting characters / 4 passes; 8,191 frozen development targets, seed6. Adam uses U64 / lr0.002. Graphs remain live for 64 targets before detachment. Forward stored history and the inference architecture are retained.

| Ours: credit | Dev bpc ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ | Inference MFLOPs / char ↓ |
| --- | --- | --- | --- | --- |
| 16 | 3.733 | 22.753 | 2.779 | 0.5059 |
| 64 | 3.740 | 23.019 | 2.811 | 0.5060 |

Sealed historical keys and values still affect predictions, but detachment removes later loss paths to their old producers. Longer credit restores those paths for more writes inside each optimizer window; it does not backpropagate through unlimited history. Temporal races, sparse receiver commits, separate Q/K/V and losing-route credit remain active.

Matched 16-credit records are shown when completed under identical settings. Additional backward/normalization/clip/Adam work is counted and peak memory is guarded. Inference traces average different representative spans; the operation definitions and inference architecture are the same. One seed, development selection, no frontier claim. Forward-partition equality and full 64-credit gradient/update contracts precede fitting.

## Appendix B (continued). What most reduces research uncertainty

The main direction now tests native content-and-time computation directly. Episodic race attention remains a preserved comparison. Its small language improvements do not yet establish that an attention scaffold is the best use of this substrate.

| Priority | Experiment | Doubt resolved |
| --- | --- | --- |
| 1 | Integrated order/time learning; refitted credit/time controls; three seeds | Can deep sparse temporal state learn useful representations? |
| 2 | Native-core text8 adapter and small-to-larger data ladder | Does the native construction learn economically without a KV attention bank? |
| 3 | Occupy 4, 16, 64 stream states at fixed event/query budgets | Does useful state grow without proportional per-event activity? |
| 4 | Chronological real streams and predict-before-update adaptation | Does the advantage survive real data and online change? |
| 5 | Timestamp-aware AWS controls, equal-budget/quality curves | Is the quality/resource advantage reproducible? |
| 6 | Whole-system FPGA/ASIC timing, traffic and energy measurement | Does the physical substrate deliver the projected savings? |

The first native branch uses eight event blocks, two independent receiver heads, observed source addresses, persistent content/state, analytic temporal evolution and counterfactual learning. It has no per-position KV attention. Other sources keep independent progress; source-local causal dependencies and internal joins remain charged.

The same native core receives token content for the language test. Sharing receiver maps across tokens changes capacity and parameter exposure; it is a whole-construction comparison, not an isolated attention-removal ablation. Persistent state is not a full-cache equivalence claim.

Contracts/smokes precede fixed-budget pilots, conditional capacity/data scaling and independent replication. Whole fitting, per-target work, inference, occupancy, memory and confidence intervals are published from completed files. Synthetic learning, real-data Pareto advantage and physical joules are separate milestones. The executable protocol, gates and current host limitations are documented in experiments/RESEARCH_VALUE_PLAN.md.

## Appendix B (continued). Ours: content-gated temporal reception

Native eight-block independent-head models reuse key/query matches for two/four additional scalar clock policies. Local rotating/decaying clock vectors gate content-dependent projections and compose the next message. The waiting control retains the same clocks and joins but removes temporal reception/readout. The native parent has no extra clock branch.

![delay language quality work](report/figures/delay_language_quality_work.png)

| Ours | Fit chars / passes | Dev bpc ↓ | CPU fit GFLOPs ↓ | Fit MFLOPs / target ↓ | Infer MFLOPs / char ↓ |
| --- | --- | --- | --- | --- | --- |
| Native | 2,048/4 | 3.765 | 3.778 | 0.461 | 0.0974 |
| R2 uniform | 2,048/4 | 3.796 | 4.032 | 0.492 | 0.1056 |
| R4 uniform | 2,048/4 | 3.795 | 4.183 | 0.511 | 0.1109 |
| R4 late | 2,048/4 | 3.795 | 3.979 | 0.486 | 0.1041 |
| R4 uniform wait | 2,048/4 | 3.764 | 3.897 | 0.476 | 0.1007 |

| Ours | Projected whole fit GFLOPs | Receivers / commits per token | Matches / clocks per token | Parameters |
| --- | --- | --- | --- | --- |
| Native | 3.777 | 32/16 | 32/16 | 54,907 |
| R2/uniform/on | 4.029 | 32/16 | 32/48 | 57,083 |
| R4/uniform/on | 4.178 | 32/16 | 32/80 | 59,259 |
| R4/late/on | 3.976 | 32/16 | 32/48 | 57,083 |
| R4/uniform/waiting | 3.891 | 32/16 | 32/80 | 59,259 |

Completed fits only; identical frozen 8,191-target development protocol, four passes, U64/lr.002/warm512, ordinary credit16. Different fitting sizes are explicitly marked. All scalar policies, projections, temporal bases, gates, counterfactual content teachers and actual Adam remain charged. Projected arithmetic removes only numeric clock simulation. Physical rate setting, clock circuits, traffic, precision and measured joules remain separate. Exploratory development results; these do not alone establish comparable-quality Transformer superiority.

Current matched-fit interpretation: best completed added-clock row is 0.0010 bpc better than native, with 3.13% more fitting work. Pending allocations and waiting controls cannot establish a benefit yet.

## Appendix B (continued). Ours: count-carrying native receivers

The unchanged native eight-block core supplies the base predictive; addressed context-suffix receivers of orders 1..K carry sufficient statistics and deliver by an escape-race cascade with learned discount and concentration (Theory §§376–380, 387). The escape-gate variant makes discount and concentration per-position functions of the native predictive and count evidence; the count-message variant adds the counts to the base logits (Theory §389). Fitting counts are leave-one-out; development counts are prequential persistent state with frozen weights. Count tables are capacity; each target touches K addresses.

| Model | Fit chars / passes | Dev bpc ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ | Infer MFLOPs / char ↓ |
| --- | --- | --- | --- | --- | --- |
| Native alone 2,048 | 2,048/4 | 3.765 | 3.778 | 0.461 | 0.0974 |
| Count-carrying native K4 2,048 | 2,048/4 | 2.734 | 3.806 | 0.465 | 0.0982 |
| Same, untrained base 2,048 | 2,048/0 | 2.741 | Not trained | Not trained | 0.0982 |
| Count-carrying native K4 + count message 2,048 | 2,048/4 | 2.744 | 4.032 | 0.492 | 0.1102 |
| Count-carrying native K4 + escape gate 2,048 | 2,048/4 | 2.695 | 3.824 | 0.467 | 0.0993 |
| Count-carrying native K4 + escape gate [credit64] 2,048 | 2,048/4 | 2.694 | 3.843 | 0.469 | 0.0993 |
| Count-carrying native K4 + escape gate [minimal core p2/d1] 2,048 | 2,048/4 | 2.694 | 0.072 | 0.009 | 0.0031 |
| Count-carrying native K4 + escape gate [minimal core p2/d1] [credit64] 2,048 | 2,048/4 | 2.694 | 0.072 | 0.009 | 0.0031 |
| Count-carrying native K4 + gate + message 2,048 | 2,048/4 | 2.703 | 4.053 | 0.495 | 0.1113 |
| KN counts, frozen o4 | 2,048/1 | 3.615 | Not FLOPs | Not FLOPs | Not FLOPs |
| Calibration ceiling: adaptive interpolated KN o6 | 2,048/1 | 2.521 | Not FLOPs | Not FLOPs | Not FLOPs |
| Native alone 8,192 | 8,192/4 | 3.557 | 15.116 | 0.461 | 0.0974 |
| Count-carrying native K4 8,192 | 8,192/4 | 2.671 | 15.227 | 0.465 | 0.0982 |
| Same, untrained base 8,192 | 8,192/0 | 2.682 | Not trained | Not trained | 0.0982 |
| Count-carrying native K4 + escape gate 8,192 | 8,192/4 | 2.595 | 16.853 | 0.514 | 0.1155 |
| Count-carrying native K4 + escape gate 8,192 | 8,192/4 | 3.474 | 16.789 | 0.512 | 0.1155 |
| Count-carrying native K4 + escape gate 8,192 | 8,192/4 | 2.601 | 15.314 | 0.467 | 0.0993 |
| Count-carrying native K4 + escape gate [minimal core p16/d1] 8,192 | 8,192/4 | 2.592 | 4.181 | 0.128 | 0.0358 |
| Count-carrying native K4 + escape gate [minimal core p16/d1] 8,192 | 8,192/4 | 3.492 | 4.161 | 0.127 | 0.0358 |
| Count-carrying native K4 + escape gate [minimal core p2/d1] 8,192 | 8,192/4 | 2.588 | 0.288 | 0.009 | 0.0031 |
| Count-carrying native K4 + escape gate [minimal core p2/d1] 8,192 | 8,192/4 | 2.588 | 5.995 | 0.183 | 0.0632 |
| Count-carrying native K4 + escape gate [minimal core p2/d1] 8,192 | 8,192/4 | 2.593 | 1.744 | 0.053 | 0.0184 |
| Count-carrying native K4 + escape gate [minimal core p2/d1] 8,192 | 8,192/4 | 3.471 | 1.744 | 0.053 | 0.0184 |
| KN counts, frozen o4 | 8,192/1 | 3.081 | Not FLOPs | Not FLOPs | Not FLOPs |
| Calibration ceiling: adaptive interpolated KN o6 | 8,192/1 | 2.414 | Not FLOPs | Not FLOPs | Not FLOPs |
| Count-carrying native K4 32,768 | 32,768/4 | 2.560 | 60.790 | 0.464 | 0.0982 |
| Same, untrained base 32,768 | 32,768/0 | 2.593 | Not trained | Not trained | 0.0982 |
| Count-carrying native K4 + escape gate 32,768 | 32,768/4 | 2.447 | 60.790 | 0.464 | 0.0993 |
| Count-carrying native K4 + escape gate [minimal core p2/d1] 32,768 | 32,768/4 | 2.401 | 1.151 | 0.009 | 0.0031 |
| KN counts, frozen o5 | 32,768/1 | 2.704 | Not FLOPs | Not FLOPs | Not FLOPs |
| Calibration ceiling: adaptive interpolated KN o8 | 32,768/1 | 2.271 | Not FLOPs | Not FLOPs | Not FLOPs |

Same 8,191 development targets for every row; one seed. Count increments/lookups are integer table work reported in the result files, not FLOPs. The initialized-base/escape row measures whole-model fitting benefit; it does not isolate the native base. Count rows are dev-selected-order references, not neural controls. How to read these rows (Theory §§393–394): at a few thousand to tens of thousands of characters, smoothed counting is a near-optimal estimator, and no learner (Transformers included) is expected to exceed it. At 10M characters the repository's dense Transformer control is still .12 bpc worse than frozen counts (§381). The strongest such reference, stream-adaptive interpolated Kneser–Ney, scores 2.521 / 2.414 / 2.271 bpc at 2K / 8K / 32K and 2.101 at 131K (results/count_reference/curie_adaptive_kn_language_reference_20261002T131500Z.json). It is a calibration ceiling, not a competitor. Distances to it measure remaining smoothing, and the rows here are mechanism diagnostics, not a verdict on the architecture, whose claims are tested on tasks where learning matters (§394). Exploratory development evidence.

At 2,048 fitting characters, fitting the native base and escape parameters improves 0.0071 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 2,048 fitting characters, fitting the native base and escape parameters improves 0.0542 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 2,048 fitting characters, fitting the native base and escape parameters improves 0.0455 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 2,048 fitting characters, fitting the native base and escape parameters improves 0.0381 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 2,048 fitting characters, fitting the native base and escape parameters improves -0.0032 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 32,768 fitting characters, fitting the native base and escape parameters improves 0.0329 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 32,768 fitting characters, fitting the native base and escape parameters improves 0.1930 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 32,768 fitting characters, fitting the native base and escape parameters improves 0.1454 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 8,192 fitting characters, fitting the native base and escape parameters improves 0.0108 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 8,192 fitting characters, fitting the native base and escape parameters improves 0.0988 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 8,192 fitting characters, fitting the native base and escape parameters improves 0.0988 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 8,192 fitting characters, fitting the native base and escape parameters improves 0.0922 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 8,192 fitting characters, fitting the native base and escape parameters improves 0.0890 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 8,192 fitting characters, fitting the native base and escape parameters improves 0.0915 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 8,192 fitting characters, fitting the native base and escape parameters improves -0.2996 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 8,192 fitting characters, fitting the native base and escape parameters improves -0.2977 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 8,192 fitting characters, fitting the native base and escape parameters improves -0.3213 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 8,192 fitting characters, fitting the native base and escape parameters improves 0.0809 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 2,048 fitting characters, fitting the native base and escape parameters improves 0.0464 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

At 2,048 fitting characters, fitting the native base and escape parameters improves 0.0543 bpc over their untrained initialization. The complete composed predictor improves over native-alone, while this smaller learning contribution is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.

## Appendix B. Longer credit versus learned count smoothing

Completed four-arm integrated comparison, seed6: same2K fitting characters/four passes, 8,191 development targets, K4 escape gate, U64/lr.002/warmup512. Each fit processes8,188 targets and128 optimizer updates. Full H2/d16/depth8 and minimal H2/d2/depth1 each compare16 versus64 tokens of graph reach; persistent state and forward mechanisms remain unchanged within each core.

| Model/credit | Dev bpc | Whole fit GFLOPs | Fit MFLOPs/target | Infer MFLOPs/char |
| --- | --- | --- | --- | --- |
| Ours: full core /credit16 | 2.6951 | 3.824 | 0.467 | 0.0993 |
| Minimal core control /credit16 | 2.6938 | 0.072 | 0.009 | 0.0031 |
| Ours: full core /credit64 | 2.6943 | 3.843 | 0.469 | 0.0993 |
| Minimal core control /credit64 | 2.6937 | 0.072 | 0.009 | 0.0031 |

| Model/credit | Parameters | Core state slots | Updates/scores/teacher values per target |
| --- | --- | --- | --- |
| Ours: full core /credit16 | 54,955 | 32 | 16/32/32 |
| Minimal core control /credit16 | 467 | 4 | 2/4/4 |
| Ours: full core /credit64 | 54,955 | 32 | 16/32/32 |
| Minimal core control /credit64 | 467 | 4 | 2/4/4 |

Full-core longer-credit gain0.0008bpc; minimal-core gain0.0001bpc. Full versus minimal advantage at64:-0.0006bpc, with1.005× full-core fitting work versus16. The predeclared follow-up gate fails.

All neural fitting forward/loss/backward/normalization/clipping/Adam and admitted losing-value credit are charged in CPU emulator units; projected clockless work is a separate result ledger. Each target also looks up4 count addresses; integer counts, discovery, traffic and energy stay separate. This reuses development data and is a single-seed screen, not confirmation, semantic-feature proof or supremacy. Passing requires at least.02bpc full-core credit gain AND at least.02bpc advantage over the matched minimal64 core at no more than2× full-core fitting work. No automatic larger fit.

## Appendix B. Does the trained deep state contribute?

Frozen saved full/minimal64-credit models from the matched2K fits above. Same32 development targets at positions128–159, same race noise and causal count vectors. Erase stored content once at the slice start, retaining arrival times and absolute count cursor; state may rebuild. The learned escape gate responds to the changed neural base.

| Model/intervention | Composed bpc | Raw base bpc | Change in composed bpc |
| --- | --- | --- | --- |
| Full core/intact | 3.4541 | 4.9093 | +0.0000 |
| Full core/Erase source context | 3.4916 | 5.0308 | +0.0375 |
| Full core/Erase receiver content | 3.4495 | 4.9933 | -0.0046 |
| Full core/Erase all content | 3.5022 | 5.1477 | +0.0481 |
| Minimal control/intact | 3.4904 | 4.9638 | +0.0000 |
| Minimal control/Erase source context | 3.4885 | 4.9424 | -0.0018 |
| Minimal control/Erase receiver content | 3.4901 | 4.9649 | -0.0002 |
| Minimal control/Erase all content | 3.4882 | 4.9401 | -0.0022 |

Full-core stored content helps this slice by.0481bpc, mainly through the carried source context; erasing receiver content alone does not hurt. Useful recurrence is present, but this does not establish hierarchical semantics, generalization across slices, or a full-development lead. Per-layer receiver erasures are preserved in the diagnostic JSON.

| Model | Mean base responsibility | Median base responsibility | Dynamic/base-only gradient norm |
| --- | --- | --- | --- |
| Full core | 0.0717 | 0.0045 | 0.207 |
| Minimal control | 0.0571 | 0.0031 | 0.195 |

Credit diagnostic uses64 fitting targets at positions64–127. With gate values fixed, the exact base-logit derivative is responsibility × (q−onehot); numerical errors are below5e−9. Actual dynamic-gate gradients reach every layer. Responsibility suppression is correct mixture credit, not a demonstrated autograd bug; Adam can compensate for uniform scaling. The distribution and usefulness of the residual signal, information retention and retrieval remain hypotheses.

No optimizer steps or weight changes. Raw base is trained as a conditional residual, so its standalone bpc is diagnostic. Audit12.85s/522,736KiB; no FLOP or superiority claim. Source: local_deep_core_attribution_20261002T075000Z.json. The synthetic long-range queue was retired before launch after unsupported chance claims and credit/update confounding were found; repaired-driver contracts require matched optimizer windows and separately measured local controls.

## Appendix B. Integrated memory repair: accounting prerequisite

Completed short accounting smokes, not quality pilots. Unchanged full native core versus context-addressed outcome slots: H2/d16/depth8, same192 fitting characters/one pass,191 targets, three U64 Adam updates (partial63), c16/lr.002/no warmup, seed6, same core initialization and training race RNG. Frozen cold-state development has128 targets. Both zero-repair initial scores match exactly; every actual fitting operation is traced.

| Model | Smoke dev bpc | Whole fit GFLOPs | Fit MFLOPs/target | Infer MFLOPs/target |
| --- | --- | --- | --- | --- |
| native | 4.5432 | 0.08838 | 0.4627 | 0.0974 |
| addressed | 4.5384 | 0.08983 | 0.4703 | 0.1016 |

| Model | Core slots | Updates/keys/teacher values per target | Extra slots occupied/capacity | State tensor bytes |
| --- | --- | --- | --- | --- |
| native | 32 | 16/32/32 | 0/0 | 2,448 |
| addressed | 32 | 16/32/32 | 150/4096 | 21,648 |

Addressed whole-fitting work ratio1.0164×; observed smoke gain0.0047bpc. Three updates are insufficient to assess useful deeper features. Fixed hashed addresses test memory capacity and evidence paths; they do not implement learned context pooling or KV race attention. Native temporal races, evolving messages, receiver key/value separation and unrealized-route surrogate credit remain.

Full-shape addressed/tapped contracts pass exact zero forward/parent gradients, trained causality/chunk invariance, target-weighted normalization and serialized next-Adam predictions, parameters and moments. Context-slot and tap-buffer tensors are now counted and detached. Whole-driver recovery/selection and consistent warm-state comparisons remain admission gaps before long fits; no automatic scale-up.

CPU emulator, 2FLOPs/MAC plus unit-weight specials, all fitting forward/loss/backward/normalization/clipping/Adam; excludes development, RNG, hash/integer operations and traffic. Inference is one accounted target after127 warm tokens, not throughput. Activity columns are derived from fixed code/dimensions. Empty text filler-group NaN in original JSONs is undefined, not a score; preserved beside correction to null in the driver. Cold neural memory differs from fit-prefilled count references. Source: local_deep_feature_preflight_20261002T081500Z.json.

## Appendix B. Restoring addressed value credit: completed fits

Five matched integrated arms:1,024 fitting characters/four passes,4,092 targets/64 Adam updates,2,047 dev targets, H2/pool2/c16/U64/warm512/lr.002/seed6. Full native temporal core versus original projected-value slots, late-projected raw-feature slots and two shallow controls. Core initialization and race RNG are matched. Minimum cold dev bpc over fixed passes selects each checkpoint; frozen fit replay is secondary and never selects weights.

| Model | Cold dev bpc | Fit-replay dev bpc | Selected pass |
| --- | --- | --- | --- |
| Native full p16/L8 | 3.9681 | 3.9674 | 4 |
| Stored projection p16/L8 | 3.9030 | 3.8967 | 4 |
| Late projection p16/L8 | 3.9421 | 3.9809 | 4 |
| Late p16/L1 | 4.0862 | 4.0801 | 4 |
| Late p2/L1 | 4.5396 | 4.5274 | 4 |

![value credit learning 20261002T090800Z](report/figures/value_credit_learning_20261002T090800Z.png)

Late-full gain over native:+0.0260bpc; over stored projection:-0.0391; over minimal:+0.5975; over same-width shallow:+0.1441. The predeclared nomination gate fails.

Late projection preserves fixed-weight computation by linearity and restores fixed-feature projection credit from old detached slots. It does not restore historical core-producer credit or add linear-reader expressivity. Fixed hash addresses are not learned pooling/KV race attention. The same-width shallow arm isolates depth better than the width-changing minimal control. Reused dev and one seed: exploratory quality/learning evidence, not semantic proof or supremacy.

## Appendix B. Value-credit resources and protocol boundaries

| Model | Whole fit GFLOPs est. | Fit MFLOPs/target est. | Cold infer MFLOPs/target | Replay GFLOPs est. | Warm infer MFLOPs/target |
| --- | --- | --- | --- | --- | --- |
| Native full p16/L8 | 1.893 | 0.463 | 0.0974 | 0.100 | 0.0974 |
| Stored projection p16/L8 | 1.924 | 0.470 | 0.1016 | 0.104 | 0.1016 |
| Late projection p16/L8 | 1.933 | 0.472 | 0.0999 | 0.104 | 0.1006 |
| Late p16/L1 | 0.348 | 0.085 | 0.0203 | 0.022 | 0.0210 |
| Late p2/L1 | 0.018 | 0.004 | 0.0012 | 0.001 | 0.0012 |

| Model | Core slots | Extra occupied/capacity | Updates/keys/teacher values per fit target | Context reads/writes per fit target |
| --- | --- | --- | --- | --- |
| Native full p16/L8 | 32 | 0/0 | 16/32/32 | 0.000/0.000 |
| Stored projection p16/L8 | 32 | 757/4096 | 16/32/32 | 1.000/0.999 |
| Late projection p16/L8 | 32 | 757/4096 | 16/32/32 | 1.000/0.999 |
| Late p16/L1 | 4 | 757/4096 | 2/4/4 | 1.000/0.999 |
| Late p2/L1 | 4 | 757/4096 | 2/4/4 | 1.000/0.999 |

Every column uses the same unit/denominator for all five models. Whole fitting costs are representative first/mature/partial-window estimates from actual forward/loss/backward/normalization/clipping/Adam audits, with admitted losing-value work charged.2FLOPs/MAC plus unit-weight specials; CPU emulator here. Projected clockless costs are separate in the completed JSON ledger. Variable retrieval occupancy and graph reach are not fully enumerated.

Warm evaluation replays all1,024 fit tokens with frozen selected weights, then carries predictive state/time into dev. Hash history and previous-address are cleared at the boundary to forbid an invented cross-split outcome. Dev race noise is coupled to cold scoring. Replay cost is extra work, estimated from first/last replay chunks; it is neither fitting FLOPs nor free access to historical data. Neural slots update causally during dev with frozen parameters.

Late/original full fitting work ratio:1.004×. All direct/core and additional memory work must earn predictive value. No automatic scale-up. The nomination gate requires at least.02bpc cold gains against all four controls and no more than2× original-full fitting work under this common estimate convention.

Hash/integer bookkeeping, RNG, traffic, Python metadata and physical energy remain separate. Cold/replay scores here use2,047 dev targets and cannot be directly juxtaposed with saved8,191-target count/dense references. Original-minimal projection contrast remains open; AWS integrated capacity/exposure work remains independent. Complete work/activity/storage and exact-source checkpoint provenance are retained in local_value_credit_analysis_20261002T090800Z.json.

## Appendix B. Fixed-feature credit and compiled inference

Frozen-reader compilation folds A=R_vW once and removes the unused W matrix. Coupled predictions and raw-slot state preserve the unfused construction within numerical tolerance. Under2FLOPs/MAC, fold2d³ replaces2d² per occupied read, breaking even after d occupied reads. This changes deployment work, not the fitting optimizer; compiled models refuse training.

| Late model | Unfused infer KFLOPs/target | Fused infer KFLOPs/target | Removed weights | Fold KFLOPs | Break-even occupied reads |
| --- | --- | --- | --- | --- | --- |
| late_full | 100.180 | 99.540 | 1024 | 65.536 | 32 |
| late_shallow | 20.562 | 19.922 | 1024 | 65.536 | 32 |
| late_minimal | 1.224 | 1.214 | 16 | 0.128 | 4 |

Inference samples include16 actual targets after128 warm tokens; the saved projection work equals2d² times observed occupied reads. Matrix-fold work is charged above; constructor initialization/copy/RNG are separate. This is arithmetic/weight reduction with frozen quality, not measured latency, energy or competitive superiority.

| Text-trained model | Prefix-pair KL, gap8 | Prefix-pair KL, gap32 | Prefix-pair KL, gap64 |
| --- | --- | --- | --- |
| native_full | 2.14e-04 | 1.84e-03 | 3.46e-04 |
| addressed_full | 1.94e-03 | 5.28e-04 | 1.10e-03 |
| late_full | 1.87e-03 | 4.80e-04 | 5.50e-04 |
| late_shallow | 1.88e-06 | 3.18e-07 | 2.90e-07 |
| late_minimal | 8.10e-14 | 0.00e+00 | 0.00e+00 |

Each parity diagnostic balances all four input bit pairs with identical noise/query suffix. Actual causal query count vectors are identical for orders1–8, but targets are opposite across paired prefixes. A predictor restricted to those suffix/count inputs has at least1bit target logloss; the full prefix determines parity exactly. Cue-free prefixes make the count contract explicit, replacing the retired synthetic all-orders independence assertion.

These frozen text checkpoints were never trained on parity. KL and top-context differences diagnose prefix dependence/retention, not parity learning or semantic abstraction. Zero optimizer steps, parameter fingerprints unchanged. A later task fit needs matched full/shallow controls, new-prefix generalization and complete prefix computation/credit accounting. Historical feature-producer credit, learned address pooling and KV races remain open. Source: local_value_credit_frozen_20261002T090800Z.json; theory63 records exact scope.

## Appendix B. Frozen feature geometry and matched count calibration

| Model | Frozen fit bpc | Cold dev bpc | Fit head gradient norm | Dev feature participation rank |
| --- | --- | --- | --- | --- |
| native_full | 3.6920 | 3.9681 | 0.597 | 2.31 |
| addressed_full | 3.6595 | 3.9030 | 0.536 | 1.95 |
| late_full | 3.7272 | 3.9421 | 0.637 | 1.60 |
| late_shallow | 3.8785 | 4.0862 | 0.249 | 4.53 |
| late_minimal | 4.5775 | 4.5396 | 0.323 | 1.21 |

At the selected weights, replay1,023 fitting and2,047 development targets from cold state with the primary evaluation RNG. Capture actual head inputs, then compute fixed-feature linear-head loss derivatives and covariance in float64. No head fitting, optimizer step or parameter changes. Frozen dev scores reproduce the primary results within1e−5bpc.

The participation rank is (trace C)²/trace(C²), a measure of feature geometry. Finite joint training need not make the final head stationary; its gradient norm and covariance rank alone do not identify a bug or semantic abstraction. Frozen fitting loss helps separate training fit from development generalization without using online training scores.

| Count order | Frozen KN bpc | Frozen WB bpc | Frozen AD bpc | Adaptive WB bpc | Adaptive AD bpc |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.8263 | 3.8008 | 3.8692 | 3.4806 | 3.5024 |
| 2 | 3.7986 | 3.9596 | 3.9576 | 3.1259 | 3.1566 |
| 3 | 3.7530 | 4.1254 | 4.0371 | 3.0285 | 3.0271 |
| 4 | 3.7515 | 4.2123 | 4.0771 | 3.0873 | 3.0308 |

Same1,024 fit characters and2,047 dev targets, reused count-reference implementation, fixed discount.75, all orders/methods shown separately. Count construction uses one pass; learned models use four gradient passes. Counts are fit-prefilled; adaptive variants also update from observed dev outcomes after prediction. These state/history policies differ from cold neural scoring; frozen neural fit replay is reported separately above.

KN: interpolated Kneser–Ney; WB: Witten–Bell; AD: absolute discount. Integer represented count increments/lookups and measured construction/scoring wall time are in the completed audit, without conversion to neural FLOPs. Equal scalar loss calibrates prediction quality; it does not establish identical features or a same-quality compute advantage. No per-target order selection or official-test access. Full spectra, data/checkpoint hashes and diagnostic boundaries are retained in local_value_credit_frozen_20261002T090800Z.json.

## Appendix B. Beyond query counts: balanced joint learning

Each episode observes a bit followed by its complement, then another such pair, eight shared noise symbols and a unique query cue. The target is the relation between the earlier bits. All four bit combinations occur for each noise suffix. Observed symbol totals and actual order-0..8 query count vectors match; all 36 declared WB/AD controls score one bit.

| Integrated model | Dev bits/query | Dev accuracy | Selected pass |
| --- | --- | --- | --- |
| native_full | 0.9995 | 50.00% | 8 |
| tapped_full | 1.0002 | 48.44% | 14 |
| tapped_shallow | 1.0002 | 50.00% | 16 |

![local balanced joint 20261002T121500Z analysis learning](report/figures/local_balanced_joint_20261002T121500Z_analysis_learning.png)

The native and learned-delay variants fail the predeclared 75%/.8-bit dependency gate. Eight fitting suffix groups give 32 targets/pass; 16 passes,512 target presentations,64 Adam updates,64 held-out dev targets. All arms have query-only binary supervision and full 15-event credit; the target is never an input. Noise couples the quartet and variants.

The one-bit bound restricts the predictor to query suffix/count inputs, including root marginals. It excludes arbitrary inspection of the full prefix or other count addresses. Counts should lead in their supported local regime; this test asks for useful nonlocal prediction. One selected seed/development set, not semantic language or supremacy.

## Appendix B. Balanced-core fitting and inference work

| Model | Whole fit GFLOPs est. | Fit MFLOPs/target est. | Inference MFLOPs/query | State tensor bytes |
| --- | --- | --- | --- | --- |
| native_full | 0.5603 | 1.0943 | 0.2424 | 720 |
| tapped_full | 0.6186 | 1.2081 | 0.2750 | 4560 |
| tapped_shallow | 0.1998 | 0.3903 | 0.0978 | 496 |

All rows use 512 query-target presentations as the fitting denominator. Inference includes all 15 episode inputs and query loss. Full prefix forward/backward, losing receiver values, normalization/clipping and Adam are charged. First/last complete optimizer windows supply the whole-fit estimates; routing occupancy is not exhaustively traced.2FLOPs/MAC plus unit specials, including declared delay-index rounding. Actual keys/commits/input events and wall/RSS are saved separately.

Delay taps add bounded per-layer input buffers, trainable delays and projections while retaining native temporal races and private persistent receivers. Zero taps exactly nest native prediction and parent gradients. Seven contracts include actual interrupted-driver and full-shape Adam recovery. The same-width shallow control retains these mechanisms but has fewer layers; initialization also changes, so this is an architecture comparison.

Memory traffic, integer/index bookkeeping, RNG, checkpoint I/O and hardware energy remain outside floating arithmetic. More retention capacity or a valid derivative is not a completed quality advantage. All unsuccessful arms and previous language evidence remain preserved; full source/data/result hashes are in the completed analysis JSON.

## Appendix B. Frozen retention versus interaction decoding

| Encoder / degree | Fit bits/query | Fresh bits/query | Fresh accuracy |
| --- | --- | --- | --- |
| native full selected / 1 | 0.994 | 1.063 | 50.78% |
| native full selected / 2 | 0.920 | 12.138 | 50.78% |
| native full initial / 1 | 0.999 | 1.026 | 50.00% |
| native full initial / 2 | 0.876 | 25.333 | 50.78% |
| tapped full selected / 1 | 0.999 | 1.011 | 49.22% |
| tapped full selected / 2 | 0.943 | 21.694 | 50.00% |
| tapped shallow selected / 1 | 1.000 | 1.011 | 50.00% |
| tapped shallow selected / 2 | 0.980 | 7.876 | 50.00% |

| Frozen encoder | First-bit probe accuracy | Second-bit probe accuracy |
| --- | --- | --- |
| native_full_selected | 50.78% | 51.56% |
| native_full_initial | 50.78% | 48.44% |
| tapped_full_selected | 50.78% | 53.12% |
| tapped_shallow_selected | 50.78% | 51.56% |

Freeze the selected encoders, plus native initial weights as a reservoir control. Fit zero-initialized affine or standard degree-2 polynomial residuals on actual query features. Fitting standardization only; four race-noise views of the same32 fitting episodes. Regularized full-batch L-BFGS, fixed lambda1e-5/100 iterations; reported derivatives do not establish convergence. New seed73001 has32 paired suffix groups/128 targets and never selects readout settings or weights.

Neither readout generalizes; polynomial heads become badly overconfident. Separate one-bit probes also remain near chance. They use extra bit supervision only for diagnosis; their outputs never enter the relation predictor. These failures motivate evidence-access tests rather than assuming a larger decoder solves the current model.

Zero core optimizer steps and encoder fingerprints preserved. Eight predeclared readout arms on one fresh synthetic distribution, not general confirmation or a semantic claim. Failed probes do not prove all information was erased or a universal learning ceiling.

## Appendix B. Complete frozen-readout pipeline resource boundary

| Encoder / degree | Total fitting GFLOPs est. | MFLOPs/distinct fit query | Inference MFLOPs/query est. |
| --- | --- | --- | --- |
| native full selected / 1 | 0.5918 | 18.495 | 0.2424 |
| native full selected / 2 | 0.6061 | 18.942 | 0.2428 |
| native full initial / 1 | 0.0315 | 0.985 | 0.2424 |
| native full initial / 2 | 0.0456 | 1.426 | 0.2428 |
| tapped full selected / 1 | 0.6543 | 20.447 | 0.2751 |
| tapped full selected / 2 | 0.6686 | 20.894 | 0.2755 |
| tapped shallow selected / 1 | 0.2129 | 6.654 | 0.0979 |
| tapped shallow selected / 2 | 0.2270 | 7.095 | 0.0983 |

Every total includes the entire earlier encoder fit, four-view feature-prefix replay and actual residual-head optimization/feature construction. Initial weights pay zero encoder fitting. The common denominator here is32 DISTINCT fitting queries for every row, rather than differing L-BFGS closure counts. This denominator differs explicitly from the presentation-based table above. Closure target evaluations and exact head work are saved separately. The same feature replay is charged once to each hypothetical standalone arm.

Every inference estimate includes processing the complete prefix plus the chosen query head. Encoder replay/inference estimates use coupled first-quartet operation audits; head construction, loss, backward and optimizer operations are traced directly. Separate retention-probe supervision/work is diagnostic overhead, not free parity training. Raw candidate/traffic/metadata/RNG costs and measured energy remain separate.

Local vector products are supporting primitives, not substitutes for the native temporal core or evidence of learned KV routing. Numerical contracts verify polynomial adjoints, regularized fixed-feature gradients, zero nesting, causal integrated prediction and restored head outputs. Preserve these negative results beside the subsequent protected-memory hypothesis; no favorable cell is inferred from unfinished training.

## Appendix B. Protected outcomes and joint terminal races

Each observed token writes its observed successor at its predecessor address, never a target-derived or hand-selected bit address. All occupied addresses are candidates. Two learned query/key exponential races deliver small interpreted symbol values to a generic bilinear query decoder. Native temporal computation, sparse vector receivers and earlier counterfactual learning remain. No XOR extraction enters prediction.

| Integrated arm | Selected dev bits | Reserved bits | Reserved accuracy | Selected pass |
| --- | --- | --- | --- | --- |
| joint_full | 0.077 | 0.102 | 96.09% | 16 |
| local_full | 0.933 | 0.750 | 78.12% | 13 |
| joint_shallow | 0.062 | 0.119 | 95.31% | 16 |

![local joint outcome 20261002T125700Z analysis learning](report/figures/local_joint_outcome_20261002T125700Z_analysis_learning.png)

Reserved seed74001/32 groups/128 queries scores fixed selected weights after all fits; no retuning. Joint-minus-local loss improvement:+0.648 bits; full-minus-shallow improvement:+0.017 bits. The joint-credit gate passes; all arms remain visible.

One fitted seed and structured generator. Sixteen fit groups/64 queries,16 passes, U4/lr.01/p4/H2/pool2,fullL2/shallowL1. The one-bit bound applies to identical query suffix/count inputs including root marginals, not arbitrary inspection of other count addresses. No natural-language, dense-control or resource-supremacy claim.

## Appendix B. Joint-credit work, capacity and activity

| Arm | Whole fit GFLOPs est. | Fit MFLOPs/query est. | Inference MFLOPs/query |
| --- | --- | --- | --- |
| joint_full | 0.2379 | 0.2323 | 0.0534 |
| local_full | 0.2319 | 0.2265 | 0.0534 |
| joint_shallow | 0.1534 | 0.1498 | 0.0361 |

| Arm | Core / raw address capacity | Mean raw occupied | Core commits / raw writes per query | Terminal keys / values per query | Training loss pairs per query |
| --- | --- | --- | --- | --- | --- |
| joint_full | 8 / 27 | 10.56 | 60 / 14 | 21.12 / 2 | 112.06 |
| local_full | 8 / 27 | 10.56 | 60 / 14 | 21.12 / 2 | 0.00 |
| joint_shallow | 4 / 27 | 10.56 | 30 / 14 | 21.12 / 2 | 112.06 |

All fitting columns use1,024 query presentations,256 Adam updates and all15 observed prefix events per query. Whole-fit first/last-window estimates include native candidates, losing proposals, protected-state discovery, terminal decoder/loss, backward, clipping and Adam. Inference scores every occupied key and delivers only two values. Available addresses, occupied state, scored keys, commits, raw writes and value deliveries are distinct; sparse activity does not imply zero key or learning cost.

Joint training enumerates C² terminal losses and differentiates their categorical expected risk, giving exact conditional terminal content-choice credit. Local training uses a sampled pair and the existing value-linearized race surrogate. Their initial forward predictions and inference policy match; learning estimators differ. Earlier native route surrogates, raw fixed-address writes and downstream timing credit remain separate limitations. No exact whole-sequence gradient is claimed.

Joint/local whole-fitting work ratio:1.026. 2FLOPs/MAC plus unit specials; traffic, raw integer state, Python objects, RNG and energy separate. Five contracts and three accounting smokes precede fits. The same-width shallow comparison changes initialization too. A successful protected-state read does not prove learned context pooling, arbitrary-distance KV retrieval or useful deep producer credit.

## Appendix B. Fixed joint-credit confirmation

| Fit seed / credit | Selected dev bits | New suffix bits | New suffix accuracy | Pass | Zero-value accuracy |
| --- | --- | --- | --- | --- | --- |
| s7 / joint | 0.055 | 0.050 | 99.22% | 16 | 98.83% |
| s7 / local | 0.009 | 0.003 | 100.00% | 15 | 100.00% |
| s8 / joint | 0.000 | 0.000 | 100.00% | 16 | 75.00% |
| s8 / local | 0.001 | 0.004 | 100.00% | 16 | 99.22% |

| Fit seed | Joint improvement bits/query | Paired suffix interval | Joint/local fitting work |
| --- | --- | --- | --- |
| 7 | -0.046 | [-0.093, -0.016] | 1.025 |
| 8 | +0.003 | [+0.000, +0.008] | 1.026 |

Two additional fitted seeds7/8 use the unchanged full p4/L2 configuration and16 passes. All four declared fits are shown. Development seed72001 selects the minimum across fixed passes; seed75001/64 suffix groups/256 targets is reserved for fixed selected models. No parameter, learning rate, stopping rule or evaluation setting is retuned on this set.

0 of2 declared joint-versus-local gates pass. A seed must reach at least75%/.8bits and improve local credit by.05bits. Paired suffix bootstrap intervals are conditional on the fitted seed and synthetic generator; they are not a broad confidence interval over learning algorithms or tasks.

Initial predictions, race choices and probabilities match within each joint/local pair. Inference mechanisms are identical. The differing training risk/credit objectives therefore test this terminal learning intervention, while protecting all earlier negative evidence.

Zero-value evaluation removes delivered content while keeping native context, query/key races and contextual decoder. No fitting or model selection; every parameter is restored and verified afterwards. Three arms retain98.8–100% accuracy: native context has learned useful nonlocal prediction. Joint seed8 drops to75%. Value delivery is therefore not necessary in every selected fit; neither its training benefit nor a depth premium is isolated.

Query-suffix/count bound only. The protected bank has fixed observed predecessor addresses and no learned writes; the standard bilinear decoder and exact terminal content credit do not establish deep core learning, clock-gradient accuracy or natural-text gains.

## Appendix B. Confirmation fitting and inference resources

| Seed / credit | Whole fit GFLOPs est. | Fit MFLOPs/query est. | Inference MFLOPs/query | Fit wall seconds |
| --- | --- | --- | --- | --- |
| s7 / joint | 0.2373 | 0.2318 | 0.0534 | 302.3 |
| s7 / local | 0.2314 | 0.2260 | 0.0534 | 315.3 |
| s8 / joint | 0.2380 | 0.2324 | 0.0534 | 307.6 |
| s8 / local | 0.2320 | 0.2266 | 0.0534 | 309.0 |

Every row fits64 distinct queries for16 passes:1,024 target presentations and256 Adam updates, all15 prefix events per query. Whole-fit and per-target columns share that denominator. First/last complete optimizer-window estimates include prefix/native races, all terminal candidate values, joint pair losses where applicable, backward, normalization/clipping and Adam. Measured wall and RSS are separate from FLOPs.

All models have8 native receivers plus27 raw outcome addresses. Each query causes 60 native state commits and14 raw outcome writes. Terminal inference scores2C keys and delivers2 values. Joint training enumerates C² losses; both training arms inspect C candidate values. C depends on observed occupancy and is recorded in activity traces.

Additional counterfactual arithmetic buys prediction improvement only where completed results support it. A quality advantage at this budget is not an iso-quality compute advantage; shorter or better-controlled local fits have not been optimized. No measured energy, candidate-discovery latency or hardware throughput claim is inferred.

One guarded one-thread CPU job at a time, watchdog active,8GiB available-memory floor. Unique queues, checkpoints, source/data hashes and failed results preserved. This confirmation adds seed evidence for a small structured nonlocal relation, not architectural supremacy or superiority to an unrestricted count-memory algorithm.

## Appendix B. A terminal symmetry, not a learning impossibility

| Numerical contract | Completed result |
| --- | --- |
| Actual outcome bank | Uniform two-value law identical across second-bit alternatives |
| Fixed zero-context risk | 1.000000bits |
| Maximum decoder gradient | 6.939e-18 |
| Maximum policy gradient | 0.000e+00 |
| Analytic marker-policy witness | 6.921e-09expected bits |

At query time the four distinguished outcome addresses contain b1,b2,1−b2,n0. Their multiset is b1,0,1,n0. Remaining entries depend only on the shared noise suffix. Uniform independent value reads therefore have the same full joint distribution for both second-bit alternatives. If additional context also lacks that bit, no decoder of those values can predict the balanced relation better than one bit.

At zero residual, every candidate pair has the parent logit. Exact terminal key credit is initially zero. With fixed zero-logit uninformative context, uniform policies and balanced examples, decoder gradients are zero too. The actual bank and decoder contracts verify this conditional stationary point in float64 on four suffix groups. This is not a stationary-point proof for the whole trainable native core.

An explicit target-independent existence witness selects marker addresses24/25 and interprets symbols0/1 as signed values. A bilinear interaction can predict their relation with very small expected loss. The witness is analytic and never enters the fitted model or benchmark score. Finite score clamps leave a nonzero routing error floor, so unlimited decoder confidence need not reduce unconditional expected loss.

The useful solution and terminal symmetry coexist. Random key contrast may break the alias, which is compatible with late fitted transitions but does not prove their cause. A new controlled conditioning study must retain all seeds and mechanisms; these fixed confirmation fits are neither extended nor tuned after observation.

Theory70 states the information conditions and proof. Numerical result records source hashes, targets, wall and RSS. The obstruction concerns addressed value pooling and terminal initialization; it does not imply that all counting, temporal computation or architectural adaptation faces a mathematical ceiling.

## Appendix B. Practical headroom: a stronger joint-event table

| Fixed table evaluation | Queries | Accuracy | NLL | Maximum accuracy headroom |
| --- | --- | --- | --- | --- |
| development | 256 | 99.609% | 0.051632 | 0.391pp |
| confirmation | 1024 | 99.512% | 0.055258 | 0.488pp |

This control consumes observed timestamps as well as text. It retains the complete question string and the latest observed time for each of four marks. A generic fit learns one mark-age split per question: which mark, threshold, split direction and both leaf probabilities. The generator word/mark mapping and recency threshold are not supplied.

Same512 distinct fitting episodes/seed1301,one fitting pass,20 question strings; fit wall0.027026s,maxRSS67,940KiB. Fixed learner; dev2301/256 and confirmation3301/1,024 episodes do not tune it. Contracts verify label mutation, appended future observations and common clock shifts cannot change causal prediction features.

Less than half a percentage point of accuracy headroom remains. The proposed20-point learned-control advantage gate is impossible against this reference on these episodes, even for a perfect learner. A gain over the time-blind table would therefore measure extra supplied information rather than demonstrate the requested practical advantage.

The joint recency task remains a capability and mechanism diagnostic. Likelihood may still improve; neither saturated accuracy nor tiny fitting cost supports general supremacy. Strong real-stream calibration with causal common inputs takes priority. Old successful mechanism evidence and negative results are preserved.

Logical content inspections, timestamp writes, age subtraction, sorting comparisons, threshold candidates, count/probability work and wall are saved. They are not converted to neural FLOPs. Full question lookup is a task-specific conventional control, not a general language model or the integrated research architecture. Theory72 records scope.

## Appendix B. Real gesture practical calibration

A meaningful practical region:984 first-second fitting gestures from users1–19 and192 development gestures from users20–23. The common causal representation has4×4 spatial cells, two polarities and20 observed50ms count closures. No official test was opened.

| Control | Dev accuracy % | Dev NLL | Fit seconds | Dev inference seconds |
| --- | --- | --- | --- | --- |
| Calibrated counts | 58.85 | 1.3734 | 0.042 | 0.000 |
| linear_C0.1 | 61.46 | 1.2036 | 0.742 | 0.001 |
| linear_C1 | 59.38 | 1.5903 | 0.439 | 0.001 |
| linear_C10 | 57.81 | 2.4610 | 0.191 | 0.001 |
| rbf_C1_g0.25 | 67.19 | 0.8804 | 0.860 | 0.068 |
| rbf_C1_g1 | 74.48 | 0.7358 | 1.142 | 0.078 |
| rbf_C1_g4 | 73.44 | 0.8643 | 1.816 | 0.077 |
| rbf_C10_g0.25 | 68.23 | 0.8122 | 0.799 | 0.073 |
| rbf_C10_g1 | 73.44 | 0.7065 | 1.259 | 0.072 |
| rbf_C10_g4 | 72.40 | 0.8338 | 1.808 | 0.086 |

The lowest development-NLL control is rbf_C10_g1; its accuracy exceeds calibrated counts by14.58points. Maximum grid accuracy is74.48%, so this observation protocol has headroom rather than the99.5% saturation of the cheap recency table. This admits an integrated learning test, not a native advantage claim. The older weak dense gesture screens are not the practical ceiling.

All nine fixed learned cells are preserved. Linear/RBF controls use fitting-only log-count centering/scaling. Counts use all640 time bins and3-fold fitting-only alpha/temperature calibration; this avoids treating correlated camera events as independent label evidence. SVM probability calibration is included in measured fitting.

Common raw preprocessing costs123.976s. The complete control campaign takes15.926s with411.1MiB peak process RSS. Table inference uses the complete192-example batch. Third-party solver FLOPs are unmeasured, not zero; CPU wall, arithmetic, storage and energy are distinct. Development selection and one fitted seed are exploratory evidence, not independent confirmation.

## Appendix B. Integrated real-packet quality and work

| Model | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| Calibrated counts | 58.85 | 1.3734 | Unmeasured | Unmeasured | Unmeasured |
| linear_C0.1 | 61.46 | 1.2036 | Unmeasured | Unmeasured | Unmeasured |
| linear_C1 | 59.38 | 1.5903 | Unmeasured | Unmeasured | Unmeasured |
| linear_C10 | 57.81 | 2.4610 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C1_g0.25 | 67.19 | 0.8804 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C1_g1 | 74.48 | 0.7358 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C1_g4 | 73.44 | 0.8643 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C10_g0.25 | 68.23 | 0.8122 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C10_g1 | 73.44 | 0.7065 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C10_g4 | 72.40 | 0.8338 | Unmeasured | Unmeasured | Unmeasured |
| Ours p16/L2/H2 | 65.10 | 0.9632 | 20.075 | 2.550 | 0.592 |

Same984 distinct fitting gestures and192 subject-disjoint development targets. Ours fits eight fixed passes:7,872 target presentations and496 Adam updates,U16, lr.003, seed6. Conventional solvers have their own convergence/calibration policies; equal pass or fitting-work protocols are not claimed. Every learned baseline is retained.

Selected native pass8: accuracy difference versus minimum-NLL control-8.33points, NLL improvement -0.2567. Development quality dominance flag: False. Independent practical advantage remains unproved; no pending, best-seed or official-test prediction fills this table.

Native estimates include complete prefix/query forward/backward, all scored keys, candidate values, counterfactual value credit, normalization/clipping and Adam. First/last windows are sampled by actual16/8 target size;2FLOPs/MAC and unit-weight special functions. Solver arithmetic is unmeasured. Preprocessing, validation, traffic and measured CPU latency must be charged separately; isolated inference estimates cannot establish total resource or energy advantage.

## Appendix B. Real-packet learning, capacity and total workflow

![local dvs native analysis 20261002T151000Z learning](report/figures/local_dvs_native_analysis_20261002T151000Z_learning.png)

| Completed native quantity | Value |
| --- | --- |
| Parameters / available receivers | 15,523 / 8 |
| Persistent state tensor bytes | 720 |
| Keys / selected updates / counterfactual values per fitting prefix | 168.0 / 84.0 / 168.0 |
| Fit plus nonfitting forward GFLOPs estimate | 21.256 |
| Native workflow / whole control grid seconds | 1557.590 / 15.926 |
| Common raw preprocessing seconds | 123.976 |
| Native / grid workflow plus preprocessing seconds | 1681.566 / 139.902 |
| Native peak process RSS MiB | 347.8 |

The unchanged native core computes through temporal races, separate keys/values, sparse persistent receiver updates and local mixing. Nonempty observed packets arrive at their physical closure and an observed query at1s; state resets per gesture. Race noise depends on pass/fitting seed, never clip identity, index or label. Counts are observed camera content rather than fitted statistical prediction experts.

Workflow wall includes fitting, validation, checkpoints and operation profiling; the complete control-grid wall includes all fixed solvers, calibration, scoring and serialization. Nonfitting neural forward work is estimated using completed prefix samples; raw preprocessing is common and added once to each pipeline. Small tensor state does not mean small process RSS, zero scoring cost, useful extra depth or zero optimizer work. Earlier numerical admission/research costs remain separately saved.

## Appendix B. Frozen practical inference and storage

| Model | Dev accuracy % | Dev NLL | Sequential ms / prefix | Saved model KiB |
| --- | --- | --- | --- | --- |
| Calibrated counts | 58.85 | 1.3734 | 0.013 | 55.4 |
| linear_C0.1 | 61.46 | 1.2036 | 0.163 | 66.1 |
| linear_C1 | 59.38 | 1.5903 | 0.161 | 66.1 |
| linear_C10 | 57.81 | 2.4610 | 0.167 | 66.1 |
| rbf_C1_g0.25 | 67.19 | 0.8804 | 0.717 | 4632.4 |
| rbf_C1_g1 | 74.48 | 0.7358 | 0.670 | 4730.5 |
| rbf_C1_g4 | 73.44 | 0.8643 | 0.711 | 5081.4 |
| rbf_C10_g0.25 | 68.23 | 0.8122 | 0.608 | 4451.8 |
| rbf_C10_g1 | 73.44 | 0.7065 | 0.671 | 4766.6 |
| rbf_C10_g4 | 72.40 | 0.8338 | 0.680 | 5081.4 |
| native | 65.10 | 0.9632 | 55.076 | 103.9 |

All models consume the same saved observed first-second packet counts, one prefix at a time on one CPU thread. Fit-only feature transformation and native race simulation are timed. Three deterministic repeats measure execution variation; they are not independent fits. Warmup, loading and common raw event coalescing are reported separately.

Every frozen probability matches the completed development result and every repeated prediction is identical. Native weights are the fixed minimum-NLL selected checkpoint; no model is refitted, temperature-adjusted or selected during this audit. All conventional cells remain visible, including faster or better alternatives.

Storage is uncompressed joblib serialization of each fitted model plus its necessary fitting-only transform. It is not resident process memory or memory traffic. Native parameter tensor bytes and persistent state bytes are separate quantities. CPU emulator latency cannot be relabelled as event-hardware latency or measured energy.

Subject-disjoint development evidence only. A quality/resource tradeoff here requires frozen independent confirmation before promotion. No test leakage, broad supremacy, useful-depth premium or dormant-capacity advantage is inferred from this audit.

## Appendix B. Real learned features versus the initial reservoir

| Frozen encoder | Original accuracy % | Fresh-head accuracy % | Fresh-head NLL | Head | Replay seconds | Head grid seconds |
| --- | --- | --- | --- | --- | --- | --- |
| initial | 12.50 | 57.81 | 1.1311 | rbf | 66.606 | 2.949 |
| selected | 65.10 | 67.19 | 0.9999 | linear | 66.168 | 1.915 |

Both encoders expose the same32-dimensional query feature and use the same984 fitting/192 development gestures and race draws. Every original prediction is reconstructed exactly by its saved head; replay and new head fitting preserve every encoder parameter. The initial encoder is the exact pre-fitting reservoir, not a separate tuned control.

Three linear C values and six RBF C/gamma cells are evaluated by3-fold fitting-only decoder NLL. The selected head is then fitted once on all fitting features. No head hyperparameter is selected on development. Scaling is fitting-fold only. Conditional decoder CV is not unbiased end-to-end validation because the selected encoder already saw all fitting labels during its original fit.

The fitted representation gives a better selected readout than the initial reservoir: 67.19% versus57.81%. This supports useful feature learning under this probe protocol. Replacing the fitted head increases accuracy only2.08points and worsens NLL relative to the original65.10%/.963161. The final decoder alone does not close the strong raw-input control gap; failed finite heads also do not prove information is absent.

This is a frozen diagnostic, not an architectural substitution or practical advantage. The selected encoder still pays its original20.075GFLOPs/1557.590s workflow, plus full replay and every readout fit shown here. Solver arithmetic remains unmeasured. Useful extra depth, semantic language features, whole-route gradient accuracy and independent confirmation are separate questions. No official test was opened.

Theory75 states admission and scope. Selected initial/fitted readout artifacts, all fitting-CV cells, native/checkpoint hashes, probability arrays, wall and RSS are saved.

## Appendix B. Shared fitting noise: covariance hypothesis fails its gate

| Saved seed | All variance ratio | Route ratio | Time ratio | Message ratio | Gate |
| --- | --- | --- | --- | --- | --- |
| 6 | 0.99787 | 1.00166 | 1.04860 | 0.99099 | FAIL |
| 7 | 1.00178 | 0.99119 | 0.97407 | 0.99542 | FAIL |

Four fixed fitting prefixes per saved native seed6/7, 32 independent whole-history draws and zero optimizer updates. Each clip gradient is recorded under the same stream. Shared-batch trace covariance is measured; independent-coupling covariance is estimated from those same marginal samples. Their difference equals the cross-clip covariance sum.

Both route-map ratios miss the declared 1.20 admission gate. Proposed fresh_shared/ fresh_independent fits are stopped. Neither model shows a large cross-clip covariance penalty on these prefixes. This is not a corpus-wide result or a test of repeated-noise adaptation across optimizer updates. Small-group variance ratios retain their finite-sample scope.

Theory93 derives the effective-batch covariance law and separates per-window freshness from cross-row decorrelation. Independent noise need not help when cross-covariance is negative. Flipout is a primary-paper analogy about shared perturbation correlation, not an implementation or transferred quality guarantee for these event races.

Two algebra contracts pass. Audit 56.443s/396.2MiB; 0.637545 known diagnostic GFLOPs est. Complete first-draw per-clip forward/backward traces times draws; reporting reductions separate. Frozen weights preserved. No training, official test or superiority claim.

## Appendix B. Causal persistent-state access: positive probe, qualified cause

| Frozen encoder | Feature access | Dimensions | Dev accuracy % | Dev NLL |
| --- | --- | --- | --- | --- |
| initial | Query | 32 | 57.81 | 1.1311 |
| initial | Query+state | 176 | 63.54 | 0.8862 |
| selected | Query | 32 | 67.19 | 0.9999 |
| selected | Query+state | 176 | 71.35 | 0.8591 |

| Encoder | Original fit GF est. | Fit MF / presentation | Core replay GF est. | Replay MF / prefix | Decoder/grid GF |
| --- | --- | --- | --- | --- | --- |
| initial | 0.000000 | 0.000000 | 1.735341 | 1.475630 | Unknown |
| selected | 20.075193 | 2.550202 | 1.735341 | 1.475630 | Unknown |

Same 984 fitting/192 development gestures. Augmentation adds all eight pre-query receiver memories, their ages at source query admission and occupancy flags to the actual 32-dimensional query feature. No future or target-dependent feature. Three contracts reproduce serial state, original probabilities and saved query-probe probabilities while keeping every encoder parameter bitwise fixed. Nine-cell three-fold fitting-only head selection.

Initial augmentation improves NLL by .244819; trained augmentation by .140774. Both pass the declared diagnostic access signal. The trained augmented probe reaches 71.35%/ .859084, versus original native 65.10%/.963161, compact prototype 66.67%/.902951 and strong full RBF 73.44%/.706478. Decoder/grid costs are unknown, so these are quality comparisons and diagnostic signals, not practical advantage.

Important qualification from the completed partition controls: stronger query-only regularization alone reaches .889842; adding full state at that same C contributes only .030759 further NLL improvement. At the weaker C, full-state NLL worsens to 1.278955. Most apparent trained-state gain is therefore compatible with regularization rather than uniquely missing information. Early/later state partitions retain smaller gains.

Theory94/95. Dense all-state probing is a diagnostic, not sparse inference or a new main architecture. Selected encoder saw all fitting labels and was dev-selected; decoder CV is not unbiased end-to-end validation. Known original fit uses 7,872 presentations; current two-pass core replay uses 1,176 prefixes. Solver/materialization/ traffic/energy and total probe fit/inference remain unmeasured rather than zero.

## Appendix B. Retained payloads, clocks, layers and regularization

![local dvs state partition probe 20261002T214000Z learning](report/figures/local_dvs_state_partition_probe_20261002T214000Z_learning.png)

| Encoder | Added state/control | Dims | Accuracy % | NLL | Kind/C |
| --- | --- | --- | --- | --- | --- |
| initial | Payloads | 160 | 63.54 | 0.8862 | rbf/10.0 |
| initial | Clocks | 48 | 55.21 | 1.1245 | rbf/10.0 |
| initial | Layer0 | 104 | 66.15 | 0.9048 | rbf/10.0 |
| initial | Layer1 | 104 | 64.58 | 0.9545 | rbf/10.0 |
| initial | Query@state setting | 32 | 57.81 | 1.1311 | rbf/10.0 |
| initial | State@query setting | 176 | 63.54 | 0.8862 | rbf/10.0 |
| selected | Payloads | 160 | 71.35 | 0.8603 | linear/0.1 |
| selected | Clocks | 48 | 67.71 | 1.0178 | linear/1.0 |
| selected | Layer0 | 104 | 70.83 | 0.8383 | linear/0.1 |
| selected | Layer1 | 104 | 69.79 | 0.8403 | linear/0.1 |
| selected | Query@state setting | 32 | 67.19 | 0.8898 | linear/0.1 |
| selected | State@query setting | 176 | 68.23 | 1.2790 | linear/1.0 |

Four partitions retain the query: both payload layers, ages/occupancy only, layer0 state or layer1 state. Same nine-cell fitting-only decoder selection per partition, both initial and trained encoders. Two fixed configuration swaps per encoder test regularization without another grid or development selection.

Trained payload .860321, layer0 .838315 and layer1 .840329; clocks-only 1.017809. The query C.1 control .889842 explains most full-state gain. These finite probe differences do not isolate a causal depth failure or population conditional information. All outcomes remain visible; core producers/routes were never retrained.

Theory95. Cached features add zero CORE replay, not zero fitting work. Prior encoder fit/core replay retained; additional solver arithmetic unknown. Conditional folds share the label-trained encoder. Producer-held fitting examples are the next decoder-selection check. No official test or supremacy claim.

## Appendix B. Completed packet-scale clock comparison

| Model | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| Calibrated counts | 58.85 | 1.3734 | Unmeasured | Unmeasured | Unmeasured |
| linear_C0.1 | 61.46 | 1.2036 | Unmeasured | Unmeasured | Unmeasured |
| linear_C1 | 59.38 | 1.5903 | Unmeasured | Unmeasured | Unmeasured |
| linear_C10 | 57.81 | 2.4610 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C1_g0.25 | 67.19 | 0.8804 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C1_g1 | 74.48 | 0.7358 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C1_g4 | 73.44 | 0.8643 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C10_g0.25 | 68.23 | 0.8122 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C10_g1 | 73.44 | 0.7065 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C10_g4 | 72.40 | 0.8338 | Unmeasured | Unmeasured | Unmeasured |
| Ours original | 65.10 | 0.9632 | 20.075 | 2.550 | 0.592 |
| Ours packet-scale clock | 66.15 | 1.0420 | 20.075 | 2.550 | 0.592 |
| Compact prototype33 | 66.67 | 0.9030 | Unmeasured | Unmeasured | Unmeasured |

Both native fits use984 gestures/eight fixed passes,7,872 presentations and496 updates; 192 subject-disjoint dev targets select pass8 by minimum NLL. Only initial decay rates and rotation frequencies are scaled to the observed50ms packets. Temporal races, key/value separation, sparse updates and counterfactual learning remain. Parameters15,523; available receivers8; per-prefix168 key scores,84 selected updates,168 candidate values.

Clock fit wall1565.128s; peak RSS347.2MiB. Accuracy rises1.04points versus original while NLL worsens. Both trail the selected full kernel and compact prototype in both quality measures. This initialization change does not establish advantage; the original lower-NLL result and all control cells are retained.

Same units and native target denominators in every work column. Solver FLOPs remain unmeasured; unequal solver policies and development tuning preclude iso-FLOP superiority. Raw preprocessing, validation, checkpoints, traffic and energy remain separate. One seed; no official-test access. Source: local_dvs_clock_full_20261002T153000Z.json; theory76.

## Appendix B. Full and partial-window batched learning admission

| Credit | Initial fit NLL | Selected fit NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| local | 2.6409 | 2.1145 | 0.107699 | 2.2437 | 0.5917 |
| pairs | 2.6409 | 2.1133 | 0.108205 | 2.2543 | 0.5917 |

| Credit | Workflow seconds | Peak RSS MiB | U16 updates | U8 updates |
| --- | --- | --- | --- | --- |
| local | 20.821 | 334.8 | 2 | 2 |
| pairs | 20.524 | 334.9 | 2 | 2 |

Each arm uses24 fitting gestures/two fixed passes and8 development targets:48 target presentations/four Adam updates. Both select pass2 and score25% dev accuracy; dev NLL 2.3115 local and2.3070 pairs. These tiny fixed smokes verify learning and resource readiness, not prediction advantage. All optimizer stages have complete operator coverage. Accounting includes candidate values, backward, normalization/clipping and Adam.

Same p16/L2/H2/pool2 integrated architecture:15,523 parameters,8 available receivers, 21 events per prefix,168 scored keys/84 selected state updates/168 candidate values, 720 persistent-state tensor bytes. Independent-clip batching passes forward/state and every-parameter gradient equality plus actual interrupted model/Adam/cursor recovery.

Optional pair credit enumerates the actual final-query two-head outcome losses. Its conditional risk and all derivatives match explicit enumeration; earlier routes retain local surrogate credit. Inference still delivers two hard winners per layer. No exact whole-core gradient or useful-depth claim follows. The bounded matched pilot is next; stronger full-data controls are reported above and are not comparable tiny-fit controls.

One guarded one-thread job at a time; RSS watchdog and8GiB available-memory floor. Theory77; completed admission record local_dvs_batched_smoke_admission_20261002T163600Z.json. All negative full-fit results remain visible; no pending score fills an evidence table.

## Appendix B. Completed matched credit pilot: terminal pairs / seed6

![local dvs credit comparison pairs 20261002T174000Z learning](report/figures/local_dvs_credit_comparison_pairs_20261002T174000Z_learning.png)

| Credit | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| local | 54.17 | 1.3059 | 2.285696 | 2.2321 | 0.5917 |
| terminal pairs | 56.77 | 1.3498 | 2.296461 | 2.2426 | 0.5918 |

| Credit | Workflow seconds | Peak RSS MiB | Keys / fit target | Commits / fit target | Values / fit target |
| --- | --- | --- | --- | --- | --- |
| local | 94.464 | 338.9 | 168 | 84 | 168 |
| terminal pairs | 90.585 | 339.4 | 168 | 84 | 168 |

Fixed256 fit/192 development gestures, four passes/1,024 target presentations/64 Adam updates; same initialization, causal packets, draws and minimum-devNLL selection. Unchanged p16/L2/H2/pool2 architecture:15,523 parameters/eight available receivers; inference168 scored keys/84 commits/168 candidate values per21-event prefix. Fitting activity includes shadow replay where present.

Promotion gate FAILS: NLL improvement -0.043869, accuracy decline-2.604pp, whole-fit work ratio1.0047. Required gain>=.02, decline<=1pp, work ratio<=1.10, peak RSS<900,000KiB. A failed gate stops unchanged confirmation/scale-up; all passes and negative findings retained.

All forward/replay, backward, normalization/clipping and Adam paid;2FLOPs/MAC plus unit specials. Strong984-fit RBF73.44%/.7065 and compact66.67%/.9030 references have unequal fitting data; their solver FLOPs remain unmeasured. See common-unit full-fit tables above. One seed, no official test or superiority claim. local_dvs_credit_comparison_pairs_20261002T174000Z.json; theory77/80–84.

## Appendix B. Completed matched credit pilot: state choice / seed6

![local dvs credit comparison state choice s6 20261002T192100Z learning](report/figures/local_dvs_credit_comparison_state_choice_s6_20261002T192100Z_learning.png)

| Credit | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| local | 54.17 | 1.3059 | 2.285696 | 2.2321 | 0.5917 |
| state choice | 58.85 | 1.1940 | 3.059796 | 2.9881 | 0.5917 |

| Credit | Workflow seconds | Peak RSS MiB | Keys / fit target | Commits / fit target | Values / fit target |
| --- | --- | --- | --- | --- | --- |
| local | 94.464 | 338.9 | 168 | 84 | 168 |
| state choice | 103.948 | 339.8 | 336 | 168 | 336 |

Fixed256 fit/192 development gestures, four passes/1,024 target presentations/64 Adam updates; same initialization, causal packets, draws and minimum-devNLL selection. Unchanged p16/L2/H2/pool2 architecture:15,523 parameters/eight available receivers; inference168 scored keys/84 commits/168 candidate values per21-event prefix. Fitting activity includes shadow replay where present.

Promotion gate PASSES: NLL improvement 0.111931, accuracy decline-4.688pp, whole-fit work ratio1.3387. Required gain>=.02, decline<=1pp, work ratio<=1.50, peak RSS<900,000KiB. A failed gate stops unchanged confirmation/scale-up; all passes and negative findings retained.

All forward/replay, backward, normalization/clipping and Adam paid;2FLOPs/MAC plus unit specials. Strong984-fit RBF73.44%/.7065 and compact66.67%/.9030 references have unequal fitting data; their solver FLOPs remain unmeasured. See common-unit full-fit tables above. One seed, no official test or superiority claim. local_dvs_credit_comparison_state_choice_s6_20261002T192100Z.json; theory77/80–84.

## Appendix B. Completed matched credit pilot: state choice / seed7

![local dvs credit comparison state choice s7 20261002T193500Z learning](report/figures/local_dvs_credit_comparison_state_choice_s7_20261002T193500Z_learning.png)

| Credit | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| local | 55.21 | 1.3065 | 2.285696 | 2.2321 | 0.5919 |
| state choice | 55.21 | 1.3362 | 3.059796 | 2.9881 | 0.5918 |

| Credit | Workflow seconds | Peak RSS MiB | Keys / fit target | Commits / fit target | Values / fit target |
| --- | --- | --- | --- | --- | --- |
| local | 96.754 | 338.9 | 168 | 84 | 168 |
| state choice | 101.906 | 339.5 | 336 | 168 | 336 |

Fixed256 fit/192 development gestures, four passes/1,024 target presentations/64 Adam updates; same initialization, causal packets, draws and minimum-devNLL selection. Unchanged p16/L2/H2/pool2 architecture:15,523 parameters/eight available receivers; inference168 scored keys/84 commits/168 candidate values per21-event prefix. Fitting activity includes shadow replay where present.

Promotion gate FAILS: NLL improvement -0.029693, accuracy decline0.000pp, whole-fit work ratio1.3387. Required gain>=.02, decline<=1pp, work ratio<=1.50, peak RSS<900,000KiB. A failed gate stops unchanged confirmation/scale-up; all passes and negative findings retained.

All forward/replay, backward, normalization/clipping and Adam paid;2FLOPs/MAC plus unit specials. Strong984-fit RBF73.44%/.7065 and compact66.67%/.9030 references have unequal fitting data; their solver FLOPs remain unmeasured. See common-unit full-fit tables above. One seed, no official test or superiority claim. local_dvs_credit_comparison_state_choice_s7_20261002T193500Z.json; theory77/80–84.

## Appendix B. Completed matched credit pilot: state clock / seed6

![local dvs credit comparison state clock 20261002T174600Z learning](report/figures/local_dvs_credit_comparison_state_clock_20261002T174600Z_learning.png)

| Credit | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| local | 54.17 | 1.3059 | 2.285696 | 2.2321 | 0.5917 |
| state clock | 47.40 | 1.4507 | 3.060638 | 2.9889 | 0.5918 |

| Credit | Workflow seconds | Peak RSS MiB | Keys / fit target | Commits / fit target | Values / fit target |
| --- | --- | --- | --- | --- | --- |
| local | 94.464 | 338.9 | 168 | 84 | 168 |
| state clock | 99.314 | 339.5 | 336 | 168 | 336 |

Fixed256 fit/192 development gestures, four passes/1,024 target presentations/64 Adam updates; same initialization, causal packets, draws and minimum-devNLL selection. Unchanged p16/L2/H2/pool2 architecture:15,523 parameters/eight available receivers; inference168 scored keys/84 commits/168 candidate values per21-event prefix. Fitting activity includes shadow replay where present.

Promotion gate FAILS: NLL improvement -0.144750, accuracy decline6.771pp, whole-fit work ratio1.3390. Required gain>=.02, decline<=1pp, work ratio<=1.50, peak RSS<900,000KiB. A failed gate stops unchanged confirmation/scale-up; all passes and negative findings retained.

All forward/replay, backward, normalization/clipping and Adam paid;2FLOPs/MAC plus unit specials. Strong984-fit RBF73.44%/.7065 and compact66.67%/.9030 references have unequal fitting data; their solver FLOPs remain unmeasured. See common-unit full-fit tables above. One seed, no official test or superiority claim. local_dvs_credit_comparison_state_clock_20261002T174600Z.json; theory77/80–84.

## Appendix B. Actual-write choice: independent-seed gate fails

| Seed | NLL improvement | Accuracy gain pp | Fit work ratio | Declared gate |
| --- | --- | --- | --- | --- |
| 6 | +0.111931 | +4.6875 | 1.33867 | Pass |
| 7 | -0.029693 | -0.0000 | 1.33867 | FAIL |

Same256 fitting/192 development gestures, four fixed passes and1,024 presentations per arm/seed. Seed6 improves54.17%/1.305937 to58.85%/1.194006. Seed7 gives55.21%/1.306508 local versus 55.21%/1.336201 corrected credit. Every pass and both completed comparisons are retained.

The unchanged method fails its independent confirmation gate. No full984-fit campaign, extra pass extension or best-seed promotion is admitted. Numerical credit correctness remains established for the isolated categorical component; repeatable predictive benefit is not. Averaging seeds does not override the predeclared gate.

Next distinguish information access, branch/content credit exposure, common-clock noise and private-map dilution on fitting-only frozen probes. Alternative no-grad forwards teach choices without directly teaching losing content on that realization. Winner-sampled branch derivatives can nevertheless be correct in expectation; rare exposure and other approximate teachers require separate tests. A larger pool is not an automatic repair.

Theory87/88 prioritize one diagnosed repair, numerical contracts, a tiny integrated fit and a prespecified matched-work comparison before independent confirmation. Joint versus alternating route/message updates is a conditional schedule experiment, not a change to the coupled temporal forward computation or a currently demonstrated advantage.

## Appendix B. Why actual write and joint-clock credit need a test

| Frozen model | Opposed directions | Mean |value effect| | Mean |write effect| | Mean |interaction| |
| --- | --- | --- | --- | --- |
| native_full_20261002T145300Z | 4/15 | 0.01699 | 0.00602 | 0.00681 |
| batched_local_pilot_s6_20261002T164700Z | 5/24 | 0.00693 | 0.00130 | 0.00004 |
| batched_pairs_pilot_s6_20261002T164700Z | 5/24 | 0.01225 | 0.00165 | 0.00003 |

Replay32 fixed event/head sites on the first four previously used development clips/model. Both legal winner choices change delivered content and actual persistent writes, with full suffix replay at unchanged current first time and future random draws. Two additional value/write hybrids are diagnostic, not legal routes. Nonzero-comparable direction counts exclude tiny/zero gradients. Absolute effects are not additive attribution percentages.

In the original full model, one alternative improves value-only loss by.01016 but worsens write-only loss by.10514; its legal complete-route effect is+.09193 NLL. The averaged local value teacher favors that harmful alternative. These four original-model clips are correctly classified: this is a conditional credit defect, not attribution of the overall error rate.

Theory78 independently proves a convex affine cross-entropy example where the two-route local teacher reverses exact expected-loss descent. Actual reference/backward contracts pass. The full-state correction replaces one earlier score derivative by pi_i(F_i-b) minus lambda_i*T times the probability-weighted centered loss. It retains sampled content derivatives and includes common-clock credit through later timing jumps. The isolated-node joint likelihood component is exact in expectation; other route teachers remain local, and variance is unresolved.

Routes, messages and time parameters do learn: frozen original/256-fit local/pair models change35.8%/26.2%/25.2% of16,128 audited choices versus initialization, with nonzero updates in every parameter group. Removing routing score-gradient paths leaves the factual forward unchanged but removes incoming-content/message gradients. This does not prove their fitted utility or diagnose the full quality gap.

Learned-window gradients were tested as smooth primitives; fixed-count repeated-arrival fits are a separate mechanism. New silence-burst/popcorn contracts cover scheduled deadlines, causal EOF, fixed-partition gradients and exact finite timeout-bank risk. No integrated learned window or popcorn fit is established. Hard merge/split credit and an irregular adapter remain prerequisites; fixed50ms packet schedules do not test natural silence.

Sources: completed frozen route-content audit171500Z, suffix audit172000Z, curvature contracts165900Z, silence-burst contracts170500Z and joint-clock contracts173000Z; theory78–81. Numerical diagnostics are not held-out superiority evidence.

## Appendix B. Frozen variance audit after the failed clock pilot

| Selected checkpoint | Choice RMS | Common-clock RMS | Clock / choice RMS | Summed variance |
| --- | --- | --- | --- | --- |
| Local | 0.000290 | 0.385649 | 1332 | 2.385095 |
| Joint-clock treatment | 0.000469 | 0.423377 | 904 | 3.252833 |

First four previously used fitting prefixes/model, event9/both layers/heads. Replay both legal delivered values and actual persistent writes through the complete suffix at8/16 exponential-time quadrature nodes, with fixed future draws. Each model pays192 complete legal shadow forwards. No optimizer or new quality score. Magnitudes and summed variances are in score space with the same per-clip/probe convention for both rows.

Decompose the conditional-winner joint score into choice credit pi_i(F_i-R) and common-clock credit pi_i(1-Lambda*T)(R-b). If branch losses are independent of T, the common-clock mean is zero and baseline mismatch alone produces variance ||pi||^2(R-b)^2. Choosing b=R removes it exactly; the constant-outcome numerical contract passes. Time-dependent later route jumps can make the common-clock mean useful, so deleting it universally is not justified.

The current intermediate decoder baseline differs substantially from actual suffix risk. Across these limited probes the common-clock RMS is900–1,300 times the choice RMS; an oracle baseline computed with all quadrature replays removes more than99.9999% of estimated variance. That oracle is a diagnostic ceiling with all replay cost paid, not a deployable cheap baseline, optimizer-noise measurement or fitted improvement.

Eight-versus16-node mean-gradient L2 differences range4.3e-8 to7.3e-7. This agrees numerically on these probes but does not establish convergence across discontinuous histories. Per-pass shared draws also prevent assuming ordinary minibatch variance reduction; actual batch covariance was not measured. Neither audit proves the cause of the held-out regression or the primary bottleneck on other benchmarks.

Decision: stop the failed joint-clock campaign. Next isolate exact conditional actual-write choice credit while retaining the native pathwise clock derivative and all temporal/sparse mechanisms. This retains a known approximation for downstream timing jumps; numerical contracts and one small smoke precede any matched fit. A later independent-noise/control-variate comparison needs fresh matched local controls and full recovery/work accounting. No automatic wider capacity or pass extension.

Completed diagnosticlocal_dvs_joint_credit_variance_20261002T175100Z.json;23.205s/307.3MiB peak RSS, one guarded CPU job. Theory82/83 and COUNTERFACTUAL_CREDIT_PLAN.md preserve proof, negative fit and limited diagnostic scope.

## Appendix B. Actual-write choice credit: integrated admission

| Credit | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| Write-choice + native timing | 25.00 | 2.3113 | 0.143986 | 2.999703 | 0.591695 |

Fixed24 fit/eight dev/two passes,48 presentations/four Adam updates: two U16 and two partial U8 windows. Fitting NLL2.6409 to2.1106; workflow23.041s, peak RSS335.5MiB. This is readiness, not benchmark advantage.

Unchanged native p16/L2/H2/pool2:15,523 parameters/eight available receivers. Fitting pays336 key scores/168 state commits/336 candidate values per target including the complete shadow. Inference retains168 keys/84 commits/168 values over21 events;720 persistent-state tensor bytes.

At one event9 race head/window, substitute the exact conditional expected-loss derivative for both legal delivered values and actual persistent writes. Retain the separate native winner-delay derivative, winning content/state credit and all other local teachers. The obsolete intermediate decoder baseline is unnecessary for categorical enumeration.

Contracts independently enumerate the choice risk, differentiate smooth raw clocks, preserve winning payload and other-head credit, reproduce factual state/logits and actual alternative commits, and verify every parameter-gradient change equals the upstream VJP of the replaced score residual. Interrupted Adam/cursor/RNG recovery and complete operator accounting pass. Downstream timing jumps and other local teachers remain approximate.

Theory84; contracts191100Z and smoke191400Z. All losses summed before gradient normalization, global clipping and Adam; all shadow work charged. No official test or exact whole-model gradient claim. Fixed matched pilot and second-seed gate follow.

## Appendix B. Calibration: noisy routing changes global gradient balance

| Frozen weights / credit | Total norm | Clip scale | Cosine to local | Replacement norm |
| --- | --- | --- | --- | --- |
| Pilot256 / local | 3.5519 | 0.2815 | 1.0000 | 0.0000 |
| Pilot256 / choice | 3.5519 | 0.2815 | 1.0000 | 0.0052 |
| Pilot256 / clock | 3.8068 | 0.2627 | 0.8413 | 2.0875 |
| Full984 / local | 3.3901 | 0.2950 | 1.0000 | 0.0000 |
| Full984 / choice | 3.3915 | 0.2949 | 1.0000 | 0.0215 |
| Full984 / clock | 6.4002 | 0.1562 | 0.3940 | 5.9463 |

| Frozen weights / credit | Context norm | Message norm | Route norm | Time norm | Decoder norm |
| --- | --- | --- | --- | --- | --- |
| Pilot256 / local | 3.1646 | 0.6550 | 0.0929 | 0.0239 | 1.4707 |
| Pilot256 / choice | 3.1646 | 0.6551 | 0.0928 | 0.0238 | 1.4707 |
| Pilot256 / clock | 3.1557 | 1.0448 | 1.1304 | 0.0284 | 1.4707 |
| Full984 / local | 2.9646 | 0.4905 | 0.1972 | 0.0238 | 1.5568 |
| Full984 / choice | 2.9656 | 0.4922 | 0.2016 | 0.0239 | 1.5568 |
| Full984 / clock | 5.4179 | 1.8514 | 2.3982 | 0.0729 | 1.5568 |

Same selected weights/first16 fitting prefixes/draw314159/event9 layer0 head0 for each local, actual-write choice+native timing and joint-clock intervention. Factual logits and losses are identical. All gradients summed then divided by16 before computing the native norm1 clip multiplier. No optimizer, fitting selection or new quality score.

For Full984, the noisy joint-clock correction raises total gradient norm3.3901 to 6.4002, rotates its direction to cosine.3940 and reduces the global clip multiplier .2950 to.1562. The decoder gradient is unchanged before clipping, so the global multiplier also attenuates its useful supervised update. This directly measures cross-path balance on one draw; it is not attribution of the held-out regression.

Write-choice credit keeps total norm3.3915, cosine.99998 and clip multiplier.2949 while correcting actual legal memory utility. Positive scaling alone cannot repair a wrong direction or manufacture useful information. Per-sample norm equalization can bias a zero-mean signal; a prefix-only clock baseline or independent calibration requires a separate matched protocol. Shared weights retain both message and timing credit.

Completed auditlocal_dvs_credit_gradient_balance_20261002T191600Z.json;31.254s/356.6MiB. Every intervention's forward/replay, backward and normalization operator coverage passes. No measured expected variance, batch covariance, Adam trajectory or fitted benefit from gradient normalization follows.

## Appendix B. Credit calibration: coordinates and limits

Completed calibration contracts: 5 passed. Common log-rate c and relative logits u define rates exp(c)*softmax(u). The joint winner/time score pulls back to common-clock credit(1-Lambda*T)*(R-b), and categorical credit pi_i*(F_i-R). The choice component is independent of common baseline error; the clock component remains useful and noisy.

Recomposing c=logsumexp(old scores) and u=old scores preserves BOTH the old forward and the old gradient. A coordinate identity does not remove clock noise. Separately learned output maps require an architectural comparison; shared incoming content can still receive both derivatives. Eligible-key normalization and discovery remain paid.

A positive score-space rescaling can oppose parameter descent through coupled Jacobians. The contract gives true parameter gradient[-1,-1] versus rescaled[8,-1], dot product-7. Per-sample unit normalization also biases an explicitly zero-mean scalar estimator. RMS equalization is therefore not a universal calibration rule.

Primary-paper analogues suggest variance-trained legal control variates(RELAX), conditional averaging, phasic updates with constrained route drift(PPG), exposure accounting(MoE) and event derivative jumps(EventProp). These are distinct repairs. MoE common-logit z-loss would regularize this race clock itself; higher-order optionality needs correct stochastic derivatives(DiCE), not repeated differentiation of detached first-order teachers.

Theory85 mathematical contracts and theory88 linked primary papers; no fitted benefit from these normalization/alternation proposals. Parameter-space covariance and branch exposure on independent-noise fitting probes determine the next single repair. Keep coupled forward messages/races/timing, sparse addressed writes and all existing negative evidence.

## Appendix B. Larger alternative pools: paired-credit readiness

| Credit / pool | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| Enumerated2 / readiness | 25.00 | 2.3113 | 0.143986 | 2.999703 | 0.591695 |
| Paired8 / readiness | 12.50 | 2.2989 | 0.414378 | 8.632876 | 0.872348 |

Both24 fit/eight dev/two passes/48 presentations/four Adam updates,U16+partial U8. Paired8 fitting NLL2.5170 to2.1739; 27.533s/349.6MiB peak RSS. These tiny fits establish readiness, not quality rankings, larger-pool headroom or practical superiority.

Paired8 has42,091 parameters/32 available receivers versus15,523/eight for pool2. Per21-event inference prefix:672 scored keys/84 selected commits/672 candidate values; persistent state2,376 bytes. Fitting includes the full shadow: 1344 keys/168 commits/1344 values per target. More state does not mean free key discovery or training.

One full alternative write/suffix forward per fitting window supplies loss-difference credit with recorded proposal propensity. Epsilon.1 gives importance multiplier at most 1/.9; eight auxiliary proposal exponentials/target are recorded separately from arithmetic. Finite2/3/8/64-candidate contracts match independent expected-risk derivatives; pool2 nests enumerated credit exactly. Pool8 legal writes, every-parameter pool2 nesting and actual Adam/cursor/RNG recovery with full/partial windows pass; all operators are covered.

Contracts195800Z, smoke201100Z; theory86/87. Inference and native timing remain unchanged at each shape. No-grad alternatives do not directly teach losing payload maps. Other local teachers and future timing jumps remain approximate. The failed initial195200Z operator-accounting attempt is preserved; covered equivalent subtraction passes a fresh run.

## Appendix B. Fitting-only branch exposure and parameter interference

| Saved seed | Opposed selected choice | Opposed global difference | Mean message cosine | Min sampled/averaged cosine |
| --- | --- | --- | --- | --- |
| 6 | 0/16 | 16/16 | -0.0481 | 0.999995 |
| 7 | 3/16 | 16/16 | -0.0739 | 0.999970 |

First two fitting prefixes of each saved native local seed6/7 model; four independent whole-history draws, events9/19, both layers, head0. Actual branch writes and full suffixes preserve current first time. Native gradients are compared with an explicit diagnostic blocking ALL race-score paths, including raw-clock score sensitivities. No optimizer or development-label selection; these prefixes do not represent all gestures.

Route/clock-path differences mildly oppose message gradients in both saved models. Averaging both legal message branches leaves the full parameter-gradient direction almost unchanged on these prefixes. Negative inner product alone does not prove harmful interference or justify deleting a chain-rule term. Finite whole-history covariance estimates include changing entering states and clocks; they are not current-node conditional variance.

Independent ordinary-autograd gather contracts reproduce all parameters and full state for six score-blocked factual/legal branches. Detached-probability weighting separates branch derivatives from categorical derivatives. All captured native/branch forwards, backwards and residual VJPs have complete operation coverage; reporting reductions are outside that ledger.

Completed audit203600Z: 273.721s/371.5MiB. Contracts203400Z. No causal explanation of seed7 failure, global exact-gradient claim or benchmark advantage. User-proposed phase offsets have a separate coupled-time hypothesis.

## Appendix B. Exact probabilities do not remove winner-dependent credit bias

| Conditional gradient | Candidate0 score | Candidate1 score |
| --- | --- | --- |
| True hard-outcome risk | +.018750 | -.018750 |
| Original teacher expectation | +.018750 | -.018750 |
| Exact-pi teacher expectation | -.028125 | +.028125 |

Four completed contracts use actual backward implementations and independent autograd. A convex quadratic loss.5*(value-.6)^2 with values0/1 and race rates1/3 gives probabilities .25/.75 and legal losses.18/.08. The original local teacher happens to equal the true categorical derivative here; the exact-probability replacement reverses its direction.

Replacing a random rate coefficient by known pi preserves the old expectation when downstream error is fixed. In a nonlinear model that error depends on the winner, its write and subsequent routes. The full gradient is therefore not generally unchanged-expectation or zero variance. In the quadratic witness, replacement optimizes loss at the MEAN value; hard-delivery expected loss also contains a value-variance derivative.

Protocol correction beside theory59 section400: the earlier curie192000Z fidelity audit forces each candidate with its individual arrival time, changing identity AND timing. Conditional first time has mean.25 for both winners; unconditioned individual times have means1/.333333. Its original sign/magnitude numbers remain combined-intervention evidence, not an exact fixed-time route-credit diagnosis. Existing models/results are preserved.

Theory89; contracts204000Z. Exact-pi remains an empirical candidate separately owned by the other host. Neither this counterexample nor the earlier diagnostic establishes general superiority of either local teacher. Correct counterfactuals match actual sparse delivery, legal writes, time law and downstream utility.

## Appendix B. Evolution offset preserves signal-time coupling

| Schedule | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| alternating | 25.00 | 2.3294 | 0.107416 | 2.237827 | 0.596655 |
| joint | 25.00 | 2.3032 | 0.107803 | 2.245888 | 0.596655 |

Phase=frequency*physical_age+beta, beta=pi*tanh(raw_offset). Physical-age damping, emission delays, receiver readiness and stored timestamps remain. The32 offsets calibrate reception phase; they do not create a separate signal clock. Time still drives representation evolution and receives timing derivatives. Initial zero reproduces the original model.

Both24fit/eightdev/two passes/48 presentations/four updates,U16+partialU8; p16/L2/H2/pool2 has15,555 parameters/eight receivers,168 inference keys/84 commits/168 candidate values over21 events,720 persistent-state bytes. Joint fitting NLL2.64095 to2.09915;alternating to2.14262. Both smokes take21.0 seconds. Tiny development scores establish readiness, not quality ranking or advantage against strong controls.

Alternating message/route windows train the offset only with message parameters. Inactive parameters and Adam momentum/steps remain fixed. Shared content/context maps receive the full derivative every window; their updates can still change routes. No content detach or changed credit estimator. Private blocks receive fewer updates and active-gradient clipping differs at the same total presentation count.

Four contracts205000Z: zero-offset state/logit/original-gradient identity; nonzero serial/batched identity; physical-age/offset finite differences and independent directions with damping; phase ownership and actual interrupted recovery/accounting. Theory90. Pure rotation can make offset/time locally redundant; constant phase is a known calibration operator, not a universal capacity or convergence theorem.

## Appendix B. Completed coupled-time offset pilot: seed6

![local dvs evolution offset comparison s6 20261002T210100Z learning](report/figures/local_dvs_evolution_offset_comparison_s6_20261002T210100Z_learning.png)

| Schedule | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| local | 54.17 | 1.3059 | 2.285696 | 2.232125 | 0.591737 |
| joint | 57.29 | 1.2582 | 2.287775 | 2.234155 | 0.596711 |
| alternating | 53.65 | 1.2630 | 2.281584 | 2.228110 | 0.596655 |

| Offset schedule | NLL improvement | Accuracy gain pp | Fit work ratio | Gate |
| --- | --- | --- | --- | --- |
| joint | +0.047767 | +3.1250 | 1.000910 | Pass |
| alternating | +0.042929 | -0.5208 | 0.998201 | Pass |

Fixed256 fit/192 dev/four passes/1,024 presentations/64 Adam updates. Same native initial predictions, data, draws and minimum-devNLL selection. Native15,523 parameters versus15,555 with offsets; eight available receivers and identical selected activity. All offset operators, backward, clipping and active Adam work are charged.

Candidate gate requires>=.02NLL improvement,<=1pp accuracy decline,<=1.50 fitting work ratio and<900,000KiB RSS. Selected schedule: joint. Seed7 joint confirmation fails:1.327347 versus native1.306508NLL and2.0833pp accuracy decline. No unchanged full-data stage or extra epochs is admitted. Two seed6 schedules were declared before either result; all failures remain.

Preserved strong984-fit RBF73.44%/.7065 and compact66.67%/.9030 controls have unequal fitting data; solver FLOPs unmeasured. Alternation also changes private-block update counts/clipping, so this is an algorithm comparison, not isolated interference attribution. Physical time still drives content; no official test or supremacy claim.

## Appendix B. Completed coupled-time offset pilot: seed7

![local dvs evolution offset comparison s7 20261002T210800Z learning](report/figures/local_dvs_evolution_offset_comparison_s7_20261002T210800Z_learning.png)

| Schedule | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| local | 55.21 | 1.3065 | 2.285696 | 2.232125 | 0.591863 |
| joint | 53.12 | 1.3273 | 2.287775 | 2.234155 | 0.596823 |

| Offset schedule | NLL improvement | Accuracy gain pp | Fit work ratio | Gate |
| --- | --- | --- | --- | --- |
| joint | -0.020839 | -2.0833 | 1.000910 | FAIL |

Fixed256 fit/192 dev/four passes/1,024 presentations/64 Adam updates. Same native initial predictions, data, draws and minimum-devNLL selection. Native15,523 parameters versus15,555 with offsets; eight available receivers and identical selected activity. All offset operators, backward, clipping and active Adam work are charged.

Candidate gate requires>=.02NLL improvement,<=1pp accuracy decline,<=1.50 fitting work ratio and<900,000KiB RSS. Selected schedule: none. Seed7 joint confirmation fails:1.327347 versus native1.306508NLL and2.0833pp accuracy decline. No unchanged full-data stage or extra epochs is admitted. Two seed6 schedules were declared before either result; all failures remain.

Preserved strong984-fit RBF73.44%/.7065 and compact66.67%/.9030 controls have unequal fitting data; solver FLOPs unmeasured. Alternation also changes private-block update counts/clipping, so this is an algorithm comparison, not isolated interference attribution. Physical time still drives content; no official test or supremacy claim.

## Appendix B. When added representation freedom can help

Theory91 derives a local, weighted score/content fitting criterion. With existing Jacobian A, added offset Jacobian B and desired change b, first fit existing coordinates; let r be the remaining residual. Eliminating those coordinates gives offset sensitivity h=B-transpose*r and positive definite Schur matrix S. The exact best local improvement is one half h-transpose*S-inverse*h. It is strict precisely when h is nonzero.

Additional coordinates can address a missing useful direction, or lower a movement penalty along an already available direction. The second case improves conditioning without proving increased representational rank. Shared offsets act at many receptions, whose demands may cancel. A local post-race offset preserves that already selected race while changing future state and routes; the full sequence remains coupled.

For a damped rotating carrier z, the age/phase Jacobian determinant is minus damping rate times squared message norm. The raw bounded offset adds pi*sech(raw-offset)^2. Nonzero damping gives independent local age and phase directions; weak damping, decayed messages or saturated offsets can still make them poorly conditioned. Physical-time derivatives and actual timestamps remain intact.

Four completed independent matrix contracts check 16 direct-versus-eliminated solves, zero-projection/no-benefit, redundant coordinates with lower penalized movement cost, and the rotor determinant. These are mathematical identities, not a demonstrated learning cure. Exact gradients of one loss can have opposed route/content terms and still yield descent when added; opposition alone does not justify deleting credit.

Contracts211000Z; theory91. Joint offset seed6 gains .047767 NLL at 1.000910 fitting-work ratio; unchanged seed7 loses .020839 and 2.0833 accuracy points. No full-data stage is admitted. The local criterion does not guarantee validation generalization, convergence of surrogate race credit or practical superiority.

## Appendix B. Counterfactual replay must match the first-time law

| Time-only loss, rates 1/3 | First score gradient | Second score gradient |
| --- | --- | --- |
| True expected loss gradient | -.062500 | -.187500 |
| Incorrect replay choice addition | +.125000 | -.125000 |
| Factual pathwise + wrong replay | +.062500 | -.312500 |
| Correct joint-score expectation | -.062500 | -.187500 |

The first arrival conditioned on either winner has Exp(sum rates) law. Forcing a losing alternative at its individual Exp(its rate) arrival changes both identity and clock. Detached softmax weighting does not turn those combined interventions into exact conditional choice credit. The table uses loss equal to raw arrival time; the first score direction reverses under the incorrect combined estimator.

Four completed contracts include independent analytic expectations and the actual external local-expectation replay wrapper with one time-recording event at four seeds. Forcing the factual winner reproduces the factual delay, while every losing alternative uses its later individual arrival. Existing softmax-algebra and factual-winner tests cannot establish the expected-risk claim. External model, queue and empirical results are preserved; this is a credit-protocol correction, not a fitted quality result.

Correct conditional choice replays retain the factual first time and actual selected memory write. Clock credit needs its own consistent estimator. Factorized winner/first time sampling or the joint likelihood can supply coherent derivations, with all future discrete credit and deterministic branch derivatives included. Do not double-count one sampled time through both likelihood and reparameterized derivatives.

A critic residual correction is unbiased for a declared replay sum when the critic is fixed before its correction subset is drawn. Training on that same subset first can introduce bias: a two-site counterexample gives -1 for a true zero target. Even a sampling-correct critic preserves the base target; it cannot cure wrong replay time laws. Independent conditioning and critic work require separate contracts.

Theory92; contracts212000Z; qualifications beside theory59 sections402/403. Four contracts pass; audited external source versions are recorded. No integrated repair fit, whole-model exactness, guaranteed variance reduction or superiority claim.

## Appendix B. Exact prefix reuse reduces integrated fitting work

| Replay | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| full | 25.00 | 2.3113 | 0.143986 | 2.999703 | 0.591695 |
| reuse | 25.00 | 2.3113 | 0.128525 | 2.677607 | 0.591695 |

Detached causal-prefix reuse saves10.738% counted fitting work with identical entire learning curves, predictions, model weights, Adam, cursor and RNG recovery. Same24-fit/eight-dev/two-pass/48-target native p16/L2/H2/pool2 protocol. Both rows use the same full-fit and presentation denominators.

Reuse skips repeated alternative-prefix computation while preserving the exact actual-write estimator. This is completed implementation advantage against its full-replay reference, not quality advantage against the strong gesture controls. Snapshot/copy memory traffic and energy remain unmeasured; all retained suffix and optimizer work is charged.

Other-host evidence retained from AWS_PREFIX_REPLAY_FINDINGS_20261002.md and aws_prefix_replay_smokes_20261002T201100Z_analysis.json, with verified parent hashes. Observed13.609 versus12.653 seconds is one concurrent-host observation, not a general latency claim. Exact reuse may support future credit comparisons after matching contracts.

## Appendix B. Other-host completed native capacity comparisons

| Model | Dev accuracy % | Dev NLL | Whole fit GFLOPs est. | Fit MFLOPs / target est. | Infer MFLOPs / prefix est. |
| --- | --- | --- | --- | --- | --- |
| Calibrated counts | 58.85 | 1.3734 | Unmeasured | Unmeasured | Unmeasured |
| rbf_C10_g1 | 73.44 | 0.7065 | Unmeasured | Unmeasured | Unmeasured |
| Ours original | 65.10 | 0.9632 | 20.075 | 2.550 | 0.592 |
| Compact prototype33 | 66.67 | 0.9030 | Unmeasured | Unmeasured | Unmeasured |
| Native p16/L2/pool2 | 66.15 | 1.0420 | 20.075 | 2.550 | 0.592 |
| Native p16/L2/pool8 | 62.50 | 1.0880 | 57.298 | 7.279 | 0.874 |
| Native p16/L4/pool2 | 57.29 | 1.2216 | 37.869 | 4.811 | 1.068 |

| Native capacity | Parameters | Available receivers | Keys / prefix | Commits / prefix | State bytes |
| --- | --- | --- | --- | --- | --- |
| Native p16/L2/pool2 | 15523 | 8 | 168 | 84 | 720 |
| Native p16/L2/pool8 | 42091 | 32 | 672 | 84 | 2376 |
| Native p16/L4/pool2 | 28539 | 16 | 336 | 168 | 1296 |

These completed stages use the same984-fit/192-dev, eight fixed passes/7,872 presentations/496 updates and seed6 as the packet-clock reference. Wider eligible pool and extra depth retain native timing/races/persistent addressed state/key-value separation/counterfactual credit. All scored keys and losing values remain charged.

Pool8 scores62.50%/1.0880NLL versus pool2 reference66.15%/1.0420, at57.2985 versus20.0747 whole fitting GFLOPs. Depth4 scores57.29%/1.2216 at37.8686GFLOPs. Neither completed variant improves the baseline. Extra private capacity also adds untied trainable maps; this is not isolated counterfactual-support or useful-capacity evidence. Other-host tied-map/replication work remains pending and separate.

Completed curie_dvs_clock_p16d2pool8 and p16d4pool2 results172500Z; theory59§§396/398. Conventional solver work is unmeasured, not zero. Raw controls retained in the full comparison above; selected references here use the same column units. No official-test or supremacy claim.

## Appendix B. Strong compact controls rule out an easy storage claim

| Lowest-NLL cell / family size | Dev accuracy % | Dev NLL | Model KiB | In budget | CPU ms / prefix |
| --- | --- | --- | --- | --- | --- |
| class_prototype_m11_g1_C10 | 59.38 | 1.0926 | 39.2 | True | 0.210 |
| class_prototype_m22_g1_C10 | 66.67 | 0.9549 | 67.2 | True | 0.228 |
| class_prototype_m33_g1_C10 | 66.67 | 0.9030 | 95.2 | True | 0.230 |
| class_prototype_m44_g1_C10 | 67.19 | 0.8865 | 123.2 | False | 0.230 |
| nystroem_m8_g0.25_C10 | 55.21 | 1.3055 | 32.3 | True | 0.836 |
| nystroem_m16_g0.25_C10 | 62.50 | 1.1175 | 53.4 | True | 0.859 |
| nystroem_m24_g0.25_C10 | 62.50 | 1.0640 | 75.1 | True | 0.852 |
| nystroem_m32_g1_C10 | 66.15 | 0.9954 | 97.2 | True | 0.854 |

Native uncompressed saved-model budget103.9KiB. The fixed72-cell grid uses random8/16/24/32 Nyström landmarks or1/2/3/4 learned prototypes per class, three gamma scales and three logistic C values. Prototypes, landmarks and normalization use fitting data only; each model includes its transform in the same uncompressed joblib serialization. Every cell remains in the completed JSON.

Selected in-budget control class_prototype_m33_g1_C10: 66.67%/0.9030NLL, 95.2KiB, 0.230ms/prefix. This exceeds the original native 65.10%/.963161 while using less storage and far less CPU time. Thus the original native result does not establish advantage even in this bounded-storage region.

The whole compact grid costs10.043s/183.9MiB peak RSS; common raw preprocessing still costs123.976s. Shared prototype construction, kernel features, all fitting/tuning and scoring are paid in workflow wall. Solver FLOPs are unknown, not zero. The table summarizes each family/size by its minimum development NLL, with all gamma/C settings and probabilities retained.

Exploratory development calibration, not independent model confirmation. Single timing passes are not repeated latency claims. Class prototypes are supervised fitting controls, not target-derived inference inputs. Neither the full kernel nor these compact alternatives is omitted when evaluating the next native initialization.

## Appendix B (continued). Diagnostic: count receivers over the temporal carrier

Labelled diagnostic, not the integrated native architecture. The input-gated temporal carrier supplies the base predictive to the same escape-race count cascade (Theory §§386–388). It tests whether sufficient-statistic receivers remove the memorization tax: if counts hold the exact local statistics, a small learned base should lose far less than the carrier alone does.

| Model | Fit chars / passes | Dev bpc ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ | Infer MFLOPs / char ↓ |
| --- | --- | --- | --- | --- | --- |
| Minimal base w2/d1 + counts K5 + escape gate | 131,072/4 | 2.178 | 3.6 | 0.007 | 0.0028 |
| Minimal base w2/d1 + counts K5 + gate + message | 131,072/4 | 2.124 | 20.5 | 0.039 | 0.0177 |
| Carrier alone w32 | 131,072/4 | 2.848 | 80.9 | 0.154 | 0.0494 |
| Carrier w32 + counts K5 | 131,072/4 | 2.313 | 82.3 | 0.157 | 0.0504 |
| Same, untrained base w32 | 131,072/0 | 2.371 | Not trained | Not trained | 0.0504 |
| Carrier w32 + counts K5 + escape gate | 131,072/4 | 2.136 | 83.8 | 0.160 | 0.0518 |
| Carrier w32 + counts K5 + gate + message | 131,072/4 | 2.127 | 100.8 | 0.192 | 0.0667 |
| Carrier alone w128 | 131,072/4 | 2.587 | 1046.7 | 1.996 | 0.6389 |
| Carrier w128 + counts K5 | 131,072/4 | 2.313 | 1048.1 | 1.999 | 0.6399 |
| Same, untrained base w128 | 131,072/0 | 2.370 | Not trained | Not trained | 0.6399 |
| Carrier w128 + counts K5 + escape gate | 131,072/4 | 2.130 | 1049.6 | 2.002 | 0.6412 |
| Carrier w128 + counts K5 + gate + message | 131,072/4 | 2.129 | 1066.6 | 2.034 | 0.6562 |
| Carrier alone w256 | 131,072/4 | 2.572 | 4025.5 | 7.678 | 2.4571 |
| Carrier w256 + counts K5 | 131,072/4 | 2.316 | 4026.9 | 7.681 | 2.4580 |
| Same, untrained base w256 | 131,072/0 | 2.370 | Not trained | Not trained | 2.4580 |
| KN counts, frozen o5 | 131,072/1 | 2.349 | Not FLOPs | Not FLOPs | Not FLOPs |
| Calibration ceiling: adaptive interpolated KN o8 | 131,072/1 | 2.101 | Not FLOPs | Not FLOPs | Not FLOPs |

Same 8,191 development targets; seed 6, one seed per row; same depth, chunk, learning rate and passes per width. Predeclared: P1 composed w128 < 2.326; P2 composed w32−w256 gap < half the carrier gap; both hold formally for the scalar cascade, but the trained bases alone score 8.17 (w32) / 11.34 (w128) bpc, worse than uniform: the standalone base is trained as a conditional residual, so this alone cannot establish an inert base (Theory §389.1). With the escape gate (+ count message) every width reaches 2.12–2.14, and a minimal 2-wide, one-layer base matches w128 (2.124 vs 2.129) at 1/50 of the work: the gain is learned count smoothing, not the temporal carrier. Count references are near-optimal estimators at this size. The rows above exceed the frozen-KN and Witten–Bell references, and the stronger stream-adaptive interpolated Kneser–Ney is the calibration ceiling here (§393). This is a mechanism diagnostic in a regime where counting is expected to be near-optimal for any learner, not an architecture verdict (§394). Count increments/lookups (5 per target) are integer table work outside FLOPs. Exploratory development evidence; no comparable-quality Transformer claim.

## Appendix B (continued). Ours: native temporal-core language

27-character text8 content enters one conversation address in the native event core. Independent receiver heads, persistent state, learned delays, evolving channels and losing-route credit remain; no historical KV attention or dense carrier is added. The receiver maps now learn across all token marks.

![native language quality work](report/figures/native_language_quality_work.png)

| Ours | Fit chars / passes | Dev bpc ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ | Infer MFLOPs / char ↓ |
| --- | --- | --- | --- | --- | --- |
| d16/p2/s6 | 2,048/4 | 3.765 | 3.778 | 0.461 | 0.0974 |
| d16/p2/s6 | 8,192/4 | 3.557 | 15.116 | 0.461 | 0.0974 |

Fixed passes and frozen 8,191-target development selection; official test untouched. Existing stronger receiver and dense-reference evidence is preserved in the common ledger. Architecture, capacity and weight sharing differ from the KV variants. Work uses audited representative optimizer windows including backward, teachers and Adam; special functions have unit weight here, with separate JSON counts. These are exploratory scaling points, not a frontier score or a measured energy advantage. With H2/depth8/pool2, available receivers are 32, selected commits 16/token, scored keys and training proposals 32/token. Parameter counts and occupied state are retained in completed records.

## Appendix B (continued). Ours: native event/state capability

Each source writes three signed marks then receives an explicit causal query. Fits use 512 queries / 2,048 input events per pass, eight passes; development has 256 queries. All source populations are occupied. The order task predicts the last two signs; the timing task predicts a two-mode temporal trace. All events, timestamps and query flags are observed; target labels never address a module.

![native event quality work](report/figures/native_event_quality_work.png)

| Ours: task / sources / seed | Credit / time | Dev accuracy ↑ | Whole fit GFLOPs ↓ | Fit MFLOPs / query ↓ | Infer MFLOPs / event ↓ |
| --- | --- | --- | --- | --- | --- |
| order/S4/s6 | CF/observed | 100.0% | 7.099 | 1.733 | 0.0923 |

Equal declared query/event budgets; available receivers scale with source population while per-event key scores and selected commits are logged explicitly. All fitting arithmetic is summed from executed operators, including complete producer graphs, losing proposals, clipping and Adam; CPU numerical clocks and unit-weight specials are included. Bootstrap intervals by population and raw state-clearing/time/long-gap probes remain in the completed JSON. More addresses change training exposure. Synthetic capability and bounded activity do not establish superiority over timestamp-aware recurrent/attention controls, real-stream generalization or clockless energy savings.

## Appendix B (continued). Ours: occupied state and activity

Every source stores three content events and receives a query. These are useful-state tests rather than padded unused capacity. Event budgets, active depth, head count and candidate pool are fixed. More addresses reduce observations per local parameter; addressing is input information that strong controls must also receive.

| Ours: task / sources / seed | Available / occupied receivers | Commits / scores per event | Persistent state KiB | Parameters |
| --- | --- | --- | --- | --- |
| order/S4/s6 CF/observed | 128/122 | 16/32 | 9.14 | 159,716 |

| Ours: task / sources / seed | Whole fit GFLOPs | Fit MFLOPs / input event | Infer MFLOPs / query |
| --- | --- | --- | --- |
| order/S4/s6 CF/observed | 7.099 | 0.433 | 0.369 |

| Ours: task / sources / seed | Selected dev accuracy | Clear-history accuracy | Independent confirmation |
| --- | --- | --- | --- |
| order/S4/s6 CF/observed | 100.0% | 27.7% | Not read |

Occupied receivers and stored float bytes come from the audited inference population, not parameter storage, Python metadata, graphs or host RSS. Inference/query includes all four producer/query events. Full fitting includes every event and optimizer operation. Arithmetic plus unit-weight specials; traffic/RNG/energy separate. Development selects checkpoints; independent confirmation is read once only on fixed promoted seed protocols. Population bootstrap intervals in the quality plot condition on the selected model and do not correct development selection or substitute for independent seed uncertainty.

## Appendix B (continued). Ours: compact delayed learning

Available historical content is not automatically learned historical content. After graph detachment, a retrieved old key/value can affect the prediction while its write map receives no later loss credit. Increasing the ordinary credit span from 16 to 64 did not improve the completed matched 2K pilot.

![historical write credit flow](report/figures/historical_write_credit_flow.png)

The tested integrated construction saves the normalized feature φ of each historical write. Later queries send a compact producer teacher to sealed key/value maps while preserving independent heads, temporal races, incoming content and persistent addressed state.

| Ours: added teacher | Factorization | Extra work / query |
| --- | --- | --- |
| Old keys | α q (Σ eᵢ φᵢ)ᵀ | O(C_old d + d²) |
| Old winning value | α a φ_wᵀ | O(d²) |
| Saved eligibility | One detached d-vector / write | 50% extra K/V float storage |

A shared offset of all race scores leaves winner probabilities unchanged but rescales arrival time. Content contrasts can teach which value wins; a separate clock signal can teach when it arrives and how persistent state evolves. The new write teacher retains both signals.

The shared query factors all admitted old key teachers into one matrix update. Live writes keep ordinary gradients without duplicate credit. Six read-only numerical checks and guarded full eight-block update/recovery contracts passed; disabling the new teacher exactly reproduces parent outputs, RNG, gradients and two Adam windows.

This is exact for a hypothetical common perturbation of historical write maps with saved features fixed. Transport to current maps is a delayed local surrogate: old weight versions and omitted representation paths remain limitations. Additional learning arithmetic and eligibility traffic are real costs. Existing counterfactual score credit is retained. Fitting work uses representative audited windows, with exact credit-coverage counters logged separately. Matched quality/work results require completed fits; improved language accuracy and native clockless learning are not established by these contracts.

## Appendix B (continued). Ours: historical write credit

H2, d32/head, 8 event blocks, pool2; 2,048 fitting characters / 4 passes; 8,191 frozen development targets, seed6. Credit16, Adam U64, lr0.002, warmup512. Forward races/history are preserved; sealed writes gain a compact local producer teacher.

![historical write quality work 003fd093a8](report/figures/historical_write_quality_work_003fd093a8.png)

| Ours: credit | Dev bpc ↓ | Whole fit GFLOPs ↓ | Fit MFLOPs / target ↓ | Inference MFLOPs / char ↓ |
| --- | --- | --- | --- | --- |
| parent | 3.733 | 22.753 | 2.779 | 0.5059 |
| α=0.25 | 3.722 | 23.469 | 2.866 | 0.5059 |
| α=1 | 3.724 | 23.464 | 2.866 | 0.5058 |

A historical normalized write feature φ is retained alongside each cached key/value. The old key-map teacher factors as α q (Σ eᵢ φᵢ)ᵀ: one weighted feature sum and one matrix outer product per query. The old winning value map receives α a φᵀ. Live writes keep ordinary credit without duplication; no full-history graph is reopened.

One extra feature vector raises K/V float storage by 50%. The update is exact for a common hypothetical perturbation of historical maps at fixed saved features; transporting it to current maps is a delayed local surrogate with stale weight versions and omitted representation paths. Additional teacher/optimizer arithmetic is counted; feature traffic and physical energy are separate. Parent reuse requires exact forward/RNG/gradient/Adam nesting at α=0. One-seed exploratory development selection, not comparable-quality supremacy.

## Appendix B (continued). Ours: frozen information-flow diagnosis

Saved selected H2/H4 eight-block 8K checkpoints; 256 targets from the development prefix. Weights remain fixed. Each intervention removes one path only during this short evaluation; these are diagnostic probes rather than refitted architecture comparisons.

| Ours | Frozen intervention | Window bpc ↓ | Context RMS mean |
| --- | --- | --- | --- |
| H2 | unaltered | 3.316 | 1.770 |
| H2 | source off | 4.013 | 1.676 |
| H2 | kv off | 3.312 | 1.750 |
| H2 | channel identity | 3.773 | 1.488 |
| H4 | unaltered | 3.331 | 2.144 |
| H4 | source off | 4.001 | 2.066 |
| H4 | kv off | 3.375 | 2.073 |
| H4 | channel identity | 3.786 | 1.477 |

Removing source-message carry and learned channel mixing worsens both saved checkpoints in this window. These content/state paths are useful here; short-window removal probes do not warrant discarding them or demonstrate a refitted architecture advantage.

A route-credit audit replays each admitted historical value at three depths/head0 for the two-head model. It holds one realized race time and continuation seed fixed, then compares the centered local content teacher with the conditional categorical loss gradient.

| Ours: local audit | Probes | Mean cosine | Opposed directions | Mean oracle gap (nats) |
| --- | --- | --- | --- | --- |
| H2 / head0 | 24 | 0.717 | 1 / 24 | 0.071 |

Higher cosine means better alignment in this narrow replay. The oracle gap is realized loss minus the best candidate replay, not achieved improvement. Different routing can change subsequent candidate/RNG paths. The audit conditions away time derivatives and does not estimate the full expected gradient, long-history utility or discovery coverage. Interventions have no refitting and no confidence intervals; do not use these window scores as promotion metrics or dense-model superiority evidence.

## Appendix B (continued). Ours: separate online neural learning

Both arms start from the same selected integrated-model checkpoint and maintain persistent event memory on the same new development stream. The frozen arm retains its parameters. The online arm updates the complete neural backbone after making the causal predictions in each 16-character block, with no replay.

![integrated online language](report/figures/integrated_online_language.png)

| Ours: mode | Stream bpc ↓ | Whole stream MFLOPs ↓ | Parameter change L2 | Updates |
| --- | --- | --- | --- | --- |
| Ours: frozen | 3.191 | 210.903 | 0.000 | 0 |
| Ours: online | 3.096 | 2,264.163 | 1.551 | 512 |

The 8,191 targets come from [90,065,536, 90,073,728). Learning rate 0.0001 was fixed before this stream; Adam starts with fresh moments. Both arms use paired block race noise. Parameter updates precede only future blocks; the current target cannot change its own prediction. Their contexts can diverge after learning.

This measures block-delayed online adaptation, not instant per-character updates or a frozen official-test score. One inherited model/window/rate. All neural parameters may learn, including content, keys, clocks and retention; this differs from the earlier statistical expert-mixing-only ablation. Inherited fitting is additional and identical in both arms. Work estimates observe the actual first/middle/last blocks and include counterfactual learning, backward, clipping and Adam; RNG, indexing and physical traffic remain additional. Lower online loss, if observed, is evidence only for this protocol.

## Appendix B (continued). Separate statistical language baseline

This count/copy predictor does not use the learned Sleeping Machines event backbone. It is a separate statistical system: order-0 through order-6 counts, backoff probabilities and a bounded causal copy cache, combined by learned mixing weights. Its quality/work results must not be attributed to the event architecture.

The statistical predictor and saved neural references score the same 999,999 character targets, starting from a cold context. Parameters are frozen during testing. Earlier observed test characters can supply causal context, including the mixture's bounded 256-character copy cache. Lower bits per character means better prediction.

| Predictor | Test bpc ↓ | Fitting and selection budget | Estimated training FLOPs ↓ |
| --- | --- | --- | --- |
| Ours: separate statistical count/copy baseline | 1.727 | 10M count fitting + three 1M mixing-rate trials | 4.47G + integer count construction |
| Ours: count/copy plus causal word context | 1.719 | 10M count fitting + three 1M mixing-rate trials | 4.80G + integer count construction |

The neural totals charge all original optimizer steps that produced the inherited E174 checkpoints: forward, estimated 2×-forward backward, clipping and Adam. Alignment evaluation is excluded. The statistical floating estimate charges expert probability preparation and all three mixing-rate trials, including local gradients and weight updates. Exp/log/root evaluations count as one operation in these language estimates. Count construction additionally uses about 80M integer count presentations, plus sorting, lookup and hashing; that work is not quantified as FLOPs. The floating totals alone cannot establish total compute, runtime or energy savings.

The count/copy mixture improves on LSTM by 0.072 bpc and Transformer by 0.181 bpc. These results establish useful specialized prediction; generic learned representations are assessed in the separate language screen.

The statistical baseline's count arrays occupy 66.55 MB. Vocabulary, capacities, optimization and fitting budgets differ from the neural references. The comparison does not measure total training energy or a matched-capacity advantage.

All three use the same historical 27-character alphabet. Modern shared subword tokenization is a separate comparison gate for the learned event architecture, described later in this appendix.

One exploratory seed. Text8 offsets: count fitting [0,10M), mixing-weight validation [90M,91M), test [95M,96M); test index zero is excluded for all three predictors. The mixture selects its update rate on validation. Saved neural weights are unchanged. Results: E173/E174; stream contract: E175. At 90M training characters, reference test scores are 1.661 for the LSTM, 1.604 for the four-layer Transformer. Estimated training work (forward, backward, Adam and gradient clipping): LSTM: 3.89 PFLOP, TF: 8.00 PFLOP. Shape-based estimates count multiply-add as two operations; backward is approximated as twice forward. Validation/test inference is excluded. These are single-seed comparisons; capacities and fitting budgets are not matched. Calibration (Theory §381): untuned modified Kneser–Ney counts of the same 90M characters score 1.652 on the same test targets, so these controls sit near count level and are not frontier bars. Target-leaked historical mixtures have been quarantined; a corrected 90M mixture comparison remains open.

## Appendix B (continued). Learned language and depth

A bounded screen trains the common event backbone to predict the next character. Both configurations use width 32, the same 8,192 training characters, four passes, 32-character contexts and 1,024 validation predictions. All eight layers' value, route and memory-time parameters update. Lower bits per character means better prediction.

![e133 generic language](report/figures/e133_generic_language.png)

| Ours: depth | Learned parameters | Validation bpc: lower is better | Total CPU wall time |
| --- | --- | --- | --- |
| 1 | 7,671 | 3.464 | 27.4 s |
| 8 | 53,430 | 3.395 | 170.5 s |

Eight layers improve validation loss from 4.752 to 3.395 bpc, versus 3.464 with one layer. Shuffling preceding characters while preserving the last character, count and timestamps increases the deeper model's loss to 3.805; replacing preceding context raises it to 3.777. These frozen input probes show context sensitivity, not a retrained baseline comparison.

The deeper model has more parameters and takes more CPU time. The quality/time panel includes fitting and evaluation, with backpropagation and optimizer updates executed during fitting. These bounded-query models replay preceding context. The following persistent implementation consumes each character once. Physical memory traffic and joules remain unmeasured.

One seed; different parameter counts. This establishes a generic learned-language foothold and a small depth gain, not competitive large-scale representation, a matched tuned dense-model advantage or a scaling law. The statistical 10M-character mixture remains a separate result. Official test data are untouched. E133 preserves commands, source/data hashes, layer diagnostics and work coverage.

## Appendix B (continued). Persistent learned language

The event-state language model consumes each character once and retains local modal memories and its delayed-message queue. Chunk boundaries truncate learning credit without discarding the observed history. Only actual event arrivals evaluate layers; text time is measured in token intervals.

![e176 stream language learning](report/figures/e176_stream_language_learning.png)

| Ours: representation | Parameters | Fitting bpc ↓ | Validation bpc ↓ | Deliveries/pass |
| --- | --- | --- | --- | --- |
| Characters | 28,403 | 3.142 | 3.351 | 65,784 |
| Causal prefix tokens | 35,163 | 3.085 | 3.517 | 45,816 |

Validation loss falls from 5.329 to 3.351 bpc. All eight layer teachers are nonzero in every fitting pass. Each pass consumes 8,223 characters including warmup and makes 65,784 block deliveries. The result establishes learning with persistent causal state and no prefix replay.

A train-only 131-token prefix dictionary reduces layer deliveries by 30.4% and observed CPU time by 26.2%, while validation bpc is 3.517. Exact partial-token marginalization scores identical raw targets. Its larger vocabulary fits better but generalizes less well in this small screen: compression alone does not explain or resolve the quality gap.

One seed; 28,403 parameters, width 32, sixteen temporal modes per block. Four passes, 128 Adam steps/pass, credit truncated every 64 characters, 31 warm characters and 1,024 validation targets. Target offsets match the bounded E133 screen; topology, capacity, history and update counts differ, so this is not a matched intervention. Total CPU wall time 364.9 s including fitting/evaluation; peak RSS 364.8 MiB. These are event counts and observed resources, not total arithmetic, physical memory traffic or energy. No official test or large-corpus claim. E176.

## Appendix B (continued). Learned event language: training and inference work

These counts belong to the learned eight-layer persistent Sleeping Machines event model that reaches 3.351 validation bpc. It has no count/copy/word experts. Primary counts describe the logical event algorithm; simulator dispatch/allocation is excluded.

| Ours: fitting stage | Estimated arithmetic FLOPs |
| --- | --- |
| Forward and loss | 1.96G |
| Backward | 4.77G |
| Gradient clipping | 43.52M |
| Optimizer | 173.95M |
| Warmup | 7.42M |
| Total | 6.96G |

The entire completed budget includes 32,768 fitting targets, 512 Adam/clipping steps and four stream warmups. Special functions add 85.431M evaluations, reported separately from arithmetic FLOPs.

| Ours: inference boundary | Per character |
| --- | --- |
| Prediction plus NLL-scoring arithmetic | 59,741 FLOPs |
| Additional special functions | 1,212 evaluations |

A 64-character saved-checkpoint trace measures forward/loss, backward, clipping and warm Adam; the fitting ledger scales those stages by the original 512 steps and separately charges warmup. This is a representative arithmetic estimate, not a whole-run trace. The traced chunk has complete floating-operator formula coverage. Index/queue work, comparisons, memory traffic and evaluation passes are outside the fitting arithmetic boundary.

The 10M-character LSTM/Transformer checkpoints have different data, capacity and achieved quality. Dividing their full fitting budgets by this small run would not establish a fair training advantage. Full learned-event runs must complete before a larger aligned quality/work comparison.

Evidence: event_language_work/local_event_language_work_20260930T124646Z.json; quality: E176. Two arithmetic FLOPs per multiply-add. Additional unit-weight special functions would give a different logical-operation total; no event-device runtime or energy has been measured.

## Appendix B (continued). Statistical online-learning pilot

The official language comparison freezes parameters during evaluation. A new development-only pilot asks whether causal local learning improves prediction: score each character first, reveal it, then update only the expert mixing weights. Count experts stay frozen and both arms use identical causal copy-cache behavior.

| Ours: statistical expert set | Frozen bpc ↓ | Online bpc ↓ | Extra update FLOPs |
| --- | --- | --- | --- |
| Without word | 2.605 | 2.564 | 2.75M |
| With causal word | 2.537 | 2.487 | 3.21M |

This pilot reuses a 100,000-character count checkpoint and its learning rate selected on an earlier 2,048-character validation window. It scores 8,191 targets in a fresh 8,192-character development window [90,032,768,90,040,960), excluding the first target. Each adaptive arm performs 8,192 updates, including first-position warmup. Official test data are untouched.

One checkpoint and one window; a specialized adaptive readout, not deep learned TTT or a frontier result. The extra arithmetic column counts only local gradient/weight updates, on top of shared prediction and inherited fitting costs. Total pilot CPU wall time is 0.587 s. Evidence: online_language/local_online_language_20260930T121741Z.json.

### Compute allocation is the next architectural hypothesis

Independent budgets for active width/depth, dormant capacity, temporal memory, retrieval, credit and adaptation may let extra work buy more prediction quality. The new theory derives conditional marginal-value allocation and retrieval-error bounds, and identifies exposure, routing overhead and hardware utilization as possible limits. A small-scale win does not prove a better scaling exponent or a widening frontier advantage.

Next learned-language comparisons need shared modern subword tokenization, Unicode/byte coverage, suitable rotary/relative position, packed optimized Transformer kernels, sparse MoE and modern recurrent/attention-hybrid controls. Existing event memory already uses relative-time rotations. Token coordinates must remain distinct from learned scheduling delays.

Known sequences can be fitted with input-known affine scans and then generated sequentially with persistent state. Sequence-parallel fitting also permits Transformer adaptation between tokens/chunks. Compare frozen and online arms at identical observations and adaptation budgets; charge cache consistency and update work.

[Theory §§280–286](experiments/theory/43_compute_allocation_and_frontier_scaling.md) and the [frontier compute protocol](experiments/FRONTIER_COMPUTE_PROTOCOL.md) specify these tests. Modern architecture and larger-scale comparison arms remain proposed, not completed.

## Appendix C. Preserved historical results and revisions

Earlier result files remain part of the research record. The following numbers explain older report headlines and why their interpretation changed. They are preserved here with the identified protocol errors; they are not current valid benchmark comparisons.

### Earlier statistical language results

The target-dependent partial-word mixtures have been removed from the active results tree and all numerical comparisons. Raw records are quarantined for audit only. The corrected E17310M results are1.727 without word context and1.719 with causal word context; a corrected90M mixture comparison remains open.

### Earlier event world-model results

| Historical predictor | Day 6 log-likelihood ↑ | Day 7 log-likelihood ↑ |
| --- | --- | --- |
| Ours: event hazard + rate/flow state | -2.184 | -1.999 |
| Ours: event hazard + per-type state | -2.159 | -1.973 |
| Ours: event hazard + per-type state; finer gap bank | -1.912 | -1.734 |
| Saved Transformer Hawkes reference | -1.971 | -1.816 |

The event hazard models use sparse conditional memories and local rate/flow state. Their frozen test parameters and causal state updates are useful mechanisms. However, the event-size threshold was fitted across all seven pilot days, including the evaluation days, for both the event and neural references. Day resets and warmup exclusions also differ. The recorded gap needs fitting-only preprocessing and aligned rescoring before it supports a held-day advantage.

[Source and numerical review](experiments/EXPERIMENTAL_REVIEW.md); [E57 world-model records](experiments/results/e57/). Preserved timing, composition, retrieval and modular results remain in the main report. New learned-model benchmarks add evidence; they do not erase these earlier runs.

## Appendix D. Evidence and metric definitions

| Metric | Interpretation |
| --- | --- |
| Bits per character | Held-out negative log probability in base two; lower is better next-character prediction. |
| Accuracy | Fraction of correct class decisions on the declared development or test protocol. |
| Event likelihood | Scores both the next event type and waiting time, including the observed silence. |
| Arithmetic FLOPs | Multiply-add counts as two operations. Tables state whether special functions are separate or assigned unit cost; these conventions must be aligned before forming ratios. |
| Logical operations | The structured-task ledger assigns 2 units per MAC and 1 per other scalar arithmetic/nonlinear operation or estimated sort comparison. |
| Resource boundary | Event-target arithmetic includes required active algorithm work. Memory traffic, queues, indexing and simulator overhead are reported separately where measured. Activity counts alone do not determine total FLOPs. |
| Energy | Measured total joules over an explicit boundary. Operation estimates and CPU timings support work comparisons, but are not joule measurements. |

The evidence is preserved in versioned result summaries with configurations, split identities, learning curves and source hashes. E173/E174 support the language comparison; E61 supports retrieval; E34/E53/E54 support native composition; E41 supports the original periodic computation. E121/E124 establish consolidated arithmetic and its certificate; E123 supplies the new dense controls and E124 the operation ledger. E118/E119/E122/E125/E126 support deep speech, readout and causal-context comparisons; E127–E131 audit credit geometry, hard race boundaries and separate key/value learning; E132 checks a joint race-credit formalism, E133 supplies the language/depth screen, and E134–E135 test whole-value credit and content-selective temporal memory. E136 audits reversible augmented transport and its supervised memory boundary, including twelve-layer query/learning interventions. E137 tests compact memory queries and class-visible credit geometry. E138–E141 examine richer source messages and trainable signed temporal memory, with exact local teacher and initial-nesting contracts. E142 establishes signed-state and first-coalescing identities; E143 tests a larger nonlinear temporal residual learner, and E144 audits simultaneous state/query pooling. E171 reproduces the consolidated screens and selected speech answers, and checks causal input boundaries. E172 records complete training-step arithmetic; E175 checks the generic persistent language stream.

The project theory index contains formal assumptions and proofs. Research findings retain detailed analyses and the full experimental record. The model documentation describes reproducible configurations and operational procedures. This report presents the project, its evidence and its potential.

## Appendix B. AWS coarse temporal screens: two seeds pass

| Native packet/clock | Accuracy % | Dev NLL | Whole fit GF est. | Fit MF / target | Infer MF / prefix |
| --- | --- | --- | --- | --- | --- |
| s6 20 bins/.05 | 48.96 | 1.4242 | 2.285696 | 2.232125 | 0.591709 |
| s6 4 bins/.05 | 57.81 | 1.2221 | 0.548517 | 0.535661 | 0.138119 |
| s6 4 bins/.25 | 55.73 | 1.2749 | 0.548517 | 0.535661 | 0.138133 |
| s7 20 bins/.05 | 49.48 | 1.4347 | 2.285696 | 2.232125 | 0.591695 |
| s7 4 bins/.05 | 53.12 | 1.3233 | 0.548517 | 0.535661 | 0.137993 |
| s7 4 bins/.25 | 55.21 | 1.2208 | 0.548517 | 0.535661 | 0.137965 |

Same integrated p16/L2/H2/pool2 core, 256 fitting gestures, 192 subject-disjoint development gestures, four passes, 1,024 fitting presentations and U16. Coalescing five 50ms count packets into each 250ms packet preserves total causal counts and the 1s query but removes timing within each quarter second. Learned temporal computation, races, separate keys/values, sparse addressed writes and counterfactual credit remain. The two coarse arms differ only in clock initialization.

Both coarse arms pass the prespecified screen in seeds6 and7: at least .02 NLL improvement, at most 1 percentage point accuracy decline and at most .50 fitting work ratio. Work ratio .239978 means 76.002% less counted fitting work. Events 21,504 to 5,120; key scores 172,032 to 40,960; writes 86,016 to 20,480. Eight available receivers and four selected writes per event in every arm.

Seed6 one-thread AWS job walls are 42.846/14.031/14.046s, under the authorized three-slot scheduler. These are observations, not universal hardware speedups. The subsequent full-data matrix is complete on the next page; the earlier pending replication and full-data statuses are superseded by completed evidence.

Exploratory reused development set; no independent confirmation or control advantage. All passes, candidate values, backward/Adam/clip are charged in consistent arithmetic plus unit-special estimates. Whole-job wall includes preprocessing/evaluation; NumPy preprocessing FLOPs, traffic and energy remain unknown. Both completed analysis files and the verbatim pre-integration AWS appendix are retained in version control.

## Appendix B. Full coarse matrix: positive means, failed all-seed gates

| Native packet/clock | Accuracy % | Dev NLL | Whole fit GF est. | Fit MF / target | Infer MF / prefix |
| --- | --- | --- | --- | --- | --- |
| s6 20 bins/.05 | 57.81 | 1.1798 | 17.573520 | 2.232409 | 0.591779 |
| s6 4 bins/.05 | 61.46 | 1.0718 | 4.218015 | 0.535825 | 0.138035 |
| s6 4 bins/.25 | 66.67 | 0.8910 | 4.218015 | 0.535825 | 0.138119 |
| s7 20 bins/.05 | 55.73 | 1.1726 | 17.573520 | 2.232409 | 0.591737 |
| s7 4 bins/.05 | 70.31 | 0.9333 | 4.218015 | 0.535825 | 0.137923 |
| s7 4 bins/.25 | 61.98 | 1.0569 | 4.218015 | 0.535825 | 0.138007 |
| s8 20 bins/.05 | 68.75 | 0.8986 | 17.573520 | 2.232409 | 0.591751 |
| s8 4 bins/.05 | 60.42 | 1.0286 | 4.218015 | 0.535825 | 0.138007 |
| s8 4 bins/.25 | 69.27 | 0.9207 | 4.218015 | 0.535825 | 0.138035 |

Nine completed fits: same984 fitting/192 development gestures, eight passes, 7,872 fitting presentations per run. Mean fine accuracy/NLL 60.764%/1.083660; 4-bin/.05 64.062%/1.011241; 4-bin/.25 65.972%/.956220. Coarse fitting work is 4.218015 versus 17.573520 GFLOPs per fit: ratio .240021. Every per-target column uses the same denominator across models; no whole-fit/per-target unit mixing.

Both frozen all-seed gates FAIL. Seed8 fine 68.75%/.898591 has lower NLL than 4-bin/.05 60.417%/1.028562 and 4-bin/.25 69.271%/.920744. Diagnostic crossed seed/development-user intervals include zero. Preserve positive mean quality and work savings alongside this failure; no unchanged extension follows.

Original stronger local references remain valid: native65.10%/.963161 and clock66.15%/1.041987. They have different fitting implementations and work totals and are not replaced by the weaker matched fine arm. Same8 receivers, four writes per event, 21 versus5 events per query. Saved full-coarse analysis retains every curve, activity ledger, result digest, wall/RSS and uncertainty scope. No official test.

## Appendix B. Strong coarse controls and frozen readout evidence

| Native packet/clock | Accuracy % | Dev NLL | Whole fit GF est. | Fit MF / target | Infer MF / prefix |
| --- | --- | --- | --- | --- | --- |
| Raw RBF 1 bins | 68.75 | 0.8068 | Unknown | Unknown | Unknown |
| Raw RBF 4 bins | 77.60 | 0.6867 | Unknown | Unknown | Unknown |
| Raw RBF 20 bins | 74.48 | 0.7080 | Unknown | Unknown | Unknown |

| Frozen encoder/access | Mean accuracy % | Mean dev NLL | Nomination gate |
| --- | --- | --- | --- |
| Initial/query32 | 61.632 | 1.036019 | Diagnostic |
| Initial/all resident state | 68.750 | .886478 | Diagnostic |
| Trained/query32 | 69.097 | .856456 | Pass |
| Trained/all resident state | 72.743 | .829107 | FAIL |

Raw controls use984 fitting/192 development examples and fitting-user GroupKFold selection. The4-bin RBF result77.604%/.686661 is a strong completed reference under that protocol; the earlier local RBF73.44%/.706478 uses different selection and remains visible. Solver FLOPs and total fit/inference work are unknown, not zero.

All three matched-clock native encoders and their initial reservoirs are frozen. Query RBF probes improve native mean65.972%/.956220 by3.125 points/.099764 NLL; individual NLL gains .056455/.163228/.079610 pass the context nomination gate. All resident reads add128 memory values, eight ages and eight occupancy flags. Their extra mean .027349 NLL gain misses the .05 resident gate.

These are information/readout diagnostics. Dense resident access does not demonstrate sparse dormant-value retrieval. Conditional fitting-only decoder selection shares label-trained encoders; it is not producer-cross-fitted validation. The next integrated quadratic head fails, as recorded on the next page.

Original native fits4.218015GF each, six1,176-prefix replays, 72 fold fits and12 refits retained. Probe53.605s/516,444KiB; state/winner/probability contracts pass. Replay solver/materialization/traffic/energy totals remain unknown. Full per-seed readout outcomes and costs are preserved in AWS_COARSE_READOUT_FINDINGS_20261002.md.

## Appendix B. Integrated quadratic head: contracts pass, quality fails

| Native packet/clock | Accuracy % | Dev NLL | Whole fit GF est. | Fit MF / target | Infer MF / prefix |
| --- | --- | --- | --- | --- | --- |
| affine | 55.73 | 1.2749 | 0.548517 | 0.535661 | 0.138133 |
| quadratic | 55.21 | 1.4415 | 0.594382 | 0.580452 | 0.201506 |

Same4-bin/.25-clock native p16/L2/H2/pool2, 256 fitting/192 development gestures, seed6, four passes and1,024 fitting presentations. A standard local degree2 query residual starts at zero: initial logits/RNG and every old gradient nest the affine model exactly. Six numerical/deep-gradient/recovery/accounting contracts and24fit/8dev/two-pass learning smoke pass before this screen.

The quality gate FAILS: NLL worsens .166618 and accuracy declines .521 percentage point. Fitting ratio1.083617 and inference ratio1.458782 satisfy resource admission but do not rescue prediction quality. There is no unchanged seed7/full-data/epoch extension and no replacement of the leading result.

The5808 added head weights bring parameters from15,523 to21,331. Eight available receivers, candidate keys/values and four selected writes per event are unchanged. Every one of the five sequential readout calls is charged. Physical time, hard races, persistent addressed state, separate keys/values and counterfactual learning remain; positive frozen RBF information alone does not prove this coupled encoder/head optimizer can exploit it.

Full-coarse seed8 failure and initial-reservoir/readout controls remain beside this negative result. The AWS team owns the separately frozen affine versus polynomial convex-fit diagnostic. No pending cell is reported as evidence; solver work, preprocessing FLOPs, traffic and energy remain separately scoped.

## Appendix B. Producer-held selection: both rules agree

| Seed | Selection rule | C at984 | Accuracy % | Dev NLL |
| --- | --- | --- | --- | --- |
| 6 | conditional cv | 0.1 | 54.17 | 1.222593 |
| 6 | producer held | 0.1 | 54.17 | 1.222593 |
| 7 | conditional cv | 0.1 | 53.65 | 1.177975 |
| 7 | producer held | 0.1 | 53.65 | 1.177975 |

| Seed | Original core fit GF | Feature replay GF | Port-check replay GF | Solver/grid GF |
| --- | --- | --- | --- | --- |
| 6 | 2.285696 | 0.888889 | 0.299717 | Unknown |
| 7 | 2.285696 | 0.888889 | 0.299722 | Unknown |

Use the two saved256-fit native pilot encoders after four fixed passes, without development checkpoint selection. Conditional three-fold decoder CV uses producer-seen fitting labels. Producer-held selection fits on those256 examples and scores728 fitting examples whose labels the producer never saw. Both rules choose the strongest nominal C.1 in both seeds; no selection disagreement or causal selection failure is demonstrated here.

Three mean-L2 coefficients are invariant across fold sizes: lambda=1/(984C), solver C=1/(n lambda). The selected affine heads are then refitted on all984 fitting examples. Final54.17%/1.222593 and53.65%/1.177975 remain weak. Because the new heads see984 labels versus the original pilots256, this is neither an equal-data benchmark nor evidence of practical advantage.

Four contracts include a constructed feature-selection confidence counterexample, sample-size invariant regularization, reproduction of fixed-pass native probabilities and folding scaled affine coefficients into the original native head. Every non-head parameter is bitwise preserved; batched and serial native probabilities match. Parameter count and the inference architecture remain unchanged.

Theory96. The mathematical counterexample establishes a possible failure, not its occurrence in these pilots. Original encoder fits, feature replay and head-port verification replay are separate costs; solver/transformation/traffic/ energy work remains unknown. Same reused192 development examples; no official test or new main model. No guessed global regularization fit is admitted from this result.

## Appendix B. Receiving losing arrivals requires boundary credit

| Constructed gradient | Complete conditional reference | Ordinary history-cell autograd |
| --- | --- | --- |
| Width dL/dH | -27.662490 | 0.656618 |
| Zero-width right derivative | -34.323670 | Birth credit required |

For a common-start exponential race, winner W and first time T factor as W~Categorical(lambda/sum lambda), T=E0/sum lambda. Conditional losing residual arrivals are independent Exp(lambda_j). Winner-only replay omits those residuals when a layer receives several messages. Under the native increasing bounded delay map, a deadline H after the first arrival gives an explicit raw residual cutoff R(T,H) and losing membership q_j=1-exp(-lambda_j R).

The complete expected gradient decomposes into ordinary fixed-history derivatives, winner choice and paired boundary flux. At a losing arrival crossing the deadline, dq_j multiplies the downstream loss difference between heard and unheard outcomes, including the actual separate receiver write. This produces faster/slower emitter and reception-width credit while preserving coupled contents and physical times.

Seven numerical contracts verify joint arrival density, physical cutoff/cap, membership normalization/activity, every score/content/decay/width derivative, independent finite differences, zero-width birth and actual-write effects. Three candidates, decaying messages and a later addressed-memory read; quadrature agrees with the boundary/winner/history formula. Maximum width finite-difference error 4.41e-9. The constructed sign reversal shows ordinary autograd can widen or narrow in the wrong direction; it does not identify a benchmark failure.

Enumerating membership histories grows exponentially. Paired boundary sampling can avoid enumeration, but every candidate discovery, actual timed counterfactual and future replay is paid and variance remains. Conditional inverse-CDF derivatives already redistribute heard arrival times; adding the full unconditional boundary term to them double counts credit. Neither primitive supplies a free whole-model gradient.

Theory97, guarded .743s/295,904KiB reference. No optimizer or fitted native window. Fixed-first deadline differs from silence-reset popcorn, whose scheduler and merge/split credit remain separate. FLOPs/traffic/energy unmeasured, not zero. The next pages test real native information/access before any training admission.

## Appendix B. Native reception intervention, seed6: no smoke admission

| H ms | One-site mode | Initial NLL | Trained NLL | Initial acc % | Trained acc % | Extra arrivals |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Winner only | 2.820885 | 1.071110 | 6.25 | 50.00 | 0/16 |
| 1 | Heard sum + writes | 2.817558 | 1.070983 | 6.25 | 50.00 | 0/16 |
| 1 | Heard mean + writes | 2.824706 | 1.070983 | 6.25 | 50.00 | 0/16 |
| 1 | Winner x heard count | 2.811214 | 1.070983 | 6.25 | 50.00 | 0/16 |
| 1 | Winner + same wait | 2.820885 | 1.070983 | 6.25 | 50.00 | 0/16 |
| 1 | Heard sum, winner write | 2.817760 | 1.070983 | 6.25 | 50.00 | 0/16 |
| 3 | Heard sum + writes | 2.799750 | 1.104131 | 6.25 | 50.00 | 4/16 |
| 3 | Heard mean + writes | 2.819896 | 1.069441 | 6.25 | 50.00 | 4/16 |
| 3 | Winner x heard count | 2.800960 | 1.105012 | 6.25 | 50.00 | 4/16 |
| 3 | Winner + same wait | 2.820911 | 1.070549 | 6.25 | 50.00 | 4/16 |
| 3 | Heard sum, winner write | 2.800910 | 1.104000 | 6.25 | 50.00 | 4/16 |

| Mean/winner H ms | First-prefix core MF est. | Minimum ready time s | Maximum ready time s |
| --- | --- | --- | --- |
| 0 | 0.591695 | 1.004001 | 1.012672 |
| 1 | 0.591874 | 1.005001 | 1.012672 |
| 3 | 0.591874 | 1.007001 | 1.014160 |

Frozen fixed-four-pass producer and its initial reservoir; 16 prespecified evenly spaced FIT examples unused by the256-label producer. No optimizer, decoder refit, development or official-test evaluation. One actual site: observed packet event19/layer0/head0; the following deeper computation/query sees real changed receiver memories. Transport runs to the actual deadline.

Prespecified1ms heard-mean versus same-wait winner gate FAILS in both trained seeds: zero extra arrivals and zero NLL difference. At3ms seed6 hears extras4/16, seed7 7/16; mean-over-wait NLL gains .001108/.002312. Seed6 sum worsens loss. Those diagnostic settings cannot replace the frozen gate. Failure concerns this site/width/sample, not all learned windows.

Theory98. Eight available receivers, four original writes/event and 168 scored keys/prefix; only this site can add one actual write. Original trained fit2.285696GF/2.232125MF per fitting presentation retained. Table inference MF is the first selected prefix only, not full-audit FLOPs. All44 model/configurations104.222s/396,720KiB; twelve zero-nesting/gradient/ causality/write/target-invariance/refused-training contracts pass. Frozen weights remain bitwise equal. Unknown audit/scheduler/traffic/energy work is not zero. Processing readiness is charged beyond the1s observation cutoff.

## Appendix B. Native reception intervention, seed7: no smoke admission

| H ms | One-site mode | Initial NLL | Trained NLL | Initial acc % | Trained acc % | Extra arrivals |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Winner only | 2.461633 | 1.025841 | 12.50 | 56.25 | 0/16 |
| 1 | Heard sum + writes | 2.459791 | 1.025891 | 12.50 | 56.25 | 0/16 |
| 1 | Heard mean + writes | 2.460627 | 1.025891 | 12.50 | 56.25 | 0/16 |
| 1 | Winner x heard count | 2.461396 | 1.025891 | 12.50 | 56.25 | 0/16 |
| 1 | Winner + same wait | 2.461622 | 1.025891 | 12.50 | 56.25 | 0/16 |
| 1 | Heard sum, winner write | 2.459954 | 1.025891 | 12.50 | 56.25 | 0/16 |
| 3 | Heard sum + writes | 2.458680 | 1.022673 | 12.50 | 56.25 | 7/16 |
| 3 | Heard mean + writes | 2.455532 | 1.023701 | 12.50 | 56.25 | 7/16 |
| 3 | Winner x heard count | 2.454758 | 1.024533 | 12.50 | 56.25 | 7/16 |
| 3 | Winner + same wait | 2.452995 | 1.026013 | 12.50 | 56.25 | 7/16 |
| 3 | Heard sum, winner write | 2.458298 | 1.022674 | 12.50 | 56.25 | 7/16 |

| Mean/winner H ms | First-prefix core MF est. | Minimum ready time s | Maximum ready time s |
| --- | --- | --- | --- |
| 0 | 0.591695 | 1.004017 | 1.015801 |
| 1 | 0.591874 | 1.005011 | 1.016597 |
| 3 | 0.591874 | 1.007011 | 1.019004 |

Frozen fixed-four-pass producer and its initial reservoir; 16 prespecified evenly spaced FIT examples unused by the256-label producer. No optimizer, decoder refit, development or official-test evaluation. One actual site: observed packet event19/layer0/head0; the following deeper computation/query sees real changed receiver memories. Transport runs to the actual deadline.

Prespecified1ms heard-mean versus same-wait winner gate FAILS in both trained seeds: zero extra arrivals and zero NLL difference. At3ms seed6 hears extras4/16, seed7 7/16; mean-over-wait NLL gains .001108/.002312. Seed6 sum worsens loss. Those diagnostic settings cannot replace the frozen gate. Failure concerns this site/width/sample, not all learned windows.

Theory98. Eight available receivers, four original writes/event and 168 scored keys/prefix; only this site can add one actual write. Original trained fit2.285696GF/2.232125MF per fitting presentation retained. Table inference MF is the first selected prefix only, not full-audit FLOPs. All44 model/configurations104.222s/396,720KiB; twelve zero-nesting/gradient/ causality/write/target-invariance/refused-training contracts pass. Frozen weights remain bitwise equal. Unknown audit/scheduler/traffic/energy work is not zero. Processing readiness is charged beyond the1s observation cutoff.

## Appendix B. AWS frozen polynomial decoder: both nomination gates fail

| Encoder | Seed | Degree | Accuracy % | Dev NLL | Native infer MF/query |
| --- | --- | --- | --- | --- | --- |
| Initial | 6 | 1 | 57.81 | 1.162621 | Unmeasured |
| Initial | 6 | 2 | 64.06 | 1.065509 | Unmeasured |
| Fitted | 6 | 1 | 68.75 | 0.835175 | 0.138119 |
| Fitted | 6 | 2 | 68.75 | 0.870350 | 0.201534 |
| Initial | 7 | 1 | 56.77 | 1.154207 | Unmeasured |
| Initial | 7 | 2 | 59.90 | 0.994128 | Unmeasured |
| Fitted | 7 | 1 | 66.67 | 0.939479 | 0.138007 |
| Fitted | 7 | 2 | 69.79 | 0.899185 | 0.201422 |
| Initial | 8 | 1 | 60.94 | 1.097227 | Unmeasured |
| Initial | 8 | 2 | 64.58 | 0.976572 | Unmeasured |
| Fitted | 8 | 1 | 68.75 | 0.936226 | 0.138035 |
| Fitted | 8 | 2 | 71.35 | 0.885890 | 0.201450 |

All three coarse trained encoders and exact initial reservoirs,984 fitting/ 192 development examples. Local affine/quadratic heads selected by fitting-user GroupKFold over C(.01,.1,1);108 fold fits,12 final fits and six feature replays. No encoder retraining, dense prefix carrier or resident-memory read. Algebraic folding reproduces actual native predictions/state; every sequential head is charged.

Fitted affine mean68.056%/.903627; quadratic69.965%/.885142. Only .018485 mean NLL gain and1.910 accuracy points, with seed6 NLL regression .035176, fail the polynomial nomination. Gains over parent native .020633/.157749/ .034854 miss the each-seed .05 requirement. Initial affine58.507%/1.138019 and quadratic62.847%/1.012070 remain visible. Both gates FAIL; no unchanged scaling.

Original trained encoder fit4.218015GF each, .535825MF per7,872 fitting presentations; initial optimizer work zero. Combined fitting and solver FLOPs UNKNOWN, so parent fit alone is not total work. Whole study126.813s/608,044KiB includes every fit/replay/evaluation/export; one convergence warning retained. Tensor bytes exclude shared input normalization and metadata. Conditional decoder folds reuse label-trained encoders; no pipeline cross-fit or test. RBF context evidence remains stronger; raw coarse77.604%/.686661 and historical compact66.667%/.902951 controls preserved. AWS owns the next compact context head.

## Appendix B. Race-scaled reception: available arrival support

| Seed | Encoder | Entropy | Top p>=.99 % | 1ms heard | 3ms heard | c4 heard |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | initial | 0.750 | 5.13 | 1.197 | 1.523 | 1.510 |
| 6 | fixed_pass4 | 0.335 | 44.47 | 1.120 | 1.341 | 1.228 |
| 7 | initial | 0.726 | 7.02 | 1.202 | 1.513 | 1.493 |
| 7 | fixed_pass4 | 0.399 | 34.59 | 1.144 | 1.394 | 1.272 |

A relative raw deadline (1+c)T maps through the actual bounded temporal delay. Its heard set is unchanged under a common entering-score shift: all raw clocks scale together. Physical computational times still change. The exact expected receiver count is 1+sum_w pi_w sum_(j!=w) c*pi_j/(1+c*pi_j). The maximum additional physical waiting is .010*(sqrt(1+c)-1)/(sqrt(1+c)+1):1.716ms at c1,3.820ms at c4. The local11ms delay bound remains. This fixes speed-dependent membership, not utility.

43,008 actual native races across all84 sites,16 producer-unseen FIT inputs, eight noise histories and four encoders. The earlier event19/layer0/head0 site is more concentrated than the whole model: top probability>=.99 in71.88%/57.03% of trained races. First events have the most alternative support. A previous one-history1ms lack of reception is not proof that all alternatives are absent.

Trained relative c4 mean additional waiting1.406/1.675ms. These are frozen arrival profiles, with no changed deliveries/writes, new loss, optimizer, DEV or test evaluation. No learned window or quality nomination follows from counts.

Theory99: six numerical contracts and original logit/state checks pass. 59.553s/335,536KiB; exact candidate/clock arrays and checksums retained. Theory100 independently verifies collector end-RNG preservation. Prior trained core fits 2.285696GF each remain paid. Whole audit FLOPs, traffic and energy unknown, not zero.

## Appendix B. Memory-conditioned keys explain routing confidence

| Seed | Encoder | Static gap | Memory gap | Static entropy | Full entropy |
| --- | --- | --- | --- | --- | --- |
| 6 | initial | 0.070 | 1.350 | 0.999 | 0.744 |
| 6 | fixed_pass4 | 0.110 | 5.322 | 0.996 | 0.333 |
| 7 | initial | 0.105 | 1.450 | 0.998 | 0.732 |
| 7 | fixed_pass4 | 0.128 | 4.340 | 0.996 | 0.402 |

Native entering score = query dot static key /sqrt(payload) + clock bias + query dot key_read(persistent memory)/sqrt(payload), then the existing clamp. Exact hooks reconstruct all candidate scores within1.91e-6. Trained memory gaps exceed static gaps in93.75%/92.86% of races. First-event memory is zero and routing is nearly uniform. Persistent-memory reads are the observed confidence source.

Component-only entropies hold the current query fixed; they are algebraic local counterfactuals, not predictions from a modified model history. Large memory scores may express useful specialization or brittle commitment. This decomposition does not establish harmful confidence or a representation-learning regression.

Conventional router z-loss penalizes logsumexp(scores) for numerical stability (ST-MoE, Zoph et al.,2022). Here that quantity is log total rate, so it controls actual first-arrival time. Subtracting it forces total rate1 and changes computation. Choice uncertainty and common speed must be distinguished before importing normalization.

Theory100,5,376 race score pairs /10,752 candidate scalars; four frozen encoders x16 unused FIT prefixes xone noise history. Eight exact logit/state/end-RNG contracts;32.992s/328,876KiB. No loss, fitting, DEV/test or causal harm claim. All weights fixed; prior fits and diagnostic work retained. Full audit FLOPs unknown.

## Appendix B. Clock-preserving route calibration, seed6

| Encoder | Temp. | Scope | FIT NLL | FIT accuracy % | Prefix MF est. |
| --- | --- | --- | --- | --- | --- |
| initial | 1 | all | 2.779015 | 0.00 | 0.591695 |
| initial | 2 | all | 2.773072 | 0.00 | 0.594131 |
| initial | 4 | all | 2.771472 | 0.00 | 0.594131 |
| initial | 2 | layer0 | 2.769006 | 0.00 | 0.592913 |
| fixed_pass4 | 1 | all | 1.376994 | 31.25 | 0.591695 |
| fixed_pass4 | 2 | all | 1.365840 | 34.38 | 0.594131 |
| fixed_pass4 | 4 | all | 1.369226 | 32.81 | 0.594131 |
| fixed_pass4 | 2 | layer0 | 1.362028 | 32.81 | 0.592913 |

At each entering state, keep the original first raw time T and total rate Lambda. A losing exponential residual supplies a uniform independent of T; conditional inverse CDF changes the categorical winner to softmax(score/temperature). No extra random draws, losing value deliveries or receiver updates. The actual selected receiver changes, so subsequent memories and later clocks may change.

Separate16 unused FIT examples, four prespecified noisy histories; every initial/fixed-four-pass model and setting retained. This is mean per-history loss/accuracy, not ensemble inference. No optimizer, new decoder, DEV or test. Temperature2/all NLL gain 0.011155, accuracy change 3.125 points; fixed both-seed smoke gate FAILS.

Theory101. Distribution screens and exact tau1 output/state/all-gradient/RNG contracts pass29.010s/354,328KiB. Positive-temperature training is refused until correct choice/common-clock gradients and optimizer/recovery/accounting exist. Each prefix scores168 keys, writes84 receivers, has8 available units/720state bytes. Original trained fit2.285696GF/2.232125MF per presentation; table work is one traced inference prefix including calibration, not full-audit cost or benchmark advantage.

## Appendix B. Clock-preserving route calibration, seed7

| Encoder | Temp. | Scope | FIT NLL | FIT accuracy % | Prefix MF est. |
| --- | --- | --- | --- | --- | --- |
| initial | 1 | all | 2.444056 | 18.75 | 0.591695 |
| initial | 2 | all | 2.445819 | 18.75 | 0.594131 |
| initial | 4 | all | 2.451157 | 18.75 | 0.594131 |
| initial | 2 | layer0 | 2.441686 | 18.75 | 0.592913 |
| fixed_pass4 | 1 | all | 1.189652 | 43.75 | 0.591849 |
| fixed_pass4 | 2 | all | 1.187395 | 43.75 | 0.594131 |
| fixed_pass4 | 4 | all | 1.205503 | 43.75 | 0.594131 |
| fixed_pass4 | 2 | layer0 | 1.193236 | 43.75 | 0.593067 |

At each entering state, keep the original first raw time T and total rate Lambda. A losing exponential residual supplies a uniform independent of T; conditional inverse CDF changes the categorical winner to softmax(score/temperature). No extra random draws, losing value deliveries or receiver updates. The actual selected receiver changes, so subsequent memories and later clocks may change.

Separate16 unused FIT examples, four prespecified noisy histories; every initial/fixed-four-pass model and setting retained. This is mean per-history loss/accuracy, not ensemble inference. No optimizer, new decoder, DEV or test. Temperature2/all NLL gain 0.002257, accuracy change 0.000 points; fixed both-seed smoke gate FAILS.

Theory101. Distribution screens and exact tau1 output/state/all-gradient/RNG contracts pass29.010s/354,328KiB. Positive-temperature training is refused until correct choice/common-clock gradients and optimizer/recovery/accounting exist. Each prefix scores168 keys, writes84 receivers, has8 available units/720state bytes. Original trained fit2.285696GF/2.232125MF per presentation; table work is one traced inference prefix including calibration, not full-audit cost or benchmark advantage.

## Appendix B. AWS frozen replay critics and parameter calibration

| Frozen conditional score study | Seed | k2/plain k4 MSE | Gate |
| --- | --- | --- | --- |
| Initial fine norm | 7 | 2.083742 | FAIL |
| Initial fine norm | 8 | 2.081475 | FAIL |
| Trained coarse norm | 7 | 2.429824 | FAIL |
| Trained coarse norm | 8 | 2.247774 | FAIL |
| Signed/label critic | 7 | 2.143301 | FAIL |
| Signed/label critic | 8 | 2.493344 | FAIL |
| Training-only shrinkage | 7 | 2.149378 | FAIL |
| Training-only shrinkage | 8 | 2.493344 | FAIL |

Correct first-time-preserving actual-write replay and critics frozen before subset sampling retain unbiased credit. Every reduced-replay nomination fails. The initial fine screen uses84 races; trained coarse uses20. Signed-message and label-aware critic inputs are detached learning features, with32 critic-training and32 heldout FIT prefixes. Sixteen original numerical contracts precede screens.

Score-coordinate MSE is insufficient for shared parameters: cross-site covariance contributes. Actual training-only calibrated parameter variance ratios 2.383887/2.189517 in seed7 and37.462736/1.674462 in seed8 also fail. Exhaustive finite-population and cached factual-probability contracts pass.160 new VJPs paid.

All outcomes, targets, critics and original costs retained. No new producer fit, DEV/test quality or reduced-work learning claim. Diagnostic FLOPs unmeasured, not zero. This rejects unchanged critic reduction rather than counterfactual credit in general. AWS_REPLAY_VARIANCE_FINDINGS and AWS_REPLAY_CALIBRATION_FINDINGS carry lineage.

## Appendix B. Replay allocation: fresh score gains, parameter failure

| Proposal | Seed | Score variance/k4 | Two parameter ratios |
| --- | --- | --- | --- |
| Ordinal k2 | 7 | 2.074656 | Not measured |
| Ordinal k2 | 8 | 3.053016 | Not measured |
| Feature k2 | 7 | 1.206815 | Not measured |
| Feature k2 | 8 | 1.519582 | Not measured |
| Weighted distinct k3 | 7 | 0.768144 | 0.600428/4.011873 |
| Weighted distinct k3 | 8 | 0.693763 | 0.639390/1.009399 |

Ordinal and feature-conditioned k2 proposals sample WITH replacement, using exact1/(kp) correction and a positive floor. Baseline uniform k4 is WITHOUT replacement; uniform k2 with replacement is2.375x. Feature-conditioned priorities improve over that reference but fail reduced-budget gates. Diagnostic exact-return oracle ratios .631115/.519480 need all costly utilities and are not deployable.

The frozen feature predictor then selects three distinct weighted sites on fresh FIT64..95, with exact marginal/pair inclusion probabilities and1/inclusion correction. Conditional score variance falls23.19%/30.62% versus uniform four; uniform three would increase it41.67%. Preserve this positive allocation result. The combined gate fails on actual parameter cases shown above.

Exact expectation/shared-vector variance/uniform nesting contracts pass. 2560 actual diagnostic lanes and80 confirmation VJPs; distinct screen3.578s/ 518,380KiB. Inclusion computation .742/.747ms per prefix, plus tree/discovery/VJP costs. Proposed6 versus8 execution lanes is projected, not measured reduced-work training. Full diagnostic FLOPs unknown; no quality/supremacy claim or unchanged fit.

## Appendix B. AWS compact context head: practical controls remain stronger

| Predictor | Dev accuracy % | Dev NLL | Combined fit GF | Fit MF/target |
| --- | --- | --- | --- | --- |
| s6_initial | 58.85 | 1.268551 | Unknown | Unknown |
| s6_fitted | 67.71 | 0.901193 | Unknown | Unknown |
| s7_initial | 62.50 | 1.145877 | Unknown | Unknown |
| s7_fitted | 64.58 | 0.938003 | Unknown | Unknown |
| s8_initial | 55.73 | 1.170969 | Unknown | Unknown |
| s8_fitted | 68.75 | 0.915083 | Unknown | Unknown |
| raw4 | 65.62 | 0.888167 | Unknown | Unknown |
| raw20 | 66.15 | 0.902387 | Unknown | Unknown |

33-anchor dense local RBF readout uses all distances/basis outputs on frozen native query contexts, plus mandatory raw4/raw20 prototype controls.984 FIT/ 192 DEV, three native trained/initial pairs, fitting-user three-fold regularization selection.72 fold fits,8 refits and352 class-specific KMeans fits are paid. Six Torch and48 query/state contracts pass before decoder fitting.

Both fixed stage/storage gates FAIL. Learned-over-initial mean NLL gain .277040 is supported. Native standalone exports ~115KB versus raw4 40.966KB and raw20 184.334KB; fine-control byte savings do not overcome stronger coarse quality/storage. Raw4 SVC77.604%/.686661 and earlier valid leaders remain.

Whole/per-target combined fitting and inference FLOPs unknown for EVERY arm because solver/replay work is unmeasured. Trained native parent fit4.218015GF/ .535825MF per7872 presentations retained, not total fit. Initial/raw producer fit0 is not zero solver work. Study23.296s/690,896KiB. Portable prediction/roundtrip and three sequential latency repeats pass; exports include normalization/core/ anchors/decoder metadata. This is a dense readout diagnostic, not sparse race attention.

## Appendix B. Integrated exact replay: first seed does not improve quality

| Native seed7 method | Dev acc % | Dev NLL | Whole fit GF | Fit MF/presentation | Infer MF/prefix |
| --- | --- | --- | --- | --- | --- |
| Original route teacher | 57.81 | 1.105326 | 20.074 | 2.550 | 0.591709 |
| Factorized clock only | 56.77 | 1.116189 | 18.244 | 2.318 | 0.591737 |
| All-race exact replay | 58.33 | 1.141794 | 1040.397 | 132.164 | 0.591779 |

Same984 fitting/192 development inputs, eight passes/7872 fitting presentations, p16/layer2/head2/pool2. Every method retains computational delays, temporal races, persistent sparse writes and separate keys/values. All-race replay forces alternatives at the factual first time with real writes; vectorized shadow histories include unrealized downstream effects. Factorized-clock-only removes the original local route teacher.

All-race replay58.333%/1.141794 versus factorized56.771%/1.116189 and original57.812%/1.105326 is not a quality improvement under the prespecified loss criterion. Approximately1040GF counted fit versus18.24GF factorized is paid despite batched CPU execution. A large fit/dev gap motivates independent generalization diagnostics; it does not prove routing credit cannot help.

Whole fitting and per-presentation fitting columns share units and7872 denominator. Inference mean first11 DEV prefixes;2FLOPs/MAC plus unit specials. Eight available receivers,168 scored keys/84 writes per21-event prefix. Candidate discovery, replay, backward and Adam charged; preprocessing, traffic, RNG and energy separate. Source/class/estimator variants and one-seed exploratory scope retained. Other host owns remaining replay/regularization comparisons.

## Exact native terminal-query admission: arithmetic saving, wall loss

All576 paired full-prefix logits/ALLstate checks are bitwise identical across
three saved coarse native seeds. Skip four unused affine classifiers, preserve
all clocks/races/keys/messages/writes. Original full-prefix arithmetic+special
operations .138119/.138007/.138035MF ->.135259/.135147/.135175MF (2.07%less).
Original quality and4.218015GFwholefit/.535825MFperfit-target unchanged. Complete
ATenaudit, query/target/parameter/training-rejection checks pass. Observed Python
wall is~2%SLOWER, so no practicalspeed or supremacyclaim. Full common-unit table
and counterbalanced timings: experiments/AWS_QUERY_ONLY_FINDINGS_20261002.md.
