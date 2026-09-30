# Sleeping Machines

Computing with time: learned delays, vector messages and local memory

**Time is part of the computation.** Messages carry both a vector and an arrival time. Learned delays change arrival order, shape temporal memory and decide which competing message wins. Nodes can wait, accumulate evidence and transform a message's content and timing. A clock race can select a class; learned phase transformations can compose an arithmetic rule. Unrealized alternatives teach better choices. Dormant capacity and sparse activation reduce cost, while the organizing idea is to make timing itself a trainable computational medium.

## The strongest demonstrated results

- **Better real-language prediction.** With 10M training characters, the native count/copy mixture reaches **1.727 test bits per character**, ahead of LSTM (1.799) and four-layer Transformer (1.908) on identical text8 test targets. Lower bits per character means better prediction.
- **Accurate retrieval with far fewer examples.** Local race retrieval learns perfect recall at four times the training context within 4,000 examples in all five runs. The consolidated model preserves 100% on its standard and longer contexts.
- **Rule learning and deep composition.** The consolidated periodic path reaches **100% across all 3,440 unseen modular triples**. Native depth-four order models reach 99.9–100%; shared-motif composition reaches about 99.65% from one pass.

![accomplishments](report/figures/accomplishments.png)

Language: 999,999 identical targets, frozen test parameters and cold test context. Counts fit 10M characters; mixture weights additionally fit 1M validation labels. Neural baselines use different capacities, fitting passes and validation budgets. This is a specialized count/copy mixture result. Generic deep-language learning is measured separately. Retrieval, composition and arithmetic are controlled synthetic tasks.

## Consolidated models: accuracy versus computation

The common implementation now retains perfect modular generalization and longer-context retrieval. New Transformer and LSTM controls use the same synthetic examples. **Higher and further left is better:** more accurate answers from less counted work. The logarithmic axis makes large cost differences visible.

![consolidated work frontiers](report/figures/consolidated_work_frontiers.png)

| Common configuration | Held-out capability | Estimated work per query |
| --- | --- | --- |
| Periodic path; 69 learned scalars | 3,440/3,440 unseen triples | 188 logical operations |
| Two-layer carrier + hard pointer | 100% at four times context | 728,602 logical operations |

The periodic path executes directly through the shared model interface. A two-layer carrier variant is also shown: its phase state supplies the same answers while the carrier adds cost. Speech and other representation tasks use deeper carrier configurations. The architecture chooses the required primitives and depth per task; each task has separately trained weights.

**Learning work is also selective.** The periodic teacher makes 29,003 mistaken-example updates and 145,015 learned-scalar update visits. The dense arithmetic controls make 4,800 Adam steps: 92.2M parameter visits for the LSTM and 132.5M for the Transformer. These count parameter updates, excluding optimizer state and backward arithmetic; they are not training FLOPs or joules.

Arithmetic: 1,473 fitting triples, a 200-epoch budget, all 3,440 unseen triples; supplied period 17. The phase-only path stops after 47 passes, when an entire fitting pass makes no updates. Recall: 4,000 pointer-fitting examples plus 512 neural-fitting examples; the dense controls receive all 4,512 examples for eight epochs. Width 32 and two generic layers where present, seed 6, one small dense setting. Work is an analytic logical-operation estimate, including configured vector maps, routers, scans, normalization, clock candidates and pointer search. These inference counts are not measured joules or backward/optimizer counts. Full work definitions appear in the evidence appendix.

## Native components: quality, work and sample efficiency

The native components establish why selective temporal computation is promising. Timing patterns, composition and deep order reach high accuracy with sparse event activity. Retrieval and rule learning also show that a reusable computation can generalize beyond the observed examples.

![supremacy map](report/figures/supremacy_map.png)

Native work panels show event deliveries and dense multiply-adds: different activity measures. Native delivery counts omit candidate scans and local array arithmetic, so their ratios do not measure total work or energy savings. Synthetic evaluation sets were reused during development. Market points are causal development screens with prior-day evidence. The preceding page uses a common logical operation ledger; Appendix B includes complete optimizer-step arithmetic estimates.

## How the model computes and learns

The architecture computes through local memory, vector payloads and timing. A learned delay changes which arrivals interact and which route wins; it participates in the function being learned. Activity sparsity controls how much computation occurs. A message can carry a learned representation, a pointer, evidence or a periodic state. Layers need not all perform the same operation. Their job is to preserve useful information and recruit the computation needed for the task.

![shared architecture](report/figures/shared_architecture.png)

- **Local temporal memory** accumulates observed content and elapsed time without evaluating empty time ticks.
- **Computation through delays** uses waiting times, arrival order and clock races to transform information and select outcomes.
- **Hard races** choose the emitted vector and delay. Losing alternatives provide training credit without becoming identical forward messages.
- **Reusable memories** include conditional outcome statistics, relative pointers and learned phase transformations.
- **Trainable depth** uses bounded carrier updates to preserve representations and credit through a hierarchy.
- **Appropriate supervision** teaches a completed class decision or the next event's type and waiting time, including silence.

The consolidated model is trained independently on each task. Input queries contain only observations already available at their cutoff. The label and next event remain targets. Its current query interface processes event packets; persistent scheduling across overlapping queries is an additional systems capability.

Small local vector maps and query readouts can remain dense. The intended efficiency comes from selective event/state computation and appropriate primitives; total memory access and communication must also be measured when assessing an implementation.

## A mathematical foundation for trainable computation

| Principle | What it enables |
| --- | --- |
| Stable transport through depth | A reversible packet/memory program preserves conditional value and credit norms at any depth. Twelve-layer speech prototypes learn with observable memory queries. Readout alignment, route support and transfer remain separate requirements. |
| Active communication support | Inputs need causal paths through which to interact. A context channel supplies joint information when sparse packets leave local groups disconnected. |
| Credit to unrealized alternatives | A losing payload or timing choice can show how a different route would change the outcome, while forward computation remains a hard race. |
| Periodic state as an isometry | Learned rotations/reflections have unit-magnitude occurrence derivatives. Their composition supports reusable arithmetic instead of a table of observed tuples. |
| Certified composition | Target-constrained min/max composition of phase errors certifies the fitted modular rule across all 4,913 possible tuples; exhaustive checking confirms it. |
| Natural supervised credit | Categorical and event likelihoods both credit predicted sufficient statistics minus observations. Silence enters through integrated exposure. |
| Useful optionality | Reserve consists of distinct, attainable future corrections under a causal work budget. Reachability and transferable learning matter alongside immediate loss. |

### From mathematics to an engineering discipline

The theory connects representation, topology, clocks and optimization. Expressivity describes what a network can compute; transport describes whether information and credit survive; the objective describes what the teacher asks it to learn. These pieces must agree. The periodic certificate is one concrete case where the formal model explains and verifies a learned computation.

A common implementation makes these principles reusable across tasks. Efficient primitives can own a computation when its structure is known; a deep carrier can learn representations when it is not. The research objective is to combine this flexibility with affordable route discovery and increasingly capable models.

Formal derivations and their assumptions are indexed in the project's theory notes. Conditional stability and a certificate for a fitted rule do not establish global optimizer convergence or a scaling law.

## Potential: what the demonstrated capabilities put within reach

The opportunity is intelligence that learns reusable structure and spends computation in proportion to useful activity. Strong predictive mixtures, reliable retrieval, efficient temporal composition and certified periodic computation already provide working foundations. The shared implementation gives those mechanisms a common place to develop.

### Compact prediction and memory

Local predictive memories can support compression, stream forecasting and adaptation to recurring patterns. Learned pointer rules can keep their meaning as context grows. These are useful ingredients for models that retain experience without recomputing an entire dense history at each query.

### A reusable event-to-decision module

Sound, event-camera vision, touch and telemetry all arrive as evolving evidence. A capable recognizer could maintain context, identify meaningful patterns and answer as soon as confidence is sufficient. Deep speech learning and temporal composition establish parts of this capability. The new temporal encoder improves private speech accuracy by 7.42 points; its gain also holds on disjoint utterances, and its learned clocks contribute to classification. Reliable early decisions and broader generalization are central development goals.

### Training efficiency creates capability

Cheaper updates can buy more data, depth and experimentation from the same budget. Better sample efficiency makes each experience more useful. The exact event-memory scan already reduces audited forward/backward CPU time by 1.68×, preserving the checked predictions and gradients. Local timing and routing teachers provide additional routes to efficient learning.

### Industrial and scientific applications

Machines and instruments could maintain local predictive models, recognize changes and adapt from new operating conditions. Laboratories could use event histories to select informative measurements and control experiments. These applications connect fast local responses with longer-term memory, close to the source of the observations.

## Potential: a new foundation for frontier models

If the architecture combines frontier predictive quality, reliable deep learning and lower total training and inference cost, it would change the practical recipe for building frontier models. Useful capacity, active computation and learning cost could become more independently controllable.

### Useful quality at a lower energy cost

A broad advantage can begin with comparable quality at substantially lower measured training or inference energy. A modest quality tradeoff with a large energy saving can also unlock new applications. Tenfold savings, or larger, would transform feasible deployments and research budgets; these are conditional scenarios, not current measured energy ratios.

### The economics of creating intelligence

A fixed power and capital budget could produce a more capable model, more specialized models or more research. Teams constrained by compute could enter new scales and applications. Efficient learning would expand what is feasible, including the size and sophistication of frontier training runs.

### Capacity that can remain dormant

Large stores of memories and skills could stay available while only relevant portions participate in a decision. Straightforward situations could resolve with little work; difficult ones could recruit more computation. Cheap search, communication and selective credit would make this a different scaling regime from repeatedly activating an entire dense model.

### Autonomy in mobile platforms

Robots, vehicles, phones, wearables and remote instruments could sustain perception, memory and adaptation within a mobile power budget. Local intelligence could retain context continuously, react quickly and learn from experience while reducing dependence on continuous connectivity.

### Hardware and infrastructure

If sparse communication and persistent local state determine cost, processors and data centers would increasingly optimize those operations: message delivery, delay queues, memory access, candidate lookup and selective updates. Investment would follow measured useful learning per watt and per unit of capital. The architecture could change which accelerators are most valuable and where capable models can operate.

These larger outcomes depend on demonstrating quality, scaling, retention and total resource cost together. The existing results provide concrete footholds for that research program.

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

The strongest completed speech result in the model family is **79.69%** on 512 private held-speaker utterances (selected single six-block prefix; trained with twelve blocks). Accurate general recognition remains an open capability; published official-test results below are reference targets, evaluated on a different partition.

| Private development configuration | Correct | Accuracy ↑ |
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
| EventSSM: asynchronous learned state-space layers | 95.9% |
| S7: input-dependent temporal state | 96.3% |
| 2026 multiscale residual encoder; publisher abstract | 96.44% |

Our private sample uses training-file speakers 3/6; official test accuracy is unmeasured. The temporal residual uses three passes and a fresh optimizer; its larger capacity and budget are not a matched single-factor comparison. Its listed score selects the best private-development epoch; the curve shows all three. The original parent was fitted on 4,096 examples. Continuation timers exclude loading; the residual timer includes it. Both include evaluation and are not energy measurements. One seed. References: [EventSSM](https://arxiv.org/html/2404.18508v2), [S7](https://arxiv.org/html/2410.03464v1), [multiscale encoding](https://www.sciencedirect.com/science/article/abs/pii/S0893608026003345). The last reference's full training/selection protocol has not yet been inspected.

## Appendix A (continued). Learned timing and transfer

**Learned delays contribute to the answer.** Restoring the hidden clocks to their initial values, with source vectors, state/value maps and the trained classifier retained, loses 13 correct answers. Timing changes the temporal interactions used by the representation; it performs computation.

![e145 learned timing and transfer](report/figures/e145_learned_timing_and_transfer.png)

| Same trained readout; one subsystem reset | Correct | Accuracy ↑ |
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

| Private development configuration | Correct | Accuracy ↑ | NLL ↓ |
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

| One matched fitting pass | Parameters | Fit accuracy ↑ | Private accuracy ↑ | NLL ↓ |
| --- | --- | --- | --- | --- |
| Six-block control | 395,814 | 90.72% | 78.12% | 0.639 |
| Twelve blocks: bounded outputs | 696,888 | 90.72% | 78.12% | 0.639 |
| Twelve blocks: directional units | 696,120 | 90.77% | 78.32% | 0.644 |

![e164 depth use and work](report/figures/e164_depth_use_and_work.png)

Directional conditioning normalizes temporal-state features before their output map. An invertible coordinate change rescales classifier-sensitive directions and preserves hidden null directions for later computation. The fixed transform folds into an ordinary map at deployment. Initial outputs and old teachers remain exact; fitting replays verify the actual proposed update.

The directional model reaches 509/657 on the reused audit. Removing its six appended blocks changes 11 predictions: 0 are correct only with the blocks and 9 only without them. This measures fitted contribution with the trained prefix/head retained; it is not a retrained architecture comparison.

All arms inherit the paired-head checkpoint and use 6,144 fitting utterances, the same order, channel/time transformations and one encoder pass. Old-group LR is 0.0000203125; directional new groups use 0.0001953125 from fitting-only replay. Changed normalization and update coordinates form one intervention. New blocks initially add 36 ms latency; labels supervise completed untimed utterances. Audit reuse is explicit; official-test parity remains unmeasured. CPU points are one warmed observation per model, packing/query included and loading excluded; energy is unmeasured. Weight-coordinate folding, all source work and added depth must be charged during training. Local maps remain dense, with no empty ticks or event-pair attention. Theory §§249–264.

## Appendix B. Breadth of the common implementation

The common event backbone has development screens across language, event prediction, temporal composition, images and event cameras, in addition to speech, retrieval and arithmetic. These bounded screens establish implementation breadth; the stronger native comparison results use their own complete protocols.

**How to read the comparison:** accuracy is the percentage of correct answers, so **higher is better**. Bits per character (bpc) and nats/event measure prediction error, so **lower is better**. FLOPs estimate arithmetic work: **lower means less computation**. Each work pair lists the common model first and the Transformer (TF) second.

| Task and quality direction | Common quality | Transformer quality | Inference contractions/query: common / TF ↓ | Complete step arithmetic/query: common / TF ↓ |
| --- | --- | --- | --- | --- |
| Text8<br/>Prediction error: lower is better | 2.915 bpc | 3.729 bpc | 1.26M / 1.84M | 11.32M / 5.91M |
| Market event prediction<br/>Prediction error: lower is better | 3.823 nats/event | 4.208 nats/event | 1.26M / 1.84M | 11.31M / 5.91M |
| Temporal composition<br/>Accuracy: higher is better | 97.3% | 90.2% | 0.29M / 0.38M | 2.38M / 1.21M |
| MNIST<br/>Accuracy: higher is better | 75.8% | 69.9% | 1.66M / 2.57M | 13.72M / 9.55M |
| Event-camera gestures<br/>Accuracy: higher is better | 59.1% | 15.9% | 23.99M / 124.23M | 215.70M / 476.33M |

The common screens use eight layers and the Transformer references two, both at width 32 for eight epochs. They share neural-fitting examples, held-out examples, encoding, objective and schedule, with fitting batches of 16/64 respectively. These are small reference settings. Two-layer follow-ups retain 100% recall at both context lengths and reach 96.1% composition versus 97.3% at depth eight, using four times fewer hidden carrier emissions.

**Why training can cost more:** the race core evaluates all three candidate vector payloads during training, versus only the winner during inference. Its eight layers also exceed the reference's two. With short contexts, that work outweighs the saved attention cost; these rows do not show a training efficiency advantage. On the longer event-camera prefixes, the common model's complete step uses 45.3% of the reference arithmetic on the audited four-query batch.

Seed 6; neural fit/development counts: text 2,048/256, market 512/256, temporal 1,024/256, MNIST 1,024/256, gestures 88/44. The common text model also has a separately fitted 32,768-character evidence bank; market evidence is fitted on a prior day. The references have no such bank. MNIST uses pooled training-set images; gestures use first-second prefixes and disjoint users. No official real-data test sets are used here. The market fixed-evidence reference is 3.670 nats/event.

M = million, G = billion. Inference counts estimate contractions per unpadded prefix, excluding other arithmetic. Complete steps include forward/loss, backward, clipping and Adam, with actual padding and all training alternatives. Each is one four-query fitting batch divided by four. These are different accounting boundaries, not whole-run budgets. The next page defines complete-step coverage.

## Appendix B (continued). Complete training steps

The ledger charges the complete learning step for both models: prediction and loss, reverse credit, gradient clipping and Adam. It traces the actual tensor operators in a representative four-query fitting batch. Every observed floating operator has a declared formula or a classification as comparison, special function or data movement. Lower arithmetic work is better.

![e172 complete training work](report/figures/e172_complete_training_work.png)

| Model/task | Forward + loss | Backward | Clip | Adam | Total |
| --- | --- | --- | --- | --- | --- |
| Language: Common | 3.598 | 7.518 | 0.040 | 0.161 | 11.316 |
| Language: TF | 1.907 | 3.906 | 0.020 | 0.081 | 5.915 |
| Market: Common | 3.597 | 7.517 | 0.040 | 0.160 | 11.314 |
| Market: TF | 1.907 | 3.905 | 0.020 | 0.079 | 5.911 |
| Temporal: Common | 0.694 | 1.487 | 0.040 | 0.158 | 2.380 |
| Temporal: TF | 0.370 | 0.739 | 0.020 | 0.079 | 1.208 |
| Mnist: Common | 4.376 | 9.141 | 0.041 | 0.163 | 13.721 |
| Mnist: TF | 3.067 | 6.361 | 0.024 | 0.096 | 9.549 |
| Dvs: Common | 69.925 | 145.578 | 0.040 | 0.159 | 215.701 |
| Dvs: TF | 143.212 | 333.017 | 0.020 | 0.081 | 476.330 |

All table values are MFLOPs per query. A multiply-add counts as two arithmetic operations. Fused attention, normalization and activation kernels use shape-based mathematical formulas. Exponentials, logarithms, trigonometric functions, roots and comparisons are recorded separately; memory traffic and execution overhead are outside the arithmetic total. This is formula coverage of the observed CPU steps, not measured hardware instructions or joules.

The audit does not reconstruct total historical training work. Fitting evidence banks, preprocessing, calibration, inherited weights, evaluation and search add work beyond these steps. Different original batch sizes also change optimizer amortization. A future whole-run ledger must record all of these costs alongside energy and quality.

## Appendix B (continued). Language benchmark protocol

The native predictor and saved neural references score the same 999,999 character targets, starting from a cold context. Parameters are frozen during testing. Earlier observed test characters can supply causal context, including the mixture's bounded 256-character copy cache. Lower bits per character means better prediction.

| Predictor | Test bpc ↓ | Fitting and selection budget |
| --- | --- | --- |
| Native count/copy mixture | 1.727 | 10M count fitting + 1M mixing-weight fitting |
| LSTM, width 512; one recurrent layer | 1.799 | 10M characters, six passes; 200k validation selection |
| Transformer, width 256; four layers | 1.908 | 10M characters, four passes; 200k validation selection |

The count/copy mixture improves on LSTM by 0.072 bpc and Transformer by 0.181 bpc. These results establish useful specialized prediction; generic learned representations are assessed in the separate language screen.

The native mixture combines order-0 through order-6 conditional counts, Witten–Bell prediction and a bounded copy predictor. Its count arrays occupy 66.55 MB. Vocabulary, capacities, optimization and fitting budgets differ from the neural references. The comparison does not measure total training energy or a matched-capacity advantage.

### Characters, subwords and a persistent stream

All four predictors use the same 27-character alphabet. Characters are tokens, but a subword representation can reduce the number of arrivals and expose longer patterns within a fixed credit window. It also enlarges the output vocabulary. The useful comparison is quality and total work per original character, with train-only tokenizer fitting and declared buffering latency.

A generic streaming event-state implementation retains modal memory and pending delayed messages across chunks. An eight-layer contract confirms identical predictions under chunk splitting, causal prefix invariance and nonzero learning signals in every layer. Fifteen input arrivals cause 120 layer deliveries, with no repeated prefix processing. The trained eight-layer stream reaches 3.351 validation bpc in the separately described small screen.

One exploratory seed. Text8 offsets: count fitting [0,10M), mixing-weight validation [90M,91M), test [95M,96M); test index zero is excluded for all three predictors. The mixture selects its update rate on validation. Saved neural weights are unchanged. Results: E173/E174; stream contract: E175.

## Appendix B (continued). Learned language and depth

A bounded screen trains the common event backbone to predict the next character. Both configurations use width 32, the same 8,192 training characters, four passes, 32-character contexts and 1,024 validation predictions. All eight layers' value, route and memory-time parameters update. Lower bits per character means better prediction.

![e133 generic language](report/figures/e133_generic_language.png)

| Depth | Learned parameters | Validation bpc: lower is better | Total CPU wall time |
| --- | --- | --- | --- |
| 1 | 7,671 | 3.464 | 27.4 s |
| 8 | 53,430 | 3.395 | 170.5 s |

Eight layers improve validation loss from 4.752 to 3.395 bpc, versus 3.464 with one layer. Shuffling preceding characters while preserving the last character, count and timestamps increases the deeper model's loss to 3.805; replacing preceding context raises it to 3.777. These frozen input probes show context sensitivity, not a retrained baseline comparison.

The deeper model has more parameters and takes more CPU time. The quality/time panel includes fitting and evaluation, with backpropagation and optimizer updates executed during fitting. These bounded-query models replay preceding context. The following persistent implementation consumes each character once. Physical memory traffic and joules remain unmeasured.

One seed; different parameter counts. This establishes a generic learned-language foothold and a small depth gain, not competitive large-scale representation, a matched tuned dense-model advantage or a scaling law. The native 10M-character mixture remains a separate result. Official test data are untouched. E133 preserves commands, source/data hashes, layer diagnostics and work coverage.

## Appendix B (continued). Persistent learned language

The event-state language model consumes each character once and retains local modal memories and its delayed-message queue. Chunk boundaries truncate learning credit without discarding the observed history. Only actual event arrivals evaluate layers; text time is measured in token intervals.

![e176 stream language learning](report/figures/e176_stream_language_learning.png)

| Pass | Fitting bpc ↓ | Validation bpc ↓ | Deliveries/pass |
| --- | --- | --- | --- |
| 1 | 3.746 | 3.831 | 65,784 |
| 2 | 3.400 | 3.528 | 65,784 |
| 3 | 3.251 | 3.420 | 65,784 |
| 4 | 3.142 | 3.351 | 65,784 |

Validation loss falls from 5.329 to 3.351 bpc. All eight layer teachers are nonzero in every fitting pass. Each pass consumes 8,223 characters including warmup and makes 65,784 block deliveries. The result establishes learning with persistent causal state and no prefix replay.

One seed; 28,403 parameters, width 32, sixteen temporal modes per block. Four passes, 128 Adam steps/pass, credit truncated every 64 characters, 31 warm characters and 1,024 validation targets. Target offsets match the bounded E133 screen; topology, capacity, history and update counts differ, so this is not a matched intervention. Total CPU wall time 364.9 s including fitting/evaluation; peak RSS 364.8 MiB. These are event counts and observed resources, not total arithmetic, physical memory traffic or energy. No official test or large-corpus claim. E176.

## Appendix C. Evidence and metric definitions

| Metric | Interpretation |
| --- | --- |
| Bits per character | Held-out negative log probability in base two; lower is better next-character prediction. |
| Accuracy | Fraction of correct class decisions on the declared development or test protocol. |
| Event likelihood | Scores both the next event type and waiting time, including the observed silence. |
| Logical operations | The consolidated ledger assigns 2 units per MAC and 1 per other scalar arithmetic/nonlinear operation or estimated sort comparison. |
| Resource boundary | Logical memory reads are reported separately. Transfers, allocations, kernel launch and instrumentation are outside the arithmetic ledger. Division, exponential and remainder costs have unit weights. |
| Energy | Measured total joules over an explicit boundary. Operation estimates and CPU timings support work comparisons, but are not joule measurements. |

The evidence is preserved in versioned result summaries with configurations, split identities, learning curves and source hashes. E173/E174 support the language comparison; E61 supports retrieval; E34/E53/E54 support native composition; E41 supports the original periodic computation. E121/E124 establish consolidated arithmetic and its certificate; E123 supplies the new dense controls and E124 the operation ledger. E118/E119/E122/E125/E126 support deep speech, readout and causal-context comparisons; E127–E131 audit credit geometry, hard race boundaries and separate key/value learning; E132 checks a joint race-credit formalism, E133 supplies the language/depth screen, and E134–E135 test whole-value credit and content-selective temporal memory. E136 audits reversible augmented transport and its supervised memory boundary, including twelve-layer query/learning interventions. E137 tests compact memory queries and class-visible credit geometry. E138–E141 examine richer source messages and trainable signed temporal memory, with exact local teacher and initial-nesting contracts. E142 establishes signed-state and first-coalescing identities; E143 tests a larger nonlinear temporal residual learner, and E144 audits simultaneous state/query pooling. E171 reproduces the consolidated screens and selected speech answers, and checks causal input boundaries. E172 records complete training-step arithmetic; E175 checks the generic persistent language stream.

The project theory index contains formal assumptions and proofs. Research findings retain detailed analyses and the full experimental record. The model documentation describes reproducible configurations and operational procedures. This report presents the project, its evidence and its potential.
