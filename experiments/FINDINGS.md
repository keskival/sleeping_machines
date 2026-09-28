# Findings log

One entry per result: what we ran, what came out, what it teaches us, what changes.
Newest first. Numbers are single seeds unless stated.

## 2026-09-28

**xLSTM transfer and affine-scan lemma (analysis, not an experiment).** Corrected the earlier overstatement: a real-mode,
fixed-schedule time-vector state is a restricted normalized exponential accumulator, not a full xLSTM layer. It lacks
sLSTM's learned gate/memory mixing and mLSTM's query-addressed matrix memory; E77's Hopfield key/value updates are an
explicit separate operation. xLSTM's 7B model and its 672-run, 80M–7B parameter scaling study are precedents for deep
non-Transformer scale methodology, not transferred evidence for E77 ([7B model](https://arxiv.org/abs/2503.13427),
[scaling study](https://arxiv.org/abs/2510.02228)). New derivation: for fixed event bins, each E74 state step is the
diagonal affine map $(A_k,x_k):z\mapsto A_kz+x_k$. Composition
$(A_j,x_j)\circ(A_i,x_i)=(A_jA_i,A_jx_i+x_j)$ is associative, so a prefix scan can compute all state bins with linear
work and logarithmic parallel depth; the count state follows the same construction. This preserves the recurrence in
exact arithmetic but keeps dense prefix-state storage and does not parallelize topology changes, thresholds, or resets.
**Next:** compare sequential and scan values/gradients on fixed schedules, then time a small serialized pilot. No speedup
or energy saving has been measured.

**Hopfield event-message architecture and depth bound (not yet measured on language).** E77 now applies a causal key/value
associative update over emitted event payloads at each event layer, then forwards the updated payloads. Queries, keys,
and values use separate trainable maps; a gated identity path preserves the original event, with update scale 1/depth.
Scores become arrival advantages through $\\delta=\\tau(s_{max}-s)$, whose exponential decay reproduces the softmax
weights. This connects the Hopfield address and payload operations to the project's delay/vector message primitive.
For a fixed memory, the query Jacobian is a key/value cross-covariance and is bounded by
\\beta D_KD_V/4, independent of the number of memories. If the complete fixed-topology correction Jacobian is bounded
by $K$, 1/depth residual scaling gives singular values of the depth Jacobian in $[e^{-2K},e^K]$. E77 records key/value
diameter upper bounds and the implied query-Jacobian bound as direct diagnostics. Sequence-level sensitivity also
depends on maximum key fan-out $H=\max_j\sum_i p_{ij}$, since one stored payload may influence many later queries;
§107(h) derives a conservative operator bound including $H$ and the query/key/value, gate, and output gains. This is a
conditional fixed-topology state-path guarantee, not proof that event routes are discovered or that hard event births are differentiable. Exact
all-past-event retrieval is the trainability reference; optional top-k limits aggregation but still scores every pair.
E77 logs event pair count, retrieval entropy, score spread, update size, key fan-out, the sequence Jacobian bound and
residual-scaled layer sum, and query/key/value/output/gate gradient norms. The revised E77 architecture is queued, not
yet run. A depth 2/4/8/16 Transformer control now matches E77's 256-character windows, sampled training batches, 5M token
exposure, batch size and optimizer-update count. Widths 104/104/112/136 match E77's parameter count within 4% at all
four depths.
Separate ablations now turn off token retrieval and event-Hopfield updates one at a time.

**Hopfield stability/precision tradeoff (derived).** With values equal to keys, softmax retrieval is $\nabla\Phi$ for
the convex log-partition $\Phi(q)=\beta^{-1}\log\sum_j e^{\beta q^\top k_j}$, and its Hessian is the PSD key
covariance. The update is contractive if $\beta D_K^2/4<1$. For two choices, score credit is $\beta p(1-p)$:
it peaks at a tie and vanishes as retrieval becomes certain. More generally, a score margin $m$ gives non-winner mass
at most $(N-1)e^{-\beta m}$, so raising precision narrows the region that receives learning credit. Separate key and
value codes enable arbitrary hetero-association but remove the general energy-descent guarantee. These derivations
motivate a measured temperature/near-miss sweep before claiming that sharper, sparser retrieval will train as well.

**Adaptive retrieval and depth, theory-to-code update.** E77 has event-state layers, a recurrent state route, event-level Hopfield payload updates, and a token query/key/value retrieval interface. Event-state layers remain non-attention layers; the associative update is a separate operation at emitted events. E77 accepts configurable depth and adds sparse raw-event skips to deeper blocks. This prototype is not yet trained or validated. Sparse describes its synaptic fan-out and event-stream wiring, not a measured sparse execution kernel: `TVLayer` allocates dense time-by-batch-by-unit states, while event and token retrieval score all eligible pairs. The attention window or event top-k limits value aggregation, not candidate search. **Design correction:** sparse aggregation does not imply sparse search or low energy; indexed candidate search and an event-driven simulator remain open engineering problems.

**E79 data/capacity scaling and copy-window ablation.** The race mixture's 256-character-copy setting scores 1.808 / 1.613 / 1.504 frozen bpc (1.782 / 1.593 / 1.483 with online weight adaptation) at 1M / 10M / 90M training characters. Expert count rises with data (K=5/6/7), so this is evidence that the native expert mixture improves as both data and capacity grow, not an isolated data-scaling law. Removing the copy-window cap changes frozen test bpc from 1.808 to 1.779 at 1M, 1.613 to 1.612 at 10M, and 1.504 to 1.512 at 90M. Thus unbounded copy helps modestly at 1M but has no consistent advantage at larger scales; the 0.008 difference at 90M is one seed and has no uncertainty estimate. **Learned:** keep a 256-character copy window in scale comparisons, and treat long-range associative retrieval as its own capacity path. E79 is ahead of the completed 1M one-layer LSTM (2.179 bpc), but adds count/copy memory and has no matched compute/energy accounting. E61 learned query/key race retrieval solved a synthetic recall task in 1–4k examples and extrapolated to 4× context. E77 has no completed LM result yet; these findings motivate its deeper adaptive model but do not establish its superiority.

**E64b gradient language-model baselines.** The validation-selected 1M, 20-pass LSTM completed at 2.1385 valid / 2.1794 test bpc (338,395 parameters). The 2-layer width-256 Transformer completed at 2.3403 valid / 2.3667 test bpc (1,658,907 parameters); both best checkpoints are at their final validation point, so strict convergence is not established. At 10M, the 2-layer width-512 LSTM completed six passes at 1.7448 validation / 1.7993 test bpc (1,199,323 parameters; 7,324 updates, 7,301 seconds). Its best checkpoint is again the final validation point. The matched 4-layer width-256 Transformer (four passes, batch 32) was stopped at step 1,464/4,882 (30%) to release the single-runner slot for targeted market and speech work. At step 488 (10%, 1,832 seconds), it reached 2.6372 train / 2.5857 validation bpc; at step 976 (20%, 3,701 seconds), 2.3096 train / 2.2329 validation bpc; at step 1,464 (5,577 seconds), 2.1277 train / 2.1001 validation bpc. Validation improved by 0.1328 since 20% and 0.4856 since 10%; it remained 0.3553 bpc above the completed LSTM's 1.7448 validation score. This is an interim validation trace, not a test or endpoint comparison, and does not establish which model would finish ahead. Linear projection from 30% is about 18,590 seconds (5.16 hours) for all updates before final evaluation. Host available memory stayed above the runner's 6 GiB floor. No other job should start until the shared safe-runner lock is confirmed free.

**E79 at 10M versus the completed LSTM.** On the same 10M-character training prefix and 1M-character test segment, the frozen E79 race mixture scores 1.6130 test bpc versus 1.7993 for the LSTM: a 0.1863 bpc lead. E79 uses six native experts plus its copy memory; the LSTM has 1,199,323 parameters. This is a single-seed, same-data result, not a compute-, parameter-, or training-budget-matched architecture comparison. The incomplete 10M Transformer trace has no test or endpoint score and is not used to quantify this lead.

**E64b LSTM versus the E79 race mixture at 1M and 10M text8 characters.** The validation-selected 256-hidden-unit LSTM used 20 passes and early
stopping on 200k validation characters: 2.138 validation / 2.179 test bits per character, 338,395 parameters. E79's frozen race
mixture on the same 1M-character training and test segments scores 1.808 with its copy expert limited to the Transformer's 256-character
context (1.782 when weights also adapt online). The frozen mixture is 0.371 bpc below the LSTM and 0.559 below the two-layer Transformer;
the online-adapted score is 0.397 below the LSTM. This is the clearest current real-language performance lead for the native experts.
It is not yet a compute/energy-matched architecture comparison: its count tables and copy index are extra state. It also does not test
the deep E77 time-vector language model. At 10M, the frozen race mixture scores 1.613 test bpc against 1.799 for the completed
two-layer width-512 LSTM on the same test segment. That 0.186 bpc lead is also single-seed and lacks matched compute or parameter
accounting; the four-layer Transformer trace stopped at 30% before producing a test result. **Next:** prioritize the deep
market and speech event models, then return to additional language baselines only if they answer a specific unresolved comparison.
Report memory, scoring work, and adaptation policy alongside loss before making a scaling claim.

**E74/E75 — time-vector speech pilots do not yet learn well, and the memory ceiling was too tight for their default batches.**
E74 on 2k training / 500 held-out utterances at batch 32 completed 6 epochs in 31 minutes, but held-out speaker accuracy peaked at
0.146 and ended at 0.120. E82's partial readout sweep peaked at 0.184 after 240 updates with a non-spiking state readout and layer
normalization; the queued normalized spike readout stayed near chance. E75 at batch 16 reached only 0.044 after 2 epochs before a
38 MB CPU allocation failed; a batch-32 restart failed on a 65 MB allocation. Host available memory remained above 10 GB, so these
failures are consistent with the 3.5 GB per-process virtual-address limit, not host exhaustion. **Learned:** the current speech
representation/readout and optimization are not close to a useful baseline; exact equivariance alone is not enough. Do not spend on
full SHD runs until smaller pilots show a learning signal. **Change:** future E74/E75 pilots and ablations use batch 4, results now
include batch size in their filenames, and the safe runner also monitors whole-job RSS. These lower-batch pilots are queued; they have
not yet established whether the memory issue or the learning issue is resolved.

## 2026-09-26

**E24 — grokking on (a + b) mod 31, half the pairs for training (first runs).** Dense MLP, full-batch AdamW
(wd 1.0): train 1.0 by step 1k, test 0.00 until ~3k, then 0.87 at 20k while the weight norm falls: the regime
exists on this CPU (92 s). Race without sleep: train 0.99–1.0 within ~50 epochs, test 0.002 (below chance 0.032)
through 1,500 epochs. *Learned:* the race memorizes as a partial-key lookup: an unseen pair activates the codes
of training pairs sharing an operand, whose sums are all different, so it is reliably wrong. Contrary to §52.2,
plasticity never goes quiet (57M events by epoch 1,300, still rising) and the weight norm grows; random-feedback
hidden credit keeps the codes churning. Sleep sweep running.

**E23 — class-incremental split-MNIST (5 blocks), the readout was the problem.** Single-head forgetting is
saturated for both learners (race 0.968, MLP 0.979 at 1k frames per task; MLP 0.984 at 4k): each block drives
the old tasks to exactly 0.00, so §50's predictions cannot be tested on it. Added a task-aware readout (only the
task's classes may win; the race is re-run with the other outputs' thresholds out of reach) and plasticity per
block. Task-aware forgetting: race 0.167, MLP 0.030 (1k per task); MLP 0.029 at 4k. *Learned:* §50's prediction
is reversed so far. The race output has no bias (the bias column is zeroed, thresholds fixed), so the prior shift
of each block is carried by old classes' feature weights (THEORY §51). *Changes:* `--price` (learned output
thresholds) added; ablations queued.

**Engineering.** The container was restarted without a GPU (host driver is nouveau; not needed: the code is
numpy-only). The venv lives in the session scratchpad and had to be rebuilt; the interrupted E23 job reran.

**E17 — a continually learning race on the BTCUSDT trade stream (preregistered; 21 confirmatory days,
prequential, day-block 95% intervals).** Direction accuracy on moves ≥ 1 bp: race 0.593 (88% of episodes
decided, 6.7 s mean decision), frozen race 0.595, online logistic regression 0.583 (decides at 10 s), momentum
0.591. Preregistered rules: competitive (met, 33% earlier) and nominally better than logistic regression (+1.0
[0.6, 1.5]). A fairness check made after seeing the results overturns "better": logistic regression at the
race's coverage is as accurate (+0.1 [−0.3, +0.5]); momentum ties the race. Continual − frozen −0.2 [−0.6,
+0.1]: no benefit from test-time learning. The hold race (learning when to trade) traded 1.4% of episodes at
chance accuracy, tying logistic regression's most confident trades. Every learner loses money after a 2 bp
cost. *Learned:* the race can match baselines on a real asynchronous stream while deciding a third earlier;
nothing here is an edge. Plain 10 s momentum is ~59% right on ≥ 1 bp moves, more than the preregistered null
expected (look-ahead checked).

**E20 — exact spike-time gradients on the race architecture (debug: 10k images, 2 epochs).** Gradients verified
by finite differences (error 1e-4 to 1e-3 at step 0.01). Adam needs steps relative to each layer's weight scale
(absolute steps diverged). Depth 1: 0.922 (local rule at this size about 0.85). Depth 3 collapsed to chance (0.10)
exactly as THEORY §27/§30/§36 predicted, and the theory's remedies rescue it: fast homeostasis 0.64, Ward centering
of timing credit 0.40, **both 0.785** (slow homeostasis 0.01 fails, as §36's timescale argument says). *Learned:*
the first theory-derived fix that changes a result materially; deep exact training of race networks is possible
with activity owned by the thresholds.

**E22 — Spiking Heidelberg Digits, first pass (validation, seed 0, 10 epochs): poor.** Race depth 1: counterfactual
0.353, fired-only 0.314, **frozen hidden 0.385**; depth 2 0.356; dense MLP 0.559 / 0.589 (depth 1 / 2). Synaptic events
per utterance 87–139k (MLP ~288k multiply-accumulates). *Learned:* far behind the MLP, and hidden learning hurts.
Suspected cause: hasty decisions. The first output crossing commits on the first few hundred ms of a one-second
utterance, where early spikes are not stronger evidence, and the rule learns from those premature decisions.
Pilots with later output decisions (theta_out 3, 10) and speed–accuracy curves queued.

**E22 decision-time diagnosis (5 epochs).** Output threshold 1 / 3 / 10: 0.301 / 0.361 / 0.273; frozen hidden (threshold
3) 0.380. Speed–accuracy curves rise with later decisions but saturate near 0.36; the relative (MSPRT) rule does
not beat the absolute race. *Learned:* hasty decisions are a small part of the SHD gap; the local learning rule is
the main problem (a frozen random hidden layer still beats trained ones).

**E20/E21 diagnostics (MNIST, depth 2).** Exact gradients with no cancellation at the MLP's 5-epoch budget:
0.9675 vs MLP 0.976, so the race architecture itself is close to dense. Fermi–Dirac (soft k-winner) training,
evaluated hard: 0.942 vs 0.930 trained hard (k = 3, 2 epochs), about 60% of the cancellation cost recovered.
Both are training-time diagnostics under the 2026-09-26 direction decision (ROADMAP).

**E20 at full length (depth 2, 2 epochs, decay + clipping; the constant-rate runs diverged).** Winners per group
3 / 5 / 10: 0.930 / 0.926 / 0.951; local rule (E14, 3 epochs) 0.952; backprop MLP (5 epochs) 0.976. *Learned:*
race cancellation costs about 2 points under exact gradients, and a further ~2.5-point gap remains without it, at
an unequal budget (2 vs 5 epochs). Matched-budget runs (5 epochs, depth 2 and 4) queued.

**E17 follow-up, continual learning as tracking (THEORY §43; exploratory).** Step size η 0.001 / 0.003 /
0.01 / 0.03: 0.583 / 0.593 / 0.593 / 0.578; change-gated 0.578 (mean gate 0.15); frozen 0.595. No setting
beats freezing. *Learned:* the test was confounded, because η and the gate also applied while learning from
scratch in the pilot week; a corrected run (pilot at full rate, tracking variants afterwards) is queued. What
stands: three weeks of BTCUSDT show no drift the race can exploit, and a large step (0.03) chases noise.

**Credit conservation at full length (2 seeds, width 400, 3 epochs).** 0.960 / 0.952 / 0.941 at depths 1 / 2 /
3 vs 0.949 / 0.942 / 0.924 without: +1.0 to +1.7 at every depth, both seeds. The theory's prediction holds; the
debug size (+8–10) overstated it. Non-negative weights at depth 3: 0.932 vs 0.937 (1 seed).

**Theory §27–41 (THEORY.md) and first measurements.** Exact: a Ward identity (timing credit sums to the
deadline's credit), topical-map non-expansiveness with a certified jitter radius, and a commitment-cost
identity (near-miss credit is the gradient of σ × surprisal). Scaling predictions queued as M31–M43. Measured
so far: the input's entropy exponent α ≈ 0.95 (M39), refuting a predicted input-redundancy pyramid; smoke tests
show a common-mode share around 0.03, against a predicted value near 1 (untrained; decisive run queued).

**Engineering.** The host hung a third time: the queue ran one job while an ad-hoc debug run ran beside it.
Now every computation goes through the queue, the watchdog is at 6 GB and checks every second, and `dev.sh`
gives new containers hard memory, CPU and GPU settings.

## 2026-09-25

**Pivotal credit (THEORY §26; debug, 10k images, 2 epochs, output conservation on).** Computing
the first-order jump through the real output weights, gated by arrival before the decision, for
the top hidden layer: depth 1 0.921 (0.925 with group conservation) vs 0.895 random feedback;
with random feedback deeper ("pivot at the top"): depth 2 0.896 vs 0.880, depth 3 0.876 vs 0.870.
Failures on the way, each informative: (a) arrival gating created dead-late units (fixed by a
time-residue weight); (b) pivotal credit through real hidden-to-hidden weights collapses at
depth 3 (0.10); (c) zero-sum credit within hidden groups breaks even the working rule (0.87 →
0.10), refuting my explanation of (b); (d) the exact race Jacobian (evidence shares w/A, a
conservative backward flow) does not fix (b) either (depth 3 0.096; depth 2 0.865, worse than
pivot at the top). The deep path's instability is open. Leads; full-length runs queued.

**M23 at full length (2 seeds, width 400, 3 epochs) — the debug lead did not hold.** Depth 3:
shadow neuron window 0.4: 0.916 / 0.916; window 0.15: 0.920 / 0.924; sampled binary 0.925 /
0.921; **hard binary window 0.928 / 0.931**; residue weighting (E14) 0.924. Depth 1 shadow 0.951
(vs 0.949), depth 2 shadow 0.930 (vs 0.942). *Learned:* the shadow neuron's 5-point debug lead
reversed with full training (and it costs twice the forward compute); weighting-free, sort-free
selection by a hard window is as good or slightly better than continuous weighting at depth 3.
Debug leads from 1-epoch runs are unreliable for rankings; confirm at full length before
presenting them. (Bug found here: result filenames omitted the window, so the w0.4 depth-3 files
were overwritten; values recovered from logs; filenames now include every varied setting.)

**E14 depths 4–5 (seed 0).** Counterfactual vs fired-only: depth 4 0.910 vs 0.891 (+1.9),
depth 5 0.897 vs 0.872 (+2.5). With depths 1–3 (+0.2, +1.2, +1.5, 2 seeds) the gap grows
monotonically with depth.

**M30 — learnt capacities (debug, depth 3).** Target firing rates proportional to each node's
information about the label: 0.733 (linear) and 0.736 (log-ratio) vs 0.746 with equal targets.
*Learned:* no gain from this simple version; the capacity structure holds but this choice of
capacities is not the missing piece.

**M29 — homeostasis as Sinkhorn (debug, depth 3).** Log-ratio (Sinkhorn dual) threshold updates
balance hidden usage better than the linear rule (0.95–0.98 vs 0.89–0.93) but accuracy drops as
balance is forced (0.741 / 0.612 / 0.562 vs 0.746). *Learned:* thresholds really act as prices on
a capacity constraint (THEORY §25), but equal capacities are the wrong target; capacities should
be learnt.

**E14 complete (2 seeds, width 400, 3 epochs, validation).** Counterfactual vs fired-only vs
frozen: depth 1 0.949 / 0.947 / 0.873 (gap +0.2; seeds −0.1, +0.4); depth 2 0.942 / 0.930 / 0.708
(gap **+1.2**, both seeds +1.2); depth 3 0.924 / 0.909 / 0.565 (gap **+1.5**; seeds +1.2, +1.7).
*Learned:* confirmed at 2 seeds: counterfactual credit's advantage appears from depth 2 and holds
at depth 3, modest (~1.2–1.5 points) but consistent; a frozen stack collapses with depth, so deep
race networks depend on hidden credit. Depths 4–5 and credit conservation on top are queued.

**M28 — simplex (multiplicative) learning: negative.** Non-negative, credit-conserving nets,
debug size: exponentiated-gradient updates of each node's evidence mix (urgency additive) reach
0.06–0.31 vs 0.85 additive (depth 1) and 0.08–0.15 vs 0.75 (depth 3). *Learned:* the simplex
view is an exact description (neurons are linear in their evidence mix) but the wrong learning
geometry: multiplicative steps cannot recruit absent evidence (zero or tiny weights), which is
most of what learning has to do. THEORY §24.

**E14 full length, seed 0 (width 400, 3 epochs, validation).** Counterfactual vs fired-only:
depth 1 0.945 vs 0.946 (−0.1), depth 2 0.940 vs 0.933 (+0.6), depth 3 0.926 vs 0.909 (+1.7);
frozen hidden 0.875 (d1), 0.712 (d2). *Learned:* the gap grows steadily with depth as predicted
(§16), but at full training it is ~2 points at depth 3, not the ~19 of the 1-epoch debug run:
longer training lets fired-only credit partly catch up. Seed 1 and depths 4–5 pending; credit
conservation (+8–10 in debug) is queued on top of this.

**Formal consequences, checked (THEORY §22).** (1) Time-shift equivariance is exact: uniform
input delays of 0.05 and 0.2 leave all 300 decisions and hidden firing sets unchanged, times
shifted to within 2·10⁻⁷. (2) Weight norm is urgency: ×1.5 fires 0.084 earlier. (3) **Credit
conservation**, predicted by the dequantized-min derivative: normalising competitor credit
raises depth-3 accuracy 0.649 → 0.746 (residue weighting) and 0.698 → 0.782 (shadow neuron)
(debug size, one seed): the largest single gain at depth. (4) Monotone (non-negative-weight)
networks lose almost nothing (0.848 vs 0.852 at depth 1; 0.745 vs 0.746 at depth 3). Full runs queued.

**M23 (debug size) — sort-free near-miss selection and the two-channel neuron (user's ideas).**
Depth 3, width 200, 5k images, 1 epoch, one seed. Continuous residue weighting 0.649; hard Δ
window (binary) 0.584; sampled binary eligibility (Bernoulli(exp(−Δ/σ))) 0.548; fired-only 0.525;
**two-channel shadow neuron, window 0.4: 0.698** (window 0.05/0.15: 0.62). *Learned (tentatively):*
letting losers keep integrating and emit shadow spikes into a separate learning channel, with
closeness selected by time rather than by sorting or weighting, is at least as good as the
residue machinery at depth. It is the straight-through pattern with a real counterfactual backward
path, and the substrate-natural form of the idea (THEORY §20). Full runs queued.

**M20 — beams as asynchronous shadow events.** One discrete-event pass computes the factual
history plus one branch per hidden-group collapse ("the group's last winner does not fire"), with
shadow continuation of the group's members and per-branch output deltas. Over 50 samples (small
network, 6 branches per sample): factual winner agreement 100%, **branch winner agreement 100%**
against the batch computation of the same swaps. Cost per sample: ~506 output shadow
re-predictions and ~81 hidden shadow updates (vs 611 scheduled events in total), mostly from every
branch re-predicting every output on each output input. *Learned:* branches can run exactly and
asynchronously in the same event queue (§14.6 holds); the overhead is in re-prediction and has
obvious savings (only outputs whose ranking can change). Now a regression test.

**E16 — counterfactual routing gradient in an ordinary MoE (numpy; 8 linear experts; top-1).**
A first version showed a dramatic gain for the boundary term. That was an artifact: my gate
baseline did not implement the exact Switch-style gradient. With exact gradients, on MNIST every
router collapses to one expert (a single linear classifier already reaches ~0.89, so routing is
irrelevant there). On a synthetic task that *needs* routing (8 clusters, each with its own linear
labelling), 3 seeds, 20 epochs: gate + load balancing 0.843 ± 0.015; gate 0.816 ± 0.010;
boundary with 1 shadow expert 0.757 ± 0.029 (σ 0.3: 0.769; σ 3: 0.639); dense top-2 0.716;
boundary with 2 shadows 0.661. The boundary term helped early (5 epochs: 0.695 vs 0.652) and hurt
later. *Learned:* the counterfactual routing gradient is exact for the *current* experts but
**myopic**: an expert that is not routed an input is not learning from it, so its current loss
understates what it could become. Mostly the alternative is worse, so the term reinforces the
existing routing and reduces exploration (balance 0.88 vs 0.97). Routing is a bandit whose arms
improve when pulled (THEORY §15, §19); a useful routing term must value an alternative's
*learning potential*, not just its present loss. *Next:* an optimistic or lookahead variant
(the alternative's loss after one hypothetical update), before any MoE claim.
**Follow-up (lookahead, 10 seeds):** valuing the alternative after one hypothetical step on the
input removes the harm: 0.864 ± 0.018 (95% CI) vs gate + load balancing 0.866 ± 0.012 (paired
difference −0.002 ± 0.022, 6/10 wins), gate 0.855 ± 0.017, dense top-2 0.738 ± 0.021. At 3 seeds
it had looked like a win (0.865 vs 0.843); 10 seeds show a tie, at twice the expert compute.
*Learned:* the myopia diagnosis is right (lookahead fixes it), but on this task the counterfactual
routing gradient does not beat the standard remedy. No MoE claim.


**E15 (debug size) — credit percolation with local layer-by-layer feedback.** Depth 3, width
200, 5k images, 1 epoch. Reach per layer (input side → output side): fan-in 16 counterfactual
0.68/0.51/0.51, fired-only 0.25/0.24/0.21 (above threshold, no decay, as predicted); fan-in 2
fired-only 0.11/0.13/0.18 (decays toward the input, as predicted), counterfactual
0.20/0.16/0.23 (noisy). No clean transition yet at this size; the full sweep is queued.
**Surprise:** sparse hidden-to-hidden connectivity raises deep accuracy (≈0.79 at fan-in 2
vs 0.68 at 16), matching the E9 fan-in result: sparsity seems to help race networks in its
own right, plausibly because dense layers are decided by a few very early spikes.
**Link (user):** the same branching condition governs router training in sparse MoE (top-1
routers get no gradient toward unselected experts); residue-based near-miss credit could train
routers without running unselected experts (THEORY §17.5).

**E9 sparse fan-in (debug size).** 3k images, 1 epoch, validation: fan-in 32 gives 8.1k
synaptic events per image vs 113.9k dense (14× fewer), with *higher* accuracy (0.817 vs
0.792). For scale: the MLP that matched round-3 accuracy (32 hidden units) needs ~25k
multiply-accumulates. If fan-in 32 holds ~0.96 at ~8k events, the inference-energy verdict
flips in the race's favour. *Next:* E9 pilots (3 epochs; fan-in 16/32/64, with and without
growth) are queued; if they hold, a full 10-epoch test-set run for the energy table.

**DRTP control (debug size).** Hidden credit from a random projection of the label alone
(direct random target projection), the pure form of the template mechanism: one hidden layer
0.54 vs 0.59 for counterfactual credit (close, consistent with templates); **depth 3: 0.20 vs
0.63**. *Learned (tentatively):* in shallow nets our hidden credit behaves much like label
templates, but at depth the error-gated, near-miss structure is what keeps race networks
trainable. That is the distinctive part of the idea. Full runs (M22, E14 with crl_drtp) queued.

**M3 against a smooth race-margin objective, and along training.** Output rule +0.82 against
the margin objective. Hidden credit: random feedback −0.10, fired-only −0.17, true-weight
feedback +0.28, sign feedback +0.31 (seed 0; reliability 0.79). Along training (error
objective): random feedback ≈ 0 at 0 and 1 pretraining epochs; sign feedback rises to
+0.45–0.50 after one epoch. *Learned:* the "smoother objective" explanation of the puzzle
fails: random-feedback hidden credit follows neither objective's gradient, yet adds 16 points.
*New hypothesis (template mechanism):* with random feedback each hidden node is pushed to fire
for the classes its feedback row favours, so the hidden layer forms label-conditioned templates
that the output learns to read; label-driven feature formation, not gradient descent. It would
also explain why gradient-aligned (true-weight) feedback did worse in rounds 1–2: its targets
move as the output weights change. First diagnostic (debug size): output–feedback alignment
0.19 vs 0.04 frozen; hidden class selectivity 0.29 vs 0.24 (chance 0.11). M22 (3 seeds,
including true-weight and sign feedback under round-3 settings) queued.

**E14 (debug size) — counterfactual credit with depth.** User's hypothesis: the
counterfactual part matters only beyond one hidden layer. 5k images, 1 epoch, width 200,
one seed: gap (counterfactual − fired-only) +2.2 at depth 1, +1.9 at depth 2, **+19.3 at
depth 3** (0.630 vs 0.437). Share of hidden nodes receiving credit: counterfactual 29% / 44–62%
/ 54–78% by depth, fired-only 17% / 24% / 27–29%. *Learned (tentatively):* fired-only credit
collapses with depth because it cannot reach nodes whose influence runs through events that did
not happen; counterfactual credit keeps deep race networks trainable. Accuracy still falls with
depth at this training length. Full runs (2 seeds, 3 epochs, width 400, plus frozen) queued next.

**E13a (debug size) — reward only.** On 2k images, 2 epochs: near-miss guess 0.30,
reward-modulated winner-only 0.19, pool policy gradient 0.15, supervised reference 0.59.
*Learned (tentatively):* spreading credit over the close calls when wrong roughly doubles
what the standard spiking reinforcement rule reaches. Pool policy gradient may need another
temperature (σ variants queued). 3 seeds at full size queued.

**Queued to resolve open questions:** M3 against a smooth race-margin objective (does the
hidden credit follow a smoother objective than 0/1 error?); M3 along training; E9 sparse
fan-in (16/32/64, with and without growth) as the energy viability gate.

**M18 / M18b / M21 — 3 seeds (small network, 10k images, 3 epochs).**

| rule | accuracy | weights changed |
|---|---|---|
| counterfactual credit | 0.821 ± 0.006 | 12.1M |
| fired-only credit | 0.818 ± 0.005 | 6.8M |
| repair + homeostasis + thin margins | 0.798 ± 0.014 | 0.40M |
| repair | 0.767 ± 0.021 | 0.27M |
| repair + homeostasis | 0.756 ± 0.020 | 0.28M |
| output-only repair | 0.661 ± 0.005 | 0.40M |
| frozen hidden | 0.659 ± 0.010 | 1.4M |
| label-free competitive hidden (M21) | 0.467 ± 0.022 | 13.5M |

*Learned:* history repair comes within ~2.3 points of the gradient-like rules while changing
31× fewer weights; widening thin margins gave ~3 points, homeostasis nothing. Label-free
competitive learning makes the hidden layer much worse than leaving it random: labels are
essential for the hidden layer.

**Puzzle.** Label-driven hidden credit is worth +16 points (fired-only 0.818 vs frozen 0.659),
yet its alignment with the gradient of expected 0/1 error is ≈ 0 even at initialisation (M3 at
0 pretraining epochs: +0.03 random feedback, +0.00 fired-only; estimate reliability 0.77). So
its usefulness is not captured by alignment with that gradient. Candidate explanation: the
0/1-error gradient is dominated by the few samples on a decision boundary, while the rules
improve a smoother objective that pays off later. M19 (smooth loss) and alignment against a
smooth objective should tell.

**M21 (debug size) — can the hidden layer learn without labels?** A label-free competitive
rule (winners move toward their input, rows normalised, homeostasis) with the same output
learning reached 0.29 (η 0.02) and 0.20 (η 0.1) on 2k images, *below* a frozen random hidden
layer (0.33); fired-only credit, which uses labels, reached 0.56. *Learned:* this **contradicts**
the M3 reading below. Label information in the hidden update matters a lot, even though its
alignment with the true gradient, measured at one late snapshot, is weak. Likely reconciliation:
weak local alignment that is consistent over training (M3 measures one point on a pretrained
network). *Next:* M21 at full size (queued); measure alignment along training, not only after it.

**M18 — history repair, full size (small network, 10k images, 3 epochs, seed 0).**
Repair 0.749 vs counterfactual credit 0.818 and fired-only 0.823, while changing 46× fewer
weights (0.27M vs 12.6M). Hidden repairs add ~9 points over output-only repair (0.655).
*Learned:* single-event repair is a real hidden-layer learning signal and extremely sparse, but
not yet competitive on accuracy. The baselines also learn on thin margins and run homeostasis;
repair did neither. *Next:* M18b adds both (queued), then multi-step delegation.

**M13 — learning by half-space projection (single layer, validation).** 7 settings end at
0.899–0.905 after 5 epochs vs 0.909 for the near-miss rule. Needs a minimum correction
deadline (c ≥ 0.2–0.5), because decisions made at the first instant leave no charge to act
on, and a capped step (PA-I); without them it diverges (0.25). *Learned:* a valid
reformulation with no learning rate, not an improvement. The half-space property is also known
(RELATED_WORK.md). *Next:* keep it as the geometric core of repair, not as a rule of its own.

**M19 (debug size only).** After fixing a capped-time bug, the exact gradient of a
cross-entropy over output firing times barely aligns with the gradient of expected *error*
(−0.03 vs +0.61 for the E6 rule), at a size where the estimate is unreliable. *Hypothesis:* the
cross-entropy pushes confident samples too, while error only moves near boundaries. Full size
queued.

**M3 — does each rule point downhill? (3 seeds, 200 coordinates, estimate reliability
0.74–0.80 hidden, 0.95–0.99 output).** Output rule: +0.36 to +0.73. Hidden rules: random
feedback ≈ +0.05, fired-only ≈ −0.01, true-weight feedback ≈ +0.21, sign feedback ≈ +0.17.
Only 13–25% of hidden weights have any nonzero true gradient, rising with timing noise.
*Learned:* at this snapshot, hidden credit carries little gradient information, which fits
counterfactual and fired-only credit tying. (An earlier reading, that the hidden layer's gain
comes from its own competition and homeostasis, is contradicted by M21 above.)
The zero-gradient majority is the blind spot that motivates the tree-of-histories view (§14).

**E9 P1 — cascade.** Only 0.07% of work happens after the output decides. *Learned:* the cost
sits before the decision; only sparse fan-in and routing can reduce it.

**E6 round-3 controls.** Single layer 0.921, frozen hidden 0.895, counterfactual 0.959,
fired-only 0.961, hidden 2000 0.960. *Learned:* depth adds ~4 points under round-3 settings;
the new synapse model gave the single layer ~2.5; width and credit type add nothing.

**Energy re-check.** Round-3 hidden networks: ~108k synaptic events per image, ~3.6× the
inference energy of an equally accurate 32-unit MLP; training 1.2–1.5× cheaper than unbatched
dense training. Only the single racing layer wins at inference (~2.4×).

**Engineering.** Parallel runs hung the host twice; logs could not say why. Everything now runs
one job at a time with a watchdog. Hidden batch assumptions surfaced when moving to one frame
per update (homeostasis inside `teach`; a frame with no hidden spike). A waiter loop that
matched its own command line never started the queue; found by checking, not assuming.
