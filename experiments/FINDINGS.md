# Findings log

One entry per result: what we ran, what came out, what it teaches us, what changes.
Newest first. Numbers are single seeds unless stated.

## 2026-09-29

**E83 scalar route optionality through descendants (§§149–150).** We ran six
guarded frozen-checkpoint audits on the same stratified 32-example
held-out-speaker subset, selecting eight factual errors. The alternatives
included route swaps at two margin widths, closed-route birth, spike birth,
and their combined proposal pool. Route-only, route-birth, and spike-only
audits produced no immediate-loss or finite suffix-step advantage among their
sampled leaves. In the spike-only arm, verified interventions added events in
L1/L2/L3 on 13/11/3 leaves, but none reached L4. This shows that making more
local events is insufficient unless their descendants reach the classifier.

The combined action pool produced 31 leaves; all 31 forced interventions were
verified. Five leaves added L4 events (eight added L4 events total across
these separate replays). Four of 31 lowered deepest-head loss, four improved
the matched one-step clipped suffix-SGD progress, and three improved both.
One branch worsened current loss while increasing finite suffix progress.
The recursively backed-up scalar was positive on one of eight error trees at
each tested learning-option weight λ=0, 1, and 10, and on two trees at λ=100;
the second tree flips only because that uncalibrated high weight values its
larger suffix step over its worse current loss. The result is a small
checkpoint-conditioned route to deeper label credit, not a training gain or
SHD accuracy improvement. Action proposals were not paired with common random
numbers across arms, and no route-birth leaf was sampled in the combined arm.
The audit stores branch traces for inspection, while the proposed recursion
itself propagates one scalar and does not require a global tree.

**Theory update.** §149 now defines the terminal value as immediate
label-loss advantage plus measured finite suffix-learning progress, and backs
that scalar through counterfactual descendants. §150 distinguishes event
propagation from reachable class utility and reports the six audits. λ needs
calibration against real held-out loss after a sparse scalar update; this
virtual step remains a development diagnostic. The next decisive experiment
is a bounded update on training data with recorded proposal propensities and
held-out-speaker evaluation after training.

The jobs ran serially through `experiments/queue/run_safe.sh` with 3.6 GB
virtual-memory cap, 2.6 GB process-group RSS cap, 8 GB host-availability
floor, and 20-minute watchdog. Peak observed RSS was 407 MB and available host
memory stayed above 11.8 GB; all completed jobs exited 0. The first spike-mask
attempt exited before useful work due to a tensor-broadcast shape error; the
mask was corrected and the guarded rerun succeeded.

**E83 1,024-example spike-pair audit: event propagation is not pair synergy (§146).**
The frozen held-out-speaker audit tested every binary outcome for one selected
near-boundary pair per batch. L2 yielded 141 control and 69 late-only pairs;
L3 yielded 31 and 11. Among natural-off pairs, opening both raised downstream
spike counts in 38/64 and 19/34 L2 cases, and 6/10 and 3/5 L3 cases. The same
double opening lowered deepest-only loss in 8/64 and 4/34 L2 cases, and 5/10
and 4/5 L3 cases. At L3 the candidate pool is tiny: the late-only median
double-open loss change was −0.032, but a single +1.40 harmful outlier made
the mean harmful. These are checkpoint-conditioned interventions, not an
accuracy gain.

The four-corner difference-in-differences $\Gamma=L_{11}-L_{10}-L_{01}+L_{00}$
isolates credit specific to the pair. Across 113 natural-off pairs, no
double-open intervention improved deepest-only loss when both singleton
openings were individually non-helpful. With $|\Gamma|>0.01$ as a descriptive
threshold, only 2/42 L3 pairs and 0/210 L2 pairs crossed it; median $|\Gamma|$
was zero. L2 event openings frequently created deeper activity without
changing class loss, so spike counts alone do not measure useful propagation.
This spike-event audit selects any two distinct units within 50 ms; it does
not enforce a shared downstream receiver and therefore does not test the
topology-conditioned same-receiver route-pair mechanism.

**Theory update.** §146 derives the exact mixed derivative of the independent
logistic event relaxation: $\partial^2\tilde L/(\partial m_i\partial m_j)=
p_i(1-p_i)p_j(1-p_j)\Gamma/\tau^2$. Pair replay work is warranted where a
shared receiver and overlapping arrivals make an interaction plausible; the
four-corner estimate determines whether such pair-specific credit exists.
The next mechanism test should compare topology-matched shared-receiver pairs
against time-matched nonshared controls and log message acceptance, event
time/payload changes, deepest loss, and replay work. No trainable pair update
or SHD accuracy improvement is established.

The four jobs ran serially through `experiments/queue/run_safe.sh` with 3.6 GB
virtual-memory cap, 2.6 GB process-group RSS cap, and 8 GB minimum available
host memory. Peak observed RSS was about 334 MB; `MemAvailable` remained above
11.1 GB. No guard tripped.

**E83 layer-balanced route pairs: deep support without class learning (§143).**
This matched D4 seed-6 follow-up changed only pair selection relative to the
global-pair arm: same initialization, 120 examples, four epochs/120 updates,
one pair shadow per minibatch, and 128 held-out examples. It selected
43/34/15/28 route pairs from layers 1–4 over the run; the global sampler had
sent all 120 pair shadows to layer 1. Layer 4 then retained 100% of held-out
example support in every epoch. Yet final anytime-with-fallback accuracy was
6/128 (4.69%), compared with 8/128 (6.25%) for the no-pair control and 14/128
(10.94%) for the global-pair arm. Paired exact McNemar tests were
$p=0.791$ (balanced vs control) and $p=0.180$ (global vs control). These
single-seed differences are inconclusive.

Support came with an unstable activity regime: the balanced run's mean
per-utterance layer event counts at epoch 4 were [122, 189, 363, 1,647],
and its four-epoch averages were [142, 192, 334, 1,421]. Adjacent activity
amplification ratios from those averages were 1.35, 1.74, and 4.25. Its
training losses stayed between 6,574 and 12,932; final late-prefix NLL was
14,699, and the classifier predicted only two of 20 classes. Therefore
restoring deep support did not produce a useful representation; it exposed a
second failure mode, overactive deep event generation with collapsed class
evidence. A matched intervention associates this with layer-balanced route
updates, but does not isolate the pair sampler from the threshold/reset
dynamics or noisy route utility.

The paired replays give additional separation between action utility and
learning: joint opening lowered its matched loss in 38/120 selected pairs
(31.7%), while $\Gamma<0$ in 30/120 (25.0%). Per-epoch raw counterfactual to
pathwise gradient norm ratios ranged 0.0011–0.0146, and their cosine ranged
from -0.202 to +0.251. Useful counterfactuals exist, but they were neither
common nor a reliably aligned update. The balanced policy samples layers
uniformly by design; absent inverse-propensity weighting it estimates a
layer-equalized objective rather than the full candidate-pool gradient.

**Theory update.** §143 now formalizes the joint operating-point requirement:
track active-example support $c_\ell$, conditional event multiplicity
$\mu_\ell$, and total activity $n_\ell=c_\ell\mu_\ell$ separately. It
derives a cost-constrained route utility that includes the counterfactual
change in per-layer event/message work, and identifies the next experiment:
collect matched class-loss/work deltas before setting any activity budget.
E83 also scans hidden states on a 1 ms grid, so its current SHD training
result is not evidence of sparse asynchronous training cost.

**Frozen validation route/work audit (§143).** With no parameter updates, the
balanced checkpoint was evaluated on the same 128 held-out-speaker examples.
One pair was sampled per eligible layer and minibatch: 115 pairs total,
32/32/19/32 from layers 1–4; eligible pool sizes were 214,920/228/38/273.
Joint opening improved the matched prefix loss in 5/32 (15.6%), 10/32
(31.3%), 11/19 (57.9%), and 16/32 (50.0%) pairs by layer. Median
$L_{11}-L_{00}$ was 0.000/0.236/-0.268/-0.001, while the means were
19.93/74.07/1.36/0.46. Early-layer means are outlier-sensitive; the audit
does not estimate general utility from one collapsed checkpoint.

The measured downstream work deltas give a sharper route hypothesis. A
layer-2 pair added on average 4.24 L4 spikes and 10.76 L4 readout updates per
example; layer 3 added 1.79 and 4.04; layer 4 added 0.23 and 0.45. Layer-1
pairs added 0.78 and 1.83. Layer-2 averages conceal mostly zero late-layer
work changes plus a small number of large cascades. Next, test a matched
late-layer-only pair sampler (last half of the stack) to see whether it
retains useful deep alternatives while avoiding the early-layer cascade.
This uses validation utility to form a training hypothesis; the same
validation results are development evidence, not final test evidence.

Reconstructing the clipped four-outcome derivative from those same paired
losses and route scores ($\sigma=0.25$) gives mean $\partial L/\partial s$
of +0.291, +0.517, -0.013, -0.114 per route in layers 1–4. Gradient descent
would close routes for positive derivatives and open routes for negative
ones; negative-derivative fractions were 14.1%, 35.9%, 60.5%, 50.0%.
Thus equal shadow allocation is not equal useful credit: early proposals
mostly say “close,” while the late-layer signal is more favorable but weak
and noisy. This is one checkpoint's validation-conditioned calculation,
not a general layer ranking.

**Matched late-layer pair run and second-order proposal starvation (§144).**
The D4 seed-6 late-balanced arm changed only pair selection, restricting one
pair shadow per minibatch to L3/L4. It produced 9 deep pair shadows in epoch
1, then none in epochs 2–4. The per-epoch candidate-pair counts by layer were
`[193140, 3, 6, 18]`, `[188476, 0, 0, 0]`, `[180439, 0, 0, 0]`, and
`[180459, 0, 0, 0]`. L4 validation support was not literally zero in every
later epoch, so the sharper explanation is pair co-occupancy failure: no two
closed, near-time routes shared a deep receiver, despite occasional deep
events. Final anytime accuracy was 6/128 (4.69%), versus 8/128 (6.25%) for
the matched control. This is not evidence that late alternatives are harmful;
the treatment had no deep pair proposals for three of four epochs.
Training loss decreased 59.78→3.07, but held-out prefix NLL was 2.985/3.167
in the two time strata and accuracy stayed near the 5% chance rate.

For each minibatch/receiver group `g`, let `n_g` be the number of eligible
closed routes before the code filters adjacent arrivals by time gap and
distinct source event. The exact pair count is bounded by
`sum_g (n_g - 1)_+`. Under a sparse Poisson occupancy model, this bound's
expected value and the probability of at least two alternatives both scale
as `lambda^2/2`, while single-route availability scales as `lambda`. Thus
higher-order counterfactuals disappear faster than first-order credit as
event flux thins. Importance weighting corrects
sampling among existing pairs; it cannot repair a zero candidate set. The
frozen time-window sweep tested 25, 50, 100, 250, 500, and 1,000 ms on the
same 120-example fit subset. Pair counts at 1,000 ms were `[189987, 6, 0, 1]`
for L1–L4, compared with `[181577, 1, 0, 0]` at 25 ms. Even across a full
second there were no L3 pairs, only one L4 pair, and L2 pairs in just 6/30
batches. Near-closed route records themselves were `[191787, 111, 3, 13]` by
layer. This rules out the 25 ms cutoff as the main cause; source-route
co-occupancy is the bottleneck. Widening the window measures candidate count,
not counterfactual usefulness, since widely separated arrivals may not
interact under the receiver kernel. The next design must preserve first-order
shadows, instrument co-occupancy by layer, and use sparse prefix-expansion
replays to create downstream counterfactual events when justified. That
estimator is not implemented yet.

**Refractory-aware spike-boundary audit.** A matched no-pair control and the
late-only checkpoint were evaluated on the same 128 held-out-speaker examples
(32 batches). Each batch/layer supplied its closest spike toggle; local
boundary candidates were then restricted to margin $\le0.25$ and
nonrefractory state. Valid candidate counts by layer were `[22, 21, 8, 0]`
for control and `[21, 19, 2, 1]` for late-only. Separating the fused main loss
from the weighted auxiliary losses, the L1 main-loss toggle helped 12/22
control candidates (mean $+0.0266$) and 16/21 late-only candidates (mean
$-0.0094$, median $-0.0020$); the auxiliary term has the same direction.
Only 13 batches had a valid L1 candidate in both arms; the mean difference
between the two arm-specific selected spike-on utilities was $-0.037$ (SE
$0.035$), and the selected unit/time can differ by checkpoint. This is
suggestive rather than a reliable treatment effect. L2 is not a robust opening signal: 12/19
late-only candidates helped, yet mean main-loss change was $+0.0040$ (median
$-0.00063$), with large variance. L3/L4 have only 2/1 valid late-only
candidates. The earlier all-candidate averages were misleading because they
included forced spikes outside the margin band. A matched deepest-only replay
of these same toggles found exactly zero L1/L2 main-loss changes, although
late-only L1's all-depth mean was $-0.00936$. The L1 intervention added no
downstream hidden spikes and changed its own readout edges by 1.238/example.
Thus the apparent L1 utility is a shallow-head shortcut, not a deep credit
signal. The next test should use a deepest-only primary objective and/or a
work-capped sparse suffix-expansion replay that demonstrates downstream event
changes; neither remedy has yet been trained.

**E83 receiver-bundle route-credit screen (§142).** This matched D4 seed-6
comparison used the same initialization, 120 training examples, four epochs
(120 optimizer updates), split RNG streams, and 128 held-out examples; both
arms retained the one-route-per-layer counterfactual update. The treatment
added one receiver-bundle pair shadow per minibatch, costing three replays.
Terminal accuracy ended at 8/128 (6.25%) for control and 14/128 (10.94%) for
the pair arm. For the actual anytime prediction with terminal fallback, the
paired counts were 8 versus 14 correct, with 4 control-only and 10 pair-only
correct examples (exact McNemar $p=0.180$). This is a small exploratory
direction, not evidence of a reliable accuracy gain.

The main objective issue remains visible: control training loss fell from
1752.8 to 3.01, nearly the uninformative 20-class value $\ln20=2.996$; held-
out late-prefix NLL finished at 3.246. The pair arm ended at train loss 3.23,
late-prefix NLL 3.916, and 3.9% layer-4 support versus 1.6% for control. Both
deep supports collapsed over training. The pair arm temporarily retained
51.6% layer-4 support at epoch 2 versus 11.7% in control, but this did not
persist or improve that epoch's accuracy (3.9% versus 9.4%). A falling scalar
loss therefore did not mean that the classifier learned the task.

The global proposal policy selected 30 pairs per epoch from 197k–206k
eligible adjacent candidates, and all 120 selected pairs came from layer 1.
Only 7/120 joint openings reduced the matched loss and 8/120 had negative
interaction $\Gamma$; the first epoch accounted for most of the positive mean
interaction. The policy paired closed scores in $[-0.5,0)$ by batch/receiver,
kept adjacent source-time pairs within 25 ms, then sampled globally. This was
not a threshold-margin or vector-compatibility selector. The result shows
that generic temporal proximity produces little useful pair credit and that
global candidate counts starve deeper layers. It does not reject cooperative
routes; it identifies the proposal policy as the next variable to isolate.

The follow-up keeps one pair per minibatch and changes only selection:
choose uniformly among eligible layers, then uniformly among that layer's
pairs, while logging layerwise candidate counts and inclusion probabilities.
The route formalism also now distinguishes the hard deterministic edge gate
from its logistic training relaxation and records two additional architectural
limitations: the current receiver query is input-independent, and the same
score controls both gate admission and positive delay. The four-outcome
gradient is exact for its stated independent-gate surrogate before clipping;
the proposal sampler, clipped contrasts, and separate local SGD step are
explicitly identified as bounded learning heuristics. Alternate delay/value
policies and receiver-state-conditioned queries remain untested.

**E83 route-credit theory and compute-matched screen (§§138–141).** The
analysis quantifies the pathwise-gradient penalty from sparse deep support:
if a sample contributes gradient $X$ only when layer $\ell$ is active, then
$G=A X$ has mean $c_\ell\mu_\ell$ and covariance
$c_\ell\Sigma_\ell+c_\ell(1-c_\ell)\mu_\ell\mu_\ell^\top$. For IID minibatches,
the probability of no active example is $(1-c_\ell)^B$; with batch size four
and measured layer-4 coverage 1.56–7.03%, this is 75–94%. Under
conditional-gradient-noise dominance, task-direction SNR scales as
$\sqrt{B c_\ell}$. This is exact under the stated sampling assumptions, but
does not establish that active events carry useful class information.

The compute-matched D4 comparison held architecture, initialization, and 120
optimizer updates fixed, with nested 120-example/four-epoch versus
480-example/one-epoch subsets and the same 128 evaluation examples within
each seed. Seed 6 favored the larger subset (20/128 vs 8/128; paired exact
McNemar $p=0.0227$); seed 7 reversed direction (8/128 vs 20/128;
$p=0.0357$). Layer-4 support remained 1.56–7.03%, and late-prefix NLL was
3.02–6.58, above uniform $\log20=2.996$. The data-diversity effect is not
reproducible in these two runs and did not repair deep support or posterior
quality. The optimizer-induced cascade hypothesis also remains unverified:
global gradient clipping does not bound AdamW's gate-margin movement, and
per-update margin traces have not been collected.

Sections 139–140 separate four objects that had been conflated: the fixed
candidate mask, the event-conditioned graph, the realized route-and-fire
graph, and the graph receiving loss credit. Exact reconstruction of the
seed-6/7 D4 masks gives 90.2%/98.0% first-to-fourth unit-pair reachability;
all 140 input bands reach layer 4. Yet only 1.56–7.03% of examples actually
reach it in the matched arms. Sparse layer-1 skips raise support to 99–100%
without paired recognition improvement. Static disconnection is therefore
not the broad failure; useful events and label credit still fail downstream.

The route calculation makes the sparse-MoE analogy precise: a competing
route's relaxed gradient depends on the loss difference between paired route
outcomes. Current E83 shadows test one near-boundary route at a time, so they
miss a possible cooperative crossing where two individually subthreshold
messages jointly create a useful spike. A two-gate derivation identifies the
needed four replays (00/10/01/11) and the interaction loss. This is a
testable mechanism, not yet an established cause. Section 141 generalizes the
idea to dormant proposals: record why a candidate did not fire, replay a
sparse set through the full downstream state, and credit its local margin
only when the paired loss says the alternative helped. Route closure,
threshold failure, race loss, and refractory blocking need distinct margins;
absence by itself is not a negative label.

Finally, §140 shows why depth can add useful choices but cannot be increased
blindly: candidate paths grow combinatorially while strict-chain sample
support multiplies by conditional survival at every transition. Preserving
half of full input support at depth 8 or 16 requires average per-transition
survival of at least 0.906 or 0.955. These results motivate a fixed-checkpoint,
margin-stratified dormant-event audit before another depth sweep; they are
not evidence of supremacy or of route-pair synergy in the current model.

**E83 all-depth D4 control and strict-chain support invariant (§137).** The
matched four-epoch depth-4 `all_depths` run used 128 train / 32 held-out
speakers, seed 6, and the same event-prefix objective and route shadows as its
queued deepest-only control. Held-out spikes per utterance rounded by layer
were `[14, 6, 5, 28]`, `[16, 2, 1, 0]`, `[9, 1, 0, 0]`, and `[10, 1, 0, 0]`.
Terminal accuracy was 6.25%, 3.125%, 6.25%, and 6.25%; epoch-4 prefix NLL was
`[3.1373, 3.4115]`, above the uniform 20-class NLL of 2.996. On the first
training minibatches of epochs 3 and 4, main-loss gradient norms in layers 3
and 4 were exactly zero. These are first-minibatch gradient measurements,
not claims that every update in those epochs was zero.

The matched deepest-only arm ended at 5/32 (15.6%) versus 2/32 (6.25%) for
`all_depths`. This is a first above-chance-sized endpoint in the matched D4
screen, but it is not strong evidence yet: on the shared examples there were
five cases correct only for deepest-only and two correct only for fusion
(exact two-sided McNemar p=0.453). The deepest-only prefix NLLs were
`[4.0081, 4.7412]`, worse than uniform NLL 2.996, and accuracy moved
9.4% → 0% → 3.1% → 15.6% across its four epochs. The nominal one-sided
binomial tail for 5/32 at 5% chance is 0.020, before accounting for the
multiple arms/epochs inspected. Larger held-out evaluation is needed.

That larger readout comparison is now complete for seed 6, with 128 examples
in both the train and held-out-speaker subsets. The `all_depths` and
`deepest` arms used the same seed, selected examples, and training random
stream; only the readout fusion setting differed. At the fixed fourth-epoch
endpoint, terminal accuracy was 17/128 (13.3%) for `all_depths` and 7/128
(5.5%) for `deepest`. Layer-4 active-example coverage was 61.7% versus 4.7%,
respectively; layer-2 coverage was 100% versus 46.1%. For the actual
thresholded output policy (first crossing, otherwise terminal fallback), the
paired predictions were correct on 20/128 versus 7/128 examples, with 19
all-depth-only and 6 deepest-only correct cases (exact McNemar p=0.0146).
This is a nominal single-seed, two-speaker result, not a speaker-general
claim. The fixed-threshold race emitted on 42/128 all-depth examples, 10
correct (23.8% accuracy among emissions); their mean reported confidence was
63.8%. Deepest-only emitted on 2/128 and neither was correct. All-depth
late-prefix NLL was 17.57, while deepest-only
was 4.66, both above uniform 2.996. The objective improves class decisions
and preserves deep support in this pair, but its accumulated evidence is
poorly calibrated. Removing the layer-4 branch left all-depth terminal
accuracy unchanged at 13.3%, so this result does not establish that the
deepest branch contributes the accuracy gain.
At this endpoint, standalone head accuracies were `[7.0, 10.2, 8.6, 10.2]%`;
removing each head in turn from the fused terminal logits left
`[10.2, 7.0, 10.2, 13.3]%`. The second-layer head has the largest measured
leave-one-out effect, while layer 4 has none. These are readout-head
ablations on one trained checkpoint, not retrained depth ablations.

The seed-7 matched replication narrows the claim. At the fixed endpoint,
`all_depths` reached 9/128 terminal accuracy versus 7/128 for `deepest`; its
race-plus-fallback outputs were correct on 9 versus 6 examples, with 9
all-depth-only and 6 deepest-only cases (exact McNemar p=0.607). Layer-4
coverage was 22.7% versus 16.4%, a smaller difference than seed 6. The
all-depth race emitted 8 answers with mean confidence 63.1%, none correct;
late-prefix NLL was 5.74 versus 4.09 for deepest-only. The seed-6 paired
accuracy advantage therefore did not replicate, while the calibration failure
did. Seed-7 leave-one-head-out fused accuracies were
`[2.3, 7.8, 10.9, 7.8]%` (omitting layers 1–4) versus 7.0% with all heads;
removing layer 4 slightly improved the endpoint. The layer contribution
pattern varies by seed, with no stable deepest-layer gain.

The seed-7 `deepest` replication has also finished: terminal accuracy was
7/128 (5.5%), layer-4 coverage 16.4%, late-prefix NLL 4.09, and the fixed
0.6 race emitted four answers with none correct. Thus deepest-only remains
near chance on a second training/evaluation subset, with the same two held-out
speakers. Its matched seed-7 `all_depths` arm completed; it did not reproduce
the seed-6 paired advantage.

The seed-6 deepest-only skip ablation has finished. A sparse layer-1 event
skip raised layer-4 coverage from 4.7% to 99.2% and broke strict support
nesting as intended (29 measured nesting violations); mean deep candidate
scores rose only 1.8% (46,246 to 47,096). Terminal accuracy moved from 7/128
to 9/128, while late-prefix NLL improved from 4.66 to 3.71, still above
uniform 2.996. At the fixed 0.6 threshold the skip emitted five times and
none were correct. The paired race-plus-fallback comparison had 6 strict-only
and 8 skip-only correct cases (exact McNemar p=0.791). The skip fixes the
intermediate support bottleneck at small sparse fan-out cost, but it does not
yet produce useful class evidence or a reliable stopping signal. The 1.8%
figure counts candidate event–receiver pairs; the simulator still performs
288,008 vector-state updates per utterance on a 1 ms grid, so this is not an
end-to-end energy measurement.

The seed-7 skip replication reached 100% layer-4 coverage but only 5/128
terminal accuracy, late-prefix NLL 48.80, and zero correct answers among
seven fixed-threshold emissions. Total candidate-score work was 22.3% above
its strict-chain control; paired race-plus-fallback accuracy was 5 versus 6
correct (6 strict-only, 5 skip-only; exact McNemar p=1.0). The skip therefore
restores support in both seeds without a paired classification gain, while
its event activity and evidence scale vary sharply by seed. Support survival,
event-rate control, and class credit are separate requirements.

The analysis yields an exact structural invariant for this model: if a strict
chain layer receives no events, its zero state and positive threshold produce
no output events. Per-utterance active-example support is therefore nested
with depth; `all_depths` changes the loss paths but cannot make a silent hidden
layer receive input. Mean event count and active-example support are distinct:
event multiplicity can cascade on a shrinking active subset. This clarifies
how the D8 count cascade can coexist with deep support loss. The count-mark
arm kept layer-4 coverage at 56–94% across its four epochs and produced zero
support-nesting violations, yet accuracy fell to 0/32 after epoch 1 and
epoch-4 prefix NLL was `[8.4786, 27.28]`. Thus count input changed deep
dynamics but did not create class evidence in this seed. Added held-out
coverage and spikes-per-active-utterance fields, plus a support-nesting
violation counter. The seed-7 deepest-only and both sparse layer-1-skip
controls are complete. The skip restored layer-4 support in each seed but had
no paired accuracy gain; its seed-7 arm also had unstable late-prefix loss.
The skip masks use a separate topology RNG
so the matched adjacent-layer masks remain unchanged.
E83 now defaults to `--rng_protocol split`, which separates evaluation
selection, training subset/order, augmentation, prefix sampling, and
route-shadow sampling. Existing results used the prior shared-stream behavior;
pass `--rng_protocol legacy_shared` to reproduce it. The split implementation
is not yet validated with a paired run across evaluation limits. Use split
streams before comparing different evaluation sizes. Do not add a hidden-spike
boundary update until its paired loss deltas show useful class credit.

## 2026-09-28

**E77 1M depth-8 benchmark resource screen.** The parameter-matched 8-layer
Transformer control completed 4,882 updates on 1M text8 training characters
(4,999,168 token positions), with 1.251M parameters. Its best validation BPC
was 2.322 at update 4,881; the frozen 1M-character test BPC was 2.352. It ran
for 1,617 seconds at about 3,507 token positions/second and stayed near 1.2 GB
RSS. The matching E77 depth-8, width-128 job at 1M characters, five passes,
batch 4 and 256-character windows was stopped by `run_safe.sh` before training
when its process group reached 3.834 GB, above the 3.5 GB cap. `MemAvailable`
remained above 10 GB; the host was not at risk. **Learning:** the guard caught
a batch/activation configuration that was too large before it could become a
host-wide failure. The 1M E77 job will not be retried. A matched 100k-character
screen now uses batch 2 and 128-character windows for both E77 and Transformer;
depth 16 is deferred until that runtime is measured.

**E77 exact post-reset activity matching and depth-8 gradient reach (§§133–134).**
The raw voltage-quantile estimate failed at the default E77 width: on a matched
depth-4, 10k-character, seed-77 run it yielded selected-checkpoint test rates
$[0,0.007,0.009,0.004]$ spikes/character. Only $[1,9,12,6]$ of 13 validation
checkpoints had nonzero gradients by layer. We changed calibration to save
pre-reset voltage traces and replay the exact refractory/reset recurrence,
then search threshold against the realized event count without rerunning the
whole network. At initialization, rates matched the 0.1 target to
$[0.096,0.082,0.094,0.100]$; all four layers had gradients at all 13
checkpoints. Test rates rose to $[0.204,0.190,0.456,0.802]$ as the
representation changed. Quantile-only and exact-rate runs scored 3.765 and
3.824 BPC, respectively, on only 1k test characters; this is not a quality
comparison. It is evidence that the initialization statistic, not the
language-model loss, caused the earlier silent stack.

The exact-rate procedure also carried a width-8 model to depth 8. With 4,096
training characters, 16 updates, and one seed, all eight layers had nonzero
gradients at each of 16 validation points. Initial rates were
$[0.102,0.086,0.109,0.102,0.086,0.102,0.098,0.102]$ and selected-checkpoint
test rates were $[0.115,0.125,0.217,0.075,0.081,0.138,0.124,0.121]$.
The 4-layer control under the same tiny setup had nonzero gradients on
$[16,16,16,12]$ validation points. Test BPC was 4.319 at depth 8 and 4.254 at
depth 4, each on 512 characters; neither supports a quality or depth
comparison. **New learning:** the mechanism that reopened the depth-4 path
also reaches depth 8 while keeping event activity sparse. This is a concrete
trainability result, not scaling or supremacy evidence.

A fresh depth-4 route-shadow pass sampled 64 near-boundary route openings per
layer. The minibatch-trajectory means and cluster SEs for
$L_{open}-L_{closed}$ were $[+0.000360\pm0.000218,-0.001115\pm0.000325,
+0.000220\pm0.000278,-0.000435\pm0.000290]$. Layer 2's openings tended to
help, but adjacent updates share a moving model, so these are not independent
replicates. Across the run, the counterfactual/pathwise norm ratio averaged
0.77% and cosine 0.0040; the optimizer correction remains disabled. This
supports continued layer/condition-stratified measurement, not a larger gain.

The new local Bayesian derivation (§134) formalizes early evidence trust: for
a locally linear conditional, posterior mean gain scales with posterior
covariance, input-state novelty, and observation noise. Repeated observations
shrink gain only in the state directions actually observed. Activity alone
calibrates event scale, not task relevance; output credit is still required.
Representation drift requires discounted precision or reset. No
posterior-driven synaptic gain or online threshold homeostasis is implemented
yet. All jobs ran through the memory-guarded runner; one first attempt was
stopped at its RSS cap before the calibration search was optimized.

**E77 depth-4 event bootstrap and route-credit diagnostic (§133).** A matched
16-update, seed-77 text8 micro-pilot exposed a trainability blocker: with the
fixed threshold $\theta=1$, test activity was [0.018, 0, 0, 0] events/character
across four layers, and only layer 1 had any nonzero gradient (5/16 updates).
The deeper stack was effectively absent from learning. We added a
label-free, depth-ordered threshold calibration from empirical pre-reset
voltage quantiles, targeting 0.1 spikes/character/layer. The matched calibrated
run set thresholds [0.664, 0.229, 0.429, 0.283], produced [0.065, 0.052,
0.044, 0.035] test events/character, and had nonzero per-layer gradients on
15/16, 16/16, 14/16, and 16/16 updates. This is direct evidence that the
previous initialization silenced the deep event stack and that a measured
activity scale can restore gradient paths. The tiny run (4,096 training
characters, 512 validation/test characters, width 8, 16 updates, one seed)
does not establish language-model quality or depth scaling: test BPC was
4.341 fixed versus 4.284 calibrated, too little data for a performance claim.

The calibrated model also received 64 near-boundary route shadows per layer
(256 total). Minibatch-clustered layer means of $L_{open}-L_{closed}$ and
standard errors were: layer 1 −0.000034 ± 0.000237; layer 2 −0.000440 ±
0.000236; layer 3 −0.000045 ± 0.000103; layer 4 −0.000092 ± 0.000477. Every
approximate 95% interval includes zero. Only 3.1–15.6% of sampled openings
helped by layer; the global counterfactual/pathwise norm ratio averaged
0.61%, with cosine −0.0019. A weak prior should therefore remain broad, and
the optimizer still applies no counterfactual update. This is a posterior
uncertainty result, not a reason to turn up the gain. E77's token-level query/
key module still materializes a dense causal $L\times L$ score matrix; this
experiment measures deep hidden-route trainability, not sparse retrieval cost.
All runs used the memory-guarded queue.

**E83 causal prefix supervision and route-gradient statistics (§§130–132).** The per-utterance normalized-time sampler was future-dependent: it used each utterance's last event to choose a prefix. E83 now samples a fixed, shared 0–1000 ms physical-time window, stratified into exogenous query strata. Under log loss, the population optimum at each query is the causal posterior $P(Y\mid\mathcal F_t)$; choosing query time using future duration would instead reweight labels by duration. The code also fixed a PyTorch autograd failure in grouping equal event times (`unique_consecutive` had no derivative), replacing it with detached group IDs and differentiable group-time means. This makes the prefix-time path differentiable; it does not remove E83's 1 ms hidden-state scan.

The exact route boundary term from §§19/57 is now measured by toggling one near-boundary route, rerunning the whole downstream stack, and comparing factual and shadow prefix loss at identical sampled query times. A first Horvitz–Thompson implementation multiplied each sampled route by candidate-count/sample-count. At depth 4, 128 train / 32 held-out, two epochs and seed 6, this drew 128 shadows per epoch from about 480k eligible routes. Train loss rose 1,974 → 17,044, prefix NLL and layer-4 activity exploded, so that estimator is too noisy at this budget. The fixed-population formula in §132 explains the scale: total-estimator covariance has the factor $N^2(1-m/N)/m$.

A bounded normalized local rule now averages sampled boundary terms within each layer, clips $L_{on}-L_{off}$ to ±5, clips the global correction-gradient norm to 1, and applies a separate SGD step of 0.001. The matched seed-6 pathwise-only run stayed near chance: terminal accuracy 6.25% (2/32), with held-out prefix NLL by stratum [2.995, 3.202] in epoch 2. The local-counterfactual run also ended at 6.25%, but its epoch-2 NLL was [2.996, 19.735], so this pilot did not improve the posterior and showed severe late-window overconfidence. Race coverage was 3.12% with no correct emitted answer. It did not reproduce the HT loss explosion, but neither did it establish stable learning.

The route statistics explain why a global gain is unjustified. In epoch 1, 121 shadows had mean signed open-minus-closed loss +0.172, SD 1.176, and only 14.9% showed that opening helped. Per-layer mean deltas were [+0.051, +0.344, +0.262, −0.010]; layer 4 was less unfavorable (36% helpful), but its mean remained small relative to its SD. Epoch 2's overall mean was +0.0063 (SD 0.0644; helpful fraction 11.6%). The counterfactual/pathwise gradient cosine was 0.017 then 0.0038. Raw gradient norm ratios were 4.47% then 1.64%, but the applied clipped update norms were only 4.5e−4 and 1.5e−4 per batch. These results call for uncertainty-aware, layer-specific analysis: a weak prior at initialization should not suppress route evidence, but the sampled evidence itself remains noisy and does not support a larger update. §132 derives a hierarchical posterior/shrinkage diagnostic and separates it from trust-region step-size control.

A separate 80/40 depth-2 smoke reinforces the scale issue: mean absolute shadow delta 2.6e−4, SD 1.28e−3, 10% helpful openings, gradient norm ratio 1.17e−5, and cosine −0.002. The bounded update was nearly negligible there. Both pilots remain at chance-scale accuracy; use them to diagnose credit magnitude, not to claim trainability. The guarded depth-4 run peaked below 0.6 GB RSS with host available memory above 11.7 GB. Inspection also found that payload-agreement accounting included fallback cases in an emitted-only denominator; the E83 counter is fixed, and the saved result summary was corrected from its per-sample output vectors.

**Deep sparse-stack stability and full-sequence attention bound (§§113–114; derivation plus E114 diagnostic).** Section 113 composes local approximation and Jacobian error through residual depth; under residual scale $1/L$ and bounded per-layer constants, these bounds do not grow exponentially, but they do not show that the task gradient is useful. Section 114 lifts fixed-support softmax truncation to the full sequence Jacobian, with operator norm at most $\sqrt{RC}$ and a shared-key column term containing $F_j$. E114 checked 144 fixed-support synthetic cases using central finite differences at lengths 3, 5, and 8, with diffuse/reused keys, several retained supports, and three score temperatures. No absolute violation exceeded $10^{-8}$. Among the 90 cases with bounds at least $10^{-8}$, the largest Jacobian and forward-error ratios were 0.897 and 0.687; the largest positive absolute excesses were $1.32\times10^{-10}$ and $7.4\times10^{-16}$. Ratios for the all-keys-retained cases are omitted because the exact bound is zero and finite-difference residue makes the ratio meaningless. This supports the formulas on small synthetic inputs; it is not an architecture-scale result. Uniform state-region bounds, sparse Jacobian Lipschitz constants, parameter-VJP error, learned support changes, and task-gradient alignment remain open.

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

**Attention sparsification and counterfactual credit (§112; mathematical result, not yet an experiment).** Conditioning dense softmax on a fixed candidate set gives an exact query-gradient residual: omitted-group covariance plus a between-group covariance between key direction and value advantage. Its norm is at most β ε (3/2 − ε) D_K D_A. Mass recall therefore controls absolute gradient error but not relative error: in the two-key example, 99.9% retained mass still leaves the sparse query gradient zero while the omitted key carries the dense gradient. A new H-smooth extension closes the nonlinear-loss gap: the full-loss query-VJP error is bounded by β ε D_K[(3/2 − ε)D_A^C + H D_V²/4], with corresponding key- and value-VJP bounds. The H term is the upstream-gradient change induced by the retrieval error, not just the local support mismatch. This remains an absolute, fixed-support bound and cannot promise alignment when the true gradient is small. Adding a missing key changes the output to (1 − r)y + rv; the finite loss change differs from r·gᵀ(v − y) by at most (H/2)r²‖v − y‖². This connects the already-derived attention score credit (§105/§107) to the existing sparse-expert counterfactual route rule (§19/§57). Substituting this estimate into the existing route-boundary gradient bounds its error by (H D_V² / 2) Σ ρσ(m) r² ‖∇θ(b_u − b_c)‖, giving a principled way to allocate exact shadow runs to near-miss keys. Next compare exact dense and truncated autodiff VJPs against the new bounds while varying support mass and downstream curvature, then compare bound-based shadow allocation with fixed top-m at equal shadow work. No such attention-index experiment has run yet.

**E64b gradient language-model baselines.** The validation-selected 1M, 20-pass LSTM completed at 2.1385 valid / 2.1794 test bpc (338,395 parameters); the 2-layer width-256 Transformer completed at 2.3403 / 2.3667 (1,658,907 parameters). Both best checkpoints are at their final validation point. At 10M, the 2-layer width-512 LSTM completed six passes at 1.7448 validation / 1.7993 test bpc (1,199,323 parameters). The 4-layer width-256 Transformer then completed all 4,882 updates in four passes (batch 32; 3,238,427 parameters): best validation BPC 1.864707 at step 4,880 and held-out test BPC 1.908253. It is 0.1199 higher on validation and 0.1090 higher on test than the LSTM. This is a same-data comparison, not a capacity- or training-budget-matched comparison; each run uses one seed, and the LSTM's best validation point is its last checkpoint.

**90M E79 queue state.** The native race-mixture 90M result already exists at `experiments/results/e79/race_mixer_D90000000_K7_e77none.json`; the queue runner lost its successful-completion marker. The duplicate E79 90M queue line is now commented out so a resumed chain will not repeat it. This was not a 90M Transformer run. The Transformer currently advancing is the 10M, four-layer E64b baseline.

**E79 at 10M versus the completed LSTM.** On the same 10M-character training prefix and 1M-character test segment, the frozen E79 race mixture scores 1.6130 test bpc versus 1.7993 for the LSTM: a 0.1863 bpc lead. E79 uses six native experts plus its copy memory; the LSTM has 1,199,323 parameters. This is a single-seed, same-data result, not a compute-, parameter-, or training-budget-matched architecture comparison. The completed 10M four-layer Transformer scores 1.9083 test bpc on that segment; E79 is 0.2953 lower, but uses six experts plus copy memory and is not matched for parameters, training, or compute.

**E64b LSTM versus the E79 race mixture at 1M and 10M text8 characters.** The validation-selected 256-hidden-unit LSTM used 20 passes and early
stopping on 200k validation characters: 2.138 validation / 2.179 test bits per character, 338,395 parameters. E79's frozen race
mixture on the same 1M-character training and test segments scores 1.808 with its copy expert limited to the Transformer's 256-character
context (1.782 when weights also adapt online). The frozen mixture is 0.371 bpc below the LSTM and 0.559 below the two-layer Transformer;
the online-adapted score is 0.397 below the LSTM. This is the clearest current real-language performance lead for the native experts.
It is not yet a compute/energy-matched architecture comparison: its count tables and copy index are extra state. It also does not test
the deep E77 time-vector language model. At 10M, the frozen race mixture scores 1.613 test bpc against 1.799 for the completed
two-layer width-512 LSTM on the same test segment. That 0.186 bpc lead is also single-seed and lacks matched compute or parameter
accounting; the four-layer Transformer trace has no endpoint or test result (last logged at 90%; the serialized lock remains held, though its process is not visible in the current process listing). **Next:** prioritize the deep
market and speech event models, then return to additional language baselines only if they answer a specific unresolved comparison.
Report memory, scoring work, and adaptation policy alongside loss before making a scaling claim.

**E83 — sequence objective audit and depth-4 screening controls.** SHD supplies one class label per spoken utterance. The original E83 implementation trained one loss per utterance, but averaged softmax(V) across the entire shared simulation grid. Empty prefixes and the artificial delay tail contributed uniform class guesses; a short utterance's output could therefore change when batched with a longer one. The completed depth-2 jobs are retained as debugging data but excluded from trainability claims.

E83 now compares a causal confidence race against two established utterance readouts. The race stays silent until the first temperature-scaled class probability crosses a threshold; only that first class is emitted. Its train loss scores the probability that the correct class wins the event race, includes survival (no output yet) before a winner, and discounts late correct outputs. It does not train against the final class at every prefix. Per-utterance endpoint masks prevent batch padding from changing the objective. Cramer et al. used max-over-time readout potentials for their SHD SNN experiments and reported better results than last-step loss; Spyx uses cross-entropy on integrated readout potentials ([Cramer et al.](https://kip.uni-heidelberg.de/Veroeffentlichungen/download.php/6616/temp/4143-3.pdf), [Spyx SHD tutorial](https://spyx.readthedocs.io/en/latest/examples/surrogate_gradient/SurrogateGradientTutorial/)).

The first race prototype failed: at 80 train / 40 held-out examples for one epoch, it emitted for every item and reached 2.5% emitted accuracy. A revised shared confidence gate and temperature-scaled class race reached 92.5% coverage but only 10.8% accuracy among emitted answers, with 2.5% max-over-time accuracy. These tiny one-epoch runs are smoke diagnostics, not model comparisons. A guarded, matched depth-4 screening run compared two-epoch objectives on 512 training and 128 held-out-speaker examples. Integral pooling ended at 4.69% max-over-time accuracy; terminal fallback reached 5.47%, race coverage was 17.97%, and emitted accuracy 4.35%. Max pooling ended at 4.69% max-over-time accuracy; at the default race threshold its fallback accuracy was 3.91%. The `anytime` race-plus-fallback objective reached 6.25% max-over-time accuracy (8/128) and 5.47% emitted accuracy at 100% coverage; all thresholds from 0.3 through 0.9 still emitted every item, with peak confidence 1.0. Layer 4 firing increased from 932 to 1,024 spikes/utterance. Under a 5% chance model, 8/128 has a 0.31 upper-tail probability and a 3.2–11.8% Wilson interval, so this is not reliable evidence above chance; the decisive signal is false confidence, not early useful recognition. The matched stable-cause race-only control also finished near chance: 3.12% max accuracy, 97.66% coverage, 4.8% emitted accuracy, with mean peak confidence 0.981. A readout-only shadow probe at seed-2 initialization (not trained weights) forced 32 near-gate routes on four held-out-speaker utterances. 31/32 changed max-pooled CE by exactly zero; one reduced it by 0.045. The estimated counterfactual-gradient norm was 2.2% of pathwise norm, cosine 0.012. This small mechanism probe is consistent with a max-pooling winner-gap dead zone (§129), but says nothing conclusive about hidden route births or trained checkpoints. All jobs stayed under the one-process memory-guarded runner. E83 stores per-sample vector outputs and final-layer firing rasters for separate visualization.

**Anytime stream-to-class theory (§§122–§128).** With a true posterior at every prefix, emitting at the first crossing of $1-\epsilon$ bounds the error among emitted answers by $\epsilon$; ordinary full-recording calibration does not establish this, so first-crossing calibration must be measured. For a marked point process, posterior evidence consists of both event log-likelihood jumps and survival evidence from intervals with no event. If each message changes only $r$ class logits, indexed max and log-sum-exp trees can maintain the exact confidence threshold in $O(r\log C)$ updates, with $O(C)$ paid only when materializing a full class-probability payload. A portable reference head is added at `experiments/sparse_anytime_readout.py`; it is not yet connected to E83 or benchmarked. Section 128 separates posterior estimation from the stopping policy: sampled-prefix log loss is proper for $P(Y\mid E_{\le t})$ even with only one utterance label, while the race objective alone does not identify calibrated prefix probabilities. This applies to SHD and event-camera clips, which share one-label-per-stream supervision but differ in event marks, rates, and nuisance variation.

**E83 gradient-graph and readout audit (§§127, 129).** The fixed-support controls all share a hard routing implementation: `TVLayer` removes nonpositive message scores using `keep = r.detach() > 0`, and computes spike identities in `no_grad`. Thus closed routes receive exactly zero task gradient; open routes can receive pathwise credit through their delay and payload; firing-time refinement differentiates a spike only after it has fired. Intermediate readouts improve credit for realized events but do not create the missing route-birth/firing boundary term. The route masks are also fixed random/tonotopic buffers, so topology itself cannot be recruited. A subsequent readout-only shadow probe forced 32 near-gate final routes at seed-2 initialization, on four held-out-speaker utterances. 31 interventions changed terminal max-pooled CE by exactly zero; the one nonzero intervention reduced it by 0.045. This is consistent with the max-pooling winner-gap dead zone derived in §129: a route can alter its local trace while remaining below the current per-class temporal maximum, leaving terminal scores and loss unchanged. The measured boundary-gradient norm was 2.2% of pathwise norm, cosine 0.012. Because this probe used untrained weights and no hidden spike insertions, it is only a mechanistic clue. **Next discriminating work:** save a trained checkpoint, shadow near-threshold hidden spikes through the remaining layers, and compare max, integral, and smooth-max posterior heads under the same event support and sampled-prefix proper log loss. Report exact loss differences, winner gaps, and gradient alignment before adding counterfactual credit. Use a fixed noise band and shadow budget. No more objective sweep answers this mechanism question.

**E83 depth-four failure mechanism, spike-boundary audit, and depth-eight rate profile (§137).** The seed-6 128/32 pathwise event-prefix run reports held-out spikes per utterance rounded to `[4, 2, 0, 0]`; layers 3–4 average below 0.5 events per utterance. On epoch 2's first minibatch, the main-loss gradient norms were exactly zero for every hidden layer, while the auxiliary gradient norms for layers 3–4 were also zero. This confirms that the hard support cuts label credit on silent deep-path batches. The normalized route-shadow run had more held-out activity `[21, 15, 4, 14]` but still 6.25% terminal accuracy and late-prefix NLL 19.735: routing credit alone did not solve recognition. A paired hidden-spike audit on 128 held-out utterances found near-threshold margin candidates (within ±0.25) averaging 349/batch, 67/batch, 7.9/batch, and 6.6/batch across the four layers; layer 4 had none in 24/32 batches. Spike-on improved the loss in only 16/32, 17/32, 14/32, and 14/32 interventions, with near-zero mean effects. This confirms scarce deep support but does not justify a single-spike update. A separate input audit found that E83 drops the merged event-count mark: it is returned by preprocessing, but the network uses only band identity. Multiple raw spikes occur in 55.6% of fitting groups and 43.5% of held-out-speaker groups, so this is a concrete information bottleneck whose class value remains to be tested. The depth-eight all-depths pilot oscillated between early extinction and an activity cascade: layer counts moved from `[16, 9, 3, 6, 27, 51, 141, 250]` at epoch 1 to `[24, 3, 1, 1, 4, 6, 32, 65]` at epoch 2, surged to `[59, 35, 97, 286, 1137, 1929, 4125, 4888]` at epoch 3, and fell to `[27, 3, 2, 2, 3, 11, 47, 58]` at epoch 4. Late-prefix NLL swung 39,814.7 → 1,692.1 → 19,787,863.2 → 22.9; terminal accuracy remained 3.1–12.5%. We stopped after epoch 4 for instability. The next controlled representation test preserves the count as a sparse vector mark before adding spike credit or threshold calibration.

**E83 all-depth readout audit (§145): the L1 signal is a shallow bypass.** On the same seed-6 checkpoints and held-out examples, we replayed the valid nonrefractory, in-band spike toggles against both fused all-depth and deepest-only main loss. The late-only L1 all-depth delta averaged −0.00936 (16/21 helpful), while all 21 matched deepest-only L1 deltas were exactly zero; every valid L1/L2 deepest-only delta was zero in both arms. Work deltas show why: L1 spike-on added 0.1429 L1 spikes/example but no downstream hidden spikes; it changed 1.2381 L1 sparse-readout edges/example. Thus the fused loss directly rewards the L1 classifier head even when no event cascade reaches deeper layers. The observed local utility is not evidence of deep compositional credit. Deepest-only L3/L4 counts were just 2/1 and cannot support estimates. The next depth test must train the primary deepest-only objective or explicitly create a plausible sparse prefix event, replay its suffix, and measure downstream changes under a declared work cap. This was a frozen validation replay, not an accuracy gain.

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
