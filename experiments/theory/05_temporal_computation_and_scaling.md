# Temporal computation and counterfactual routing

[Theory index](../THEORY.md) · Global sections 56–65; section numbers remain stable. · Next: [05b scaling grokking and composition](05b_scaling_grokking_and_composition.md)

## 56. Computing with time: what clocklessness forbids, what a reference adds, and how credit flows

*Written 2026-09-26. Builds on the space-time algebra of J. E. Smith (ISCA 2018; "(Newtonian) Space-Time
Algebra", arXiv 2001.04242) and on race logic as tropical algebra (Madhavan, Sherwood & Strukov 2014; Madhavan
et al., "Temporal State Machines", 2021). Those works define the primitives and their algebra. This section adds
three things: what shift invariance forbids and the minimal fix (§56.2–56.3), the credit structure of
space-time networks (§56.4), and the resulting pull-only learning principle (§56.5).*

### 56.1 Primitives

Values are spike times in T = ℝ ∪ {∞} (∞ = no spike), each line spiking at most once per episode.

| primitive | output time | neural form | tropical form |
|---|---|---|---|
| delay δ_c | a + c | axon or dendrite | a ⊗ c (min-plus) |
| first-of (OR) | min(a, b) | either input fires the node | a ⊕ b |
| coincidence (AND) | max(a, b) | threshold 2 with long PSPs | max-plus ⊕ |
| veto (inhibit) | a if a < b, else ∞ | inhibitory synapse that arrives first | not tropical: breaks monotonicity |

The race of §1–§55 is min over a group, with cancellation of the rest: first-of with a label.

### 56.2 Clockless means shift-equivariant, and shift-equivariant networks cannot add times

With no clock, nothing in the network knows absolute time: shifting every input by c shifts every spike by c.
Every primitive above commutes with the shift, so every network built from them computes a function with
f(a₁ + c, …, a_n + c) = f(a₁, …, a_n) + c (Smith's invariance). Consequences:

- **a + b is not computable**, for two variable times: it would shift by 2c. More generally, only functions whose
  pieces have unit total slope. Differences can be computed (compare a against b), sums cannot.
- **Absolute magnitudes are not computable.** "Fire 3 ms after a" is fine; "fire at time 3" is not, because
  nothing marks time 0. §38's blindness to absence is the same fact: an absent spike can't be noticed without a
  reference that says when it was due.

So an asynchronous substrate of pure delays and races computes **relations among times, never their sums**.
Sleep Sort sorts because sorting is shift-equivariant. Modular addition is not, and needs more.

### 56.3 The minimal reference: one oscillator, and time becomes a group

Break the symmetry with the smallest possible reference: a free-running oscillator of period P, shared by
everything and amortized over all queries (a brain rhythm). Now a spike has a **phase** φ = t mod P, and a
unit can act at a phase set by one input and read at a time set by another (the E25 ring: operand a resets the
phase, operand b reads it). With the reference:

- The symmetry drops from all shifts ℝ to shifts by whole periods, Pℤ. Functions are equivariant only under
  those, so phase addition (a + b mod P) becomes computable: it is invariant under Pℤ shifts of both operands.
- **Races on a circle need an anchor.** "First" is not defined on a circle: every phase is after every other.
  Order exists only relative to a reference event (the read): the winner is the first detector phase *after*
  the read. Every cyclic race therefore has an anchor, and learning targets must be defined relative to it
  (E26: the teacher listens half a unit after the read).
- **What one reference buys is exactly one character.** With a single oscillator, the relations reachable at
  O(1) events per query are those factoring through one phase composition, y = h(f(a) + g(b) mod P): E25's
  learnable class, now derived from the symmetry instead of observed. Several oscillators with incommensurate
  periods give several characters; the DFT-rank argument of §53.5 bounds what they can compress.

The manifesto's "relative, causal, temporal referencing" is therefore not optional: a clockless system computes
relations; a system with one rhythm computes group operations; nothing in between computes sums.

### 56.4 Credit flows along one causal chain

A spike's time in a network of delays, first-ofs and coincidences is a tropical polynomial in the delays: the
sum of the delays along one path, the **critical path** (the argmin through first-ofs, the argmax through
coincidences). Its derivative with respect to a delay is 1 on the critical path and 0 elsewhere (almost
everywhere). So exact credit in a space-time network is:

- **Sparse by construction:** one path per spike, length = depth, not fan-in × depth as in a dense backward
  pass. The critical path is recorded for free during the forward race: each node remembers which input set its
  time (the E4 "which input arrived last before threshold" trace).
- **Local:** each node needs only whether it is on the path and the sign of the error at the end.
- **Discontinuous at ties:** when two paths tie, credit switches. This is where learning can oscillate, and
  where timing noise (§41, §48) smooths the switch into a probability.

(For integrate-to-threshold nodes the path generalizes to the causal set, §34.1; Mostafa 2018's exact TTFS
gradient has this structure.)

### 56.5 Teach the event that should have won; never touch the losers

Every error in a race network is a race lost by the right event: the teacher's spike came after the winner's
(or never came), or a spike that should have been vetoed was not. The native fix is to move **only the event
that should have won**, along its critical path, toward an **anchored target time** (the reference + margin),
and to leave the losers alone:

- Wrong class won: move the teacher's path so the teacher arrives at anchor + m.
- A spike fired that should not have: move the **veto's** path so the inhibitor arrives first. Veto is the
  only native way to make something later, and it makes suppression an act of winning too.
- Teacher never fired: move its path toward coincidence. The target of an arrival is its partner in the window,
  not "earlier": arrivals move toward each other. A one-sided "make it earlier" rule drifts, since shortening
  delays or loop periods makes everything earlier (E28 delays past the anchor, E29 periods to the floor).

A false positive does require acting on the loser, and here the distinction is how. Shifting its delays later
(a **push**) displaces it in time; that is what collapsed E26. Adding an inhibitory condition that blocks it on
the offending inputs (a **veto**) specializes it and leaves its timing alone. So the rule is: *losers are
specialized by inhibition, never displaced in time.*

This is regression to an anchored time on one causal chain, not a margin against competitors. §54 is the
evidence that it matters: the two-sided rule (pull the teacher, push the winner) collapses all detectors onto
one phase, while pull-only learns the relation. The push fails because a loser's time has no target of its own;
it is set by other classes' errors, and each push hands the lead to the next loser. In a race the losers are
beaten, not punished.

**Why this is sparse and asynchronous by construction:** updates happen only on errors (a lost race), touch
only one causal chain (depth-many delays), and need only a local anchor. No sums over samples, no backward pass
through all synapses, no clock except the one reference that §56.3 shows is needed anyway.

### 56.6 Predictions (M56, E27)

(i) A network with learnable delays on excitatory (coincidence) and veto synapses, trained by §56.5 only,
    learns temporal patterns of the form "B within Δ after A, unless C in between", generalizing to unseen
    timings.
(ii) Fixing false positives by pushing the loser's delays later, instead of by veto, breaks it (as in E26).
(iii) Without veto synapses, patterns that need "unless" are not learnable at any size (monotonicity).
(iv) Updates per sample fall to ≈ the error rate × depth; synaptic events per sample stay ≈ input spikes +
     O(1).
(v) Without a reference, a sum of two times is not computable (§56.2): a linear-time race network given the
    operands as spike times, with no oscillator, cannot learn a + b as an output time at any size, while it can
    learn comparisons (which of a, b is larger, by how much, within a window).
## 57. Routing needs counterfactuals, and cancellation supplies them without extra events

*Written 2026-09-27, prompted by the observation that a race network is a routing network: which node wins is a
discrete route choice, and critical-path credit (§56.4) only tunes timing along the route taken.*

### 57.1 The problem

§56.4's derivative is 1 along the critical path and 0 elsewhere. It says how to move the route that was taken,
never whether another route should have been taken. The same wall appears in sparse mixture-of-experts: the
gate's gradient exists only for experts that ran, which is why top-k routing with k ≥ 2 is used, to compare at
least two alternatives. Routing credit requires counterfactual information: what the untaken routes would have
produced.

### 57.2 Three sources of counterfactual information in a race

| source | what it tells | cost |
|---|---|---|
| k winners per group (k ≥ 2) | the runners-up's actual outputs | (k − 1) extra spikes per group, all downstream events they cause |
| cancelled near-misses | how close each cancelled node came (its frozen charge) and on which inputs (its best partial window) | none: the charge is state the node already has at cancellation |
| timing noise σ | which route would have won under a small perturbation | extra runs, or temperature over time |

The second is native and free. A cancelled node stops integrating, but the charge it has reached and the inputs
that produced it are its state at the moment of cancellation. They are exactly the counterfactual "had I been
allowed to continue, I would have fired on these inputs", ranked by how near the node came.

### 57.3 The rule

On an error, the teacher's missing input tells the hidden layer what route was needed (through the teacher's
synapses, w2). Pull the **cancelled node the teacher wants with the highest near-miss**, on **its best partial
window only**, so it wins its group next time; do not touch the route that was taken, and do not push the winner.
Strengthening all inputs that reached the cancelled node (distractors included) is destructive (E28 pilot: 0.18,
at the "none" floor); restricting the pull to the partial window turns it into a gain (0.41 vs 0.34 for top-k
fired credit, 0.35 at depth 1; one seed, 8k episodes; full runs queued).

### 57.4 Predictions (M57, E28)

*Status (full runs, 3 seeds): (i) confirmed against critical-path-only credit (0.31–0.33 vs 0.17); near-miss did not
beat fired credit at k = 2 (0.31 both), (ii) k = 1 near-miss matched k = 2 fired at 28% fewer events but with high
variance, (iii) not confirmed (depth 2 ≈ depth 1), (iv) confirmed (push 0.17). All arms far below ceiling; the
readout lacked §60's conservation and capacity. With §60 applied to every node (E28b, 3 seeds): depth 2 with routing
credit 0.72–0.74 vs depth 1 0.68 vs depth 2 path-only 0.17: (i) confirmed strongly, (ii) near-miss ≈ fired, (iii)
a modest lead for depth that did not hold with 15 classes from 6 shared motifs (E28c, 5 seeds: depth 1 0.51 vs
depth 2 0.44 at 60k): depth is learnable, not yet advantageous.*

(i) Near-miss routing credit beats critical-path-only credit at depth 2, and matches or beats top-k fired credit.
(ii) Near-miss credit with k = 1 approaches its k = 2 result: counterfactuals from cancellation replace
     counterfactuals from extra firing, at fewer events.
(iii) Depth 2 with near-miss credit beats depth 1 on hierarchical motifs.
(iv) Displacing false winners in time breaks it, as in E26 and E27.
## 58. What counts as generalization: restriction, forced generalization, and grokking

*Written 2026-09-27, after the objection that E25/E26 generalize because their structure restricts them to the
answer's form.*

A learner can reach the relation on unseen pairs for three different reasons, and only the last is grokking.

1. **Restriction.** The hypothesis class contains the relation and little else. E25/E26's single loop with a
   single-phase readout can only express one-character relations (§53.2); within that class, any good fit on
   enough data is the relation. This measures the prior, not the learner.
2. **Forced generalization.** The class does contain memorizers, but only below its capacity. E25 shows it
   exactly: with 3p parameters, fits that memorize exist below n ≈ 3p (p = 31, 5% of pairs: train 1.0, test at
   chance) and stop existing above it. Generalization above capacity is counting, not learning.
3. **Grokking.** At n well below capacity, memorizers consistent with the training set exist in the class, and
   the learner still reaches the relation. That is a property of the learning dynamics (its implicit preference),
   and it is what the dense MLP shows on E24 (≈ 17k parameters vs 480 training pairs at p = 31, frac 0.5).

**Criterion.** Report the capacity ratio ρ = n / (effective parameter count) with every generalization result.
Grokking claims need ρ ≪ 1, a class that provably contains memorizers at that n, and a demonstration that some
learner in the same class does memorize (a table-like baseline trained the same way).

**Consequence for this project.** A loop in the substrate is not itself the bias: a loop's period is a
learnable delay, and delays are scale-free (any period works with rescaled injection delays), so "a recurrent
delay loop exists" is as generic as recurrence. The bias in E26 is the **single-phase readout**, which removes
every non-cyclic hypothesis. The honest test (E29) is a general race network, the one that memorizes E24 at
test 0.00, given recurrent delay loops as a resource and native credit (pull-only, errors only, near-miss
routing). It passes only if it reaches the relation at ρ ≪ 1 while the same network without loops, or without
counterfactual credit, memorizes.
## 59. A two-counter machine wired from the basis, and restoration in time

*Written 2026-09-27 with E30 (`e30_minsky.py`, results in `results/e30/minsky.json`).*

**Construction.** A netlist of four node types (Delay; Or; And with a PSP window per input; Veto), plus a reference
oscillator, runs two-counter Minsky programs. Counter value n is the phase OFF + n·q of a spike in a hold loop of
period T. Per cycle, the spike takes one of three Delay paths: hold (T), increment (T + q), decrement (T − q);
path choice is an And with the instruction's enable line (long PSP) and a Veto on the hold path. The zero test is
an And of the counter with the reference (window q/2); branching is a Veto ("next unless zero") and an And ("next
if zero"); a decrement blocked at zero returns to hold through And(zero, gated counter). No node reads a clock,
stores a number, or branches in code.

**Result.** Exact on every test (add, double, parity; 10 inputs including zero edge cases): correct counters and
the exact cycle count, ~5k events per run. Two-counter machines are Turing-complete (Minsky 1967), so the basis
{delay, first-of, coincidence, veto, hold} + one reference is Turing-complete given unbounded phase precision
(prior art for spiking networks with exact delays: Maass 1996). Two wiring lessons are general: coincidence
windows must be per input (a PSP length per synapse), or stale spikes from earlier cycles pair with new ones;
and every veto that blocks a path must hand the spike to another path, or state is destroyed.

**Precision is the tape, and restoration makes length free.** With jitter σ on every hop, phases random-walk.
Without restoration, success falls with program length (q/σ = 20: 0.60 over 25 steps, 0.30 over 81). With one
restoring coincidence per counter per cycle (the loop delivers the spike q/2 early; a comb tick at the nominal
phase re-emits it), success is length-independent (q/σ = 20: 1.00 at both lengths; q/σ = 10: 0.975 at both). (A random walk fits the unrestored runs with one parameter: a counter phase after s steps is N(0, σ√(h s)) and
fails beyond q/2; q/σ = 20 with 60% success at 25 steps gives h ≈ 5.8 jittered hops per cycle, which predicts 35%
at 81 steps, observed 30%; the netlist has about six hops per counter per cycle.) This
is digital restoration done in time: the cost is one comb coincidence per counter per cycle, the capacity T/q
states per counter, and the error rate per step a function of q/σ only. The residual failures at q/σ ≤ 7 come from
the unrestored control margins inside a cycle.

**Consequence.** An asynchronous temporal substrate can compute arbitrarily long with noisy timing, and the
resource accounting is explicit: precision (T/q) replaces tape, cycles replace steps, events replace energy, and
restoration events are the price of reliability.
## 60. Pull-only on weights needs conservation; capacity sets the readout

*Written 2026-09-27 from E26b and the E29 readout diagnosis.*

**E26b refutes §54's transfer as stated.** Dropping the competitor push from the main weight-based race collapses
SHD from 0.35 to 0.061 (`--compete 0`, depth 1, 10 epochs). The positional principle holds for timing parameters
(delays and phases, where a pull cannot inflate anything: E26), but not for weights: in the weight race the push
is the only force bounding the weights, and without it every class node inflates and fires.

**The native counter-force is conservation.** E29's readout, isolated on a frozen hidden layer with unique codes,
shows the sequence: pull-only weights saturate (every class fires at once, the winner is timing noise); a
per-node conserved synaptic budget (heterosynaptic: a pull on the co-active synapses is paid by the node's
others) stops that, but only if each pull moves a fixed fraction of the budget (otherwise each pull rescales the
whole node and it remembers only its last sample); weakening a false winner must also conserve (otherwise it is
a one-way drain); per-node prices (thresholds raised by false wins, lowered by misses; §36, §51) break up the
remaining hub classes. Veto at the readout over-suppresses when codes overlap.

**Capacity, not the rule, set the plateau.** A first-to-threshold class node is a linear threshold unit on the
hidden spikes; with 96 hidden nodes it can separate only ~2·96 random patterns (Cover 1965), fewer than the 480
training pairs, and the readout plateaued near 0.45 for every rule variant. With 384 hidden nodes the same rule
memorizes (train 0.88 by epoch 7). Grokking tests need hidden layers above this bound, or the "memorizer" in
§58's criterion does not exist.

**Revised principle (§54, §56.5):** never displace in time; for weights, pull under conservation; remove false
winners by veto where codes are specific and by conserved weakening plus prices where they overlap.
## 61. Two kinds of temporal tolerance: aligning destroys the interval, holding keeps it

*Written 2026-09-27 from the E27 error analysis.*

A coincidence node can accept "B within Δ after A" in two ways.

- **Align:** delay A by about the typical gap so it meets B, with a narrow window. The interval [t_A, t_B] is
  compressed into the window; the node no longer knows what happened in between.
- **Hold:** give A a PSP of length Δ and fire when B arrives while it is still open. The node is armed exactly over
  the raw interval [t_A, t_B].

The two are equivalent for detecting the pair and not equivalent for anything that depends on the interval's
content. A veto for "unless C in between" must arrive while the node is armed; under alignment the armed span is
a narrow window near t_B + d_B, while C can fall anywhere in (t_A, t_B), so no single veto delay covers it. Under
holding, a veto with no delay does. E27 shows the symptom: with alignment the veto synapse on the right channel
reaches full strength for every pattern, yet vetoed near-misses remain the main error (192 of 318 errors after
100k episodes). With holding, veto adds +5 points over no veto (pilot, 2 seeds) versus +1.8 under alignment,
though the pilot's windows only grow, which caps its overall accuracy (0.71–0.73).

**Principle.** Delays are for *where in time* an event should act; PSP durations are for *how long* a node should
remember that it happened. Any operator whose meaning involves the interval between events (veto, ordering,
"no C since A") needs duration, not delay. Learning therefore has two timing parameters per synapse, delay and
duration, and they answer different credit questions: a coincidence missed by misalignment moves the delay; one
missed because the partner came too late for the PSP lengthens the duration; a false fire from a partner that came
too late shortens it.
## 62. Why depth does not pay yet (corrected: the hidden nodes are parts; the readout discards their order)

*Written 2026-09-27 from E28c and a hidden-selectivity diagnosis (15 classes built from 6 motifs, 192 hidden,
20k episodes; selectivity = max over conditions of P(fire | condition) − P(fire | not)).*

| hidden layer | median motif selectivity | median class selectivity | motif-selective nodes (> 0.3) | class-selective nodes (> 0.3) |
|---|---|---|---|---|
| untrained | 0.07 | 0.13 | 7 | 28 |
| trained, label-gated credit (fired) | 0.12 | 0.21 | 24 | 55 |
| trained, label-free (every winner pulls its window) | 0.11 | 0.19 | 22 | 55 |

Training makes hidden nodes more class-selective than motif-selective, with or without labels: they duplicate what
depth 1 does, so depth adds nothing (E28c: depth 1 0.51 vs depth 2 0.44). Label-free learning from recurrence
(the STDP route to repeated patterns) does not change this, although motifs are ~50× more frequent than chance
coincidences of the same channels.

**Correction (same day).** The selectivity metric conflates: a genuine motif detector also looks class-selective,
because it fires for every class containing its motif. Looking at receptive fields instead (the two strongest input
channels of each hidden node that some class relies on): after training, 86 of 124 such nodes take both inputs from
the same motif (parts), 16 from different motifs, 22 use a non-motif channel; untrained, 2 of 35 were parts. **The
hidden layer does learn parts.** What fails is the readout introduced by §60: it accumulates without a window,
which counts which parts occurred and discards their order, while E28's classes are ordered pairs (and the decoys
are reversed pairs). Non-leaky accumulation was right for a static task (E29) and is wrong for a temporal-order
task, where the readout must align one part's spike with the other's by a delay inside a window (§61). The original
diagnosis below is kept for the record.

**Second correction (error breakdown, same day).** Depth 2 makes no order errors at all (0% reversed-class answers,
vs 13% at depth 1); 56% of its answers are a class sharing one motif with the true class. The failure is
**conjunction**, not order: a single detected part fires a class. It survives capping single synapses below
threshold (~14 redundant detectors per motif still sum past it), windowed readouts, and global competition across
the hidden layer. The open problem is precise: a native readout that requires two *distinct* parts when parts are
represented redundantly. (Dense readouts get this from signed weights that learn to subtract single-part evidence;
the race readout has positive weights and one threshold.) Tried and negative: input consumption (the first hidden node to
fire uses up its input spikes, the weaving operator's self-cancellation), 0.16–0.35, because the earliest
coincidences are often distractors and they consume the motifs' spikes. Not yet tried: a readout node whose inputs
are vetoed by themselves after the first part (refractory per part group), so only a different part completes it.

**Original diagnosis (superseded).** A motif is "j within 0.3–1.5 after i"; the hidden window is 0.6. A pull moves only arrivals already
inside the firing window, so a node whose window catches i but not j can never align j onto it: the recurrence is
there, but the credit cannot reach it. This is §61's distinction again: to discover a motif a node must either hold
long enough to see both spikes (duration) and then align (delay), or receive credit from near-coincidences across a
span longer than its window. **Prediction (M62):** hidden nodes with learnable duration (hold), shortened as
their delays align, become motif-selective, and depth 2 then beats depth 1 on shared-motif tasks.

**Test (same day): not supported.** Hold-then-align (windows start at 2.0; each pull aligns the window's arrivals
and shrinks the window toward their spread) leaves accuracy unchanged (depth 2: 0.38–0.48, 2 seeds; depth 1 0.50)
and, label-free, raises class selectivity more than motif selectivity (median 0.28 vs 0.16; strongly
motif-selective nodes fall from 22 to 9). Longer windows let a node see more of an episode, and it uses that to
become a better class detector. Reach was not the missing piece; the missing piece is a pressure that makes a node
prefer a part over a whole. In dense networks that pressure is width-limited capacity shared across many
outputs; here every hidden node can afford to be a class template. Open.
## 63. A measured frontier: event learner vs clocked dense model on a timing task (E32)

*Written 2026-09-27. Task: E27's ("B within Δ after A unless C", 4 patterns + none, 12 channels, episodes of 10
time units). Dense: 1-D temporal conv on binned spikes (F filters, receptive field 4 time units, ReLU, max over
time, linear readout), backprop + Adam, 200k training episodes as for E27, 2 seeds; cost = multiply-adds (MACs) of
the forward pass. Event: E27's learner, 5 seeds; cost = synaptic events actually delivered (excitatory synapses of
the channels that spiked, existing veto synapses reached, detector spikes).*

| model | test accuracy | cost per episode |
|---|---|---|
| event learner (delays, windows, veto; errors-only updates) | 0.914 (0.887–0.934) | 10.2 synaptic events |
| dense, F = 16, δ = 0.05 | 0.995 | 3.07M MACs |
| dense, F = 16, δ = 0.25 | 0.984 | 123k |
| dense, F = 16, δ = 1.0 | 0.935 | 7.8k |
| dense, F = 4, δ = 1.0 | 0.895 (0.872, 0.918) | 1.9k |
| dense, F = 4, δ = 2.0 | 0.845 | 500 |
| dense, F = 2, δ = 1.0 | 0.686 | 970 |

**At matched accuracy (≈ 0.91)** the event learner uses ≈ 190× fewer operations per episode than the cheapest
clocked dense model found (1.9k MACs), ≈ 80× less energy at published per-operation costs (Loihi synaptic event
≈ 24 pJ vs a 45 nm MAC with its weight read ≈ 10 pJ). With the episode inside 99% silence (a sparse stream) the
clocked model's cost grows 100× and the event learner's does not: ≈ 19,000× in operations, ≈ 8,000× in energy.
Training: ≈ 0.09 updates per episode touching ≲ 10 parameters (≈ 1.8·10⁵ parameter changes in all) vs ≈ 1.2·10⁹
MACs of backprop, ≈ 6,000×.

**What this does not show.** (1) Accuracy: the dense model goes far higher (0.995) at far higher cost; the event
learner has no knob to buy accuracy with more compute yet. (2) Most of the separation is the clock, not the
learner: a dense conv evaluated only where spikes are (a sparse, event-driven digital implementation) would cost
≈ spikes × F × width ≈ 6.6 × 4 × 4 ≈ 106 MACs, only ≈ 10× more than the event learner. That is §55's point: the
advantage belongs to the event paradigm however it is implemented, and grows with silence. (3) One task, designed
around the primitives; the cheapest dense model was searched only over F ∈ {1, 2, 4, 16} and δ ∈ {0.05 … 2}.
## 64. Order is asymmetry of duration; one node computes an interval predicate; few mistakes suffice

*Written 2026-09-27, with E27's directional-hold result (0.9995 and 1.000 on 2 seeds, ~400 updates in 40k
episodes; without veto 0.81).*

### 64.1 The recurring construction

Three separate constructions needed the same thing. E30's Minsky machine worked only when coincidence windows were
set per input (a long PSP on the enable line, none on the counter spike); with symmetric windows, stale spikes paired
across cycles. E27's detector reached 1.0 only when A's input opened a long PSP and B's was instantaneous; its
symmetric version accepted reversed order (0.71) and its aligned version lost the interval the veto needs (0.91).
In each case **order is encoded by an asymmetry of PSP durations, and exclusion by a veto inside the held
interval.** A symmetric coincidence is order-blind by construction.

### 64.2 One node, one interval predicate

Give every synapse three timing parameters: delay d, duration (PSP length) w, and sign (excitatory or veto). A node
with an excitatory A-synapse (delay d_A, duration w), an instantaneous excitatory B-synapse (delay d_B) and veto
synapses (delays u_C) fires exactly when

  t_B + d_B − (t_A + d_A) ∈ [0, w]   and   no C with t_C + u_C in (t_A + d_A, t_B + d_B).

That is the predicate "B within [d_A − d_B, d_A − d_B + w] after A, and no C in between (shifted by u_C)", the
atom of the manifesto's causal logic ("emit A after x if no B") generalized to an interval. **Corrected claim, verified (E33).** A relation fixing one bounded difference with exclusions ("q within
[lo, hi] after p, no C in between") is one node. A relation fixing the order of m points is a chain of m − 1
nodes, and not one node: a node's hold condition ("each held input arrived within its window before the trigger")
is invariant under swapping the arrival order of two held inputs inside their windows, so it cannot fix their
mutual order; a chain can, because each node's output spike carries "so far in order" forward in time to the node
that checks the next point. **Temporal depth equals the length of the order chain.** *(Superseded by §71: depth 2 suffices for every conjunction of order constraints; one node orders at most three events.)* E33 wires all thirteen of
Allen's interval relations this way (before, meets and their inverses: 1 node; overlaps, starts, during,
finishes, equals and inverses: 3-node chains; causal, no negative delays) and checks them against their
definitions on ~19,900 random interval pairs with planted equalities (tolerance 0.05; 99 samples within a
tolerance band skipped as ambiguous): exact, every relation with 475–3,964 positive cases. (Allen's algebra is
prior art; the node-level construction and the depth statement are the claim.) First versions had three bugs,
each instructive: delaying the trigger instead of the held input weakens "after" into "not too long before"; a
chain cannot subtract a tolerance from a spike time (non-causal), it can only absorb it in later delays; and
planted equalities must keep intervals valid.

### 64.3 Why so few updates

The hypothesis class of one hold+veto node on N channels is small: an interval on one time difference (two
parameters) and a set of veto channels (a disjunction over N): VC dimension about 2 + N. Mistake-driven learning
of an interval and of a monotone disjunction needs O(VC · log(1/ε)) mistakes (Littlestone's bounds for
monotone disjunctions are logarithmic in N per relevant channel), independent of how many episodes stream past.
E27 hold: ~400 updates in 40k episodes (~1%), for K = 4 detectors on N = 12 channels, i.e. ~100 mistakes per
detector. The dense conv baseline has 16 × 12 × 16 + readout ≈ 3k parameters at δ = 0.25 and needs ~10⁵ episodes.
**The event learner's sample and update efficiency is an Occam effect of the primitive:** when the task is made
of interval predicates, a node that computes exactly one needs only as many mistakes as its few parameters.
The flip side is the same statement: tasks not made of such predicates need depth, which is where §62 stands.
## 65. Depth and routing, learned: hold/trigger chains (E34) and nothing-given detectors (E35)

*Written 2026-09-27.*

**E34: depth pays when composition is a chain.** E28's class nodes accumulated evidence, so any one part could fire a
class (§62). A hold/trigger class node cannot be satisfied by one part: one part's spike must be held and a different,
later part's spike must trigger. On E28's task (15 classes from 6 motifs; 5 seeds, 40k episodes): depth 2 0.97
(0.92–0.99) with part windows [0, 1.5], 0.86 with a generic bank of windows {1, 2, 4}, 0.79 with one wide window 4;
depth 1 with the same nodes 0.39; E28's accumulating readout 0.44–0.51. Routing (which part holds, which triggers)
is learned by counterfactual pulls under conserved budgets. Learning part windows from class routes failed
(0.08–0.13): a part shared across classes must not be shaped by one class's errors, the same lesson as §62.

**E35: nothing given.** Detectors with learned hold, trigger and veto weights over all channels and learned hold
durations solve E27's task at 1.000 on four seeds and 0.9995 on the fifth (7.5 synaptic events per episode, 443–1,530
updates in 200k episodes). Given E27 told its detectors their channels, this removes the structural prior that made
E32's comparison unfair; the headline comparison (§63's table) holds with nothing given.

**Why routing is cheap here.** Each detector's routing is a choice of one hold and one trigger channel: 2 · log₂N
bits. Counterfactual pulls find it with O(N) mistakes per detector (E35: ~100–400 per detector), consistent with
§64.3's Occam argument extended to routing.
