# Marked events and stable deep learning

29 September 2026. Continues §§155–163. This note separates measured outcomes,
a local loss calculation, and architectural hypotheses.

## 164. A mark is also a routing intervention when keys and values share a vector

In TVLayer the input vector serves two roles:

\[
 r_{ij}=q_j^\top v_i+c_{ij},\qquad b_i=Bv_i.
\]

Positive scores admit a message and set its delay; b is integrated into the
receiver state. With multiplicity mark m and v=e_i+W_m m, the mark changes
both the route score by q_j^T W_m m and the state increment by B W_m m.
Zero initialization of W_m preserves the initial function, but does not
preserve its routing during learning. Hard admission boundaries can change
when the first quantity crosses the original score margin.

The new `address_neutral` input mode instead uses

\[
 r_{ij}=q_j^\top e_i+c_{ij},\qquad b_i=B(e_i+W_m m).
\]

At fixed model parameters and input event identity/time, first-layer admission
and delay are exactly invariant to m. The payload and its derivative still
carry m. First-layer firing, subsequent representations, and downstream
routing can change. Training also changes e, q, and c; this is not a guarantee
of identical trained route histories. The boundary-credit trace stores the
actual addressing vector. A small executable contract checks admission/delay
invariance, retained payload credit, and score reconstruction.

### Matched SHD evidence

All arms use seed 6, 512 fitting utterances, 256 held-out-speaker development
utterances, depth 4, width 16, vector width 8, two epochs, corrected grid
emission, a deepest-layer primary loss, and auxiliary weight 0.2. Route
counterfactual updates are disabled to isolate input-mark placement. The
initial deepest-loss gradient norms coincide in all three arms. The extra
count projection has eight trainable parameters and starts at zero.

| Input mark | Terminal correct | Anytime correct | L4 support at final epoch | Late-prefix NLL | Train loss, epochs 1 → 2 |
|---|---:|---:|---:|---:|---:|
| Absent | 14/256 | 14/256 | 12.11% | 2.9927 | 128.27 → 3.40 |
| Shared address and payload | 14/256 | 14/256 | 0% | 2.9984 | 121.74 → 3.00 |
| Payload only at input | 13/256 | 14/256 | 91.80% | 297.1397 | 232.64 → 497.94 |

The uninformative 20-class log loss is log(20)=2.9957. None establishes
recognition above chance. Preserving a count mark in the shared input vector
coincides with extinction of L4 activity; separating the input address
maintains extensive activity but gives strongly miscalibrated evidence.
These trained trajectories support a coupling/stability hypothesis, not a
proof that input routing alone caused extinction. Later firing and routing
are affected in both treatments. The address-neutral arm was selected after
seeing the first pair; this is sequential development evidence.

Average recorded L1 gradient norm in the final epoch was approximately
454 / 2.98 / 23.85 million across the three arms, before global norm clipping.
The final arm's L4 norm was about 13,537. Large gradients and deep event
support therefore do not establish useful optimization. The auxiliary loss
excludes L4 by definition: its zero direct L4 gradient is expected.

Saved results and configuration checks are pinned in
`results/e83/countmark_matched_20260929.json`. The scripts report roughly
293,272 hidden state-vector updates per utterance because the reference
still scans a 1 ms grid. These runs make no asynchronous energy claim.

## 165. Why suppressing evidence can be an easy loss reduction

Let C classes be balanced and let centered class evidence z satisfy
sum_c z_c=0. Along an available evidence-gain direction alpha, cross-entropy
has the local expansion

\[
 \ell(\alpha z,y)=\log C-\alpha z_y
   +\frac{\alpha^2}{2C}\|z\|^2+O(\alpha^3\|z\|_\infty^3).
\]

This follows by differentiating log-sum-exp at the uniform distribution.
Writing A=E[z_Y] and V=E[||z||²]/C gives expected loss
log C-alpha A+alpha² V/2+O(alpha³ E||z||∞³), assuming the stated moments exist.
Uninformative, label-independent evidence has A=0. Its variance costs loss;
reducing its gain toward zero is beneficial. More generally, positive label
alignment A must offset the variance term before stronger evidence helps.
The local stationary gain is approximately A/V when V>0 and the expansion
is valid. This is not a global optimum or a convergence theorem.

Hard event deletion is not continuous gain scaling. Nevertheless, this
calculation identifies a real competing incentive: an untrained active
classifier may lose much more than a silent near-uniform classifier. Closing
routes can remove that bad evidence before downstream features learn their
use. The observed descent toward log(20) with disappearing support is
consistent with this mechanism; it does not uniquely identify its cause.
The high-support arm shows the other side: retaining events without
controlling feature and evidence scale can retain confidently wrong outputs.

### A constructive next design

Implement the sparse serial continuation already specified in §157. An
arriving event carries a payload through each learned residual transformation;
creating a new threshold crossing is no longer necessary for the next layer
to receive learnable information. Optional branches still use timed races,
holds, and counterfactual credit. Bound event multiplication explicitly: one
carrier per input event per layer costs O(DE); unbounded branching does not.
Merge or select carriers only under a declared information/work budget.

Temporal stability alone is insufficient for bounded payloads. For a scalar
state coordinate driven by events,

\[
 z_r(t)=\sum_{t_i\le t}e^{(-a_r+i\omega_r)(t-t_i)}(Bv_i)_r,
 \qquad c_r(t)=\sum_{t_i\le t}e^{-a_r(t-t_i)},
\]

triangle inequality gives

\[
 \frac{|z_r(t)|}{\epsilon+c_r(t)}\le\max_{t_i\le t}|(Bv_i)_r|\quad(\epsilon>0).
\]

Thus a decayed mass channel can remove accumulation gain caused only by
message count, while a separate bounded mass feature retains rate information.
Both states admit updates at actual events with exact exponential decay;
silent units need no tick updates. This bound neither controls B/C projection
norms nor proves a full sequence Jacobian bound. Those gains and shared-event
fan-out require the depth conditions in §§107–114/157. A normalized state is
a proposed design here, not a measured accuracy improvement.

First demonstrate small-set terminal learning through the deepest layer,
with layerwise class-information probes and separate train/validation curves.
Then add causal prefix queries and calibrate first-confidence-crossing
accuracy, coverage, and latency. Prefix log loss remains proper (§130);
terminal supervision is a diagnostic isolation of representation learning.
After useful value transformations learn, route updates should preserve the
carrier within a trust region and value alternatives by cross-example
transfer (§161), avoiding reward for merely fitting the current sample.
The decisive milestones are retained class information, useful serial depth,
then calibrated stopping and measured total learning/inference work.
