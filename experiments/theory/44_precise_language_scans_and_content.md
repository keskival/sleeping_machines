# Precise language clocks, causal scans and content-bearing messages

Derived 30 September 2026. This note describes the current learned language
implementation and its numerical contracts. It does not establish frontier
quality, general sparse routing, physical energy savings or a scaling law.

## 287. Bounded delays preserve a layer's source order

Token i arrives at integer time i. Each of D blocks adds a delay in
[0.001, 0.011]. If D * 0.011 < q < 1, the token's final message arrives before
its query at i+q and before the next token. Within any layer, the latest
possible arrival of token i precedes the earliest arrival of token i+1.
Consequently every layer processes tokens in source order, and no message
remains pending at a token boundary. Width-256, six-layer fitting satisfies
this contract with q=0.5. It does not cover arbitrary delays or route graphs.

Within a layer, the incoming content and times are already known once the
previous layer is evaluated. Its state recurrence is therefore an affine scan,
despite nonlinear content and clock maps between layers. A whole chunk can be
evaluated per layer without observing future tokens in earlier predictions.
This permits parallel fitting and sequential causal generation of the same
function, under the declared local clock-credit surrogate.

## 288. The affine scan includes cross-chunk state and its teacher

For one real-pair mode let r>0 be the decay rate, omega its frequency and R a
rotation. The serial recurrence is

\[
 h_i=e^{-r(t_i-t_{i-1})}R(\omega(t_i-t_{i-1}))h_{i-1}+W x_i.
\]

With chunk origin t_0, define z_i=R(-omega(t_i-t_0))h_i and
b_i=R(-omega(t_i-t_0))W x_i. Then z_i=a_i z_{i-1}+b_i, where
a_i=exp(-r(t_i-t_{i-1})). Affine pairs compose associatively:

\[
 (a_2,b_2)\circ(a_1,b_1)=(a_2a_1,b_2+a_2b_1).
\]

The first drive includes incoming state transported from its saved arrival to
t_0. Inclusive prefix composition and rotation back reproduce serial state.
Differentiation includes the rates, frequencies, incoming values and intervals;
state is detached only at the declared truncated-credit boundary. Equal outputs
alone would not establish equal training: parameter gradients, warm state,
chunk partitions and future perturbations are checked separately.

## 289. Absolute time precision is separate from payload precision

Float32 spacing near positive position p is
2^(floor(log2(p))-23). At 2^16 it is 0.0078125, already exceeding the smallest
delay; at 1,000,000 it is 0.0625, exceeding every delay. Adding a small learned
delay to an absolute float32 token coordinate can therefore erase computation.

The corrected implementation stores clocks and computes phase angles in
float64, casts elapsed decay intervals and sin/cos results to the float32
payload dtype, and uses chunk-relative phase coordinates. It does not upcast
all parameters or vector work. Serial reference and parallel predictions,
states and gradients agree at both position 0 and 10M within recorded
numerical tolerances. Small measured numerical differences are retained in the
contract results, rather than described as bitwise equivalence.

## 290. A message retains content and mixes it with memory

The block does not replace each input by a constant learned node vector. Its
current content transformation is

\[
 u_i=\operatorname{LayerNorm}(W_o h_i+d\odot x_i),\qquad
 x'_i=x_i+\alpha u_i\odot\sigma(W_g\operatorname{GELU}(u_i)+b_g).
\]

The input x_i contributes to the drive W x_i, directly to the mixed read,
and through the residual x_i. Multiple messages accumulate in transported
modal state. Output gating is content dependent; the current drive is additive
and decay rates are learned but not individually selected by a write/forget
gate. Richer selective writing is a possible ablation, not an existing result.

Finite modal state compresses history and need not preserve every distinction.
Residual content, trainable mixing and memory therefore address a real
representation problem without proving lossless memory. The fixed-drive
temporal-orbit argument in §235 isolates the effect of changing one delay; it
does not restrict the full encoder's input-dependent payload.

## 291. Token representations and relative distance

The language model learns a 27-character embedding table and a vocabulary
readout. It has token embeddings, but no learned subword tokenizer. Preserving
this alphabet keeps raw-character likelihood comparable with saved controls.

A message's age changes state by exp(-r age)R(omega age). This supplies
relative-distance information inside persistent state. Transformer rotary
position encoding instead rotates queries/keys to encode relative distance in
attention scores; the two operations are related positional mechanisms, not
identical architectures. Useful content selection, memory retention and
representation capacity still have to be established by learning experiments.
RoPE: https://arxiv.org/abs/2104.09864. Selective state-space precedent:
https://arxiv.org/abs/2312.00752.

The final block's clock has no language teacher in this bounded schedule:
every message finishes before the fixed query and the query reads its completed
payload. That time does not change the readout or a later receiving block.
Zero final-clock change is expected here, rather than evidence of a severed
content teacher.

## 292. Initialization must match the task's time unit

The inherited event block initializes rates from 0.1 to 50 per time unit,
equivalent to modal e-folding times 10 down to 0.02. In serialized text the
time unit is a character. The contribution of a single drive after lag k has
norm multiplier exp(-r k); with r=0.1 it is only 0.00166 at k=64. Completed
small fits leave their longest individual timescales at a few characters.
Depth can compose these responses, so this is not a hard context bound.

The matched initialization ablation keeps capacity, weights' random draws,
optimizer, data, seed, pass count and credit horizon fixed. `long_decay`
initializes timescales 1..1024 characters; `long_spectrum` also initializes
resolved periods 4..2048 characters. Both leave rates/frequencies trainable.
Longer retention improves potential transport, while accumulated interference
and readout conditioning can worsen quality. Development decides the result.

Credit is still truncated every 64 characters. Persisting a memory beyond that
boundary carries information, but does not send a future teacher back through
the detached original write. Longer retention and longer credit are distinct
interventions. A longer-credit experiment must also account for changed
optimizer step frequency, total work and host memory. Small-scale gates are
practical promotion rules; they cannot guarantee superiority over a larger
language model or over contemporary frontier systems.
