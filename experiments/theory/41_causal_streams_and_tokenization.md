# Causal streams, tokenization and comparable probability scores

This note distinguishes a representation change from an information advantage,
and gives a causal token experiment with exact per-character likelihood.

## 269. The observation filtration includes preprocessing

Let \(X_0,X_1,\ldots\) be raw observations and
\(\mathcal F_i=\sigma(D_{\rm fit},X_0,\ldots,X_i)\), where fitted vocabulary,
thresholds and parameters are functions of the declared fitting data only.
A predictor for \(X_{i+1}\) must be measurable with respect to \(\mathcal F_i\).
Freezing weights at evaluation is insufficient if an input feature or fitted
threshold depends on held-out observations.

For example, a current-character word boundary
\(b_i=\max\{j\le i:X_j=\text{space}\}\) reveals whether the target is a space
when predicting \(X_i\). Its causal counterpart uses \(j<i\). Counts trained
on the first definition cannot simply be queried using the second: the feature
distributions and conditional tables differ. E173 rebuilds the fitting tables
under the causal definition. Numerical target mutations complement this source
argument; finite mutation probes are not a universal proof for every program.

The same requirement applies to trade-size quantiles, tokenizer merges, class
vocabularies and normalization statistics. Fit/development/test identities must
be preserved independently of the prediction implementation.

## 270. Persistent computation and truncated credit are different operations

An event stream maintains \(S=(h_1,\ldots,h_D,Q,t_1,\ldots,t_D,o)\): local
memories, pending messages, last arrival times and the latest deep output.
At each observed source arrival, process pending messages only through the
declared query deadline. Assume positive delays, causal local maps, deterministic
tie-breaking and a deadline before the next unobserved source arrival.

**Prefix invariance.** Two input streams with the same observed prefix have
identical predictions through that prefix. Induct on ordered message deliveries:
the next heap key, its payload and receiver state depend only on earlier
deliveries and already observed source events. A future source event cannot
enter the queue before observation. Equal states therefore produce equal next
states. A chunk split that preserves the entire state preserves the same program.

Detaching tensors at a credit boundary changes the derivative, not this forward
program. Resetting memories, queue or partial tokens changes the program.
E175 checks prefix/chunk invariance for the event-state language implementation;
E176 carries state through an entire character stream with 64-character truncated
backpropagation. Its scalar timestamps serialize text; they are not measured
speech timestamps. Local dense maps and readouts remain charged work.

## 271. What tokenization can improve

Characters are already tokens. An invertible encoding does not add observed
information; it changes sequence length, available representation, optimizer
geometry and work. A fixed token-credit window covers more characters when
tokens span multiple characters. A fixed *character* credit window avoids that
particular budget change. Vocabulary size also changes output and optimizer work.

Consequently, compare the same raw-text targets and report bits/character,
unique characters, presentations, raw-history/credit limits, released tokens,
vocabulary parameters and total computation. Bits/token alone is incomparable
across tokenizers. Train-only fitting and declared release latency are necessary.

[ByT5](https://arxiv.org/abs/2105.13626) establishes that byte input can support
useful language models; it also studies its longer sequence cost.
[Charformer](https://arxiv.org/abs/2106.12672) studies learned aggregation into
larger units. These suggest both a fitted tokenizer control and a learned causal
aggregation direction. Their encoder/decoder experiments do not prove that our
autoregressive event backbone will improve under either intervention.

Ordinary token vocabularies can permit multiple token sequences decoding to the
same text. A canonical token-path score then need not equal the probability of
the raw text, which sums over all decodings. Offline segmentation can also use
future characters to decide a preceding boundary. These are manageable protocol
choices, but a clean initial experiment can avoid both issues constructively.

## 272. A complete prefix dictionary gives stopping-time tokens

Take an alphabet \(\mathcal A\). Build a finite rooted tree whose every internal
node has all \(|\mathcal A|\) children. Tokens are its leaves \(\mathcal V\).
The dictionary is prefix free and complete: every infinite raw sequence reaches
exactly one leaf after finitely many characters. Token completion time is a
stopping time of the raw observation filtration. Release the vector only at that
time, preserving the last raw character's timestamp.

Fit the tree on training text by repeatedly expanding a frequent leaf into all
alphabet children. This is a frequency-prioritized prefix dictionary, not a claim
of an optimal tokenizer. Vocabulary counts choose the representation only; they
do not supply prediction probabilities. A 27-symbol tree with four expansions
has \(27+4(27-1)=131\) leaves. Rare phrases remain representable and no unknown
token or future lookahead is required. Dictionary depth bounds buffering latency.

**Unique decoding.** No leaf is a prefix of another. Starting at the root,
follow observed characters until a leaf, reset to the root and repeat. This is
the unique parse. Concatenating arbitrary leaf tokens yields exactly that parse.
Thus token sequences and complete raw phrase sequences are in bijection.

## 273. Exact likelihood and teacher inside an unfinished token

Let \(q_v(h)\) be a learned softmax distribution over the next leaf, given the
event backbone's state after completed tokens. For an observed unfinished prefix
\(a\), define the subtree mass

\[
 M(a)=\sum_{v:a\preceq v}q_v(h),\qquad M(\varnothing)=1.
\]

The next-character probability is

\[
 P(c\mid a,h)=M(ac)/M(a).
\]

Every internal node has all children, so these probabilities sum to one.
No future character or completed future token is an input to the predictor.
Inside a phrase \(v=c_1\cdots c_k\), the log probabilities telescope:

\[
 \sum_{j=1}^k\log P(c_j\mid c_{<j},h)=\log q_v(h).
\]

A finite evaluation stream ending inside a phrase is scored exactly by its
subtree mass. No terminal character needs to be dropped or its future revealed.
This permits exactly the same raw-character target positions for character and
variable-length token models, including warm context and final partial phrases.

For leaf logits \(z_v\), define the posterior over compatible leaves
\(\pi_v(a)=q_v1[a\preceq v]/M(a)\). Then

\[
 \frac{\partial[-\log P(c\mid a,h)]}{\partial z_v}
 =\pi_v(a)-\pi_v(ac).
\]

Each observed character removes incompatible alternatives and credits the
remaining ones. Over a completed token, these local teachers telescope to
\(q_v-1[v=v_*]\). This is an exact conditional teacher, not an inverse-confidence
heuristic. Timing/routing teachers still inherit the event backbone's declared
hard-schedule surrogate; exact output likelihood does not repair every routing
boundary derivative.

## 274. Efficiency and the decisive experiment

Let \(\rho=M/N\) be released tokens per raw character, \(C\) body work per token,
\(H(V)\) head work for vocabulary size \(V\), and \(T\) tokenizer/marginalization
work per raw character. The schematic inference cost is

\[
 W/N\approx\rho[C+H(V)]+T.
\]

Compression saves body events, while a larger vocabulary and subtree work spend
some savings. Training adds reverse credit, optimizer states and vocabulary
fitting; its result cannot be inferred from the event ratio. Cached leaf logits
can serve successive raw-character queries until a token completes. Pending
events must still respect raw arrival times rather than retiming a whole phrase
as if observed at its start.

The first comparison retains depth eight, raw fitting/development ranges, four
passes, 64-character credit truncation and a common initialized body. The
character arm has 27 leaves; the compressed arm has 131. Vocabulary capacity
differs explicitly. No count/copy/word output expert is added. E177 checks
normalization, token/raw-score identity, exact character-control equivalence,
causal chunking and all-layer teachers before empirical training.

What it answers: whether one causal compression mechanism improves the quality /
event-work tradeoff of this small generic stream. A failure would reject that
intervention at that budget, not all subword or learned aggregation models. A win
would justify a larger matched study, not establish frontier scaling or joules.
