# Language capacity, causal online learning and larger-data promotion

Derived 30 September 2026. This note motivates testable changes to the integrated
architecture without replacing its timed races, sparse state updates or
counterfactual teacher. Elementary linear algebra here is not a novelty claim.

## 303. A concrete message-width restriction at the output

For each realized race path, the integrated model produces 27 character logits
through a linear head:

    z = W h + b,       h in R^d, W in R^(27 x d).

Softmax ignores a common additive logit. Let H = I - 11^T/27. The attainable
centered logits lie in the affine space H b + image(H W), whose dimension is
at most min(d, 26). For d=16, the head therefore restricts every context's
log-odds vector to at most 16 affine directions. Increasing dormant receiver
count alone does not remove this restriction: all selected messages still
meet the same 16-dimensional readout.

At d=32, the centered head can have rank 26, removing this particular output
subspace restriction. That is a concrete reason to test richer messages before
very long fitting. It does not prove that text8 requires every direction, that
the hidden dynamics will supply the necessary features, or that this head
restriction caused the current score. Sixteen float components are not simply
"too few bits" for a 27-character target.

## 304. Larger useful capacity and additional data are a conditional hypothesis

The width-32 model has 1,388,871 parameters versus 361,367 at width 16, while
retaining 324 units, six selected state updates and twelve addressed key scores
per input. Its local state, gates, keys and content maps are also wider. Local
matrix arithmetic scales approximately as d squared; preserved event sparsity
does not make extra representation work free. This is an integrated width
intervention, not an isolated output-head ablation.

A larger representation can lower approximation error, while additional data
can constrain more of its parameters. Their benefit is conditional on useful
credit, exploration, retention and optimization. Increasing width changes
normalization and modal initialization too, so do not assume exact nesting of
the full model classes or a theorem that test error decreases with width/data.

The matched first comparison is width 16 versus 32 at 32,768 fitting characters,
four passes, the same 8,191 cold development targets, seed and 16-character
credit horizon. Only a completed improvement promotes width 32 to 131K. A gain
of 0.02 bpc is an exploratory progress gate, not statistical significance.
The subsequent 1M gate requires at least 0.1 bpc gain over width-32/32K and
development bpc at most 3.0. Preserve every completed/negative row.
The prepared AWS ladder uses seed 6. Local follow-ups add matched width-16 and
width-32 controls at seed 7, giving independent initialization evidence before
that host's larger-data promotions rather than repeating the identical seed.

The pool-4/8K run completed at 3.426 versus pool-2's 3.398. This tests more
addressed alternatives at a small data budget, not wider messages or all
possible ways to add useful capacity. Its negative direction remains evidence.

## 305. Iso-FLOP advantage concerns the best achieved risk under a budget

Write whole fitting work as N c(d,C,H) plus declared setup/evaluation costs,
where N is actual target presentations, C addressed alternatives and H credit
horizon. Compare configurations through the best attainable held-out risk
under that complete budget, not per-target arithmetic alone. A low per-target
cost creates room for additional data, content width or useful credit; whether
these improve quality sufficiently is an empirical question.

The two completed pool-2 data points (3.398 at 8K, 3.121 at 32K) show learning
at fixed capacity with nearly constant per-target work. Two points do not
identify an asymptote or justify extrapolating past the Transformer/LSTM scores.
The report plots completed work/quality pairs with development and test scores
in separate panels and preserves all their data/model/estimate conventions.

## 306. Causal online adaptation has a different evaluation contract

For blocks B_j and parameters theta_j available before block j, the score is

    R_online = sum_j sum_(t in B_j) -log2 p_theta_j(y_t | observed prefix_t).

Predictions use only causal source inputs, and theta_(j+1) is obtained only
after B_j's prediction losses have been recorded. Feedback is delayed by up
to sixteen targets in the current experiment. Future labels cannot affect
their own prediction. This is not immediate per-character SGD.

Both arms inherit the same selected neural checkpoint and start cold event
state on a new development interval. Frozen weights still permit event-memory
updates; online learning adapts the full neural backbone with fresh Adam and
a learning rate fixed before this stream. Paired block race noise reduces
initial routing differences; contexts may diverge after parameter updates.
No replay or official-test adaptation is allowed in this pilot.

Charge counterfactual values, backward, clipping and Adam as well as prediction.
Inherited fitting is identical and additional to both arms. A successful online
score is evidence for adaptive deployment under this feedback policy; it cannot
replace a frozen official-test score or the earlier statistical-mixing result.

## 307. Long-run reproducibility is part of the experiment

The new driver checks the chosen width, exact next-update recovery and causal
predictions before fitting. A guarded end-to-end probe was interrupted in the
second epoch and resumed from disk. Its curve, final score, parameter changes,
random-draw counts and work ledger match the uninterrupted control exactly.
Source/data hashes, model/Adam state, persistent messages and arrivals, selected
weights, pass/target pointers and RNG must remain fixed during recovery.

The AWS definition is 10M/four passes, 200K validation and 999,999 frozen test
targets with width 32. It is a defined benchmark, not a completed result. The
CPU emulator's runtime must be estimated from the larger-model pilot on that
host; a GPU provision alone does not accelerate the serial implementation.
FLOPs, elapsed time, traffic and physical joules remain separate quantities.

## 308. Queries and temporal softmax do not specify the memory bank

At depth l, the implemented query is q_l = Q_l layer_norm(x_l). The candidate
key is k_i = prototype_i + K_i h_i, using that receiver's persistent state.
The score s_i = clamp(q_l dot k_i / sqrt(d) + bias_i, -12, 12) sets rate
lambda_i = exp(s_i). With independent Exp(1) samples E_i, the winner minimizes
E_i/lambda_i. Exactly P(W=i) = softmax(s)_i. Clipping changes the scored
distribution, not this identity. The clock bounds preserve winner identity.

Each depth addresses only the pool associated with the currently observed
character: two candidates in the primary configuration, twelve scored keys and
six selected state updates per character. A query cannot search arbitrary past
positions or other character pools. The winner mixes incoming content with
retained state and emits a new value; it does not discard the incoming vector.

For fixed candidates, E[v_W] = sum_i softmax(s)_i v_i. One sampled winner is
not the deterministic weighted sum, and nonlinear downstream layers/losses do
not commute with this expectation. The local counterfactual teacher has the
restricted scope of §§296 and 301. No proof of equivalence to a full
Transformer follows from the race identity alone.

The earlier `sleeping_machines/race_language.py` probe really does keep separate
historical prefix keys and observed-successor values, using a bounded inverted
index and race versus softmax aggregation controls. It also executes a dense
carrier and was smoke-tested, not established as the integrated benchmark.
Do not describe the prioritized sparse receiver model as that cached-token
attention experiment. Temporal competition and memory organization are two
independent design choices; a token bank could also use temporal competition.

## 309. Bounded persistent storage is not equal recall capacity

The saved four-layer, width-256 Transformer has `ctx=256` characters, learned
positions and sliding-window scoring. Its implementation recomputes windows;
there is no persistent KV cache. This window is a reference-model hyperparameter,
not an intrinsic text8 limit. The integrated model carries recurrent state until
the fitting pass or evaluation stream resets. Its sixteen-character truncated
gradient horizon is not a forward-history limit. Learned decay and compression
still restrict which earlier information remains usefully recoverable.

Counting only raw tensors for one stream with all 324 receivers populated:

    bytes_ours(d) = 324 (4d + 8) + 4d.

Float32 state vectors, float64 last-arrival clocks and the last deep message give
23,392 bytes (22.84 KiB) at d=16, and 44,192 bytes (43.16 KiB) at d=32.
These upper bounds on the declared state-tensor layout do not grow with stream
length; Python dictionaries, tensor objects, integer indices and counters are
additional. Parameters, optimizer, gradients and temporary activations are
excluded. This is not a measured process-RSS claim.

A conceptual standard FP32 KV cache for the saved Transformer shape would hold
2 L T d floats: 2 × 4 × 256 × 256 × 4 = 2,097,152 bytes, or 2 MiB.
That is 47.46 times the width-32 raw recurrent state. The denominator is not
equal usable memory capacity: per-token keys/values and compressed receiver
state retain different information. Quantization, sharing and changed context
would change the KV allocation. The existing reference has no measured cache
allocation, so this ratio must be labelled conceptual.

Fixed-size recurrent compression has a structural resemblance to selective
state-space memory; see Gu and Dao, Mamba, https://arxiv.org/abs/2312.00752.
That does not make the integrated hard-race, addressed-unit construction a Mamba
implementation. Separate learned clocks, sparse receiver updates and conserved
counterfactual route credit remain its proposed construction. Establish useful
long-context retention with matched retrieval/recall tests before asserting a
comparable-capacity memory advantage. Preserved structured pointer results
remain evidence for those earlier models, not a substitute for this test.

## 310. Test episodic race memory without replacing the sparse backbone

The user requests a per-position KV analogue, small data first, and a separate
architectural FLOP estimate even if emulation is inefficient. Compression is
not required by temporal races. Retaining separate historical keys and values
uses storage linear in history, but selected value delivery can remain bounded.
Candidate discovery and scoring require their own explicit cost/coverage model.

The proposed small intervention adds learned query/key/value maps at each of
the integrated model's six event depths, retaining receiver races, content
mixing, sparse state updates and counterfactual teaching. Historical entries
are created only from already observed input messages. Each depth races over
a shortlist of recent entries and matching-character indexed entries, then
gates one historical value into the next message. All entries are retained;
the index does not claim access to arbitrary semantic matches. The candidate
budget, coverage, age and stored bytes are reported separately from deliveries.

Inference reads only the selected value. Training reads all admitted values
for the conserved teacher and charges that work. Token keys/values are frozen
when the truncated credit boundary passes, as cached activations are not
recomputed after each parameter update. Future targets are never inserted.
Winner selection remains stochastic and is not equivalent to deterministic
Transformer attention. The causal delay bound must include the added races.

Compare against the existing integrated backbone at the same small fitting
characters, passes, development interval, width and seed. Record every result,
including a negative intervention. Final-architecture arithmetic counts actual
query/key/value projections, candidate dot products, gated winner delivery,
counterfactual learning, clipping and optimizer; index/RNG/traffic are separate.
Architectural normalization via physical competition can remove the explicit
normalizing sum and clock-simulation arithmetic, but not all query/key work.
No measured joules or quality advantage is implied by this allocation model.
