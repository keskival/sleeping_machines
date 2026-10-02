# Joint clock/content/memory credit and training support

The user's interpretation is right about the desired coupling: incoming
representations parameterize clock rates and candidate contents, and previous
times affect rotation, decay, causal joins and memory ages. It does not follow
that every payload direction changes a current route: key/value separation,
projection nullspaces and saturated/clamped scores can leave timing unchanged.
Useful selective coupling is the objective, not a requirement that every
content perturbation move every clock.

## What the completed model actually learns

The frozen full DVS fit changes35.7949% of16,128 dev hard choices from its
initialization under identical eval draws. Route/message/time parameter groups
all change and receive nonzero local gradients. Removing sampled score/time
edges leaves forward logits exactly unchanged while changing gradients in
incoming/context, message and time maps. This demonstrates coupled training
paths; it is not an orthogonal gradient decomposition or an accuracy ablation.
Mean race entropy is.176990nats versus maximum log(2)=.693147;13.5851% of
candidate scores reach the clamp. Concentration can represent specialization;
the audit alone does not prove collapse or identify a dominant bottleneck.
Terminal pair risk supplies nonzero direct losing-value credit, as intended.

The unchanged256-fit/four-pass seed6 comparison fails its predeclared NLL
gate: local1.3059375/54.1667%, terminal pairs1.3498068/56.7708%. Both select
pass4. A better accuracy component does not satisfy the fixed.02-NLL gate.
No second fitted seed or longer credit campaign is admitted on this result.
The saved984-fit strong/compact controls remain stronger and have different
fitting data. Do not manufacture a resource advantage from those denominators.

## Turning unrealized histories into faster/slower emitter derivatives

At one isolated race, s_i=f_i(x,m;theta), lambda_i=exp(s_i), Lambda=sum lambda,
raw T_i=E_i/lambda_i. Increasing s_i reduces raw waiting time; physical delay
is the existing monotone d(T)=.001+.010T/(1+T). Its realized interior derivative
is -d'(T)T for the winner and zero for losers until their ordering changes.
Ordinary backprop then reaches theta and the shared content/memory arguments.

For fixed entering prefix and raw first time T, actual alternative histories
have losses F_i(T,xi), where each route delivers its message, commits its
memory/address/metadata and continues with the same independent future draws.
R_T=sum_i pi_i F_i, pi_i=lambda_i/Lambda. The categorical component is

    g_choice,i=pi_i(F_i-R_T).

A lower-loss alternative receives negative score credit, so gradient descent
makes it faster relative to the other candidates. Candidate construction and
shared arguments also receive their direct content/state derivatives. This is
one joint objective, not independent route and weight optimization problems.
The components conjoin in shared parameters; none guarantees a favorable
global optimizer step when other examples and parameters also change.

Relative-choice credit is not all clock credit. The joint raw winner/time
density lambda_W exp(-Lambda T) yields, as derived in theory57,

    g_joint = E[(e_W-lambda*T) F_W(T,xi)].

Conditional winner enumeration gives

    g_joint,i(T)=pi_i F_i(T,xi)-lambda_i T R_T.

The choice component sums to zero along a common score shift; time-scale
credit generally does not. For a smooth conditional suffix one may instead
combine choice credit with pathwise timing derivatives. A discontinuous
downstream event/route suffix also has timing boundary terms. The likelihood
form handles those under its local fixed-prefix assumptions without assuming
that the interior derivative captures a jump. Neither expression is an exact
whole-model gradient while earlier routes remain locally approximated.

## Concrete missing-write evidence in the current real-stream model

The new frozen audit evaluates4 previously used dev clips, events9/19, both
layers and heads:32 sites/model,64 legal alternatives plus64 diagnostic hybrids.
Raw first time stays fixed, and actual alternative payload and memory commit
are propagated through the whole suffix. This repeats the factorial method
in theory57 on the current native DVS model, not a shadow decoder.

In the full-fit clip2/event9/layer1/head0 probe, alternative delivery alone
changes NLL by-.010163; alternate write alone by+.105139; their interaction
by-.003049; the legal full route by+.091928. The value teacher favors the
alternative although its actual suffix is worse. Mean absolute effects over
all32 sites are.016994 delivery,.006022 write,.006806 interaction; these
absolute summaries are not additive causal percentages.4of15 nonzero
comparable directions oppose full conditional categorical credit. Local and
pair pilots each have5of24 opposed directions on these sites.

This establishes a concrete estimator blind spot, not its prevalence over
unseen inputs or its share of the benchmark gap. Some loss differences and
single policy-step effects are tiny; retain raw values beside direction counts.
The first4 dev clips are correctly classified in the full fit; the audit tests
NLL credit and is not a targeted-error sample or independent confirmation.
Changing the memory can alter future keys, values, clock times and routes; an
immediate value-only Taylor teacher omits that complete alternative history.

## Breadth, depth and joint order are separate learning budgets

Record four quantities: eligible candidate identities C; evaluated/trained
candidate values V; temporal suffix replay horizon r; joint decision order q.
Also record inference winners, scored keys, writes and stored capacity. The
current native DVS race hasC=2 per head, not two historical-token alternatives:
each receiver carries persistent state. Increasing C changes available state
and learning exposure unless maps/state support are carefully controlled.
Increasing r may repair delayed-memory credit without increasing C. Increasing
q tests interacting simultaneous choices, with combinatorial fitting cost.
These are different experimental interventions.

Dense attention forms y=sum_i a_i v_i, a=softmax(s). Holding values independent
of scores, its derivative is a_i g.(v_i-y), and value credit is a_i g. These are
exact derivatives of a mixed forward output, not exact losses for alternate
hard histories. Every allowed position contributes; tiny weights can still
make effective gradient support narrow. [Attention](https://arxiv.org/html/1706.03762v7).

Sparse attention has that direct support only within its pattern, while deeper
paths or summaries can extend access. The original factorized design uses
roughly sqrt(N) positions per head for two-step global connectivity.
[Sparse Transformer](https://arxiv.org/html/1904.10509v1).
Native Sparse Attention separately uses compressed summaries, selected token
blocks and a local window; summary coverage differs from fine-grained value
coverage. [NSA](https://arxiv.org/html/2502.11089v1).

Top-k MoE commonly scores M experts but evaluates k outputs. The selected
softmax gate and load-balancing objective can give router gradients across
expert logits without knowing unselected expert output utility; unselected
expert networks receive no direct task-output gradient for that token.
MoE expert support is distinct from surrounding attention context support.
[Switch](https://www.jmlr.org/papers/v23/21-0998.html).

## Broad early support, sparse inference, and clock precision

There are precedents for an early broad gate followed by sparsification:
EvoMoE trains with a high-temperature dense-to-sparse gate and transitions to
top1. That changes actual activated experts, not just a training-only shadow
budget. [EvoMoE](https://arxiv.org/pdf/2112.14397).
Our prospective comparison should instead hold inference winner count and
eligible state fixed while varying paid counterfactual replay/discovery support.
Preserve a full-support exploration proposal; late useful features may need
rediscovery, so monotone shrinkage is a hypothesis, not an optimization law.

Changing our routing temperature changes physical raw rates unless controlled.
One prospective clock-preserving transformation is

    s'_i = logsumexp(s) + s_i/tau - logsumexp(s/tau).

It keeps sum exp(s')=sum exp(s), hence the raw first-time distribution, while
changing categorical choice entropy. It does not preserve individual
sample-path clocks or downstream histories. It has not been fitted or admitted.

For a fixed local prefix and raw T, sample candidate I from full-support q.
The estimator pi_I/q_I * F_I * (e_I-pi) has the exact conditional choice mean.
Joint clock credit instead uses pi_I/q_I * F_I * (e_I-lambda*T). Charge the
actual branch replay. A prefix-only baseline and residual control variate can
reduce variance under theory57's independence conditions; detaching a
winner/time-dependent suffix adjoint does not establish that independence.
Shrink shadow count based on measured variance and retained candidate coverage,
not only winner confidence. No sampler is installed by this note.

Frozen inference can omit training counterfactual evaluation. Actual discovery,
key scoring, reads, timing and writes remain. Online learning can use sampled
score-function credit or richer shadows; full candidate enumeration is not a
mathematical requirement, but a quality/variance/work choice.

## Progress criterion

No general impossibility for the temporal/sparse substrate is established.
Restricted information bounds and explicit teacher failures are real; they do
not prove a global architectural ceiling. Tuning alone has not been shown
sufficient either. The next integrated repair should test state-aware or paid
suffix-residual credit against the preserved value-only estimator, at unchanged
capacity/data/inference activity, after derivative/recovery/learning contracts.
Wider associative candidate discovery is a separate comparison. Preserve the
architectural case and strongest controls; claim advantage only from completed
comparable-quality/data and resource evidence.
