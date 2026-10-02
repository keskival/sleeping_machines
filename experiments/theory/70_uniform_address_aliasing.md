# Uniform address aliasing and the joint-learning cold start

The protected-outcome experiment demonstrates an attainable relation, but
seed7 starts learning much later than seed6. Its final confirmation is still
pending when this derivation is written. This note explains a possible
optimization bottleneck without attributing every delay to it.

## A distributional alias, not just a small mean

An episode is24,b1,1-b1,25,b2,1-b2,n0,...,n7,26, with noise symbols2..23.
The causal last-successor bank at the query has

    outcome[24] = b1,       outcome[25] = b2,
    outcome[b2] = 1-b2,     outcome[1-b2] = n0.

All remaining occupied addresses and outcomes depend only on the common
noise suffix. Both0 and1 are occupied regardless of either bit. Hence the
multiset of the four displayed outcomes is

    {b1, b2, 1-b2, n0} = {b1, 0, 1, n0}.

It is independent of b2 conditional on b1 and the suffix. The same is true
of the whole outcome multiset and of its occupied count C. For ANY learned
symbol interpretation v, a uniformly sampled delivered value has the same
distribution for the two b2 alternatives. Two independent uniform reads
also have identical joint value distributions. This is stronger than saying
their mean is equal: no decoder of those delivered values can recover b2.
If the decoder's additional context does not itself expose b2, the balanced
relation Y=b1 XOR b2 remains conditionally uniform, with one-bit entropy.

This is a statement about uniform addressed reads under the specified
information condition. It is not a theorem that the native context lacks
b2, that the observed bank lacks it, or that all learned reads fail. Address
identities distinguish the evidence; discarding them in uniform value pooling
creates the alias. Native timing and protected address contents still exist.

## Why exact terminal gradients can still start slowly

The decoder residual begins at zero to nest the original parent prediction.
Every pair then has the same logit z_ij=base(h), so its loss is identical.
For either categorical policy, the exact conditional derivative

    dL/ds_i = pi_i * (E[loss | first=i] - L)

is zero at that initialization, regardless of key asymmetry. Key learning
starts only after decoder values distinguish candidate pairs. If policies
are exactly uniform and context exposes no b2, the decoder's balanced
population signal cannot reveal the relation either. In particular, with
fixed uninformative context and zero base logit, zero residual is a terminal
stationary point: each decoder gradient is a balanced label times a feature
whose distribution is independent of b2, and each key gradient is zero.
This is not a stationary-point proof for the entire trainable native core.
A terminal information-poor stationary point can coexist with correct
gradients and a useful representable solution. No generic gradient
implementation bug is needed for this cold start.

Current keys/queries are randomly asymmetric, not exactly uniform. They can
break the alias and produce decoder signal. The amount and alignment of
that initial distinction may vary by seed; empirical late transitions
are compatible with that explanation, but do not identify it uniquely.
An actual conditioning experiment should vary target-independent key/query
contrast while retaining zero-output nesting, candidate coverage, the same
decoder and native mechanisms, and compare all fit seeds under a new fixed
protocol. Do not extend or tune current confirmation fits after their scores.

## Existence of a useful solution

Keys in four dimensions can separate marker24 and25 for two queries without
using targets. Select24 with the first race and25 with the second; interpret
symbols0/1 as signed values -1/+1 in one value dimension. A bilinear decoder
z=-k*v1*v2 predicts the XOR relation, with correctly retrieved conditional
logistic loss tending to zero as k increases. With finite score contrast24
under the current clamps, each
correct-marker probability is1/(1+(C-1)*exp(-24)); sparse terminal delivery
can approach deterministic useful retrieval. The finite clamp leaves a
nonzero routing error floor, so unconditional expected loss need not tend
to zero as k grows and its optimum generally uses finite confidence.
This is an existence witness,
not a fitted score or a proposed marker-filtered implementation. Actual
seed6 routes often choose24/0 or24/1 and exploit other observed successor
information, so their learned behavior must be reported as measured.

The stationary alias and the useful solution coexist. Architecture and
credit can make progress; optimization must connect them. Protecting more
raw evidence alone, exact conditional derivatives alone, and adding depth
alone do not guarantee that connection. The completed joint/local and
shallow controls price these distinctions; broader contextual evidence and
learned write-route coverage remain the integrated research priority.
