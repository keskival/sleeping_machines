# Evidence access, precision and joint credit

This note interprets the completed value-credit study and states a next
comparison. It introduces no fitted architecture or queued run. Temporal
computation, sparse addressed updates, separate keys and values, persistent
state and credit to unrealized routes remain the research substrate.

## A gradient repair cannot recover evidence that is not read

For a fixed realized address, a read from a detached slot gives the derivative
of the chosen delivery under the declared surrogate. Moving its value map to
read time restores a linear-map derivative, as Note 63 verifies. It does not
change which evidence that address contains. A unique earlier cue absent from
the queried slot must reach the output through another demonstrated path.
Increasing value width or read-map credit alone cannot repair that missing
input. This is a conditional information limitation of this path, not a
limitation of the full recurrent model or the proposed temporal substrate.

The completed prefix probes establish that some old information does survive
in the core. They do not establish that both relevant bits survive in a form
usable by a trained interaction, or that the decoder has learned that
interaction. These are three distinct measurements: retention, joint usable
information and useful prediction.

## A mean and an occupied bit can hide evidence precision

The addressed reader receives a feature mean and 1[n > 0]. It stores n in
metadata but does not deliver its magnitude directly. For identical mean m,
one observation and nine observations give identical inputs to this reader.
Even a nonlinear reader cannot distinguish those inputs in isolation.

For a categorical observation with prior mass alpha and prior mean p0, the
posterior predictive mean is

    p_next = (n * m + alpha * p0) / (n + alpha).

With alpha = 2, p0 = 1/2 and m = 1, the one-observation prediction is 2/3;
the nine-observation prediction is 10/11. Mean and occupancy alone lose a
statistic needed for this particular calibrated prediction. This is an exact
counterexample for the restricted reader inputs. Core history, event time or
feature values may encode n indirectly, so it is not an impossibility theorem
for the implemented full model. Counts of learned nonstationary features are
also not automatically Bayesian sufficient statistics for text.

Pooling evidence has the same boundary. Combining slot means m_k requires
sum(n_k * m_k) / sum(n_k), when those observations share an estimand; averaging
means without n_k overweights sparsely observed slots. When contexts have
different predictive laws, even count-weighted pooling is biased. Learned
keys must discover useful sharing, rather than replacing conditional evidence
with indiscriminate averaging. These principles already motivate statistic
values in Notes 58–59; the new issue is their exact absence at this reader.

A confidence-aware delivery or timing map is a candidate repair, not a
completed explanation of the .039123 bpc late-projection deficit. Test it
against the original addressed model and unchanged late variant, keeping
hash keys and native mechanisms fixed. Report confidence inputs, additional
storage/operations and exposure explicitly. A fixed-key control cannot be
promoted into evidence for learned-key race attention.

## Counterfactual route credit can still require decoder learning

In Note 64's balanced parity construction, let independent route policies
deliver the relevant bits with probabilities r1 and r2. A bilinear decoder
with sign logit w F1 F2 has expected derivative at w = 0 equal to

    dL/dw = -r1 * r2 / 2.

At exactly w = 0, its loss is log(2) for every delivered pair. Thus even
joint route counterfactual losses are equal at that instant: their immediate
route advantages vanish, while the interaction-weight derivative can be
nonzero. Merely enumerating pairs does not solve an unlearned decoder. Once
w moves, conditional pair utility can become informative. Candidate exposure,
decoder learning and route credit must therefore be inspected together.

This is a scoped calculation for balanced signs, this decoder and independent
distractor selections. It does not show that actual random initialization,
native recurrent interaction or every alternative teacher has zero utility.
It explains why nonzero layer gradients and an exact local route formula need
not imply efficient discovery of a useful feature conjunction. Neither the
calculation nor the failed projection fit supplies a general mathematical
barrier to progress.

## Bounded next experiment and decision criteria

Prioritize the unchanged integrated AWS capacity/exposure campaign already in
HANDOFF; do not displace it with a local diagnostic control. Before another
language scale-up, a proposed local stage should use the exact balanced-prefix
construction already contracted, with held-out prefixes and identical query
suffix/count vectors. It must first establish that the complete input makes
the target identifiable and that both relevant bits reach a query-time read.

Compare full and same-width shallow native cores under matched target counts,
optimizer updates and candidate budgets. Keep original and late fixed-key
memory as labelled information-path controls. Compare unchanged local
counterfactual credit with a specified joint alternative only after its
numerical/optimizer contracts and accounting smoke pass. Record decoder and
route updates separately, along with retained feature evidence and losses;
a realized-route gradient is not an exact whole-route intervention estimator.

Target-only supervision is a different objective from whole-stream prediction;
do not silently give one model this advantage. All prefix processing, candidate
discovery, losing values, interaction updates, optimizer work and development
replay must be charged. Paired held-out targets can show learning beyond the
restricted query counts; this is still not evidence of semantic language
abstraction or resource supremacy. No new fit is admitted by this note.

Exact historical producer credit remains a separate resource question. A
naive full Jacobian for 4,096 slots of width 32 and 56,987 parameters takes
4096 * 32 * 56987 * 4 = 29,877,600,256 bytes in float32, before optimizer or
runtime memory. That is one expensive implementation, not a lower bound;
structured statistics, bounded replay or compressed eligibility may change
the tradeoff. The linear projection contraction in Note 63 is a tested
example of avoiding such a Jacobian for one parameter block. Generalizing it
to nonlinear historical producers requires another derivation and matched
evidence, not an unchecked unbounded history graph on this host.
