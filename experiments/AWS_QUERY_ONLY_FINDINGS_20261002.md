# Exact query-only native inference: fewer operations, slower observed wall time

Same three saved coarse4/.25-clock native producers, no new training. Only the
classifier is admitted at the explicit observed terminal query. All five core
events, physical clocks, races, candidate scoring, messages and selected writes
remain. Original intermediate logits have no feedback into the core.

| Seed | Accuracy % | NLL | Whole fit GF | Fit MF/target | Original infer MF/prefix | Query-only infer MF/prefix |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 66.667 | .890983 | 4.218015 | .535825 | .138119 | .135259 |
| 7 | 61.979 | 1.056934 | 4.218015 | .535825 | .138007 | .135147 |
| 8 | 69.271 | .920744 | 4.218015 | .535825 | .138035 | .135175 |

Each quality/state comparison covers all192 DEV prefixes, bitwise logits and
ALL persistent state identical, parameter dictionaries identical. Saved original
probabilities match within established FP tolerance. Target mutation, repeated
calls, missing-query and training-rejection contracts pass.Exactly576
paired full-prefix logits/state comparisons, three192-prefix counterbalanced
wall repeats per implementation and seed. No official test or quality selection.

Executed ATen ledger first11 prefixes/seed pays ALLcore operations and special
functions with full formula coverage, 2FLOPs/MAC convention. Eliminating four
32x11 affine classifiers saves2860 counted operations/prefix (2.071–2.072%).
Same33-channel observations, p16/L2/H2/pool2,8 available receivers,4selected
writes/event; no sparse discovery or losing-value work is silently removed.
This is an exact arithmetic reduction, not a practical wall-time advantage:
measured wrapper is about2% SLOWER on this Python path. Timing dispatch overhead
outweighs the small affine saving. All raw timings retained; traffic and energy
unknown. Native fitting/parameters unchanged; inference-only wrapper rejects
training and nonterminal/missing observed queries. It cannot be substituted for
ongoing-label or silence-supervised streams without a separate contract.

65.025s/465880KiB,unique guarded queue,2GiB RSS/6GiB virtual,8GiB available
floor,600s timeout,oneCPUthread. No general architectural replacement or
supremacy claim. A lower-overhead native query gate might make the arithmetic
saving practical, but that is a future engineering hypothesis, not this result.
