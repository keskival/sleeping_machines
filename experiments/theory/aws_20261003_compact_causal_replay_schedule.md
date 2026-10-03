# Compact causal replay: retain prefix savings without repeated suffix loops

## Failure addressed and mechanisms retained

The completed118/119 causal-prefix contracts preserve every double gradient,
factual state/RNG and pending Adam recovery. They reduce shadow events, but
production T16 CPU wall regresses: private winner-reuse1.221s versus grouped
cache2.579s; shared1.218s versus2.616s. Grouping replay by start token launches
16+15+...+1=136 token-loop iterations instead of16. This implementation is
NOT promoted. Source: theory119 and its frozen completed resource JSON.

A proposed sibling kernel keeps the SAME exact conditional winner estimator,
all losing alternatives, temporal races, separate keys/values, private evolving
receiver memory and factual-only differentiable content path. It changes only
the no-grad shadow schedule. It does not implement the new live losing-content
policy; that distinct estimator has its own negative relative admission131.
No changes to currently active10M sources or protocols follow from this note.

## One chronological loop with growing active lanes

For T tokens, L layers, H heads and U choices, index each losing lane by
(tau,d,h,i), where tau is the factual token of the forced race and i differs
from its factual winner. There are LH(U-1) lanes activated at each tau.

1. Run the factual chunk exactly as today, caching detached PRE-token state
   and the shared RNG boundary at each token. Record winner and score vectors.
2. Allocate stable global lane identifiers; retain a compact active-prefix
   layout ordered by activation token tau. Stable identifiers preserve the
   mapping from returns to scored routes even when tensor rows change.
3. At absolute token t, append ONLY the lanes with tau=t. Initialize their
   memories, arrival times, source contexts, presence flags and queue state
   from factual snapshot t. Older active lanes retain THEIR evolving state.
4. Execute one native token step over all active lanes. Use the SAME factual
   absolute-token/head/layer noise as full replay, shared across lanes. Apply
   the forced identity only when (t,d,h) equals a lane's designated race.
5. Accumulate detached target losses only for active lanes, then proceed to t+1.
   Every lane is alive until chunk end; there is no termination/pruning rule.
6. Scatter the resulting suffix returns into the existing route-return table;
   reuse factual winner returns. Restore the factual end RNG and unchanged
   factual state. Build the SAME route-credit objective and factual backward.

This schedules T token-loop iterations, with active batch size
  A_t = LH(U-1)(t+1).
Total shadow event-lanes are
  sum_t A_t = LH(U-1)T(T+1)/2,
versus winner reuse without prefix caching
  LH(U-1)T^2.
Their ratio is (T+1)/(2T). At L8/H2/U2/T16 this is2176 versus4096 shadow
events, with256 total lanes,16 token-loop iterations rather than136. The
original all-choice implementation has8192 shadow events. These are EXACT
combinatorial schedule counts, not measured whole-fit FLOPs or CPU speedups.

## Important constraints and costs

Matching a start-token seed alone is insufficient if any draw depends on batch
size. Cache or regenerate the factual absolute-token noise schedule, and assert
all shared draws, force locations and shadow race histories. Clock times and
arrival timestamps remain absolute; force indexing must not reset incorrectly
at lane activation. New lanes need complete snapshot state, not memory alone.
No stale scratch tensors may cross activation or chronological boundaries.

Compaction must happen BEFORE payload/race arithmetic. Padding every lane and
masking its output afterward reinstates prefix work and does not implement
these savings. A growing prefix view can avoid repeatedly copying old state;
initialization, gathers/scatters, cached snapshot storage, RNG recording,
allocation and any hidden full-size operations still need accounting. The
maximum live state is still256 lanes; lower event work does not imply half RSS.
The factual backward, optimizer, discovery/key scoring and counterfactual table
must be counted under the same full-fit/per-target boundaries as controls.

Different growing batch shapes can change float32 rounding; exact estimator
identity does not imply bitwise old/new float32 updates. Preserve116/117 gates
and precision qualifications. No statistical sampling or route is removed.

## Required admission

A unique guarded sibling diagnostic must compare original/all-choice,
winner-reuse, grouped-prefix and compact-prefix on fixed private/shared
nonempty state. Test T1/T3 and productionT16, same represented float32/double
weights, every parameter gradient, actual forced identities, ALL shadow route
histories, factual state/logits/end RNG and paid traced/untraced optimizer
recovery. Preserve old strict coordinate outcomes beside global/update errors.
Benchmark actual one-thread median wall and RSS at production shapes; charge
all forward/shadow/backward/normalization/clip/Adam work. Only a completed
learning/work smoke with no wall regression can nominate a new separately
named >=10M arm. No current winner-reuse fit is restarted for this hypothesis.

All three AWS guarded slots remain occupied. This note is a derived schedule
and admission proposal, not an executed kernel or queued quality experiment.
The already queued centering diagnostic remains unchanged.
