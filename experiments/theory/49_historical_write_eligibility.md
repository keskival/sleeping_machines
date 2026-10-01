# Historical write eligibility and delayed sparse learning

## 326. Available history is not learned history

**Failure addressed.** The completed H2/H4 eight-block language fits missed
the saved indexed-control gate (§325). The matched H2 2K pilot with credit64
reached 3.740341 development bpc versus 3.732586 for credit16: extending the
graph alone did not help. Frozen tests nevertheless identify a real limitation:
sealed historical K/V entries can affect a prediction but its loss cannot teach
their old write maps. The retained source carry and channel mixing are useful
in the frozen interventions, so this change preserves them.

**Retained construction.** Independent per-head Q/K/V and temporal races;
addressed persistent receiver updates; analytic temporal evolution; incoming
content and learned cross-head mixing; all indexed history; losing-route
counterfactual score credit. Forward keys, values, routes, delays and RNG are
unchanged at fixed weights. The change adds a detached feature per write and
direct delayed credit to sealed write maps. It does not replace races with dense
attention or reopen the old receiver/representation graph.

At write u, save the already computed normalized feature φ_u alongside
k_u = W_K^(u) φ_u and v_u = W_V^(u) φ_u. At query t, the actual cached values
remain authoritative. Define a *conditional historical perturbation* Δ:

    k_u(Δ_K) = k_u + Δ_K φ_u,
    v_u(Δ_V) = v_u + Δ_V φ_u.

This hypothetical perturbation acts on old writes with features held fixed.
Its derivative at Δ=0 is transported to the current shared write map; it is
not the derivative of the actual cached primal with respect to today's map.
Old weight versions and omitted representation paths make this a delayed local
surrogate. Linear outer-product eligibility is an established learning
primitive, not a standalone novelty claim. The research question is whether
this compact credit works inside the integrated temporal/sparse construction.

## 327. One key-map adjoint for a whole admitted race

Let q be the query, C_o the number of admitted sealed keys, and e_u the local
upstream derivative of raw q·k_u, including score scaling, clipping, recency
and the existing temporal-route teacher. For credit strength α:

    g_K = α Σ_u e_u q φ_uᵀ
        = α q (Σ_u e_u φ_u)ᵀ.

The shared query factors the producer teacher into one weighted feature sum
and one matrix outer product: Θ(C_o d + d²), rather than Θ(C_o d²).
The ordinary query and live-key adjoints stay intact. Only sealed writes receive
this additional teacher, avoiding double credit to live writes.

If historical winner w delivers a value with upstream vector adjoint a:

    g_V = α a φ_wᵀ.

The extra value-map work is Θ(d²) for the winner. All admitted losing values
remain charged under the existing score teacher. No saved φ receives a gradient,
and no whole-history autograd graph is claimed. The local score teacher itself
is still a conserved counterfactual surrogate, not an exact deep boundary
gradient. α=0 must exactly nest the frozen parent through forward, gradients,
RNG and multiple Adam windows; α=1 and α=.25 test full and tempered transport.

## 328. Storage, stale writes and the experiment gate

One saved d-vector makes the float K/V/eligibility payload 3d instead of 2d:
**50% extra tensor storage**, with unchanged integer positions/index. Feature
copies and reads are traffic, even when they are not floating arithmetic.
The emulator retains eligibility during evaluation for online-ready state;
evaluation introduces no extra key/value arithmetic or learning feature reads.
An inference-only deployment could omit eligibility but is not the measured
storage path. Additional training FLOPs, clipping, optimizer and allocations
are audited; a cheaper clock does not erase those costs.

A limited conditioning bound helps describe stale writes. Hold q and φ fixed,
let ||q||≤Q, ||φ||≤F, and ||W_K^(t)-W_K^(u)||₂≤η_K. Then

    |δ score_u| ≤ QFη_K / √d = ε_s.

The bounded score transform is nonexpansive. For softmax winner probabilities,
||δp||₁≤2ε_s conservatively. With ||v_u||≤V and value-map drift ≤η_V,
the change of a *linear expected retrieval* is at most 2Vε_s+Fη_V.
This is not a gradient-bias bound, a hard-route pathwise bound, or a bound on
deep evolving-state predictions; query/state drift is excluded. Tempering is
an experiment, not a consequence guaranteeing descent.

Protocol: first fixed-feature perturbation/finite-difference tests, exact parent
nesting and no-double-credit tests; then guarded full-depth update/recovery
contracts and small fits. Reuse the completed H2 d32/depth8/pool2, credit16,
U64/lr.002/warm512, seed6 four-pass 2K reference only after exact nesting.
Run matched α=1 and .25 pilots on the same disjoint 8K development set.
Promote only a completed ≥.02 bpc pilot gain to one 8K fit. Larger fits require
the existing indexed-control +.10 bpc gate, measured memory and time. Preserve
negative evidence. This tests delayed local credit, not unlimited learned
history, ASIC energy or frontier supremacy.

## 329. Content contrasts and the common clock mode

A temporal race has a degree of freedom beyond its categorical probabilities.
For unsaturated scores s_i and fixed exponential noise E_i, adding the same b
to every score leaves the winner unchanged but rescales its raw arrival:

    p_i(s+b1) = p_i(s),    T(s+b1) = exp(-b) T(s).

For the implemented bounded delay D(T)=.001+.010 T/(1+T):

    dD/db = -.010 T/(1+T)².

The conserved content counterfactual teacher has Σ_i e_i^content=0. The
interior timing teacher contributes Σ_i e_i^time = a_D dD/db. Thus probability
invariance does not make the common score mode useless: time can change
receiver evolution and when messages are integrated. No extra value is read
to express this common clock change. The CPU still pays the numerical clock
and state-evolution costs; it is not an energy measurement.

When all admitted entries are sealed, their direct key-map teacher can be
decomposed around any feature center μ:

    g_K = (α/√d) q [Σ_i e_i(φ_i-μ) + μ Σ_i e_i]ᵀ.

The contrast term teaches which historical feature should win; the mean term
can teach the common clock mode. This is an identity, not a new estimator or
implemented centering optimization. With mixed live/sealed entries, conservation
is over the whole race, not the old subset; unequal α weights and score clipping
also change the cancellation. Do not silently zero the mean term in the new
teacher. Receiver/transport nonlinearities and deep route boundaries retain
the limitations above. A read-only common-score-shift test verifies winner
invariance, delay rescaling and the timing/content adjoint sums before claiming
this degree of freedom; its quality contribution remains unmeasured.
