# Lazy memory coordinates, actual write utility and the divergence diagnosis

3 October 2026. Read140/141 and §§413 before changing the current operator.
This note adds reproducible conditional-credit witnesses; it changes no
active model, driver, loss, optimizer or queue.

## Completed progress and the precise unresolved failure

The completed p32/D4/H2/pool2 one-pass10M native language arm with message
linearized categorical credit improves test2.507→**2.370 bpc**, ahead of the
saved matched-one-pass Transformer2.427 and no-selection control2.439.
Parameters108875, same1220updates/9994240sampled fitting positions; counted
fitting work rises.722→.725MFLOPs/position. Retain this positive integrated
result and the supported raw estimated work gap. It remains a single-seed
comparison with different work-estimation conventions and trails LSTM2.171.
At the controls' matched T256 window, native2.371491 versus Transformer2.427
retains .0554bpc advantage; the.057 difference uses nativeT128. The
report appendix shows both window lengths and explicit state/key/value/
selected-write counts alongside the same-unit whole-fit/per-position work.

The subsequent `linear_rw` fit failed around window250 with nonfinite
gradients. The producer's job log is unavailable here; the shared §413
account and absent completed result are retained as a failed fit, never a
completed quality cell. Its memory coefficient is G_j·(m_new,j-m_j).
`linear_rwn` now drops the transported-old-memory change and uses only
G_j·written_j. Both keep the same hard forward state/messages/times, with
training-only auxiliary score credit. The v5 comparison remains priority.

Lazy coordinates and omitted timestamps are a plausible instability
mechanism, but divergence alone does not establish that causal explanation.
There are also factual memory magnitudes, long BPTT cotangents, surrogate
curvature, emitter coupling, clipping/Adam and actual future key effects.
The following cases distinguish these, rather than treating every old-value
transport change as fictitious or every newly-written value as bounded.

## When a refresh is genuinely only a coordinate change

For one scalar/complex pair with fixed decay r, forget f and frequency w,
write the homogeneous transport as A(a)=exp((-r*f+i*w)*a). A stored pair
(m,t) read at b gives A(b-t)*m. With zero new content, resetting it at a to
(A(a-t)*m,a) leaves that read identical because

    A(b-a)*A(a-t)*m = A(b-t)*m.

For the scalar case, along the exact continuous coordinate transformation

    t(alpha)=t+alpha*(a-t), m(alpha)=exp(-r*f*alpha*(a-t))*m,
    g_t=r*f*m(alpha)*g_m,
    g_m*dm/dalpha + g_t*dt/dalpha = 0.

Memory-only local credit discards that cancellation. Two already-seen
slots m=(1,.5), t=(0,0), r=(.5,.25), f=1, reset at2 and read at5 give equal
true affine conditional losses at both routes. The exact score gradient is
zero, while the stored-memory-only surrogate at factual winner0 is
(-.0211699337,+.0211699337). Written-only is zero here, correctly.
Enumerating both factual winners gives expected stored-only credit
(-.0054545590,+.0054545590), still biased despite zero exact utility.

Adding a finite stamp delta to a finite memory delta is still only a Taylor
approximation. The exact cancelling tangent follows the curved coordinate
path; a single endpoint linearization can be nonzero even for this pure
rebase. A common-time coordinate comparison or actual finite suffix loss
observes exact equality; a generic linear stamp auxiliary does not prove it.

## Why that cancellation is not generally a native forward invariant

**Stored keys.** The current eager/compiled body reads
`key + key_read @ m` before candidate age transport. A reset changes that
stored-key argument. There is no timestamp compensation on this path.
In an affine conditional loss on the stored feature sum, the same two-slot
witness has exact score gradient(-.1088464722,+.1088464722); stored-coordinate
memory credit recovers it exactly, and written-only returns zero. This is
a constructed read-path witness, not a measured language gradient. It
demonstrates that a stored-coordinate change need not be fictitious when
downstream computation consumes those coordinates directly.

**Changing forget gates.** The native candidate multiplies the entire age
by the *current* input-dependent forget. For current f_a and later f_b,
an intermediate zero-input refresh gives

    A_fb(b-a)*A_fa(a-t)*m,

whereas an unrefreshed later read gives A_fb(b-t)*m. Their decay ratio is
exp(-r*(f_a-f_b)*(a-t)); the rotations still compose. They differ unless
the generators agree or the state contribution is zero. With f_a=.2 and
f_b=1, the same witness's exact score0 gradient is **+.00753586447**; the
memory-only linear surrogate is **-.00670359200**, and written-only is zero.
Its mean over both factual winners is **-.00240225543**, also opposite to
the exact positive gradient: this is a conditional bias, not only the sign
of one noisy outcome.
Thus omitted timestamp effects can reverse a local sign, while dropping
all homogeneous-state effects can remove useful credit as well.

All witnesses keep messages and first times fixed, and all slots are
already seen. They deliberately exclude bool readiness/topology changes,
future hard route boundaries and parameter pullbacks. Actual forced-write
suffix replay observes those paths; a memory linearization does not become
an exact sequence teacher merely by being more stable.

## Bounds and calibration: what is actually controlled

For the native contractive homogeneous operator, ||A||<=1 and

    ||(A-I)*m|| <= 2*||m||,
    ||m_new-m|| <= 2*||m|| + ||written||.

Longer gaps can increase a particular transport delta, but they do not
create an unbounded delta at fixed memory norm. Memory itself and its loss
cotangent can be large. Rotations make monotone growth in gap invalid too.
Written-only removes that memory-dependent term; it does not bound the
remaining gradient.

The gate satisfies0<write<2, but the written content is write*(Input*x_u).
`x_u` comes from the unnormalized channel-mix output. LayerNorm is applied
to the control inputs and later message read, not to this write operand;
Input and channel mix are trainable maps. Consequently

    ||written|| <= 2*||Input||*||x_u||

is a conditional bound, not a uniform architecture bound. A zero control
map gives write=1, and scaling incoming (1,-1) by1000 scales the write
norm by1000 despite normalized control inputs. The shared observation of a
smaller write-credit norm is useful empirical calibration evidence, not a
proof that the revised operator cannot diverge.

At fixed prefix/time, exact conditional route credit is pi_i*(Q_i-Qbar).
If local coefficients a_i approximate Q_i up to a shared baseline, define
residual e_i=Q_i-a_i. The error in the categorical score teacher is
pi_i*(e_i-E_pi e), with squared pi-whitened norm **Var_pi(e)**. For a smooth
fixed topology, a full-state first-order Taylor residual is bounded by
(H/2)*||z_i-z_factual||^2 under a Hessian norm bound H. Omitted coordinates
add their first-order terms; discrete readiness/routes have no such smooth
bound. Pullback through the actual emitter Jacobian and clipping/warm Adam
must be measured separately (85/138/139). A smaller coefficient norm alone
does not establish a more accurate teacher or a better parameter update.
Positive scalar attenuation reduces an opposed component's magnitude but
does not fix its sign. In the changing-forget witness, any positive scale
of the isolated biased teacher still ascends the exact conditional risk to
first order. Recover the missing utility before assuming scale tuning alone
resolves the problem; the full architecture has the required state paths.

## Bounded next decision, without displacing the active fits

1. Finish the integrated v5 credit/capacity/width/depth arms and AWSr2
   contracts/pilots. Keep `linear` as the completed valid reference; preserve
   failed `linear_rw` and pending `linear_rwn` status distinctly.
2. Before further write-credit changes, use the same actual saved trained
   weights on fixed FIT contexts, not heldout labels or reconstructed states.
   Sample declared small sites; compare message-only, full stored-memory and
   written-only coefficients with actual alternative-write suffix losses at
   fixed first time and identical future draws. Include real timestamps and
   seen bits. Retain the omitted homogeneous contribution separately.
3. Charge all factual/shadow passes and backward/optimizer work. Measure
   parameter-space alignment and actual warm-Adam prediction changes, memory
   norms/gaps/forget values, score boundary exposure and component cotangents.
   Use only independent FIT pilot statistics for any later scale choice;
   per-draw norm balancing changes bias and is not automatically descent.

No additional native job is launched or new training variant queued from
this context. Physical curie remains occupied/reserved; its checkpoint and
runner log are unavailable here. This is a measurement/derivation repair
within the retained temporal/race/private-state architecture, not a proposal
to replace it with dense or synchronous computation.

`python3 experiments/check_lazy_write_geometry.py` reproduces all six
standard-library contract classes, maximum score finite-difference error
2.419e-12. It writes no result file and loads no Torch/NumPy. These are
ideal scalar/complex conditional-loss witnesses, not native gradients,
divergence attribution or a new benchmark claim.
