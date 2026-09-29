# Separate keys, values and hard race boundaries

## 194. Positive interior credit can still lose at a race boundary

E128 certifies a positive common interior direction for new context parameters.
E129 extends the conditions to all 40 class/fitting-speaker-group pairs. Its
normalized-gradient common improvement is bounded between 0.02431 and 0.02617.
The counterfactual mixture retains a positive exact predicted improvement in
every condition. However, all twelve finite radii fail the declared conditional
loss constraints, even while reducing average fitting loss. The model restores
its initial state rather than treating a failed line search as a fitted result.

E130 separates winner changes from arrival-order changes with four executions
of the same fitting-only perturbation. At radius $2.44\times10^{-5}$ the ordinary
hard model changes one winner and its worst conditional loss increases by
$3.01\times10^{-5}$. Holding winners at their baseline choices makes the worst
conditional change negative ($-1.96\times10^{-8}$). Holding arrival order alone
leaves the positive loss change. The same winner-conditioned improvement holds
at the three larger tested radii. At radius 0.001, holding winners cuts the
conditional first-order prediction error from 0.00251 to 0.0000451.

Frozen histories are diagnostic graphs, not causal deployment models. They
identify the finite-update obstruction in this tested direction: a change in
winner can replace useful content even when its interior gradient predicts
improvement. This does not establish the cause of every SHD error or prove that
all beneficial representation changes should preserve the old winners.

For a hard race with margin $m=s_r-s_k$, a parameter displacement changes the
winning value discontinuously at $m=0$ if $v_r\ne v_k$. Transport bounds inside
one cell do not bound this jump. The relevant learning record is therefore
the joint distribution of margins, alternative payload differences and
downstream utility, rather than gradient norm or margin density alone. A large
margin can protect an incorrect route; a tiny margin can expose a harmful
replacement. Neither stability nor sensitivity by itself supplies useful credit.

## 195. A separate key stream creates a smooth value-learning phase

Represent an emitted packet by $(\kappa_i,\nu_i,t_i)$: a routing key, a value
payload and an arrival time. Maintain causal local memory for keys and values,
with optional causal global value memory. At layer $l$ compute

$$r_i=\arg\min_k d_k(f_i^\kappa),\quad
 \kappa_i'=\kappa_i+\alpha K_{r_i}(f_i^\kappa),\quad
 t_i'=t_i+d_{r_i}(f_i^\kappa),\tag{195.1}$$
$$\nu_i'=\nu_i+\alpha V_{r_i}(f_i^\nu,f_i^{G,\nu}).\tag{195.2}$$

The same hard choice selects the key and value update. One paired packet and
one delay are emitted. All memories observe only arrived packets, including
the declared current-event/tie convention. There is no dense event-pair map,
hidden time grid, mixture of losing forward payloads, or per-example winner
lookup. The key computation runs on each actual new input.

Initialize the key stream from the trained local carrier and freeze its
parameters during a value-learning phase. Initialize the additional value
context columns to zero. With the existing readout, this nests the parent
predictor on the audited inputs. The key stream has no global context columns;
the demonstrated nesting uses a parent with those added columns zero.

For a fixed observed query, every value-parameter update now leaves keys,
winning choices, clocks and receiver order invariant **by construction**.
This protects function of the routing policy, rather than only its parameter
tensor: in the coupled model, frozen router weights can still select a different
winner when a learned value changes their input. The new value maps can learn
joint information while the established routing policy continues to compute.

### Depth bound across the value-learning phase

The actual key schedule fixes the value-memory operators. Each is a positive
causal averaging map with row sums at most one. With separately row-bounded
local and context maps, $F_l$ is $a_l$-Lipschitz in the packet maximum norm,
where $a_l=1$ locally and two with context. Therefore

$$\prod_l(1-a_l\beta/L)\|x-y\|_\infty
 \le\|\Phi(x)-\Phi(y)\|_\infty
 \le\prod_l(1+a_l\beta/L)\|x-y\|_\infty.\tag{195.3}$$

Unlike the coupled value/key update, the schedule remains fixed with respect
to all value-map updates in this phase. The conditional transport certificate
thus applies throughout their bounded parameter region for the supplied query,
without an accidental value-induced route crossing. It bounds value-state and
adjoint transport; individual parameter directions can still lack support,
and it is not a global learning/generalization theorem. If key parameters or
observed inputs change, their schedule conditions must be analyzed separately.

E131 verifies exact initial logits, summaries, winners and clocks on its real
checkpoint batch. New value-map credit has norm 8.2404. At radius 0.0001 its
predicted loss change is -0.0008240 and the realized change is -0.0008225.
Small and large value perturbations leave every audited key choice and clock
exactly unchanged. Query separation and checkpoint roundtrip also pass.

The prototype adds 52,616 frozen key parameters and trains 6,336 added value
scalars. Key and value streams each execute selected local maps; their memory,
sorting, map and storage costs are charged. Both streams are event-driven with
linear memory-state work and no event-pair quadratic attention. This experiment
does not claim that duplicating a key stream reduces inference energy. Key
compression, trainable key policy and persistent state are separate next steps.

## 196. Learning keys requires the joint counterfactual computation

The value-learning phase optimizes its emitted value maps. A later key-policy
phase must credit alternatives in the joint packet and its actual suffix.
For vector teachers $g_\kappa,g_\nu$ and clock teacher $q$, a local alternative
score is

$$A_k=\alpha g_\kappa^T(\kappa_k-\kappa_r)
       +\alpha g_\nu^T(\nu_k-\nu_r)+q(d_k-d_r).\tag{196.1}$$

This is an interior linearization of alternative utility. A realized paired
suffix replay instead measures $U_k=L(\mathcal H_k)-L(\mathcal H_r)$, including
changed later winners and arrivals. With differentiable relaxed route weights,
the policy teacher is $\sum_k \nabla p_k\,\operatorname{sg}(U_k)$ under the
declared proposal/support law. At fixed common clock and smooth suffix, a
curvature bound controls $U_k-A_k$. When the alternative crosses downstream
hard races, that local Taylor bound is inapplicable and the paired result
captures the additional jump.

Reachability matters: an overridden history is a candidate only if a declared
clock/gate/content change can realize it within the model's delay and causal
bounds. A forced route is a diagnostic until that eligibility is established.
Use a bounded, logged near-margin proposal budget and measured downstream
utility; account for its replay cost. The scheme must preserve both local
ordinary value learning and credit to losing joint alternatives.

Future optionality can replace immediate $U_k$ with independently measured
post-adaptation continuation utility at a stated horizon. It need not demand
an immediate loss decrease. That quantity remains distinct from the smooth
value-phase descent certificate. Key/value separation supplies controllable
learning directions; properly supported policy credit must decide when a new
hard computation is better.
