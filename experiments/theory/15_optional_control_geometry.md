# Optionality as a reachable correction geometry

29 September 2026. This develops §§159–163 into a tractable local geometric
surrogate. It is a derivation and a small numerical contract, **not an SHD
training result**. The route/time history must remain fixed for its linear
model, or a separate model must be formed for each realizable history.

## 171. A budgeted correction ellipsoid, not raw route variance

Let u collect admissible trainable controls for a route and let its local
logit response on a declared example set be delta z=J u. Give controls an
explicit cost u^T M^{-1} u <= b, M positive definite. The set of reachable
corrections is an ellipsoid (possibly confined to a subspace), with Gramian

\[
 G=J M J^T.
\]

Its support in a future correction direction d is exactly

\[
 h(d)=\sup_{u^TM^{-1}u\le b}d^TJu
      =\sqrt{b\,d^TGd}.
\]

Proof: write u=M^(1/2)v and use Cauchy–Schwarz; equality holds along
M^(1/2)J^T d unless that vector vanishes. This makes “more options for later
correction” quantitative: they must cover relevant directions at feasible
control cost. Noise in unreachable directions and duplicate route names
do not increase the set. This is classical controllability geometry applied
to learning controls, not a new controllability theorem. See
[Boyd's controllability notes](https://ee263.stanford.edu/archive/contr.pdf).

For a local quadratic loss with gradient g, PSD curvature F, and control
penalty lambda>0, minimizing

\[
 g^TJu+\tfrac12u^TJ^TFJu+
 \tfrac\lambda2 u^TM^{-1}u
\]

gives the attainable local improvement

\[
 R(g)=\tfrac12g^T J
 (J^TFJ+\lambda M^{-1})^{-1}J^Tg.
\]

It follows by completing the square. R≥0, because doing nothing is allowed.
Under an *independently estimated*, declared future-gradient second moment
Q=E[gg^T], expected reserve is half the trace of the middle operator times
Q. An empirical covariance from the same example used to choose u can
recover the gradient-noise bias in §161. F, M, Q and the update budget must
be specified; raw gradient norm and route entropy are not substitutes.
Fisher/curvature conditioning has extensive prior art, including
[Martens and Grosse, 2015](https://proceedings.mlr.press/v37/martens15.html).

This reserve is about **possible future learning**, rather than requiring
an immediate classification improvement at the current parameters. Its
local model still omits nonlinear changes outside the certified region.
Comparing these ellipsoids between routes needs the same output coordinates,
control cost convention, future-example distribution, and work budget.

## 172. What can propagate locally, and why shared controls matter

For a linearized chain with independent control blocks,

\[
 \delta h_{l+1}=A_l\delta h_l+B_l u_l,\qquad
 \sum_l u_l^TM_l^{-1}u_l\le b,
\]

the reachable Gramian obeys the exact local recursion

\[
 G_{l+1}=A_lG_lA_l^T+B_lM_lB_l^T,\qquad G_0=0.
\]

Thus no global route tree is necessary for this conditional linear history.
A factor Z with G=ZZ^T can propagate as concatenated factors
[A Z, B M^(1/2)], followed by an explicitly approximate rank reduction if
required. A compact directional sketch is an alternative. The matrix
retains the direction of opportunities; a trace alone discards it. For a
fixed terminal direction d, reverse-propagated d_l permits an exact scalar
sum of d_l^T B_l M_l B_l^T d_l. That scalar is sufficient **for that direction**,
not for arbitrary future unknown errors. Isotropic future directions are a
special case where the expected squared support is proportional to tr G.
The severe anisotropy measured in §170 makes isotropy a poor default here.

### Shared trainable knobs invalidate an independent-event sum

If the same u acts at many events or stages, instead maintain its response
J_{l+1}=A_l J_l+B_l. Then

\[
 G_{l+1}=A_lG_lA_l^T+B_lMB_l^T
       +A_lJ_lMB_l^T+B_lMJ_l^TA_l^T.
\]

The cross terms can help or cancel. A one-dimensional witness has A=1,
J_l=1, B_l=-1, M=1: the real response is zero and G_{l+1}=0. Dropping the
cross terms predicts G=2 and invents correction capacity. This is directly
relevant to E118: a router weight change moves the scores of many events,
not just one convenient near-losing message. Local counterfactual advantages
must be composed through the actual shared parameter response.

For mutually exclusive discrete routes, reachable sets form a **union**, not
a sum: h_union(d)=max_r h_r(d) when each alternative uses the same budget.
Adding their Gramians would falsely permit simultaneous controls in routes
that cannot both be taken. Duplicating an alternative must leave reserve
unchanged. Parallel active routes need their real joint budget and shared
control model before such addition is valid.

### Connection to the first actual-router replay

E118's first diagnostic perturbs real shared router parameters along a
normalized gradient on four fitting examples and replays the entire model,
including all changed winners and event times. It then evaluates four
held-out-speaker examples without updating a checkpoint. At router-step
norm 0.01, loser-score credit gives fitting loss change −0.02898 and held-out
change +0.05080; ordinary pathwise credit gives +0.01464 and −0.00608.
At 0.001 and 0.1 the signs have the same pattern in this tiny diagnostic.
These four-example outcomes are not population estimates or a policy
selection criterion. They illustrate two distinct gaps: local tangent credit
can miss discrete winner changes, and fitting progress can fail to transfer.

The next optionality estimator should therefore carry a small approximation
to **feasible, cost-normalized correction directions**, value it on other
examples, and account for shared knobs. Adding an unsigned variance bonus
or counting more near-tied routes would not address these measured gaps.
