# Useful extra freedom: attainable changes, conditioning and coupled credit

2 October2026. Derivation supplement to source-frozen note90; no new model,
credit estimator or fit is introduced here. Seed6 offset selection is exploratory;
the unchanged joint candidate's seed7 confirmation fails below. This note answers
when additional representation coordinates can reduce a forced timing tradeoff,
and why gradient opposition alone does not answer that question.

## An exact local improvement formula

Fix an entering prefix and a smooth realized event history. Stack desired score
and representation changes into b, with explicit units/weights: for example
b=[W_s^(1/2)*desired_score_change; W_z^(1/2)*desired_content_change]. Existing
coordinates x have stacked Jacobian A; new offset coordinates y have B. Consider

    Q(x,y) = 1/2 ||A*x+B*y-b||^2 + lambda/2 ||x||^2 + mu/2 ||y||^2,
    lambda>0, mu>0.

This is a local fitting surrogate, not the hard-route sequence risk. Set
H=A^T*A+lambda*I, x0=H^(-1)*A^T*b and residual r=b-A*x0. Eliminating x yields

    S = B^T*(I-A*H^(-1)*A^T)*B + mu*I,
    h = B^T*r,
    y_star = S^(-1)*h,
    min_x Q(x,0) - min_(x,y) Q(x,y) = 1/2 h^T*S^(-1)*h >= 0.

Proof: complete the square in x; the remaining quadratic is
Q(x0,0)-h^T*y+1/2*y^T*S*y. H and its Schur complement S are positive definite.
Improvement is strict exactly when B^T*r is nonzero. Extra coordinates that only
point orthogonally to the unresolved task residual cannot help this local
objective. If a constrained score-preserving content direction is missing and
the residual projects onto the offset, the added coordinate can help. The formula
also charges a chosen parameter-change penalty instead of assuming arbitrary
movement is free. With penalties, even a redundant coordinate can improve the
surrogate by changing the effective movement cost; this is conditioning rather
than a proof of increased representational rank. With zero penalties use the
orthogonal projector onto col(A) and pseudoinverse: benefit comes only from the
component of B outside col(A) that addresses the residual.

For a locally post-race reception calibration, B has zero score rows at that
specific race, with the entering history fixed. It need not have zero score rows
for later races or the complete prefix: updated memory/content changes future
keys, clocks and winners. Using full-sequence Jacobians preserves the same
augmented-quadratic argument, but removes the claim of score-preserving motion.
A shared offset is not an independent per-event control. Its column must include
all sites where it is reused; opposing site demands may cancel B^T*r.

## A quantitative independence and conditioning result

For one damped rotating pair, z=exp(-rho*a) R(omega*a+beta)m, with J a ninety-degree
rotation. Holding physical a, m and parameters otherwise fixed,

    dz/da = -rho*z + omega*J*z,
    dz/dbeta = J*z,
    det([dz/da,dz/dbeta]) = -rho*||z||^2.

Thus rho>0 and z!=0 give two independent local directions. Waiting couples
rotation to damping; the calibration can rotate without that damping change.
For raw offset u with beta=pi*tanh(u), the determinant acquires the factor
pi*sech(u)^2. Large |u|, weak damping or a heavily decayed message can make the
coordinates poorly conditioned even when rank is formally two. Zero damping
makes the one-pair age/phase directions collinear. This exact determinant was
checked in note90's completed205000Z contracts; the raw-coordinate factor follows
by the chain rule. Rank alone does not promise a useful optimization magnitude.
The age derivative remains intact: this is partial freedom, not removal of time
as computation. Adjacent unconstrained linear maps can absorb constant phase in
restricted models, so the whole architecture's rank must not be inferred from
this two-column calculation alone.

## Why negative route/content cosine is insufficient

For a genuinely differentiable single loss, g=g_route+g_content. Its exact
ordinary gradient step changes loss by -eta*||g||^2+O(eta^2), even when the two
terms have negative inner product. Cancellation can reflect a real opposing
consequence of one parameter change, rather than corrupt credit. A block step
on exact current gradients has first-order decrease -eta*||g_block||^2; this does
not prove alternating blocks are faster than joint descent. Stale Adam momentum,
finite steps and surrogate race teachers invalidate the elementary guarantee.
Route/content teachers in this repository have explicit conditional scopes;
their sum must not automatically be called an exact expected-risk gradient.

A desired improvement diagnostic needs the loss derivative of the proposed
update and its downstream effect, not just gradient norms. Cosines also depend
on parameter scaling/metric. For positive definite preconditioner P, the relevant
cross term is g_route^T*P*g_content; Euclidean opposition is not invariant under
reparameterization. Positive score-space normalization cannot repair a wrong
teacher direction, and projection may remove a correct causal term. Physical
age derivatives at a fixed event history exclude winner/boundary changes;
expected hard-event risk additionally requires correctly scoped choice and
timing credit. Extra coordinates do not repair those missing terms by theorem.

## Decision implications with limited compute

Use independently derived contracts before quality fits. The offset experiment
holds the native credit rule fixed and retains actual races, sparse writes,
physical-age evolution and all losing-candidate fitting work. Its zero setting
reproduces the original model. Four readiness contracts and two tiny learning
smokes passed; two seed6 schedules were declared before either pilot. Joint
selected1.258170NLL versus native1.305937 at1.000910 fitting-work ratio; alternating
selected1.263009 at.998201. Both pass the fixed gate, but no independent benefit
of alternation is demonstrated. Private blocks get half as many updates under
alternation and global clipping differs. Preserve this limitation beside the
numbers. Only the selected unchanged joint schedule received seed7 confirmation. It fails:
native55.2083%/1.306508 versus offset53.1250%/1.327347, NLL improvement
-.020839 and accuracy decline2.0833pp, at1.000910 fitting-work ratio. Both
completed comparisons and all four passes remain preserved. No unchanged
full-data scale-up, extra epochs or third-seed fishing is admitted.

If confirmation fails, do not reinterpret local geometry as empirical validation
or add epochs to rescue the same gate. If it passes, full-data quality and total
work remain separate requirements before a practical advantage claim. An offset
improvement could arise from conditioning, a useful new tangent direction or
finite-step regularization; the present pilot does not isolate these causes.
A future fitting-only Jacobian probe can estimate B^T*r and the regularized Schur
benefit on a fixed small prefix set, but it is not yet an admitted training
campaign and must be charged if used. The present evidence does not establish
surface n-gram dependence, a universal gradient regression or mathematical
impossibility of deeper features.
