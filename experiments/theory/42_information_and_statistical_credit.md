# Information in asynchronous teachers and statistically useful depth

The token control fits better and predicts new text worse, while executing fewer
deep events. Nonzero gradients and loss reduction therefore need a statistical
interpretation. This note derives two identities that separate lost information,
changed representation and noisy fitted correction.

## 275. A stopped token conserves the information in its character teachers

Fix the state before a phrase, with learned leaf probabilities \(q_v(\theta)\).
Generate a leaf \(V\sim q\), revealing its characters in order. Let
\(\mathcal G_k\) be the information revealed through character \(k\), stopped at
the leaf depth \(K\). Assume a finite complete prefix dictionary and positive
probabilities. The posterior leaf distribution
\(\pi_k(v)=P(V=v\mid\mathcal G_k)\) is a bounded martingale.

For logits \(z_v\), let \(A_v=\nabla_\theta z_v\), evaluated at the fixed
pre-phrase state. The next-character log-score increment is

\[
 s_k=A^T(\pi_k-\pi_{k-1}),\quad k\le K.
\]

By conditional expectation, \(E[s_k\mid\mathcal G_{k-1}]=0\). Distinct
increments are orthogonal in expectation: for \(i<j\),
\(E[s_i s_j^T]=E[s_iE[s_j^T\mid\mathcal G_{j-1}]]=0\).
Their stopped sum telescopes to \(\nabla_\theta\log q_V\). Consequently,

\[
 \boxed{\mathcal I_{\rm leaf}
 =E[\nabla\log q_V\nabla\log q_V^T]
 =E\!\left[\sum_{k=1}^{K}s_k s_k^T\right].} \tag{275.1}
\]

Thus exact partial-phrase supervision neither discards nor creates Fisher
information relative to that same leaf model. Fewer released tokens do not
automatically mean proportionally less statistical evidence. This is a
specialization of score-martingale/conditional-information identities, not a
claim that the underlying probability theorem is new.

The identity uses expectations under the model and a fixed parameter/state
during each phrase. Misspecified empirical score covariance need not obey the
same martingale orthogonality; online parameter changes and truncated credit
also require their actual boundary treatment. Comparing a character backbone
with a phrase backbone changes the family \(q\) and its state Jacobians, so
275.1 does not establish equal trainability or quality between architectures.

## 276. Compression changes when the model can act

Invertible tokenization preserves the raw string, but a deep state updated only
at stopping times \(T_1,T_2,\ldots\) has a different control schedule. Between
releases, the prefix model conditions a fixed leaf distribution by eliminating
incompatible leaves. It does not learn an arbitrary new deep-state transformation
for each partial prefix. More output logits can represent a richer joint phrase
distribution while providing fewer state updates per raw character.

The design tradeoff is therefore **control frequency versus local statistical
capacity**, not merely information loss. A free 131-way head has more independent
readout directions than a 27-way head. Rare compound leaves can have weakly
estimated embeddings/readouts even when the parse itself is lossless.

A constructive next architecture separates scales. A small learned character
state updates on real character arrivals, creates a vector summary and sends it
to deep blocks at causal release times. A shared 27-way decoder reads the latest
deep state and current local character state. This retains current-prefix control,
shares output statistics across phrases and reduces deep-body events. It charges
the local state/decoder at every character; it introduces no empty time ticks.
The fitting dictionary is one boundary control; a learned stopping-time policy
requires its own event-creation and timing counterfactual teachers.

The interpretation applies to speech coalescing too: original channel/time
information must enter learned source vectors before release. A perfect delivery
schedule cannot recover an input quotient that removed class-relevant structure.

## 277. A statistical variance term can make active depth harmful

Consider a local quadratic population risk
\(R(\theta+\delta)-R(\theta)=\mu^T\delta+\tfrac12\delta^TH\delta\),
\(H\succeq0\). A fitted average teacher has mean \(\mu\) and covariance
\(C\). For a fixed symmetric positive preconditioner \(P\), independent of
the teacher draw, let \(\delta=-\eta P\bar g\). Direct expansion gives

\[
 E[\Delta R]=-\eta\mu^TP\mu+
 \frac{\eta^2}{2}\left[
 \mu^TPHP\mu+\operatorname{tr}(HP C P)\right]. \tag{277.1}
\]

The descent term measures a reproducible direction; the trace term is the
curvature cost of fitting noise. If both terms are nonzero, expected descent
requires

\[
 \eta<\frac{2\mu^TP\mu}
 {\mu^TPHP\mu+\operatorname{tr}(HP C P)}. \tag{277.2}
\]

For a smooth nonquadratic risk this is a local expansion with a Taylor remainder,
not a global learning guarantee. Fresh Adam depends nonlinearly on the same
teacher; substituting its update for a fixed \(P\) is invalid. Actual Adam
steps still need replay, as in §§249–252. The formula explains why many nonzero
teachers can improve fitting yet reduce transfer: added directions can enlarge
the variance term without enlarging the reproducible descent term.

In categorical function coordinates, local curvature is
\(F(p)=\operatorname{diag}(p)-pp^T\); a parameter Jacobian pulls it back to
\(J^TF(p)J\). Path counts or raw teacher norms alone give no lower bound on
useful eigenvalues or on \(\mu^TP\mu\). Deep optionality should be assessed by
reachable, reproducible class corrections under a work budget.

## 278. Measure teacher reproducibility before spending another full run

For independent teacher samples \(g_1,\ldots,g_m\) and fixed \(P\),

\[
 \widehat{\mu^TP\mu}=
 \frac{(\sum_i g_i)^TP(\sum_i g_i)-\sum_i g_i^TPg_i}{m(m-1)}. \tag{278.1}
\]

This cross-sample statistic removes the positive diagonal noise term in the
squared fitted mean. Under the stated independent sampling assumptions it is
unbiased; a negative realized estimate means no certified positive signal from
that finite sample. For equal disjoint groups, the cross inner product of group
means is a related measure of reproducibility. Neither is a guarantee of held
descent or an Adam-step estimate.

Real sequence teachers are correlated. Exactly,
\(C=m^{-2}\sum_{i,j}\operatorname{Cov}(g_i,g_j)\); a scalar effective sample
size generally differs by direction. Speech utterances, speakers, clean/augmented
views and consecutive text positions must not be counted as independent event
labels. A paired utterance teacher is one sampled unit. Sampling a finite fitting
set without replacement and speaker clustering also qualify the independence
interpretation; retain measured cross-statistics as diagnostics.

The concrete next SHD diagnostic warms the owned correction head using the
already calibrated fitting-only first update. It freezes that checkpoint, then
measures raw paired teachers in disjoint fitting batches, separately for the
head, source projection and all six added blocks. It checks cross-sample and
split-group agreement, layer support and clipping magnitudes. No private
development or official test labels choose it. This distinguishes absent credit
from active but statistically inconsistent credit before a longer full-depth run.
