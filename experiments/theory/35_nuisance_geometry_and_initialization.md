# Initialization must see the nuisance channel of its later training

Derived 30 September 2026 after E150's clean fitted head reaches 6,011/6,144
fitting decisions (97.84%) but 374/512 development decisions (73.05%). Its
first augmented pass has online NLL 5.0118 and reaches 365/512 (71.29%).
This motivates an explicit fitting-only paired-view intervention. It does not
identify the complete cause of the speech generalization gap.

## 245. Exact categorical risk of a feature nuisance channel

For a label-preserving transformation T, let h_T be the encoder's query,
bar h=E_T h_T, L_T=W h_T+b and bar L=W bar h+b. Cross-entropy's label term is
linear in logits. Consequently,

\[
 E_T\ell(L_T,y)-\ell(\bar L,y)
 =E_T D_{KL}(p(\bar L)\Vert p(L_T))\geq0. \tag{245.1}
\]

Proof: use the log-sum-exp Bregman identity of §238; the linear remainder
averages to zero because E_T(L_T-bar L)=0. Unlike a second-order approximation,
this is exact for finite feature displacements and an affine head. It compares
the transformed distribution to its mean feature, not to the clean view: a
nonzero mean displacement has its own class effect.

Let Sigma_T=Cov_T(h_T), and P remove the class-wide logit offset. Then §238 gives

\[
 E_T\ell(L_T,y)-\ell(\bar L,y)
 \leq\tfrac14\operatorname{tr}(PW\Sigma_T W^TP). \tag{245.2}
\]

If all categorical probabilities along every interpolation are at least rho,
the lower bound is rho/2 times the same trace: for a centered contrast v,
Var_p(v)>=rho min_a sum_j(v_j-a)^2=rho ||v||^2. This condition can be weak
for a confident classifier. The exact KL/Jensen measurement remains usable.

For two equiprobable views h,h', conditional covariance is
(h'-h)(h'-h)^T/4. Their exact average CE minus CE at their mean equals the
average KL from the mean-view prediction; its upper bound is
||P W(h'-h)||^2/16. Thus a head can be excellent on clean features and incur
large transformed risk without any route being missing or label credit being
zero. The proposed measurement observes this directly on fitting data.

## 246. Total feature variation and nuisance variation differ

For clean/augmented pairs a_i,b_i, define midpoint m_i=(a_i+b_i)/2 and
global mean mu=mean_i m_i. Equal weighting of the 2n views gives exactly

\[
 \Sigma_{view}=\Sigma_{between}+\Sigma_{within},\quad
 \Sigma_{within}=\tfrac1{4n}\sum_i(b_i-a_i)(b_i-a_i)^T. \tag{246.1}
\]

This is the law of total covariance applied to a declared finite view channel.
Its within component includes the clean-to-transform drift, rather than
assuming that augmentation has zero mean relative to clean inputs.
Nuisance response depends on W Sigma_within W^T, not only the query variance
or the head's physical norm. Directions with little clean variation but much
transformed variation can be strongly amplified by a clean-only inverse metric.
Generalized eigenvalues of (Sigma_within, Sigma_clean+lambda I) expose that
relative amplification; they are a diagnostic, not task accuracy by themselves.

## 247. Correct the initialization objective before deep updates

E152 keeps the inherited E143 encoder fixed while producing one clean and
one augmented fitting view per utterance. It reconstructs the E150 clean-head
fit on the same clean cache, then measures both its transformed risk and a
paired-view head. The augmented views reproduce E150's first-pass transformation
draws; the training RNG is replayed unchanged. Development remains excluded.

Use the robust metric

\[
 Q=(\Sigma_{view}+\gamma\Sigma_{within}+\lambda I)^{-1/2},
 \quad\gamma=1. \tag{247.1}
\]

Convex supervised/teacher head fitting uses both views and a declared whitened
weight penalty 1e-4, versus E150's clean-view 1e-5. Head folding remains exact.
In physical coordinates, ||A||_F^2 is

\[
 \operatorname{tr}[W(\Sigma_{view}+\gamma\Sigma_{within}+\lambda I)W^T].
 \tag{247.2}
\]

The change of coordinates alone would not impose robustness on an unregularized
converged fit. Its specified penalty and augmented targets do. This guards
task-visible nuisance amplification while retaining one ordinary deployed head.
Two subsequent single-encoder passes use ordinary label CE. The extra teacher
view evaluation and head fitting are charged; this is not an equal-budget
attribution against E150's three-pass continuation. The initial fixed-feature
comparison isolates the specified head objective under common features.

## 248. What this advance means for event routing

Useful route alternatives must retain class-sensitive directions while limiting
directions that change under a label-preserving event transformation. A large
local teacher or control Gramian can reflect the latter. Rewarding unsigned
variance or head fit alone can therefore select unstable learning reserve.
The paired risk and covariance give a concrete fitting-only metric for that
distinction. They do not replace future-adaptation optionality (§§159–163) by
ordinary variance, or prove that a transformation really preserves labels.
Channel shifts/cropping and time stretches retain the project's existing
augmentation assumptions and their possible information loss.

The noise/regularization connection is established prior art, including
[Bishop (1995)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf).
The contribution here is an exact finite-view categorical identity and its
implementation in the specific event-encoder absorption experiment, not a
claim to invent noise regularization. E153 checks that identity, its bound,
the paired covariance decomposition and the folded regularizer. Only completed
E152 outcomes determine whether this correction improves speech recognition.

**Completed E153 check:** finite paired CE/KL identity error 2.11e-15;
covariance decomposition error 4.76e-16; folded logits error 2.67e-15;
regularizer identity error 1.43e-14. The declared finite-view KL bound holds.
