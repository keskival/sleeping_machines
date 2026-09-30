# Absorbing parallel classifiers and moving credit into one event encoder

Derived 30 September 2026. E143 obtains 408/512 private SHD decisions with a
frozen D8 parent plus a trained D6 temporal correction. This note specifies a
single-encoder continuation using fitting features and parent logits. Neither
projection nor head optimization uses development labels. The inherited
training lineage remains part of its cost.

## 237. Functional absorption is a projection question

Write the combined and replacement logits as

\[
 L(x)=f(x)+Wh_\theta(x)+b,\quad
 \widetilde L(x)=(W+V)h_\theta(x)+b+a. \tag{237.1}
\]

The replacement has one event encoder and one affine head. Its sources,
modal states, gates and actual winning clocks initially remain unchanged.
Let P=I-11^T/C remove the categorical logit gauge. Only Pf needs representation:
the component parallel to 1 changes neither probabilities nor decisions.

With fitting queries h_i, fit centered targets Pf_i. Set mu=mean(h),
a0=mean(Pf), X_i=h_i-mu, Y_i=Pf_i-a0 and Sigma=X^T X/n. In row-vector
convention, positive-ridge regression gives

\[
 B=(\Sigma+\lambda I)^{-1}X^TY/n,\quad
 a=a_0-\mu B,\quad V=B^T. \tag{237.2}
\]

Lambda is declared as 1e-4 times mean fitting feature variance; the intercept
is unpenalized. In Sigma's eigenbasis, ridge multiplies the empirical linear
projection by sigma/(sigma+lambda). The remaining parent contribution is
measured, rather than assumed absent. Without ridge, an exactly affine supported
contribution is absorbed up to its gauge; with ridge, shrinkage bias is explicit.
For a nonlinear parent, additional encoder training has a concrete target:
make missing class information reachable in h. Its hidden states and old
packetization need not be reproduced.

## 238. Certificates for probabilities and decisions

Set e=P(tilde L-L), p=softmax(L) and tilde p=softmax(tilde L). Then

\[
 D_{KL}(p\Vert\widetilde p)\leq\|e\|_2^2/4. \tag{238.1}
\]

Proof: KL is the Bregman remainder of log-sum-exp. Its Hessian is
diag(p)-pp^T. The quadratic form on v is the variance of a categorical draw
of v's coordinates, bounded by (max(v)-min(v))^2/4 <= ||v||_2^2/2.
Taylor's integral remainder supplies the other factor 1/2. This bound holds
for finite errors, although it can be loose.

For the teacher's top-two logit gap m,

\[
 2\|e\|_\infty<m\implies
 \arg\max\widetilde L=\arg\max L. \tag{238.2}
\]

Logit residual, probability distortion and decision retention are distinct.
They do not say whether the teacher was correct. With h deterministic, the
unrestricted best teacher imitation is E[p(X)|h]; its minimum expected KL is
I(T;X|h), where T is a categorical draw from the teacher. An affine head has
an additional function-class restriction. This teacher-information identity
does not identify true-label Bayes error on new speakers.

## 239. Conditioning can be folded away

Use Q=(Sigma+lambda I)^(-1/2) and z=(h-mu)Q. A whitened head z A+c folds into
the deployed affine head exactly:

\[
 W_{full}^T=QA,\quad b_{full}=c-\mu QA. \tag{239.1}
\]

Whitened fitting covariance has eigenvalues sigma/(sigma+lambda), at most
one. It does not create support in a zero-variance direction. Offline head
conditioning therefore needs no extra deployed map, state or traffic. Remove
class-wide offsets from A and c to eliminate the likelihood gauge.

Fixed-feature initialization minimizes soft-target CE plus a declared quadratic
penalty in whitened weights, with target

\[
 t_i=(\operatorname{onehot}(y_i)+\beta p_i)/(1+\beta),\quad\beta=0.5. \tag{239.2}
\]

Up to scale and a constant, this is supervised CE plus beta times teacher KL.
The teacher is computed once on fitting data and discarded before end-to-end
training. That subsequent learner uses ordinary utterance-label CE alone.
For fixed z, the data Hessian is the mean of
(diag(q_i)-q_i q_i^T) tensor (z_i z_i^T); the penalty is also convex. Hence the
head fit has a convex objective. This specifies an initialization, not global
deep-network convergence. Regularization's function effect is retained.

## 240. Absorption changes who receives the teacher

With label residual r=p-onehot(y), a frozen-parent residual encoder receives
W^T r. At exactly preserved probabilities, the single encoder instead receives

\[
 g_h^{single}=(W+V)^Tr=g_h^{residual}+V^Tr. \tag{240.1}
\]

Even at identical predictions, its parameter teacher changes by
J_theta(h)^T V^T r. Credit associated with the absorbed classifier now reaches
the trainable temporal representation. Function preservation is not gradient
preservation. Approximate absorption adds (W+V)^T(tilde p-p) as well.

For route-control Jacobian K, observed route displacements change from P W K
to P(W+V)K. This can reveal previously invisible control directions without
adding routes. The Gramian also has cross terms: its rank and useful reserve
are not automatically monotone. Thus the temporal orbit of §§234–236 must be
observed through the complete trained head, not judged by local route counts.

**Numerical and empirical scope.** E149 checks an ill-conditioned affine
component: maximum centered logit error 6.45e-7 with tiny positive ridge;
all 240 decisions retained and certified. Whitened/deployed logits differ
by at most 2.14e-14; random finite perturbations satisfy the KL bound.
E150 predeclares the E143 pass-three checkpoint, 6,144 fitting and 512
development utterances, three single-encoder passes and fresh Adam with
initial LR 0.000325. Calibration is reported separately. The inclusion
checks imply no speech accuracy or deep convergence outcome.
