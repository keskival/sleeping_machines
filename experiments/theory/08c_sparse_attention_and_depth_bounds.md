# Sparse attention and depth bounds

[Theory index](../THEORY.md) · Previous: [08b memory retrieval and deep learning](08b_memory_retrieval_and_deep_learning.md) · Global sections 112–114; section numbers remain stable.

## 112. Mass truncation and counterfactual key recruitment

The exact dense key/value score credit is already derived in §105(f) and
§107(g), and the noisy-choice route boundary term plus cancellation-based
near-miss credit are already derived in §§19 and 57. Here I combine them in two
ways: an exact decomposition of the query-gradient error under support
truncation, and a curvature bound connecting a missing key's local attention
credit to the finite counterfactual route loss used by the existing router.

### Fixed-support query-gradient error

Let $p$ be the dense softmax over keys and let the retained set $C$ have mass
$1-\epsilon$; let $O$ be its complement. Write $p_C$ and $p_O$ for the
conditional distributions on the two groups. For one shared upstream loss
vector $g$, set $a_j=g^\top v_j$, and define conditional means and cross
covariances

$$
\bar k_R=\mathbb E_R[k],\qquad \bar a_R=\mathbb E_R[a],\qquad
\Sigma_R=\mathbb E_R[(k-\bar k_R)(a-\bar a_R)],\quad R\in\{C,O\}.
$$

The dense and truncated query gradients are respectively
$\beta\operatorname{Cov}_p(k,a)$ and $\beta\Sigma_C$. The law of total
covariance gives the exact residual

$$
\nabla_q L_{\rm dense}-\nabla_q L_C
=\beta\epsilon\left[\Sigma_O-\Sigma_C+
(1-\epsilon)(\bar k_C-\bar k_O)(\bar a_C-\bar a_O)\right].
$$

So omitted mass alone is not the full optimization signal: the error contains
both the omitted group's within-group key/advantage covariance and a
between-group covariance. Let $D_K$ be the diameter of all keys, $D_V$ the
diameter of all values, and $D_A=\max_j a_j-\min_j a_j\le\|g\|D_V$. The
covariance variance bound yields

$$
\|\nabla_q L_{\rm dense}-\nabla_q L_C\|
\le \beta\epsilon(3/2-\epsilon)D_KD_A.
$$

Also $y=(1-\epsilon)y_C+\epsilon y_O$, hence
$\|y-y_C\|\le\epsilon D_V$. With the same $g$, the total block-norm error in
all value gradients is exactly $2\epsilon\|g\|$; for independently
parameterized keys, $\nabla_{k_j}L=\beta q\,p_j(a_j-\bar a)$, so the sum of
key-gradient block errors is at most
$\beta\epsilon(5/2-\epsilon)\|q\|D_A$. For the retained keys, this follows
from $\mathbb E_C|a-\bar a_C|\le D_A/2$ and
$|\bar a_C-\bar a_O|\le D_A$; the omitted keys contribute at most
$\epsilon D_A$ in scalar coefficient mass. These are local VJP comparisons
at fixed support. A nonlinear downstream loss changes $g$ when $y$ changes,
and that additional curvature term is outside this statement.

### Downstream curvature: the full smooth-loss VJP error

The fixed-$g$ comparison can be extended to the actual scalar objective. Let
$F$ be twice differentiable with $\|\nabla^2F(z)\|_2\le H$ along the segment
joining dense output $y$ to truncated output $y_C$. Write
$g=\nabla F(y)$, $g_C=\nabla F(y_C)$, and let $D_A^C$ be the range of
$g_C^\top v_j$ over *all* keys, with $0\le\epsilon<1$. Since
$\|y-y_C\|\le\epsilon D_V$,
\[
\|g-g_C\|\le H\epsilon D_V.
\]
Decompose the query VJP by first holding the upstream vector at $g_C$, then
changing it from $g_C$ to $g$. The first term is the fixed-support residual
above. The second is the dense covariance with scalar advantage
$(g-g_C)^\top v_j$, whose range is at most $\|g-g_C\|D_V$. Therefore
\[
\|\nabla_qF(y)-\nabla_qF(y_C)\|
\le \beta\epsilon D_K\left[
  (3/2-\epsilon)D_A^C+{H D_V^2\over4}
\right].
\]
This is an absolute-error bound; when the dense query gradient is near zero it
does not imply relative accuracy or preserve gradient direction.

The same decomposition gives bounds for the other independently parameterized
attention blocks. For values, the dense VJP is $p_jg$ and the truncated VJP
is $p_jg_C/(1-\epsilon)$ on $C$ and zero on $O$. The $\ell_1$ difference of
the padded probability vectors is exactly $2\epsilon$, so
\[
\sum_j\|\nabla_{v_j}F(y)-\nabla_{v_j}F(y_C)\|
\le 2\epsilon\|g_C\|+H\epsilon D_V.
\]
For independent keys, the score-gradient coefficient is
$p_j(a_j-\mathbb E_p a)$, with $a_j=g^\top v_j$. Changing support at fixed
$g_C$ uses the existing bound above; changing $g_C$ to $g$ on the dense
support adds at most $\tfrac12D_V\|g-g_C\|$ in coefficient $\ell_1$ norm,
by the scalar range bound for mean absolute deviation. Thus
\[
\sum_j\|\nabla_{k_j}F(y)-\nabla_{k_j}F(y_C)\|
\le \beta\|q\|\epsilon\left[
  (5/2-\epsilon)D_A^C+{H D_V^2\over2}
\right].
\]
These are local bounds for fixed keys, values, and candidate support. For
gradients with respect to the inputs of linear query/key/value projections,
multiply by the corresponding transpose operator norms; weight-gradient bounds
also require the projection inputs' norms. They close the smooth downstream-loss
gap left by the fixed-$g$ lemma, while retaining its main diagnosis: high
retained mass controls an absolute error, but does not imply relative accuracy
or gradient alignment.

The exact residual identifies what a candidate rule should preserve. High mass
recall is a useful worst-case certificate, but the query-gradient target is the
omitted cross-covariance above. In particular, a key index can retain nearly
all mass and still lose a directionally important between-group term; mass
controls absolute error, not relative error or gradient alignment. For example,
retain one key $(k,v)=(0,0)$ with mass $1-\epsilon$ and omit $(1,1)$ with mass
$\epsilon$, using $L(y)=y$. The truncated output is zero and its query gradient
is zero; dense attention has query gradient $\beta\epsilon(1-\epsilon)$.
Although almost all forward mass is retained, the omitted key carries the whole
nonzero query-learning signal. Its insertion advantage is $A=1$, so the
existing counterfactual route mechanism sees the missing route directly.

### A missing key's attention credit is a counterfactual route advantage

Let the current support have partition sum $Z_C$ and output $y_C$. Add a
proposed missing key $u$ with score $s_u$ and value $v_u$. Its resulting
softmax mass is $r_u=e^{s_u}/(Z_C+e^{s_u})$, and the augmented output is exactly

$$
y_{C+u}=(1-r_u)y_C+r_uv_u.
$$

For the downstream loss $F(y)$, define the already-known local attention
advantage $A_u=\nabla F(y_C)^\top(v_u-y_C)$. If the Hessian of $F$ has operator
norm at most $H$ along the segment from $y_C$ to $y_{C+u}$, Taylor's theorem
gives

$$
\left|F(y_{C+u})-F(y_C)-r_uA_u\right|
\le {H\over2}r_u^2\|v_u-y_C\|^2.
$$

Moreover, along the insertion path $y_r=(1-r)y_C+rv_u$,
$\frac{dF(y_r)}{ds_u}=r(1-r)\nabla F(y_r)^\top(v_u-y_C)$, exactly the standard
softmax score gradient. Thus the dense attention residual is the tangent of
the finite route-insertion loss; its approximation error is controlled by the
curvature bound above. This gives the established §19/§57 counterfactual router
a cheap attention-specific loss estimate $r_uA_u$. If the curvature remainder
is large, the existing shadow execution can supply the finite loss difference.
More specifically, substitute this estimate into the existing §19 boundary
gradient. Let $b_u$ be the candidate-router score (it may equal the attention
logit $s_u$), let $m_u=b_c-b_u$ be the margin to the chosen route, and let
$\rho_\sigma(m_u)$ be its noise density. Replacing the exact route loss
difference $\Delta F_u$ by $r_uA_u$ changes that gradient
by at most

$$
\|G_{\rm CF}-\widehat G_{\rm CF}\|
\le {H D_V^2\over2}\sum_{u\in N}
\rho_\sigma(m_u)r_u^2\|\nabla_\theta(b_u-b_c)\|,
$$

for the near-miss set $N$. This yields a bound-driven compute rule: shadow-run
the alternatives with the largest individual remainder bound, and stop when
the sum of unshadowed bounds is below a chosen credit-error budget. It
specializes §19's exact-versus-approximate counterfactual tradeoff to attention
keys; it does not solve candidate generation. Score work, payload access, and
shadow execution still need to be counted.

**Falsifiable consequence.** On diagnostic batches with exact dense scores,
first verify the covariance residual against autodiff while sweeping support
mass and temperature. Then apply the existing counterfactual route learner to
proposed key/value candidates. Compare the cheap $r_uA_u$ ranking with exact
finite insertion losses, measure the omitted-gradient residual recovered by
recruited keys, and report route recall, candidate scoring, shadow work, and
loss across depths. Compare fixed top-$m$ shadowing against the remainder-bound
allocation above at equal shadow budget. This measures whether curvature-aware
counterfactual work reduces router-gradient error; separately count the search
work needed to obtain candidates.
## 113. Deep-stack perturbation: when local sparsification errors stay controlled

Sections 107(f–h) bound propagation through a fixed-topology event stack, and
§112 bounds the local output and VJP errors caused by truncating attention
support. This section composes those two kinds of result. It does not repeat the
route-changing counterfactual derivative from §§19 and 57: the theorem below is
inside a region where each dense and sparse block is differentiable and its
candidate support is fixed.

### Residual-stack statement

Consider depth $L$ residual stacks with the same initial state,

$$
h_{\ell+1}=h_\ell+\alpha_\ell f_\ell(h_\ell;\theta_\ell),\qquad
\tilde h_{\ell+1}=\tilde h_\ell+\alpha_\ell\tilde f_\ell(\tilde h_\ell;\theta_\ell),
\quad \ell=0,\ldots,L-1,
$$

where $f_\ell$ is a dense block and $\tilde f_\ell$ is its sparse-support
approximation. Assume $\alpha_\ell\ge0$. Throughout a region containing both
trajectories and the line segments used in the comparisons, suppose

$$
\|D f_\ell\|,\|D\tilde f_\ell\|\le K_\ell,\qquad
\|f_\ell(x)-\tilde f_\ell(x)\|\le\epsilon_\ell,
$$

and, for the state Jacobians,

$$
\|D f_\ell(x)-D\tilde f_\ell(x)\|\le\eta_\ell,\qquad
\|D\tilde f_\ell(x)-D\tilde f_\ell(y)\|\le M_\ell\|x-y\|.
$$

Set $d_\ell=\|h_\ell-\tilde h_\ell\|$ and
$R_{a:b}=\prod_{j=a}^{b}(1+\alpha_jK_j)$, with an empty product equal to one.
Then

$$
d_\ell\le\sum_{i<\ell}\alpha_i\epsilon_i R_{i+1:\ell-1}.
$$

Let $A_\ell=I+\alpha_\ell Df_\ell(h_\ell)$ and
$\tilde A_\ell=I+\alpha_\ell D\tilde f_\ell(\tilde h_\ell)$, and let
$P=A_{L-1}\cdots A_0$ and $\tilde P=\tilde A_{L-1}\cdots\tilde A_0$.
Telescoping the products gives

$$
\|P-\tilde P\|
\le\sum_{i=0}^{L-1}\alpha_i(\eta_i+M_id_i)
\prod_{j\ne i}(1+\alpha_jK_j).
$$

If a scalar loss $\Phi$ has $H$-Lipschitz gradient along the output segment
and $\|\nabla\Phi(h_L)\|\le G$, then the input-gradient discrepancy satisfies

$$
\|P^\top\nabla\Phi(h_L)-\tilde P^\top\nabla\Phi(\tilde h_L)\|
\le G\|P-\tilde P\|+H d_L\|\tilde P\|.
$$

For an individual layer parameter block, put
$B_m=\alpha_mD_{\theta_m}f_m(h_m)$ and
$\tilde B_m=\alpha_mD_{\theta_m}\tilde f_m(\tilde h_m)$. If
$\|B_m-\tilde B_m\|\le\zeta_m$, $\|\tilde B_m\|\le b_m$, and $P_{>m}$
denotes the product of state Jacobians strictly after layer $m$, then

$$
\|\nabla_{\theta_m}\Phi-\nabla_{\theta_m}\tilde\Phi\|
\le \zeta_m\|P_{>m}\|G
+b_m\left(\|P_{>m}-\tilde P_{>m}\|G
+\|\tilde P_{>m}\|H d_L\right).
$$

### Derivation

The state recurrence follows by adding and subtracting
$f_\ell(\tilde h_\ell)$:

$$
d_{\ell+1}\le(1+\alpha_\ell K_\ell)d_\ell+\alpha_\ell\epsilon_\ell,
\qquad d_0=0.
$$

Unrolling gives the stated sum. Next, split the Jacobian mismatch at the same
state $h_\ell$ and use the Lipschitz assumption:

$$
\|A_\ell-\tilde A_\ell\|
\le\alpha_\ell\left(
\|Df_\ell(h_\ell)-D\tilde f_\ell(h_\ell)\|
+\|D\tilde f_\ell(h_\ell)-D\tilde f_\ell(\tilde h_\ell)\|
\right)
\le\alpha_\ell(\eta_\ell+M_\ell d_\ell).
$$

For ordered products, insert one factor difference at a time:

$$
P-\tilde P=\sum_i
A_{L-1}\cdots A_{i+1}(A_i-\tilde A_i)
\tilde A_{i-1}\cdots\tilde A_0.
$$

Since each factor norm is at most $1+\alpha_jK_j$, taking norms gives the
product bound. For the input gradient, add and subtract
$\tilde P^\top\nabla\Phi(h_L)$ and use the $H$-Lipschitz gradient property.
For a parameter block,
$\nabla_{\theta_m}\Phi=B_m^\top P_{>m}^\top\nabla\Phi(h_L)$; inserting first
$\tilde B_m$, then $\tilde P_{>m}$, then
$\nabla\Phi(\tilde h_L)$ yields the three terms in the bound.

This separates three causes of gradient error: the layer's own parameter-VJP
mismatch, the downstream Jacobian-product mismatch, and the changed output
gradient caused by forward-state drift. If the parameter Jacobian discrepancy
at a matched state is at most $\mu_m$ and the sparse parameter Jacobian is
$N_m$-Lipschitz in state, one may take
$\zeta_m\le\alpha_m(\mu_m+N_md_m)$.

### Consequence for depth scaling

If $K_\ell\le K$, $\alpha_\ell=1/L$, and
$\epsilon_\ell\le\epsilon$, then $d_L\le e^K\epsilon$. If also
$\eta_\ell\le\eta$ and $M_\ell\le M$, then

$$
\|P-\tilde P\|\le e^K(\eta+M e^K\epsilon),
$$

which has no explicit exponential dependence on depth. The conclusion relies
on residual scaling and uniformly bounded local errors and Jacobian
smoothness. It is not true for an arbitrary unscaled serial stack: its
Jacobian products may grow or collapse exponentially. Nor does the result say
that the *training gradient* is useful or aligned with the task; it says that a
fixed-support sparse approximation perturbs a dense model's local gradients by
a controlled amount under the stated conditions.

### What remains to prove for attention blocks

The §112 covariance and curvature bounds provide local query/key/value VJP
estimates, not an operator-norm bound for an entire sequence layer. To plug
them into this theorem, one must account for shared-key fan-out, all query
positions, projection norms, gates, and payload paths to certify $\epsilon_\ell$
and $\eta_\ell$. The full-sequence Jacobian accounting in §107(h) handles
fan-out for the dense fixed-topology block, but it does not yet bound the
dense-versus-truncated Jacobian difference. Candidate-set changes and event
birth/death also leave the theorem's smooth region; their route credit remains
the existing boundary mechanism of §§19 and 57. This identifies the analytical
bridge still missing between a good local attention approximation and
trainability of a deep sparse stack.

**Test implied by the theorem.** On a small sequence, compare dense and
truncated blocks by measuring layer output discrepancy, full sequence-Jacobian
operator discrepancy, key fan-out, and parameter-gradient discrepancy at every
depth. Check the displayed inequalities on fixed support first, then introduce
support changes and separately measure counterfactual-boundary coverage. In
E83/E84, measure local layer discrepancies and Jacobian gains where possible;
accuracy and gradient norms alone cannot verify this stability statement.

## 114. From per-query truncation to a full-sequence Jacobian certificate

Section 112 bounds a single query's output and query/key/value VJPs. Here I
lift those estimates to the whole sequence layer, including the fan-out of each
shared key. This closes the fixed-support operator-norm gap identified in
§113; candidate-set changes remain outside the derivative and continue to use
the established boundary credit of §§19 and 57.

### Setup and statement

For input vectors $x_1,\ldots,x_T$, use shared linear projections

$$
q_i=W_Qx_i,\qquad k_j=W_Kx_j,\qquad v_j=W_Vx_j,
$$

and scores $s_{ij}=\beta q_i^\top k_j+b_{ij}$, where $b_{ij}$ is fixed
(it may encode a positional bias or causal mask) and $\beta\ge0$. Let $p_{ij}$
be the dense softmax over valid keys for query $i$. A fixed candidate set $C_i$
has dense mass $1-\epsilon_i>0$, and the sparse distribution is the dense
distribution conditioned on that set:

$$
\tilde p_{ij}=\frac{p_{ij}\mathbf 1[j\in C_i]}{1-\epsilon_i},\qquad
y_i=\sum_jp_{ij}v_j,\qquad \tilde y_i=\sum_j\tilde p_{ij}v_j.
$$

Let $D_K=\max_{j,k}\|k_j-k_k\|$, $D_V=\max_{j,k}\|v_j-v_k\|$, and
let $\|W\|$ denote spectral operator norm. For a fixed output projection $W_O$
(identity if absent), define

$$
c_i^Q=\beta\epsilon_i(3/2-\epsilon_i)D_KD_V,
$$

$$
b_{ij}=\|W_O\|\left[
\mathbf 1[i=j]c_i^Q\|W_Q\|
+|p_{ij}-\tilde p_{ij}|\|W_V\|
+\beta D_V\|q_i\|\|W_K\|
  \left(|p_{ij}-\tilde p_{ij}|+\epsilon_i\tilde p_{ij}\right)
\right].
$$

Then the forward error is

$$
\|Y-\tilde Y\|_2\le\|W_O\|D_V
\left(\sum_i\epsilon_i^2\right)^{1/2},
$$

and the dense-versus-sparse Jacobian of the entire sequence map obeys

$$
\|D_X\mathcal A(X)-D_X\tilde{\mathcal A}(X)\|_{2\to2}
\le\sqrt{R C},\qquad
R=\max_i\sum_j b_{ij},\quad C=\max_j\sum_i b_{ij}.
$$

The row sum has the simpler bound

$$
R\le\|W_O\|\max_i\epsilon_i\left[
\beta(3/2-\epsilon_i)D_KD_V\|W_Q\|
+2\|W_V\|+3\beta D_V\|q_i\|\|W_K\|
\right].
$$

The column sum is the measurable fan-out certificate

$$
C\le\|W_O\|\max_j\left[
c_j^Q\|W_Q\|
+\|W_V\|\sum_i|p_{ij}-\tilde p_{ij}|
+\beta D_V\|W_K\|F_j
\right],
\qquad
F_j=\sum_i\|q_i\|
\left(|p_{ij}-\tilde p_{ij}|+\epsilon_i\tilde p_{ij}\right).
$$

Thus omitted mass at each query is not by itself a depth certificate. The
column statistics $\sum_i|p_{ij}-\tilde p_{ij}|$ and $F_j$ measure how many
queries reuse a key in directions where truncation changes its influence. If
one key is consequential to many queries, $C$ can grow even when every
individual $\epsilon_i$ is small.

### Proof

Write $y_i=(1-\epsilon_i)y_{C_i}+\epsilon_i y_{O_i}$ for the conditional
means on retained and omitted keys. Their value diameter is at most $D_V$,
so $\|y_i-\tilde y_i\|\le\epsilon_iD_V$; stacking these bounds proves the
forward result.

For a fixed output cotangent $g$, the query derivative is the transpose of
$\beta\operatorname{Cov}_{p_i}(k,v)$. Applying the fixed-support query-VJP
bound from §112 and $\operatorname{range}_j(g^\top v_j)\le\|g\|D_V$ gives
$\|D_{q_i}y_i-D_{q_i}\tilde y_i\|\le c_i^Q$. The value block for input key
$j$ differs by $(p_{ij}-\tilde p_{ij})W_V$, whose norm is the second term of
$b_{ij}$. The key block before projection is
\[
\beta\left[p_{ij}(v_j-y_i)-\tilde p_{ij}(v_j-\tilde y_i)\right]q_i^\top.
\]
Add and subtract $\tilde p_{ij}(v_j-y_i)$; since
$\|v_j-y_i\|\le D_V$ and $\|y_i-\tilde y_i\|\le\epsilon_iD_V$, its norm
after $W_K$ and $W_O$ is bounded by the last term of $b_{ij}$. The query
projection contributes only to block $(i,i)$, yielding
\[
\|(D_X\mathcal A-D_X\tilde{\mathcal A})_{ij}\|\le b_{ij}.
\]
For any perturbation with block norms $u_j=\|\delta x_j\|$, output block
norms are bounded by the nonnegative matrix $B=(b_{ij})$ acting on $u$.
Therefore the block operator norm is at most
$\|B\|_2\le\sqrt{\|B\|_1\|B\|_\infty}=\sqrt{CR}$, the standard
row/column-sum bound. Summing $|p_{ij}-\tilde p_{ij}|$ across a query gives
$2\epsilon_i$, and $\sum_j\tilde p_{ij}=1$; these identities yield the
stated simpler row bound and the explicit column statistic $F_j$.

For a residual block with the *same* identity path,
$X\mapsto X+\alpha\mathcal A(X)$ versus
$X\mapsto X+\alpha\tilde{\mathcal A}(X)$, the identity cancels in the
difference: multiply both displayed local error bounds by $|\alpha|$. Taking
their suprema over the region in §113 supplies its forward error $\epsilon_\ell$
and Jacobian mismatch $\eta_\ell$. A fixed scalar output gate multiplies them
by its gain. An input-dependent gate adds the product-rule term
$Dg_i\,(y_i-\tilde y_i)W_O$, which must be bounded separately; it is not
hidden inside this certificate.

### What this closes and what remains

This supplies a full sequence operator bound for fixed-support, linearly
projected softmax attention, including shared-key fan-out. It can now be
composed with §113's residual-depth perturbation theorem. For a concrete deep
model, the quantities must hold uniformly over the relevant state region; the
Jacobian-Lipschitz constant $M_\ell$ in §113 and parameter-VJP discrepancies
still need bounds. Learned support changes, hard event births, and the usefulness
or alignment of the resulting task gradient are also open. The practical
certificate should log $R$, $C$, and the worst-key $F_j$ at each layer, alongside
omitted mass and the actual dense/sparse Jacobian discrepancy on small
diagnostic sequences.

**Numerical diagnostic (E114).** A central finite-difference sweep over 144
fixed-support cases (sequence lengths 3, 5, and 8; diffuse and reused keys;
multiple retained supports and temperatures) found no violation above a
$10^{-8}$ absolute tolerance. For cases whose bound is at least $10^{-8}$,
the largest observed Jacobian-error/bound ratio is 0.897 and the largest
forward-error/bound ratio is 0.687. The largest positive absolute excesses
are $1.32\times10^{-10}$ for the Jacobian and $7.4\times10^{-16}$ for the
forward map. The all-keys-retained cases have a zero theoretical bound, so
their finite-difference residue makes a relative ratio meaningless; those
cases are checked by absolute error instead. This is a small synthetic check,
not an architecture-scale result. Next compare the block formulas with
autodiff Jacobians, then compare the fan-out term with §107(h). Only after
fixed-support checks should candidate membership change; measure those
boundary updates with the existing §§19/57 counterfactual signal.
