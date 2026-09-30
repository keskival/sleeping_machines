# Identity depth growth with live event-state output teachers

Derived 30 September 2026. This supplies an explicit growth construction for
the modal event encoder, rather than assuming that arbitrary extra layers will
train. It preserves an untimed completed-query classifier at initialization.
Physical event latency increases; a fixed deadline is a different objective.

Function-preserving network growth is established prior art, including
[Net2Net](https://arxiv.org/abs/1511.05641). The specific work here is the
event-state/gated/LayerNorm initialization, old-teacher equivalence, live new
output teachers and the explicit event-latency boundary.

## 241. Preserve the function without deleting the new learning signal

A block has state s_i, message x_i and emitted value

\[
 v_i=x_i+\alpha\,\mathrm{LN}(Cs_i+D x_i)
 \odot\sigma(G\,\mathrm{gelu}(\mathrm{LN}(Cs_i+D x_i))+b_G).
 \tag{241.1}
\]

Initialize only C and D at zero. Keep a nonzero input map B and stable modal
state dynamics. Set LayerNorm bias zero, gain gamma=sqrt(eps), gate bias zero,
and clock weights/biases zero. Then y=0, the gate is exactly 1/2, v_i=x_i and
the winning delay is the constant 6 ms of the existing bounded clock rule.
All actual supplied packets still produce one winning vector; losing clocks
are not propagated as if they won.

Append K such blocks without changing old blocks or their residual gains.
All emitted values and the completed count-mean query remain identical.
The final event times increase by 6K ms. This does not preserve output latency,
event traces or the answer under an earlier deadline. There are no empty time
ticks or event-pair attention maps; additional state/maps cost real work.

## 242. Exact old teachers and nonzero new output-map credit

At y=0, LayerNorm's derivative is

\[
 D\mathrm{LN}(0)=\gamma/\sqrt{\epsilon}\,P_d=P_d,
 \quad P_d=I-11^T/d. \tag{242.1}
\]

Since C=D=0, each new block's value derivative with respect to its incoming
message is I, and its value derivative with respect to incoming time is zero.
Thus the classifier's teachers to every old parameter are unchanged exactly.
The zeroed blocks do not impose a serial product of zero payload Jacobians.

For complete incoming label teacher g_i, output-map credit is instead

\[
 \nabla_C\ell=\alpha/2\sum_i(P_d g_i)s_i^T,
 \quad\nabla_D\ell=\alpha/2\sum_i(P_d g_i)\odot x_i. \tag{242.2}
\]

Here D is diagonal in the implementation. B has already created modal states,
so C can receive a teacher before the new branch has emitted a nonzero
correction. All appended blocks see the same identity-transported message
teacher, with their respective state features. They can all start output
learning in the first update. Initially B, modal poles, gates and clocks have
zero effect through these zero output maps; after C/D changes, their credit
can become observable. The final untimed output clock stays unobserved.

The summed correlations in (242.2) can vanish, and a class-degenerate head
can make P_d g_i zero. Nonzero input maps alone are not a universal gradient
lower bound. E151 measures the actual output teachers in its declared example.
Setting gamma=sqrt(eps) avoids LayerNorm's otherwise large 1/sqrt(eps)
output-map derivative at zero. This is a scale choice derived from the local
Jacobian, rather than a guessed noisy initialization.

## 243. A trainable neighborhood around the identity

For payload Jacobians J_l=I+E_l with ||E_l||_2<=eta_l<1, singular-value bounds
give

\[
 \prod_l(1-\eta_l)\leq\sigma_{min}(J_L\cdots J_1),\qquad
 \|J_L\cdots J_1\|_2\leq\prod_l(1+\eta_l). \tag{243.1}
\]

Keeping sum eta_l bounded as depth grows retains finite transport bounds.
The identity initialization starts with eta_l=0 for the appended payload path.
This establishes a locally conditioned starting point, not a global training
guarantee: state sensitivities, changed event order, learned norms and parameter
updates must also be controlled. A fixed small residual gain alone does not
enforce the hypotheses. The implementation does not claim such enforcement.

## 244. Preserve optimizer ownership when growing

Equal old gradients do not imply equal updates if all old/new teachers are
globally clipped together. New output teachers can change the clipping factor
of old parameters, as §§224–225 showed. Exact first-update preservation requires
the old optimizer state, grouping, rate and clipping operation to remain the
same; new parameters get a separate declared update group and clipping budget.
Later updates intentionally couple the representations and need not match.
An optimizer-reset depth experiment must be identified as such.

`depth_growth.py` copies the encoder, retains every existing parameter and
residual gain, then appends the specified blocks. E151 checks D6 -> D12
function, payload and old-parameter teachers, constant added delay, live C
teachers in all six new blocks, and formula (242.2). A successful numerical
contract establishes the construction. Only completed SHD training can
establish that twelve blocks improve recognition or computation at a quality
target. Existing twelve-layer scattering prototypes are a different model.

**Completed E151 check:** class logits, emitted payloads and all old-parameter
teachers agree exactly in float64. The additional clock shift differs from
36 ms by at most 3.47e-17 seconds. All six new output maps receive nonzero
teachers (norms 0.0566–0.0759); explicit local credit agrees to 1.11e-16.
