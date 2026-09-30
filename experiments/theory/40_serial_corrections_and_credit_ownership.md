# Train the full serial suffix through its own observed correction

Derived 30 September 2026. The previous appended blocks can alter class
decisions, but decrease held accuracy relative to their own fitted prefix.
We need to identify the risk mechanism and give the full suffix a useful
teacher, rather than count layers or nonzero parameter updates.

## 265. Exact loss decomposition of a finite learned correction

For baseline logits z, correction d, label y and p=softmax(z),

\[
 CE(z+d,y)-CE(z,y)=(p-e_y)^T d+KL(p\Vert softmax(z+d)). \tag{265.1}
\]

The first term is signed observed label credit; the second is nonnegative
finite curvature. This identity follows directly from log-sum-exp's Bregman
divergence and remains exact for any finite d. It evaluates the complete
learned event correction, including its clocks and changed internal states.
No frozen-route linearization, Hessian approximation or event-pair matrix
is required. A harmful correction may have wrong average direction, excessive
curvature, or different directions on fitting and held populations.

For z+alpha d, fixed z/d give convex CE in alpha and derivative at zero
(p-e_y)^T d. Thus fitting-only scale calibration can test excessive size;
it cannot repair a systematically wrong direction on new speakers. E167
decomposes clean/augmented fitting examples and declares four fixed scales.
It does not use development/audit labels to choose a direction or update.

## 266. Separate the baseline teacher from the suffix correction teacher

Let a trained prefix emit messages x_i at actual times t_i and provide z0.
Let a six-block suffix consume those messages/times and produce query q_theta.
Use the completed classification query

\[
 z=z_0+U q_\theta+b,\qquad U_0=0,\ b_0=0. \tag{266.1}
\]

Its hidden maps/states are initialized nonzero. At initialization the old
classifier is exact even though new suffix messages/clocks are nontrivial.
The correction head receives r q^T and r immediately. After its first nonzero
update, the suffix query receives U^T r, which propagates through all six
nonzero serial state/value maps and winning-clock dependencies. This avoids
the double-zero serial gate; it deliberately does not claim that hidden
teachers are nonzero before the head update. The last untimed clock remains
unobserved. Nonzero transport/feature correlations still require measurement.

Freezing the prefix during this phase prevents its larger observer/teacher
from changing the feature distribution or absorbing the new learner's update.
This is progressive deep training, with all twelve blocks trained in their
declared stages, not simultaneous from-scratch twelve-block training. A later
joint phase is a separate protocol requiring retained old optimizer state
and measured cross-block update transfer.

## 267. Paired supervision and correction ownership

For a clean/label-preserving augmented utterance pair, minimize

\[
 \tfrac12 CE(z_{0,c}+Uq_c+b,y)+
 \tfrac12 CE(z_{0,a}+Uq_a+b,y). \tag{267.1}
\]

This is the ordinary completed-utterance categorical objective over both
views. Its exact Jensen/KL nuisance term (§245) is already included; no
arbitrary variance reward or early-timestep labels are added. With features
and baseline fixed, head fitting is convex and includes U=b=0 as a feasible
baseline. Neural suffix adaptation is nonconvex and does not inherit global
convergence or transfer guarantees. Acceptable actual step size still needs
fitting-only replay; a clipped teacher alone is insufficient.

The serial suffix has an independently learned class observer U rather than
forcing every new temporal representation through the old W. All supplied
messages and delays enter its state computation. Its readout skip retains
the established computation while the deeper learner acquires useful credit.
This is a standard residual-learning idea used in a concrete asynchronous
serial structure; neither residual learning nor progressive growth is claimed
as novel by itself.

## 268. Sparse computation and the empirical acceptance gate

The prefix executes once and passes vector/timing packets to the suffix.
No second raw-input model, empty time ticks or event-pair attention is used.
Both class maps run only at the completed query. Local vector maps, the new
stem projection, all state updates, clock candidates and both views cost work.
Sequential gradient accumulation over the two views keeps peak memory close
to one view; it does not make the second pass free.

`serial_residual.py` implements this operator. Check exact initial classifier,
immutable prefix, immediate head credit and all six hidden teachers after
the head update. Calibrate resources/actual fitting steps, then train a declared
multi-pass full model and audit its learned correction contribution. Acceptance
requires the full deeper model to improve held quality, not only its pruned
prefix or fitting accuracy. Official-test parity remains the later benchmark gate.
