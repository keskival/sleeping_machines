# New depth must be scaled in the units of its class observer

Derived 30 September 2026. The original D12 identity construction preserves
logits and old teachers, but E157's actual first-update replay passes none of
the six declared scales down to 1/1024. Even the smallest scale has fitting
anchor KL 0.03428, above the 0.02 budget. Its pre-clipping teacher norm is
51,619.8, compared with 69.93 for the D6 base. This identifies an additional
conditioning issue at the new output maps, despite correct gradient support.

## 253. A finite bound that includes the readout

For LayerNorm with equal gain gamma and zero bias,
||LN(z)||_2 <= gamma sqrt(d), since its normalized variance is at most one.
Multiplication by sigmoid gates cannot increase that norm. For K appended
residual blocks with gains alpha_l, the count-mean query displacement obeys

\[
 \|\Delta h\|_2\leq\gamma\sum_l\alpha_l\sqrt{d_l},\qquad
 \|P W\Delta h\|_2\leq\|PW\|_2\gamma\sum_l\alpha_l\sqrt{d_l}.
 \tag{253.1}
\]

The mean is a convex combination, packet counts are unchanged and residual
displacements telescope along each supplied packet. State values, C/D maps
and delays can be arbitrary in this bound: the normalized emitted correction
is bounded. Clocks may change the state used by the correction but do not drop
packets or change the query's count weights in this encoder.

The comparison uses the same old-prefix parameters and head, with appended
blocks omitted in one path. It is not a bound between independently trained
D6/D12 models, nor between fixed-deadline queries at different arrival times.
Nonzero norm bias adds sum alpha_l ||bias_l||. Learned gains must be included
at their current values; initialization alone is not a whole-training bound.

## 254. Choose a nonzero gain using observed class sensitivity

Let H=||P W||_2, S=sum alpha_l sqrt(d_l), and declare initial logit budget tau.
Initialize new gains to

\[
 \gamma_0=\min(\sqrt{\epsilon},\tau/(HS)). \tag{254.1}
\]

At exact identity, all new outputs still vanish. Their C teachers remain live,
now scaled by gamma0/sqrt(eps) in (242.2). After arbitrary map changes with
the norm gain/bias held at their initialized values, the added logit contrast
norm is bounded by tau, giving categorical KL <= tau^2/4 (§238).
This is class-observable initialization, not a raw hidden-variance heuristic.
H is measured once from fitting-derived head parameters, with no held labels.

E157 uses tau=0.05, H=912.9976, six new blocks at alpha=1/6 and width 128.
It gives gamma0=4.84056e-6, versus the original sqrt(eps)=0.00316228.
The construction retains old parameters/gains and all old teachers.

## 255. The normalization parameters need compatible update units

Merely initializing gamma small and then taking the same physical Adam step
on it would immediately lose that scale. Let s=gamma0/sqrt(eps). New norm
gain and bias therefore use learning rate s times the new-map learning rate.
For fresh moments, each coordinate's first movement is at most eta s, with
zero initial gain teacher and potentially nonzero bias teacher. Relative to
gamma0, this is at most eta/sqrt(eps) for norm bias coordinates. This bounds
their first-update contribution in the class observer's units.

The implementation retains ordinary Adam epsilon and moments; this is a
declared update-unit scale, not a claim of exact coordinate invariance of Adam.
Later gain teachers and moments can change the bound. All old parameters keep
their own clipping operation; new maps/norms are clipped together separately.
Their future cross-coordinate statistics still need measured actual descent.

## 256. Real replay separates live credit from safe finite learning

The conditioned D12 replay passes the same 1/16 factor as D6: update-batch
CE becomes 0.332072 from 0.622628, with anchor mean KL 0.004448. D6 yields
0.333181 and KL 0.004412. Original D12 at that factor gives CE 0.088498 but
anchor KL 0.074819: a strong same-batch fit does not meet the transfer check.
Its smallest tested factor still gives KL 0.034277; no original proposal is
admissible under this declared budget.

All policies use the same checkpoint, fitting inputs, teacher and actual
counterfactual replay. The map/gain/optimizer scale is the intervention;
random seeds are not varied. E155 trains the conditioned D12 and a matched
D6 continuation. A successful first-step replay does not predict their final
accuracy. The recorded E155 D12 pass gives 350/512 and online NLL 3.5073,
but subsequent provenance inspection finds that the scheduler's restored
initial_lr overwrote the intended old rate 0.0000203125 with 0.000325. Its
epoch records the latter actual rate. That run does not test the calibrated
continuation. E159 resets scheduler base-rate metadata, verifies every actual
starting group rate and repeats the predeclared matched D6/D12 experiment.
The completed E155 evidence/source remains unchanged.
