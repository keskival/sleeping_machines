"""Fold a parallel logit component into an existing affine event readout.

Only fitting features/teacher logits enter the projection. Whitening is an
offline optimizer coordinate choice; its affine map is folded into the head.
This does not make a nonlinear teacher exactly representable by a fixed query.
"""
import torch


def project_component(features, component, relative_ridge=1e-4):
    if features.ndim != 2 or component.ndim != 2 or len(features) != len(component):
        raise ValueError("Aligned query features and class logits required")
    if len(features) < 2 or relative_ridge <= 0:
        raise ValueError("At least two fitting examples and positive ridge required")
    h, target = features.double(), component.double()
    # Categorical likelihood ignores the common logit offset.
    target = target-target.mean(-1, keepdim=True)
    mean, intercept = h.mean(0), target.mean(0)
    x, y = h-mean, target-intercept
    covariance = x.T@x/len(x)
    ridge = relative_ridge*float(covariance.trace()/h.shape[1])
    if ridge <= 0:
        raise ValueError("Readout features have no variation")
    values, vectors = torch.linalg.eigh(covariance)
    metric = (vectors*(values.clamp_min(0)+ridge).rsqrt())@vectors.T
    weight = torch.linalg.solve(covariance+ridge*torch.eye(h.shape[1], dtype=h.dtype), x.T@y/len(x))
    bias = intercept-mean@weight
    error = h@weight+bias-target
    stats = dict(relative_ridge=relative_ridge,absolute_ridge=ridge,
        feature_eigenvalue_min=float(values.min()),feature_eigenvalue_max=float(values.max()),
        centered_component_mse=float(error.square().mean()),
        mean_logit_kl_upper_bound=float(error.square().sum(-1).mean()/4),
        centered_component_rms=float(target.square().mean().sqrt()))
    return weight, bias, mean, metric, stats


def optimize_head(features, teacher_logits, labels, initial_weight, initial_bias,
                  mean, metric, teacher_weight=.5, iterations=120, ridge=1e-5):
    """Convex fit-only head optimization in whitened coordinates.

    Head weights use the Linear convention [classes, width]. Positive ridge is
    applied to centered whitened weights, with an unpenalized intercept.
    Teacher forward work is charged by the caller and absent from deployment.
    """
    if teacher_weight < 0 or min(iterations, ridge) <= 0:
        raise ValueError("Nonnegative teacher weight and positive fit settings required")
    h, teacher = features.double(), teacher_logits.double()
    z = (h-mean)@metric
    coef = torch.linalg.solve(metric, initial_weight.double().T).detach().contiguous().requires_grad_()
    bias = (initial_bias.double()+mean@initial_weight.double().T).detach().requires_grad_()
    target = (torch.nn.functional.one_hot(labels,teacher.shape[1]).double()+
        teacher_weight*teacher.softmax(-1))/(1+teacher_weight)
    opt = torch.optim.LBFGS([coef,bias],lr=1.,max_iter=iterations,
        tolerance_grad=1e-7,tolerance_change=1e-10,line_search_fn="strong_wolfe")
    history=[]
    def closure():
        opt.zero_grad()
        centered = coef-coef.mean(-1,keepdim=True)
        score=z@centered+bias-bias.mean()
        loss=-(target*score.log_softmax(-1)).sum(-1).mean()+ridge*centered.square().sum()/2
        loss.backward();history.append(float(loss.detach()))
        return loss
    opt.step(closure)
    with torch.no_grad():
        coef=coef-coef.mean(-1,keepdim=True)
        intercept=bias-bias.mean()
        weight=metric@coef
        folded_bias=intercept-mean@weight
        score=h@weight+folded_bias
        target_loss=float(-(target*score.log_softmax(-1)).sum(-1).mean())
    return weight.T,folded_bias,dict(teacher_weight=teacher_weight,iterations=iterations,
        closure_evaluations=len(history),objective_initial=history[0],objective_final=history[-1],
        unregularized_soft_target_ce=target_loss,coordinate_ridge=ridge)


def absorption_certificate(teacher, student):
    """Sufficient label-preservation and global categorical KL bounds."""
    error=student-teacher
    error=error-error.mean(-1,keepdim=True)
    top=teacher.topk(2,-1).values
    margin=top[:,0]-top[:,1]
    certified=2*error.abs().amax(-1)<margin
    kl=(teacher.softmax(-1)*(teacher.log_softmax(-1)-student.log_softmax(-1))).sum(-1)
    bound=error.square().sum(-1)/4
    return dict(n=len(teacher),teacher_decisions_certified=int(certified.sum()),
        changed_decisions=int((teacher.argmax(-1)!=student.argmax(-1)).sum()),
        teacher_to_student_kl=float(kl.mean()),mean_kl_upper_bound=float(bound.mean()),
        maximum_bound_violation=float((kl-bound).max()),
        certified_decision_failures=int((certified&(teacher.argmax(-1)!=student.argmax(-1))).sum()))
