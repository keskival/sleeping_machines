"""Fitting-only paired-view geometry for an affine event-query classifier."""
import torch


def paired_view_metric(clean, transformed, relative_ridge=1e-4, noise_weight=1.):
    if clean.shape!=transformed.shape or clean.ndim!=2 or relative_ridge<=0 or noise_weight<0:
        raise ValueError('Aligned views and positive regularization required')
    a,b=clean.double(),transformed.double()
    both=torch.cat((a,b));mean=both.mean(0);x=both-mean
    covariance=x.T@x/len(x)
    delta=b-a
    within=delta.T@delta/(4*len(a))
    ridge=relative_ridge*float(covariance.trace()/a.shape[1])
    robust=covariance+noise_weight*within+ridge*torch.eye(a.shape[1],dtype=a.dtype)
    values,vectors=torch.linalg.eigh(robust)
    metric=(vectors*values.rsqrt())@vectors.T
    between=(a+b)/2-mean
    decomposition_error=float((covariance-between.T@between/len(a)-within).abs().max())
    return mean,metric,dict(noise_weight=noise_weight,relative_ridge=relative_ridge,
        absolute_ridge=ridge,between_trace=float((between.T@between/len(a)).trace()),
        within_trace=float(within.trace()),view_covariance_trace=float(covariance.trace()),
        covariance_decomposition_max_error=decomposition_error)


def paired_logit_risk(clean_logits, transformed_logits, labels):
    a,b=clean_logits.double(),transformed_logits.double();middle=(a+b)/2
    loss=torch.nn.functional.cross_entropy
    jensen=(loss(a,labels,reduction='none')+loss(b,labels,reduction='none'))/2-loss(middle,labels,reduction='none')
    p=middle.softmax(-1)
    kl=(p*((middle.log_softmax(-1)-(a.log_softmax(-1)+b.log_softmax(-1))/2))).sum(-1)
    delta=b-a;delta=delta-delta.mean(-1,keepdim=True)
    # Each of the two centered logit perturbations is +/- delta/2.
    upper=delta.square().sum(-1)/16
    return dict(n=len(a),mean_jensen_ce_penalty=float(jensen.mean()),
        mean_exact_kl_penalty=float(kl.mean()),mean_kl_upper_bound=float(upper.mean()),
        identity_max_error=float((jensen-kl).abs().max()),
        maximum_bound_violation=float((jensen-upper).max()),
        paired_prediction_changes=int((a.argmax(-1)!=b.argmax(-1)).sum()))
