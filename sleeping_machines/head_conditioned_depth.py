"""Identity depth growth in the metric of the existing class observer.

The added blocks retain live output teachers, but their initial LayerNorm
gain is scaled by the actual class-contrast head norm. Its optimizer coordinate
scale is returned explicitly. This bounds initial normalized residual output,
not all later learned updates or fixed-deadline latency.
"""
import math
import torch
from .depth_growth import grow_event_encoder


def grow_head_conditioned(encoder,depth,initial_logit_budget=.05):
    if depth<=len(encoder.layers) or initial_logit_budget<=0:
        raise ValueError('Positive budget and greater depth required')
    grown=grow_event_encoder(encoder,depth)
    head=encoder.head.weight.detach().double()
    contrast=head-head.mean(0,keepdim=True)
    observer=float(torch.linalg.svdvals(contrast)[0])
    added=grown.layers[len(encoder.layers):]
    denominator=sum(layer.gain*math.sqrt(layer.width) for layer in added)*observer
    standard=math.sqrt(added[0].norm.eps)
    gain=min(standard,initial_logit_budget/max(denominator,1e-30))
    with torch.no_grad():
        for layer in added:layer.norm.weight.fill_(gain)
    return grown,dict(initial_logit_budget=initial_logit_budget,
        class_contrast_head_operator_norm=observer,initial_norm_gain=gain,
        original_identity_norm_gain=standard,norm_learning_rate_scale=gain/standard,
        initial_new_residual_logit_bound=gain*denominator)
