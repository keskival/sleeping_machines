"""Small query-local affine/quadratic residual over a frozen event encoder.

The degree-2 map is a standard polynomial feature primitive. Temporal encoder
and arrival dynamics are preserved; this diagnostic adds no oracle bit input.
"""
import torch
from torch import nn


class PolynomialQueryReadout(nn.Module):
    def __init__(self,width,degree=2):
        super().__init__()
        if width<1 or degree not in (1,2):raise ValueError('Positive width and degree1/2 required')
        self.width,self.degree=width,degree
        self.register_buffer('center',torch.zeros(width,dtype=torch.float64))
        self.register_buffer('scale',torch.ones(width,dtype=torch.float64))
        self.register_buffer('pairs',torch.triu_indices(width,width))
        size=1+width+(self.pairs.shape[1] if degree==2 else 0)
        self.coefficients=nn.Parameter(torch.zeros(size,dtype=torch.float64))

    def fit_statistics(self,features):
        with torch.no_grad():
            x=features.to(torch.float64)
            self.center.copy_(x.mean(0));self.scale.copy_(x.std(0,unbiased=False).clamp_min(1e-3))

    def design(self,features):
        h=(features.to(torch.float64)-self.center)/self.scale
        parts=[torch.ones_like(h[...,:1]),h]
        if self.degree==2:parts.append(h[...,self.pairs[0]]*h[...,self.pairs[1]])
        return torch.cat(parts,-1)

    def forward(self,features,base_logit):
        return base_logit.to(torch.float64)+self.design(features)@self.coefficients


@torch.no_grad()
def query_features(encoder,row,noise_seed):
    """Process observed inputs only; return the actual final native head input."""
    encoder.eval();box={}
    def observe(module,args):box['feature']=args[0].detach()
    handle=encoder.head.register_forward_pre_hook(observe)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(noise_seed+row['group'])
            logits,state=encoder.forward_chunk(torch.tensor(row['inputs']))
        return box['feature'],logits[-1,1]-logits[-1,0],state
    finally:handle.remove()


@torch.no_grad()
def integrated_query_logit(encoder,readout,row,noise_seed):
    feature,base,state=query_features(encoder,row,noise_seed)
    return readout(feature,base),state
