"""Identity growth with normalized features and directional output units.

A fixed fitting-derived class observer rescales its visible singular directions;
its nullspace retains unit scale. This invertible output-map parametrization
preserves expressivity. State features are normalized before the zero output
map; supplied packets still emit only their winning vector and clock.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from .event_state import EventStateBlock


class ObserverUnits(nn.Module):
    def __init__(self,head):
        super().__init__()
        contrast=head.detach().double()-head.detach().double().mean(0,keepdim=True)
        _,singular,vh=torch.linalg.svd(contrast,full_matrices=False)
        self.register_buffer('basis',vh.T.to(head))
        self.register_buffer('shrink',(1-torch.rsqrt(1+singular.square())).to(head))

    def forward(self,weight):
        return weight-self.basis@(self.shrink[:,None]*(self.basis.T@weight))


class ObserverStateBlock(EventStateBlock):
    def __init__(self,width,modes,options,gain,head):
        if width!=2*modes:raise ValueError('This variant requires width=2*modes')
        super().__init__(width,modes,options,gain=gain)
        self.to(device=head.device,dtype=head.dtype)
        del self.direct
        self.register_buffer('direct',torch.zeros(width))
        nn.init.zeros_(self.output.weight)
        nn.init.ones_(self.norm.weight);nn.init.zeros_(self.norm.bias)
        nn.init.zeros_(self.gate.bias)
        parametrize.register_parametrization(self.output,'weight',ObserverUnits(head))

    def emit(self,x,states,times,credit=True):
        features=self.norm(states)*torch.sigmoid(self.gate(F.gelu(x)))
        value=x+self.gain*self.output(features)
        delays=.001+.010*torch.sigmoid(-self.clock(value))
        winner=delays.argmin(-1)
        chosen=delays.gather(1,winner[:,None]).squeeze(-1)
        arrival=times+chosen
        if self.training and credit and self.options>1:
            probability=torch.softmax(-delays/.002,-1)
            arrival=arrival+((probability-probability.detach())*
                (delays-chosen[:,None]).detach()).sum(-1)
        rank=(len(self.output.parametrizations.weight[0].shrink)
            if parametrize.is_parametrized(self.output,'weight') else 0)
        return value,arrival,dict(winner=winner,delays=delays,emitted_vectors=len(x),
            clock_candidates=len(x)*self.options,
            observer_weight_fold_macs=2*self.width*self.width*rank)


def grow_observer_conditioned(encoder,depth):
    if depth<=len(encoder.layers):raise ValueError('Greater depth required')
    grown=copy.deepcopy(encoder);first=encoder.layers[0]
    for _ in range(depth-len(encoder.layers)):
        layer=ObserverStateBlock(first.width,first.modes,first.options,
            first.gain,encoder.head.weight)
        layer.to(device=first.input.weight.device,dtype=first.input.weight.dtype)
        grown.layers.append(layer)
    grown.train(encoder.training)
    contrast=encoder.head.weight.detach().double()
    contrast=contrast-contrast.mean(0,keepdim=True)
    singular=torch.linalg.svdvals(contrast)
    return grown,dict(policy='Pre-normalized states; zero output maps in fixed observer units',
        norm_learning_rate_scale=1.,initial_norm_gain=1.,
        class_contrast_head_operator_norm=float(singular[0]),
        largest_observed_output_sensitivity=float((singular/torch.sqrt(1+singular.square())).max()),
        output_coordinate_minimum_scale=float(torch.rsqrt(1+singular.square()).min()),
        observer_rank_upper_bound=len(singular),preserved_hidden_nullspace=True,
        inference_fold='Materialize physical output maps once; no observer transform per packet')


def materialize_observer_maps(encoder):
    model=copy.deepcopy(encoder)
    for layer in model.layers:
        if parametrize.is_parametrized(layer.output,'weight'):
            # Replace the copied module: removing parametrizations in-place can
            # alter the dynamic Python class shared with the training instance.
            weight=layer.output.weight.detach()
            linear=nn.Linear(weight.shape[1],weight.shape[0],bias=False,
                device=weight.device,dtype=weight.dtype)
            with torch.no_grad():linear.weight.copy_(weight)
            layer.output=linear
    return model
