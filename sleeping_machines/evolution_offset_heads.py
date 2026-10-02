"""Reception phase calibration while retaining physical temporal evolution."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .addressed_event_heads import AddressedEventHeads
from .parallel_head_race_language import precise_rotate


class EvolutionOffsetHeads(AddressedEventHeads):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.raw_evolution_offset=nn.Parameter(torch.zeros(self.depth,self.heads,self.payload//2))

    def evolution_phase(self,depth):
        return math.pi*torch.tanh(self.raw_evolution_offset[depth].to(torch.float64))

    def transport(self,value,age,depth,head):
        age=age.clamp_min(0)
        rate=F.softplus(self.transport_rate[depth,head])+1e-6
        decayed=value*torch.exp(-age.to(value.dtype)*rate).repeat_interleave(2)
        angles=age*self.transport_frequency[depth,head].to(torch.float64)+self.evolution_phase(depth)[head]
        return precise_rotate(decayed,angles)


def parameter_block(name):
    if name.startswith('queries.') or any(s in name for s in ('.key','.clock_bias')):return 'route'
    if name.startswith('units.') or name.startswith('head.') or name.startswith('transport_') or name=='raw_evolution_offset':return 'message'
    return 'shared'
