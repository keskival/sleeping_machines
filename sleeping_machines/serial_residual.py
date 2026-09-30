"""Serial event-depth extension with an explicitly owned correction query.

The suffix consumes the trained prefix's vector messages and actual arrival
times. It is a serial deeper event network, not a second raw-input encoder.
Nonzero suffix state/maps supply features to a zero correction head. Prefix
values/classifier are immutable during the declared extension-training phase.
"""
import copy
import torch
from torch import nn
from .event_state import EventStateEncoder


class SerialResidualEncoder(nn.Module):
    def __init__(self,prefix,suffix_depth=6):
        super().__init__()
        self.prefix=copy.deepcopy(prefix).requires_grad_(False).eval()
        width=prefix.embedding.embedding_dim;first=prefix.layers[0]
        self.suffix=EventStateEncoder(width,width=width,modes=first.modes,
            depth=suffix_depth,classes=prefix.head.out_features,options=first.options)
        self.suffix.to(device=prefix.head.weight.device,dtype=prefix.head.weight.dtype)
        with torch.no_grad():
            self.suffix.project.weight.copy_(torch.eye(width,device=prefix.head.weight.device,dtype=prefix.head.weight.dtype))
            self.suffix.project.bias.zero_()
            self.suffix.head.weight.zero_();self.suffix.head.bias.zero_()

    def train(self,mode=True):
        super().train(mode);self.prefix.eval();return self

    def forward(self,source,raw_times,assignment,times,receivers,counts,size,sequential=False):
        with torch.no_grad():
            baseline,messages,arrival,prefix_stats=self.prefix(source,raw_times,assignment,
                times,receivers,counts,size,sequential)
        correction,payload,arrival,suffix_stats=self.suffix(messages,arrival,receivers,counts,size,sequential)
        suffix_stats[0]['stem_projection_macs']=len(messages)*self.suffix.project.in_features*self.suffix.project.out_features
        return baseline+correction,payload,arrival,dict(prefix=prefix_stats,suffix=suffix_stats,
            baseline_logits=baseline,correction_logits=correction,
            total_depth=len(self.prefix.layers)+len(self.suffix.layers))
