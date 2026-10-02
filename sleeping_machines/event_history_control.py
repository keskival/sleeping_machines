"""Diagnostic learned decoder over causal three-mark history; no race mechanisms."""
import math
import torch
from torch import nn


class EventHistoryControl(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.decoder=nn.Sequential(nn.Linear(9,width),nn.GELU(),nn.Linear(width,4))

    def predict_population(self,row):
        state={};logits=[];targets=[];last=-math.inf
        for event in row:
            if not math.isfinite(event.time) or event.time<last:raise ValueError('Causal finite timestamps required')
            last=event.time;now=torch.tensor(event.time,dtype=self.decoder[0].weight.dtype)
            history=state.setdefault(event.source,[])
            if event.mark[1]==0:
                history.append((torch.tensor(event.mark[0],dtype=now.dtype),now))
                del history[:-3]
            else:
                marks=[];ages=[];valid=[]
                for index in range(3):
                    if index<len(history):
                        value,timestamp=history[index];marks.append(value);ages.append((now-timestamp)/10);valid.append(now.new_tensor(1.))
                    else:
                        marks.append(now.new_zeros(()));ages.append(now.new_zeros(()));valid.append(now.new_zeros(()))
                features=torch.stack(marks+ages+valid)
                logits.append(self.decoder(features))
                # Target read occurs only after constructing predictions.
                targets.append(event.target)
        return torch.stack(logits),torch.tensor(targets),state
