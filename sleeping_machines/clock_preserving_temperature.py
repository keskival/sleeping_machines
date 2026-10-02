"""Frozen categorical calibration preserving the exponential first-time law.

The inverse conditional transform reuses a losing residual for an independent
uniform. This is a forward sampling contract, not an autograd route estimator.
"""
import torch
from .addressed_event_heads import AddressedEventHeads
from .race_window import bounded_delay


def calibrated_from_raw(scores,raw_times,temperature):
    if scores.shape!=raw_times.shape or scores.shape[-1]<2:
        raise ValueError('At least two common-start exponential candidates required')
    if not torch.isfinite(scores).all() or not torch.isfinite(raw_times).all() or not (raw_times>=0).all():
        raise ValueError('Finite scores and nonnegative sampled raw clocks required')
    if temperature.ndim or not torch.isfinite(temperature) or not temperature>0:
        raise ValueError('Finite positive scalar temperature required')
    s=scores.to(torch.float64);pi=s.softmax(-1);first,winner=raw_times.min(-1)
    loser=torch.where(winner==0,1,0)
    residual=raw_times.gather(-1,loser[...,None]).squeeze(-1)-first
    rate=s.exp().gather(-1,loser[...,None]).squeeze(-1)
    within=-torch.expm1(-rate*residual)
    lower=(pi.cumsum(-1)-pi).gather(-1,winner[...,None]).squeeze(-1)
    mass=pi.gather(-1,winner[...,None]).squeeze(-1)
    uniform=(lower+mass*within).clamp(0.,1.)
    calibrated=(s/temperature).softmax(-1)
    chosen=(uniform[...,None]>calibrated.cumsum(-1)).sum(-1).clamp_max(s.shape[-1]-1)
    if bool(temperature==1):chosen=winner
    return first,chosen,uniform


class ClockPreservingTemperatureHeads(AddressedEventHeads):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.route_temperature=1.;self.temperature_scope='all';self._head_counter=0

    def consume_event(self,source,timestamp,content,state):
        self._head_counter=0
        return super().consume_event(source,timestamp,content,state)

    def race(self,scores,values=None):
        depth=self._head_counter//self.heads;self._head_counter+=1
        if self.route_temperature==1. or (self.temperature_scope=='layer0' and depth!=0):
            return super().race(scores,values)
        if self.training or values is not None:
            raise ValueError('Frozen calibrated routing only; positive-temperature training not installed')
        if self.temperature_scope not in ('all','layer0'):raise ValueError('Declared scope required')
        rates=scores.to(torch.float64).exp();raw=torch.empty_like(rates).exponential_()/rates
        first,winner,_=calibrated_from_raw(scores,raw,torch.tensor(self.route_temperature,dtype=torch.float64))
        return None,bounded_delay(first),winner
