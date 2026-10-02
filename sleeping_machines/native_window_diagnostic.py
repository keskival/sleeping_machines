"""Frozen one-site native reception intervention; no training credit installed."""
import math
import torch
from torch.nn import functional as F
from .addressed_event_heads import AddressedEventHeads
from .race_window import bounded_delay


class NativeWindowDiagnostic(AddressedEventHeads):
    """Receive at event19/layer0/head0; all other native operators unchanged.

    Positive widths are evaluation-only. Selected receiver memory proposals keep
    the original native admission timestamp; emitted values are physically
    transported to the window deadline before the ordinary head/depth join.
    """
    modes=('window_sum','window_mean','winner_gain','winner_wait','delivery_only_sum')

    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.window_width=0.;self.window_mode='window_mean'

    def consume_event(self,source,timestamp,content,state):
        if self.window_width==0:return super().consume_event(source,timestamp,content,state)
        if self.training:raise ValueError('Frozen diagnostic only; positive-width training not installed')
        if not 0<self.window_width<=.011 or self.window_mode not in self.modes:
            raise ValueError('Declared positive width and control mode required')
        source,timestamp=int(source),float(timestamp)
        if not 0<=source<self.sources or not math.isfinite(timestamp) or timestamp<state.last_input_time:
            raise ValueError('Known address and finite nondecreasing times required')
        mark=torch.as_tensor(content,dtype=self.embedding.weight.dtype,device=self.embedding.weight.device)
        if mark.shape!=(self.content_dim,) or not bool(torch.isfinite(mark).all()):
            raise ValueError('Finite observed content required')
        x=self.embedding.weight[source]+self.content(mark)
        arrival=torch.tensor(timestamp,dtype=torch.float64,device=x.device)
        if source in state.contexts:
            previous,times=state.contexts[source];ready=times.max();read_time=torch.maximum(arrival,ready)
            state.queue_wait_sum+=float((read_time-arrival).detach());arrival=read_time
            context=self.align(list(previous.split(self.payload)),times,arrival,self.depth-1)
            x=F.layer_norm(x+torch.sigmoid(self.source_gate(x))*context,(self.total_payload,))
        for depth in range(self.depth):
            mixed=self.channel_mix[depth](x);all_features=F.layer_norm(mixed,(self.total_payload,))
            values,arrivals=[],[]
            for head in range(self.heads):
                incoming=mixed[head*self.payload:(head+1)*self.payload]
                units=self.units[depth][head][source];query=self.queries[depth][head](all_features)
                scores=torch.stack([(query@(unit.key+unit.key_read(state.memories.get(
                    (depth,head,source,i),incoming.new_zeros(self.payload)))))/math.sqrt(self.payload)+unit.clock_bias
                    for i,unit in enumerate(units)]).clamp(-12,12)
                if (state.events,depth,head)==(19,0,0):
                    rates=scores.to(torch.float64).exp()
                    raw=torch.empty_like(rates).exponential_()/rates;first,winner=raw.min(0);index=int(winner)
                    delays=bounded_delay(raw);deadline=arrival+bounded_delay(first)+self.window_width
                    heard=[i for i in range(self.pool) if bool(arrival+delays[i]<=deadline)]
                    assert index in heard
                    project=heard if self.window_mode in ('window_sum','window_mean','delivery_only_sum') else [index]
                    proposals={i:units[i].propose(incoming,state.memories.get((depth,head,source,i),incoming.new_zeros(self.payload)),
                        state.arrivals.get((depth,head,source,i)),arrival) for i in project}
                    commit=heard if self.window_mode in ('window_sum','window_mean') else [index]
                    for i in commit:
                        address=(depth,head,source,i);state.memories[address]=proposals[i][1];state.arrivals[address]=arrival
                        state.visited_units.add(address)
                    transported=[self.transport(proposals[i][0],deadline-(arrival+delays[i]),depth,head) for i in project]
                    value=torch.stack(transported).sum(0)
                    if self.window_mode=='window_mean':value=value/len(heard)
                    if self.window_mode=='winner_gain':value=value*len(heard)
                    state.selected_updates+=len(commit);state.candidate_scores+=self.pool
                    state.window_diagnostic=dict(heard=heard,winner=index,selected_writes=len(commit),
                        projected_values=len(project),deadline=float(deadline),incoming_admission=float(arrival),
                        first_arrival=float(arrival+bounded_delay(first)),local_window_s=self.window_width)
                    values.append(value);arrivals.append(deadline)
                else:
                    _,delay,winner=self.race(scores);index=int(winner)
                    value,memory=units[index].propose(incoming,state.memories.get((depth,head,source,index),incoming.new_zeros(self.payload)),
                        state.arrivals.get((depth,head,source,index)),arrival)
                    address=(depth,head,source,index);state.memories[address]=memory;state.arrivals[address]=arrival
                    state.visited_units.add(address);state.candidate_scores+=self.pool;state.selected_updates+=1
                    values.append(value);arrivals.append(arrival+delay)
            arrival=torch.stack(arrivals).max();x=self.align(values,arrivals,arrival,depth)
        state.contexts[source]=(torch.cat(values),torch.stack(arrivals));state.events+=1;state.last_input_time=timestamp
        return self.head(x),arrival
