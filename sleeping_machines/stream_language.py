"""Persistent language stream using the shared signed event-state primitive.

One source token is consumed once. Pending vector messages execute in actual
arrival order; only event arrivals evaluate blocks. Time units are token
intervals, not speech milliseconds. A query processes pending messages up to
its declared deadline and reads the last completed deep message. This is a
new generic path; it is not the native count/copy mixture or an old benchmark.
"""
from dataclasses import dataclass, field
import heapq
import torch
from torch import nn
from torch.nn import functional as F
from .event_state import EventStateBlock
from .rotating_memory import rotate_pairs


@dataclass
class LanguageStreamState:
    modes: list
    last_arrival: list
    output: torch.Tensor
    pending: list = field(default_factory=list)
    position: int = 0
    serial: int = 0
    deliveries: int = 0

    def detach(self):
        """Truncate credit without resetting causal memory or pending messages."""
        return LanguageStreamState([x.detach() for x in self.modes],
            [None if t is None else t.detach() for t in self.last_arrival],
            self.output.detach(),[(key,serial,node,x.detach(),t.detach())
                                 for key,serial,node,x,t in self.pending],
            self.position,self.serial,self.deliveries)


class StreamingEventLanguageModel(nn.Module):
    def __init__(self,vocabulary=27,width=32,modes=16,depth=8,query_budget=.5):
        super().__init__()
        if min(vocabulary,width,modes,depth)<1 or not 0<=query_budget<1:
            raise ValueError('Positive capacity and a query before the next token required')
        self.vocabulary,self.width,self.depth=vocabulary,width,depth
        self.query_budget=float(query_budget)
        self.embedding=nn.Embedding(vocabulary,width)
        self.layers=nn.ModuleList([EventStateBlock(width,modes,3,gain=1/depth)
                                  for _ in range(depth)])
        self.head=nn.Linear(width,vocabulary)

    def new_state(self):
        p=self.embedding.weight
        return LanguageStreamState([p.new_zeros(2*layer.modes) for layer in self.layers],
            [None]*self.depth,p.new_zeros(self.width))

    def _enqueue(self,state,node,payload,arrival):
        # Discrete schedule follows the realized forward time; tensor time
        # retains the local delay derivative used by the event-state primitive.
        heapq.heappush(state.pending,(float(arrival.detach()),state.serial,node,payload,arrival))
        state.serial+=1

    def advance(self,state,cutoff,readout=True):
        """Execute arrivals already due. No evaluation on empty time ticks."""
        while state.pending and state.pending[0][0]<=cutoff:
            _,_,node,x,t=heapq.heappop(state.pending)
            if node==self.depth:
                state.output=x
                continue
            layer=self.layers[node]
            previous=state.last_arrival[node]
            h=state.modes[node]
            if previous is not None:
                dt=(t-previous).clamp_min(0)
                rate=F.softplus(layer.raw_rate)+1e-6
                decay=torch.exp(-dt*rate).repeat_interleave(2)
                h=rotate_pairs(h*decay,dt*layer.frequency)
            h=h+layer.input(x)
            state.modes[node]=h;state.last_arrival[node]=t
            value,arrival,_=layer.emit(x[None],h[None],t[None])
            state.deliveries+=1
            self._enqueue(state,node+1,value[0],arrival[0])
        return self.head(state.output) if readout else None

    def consume(self,token,state):
        """Observe one token, then predict the next before it is observed.

        The scalar arrival index is a declared serialization of text. It is
        not a claim about physical language timestamps or a learned tokenizer.
        """
        if not 0<=int(token)<self.vocabulary:raise ValueError('Unknown token')
        arrival=float(state.position)
        self.advance(state,arrival,readout=False)
        index=torch.tensor(int(token),device=self.embedding.weight.device)
        self._enqueue(state,0,self.embedding(index),self.embedding.weight.new_tensor(arrival))
        prediction=self.advance(state,arrival+self.query_budget)
        state.position+=1
        return prediction

    def forward_chunk(self,tokens,state=None):
        state=self.new_state() if state is None else state
        logits=[self.consume(token,state) for token in tokens]
        if not logits:raise ValueError('Empty token chunk')
        return torch.stack(logits),state
