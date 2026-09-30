"""Integrated sparse temporal language network, without a dense carrier.

Each observed event traverses hard races over small addressed pools. Exactly
one content-bearing unit per depth updates persistent state and emits a value.
Unselected units remain dormant. Training evaluates the addressed losing units
for a conserved counterfactual score teacher, and charges that work. Pools are
indexed by the observed source character: topology discovery is not claimed.
"""
from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F

from .parallel_stream_language import precise_rotate


class TemporalRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx,scores,values):
        rates=scores.to(torch.float64).exp()
        noise=torch.empty_like(rates).exponential_()
        times=noise/rates
        time,winner=times.min(dim=0)
        ctx.save_for_backward(rates,time,winner,values)
        # Monotone bounded delay preserves race identity and source order.
        delay=.001+.010*time/(1+time)
        return values[winner],delay,winner

    @staticmethod
    def backward(ctx,error_value,error_delay,error_winner):
        rates,time,winner,values=ctx.saved_tensors
        credit=torch.zeros_like(rates)
        value_credit=torch.zeros_like(values)
        if error_value is not None:
            centered=values-values.mean(0)
            direction=centered @ error_value
            credit=rates*time*direction.to(rates.dtype)
            # Conserved counterfactual score teacher. This is a local surrogate
            # for nonlinear sampled losses, not exact credit for route changes.
            credit=credit.scatter_add(0,winner[None],-credit.sum()[None])
            value_credit=value_credit.index_add(0,winner[None],error_value[None])
        if error_delay is not None:
            # Exact interior derivative of the realized winner's arrival time.
            timing=-error_delay*.010*time/(1+time).square()
            credit=credit.scatter_add(0,winner[None],timing[None])
        return credit.to(values.dtype),value_credit


class TemporalUnit(nn.Module):
    def __init__(self,payload,gain):
        super().__init__();self.payload,self.gain=payload,gain
        self.key=nn.Parameter(torch.randn(payload)*.1)
        self.key_read=nn.Linear(payload,payload,bias=False)
        self.clock_bias=nn.Parameter(torch.tensor(0.))
        self.input=nn.Linear(payload,payload,bias=False)
        self.output=nn.Linear(payload,payload,bias=False)
        self.gate=nn.Linear(payload,payload)
        self.control=nn.Linear(payload,2)
        nn.init.zeros_(self.control.weight);nn.init.zeros_(self.control.bias)
        modes=payload//2
        tau=torch.logspace(0,2,modes)
        self.raw_rate=nn.Parameter(torch.expm1(1/tau).log())
        self.frequency=nn.Parameter(torch.linspace(-math.pi/2,math.pi/2,modes))

    def propose(self,x,memory,previous,arrival):
        controls=self.control(F.layer_norm(x,(self.payload,)))
        forget=F.softplus(controls[0])/math.log(2);write=2*torch.sigmoid(controls[1])
        if previous is not None:
            age=(arrival-previous).clamp_min(0)
            rate=F.softplus(self.raw_rate)+1e-6
            memory=precise_rotate(memory*torch.exp(-age.to(x.dtype)*rate*forget).repeat_interleave(2),
                                  age*self.frequency.to(torch.float64))
        memory=memory+write*self.input(x)
        y=F.layer_norm(self.output(memory)+x,(self.payload,))
        value=x+self.gain*y*torch.sigmoid(self.gate(F.gelu(y)))
        return value,memory


@dataclass
class SparseLanguageState:
    memories: dict = field(default_factory=dict)
    arrivals: dict = field(default_factory=dict)
    position: int = 0
    deliveries: int = 0
    candidate_scores: int = 0
    counterfactual_values: int = 0
    selected_updates: int = 0
    visited_units: set = field(default_factory=set)
    context: object = None

    def detach(self):
        return SparseLanguageState({k:v.detach() for k,v in self.memories.items()},
            {k:v.detach() for k,v in self.arrivals.items()},self.position,self.deliveries,
            self.candidate_scores,self.counterfactual_values,self.selected_updates,set(self.visited_units),
            None if self.context is None else self.context.detach())


class SparseRaceLanguageModel(nn.Module):
    def __init__(self,payload=16,depth=6,pool=2,vocabulary=27):
        super().__init__()
        if payload<2 or payload%2 or min(depth,pool,vocabulary)<1 or depth*.011>=.5:
            raise ValueError('Even positive payload and bounded-depth query contract required')
        self.payload,self.depth,self.pool,self.vocabulary=payload,depth,pool,vocabulary
        self.embedding=nn.Embedding(vocabulary,payload)
        self.source_gate=nn.Linear(payload,payload)
        nn.init.zeros_(self.source_gate.weight);nn.init.constant_(self.source_gate.bias,-2.)
        self.queries=nn.ModuleList([nn.Linear(payload,payload,bias=False) for _ in range(depth)])
        self.units=nn.ModuleList([nn.ModuleList([
            nn.ModuleList([TemporalUnit(payload,1/depth) for _ in range(pool)])
            for _ in range(vocabulary)]) for _ in range(depth)])
        self.head=nn.Linear(payload,vocabulary)

    @property
    def capacity_units(self):return self.depth*self.vocabulary*self.pool

    def new_state(self):return SparseLanguageState()

    def consume(self,token,state):
        token=int(token);x=self.embedding.weight[token]
        if state.context is not None:
            x=F.layer_norm(x+torch.sigmoid(self.source_gate(x))*state.context,(self.payload,))
        arrival=torch.tensor(float(state.position),dtype=torch.float64,device=x.device)
        for depth,pools in enumerate(self.units):
            units=pools[token]
            query=self.queries[depth](F.layer_norm(x,(self.payload,)))
            scores=torch.stack([(query @ (unit.key+unit.key_read(
                state.memories.get((depth,token,i),x.new_zeros(self.payload))))) /
                math.sqrt(self.payload)+unit.clock_bias for i,unit in enumerate(units)])
            # Bounded physical rates protect the numerical emulator; no sum or
            # division normalizes probabilities in the winner path.
            scores=scores.clamp(-12,12)
            if self.training:
                proposals=[unit.propose(x,state.memories.get((depth,token,i),x.new_zeros(self.payload)),
                    state.arrivals.get((depth,token,i)),arrival) for i,unit in enumerate(units)]
                candidates=torch.stack([p[0] for p in proposals])
                value,delay,winner=TemporalRoute.apply(scores,candidates)
                index=int(winner);memory=proposals[index][1]
                state.counterfactual_values+=self.pool
            else:
                rates=scores.to(torch.float64).exp()
                time,winner=(torch.empty_like(rates).exponential_()/rates).min(0)
                index=int(winner);delay=.001+.010*time/(1+time)
                value,memory=units[index].propose(x,state.memories.get((depth,token,index),x.new_zeros(self.payload)),
                    state.arrivals.get((depth,token,index)),arrival)
            address=(depth,token,index)
            state.memories[address]=memory;state.arrivals[address]=arrival
            state.visited_units.add(address);state.candidate_scores+=self.pool
            state.selected_updates+=1;state.deliveries+=1
            x,arrival=value,arrival+delay
        state.context=x;state.position+=1
        return self.head(x)

    def forward_chunk(self,tokens,state=None):
        state=self.new_state() if state is None else state
        indices=torch.as_tensor(tokens,dtype=torch.long,device=self.embedding.weight.device)
        if indices.ndim!=1 or not len(indices):raise ValueError('Nonempty character stream required')
        return torch.stack([self.consume(token,state) for token in indices.tolist()]),state
