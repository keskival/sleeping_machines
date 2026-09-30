"""Causal, indexed race retrieval alongside the unchanged language carrier.

This is an architectural probe, not a claim of a wholly sparse language model.
An inverted index uses the last observed character to shortlist earlier prefix
representations. Keys describe those prefixes; values describe their observed
successors. A query never inserts an unknown successor. Each exponential race
delivers one value. A conserved counterfactual teacher reads the shortlisted
losing values during training and is charged separately.
"""
from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F

from .selective_stream_language import SelectiveEventLanguageModel


class WinnerRace(torch.autograd.Function):
    """Winner delivery with a local, time-based counterfactual score teacher.

    The teacher's expectation equals the soft-attention output Jacobian for a
    fixed candidate set and fixed upstream cotangent. With a cotangent depending
    on the sampled output it is a surrogate, not an unbiased loss gradient.
    Value gradients follow the actual delivered winners. No explicit softmax is
    evaluated in this arm. RNG and emulator traffic are additional costs.
    """
    @staticmethod
    def forward(ctx, scores, values, races):
        rates = torch.exp((scores-scores.detach().max()).to(torch.float64))
        noise = torch.empty((races,len(scores)),dtype=torch.float64,device=scores.device).exponential_()
        times, winners = (noise/rates).min(dim=1)
        ctx.save_for_backward(rates, times, winners, values)
        return values.index_select(0,winners).mean(dim=0)

    @staticmethod
    def backward(ctx, error):
        rates, times, winners, values = ctx.saved_tensors
        count = len(times)
        # A deterministic baseline reduces clock-teacher variance. The original
        # uncentered time sum has one shared Exp(1) amplitude, not independent
        # per-key noise. Conservation holds for every realization below.
        centered = values-values.mean(dim=0)
        directional = centered @ error
        weighted_sum = (rates*directional.to(rates.dtype)).sum()
        score_credit = rates*times.mean()*directional.to(rates.dtype)
        score_credit = score_credit.scatter_add(0,winners,-times*weighted_sum/count)
        value_credit = torch.zeros_like(values).index_add(
            0,winners,error.expand(count,-1)/count)
        return score_credit.to(values.dtype), value_credit, None


@dataclass
class RaceStreamState:
    core: object
    buckets: dict = field(default_factory=dict)
    previous_key: object = None
    previous_token: object = None
    previous_position: object = None
    retrieval_queries: int = 0
    scored_keys: int = 0
    delivered_values: int = 0
    teaching_value_visits: int = 0

    @property
    def deliveries(self):
        return self.core.deliveries+self.delivered_values

    def detach(self):
        return RaceStreamState(self.core.detach(),
            {i:[(k.detach(),v.detach(),t) for k,v,t in rows] for i,rows in self.buckets.items()},
            None if self.previous_key is None else self.previous_key.detach(),
            self.previous_token,self.previous_position,self.retrieval_queries,
            self.scored_keys,self.delivered_values,self.teaching_value_visits)


class RaceLanguageModel(nn.Module):
    def __init__(self,width=128,modes=64,depth=6,payload=16,capacity=8,races=4,attention='race'):
        super().__init__()
        if min(payload,capacity,races)<1 or attention not in ('race','softmax'):
            raise ValueError('Positive payload/capacity/races and a declared attention arm required')
        self.core = SelectiveEventLanguageModel(width=width,modes=modes,depth=depth)
        self.payload,self.capacity,self.races,self.attention = payload,capacity,races,attention
        self.query = nn.Linear(width+27,payload)
        self.key = nn.Linear(width+27,payload)
        self.value = nn.Linear(width,payload)
        self.readout = nn.Linear(payload,27,bias=False)
        nn.init.normal_(self.readout.weight,std=.01)
        self.raw_recency = nn.Parameter(torch.tensor(-4.))
        self.scale = nn.Parameter(torch.tensor(0.))

    def new_state(self):
        return RaceStreamState(self.core.new_state())

    def forward_chunk(self,tokens,state=None):
        state = self.new_state() if state is None else state
        indices = torch.as_tensor(tokens,dtype=torch.long,device=self.core.embedding.weight.device)
        origin = state.core.position
        base,state.core = self.core.forward_chunk(indices,state.core)
        embeddings = self.core.embedding(indices)
        features = F.layer_norm(torch.cat((embeddings,base),dim=-1),(self.core.width+27,))
        queries,keys,values = self.query(features),self.key(features),self.value(embeddings)
        outputs = []
        for j,token in enumerate(indices.tolist()):
            position = origin+j
            if state.previous_key is not None:
                rows = state.buckets.setdefault(state.previous_token,[])
                # The current observed token supplies the previous prefix's
                # successor value; it cannot be used before this observation.
                rows.append((state.previous_key,values[j],state.previous_position))
                if len(rows)>self.capacity:del rows[0]
            rows = state.buckets.get(token,[])
            correction = base[j].new_zeros(27)
            if rows:
                bank_keys = torch.stack([r[0] for r in rows])
                bank_values = torch.stack([r[1] for r in rows])
                # Integer relative positions avoid cancellation of absolute
                # sub-token clocks, including beyond 10M characters.
                age = torch.tensor([position-r[2] for r in rows],device=base.device,dtype=base.dtype)
                scores = bank_keys @ queries[j]/math.sqrt(self.payload)
                scores = scores-F.softplus(self.raw_recency)*torch.log1p(age)
                if self.attention=='race':
                    message = WinnerRace.apply(scores,bank_values,self.races)
                    delivered = self.races
                else:
                    message = torch.softmax(scores,dim=0) @ bank_values
                    delivered = len(rows)
                correction = torch.sigmoid(self.scale)*self.readout(message)
                state.retrieval_queries += 1
                state.scored_keys += len(rows)
                state.delivered_values += delivered
                if self.training:state.teaching_value_visits += len(rows)
            outputs.append(base[j]+correction)
            state.previous_key,state.previous_token,state.previous_position = keys[j],token,position
        return torch.stack(outputs),state
