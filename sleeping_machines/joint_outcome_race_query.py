"""Protected causal outcomes, two learned key races and a bilinear query.

Observed predecessor addresses are generic, never label-selected. The native
temporal core remains; exact terminal expected-risk credit has a separate scope
from its existing earlier-route surrogate.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .native_stream_language import NativeStreamLanguageModel,NativeLanguageState


class OutcomeRaceState(NativeLanguageState):
    def __init__(self):
        super().__init__();self.outcomes={};self.previous_token=None;self.outcome_writes=0
        self.terminal_key_scores=0;self.terminal_deliveries=0;self.terminal_pairs=0
        self.terminal_candidate_values=0;self.terminal_completion_delay=0.
        self.terminal_addresses=[];self.terminal_probabilities=[]

    def storage(self):
        result=super().storage();result.update(outcome_addresses=len(self.outcomes),outcome_capacity=27,
            raw_outcome_packed_integer_bytes=8*len(self.outcomes)+8)
        result['scope']+='; raw predecessor/successor integer state separate, not tensor bytes or Python heap'
        return result


class JointOutcomeRaceQuery(nn.Module):
    def __init__(self,payload=4,depth=2,pool=2,heads=2,key_width=4,value_width=4,credit='joint'):
        super().__init__()
        if credit not in ('joint','local'):raise ValueError('Declared terminal credit required')
        self.core=NativeStreamLanguageModel(payload,depth,pool,heads)
        self.total_payload=self.core.total_payload;self.credit=credit;self.key_width=key_width;self.value_width=value_width
        self.keys=nn.Embedding(27,key_width);nn.init.normal_(self.keys.weight,std=.1)
        self.values=nn.Embedding(27,value_width)
        self.query=nn.ModuleList([nn.Linear(self.total_payload+27,key_width,bias=False) for _ in range(2)])
        self.context_residual=nn.Linear(self.total_payload,1);self.linear_values=nn.Parameter(torch.zeros(2,value_width))
        self.interaction=nn.Parameter(torch.zeros(value_width,value_width))
        nn.init.zeros_(self.context_residual.weight);nn.init.zeros_(self.context_residual.bias)
        self._expected=None

    def new_state(self):return OutcomeRaceState()

    def pair_logits(self,feature,base,values):
        first=values@self.linear_values[0];second=values@self.linear_values[1]
        return base+self.context_residual(feature).squeeze()+first[:,None]+second[None,:]+values@self.interaction@values.T

    def query_loss(self,target,sampled_logits):
        if self.credit=='local':return F.cross_entropy(sampled_logits[None],torch.tensor([target]))
        pair,first,second=self._expected
        sign=float(2*target-1)
        return (first[:,None]*second[None,:]*F.softplus(-sign*pair.to(torch.float64))).sum()

    def forward_chunk(self,tokens,state=None):
        state=self.new_state() if state is None else state
        seq=torch.as_tensor(tokens,dtype=torch.long);box={}
        if not len(seq) or bool((seq[:-1]==26).any()):raise ValueError('A query block must end at its unique cue')
        def observe(module,args):box['feature']=args[0]
        hook=self.core.head.register_forward_pre_hook(observe)
        try:logits,state=self.core.forward_chunk(seq,state)
        finally:hook.remove()
        for token in seq:
            token=int(token)
            if state.previous_token is not None:
                state.outcomes[state.previous_token]=token;state.outcome_writes+=1
            state.previous_token=token
        self._expected=None
        if int(seq[-1])!=26:return logits,state
        addresses=sorted(state.outcomes);indices=torch.tensor(addresses,dtype=torch.long)
        raw=torch.tensor([state.outcomes[k] for k in addresses],dtype=torch.long)
        feature=box['feature'];base=logits[-1,1]-logits[-1,0]
        mark=F.one_hot(seq[-1],27).to(feature.dtype);query_input=torch.cat([feature,mark])
        keys=self.keys(indices);scores=[(keys@q(query_input)/math.sqrt(self.key_width)).clamp(-12,12) for q in self.query]
        state.terminal_key_scores+=2*len(addresses);state.candidate_scores+=2*len(addresses)
        candidate_values=self.values(raw) if self.training else None
        if self.training:
            state.terminal_candidate_values+=len(addresses);state.counterfactual_values+=len(addresses)
        delivered=[];delays=[]
        for score in scores:
            if self.training and self.credit=='local':
                value,delay,winner=self.core.race(score,candidate_values)
            else:
                _,delay,winner=self.core.race(score)
                value=(candidate_values[int(winner)] if candidate_values is not None else self.values(raw[int(winner)]))
            delivered.append(value);delays.append(delay)
            state.terminal_addresses.append(addresses[int(winner)])
            state.terminal_probabilities.append(score.detach().to(torch.float64).softmax(-1).tolist())
        state.terminal_deliveries+=2;state.terminal_completion_delay+=float(torch.stack(delays).max().detach())
        residual=self.context_residual(feature).squeeze()+sum(v@w for v,w in zip(delivered,self.linear_values))+delivered[0]@self.interaction@delivered[1]
        final=logits[-1].clone();final[1]=final[1]+residual
        logits=torch.cat([logits[:-1],final[None]])
        if self.training and self.credit=='joint':
            self._expected=(self.pair_logits(feature,base,candidate_values),*[score.to(torch.float64).softmax(-1) for score in scores])
            state.terminal_pairs+=len(addresses)**2
        return logits,state
