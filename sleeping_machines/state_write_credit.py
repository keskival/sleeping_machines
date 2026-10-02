"""Hard temporal races with addressed memory/arrival boundary credit.

Winner-only inference is inherited unchanged. Training keeps auxiliary state
views to receive losing-address adjoints. This is a local full-state surrogate,
not an unbiased arbitrary sequence-gradient estimator.
"""
from dataclasses import dataclass,field
import math

import torch
from torch.nn import functional as F

from .addressed_event_heads import AddressedEventState
from .split_event_heads import SplitEventHeads


class StateWriteRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx,scores,values,old_memory,new_memory,old_times,arrival,strength):
        rates=scores.to(torch.float64).exp()
        time,winner=(torch.empty_like(rates).exponential_()/rates).min(0)
        memories=old_memory.clone();memories[winner]=new_memory[winner]
        times=old_times.clone();times[winner]=arrival
        ctx.save_for_backward(rates,time,winner,values,old_memory,new_memory,old_times,arrival)
        ctx.strength=strength;ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(winner)
        return values[winner],.001+.010*time/(1+time),winner,memories,times

    @staticmethod
    def backward(ctx,error_value,error_delay,error_winner,error_memory,error_times):
        rates,time,winner,values,old,new,old_times,arrival=ctx.saved_tensors
        score=torch.zeros_like(rates);value_grad=torch.zeros_like(values)
        if error_value is not None:
            utility=(values-values.mean(0))@error_value
            score=rates*time*utility.to(rates.dtype)
            score=score.scatter_add(0,winner[None],-score.sum()[None])
            value_grad=value_grad.index_add(0,winner[None],error_value[None])
        if error_delay is not None:
            timing=-error_delay*.010*time/(1+time).square()
            score=score.scatter_add(0,winner[None],timing[None])
        utility=torch.zeros_like(rates)
        old_grad=new_grad=times_grad=arrival_grad=None
        if error_memory is not None:
            utility=utility+(error_memory*(new-old)).sum(-1).to(rates.dtype)
            old_grad=error_memory.clone();old_grad[winner]=0
            new_grad=torch.zeros_like(new);new_grad[winner]=error_memory[winner]
        if error_times is not None:
            utility=utility+error_times*(arrival-old_times)
            times_grad=error_times.clone();times_grad[winner]=0
            arrival_grad=error_times[winner]
        if ctx.strength:
            extra=ctx.strength*rates*time*(utility-utility.mean())
            extra=extra.scatter_add(0,winner[None],-extra.sum()[None])
            score=score+extra
        return score.to(values.dtype),value_grad,old_grad,new_grad,times_grad,arrival_grad,None


@dataclass
class StateCreditState(AddressedEventState):
    credit_memories:dict=field(default_factory=dict)
    credit_arrivals:dict=field(default_factory=dict)
    state_credit_races:int=0

    def detach(self):
        super().detach()
        self.credit_memories={k:v.detach() for k,v in self.credit_memories.items()}
        self.credit_arrivals={k:v.detach() for k,v in self.credit_arrivals.items()}
        return self

    def clear_source(self,source):
        self.contexts.pop(source,None)
        for name in ('memories','arrivals','credit_memories','credit_arrivals'):
            setattr(self,name,{k:v for k,v in getattr(self,name).items() if k[2]!=source})

    def storage(self):
        row=super().storage()
        row.update(auxiliary_credit_addresses=len(self.credit_memories),
            auxiliary_credit_logical_tensor_bytes=sum(v.numel()*v.element_size()
                for table in (self.credit_memories,self.credit_arrivals) for v in table.values()),
            auxiliary_scope='Training-only logical views, including overlap with physical states; '
                            'not unique allocated bytes or inference activity. RSS and graphs are separate.')
        return row


class StateCreditEventHeads(SplitEventHeads):
    def __init__(self,*args,state_credit=1.,**kwargs):
        super().__init__(*args,**kwargs)
        if not math.isfinite(state_credit) or not 0<=state_credit<=1:
            raise ValueError('State credit strength must be in[0,1]')
        if self.credit!='counterfactual':raise ValueError('Retain the counterfactual parent teacher')
        self.state_credit=state_credit

    def new_state(self):return StateCreditState()

    def consume_event(self,source,timestamp,content,state):
        # The zero-added-credit model and all inference use the frozen parent.
        if not self.training or not self.state_credit:
            return super().consume_event(source,timestamp,content,state)
        source,timestamp=int(source),float(timestamp)
        if not 0<=source<self.sources or not math.isfinite(timestamp) or timestamp<state.last_input_time:
            raise ValueError('Known observed address and finite nondecreasing timestamps required')
        mark=torch.as_tensor(content,dtype=self.embedding.weight.dtype,device=self.embedding.weight.device)
        if mark.shape!=(self.content_dim,) or not bool(torch.isfinite(mark).all()):
            raise ValueError('Finite fixed-width observed content required')
        x=self.embedding.weight[source]+self.content(mark)
        arrival=torch.tensor(timestamp,dtype=torch.float64,device=x.device)
        if source in state.contexts:
            previous,times=state.contexts[source];read_time=torch.maximum(arrival,times.max())
            state.queue_wait_sum+=float((read_time-arrival).detach());arrival=read_time
            context=self.align(list(previous.split(self.payload)),times,arrival,self.depth-1)
            x=F.layer_norm(x+torch.sigmoid(self.source_gate(x))*context,(self.total_payload,))
        for depth in range(self.depth):
            mixed=self.channel_mix[depth](x);features=F.layer_norm(mixed,(self.total_payload,))
            values,arrivals=[],[]
            for head in range(self.heads):
                incoming=mixed[head*self.payload:(head+1)*self.payload]
                units=self.units[depth][head][source];query=self.queries[depth][head](features)
                addresses=[(depth,head,source,i) for i in range(self.pool)]
                memories=[state.credit_memories.get(a,state.memories.get(a,incoming.new_zeros(self.payload)))
                          for a in addresses]
                times=[state.credit_arrivals.get(a,state.arrivals.get(a,arrival)) for a in addresses]
                scores=torch.stack([(query@(unit.key+unit.key_read(memories[i])))/math.sqrt(self.payload)
                                    +unit.clock_bias for i,unit in enumerate(units)]).clamp(-12,12)
                # Unwritten zero slots have virtual clocks solely for training
                # adjoints; physical occupancy and event commits stay unchanged.
                proposals=[unit.propose(incoming,memories[i],times[i],arrival) for i,unit in enumerate(units)]
                value,delay,winner,updated,updated_times=StateWriteRoute.apply(scores,
                    torch.stack([p[0] for p in proposals]),torch.stack(memories),
                    torch.stack([p[1] for p in proposals]),torch.stack(times),arrival,self.state_credit)
                index=int(winner)
                for i,address in enumerate(addresses):
                    state.credit_memories[address]=updated[i];state.credit_arrivals[address]=updated_times[i]
                address=addresses[index];state.memories[address]=updated[index];state.arrivals[address]=updated_times[index]
                state.visited_units.add(address);state.candidate_scores+=self.pool;state.selected_updates+=1
                state.counterfactual_values+=self.pool;state.state_credit_races+=1
                values.append(value);arrivals.append(arrival+delay)
            arrival=torch.stack(arrivals).max();x=self.align(values,arrivals,arrival,depth)
        state.contexts[source]=(torch.cat(values),torch.stack(arrivals))
        state.events+=1;state.last_input_time=timestamp
        return self.head(x),arrival
