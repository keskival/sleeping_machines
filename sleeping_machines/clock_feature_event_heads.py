"""Native sparse events with fast delay-computed features per independent head.

Each existing query/key match drives several scalar clock policies. Local
rotating/decaying feature states evolve until these arrivals. All clock outputs
are awaited; the content message evolves while waiting for the last clock.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .addressed_event_heads import AddressedEventHeads, AddressedEventState
from .clock_feature_races import ContentAndClockRaces
from .native_stream_language import NativeStreamLanguageModel


@dataclass
class ClockFeatureState(AddressedEventState):
    extra_clock_races: int = 0
    extra_rate_settings: int = 0

    def packed_storage(self):
        stats = self.storage()
        stats['differentiable_entries'] = sum(int(t.requires_grad) for t in self.memories.values())
        return stats


class ClockFeatureEventHeads(AddressedEventHeads):
    def __init__(self,*args,clock_features=4,clock_readout=True,clock_allocation='uniform',**kwargs):
        if clock_features not in (0,2,4):
            raise ValueError('Use the bounded zero/two/four-clock comparison')
        if clock_allocation not in ('uniform','late'):
            raise ValueError('Uniform or deeper-half clock allocation required')
        super().__init__(*args,**kwargs)
        if clock_features and self.credit != 'counterfactual':
            raise ValueError('Extra-clock construction currently requires counterfactual content credit')
        self.clock_features, self.clock_readout = clock_features, bool(clock_readout)
        self.clock_allocation = clock_allocation
        self.clock_counts = [clock_features if clock_allocation=='uniform' or d>=self.depth//2 else 0 for d in range(self.depth)]
        if not clock_features:
            return  # Exact parent parameters, initialization and random stream.
        self.clock_temperature = nn.ParameterList()
        self.clock_bias = nn.ParameterList()
        self.clock_decay = nn.ParameterList()
        self.clock_frequency = nn.ParameterList()
        self.clock_projection = nn.ModuleList()
        self.clock_reception = nn.ModuleList()
        for count in self.clock_counts:
            shape = (self.heads,count)
            temperatures = torch.logspace(math.log10(.25),math.log10(2.),count)
            self.clock_temperature.append(nn.Parameter(torch.expm1(temperatures).log().expand(shape).clone()))
            self.clock_bias.append(nn.Parameter(torch.full(shape,-math.log(self.pool))))
            self.clock_decay.append(nn.Parameter(torch.expm1(torch.full(shape,.25)).log()))
            frequencies = torch.logspace(math.log10(math.pi/4),math.log10(math.pi),count)
            self.clock_frequency.append(nn.Parameter(frequencies.expand(shape).clone()))
            self.clock_projection.append(nn.ModuleList([nn.Linear(2*count,self.payload,bias=False)
                if count else nn.Identity() for _ in range(self.heads)]))
            self.clock_reception.append(nn.ModuleList([nn.Linear(self.payload,2*count,bias=False)
                if count else nn.Identity() for _ in range(self.heads)]))
        for modules in (self.clock_projection,self.clock_reception):
            for row in modules:
                for layer in row:
                    if hasattr(layer,'weight'):nn.init.zeros_(layer.weight)

    def new_state(self):
        return ClockFeatureState()

    def clock_features_at(self,delays,depth,head):
        # The fixed clock span calibrates the temporal basis. Numerically this
        # is the local time of a fast state, reset for this event and latched at
        # arrival; it is not a token embedding or a second key/query match.
        age = ((delays-.001)*100).to(self.embedding.weight.dtype)
        phase = age*self.clock_frequency[depth][head]
        amplitude = torch.exp(-age*F.softplus(self.clock_decay[depth][head]))
        return torch.stack((amplitude*phase.cos(),amplitude*phase.sin()),-1).flatten()

    def temporal_reception(self,value,features,depth,head):
        representation = self.clock_reception[depth][head](value).reshape(-1,2)
        phase = features.reshape(-1,2)
        gates = torch.sigmoid((representation*phase).sum(-1)/math.sqrt(2.))
        reception = (phase*gates[:,None]).flatten()
        return value+self.clock_projection[depth][head](reception)/math.sqrt(self.clock_counts[depth])

    def consume_event(self,source,timestamp,content,state):
        if not self.clock_features:
            return super().consume_event(source,timestamp,content,state)
        source,timestamp = int(source),float(timestamp)
        if not 0 <= source < self.sources or not math.isfinite(timestamp) or timestamp < state.last_input_time:
            raise ValueError('Known address and finite nondecreasing observed timestamps required')
        mark = torch.as_tensor(content,dtype=self.embedding.weight.dtype,device=self.embedding.weight.device)
        if mark.shape != (self.content_dim,) or not bool(torch.isfinite(mark).all()):
            raise ValueError('Finite fixed-width observed content required')
        x = self.embedding.weight[source]+self.content(mark)
        arrival = torch.tensor(timestamp,dtype=torch.float64,device=x.device)
        if source in state.contexts:
            previous,times = state.contexts[source]
            read_time = torch.maximum(arrival,times.max())
            state.queue_wait_sum += float((read_time-arrival).detach())
            arrival = read_time
            context = self.align(list(previous.split(self.payload)),times,arrival,self.depth-1)
            x = F.layer_norm(x+torch.sigmoid(self.source_gate(x))*context,(self.total_payload,))
        for depth in range(self.depth):
            count = self.clock_counts[depth]
            mixed = self.channel_mix[depth](x)
            all_features = F.layer_norm(mixed,(self.total_payload,))
            outputs,arrivals = [],[]
            for head in range(self.heads):
                incoming = mixed[head*self.payload:(head+1)*self.payload]
                units = self.units[depth][head][source]
                query = self.queries[depth][head](all_features)
                scores = torch.stack([(query @ (unit.key+unit.key_read(state.memories.get(
                    (depth,head,source,i),incoming.new_zeros(self.payload)))))/math.sqrt(self.payload)
                    +unit.clock_bias for i,unit in enumerate(units)]).clamp(-12,12)
                policy = scores[None,:]
                if count:
                    temperatures = F.softplus(self.clock_temperature[depth][head])
                    extra = (temperatures[:,None]*scores[None,:]+self.clock_bias[depth][head,:,None]).clamp(-12,12)
                    policy = torch.cat((policy,extra))
                if self.training:
                    proposals = [unit.propose(incoming,state.memories.get((depth,head,source,i),
                        incoming.new_zeros(self.payload)),state.arrivals.get((depth,head,source,i)),arrival)
                        for i,unit in enumerate(units)]
                    value,delays,winner = ContentAndClockRaces.apply(policy,torch.stack([p[0] for p in proposals]))
                    index = int(winner)
                    memory = proposals[index][1]
                    state.counterfactual_values += self.pool
                else:
                    _,delays,winner = ContentAndClockRaces.apply(policy,scores.new_zeros((self.pool,0)))
                    index = int(winner)
                    value,memory = units[index].propose(incoming,state.memories.get((depth,head,source,index),
                        incoming.new_zeros(self.payload)),state.arrivals.get((depth,head,source,index)),arrival)
                # Clock-only features cannot arrive before their clocks fire.
                joined = delays.max()
                if count:value = self.transport(value,joined-delays[0],depth,head)
                if count and self.clock_readout:
                    features = self.clock_features_at(delays[1:],depth,head)
                    # Content/state-dependent reception into each temporal pair.
                    # The learned representation sees the selected mixed value;
                    # clock vectors were latched at their respective arrivals.
                    value = self.temporal_reception(value,features,depth,head)
                address = (depth,head,source,index)
                state.memories[address],state.arrivals[address] = memory,arrival
                state.visited_units.add(address)
                state.candidate_scores += self.pool
                state.selected_updates += 1
                state.extra_clock_races += count
                state.extra_rate_settings += count*self.pool
                outputs.append(value)
                arrivals.append(arrival+joined)
            arrival = torch.stack(arrivals).max()
            x = self.align(outputs,arrivals,arrival,depth)
        state.contexts[source] = (torch.cat(outputs),torch.stack(arrivals))
        state.events += 1
        state.last_input_time = timestamp
        return self.head(x),arrival


class ClockFeatureLanguageModel(ClockFeatureEventHeads):
    forward_chunk = NativeStreamLanguageModel.forward_chunk

    def __init__(self,payload=16,depth=8,pool=2,heads=2,vocabulary=27,matching=0,recent=0,
                 clock_features=4,clock_readout=True,clock_allocation='uniform'):
        if matching or recent:
            raise ValueError('No per-position KV bank in the native clock feature model')
        super().__init__(sources=1,content_dim=vocabulary,classes=vocabulary,payload=payload,
                         depth=depth,pool=pool,heads=heads,clock_features=clock_features,
                         clock_readout=clock_readout,clock_allocation=clock_allocation)
        self.vocabulary = vocabulary
