"""Native bounded emitter prototype with exact zero bridge and paid raw taps."""
import math
import torch
from .addressed_event_heads import AddressedEventHeads
from .bounded_score_sensitivity import bounded_score_bridge


class NativeBoundedScoreDiagnostic(AddressedEventHeads):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.bridge=0.;self.records=[];self.observed_trace=None
        self.condition_previous_clocks=False;self._query={};self._read={};self._hooks=[]
        for depth in range(self.depth):
            for head in range(self.heads):
                def qhook(module,args,out,d=depth,h=head):self._query[d,h]=out
                self._hooks.append(self.queries[depth][head].register_forward_hook(qhook))
                for source in range(self.sources):
                    for index,unit in enumerate(self.units[depth][head][source]):
                        def khook(module,args,out,d=depth,h=head,s=source,i=index):self._read[d,h,s,i]=out
                        self._hooks.append(unit.key_read.register_forward_hook(khook))

    def consume_event(self,source,timestamp,content,state):
        self._source=int(source)
        return super().consume_event(source,timestamp,content,state)

    def race(self,scores,values=None):
        if self.training and (self.bridge!=0. or self.condition_previous_clocks):
            raise ValueError('Positive bridge/conditioned training not installed')
        index=len(self.records);depth=(index%(self.depth*self.heads))//self.heads;head=index%self.heads
        query=self._query[depth,head];source=self._source
        raw=torch.stack([(query@(unit.key+self._read[depth,head,source,i]))/math.sqrt(self.payload)+unit.clock_bias
            for i,unit in enumerate(self.units[depth][head][source])])
        if not torch.equal(raw.clamp(-12,12),scores):raise AssertionError('Exact preclamp reconstruction required')
        effective=scores if self.bridge==0. else bounded_score_bridge(raw,self.bridge)
        value,delay,winner=super().race(effective,values)
        if self.observed_trace is not None:
            observed=self.observed_trace[index];delay=observed['delay'];winner=observed['winner']
            if value is not None:raise ValueError('Fixed histories require inference value selection')
        self.records.append(dict(raw=raw,scores=effective,delay=delay.detach().clone(),winner=winner.detach().clone()))
        return value,delay.detach() if self.condition_previous_clocks else delay,winner

    def clear_trace(self):
        self.records=[];self.observed_trace=None;self.condition_previous_clocks=False
        self._query.clear();self._read.clear()
