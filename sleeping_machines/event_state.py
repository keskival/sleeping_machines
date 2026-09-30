"""Learned signed event-state blocks with nonlinear payloads and winning clocks.

This is a candidate encoder, not the trained common SHD model. It composes the
affine temporal operator of THEORY §219 with pointwise nonlinear representation
learning. State is segmented by receiver; only supplied events are evaluated.
Clock credit is a declared local surrogate, not an exact hard-order estimator.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .event_memory import affine_prefix
from .rotating_memory import rotate_pairs


def signed_state_scan(drive, times, receivers, rates, frequencies, sequential=False):
    """Unnormalized damped modal state, with actual per-receiver event order."""
    if not len(drive) or times.shape != receivers.shape or len(times) != len(drive):
        raise ValueError("Nonempty, consistently addressed events required")
    if drive.shape[-1] != 2*len(rates) or frequencies.shape != rates.shape:
        raise ValueError("Each temporal mode requires two drive coordinates")
    chronological = torch.argsort(times, stable=True)
    grouped = torch.argsort(receivers[chronological], stable=True)
    order = chronological[grouped]
    t, k, u = times[order], receivers[order], drive[order]
    first = torch.cat((torch.ones(1,dtype=torch.bool,device=k.device),k[1:] != k[:-1]))
    dt = torch.diff(t,prepend=t[:1]).clamp_min(0)
    decay = torch.exp(-dt[:,None]*rates[None])*(~first[:,None])
    angle = t[:,None]*frequencies[None]
    z = rotate_pairs(u,-angle).reshape(len(u),len(rates),2)
    if sequential:
        state, rows = torch.zeros_like(z[0]), []
        for i in range(len(z)):
            state = decay[i,:,None]*state+z[i]
            rows.append(state)
        z,work = torch.stack(rows),len(rows)
    else:
        z,work = affine_prefix(decay,z)
    states = rotate_pairs(z.flatten(1),angle)
    inverse = torch.argsort(order)
    return states[inverse],work


class EventStateBlock(nn.Module):
    """A global/local addressed SSM operator, gated output and a message race.

    One receiver may be an utterance, node or explicitly selected state address.
    With one constant-delay route, no nonlinear output and residual=False,
    this contains the stable diagonal complex SSM operator. Gating and competing
    clocks extend it. They do not supply a universal deep optimization guarantee.
    """
    def __init__(self,width=128,modes=64,options=3,gain=.25,nonlinear=True,residual=True):
        super().__init__()
        if min(width,modes,options)<1 or gain <= 0:
            raise ValueError("Positive dimensions and residual gain required")
        self.width,self.modes,self.options=width,modes,options
        self.gain,self.nonlinear,self.residual=gain,nonlinear,residual
        initial_rate=torch.logspace(math.log10(.1),math.log10(50.),modes)
        self.raw_rate=nn.Parameter(torch.expm1(initial_rate).log())
        self.frequency=nn.Parameter(torch.linspace(-30.,30.,modes))
        self.input=nn.Linear(width,2*modes,bias=False)
        self.output=nn.Linear(2*modes,width,bias=False)
        self.direct=nn.Parameter(torch.ones(width))
        self.norm=nn.LayerNorm(width)
        self.gate=nn.Linear(width,width,bias=True)
        self.clock=nn.Linear(width,options,bias=True)
        nn.init.zeros_(self.clock.weight)
        nn.init.zeros_(self.clock.bias)

    def emit(self,x,states,times,credit=True):
        y=self.output(states)+self.direct*x
        if self.nonlinear:
            y=self.norm(y)
            y=y*torch.sigmoid(self.gate(F.gelu(y)))
        value=x+self.gain*y if self.residual else y
        delays=.001+.010*torch.sigmoid(-self.clock(value))
        winner=delays.argmin(-1)
        chosen=delays.gather(1,winner[:,None]).squeeze(-1)
        arrival=times+chosen
        if self.training and credit and self.options>1:
            p=torch.softmax(-delays/.002,-1)
            arrival=arrival+((p-p.detach())*(delays-chosen[:,None]).detach()).sum(-1)
        return value,arrival,{"winner":winner,"delays":delays,
            "emitted_vectors":len(x),"clock_candidates":len(x)*self.options}

    def forward(self,x,times,receivers,counts=None,sequential=False):
        drive=self.input(x)
        if counts is not None:drive=drive*counts[:,None]
        states,work=signed_state_scan(drive,times,receivers,
            F.softplus(self.raw_rate)+1e-6,self.frequency,sequential)
        value,arrival,stats=self.emit(x,states,times)
        stats.update(state_scan_compositions=work,temporal_modes=self.modes,
            input_projection_macs=len(x)*self.width*2*self.modes,
            output_projection_macs=len(x)*self.width*2*self.modes,
            nonlinear_gate_macs=len(x)*self.width*self.width if self.nonlinear else 0)
        return value,arrival,stats


class EventStateEncoder(nn.Module):
    """Generic continuous-message encoder; each block emits one vector and time.

    Six width-128 blocks are a candidate reference-sized structure. Source
    projection/coalescing and task supervision belong to the caller. The final
    readout here is a completed-query count mean, not an early-confidence policy.
    No count/pointer expert or task-specific predictor is built into the encoder.
    """
    def __init__(self,input_dim,width=128,modes=64,depth=6,classes=20,options=3):
        super().__init__()
        if min(input_dim,depth,classes)<1:
            raise ValueError("Positive encoder dimensions required")
        self.project=nn.Linear(input_dim,width)
        self.layers=nn.ModuleList([EventStateBlock(width,modes,options,gain=1/depth)
            for _ in range(depth)])
        self.head=nn.Linear(width,classes)

    def forward(self,messages,times,receivers,counts,size,sequential=False):
        x=self.project(messages)
        stats=[]
        for layer in self.layers:
            x,times,row=layer(x,times,receivers,counts,sequential)
            stats.append(row)
        mass=x.new_zeros(size).index_add(0,receivers,counts)
        if torch.any(mass<=0):raise ValueError("Empty queries need an explicit silence policy")
        query=x.new_zeros((size,x.shape[-1])).index_add(0,receivers,x*counts[:,None])/mass[:,None]
        return self.head(query),x,times,stats


class CoalescedEventStateEncoder(nn.Module):
    """Source learning precedes causal pooling; first modal states are exact.

    Each packet has a closure time and raw-event assignments. Raw marks are
    transported to that closure before summation. The first affine state's
    packet endpoints agree with processing every raw impulse, including its
    decay/frequency/source teachers. Nonlinear output is evaluated only at
    packet closures: this does not equal a raw-event nonlinear deep network.
    No uniform empty frames or event-pair attention are evaluated.
    """
    def __init__(self,sources=700,width=128,modes=64,depth=6,classes=20,options=3):
        super().__init__()
        if min(sources,depth,classes)<1:
            raise ValueError("Positive encoder dimensions required")
        self.embedding=nn.Embedding(sources,width)
        nn.init.normal_(self.embedding.weight,std=.1)
        self.layers=nn.ModuleList([EventStateBlock(width,modes,options,gain=1/depth)
            for _ in range(depth)])
        self.head=nn.Linear(width,classes)

    def forward(self,source,raw_times,assignment,times,receivers,counts,size,sequential=False):
        first=self.layers[0]
        lag=times[assignment]-raw_times
        if torch.any(lag < -1e-6) or torch.any(counts<=0):
            raise ValueError("Causal, nonempty packet assignments required")
        # Source-address lookup avoids a dense projection at every raw spike.
        table=first.input(self.embedding.weight)
        raw_drive=F.embedding(source,table)
        angle=lag[:,None]*first.frequency[None]
        decay=torch.exp(-lag[:,None]*(F.softplus(first.raw_rate)+1e-6)[None])
        transported=rotate_pairs(raw_drive,angle).reshape(len(source),first.modes,2)
        transported=(transported*decay[:,:,None]).flatten(1)
        drive=transported.new_zeros((len(times),2*first.modes)).index_add(0,assignment,transported)
        states,work=signed_state_scan(drive,times,receivers,
            F.softplus(first.raw_rate)+1e-6,first.frequency,sequential)
        raw=F.embedding(source,self.embedding.weight)
        x=raw.new_zeros((len(times),self.embedding.embedding_dim)).index_add(0,assignment,raw)/counts[:,None]
        x,times,row=first.emit(x,states,times)
        row.update(state_scan_compositions=work,source_events=len(source),
            source_table_projection_macs=table.numel()*self.embedding.embedding_dim,
            source_transported_scalars=transported.numel(),
            source_payload_sum_scalars=raw.numel(),
            output_projection_macs=len(x)*first.width*2*first.modes,
            nonlinear_gate_macs=len(x)*first.width*first.width)
        stats=[row]
        for layer in self.layers[1:]:
            x,times,row=layer(x,times,receivers,counts,sequential)
            stats.append(row)
        mass=x.new_zeros(size).index_add(0,receivers,counts)
        if torch.any(mass<=0):raise ValueError("Empty queries need an explicit silence policy")
        query=x.new_zeros((size,x.shape[-1])).index_add(0,receivers,x*counts[:,None])/mass[:,None]
        return self.head(query),x,times,stats
