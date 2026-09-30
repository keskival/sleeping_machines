"""Generic observable-state event classifier; SHD driver declares its data."""
import copy
import math
import torch
from torch import nn
from sleeping_machines.shared_event import SharedEventModel
from e136_event_scattering import scatter
from e134_value_phase import SOURCE


class ScatteringClassifier(nn.Module):
    def __init__(self, depth=12, readout='state'):
        super().__init__()
        if depth<1 or readout not in ('state','packets'):raise ValueError('Invalid configuration')
        torch.manual_seed(6)
        self.key=SharedEventModel(depth=8,memory_backend='linear',cf_credit=False)
        saved=torch.load(SOURCE,weights_only=False,map_location='cpu')
        self.key.load_state_dict(saved['state_dict']);self.key.requires_grad_(False)
        self.embedding=copy.deepcopy(self.key.embedding).requires_grad_(True)
        self.depth,self.readout=depth,readout
        self.dim,self.receivers,self.options=self.key.dim,self.key.groups+1,3
        self.angle_weight=nn.Parameter(torch.randn(depth,3,self.dim)*.03)
        self.angle_bias=nn.Parameter(torch.tensor([.2,.5,.8]).repeat(depth,1))
        self.features=self.dim+1+depth*self.receivers*self.options*self.dim
        self.head=nn.Linear(self.features,self.key.classes)
        nn.init.normal_(self.head.weight,std=.001);nn.init.zeros_(self.head.bias)
        self.register_buffer('center',torch.zeros(self.features))
        self.register_buffer('scale',torch.ones(self.features))

    def forward(self,b,t,c,ids,size,trace=False):
        self.key.eval()
        phi=torch.cat((t.new_ones((len(t),1)),torch.exp(-t[:,None]/self.key.time_constants)),1)
        x=(self.embedding(b)[:,:,None]*phi[:,None,:]).flatten(1)*c.sqrt()[:,None]
        with torch.no_grad():kx=(self.key.embedding(b)[:,:,None]*phi[:,None,:]).flatten(1)
        kt=t;stats=[];traces=[];states=[];width=self.key.bands//self.key.groups
        for j in range(self.depth):
            group=(b+(width//2 if j%2 else 0))//width
            receiver=ids*self.receivers+group
            with torch.no_grad():
                before=kx
                kx,kt,st,schedule=self.key.layers[j%len(self.key.layers)](kx,kt,c,receiver,emit_schedule=True)
                winner=schedule['winner']
            row=self.angle_weight[j][winner];bias=self.angle_bias[j][winner]
            angle=(math.pi/2)*torch.tanh((before*row).sum(-1)+bias)
            address=receiver*self.options+winner
            # The key layer has completed its winning delay; memory consumes
            # arrivals in the INPUT order supplied by that same key program.
            order=schedule['time_order'];inverse=torch.argsort(order)
            outgoing,final,work=scatter(x[order],angle[order],address[order],
                    x.new_zeros((size*self.receivers*self.options,self.dim)))
            x=outgoing[inverse];states.append(final.reshape(size,-1))
            st=dict(st);st['key_scan_compositions']=st['scan_compositions']
            st['scan_compositions']+=work;st['scattering_compositions']=work
            st['winning_state_updates']=len(x);stats.append(st)
            if trace:traces.append({'winner':winner,'times':kt,'angle':angle,'out':x})
        mass=x.new_zeros(size).index_add(0,ids,c)
        pooled=x.new_zeros((size,self.dim)).index_add(0,ids,x)/mass.sqrt()[:,None]
        summary=torch.cat((pooled,mass.log1p()[:,None],*states),1)
        z=(summary-self.center)/self.scale
        if self.readout=='packets':z=torch.cat((z[:,:self.dim+1],torch.zeros_like(z[:,self.dim+1:])),1)
        logits=self.head(z)
        return logits,summary,{'layers':stats,'packets':len(t),'max_payload':float(x.detach().abs().max()),
          'mean_added_delay_ms':float((kt-t).mean()*1000),
          'retained_state_scalars':size*self.depth*self.receivers*self.options*self.dim},traces

    @torch.no_grad()
    def calibrate(self,items,batch_fn):
        sums=torch.zeros(self.features);squares=torch.zeros_like(sums);n=0
        for start in range(0,len(items),4):
            rows=items[start:start+4];data=batch_fn(rows)
            _,summary,_,_=self(*data[:4],len(rows))
            sums+=summary.sum(0);squares+=summary.square().sum(0);n+=len(rows)
        mean=sums/n;var=(squares/n-mean.square()).clamp_min(0)
        self.center.copy_(mean);self.scale.copy_(var.sqrt().clamp_min(.1))
