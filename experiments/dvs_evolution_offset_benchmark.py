"""Native physical-time phase offsets, with optional alternating parameter blocks."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
from sleeping_machines.evolution_offset_heads import EvolutionOffsetHeads,parameter_block
from sleeping_machines.fast_native_core import fast_class
from sleeping_machines.batched_evolution_offset_fit import forward

BASE_PARSER=N.parser;BASE_SOURCES=N.sources


def parser():
    p=BASE_PARSER();p.add_argument('--update-schedule',choices=['joint','alternating'],default='joint');return p


def make_model(a,fast=True):
    torch.manual_seed(a.seed);cls=fast_class(EvolutionOffsetHeads) if fast else EvolutionOffsetHeads
    return cls(sources=1,content_dim=33,classes=11,payload=a.payload,depth=a.depth,heads=a.heads,pool=a.pool)


def sources():
    names=['sleeping_machines/evolution_offset_heads.py','sleeping_machines/batched_evolution_offset_fit.py',
        'sleeping_machines/batched_addressed_fit.py','experiments/dvs_evolution_offset_benchmark.py',
        'experiments/dvs_evolution_offset_contracts.py','experiments/theory/90_coupled_time_evolution_offsets.md']
    return {**BASE_SOURCES(),**{n:N.sha(ROOT/n) for n in names}}


def phase(model,optimizer,schedule):
    if schedule=='joint':return 'joint'
    shared_step=optimizer.state.get(model.content.weight,{}).get('step',0)
    return 'message' if int(shared_step)%2==0 else 'route'


def train_window(model,optimizer,rows,a,epoch,trace=False):
    active=phase(model,optimizer,a.update_schedule);optimizer.zero_grad(set_to_none=True);stages={};box={}
    def traced(name,fn):
        if trace:stages.setdefault(name,[]).append(N.capture(fn))
        else:fn()
    def compute():
        logits,state,_=forward(model,rows,100000+a.seed+10000*epoch)
        box.update(loss=F.cross_entropy(logits,torch.tensor([r['target'] for r in rows]),reduction='sum'),state=state)
    traced('forward_and_loss',compute)
    if not torch.isfinite(box['loss']):raise FloatingPointError('Nonfinite offset loss')
    traced('backward',lambda:box['loss'].backward())
    def normalize():
        for name,p in model.named_parameters():
            if active!='joint' and parameter_block(name) not in (active,'shared'):p.grad=None
            if p.grad is not None:p.grad.div_(len(rows))
    traced('gradient_normalization',normalize)
    traced('gradient_clipping',lambda:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    traced('optimizer',optimizer.step)
    if optimizer.state[model.content.weight]['step']<1:raise RuntimeError('Shared phase cursor required')
    events=box['state']['events'];selected=events*model.depth*model.heads
    return dict(targets=len(rows),loss_sum=float(box['loss'].detach()),events=events,key_scores=selected*model.pool,
        selected_updates=selected,counterfactual_values=selected*model.pool,
        stages={n:N.merge(v) for n,v in stages.items()},phase=active)


@contextmanager
def activate():
    old=N.parser,N.sources,N.make_model,N.train_window
    N.parser,N.sources,N.make_model,N.train_window=parser,sources,make_model,train_window
    try:yield
    finally:N.parser,N.sources,N.make_model,N.train_window=old


def run(a,directory=None):
    with activate():result=N.run(a,directory)
    if result['status']=='completed':
        result['evolution_offset_protocol']=dict(kind='bounded_reception_phase',extra_parameters=a.depth*a.heads*(a.payload//2),
            equation='phase=frequency*physical_age+pi*tanh(raw_offset); damping uses physical_age',
            stored_timestamps_unchanged=True,independent_signal_clock=False,offset_active_during_message_phase_only=a.update_schedule=='alternating',
            schedule=a.update_schedule,shared_maps_active_each_update=True,native_route_and_timing_credit_retained=True,
            inactive_adam_state_frozen=True,full_model_gradient_exact=False)
        folder=Path(directory) if directory else ROOT/'experiments/results/dvs_native'
        (folder/(a.tag+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':run(parser().parse_args())
