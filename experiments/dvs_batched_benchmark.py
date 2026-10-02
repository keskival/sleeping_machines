"""Native fit with independent-clip vectorization and optional terminal pair risk."""
from contextlib import contextmanager
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
from sleeping_machines.batched_addressed_fit import forward

BASE_PARSER=N.parser;BASE_SOURCES=N.sources


def parser():
    p=BASE_PARSER();p.add_argument('--terminal-risk',choices=['local','pairs'],default='local');return p


def sources():
    names=['sleeping_machines/batched_addressed_fit.py','experiments/dvs_batched_benchmark.py',
        'experiments/dvs_batched_contracts.py','experiments/theory/77_batched_fit_and_terminal_risk.md']
    return {**BASE_SOURCES(),**{n:N.sha(ROOT/n) for n in names}}


def loss(logits,pairs,targets):
    if pairs is None:return F.cross_entropy(logits,targets,reduction='sum')
    values,weights=pairs;B,C,K=values.shape
    losses=F.cross_entropy(values.reshape(-1,K),targets[:,None].expand(B,C).reshape(-1),reduction='none').reshape(B,C)
    return (weights*losses).sum()


def train_window(model,optimizer,rows,a,epoch,trace=False):
    optimizer.zero_grad(set_to_none=True);stages={};box={}
    def traced(name,fn):
        if trace:stages.setdefault(name,[]).append(N.capture(fn))
        else:fn()
    def compute():
        logits,state,pairs=forward(model,rows,100000+a.seed+10000*epoch,a.terminal_risk=='pairs')
        targets=torch.tensor([row['target'] for row in rows])
        box.update(loss=loss(logits,pairs,targets),state=state)
    traced('forward_and_loss',compute)
    if not torch.isfinite(box['loss']):raise FloatingPointError('Nonfinite batched loss')
    traced('backward',lambda:box['loss'].backward())
    def normalize():
        for parameter in model.parameters():
            if parameter.grad is not None:parameter.grad.div_(len(rows))
    traced('gradient_normalization',normalize)
    traced('gradient_clipping',lambda:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    traced('optimizer',optimizer.step)
    events=box['state']['events'];selected=events*model.depth*model.heads
    return dict(targets=len(rows),loss_sum=float(box['loss'].detach()),events=events,
        key_scores=selected*model.pool,selected_updates=selected,counterfactual_values=selected*model.pool,
        stages={name:N.merge(values) for name,values in stages.items()})


@contextmanager
def activate():
    previous=N.parser,N.sources,N.train_window
    N.parser,N.sources,N.train_window=parser,sources,train_window
    try:yield
    finally:N.parser,N.sources,N.train_window=previous


if __name__=='__main__':
    with activate():N.run(parser().parse_args())
