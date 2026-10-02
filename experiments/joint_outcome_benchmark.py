"""Reuse the frozen paired-task driver with an explicit terminal-credit adapter.

Only model factory, query loss and observer counters differ. The original
driver's budget, checkpoint/recovery and sampling code is preserved.
"""
import hashlib
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import balanced_joint_benchmark as B
from sleeping_machines.joint_outcome_race_query import JointOutcomeRaceQuery
_parent_sources=B.sources


def sources():
    names=['experiments/joint_outcome_benchmark.py','sleeping_machines/joint_outcome_race_query.py',
        'experiments/theory/68_joint_addressed_outcome_races.md']
    return {**_parent_sources(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def make_model(a):
    torch.manual_seed(a.seed)
    return JointOutcomeRaceQuery(a.payload,a.depth,a.pool,a.heads,getattr(a,'key_width',4),
        getattr(a,'value_width',4),getattr(a,'read_credit','joint'))


def train_window(model,opt,rows,a,epoch):
    model.train();opt.zero_grad(set_to_none=True);loss_sum=0.;events=keys=updates=teachers=0;max_bytes=0
    for row in rows:
        logits,state=B.episode(model,row,100000+a.seed+10000*epoch)
        loss=model.query_loss(row['target'],logits)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite declared query objective')
        loss.backward();loss_sum+=float(loss.detach());model._expected=None
        events+=state.events;keys+=state.candidate_scores;updates+=state.selected_updates
        teachers+=state.counterfactual_values;max_bytes=max(max_bytes,state.storage()['persistent_tensor_bytes'])
    for p in model.parameters():
        if p.grad is not None:p.grad.div_(len(rows))
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
    return dict(targets=len(rows),loss_sum=loss_sum,input_events=events,key_scores=keys,selected_updates=updates,
        counterfactual_values=teachers,max_state_tensor_bytes=max_bytes,gradient_norm_before_clip=norm)


@torch.no_grad()
def evaluate(model,rows):
    model.eval();losses=[];probabilities=[];hits=[];events=keys=updates=raw_writes=terminal_keys=deliveries=0
    max_bytes=max_integer_bytes=0;chosen=[];distributions=[]
    for row in rows:
        z,state=B.episode(model,row,314159)
        losses.append(float(F.cross_entropy(z[None],torch.tensor([row['target']])))/math.log(2))
        probabilities.append(float(z.softmax(-1)[1]));hits.append(int(z.argmax())==row['target'])
        events+=state.events;keys+=state.candidate_scores;updates+=state.selected_updates;raw_writes+=state.outcome_writes
        terminal_keys+=state.terminal_key_scores;deliveries+=state.terminal_deliveries
        storage=state.storage();max_bytes=max(max_bytes,storage['persistent_tensor_bytes'])
        max_integer_bytes=max(max_integer_bytes,storage['raw_outcome_packed_integer_bytes'])
        chosen.append(state.terminal_addresses);distributions.append(state.terminal_probabilities)
    return dict(targets=len(rows),query_bits=float(np.mean(losses)),accuracy=float(np.mean(hits)),
        per_target_bits=losses,class1_probabilities=probabilities,input_events=events,key_scores=keys,
        selected_updates=updates,max_state_tensor_bytes=max_bytes,outcome_writes=raw_writes,
        terminal_key_scores=terminal_keys,terminal_value_deliveries=deliveries,
        raw_outcome_packed_integer_bytes_max=max_integer_bytes,terminal_chosen_addresses=chosen,
        terminal_probabilities=distributions,
        scope='Sampled hard terminal races. Key scores include native and terminal candidates; selected_updates are native commits only. Raw writes and delivered values separate; integer state excludes Python heap.')


def activate():
    B.make_model=make_model;B.train_window=train_window;B.evaluate=evaluate;B.sources=sources


def parser():
    p=B.parser();p.add_argument('--read-credit',choices=('joint','local'),default='joint')
    p.add_argument('--key-width',type=int,default=4);p.add_argument('--value-width',type=int,default=4)
    return p


if __name__=='__main__':
    activate();B.run(parser().parse_args())
