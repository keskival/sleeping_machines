"""Replay the first actual Adam step using only fitting speech and a KL budget.

All proposals share initial parameters, optimizer state, examples and gradient.
This is learning-rate calibration by finite replay, not a seed/development sweep.
The chosen factor is the largest declared factor satisfying fitting-batch
descent and the declared anchor probability-distortion budget.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from e139_fine_packet_model import load_marked,augment_marked
from e150_single_state_shd import forward_query
from sleeping_machines.event_state import CoalescedEventStateEncoder
from sleeping_machines.depth_growth import grow_event_encoder


@torch.no_grad()
def logits(encoder,items,window):
    encoder.eval();scores=[]
    for start in range(0,len(items),4):scores.append(forward_query(encoder,items[start:start+4],window)[0])
    return torch.cat(scores)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--run',default='experiments/results/e152/nuisance_state_n6144_s6_e2_20260930.json')
    p.add_argument('--kl-budget',type=float,default=.02)
    p.add_argument('--depth',type=int,default=6)
    a=p.parse_args();out=Path('experiments/results/e156')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists() or a.kl_budget<=0:raise ValueError('Unique tag and positive KL budget')
    run=json.loads(Path(a.run).read_text())
    if run['status']!='completed':raise ValueError('Completed run required')
    selected=min(run['curve'],key=lambda r:(-r['dev']['correct'],r['dev']['nll']))
    checkpoint=Path(a.run).with_name(Path(a.run).stem+f".epoch{selected['epoch']}.pt")
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['encoder_config']
    torch.set_num_threads(1);torch.manual_seed(6);started=time.perf_counter()
    encoder=CoalescedEventStateEncoder(sources=720,width=cfg['width'],modes=cfg['modes'],depth=cfg['depth'])
    encoder.load_state_dict(saved['encoder_state_dict'])
    old_names=[name for name,_ in encoder.named_parameters()]
    if a.depth>cfg['depth']:encoder=grow_event_encoder(encoder,a.depth)
    elif a.depth!=cfg['depth']:raise ValueError('No layer removal in this protocol')
    named=dict(encoder.named_parameters());old_params=[named[name] for name in old_names]
    new_params=[q for name,q in named.items() if name not in set(old_names)]
    initial=copy.deepcopy(encoder.state_dict())
    fit=load_marked(6144,'fit_spk',6)
    order_rng=np.random.default_rng(8);order_rng.bit_generator.state=saved['order_rng']
    aug_rng=np.random.default_rng(128);aug_rng.bit_generator.state=saved['augmentation_rng']
    order=order_rng.permutation(len(fit))
    views=[augment_marked(fit[j],aug_rng) for j in order[:36]]
    batch,anchors=views[:4],views[4:]
    encoder.train();score,_,labels,_=forward_query(encoder,batch,cfg['window'])
    loss=F.cross_entropy(score,labels);loss.backward()
    unclipped=float(torch.sqrt(sum(q.grad.square().sum() for q in encoder.parameters() if q.grad is not None)))
    torch.nn.utils.clip_grad_norm_(old_params,1.,error_if_nonfinite=True)
    if new_params:torch.nn.utils.clip_grad_norm_(new_params,1.,error_if_nonfinite=True)
    gradients={name:None if q.grad is None else q.grad.detach().clone() for name,q in encoder.named_parameters()}
    encoder.zero_grad(set_to_none=True);encoder.eval()
    physical_score,_,_,_=forward_query(encoder,batch,cfg['window'])
    F.cross_entropy(physical_score,labels).backward()
    physical_gradients={name:None if q.grad is None else q.grad.detach().clone() for name,q in encoder.named_parameters()}
    before=logits(encoder,batch+anchors,cfg['window'])
    targets=torch.tensor([r[3] for r in batch+anchors])
    original_batch_nll=float(F.cross_entropy(before[:4],targets[:4]))
    original_anchor_nll=float(F.cross_entropy(before[4:],targets[4:]))
    rows=[];chosen=None
    for factor in (1.,.25,.0625,.015625,.00390625,.0009765625):
        encoder.load_state_dict(initial)
        for name,q in encoder.named_parameters():q.grad=gradients[name]
        opt=torch.optim.Adam(old_params,lr=.000325);opt.load_state_dict(copy.deepcopy(saved['optimizer']))
        if new_params:opt.add_param_group(dict(params=new_params,lr=.000325))
        for group in opt.param_groups:group['lr']*=factor
        rate=opt.param_groups[0]['lr'];opt.step()
        after=logits(encoder,batch+anchors,cfg['window'])
        p=before.softmax(-1)
        kl=(p*(before.log_softmax(-1)-after.log_softmax(-1))).sum(-1)
        step_norm=float(torch.sqrt(sum((q.detach()-initial[name]).square().sum() for name,q in encoder.named_parameters())))
        prediction=float(sum((gradients[name]*(q.detach()-initial[name])).sum() for name,q in encoder.named_parameters() if gradients[name] is not None))
        physical_prediction=float(sum((physical_gradients[name]*(q.detach()-initial[name])).sum() for name,q in encoder.named_parameters() if physical_gradients[name] is not None))
        batch_nll=float(F.cross_entropy(after[:4],targets[:4]))
        anchor_nll=float(F.cross_entropy(after[4:],targets[4:]))
        accepted=batch_nll<=original_batch_nll and float(kl[4:].mean())<=a.kl_budget
        row=dict(factor=factor,learning_rate=rate,batch_nll=batch_nll,anchor_nll=anchor_nll,
            anchor_mean_kl=float(kl[4:].mean()),anchor_max_kl=float(kl[4:].max()),
            parameter_step_norm=step_norm,clipped_gradient_linear_loss_prediction=prediction,
            physical_interior_linear_loss_prediction=physical_prediction,
            actual_batch_loss_change=batch_nll-original_batch_nll,accepted=accepted)
        rows.append(row)
        if chosen is None and accepted:chosen=row
    if chosen is None:raise ValueError('No declared step meets fitting descent and KL budget')
    result=dict(status='completed',depth=a.depth,selected_starting_epoch=selected['epoch'],kl_budget=a.kl_budget,
        selected_factor=chosen['factor'],selected_learning_rate=chosen['learning_rate'],rows=rows,
        initial_batch_nll=original_batch_nll,initial_anchor_nll=original_anchor_nll,
        initial_unclipped_gradient_norm=unclipped,
        fitting_batch_absolute_ids=[fit[j][8] for j in order[:4]],anchor_absolute_ids=[fit[j][8] for j in order[4:36]],
        checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        optimizer_has_moments=bool(saved['optimizer']['state']),
        protocol=__doc__.strip(),scope='Finite first-update fitting replay; anchors are fitting-only and not new-speaker evidence. One fixed calibration factor is not a whole-training trust-region guarantee',
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            [Path(__file__),Path('experiments/e150_single_state_shd.py'),Path('sleeping_machines/event_state.py'),Path('sleeping_machines/depth_growth.py')]},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
