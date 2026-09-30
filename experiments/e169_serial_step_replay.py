"""Fitting-only actual first Adam step for the serial correction suffix."""
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
from e143_event_state_shd import event_batch
from sleeping_machines.event_state import CoalescedEventStateEncoder
from sleeping_machines.serial_residual import SerialResidualEncoder


def build(checkpoint):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['encoder_config']
    prefix=CoalescedEventStateEncoder(sources=720,width=cfg['width'],modes=cfg['modes'],depth=saved['actual_depth'])
    prefix.load_state_dict(saved['encoder_state_dict'])
    return SerialResidualEncoder(prefix,suffix_depth=6),saved,cfg


def query(model,rows,window):
    score,_,_,stats=model(*event_batch(rows,window),len(rows))
    return score,torch.tensor([row[3] for row in rows]),stats


@torch.no_grad()
def logits(model,rows,window):
    model.eval();return torch.cat([query(model,rows[start:start+4],window)[0] for start in range(0,len(rows),4)])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    out=Path('experiments/results/e169')/(args.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(args.tag).name!=args.tag or out.exists():raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(6);started=time.perf_counter()
    checkpoint=Path('experiments/results/e165/selected_prefix_20260930.pt')
    model,saved,cfg=build(checkpoint)
    initial=copy.deepcopy(model.state_dict());fit=load_marked(6144,'fit_spk',6)
    order_rng=np.random.default_rng(8);order_rng.bit_generator.state=saved['order_rng']
    aug_rng=np.random.default_rng(128);aug_rng.bit_generator.state=saved['augmentation_rng']
    order=order_rng.permutation(len(fit));clean=[fit[j] for j in order[:36]]
    transformed=[augment_marked(row,aug_rng) for row in clean]
    update=clean[:4]+transformed[:4];anchors=clean[4:]+transformed[4:]
    model.train();loss_sum=0.
    for batch in (clean[:4],transformed[:4]):
        score,y,_=query(model,batch,cfg['window']);loss=F.cross_entropy(score,y)/2
        loss.backward();loss_sum+=float(loss.detach())
    parameters=list(model.suffix.parameters())
    unclipped=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
    gradients={name:None if q.grad is None else q.grad.detach().clone() for name,q in model.suffix.named_parameters()}
    before=logits(model,update+anchors,cfg['window']);labels=torch.tensor([row[3] for row in update+anchors])
    old_ce=float(F.cross_entropy(before[:8],labels[:8]));rows=[];chosen=None
    for rate in (.001,.00025,.0000625,.000015625,.00000390625,.0000009765625):
        model.load_state_dict(initial)
        for name,q in model.suffix.named_parameters():q.grad=gradients[name]
        optimizer=torch.optim.Adam(parameters,lr=rate);optimizer.step()
        after=logits(model,update+anchors,cfg['window'])
        p=before.softmax(-1);kl=(p*(before.log_softmax(-1)-after.log_softmax(-1))).sum(-1)
        new_ce=float(F.cross_entropy(after[:8],labels[:8]))
        row=dict(learning_rate=rate,batch_ce_before=old_ce,batch_ce_after=new_ce,
            anchor_mean_kl=float(kl[8:].mean()),accepted=new_ce<=old_ce and float(kl[8:].mean())<=.02)
        rows.append(row)
        if chosen is None and row['accepted']:chosen=row
    if chosen is None:raise ValueError('No fitting-admissible serial step')
    result=dict(status='completed',selected_learning_rate=chosen['learning_rate'],rows=rows,
        initial_paired_loss=loss_sum,unclipped_teacher_norm=unclipped,
        update_fitting_ids=[row[8] for row in clean[:4]],anchor_fitting_ids=[row[8] for row in clean[4:]],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        protocol='Largest declared first-step rate decreasing paired fitting CE with fitting-anchor KL <= 0.02; prefix immutable; no held labels or updates',
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__),Path('sleeping_machines/serial_residual.py'),Path('sleeping_machines/event_state.py'),Path('experiments/e139_fine_packet_model.py'))},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
