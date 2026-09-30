"""Fitting-only cross-example teacher statistics after a live serial head step.

No new model tuning or held data. Raw paired CE teachers are measured at one
frozen checkpoint. Cross-inner-products remove the diagonal noise term under
explicit independence assumptions; real speaker/finite-set correlation limits
that statistical interpretation. They do not estimate actual Adam descent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e169_serial_step_replay import build,query
from e139_fine_packet_model import load_marked,augment_marked


def summarize(matrix):
    g=matrix.double();m=len(g);mean=g.mean(0)
    energy=g.square().sum(1);mean2=float(mean.square().sum())
    cross=(float(g.sum(0).square().sum())-float(energy.sum()))/(m*(m-1))
    variance=float((g-mean).square().sum())/(m-1)
    a,b=g[::2].mean(0),g[1::2].mean(0)
    cosine=float(a@b)/(float(a.norm()*b.norm())+1e-300)
    return dict(teachers=m,coordinates=g.shape[1],all_nonzero=bool((energy>0).all()),
        squared_fitted_mean=mean2,cross_sample_signal=cross,
        teacher_variance_trace=variance,variance_of_mean_under_independence=variance/m,
        signal_fraction_of_squared_mean=cross/max(mean2,1e-300),
        split_cross_inner_product=float(a@b),split_cosine=cosine,
        mean_teacher_norm=float(energy.sqrt().mean()),max_teacher_norm=float(energy.sqrt().max()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--limit',type=int,default=128);a=p.parse_args()
    if a.limit<16 or a.limit%8:raise ValueError('At least sixteen fitting utterances; multiple of eight')
    out=Path('experiments/results/e179')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists() or Path(a.tag).name!=a.tag:raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(6);started=time.perf_counter()
    checkpoint=Path('experiments/results/e165/selected_prefix_20260930.pt')
    model,saved,cfg=build(checkpoint);fit=load_marked(6144,'fit_spk',6)
    order_rng=np.random.default_rng(8);order_rng.bit_generator.state=saved['order_rng']
    aug_rng=np.random.default_rng(128);aug_rng.bit_generator.state=saved['augmentation_rng']
    order=order_rng.permutation(len(fit))
    clean=[fit[j] for j in order[:36+a.limit]]
    transformed=[augment_marked(row,aug_rng) for row in clean]
    calibration=json.loads(Path('experiments/results/e169/serial_step_replay_20260930.json').read_text())
    assert [row[8] for row in clean[:4]]==calibration['update_fitting_ids']
    assert [row[8] for row in clean[4:36]]==calibration['anchor_fitting_ids']
    params=list(model.suffix.parameters());optimizer=torch.optim.Adam(params,lr=calibration['selected_learning_rate'])
    model.train();optimizer.zero_grad(set_to_none=True)
    for batch in (clean[:4],transformed[:4]):
        z,y,_=query(model,batch,cfg['window']);(F.cross_entropy(z,y)/2).backward()
    warm_norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));optimizer.step()
    groups={'projection':list(model.suffix.project.parameters()),
            **{f'block_{i+1}':list(layer.parameters()) for i,layer in enumerate(model.suffix.layers)},
            'head':list(model.suffix.head.parameters())}
    before={n:q.detach().clone() for n,q in model.named_parameters()}
    rows={key:[] for key in groups};batch_records=[];model.eval()
    for begin in range(36,36+a.limit,4):
        model.zero_grad(set_to_none=True);paired_loss=0.
        for batch in (clean[begin:begin+4],transformed[begin:begin+4]):
            z,y,_=query(model,batch,cfg['window']);loss=F.cross_entropy(z,y)/2
            paired_loss+=float(loss.detach());loss.backward()
        for key,parameters in groups.items():
            rows[key].append(torch.cat([q.grad.detach().flatten().clone() if q.grad is not None
                                        else torch.zeros_like(q).flatten() for q in parameters]))
        raw_norm=sum(float(q.grad.square().sum()) for q in params if q.grad is not None)**.5
        batch_records.append(dict(ids=[row[8] for row in clean[begin:begin+4]],
            labels=[int(row[3]) for row in clean[begin:begin+4]],paired_ce=paired_loss,
            unclipped_teacher_norm=raw_norm,clipping_factor=min(1.,1/(raw_norm+1e-6))))
        if len(batch_records)%4==0:print(json.dumps({'measured_batches':len(batch_records)}),flush=True)
    assert all(torch.equal(before[n],q) for n,q in model.named_parameters())
    result=dict(status='completed',args=vars(a),groups={key:summarize(torch.stack(values)) for key,values in rows.items()},
        batches=batch_records,warm_step=dict(rate=calibration['selected_learning_rate'],teacher_norm=warm_norm,
            fitting_ids=calibration['update_fitting_ids']),
        protocol=dict(depth=12,frozen_prefix_depth=6,trainable_suffix_depth=6,
            probe_utterances=a.limit,paired_views_per_utterance=2,batch_utterances=4,
            calibration_update_and_anchor_ids_excluded=True,parameter_updates_during_probe=0,
            held_data_read=False,metric='raw CE parameter-gradient geometry; fixed identity preconditioner'),
        scope=__doc__.strip(),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (
            Path(__file__),Path('experiments/e169_serial_step_replay.py'),Path('sleeping_machines/serial_residual.py'))},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'groups':result['groups'],'wall_s':result['wall_s']}),flush=True)


if __name__=='__main__':main()
