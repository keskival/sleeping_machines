"""Decompose full-depth correction risk on fitting speech and its transforms.

This uses a declared fitting-only subset, one fresh transformation per example,
and the unchanged completed E163 model. Exact linear-credit and KL-curvature
terms distinguish a wrong correction direction from an excessive correction.
No held labels select an update, scale or model.
"""
import argparse
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
from e150_single_state_shd import forward_query,metrics
from sleeping_machines.event_state import CoalescedEventStateEncoder
from sleeping_machines.observer_conditioned_depth import grow_observer_conditioned


def decompose(prefix,full,labels):
    correction=full-prefix;p=prefix.softmax(-1)
    residual=p-F.one_hot(labels,p.shape[-1])
    linear=(residual*correction).sum(-1)
    curvature=(p*(prefix.log_softmax(-1)-full.log_softmax(-1))).sum(-1)
    change=F.cross_entropy(full,labels,reduction='none')-F.cross_entropy(prefix,labels,reduction='none')
    return dict(prefix=metrics(prefix,labels),full=metrics(full,labels),
        average_ce_change=float(change.mean()),mean_linear_credit=float(linear.mean()),
        mean_kl_curvature=float(curvature.mean()),
        exact_decomposition_max_error=float((change-linear-curvature).abs().max()),
        harmful_examples=int((change>0).sum()),helpful_examples=int((change<0).sum()),
        mean_centered_correction_square=float((correction-correction.mean(-1,keepdim=True)).square().sum(-1).mean()),
        fixed_correction_scales={str(scale):metrics(prefix+scale*correction,labels) for scale in (0.,.25,.5,1.)})


@torch.no_grad()
def pair_scores(model,items,bs,window,prefix_depth):
    layers=model.layers;old=torch.nn.ModuleList(list(layers)[:prefix_depth]);full_scores=[];prefix_scores=[];labels=[]
    for start in range(0,len(items),bs):
        batch=items[start:start+bs]
        model.layers=layers;full,_,y,_=forward_query(model,batch,window)
        model.layers=old;prefix,_,_,_=forward_query(model,batch,window)
        full_scores.append(full);prefix_scores.append(prefix);labels.append(y)
    model.layers=layers
    return torch.cat(prefix_scores).double(),torch.cat(full_scores).double(),torch.cat(labels)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--limit',type=int,default=1024);args=parser.parse_args()
    out=Path('experiments/results/e167')/(args.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(args.tag).name!=args.tag or out.exists() or args.limit<1:raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(167);started=time.perf_counter()
    path=Path('experiments/results/e163/observer_depth_n6144_s6_e1_20260930.json')
    reference=json.loads(path.read_text());checkpoint=path.with_suffix('.pt')
    if reference['status']!='completed':raise ValueError('Completed model required')
    for source,digest in reference['source_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:raise ValueError('Changed source')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['encoder_config']
    model=CoalescedEventStateEncoder(sources=720,width=cfg['width'],modes=cfg['modes'],depth=cfg['depth'])
    model,_=grow_observer_conditioned(model,saved['actual_depth'])
    model.load_state_dict(saved['encoder_state_dict']);model.eval()
    items=load_marked(args.limit,'fit_spk',6)
    if [r[8] for r in items]!=reference['fit_absolute_ids'][:args.limit]:raise ValueError('Different fitting subset')
    clean_prefix,clean_full,labels=pair_scores(model,items,4,cfg['window'],cfg['depth'])
    rng=np.random.default_rng(167);aug_prefix=[];aug_full=[]
    for start in range(0,len(items),4):
        batch=[augment_marked(row,rng) for row in items[start:start+4]]
        prefix,full,_=pair_scores(model,batch,4,cfg['window'],cfg['depth'])
        aug_prefix.append(prefix);aug_full.append(full)
    aug_prefix=torch.cat(aug_prefix);aug_full=torch.cat(aug_full)
    result=dict(status='completed',fitting_absolute_ids=[r[8] for r in items],
        clean=decompose(clean_prefix,clean_full,labels),augmented=decompose(aug_prefix,aug_full,labels),
        paired=decompose(torch.cat((clean_prefix,aug_prefix)),torch.cat((clean_full,aug_full)),labels.repeat(2)),
        protocol=__doc__.strip(),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_sha256={**reference['source_sha256'],str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:{k:v for k,v in result[key].items() if k not in ('prefix','full','fixed_correction_scales')}
        for key in ('clean','augmented','paired')}),flush=True)


if __name__=='__main__':main()
