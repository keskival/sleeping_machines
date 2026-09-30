"""Audit trained directional depth and deletion of its six appended blocks.

The 657 same-speaker audit utterances were already used by E147/E154. They
are disjoint from current fitting/development, not an untouched official test.
Selection uses the declared saved continuation only. CPU timings are single
warmed forward observations including packing/query, excluding data loading.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from e139_fine_packet_model import load_marked
from e150_single_state_shd import evaluate,forward_query
from sleeping_machines.event_state import CoalescedEventStateEncoder
from sleeping_machines.observer_conditioned_depth import grow_observer_conditioned,materialize_observer_maps


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    out=Path('experiments/results/e164')/(args.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(args.tag).name!=args.tag or out.exists():raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(164);started=time.perf_counter()
    path=Path('experiments/results/e163/observer_depth_n6144_s6_e1_20260930.json')
    run=json.loads(path.read_text())
    reference_path=Path('experiments/results/e154/single_encoder_audit_20260930.json')
    reference=json.loads(reference_path.read_text())
    if run['status']!='completed' or reference['status']!='completed':raise ValueError('Completed runs required')
    for source,digest in run['source_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:raise ValueError('Changed executed source')
    selected=min(run['curve'],key=lambda r:(-r['dev']['correct'],r['dev']['nll']))
    checkpoint=path.with_name(path.stem+f".epoch{selected['epoch']}.pt")
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['encoder_config']
    encoder=CoalescedEventStateEncoder(sources=720,width=cfg['width'],modes=cfg['modes'],depth=cfg['depth'])
    encoder,_=grow_observer_conditioned(encoder,saved['actual_depth'])
    encoder.load_state_dict(saved['encoder_state_dict']);encoder.eval()
    desired=set(reference['absolute_ids'])
    items=[row for row in load_marked(100000,'val_spk',7) if row[8] in desired]
    if [r[8] for r in items]!=reference['absolute_ids']:raise ValueError('Changed audit membership/order')
    if desired&set(run['fit_absolute_ids']+run['dev_absolute_ids']):raise ValueError('Overlap')
    deployment=materialize_observer_maps(encoder).eval()
    with torch.no_grad():
        fold_error=float((forward_query(encoder,items[:4],cfg['window'])[0]-
            forward_query(deployment,items[:4],cfg['window'])[0]).abs().max())
    if fold_error>1e-6:raise ValueError('Deployment fold changed logits')
    rows={}
    for name in ('directional_d12','directional_prefix'):
        if name=='directional_prefix':deployment.layers=torch.nn.ModuleList(list(deployment.layers)[:cfg['depth']])
        evaluate(deployment,items[:16],4,cfg['window']);before=time.perf_counter()
        score=evaluate(deployment,items,4,cfg['window']);elapsed=time.perf_counter()-before
        rows[name]=dict(audit=score,forward_wall_s=elapsed,utterances_per_second=len(items)/elapsed,
            deployed_parameters=sum(p.numel() for p in deployment.parameters()))
    full=rows['directional_d12']['audit'];prefix=rows['directional_prefix']['audit']
    paired={}
    for name,score in (('its_trained_prefix',prefix),
        ('calibrated_d6',reference['rows']['matched_d6']['audit']),
        ('combined',reference['rows']['combined']['audit'])):
        if full['labels']!=score['labels']:raise ValueError('Different labels/order')
        paired[name]=dict(
            full_only_correct=sum(p==y and q!=y for p,q,y in zip(full['predictions'],score['predictions'],full['labels'])),
            reference_only_correct=sum(p!=y and q==y for p,q,y in zip(full['predictions'],score['predictions'],full['labels'])),
            changed_predictions=sum(p!=q for p,q in zip(full['predictions'],score['predictions'])))
    result=dict(status='completed',n=len(items),absolute_ids=[r[8] for r in items],rows=rows,paired=paired,
        selected_epoch=selected['epoch'],development_correct=selected['dev']['correct'],
        deployment_fold_max_error=fold_error,protocol=__doc__.strip(),
        deletion_scope='Remove the appended trained blocks only; retain the trained prefix/head. Fitted contribution probe, not a matched retrained architecture control',
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        result_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,reference_path)},
        source_sha256={**run['source_sha256'],str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        hardware=dict(platform=platform.platform(),torch=torch.__version__,threads=1),energy_joules=None,
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rows={key:{k:v for k,v in value.items() if k!='audit'}|
        {k:value['audit'][k] for k in ('correct','nll')} for key,value in rows.items()},paired=paired)),flush=True)


if __name__=='__main__':main()
