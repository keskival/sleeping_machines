"""Compare completed single encoders with E143 on the existing disjoint audit.

Selection uses only the private 512 development examples. The 657 examples
were previously audited by E147, so this is a reused audit, not untouched test
evidence. Forward timing includes packing/model/query work, excludes loading,
and is not measured energy or end-to-end service latency.
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
from e143_event_state_shd import evaluate as evaluate_combined
from e150_single_state_shd import evaluate as evaluate_single
from sleeping_machines.event_state import CoalescedEventStateEncoder
from sleeping_machines.shared_event import SharedEventModel
from sleeping_machines.depth_growth import grow_event_encoder


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=Path('experiments/results/e154')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(6);started=time.perf_counter()
    teacher_path=Path('experiments/results/e143/d8_parent_d6_state_residual_n6144_s6_e3_20260930.json')
    runs={'combined':teacher_path,
        'single_clean':Path('experiments/results/e150/single_state_n6144_s6_e3_20260930.json'),
        'single_paired':Path('experiments/results/e152/nuisance_state_n6144_s6_e2_20260930.json'),
        'grown_d12':Path('experiments/results/e159/calibrated_d12_n6144_s6_e1_20260930.json'),
        'matched_d6':Path('experiments/results/e159/calibrated_d6_n6144_s6_e1_20260930.json')}
    teacher=json.loads(teacher_path.read_text());cfg=teacher['args']
    excluded=set(teacher['fit_absolute_ids'])|set(teacher['dev_absolute_ids'])
    items=[r for r in load_marked(100000,'val_spk',7) if r[8] not in excluded]
    if len(items)!=657:raise ValueError('Unexpected reused audit split')
    rows={};files={};source_hashes={}
    for name,path in runs.items():
        run=json.loads(path.read_text())
        if run['status']!='completed':raise ValueError('Completed result required')
        if set(run['fit_absolute_ids'])&set(r[8] for r in items):raise ValueError('Audit overlaps fit')
        for source,digest in run['source_sha256'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:
                raise ValueError('Changed source: '+source)
            source_hashes[source]=digest
        selected=min(run['curve'],key=lambda r:(-r['dev']['correct'],r['dev']['nll']))
        checkpoint=path.with_name(path.stem+f".epoch{selected['epoch']}.pt")
        if not checkpoint.exists():checkpoint=path.with_suffix('.pt')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['result']['final']['epoch']!=selected['epoch']:raise ValueError('Wrong selected checkpoint')
        encoder=CoalescedEventStateEncoder(sources=720,width=cfg['width'],modes=cfg['modes'],depth=cfg['depth'])
        if saved.get('actual_depth',cfg['depth'])>cfg['depth']:
            encoder=grow_event_encoder(encoder,saved['actual_depth'])
        encoder.load_state_dict(saved['encoder_state_dict']);encoder.eval()
        core=None
        if name=='combined':
            parent=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
            core=SharedEventModel(depth=8,memory_backend='linear')
            core.load_state_dict(parent['state_dict']);core.requires_grad_(False);core.eval()
        def evaluate(batch):
            return (evaluate_combined(core,encoder,batch,4,cfg['window']) if core is not None else
                evaluate_single(encoder,batch,4,cfg['window']))
        evaluate(items[:16])
        before=time.perf_counter();score=evaluate(items);elapsed=time.perf_counter()-before
        rows[name]=dict(selected_epoch=selected['epoch'],development_correct=selected['dev']['correct'],
            audit=score,forward_wall_s=elapsed,utterances_per_second=len(items)/elapsed,
            deployed_parameters=sum(q.numel() for q in encoder.parameters())+
                (sum(q.numel() for q in core.parameters()) if core is not None else 0))
        if name=='grown_d12':
            layer_rows=[dict(index=i,output_map_norm=float(layer.output.weight.norm().detach()),
                normalization_gain_norm=float(layer.norm.weight.norm().detach()),
                normalization_bias_norm=float(layer.norm.bias.norm().detach()))
                for i,layer in enumerate(encoder.layers)]
            rows[name]['layer_parameters']=layer_rows
            encoder.layers=torch.nn.ModuleList(list(encoder.layers)[:cfg['depth']])
            evaluate(items[:16]);before=time.perf_counter()
            prefix_score=evaluate(items);prefix_elapsed=time.perf_counter()-before
            rows['grown_d12_prefix']=dict(selected_epoch=selected['epoch'],audit=prefix_score,
                forward_wall_s=prefix_elapsed,utterances_per_second=len(items)/prefix_elapsed,
                deployed_parameters=sum(q.numel() for q in encoder.parameters()),
                intervention='Delete only the six appended blocks from the trained D12 checkpoint; retain its trained prefix and head; no retraining or selection')
        files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        files[str(checkpoint)]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        del saved,encoder,core
    reference=rows['combined']['audit'];paired={}
    for name in ('single_clean','single_paired','grown_d12','matched_d6'):
        score=rows[name]['audit']
        if score['labels']!=reference['labels']:raise ValueError('Changed audit order')
        y=score['labels'];new=score['predictions'];old=reference['predictions']
        paired[name]=dict(new_only_correct=sum(p==label and q!=label for p,q,label in zip(new,old,y)),
            combined_only_correct=sum(p!=label and q==label for p,q,label in zip(new,old,y)))
    full=rows['grown_d12']['audit'];prefix=rows['grown_d12_prefix']['audit']
    if full['labels']!=prefix['labels']:raise ValueError('Changed prefix audit order')
    paired['depth_vs_its_prefix']=dict(
        full_only_correct=sum(p==y and q!=y for p,q,y in zip(full['predictions'],prefix['predictions'],full['labels'])),
        prefix_only_correct=sum(p!=y and q==y for p,q,y in zip(full['predictions'],prefix['predictions'],full['labels'])),
        changed_predictions=sum(p!=q for p,q in zip(full['predictions'],prefix['predictions'])))
    result=dict(status='completed',n=len(items),absolute_ids=[r[8] for r in items],rows=rows,paired=paired,
        selection='Best recorded saved epoch within each named run, ties broken by development NLL; unchanged starting head reported separately; audit never used to update or select',
        protocol=__doc__.strip(),hardware=dict(platform=platform.platform(),torch=torch.__version__,threads=1),
        file_sha256=files,source_sha256={**source_hashes,str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,energy_joules=None)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{'correct':v['audit']['correct'],'nll':v['audit']['nll'],
        'wall_s':v['forward_wall_s'],'parameters':v['deployed_parameters']} for k,v in rows.items()}),flush=True)


if __name__=='__main__':main()
