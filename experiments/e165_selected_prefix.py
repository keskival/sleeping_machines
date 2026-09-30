"""Export a development-selected prefix of the completed directional model.

Both full and prefix development scores were recorded by E163 before the
reused E164 audit. Choose between those two candidates by development correct,
then NLL; never use audit labels for selection. This post-hoc deletion is an
exploratory deployment decision, not a matched retrained depth result.
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
import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    out=Path('experiments/results/e165')/(args.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(args.tag).name!=args.tag or out.exists() or out.with_suffix('.pt').exists():
        raise ValueError('Unique outputs required')
    torch.set_num_threads(1);started=time.perf_counter()
    paths={name:Path('experiments/results')/path for name,path in (
        ('training','e163/observer_depth_n6144_s6_e1_20260930.json'),
        ('audit','e164/observer_depth_audit_20260930.json'),
        ('reference','e154/single_encoder_audit_20260930.json'))}
    records={name:json.loads(path.read_text()) for name,path in paths.items()}
    sources={}
    for record in records.values():
        if record['status']!='completed':raise ValueError('Completed evidence required')
        for path,digest in record['source_sha256'].items():
            if sha(path)!=digest:raise ValueError('Changed executed source: '+path)
            sources[path]=digest
    run=records['training'];audit=records['audit'];reference=records['reference']
    final=run['final'];candidates={'full':final['dev'],'prefix':final['prefix_dev']}
    selected=min(candidates,key=lambda name:(-candidates[name]['correct'],candidates[name]['nll']))
    if selected!='prefix':raise ValueError('This exporter is for a selected prefix')
    checkpoint=paths['training'].with_name(paths['training'].stem+f".epoch{final['epoch']}.pt")
    if sha(checkpoint)!=audit['checkpoint_sha256']:raise ValueError('Changed audited checkpoint')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    depth=run['starting_prefix_depth'];state={}
    for name,tensor in saved['encoder_state_dict'].items():
        if name.startswith('layers.') and int(name.split('.')[1])>=depth:continue
        state[name]=tensor
    optimizer=copy.deepcopy(saved['optimizer']);optimizer['param_groups']=optimizer['param_groups'][:1]
    old_ids=set(optimizer['param_groups'][0]['params'])
    optimizer['state']={key:value for key,value in optimizer['state'].items() if key in old_ids}
    score=audit['rows']['directional_prefix']['audit'];pairs={}
    for name,row in (('combined',reference['rows']['combined']),('calibrated_d6',reference['rows']['matched_d6'])):
        old=row['audit']
        if old['labels']!=score['labels']:raise ValueError('Changed audit ordering')
        pairs[name]=dict(selected_only_correct=sum(p==y and q!=y for p,q,y in zip(score['predictions'],old['predictions'],score['labels'])),
            reference_only_correct=sum(p!=y and q==y for p,q,y in zip(score['predictions'],old['predictions'],score['labels'])))
    result=dict(status='completed',selected='prefix',trained_depth=run['args']['depth'],deployed_depth=depth,
        deployed_parameters=audit['rows']['directional_prefix']['deployed_parameters'],
        candidates=candidates,final=dict(epoch=final['epoch'],dev=candidates[selected]),
        reused_audit=score,reused_audit_paired=pairs,
        observed_forward_wall_s=audit['rows']['directional_prefix']['forward_wall_s'],
        selection=__doc__.strip(),fit_absolute_ids=run['fit_absolute_ids'],dev_absolute_ids=run['dev_absolute_ids'],
        training_cost_scope='Retain E143 inherited encoder training, paired head/teacher work, first-step replays and the full twelve-block E163 pass; deployment pruning does not erase those costs',
        checkpoint_sha256=sha(checkpoint),result_sha256={str(path):sha(path) for path in paths.values()},
        source_sha256={**sources,str(Path(__file__)):sha(__file__)},energy_joules=None,
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    payload=dict(encoder_state_dict=state,optimizer=optimizer,encoder_config=saved['encoder_config'],
        actual_depth=depth,result=result,args=vars(args),
        order_rng=saved['order_rng'],augmentation_rng=saved['augmentation_rng'],torch_rng=saved['torch_rng'])
    torch.save(payload,out.with_suffix('.pt'))
    result['exported_checkpoint_sha256']=sha(out.with_suffix('.pt'))
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(selected=selected,development_correct=candidates[selected]['correct'],
        development_nll=candidates[selected]['nll'],audit_correct=score['correct'],
        parameters=result['deployed_parameters'],paired=pairs)),flush=True)


if __name__=='__main__':main()
