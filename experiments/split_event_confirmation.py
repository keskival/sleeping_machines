"""Guarded frozen-checkpoint confirmation; no optimizer or refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.split_event_benchmark import episodes,evaluate,sources
from experiments.native_event_tasks import data_hash
from sleeping_machines.split_event_heads import SplitEventHeads


def weight_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def restore(saved,public):
    """Require completed public/checkpoint lineage and reproduce selected dev loss."""
    row=saved['result'];a=row['args']
    if row.get('status')!='completed' or public.get('status')!='completed':
        raise ValueError('Completed checkpoint and published parent required')
    for key in ('args','source_sha256','data_sha256','selected_epoch','parameters','work'):
        if row[key]!=public[key]:raise ValueError('Checkpoint/public lineage differs: '+key)
    if row['source_sha256']!=sources():raise ValueError('Frozen parent source changed')
    for key,seed in (('fit',1201),('dev',2201)):
        rows=episodes(a['task'],a['sources'],a[key+'_targets'],seed)
        if data_hash(rows)!=row['data_sha256'][key]:raise ValueError('Parent data changed')
    model=SplitEventHeads(sources=a['sources'],classes=4 if a['task']=='order' else 2,
        payload=a['payload'],depth=a['depth'],pool=a['pool'],heads=a['heads'],credit=a['credit'],
        shared_maps=a['shared_maps'],protected_pairs=a['protected_pairs'])
    model.load_state_dict(saved['best_state']);model.eval();before=weight_hash(model)
    if sum(p.numel() for p in model.parameters())!=public['parameters']:
        raise ValueError('Parent parameter count differs')
    dev=evaluate(model,episodes(a['task'],a['sources'],a['dev_targets'],2201),a['time_input'],
                 paired=a['task']=='paired_timing')
    for key in ('accuracy','nll','n','correct'):
        if abs(dev[key]-public['final']['dev'][key])>1e-7:
            raise ValueError('Selected development score differs: '+key)
    if before!=weight_hash(model):raise ValueError('Frozen evaluation changed weights')
    return model,dev


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--parent',required=True);p.add_argument('--contracts-only',action='store_true')
    a=p.parse_args();out=ROOT/'experiments/results/event_confirmation'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Fresh unique confirmation tag required')
    parent=ROOT/a.parent;checkpoint=parent.with_suffix('.progress.pt')
    if not checkpoint.exists():raise ValueError('Parent checkpoint absent; never silently refit')
    public=json.loads(parent.read_text());saved=torch.load(checkpoint,weights_only=False)
    settings=public['args'];torch.set_num_threads(1);started=time.perf_counter()
    rng=torch.get_rng_state().clone()
    with torch.random.fork_rng():
        model,dev=restore(saved,public);before=weight_hash(model)
        final=dict(dev=dev)
        if not a.contracts_only:
            final['confirmation']=evaluate(model,episodes(settings['task'],settings['sources'],1024,3201),
                                            settings['time_input'],paired=settings['task']=='paired_timing')
        if before!=weight_hash(model):raise ValueError('Confirmation changed weights')
    if not torch.equal(rng,torch.get_rng_state()):raise ValueError('Outer RNG changed')
    provenance={**sources(),'experiments/split_event_confirmation.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result=dict(status='completed',args={**settings,**vars(a)},source_sha256=provenance,
        parent_result_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        data_sha256=public['data_sha256'],parameters=public['parameters'],final=final,
        work=public['work'],extra_optimizer_steps=0,
        protocol={**public['protocol'], 'synthetic_holdout_read':not a.contracts_only,
            'confirmation_data_seed':3201 if not a.contracts_only else None,
            'weights_frozen_before_confirmation':True,'parent_development_score_reproduced':True,
            'historical_fitting_work_charged':True},
        historical_fitting_wall_s=public['wall_s'],wall_s=time.perf_counter()-started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        parameter_weights_preserved=True,torch_rng_preserved=True,
        evaluation_work_scope='Actual dev/confirmation wall and target counts; inherited whole-fit and '
            'per-query inference traces are historical parent work, not zero-cost retraining or measured energy')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(tag=a.tag,contracts_only=a.contracts_only,
                         accuracy=final.get('confirmation',dev)['accuracy'])),flush=True)


if __name__=='__main__':main()
