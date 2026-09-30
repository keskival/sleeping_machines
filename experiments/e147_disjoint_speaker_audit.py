"""Evaluate the selected E143 model on all remaining speaker3/6 utterances.

Disjoint from the current 512-example development sample and all current fit
examples. This is a same-speaker transfer audit, not a new-speaker or official
test result; historical project-wide non-exposure is not asserted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
from e143_event_state_shd import evaluate
from e139_fine_packet_model import load_marked
from sleeping_machines.shared_event import SharedEventModel
from sleeping_machines.event_state import CoalescedEventStateEncoder


def main():
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True);a=p.parse_args()
    out=Path("experiments/results/e147")/(a.tag+".json");out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError("Unique output required")
    run_path=Path('experiments/results/e143/d8_parent_d6_state_residual_n6144_s6_e3_20260930.json')
    run=json.loads(run_path.read_text())
    if run['status']!='completed':raise ValueError('Completed run required')
    selected=max(run['curve'],key=lambda r:r['dev']['correct']);cfg=run['args']
    base=run_path.with_suffix('.pt');checkpoint=base.with_name(base.stem+f".epoch{selected['epoch']}.pt")
    if not checkpoint.exists():checkpoint=base
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['result']['final']['epoch']!=selected['epoch']:raise ValueError('Wrong selected checkpoint')
    for source,digest in run['source_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed source: {source}')
    torch.set_num_threads(1);torch.manual_seed(cfg['seed']);started=time.perf_counter()
    parent=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    core=SharedEventModel(depth=8,memory_backend='linear')
    core.load_state_dict(parent['state_dict']);core.requires_grad_(False);core.eval()
    encoder=CoalescedEventStateEncoder(sources=720,width=cfg['width'],modes=cfg['modes'],depth=cfg['depth'])
    encoder.load_state_dict(saved['encoder_state_dict'])
    excluded=set(run['fit_absolute_ids'])|set(run['dev_absolute_ids'])
    all_held=load_marked(100000,'val_spk',cfg['seed']+1)
    items=[r for r in all_held if r[8] not in excluded]
    if not items:raise ValueError('No disjoint utterances')
    learned=evaluate(core,encoder,items,cfg['bs'],cfg['window'])
    with torch.no_grad():encoder.head.weight.zero_();encoder.head.bias.zero_()
    reference=evaluate(core,encoder,items,cfg['bs'],cfg['window'])
    assert reference['correct']==reference['parent_correct'] and reference['nll']==reference['parent_nll']
    labels=learned['labels'];new=learned['predictions'];old=reference['predictions']
    new_only=sum(p==y and q!=y for p,q,y in zip(new,old,labels))
    parent_only=sum(p!=y and q==y for p,q,y in zip(new,old,labels))
    per_class=[]
    for value in range(20):
        idx=[i for i,y in enumerate(labels) if y==value]
        per_class.append(dict(label=value,n=len(idx),parent_correct=sum(old[i]==value for i in idx),
            residual_correct=sum(new[i]==value for i in idx)))
    result=dict(status='completed',selected_epoch=selected['epoch'],checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        n=len(items),absolute_ids=[r[8] for r in items],parent=reference,residual=learned,
        new_only_correct=new_only,parent_only_correct=parent_only,
        gain_percentage_points=100*(new_only-parent_only)/len(items),per_class=per_class,
        selection='Architecture and checkpoint selected on prior 512 private-development examples; audit includes all remaining eligible speaker3/6 utterances, no audit updates or checkpoint reselection',
        scope=__doc__.strip(),wall_s=time.perf_counter()-started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            [Path(__file__),Path('experiments/e143_event_state_shd.py'),Path('sleeping_machines/event_state.py'),Path('experiments/e139_fine_packet_model.py')]})
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('n','new_only_correct','parent_only_correct','gain_percentage_points','wall_s','max_rss_kb')}),flush=True)
    print(json.dumps({'parent_accuracy':reference['accuracy'],'residual_accuracy':learned['accuracy'],'parent_nll':reference['nll'],'residual_nll':learned['nll']}),flush=True)


if __name__=='__main__':main()
