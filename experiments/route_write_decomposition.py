"""Guarded frozen 2x2 delivery/commit replay; no optimizer or model change."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.frozen_native_route_audit import weight_hash
from native_event_contracts import predict
from native_event_tasks import episodes
from sleeping_machines.addressed_event_heads import AddressedEventHeads


def replay(model,row,node,seed,delivery=None,commit=None,gradients=False):
    original=model.race;trace={};count=0
    def race(scores,values=None):
        nonlocal count
        index=count;count+=1
        value,delay,winner=original(scores,values)
        if index==node:
            trace.update(values=values.detach().clone(),scores=scores.detach().clone(),
                winner=int(winner),delay=float(delay.detach()))
            if gradients:
                value.retain_grad();trace['live_value']=value
            if delivery is not None:value=values[delivery]
            if commit is not None:winner=torch.tensor(commit,device=scores.device)
        return value,delay,winner
    model.race=race
    try:
        model.zero_grad(set_to_none=True);model.train();torch.manual_seed(seed)
        with torch.set_grad_enabled(gradients):
            logits,target,state,_=predict(model,row)
            loss=F.cross_entropy(logits,target,reduction='sum')
            if gradients:
                loss.backward();trace['error_value']=trace.pop('live_value').grad.detach().clone()
        trace.update(loss=float(loss.detach()),races=count,commits=state.selected_updates,
            rng=torch.get_rng_state().clone(),storage=state.storage())
        return trace
    finally:model.race=original


def components(f00,f10,f01,f11,linear):
    value=f10-f00;write=f01-f00;interaction=f11-f10-f01+f00
    return dict(total_branch_loss_difference=f11-f00,delivered_value_effect=value,
        persistent_commit_effect=write,delivery_commit_interaction=interaction,
        local_delivered_value_linearization=linear,value_linearization_residual=value-linear,
        teacher_residual=f11-f00-linear,
        residual_reconstruction=(value-linear)+write+interaction)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--checkpoint',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Fresh unique audit tag required')
    checkpoint=ROOT/a.checkpoint;saved=torch.load(checkpoint,weights_only=False)
    source=saved['result'];settings=source['args']
    if source['status']!='completed' or settings['pool']!=2:raise ValueError('Completed two-candidate checkpoint required')
    for name,sha in source['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Source lineage changed: '+name)
    torch.set_num_threads(1);started=time.perf_counter();probes=[]
    with torch.random.fork_rng():
        model=AddressedEventHeads(sources=settings['sources'],payload=settings['payload'],depth=settings['depth'],
            pool=2,heads=settings['heads'],classes=4 if settings['task']=='order' else 2,credit=settings['credit'])
        model.load_state_dict(saved['best_state']);before=weight_hash(model)
        row=episodes(settings['task'],settings['sources'],settings['sources'],2201)[0]
        nodes=[(event,depth,(event*settings['depth']+depth)*settings['heads'])
            for event,e in enumerate(row) if e.source==0 for depth in (0,settings['depth']//2,settings['depth']-1)]
        for event,depth,node in nodes:
            base=replay(model,row,node,619,gradients=True);winner=base['winner'];alternative=1-winner
            value=replay(model,row,node,619,delivery=alternative,commit=winner)
            write=replay(model,row,node,619,delivery=winner,commit=alternative)
            full=replay(model,row,node,619,delivery=alternative,commit=alternative)
            for r in (value,write,full):
                assert (r['races'],r['commits'],r['delay'])==(base['races'],base['commits'],base['delay'])
                assert torch.equal(r['rng'],base['rng'])
            linear=float((base['values'][alternative]-base['values'][winner])@base['error_value'])
            c=components(base['loss'],value['loss'],write['loss'],full['loss'],linear)
            if abs(c['teacher_residual']-c['residual_reconstruction'])>1e-10:
                raise ValueError('Decomposition identity failed')
            probes.append(dict(event=event,depth=depth,head=0,node=node,winner=winner,alternative=alternative,
                branch_losses=dict(base=base['loss'],value_only=value['loss'],commit_only=write['loss'],both=full['loss']),
                components=c,races_per_replay=base['races'],common_time_noise_verified=True))
        assert before==weight_hash(model)
    names=probes[0]['components']
    result=dict(status='completed',args=vars(a),probes=probes,
        summary={k:sum(abs(p['components'][k]) for p in probes)/len(probes) for k in names},
        parameter_weights_preserved=True,torch_rng_preserved=True,
        source_sha256={**source['source_sha256'],**{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('experiments/route_write_decomposition.py','experiments/frozen_native_route_audit.py')}},
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        accounting=dict(forward_replays=4*len(probes),backward_replays=len(probes),
            total_races=4*sum(p['races_per_replay'] for p in probes),
            scope='Actual replay counts and whole wall/RSS; arithmetic FLOPs not instrumented'),
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen factorial intervention; hybrid delivery/commit combinations are diagnostic counterfactuals, not legal new model routes. '
              'Value residual includes nonlinear response and downstream route switches, not just smooth curvature. '
              'One population/address/noise seed, fixed selected-node time; no expected-gradient or quality claim.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
