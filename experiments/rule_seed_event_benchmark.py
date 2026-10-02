"""Guarded protected-state/shared-processing and identifiable-timing experiments.

Each host runs one unique queue job. All fitting arithmetic is audited, not
extrapolated. Sources address observed streams, not labels or oracle routes.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from native_event_tasks import episodes as original_episodes,data_hash
from experiments.paired_timing_tasks import paired_timing_episodes,pair_bootstrap
from rule_seed_event_contracts import contracts,predict
from parallel_head_accumulated_language import merge
from race_language_screen import capture
from sleeping_machines.rule_seed_event_heads import RuleSeedEventHeads as SplitEventHeads


def episodes(task,sources,targets,seed,stretch=1.):
    if task=="paired_timing":
        if stretch!=1.:raise ValueError("Paired timing has declared short/long ages")
        return paired_timing_episodes(sources,targets,seed)
    return original_episodes(task,sources,targets,seed,stretch)


def sources():
    names=('experiments/rule_seed_event_benchmark.py','experiments/paired_timing_tasks.py','experiments/native_event_tasks.py',
        'experiments/rule_seed_event_contracts.py','sleeping_machines/rule_seed_event_heads.py','sleeping_machines/split_event_heads.py','sleeping_machines/addressed_event_heads.py','sleeping_machines/addressed_event_heads.py',
        'sleeping_machines/parallel_head_race_language.py','sleeping_machines/sparse_race_language.py',
        'sleeping_machines/parallel_stream_language.py','sleeping_machines/operation_audit.py',
        'experiments/race_language_screen.py','experiments/parallel_head_accumulated_language.py')
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


@torch.no_grad()
def evaluate(model,rows,mode='observed',clear=False,paired=False):
    with torch.random.fork_rng():
        torch.manual_seed(314159);model.eval();started=time.perf_counter()
        loss,correct,n,events,updates,scores=0.,0,0,0,0,0;cluster=[];peak=0;wait=0.
        for index,row in enumerate(rows):
            if paired:
                if index%2==0:pair_rng=torch.get_rng_state()
                else:torch.set_rng_state(pair_rng)
            z,y,state,_=predict(model,row,mode,clear)
            loss+=float(F.cross_entropy(z,y,reduction='sum'))
            hit=int((z.argmax(-1)==y).sum());correct+=hit;n+=len(y)
            cluster.append(hit/len(y));events+=state.events;updates+=state.selected_updates
            scores+=state.candidate_scores;peak=max(peak,state.storage()['persistent_tensor_bytes'])
            wait+=state.queue_wait_sum
        rng=np.random.default_rng(419)
        sampled=rng.choice(cluster,(1000,len(cluster)),replace=True).mean(1)
        return dict(n=n,nll=loss/n,correct=correct,accuracy=correct/n,episodes=len(rows),
            episode_accuracy=cluster,episode_bootstrap_95_percent_interval=(pair_bootstrap(cluster) if paired else
                np.quantile(sampled,(.025,.975)).tolist() if len(cluster)>1 else None),
            independent_clusters=len(cluster)//2 if paired else len(cluster),
            events=events,selected_updates=updates,key_scores=scores,persistent_tensor_bytes=peak,
            source_queue_wait_sum=wait,wall_s=time.perf_counter()-started,
            scope='Frozen weights; complete population-pair bootstrap for paired timing, otherwise whole populations; unavailable with one cluster; not seed uncertainty')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--common-source-seed',type=int,choices=(0,1),required=True)
    p.add_argument('--tag',required=True);p.add_argument('--task',choices=('order','timing','paired_timing'),default='order')
    p.add_argument('--sources',type=int,choices=(4,16,64),default=4)
    p.add_argument('--fit-targets',type=int,default=512);p.add_argument('--dev-targets',type=int,default=256)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--seed',type=int,default=6)
    p.add_argument('--payload',type=int,default=16);p.add_argument('--depth',type=int,default=8)
    p.add_argument('--heads',type=int,default=2);p.add_argument('--pool',type=int,default=2)
    p.add_argument('--update-targets',type=int,default=64);p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--credit',choices=('counterfactual','pathwise'),default='counterfactual')
    p.add_argument('--time-input',choices=('observed','rank'),default='observed')
    p.add_argument('--contracts-only',action='store_true');p.add_argument('--confirm',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--shared-maps',action='store_true')
    p.add_argument('--protected-pairs',type=int,default=0)
    a=p.parse_args()
    directory=ROOT/'experiments/results/event_variants';directory.mkdir(exist_ok=True)
    out=directory/(a.tag+'.json');running=out.with_suffix('.running.json');checkpoint=out.with_suffix('.progress.pt')
    if Path(a.tag).name!=a.tag or out.exists() or (running.exists() and not a.resume):
        raise ValueError('Unused tag or exact-source recovery required')
    if (not a.sources<=a.fit_targets<=2048 or not a.sources<=a.dev_targets<=1024 or
            not 1<=a.epochs<=16 or any(v%a.sources for v in (a.fit_targets,a.dev_targets,a.update_targets))
            or a.fit_targets%a.update_targets):
        raise ValueError('Bounded, complete source populations and optimizer windows required')
    if a.task=='paired_timing' and any(v%(2*a.sources) for v in (a.fit_targets,a.dev_targets)):
        raise ValueError('Complete short/long population pairs required')
    torch.set_num_threads(1);torch.manual_seed(a.seed);started=time.perf_counter()
    checks=contracts(a.sources,a.payload,a.depth,a.pool,a.heads,a.shared_maps,a.protected_pairs,a.common_source_seed)
    provenance=sources()
    result=dict(status='running',args=vars(a),source_sha256=provenance,numerical_contracts=checks)
    if a.contracts_only:
        result.update(status='completed',wall_s=time.perf_counter()-started,
            max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        out.write_text(json.dumps(result,indent=2)+'\n');return
    fit=episodes(a.task,a.sources,a.fit_targets,1201);dev=episodes(a.task,a.sources,a.dev_targets,2201)
    model=SplitEventHeads(sources=a.sources,classes=4 if a.task=='order' else 2,
        payload=a.payload,depth=a.depth,pool=a.pool,heads=a.heads,credit=a.credit,
        shared_maps=a.shared_maps,protected_pairs=a.protected_pairs,common_source_seed=a.common_source_seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    best=float('inf');best_state=None;ledger={k:[] for k in ('forward_and_loss','backward',
        'gradient_normalization','gradient_clipping','optimizer')}
    rng=np.random.default_rng(a.seed+10);cursor=dict(epoch=1,next_episode=0,order=None)
    activity=dict(events=0,key_scores=0,selected_updates=0,counterfactual_proposals=0)
    previous_wall=0.
    if a.resume:
        saved=torch.load(checkpoint,weights_only=False);result=saved['result']
        if result['source_sha256']!=provenance:raise ValueError('Source changed during recovery')
        if {k:v for k,v in result['args'].items() if k!='resume'}!={k:v for k,v in vars(a).items() if k!='resume'}:
            raise ValueError('Settings changed during recovery')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best_state=saved['best_state'];best=saved['best'];ledger=saved['ledger'];activity=saved['activity']
        cursor=saved['cursor'];rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
        previous_wall=result['wall_s']
    else:
        result.update(parameters=sum(p.numel() for p in model.parameters()),curve=[],
            hardware=dict(platform=platform.platform(),torch=torch.__version__,device='cpu',threads=1),
            data_sha256=dict(fit=data_hash(fit),dev=data_hash(dev)),
            protocol=dict(official_test_read=False,synthetic_holdout_read=a.confirm,
                fit_data_seed=1201,development_data_seed=2201,confirmation_data_seed=3201 if a.confirm else None,
                queries_per_pass=a.fit_targets,events_per_pass=4*a.fit_targets,
                input='Observed source address, physical timestamp, signed mark and query flag; targets never included',
                state='Independent source contexts and source/head receiver memories, reset between whole populations',
                occupied_capacity='Every source writes three marks and is queried; capacity is not an unused padded pool',
                learning='All per-population producer graphs retained to queries; sums normalized by actual query targets; Adam per declared window',
                architecture='Native addressed independent heads with declared protected pairs and shared/private rules; no KV or dense carrier',
                state_subspaces='Identity silent prefix in receiver memory and head/context transport; temporal suffix still computes with time; event writes/mixing can change all coordinates',
                parameter_sharing=dict(shared_receiver_maps=a.shared_maps,common_source_seed=bool(a.common_source_seed),private_addressed_states=True),
                paired_timing=a.task=='paired_timing',
                evaluation_noise='Common random numbers within each short/long pair' if a.task=='paired_timing' else 'Fixed independent population stream',
                paired_timing_distribution='Identical marks, addresses, rank order and opposite labels for short2.2/long8.2 query ages; bounded mark rejection' if a.task=='paired_timing' else None,
                selection='Lowest frozen development NLL over fixed passes; no test selection',
                scope='Synthetic capability/mechanism pilot; cannot establish real-data or energy supremacy'))

    def persist():
        result.update(wall_s=previous_wall+time.perf_counter()-started,
            max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        target=out if result['status']=='completed' else running
        temporary=target.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(target)
        temporary=checkpoint.with_suffix('.pt.tmp')
        torch.save(dict(result=result,model=model.state_dict(),optimizer=optimizer.state_dict(),
            best_state=best_state,best=best,ledger=ledger,activity=activity,cursor=cursor,
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state()),temporary);temporary.replace(checkpoint)

    persist();window=a.update_targets//a.sources
    for epoch in range(cursor['epoch'],a.epochs+1):
        order=cursor['order'] if cursor['order'] is not None else rng.permutation(len(fit)).tolist()
        start=cursor['next_episode'];total=0.;model.train()
        for begin in range(start,len(order),window):
            optimizer.zero_grad(set_to_none=True);pending=0
            for index in order[begin:begin+window]:
                box={}
                def forward():
                    z,y,state,_=predict(model,fit[index],a.time_input)
                    box.update(loss=F.cross_entropy(z,y,reduction='sum'),state=state)
                ledger['forward_and_loss'].append(capture(forward))
                if not torch.isfinite(box['loss']):raise FloatingPointError('Nonfinite loss')
                ledger['backward'].append(capture(lambda:box['loss'].backward()))
                pending+=a.sources;total+=float(box['loss'].detach())
                state=box['state'];activity['events']+=state.events
                activity['key_scores']+=state.candidate_scores;activity['selected_updates']+=state.selected_updates
                activity['counterfactual_proposals']+=state.counterfactual_values
                state.detach();del box,state
            def normalize():
                for parameter in model.parameters():
                    if parameter.grad is not None:parameter.grad.div_(pending)
            ledger['gradient_normalization'].append(capture(normalize))
            ledger['gradient_clipping'].append(capture(lambda:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)))
            ledger['optimizer'].append(capture(optimizer.step))
            # Compact the exact sum as work proceeds; checkpoints stay bounded.
            for key in ledger:
                ledger[key]=[merge(ledger[key])]
            cursor=dict(epoch=epoch,next_episode=begin+window,order=order);persist()
        score=evaluate(model,dev,a.time_input,paired=a.task=='paired_timing')
        result['curve'].append(dict(epoch=epoch,dev=score,training_nll=total/a.fit_targets,
            training_nll_scope='This process segment; resumed partial pass excludes its earlier losses'))
        if score['nll']<best:best=score['nll'];best_state=copy.deepcopy(model.state_dict());result['selected_epoch']=epoch
        cursor=dict(epoch=epoch+1,next_episode=0,order=None);persist()
        print(json.dumps(dict(epoch=epoch,dev_accuracy=score['accuracy'],dev_nll=score['nll'])),flush=True)
    model.load_state_dict(best_state)
    result['final']=dict(dev=evaluate(model,dev,a.time_input,paired=a.task=='paired_timing'),
        cleared_state=evaluate(model,dev,a.time_input,True,paired=a.task=='paired_timing'),
        rank_time=evaluate(model,dev,'rank',paired=a.task=='paired_timing'))
    if a.task=='order':
        result['final']['long_gaps']={str(scale):evaluate(model,episodes(a.task,a.sources,a.dev_targets,2201,scale),a.time_input)
            for scale in (8.,64.)}
    if a.confirm:
        result['final']['confirmation']=evaluate(model,episodes(a.task,a.sources,1024,3201),a.time_input,paired=a.task=='paired_timing')
    model.eval();box={}
    with torch.no_grad():
        def inference():box['z'],box['y'],box['state'],_=predict(model,dev[0],a.time_input)
        inference_trace=capture(inference)
    train={k:merge(v) for k,v in ledger.items()}
    arithmetic=sum(t['arithmetic_flops'] for t in train.values())
    special=sum(t['special_function_evaluations'] for t in train.values())
    result['work']=dict(fitting_query_targets=a.fit_targets*a.epochs,fitting_events=4*a.fit_targets*a.epochs,
        optimizer_steps=(a.fit_targets//a.update_targets)*a.epochs,
        total_training_arithmetic_flops=arithmetic,training_special_function_evaluations=special,
        total_training_unit_special_flops=arithmetic+special,training_stages=train,
        inference_trace=inference_trace,
        inference_arithmetic_flops_per_event=inference_trace['arithmetic_flops']/len(dev[0]),
        inference_special_functions_per_event=inference_trace['special_function_evaluations']/len(dev[0]),
        inference_arithmetic_flops_per_query=inference_trace['arithmetic_flops']/a.sources,
        inference_special_functions_per_query=inference_trace['special_function_evaluations']/a.sources,
        exact_fitting_activity=activity,available_receivers=a.sources*a.depth*a.heads*a.pool,
        selected_updates_per_event=a.depth*a.heads,key_scores_per_event=a.depth*a.heads*a.pool,
        inference_storage=box['state'].storage(),
        scope='Exact executed fitting sums including all input events, backward, losing proposals, normalization, clipping and Adam; 2 FLOPs/MAC, specials separate plus unit weight; CPU numeric clocks included; dev/RNG/traffic/energy separate')
    result['status']='completed';persist();print(json.dumps(dict(completed=a.tag,accuracy=result['final']['dev']['accuracy'])),flush=True)


if __name__=='__main__':main()
