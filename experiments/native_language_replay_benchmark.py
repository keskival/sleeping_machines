"""Source-exact, restartable native language full-write replay benchmark."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import native_language_helpers as baseline
from e120_shared_tasks import text_slice
from causal_language_replay_accumulator import ReplayAccumulator
from sleeping_machines.fast_native_core import FastNativeStreamLanguageModel
from race_language_screen import capture
from parallel_head_accumulated_language import merge


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--fit',type=int,default=10000000);p.add_argument('--dev',type=int,default=1000000)
    p.add_argument('--fit-start',type=int,default=0);p.add_argument('--dev-start',type=int,default=90000000)
    p.add_argument('--payload',type=int,default=16);p.add_argument('--depth',type=int,default=8)
    p.add_argument('--heads',type=int,default=2);p.add_argument('--pool',type=int,default=2)
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--chunk',type=int,default=16)
    p.add_argument('--update-targets',type=int,default=256);p.add_argument('--warmup-targets',type=int,default=4096)
    p.add_argument('--lr',type=float,default=.002);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--receiver-sharing',choices=('private','depth'),required=True)
    p.add_argument('--contracts',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-microchunks',type=int);p.add_argument('--stop-after-updates',type=int)
    p.add_argument('--no-trace',action='store_true',help='numerical comparison only; no work claim')
    return p


def sources():
    names=['experiments/native_language_replay_benchmark.py','experiments/causal_language_replay_accumulator.py',
        'experiments/causal_language_replay_helpers_rng.py','sleeping_machines/causal_language_shadow_rng.py',
        'experiments/parallel_head_gradient_accumulation.py','sleeping_machines/fast_native_core.py',
        'sleeping_machines/factorized_race.py','experiments/race_language_screen.py',
        'experiments/parallel_head_accumulated_language.py','experiments/theory/114_language_replay_benchmark_driver_contract.md']
    return {**baseline.source_hashes(),**{n:N.sha(ROOT/n) for n in names}}


def make(a):
    torch.manual_seed(a.seed);m=FastNativeStreamLanguageModel(payload=a.payload,depth=a.depth,pool=a.pool,heads=a.heads)
    if a.receiver_sharing=='depth':
        for h in range(m.heads):
            master=m.units[0][h][0][0]
            for layer in m.units:
                for unit in layer[h][0]:
                    for name in ('input','output','gate','control','key_read'):setattr(unit,name,getattr(master,name))
    return m


def digest_tokens(tokens):return hashlib.sha256(tokens.numpy().astype('uint8').tobytes()).hexdigest()


def run(a,directory=None,train=None,dev=None):
    torch.set_num_threads(1);started=time.perf_counter()
    directory=Path(directory) if directory else ROOT/'experiments/results/native_language_replay'
    directory.mkdir(parents=True,exist_ok=True);out=directory/(a.tag+'.json');cp=out.with_suffix('.progress.pt')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused plain completed-result tag required')
    if a.depth!=8 or a.update_targets%a.chunk or a.chunk<1 or not 1<=a.epochs<=16:
        raise ValueError('Depth8/multiple credit-window bounded protocol required')
    if not 2<=a.fit<=10000000 or not 2<=a.dev<=1000000 or a.warmup_targets<0:
        raise ValueError('Bounded data/warmup protocol required')
    if max(a.fit_start,a.dev_start)<min(a.fit_start+a.fit,a.dev_start+a.dev):raise ValueError('Disjoint data intervals required')
    contract=json.loads((ROOT/a.contracts).read_text());assert contract['status']=='completed'
    for name,sha in contract['source_sha256'].items():
        if N.sha(ROOT/name)!=sha:raise ValueError('Changed contracted source '+name)
    train=torch.tensor(text_slice(a.fit_start,a.fit)) if train is None else train.clone().long()
    dev=torch.tensor(text_slice(a.dev_start,a.dev)) if dev is None else dev.clone().long()
    if len(train)!=a.fit or len(dev)!=a.dev:raise ValueError('Exact data size required')
    data=dict(fit_sha256=digest_tokens(train),dev_sha256=digest_tokens(dev),
        fit_interval=[a.fit_start,a.fit_start+a.fit],dev_interval=[a.dev_start,a.dev_start+a.dev],
        scope='text8' if directory.name=='native_language_replay' else 'injected numerical data; no benchmark claim')
    settings={k:v for k,v in vars(a).items() if k not in ('tag','resume','stop_after_microchunks','stop_after_updates')}
    hashes=sources();m=make(a);opt=torch.optim.Adam(m.parameters(),lr=a.lr)
    learner=ReplayAccumulator(m,opt,a.lr,a.warmup_targets);best_state=None;best=math.inf;state=None;current=None
    cursor=dict(epoch=1,next_target=0,epoch_loss=0.,epoch_updates=0);prior_wall=0.
    if a.resume:
        saved=torch.load(cp,weights_only=False)
        if saved['settings']!=settings or saved['source_sha256']!=hashes or saved['data']!=data:
            raise ValueError('Changed source/settings/data recovery protocol')
        m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);learner.restore_counters(saved['counters'])
        for name,p in m.named_parameters():p.grad=saved['gradients'].get(name)
        state=saved['stream_state'];cursor=saved['cursor'];current=saved['current_window'];r=saved['result']
        best_state=saved['best_state'];best=saved['best'];torch.set_rng_state(saved['torch_rng']);prior_wall=r['wall_s']
    else:
        if cp.exists():raise ValueError('Explicit exact recovery required')
        r=dict(status='running',args=vars(a),data=data,source_sha256=hashes,
            parameters=sum(p.numel() for p in m.parameters()),initial_dev=baseline.evaluate(m,dev,a.chunk),curve=[],
            work_samples=[],window_size_counts={},activity=dict(targets=0,key_scores=0,selected_updates=0,shadow_lanes=0,shadow_events=0),
            protocol=dict(credit='all-target first-time-preserving full-write local expectation plus factorized clock/content',
                chronological_rng='factual advances once; all shadows fork same entering RNG',
                actual_state='persistent across credit/update windows; graph detached each credit chunk',official_test_read=False,
                fitting_targets_per_pass=a.fit-1,development_targets=a.dev-1,credit_targets=a.chunk,optimizer_targets=a.update_targets,
                scope='Single-seed DEV-selected native benchmark; no whole-stream exact-gradient/supremacy claim'))
    def persist(final=False):
        r.update(wall_s=prior_wall+time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        target=out if final else out.with_suffix('.running.json');tmp=target.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(target)
        payload=dict(settings=settings,data=data,source_sha256=hashes,model=m.state_dict(),optimizer=opt.state_dict(),
            gradients={n:p.grad.detach().clone() for n,p in m.named_parameters() if p.grad is not None},
            stream_state=state,cursor=cursor,current_window=current,counters=learner.counters(),
            best_state=best_state,best=best,result=r,torch_rng=torch.get_rng_state())
        tmp=cp.with_suffix('.pt.tmp');torch.save(payload,tmp);tmp.replace(cp)
    while cursor['epoch']<=a.epochs:
        epoch=cursor['epoch']
        if state is None:state=m.new_state()
        while cursor['next_target']<a.fit-1:
            start=cursor['next_target']
            if current is None:
                expected=min(a.update_targets,a.fit-1-start);window=cursor['epoch_updates']
                last=math.ceil((a.fit-1)/a.update_targets)-1
                trace=not a.no_trace and ((epoch==1 and window==0) or (epoch==a.epochs and window==last)
                    or expected<a.update_targets)
                current=dict(epoch=epoch,window=window,expected_targets=expected,trace=trace,accumulate=[])
            end=min(start+a.chunk,start+a.update_targets-learner.pending_targets,a.fit-1);count=end-start;box={}
            before_lanes=learner.shadow_lanes;before_events=learner.shadow_events
            def micro():box['value']=learner.accumulate(train[start:end],train[start+1:end+1],state)
            if current['trace']:current['accumulate'].append(capture(micro))
            else:micro()
            loss,state,_=box['value'];cursor['next_target']=end;cursor['epoch_loss']+=loss
            r['activity']['targets']+=count;r['activity']['key_scores']+=count*a.depth*a.heads*a.pool
            r['activity']['selected_updates']+=count*a.depth*a.heads
            r['activity']['shadow_lanes']+=learner.shadow_lanes-before_lanes;r['activity']['shadow_events']+=learner.shadow_events-before_events
            if learner.pending_targets==current['expected_targets']:
                if current['trace']:
                    stages=dict(full_credit_accumulate=merge(current['accumulate']),
                        gradient_normalization=capture(learner.normalize),
                        gradient_clipping=capture(lambda:torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)),
                        optimizer=capture(learner.step))
                    if not all(t['formula_coverage_complete'] for t in stages.values()):raise ValueError('Incomplete fitting accounting')
                    r['work_samples'].append(dict(epoch=epoch,window=current['window'],targets=current['expected_targets'],stages=stages))
                else:learner.update()
                key=str(current['expected_targets']);r['window_size_counts'][key]=r['window_size_counts'].get(key,0)+1
                cursor['epoch_updates']+=1;current=None
            if (a.stop_after_microchunks is not None and learner.credit_chunks>=a.stop_after_microchunks) or \
               (a.stop_after_updates is not None and learner.updates>=a.stop_after_updates):
                persist();return r
            if learner.credit_chunks%16==0:persist()
        score=baseline.evaluate(m,dev,a.chunk)
        r['curve'].append(dict(epoch=epoch,dev=score,online_bpc=cursor['epoch_loss']/(a.fit-1)/math.log(2),updates=learner.updates))
        if score['bpc']<best:best=score['bpc'];best_state=copy.deepcopy(m.state_dict());r['selected_epoch']=epoch
        state=None;cursor=dict(epoch=epoch+1,next_target=0,epoch_loss=0.,epoch_updates=0);persist()
    online=copy.deepcopy(m.state_dict());m.load_state_dict(best_state);r['final']=dict(dev=baseline.evaluate(m,dev,a.chunk))
    estimates={}
    if not a.no_trace:
        for size,number in r['window_size_counts'].items():
            samples=[row for row in r['work_samples'] if row['targets']==int(size)]
            if not samples:raise ValueError('Missing actual optimizer-window size accounting')
            for stage in samples[0]['stages']:
                estimates[stage]=estimates.get(stage,0.)+number*float(np.mean([s['stages'][stage]['arithmetic_flops']+
                    s['stages'][stage]['special_function_evaluations'] for s in samples]))
    inf_state=m.new_state();m.eval();probe=min(a.chunk,a.dev-1)
    with torch.no_grad(),torch.random.fork_rng():
        torch.manual_seed(314159)
        def inference():m.forward_chunk(dev[:probe],inf_state)
        itr=capture(inference)
    if not itr['formula_coverage_complete']:raise ValueError('Incomplete native inference accounting')
    total=sum(estimates.values()) if not a.no_trace else None;targets=r['activity']['targets']
    r['work']=dict(fitting_targets=targets,optimizer_updates=learner.updates,
        whole_fit_unit_special_flops_estimate=total,fit_unit_special_flops_per_target_estimate=None if total is None else total/targets,
        stage_estimates=estimates,inference_unit_special_flops_per_target_estimate=(itr['arithmetic_flops']+itr['special_function_evaluations'])/probe,
        available_receivers=a.depth*a.heads*a.pool,inference_scope='Cold native selected-value prefix; no shadow learning in inference',
        estimate_scope='First/last full and all partial actual optimizer-window audits by target-size; all replay/factual backward, '
            'normalization/clipping/Adam paid.2FLOPs/MAC plus unit-special; DEV passes/RNG/traffic/energy separate; no physical projection.')
    r['learner_counters']=learner.counters();r['status']='completed';m.load_state_dict(online);persist(final=True)
    return r


if __name__=='__main__':run(parser().parse_args())
