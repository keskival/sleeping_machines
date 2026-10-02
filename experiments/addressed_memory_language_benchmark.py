"""Checkpointed matched native/addressed/late-projection learning diagnostics.

Whole fit uses audited representative first/mature/partial windows; estimates
are labelled explicitly. Cold development alone selects the saved checkpoint.
Frozen fit replay is a separately charged, causal secondary evaluation.
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

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import native_language_helpers as baseline
from e120_shared_tasks import text_slice
from race_language_screen import capture
from parallel_head_gradient_accumulation import GradientAccumulator
from native_language_gradient_contracts import accumulation_contracts
from sleeping_machines.native_stream_language import NativeStreamLanguageModel
from sleeping_machines.context_addressed_memory import ContextAddressedNativeModel
from sleeping_machines.late_projected_context_memory import LateProjectedContextModel


def sources():
    names=['experiments/addressed_memory_language_benchmark.py',
           'sleeping_machines/context_addressed_memory.py',
           'sleeping_machines/late_projected_context_memory.py',
           'experiments/native_language_gradient_contracts.py',
           'experiments/parallel_head_gradient_accumulation.py']
    return {**baseline.source_hashes(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def make_model(a):
    kwargs=dict(payload=a.payload,depth=a.depth,pool=a.pool,heads=a.heads)
    if a.model=='native':return NativeStreamLanguageModel(**kwargs)
    cls=ContextAddressedNativeModel if a.model=='addressed' else LateProjectedContextModel
    return cls(**kwargs,order=a.order,buckets=a.buckets)


def activity(state,model):
    return {**baseline.activity(state,model),
        'context_reads':getattr(state,'memory_reads',0),
        'context_writes':getattr(state,'memory_writes',0)}


def evaluate(model,tokens,chunk,prefix=None,audit_prefix=False):
    """Cold or frozen-replay evaluation with identical dev race-noise draws."""
    with torch.no_grad(),torch.random.fork_rng():
        model.eval();state=model.new_state();replay_tokens=0;replay_traces={};infer=None
        if prefix is not None:
            torch.manual_seed(271828)
            for begin in range(0,len(prefix),chunk):
                end=min(begin+chunk,len(prefix))
                if audit_prefix and (begin==0 or end==len(prefix)):
                    box={}
                    def replay():box['output']=model.forward_chunk(prefix[begin:end],state)
                    trace=capture(replay);_,state=box['output']
                    replay_traces['first' if begin==0 else 'last']=dict(tokens=end-begin,trace=trace)
                else:_,state=model.forward_chunk(prefix[begin:end],state)
            replay_tokens=len(prefix)
            if hasattr(state,'history'):
                state.history=[];state.previous_address=None
            state.detach()
        before=activity(state,model);torch.manual_seed(314159);total=0.
        for begin in range(0,len(tokens)-1,chunk):
            end=min(begin+chunk,len(tokens)-1)
            if audit_prefix and begin==min(256,((len(tokens)-2)//chunk)*chunk):
                box={}
                def sample():
                    z,box['state']=model.forward_chunk(tokens[begin:end],state)
                    box['loss']=F.cross_entropy(z,tokens[begin+1:end+1],reduction='sum')
                infer=dict(tokens=end-begin,trace=capture(sample));state=box['state'];loss=box['loss']
            else:
                z,state=model.forward_chunk(tokens[begin:end],state)
                loss=F.cross_entropy(z,tokens[begin+1:end+1],reduction='sum')
            total+=float(loss)
        after=activity(state,model)
        result=dict(n=len(tokens)-1,bpc=total/(len(tokens)-1)/math.log(2),
            activity={k:after[k]-before[k] for k in after if k!='receiver_persistent_tensor_bytes'},
            persistent_state=state.storage(),replay_tokens=replay_tokens,
            boundary='retain core state/time; clear hash history/previous-address, no invented fit/dev outcome write',
            policy='cold' if prefix is None else 'frozen_fit_replay')
        if audit_prefix:
            first=replay_traces['first'];last=replay_traces.get('last',first)
            units=lambda r:r['arithmetic_flops']+r['special_function_evaluations']
            estimate=units(first['trace'])+(replay_tokens-first['tokens'])*units(last['trace'])/last['tokens']
            result['replay_work']=dict(cpu_unit_special_flops_estimate=estimate,
                amortized_cpu_unit_special_flops_per_dev_target=estimate/(len(tokens)-1),
                traces=replay_traces,inference_sample=infer,
                scope='First/last replay chunks audited; rest estimated at last-chunk rate, no loss/backward/optimizer. '
                      'Additional frozen replay work, excluded from fitting totals; hash/RNG/traffic separate.')
        return result


def atomic_json(path,value):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def merge(records):
    combined=dict(arithmetic_flops=0,special_function_evaluations=0,
        formula_coverage_complete=True,unsupported_floating_operators={},operators={},exponential_random_draws=0)
    for record in records:
        for k in ('arithmetic_flops','special_function_evaluations','exponential_random_draws'):
            combined[k]+=record[k]
        combined['formula_coverage_complete'] &= record['formula_coverage_complete']
        combined['unsupported_floating_operators'].update(record['unsupported_floating_operators'])
        for name,row in record['operators'].items():
            dest=combined['operators'].setdefault(name,{k:(v if isinstance(v,str) else 0) for k,v in row.items()})
            for k,v in row.items():
                if not isinstance(v,str):dest[k]+=v
    return combined


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',required=True);p.add_argument('--heads',type=int,choices=(2,4),default=2)
    p.add_argument('--model',choices=('native','addressed','late'),required=True)
    p.add_argument('--order',type=int,default=3);p.add_argument('--buckets',type=int,default=4096)
    p.add_argument('--stop-after-updates',type=int,default=0,help='Recovery-contract interruption at an optimizer boundary')
    p.add_argument('--fit',type=int,default=2048);p.add_argument('--dev',type=int,default=8192)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--chunk',type=int,default=16)
    p.add_argument('--update-targets',type=int,choices=(16,64,128,256),default=64)
    p.add_argument('--warmup-targets',type=int,default=512)
    p.add_argument('--payload',type=int,default=16);p.add_argument('--depth',type=int,default=8)
    p.add_argument('--pool',type=int,default=2);p.add_argument('--matching',type=int,choices=(0,),default=0)
    p.add_argument('--recent',type=int,choices=(0,),default=0);p.add_argument('--seed',type=int,default=6)
    p.add_argument('--lr',type=float,default=.002);p.add_argument('--contracts-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.memory='receiver';a.candidate_index='observed_stream_address';a.cache_storage='persistent';a.native_event_core=True
    directory=ROOT/'experiments/results/addressed_memory_language';directory.mkdir(parents=True,exist_ok=True)
    out=directory/(a.tag+'.json');running=out.with_suffix('.running.json');checkpoint=out.with_suffix('.progress.pt')
    if Path(a.tag).name!=a.tag or out.exists() or (running.exists() and not a.resume):
        raise ValueError('Use an unused tag, or exact-source checkpoint recovery')
    if not 2*a.chunk+1<=a.fit<=131072 or not a.chunk+2<=a.dev<=8192 or a.epochs<1:
        raise ValueError('Bounded disjoint development ladder only')
    if a.update_targets%a.chunk or a.warmup_targets<0:
        raise ValueError('Optimizer interval must be a multiple of credit length')
    torch.set_num_threads(1);torch.manual_seed(a.seed);started=time.perf_counter()
    checks={**baseline.contracts(a.payload,a.depth,a.pool,a.matching,a.recent,a.heads),
            **accumulation_contracts(a.payload,a.depth,a.pool,a.matching,a.recent,a.heads)}
    provenance=sources()
    result=dict(status='running',args=vars(a),source_sha256=provenance,numerical_contracts=checks)
    if a.contracts_only:
        result.update(status='completed',wall_s=time.perf_counter()-started)
        out.write_text(json.dumps(result,indent=2)+'\n');return
    train=torch.tensor(text_slice(0,a.fit));dev=torch.tensor(text_slice(90_000_000,a.dev))
    model=make_model(a)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    learner=GradientAccumulator(model,opt,a.lr,a.warmup_targets)
    torch.manual_seed(a.seed+100000)
    initial=copy.deepcopy(model.state_dict());best=float('inf');best_state=None;traces={};previous_wall=0.
    cursor=dict(epoch=1,next_target=0,total_loss=0.);event=None
    if a.resume:
        saved=torch.load(checkpoint,weights_only=False)
        result=saved['result']
        if result['source_sha256']!=provenance:raise ValueError('Recover exact source revision')
        expected={k:v for k,v in result['args'].items() if k not in ('resume','stop_after_updates')}
        actual={k:v for k,v in vars(a).items() if k not in ('resume','stop_after_updates')}
        if expected!=actual:raise ValueError('Recovery settings changed')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        cursor=saved['cursor'];event=saved['stream_state'];traces=saved['traces']
        best_state=saved['best_state'];best=min((r['dev']['bpc'] for r in result['curve']),default=float('inf'))
        learner.total_targets=saved['total_targets'];learner.updates=saved['updates']
        initial=saved['initial']
        torch.set_rng_state(saved['torch_rng']);previous_wall=result['wall_s']
    else:
        result.update(parameters=sum(p.numel() for p in model.parameters()),initial_dev=evaluate(model,dev,a.chunk),curve=[],
            protocol=dict(fitting=[0,a.fit],development=[90_000_000,90_000_000+a.dev],
                official_test_read=False,test=None,cold_context=True,credit_truncation=a.chunk,
                optimizer_update_targets=a.update_targets,warmup_targets=a.warmup_targets,
                tokenizer='27-character text8',selection='minimum frozen full-development bpc over fixed passes',
                heads=a.heads,per_head_payload=a.payload,total_payload=a.heads*a.payload,
                head_channels='independent receiver query/key/content/gate maps per head; separate timestamped channels; learned next-block projection',
                retained_cache='native receiver/context state, plus fixed hashed context sums for addressed variants; no KV race attention or learned pooling',
                learning='per-credit-chunk backward and detach; target-weighted accumulated gradients; once-per-window normalization, clipping and Adam; all admitted counterfactual values charged',
                timing='native per-source causal read; max head arrival; analytic stored-channel evolution; one unit of time per observed token',
                training_rng_reset=a.seed+100000,
                warm_evaluation_selection=False,
                scope='Exploratory same-seed native/addressed language diagnostic; same conditional data/passes/update budgets; reused dev, no semantic proof; CPU emulator, not hardware joules'),
            fitting_data_sha256=hashlib.sha256(train.numpy().astype('uint8').tobytes()).hexdigest(),
            development_data_sha256=hashlib.sha256(dev.numpy().astype('uint8').tobytes()).hexdigest(),
            hardware=dict(host=__import__('os').uname().nodename,device='cpu',threads=1,platform=platform.platform(),torch=torch.__version__))
    def persist():
        assert learner.pending_targets==0,'Checkpoint only after a completed optimizer window'
        result.update(wall_s=previous_wall+time.perf_counter()-started,
            max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        target=out if result['status']=='completed' else running
        temporary=checkpoint.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),result=result,
            best_state=best_state,stream_state=event,cursor=cursor,traces=traces,
            initial=initial,
            total_targets=learner.total_targets,updates=learner.updates,torch_rng=torch.get_rng_state()),temporary)
        temporary.replace(checkpoint)
        atomic_json(target,result)
    persist()
    full,remainder=divmod(a.fit-1,a.update_targets)
    for epoch in range(cursor['epoch'],a.epochs+1):
        start=cursor['next_target'] if epoch==cursor['epoch'] else 0
        total=cursor['total_loss'] if start else 0.
        event=event if start else model.new_state()
        for window_start in range(start,a.fit-1,a.update_targets):
            window_end=min(window_start+a.update_targets,a.fit-1)
            number=window_start//a.update_targets+1
            name=('first' if number==1 else 'partial' if window_end-window_start<a.update_targets else
                  'warm' if number==max(2,full//2) else None) if epoch==1 else None
            before=activity(event,model);forwards=[];backwards=[]
            for micro_start in range(window_start,window_end,a.chunk):
                micro_end=min(micro_start+a.chunk,window_end)
                if name:
                    model.train()
                    if learner.pending_targets==0:opt.zero_grad(set_to_none=True)
                    box={}
                    def forward():
                        z,box['state']=model.forward_chunk(train[micro_start:micro_end],event)
                        box['loss']=F.cross_entropy(z,train[micro_start+1:micro_end+1],reduction='sum')
                    forwards.append(capture(forward))
                    if not torch.isfinite(box['loss']):raise FloatingPointError('Nonfinite loss')
                    backwards.append(capture(lambda:box['loss'].backward()))
                    learner.pending_targets+=micro_end-micro_start;learner.total_targets+=micro_end-micro_start
                    total+=float(box['loss'].detach());event=box['state'].detach()
                else:
                    loss,event,_=learner.accumulate(train[micro_start:micro_end],train[micro_start+1:micro_end+1],event)
                    total+=loss
            if name:
                trace=dict(forward_and_loss=merge(forwards),backward=merge(backwards),
                    gradient_normalization=capture(learner.normalize),
                    gradient_clipping=capture(lambda:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)),
                    optimizer=capture(learner.step))
                after=activity(event,model);delta={k:after[k]-before[k] for k in after}
                trace['activity']=delta;trace['architecture_forward_and_loss']=baseline.architecture_forward(trace['forward_and_loss'],delta)
                traces[name]=trace
            else:learner.update()
            cursor=dict(epoch=epoch,next_target=window_end,total_loss=total)
            if number%8==0 or (a.stop_after_updates and learner.updates==a.stop_after_updates):
                result['progress']=dict(epoch=epoch,targets=window_end,online_bpc=total/window_end/math.log(2),optimizer_updates=learner.updates)
                persist();print(json.dumps(result['progress']),flush=True)
            if a.stop_after_updates and learner.updates==a.stop_after_updates:
                print(json.dumps(dict(checkpointed=a.tag,updates=learner.updates)),flush=True);return
        score=evaluate(model,dev,a.chunk)
        result['curve'].append(dict(epoch=epoch,dev=score,online_bpc=total/(a.fit-1)/math.log(2),
            activity=activity(event,model),state_storage=event.packed_storage(),
            optimizer_steps=math.ceil((a.fit-1)/a.update_targets)))
        if score['bpc']<best:best=score['bpc'];best_state=copy.deepcopy(model.state_dict());result['selected_epoch']=epoch
        event=None;cursor=dict(epoch=epoch+1,next_target=0,total_loss=0.);persist()
        print(json.dumps(dict(epoch=epoch,dev_bpc=score['bpc'])),flush=True)
    model.load_state_dict(best_state);result['final']=dict(dev=evaluate(model,dev,a.chunk),
        frozen_fit_replay_dev=evaluate(model,dev,a.chunk,prefix=train,audit_prefix=True))
    repetitions=dict(first=a.epochs,warm=max(0,full-1)*a.epochs)
    if remainder:repetitions['partial']=a.epochs
    stages=('forward_and_loss','backward','gradient_normalization','gradient_clipping','optimizer')
    cpu={s:sum(repetitions[k]*t[s]['arithmetic_flops'] for k,t in traces.items()) for s in stages}
    special=sum(repetitions[k]*sum(t[s]['special_function_evaluations'] for s in stages) for k,t in traces.items())
    arch=dict(cpu);arch['forward_and_loss']=sum(repetitions[k]*t['architecture_forward_and_loss']['arithmetic_flops'] for k,t in traces.items())
    arch_special=special-sum(repetitions[k]*t['activity']['scored_clock_rates'] for k,t in traces.items())
    actual={k:sum(r['activity'][k] for r in result['curve']) for k in result['curve'][0]['activity']
            if k not in ('kv_winner_age_max','kv_stored_entries','kv_raw_key_value_bytes','receiver_persistent_tensor_bytes')}
    model.eval();event=model.new_state()
    with torch.no_grad():
        warm=min(256,a.dev-1-a.chunk);_,event=model.forward_chunk(dev[:warm],event)
        before=baseline.activity(event,model)
        def inference():
            z,_=model.forward_chunk(dev[warm:warm+a.chunk],event)
            F.cross_entropy(z,dev[warm+1:warm+a.chunk+1],reduction='sum')
        scoring=capture(inference);after=baseline.activity(event,model)
    delta={k:after[k]-before[k] for k in after};projected=baseline.architecture_forward(scoring,delta)
    result['work']=dict(fitting_targets=(a.fit-1)*a.epochs,optimizer_steps=learner.updates,
        cpu_emulator=dict(training_stages=cpu,total_training_arithmetic_flops=sum(cpu.values()),
            training_special_function_evaluations=special,total_training_unit_special_flops=sum(cpu.values())+special,
            inference_arithmetic_flops_per_character=scoring['arithmetic_flops']/a.chunk,
            inference_special_functions_per_character=scoring['special_function_evaluations']/a.chunk),
        projected_event_architecture=dict(training_stages=arch,total_training_arithmetic_flops=sum(arch.values()),
            training_special_function_evaluations=arch_special,total_training_unit_special_flops=sum(arch.values())+arch_special,
            inference_arithmetic_flops_per_character=projected['arithmetic_flops']/a.chunk,
            inference_special_functions_per_character=projected['special_function_evaluations']/a.chunk,
            physical_clock_rate_settings=actual['scored_clock_rates'],physical_races=actual['race_count'],
            assumptions='Only explicit numeric race-clock simulation removed; all head projections, state evolution, counterfactual teachers, gradient accumulation/normalization, clipping and Adam charged; index/RNG/traffic separate'),
        actual_fitting_activity=actual,traces=traces,repetitions=repetitions,inference_trace=scoring,inference_activity=delta,
        estimate_convention='Observed first/mature/partial epoch1 optimizer windows extrapolated over matching window/pass counts; variable retrieval occupancy/graph reach are not fully enumerated',
        scope='Whole fitting FLOPs ESTIMATE from representative audited windows, including admitted losing-value credit and Adam; credit graphs freed every microchunk. Two FLOPs/MAC, specials separate plus unit-weight total. Excludes dev passes/RNG/hash/traffic; replay charged separately; not measured joules.')
    result['parameter_delta']={n:float((v-initial[n]).norm()) for n,v in model.state_dict().items() if v.is_floating_point()}
    result['status']='completed';persist();print(json.dumps(dict(completed=a.tag,bpc=result['final']['dev']['bpc'])),flush=True)


if __name__=='__main__':main()
