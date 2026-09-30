"""Matched guarded development screen; identical index, softmax versus races.

Jointly train the unchanged carrier and small contextual retrieval module from
scratch. Reuse the completed carrier-only screen. No official test is read.
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
from e120_shared_tasks import text_slice
from sleeping_machines.race_language import RaceLanguageModel
from sleeping_machines.operation_audit import OperationAudit
import parallel_event_language as accounting
from race_language_contracts import hashes


class RaceAudit(OperationAudit):
    def formula(self,func,args,kwargs,out):
        operation=str(func).split('.')[1].rstrip('_')
        if operation=='exponential':
            return 0,0,0,'RNG generation (additional unquantified cost)'
        if operation=='mv':
            return 2*args[0].numel(),0,0,'matrix-vector contraction (2 FLOPs/MAC)'
        if operation=='dot':
            return 2*args[0].numel()-1,0,0,'vector dot product'
        if operation=='ger':
            return out.numel(),0,0,'outer-product multiplication'
        return super().formula(func,args,kwargs,out)


def capture(action):
    with RaceAudit() as audit:action()
    result=audit.result()
    if not result['formula_coverage_complete']:raise ValueError(result['unsupported_floating_operators'])
    result['exponential_random_draws']=sum(r['floating_output_elements'] for n,r in result['operators'].items()
                                         if r['classification'].startswith('RNG generation'))
    return result


@torch.no_grad()
def evaluate(model,tokens,chunk):
    # Same fixed random stream for every epoch's frozen development evaluation.
    # fork_rng ensures evaluation never changes the subsequent fitting stream.
    with torch.random.fork_rng():
        torch.manual_seed(314159)
        model.eval();state=model.new_state();loss=0.
        for start in range(0,len(tokens)-1,chunk):
            end=min(start+chunk,len(tokens)-1)
            logits,state=model.forward_chunk(tokens[start:end],state)
            loss+=float(F.cross_entropy(logits,tokens[start+1:end+1],reduction='sum'))
        return dict(n=len(tokens)-1,bpc=loss/(len(tokens)-1)/math.log(2),
            retrieval_queries=state.retrieval_queries,scored_keys=state.scored_keys,
            delivered_values=state.delivered_values,event_deliveries=state.deliveries)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True);parser.add_argument('--contracts',required=True)
    parser.add_argument('--attention',choices=('race','softmax'),required=True)
    parser.add_argument('--fit',type=int,default=131072);parser.add_argument('--dev',type=int,default=8192)
    parser.add_argument('--epochs',type=int,default=4);parser.add_argument('--chunk',type=int,default=64)
    parser.add_argument('--width',type=int,default=128);parser.add_argument('--depth',type=int,default=6)
    parser.add_argument('--payload',type=int,default=16);parser.add_argument('--capacity',type=int,default=8)
    parser.add_argument('--races',type=int,default=4);parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=6)
    args=parser.parse_args();out=ROOT/'experiments/results/parallel_language'/f'{args.tag}.json'
    if Path(args.tag).name!=args.tag or out.exists() or out.with_suffix('.running.json').exists():
        raise ValueError('Preserve prior results; use a fresh unique tag')
    if min(args.fit,args.dev,args.epochs,args.chunk,args.width,args.depth,args.payload,args.capacity,args.races)<1:
        raise ValueError('Positive dimensions required')
    if args.fit>1048576 or args.dev>200000 or args.fit<args.chunk*2+1:
        raise ValueError('This development driver is capped; no official-test mode')
    sources=hashes();contract=json.loads((ROOT/args.contracts).read_text())
    if contract['status']!='completed' or contract['source_sha256']!=sources:
        raise ValueError('Completed contracts must match every current source')
    torch.set_num_threads(1);torch.manual_seed(args.seed);started=time.perf_counter()
    train=torch.tensor(text_slice(0,args.fit));dev=torch.tensor(text_slice(90000000,args.dev))
    model=RaceLanguageModel(width=args.width,modes=args.width//2,depth=args.depth,
        payload=args.payload,capacity=args.capacity,races=args.races,attention=args.attention)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr);initial=copy.deepcopy(model.state_dict())
    result=dict(status='running',args=vars(args),parameters=sum(p.numel() for p in model.parameters()),
        initial_dev=evaluate(model,dev,args.chunk),curve=[],source_sha256=sources,
        contract_result=args.contracts,contract_sha256=hashlib.sha256((ROOT/args.contracts).read_bytes()).hexdigest(),
        fitting_data_sha256=hashlib.sha256(train.numpy().astype('uint8').tobytes()).hexdigest(),
        protocol=dict(fitting=[0,args.fit],development=[90000000,90000000+args.dev],
            official_test_read=False,test=None,statistical_experts=[],cold_context=True,
            excluded_first_target=True,credit_truncation=args.chunk,
            selection=f'minimum full-development bpc over {args.epochs} declared passes',
            index='last observed character; bounded recent prefix/successor keys per bucket',
            key_value_separation=True,training='joint from scratch; local conserved race-score surrogate',
            caveat='Fixed-index sparse retrieval; dense carrier still executes every layer; no learned routing or hardware energy claim'),
        hardware=dict(device='cpu',threads=1,platform=platform.platform(),torch=torch.__version__))
    best=float('inf');best_state=None
    def persist(completed=False,state=None,epoch=None,start=None):
        result.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        target=out if completed else out.with_suffix('.running.json')
        temporary=target.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(target)
        checkpoint=out.with_suffix('.progress.pt');temporary=checkpoint.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),result=result,
            stream_state=None if state is None else state.detach(),epoch=epoch,start=start,
            torch_rng=torch.get_rng_state(),best_state=best_state),temporary);temporary.replace(checkpoint)
    persist();print(json.dumps(dict(started=args.tag,initial_dev=result['initial_dev'])),flush=True)
    for epoch in range(1,args.epochs+1):
        state=model.new_state();model.train();total=0.
        for step,start in enumerate(range(0,len(train)-1,args.chunk),1):
            end=min(start+args.chunk,len(train)-1);opt.zero_grad(set_to_none=True)
            logits,state=model.forward_chunk(train[start:end],state)
            loss=F.cross_entropy(logits,train[start+1:end+1])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fit')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();total+=float(loss.detach())*(end-start);state=state.detach()
            if step%256==0:
                result['progress']=dict(epoch=epoch,targets=end,steps=step,online_bpc=total/end/math.log(2))
                persist(state=state,epoch=epoch,start=end);print(json.dumps(result['progress']),flush=True)
        score=evaluate(model,dev,args.chunk)
        result['curve'].append(dict(epoch=epoch,dev=score,online_bpc=total/(len(train)-1)/math.log(2),
            teaching_value_visits=state.teaching_value_visits,scored_keys=state.scored_keys,
            delivered_values=state.delivered_values,optimizer_steps=step))
        if score['bpc']<best:best=score['bpc'];best_state=copy.deepcopy(model.state_dict());result['selected_epoch']=epoch
        persist();print(json.dumps(result['curve'][-1]),flush=True)
    model.load_state_dict(best_state);result['final']=dict(dev=evaluate(model,dev,args.chunk))
    result['retrieval_parameter_change_norms']={n:float((p.detach()-initial[n]).norm())
        for n,p in model.named_parameters() if not n.startswith('core.')}
    # Reuse the unchanged ledger with a local RNG-aware audit. The warmed trace
    # is only representative: candidate occupancy increases with the stream.
    accounting.capture=capture
    result['work']=accounting.fitting_work(model,opt,train,args)
    result['work']['scope']+=' Retrieval trace is representative, not an occupancy-exact whole-run count. RNG draw counts and simulator value stacking are additional.'
    result['saturated_index_work_scenario']=accounting.fitting_work(model,opt,torch.ones_like(train),args)
    result['saturated_index_work_scenario']['scope']='Synthetic repeated-character saturation scenario, same step budget and trained parameters; not actual fitting data or a certified upper bound.'
    result['status']='completed';persist(completed=True);out.with_suffix('.running.json').unlink(missing_ok=True)
    print(json.dumps(dict(completed=args.tag,final=result['final'],work=result['work']['total_training_arithmetic_flops'])),flush=True)


if __name__=='__main__':main()
