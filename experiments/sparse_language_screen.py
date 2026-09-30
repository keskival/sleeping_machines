"""Guarded development fitting of the full sparse temporal language candidate."""
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
from sleeping_machines.sparse_race_language import SparseRaceLanguageModel
from sparse_language_contracts import hashes
from race_language_screen import capture
import parallel_event_language as accounting


@torch.no_grad()
def evaluate(model,tokens,chunk):
    with torch.random.fork_rng():
        torch.manual_seed(314159);model.eval();s=model.new_state();total=0.
        for start in range(0,len(tokens)-1,chunk):
            end=min(start+chunk,len(tokens)-1);z,s=model.forward_chunk(tokens[start:end],s)
            total+=float(F.cross_entropy(z,tokens[start+1:end+1],reduction='sum'))
        return dict(n=len(tokens)-1,bpc=total/(len(tokens)-1)/math.log(2),
            event_deliveries=s.deliveries,candidate_scores=s.candidate_scores,
            selected_state_updates=s.selected_updates,distinct_units_visited=len(s.visited_units),
            capacity_units=model.capacity_units,active_units_per_token=model.depth,
            training_counterfactual_values=s.counterfactual_values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',required=True);p.add_argument('--contracts',required=True)
    p.add_argument('--fit',type=int,default=8192);p.add_argument('--dev',type=int,default=8192)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--chunk',type=int,default=16)
    p.add_argument('--payload',type=int,default=16);p.add_argument('--depth',type=int,default=6)
    p.add_argument('--pool',type=int,default=2);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=6);a=p.parse_args()
    out=ROOT/'experiments/results/parallel_language'/f'{a.tag}.json'
    if Path(a.tag).name!=a.tag or out.exists() or out.with_suffix('.running.json').exists():raise ValueError('Unique unused tag required')
    if min(a.fit,a.dev,a.epochs,a.chunk,a.payload,a.depth,a.pool)<1 or a.fit>1048576 or a.dev>200000:
        raise ValueError('Positive, capped development settings required')
    if a.fit<2*a.chunk+1:raise ValueError('At least two credit chunks required')
    sources=hashes();contract=json.loads((ROOT/a.contracts).read_text())
    if contract['status']!='completed' or contract['source_sha256']!=sources:raise ValueError('Source-matching completed contracts required')
    torch.set_num_threads(1);torch.manual_seed(a.seed);started=time.perf_counter()
    train=torch.tensor(text_slice(0,a.fit));dev=torch.tensor(text_slice(90000000,a.dev))
    model=SparseRaceLanguageModel(a.payload,a.depth,a.pool)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr);initial=copy.deepcopy(model.state_dict())
    result=dict(status='running',args=vars(a),parameters=sum(x.numel() for x in model.parameters()),
        capacity_units=model.capacity_units,initial_dev=evaluate(model,dev,a.chunk),curve=[],
        source_sha256=sources,contract_result=a.contracts,
        fitting_data_sha256=hashlib.sha256(train.numpy().astype('uint8').tobytes()).hexdigest(),
        hardware=dict(device='cpu',threads=1,platform=platform.platform(),torch=torch.__version__),
        protocol=dict(fitting=[0,a.fit],development=[90000000,90000000+a.dev],test=None,
            official_test_read=False,cold_context=True,excluded_first_target=True,statistical_experts=[],
            index='observed-character pools; learned contextual state keys and race clocks within each pool',
            credit_truncation=a.chunk,selection=f'minimum frozen full-development bpc over {a.epochs} passes',
            forward='one selected persistent unit per depth, bounded temporal race delays',
            learning='all addressed counterfactual values; conserved local surrogate plus interior timing credit',
            scope='Integrated sparse/timed mechanism candidate; fixed pool topology, no full topology growth or measured hardware energy'))
    best=float('inf');best_state=None
    def save(completed=False,state=None,epoch=None,next_start=None):
        result.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        target=out if completed else out.with_suffix('.running.json');temporary=target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(target)
        checkpoint=out.with_suffix('.progress.pt');temporary=checkpoint.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),result=result,
            stream_state=None if state is None else state.detach(),epoch=epoch,next_start=next_start,
            best_state=best_state,torch_rng=torch.get_rng_state()),temporary);temporary.replace(checkpoint)
    save();print(json.dumps(dict(started=a.tag,initial_dev=result['initial_dev'],parameters=result['parameters'])),flush=True)
    for epoch in range(1,a.epochs+1):
        model.train();s=model.new_state();total=0.
        for step,start in enumerate(range(0,len(train)-1,a.chunk),1):
            end=min(start+a.chunk,len(train)-1);opt.zero_grad(set_to_none=True)
            z,s=model.forward_chunk(train[start:end],s);loss=F.cross_entropy(z,train[start+1:end+1])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fitting loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();total+=float(loss.detach())*(end-start);s=s.detach()
            if step%64==0:
                result['progress']=dict(epoch=epoch,targets=end,online_bpc=total/end/math.log(2))
                save(state=s,epoch=epoch,next_start=end);print(json.dumps(result['progress']),flush=True)
        score=evaluate(model,dev,a.chunk)
        result['curve'].append(dict(epoch=epoch,dev=score,online_bpc=total/(len(train)-1)/math.log(2),
            deliveries=s.deliveries,candidate_scores=s.candidate_scores,
            counterfactual_values=s.counterfactual_values,selected_updates=s.selected_updates,
            distinct_units_visited=len(s.visited_units),optimizer_steps=step))
        if score['bpc']<best:best=score['bpc'];best_state=copy.deepcopy(model.state_dict());result['selected_epoch']=epoch
        save();print(json.dumps(result['curve'][-1]),flush=True)
    model.load_state_dict(best_state);result['final']=dict(dev=evaluate(model,dev,a.chunk))
    result['parameter_change_norms_by_category']={}
    for n,x in model.named_parameters():
        category='unit' if n.startswith('units') else n.split('.')[0]
        result['parameter_change_norms_by_category'][category]=result['parameter_change_norms_by_category'].get(category,0.)+float((x.detach()-initial[n]).norm())
    # Different sparse addresses/credit histories change optimizer activity.
    # This representative extrapolation is explicitly not a whole-run trace.
    accounting.capture=capture;result['work']=accounting.fitting_work(model,opt,train,a)
    result['work']['scope']+=' Sparse receiver/optimizer activity depends on content and credit history; representative extrapolation, not exact whole-run accounting. RNG, indexing and emulator traffic remain additional.'
    result['training_random_draws']=sum(r['candidate_scores'] for r in result['curve'])
    result['status']='completed';save(completed=True);out.with_suffix('.running.json').unlink(missing_ok=True)
    print(json.dumps(dict(completed=a.tag,dev=result['final']['dev'],work=result['work']['total_training_arithmetic_flops'])),flush=True)


if __name__=='__main__':main()
