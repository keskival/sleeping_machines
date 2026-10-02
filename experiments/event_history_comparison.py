"""Guarded fresh-data evaluation of saved native models and learned history controls."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import torch
from torch.nn import functional as F
from experiments.native_event_tasks import episodes,data_hash
from experiments.rule_seed_event_benchmark import evaluate as native_evaluate
from experiments.race_language_screen import capture
from experiments.parallel_head_accumulated_language import merge
from sleeping_machines.event_history_control import EventHistoryControl
from sleeping_machines.rule_seed_event_heads import RuleSeedEventHeads


def evaluate(model,rows):
    model.eval();loss=0.;correct=0;total=0;scores=[]
    with torch.no_grad():
        for row in rows:
            z,y,_=model.predict_population(row);loss+=float(F.cross_entropy(z,y,reduction='sum'))
            hits=int((z.argmax(-1)==y).sum());correct+=hits;total+=len(y);scores.append(hits/len(y))
    return dict(n=total,nll=loss/total,accuracy=correct/total,correct=correct,episode_accuracy=scores)


def recovery_contract(model):
    row=episodes('order',64,64,931)[0];opt=torch.optim.Adam(model.parameters(),lr=.003)
    def step(net,optimizer):
        net.train();optimizer.zero_grad(set_to_none=True)
        z,y,_=net.predict_population(row);F.cross_entropy(z,y).backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in net.parameters())
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    step(model,opt);buffer=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),buffer)
    buffer.seek(0);saved=torch.load(buffer,weights_only=False);other=copy.deepcopy(model)
    other.load_state_dict(saved['model']);second=torch.optim.Adam(other.parameters(),lr=.003);second.load_state_dict(saved['optimizer'])
    step(model,opt);step(other,second)
    for x,y in zip(model.parameters(),other.parameters()):assert torch.equal(x,y)
    return dict(exact_next_optimizer_update=True,all_decoder_gradients_finite=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--model',choices=('history32','history128','native_private','native_shared'),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--parent')
    p.add_argument('--contracts-only',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();out=ROOT/'experiments/results/event_history_comparison'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique result tag required')
    torch.set_num_threads(1);torch.manual_seed(a.seed);started=time.perf_counter()
    names=('experiments/event_history_comparison.py','sleeping_machines/event_history_control.py',
        'experiments/native_event_tasks.py','experiments/rule_seed_event_benchmark.py',
        'experiments/race_language_screen.py','experiments/parallel_head_accumulated_language.py',
        'sleeping_machines/operation_audit.py')
    provenance={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    result=dict(status='running',args=vars(a),source_sha256=provenance)
    fit=episodes('order',64,64 if a.smoke else 512,1201)
    dev=episodes('order',64,64 if a.smoke else 1024,2201)
    if a.model.startswith('native'):
        parent=ROOT/a.parent;public=json.loads(parent.read_text());saved=torch.load(parent.with_suffix('.progress.pt'),weights_only=False)
        for key in ('args','source_sha256','data_sha256','parameters','selected_epoch','work'):
            if saved['result'][key]!=public[key]:raise ValueError('Checkpoint lineage mismatch: '+key)
        settings=public['args']
        if public['status']!='completed' or (settings['seed'],settings['sources'],settings['fit_targets'],settings['dev_targets'],settings['epochs'],settings['common_source_seed'],settings['shared_maps'])!=(a.seed,64,512,1024,4,1,a.model=='native_shared'):
            raise ValueError('Matched completed native parent required')
        for name,sha in public['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen parent source changed')
        if public['data_sha256']!={'fit':data_hash(fit),'dev':data_hash(dev)}:raise ValueError('Parent data differ')
        model=RuleSeedEventHeads(sources=64,payload=8,depth=8,heads=2,pool=2,shared_maps=settings['shared_maps'],common_source_seed=1)
        model.load_state_dict(saved['best_state']);model.eval()
        check=native_evaluate(model,dev)
        for key in ('accuracy','nll'):
            if abs(check[key]-public['final']['dev'][key])>1e-7:raise ValueError('Selected dev score differs')
        result.update(final=dict(dev=check),work=public['work'],parameters=public['parameters'],
            parent_result_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),historical_fitting_wall_s=public['wall_s'],
            numerical_contracts=dict(selected_dev_score_reproduced=True),extra_optimizer_steps=0)
        if not a.contracts_only:result['final']['confirmation']=native_evaluate(model,episodes('order',64,4096,4201))
    else:
        width=int(a.model.removeprefix('history'));model=EventHistoryControl(width)
        with torch.random.fork_rng():checks=recovery_contract(copy.deepcopy(model))
        result['numerical_contracts']=checks
        result['prerequisite_optimizer_steps']=3
        if not a.contracts_only:
            optimizer=torch.optim.Adam(model.parameters(),lr=.003);curve=[];best=float('inf');best_state=None
            ledger={k:[] for k in ('forward_and_loss','backward','normalize_clip_optimizer')};rng=np.random.default_rng(a.seed)
            for epoch in range(1,2 if a.smoke else 5):
                model.train()
                for index in rng.permutation(len(fit)):
                    row=fit[index]
                    optimizer.zero_grad(set_to_none=True);box={}
                    def forward():
                        z,y,_=model.predict_population(row);box['loss']=F.cross_entropy(z,y,reduction='sum')
                    ledger['forward_and_loss'].append(capture(forward))
                    ledger['backward'].append(capture(lambda:box['loss'].backward()))
                    def update():
                        for parameter in model.parameters():parameter.grad.div_(64)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    ledger['normalize_clip_optimizer'].append(capture(update))
                score=evaluate(model,dev);curve.append(dict(epoch=epoch,dev=score))
                if score['nll']<best:best=score['nll'];best_state=copy.deepcopy(model.state_dict());selected=epoch
            model.load_state_dict(best_state)
            stages={k:merge(v) for k,v in ledger.items()};total=sum(v['arithmetic_flops'] for v in stages.values())
            special=sum(v['special_function_evaluations'] for v in stages.values());inference=capture(lambda:model.predict_population(dev[0]))
            targets=(64 if a.smoke else 512)*(1 if a.smoke else 4)
            result.update(parameters=sum(x.numel() for x in model.parameters()),curve=curve,selected_epoch=selected,
                final=dict(dev=evaluate(model,dev)),work=dict(fitting_query_targets=targets,total_training_arithmetic_flops=total,
                    training_special_function_evaluations=special,total_training_unit_special_flops=total+special,training_stages=stages,
                    inference_arithmetic_flops_per_query=inference['arithmetic_flops']/64,inference_trace=inference),
                extra_optimizer_steps=len(fit)*(1 if a.smoke else 4),persistent_history_tensor_bytes=64*3*2*4)
            if not a.smoke:result['final']['confirmation']=evaluate(model,episodes('order',64,4096,4201))
    result.update(status='completed',wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        protocol=dict(fit_sha256=data_hash(fit),dev_sha256=data_hash(dev),confirmation_seed=4201 if 'confirmation' in result.get('final',{}) else None,
            weights_selected_on_dev=True,scope='Diagnostic history control knows a three-mark history bound and shares decoder across observed source addresses. Explicit mark/age/valid features; no race or counterfactual mechanisms. Same512 fit queries/four passes/U64 and data as saved native; native historical fitting remains charged. Fresh4096-query confirmation, not a real-data or energy benchmark.'))
    # Refuse partially classified work before saving a successful result.
    def verify(value):
        if isinstance(value,dict):
            if value.get('formula_coverage_complete') is False:raise ValueError('Unsupported work operators')
            for child in value.values():verify(child)
        elif isinstance(value,list):
            for child in value:verify(child)
    verify(result);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(tag=a.tag,status='completed',confirmation=result.get('final',{}).get('confirmation',{}).get('accuracy'))))


if __name__=='__main__':main()
