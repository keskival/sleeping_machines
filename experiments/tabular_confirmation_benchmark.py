"""Frozen banknote confirmation and strong controls; run only through run_safe."""
import argparse
import copy
import hashlib
import json
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
from native_tabular_data import load
from experiments.tabular_confirmation_data import confirmation_rows,prediction_rows
from experiments.tabular_confirmation_controls import candidates
from native_tabular_model import NativeTabularModel,contracts
from clock_feature_language_helpers import source_hashes as core_hashes
from parallel_head_accumulated_language import merge
from race_language_screen import capture


def sources():
    names=('experiments/native_tabular_data.py','experiments/native_tabular_model.py',
           'experiments/native_tabular_benchmark.py','experiments/tabular_data_manifest.json',
           'experiments/tabular_confirmation_benchmark.py','experiments/tabular_confirmation_data.py',
           'experiments/tabular_confirmation_controls.py','experiments/tabular_confirmation_contracts.py',
           'experiments/tabular_confirmation_requirements.txt')
    return {**core_hashes(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def metrics(pred,y,classification):
    if classification:
        logp=pred-pred.max(1,keepdims=True);logp=logp-np.log(np.exp(logp).sum(1,keepdims=True))
        return dict(n=len(y),nll=float(-logp[np.arange(len(y)),y.astype(int)].mean()),
            accuracy=float((pred.argmax(1)==y).mean()))
    return dict(n=len(y),rmse=float(np.sqrt(np.mean((pred.reshape(-1)-y)**2))),
        mae=float(np.abs(pred.reshape(-1)-y).mean()))


@torch.no_grad()
def evaluate(model,x,y,classification,indices=None):
    with torch.random.fork_rng():
        torch.manual_seed(314159);model.eval();started=time.perf_counter();pred=[];peak=0
        for row in x:
            z,state=model(row);pred.append(z.numpy());peak=max(peak,state.storage()['persistent_tensor_bytes'])
        return dict(**metrics(np.stack(pred),y,classification),wall_s=time.perf_counter()-started,
                    persistent_tensor_bytes=peak,
                    rows=prediction_rows(np.stack(pred),y,list(range(len(y))) if indices is None else indices))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--dataset',choices=('banknote',),default='banknote')
    p.add_argument('--model',choices=('ours','trees','catboost','logistic'),default='ours')
    p.add_argument('--clock-features',type=int,choices=(0,2,4),default=0)
    p.add_argument('--clock-allocation',choices=('uniform','late'),default='uniform')
    p.add_argument('--payload',type=int,default=8);p.add_argument('--depth',type=int,default=8)
    p.add_argument('--heads',type=int,default=2);p.add_argument('--pool',type=int,default=2)
    p.add_argument('--fit',type=int,default=128);p.add_argument('--dev',type=int,default=128)
    p.add_argument('--epochs',type=int,choices=(1,4),default=4);p.add_argument('--seed',type=int,default=6)
    p.add_argument('--update-rows',type=int,default=16);p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--contracts-only',action='store_true')
    p.add_argument('--confirm',action='store_true')
    p.add_argument('--reuse-result',help='Completed matching first-screen native result; reuse its selected checkpoint')
    a=p.parse_args()
    directory=ROOT/'experiments/results/tabular_confirmation';directory.mkdir(parents=True,exist_ok=True)
    out=directory/(a.tag+'.json');running=out.with_suffix('.running.json')
    if Path(a.tag).name!=a.tag or out.exists() or running.exists():raise ValueError('Fresh tag required; preserve failed outputs')
    if not 1<=a.fit<=512 or not 1<=a.dev<=256 or not 1<=a.update_rows<=a.fit:raise ValueError('Small bounded screen required')
    if a.reuse_result and (a.model!='ours' or a.contracts_only):raise ValueError('Reuse is a native final-evaluation path only')
    if a.confirm and (a.fit!=128 or a.dev!=128 or a.epochs!=4 or a.clock_features!=0
            or (a.payload,a.depth,a.heads,a.pool,a.update_rows,a.lr)!=(8,8,2,2,16,.003)):
        raise ValueError('Confirmation requires the frozen native128-row four-pass R0 protocol')
    torch.set_num_threads(1);torch.manual_seed(a.seed);started=time.perf_counter()
    result=dict(status='running',args=vars(a),source_sha256=sources(),hardware=dict(
        platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,
        device='cpu',threads=1),curve=[])
    if a.contracts_only:
        from experiments.tabular_confirmation_data import contracts as data_contracts
        checks=data_contracts()
        if a.model=='ours':
            from experiments.tabular_confirmation_contracts import optimizer_recovery
            checks.update(contracts(clock_features=a.clock_features),**optimizer_recovery())
        else:
            controls=candidates(a.model,a.seed)
            assert len(controls)==4
            checks.update(four_fixed_control_candidates=True)
        result.update(status='completed',numerical_contracts=checks,
            wall_s=time.perf_counter()-started)
        out.write_text(json.dumps(result,indent=2)+'\n');return
    prep=time.perf_counter();data=load(a.dataset,a.fit,a.dev);prep=time.perf_counter()-prep
    result['protocol']=dict(**data['protocol'],preprocessing_wall_s=prep,
        input='Feature identity, train-scaled numeric value, missing flag; independent row reset',
        timestamps='Canonical feature-processing coordinates, not physical asynchronous observations',
        tuning='Exactly four development candidates/checkpoints for pilots; weights/settings frozen before reserved test',
        resource_boundary='Whole neural optimizer fit; preprocessing separately timed/counted; no hardware energy claim',
        feature_scaling_operations=dict(fit_elements=a.fit*len(data['feature_names']),
            transformed_fit_dev_elements=(a.fit+a.dev)*len(data['feature_names'])),
        scope='Prespecified banknote replication/confirmation on fixed grouped split; seed variability is separate from row uncertainty')
    x=torch.tensor(data['x_fit'],dtype=torch.float32);v=torch.tensor(data['x_dev'],dtype=torch.float32)
    y=data['y_fit'];d=data['y_dev'];classification=a.dataset=='banknote'
    score=lambda r:r['nll' if classification else 'rmse'];best=float('inf')
    def persist():
        result.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        path=out if result['status']=='completed' else running;tmp=path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    persist()
    if a.model!='ours':
        import sklearn
        from threadpoolctl import threadpool_limits
        result['hardware']['sklearn']=sklearn.__version__
        if a.model=='catboost':
            import catboost
            result['hardware']['catboost']=catboost.__version__
        total_fit=0.
        for candidate,model in candidates(a.model,a.seed):
            with threadpool_limits(limits=1):
                t=time.perf_counter();model.fit(x.numpy(),y);total_fit+=time.perf_counter()-t
                t=time.perf_counter();pred=np.log(np.maximum(model.predict_proba(v.numpy()),1e-30))
                m=metrics(pred,d,True);m['wall_s']=time.perf_counter()-t
                m['rows']=prediction_rows(pred,d,data['protocol']['dev_indices'])
            result['curve'].append(dict(candidate=candidate,dev=m))
            if score(m)<best:best=score(m);chosen=model;result['selected_candidate']=candidate;selected=m
            persist()
        result.update(final=dict(dev=selected),work=dict(control_fit_wall_s_all_candidates=total_fit,
            neural_flops=None,convention='All four independent control candidate fits charged; control FLOPs unavailable'))
        if a.model=='trees':
            nodes=[p.nodes for stage in chosen._predictors for p in stage]
            result['work'].update(selected_tree_nodes=sum(len(n) for n in nodes),
                selected_tree_node_bytes=sum(n.nbytes for n in nodes),selected_tree_count=len(nodes))
        if a.model=='catboost':result['work']['selected_tree_count']=chosen.tree_count_
    elif a.reuse_result:
        source=ROOT/a.reuse_result
        previous=json.loads(source.read_text())
        if previous['status']!='completed' or previous['args']['model']!='ours':
            raise ValueError('Completed native fitting result required')
        for name,sha in previous['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Reuse source changed: '+name)
        fields=('dataset','model','clock_features','clock_allocation','payload','depth','heads','pool','fit','dev','epochs','seed','update_rows','lr')
        if any(previous['args'][k]!=getattr(a,k) for k in fields):raise ValueError('Reuse settings differ')
        if any(previous['protocol'][k]!=data['protocol'][k] for k in ('fit_sha256','dev_sha256')):
            raise ValueError('Reuse data differ')
        checkpoint=source.with_suffix('.progress.pt')
        saved=torch.load(checkpoint,weights_only=False)
        model=NativeTabularModel(x.shape[1],2,a.payload,a.depth,a.heads,a.pool,a.clock_features,a.clock_allocation)
        model.load_state_dict(saved['best_state']);model.eval()
        check=evaluate(model,v,d,True,data['protocol']['dev_indices'])
        for field in ('accuracy','nll'):
            if abs(check[field]-previous['final']['dev'][field])>1e-7:raise ValueError('Reused frozen score differs')
        result.update(final=dict(dev=check),parameters=previous['parameters'],
            work=copy.deepcopy(previous['work']),activity=previous['activity'],
            selected_epoch=previous['selected_epoch'],curve=previous['curve'],
            reused_fit=dict(result=a.reuse_result,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),new_optimizer_steps=0))
    else:
        result['numerical_contracts']=contracts(clock_features=a.clock_features)
        model=NativeTabularModel(x.shape[1],2 if classification else 1,a.payload,a.depth,a.heads,a.pool,a.clock_features,a.clock_allocation)
        opt=torch.optim.Adam(model.parameters(),lr=a.lr);best_state=None
        result['parameters']=sum(p.numel() for p in model.parameters())
        stages={k:merge([]) for k in ('forward_and_loss','backward','normalization_clipping','optimizer')}
        activity=dict(events=0,selected_updates=0,candidate_scores=0,counterfactual_values=0,extra_clock_races=0,extra_rate_settings=0)
        def add(name,action):stages[name]=merge([stages[name],capture(action)])
        rng=np.random.default_rng(a.seed+10);fit_wall=0.
        for epoch in range(1,a.epochs+1):
            model.train();order=rng.permutation(a.fit);t=time.perf_counter()
            for begin in range(0,len(order),a.update_rows):
                opt.zero_grad(set_to_none=True);ids=order[begin:begin+a.update_rows]
                for i in ids:
                    box={}
                    def forward():
                        z,state=model(x[i]);box['state']=state
                        box['loss']=F.cross_entropy(z[None,:],torch.tensor([int(y[i])])) if classification else (z[0]-float(y[i])).square()
                    add('forward_and_loss',forward);add('backward',box['loss'].backward)
                    for name in activity:activity[name]+=getattr(box['state'],name)
                def normalize():
                    for param in model.parameters():
                        if param.grad is not None:param.grad.div_(len(ids))
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                add('normalization_clipping',normalize);add('optimizer',opt.step)
            fit_wall+=time.perf_counter()-t;m=evaluate(model,v,d,classification,data['protocol']['dev_indices'])
            result['curve'].append(dict(epoch=epoch,dev=m))
            if score(m)<best:best=score(m);best_state=copy.deepcopy(model.state_dict());result['selected_epoch']=epoch
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),best_state=best_state,
                            epoch=epoch,torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state),out.with_suffix('.progress.pt'))
            persist()
        model.load_state_dict(best_state);model.eval()
        with torch.random.fork_rng(),torch.no_grad():
            torch.manual_seed(314159);box={}
            def forward_inference():box['z'],box['state']=model(v[0])
            inference=capture(forward_inference)
        merged=merge(list(stages.values()));flops=merged['arithmetic_flops']+merged['special_function_evaluations']
        result.update(final=dict(dev=evaluate(model,v,d,classification,data['protocol']['dev_indices'])),activity=activity,
            work=dict(fitting_rows=a.fit*a.epochs,whole_neural_fit_unit_special_flops=flops,
                neural_fit_unit_special_flops_per_row=flops/(a.fit*a.epochs),
                inference_unit_special_flops_per_row=inference['arithmetic_flops']+inference['special_function_evaluations'],
                fit_wall_s=fit_wall,stages=stages,inference_trace=inference,
                scope='Actual full PyTorch forward/loss/backward/normalization/clipping/Adam trace; specials counted once; preprocessing, evaluation, contracts and RNG overhead separate'))
    result['protocol'].update(weights_frozen_before_test=True,independent_row_reset=True,
        confirmation_prespecified=a.confirm,test_labels_scored=False)
    if a.confirm:
        # All checkpoint/candidate selection is finished before this first test scoring.
        test=confirmation_rows(data['protocol']);t=time.perf_counter()
        if a.model=='ours':
            weights_before={k:v.detach().clone() for k,v in model.state_dict().items()}
            result['final']['test']=evaluate(model,torch.tensor(test['x'],dtype=torch.float32),test['y'],True,test['indices'])
            assert all(torch.equal(weights_before[k],v) for k,v in model.state_dict().items())
        else:
            with threadpool_limits(limits=1):pred=np.log(np.maximum(chosen.predict_proba(test['x'].astype(np.float32)),1e-30))
            result['final']['test']=dict(**metrics(pred,test['y'],True),rows=prediction_rows(pred,test['y'],test['indices']))
        result['protocol'].update(test_labels_scored=True,test_sha256=test['sha256'],
            test_scope='Reserved feature-group-disjoint rows; no checkpoint, hyperparameter or architecture selected on these labels')
        result['confirmation_wall_s']=time.perf_counter()-t
        result['final']['test']['rows']['groups']=test['groups']
    result['status']='completed';persist()


if __name__=='__main__':main()
