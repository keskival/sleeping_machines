"""Small supervised native-tabular/tree screen; run only through run_safe."""
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
from native_tabular_model import NativeTabularModel,contracts
from clock_feature_language_helpers import source_hashes as core_hashes
from parallel_head_accumulated_language import merge
from race_language_screen import capture


def sources():
    names=('experiments/native_tabular_data.py','experiments/native_tabular_model.py',
           'experiments/native_tabular_benchmark.py','experiments/tabular_data_manifest.json')
    return {**core_hashes(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def metrics(pred,y,classification):
    if classification:
        logp=pred-pred.max(1,keepdims=True);logp=logp-np.log(np.exp(logp).sum(1,keepdims=True))
        return dict(n=len(y),nll=float(-logp[np.arange(len(y)),y.astype(int)].mean()),
            accuracy=float((pred.argmax(1)==y).mean()))
    return dict(n=len(y),rmse=float(np.sqrt(np.mean((pred.reshape(-1)-y)**2))),
        mae=float(np.abs(pred.reshape(-1)-y).mean()))


@torch.no_grad()
def evaluate(model,x,y,classification):
    with torch.random.fork_rng():
        torch.manual_seed(314159);model.eval();started=time.perf_counter();pred=[];peak=0
        for row in x:
            z,state=model(row);pred.append(z.numpy());peak=max(peak,state.storage()['persistent_tensor_bytes'])
        return dict(**metrics(np.stack(pred),y,classification),wall_s=time.perf_counter()-started,
                    persistent_tensor_bytes=peak)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--dataset',choices=('banknote','wine_red'),default='banknote')
    p.add_argument('--model',choices=('ours','trees'),default='ours')
    p.add_argument('--clock-features',type=int,choices=(0,2,4),default=0)
    p.add_argument('--clock-allocation',choices=('uniform','late'),default='uniform')
    p.add_argument('--payload',type=int,default=8);p.add_argument('--depth',type=int,default=8)
    p.add_argument('--heads',type=int,default=2);p.add_argument('--pool',type=int,default=2)
    p.add_argument('--fit',type=int,default=128);p.add_argument('--dev',type=int,default=128)
    p.add_argument('--epochs',type=int,choices=(1,4),default=4);p.add_argument('--seed',type=int,default=6)
    p.add_argument('--update-rows',type=int,default=16);p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--contracts-only',action='store_true');a=p.parse_args()
    directory=ROOT/'experiments/results/native_tabular';directory.mkdir(parents=True,exist_ok=True)
    out=directory/(a.tag+'.json');running=out.with_suffix('.running.json')
    if Path(a.tag).name!=a.tag or out.exists() or running.exists():raise ValueError('Fresh tag required; preserve failed outputs')
    if not 1<=a.fit<=512 or not 1<=a.dev<=256 or not 1<=a.update_rows<=a.fit:raise ValueError('Small bounded screen required')
    torch.set_num_threads(1);torch.manual_seed(a.seed);started=time.perf_counter()
    result=dict(status='running',args=vars(a),source_sha256=sources(),hardware=dict(
        platform=platform.platform(),torch=torch.__version__,device='cpu',threads=1),curve=[])
    if a.contracts_only:
        result.update(status='completed',numerical_contracts=contracts(clock_features=a.clock_features),
            wall_s=time.perf_counter()-started)
        out.write_text(json.dumps(result,indent=2)+'\n');return
    prep=time.perf_counter();data=load(a.dataset,a.fit,a.dev);prep=time.perf_counter()-prep
    result['protocol']=dict(**data['protocol'],preprocessing_wall_s=prep,
        input='Feature identity, train-scaled numeric value, missing flag; independent row reset',
        timestamps='Canonical feature-processing coordinates, not physical asynchronous observations',
        tuning='Exactly four development candidates/checkpoints for pilots; no held-out test evaluation',
        resource_boundary='Whole neural optimizer fit; preprocessing separately timed/counted; no hardware energy claim',
        feature_scaling_operations=dict(fit_elements=a.fit*len(data['feature_names']),
            transformed_fit_dev_elements=(a.fit+a.dev)*len(data['feature_names'])),
        scope='Small single-seed supervised real-data development screen, not a benchmark supremacy result')
    x=torch.tensor(data['x_fit'],dtype=torch.float32);v=torch.tensor(data['x_dev'],dtype=torch.float32)
    y=data['y_fit'];d=data['y_dev'];classification=a.dataset=='banknote'
    score=lambda r:r['nll' if classification else 'rmse'];best=float('inf')
    def persist():
        result.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        path=out if result['status']=='completed' else running;tmp=path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    persist()
    if a.model=='trees':
        from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
        import sklearn
        from threadpoolctl import threadpool_limits
        result['hardware']['sklearn']=sklearn.__version__;total_fit=0.
        for steps in (16,32,64,128):
            cls=HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
            model=cls(max_iter=steps,max_leaf_nodes=15,min_samples_leaf=5,learning_rate=.1,
                      early_stopping=False,random_state=a.seed)
            with threadpool_limits(limits=1):
                t=time.perf_counter();model.fit(x.numpy(),y);total_fit+=time.perf_counter()-t
                t=time.perf_counter()
                pred=np.log(np.maximum(model.predict_proba(v.numpy()),1e-30)) if classification else model.predict(v.numpy())
                m=metrics(pred,d,classification);m['wall_s']=time.perf_counter()-t
            result['curve'].append(dict(max_iter=steps,dev=m))
            if score(m)<best:best=score(m);chosen=model;result['selected_candidate']=steps;selected=m
            persist()
        nodes=[p.nodes for stage in chosen._predictors for p in stage]
        result.update(final=dict(dev=selected),work=dict(tree_fit_wall_s_all_candidates=total_fit,
            selected_tree_nodes=sum(len(n) for n in nodes),selected_tree_node_bytes=sum(n.nbytes for n in nodes),
            selected_tree_count=len(nodes),neural_flops=None,
            convention='Four independently fitted boosting candidates; all candidate fitting wall time counted; neural FLOPs unavailable'))
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
            fit_wall+=time.perf_counter()-t;m=evaluate(model,v,d,classification)
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
        result.update(final=dict(dev=evaluate(model,v,d,classification)),activity=activity,
            work=dict(fitting_rows=a.fit*a.epochs,whole_neural_fit_unit_special_flops=flops,
                neural_fit_unit_special_flops_per_row=flops/(a.fit*a.epochs),
                inference_unit_special_flops_per_row=inference['arithmetic_flops']+inference['special_function_evaluations'],
                fit_wall_s=fit_wall,stages=stages,inference_trace=inference,
                scope='Actual full PyTorch forward/loss/backward/normalization/clipping/Adam trace; specials counted once; preprocessing, evaluation, contracts and RNG overhead separate'))
    result['status']='completed';persist()


if __name__=='__main__':main()
