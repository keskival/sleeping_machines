"""Frozen query versus query-plus-causal-memory diagnostic; no core updates."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import warnings
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import sleeping_machines.batched_addressed_fit as K
from dvs_frozen_feature_diagnostic import nll


@torch.no_grad()
def extract(model,rows,trace_shapes=False):
    """Read the pre-query state at source admission, then the ordinary query feature."""
    features=[];probabilities=[];work=[];box={};observed=[]
    handle=model.head.register_forward_pre_hook(lambda module,args:observed.append(args[0].detach()))
    captured=set()
    try:
        for start in range(0,len(rows),32):
            batch=rows[start:start+32]
            if not all(float(row['events'][-1][1][-1])==1. for row in batch):
                raise ValueError('Observed query flag required')
            prefix=[{**row,'events':row['events'][:-1]} for row in batch]
            def before():
                _,state,_=K.forward(model,prefix,314159)
                box['state']=state
            def full():
                z,_,_=K.forward(model,batch,314159)
                box['z']=z;box['query']=observed[-1]
            for stage,fn in [('pre_query_replay',before),('full_query_replay',full)]:
                observed.clear()
                if trace_shapes and (stage,len(batch)) not in captured:
                    tr=N.capture(fn);assert tr['formula_coverage_complete']
                    work.append(dict(stage=stage,batch_targets=len(batch),arithmetic_flops=tr['arithmetic_flops'],
                        special_function_evaluations=tr['special_function_evaluations'],
                        exponential_random_draws=tr['exponential_random_draws']))
                    captured.add((stage,len(batch)))
                else:fn()
            state=box['state'];query_time=torch.tensor([row['events'][-1][0] for row in batch],dtype=torch.float64)
            admission=torch.maximum(query_time,state['context_times'].max(-1).values)
            extras=[]
            for depth in range(model.depth):
                ages=torch.where(state['seen'][depth],(admission[:,None]-state['arrivals'][depth]).clamp_min(0),0.)
                extras.extend([state['memories'][depth].double().flatten(1),ages,state['seen'][depth].double()])
            features.append(torch.cat([box['query'].double(),*extras],-1).numpy().copy())
            probabilities.append(box['z'].softmax(-1).numpy().copy())
    finally:handle.remove()
    return np.concatenate(features),np.concatenate(probabilities),work


@torch.no_grad()
def serial_contract(model,rows,features):
    for index,row in enumerate(rows[:2]):
        _,state=N.predict(model,{**row,'events':row['events'][:-1]},314159,False)
        query_time=max(float(row['events'][-1][0]),float(state.contexts[0][1].max()))
        expected=[]
        for depth in range(model.depth):
            memory=[];ages=[];seen=[]
            for head in range(model.heads):
                for unit in range(model.pool):
                    key=(depth,head,0,unit);known=key in state.memories
                    memory.extend(state.memories[key].double().tolist() if known else [0.]*model.payload)
                    ages.append(max(0.,query_time-float(state.arrivals[key])) if known else 0.)
                    seen.append(float(known))
            expected.extend(memory+ages+seen)
        np.testing.assert_allclose(features[index,model.total_payload:],expected,rtol=1e-4,atol=1e-5)


def probe(xfit,xdev,labels,dev_labels):
    folds=list(StratifiedKFold(3,shuffle=True,random_state=681).split(np.zeros(len(labels)),labels))
    configs=[dict(kind='linear',C=c,gamma=None) for c in (.1,1.,10.)]
    configs += [dict(kind='rbf',C=c,gamma=g) for c in (1.,10.) for g in (.25,1.,4.)]
    def transform(train,evaluation):
        center=train.mean(0);scale=np.maximum(train.std(0),.5)
        return (train-center)/scale,(evaluation-center)/scale,center,scale
    def classifier(config,x):
        if config['kind']=='linear':return LogisticRegression(C=config['C'],max_iter=1000,random_state=681)
        return SVC(C=config['C'],gamma=config['gamma']/(x.shape[1]*max(float(x.var()),1e-12)),probability=True,random_state=681)
    candidates=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for config in configs:
            predictions=np.empty((len(labels),11))
            for training,held in folds:
                xt,xv,_,_=transform(xfit[training],xfit[held]);decoder=classifier(config,xt)
                decoder.fit(xt,labels[training]);predictions[held]=decoder.predict_proba(xv)
            candidates.append(dict(configuration=config,fitting_decoder_cv_nll=nll(predictions,labels)))
        chosen=min(candidates,key=lambda x:x['fitting_decoder_cv_nll'])
        xt,xv,center,scale=transform(xfit,xdev);decoder=classifier(chosen['configuration'],xt)
        decoder.fit(xt,labels);predictions=decoder.predict_proba(xv)
    return dict(accuracy=float(np.mean(predictions.argmax(1)==dev_labels)),nll=nll(predictions,dev_labels),
        probabilities=predictions.tolist()),chosen,candidates,dict(decoder=decoder,feature_center=center,feature_scale=scale),sorted(set(str(x.message) for x in caught))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--reference-probe',required=True);a=p.parse_args();started=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    previous=json.loads((ROOT/a.reference_probe).read_text());assert previous['status']=='completed'
    native=previous['args']['native'];r=json.loads((ROOT/native).read_text());assert r['status']=='completed'
    assert N.sha(ROOT/native)==previous['native_result_sha256']
    for path,h in {**previous['source_sha256'],**r['source_sha256']}.items():assert N.sha(ROOT/path)==h,('Changed source',path)
    ckpath=(ROOT/native).with_suffix('.progress.pt');assert N.sha(ckpath)==previous['native_checkpoint_sha256']
    ck=torch.load(ckpath,weights_only=False);assert ck['source_sha256']==r['source_sha256']
    old_artifact=ROOT/previous['selected_heads_artifact'];assert N.sha(old_artifact)==previous['selected_heads_sha256']
    old_heads=joblib.load(old_artifact);config=argparse.Namespace(**r['args'])
    fit,dev,metadata=N.load(config);assert metadata==r['data'] and len(fit)==984 and len(dev)==192
    labels=np.array([row['target'] for row in fit]);dev_labels=np.array([row['target'] for row in dev])
    rows=[];artifacts={};feature_artifacts={};works=[]
    for arm in ('initial','selected'):
        model=N.make_model(config)
        if arm=='selected':model.load_state_dict(ck['best_state'])
        model.eval();before={n:v.detach().clone() for n,v in model.state_dict().items()};begin=time.perf_counter()
        xf,_,work=extract(model,fit,True);xd,original,_=extract(model,dev)
        serial_contract(model,fit,xf);serial_contract(model,dev,xd)
        native_score=r['initial_dev'] if arm=='initial' else r['final']
        np.testing.assert_allclose(original,native_score['probabilities'],rtol=1e-5,atol=1e-6)
        old=next(row for row in previous['rows'] if row['encoder']==arm);head=old_heads[arm]
        old_predictions=head['decoder'].predict_proba((xd[:,:32]-head['feature_center'])/head['feature_scale'])
        np.testing.assert_allclose(old_predictions,old['frozen_decoder_development']['probabilities'],rtol=1e-5,atol=1e-6)
        replay_wall=time.perf_counter()-begin;begin=time.perf_counter()
        quality,chosen,candidates,artifact,caught=probe(xf,xd,labels,dev_labels)
        solver_wall=time.perf_counter()-begin
        assert all(torch.equal(v.detach(),before[n]) for n,v in model.state_dict().items())
        # Full fit: thirty 32-target batches plus one24; dev: six32-target batches.
        replay_work=sum((w['arithmetic_flops']+w['special_function_evaluations'])*(36 if w['batch_targets']==32 else 1) for w in work)
        works.extend([dict(encoder=arm,**w) for w in work])
        rows.append(dict(encoder=arm,query_feature_dimension=32,augmented_feature_dimension=xf.shape[1],
            query_probe_development={k:v for k,v in old['frozen_decoder_development'].items() if k!='probabilities'},
            augmented_probe_development=quality,fit_only_selected_decoder=chosen,all_fit_cv_candidates=candidates,
            retained_state_nll_improvement=old['frozen_decoder_development']['nll']-quality['nll'],
            state_access_signal_passed=old['frozen_decoder_development']['nll']-quality['nll']>=.05
                and quality['accuracy']>=old['frozen_decoder_development']['accuracy']-.01,
            replay_wall_s=replay_wall,decoder_grid_wall_s=solver_wall,warnings=caught,
            replay_whole_gflops_estimate=replay_work/1e9,decoder_solver_fit_flops=None,
            original_encoder_fit_gflops_estimate=r['work']['whole_fit_unit_special_flops_estimate']/1e9 if arm=='selected' else 0.,
            encoder_parameters_preserved=True,pre_query_state_matches_serial=True,original_probabilities_reproduced=True,
            old_query_probe_probabilities_reproduced=True,optimizer_updates=0))
        artifacts[arm]=artifact;feature_artifacts[arm+'_fit']=xf;feature_artifacts[arm+'_dev']=xd
        print(json.dumps(dict(encoder=arm,nll=quality['nll'],accuracy=quality['accuracy'],improvement=rows[-1]['retained_state_nll_improvement'])),flush=True)
    decoder_path=out.with_suffix('.heads.joblib');joblib.dump(artifacts,decoder_path)
    features_path=out.with_suffix('.features.npz');np.savez_compressed(features_path,**feature_artifacts,fit_labels=labels,dev_labels=dev_labels)
    own=['experiments/dvs_persistent_state_probe.py','sleeping_machines/batched_addressed_fit.py',
        'experiments/dvs_frozen_feature_diagnostic.py','experiments/theory/94_retained_state_and_sparse_access.md']
    result=dict(status='completed',args=vars(a),rows=rows,native_result=native,native_result_sha256=N.sha(ROOT/native),
        reference_probe_sha256=N.sha(ROOT/a.reference_probe),data=metadata,
        feature_artifact=str(features_path.relative_to(ROOT)),feature_artifact_sha256=N.sha(features_path),
        decoder_artifact=str(decoder_path.relative_to(ROOT)),decoder_artifact_sha256=N.sha(decoder_path),
        source_sha256={**r['source_sha256'],**{path:N.sha(ROOT/path) for path in own}},
        stage_work=works,formula_coverage_complete=True,contracts_passed=3,
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen dense diagnostic access to pre-query memory/age/occupancy plus unchanged query feature. '
            'Decoder selection uses3-fold FIT-only NLL; encoders previously saw all fitting labels and dev-selected weights, '
            'so not unbiased end-to-end CV or independent validation. No core refit, sparse-access architecture substitution, '
            'useful-depth theorem, official test or practical advantage. Original fitting and two core replay passes retained; '
            'feature materialization, solver/grid arithmetic, traffic and energy unmeasured, not zero.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
