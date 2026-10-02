"""Fitting-only frozen context/resident-state information probes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time
import warnings
import joblib
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_coarse_native as A
import dvs_numpy_inference_contracts as T
from sleeping_machines.numpy_addressed_inference import NumpyAddressedInference,pack

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(p,y):return dict(accuracy=float(np.mean(p.argmax(1)==y)),nll=float(-np.log(p[np.arange(len(y)),y]).mean()),targets=len(y))
def features(port,row):
    logits,state=port.predict(row['events']);arrival=float(state['context_times'].max())
    context=port.align(list(state['context']),state['context_times'],arrival,port.L-1)
    ages=np.where(state['seen'],np.maximum(0,arrival-state['arrivals']),0.)
    resident=np.concatenate([context,state['memories'].reshape(-1),ages.reshape(-1),state['seen'].reshape(-1).astype(float)])
    assert np.isfinite(resident).all()
    return context,resident,logits

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);a=parser.parse_args();started=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    rows=[];models={};checks=[];replay=[];warn=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for seed in (6,7,8):
            result_path=ROOT/f'experiments/results/dvs_native/aws_full_coarse_20261002T213100Z_coarse_matchedclock_s{seed}.json'
            parent=json.loads(result_path.read_text());assert parent['status']=='completed'
            for name,digest in parent['source_sha256'].items():assert sha(ROOT/name)==digest
            args=argparse.Namespace(**parent['args']);fit,dev,data=A.load(args);assert data==parent['data']
            checkpoint=result_path.with_suffix('.progress.pt');ck=torch.load(checkpoint,weights_only=False)
            assert ck['source_sha256']==parent['source_sha256'] and ck['data']==data and ck['cursor']['epoch']==9
            assert ck['result']['selected_epoch']==parent['selected_epoch']
            labels=np.array([r['target'] for r in fit]);dy=np.array([r['target'] for r in dev]);groups=np.array([r['identity'].split('_')[0] for r in fit]);splits=list(GroupKFold(3).split(labels,labels,groups))
            for fitted in (False,True):
                model=A.C.make_model(args);model.eval()
                if fitted:model.load_state_dict(ck['best_state'])
                checks.append(dict(seed=seed,fitted=fitted,checkpoint_sha256=sha(checkpoint),numpy_contract=T.check(model,dev[:8],False)))
                port=NumpyAddressedInference(pack(model));changed=dict(dev[0],target=(dev[0]['target']+1)%11)
                one=features(port,dev[0]);two=features(port,changed)
                for x,y in zip(one,two):np.testing.assert_array_equal(x,y)
                begin=time.perf_counter();extracted=[features(port,r) for r in fit+dev];replay.append(dict(seed=seed,fitted=fitted,targets=len(extracted),wall_s=time.perf_counter()-begin,observed_events=sum(len(r['events']) for r in fit+dev)))
                if fitted:
                    values=np.array([e[2] for e in extracted[len(fit):]]);p=np.exp(values-values.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
                    np.testing.assert_allclose(p,np.array(parent['final']['probabilities']),rtol=1e-4,atol=1e-6)
                for index,kind in enumerate(('context','resident')):
                    matrix=np.array([e[index] for e in extracted]);raw,raw_dev=matrix[:len(fit)],matrix[len(fit):];candidates=[]
                    for strength in (1.,10.):
                        predictions=np.zeros((len(fit),11));folds=[]
                        for training,validation in splits:
                            assert not set(groups[training])&set(groups[validation])
                            center=raw[training].mean(0);scale=np.maximum(raw[training].std(0),1e-6)
                            x=(raw[training]-center)/scale;xv=(raw[validation]-center)/scale;gamma=1/(x.shape[1]*x.var())
                            m=SVC(C=strength,gamma=gamma,probability=True,random_state=681);begin=time.perf_counter();m.fit(x,labels[training]);p=m.predict_proba(xv);predictions[validation]=p
                            folds.append(dict(wall_s=time.perf_counter()-begin,quality=score(p,labels[validation]),training_users=sorted(set(groups[training])),validation_users=sorted(set(groups[validation]))))
                        candidates.append(dict(C=strength,quality=score(predictions,labels),folds=folds))
                    selected=min(candidates,key=lambda r:r['quality']['nll']);center=raw.mean(0);scale=np.maximum(raw.std(0),1e-6);x=(raw-center)/scale;xd=(raw_dev-center)/scale;gamma=1/(x.shape[1]*x.var())
                    m=SVC(C=selected['C'],gamma=gamma,probability=True,random_state=681);begin=time.perf_counter();m.fit(x,labels);fit_wall=time.perf_counter()-begin;p=m.predict_proba(xd)
                    obj=dict(model=m,center=center,scale=scale,seed=seed,fitted_encoder=fitted,kind=kind,checkpoint_sha256=sha(checkpoint) if fitted else None);buf=io.BytesIO();joblib.dump(obj,buf,compress=0)
                    row=dict(seed=seed,fitted_encoder=fitted,kind=kind,dimensions=x.shape[1],selected_C=selected['C'],candidates=candidates,final=score(p,dy),probabilities=p.tolist(),probe_storage_bytes=len(buf.getvalue()),support_vectors=len(m.support_),final_probe_fit_wall_s=fit_wall,original_native_quality=parent['final'] if fitted else None,original_encoder_fit_gflops=parent['work']['whole_fit_unit_special_flops_estimate']/1e9 if fitted else 0.,solver_fitting_flops=None)
                    if fitted:row['original_native_quality']={k:parent['final'][k] for k in ('accuracy','nll')}
                    rows.append(row);models[f'{seed}_{fitted}_{kind}']=obj;print(json.dumps(dict(seed=seed,fitted=fitted,kind=kind,final=row['final'])),flush=True)
        warn=sorted(set(str(w.message) for w in caught))
    gates=[]
    for kind in ('context','resident'):
        gains=[];acc=[]
        for seed in (6,7,8):
            row=next(r for r in rows if r['seed']==seed and r['fitted_encoder'] and r['kind']==kind)
            ref=row['original_native_quality'] if kind=='context' else next(r['final'] for r in rows if r['seed']==seed and r['fitted_encoder'] and r['kind']=='context')
            gains.append(ref['nll']-row['final']['nll']);acc.append(row['final']['accuracy']-ref['accuracy'])
        gates.append(dict(kind=kind,nll_gains=gains,accuracy_gains=acc,pass_gate=np.mean(gains)>=.05 and np.mean(acc)>=.03 and min(gains)>=0))
    artifact=out.with_suffix('.models.joblib');joblib.dump(models,artifact)
    files=['experiments/aws_coarse_readout_probe.py','experiments/theory/aws_20261002_coarse_readout_diagnosis.md','sleeping_machines/numpy_addressed_inference.py','experiments/dvs_numpy_inference_contracts.py']
    result=dict(status='completed',args=vars(a),rows=rows,gates=gates,numerical_checks=checks,replay=replay,source_sha256={n:sha(ROOT/n) for n in files},model_artifact=str(artifact.relative_to(ROOT)),model_sha256=sha(artifact),warnings=warn,wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Frozen information/readout diagnostic, all fitting-only selections and native/replay work retained. Resident reads every address, not sparse inference. No new native fit, fresh confirmation or supremacy claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False,default=lambda v:bool(v) if isinstance(v,np.bool_) else str(v))+'\n')
if __name__=='__main__':main()
