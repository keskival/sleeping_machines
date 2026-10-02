"""Fitting-user selected temporal-resolution control diagnostic."""
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
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
import sklearn
ROOT=Path(__file__).resolve().parents[1]

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def aggregate(x,bins):
    assert x.ndim==3 and x.shape[1]==20 and 20%bins==0
    return x.reshape(len(x),bins,20//bins,x.shape[2]).sum(2).reshape(len(x),-1)
def transform(train,test):
    log=np.log1p(train);center=log.mean(0);scale=np.maximum(log.std(0),.5)
    x=(log-center)/scale;y=(np.log1p(test)-center)/scale
    return x,y,center,scale
def score(prob,y):
    assert prob.shape==(len(y),11) and np.all(prob>0) and np.isfinite(prob).all()
    return dict(targets=len(y),accuracy=float(np.mean(prob.argmax(1)==y)),nll=float(-np.log(prob[np.arange(len(y)),y]).mean()))
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();started=time.perf_counter()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert Path(a.tag).name==a.tag and not out.exists()
    meta_path=ROOT/'experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.json'
    meta=json.loads(meta_path.read_text());artifact=ROOT/meta['data_artifact']
    assert not meta['protocol']['official_test_read'] and digest(artifact)==meta['data_sha256']
    z=np.load(artifact,allow_pickle=False);fit,dev=z['fit_counts'],z['dev_counts'];y,yd=z['fit_labels'],z['dev_labels']
    assert len(fit)==984 and len(dev)==192
    groups=np.array([str(i).split('_')[0] for i in z['fit_ids']]);splits=list(GroupKFold(3).split(fit,y,groups))
    contracts={}
    for bins in (1,4,20):
        values=aggregate(fit,bins)
        np.testing.assert_array_equal(values.sum(1),fit.sum((1,2)))
        perm=fit.reshape(len(fit),bins,20//bins,32)[:,:,::-1,:].reshape(fit.shape)
        np.testing.assert_array_equal(aggregate(perm,bins),values)
        contracts[str(bins)]='mass conservation and within-bin permutation invariance pass'
    rows=[];saved={};warnings_seen=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for bins in (1,4,20):
            raw=aggregate(fit,bins);raw_dev=aggregate(dev,bins);candidates=[]
            for strength in (.1,1.,10.):
                predictions=np.zeros((len(y),11));folds=[]
                for training,validation in splits:
                    assert not set(groups[training])&set(groups[validation])
                    x,xv,_,_=transform(raw[training],raw[validation]);gamma=1/(x.shape[1]*x.var())
                    model=SVC(C=strength,gamma=gamma,probability=True,random_state=681)
                    begin=time.perf_counter();model.fit(x,y[training]);prob=model.predict_proba(xv);wall=time.perf_counter()-begin
                    assert model.classes_.tolist()==list(range(11));predictions[validation]=prob
                    folds.append(dict(training_users=sorted(set(groups[training])),validation_users=sorted(set(groups[validation])),quality=score(prob,y[validation]),wall_s=wall))
                candidates.append(dict(C=strength,pooled_quality=score(predictions,y),folds=folds))
            selected=min(candidates,key=lambda r:r['pooled_quality']['nll'])
            x,xd,center,scale=transform(raw,raw_dev);gamma=1/(x.shape[1]*x.var());model=SVC(C=selected['C'],gamma=gamma,probability=True,random_state=681)
            begin=time.perf_counter();model.fit(x,y);fitting_wall=time.perf_counter()-begin
            prob=model.predict_proba(xd);final=score(prob,yd)
            obj=dict(model=model,center=center,scale=scale,bins=bins,gamma=gamma)
            def predict(counts):
                flat=aggregate(counts[None],bins)
                return model.predict_proba((np.log1p(flat)-center)/scale)[0]
            predict(dev[0]);times=[]
            for _ in range(3):
                begin=time.perf_counter();sequential=np.array([predict(counts) for counts in dev]);times.append(time.perf_counter()-begin)
                np.testing.assert_allclose(sequential,prob,rtol=1e-10,atol=1e-12)
            buffer=io.BytesIO();joblib.dump(obj,buffer,compress=0)
            row=dict(bins=bins,dimensions=x.shape[1],selection='minimum pooled fitting-user GroupKFold NLL',candidates=candidates,selected_C=selected['C'],gamma=gamma,final=final,development_probabilities=prob.tolist(),support_vectors=len(model.support_),model_storage_bytes=len(buffer.getvalue()),final_fit_wall_s=fitting_wall,all_fold_fit_and_validation_wall_s=sum(f['wall_s'] for c in candidates for f in c['folds']),inference_repeat_wall_s=times,sequential_median_ms_per_query=float(np.median(times))*1000/len(dev),fitting_flops=None,inference_flops=None)
            rows.append(row);saved[str(bins)]=obj;print(json.dumps({k:v for k,v in row.items() if k in ('bins','dimensions','selected_C','final','support_vectors','model_storage_bytes','sequential_median_ms_per_query')}),flush=True)
        warnings_seen=sorted(set(str(w.message) for w in caught))
    reference=rows[-1]
    for row in rows[:-1]:row['exploratory_quality_gate']=row['final']['nll']<=reference['final']['nll']+.02 and row['final']['accuracy']>=reference['final']['accuracy']-.01
    model_path=out.with_suffix('.models.joblib');joblib.dump(saved,model_path)
    files=['experiments/aws_gesture_resolution.py','experiments/theory/aws_20261002_temporal_resolution.md']
    result=dict(status='completed',args=vars(a),rows=rows,numerical_contracts=contracts,fit_targets=984,development_targets=192,fold_fits=27,final_fits=3,protocol=meta['protocol'],data_meta_sha256=digest(meta_path),data_sha256=digest(artifact),source_sha256={n:digest(ROOT/n) for n in files},model_artifact=str(model_path.relative_to(ROOT)),model_sha256=digest(model_path),warnings=warnings_seen,sklearn=sklearn.__version__,wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Exploratory held-out-user representation diagnostic; no development selection, test access or native architecture claim. Solver FLOPs unknown, not zero; all searches retained.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
