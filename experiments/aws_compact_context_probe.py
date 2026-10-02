"""Fixed compact native query readout and raw4/raw20 controls; no producer updates."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import warnings

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_coarse_native as A
import dvs_numpy_inference_contracts as T
from sleeping_machines.numpy_addressed_inference import NumpyAddressedInference,pack
from sleeping_machines.query_feature_inference import QueryFeatureInference
from sleeping_machines.compact_context_predictor import basis,probabilities,save,load,Predictor


def quality(p,y):
    assert np.isfinite(p).all() and (p>0).all()
    return dict(accuracy=float(np.mean(p.argmax(1)==y)),nll=float(-np.log(p[np.arange(len(y)),y]).mean()))


def state_equal(a,b):
    assert set(a)==set(b)
    for k in a:
        if isinstance(a[k], np.ndarray): np.testing.assert_array_equal(a[k],b[k])
        else: assert a[k]==b[k],k


def fit_basis(raw,y,floor):
    center=raw.mean(0);scale=np.maximum(raw.std(0),floor);x=(raw-center)/scale
    anchors=np.concatenate([KMeans(n_clusters=3,n_init=5,max_iter=100,random_state=681).fit(x[y==c]).cluster_centers_ for c in range(11)])
    gamma=1/(x.shape[1]*x.var())
    return dict(center=center,scale=scale,anchors=anchors,gamma=float(gamma)),basis(x,anchors,gamma)


def transformed(raw,b): return basis((raw-b['center'])/b['scale'],b['anchors'],b['gamma'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);a=parser.parse_args()
    torch.set_num_threads(1);started=time.perf_counter();out=ROOT/'experiments/results/diagnostics'/f'{a.tag}.json';assert not out.exists()
    cache_result=ROOT/'experiments/results/diagnostics/aws_frozen_polynomial_20261002T215500Z.json'
    cache_info=json.loads(cache_result.read_text());assert cache_info['status']=='completed'
    for n,h in cache_info['source_sha256'].items(): assert A.N.sha(ROOT/n)==h,n
    cache_path=cache_result.with_suffix('.features.npz')
    assert A.N.sha(cache_path)==cache_info['artifacts'][str(cache_path.relative_to(ROOT))]['sha256']
    cache=np.load(cache_path,allow_pickle=False);y=cache['fit_labels'];yd=cache['dev_labels']
    base=ROOT/'experiments/results/dvs_native/aws_full_coarse_20261002T213100Z_coarse_matchedclock_s6.json'
    parent=json.loads(base.read_text());args=argparse.Namespace(**parent['args']);fit,dev,data=A.load(args)
    groups=np.array([r['identity'].split('_')[0] for r in fit]);splits=list(GroupKFold(3).split(y,y,groups))
    raw_parent=json.loads((ROOT/args.data).read_text());data_path=ROOT/raw_parent['data_artifact'];assert A.N.sha(data_path)==raw_parent['data_sha256']
    z=np.load(data_path,allow_pickle=False);fit_counts=z['fit_counts'];dev_counts=z['dev_counts']
    np.testing.assert_array_equal(y,z['fit_labels']);np.testing.assert_array_equal(yd,z['dev_labels'])
    packet_raw=np.log1p(A.coalesce(fit_counts,4).reshape(len(y),-1));packet_center=packet_raw.mean(0);packet_scale=np.maximum(packet_raw.std(0),.5)
    arms=[];checks=[];rows=[];artifacts={};caught=[]
    # Verify demand-only context extraction before any decoder optimization.
    for seed in (6,7,8):
        path=base.with_name(f'aws_full_coarse_20261002T213100Z_coarse_matchedclock_s{seed}.json');p=json.loads(path.read_text())
        for n,h in p['source_sha256'].items(): assert A.N.sha(ROOT/n)==h,n
        ck=torch.load(path.with_suffix('.progress.pt'),weights_only=False,map_location='cpu');args=argparse.Namespace(**p['args'])
        for fitted in (False,True):
            model=A.C.make_model(args);model.eval()
            if fitted:model.load_state_dict(ck['best_state'])
            name=f's{seed}_{"fitted" if fitted else "initial"}';packed=pack(model,maximum_events=5)
            reference=NumpyAddressedInference(packed);query=QueryFeatureInference(packed)
            for index in range(8):
                logits,state=reference.predict(dev[index]['events']);vector,other=query.predict_query(dev[index]['events']);state_equal(state,other)
                np.testing.assert_array_equal(packed['head'][0]@vector+packed['head'][1],logits)
                np.testing.assert_allclose(vector,cache[name+'_dev'][index],rtol=0,atol=0)
                changed=dict(dev[index],target=(dev[index]['target']+1)%11);again,repeated=query.predict_query(changed['events']);state_equal(other,repeated);np.testing.assert_array_equal(vector,again)
            checks.append(dict(encoder=name,state_query_target_repeat_checks=8,torch_core=T.check(model,dev[:4],False)))
            # Original affine coefficients are not used by demand-only inference.
            packed['head']=(np.empty((0,32),dtype=np.float32),None)
            arms.append(dict(name=name,seed=seed,fitted=fitted,kind='native',bins=4,packed=packed,packet_center=packet_center,packet_scale=packet_scale,
                parent_fit_gflops=p['work']['whole_fit_unit_special_flops_estimate']/1e9 if fitted else 0.,
                parent_native_quality={k:p['final'][k] for k in ('accuracy','nll')} if fitted else None,
                parent_sha256=A.N.sha(path),checkpoint_sha256=A.N.sha(path.with_suffix('.progress.pt')) if fitted else None,
                raw=cache[name+'_fit'],dv=cache[name+'_dev']))
    for bins in (4,20):
        arms.append(dict(name=f'raw{bins}',kind='raw',bins=bins,raw=np.log1p(A.coalesce(fit_counts,bins).reshape(len(y),-1)),dv=np.log1p(A.coalesce(dev_counts,bins).reshape(len(yd),-1))))
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter('always')
        for arm in arms:
            raw,dv=arm['raw'],arm['dv'];floor=1e-6 if arm['kind']=='native' else .5;begin=time.perf_counter();folds=[]
            for training,validation in splits:
                assert not set(groups[training])&set(groups[validation])
                b,x=fit_basis(raw[training],y[training],floor);folds.append((training,validation,x,transformed(raw[validation],b)))
            candidates=[]
            for strength in (.1,1.,10.):
                predictions=np.zeros((len(y),11));records=[]
                for training,validation,x,xv in folds:
                    decoder=LogisticRegression(C=strength,max_iter=1000,tol=1e-6,random_state=681);decoder.fit(x,y[training]);predictions[validation]=decoder.predict_proba(xv)
                    records.append(dict(validation_users=sorted(set(groups[validation])),iterations=decoder.n_iter_.tolist(),quality=quality(predictions[validation],y[validation])))
                candidates.append(dict(C=strength,quality=quality(predictions,y),folds=records))
            selected=min(candidates,key=lambda c:c['quality']['nll']);b,x=fit_basis(raw,y,floor);decoder=LogisticRegression(C=selected['C'],max_iter=1000,tol=1e-6,random_state=681);decoder.fit(x,y)
            assert decoder.classes_.tolist()==list(range(11));expected=decoder.predict_proba(transformed(dv,b))
            model={k:arm[k] for k in ('kind','bins','packed','packet_center','packet_scale') if k in arm}
            model.update(b,weights=decoder.coef_,bias=decoder.intercept_);artifact=out.with_name(a.tag+'_'+arm['name']+'.npz');storage=save(artifact,model)
            portable=Predictor(load(artifact));actual=np.array([portable.predict(c) for c in dev_counts]);np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
            walls=[]
            for _ in range(3):
                t=time.perf_counter();repeat=np.array([portable.predict(c) for c in dev_counts]);walls.append(time.perf_counter()-t);np.testing.assert_array_equal(repeat,actual)
            # Inference basis operation convention excludes native core/preprocessing.
            d=raw.shape[1];basis_work=33*(3*d-1)+33+2*33*11+11
            row={k:v for k,v in arm.items() if k not in ('raw','dv','packed','packet_center','packet_scale')}
            row.update(final=quality(actual,yd),probabilities=actual.tolist(),selected_C=selected['C'],candidates=candidates,storage=storage,
                fitting_wall_s=time.perf_counter()-begin,sequential_wall_s=walls,decoder_only_arithmetic_plus_exp_ops=basis_work,
                total_fitting_flops=None,total_inference_flops=None,full_port_max_probability_error=float(np.max(np.abs(actual-expected))))
            rows.append(row);artifacts[str(artifact.relative_to(ROOT))]=dict(sha256=A.N.sha(artifact),**storage);print(json.dumps(dict(name=arm['name'],final=row['final'],storage=storage)),flush=True)
        caught=sorted(set(str(w.message) for w in warning_list))
    native=[r for r in rows if r['kind']=='native' and r['fitted']];initial=[r for r in rows if r['kind']=='native' and not r['fitted']];controls=[r for r in rows if r['kind']=='raw']
    gains=[r['parent_native_quality']['nll']-r['final']['nll'] for r in native];acc=[r['final']['accuracy']-r['parent_native_quality']['accuracy'] for r in native]
    learned=float(np.mean([r['final']['nll'] for r in initial])-np.mean([r['final']['nll'] for r in native]))
    comparisons=[dict(seed=r['seed'],control=c['name'],nll_gap=r['final']['nll']-c['final']['nll'],accuracy_gap=r['final']['accuracy']-c['final']['accuracy'],export_byte_ratio=r['storage']['serialized_bytes']/c['storage']['serialized_bytes']) for r in native for c in controls]
    gate=dict(native_nll_gains=gains,native_accuracy_gains=acc,mean_learned_nll_gain=learned,
        staged_nomination_pass=bool(min(gains)>=.05 and min(acc)>=-.01 and learned>=.05),comparisons=comparisons,
        practical_storage_gate_pass=all(c['nll_gap']<=.02 and c['accuracy_gap']>=-.01 and c['export_byte_ratio']<=.8 for c in comparisons))
    sources=['experiments/aws_compact_context_probe.py','sleeping_machines/compact_context_predictor.py','sleeping_machines/query_feature_inference.py','sleeping_machines/numpy_addressed_inference.py','experiments/theory/aws_20261002_compact_context_protocol.md']
    result=dict(status='completed',tag=a.tag,rows=rows,gate=gate,numerical_checks=checks,artifacts=artifacts,fold_fits=72,final_fits=8,class_basis_fits=8*4*11,
        cache_result_sha256=A.N.sha(cache_result),cache_sha256=A.N.sha(cache_path),data=data,source_sha256={n:A.N.sha(ROOT/n) for n in sources},warnings=caught,
        wall_s=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen local native context plus dense33anchor readout. Full raw4/raw20 controls and standalone predictor exports include preprocessing. Parent fit/cache work retained; combined solver/core work not measured. No end-to-end fit, sparse RBF claim or official test.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(gate),flush=True)


if __name__=='__main__':main()
