"""Separate fixed-context convex decoder fitting from coupled native learning."""
import argparse
import copy
import hashlib
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
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_coarse_native as A
import aws_coarse_readout_probe as R
import dvs_numpy_inference_contracts as T
from sleeping_machines.numpy_addressed_inference import NumpyAddressedInference,pack
from sleeping_machines.folded_polynomial_readout import FoldedPolynomialReadout,polynomial_features

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quality(prob,y):
    assert np.isfinite(prob).all() and (prob>0).all()
    return dict(accuracy=float(np.mean(prob.argmax(1)==y)),nll=float(-np.log(prob[np.arange(len(y)),y]).mean()),targets=len(y))

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();started=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    rows=[];checks=[];replays=[];estimators={};encoders={};heads={};arrays={};warnings_seen=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for seed in (6,7,8):
            path=ROOT/f'experiments/results/dvs_native/aws_full_coarse_20261002T213100Z_coarse_matchedclock_s{seed}.json';parent=json.loads(path.read_text());assert parent['status']=='completed'
            for n,d in parent['source_sha256'].items():assert sha(ROOT/n)==d
            args=argparse.Namespace(**parent['args']);fit,dev,data=A.load(args);assert data==parent['data']
            checkpoint=path.with_suffix('.progress.pt');ck=torch.load(checkpoint,weights_only=False);assert ck['source_sha256']==parent['source_sha256'] and ck['data']==data and ck['cursor']['epoch']==9
            y=np.array([r['target'] for r in fit]);yd=np.array([r['target'] for r in dev]);groups=np.array([r['identity'].split('_')[0] for r in fit]);splits=list(GroupKFold(3).split(y,y,groups))
            for fitted in (False,True):
                model=A.C.make_model(args);model.eval()
                if fitted:model.load_state_dict(ck['best_state'])
                model.requires_grad_(False);name=f's{seed}_{"fitted" if fitted else "initial"}'
                encoders[name]=dict(args=vars(args),state=copy.deepcopy(model.state_dict()),parent_sha256=sha(path),checkpoint_sha256=sha(checkpoint) if fitted else None)
                checks.append(dict(encoder=name,original_core=T.check(model,dev[:8],False)))
                port=NumpyAddressedInference(pack(model));begin=time.perf_counter();features=np.array([R.features(port,r)[0] for r in fit+dev],dtype=np.float64)
                replays.append(dict(encoder=name,targets=len(features),events=sum(len(r['events']) for r in fit+dev),wall_s=time.perf_counter()-begin));arrays[name+'_fit']=features[:len(fit)];arrays[name+'_dev']=features[len(fit):]
                for degree in (1,2):
                    matrix=polynomial_features(features,degree);raw=matrix[:len(fit)];dv=matrix[len(fit):];candidates=[]
                    for strength in (.01,.1,1.):
                        predictions=np.zeros((len(fit),11));folds=[]
                        for training,validation in splits:
                            assert not set(groups[training])&set(groups[validation])
                            center=raw[training].mean(0);scale=np.maximum(raw[training].std(0),1e-6)
                            decoder=LogisticRegression(C=strength,max_iter=1000,tol=1e-6,random_state=681);begin=time.perf_counter();decoder.fit((raw[training]-center)/scale,y[training]);prob=decoder.predict_proba((raw[validation]-center)/scale);predictions[validation]=prob
                            folds.append(dict(wall_s=time.perf_counter()-begin,quality=quality(prob,y[validation]),iterations=decoder.n_iter_.tolist(),validation_users=sorted(set(groups[validation]))))
                        candidates.append(dict(C=strength,quality=quality(predictions,y),folds=folds))
                    selected=min(candidates,key=lambda r:r['quality']['nll']);center=raw.mean(0);scale=np.maximum(raw.std(0),1e-6);decoder=LogisticRegression(C=selected['C'],max_iter=1000,tol=1e-6,random_state=681)
                    begin=time.perf_counter();decoder.fit((raw-center)/scale,y);fit_wall=time.perf_counter()-begin;assert decoder.classes_.tolist()==list(range(11));expected=decoder.predict_proba((dv-center)/scale)
                    folded=FoldedPolynomialReadout(decoder.coef_,decoder.intercept_,center,scale,32,degree)
                    with torch.no_grad():
                        explicit=decoder.decision_function((matrix-center)/scale);actual=folded(torch.from_numpy(features)).numpy()
                    np.testing.assert_allclose(actual,explicit,rtol=1e-9,atol=1e-9)
                    native=copy.deepcopy(model);native.head=folded;predictions=[];first_state=None
                    with torch.no_grad():
                        for i,r in enumerate(dev):
                            logits,state=A.N.predict(native,r,314159,False);predictions.append(logits.softmax(-1).numpy())
                            if i==0:first_state=state
                        mutated=dict(dev[0],target=(dev[0]['target']+1)%11);other,other_state=A.N.predict(native,mutated,314159,False)
                        np.testing.assert_array_equal(other.softmax(-1).numpy(),predictions[0])
                    probabilities=np.array(predictions);max_error=float(np.max(np.abs(probabilities-expected)));assert max_error<=.001
                    reference_quality=quality(expected,yd);actual_quality=quality(probabilities,yd);assert abs(reference_quality['nll']-actual_quality['nll'])<=.005
                    inference=None;times=[]
                    if fitted:
                        with torch.no_grad():
                            inference=[A.N.capture(lambda row=r:A.N.predict(native,row,314159,False)) for r in dev[:11]]
                            for sample in inference:assert sample['formula_coverage_complete']
                            for _ in range(3):
                                begin=time.perf_counter();repeat=np.array([A.N.predict(native,r,314159,False)[0].softmax(-1).numpy() for r in dev]);times.append(time.perf_counter()-begin);np.testing.assert_array_equal(repeat,probabilities)
                    learned_weights=15523-363+11*matrix.shape[1]+11
                    row=dict(encoder=name,seed=seed,fitted_encoder=fitted,degree=degree,features=matrix.shape[1],selected_C=selected['C'],candidates=candidates,final=actual_quality,probabilities=probabilities.tolist(),probe_expected_quality=reference_quality,folded_native_max_probability_error=max_error,head_and_encoder_storage_tensor_bytes=sum(t.numel()*t.element_size() for t in native.state_dict().values()),head_tensor_bytes=sum(t.numel()*t.element_size() for t in folded.state_dict().values()),learned_prediction_weights=learned_weights,parent_native_quality={k:parent['final'][k] for k in ('accuracy','nll')} if fitted else None,parent_encoder_fit_gflops=parent['work']['whole_fit_unit_special_flops_estimate']/1e9 if fitted else 0.,decoder_solver_fitting_flops=None,combined_fitting_flops=None,final_decoder_fit_wall_s=fit_wall,decoder_iterations=decoder.n_iter_.tolist(),sequential_wall_s=times,inference_ledger=inference,inference_mflops_per_query=float(np.mean([r['arithmetic_flops']+r['special_function_evaluations'] for r in inference]))/1e6 if inference else None)
                    rows.append(row);key=name+f'_degree{degree}';estimators[key]=dict(model=decoder,center=center,scale=scale,degree=degree);heads[key]=dict(encoder=name,degree=degree,state=folded.state_dict());print(json.dumps(dict(encoder=name,degree=degree,final=actual_quality)),flush=True)
        warnings_seen=sorted(set(str(w.message) for w in caught))
    quadratic=[r for r in rows if r['fitted_encoder'] and r['degree']==2];affine=[r for r in rows if r['fitted_encoder'] and r['degree']==1];initial=[r for r in rows if not r['fitted_encoder'] and r['degree']==2]
    gains=[x['final']['nll']-y['final']['nll'] for x,y in zip(affine,quadratic)];acc=[y['final']['accuracy']-x['final']['accuracy'] for x,y in zip(affine,quadratic)]
    native_gains=[r['parent_native_quality']['nll']-r['final']['nll'] for r in quadratic];native_acc=[r['final']['accuracy']-r['parent_native_quality']['accuracy'] for r in quadratic];learned_gain=float(np.mean([r['final']['nll'] for r in initial])-np.mean([r['final']['nll'] for r in quadratic]))
    gate=dict(quadratic_over_refit_affine_nll_gains=gains,quadratic_over_refit_affine_accuracy_gains=acc,parent_native_nll_gains=native_gains,parent_native_accuracy_gains=native_acc,mean_fitted_over_initial_quadratic_nll_gain=learned_gain,polynomial_nomination_pass=bool(np.mean(gains)>=.05 and np.mean(acc)>=.03 and min(gains)>=0),staged_predictor_nomination_pass=bool(min(native_gains)>=.05 and min(native_acc)>=-.01 and learned_gain>=.05))
    model_path=out.with_suffix('.models.joblib');joblib.dump(estimators,model_path);predictor_path=out.with_suffix('.predictors.pt');torch.save(dict(encoders=encoders,heads=heads),predictor_path);feature_path=out.with_suffix('.features.npz');np.savez_compressed(feature_path,**arrays,fit_labels=y,dev_labels=yd)
    files=['experiments/aws_frozen_polynomial_probe.py','sleeping_machines/folded_polynomial_readout.py','experiments/theory/aws_20261002_frozen_polynomial_protocol.md','experiments/aws_coarse_readout_probe.py','sleeping_machines/numpy_addressed_inference.py']
    result=dict(status='completed',args=vars(a),rows=rows,gate=gate,numerical_checks=checks,replays=replays,fold_fits=108,final_fits=12,source_sha256={n:sha(ROOT/n) for n in files},artifacts={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in (model_path,predictor_path,feature_path)},warnings=warnings_seen,wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Frozen native encoder plus equivalent local folded readout. All fitting-only head selection, native fit/replay and actual sequential work retained. Combined fitting FLOPs unknown due solver. No end-to-end retraining, test access, sparse resident retrieval or supremacy claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
