"""Frozen learned magnitude priorities with legal importance correction."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
from aws_replay_importance_probe import contract,credit


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();started=time.perf_counter();torch.set_num_threads(1)
    parent=ROOT/'experiments/results/diagnostics/aws_signed_replay_variance_20261002T230400Z.json';j=json.loads(parent.read_text())
    for n,h in j['source_sha256'].items(): assert N.sha(ROOT/n)==h,n
    bankfile=parent.with_suffix('.pt');assert N.sha(bankfile)==j['artifact_sha256'];banks=torch.load(bankfile,weights_only=False)
    rows=[];models={}
    for seed in (7,8):
        examples=banks[str(seed)]['examples'];features=[f.reshape(len(f),-1).numpy() for f,pi,u in examples]
        energies=[credit(pi,u).square().sum(-1).numpy() for f,pi,u in examples]
        norms=[np.sqrt(x) for x in energies];floor=max(float(np.median(np.concatenate(norms[:32])))*.001,1e-12)
        predictor=ExtraTreesRegressor(n_estimators=64,max_depth=6,min_samples_leaf=8,random_state=681,n_jobs=1)
        predictor.fit(np.concatenate(features[:32]),np.log(np.concatenate(norms[:32])+floor));models[str(seed)]=dict(predictor=predictor,floor=floor)
        mse=base=oracle=uniform=0.;case=[]
        for index,(x,e,norm) in enumerate(zip(features[32:],energies[32:],norms[32:]),32):
            prediction=np.maximum(np.exp(predictor.predict(x))-floor,0.);prediction=np.maximum(prediction,floor)
            proposal=.9*prediction/prediction.sum()+.1/len(prediction)
            assert proposal.min()>0 and abs(proposal.sum()-1)<1e-12
            v=float(((e/proposal).sum()-e.sum())/2);b=float((len(e)/4-1)*e.sum())
            # Oracle sees all expensive utilities: a diagnostic lower bound, never a deployable predictor.
            o=float((norm.sum()**2-e.sum())/2);mse+=v;base+=b;oracle+=o;uniform+=float((len(e)-1)*e.sum()/2)
            case.append(dict(fit_index=index,k2_importance_mse=v,plain_k4_mse=b,oracle_k2_mse=o,proposal=proposal.tolist(),effective_sites=float(norm.sum()**2/e.sum())))
        ratio=mse/base;rows.append(dict(seed=seed,heldout_k2_vs_plaink4_ratio=ratio,oracle_k2_vs_plaink4_ratio=oracle/base,
            uniform_with_replacement_k2_vs_plaink4_ratio=uniform/base,nomination_passed=ratio<=1,training_norm_floor=floor,cases=case))
    out=ROOT/'experiments/results/diagnostics'/f'{a.tag}.json';assert not out.exists();modelpath=out.with_suffix('.models.joblib');joblib.dump(models,modelpath)
    result=dict(status='completed',tag=a.tag,contract=contract(),rows=rows,parent_sha256=N.sha(parent),bank_sha256=N.sha(bankfile),
        artifacts={str(modelpath.relative_to(ROOT)):dict(sha256=N.sha(modelpath),bytes=modelpath.stat().st_size)},
        source_sha256={**j['source_sha256'],'experiments/aws_replay_importance_probe.py':N.sha(ROOT/'experiments/aws_replay_importance_probe.py'),'experiments/aws_replay_priority_predictor_probe.py':N.sha(Path(__file__))},
        scope='Conditional score-space correction variance only; first32 FIT train learned magnitude, next32 holdout. Oracle requires all replay targets and is diagnostic, not a practical advantage.',
        fitting_flops=None,wall_s=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='cases'} for r in rows],indent=2))


if __name__=='__main__':main()
