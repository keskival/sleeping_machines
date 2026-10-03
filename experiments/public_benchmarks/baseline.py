"""Matched archive TRAIN/DEV baseline, never official TEST during screening.

Nearest-neighbor with fixed linear resampling; diagnostic control, not native.
"""
import argparse,json,sys,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.public_benchmarks.data import load_train,sha

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--tag',required=True);a=p.parse_args()
    started=time.perf_counter();manifest=json.loads((ROOT/'experiments/public_benchmarks/data_manifest.json').read_text());entry=next(r for r in manifest['datasets'] if r['dataset']==a.dataset)
    rows,mapping,_=load_train(a.dataset);assert sha(ROOT/'data/public_benchmarks/raw'/f'{a.dataset}_TRAIN.ts')==entry['train_sha256']
    fit=[rows[i] for i in entry['fit_indices']];dev=[rows[i] for i in entry['dev_indices']]
    length=max(len(r['values']) for r in fit)
    def features(data):
        result=[]
        for r in data:
            x=np.asarray(r['values']);result.append(np.stack([np.interp(np.linspace(0,1,length),np.linspace(0,1,len(x)),x[:,c]) for c in range(x.shape[1])],axis=1).reshape(-1))
        return np.asarray(result)
    x=features(fit);y=features(dev);center=x.mean(0);scale=np.maximum(x.std(0),1e-6);x=(x-center)/scale;y=(y-center)/scale
    labels=np.array([r['target'] for r in fit]);all_prob=[]
    for vector in y:
        distances=((x-vector)**2).sum(1);selected=int(np.argmin(distances));prob=np.full(len(mapping),1e-6);prob[labels[selected]]=1.;prob/=prob.sum();all_prob.append(prob.tolist())
    targets=np.array([r['target'] for r in dev]);prob=np.asarray(all_prob)
    result=dict(status='completed',args=vars(a),model='1NN resampled Euclidean diagnostic control',
        development=dict(targets=len(dev),accuracy=float(np.mean(prob.argmax(1)==targets)),nll=float(-np.log(prob[np.arange(len(dev)),targets]).mean()),probabilities=all_prob,identities=[r['identity'] for r in dev]),
        test=None,test_access='None',training_examples=len(fit),stored_feature_bytes=x.nbytes+labels.nbytes,
        work=dict(distance_arithmetic_flops_per_query_estimate=3*x.size,
                  scope='Subtract/square/reduction estimate only; normalization/resampling/selection not counted. Whole-fit FLOPs unknown; no complete resource claim.'),
        source_sha256={'experiments/public_benchmarks/baseline.py':sha(Path(__file__)),'experiments/public_benchmarks/data.py':sha(ROOT/'experiments/public_benchmarks/data.py')},
        data_manifest_sha256=sha(ROOT/'experiments/public_benchmarks/data_manifest.json'),wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out=ROOT/'experiments/results/public_benchmarks'/f'{a.tag}.json';out.parent.mkdir(exist_ok=True);assert not out.exists();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',dev_accuracy=result['development']['accuracy'])))
