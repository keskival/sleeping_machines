"""Exact native affine demand-readout contracts and full-prefix operation ledger."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_coarse_native as A
from sleeping_machines.query_only_native_readout import install,predict


def equal(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif hasattr(a,'__dict__'): equal(vars(a),vars(b))
    else: assert a==b


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);a=parser.parse_args();started=time.perf_counter();torch.set_num_threads(1);rows=[]
    for seed in (6,7,8):
        path=ROOT/f'experiments/results/dvs_native/aws_full_coarse_20261002T213100Z_coarse_matchedclock_s{seed}.json';parent=json.loads(path.read_text())
        for n,h in parent['source_sha256'].items():assert A.N.sha(ROOT/n)==h,n
        args=argparse.Namespace(**parent['args']);_,dev,data=A.load(args);model=A.C.make_model(args);model.eval()
        checkpoint=path.with_suffix('.progress.pt');saved=torch.load(checkpoint,weights_only=False,map_location='cpu');model.load_state_dict(saved['best_state']);optimized=install(model)
        equal(model.state_dict(),optimized.state_dict());audits={'baseline':[],'query_only':[]};probs=[]
        with torch.no_grad():
            for index,row in enumerate(dev):
                before,state=A.N.predict(model,row,314159,False);after,other=predict(optimized,row);equal(before,after);equal(state,other);probs.append(after.softmax(-1).tolist())
                if index<11:
                    audits['baseline'].append(A.N.capture(lambda:A.N.predict(model,row,314159,False)))
                    audits['query_only'].append(A.N.capture(lambda:predict(optimized,row)))
        mutated=dict(dev[0],target=(dev[0]['target']+1)%11);equal(predict(optimized,mutated)[0],predict(optimized,dev[0])[0])
        try:predict(optimized,dict(dev[0],events=dev[0]['events'][:-1]))
        except ValueError:pass
        else:raise AssertionError('Missing query must fail')
        try:optimized.train();optimized.head(torch.zeros(32))
        except RuntimeError:pass
        else:raise AssertionError('Training must fail')
        optimized.eval();walls={'baseline':[],'query_only':[]}
        np.testing.assert_allclose(np.array(probs),np.array(parent['final']['probabilities']),rtol=1e-4,atol=1e-6)
        with torch.no_grad():
            for repeat in range(3):
                for name in (('baseline','query_only') if repeat%2==0 else ('query_only','baseline')):
                    start=time.perf_counter()
                    result=np.array([(A.N.predict(model,r,314159,False) if name=='baseline' else predict(optimized,r))[0].softmax(-1).numpy() for r in dev])
                    walls[name].append(time.perf_counter()-start);np.testing.assert_array_equal(result,np.array(probs,dtype=result.dtype))
        work={name:float(np.mean([x['arithmetic_flops']+x['special_function_evaluations'] for x in ledger])) for name,ledger in audits.items()}
        assert all(x['formula_coverage_complete'] for ledger in audits.values() for x in ledger)
        rows.append(dict(seed=seed,quality=parent['final']['nll'],accuracy=parent['final']['accuracy'],bitwise_probability_state_checks=len(dev),probabilities=probs,
            whole_prefix_arithmetic_plus_special_ops=work,reduction_fraction=1-work['query_only']/work['baseline'],
            baseline_classifier_calls=5,query_classifier_calls=1,audits=audits,three_repeat_sequential_wall_s=walls,
            parent_fit_gflops=parent['work']['whole_fit_unit_special_flops_estimate']/1e9,parent_result_sha256=A.N.sha(path),checkpoint_sha256=A.N.sha(checkpoint)))
        print(json.dumps({k:v for k,v in rows[-1].items() if k not in ('audits','probabilities')}),flush=True)
    out=ROOT/'experiments/results/diagnostics'/f'{a.tag}.json';assert not out.exists()
    files=['experiments/aws_query_only_native_probe.py','sleeping_machines/query_only_native_readout.py']
    result=dict(status='completed',tag=a.tag,rows=rows,source_sha256={**parent['source_sha256'],**{f:A.N.sha(ROOT/f) for f in files}},
        scope='Exact inference arithmetic optimization only, unchanged model state/quality/training; no quality or whole-fit superiority. Memory traffic/energy not measured.',
        wall_s=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
