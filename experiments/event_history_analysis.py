"""Prespecified three-contrast crossed seed/population accuracy analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.run_aws_matrix_recovery import validate_result

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=ROOT/a.output
    if out.exists():raise ValueError('Never overwrite evidence')
    plan=json.loads((ROOT/a.manifest).read_text());groups={};rows=[];reference=None
    for job in plan['jobs']:
        if job['stage']!='pilot':continue
        d=validate_result(ROOT,job);model=d['args']['model'];seed=d['args']['seed'];q=d['final']['confirmation'];w=d['work']
        if d['protocol']['confirmation_seed']!=4201 or q['n']!=4096 or w['fitting_query_targets']!=2048:
            raise ValueError('Declared confirmation/fitting boundaries required')
        hashes=(d['protocol']['fit_sha256'],d['protocol']['dev_sha256'])
        if reference is None:reference=hashes
        if hashes!=reference:raise ValueError('Paired data identities differ')
        scores=np.asarray(q['episode_accuracy'],dtype=float)
        if scores.shape!=(64,) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any() or abs(scores.mean()-q['accuracy'])>1e-12:
            raise ValueError('Complete population scores required')
        row=dict(model=model,seed=seed,result=job['result'],result_sha256=hashlib.sha256((ROOT/job['result']).read_bytes()).hexdigest(),
            accuracy=q['accuracy'],nll=q['nll'],episode_accuracy=scores.tolist(),parameters=d['parameters'],
            whole_fit_gflops=w['total_training_arithmetic_flops']/1e9,
            fit_mflops_per_query=w['total_training_arithmetic_flops']/2048/1e6,
            inference_mflops_per_query=w['inference_arithmetic_flops_per_query']/1e6,
            training_special_function_evaluations=w['training_special_function_evaluations'],
            historical_native_fit_wall_s=d.get('historical_fitting_wall_s'),current_job_wall_s=d['wall_s'])
        groups.setdefault(model,[]).append(row);rows.append(row)
    for model in ('native_shared','native_private','history32','history128'):
        if sorted(r['seed'] for r in groups.get(model,[]))!=[6,7,8]:raise ValueError('All model/seed results required')
        groups[model].sort(key=lambda r:r['seed'])
    contrasts={}
    for control in ('native_private','history32','history128'):
        ours=groups['native_shared'];other=groups[control]
        delta=np.asarray([np.asarray(x['episode_accuracy'])-y['episode_accuracy'] for x,y in zip(ours,other)])
        rng=np.random.default_rng(7319);samples=[]
        for _ in range(10000):
            seeds=rng.integers(0,3,3);populations=rng.integers(0,64,64)
            samples.append(float(delta[seeds][:,populations].mean()))
        contrasts[control]=dict(mean_accuracy_gain_pp=float(delta.mean()*100),
            adjusted_98_33_interval_pp=(np.quantile(samples,(.05/6,1-.05/6))*100).tolist(),
            accuracy_gain_pp_by_seed=(delta.mean(1)*100).tolist(),
            control_minus_shared_nll_by_seed=[y['nll']-x['nll'] for x,y in zip(ours,other)],
            whole_fit_arithmetic_ratio_by_seed=[x['whole_fit_gflops']/y['whole_fit_gflops'] for x,y in zip(ours,other)])
    private=contrasts['native_private'];gate=private['adjusted_98_33_interval_pp'][0]>=-1 and max(private['whole_fit_arithmetic_ratio_by_seed'])<=.85
    result=dict(status='completed',rows=rows,contrasts=contrasts,intra_family_resource_gate=gate,
        scope='Three seeds crossed with64 whole fresh populations; only three training seeds and one synthetic distribution. History bound3 is prior knowledge, decoder uses timestamps/query flag and source-local storage. Historical native fitting charged, current evaluation separate. Accuracy margin1pp and15% arithmetic reduction gate scoped within native family. No broad-domain, iso-information-prior or physical-energy supremacy claim.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(intra_family_resource_gate=gate,contrasts=contrasts)))


if __name__=='__main__':main()
