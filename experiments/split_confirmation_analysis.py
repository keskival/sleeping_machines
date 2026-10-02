"""Prespecified crossed seed/population confirmation; no best-seed selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_aws_matrix_recovery import validate_result


def comparison(ours,control,paired_timing=False,draws=10000):
    if sorted(r['args']['seed'] for r in ours)!=[6,7,8] or sorted(r['args']['seed'] for r in control)!=[6,7,8]:
        raise ValueError('All prespecified seeds6/7/8 required')
    ours=sorted(ours,key=lambda r:r['args']['seed']);control=sorted(control,key=lambda r:r['args']['seed'])
    values=[]
    for a,b in zip(ours,control):
        for row in (a,b):
            if (row.get('status')!='completed' or row['protocol']['confirmation_data_seed']!=3201
                    or not row['protocol']['synthetic_holdout_read'] or row['final']['confirmation']['n']!=1024):
                raise ValueError('Completed untouched1024-query confirmation required')
            if row['data_sha256']!=ours[0]['data_sha256']:raise ValueError('Paired fitting/development data differ')
            if (row['args']['task'],row['args']['sources'])!=(ours[0]['args']['task'],ours[0]['args']['sources']):
                raise ValueError('Paired task/source identities differ')
            scores=np.asarray(row['final']['confirmation']['episode_accuracy'],dtype=float)
            if (scores.shape!=(1024//row['args']['sources'],) or not np.isfinite(scores).all()
                    or ((scores<0)|(scores>1)).any()
                    or abs(scores.mean()-row['final']['confirmation']['accuracy'])>1e-12):
                raise ValueError('Confirmation population scores inconsistent')
        x=np.asarray(a['final']['confirmation']['episode_accuracy'],dtype=float)
        y=np.asarray(b['final']['confirmation']['episode_accuracy'],dtype=float)
        if x.shape!=y.shape:raise ValueError('Paired confirmation populations differ')
        if paired_timing:
            x,y=x.reshape(-1,2).mean(1),y.reshape(-1,2).mean(1)
        values.append(x-y)
    delta=np.asarray(values);rng=np.random.default_rng(7319);samples=[]
    for _ in range(draws):
        seed_ids=rng.integers(0,3,3);population_ids=rng.integers(0,delta.shape[1],delta.shape[1])
        samples.append(float(delta[seed_ids][:,population_ids].mean()))
    return dict(mean_accuracy_gain_pp=float(delta.mean()*100),
        accuracy_gain_pp_by_seed=(delta.mean(1)*100).tolist(),
        adjusted_97_5_interval_pp=(np.quantile(samples,(.0125,.9875))*100).tolist(),
        independent_populations=delta.shape[1],seeds=3,
        ours_accuracy_by_seed=[r['final']['confirmation']['accuracy'] for r in ours],
        control_accuracy_by_seed=[r['final']['confirmation']['accuracy'] for r in control],
        whole_fit_work_ratio_by_seed=[a['work']['total_training_unit_special_flops']/b['work']['total_training_unit_special_flops'] for a,b in zip(ours,control)],
        scope='Crossed seed/population bootstrap on one fixed synthetic split; only three seeds. '
              'Bonferroni across two predeclared accuracy contrasts; timing keeps whole short/long pairs together. '
              'No broad task-class, real-data or energy supremacy follows.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite completed evidence')
    root=Path(__file__).resolve().parents[1];plan=json.loads(a.manifest.read_text());cells={}
    for job in plan['jobs']:
        if job['stage']=='pilot':cells.setdefault(job['variant'],[]).append(validate_result(root,job))
    timing=comparison(cells['paired_timing_observed'],cells['paired_timing_rank'],True)
    sharing=comparison(cells['order_S16_shared'],cells['order_S16_private'])
    gates=dict(timing=timing['adjusted_97_5_interval_pp'][0]>=25
        and min(timing['ours_accuracy_by_seed'])>=.85,
        sharing=sharing['adjusted_97_5_interval_pp'][0]>0
        and min(sharing['accuracy_gain_pp_by_seed'])>0
        and max(sharing['whole_fit_work_ratio_by_seed'])<=1.)
    a.output.write_text(json.dumps(dict(status='completed',timing=timing,sharing=sharing,
        promotion_gates=gates,promotion='Passing admits a separately frozen larger capability test; no automatic launch'),indent=2)+'\n')


if __name__=='__main__':main()
