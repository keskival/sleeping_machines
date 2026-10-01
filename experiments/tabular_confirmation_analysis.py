"""Paired banknote evidence: cross seeds and feature groups, never pool as iid."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired_summary(ours, control, draws=4000, alpha=.05/3):
    if len(ours)!=3 or len(control)!=3:
        raise ValueError('All three prespecified seeds are required')
    ours=sorted(ours,key=lambda r:r['args']['seed'])
    control=sorted(control,key=lambda r:r['args']['seed'])
    if [r['args']['seed'] for r in ours]!=[6,7,8] or [r['args']['seed'] for r in control]!=[6,7,8]:
        raise ValueError('Prespecified seed identities differ')
    reference=ours[0]['final']['test']['rows']
    nll=[];acc=[]
    for a,b in zip(ours,control):
        for r in (a,b):
            if (r.get('status')!='completed' or not r['protocol']['test_labels_scored']
                    or not r['protocol']['weights_frozen_before_test']):
                raise ValueError('Completed frozen test scores required')
            row=r['final']['test']['rows']
            for k in ('indices','targets','groups'):
                if row[k]!=reference[k]:raise ValueError('Paired row identity differs: '+k)
            for k in ('fit_sha256','dev_sha256','test_sha256'):
                if r['protocol'][k]!=ours[0]['protocol'][k]:raise ValueError('Paired data differ: '+k)
            lp=np.asarray(row['log_probabilities'],dtype=float)
            losses=-lp[np.arange(len(row['targets'])),row['targets']]
            if not np.isfinite(lp).all() or not np.allclose(np.exp(lp).sum(1),1,atol=1e-6):
                raise ValueError('Invalid saved probabilities')
            if not np.allclose(losses,row['losses'],atol=1e-6):raise ValueError('Saved losses differ')
            if not np.array_equal(lp.argmax(1)==row['targets'],row['correct']):raise ValueError('Saved hits differ')
        nll.append(np.asarray(b['final']['test']['rows']['losses'])-a['final']['test']['rows']['losses'])
        acc.append(np.asarray(a['final']['test']['rows']['correct'],dtype=float)
                   -np.asarray(b['final']['test']['rows']['correct'],dtype=float))
    nll,acc=np.asarray(nll),np.asarray(acc)
    groups=np.asarray(reference['groups']);unique=np.unique(groups)
    if len(unique)<2:raise ValueError('At least two independent groups required')
    # Same resampled row groups for every seed; the data are shared across fits.
    members=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.default_rng(4917);nll_samples=[];acc_samples=[]
    for _ in range(draws):
        seeds=rng.integers(0,3,3)
        ids=np.concatenate([members[g] for g in rng.integers(0,len(unique),len(unique))])
        nll_samples.append(float(nll[seeds][:,ids].mean()))
        acc_samples.append(float(acc[seeds][:,ids].mean()))
    return dict(mean_control_minus_ours_nll=float(nll.mean()),mean_ours_minus_control_accuracy=float(acc.mean()),
        nll_interval=np.quantile(nll_samples,(alpha/2,1-alpha/2)).tolist(),
        nll_interval_level=1-alpha,accuracy_descriptive_95_interval=np.quantile(acc_samples,(.025,.975)).tolist(),
        paired_nll_by_seed=nll.mean(1).tolist(),paired_accuracy_by_seed=acc.mean(1).tolist(),
        seeds=3,test_rows=len(groups),independent_feature_groups=len(unique),bootstrap_draws=draws,
        scope='Crossed seed/group bootstrap on one fixed data split; approximate uncertainty with only three seeds. '
              'NLL intervals use Bonferroni for three control comparisons; accuracy intervals are descriptive. '
              'No ensemble, test-time model selection or independent-data-split replication.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    a=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/a.output
    if out.exists():raise ValueError('Preserve prior analysis')
    plan=json.loads((root/a.manifest).read_text());families={k:[] for k in ('ours','trees','catboost','logistic')}
    for job in plan['jobs']:
        if job['stage']!='pilot':continue
        r=json.loads((root/job['result']).read_text())
        if r['args']['tag']!=job['tag']:raise ValueError('Result identity differs')
        families[r['args']['model']].append(r)
    comparisons={k:paired_summary(families['ours'],families[k]) for k in ('trees','catboost','logistic')}
    result=dict(status='completed',comparisons=comparisons,
        qualified_primary_gate=all(v['nll_interval'][0]>0 for v in comparisons.values()),
        scope='Prespecified single-dataset quality confirmation. Passing is evidence for this protocol, '
              'not broad tabular supremacy or a physical compute/energy advantage.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
