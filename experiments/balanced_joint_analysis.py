"""Completed-only comparison of paired nonlocal learning and full work."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Fresh unique plain tag required')
    started=time.perf_counter();plan=json.loads((ROOT/a.plan).read_text());rows=[];loaded={};common=None
    for job in plan['jobs']:
        if job['stage']!='pilot':continue
        path=ROOT/job['result'];r=json.loads(path.read_text())
        if r['status']!='completed' or r['args']['tag']!=job['tag']:raise ValueError('Completed matching pilot required')
        for name,sha in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Pilot source changed')
        args=r['args'];protocol={k:args[k] for k in ['gap','fit_groups','dev_groups','epochs','payload','heads','pool','seed','lr',
            'update_targets','fit_data_seed','dev_data_seed']}
        protocol.update(fit_hash=r['fitting_data_sha256'],dev_hash=r['development_data_sha256'])
        if common is None:common=protocol
        elif common!=protocol:raise ValueError('Unmatched data/learning settings')
        w=r['work'];final=r['final'];activity=r['activity']
        if w['fitting_targets']!=4*args['fit_groups']*args['epochs']:raise ValueError('Wrong fitting target budget')
        if not all(abs(c['query_bits']-1)<1e-12 for c in r['counts']['rows']):raise ValueError('Count control contract failed')
        if w['formula_coverage_complete'] is not True:raise ValueError('Incomplete accounting')
        arm=job['arm'];loaded[arm]=r
        rows.append(dict(arm=arm,result=job['result'],result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            model=args['model'],depth=args['depth'],payload=args['payload'],parameters=r['parameters'],
            query_bits=final['query_bits'],accuracy=final['accuracy'],selected_epoch=r['selected_epoch'],
            exploratory_dependency_gate_passed=r['exploratory_dependency_gate_passed'],
            fitting_targets=w['fitting_targets'],optimizer_updates=w['optimizer_updates'],
            whole_fit_gflops_estimate=w['whole_fit_unit_special_flops_estimate']/1e9,
            fit_mflops_per_target_estimate=w['fit_unit_special_flops_per_target_estimate']/1e6,
            inference_mflops_per_target=w['inference_unit_special_flops_per_target']/1e6,
            whole_fit_arithmetic_gflops_estimate=w['whole_fit_arithmetic_flops_estimate']/1e9,
            fitting_special_evaluations_estimate=w['whole_fit_special_evaluations_estimate'],
            input_events=activity['input_events'],key_scores=activity['key_scores'],selected_updates=activity['selected_updates'],
            counterfactual_values=activity['counterfactual_values'],state_tensor_bytes=final['max_state_tensor_bytes'],
            evaluation_input_events=r['evaluation_input_events'],wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb'],
            curve=[dict(epoch=x['epoch'],query_bits=x['dev']['query_bits'],accuracy=x['dev']['accuracy'],
                fitting_query_bits=x['fitting_query_bits']) for x in r['curve']]))
    if set(loaded)!=set(['native_full','tapped_full','tapped_shallow']):raise ValueError('All three pilots required')
    native,tapped,shallow=[loaded[n] for n in ['native_full','tapped_full','tapped_shallow']]
    for field in ('query_bits','accuracy','per_target_bits','class1_probabilities','input_events','key_scores','selected_updates'):
        if native['initial_dev'][field]!=tapped['initial_dev'][field]:raise ValueError('Tap nesting or noise mismatch at initialization')
    gains={name:control['final']['query_bits']-tapped['final']['query_bits'] for name,control in
        [('tapped_vs_native_bits',native),('tapped_vs_shallow_bits',shallow)]}
    ratios={name:tapped['work']['whole_fit_unit_special_flops_estimate']/control['work']['whole_fit_unit_special_flops_estimate']
        for name,control in [('tapped_native_fitting_work_ratio',native),('tapped_shallow_fitting_work_ratio',shallow)]}
    rng=np.random.default_rng(391);pairs={}
    for name,control in [('native',native),('shallow',shallow)]:
        d=(np.asarray(control['final']['per_target_bits'])-np.asarray(tapped['final']['per_target_bits'])).reshape(-1,4).mean(-1)
        bootstrap=d[rng.integers(0,len(d),size=(2000,len(d)))].mean(-1)
        pairs[name]=dict(mean_bits=float(d.mean()),group_bootstrap_95_percentile_interval=np.quantile(bootstrap,[.025,.975]).tolist(),
            groups=len(d),scope='Exploratory paired suffix-group uncertainty, conditional on one selected fitted seed; not selection-adjusted or confirmation')
    nominated=tapped['exploratory_dependency_gate_passed'] and all(v>=.05 for v in gains.values())
    result=dict(status='completed',args=vars(a),matched_protocol=common,common_unit_ledger=rows,
        **gains,**ratios,tapped_extension_nomination_gate_passed=nominated,paired_groups=pairs,
        any_integrated_dependency_gate_passed=any(r['exploratory_dependency_gate_passed'] for r in loaded.values()),
        restricted_query_count_lower_bound_bits=1.,count_controls=next(iter(loaded.values()))['counts'],
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={'experiments/balanced_joint_analysis.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='One-seed held-out paired dependency demonstration if gate passes. Full prefix/core/candidate/losing-value/optimizer work charged as marked estimates; no dense comparison, full count-table inspection, semantic language, iso-quality total-resource or supremacy claim. All unsuccessful arms retained.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k in
        ['tapped_vs_native_bits','tapped_vs_shallow_bits','tapped_extension_nomination_gate_passed','any_integrated_dependency_gate_passed']}),flush=True)


if __name__=='__main__':main()
