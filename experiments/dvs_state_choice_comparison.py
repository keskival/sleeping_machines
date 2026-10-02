"""Completed-only matched write-choice comparison; frozen predecessor preserved."""
import argparse
import json
from pathlib import Path
import resource
import time
from dvs_credit_comparison import completed, sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--treatment',required=True)
    a=p.parse_args();start=time.perf_counter();out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    local,treatment=completed(a.reference),completed(a.treatment)
    for key in ('data','parameters','initial_dev','initial_fit','window_size_counts'):
        if local[key]!=treatment[key]:raise ValueError('Different '+key)
    for key in ('payload','depth','heads','pool','seed','epochs','fit','dev','lr','update_targets'):
        if local['args'][key]!=treatment['args'][key]:raise ValueError('Different setting '+key)
    s=local['args']
    if (s['fit'],s['dev'],s['epochs'],s['update_targets'],s['seed']) not in [(256,192,4,16,6),(256,192,4,16,7)]:
        raise ValueError('Fixed pilot/confirmation protocol required')
    if local['args'].get('terminal_risk')!='local':raise ValueError('Local reference required')
    protocol=treatment['credit_protocol']
    if protocol['kind']!='state_choice' or not protocol['timing_gradient_retained'] or not protocol['inference_architecture_unchanged']:
        raise ValueError('Exact write-choice/native timing protocol required')
    ledger=[]
    for label,name,r in [('local',a.reference,local),('state_choice',a.treatment,treatment)]:
        selected=min(r['curve'],key=lambda x:x['dev']['nll'])
        if len(r['curve'])!=4 or selected['epoch']!=r['selected_epoch'] or selected['dev']!=r['final']:
            raise ValueError('Changed fixed selection')
        if not all(stage['formula_coverage_complete'] and not stage['unsupported_floating_operators']
            for sample in r['work_samples'] for stage in sample['stages'].values()):raise ValueError('Incomplete work')
        w=r['work'];n=w['fitting_targets']
        if n!=1024 or w['optimizer_updates']!=64 or r['window_size_counts']!={'16':64}:raise ValueError('Different exposure')
        if abs(w['whole_fit_unit_special_flops_estimate']/n-w['fit_unit_special_flops_per_target_estimate'])>1e-6:
            raise ValueError('Different work denominator')
        ledger.append(dict(arm=label,result=name,result_sha256=sha(name),distinct_fit_targets=256,fit_presentations=n,
            development_accuracy=r['final']['accuracy'],development_nll=r['final']['nll'],
            whole_fit_gflops_estimate=w['whole_fit_unit_special_flops_estimate']/1e9,
            fit_mflops_per_presentation_estimate=w['fit_unit_special_flops_per_target_estimate']/1e6,
            inference_mflops_per_target_estimate=w['inference_unit_special_flops_per_target_estimate']/1e6,
            parameters=r['parameters'],available_receivers=w['native_available_receivers'],
            key_scores_per_fit_presentation=r['activity']['key_scores']/n,
            selected_updates_per_fit_presentation=r['activity']['selected_updates']/n,
            counterfactual_values_per_fit_presentation=r['activity']['counterfactual_values']/n,
            curve=[dict(epoch=x['epoch'],nll=x['dev']['nll'],accuracy=x['dev']['accuracy']) for x in r['curve']],
            wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb']))
    gain=ledger[0]['development_nll']-ledger[1]['development_nll']
    decline=100*(ledger[0]['development_accuracy']-ledger[1]['development_accuracy'])
    ratio=ledger[1]['whole_fit_gflops_estimate']/ledger[0]['whole_fit_gflops_estimate']
    result=dict(status='completed',args={**vars(a),'kind':'state_choice','seed':s['seed']},common_unit_ledger=ledger,
        nll_improvement=gain,accuracy_decline_percentage_points=decline,whole_fit_work_ratio=ratio,
        promotion_gate_passed=gain>=.02 and decline<=1 and ratio<=1.5 and max(x['max_rss_kb'] for x in ledger)<900000,
        gate=dict(minimum_nll_improvement=.02,maximum_accuracy_decline_pp=1,maximum_work_ratio=1.5,maximum_rss_kb=900000),
        source_sha256={name:sha(name) for name in ['experiments/dvs_state_choice_comparison.py','experiments/dvs_credit_comparison.py']},
        wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Matched fixed256-fit/192-dev/four-pass screen; actual alternative writes/suffix paid, native timing kept. Full984-fit controls unequal data. Conditional choice exactness is not whole-history or downstream timing-jump exactness. No official test or practical supremacy.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
