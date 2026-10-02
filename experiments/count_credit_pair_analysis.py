"""Matched four-arm attribution of 64-credit fitting in native count-escape models."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_aws_matrix_recovery import validate_result


def analyze(plan):
    rows=[];results={}
    for key,name in plan['comparison_results'].items():
        r=validate_result(ROOT,dict(tag=Path(name).stem,result=name));a=r['args'];results[key]=r
        c=r['work']['cpu_emulator'];e=r['work']['projected_event_architecture'];n=r['work']['fitting_targets']
        rows.append(dict(arm=key,result=name,result_sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),
            bpc=r['final']['dev']['bpc'],fit=a['fit'],development_targets=r['final']['dev']['n'],passes=a['epochs'],
            payload=a['payload'],depth=a['depth'],heads=a['heads'],credit_targets=a['chunk'],parameters=r['parameters'],
            fitting_targets=n,optimizer_updates=r['work']['optimizer_steps'],
            cpu_whole_fit_gflops=c['total_training_unit_special_flops']/1e9,
            cpu_fit_mflops_per_target=c['total_training_unit_special_flops']/n/1e6,
            cpu_inference_mflops_per_character=(c['inference_arithmetic_flops_per_character']+c['inference_special_functions_per_character'])/1e6,
            projected_whole_fit_gflops=e['total_training_unit_special_flops']/1e9,
            projected_fit_mflops_per_target=e['total_training_unit_special_flops']/n/1e6,
            projected_inference_mflops_per_character=(e['inference_arithmetic_flops_per_character']+e['inference_special_functions_per_character'])/1e6,
            core_receiver_slots=a['depth']*a['heads']*a['pool'],
            selected_core_updates_per_target=r['work']['actual_fitting_activity']['receiver_updates']/n,
            core_key_scores_per_target=r['work']['actual_fitting_activity']['receiver_scores']/n,
            counterfactual_values_per_target=r['work']['actual_fitting_activity']['receiver_teacher_values']/n,
            count_work=r['count_receivers'],actual_activity=r['work']['actual_fitting_activity'],
            wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb']))
    template=results['full16'];skip={'tag','payload','depth','chunk'}
    canonical=lambda r:{k:v for k,v in r['args'].items() if k not in skip}
    for key,r in results.items():
        if canonical(r)!=canonical(template):raise ValueError('Unexpected fitting/optimizer protocol difference: '+key)
        if any(r[k]!=template[k] for k in ('fitting_data_sha256','development_data_sha256')):raise ValueError('Data mismatch')
        if any(r['work'][k]!=template['work'][k] for k in ('fitting_targets','optimizer_steps')):raise ValueError('Target/update budget mismatch')
        if (r['args']['payload'],r['args']['depth'])!=((16,8) if key.startswith('full') else (2,1)):raise ValueError('Core configuration mismatch')
        if r['args']['chunk']!=(16 if key.endswith('16') else 64):raise ValueError('Credit configuration mismatch')
    score=lambda key:results[key]['final']['dev']['bpc']
    baseline_gap=score('minimal16')-score('full16');long_gap=score('minimal64')-score('full64')
    core_gain=score('full16')-score('full64');minimal_gain=score('minimal16')-score('minimal64')
    cost=lambda key:results[key]['work']['cpu_emulator']['total_training_unit_special_flops']
    gate=long_gap>=.02 and core_gain>=.02 and cost('full64')/cost('full16')<=2
    return dict(status='completed',kind='exploratory_native_credit_core_attribution',common_unit_ledger=rows,
        full_core_credit_gain_bpc=core_gain,minimal_core_credit_gain_bpc=minimal_gain,
        full_core_advantage_bpc_at16=baseline_gap,full_core_advantage_bpc_at64=long_gap,
        difference_in_credit_gains_bpc=core_gain-minimal_gain,
        full_core_fitting_work_ratio=cost('full64')/cost('full16'),followup_gate_passed=gate,
        gate=dict(minimum_full_vs_minimal_gain_bpc=.02,minimum_full_credit_gain_bpc=.02,maximum_full_fitting_work_ratio=2),
        scope='Same seed/data/pass/update budgets; core capacity and graph reach are explicit interventions. Reused development population, not confirmation or supremacy. Gate only nominates a fresh matched follow-up; no automatic scale-up.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    path=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or path.exists():raise ValueError('Unused tag required')
    started=time.perf_counter();plan=json.loads((ROOT/a.plan).read_text());result=analyze(plan)
    result.update(args=vars(a),wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k.endswith('bpc') or k=='followup_gate_passed'}),flush=True)


if __name__=='__main__':main()
