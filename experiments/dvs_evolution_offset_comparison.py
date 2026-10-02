"""Completed, fixed-protocol native/joint-offset/alternating-offset quality gate."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from dvs_credit_comparison import completed,sha


def analyze(reference,treatments):
    base=completed(reference);rows=[]
    entries=[('local',reference,base)]+[(completed(p)['args']['update_schedule'],p,completed(p)) for p in treatments]
    schedules=[x[0] for x in entries[1:]]
    if len(set(schedules))!=len(schedules) or any(s not in ('joint','alternating') for s in schedules):raise ValueError('Distinct known schedules required')
    if base['args'].get('terminal_risk')!='local' or base['parameters']!=15523:raise ValueError('Original native local control required')
    if (base['args']['fit'],base['args']['dev'],base['args']['epochs'],base['args']['update_targets'])!=(256,192,4,16):raise ValueError('Fixed pilot required')
    if base['args']['seed'] not in (6,7):raise ValueError('Declared seeds required')
    for arm,path,r in entries:
        for key in ('data','initial_fit','initial_dev','window_size_counts'):
            if r[key]!=base[key]:raise ValueError('Unmatched '+key)
        for key in ('payload','depth','heads','pool','seed','epochs','fit','dev','lr','update_targets'):
            if r['args'][key]!=base['args'][key]:raise ValueError('Unmatched '+key)
        if arm!='local':
            protocol=r.get('evolution_offset_protocol',{})
            if r['parameters']!=15555 or protocol.get('schedule')!=arm or protocol.get('independent_signal_clock') is not False:
                raise ValueError('Coupled physical-time offset protocol required')
        if len(r['curve'])!=4 or r['selected_epoch']!=min(r['curve'],key=lambda x:x['dev']['nll'])['epoch']:raise ValueError('Fixed pass selection required')
        selected=r['curve'][r['selected_epoch']-1]['dev']
        if r['final']!=selected:raise ValueError('Selected final score changed')
        if not all(s['formula_coverage_complete'] for x in r['work_samples'] for s in x['stages'].values()):raise ValueError('Complete work required')
        w=r['work'];n=w['fitting_targets']
        if n!=1024 or w['optimizer_updates']!=64:raise ValueError('Fixed target/update exposure required')
        if abs(w['whole_fit_unit_special_flops_estimate']/n-w['fit_unit_special_flops_per_target_estimate'])>1e-6:raise ValueError('Common denominator required')
        rows.append(dict(arm=arm,result=path,result_sha256=sha(path),distinct_fit_targets=256,fit_presentations=n,
            development_accuracy=r['final']['accuracy'],development_nll=r['final']['nll'],
            whole_fit_gflops_estimate=w['whole_fit_unit_special_flops_estimate']/1e9,
            fit_mflops_per_presentation_estimate=w['fit_unit_special_flops_per_target_estimate']/1e6,
            inference_mflops_per_target_estimate=w['inference_unit_special_flops_per_target_estimate']/1e6,
            parameters=r['parameters'],available_receivers=w['native_available_receivers'],
            key_scores_per_fit_presentation=r['activity']['key_scores']/n,selected_updates_per_fit_presentation=r['activity']['selected_updates']/n,
            counterfactual_values_per_fit_presentation=r['activity']['counterfactual_values']/n,
            curve=[dict(epoch=x['epoch'],nll=x['dev']['nll'],accuracy=x['dev']['accuracy']) for x in r['curve']],wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb']))
    gates=[]
    for row in rows[1:]:
        gain=rows[0]['development_nll']-row['development_nll'];decline=100*(rows[0]['development_accuracy']-row['development_accuracy'])
        ratio=row['whole_fit_gflops_estimate']/rows[0]['whole_fit_gflops_estimate']
        gates.append(dict(schedule=row['arm'],nll_improvement=gain,accuracy_decline_percentage_points=decline,whole_fit_work_ratio=ratio,
            promotion_gate_passed=gain>=.02 and decline<=1 and ratio<=1.50 and max(rows[0]['max_rss_kb'],row['max_rss_kb'])<900000))
    admitted=[row for row in rows[1:] if next(g for g in gates if g['schedule']==row['arm'])['promotion_gate_passed']]
    winner=min(admitted,key=lambda x:(x['development_nll'],x['whole_fit_gflops_estimate']))['arm'] if admitted else None
    return dict(common_unit_ledger=rows,schedule_gates=gates,selected_schedule=winner,promotion_gate_passed=winner is not None,
        gate=dict(minimum_nll_improvement=.02,maximum_accuracy_decline_pp=1,maximum_work_ratio=1.50,maximum_rss_kb=900000))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--treatment',action='append',required=True);a=p.parse_args();started=time.perf_counter()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique tag required')
    result=analyze(a.reference,a.treatment)
    result.update(status='completed',args={**vars(a), 'seed':completed(a.reference)['args']['seed']},
        source_sha256={n:sha(n) for n in ['experiments/dvs_evolution_offset_comparison.py','experiments/dvs_credit_comparison.py']},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Fixed256-fit/192-dev/four-pass paired initialization/noise/data; native vs up to two prespecified offset schedules. Minimum devNLL selects a candidate among passing gates for seed7 confirmation, not a proven schedule mechanism, full-data strong-control advantage or official-test claim. Alternation gives fewer updates to private blocks and different active clipping; shared maps remain active. Complete forward/backward/optimizer and offset inference work charged.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
