"""Completed matched attribution of deferred value-projection credit."""
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
        warm=r['final']['frozen_fit_replay_dev'];iw=warm['replay_work']['inference_sample']
        rows.append(dict(arm=key,result=name,result_sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),
            cold_bpc=r['final']['dev']['bpc'],frozen_fit_replay_bpc=warm['bpc'],selected_epoch=r['selected_epoch'],
            fit=a['fit'],passes=a['epochs'],development_targets=r['final']['dev']['n'],
            payload=a['payload'],depth=a['depth'],heads=a['heads'],model=a['model'],parameters=r['parameters'],
            fitting_targets=n,optimizer_updates=r['work']['optimizer_steps'],
            cpu_whole_fit_gflops_estimate=c['total_training_unit_special_flops']/1e9,
            cpu_fit_mflops_per_target_estimate=c['total_training_unit_special_flops']/n/1e6,
            cpu_inference_mflops_per_target=(c['inference_arithmetic_flops_per_character']+c['inference_special_functions_per_character'])/1e6,
            projected_whole_fit_gflops_estimate=e['total_training_unit_special_flops']/1e9,
            projected_fit_mflops_per_target_estimate=e['total_training_unit_special_flops']/n/1e6,
            replay_cpu_gflops_estimate=warm['replay_work']['cpu_unit_special_flops_estimate']/1e9,
            replay_amortized_cpu_mflops_per_dev_target_estimate=warm['replay_work']['amortized_cpu_unit_special_flops_per_dev_target']/1e6,
            warm_inference_cpu_mflops_per_target=(iw['trace']['arithmetic_flops']+iw['trace']['special_function_evaluations'])/iw['tokens']/1e6,
            core_capacity=a['depth']*a['heads']*a['pool'],context_capacity=a['buckets'] if a['model']!='native' else 0,
            cold_final_state=r['final']['dev']['persistent_state'],warm_final_state=warm['persistent_state'],
            actual_fitting_activity=r['work']['actual_fitting_activity'],
            selected_updates_per_target=r['work']['actual_fitting_activity']['receiver_updates']/n,
            key_scores_per_target=r['work']['actual_fitting_activity']['receiver_scores']/n,
            teacher_values_per_fit_target=r['work']['actual_fitting_activity']['receiver_teacher_values']/n,
            context_reads_per_target=r['work']['actual_fitting_activity']['context_reads']/n,
            context_writes_per_target=r['work']['actual_fitting_activity']['context_writes']/n,
            curve=[dict(epoch=x['epoch'],cold_bpc=x['dev']['bpc']) for x in r['curve']],
            wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb']))
    template=results['native_full'];skip={'tag','model','payload','depth','resume','stop_after_updates'}
    for key,r in results.items():
        canonical=lambda x:{k:v for k,v in x['args'].items() if k not in skip}
        if canonical(r)!=canonical(template):raise ValueError('Unexpected optimizer/data protocol difference: '+key)
        for name in ('fitting_data_sha256','development_data_sha256','source_sha256'):
            if r[name]!=template[name]:raise ValueError('Data/source mismatch: '+key)
        if (r['work']['fitting_targets'],r['work']['optimizer_steps'])!=(4092,64):raise ValueError('Target/update mismatch')
        if r['final']['dev']['n']!=2047 or r['final']['frozen_fit_replay_dev']['replay_tokens']!=1024:
            raise ValueError('Evaluation/replay mismatch')
    for key in ('addressed_full','late_full'):
        if results[key]['initial_dev']['bpc']!=template['initial_dev']['bpc']:raise ValueError('Zero-repair initialization mismatch')
    score=lambda key:results[key]['final']['dev']['bpc']
    cost=lambda key:results[key]['work']['cpu_emulator']['total_training_unit_special_flops']
    gains=dict(late_vs_native_full_bpc=score('native_full')-score('late_full'),
        late_vs_addressed_full_bpc=score('addressed_full')-score('late_full'),
        late_full_vs_minimal_bpc=score('late_minimal')-score('late_full'),
        late_full_vs_same_width_shallow_bpc=score('late_shallow')-score('late_full'))
    gate=all(gains[k]>=.02 for k in gains) and cost('late_full')/cost('addressed_full')<=2
    return dict(status='completed',kind='exploratory_deferred_value_projection_attribution',common_unit_ledger=rows,
        **gains,late_vs_original_fitting_work_ratio_estimate=cost('late_full')/cost('addressed_full'),
        followup_gate_passed=gate,
        gate=dict(minimum_all_cold_quality_gains_bpc=.02,maximum_late_vs_original_fitting_work_ratio=2,
            same_width_shallow_control_required=True),
        protocol=dict(fitting=[0,1024],development=[90000000,90002048],passes=4,credit=16,update_targets=64,
            seed=6,warmup_targets=512,selection='cold development only; warm replay scored at that same checkpoint'),
        accounting_scope='Whole-fit costs are representative-window estimates, same convention for every arm; CPU and projected clockless ledgers separate. Replay is extra frozen work. Integer/hash/traffic/RNG/energy not FLOPs.',
        scope='Reused development and one seed, not confirmation/semantics/supremacy. Original addressed/minimal-original treatment contrast incomplete; shallow same-width control isolates depth better than the minimal width change. Gate nominates further controlled evidence, no automatic scale-up.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused plain tag required')
    started=time.perf_counter();plan=json.loads((ROOT/a.plan).read_text());r=analyze(plan)
    r.update(args=vars(a),official_test_read=False,wall_s=time.perf_counter()-started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={'experiments/addressed_memory_pair_analysis.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k.endswith('bpc') or k=='followup_gate_passed'}),flush=True)


if __name__=='__main__':main()
